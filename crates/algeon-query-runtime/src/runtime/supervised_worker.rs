//! Runtime-owned supervision for bounded blocking workers.
//!
//! The runtime owns worker submission, cancellation, and process-exit shutdown
//! for attempt-scoped blocking work such as source reads; calling layers produce
//! the work closure. Worker execution follows a uniform lifecycle across all
//! submitting operators.
//!
//! Pool process-exit shutdown waits for accepted blocking jobs without signaling
//! cancellation tokens. The supervisor signals cancellation prior to joining,
//! allowing workers that poll the token to terminate before starting CUDA work
//! during process exit.
//!
//! Workers execute on the device executor's bounded blocking lane rather than
//! async workers or private thread pools. Admission uses the pool's execution-class
//! permit, completion returns through a permit-free channel, and pool shutdown
//! resolves outstanding joins with typed cancellation. No outstanding join is left waiting.

use std::sync::{Arc, Condvar, Mutex};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use crate::{
    observability::RuntimeObservationHandle,
    runtime::{
        PoolBlockingWork,
        context::ExecutionCancellation,
        gpu_execution_pool::{GpuExecutionJobKind, GpuExecutorLease},
        process_exit::{
            ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
        },
    },
};

/// Terminal outcome of one supervised worker join.
pub enum SupervisedWorkerJoin<T> {
    /// The worker finished and delivered its output.
    Completed(T),
    /// The worker panicked.
    Panicked,
    /// The worker was cancelled or lost to pool shutdown/process exit before
    /// delivering completion.
    Cancelled,
}

/// Owns one pool-carried blocking worker until it has been joined.
///
/// The consumer owns this wrapper, while process-exit infrastructure retains the
/// supervisor through the process-exit registry. This handles scenarios where
/// `std::process::exit` bypasses finish or abandon handlers while workers
/// synchronize leased CUDA streams.
///
/// Supervisor instances do not escape this module. [`SupervisedWorker::submit`]
/// atomically performs registration, spawn, and handle ownership for one worker,
/// preventing concurrent submissions from overwriting active handles.
pub struct SupervisedWorker<T> {
    supervisor: Arc<SupervisedWorkerSupervisor<T>>,
}

impl<T: Send + 'static> SupervisedWorker<T> {
    /// Submits one blocking worker onto the device executor under runtime
    /// supervision.
    #[doc(hidden)]
    pub fn submit<F>(
        cancellation: ExecutionCancellation,
        executor: &GpuExecutorLease,
        observation: RuntimeObservationHandle,
        worker: F,
    ) -> Result<Self>
    where
        F: FnOnce() -> T + Send + 'static,
    {
        let supervisor = SupervisedWorkerSupervisor::new(cancellation);
        // Register before submitting: `std::process::exit` may otherwise take
        // its registry snapshot while this worker is already executing CUDA.
        supervisor.activate_process_exit();
        supervisor.spawn_on(executor, observation, worker)
    }

    #[doc(hidden)]
    pub fn join(self) -> Option<SupervisedWorkerJoin<T>> {
        self.supervisor.join()
    }
}

/// Shared between the consumer's `SupervisedWorker` and the process-exit
/// registry. Whichever of them reaches `join` first drains the worker; the
/// other waits for that join to finish and gets `None`.
struct SupervisedWorkerSupervisor<T> {
    /// The work's own token. Process exit cancels through it so a worker that
    /// has not started CUDA work exits at its next check.
    cancellation: ExecutionCancellation,
    state: Mutex<SupervisedWorkerSupervisorState<T>>,
    state_changed: Condvar,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

struct SupervisedWorkerSupervisorState<T> {
    /// Set once by the exit hook. A submission that observes it refuses to
    /// spawn, because the hook's join would otherwise miss that worker.
    process_exit_started: bool,
    /// The one join handle. `Some` from a successful spawn until a joiner
    /// takes it.
    worker: Option<PoolBlockingWork<T>>,
    /// A joiner holds the handle and is blocked in `PoolBlockingWork::join`;
    /// a second joiner waits on `state_changed` until it clears.
    joining: bool,
    joined: bool,
}

impl<T> SupervisedWorkerSupervisor<T> {
    fn new(cancellation: ExecutionCancellation) -> Arc<Self> {
        Arc::new(Self {
            cancellation,
            state: Mutex::new(SupervisedWorkerSupervisorState {
                process_exit_started: false,
                worker: None,
                joining: false,
                joined: false,
            }),
            state_changed: Condvar::new(),
            process_exit_registration: Mutex::new(None),
        })
    }

    fn request_process_exit(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.process_exit_started = true;
        // Release the state lock before cancel(), which wakes every cancellation waiter.
        drop(state);
        // Workers check this token before entering device-bound work. It
        // cannot interrupt a CUDA call already in flight, so the subsequent
        // join remains the terminal safety boundary.
        self.cancellation.cancel();
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    fn finish_join(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.joining = false;
        state.joined = true;
        self.state_changed.notify_all();
        // Release the state lock before unregistering so the notified waiters can re-lock it.
        drop(state);
        // Keep the registry entry live until the worker has completed. This
        // is also safe when the process-exit snapshot already owns the entry.
        self.unregister_process_exit();
    }

    /// Takes the handle for this caller to join, or blocks while another
    /// caller joins and returns `None` once that join is done. The wait is
    /// what lets the exit hook return only after the consumer's concurrent
    /// join has fully drained the worker.
    fn take_or_wait_for_join(&self) -> Option<PoolBlockingWork<T>> {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(worker) = state.worker.take() {
                state.joining = true;
                return Some(worker);
            }
            if state.joining {
                state = self.state_changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
                continue;
            }
            return None;
        }
    }

    /// Drains the worker once. `None` means another caller already
    /// joined it. The classification relies on `PoolBlockingWork::join`
    /// reporting a worker panic as the pool's `Internal` error and a worker
    /// lost to shutdown or process exit as `Cancelled`.
    fn join(&self) -> Option<SupervisedWorkerJoin<T>> {
        let worker = self.take_or_wait_for_join()?;
        let outcome = match worker.join() {
            Ok(value) => SupervisedWorkerJoin::Completed(value),
            Err(error) if error.code() == QueryErrorCode::Internal => SupervisedWorkerJoin::Panicked,
            Err(_) => SupervisedWorkerJoin::Cancelled,
        };
        self.finish_join();
        Some(outcome)
    }

    #[cfg(test)]
    fn joined_for_test(&self) -> bool {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).joined
    }
}

impl<T: Send + 'static> SupervisedWorkerSupervisor<T> {
    fn activate_process_exit(self: &Arc<Self>) {
        let shutdown: Arc<dyn ProcessExitShutdown> = Arc::clone(self) as _;
        let registration = register_cuda_process_exit_shutdown(shutdown);
        *self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = registration;
    }

    /// Submits `worker` to the device executor's blocking lane under this
    /// supervision.
    ///
    /// Only [`SupervisedWorker::submit`] calls this, once per supervisor, so
    /// the single worker slot can never be overwritten by a second submission.
    fn spawn_on<F>(
        self: &Arc<Self>,
        executor: &GpuExecutorLease,
        observation: RuntimeObservationHandle,
        worker: F,
    ) -> Result<SupervisedWorker<T>>
    where
        F: FnOnce() -> T + Send + 'static,
    {
        // Hold this mutex across the process-exit check and handle
        // publication. A process-exit callback either observes the fully
        // owned worker or sets the terminal boundary first, in which case no
        // CUDA-capable worker may begin.
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.process_exit_started {
            // Unregistering takes the registry locks; keep it out of the state lock window.
            drop(state);
            self.unregister_process_exit();
            return Err(QueryError::new(
                QueryErrorCode::Cancelled,
                "query_engine.runtime.supervised_worker.submit",
                "supervised blocking worker cannot start after process exit begins",
            )
            .with_fact("reason", "process_exit_started"));
        }
        let work = match executor.submit_blocking(GpuExecutionJobKind::AttemptWork, observation, worker) {
            Ok(work) => work,
            Err(error) => {
                // Unregistering takes the registry locks; keep it out of the state lock window.
                drop(state);
                self.unregister_process_exit();
                return Err(error);
            },
        };
        state.worker = Some(work);
        Ok(SupervisedWorker { supervisor: Arc::clone(self) })
    }
}

impl<T: Send + 'static> ProcessExitShutdown for SupervisedWorkerSupervisor<T> {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::SupervisedWorker
    }

    fn shutdown_for_process_exit(&self) {
        self.request_process_exit();
        let _ = self.join();
        // `join` unregisters after an owned worker completes. If exit won the
        // race before submitting, there is no worker to join but the terminal
        // boundary still makes removing this registration safe.
        self.unregister_process_exit();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::runtime::gpu_execution_pool::GpuExecutionPool;

    fn test_executor() -> (GpuExecutionPool, GpuExecutorLease) {
        let pool = GpuExecutionPool::new(0, 2).expect("test device executor");
        let executor = GpuExecutorLease::from_pool(pool.clone());
        (pool, executor)
    }

    #[test]
    fn process_exit_joins_supervised_worker_and_prevents_late_spawn() {
        let (_pool, executor) = test_executor();
        let cancellation = ExecutionCancellation::new();
        let supervisor = SupervisedWorkerSupervisor::new(cancellation.clone());
        let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(0);
        let worker_cancellation = cancellation.clone();
        let worker = supervisor
            .spawn_on(&executor, RuntimeObservationHandle::disabled(), move || {
                started_sender.send(()).unwrap();
                while !worker_cancellation.is_cancelled() {
                    std::thread::yield_now();
                }
                7_u8
            })
            .unwrap();
        started_receiver.recv_timeout(Duration::from_secs(5)).unwrap();

        supervisor.shutdown_for_process_exit();

        assert!(cancellation.is_cancelled());
        assert!(supervisor.joined_for_test());
        assert!(worker.join().is_none(), "process-exit owner already joined the worker");

        let late_spawn = supervisor.spawn_on(&executor, RuntimeObservationHandle::disabled(), || 9_u8);
        let error = match late_spawn {
            Ok(_) => panic!("supervised worker unexpectedly spawned after process exit"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("process_exit_started"));
    }

    #[test]
    fn submit_returns_independent_workers_that_each_join_once() {
        let pool = GpuExecutionPool::new(0, 3).expect("test device executor");
        let executor = GpuExecutorLease::from_pool(pool);
        let first = SupervisedWorker::submit(
            ExecutionCancellation::new(),
            &executor,
            RuntimeObservationHandle::disabled(),
            || 7_u8,
        )
        .expect("first submit");
        let second = SupervisedWorker::submit(
            ExecutionCancellation::new(),
            &executor,
            RuntimeObservationHandle::disabled(),
            || 9_u8,
        )
        .expect("second submit");
        assert!(matches!(first.join(), Some(SupervisedWorkerJoin::Completed(7_u8))));
        assert!(matches!(second.join(), Some(SupervisedWorkerJoin::Completed(9_u8))));
    }

    #[test]
    fn submit_propagates_worker_panic_through_join() {
        let (_pool, executor) = test_executor();
        let worker = SupervisedWorker::submit(
            ExecutionCancellation::new(),
            &executor,
            RuntimeObservationHandle::disabled(),
            || -> u8 { panic!("supervised worker panic probe") },
        )
        .expect("submit");
        assert!(matches!(worker.join(), Some(SupervisedWorkerJoin::Panicked)));
    }
}
