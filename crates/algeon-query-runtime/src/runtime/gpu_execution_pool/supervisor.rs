//! Supervisor thread owning the GPU execution pool Tokio runtime.

#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::{
    any::Any,
    panic,
    panic::AssertUnwindSafe,
    sync,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicU8, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::runtime::{Builder, Handle};

use super::{
    BUILD_OPERATION, CURRENT_POOL_ID, CURRENT_WORKER_INDEX, GpuExecutionPool, GpuExecutionPoolInner, SHUTDOWN_OPERATION,
};
use crate::runtime::{
    deadline::{DeadlineMutexLockError, deadline_remaining, mutex_lock_until},
    process_exit::{
        ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
    },
};

pub(super) struct RuntimeSupervisor {
    device_ordinal: usize,
    pub(super) control: Arc<RuntimeSupervisorControl>,
    join: Mutex<Option<thread::JoinHandle<()>>>,
    pub(super) join_disposition: AtomicU8,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

/// Keeps only a weak pool pointer plus the runtime-owning supervisor alive for
/// process teardown. The registry does not retain query state or CUDA objects;
/// the supervisor is enough to request runtime shutdown and join every Tokio
/// worker if the final ordinary pool reference vanished on one of those
/// workers.
struct ProcessExitGpuExecutionPool {
    inner: Weak<GpuExecutionPoolInner>,
    supervisor: Arc<RuntimeSupervisor>,
}

pub(super) const SUPERVISOR_NOT_JOINED: u8 = 0;
pub(super) const SUPERVISOR_JOINED: u8 = 1;
const SUPERVISOR_JOIN_FAILED: u8 = 2;

#[derive(Debug, Default)]
pub(super) struct RuntimeSupervisorControl {
    state: Mutex<RuntimeSupervisorState>,
    changed: Condvar,
    #[cfg(test)]
    pub(super) panic_during_teardown: AtomicBool,
    #[cfg(test)]
    pub(super) panic_after_exit: AtomicBool,
}

#[derive(Debug, Default)]
struct RuntimeSupervisorState {
    exit_requested: bool,
    exit: Option<RuntimeSupervisorExit>,
}

#[derive(Debug, Clone)]
enum RuntimeSupervisorExit {
    Clean,
    Panicked { panic_message: String },
    BuildFailed { message: String },
}

enum RuntimeSupervisorStartup {
    Ready(Handle),
    BuildFailed(String),
}

impl RuntimeSupervisor {
    pub(super) fn start(
        device_ordinal: usize,
        worker_count: usize,
        pool_id: u64,
        next_worker_index: Arc<AtomicUsize>,
    ) -> Result<(Handle, Arc<Self>)> {
        let control = Arc::new(RuntimeSupervisorControl::default());
        let supervisor = Arc::new(Self {
            device_ordinal,
            control,
            join: Mutex::new(None),
            join_disposition: AtomicU8::new(SUPERVISOR_NOT_JOINED),
            process_exit_registration: Mutex::new(None),
        });
        let thread_control = Arc::clone(&supervisor.control);
        let worker_supervisor = Arc::clone(&supervisor);
        let (startup_sender, startup_receiver) = mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name(format!("algeon-gpu-{device_ordinal}-supervisor"))
            .spawn(move || {
                (|| {
                    let runtime = Builder::new_multi_thread()
                        .worker_threads(worker_count)
                        .thread_name(format!("algeon-gpu-{device_ordinal}"))
                        .on_thread_start({
                            let next_worker_index = Arc::clone(&next_worker_index);
                            move || {
                                CURRENT_POOL_ID.with(|current| current.set(Some(pool_id)));
                                CURRENT_WORKER_INDEX.with(|current| {
                                    current.set(Some(next_worker_index.fetch_add(1, Ordering::Relaxed)));
                                });
                            }
                        })
                        .on_thread_stop(|| {
                            CURRENT_WORKER_INDEX.with(|current| current.set(None));
                            CURRENT_POOL_ID.with(|current| current.set(None));
                        })
                        .enable_all()
                        .build();
                    let runtime = match runtime {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            let message = error.to_string();
                            thread_control
                                .publish_exit(RuntimeSupervisorExit::BuildFailed { message: message.clone() });
                            let _ = startup_sender.send(RuntimeSupervisorStartup::BuildFailed(message));
                            return;
                        },
                    };
                    if startup_sender.send(RuntimeSupervisorStartup::Ready(runtime.handle().clone())).is_err() {
                        drop(runtime);
                        return;
                    }

                    let teardown = panic::catch_unwind(AssertUnwindSafe(|| {
                        thread_control.wait_for_exit_request();
                        #[cfg(test)]
                        if thread_control.panic_during_teardown.load(Ordering::Acquire) {
                            panic!("injected GPU execution supervisor teardown panic");
                        }
                        drop(runtime);
                    }));
                    let exit = match teardown {
                        Ok(()) => RuntimeSupervisorExit::Clean,
                        Err(panic) => RuntimeSupervisorExit::Panicked { panic_message: panic_message(panic.as_ref()) },
                    };
                    thread_control.publish_exit(exit);

                    #[cfg(test)]
                    if thread_control.panic_after_exit.load(Ordering::Acquire) {
                        panic!("injected GPU execution supervisor join panic");
                    }
                })();
                worker_supervisor.unregister_process_exit();
            })
            .map_err(|error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "failed to start the dedicated GPU execution runtime supervisor",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("worker_count", worker_count)
                .with_fact("reason", "runtime_supervisor_spawn_failed")
                .with_source(error)
            })?;

        let handle = match startup_receiver.recv() {
            Ok(RuntimeSupervisorStartup::Ready(handle)) => handle,
            Ok(RuntimeSupervisorStartup::BuildFailed(message)) => {
                let join_failed = join.join().err().map(|panic| panic_message(panic.as_ref()));
                let mut error = QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "failed to initialize the dedicated GPU execution runtime",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("worker_count", worker_count)
                .with_fact("reason", "runtime_build_failed")
                .with_fact("runtime_error", message);
                if let Some(join_panic) = join_failed {
                    error = error.with_fact("supervisor_join_panic", join_panic);
                }
                return Err(error);
            },
            Err(error) => {
                let join_failed = join.join().err().map(|panic| panic_message(panic.as_ref()));
                let mut query_error = QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "GPU execution runtime supervisor exited before publishing its handle",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("worker_count", worker_count)
                .with_fact("reason", "runtime_supervisor_startup_disconnected")
                .with_source(error);
                if let Some(join_panic) = join_failed {
                    query_error = query_error.with_fact("supervisor_join_panic", join_panic);
                }
                return Err(query_error);
            },
        };

        *supervisor.join.lock().unwrap_or_else(sync::PoisonError::into_inner) = Some(join);
        Ok((handle, supervisor))
    }

    pub(super) fn register_process_exit(self: &Arc<Self>, inner: Weak<GpuExecutionPoolInner>) {
        let shutdown: Arc<dyn ProcessExitShutdown> =
            Arc::new(ProcessExitGpuExecutionPool { inner, supervisor: Arc::clone(self) });
        let registration = register_cuda_process_exit_shutdown(shutdown);
        *self.process_exit_registration.lock().unwrap_or_else(sync::PoisonError::into_inner) = registration;
    }

    fn unregister_process_exit(&self) {
        let registration = self.process_exit_registration.lock().unwrap_or_else(sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    pub(super) fn request_exit(&self) -> Result<()> {
        let mut state = self.control.state.lock().map_err(|_| supervisor_state_poisoned(self.device_ordinal))?;
        state.exit_requested = true;
        self.control.changed.notify_all();
        Ok(())
    }

    pub(super) fn request_exit_from_drop(&self) {
        let mut state = self.control.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.exit_requested = true;
        self.control.changed.notify_all();
    }

    pub(super) fn wait_for_exit_until(&self, deadline: Instant) -> Result<()> {
        let mut state = mutex_lock_until(&self.control.state, deadline).map_err(|error| match error {
            DeadlineMutexLockError::Poisoned => supervisor_state_poisoned(self.device_ordinal),
            DeadlineMutexLockError::TimedOut => supervisor_exit_timeout(self.device_ordinal),
        })?;
        while state.exit.is_none() {
            let Some(remaining) = deadline_remaining(deadline) else {
                return Err(supervisor_exit_timeout(self.device_ordinal));
            };
            let (next, timeout) = self
                .control
                .changed
                .wait_timeout(state, remaining)
                .map_err(|_| supervisor_state_poisoned(self.device_ordinal))?;
            state = next;
            if timeout.timed_out() && state.exit.is_none() {
                return Err(supervisor_exit_timeout(self.device_ordinal));
            }
        }
        match state.exit.as_ref().expect("exit checked above") {
            RuntimeSupervisorExit::Clean => Ok(()),
            RuntimeSupervisorExit::Panicked { panic_message } => {
                Err(supervisor_panic_error(self.device_ordinal, "runtime_teardown", panic_message))
            },
            RuntimeSupervisorExit::BuildFailed { message } => Err(QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution supervisor retained a runtime build failure",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "runtime_supervisor_build_failed")
            .with_fact("runtime_error", message)),
        }
    }

    pub(super) fn join(&self) -> Result<()> {
        {
            let state = self.control.state.lock().map_err(|_| supervisor_state_poisoned(self.device_ordinal))?;
            if state.exit.is_none() {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    SHUTDOWN_OPERATION,
                    "GPU execution supervisor cannot be joined before it publishes exit",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "runtime_supervisor_join_before_exit"));
            }
        }

        let mut join = self.join.lock().map_err(|_| {
            QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution supervisor join mutex was poisoned",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "runtime_supervisor_join_poisoned")
        })?;
        if let Some(handle) = join.take() {
            if handle.thread().id() == thread::current().id() {
                *join = Some(handle);
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    SHUTDOWN_OPERATION,
                    "GPU execution supervisor cannot join itself",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "runtime_supervisor_self_join"));
            }
            match handle.join() {
                Ok(()) => self.join_disposition.store(SUPERVISOR_JOINED, Ordering::Release),
                Err(panic) => {
                    self.join_disposition.store(SUPERVISOR_JOIN_FAILED, Ordering::Release);
                    return Err(supervisor_panic_error(
                        self.device_ordinal,
                        "thread_join",
                        &panic_message(panic.as_ref()),
                    ));
                },
            }
        }
        let result = match self.join_disposition.load(Ordering::Acquire) {
            SUPERVISOR_JOINED => Ok(()),
            SUPERVISOR_JOIN_FAILED => Err(QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution supervisor previously failed to join",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "runtime_supervisor_join_failed")),
            unexpected => unreachable!("invalid GPU execution supervisor join disposition {unexpected}"),
        };
        self.unregister_process_exit();
        result
    }

    pub(super) fn join_for_process_exit(&self) {
        self.request_exit_from_drop();
        let mut join = self.join.lock().unwrap_or_else(sync::PoisonError::into_inner);
        let Some(handle) = join.take() else {
            return;
        };
        if handle.thread().id() == thread::current().id() {
            *join = Some(handle);
            return;
        }
        let joined = handle.join();
        self.join_disposition
            .store(if joined.is_ok() { SUPERVISOR_JOINED } else { SUPERVISOR_JOIN_FAILED }, Ordering::Release);
        drop(join);
        self.unregister_process_exit();
    }
}

impl ProcessExitShutdown for ProcessExitGpuExecutionPool {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::ExecutionPool
    }

    fn shutdown_for_process_exit(&self) {
        if let Some(inner) = self.inner.upgrade() {
            GpuExecutionPool { inner, resource_epoch: None }.close_and_join_for_process_exit();
        } else {
            self.supervisor.join_for_process_exit();
        }
    }
}

impl RuntimeSupervisorControl {
    fn wait_for_exit_request(&self) {
        let mut state = self.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        while !state.exit_requested {
            state = self.changed.wait(state).unwrap_or_else(sync::PoisonError::into_inner);
        }
    }

    fn publish_exit(&self, exit: RuntimeSupervisorExit) {
        let mut state = self.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.exit = Some(exit);
        self.changed.notify_all();
    }
}

fn supervisor_exit_timeout(device_ordinal: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        SHUTDOWN_OPERATION,
        "timed out waiting for the GPU execution runtime supervisor to exit",
    )
    .with_fact("device_ordinal", device_ordinal)
    .with_fact("reason", "runtime_supervisor_exit_timeout")
    .with_cleanup_deadline_elapsed()
}

fn supervisor_state_poisoned(device_ordinal: usize) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, SHUTDOWN_OPERATION, "GPU execution supervisor state was poisoned")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("reason", "runtime_supervisor_state_poisoned")
}

fn supervisor_panic_error(device_ordinal: usize, stage: &'static str, panic_message: &str) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, SHUTDOWN_OPERATION, "GPU execution runtime supervisor panicked")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("supervisor_stage", stage)
        .with_fact("panic_message", panic_message)
        .with_fact("reason", "runtime_supervisor_panicked")
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    panic
        .downcast_ref::<&'static str>()
        .map(|message| (*message).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}
