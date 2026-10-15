//! Supervisor thread owning the GPU execution pool Tokio runtime.
//!
//! The pool's Tokio runtime is built and dropped on a dedicated OS thread
//! named `algeon-gpu-{device}-supervisor`. The thread that constructs the pool
//! does not build or drop it. Dropping a multi-thread runtime joins its workers, and
//! that join is forbidden from inside a runtime worker and from inside another
//! Tokio runtime. Pool owners (backend close, process exit, the last `Arc`
//! drop) may be on either kind of thread, so they only request an exit and
//! the supervisor performs the drop itself.
//!
//! Teardown proceeds through distinct observable stages so cleanup coordinators
//! can bound each phase independently by requesting exit, waiting for the
//! published exit state, and joining the supervisor thread. Because the supervisor
//! publishes its exit state prior to thread termination, bounded waits can
//! detect runtime-drop panics before thread join.

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

/// Owner of the thread that holds one pool's Tokio runtime.
///
/// The supervisor outlives the pool it serves: [`GpuExecutionPoolInner`]
/// holds an `Arc` to it, and so does the process-exit registration, which
/// needs a join target even after the last ordinary pool reference is gone.
/// The runtime itself is never stored here. It lives on the supervisor
/// thread's stack, so the only way to drop it is through the exit request.
pub(super) struct RuntimeSupervisor {
    device_ordinal: usize,
    pub(super) control: Arc<RuntimeSupervisorControl>,
    /// Taken by the first successful or failed join. `None` after that, or
    /// while a build failure was already joined inside [`Self::start`].
    join: Mutex<Option<thread::JoinHandle<()>>>,
    /// One of the `SUPERVISOR_*` constants. A join outcome is recorded here
    /// because the `JoinHandle` can be consumed only once, yet `join` may be
    /// called again by a later cleanup phase and must report the same result.
    pub(super) join_disposition: AtomicU8,
    /// Present from construction until the supervisor thread finishes or an
    /// explicit join succeeds. Cleared by whichever happens first.
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

/// Handshake state shared between the supervisor thread and pool owners.
///
/// `exit_requested` flows from owners to the supervisor thread and `exit`
/// flows back. Both live under one mutex with one condition variable because
/// a waiter for either direction must not miss a notification sent between
/// its check and its wait.
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
    /// Set once, by the supervisor thread, after the runtime is dropped or
    /// the build fails. Readers treat `Some` as permission to join.
    exit: Option<RuntimeSupervisorExit>,
}

/// Terminal state the supervisor thread publishes before it returns.
#[derive(Debug, Clone)]
enum RuntimeSupervisorExit {
    Clean,
    /// Dropping the runtime unwound. The runtime is gone either way, but the
    /// panic is reported to the cleanup coordinator and is not swallowed.
    Panicked {
        panic_message: String,
    },
    /// `Builder::build` failed before any handle was published. Retained so a
    /// later `wait_for_exit_until` on a half-constructed pool reports the
    /// cause and does not wait for the deadline.
    BuildFailed {
        message: String,
    },
}

/// First message the supervisor thread sends to [`RuntimeSupervisor::start`].
enum RuntimeSupervisorStartup {
    Ready(Handle),
    BuildFailed(String),
}

impl RuntimeSupervisor {
    /// Spawns the supervisor thread, builds the runtime on it, and returns the
    /// runtime handle once the thread has published it.
    ///
    /// Worker threads record `pool_id` and a sequential worker index in
    /// thread-locals on start. The pool reads those to detect same-pool
    /// submissions, which run inline, and to label worker observations.
    ///
    /// A runtime build failure is joined here and reported as a `BUILD`
    /// error; no supervisor survives a failed start. If the thread exits
    /// before publishing either outcome, that is reported as a distinct
    /// disconnection error carrying the thread's panic message when there
    /// is one.
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
                    // A closed startup channel means `start` already returned
                    // an error; no owner exists to request exit, so return and
                    // drop the runtime before it can accept work.
                    if startup_sender.send(RuntimeSupervisorStartup::Ready(runtime.handle().clone())).is_err() {
                        return;
                    }

                    // Dropping the runtime cancels every async task at its next
                    // yield point and blocks until each `spawn_blocking` closure
                    // returns. A job body is synchronous GPU work, so a running
                    // job finishes before its task is cancelled. The pool closes
                    // submissions before requesting exit, which is why the drop
                    // waits for that request and does not wait on the handle count.
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
                // Runs on every path out of the closure, including a build
                // failure. Once the runtime is gone there is no CUDA-capable
                // worker left for the process-exit hook to join.
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

    /// Registers this pool with the CUDA process-exit hook.
    ///
    /// Called after the pool's `Arc` exists, because the registration holds a
    /// weak pointer to it: an exit while the pool is still alive runs the full
    /// close-and-drain path, while an exit after the pool was dropped only
    /// has to join the supervisor thread. Registration returns `None` when
    /// the CUDA runtime is unavailable, in which case there is no libcudart
    /// finalizer to order against.
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

    /// Asks the supervisor thread to drop the runtime. Idempotent; the thread
    /// acts on the first request and later ones find the flag already set.
    ///
    /// Callers must have closed pool submissions first, which
    /// [`GpuExecutionPoolInner::request_supervisor_exit`] enforces. Dropping
    /// the runtime cancels any task that has not completed, so an exit while
    /// submissions are still open would cancel jobs the pool had accepted.
    pub(super) fn request_exit(&self) -> Result<()> {
        let mut state = self.control.state.lock().map_err(|_| supervisor_state_poisoned(self.device_ordinal))?;
        state.exit_requested = true;
        self.control.changed.notify_all();
        Ok(())
    }

    /// [`Self::request_exit`] for `Drop` and process-exit paths, which cannot
    /// propagate an error and must still stop the runtime when the state
    /// mutex was poisoned by an earlier panic.
    pub(super) fn request_exit_from_drop(&self) {
        let mut state = self.control.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.exit_requested = true;
        self.control.changed.notify_all();
    }

    /// Blocks until the supervisor thread publishes its exit state or
    /// `deadline` passes.
    ///
    /// Success means the runtime was dropped cleanly. A teardown panic and a
    /// retained build failure are returned as errors whose `reason` fact names
    /// them; both still mean the thread is finished and may be joined, which
    /// is why [`GpuExecutionPool::close_and_wait_until`] joins after either.
    /// The deadline error carries the cleanup-deadline marker so coordinators
    /// can distinguish a slow drain from a broken one.
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

    /// Joins the supervisor thread after it has published exit.
    ///
    /// Joining before exit is published is rejected without attempting the join.
    /// A join without a deadline could block a cleanup coordinator for as long
    /// as the runtime takes to drain. The coordinator owns the deadline, so it
    /// must call [`Self::wait_for_exit_until`] first.
    /// A thread cannot join itself, so a call from the supervisor thread
    /// (possible when the final pool reference is dropped by a pool job)
    /// leaves the handle in place and reports a self-join error.
    ///
    /// The first call consumes the `JoinHandle` and records the outcome in
    /// `join_disposition`; later calls replay that outcome. A join failure is
    /// permanent because the thread's panic payload cannot be re-joined.
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

    /// Unbounded join for the process-exit hook. Returning early would let
    /// the Tokio workers run past libcudart's own `atexit` finalizer, so this
    /// never times out. It does return without joining when called from the
    /// supervisor thread itself, which can happen when process exit begins
    /// on a pool worker; that thread is already finishing and cannot be
    /// joined by itself.
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
        // Keep the join lock to the join itself; unregistering takes the
        // registration and registry locks.
        drop(join);
        self.unregister_process_exit();
    }
}

impl ProcessExitShutdown for ProcessExitGpuExecutionPool {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::ExecutionPool
    }

    fn shutdown_for_process_exit(&self) {
        // The temporary pool clone is dropped at the end of this branch. Its
        // `Drop` only re-requests exit, which is idempotent after the join.
        if let Some(inner) = self.inner.upgrade() {
            GpuExecutionPool { inner, resource_epoch: None }.close_and_join_for_process_exit();
        } else {
            self.supervisor.join_for_process_exit();
        }
    }
}

impl RuntimeSupervisorControl {
    /// Parks the supervisor thread until an owner requests exit. Recovers a
    /// poisoned mutex because the thread must still drop the runtime after a
    /// panic elsewhere left the state lock poisoned.
    fn wait_for_exit_request(&self) {
        let mut state = self.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        while !state.exit_requested {
            state = self.changed.wait(state).unwrap_or_else(sync::PoisonError::into_inner);
        }
    }

    /// Records the terminal state and wakes every `wait_for_exit_until`
    /// caller. Runs on the supervisor thread after the runtime is gone, so
    /// from this point a join cannot block on task completion.
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
