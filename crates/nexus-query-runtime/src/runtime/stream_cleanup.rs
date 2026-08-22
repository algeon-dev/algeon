//! Deferred service-stream cleanup supervisor and its fail-closed accounting.

use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};
#[cfg(test)]
use std::{cell::RefCell, sync::Barrier};

use super::{
    context::QueryServiceResourceEpoch,
    process_exit::{
        ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
    },
    service_stream::{ServiceStreamPool, ServiceStreamPoolInner, StreamPermitClass},
};
use crate::error::{QueryError, QueryErrorCode, Result};

pub(super) const STREAM_CLEANUP_SUPERVISOR_BUILD_OPERATION: &str = "nexus_query_engine.service_stream_pool.build";

type StreamCleanupCallback = Box<dyn FnOnce(StreamCleanupOutcome) + Send>;

trait ServiceStreamTestProbe: Send + Sync {
    fn fail_next_supervisor_spawn(&self) -> bool {
        false
    }
    fn force_next_cleanup_enqueue_fallback(&self) -> bool {
        false
    }
    fn force_next_synchronization_failure(&self) -> bool {
        false
    }
    fn pause_before_worker_outcome(&self) {}
    fn panic_after_worker_outcome(&self) -> bool {
        false
    }
    fn notify_shutdown_send_failure(&self) {}
    fn notify_join_waiter(&self) {}
    #[cfg(test)]
    fn as_test_probe(&self) -> Option<&TestServiceStreamProbe> {
        None
    }
}

struct NoopServiceStreamTestProbe;

impl ServiceStreamTestProbe for NoopServiceStreamTestProbe {}

#[derive(Clone)]
pub(super) struct ServiceStreamTestProbeSlot {
    probe: Arc<Mutex<Arc<dyn ServiceStreamTestProbe>>>,
}

impl std::fmt::Debug for ServiceStreamTestProbeSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ServiceStreamTestProbeSlot").finish_non_exhaustive()
    }
}

impl Default for ServiceStreamTestProbeSlot {
    fn default() -> Self {
        Self { probe: Arc::new(Mutex::new(Arc::new(NoopServiceStreamTestProbe))) }
    }
}

impl ServiceStreamTestProbeSlot {
    fn current(&self) -> Arc<dyn ServiceStreamTestProbe> {
        Arc::clone(&self.probe.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    fn fail_next_supervisor_spawn(&self) -> bool {
        self.current().fail_next_supervisor_spawn()
    }
    fn force_next_cleanup_enqueue_fallback(&self) -> bool {
        self.current().force_next_cleanup_enqueue_fallback()
    }
    fn force_next_synchronization_failure(&self) -> bool {
        self.current().force_next_synchronization_failure()
    }
    fn pause_before_worker_outcome(&self) {
        self.current().pause_before_worker_outcome();
    }
    fn panic_after_worker_outcome(&self) -> bool {
        self.current().panic_after_worker_outcome()
    }
    fn notify_shutdown_send_failure(&self) {
        self.current().notify_shutdown_send_failure();
    }
    fn notify_join_waiter(&self) {
        self.current().notify_join_waiter();
    }

    #[cfg(test)]
    pub(super) fn with_test_probe<T>(&self, action: impl FnOnce(&TestServiceStreamProbe) -> T) -> T {
        let current = self.current();
        if let Some(probe) = current.as_test_probe() {
            return action(probe);
        }
        let concrete_probe = Arc::new(TestServiceStreamProbe::default());
        let probe: Arc<dyn ServiceStreamTestProbe> = concrete_probe.clone();
        *self.probe.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = probe;
        action(&concrete_probe)
    }

    #[cfg(test)]
    pub(super) fn install_for_next_pool(probe: Arc<TestServiceStreamProbe>) {
        NEXT_POOL_TEST_PROBE.with(|slot| *slot.borrow_mut() = Some(probe));
    }

    #[cfg(test)]
    pub(super) fn install_next_pool_probe(&self) {
        let probe = NEXT_POOL_TEST_PROBE.with(|slot| slot.borrow_mut().take());
        if let Some(probe) = probe {
            *self.probe.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = probe;
        }
    }
}

#[cfg(test)]
thread_local! {
    static NEXT_POOL_TEST_PROBE: RefCell<Option<Arc<TestServiceStreamProbe>>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct TestServiceStreamProbe {
    pub(super) fail_next_supervisor_spawn: AtomicBool,
    pub(super) force_next_cleanup_enqueue_fallback: AtomicBool,
    pub(super) force_next_synchronization_failure: AtomicBool,
    pub(super) panic_after_worker_outcome: AtomicBool,
    pub(super) worker_outcome_pause: Mutex<Option<StreamCleanupWorkerOutcomePause>>,
    pub(super) shutdown_send_failed: Mutex<Option<mpsc::Sender<()>>>,
    pub(super) join_waiter: Mutex<Option<mpsc::Sender<()>>>,
}

#[cfg(test)]
impl ServiceStreamTestProbe for TestServiceStreamProbe {
    fn fail_next_supervisor_spawn(&self) -> bool {
        self.fail_next_supervisor_spawn.swap(false, Ordering::AcqRel)
    }
    fn force_next_cleanup_enqueue_fallback(&self) -> bool {
        self.force_next_cleanup_enqueue_fallback.swap(false, Ordering::AcqRel)
    }
    fn force_next_synchronization_failure(&self) -> bool {
        self.force_next_synchronization_failure.swap(false, Ordering::AcqRel)
    }
    fn pause_before_worker_outcome(&self) {
        if let Some(pause) = self.worker_outcome_pause.lock().unwrap().clone() {
            pause.arrived.wait();
            pause.release.wait();
        }
    }
    fn panic_after_worker_outcome(&self) -> bool {
        self.panic_after_worker_outcome.load(Ordering::Acquire)
    }
    fn notify_shutdown_send_failure(&self) {
        if let Some(sender) = self.shutdown_send_failed.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }
    fn notify_join_waiter(&self) {
        if let Some(sender) = self.join_waiter.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }
    fn as_test_probe(&self) -> Option<&TestServiceStreamProbe> {
        Some(self)
    }
}
type StreamCleanupTailCallback = Box<dyn FnOnce() + Send>;

/// Typed evidence handed to the pre-quiescence cleanup callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum StreamCleanupOutcome {
    Completed,
    StreamSynchronizationFailed,
    CallbackAbandoned,
}

impl StreamCleanupOutcome {
    fn from_synchronized(synchronized: bool) -> Self {
        if synchronized { Self::Completed } else { Self::StreamSynchronizationFailed }
    }
}

#[derive(Default)]
pub(super) struct StreamCleanupCompletion {
    before_quiescence: Option<StreamCleanupCallback>,
    after_quiescence: Option<StreamCleanupTailCallback>,
}

impl StreamCleanupCompletion {
    pub(super) fn new(
        before_quiescence: impl FnOnce(StreamCleanupOutcome) + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            before_quiescence: Some(Box::new(before_quiescence)),
            after_quiescence: Some(Box::new(after_quiescence)),
        }
    }

    fn run_before_quiescence(&mut self, outcome: StreamCleanupOutcome) {
        if let Some(completion) = self.before_quiescence.take() {
            completion(outcome);
        }
    }

    fn run_after_quiescence(&mut self) {
        if let Some(completion) = self.after_quiescence.take() {
            completion();
        }
    }

    /// The receiver can discard queued cleanup jobs when a prior job panics.
    /// Complete every remaining handoff while unwinding so that the typed
    /// resource and observation permits cannot be stranded. Callback panics
    /// are intentionally suppressed here: the originating supervisor failure
    /// remains the fail-closed result and a second unwind from `Drop` would
    /// abort the process.
    fn run_remaining_after_abandonment(&mut self) {
        if let Some(completion) = self.before_quiescence.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| completion(StreamCleanupOutcome::CallbackAbandoned)));
        }
        if let Some(completion) = self.after_quiescence.take() {
            let _ = catch_unwind(AssertUnwindSafe(completion));
        }
    }
}

impl Drop for StreamCleanupCompletion {
    fn drop(&mut self) {
        self.run_remaining_after_abandonment();
    }
}

/// The pool-owned receiver thread which serializes CUDA stream cleanup.
///
/// This is deliberately separate from external completion delivery. Its only
/// responsibilities are synchronization, CUDA runtime destruction, and pool
/// accounting. A clean shutdown is not inferred from stream counters: the
/// owner must first request shutdown, observe this thread's terminal state,
/// and reclaim its join handle.
pub(super) struct StreamCleanupSupervisor {
    pub(super) device: cudf::DeviceId,
    pub(super) pool_id: u64,
    pub(super) probe: ServiceStreamTestProbeSlot,
    pub(super) sender: mpsc::Sender<StreamCleanupJob>,
    pub(super) submission_gate: Mutex<()>,
    pub(super) control: Arc<StreamCleanupSupervisorControl>,
    pub(super) join: Mutex<Option<thread::JoinHandle<StreamCleanupWorkerOutcome>>>,
    pub(super) join_disposition: AtomicU64,
    pub(super) process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

pub(super) enum StreamCleanupEnqueueError {
    ProcessExit(StreamCleanupJob),
    SupervisorStopped(StreamCleanupJob),
}

/// A process-exit owner deliberately retains only the cleanup-thread control
/// plane. It keeps no CUDA runtime or pool state alive, but it can upgrade a
/// still-live pool to close it before joining the worker. If the final pool
/// reference is released by the worker itself, the supervisor remains
/// registered long enough for the process hook to join that last thread.
struct ProcessExitStreamPool {
    inner: Weak<ServiceStreamPoolInner>,
    supervisor: Arc<StreamCleanupSupervisor>,
}

pub(super) const STREAM_CLEANUP_SUPERVISOR_NOT_JOINED: u64 = 0;
pub(super) const STREAM_CLEANUP_SUPERVISOR_JOINED: u64 = 1;
pub(super) const STREAM_CLEANUP_SUPERVISOR_JOIN_FAILED: u64 = 2;

#[derive(Debug, Default)]
pub(super) struct StreamCleanupSupervisorControl {
    pub(super) state: Mutex<StreamCleanupSupervisorState>,
    pub(super) changed: Condvar,
}

#[derive(Debug, Default)]
pub(super) struct StreamCleanupSupervisorState {
    pub(super) shutdown_requested: bool,
    pub(super) joiner_active: bool,
    pub(super) terminal: Option<StreamCleanupSupervisorTerminal>,
}

#[derive(Debug, Clone)]
pub(super) enum StreamCleanupWorkerOutcome {
    Clean,
    Panicked { panic_message: String },
    Unexpected { detail: &'static str },
}

#[derive(Debug, Clone)]
pub(super) enum StreamCleanupSupervisorTerminal {
    Clean,
    Panicked { stage: &'static str, panic_message: String },
    Unexpected { detail: &'static str },
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct StreamCleanupWorkerOutcomePause {
    pub(super) arrived: Arc<Barrier>,
    pub(super) release: Arc<Barrier>,
}

pub(super) enum StreamCleanupExecutorExit {
    Shutdown,
    ReceiverDisconnected,
    // Only test control jobs request an unexpected exit; see the `#[cfg(test)]`
    // `StreamCleanupControlJob` implementations below.
    #[cfg(test)]
    Unexpected(&'static str),
}

impl std::fmt::Debug for StreamCleanupSupervisor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamCleanupSupervisor")
            .field("device_ordinal", &self.device.ordinal())
            .field("pool_id", &self.pool_id)
            .field("join_disposition", &self.join_disposition.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl StreamCleanupSupervisor {
    pub(super) fn start(device: cudf::DeviceId, pool_id: u64, probe: ServiceStreamTestProbeSlot) -> Result<Arc<Self>> {
        let (sender, receiver) = mpsc::channel();
        let control = Arc::new(StreamCleanupSupervisorControl::default());
        let supervisor = Arc::new(Self {
            device,
            pool_id,
            probe: probe.clone(),
            sender,
            submission_gate: Mutex::new(()),
            control,
            join: Mutex::new(None),
            join_disposition: AtomicU64::new(STREAM_CLEANUP_SUPERVISOR_NOT_JOINED),
            process_exit_registration: Mutex::new(None),
        });
        let thread_control = Arc::clone(&supervisor.control);
        let thread_probe = probe.clone();
        let worker_supervisor = Arc::clone(&supervisor);
        let join = if probe.fail_next_supervisor_spawn() {
            Err(std::io::Error::other("injected service stream cleanup supervisor spawn failure"))
        } else {
            thread::Builder::new().name(format!("nexus-stream-cleanup-d{}-p{pool_id}", device.ordinal())).spawn(
                move || {
                    let outcome = run_stream_cleanup_worker(receiver, &thread_control, &thread_probe);
                    worker_supervisor.unregister_process_exit();
                    outcome
                },
            )
        }
        .map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                STREAM_CLEANUP_SUPERVISOR_BUILD_OPERATION,
                "failed to start the service stream cleanup supervisor",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("stream_pool_id", pool_id)
            .with_fact("reason", "stream_cleanup_supervisor_spawn_failed")
            .with_source(error)
        })?;

        *supervisor.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(join);
        Ok(supervisor)
    }

    pub(super) fn register_process_exit(self: &Arc<Self>, inner: Weak<ServiceStreamPoolInner>) {
        let shutdown: Arc<dyn ProcessExitShutdown> =
            Arc::new(ProcessExitStreamPool { inner, supervisor: Arc::clone(self) });
        let registration = register_cuda_process_exit_shutdown(shutdown);
        let mut slot = self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *slot = registration;
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    #[cfg(test)]
    pub(super) fn enqueue(&self, job: StreamCleanupJob) -> std::result::Result<(), mpsc::SendError<StreamCleanupJob>> {
        let _submission = self.submission_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let state = self.control.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.shutdown_requested || state.terminal.is_some() {
            return Err(mpsc::SendError(job));
        }
        // A sender failure only says that the receiver has gone away. The
        // worker's joined outcome remains the sole source of terminal state.
        self.sender.send(job)
    }

    pub(super) fn enqueue_before_process_exit(
        &self,
        job: StreamCleanupJob,
        process_exit_started: &AtomicBool,
    ) -> std::result::Result<(), StreamCleanupEnqueueError> {
        let _submission = self.submission_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if process_exit_started.load(Ordering::Acquire) {
            return Err(StreamCleanupEnqueueError::ProcessExit(job));
        }
        let state = self.control.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.shutdown_requested || state.terminal.is_some() {
            return Err(StreamCleanupEnqueueError::SupervisorStopped(job));
        }
        self.sender.send(job).map_err(|error| StreamCleanupEnqueueError::SupervisorStopped(error.0))
    }

    pub(super) fn request_shutdown(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let _submission = self.lock_submission_until(deadline, operation)?;
        let mut state = self.lock_control_until(deadline, operation)?;
        if state.shutdown_requested || state.terminal.is_some() {
            return Ok(());
        }
        state.shutdown_requested = true;
        let send_failed = self.sender.send(StreamCleanupJob::Shutdown).is_err();
        drop(state);
        if send_failed {
            self.probe.notify_shutdown_send_failure();
        }
        Ok(())
    }

    fn request_shutdown_from_drop(&self) {
        let _submission = self.submission_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut state = self.control.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.shutdown_requested || state.terminal.is_some() {
            return;
        }
        state.shutdown_requested = true;
        let _ = self.sender.send(StreamCleanupJob::Shutdown);
    }

    pub(super) fn ensure_not_failed(&self, operation: &'static str) -> Result<()> {
        let state = self
            .control
            .state
            .lock()
            .map_err(|_| stream_cleanup_supervisor_state_poisoned(operation, self.device, self.pool_id))?;
        match state.terminal.as_ref() {
            None => Ok(()),
            Some(terminal) => self.terminal_result(operation, terminal),
        }
    }

    /// Commit an already-finished worker before an idle/quiescence caller
    /// reports success. A healthy worker remains untouched: only a finished
    /// handle or a join already in progress can enter the terminal path.
    #[cfg(test)]
    pub(super) fn finalize_if_finished_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let mut state = self.lock_control_until(deadline, operation)?;
        loop {
            if let Some(terminal) = state.terminal.as_ref() {
                return self.terminal_result(operation, terminal);
            }
            if state.joiner_active {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(stream_cleanup_supervisor_join_timeout(operation, self.device, self.pool_id));
                }
                let (next, timeout) = self
                    .control
                    .changed
                    .wait_timeout(state, remaining)
                    .map_err(|_| stream_cleanup_supervisor_state_poisoned(operation, self.device, self.pool_id))?;
                state = next;
                if timeout.timed_out() && state.terminal.is_none() {
                    return Err(stream_cleanup_supervisor_join_timeout(operation, self.device, self.pool_id));
                }
                continue;
            }

            // Holding control state while we inspect the handle prevents a
            // competing caller from electing itself between the finished
            // observation and this caller's election. The joiner never holds
            // the join mutex while acquiring control state.
            let finished = {
                let join = self.lock_join_until(deadline, operation)?;
                join.as_ref().map(thread::JoinHandle::is_finished).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        operation,
                        "service stream cleanup supervisor join handle was unavailable before terminal commitment",
                    )
                    .with_fact("device_ordinal", self.device.ordinal())
                    .with_fact("stream_pool_id", self.pool_id)
                    .with_fact("reason", "stream_cleanup_supervisor_join_unavailable")
                })?
            };
            if !finished {
                return Ok(());
            }

            state.joiner_active = true;
            drop(state);
            return self.join_and_commit_terminal_until(deadline, operation);
        }
    }

    /// Elect one caller to join the worker and commit its terminal state. A
    /// worker outcome is not observable until the receiver and every queued
    /// cleanup job have been dropped and the elected caller has joined it.
    pub(super) fn wait_for_exit_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let mut state = self.lock_control_until(deadline, operation)?;
        loop {
            if let Some(terminal) = state.terminal.as_ref() {
                return self.terminal_result(operation, terminal);
            }
            if !state.joiner_active {
                state.joiner_active = true;
                drop(state);
                return self.join_and_commit_terminal_until(deadline, operation);
            }

            self.probe.notify_join_waiter();
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(stream_cleanup_supervisor_join_timeout(operation, self.device, self.pool_id));
            }
            let (next, timeout) = self
                .control
                .changed
                .wait_timeout(state, remaining)
                .map_err(|_| stream_cleanup_supervisor_state_poisoned(operation, self.device, self.pool_id))?;
            state = next;
            if timeout.timed_out() && state.terminal.is_none() {
                return Err(stream_cleanup_supervisor_join_timeout(operation, self.device, self.pool_id));
            }
        }
    }

    fn join_and_commit_terminal_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let joined = self.join_worker_until(deadline, operation);
        let joined = match joined {
            Ok(joined) => joined,
            Err(error) => {
                self.relinquish_joiner();
                return Err(error);
            },
        };

        // Joining establishes the happens-before edge from receiver teardown,
        // in-flight recovery, and queued-job recovery to terminal commitment.
        let mut state = self.control.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.terminal.is_none(), "only the elected joiner may commit a cleanup terminal state");
        let terminal = match joined {
            Ok(outcome) => Self::terminal_from_worker_outcome(outcome, state.shutdown_requested),
            Err(panic) => StreamCleanupSupervisorTerminal::Panicked {
                stage: "thread_join",
                panic_message: stream_cleanup_supervisor_panic_message(panic.as_ref()),
            },
        };
        state.terminal = Some(terminal.clone());
        state.joiner_active = false;
        self.control.changed.notify_all();
        drop(state);
        self.terminal_result(operation, &terminal)
    }

    fn join_worker_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::thread::Result<StreamCleanupWorkerOutcome>> {
        let mut join = self.lock_join_until(deadline, operation)?;
        if let Some(handle) = join.as_ref() {
            while !handle.is_finished() {
                if Instant::now() >= deadline {
                    return Err(stream_cleanup_supervisor_join_timeout(operation, self.device, self.pool_id));
                }
                thread::yield_now();
            }
        }
        let handle = join.take().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "service stream cleanup supervisor join handle was unavailable before terminal commitment",
            )
            .with_fact("device_ordinal", self.device.ordinal())
            .with_fact("stream_pool_id", self.pool_id)
            .with_fact("reason", "stream_cleanup_supervisor_join_unavailable")
        })?;
        let joined = handle.join();
        self.join_disposition.store(
            if joined.is_ok() { STREAM_CLEANUP_SUPERVISOR_JOINED } else { STREAM_CLEANUP_SUPERVISOR_JOIN_FAILED },
            Ordering::Release,
        );
        self.unregister_process_exit();
        Ok(joined)
    }

    pub(super) fn join_for_process_exit(&self) {
        self.request_shutdown_from_drop();
        let mut join = self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(handle) = join.take() else {
            return;
        };
        if handle.thread().id() == thread::current().id() {
            // The worker will finish its own current cleanup and then drop
            // this handle while unwinding out of its entry point. Joining
            // oneself would deadlock and cannot add a happens-before edge.
            *join = Some(handle);
            return;
        }
        let joined = handle.join();
        self.join_disposition.store(
            if joined.is_ok() { STREAM_CLEANUP_SUPERVISOR_JOINED } else { STREAM_CLEANUP_SUPERVISOR_JOIN_FAILED },
            Ordering::Release,
        );
        drop(join);
        self.unregister_process_exit();
    }

    fn relinquish_joiner(&self) {
        let mut state = self.control.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.terminal.is_none() {
            state.joiner_active = false;
            self.control.changed.notify_all();
        }
    }

    fn terminal_from_worker_outcome(
        outcome: StreamCleanupWorkerOutcome,
        shutdown_requested: bool,
    ) -> StreamCleanupSupervisorTerminal {
        match outcome {
            StreamCleanupWorkerOutcome::Clean if shutdown_requested => StreamCleanupSupervisorTerminal::Clean,
            StreamCleanupWorkerOutcome::Clean => StreamCleanupSupervisorTerminal::Unexpected {
                detail: "cleanup supervisor exited before explicit shutdown",
            },
            StreamCleanupWorkerOutcome::Panicked { panic_message } => {
                StreamCleanupSupervisorTerminal::Panicked { stage: "cleanup_executor", panic_message }
            },
            StreamCleanupWorkerOutcome::Unexpected { detail } => StreamCleanupSupervisorTerminal::Unexpected { detail },
        }
    }

    fn terminal_result(&self, operation: &'static str, terminal: &StreamCleanupSupervisorTerminal) -> Result<()> {
        match terminal {
            StreamCleanupSupervisorTerminal::Clean => Ok(()),
            StreamCleanupSupervisorTerminal::Panicked { stage, panic_message } => {
                Err(stream_cleanup_supervisor_panic_error(operation, self.device, self.pool_id, stage, panic_message))
            },
            StreamCleanupSupervisorTerminal::Unexpected { detail } => {
                Err(stream_cleanup_supervisor_unexpected_exit_error(operation, self.device, self.pool_id, detail))
            },
        }
    }

    fn lock_submission_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, ()>> {
        loop {
            match self.submission_gate.try_lock() {
                Ok(gate) => return Ok(gate),
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(stream_cleanup_supervisor_state_poisoned(operation, self.device, self.pool_id));
                },
                Err(std::sync::TryLockError::WouldBlock) if Instant::now() >= deadline => {
                    return Err(stream_cleanup_supervisor_exit_timeout(operation, self.device, self.pool_id));
                },
                Err(std::sync::TryLockError::WouldBlock) => thread::yield_now(),
            }
        }
    }

    fn lock_control_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, StreamCleanupSupervisorState>> {
        loop {
            match self.control.state.try_lock() {
                Ok(state) => return Ok(state),
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(stream_cleanup_supervisor_state_poisoned(operation, self.device, self.pool_id));
                },
                Err(std::sync::TryLockError::WouldBlock) if Instant::now() >= deadline => {
                    return Err(stream_cleanup_supervisor_exit_timeout(operation, self.device, self.pool_id));
                },
                Err(std::sync::TryLockError::WouldBlock) => thread::yield_now(),
            }
        }
    }

    fn lock_join_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, Option<thread::JoinHandle<StreamCleanupWorkerOutcome>>>> {
        loop {
            match self.join.try_lock() {
                Ok(join) => return Ok(join),
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(stream_cleanup_supervisor_state_poisoned(operation, self.device, self.pool_id));
                },
                Err(std::sync::TryLockError::WouldBlock) if Instant::now() >= deadline => {
                    return Err(stream_cleanup_supervisor_join_timeout(operation, self.device, self.pool_id));
                },
                Err(std::sync::TryLockError::WouldBlock) => thread::yield_now(),
            }
        }
    }
}
impl Drop for ServiceStreamPoolInner {
    fn drop(&mut self) {
        self.cleanup_supervisor.request_shutdown_from_drop();
    }
}

impl ProcessExitShutdown for ProcessExitStreamPool {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::StreamPool
    }

    fn shutdown_for_process_exit(&self) {
        if let Some(inner) = self.inner.upgrade() {
            ServiceStreamPool { inner }.close_and_join_for_process_exit();
        } else {
            self.supervisor.join_for_process_exit();
        }
    }
}

impl ServiceStreamPool {
    pub(super) fn enqueue_cleanup(
        &self,
        runtime: Arc<cudf::Runtime>,
        resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
        mut completion: Option<StreamCleanupCompletion>,
        permit_class: StreamPermitClass,
    ) {
        if self.inner.process_exit_started.load(Ordering::Acquire) {
            // The process-exit owner has already joined (or is joining) the
            // only thread allowed to destroy service runtimes. Leaking this
            // terminal payload is intentional: process exit will reclaim the
            // address space, whereas dropping it could race libcudart's later
            // atexit handler.
            std::mem::forget(runtime);
            std::mem::forget(resource_epoch);
            std::mem::forget(completion);
            return;
        }
        let registered = match self.inner.state.lock() {
            Ok(mut state) => {
                state.pending_cleanups = state.pending_cleanups.saturating_add(1);
                self.inner.state_changed.notify_all();
                true
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to account deferred stream cleanup");
                false
            },
        };
        if !registered {
            let synchronized = runtime.synchronize().is_ok();
            drop(runtime);
            drop(resource_epoch);
            if let Some(completion) = completion.as_mut() {
                completion.run_before_quiescence(StreamCleanupOutcome::from_synchronized(synchronized));
                completion.run_after_quiescence();
            }
            return;
        }
        let job = StreamCleanupJob::Runtime(DeferredStreamCleanup::new(
            self.clone(),
            runtime,
            resource_epoch,
            completion,
            permit_class,
        ));
        let force_enqueue_fallback = self.inner.cleanup_probe.force_next_cleanup_enqueue_fallback();
        if force_enqueue_fallback {
            match job {
                StreamCleanupJob::Runtime(mut cleanup) => cleanup.run(),
                StreamCleanupJob::Shutdown => unreachable!("runtime cleanup cannot enqueue supervisor shutdown"),
                #[cfg(test)]
                StreamCleanupJob::Control(_) => {
                    unreachable!("new cleanup job is always a runtime cleanup")
                },
            }
            return;
        }
        match self.inner.cleanup_supervisor.enqueue_before_process_exit(job, &self.inner.process_exit_started) {
            Ok(()) => {},
            Err(StreamCleanupEnqueueError::ProcessExit(job)) => {
                // The exit flag is checked while holding the same submission
                // gate used to queue supervisor shutdown. A cleanup therefore
                // either precedes shutdown or is retained across the terminal
                // boundary; it can never fall back to CUDA work after join.
                std::mem::forget(job);
            },
            Err(StreamCleanupEnqueueError::SupervisorStopped(job)) => {
                // The only production send failure outside process exit is a
                // terminal cleanup supervisor. The synchronous fallback still
                // releases the job's permits, but the runtime cannot be
                // returned to a pool whose owner has disappeared.
                self.inner.closed.store(true, Ordering::Release);
                match job {
                    StreamCleanupJob::Runtime(mut cleanup) => cleanup.run(),
                    StreamCleanupJob::Shutdown => unreachable!("runtime cleanup cannot enqueue supervisor shutdown"),
                    #[cfg(test)]
                    StreamCleanupJob::Control(_) => {},
                }
            },
        }
    }
}

/// Owns one queued stream cleanup until both resource accounting and the
/// external completion handoff have finished. If a prior cleanup panics, the
/// receiver drops queued jobs; this guard turns that drop into fail-closed
/// resource cleanup rather than silently losing permits or completion work.
pub(super) struct DeferredStreamCleanup {
    pool: ServiceStreamPool,
    runtime: Option<Arc<cudf::Runtime>>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
    completion: Option<StreamCleanupCompletion>,
    phase: DeferredStreamCleanupPhase,
    permit_class: StreamPermitClass,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeferredStreamCleanupPhase {
    Pending,
    Active,
    AfterAccounting,
    Complete,
}

impl DeferredStreamCleanup {
    fn new(
        pool: ServiceStreamPool,
        runtime: Arc<cudf::Runtime>,
        resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
        completion: Option<StreamCleanupCompletion>,
        permit_class: StreamPermitClass,
    ) -> Self {
        Self {
            pool,
            runtime: Some(runtime),
            resource_epoch,
            completion,
            phase: DeferredStreamCleanupPhase::Pending,
            permit_class,
        }
    }

    fn run(&mut self) {
        if !self.begin_accounting() {
            self.discard_after_accounting_failure();
            return;
        }

        let synchronized = self.synchronize_runtime();
        self.finish_runtime_accounting(synchronized);
        if let Some(completion) = self.completion.as_mut() {
            // This callback releases GPU-side/query admission ownership while
            // cleanup remains active. A panic is caught by the supervisor and
            // this guard's Drop path completes the remaining handoff.
            completion.run_before_quiescence(StreamCleanupOutcome::from_synchronized(synchronized));
        }
        self.finish_active_accounting();
        if let Some(completion) = self.completion.as_mut() {
            // The typed callback only enqueues to ObservationCompletionExecutor
            // and therefore cannot reenter backend close from this supervisor.
            completion.run_after_quiescence();
        }
        self.phase = DeferredStreamCleanupPhase::Complete;
    }

    fn begin_accounting(&mut self) -> bool {
        match self.pool.inner.state.lock() {
            Ok(mut state) => {
                state.pending_cleanups = state.pending_cleanups.saturating_sub(1);
                state.active_cleanups = state.active_cleanups.saturating_add(1);
                self.phase = DeferredStreamCleanupPhase::Active;
                self.pool.inner.state_changed.notify_all();
                true
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to start accounted stream cleanup");
                false
            },
        }
    }

    fn synchronize_runtime(&self) -> bool {
        let force_failure = self.pool.inner.cleanup_probe.force_next_synchronization_failure();
        match self.runtime.as_ref().expect("deferred cleanup runtime is present").synchronize() {
            Ok(()) if force_failure => {
                eprintln!(
                    "[nexus-query-engine] warning: injecting device {} stream synchronization failure for test",
                    self.pool.device().ordinal()
                );
                false
            },
            Ok(()) => true,
            Err(error) => {
                eprintln!(
                    "[nexus-query-engine] warning: failed to synchronize device {} service stream; discarding runtime: {error}",
                    self.pool.device().ordinal()
                );
                false
            },
        }
    }

    fn finish_runtime_accounting(&mut self, synchronized: bool) {
        let mut idle_to_drop = Vec::new();
        match self.pool.inner.state.lock() {
            Ok(mut state) => {
                match self.permit_class {
                    StreamPermitClass::Root => state.root_outstanding = state.root_outstanding.saturating_sub(1),
                    StreamPermitClass::Auxiliary => state.aux_outstanding = state.aux_outstanding.saturating_sub(1),
                }
                if synchronized && !self.pool.inner.closed.load(Ordering::Acquire) {
                    state.idle.push(self.runtime.take().expect("deferred cleanup runtime is present"));
                } else {
                    state.created = state.created.saturating_sub(1);
                    if !synchronized {
                        self.pool.inner.closed.store(true, Ordering::Release);
                        state.synchronization_failures = state.synchronization_failures.saturating_add(1);
                        idle_to_drop = std::mem::take(&mut state.idle);
                        state.created = state.created.saturating_sub(idle_to_drop.len());
                    }
                }
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to finish accounted stream cleanup");
            },
        }
        drop(self.runtime.take());
        drop(idle_to_drop);
        drop(self.resource_epoch.take());
    }

    fn finish_active_accounting(&mut self) {
        match self.pool.inner.state.lock() {
            Ok(mut state) => {
                state.active_cleanups = state.active_cleanups.saturating_sub(1);
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to complete stream cleanup accounting");
            },
        }
        self.pool.inner.available.notify_all();
        self.pool.inner.state_changed.notify_all();
        self.phase = DeferredStreamCleanupPhase::AfterAccounting;
    }

    fn discard_after_accounting_failure(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| runtime.synchronize()));
            drop(runtime);
        }
        drop(self.resource_epoch.take());
        if let Some(completion) = self.completion.as_mut() {
            completion.run_before_quiescence(StreamCleanupOutcome::StreamSynchronizationFailed);
            completion.run_after_quiescence();
        }
        self.phase = DeferredStreamCleanupPhase::Complete;
    }

    fn recover_after_abandonment(&mut self) {
        let had_runtime = self.runtime.is_some();
        if let Some(runtime) = self.runtime.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| runtime.synchronize()));
            drop(runtime);
        }
        drop(self.resource_epoch.take());

        let mut idle_to_drop = Vec::new();
        match self.pool.inner.state.lock() {
            Ok(mut state) => {
                match self.phase {
                    DeferredStreamCleanupPhase::Pending => {
                        state.pending_cleanups = state.pending_cleanups.saturating_sub(1);
                        if had_runtime {
                            state.created = state.created.saturating_sub(1);
                        }
                    },
                    DeferredStreamCleanupPhase::Active => {
                        state.active_cleanups = state.active_cleanups.saturating_sub(1);
                        if had_runtime {
                            state.created = state.created.saturating_sub(1);
                        }
                    },
                    DeferredStreamCleanupPhase::AfterAccounting | DeferredStreamCleanupPhase::Complete => {},
                }
                // An abandoned job means the owning supervisor is terminally
                // unhealthy. Never reuse a runtime while its cleanup protocol
                // has been interrupted.
                self.pool.inner.closed.store(true, Ordering::Release);
                idle_to_drop = std::mem::take(&mut state.idle);
                state.created = state.created.saturating_sub(idle_to_drop.len());
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to recover abandoned stream cleanup accounting");
            },
        }
        drop(idle_to_drop);
        self.pool.inner.available.notify_all();
        self.pool.inner.state_changed.notify_all();
        if let Some(mut completion) = self.completion.take() {
            completion.run_remaining_after_abandonment();
        }
        self.phase = DeferredStreamCleanupPhase::Complete;
    }
}

impl Drop for DeferredStreamCleanup {
    fn drop(&mut self) {
        if self.phase != DeferredStreamCleanupPhase::Complete {
            self.recover_after_abandonment();
        }
    }
}
fn stream_cleanup_supervisor_exit_timeout(operation: &'static str, device: cudf::DeviceId, pool_id: u64) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for the service stream cleanup supervisor to exit",
    )
    .with_fact("device_ordinal", device.ordinal())
    .with_fact("stream_pool_id", pool_id)
    .with_fact("reason", "stream_cleanup_supervisor_exit_timeout")
    .with_cleanup_deadline_elapsed()
}

fn stream_cleanup_supervisor_join_timeout(operation: &'static str, device: cudf::DeviceId, pool_id: u64) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting to join the service stream cleanup supervisor",
    )
    .with_fact("device_ordinal", device.ordinal())
    .with_fact("stream_pool_id", pool_id)
    .with_fact("reason", "stream_cleanup_supervisor_join_timeout")
    .with_cleanup_deadline_elapsed()
}

fn stream_cleanup_supervisor_state_poisoned(
    operation: &'static str,
    device: cudf::DeviceId,
    pool_id: u64,
) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, operation, "service stream cleanup supervisor state was poisoned")
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("stream_pool_id", pool_id)
        .with_fact("reason", "stream_cleanup_supervisor_state_poisoned")
}

fn stream_cleanup_supervisor_unexpected_exit_error(
    operation: &'static str,
    device: cudf::DeviceId,
    pool_id: u64,
    detail: &str,
) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, operation, "service stream cleanup supervisor exited unexpectedly")
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("stream_pool_id", pool_id)
        .with_fact("supervisor_detail", detail)
        .with_fact("reason", "stream_cleanup_supervisor_unexpected_exit")
}

fn stream_cleanup_supervisor_panic_error(
    operation: &'static str,
    device: cudf::DeviceId,
    pool_id: u64,
    stage: &'static str,
    panic_message: &str,
) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, operation, "service stream cleanup supervisor panicked")
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("stream_pool_id", pool_id)
        .with_fact("supervisor_stage", stage)
        .with_fact("panic_message", panic_message)
        .with_fact("reason", "stream_cleanup_supervisor_panicked")
}

fn stream_cleanup_supervisor_panic_message(panic: &(dyn Any + Send)) -> String {
    panic
        .downcast_ref::<&'static str>()
        .map(|message| (*message).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

// Control jobs are a test-only probe channel: production only ever enqueues
// runtime cleanups and shutdown.
#[cfg(test)]
pub(super) trait StreamCleanupControlJob: Send {
    fn run(self: Box<Self>) -> Option<StreamCleanupExecutorExit>;
}

pub(super) enum StreamCleanupJob {
    Runtime(DeferredStreamCleanup),
    Shutdown,
    #[cfg(test)]
    Control(Box<dyn StreamCleanupControlJob>),
}

fn run_stream_cleanup_worker(
    receiver: mpsc::Receiver<StreamCleanupJob>,
    _control: &StreamCleanupSupervisorControl,
    probe: &ServiceStreamTestProbeSlot,
) -> StreamCleanupWorkerOutcome {
    let executor = catch_unwind(AssertUnwindSafe(|| run_stream_cleanup_executor(&receiver)));

    // The receiver owns all queued cleanup jobs. Drop it before constructing
    // the provisional outcome so queued recovery completes before any joiner
    // can observe and commit that outcome.
    drop(receiver);

    let outcome = match executor {
        Ok(StreamCleanupExecutorExit::Shutdown) => StreamCleanupWorkerOutcome::Clean,
        Ok(StreamCleanupExecutorExit::ReceiverDisconnected) => {
            StreamCleanupWorkerOutcome::Unexpected { detail: "cleanup receiver disconnected before explicit shutdown" }
        },
        #[cfg(test)]
        Ok(StreamCleanupExecutorExit::Unexpected(detail)) => StreamCleanupWorkerOutcome::Unexpected { detail },
        Err(panic) => StreamCleanupWorkerOutcome::Panicked {
            panic_message: stream_cleanup_supervisor_panic_message(panic.as_ref()),
        },
    };

    probe.pause_before_worker_outcome();
    if probe.panic_after_worker_outcome() {
        panic!("injected service stream cleanup supervisor join panic");
    }
    outcome
}

fn run_stream_cleanup_executor(receiver: &mpsc::Receiver<StreamCleanupJob>) -> StreamCleanupExecutorExit {
    loop {
        let job = match receiver.recv() {
            Ok(job) => job,
            Err(_) => return StreamCleanupExecutorExit::ReceiverDisconnected,
        };
        match job {
            StreamCleanupJob::Runtime(mut cleanup) => cleanup.run(),
            StreamCleanupJob::Shutdown => return StreamCleanupExecutorExit::Shutdown,
            #[cfg(test)]
            StreamCleanupJob::Control(job) => {
                if let Some(exit) = job.run() {
                    return exit;
                }
            },
        }
    }
}

#[cfg(test)]
struct ProbeCleanupJob {
    done: mpsc::Sender<thread::ThreadId>,
}

#[cfg(test)]
impl StreamCleanupControlJob for ProbeCleanupJob {
    fn run(self: Box<Self>) -> Option<StreamCleanupExecutorExit> {
        let _ = self.done.send(thread::current().id());
        None
    }
}

#[cfg(test)]
struct BlockCleanupJob {
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

#[cfg(test)]
impl StreamCleanupControlJob for BlockCleanupJob {
    fn run(self: Box<Self>) -> Option<StreamCleanupExecutorExit> {
        self.entered.wait();
        self.release.wait();
        None
    }
}

#[cfg(test)]
struct UnexpectedExitCleanupJob;

#[cfg(test)]
impl StreamCleanupControlJob for UnexpectedExitCleanupJob {
    fn run(self: Box<Self>) -> Option<StreamCleanupExecutorExit> {
        Some(StreamCleanupExecutorExit::Unexpected("test requested unexpected cleanup supervisor exit"))
    }
}

#[cfg(test)]
pub(super) fn probe_cleanup_job_for_test(done: mpsc::Sender<thread::ThreadId>) -> StreamCleanupJob {
    StreamCleanupJob::Control(Box::new(ProbeCleanupJob { done }))
}

#[cfg(test)]
#[doc(hidden)]
pub struct StreamCleanupBlocker {
    release: Option<Arc<Barrier>>,
}

#[cfg(test)]
impl Drop for StreamCleanupBlocker {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release.wait();
        }
    }
}

#[cfg(test)]
#[doc(hidden)]
pub fn block_stream_cleanup_thread_for_test(pool: &ServiceStreamPool) -> StreamCleanupBlocker {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    pool.inner
        .cleanup_supervisor
        .enqueue(StreamCleanupJob::Control(Box::new(BlockCleanupJob {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        })))
        .expect("cleanup thread accepts blocker");
    entered.wait();
    StreamCleanupBlocker { release: Some(release) }
}

#[cfg(test)]
pub(super) fn stream_cleanup_thread_id_for_test(pool: &ServiceStreamPool) -> thread::ThreadId {
    let (sender, receiver) = mpsc::channel();
    pool.inner.cleanup_supervisor.enqueue(probe_cleanup_job_for_test(sender)).expect("cleanup thread accepts probe");
    receiver.recv().expect("cleanup thread returns probe")
}

#[cfg(test)]
pub(super) fn fail_next_cleanup_supervisor_spawn_for_test() {
    let probe = Arc::new(TestServiceStreamProbe::default());
    probe.fail_next_supervisor_spawn.store(true, Ordering::Release);
    ServiceStreamTestProbeSlot::install_for_next_pool(probe);
}

#[cfg(test)]
pub(super) fn panic_cleanup_supervisor_after_outcome_for_test(pool: &ServiceStreamPool) {
    pool.inner.cleanup_probe.with_test_probe(|probe| {
        probe.panic_after_worker_outcome.store(true, Ordering::Release);
    });
}

#[cfg(test)]
pub(super) fn pause_cleanup_supervisor_before_worker_return_for_test(
    pool: &ServiceStreamPool,
) -> StreamCleanupWorkerOutcomePause {
    let pause =
        StreamCleanupWorkerOutcomePause { arrived: Arc::new(Barrier::new(2)), release: Arc::new(Barrier::new(2)) };
    pool.inner.cleanup_probe.with_test_probe(|probe| {
        *probe.worker_outcome_pause.lock().unwrap() = Some(pause.clone());
    });
    pause
}

#[cfg(test)]
pub(super) fn wait_for_cleanup_supervisor_worker_finished_for_test(pool: &ServiceStreamPool, deadline: Instant) {
    loop {
        let finished =
            pool.inner.cleanup_supervisor.join.lock().unwrap().as_ref().is_some_and(thread::JoinHandle::is_finished);
        if finished {
            return;
        }
        assert!(Instant::now() < deadline, "cleanup supervisor did not finish its provisional outcome");
        thread::yield_now();
    }
}

#[cfg(test)]
pub(super) fn observe_cleanup_supervisor_shutdown_send_failure_for_test(
    pool: &ServiceStreamPool,
) -> mpsc::Receiver<()> {
    let (sender, receiver) = mpsc::channel();
    pool.inner.cleanup_probe.with_test_probe(|probe| {
        *probe.shutdown_send_failed.lock().unwrap() = Some(sender);
    });
    receiver
}

#[cfg(test)]
pub(super) fn observe_cleanup_supervisor_join_waiter_for_test(pool: &ServiceStreamPool) -> mpsc::Receiver<()> {
    let (sender, receiver) = mpsc::channel();
    pool.inner.cleanup_probe.with_test_probe(|probe| {
        *probe.join_waiter.lock().unwrap() = Some(sender);
    });
    receiver
}

#[cfg(test)]
pub(super) fn force_cleanup_supervisor_unexpected_exit_for_test(pool: &ServiceStreamPool) {
    pool.inner
        .cleanup_supervisor
        .enqueue(StreamCleanupJob::Control(Box::new(UnexpectedExitCleanupJob)))
        .expect("cleanup supervisor accepts unexpected-exit test job");
}
