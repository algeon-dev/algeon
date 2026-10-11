//! GPU-backend close and drain coordination.
//!
//! The coordinator owns the asynchronous close worker and the two-scope
//! quiescence state. Backend construction and admission stay in
//! [`super::backend_core`]; this module only consumes the resource generation
//! once that core has closed its linearization gate.
//!
//! Cleanup runs on a dedicated thread, `algeon-gpu-close`, in bounded slices
//! of `CLEANUP_SLICE`. Each slice re-enters the idempotent drain of whichever
//! phase is still pending, which gives three properties at once: a caller's
//! `close_and_wait_until` deadline never has to be known by the worker, a
//! phase that finishes after some caller timed out is still recorded (as a
//! late success), and process exit can stop the worker at a slice boundary.
//! The phase order is fixed by dependency: the service closes admission and
//! drains its pools while the construction journal still holds every GPU
//! resource; the journal drains next; the GPU permit count must reach zero
//! before the generation, and with it the owner lease, is released; only then
//! does the worker wait for observation permits, which may outlive the GPU
//! resources by design.
//!
//! A cleanup that cannot prove completion fails closed. The generation is
//! forgotten rather than dropped, so the live-backend lease stays claimed and
//! no replacement backend can take the devices while GPU work of unknown state
//! may still exist.

use std::{
    any::Any,
    collections::BTreeMap,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use serde::Serialize;

use super::{
    backend_core::{ConstructionResourceJournal, ObservationCore, ResourceCore, ResourceGeneration},
    backend_lifecycle::BackendLifecycle,
    process_exit::{
        ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
    },
    query_service::{QueryService, QueryServiceDrainOutcome, QueryServiceDrainPhaseOutcome},
};
use crate::observability::in_runtime_observation_delivery;

const BUILD_OPERATION: &str = "query_engine.gpu_backend_core.build";
const CLOSE_OPERATION: &str = "query_engine.gpu_backend_core.close";
const CLEANUP_SLICE: Duration = Duration::from_millis(100);

/// Signals the pre-created cleanup coordinator unless construction reaches
/// publication. This is intentionally armed before the first GPU mutation.
pub(crate) struct ConstructionRollback {
    coordinator: Option<Arc<CloseCoordinator>>,
}

impl ConstructionRollback {
    pub(crate) fn new(coordinator: Arc<CloseCoordinator>) -> Self {
        Self { coordinator: Some(coordinator) }
    }

    pub(crate) fn disarm(&mut self) {
        self.coordinator.take();
    }

    pub(crate) fn attach_cleanup_facts(&mut self, error: QueryError) -> QueryError {
        let Some(coordinator) = self.coordinator.take() else {
            return error.with_fact("rollback_cleanup", "not_armed");
        };
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match coordinator.close_and_wait_until(deadline) {
            Ok(snapshot) => error
                .with_fact("rollback_cleanup", "quiescent")
                .with_fact("rollback_gpu_resource_quiescent", snapshot.gpu_resource_quiescent)
                .with_fact("rollback_observation_quiescent", snapshot.observation_quiescent),
            Err(cleanup_error) => error
                .with_fact("rollback_cleanup", "transferred")
                .with_fact("rollback_cleanup_code", cleanup_error.code().as_str())
                .with_fact("rollback_cleanup_reason", cleanup_error.fact_value("reason").unwrap_or("unknown")),
        }
    }
}

impl Drop for ConstructionRollback {
    fn drop(&mut self) {
        if let Some(coordinator) = self.coordinator.take() {
            coordinator.signal();
        }
    }
}

/// Owner of the close worker thread and the close signal.
///
/// Spawned before the first GPU resource is installed, so it exists as the
/// rollback owner for every construction failure as well as for ordinary
/// close. Held by the backend core, by the public-handle tracker's last-drop
/// callback, and by the process-exit registration.
#[derive(Debug)]
pub(crate) struct CloseCoordinator {
    /// One-shot start signal for the worker. Taken by the first `signal`, so
    /// the worker can be started at most once; `None` afterwards.
    sender: Mutex<Option<mpsc::Sender<()>>>,
    pub(crate) state: Arc<CloseCompletion>,
    /// Clone of the observation core's lifecycle. `signal` wins or loses the
    /// close gate here, which is the same gate admissions race against.
    lifecycle: BackendLifecycle,
    /// Taken by the first join after quiescence. Left in place when the
    /// joining thread is the worker itself.
    join: Mutex<Option<thread::JoinHandle<()>>>,
    /// Set by the process-exit hook. The worker checks it at every slice
    /// boundary and returns without releasing the generation.
    process_exit_requested: Arc<AtomicBool>,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

impl CloseCoordinator {
    /// Spawns the worker thread, which parks until the first `signal`.
    ///
    /// The worker owns a strong `ResourceCore` reference so the generation
    /// slot stays reachable even after every public handle and session is
    /// gone. If the worker panics, that reference is forgotten on purpose: a
    /// still-installed generation must not be dropped by an unwinding thread,
    /// because its owner lease would be released with GPU resources live.
    pub(crate) fn spawn(resource: Arc<ResourceCore>, observation: Arc<ObservationCore>) -> Result<Arc<Self>> {
        let (sender, receiver) = mpsc::channel();
        let state = Arc::new(CloseCompletion::new());
        let process_exit_requested = Arc::new(AtomicBool::new(false));
        let coordinator = Arc::new(Self {
            sender: Mutex::new(Some(sender)),
            state: Arc::clone(&state),
            lifecycle: observation.lifecycle.clone(),
            join: Mutex::new(None),
            process_exit_requested: Arc::clone(&process_exit_requested),
            process_exit_registration: Mutex::new(None),
        });
        let worker_state = Arc::clone(&state);
        let worker_lifecycle = observation.lifecycle.clone();
        let resource_for_panic = Arc::clone(&resource);
        let worker_coordinator = Arc::clone(&coordinator);
        let join = thread::Builder::new()
            .name("algeon-gpu-close".to_string())
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let _ = receiver.recv();
                    run_cleanup(resource, worker_lifecycle.clone(), Arc::clone(&worker_state), &process_exit_requested);
                }));
                if result.is_err() {
                    worker_state.fail("cleanup_worker", "cleanup worker panicked");
                    // A panic must not drop a still-installed resource
                    // generation and release its owner lease.
                    std::mem::forget(resource_for_panic);
                }
                worker_coordinator.unregister_process_exit();
            })
            .map_err(|error| {
                QueryError::new(QueryErrorCode::Internal, BUILD_OPERATION, "failed to create backend cleanup worker")
                    .with_fact("reason", "cleanup_worker_spawn_failed")
                    .with_fact("io_error", error.to_string())
            })?;
        *coordinator.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(join);
        Ok(coordinator)
    }

    /// Registers the coordinator with the CUDA process-exit hook. Called by
    /// construction only after the service install has initialized CUDA, so
    /// the engine's `atexit` handler is registered after libcudart's and runs
    /// before it. Registration is `None` when no CUDA runtime is available.
    pub(crate) fn activate_process_exit(self: &Arc<Self>) {
        let shutdown: Arc<dyn ProcessExitShutdown> = Arc::clone(self) as _;
        let registration = register_cuda_process_exit_shutdown(shutdown);
        *self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = registration;
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    /// Closes the lifecycle gate and starts the worker. Returns whether this
    /// call won the gate; only the winner sends the start signal, so the
    /// worker runs once however many callers (explicit close, last handle
    /// drop, construction rollback, process exit) signal.
    ///
    /// Never blocks: the gate is one atomic and the channel send does not
    /// wait. A worker that is already gone (spawn failure path) is recorded as
    /// a cleanup failure so waiters do not hang.
    pub(crate) fn signal(&self) -> bool {
        let gate_winner = self.lifecycle.close();
        if !gate_winner {
            return false;
        }
        self.state.mark_closing();
        let sender = self.sender.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if sender.is_none_or(|sender| sender.send(()).is_err()) {
            self.state.fail("cleanup_signal", "cleanup worker disconnected before close signal");
        }
        true
    }

    /// Signals close and blocks until observation quiescence or `deadline`.
    /// The worker thread is joined only after quiescence, so a timeout leaves
    /// it running and a later call can still observe its completion.
    pub(crate) fn close_and_wait_until(&self, deadline: Instant) -> Result<GpuBackendCloseSnapshot> {
        self.signal();
        let snapshot = self.state.wait_until(deadline, &self.lifecycle)?;
        self.join_after_quiescence()?;
        Ok(snapshot)
    }

    #[cfg(test)]
    pub(crate) fn worker_join_is_reclaimed_for_test(&self) -> bool {
        self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none()
    }

    fn join_after_quiescence(&self) -> Result<()> {
        let mut join = self.join.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, CLOSE_OPERATION, "backend cleanup worker join mutex was poisoned")
                .with_fact("reason", "cleanup_worker_join_poisoned")
        })?;
        let Some(handle) = join.take() else {
            return Ok(());
        };
        if handle.thread().id() == thread::current().id() {
            // A reentrant close invoked by the cleanup worker itself cannot
            // join its own entry point. The worker will unregister only after
            // it has completed all GPU and observation teardown.
            *join = Some(handle);
            return Ok(());
        }
        let joined = handle.join();
        // Keep the join lock to the join itself; unregistering takes the
        // registration and registry locks.
        drop(join);
        self.unregister_process_exit();
        joined.map_err(|panic| {
            QueryError::new(QueryErrorCode::Internal, CLOSE_OPERATION, "backend cleanup worker panicked during join")
                .with_fact("reason", "cleanup_worker_join_panicked")
                .with_fact("panic_message", panic_message(panic.as_ref()))
        })
    }

    fn join_for_process_exit(&self) {
        // `std::process::exit` does not unwind the caller's stack. An admitted
        // handle owned by that stack can therefore never participate in the
        // ordinary quiescence protocol. Tell the cleanup worker to stop at its
        // next bounded phase boundary and leak the resource generation; the
        // separately registered execution/stream supervisors then terminate
        // their workers without allowing CUDA destructors to run after this
        // process-exit hook returns.
        self.process_exit_requested.store(true, Ordering::Release);
        self.signal();
        let mut join = self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(handle) = join.take() else {
            return;
        };
        if handle.thread().id() == thread::current().id() {
            *join = Some(handle);
            return;
        }
        let _ = handle.join();
        // Same lock window as `join_after_quiescence`.
        drop(join);
        self.unregister_process_exit();
    }
}

impl ProcessExitShutdown for CloseCoordinator {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::BackendCoordinator
    }

    fn shutdown_for_process_exit(&self) {
        self.join_for_process_exit();
    }
}

/// The worker body: drains the generation phase by phase and records each
/// phase's progress in `state`.
///
/// Every `return` before `generation.release()` leaves the generation
/// forgotten by `FailClosedGeneration`, which is the fail-closed outcome for
/// a permanent failure and the intended outcome for process exit, where the
/// separately registered pool and stream supervisors stop their workers and
/// nothing may run CUDA destructors afterwards.
fn run_cleanup(
    resource: Arc<ResourceCore>,
    lifecycle: BackendLifecycle,
    state: Arc<CloseCompletion>,
    process_exit_requested: &AtomicBool,
) {
    lifecycle.close();
    // The slot is one `Option` take or set; a guard recovered from a panic
    // still holds the published generation, which close then drains.
    let generation = resource.generation.lock().unwrap_or_else(PoisonError::into_inner).take();
    let Some(generation) = generation else {
        state.mark_gpu_quiescent();
        wait_for_observation_forever(&lifecycle, &state, process_exit_requested);
        return;
    };
    let generation = FailClosedGeneration::new(generation);
    if let Some(service) = generation.service() {
        // `close` rejects queued waiters and new admissions at once; admitted
        // attempts keep their grants and finish through their own release.
        service.close().unwrap_or_else(|error| {
            eprintln!("[query-engine] backend close signal returned an error: {error}");
        });
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
        loop {
            let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
            let outcome: QueryServiceDrainOutcome = service.close_drain_step_until(deadline);
            let mut all_quiescent = true;
            let mut permanent_failure = None;
            for phase_outcome in outcome.phases() {
                match phase_outcome {
                    QueryServiceDrainPhaseOutcome::Quiescent { phase } => {
                        state.record_quiescent(phase.as_str());
                    },
                    QueryServiceDrainPhaseOutcome::DeadlineElapsed { phase, .. } => {
                        all_quiescent = false;
                        state.record_deadline(phase.as_str());
                    },
                    QueryServiceDrainPhaseOutcome::PermanentFailure { phase, error } => {
                        all_quiescent = false;
                        if permanent_failure.is_none() {
                            permanent_failure = Some((phase.as_str(), error));
                        }
                    },
                }
            }
            if let Some((phase, error)) = permanent_failure {
                state.fail(phase, error);
                return;
            }
            if all_quiescent {
                break;
            }
            if process_exit_requested.load(Ordering::Acquire) {
                return;
            }
        }
    }
    // The service first closes admission and drains work using the resources
    // retained by the permanent journal. Only then may the journal close and
    // verify every runtime/resource handle. Construction rollback has no
    // published service and therefore reaches this same journal path directly.
    loop {
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match generation.construction_journal().close_and_drain_until(deadline) {
            Ok(()) => {
                state.record_quiescent("construction_journal");
                break;
            },
            Err(error) if error.is_cleanup_deadline_elapsed() => {
                state.record_deadline("construction_journal");
            },
            Err(error) => {
                state.fail("construction_journal", error);
                return;
            },
        }
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
    }
    loop {
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match lifecycle.wait_for_gpu_quiescence_until(deadline) {
            Ok(()) => {
                state.record_quiescent("gpu_permit_quiescence");
                break;
            },
            Err(error) if error.is_cleanup_deadline_elapsed() => {
                state.record_deadline("gpu_permit_quiescence");
            },
            Err(error) => {
                state.fail("gpu_permit_quiescence", error);
                return;
            },
        }
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
    }
    // Releasing the generation drops the service, then the journal, then the
    // owner lease; a replacement backend can claim the devices from here on.
    generation.release();
    // The core's `Arc` is released before GPU quiescence is published so a
    // waiter that observes the fact cannot find the slot still retained.
    drop(resource);
    state.mark_gpu_quiescent();
    wait_for_observation_forever(&lifecycle, &state, process_exit_requested);
}

/// Waits, in slices, for every observation permit to drop. Unbounded because
/// an observation permit is held by attempt reports that a caller may seal at
/// any later time; only process exit ends the wait early.
fn wait_for_observation_forever(
    lifecycle: &BackendLifecycle,
    state: &CloseCompletion,
    process_exit_requested: &AtomicBool,
) {
    loop {
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match lifecycle.wait_for_observation_quiescence_until(deadline) {
            Ok(()) => {
                state.record_quiescent("observation_quiescence");
                state.mark_observation_quiescent();
                return;
            },
            Err(error) if error.is_cleanup_deadline_elapsed() => {
                state.record_deadline("observation_quiescence");
            },
            Err(error) => {
                state.fail("observation_quiescence", error);
                return;
            },
        }
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
    }
}

/// Holds the generation during cleanup and forgets it unless `release` is
/// called. Forgetting keeps the owner lease claimed for the rest of the
/// process, which is the fail-closed answer to GPU work of unknown state.
struct FailClosedGeneration {
    generation: Option<Arc<ResourceGeneration>>,
}

impl FailClosedGeneration {
    fn new(generation: Arc<ResourceGeneration>) -> Self {
        Self { generation: Some(generation) }
    }

    fn service(&self) -> Option<&QueryService> {
        self.generation.as_ref().expect("generation retained until release").service()
    }

    fn construction_journal(&self) -> &ConstructionResourceJournal {
        self.generation.as_ref().expect("generation retained until release").construction_journal()
    }

    fn release(mut self) {
        drop(self.generation.take());
    }
}

impl Drop for FailClosedGeneration {
    fn drop(&mut self) {
        if let Some(generation) = self.generation.take() {
            std::mem::forget(generation);
        }
    }
}

// Phases advance monotonically in this numeric order. `PHASE_FAILED` is
// absorbing only before GPU quiescence; see `CloseCompletion::fail`.
const PHASE_OPEN: u8 = 0;
const PHASE_CLOSING: u8 = 1;
const PHASE_GPU_QUIESCENT: u8 = 2;
const PHASE_OBSERVATION_QUIESCENT: u8 = 3;
const PHASE_FAILED: u8 = 4;

/// Close progress shared between the worker and every waiter.
///
/// The phase and failure live under one mutex with the condition variable;
/// the counters and the per-phase progress map are separate so a snapshot
/// taken while a waiter holds the wait lock still reads them.
#[derive(Debug)]
pub(crate) struct CloseCompletion {
    wait: Mutex<CloseWaitState>,
    changed: Condvar,
    /// Number of distinct phases that missed at least one cleanup slice.
    deadline_elapsed: AtomicU64,
    /// Number of those phases that later reached quiescence anyway.
    late_successes: AtomicU64,
    cleanup_phases: Mutex<BTreeMap<&'static str, CleanupPhaseProgress>>,
}

/// Phase and failure guarded by the same mutex as the [`Condvar`] predicate,
/// so a state change and its notification are atomic to waiters.
#[derive(Debug)]
struct CloseWaitState {
    phase: u8,
    failure: Option<CloseFailure>,
}

#[derive(Debug, Clone)]
struct CloseFailure {
    phase: &'static str,
    error: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CleanupPhaseProgress {
    DeadlineElapsed,
    Quiescent,
}

impl CloseCompletion {
    pub(crate) fn new() -> Self {
        Self {
            wait: Mutex::new(CloseWaitState { phase: PHASE_OPEN, failure: None }),
            changed: Condvar::new(),
            deadline_elapsed: AtomicU64::new(0),
            late_successes: AtomicU64::new(0),
            cleanup_phases: Mutex::new(BTreeMap::new()),
        }
    }

    fn mark_closing(&self) {
        self.advance(PHASE_CLOSING);
    }

    fn mark_gpu_quiescent(&self) {
        self.advance(PHASE_GPU_QUIESCENT);
    }

    fn mark_observation_quiescent(&self) {
        self.advance(PHASE_OBSERVATION_QUIESCENT);
    }

    fn advance(&self, phase: u8) {
        {
            let mut state = self.wait.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.phase < phase && state.phase != PHASE_FAILED {
                state.phase = phase;
            } else {
                return;
            }
        }
        self.changed.notify_all();
    }

    fn record_deadline(&self, phase: &'static str) {
        let mut phases = self.cleanup_phases.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !phases.contains_key(phase) {
            phases.insert(phase, CleanupPhaseProgress::DeadlineElapsed);
            self.deadline_elapsed.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_quiescent(&self, phase: &'static str) {
        let mut phases = self.cleanup_phases.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if phases.insert(phase, CleanupPhaseProgress::Quiescent) == Some(CleanupPhaseProgress::DeadlineElapsed) {
            self.late_successes.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn fail(&self, phase: &'static str, error: impl fmt::Display) {
        let mut state = self.wait.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.failure.is_none() {
            state.failure = Some(CloseFailure { phase, error: error.to_string() });
        }
        // A later observation-cleanup failure must not erase the already
        // established GPU-resource-quiescent fact. Before GPU quiescence the
        // failed sentinel also prevents any later success transition.
        if state.phase < PHASE_GPU_QUIESCENT {
            state.phase = PHASE_FAILED;
        }
        self.changed.notify_all();
    }

    pub(crate) fn snapshot(&self, lifecycle: &BackendLifecycle) -> GpuBackendCloseSnapshot {
        let state = self.wait.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.snapshot_with(state.phase, state.failure.is_some(), lifecycle)
    }

    fn snapshot_with(&self, phase: u8, cleanup_failed: bool, lifecycle: &BackendLifecycle) -> GpuBackendCloseSnapshot {
        let lifecycle = lifecycle.snapshot();
        GpuBackendCloseSnapshot {
            closing: lifecycle.closing,
            gpu_resource_quiescent: phase == PHASE_GPU_QUIESCENT || phase == PHASE_OBSERVATION_QUIESCENT,
            observation_quiescent: phase == PHASE_OBSERVATION_QUIESCENT,
            active_gpu_permits: lifecycle.active_gpu_permits,
            active_observation_permits: lifecycle.active_observation_permits,
            cleanup_failed,
            cleanup_deadline_elapsed_count: self.deadline_elapsed.load(Ordering::Relaxed),
            cleanup_late_success_count: self.late_successes.load(Ordering::Relaxed),
        }
    }

    /// Blocks until observation quiescence, a recorded cleanup failure, or
    /// `deadline`. A failure is reported in preference to a timeout whenever
    /// both are true at the moment the waiter wakes.
    pub(crate) fn wait_until(
        &self,
        deadline: Instant,
        lifecycle: &BackendLifecycle,
    ) -> Result<GpuBackendCloseSnapshot> {
        let mut state = super::deadline::mutex_lock_until(&self.wait, deadline).map_err(|error| match error {
            super::deadline::DeadlineMutexLockError::Poisoned => close_wait_poisoned_error(),
            // The phase lives behind the lock this wait could not take, so
            // the error omits the quiescence facts instead of blocking on it.
            super::deadline::DeadlineMutexLockError::TimedOut => self.timeout_error(None, lifecycle),
        })?;
        loop {
            if let Some(failure) = state.failure.clone() {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    CLOSE_OPERATION,
                    "backend cleanup failed closed",
                )
                .with_fact("reason", "cleanup_failed_closed")
                .with_fact("cleanup_phase", failure.phase)
                .with_fact("cleanup_error", failure.error));
            }
            if state.phase == PHASE_OBSERVATION_QUIESCENT {
                return Ok(self.snapshot_with(state.phase, false, lifecycle));
            }
            if super::deadline::deadline_expired(deadline) {
                return Err(self.timeout_error(Some(state.phase), lifecycle));
            }
            let Some(remaining) = super::deadline::deadline_remaining(deadline) else {
                return Err(self.timeout_error(Some(state.phase), lifecycle));
            };
            // Every wakeup re-reads the full wait state at the top of the
            // loop, so a failure recorded during the wait is reported as a
            // cleanup failure rather than a timeout.
            let (next, _) =
                self.changed.wait_timeout(state, remaining).map_err(|_error| close_wait_poisoned_error())?;
            state = next;
        }
    }

    /// Timeout error from state readable without blocking: lifecycle permits
    /// and cleanup counters always, quiescence facts only when `phase` was
    /// read under the wait lock.
    fn timeout_error(&self, phase: Option<u8>, lifecycle: &BackendLifecycle) -> QueryError {
        let lifecycle = lifecycle.snapshot();
        let mut error =
            QueryError::new(QueryErrorCode::ServiceOverloaded, CLOSE_OPERATION, "timed out waiting for backend close")
                .with_fact("reason", "backend_close_timeout");
        if let Some(phase) = phase {
            error = error
                .with_fact(
                    "gpu_resource_quiescent",
                    phase == PHASE_GPU_QUIESCENT || phase == PHASE_OBSERVATION_QUIESCENT,
                )
                .with_fact("observation_quiescent", phase == PHASE_OBSERVATION_QUIESCENT);
        }
        error
            .with_fact("active_gpu_permits", lifecycle.active_gpu_permits)
            .with_fact("active_observation_permits", lifecycle.active_observation_permits)
            .with_fact("cleanup_deadline_elapsed_count", self.deadline_elapsed.load(Ordering::Relaxed))
            .with_fact("cleanup_late_success_count", self.late_successes.load(Ordering::Relaxed))
    }
}

/// Two-scope close state. GPU quiescence can become true while observation
/// permits still exist; at that point the device-owner lease has been released.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GpuBackendCloseSnapshot {
    pub closing: bool,
    pub gpu_resource_quiescent: bool,
    pub observation_quiescent: bool,
    pub active_gpu_permits: usize,
    pub active_observation_permits: usize,
    pub cleanup_failed: bool,
    pub cleanup_deadline_elapsed_count: u64,
    pub cleanup_late_success_count: u64,
}

fn close_wait_poisoned_error() -> QueryError {
    QueryError::new(QueryErrorCode::Internal, CLOSE_OPERATION, "backend close completion mutex was poisoned")
        .with_fact("reason", "close_wait_poisoned")
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    panic
        .downcast_ref::<&'static str>()
        .map(|message| (*message).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

/// Refuses a blocking close from inside runtime observation delivery. The
/// observer runs under an observation permit, so a wait for observation
/// quiescence from there could only end at the deadline.
pub(crate) fn reject_reentrant_observation_close() -> Result<()> {
    if in_runtime_observation_delivery() {
        return Err(QueryError::new(
            QueryErrorCode::Internal,
            CLOSE_OPERATION,
            "backend close-and-wait cannot run from runtime observation delivery",
        )
        .with_fact("reason", "reentrant_observation_close"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, mpsc},
        thread,
        time::{Duration, Instant},
    };

    use super::{CleanupPhaseProgress, CloseCompletion};
    use crate::runtime::{backend_lifecycle::BackendLifecycle, query_service::QueryServiceDrainPhase};

    #[test]
    fn cleanup_deadlines_become_late_success_only_for_the_same_phase() {
        let state = CloseCompletion::new();
        let lifecycle = BackendLifecycle::new();
        let phases = [
            QueryServiceDrainPhase::StreamPoolClose.as_str(),
            QueryServiceDrainPhase::AdmissionQuiescence.as_str(),
            QueryServiceDrainPhase::ExecutionPoolQuiescence.as_str(),
            QueryServiceDrainPhase::StreamPoolQuiescence.as_str(),
            QueryServiceDrainPhase::ResourceEpoch.as_str(),
            QueryServiceDrainPhase::FinalCompletion.as_str(),
            "construction_journal",
            "gpu_permit_quiescence",
            "observation_quiescence",
        ];
        for phase in phases {
            state.record_deadline(phase);
            state.record_deadline(phase);
        }
        assert_eq!(state.snapshot(&lifecycle).cleanup_deadline_elapsed_count, phases.len() as u64);

        state.record_quiescent(phases[0]);
        let snapshot = state.snapshot(&lifecycle);
        assert_eq!(snapshot.cleanup_late_success_count, 1);
        let progress = state.cleanup_phases.lock().unwrap();
        assert_eq!(progress.get(phases[0]), Some(&CleanupPhaseProgress::Quiescent));
        assert!(phases[1..].iter().all(|phase| progress.get(phase) == Some(&CleanupPhaseProgress::DeadlineElapsed)));
        drop(progress);

        for phase in &phases[1..] {
            state.record_quiescent(phase);
            state.record_quiescent(phase);
        }
        state.mark_closing();
        state.mark_gpu_quiescent();
        state.mark_observation_quiescent();
        let snapshot = state.snapshot(&lifecycle);

        assert_eq!(snapshot.cleanup_deadline_elapsed_count, phases.len() as u64);
        assert_eq!(snapshot.cleanup_late_success_count, phases.len() as u64);
        assert!(!snapshot.cleanup_failed);
    }

    #[test]
    fn observation_failure_does_not_erase_gpu_quiescence() {
        let state = CloseCompletion::new();
        let lifecycle = BackendLifecycle::new();
        state.mark_closing();
        state.mark_gpu_quiescent();
        state.fail("observation_quiescence", "injected observation cleanup failure");

        let snapshot = state.snapshot(&lifecycle);
        assert!(snapshot.gpu_resource_quiescent);
        assert!(!snapshot.observation_quiescent);
        assert!(snapshot.cleanup_failed);
    }

    #[test]
    fn waiting_at_a_deadline_returns_current_snapshot() {
        let state = CloseCompletion::new();
        let lifecycle = BackendLifecycle::new();
        let error = state.wait_until(Instant::now(), &lifecycle).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("backend_close_timeout"));
        assert_eq!(error.fact_value("gpu_resource_quiescent"), Some("false"));
    }

    #[test]
    fn waiting_while_the_wait_lock_is_held_times_out_before_the_holder_releases_it() {
        let state = Arc::new(CloseCompletion::new());
        let lifecycle = BackendLifecycle::new();
        let (locked_tx, locked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let holder = {
            let state = Arc::clone(&state);
            thread::spawn(move || {
                let _guard = state.wait.lock().unwrap();
                locked_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
        };
        locked_rx.recv().unwrap();

        let (result_tx, result_rx) = mpsc::channel();
        let waiter = {
            let state = Arc::clone(&state);
            thread::spawn(move || {
                result_tx.send(state.wait_until(Instant::now() + Duration::from_millis(20), &lifecycle)).unwrap();
            })
        };
        let result = result_rx.recv_timeout(Duration::from_secs(10));
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        waiter.join().unwrap();

        let error = result.expect("wait_until must return while the wait lock is still held").unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("backend_close_timeout"));
        assert_eq!(error.fact_value("gpu_resource_quiescent"), None);
        assert_eq!(error.fact_value("active_gpu_permits"), Some("0"));
    }
}
