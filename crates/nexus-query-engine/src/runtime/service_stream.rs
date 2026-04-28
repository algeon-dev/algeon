//! Service-level CUDA stream pooling for native worker tasks.

use std::{
    any::Any,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
#[cfg(test)]
use std::{cell::Cell, sync::Barrier};

#[cfg(test)]
use super::DeviceFacts;
use super::{ExecutionCancellation, ExecutionContext, context::QueryServiceResourceEpoch};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    observability::{
        RuntimeEventContext, RuntimeObservationHandle, RuntimeObservationPayload, RuntimeResourceDomain,
        RuntimeResourceId, RuntimeResourceRow, RuntimeStreamLeaseObservation, RuntimeStreamLeaseScope,
        RuntimeTopologyRecord, StreamLeaseAcquisition, StreamLeaseLifecycleState,
    },
};

/// Fallback and upper bound for the default service stream pool size.
///
/// The pool creates streams lazily, so this is a concurrency cap rather than a
/// startup allocation count. [`default_service_stream_pool_size_for_device`]
/// uses this value as the fallback when device SM count detection is
/// unavailable and as the upper bound when detection succeeds.
pub const DEFAULT_SERVICE_STREAM_POOL_SIZE: usize = 64;
const STREAM_ACQUIRE_CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);
const STREAM_CLEANUP_SUPERVISOR_POLL_INTERVAL: Duration = Duration::from_millis(10);
const STREAM_CLEANUP_SUPERVISOR_BUILD_OPERATION: &str = "nexus_query_engine.service_stream_pool.build";
static NEXT_STREAM_POOL_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_CLEANUP_SUPERVISOR_SPAWN: Cell<bool> = const { Cell::new(false) };
}

/// Returns the default service stream pool size for a CUDA device.
///
/// The default is `min(device_multiprocessor_count * 2, 64)`. If the
/// multiprocessor count cannot be detected, this returns
/// [`DEFAULT_SERVICE_STREAM_POOL_SIZE`]. The detected value is cached per
/// device ordinal for the process lifetime.
pub fn default_service_stream_pool_size_for_device(device: cudf::DeviceId) -> usize {
    static DEFAULT_SIZES: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    let sizes = DEFAULT_SIZES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut sizes = match sizes.lock() {
        Ok(sizes) => sizes,
        Err(_error) => {
            return default_service_stream_pool_size_from_sm_count(detect_device_multiprocessor_count(device));
        },
    };
    *sizes
        .entry(device.ordinal())
        .or_insert_with(|| default_service_stream_pool_size_from_sm_count(detect_device_multiprocessor_count(device)))
}

pub(super) fn default_service_stream_pool_size_from_sm_count(sm_count: Option<usize>) -> usize {
    sm_count
        .filter(|count| *count > 0)
        .and_then(|count| count.checked_mul(2))
        .map(|size| size.min(DEFAULT_SERVICE_STREAM_POOL_SIZE))
        .unwrap_or(DEFAULT_SERVICE_STREAM_POOL_SIZE)
}

#[cfg(test)]
pub(super) fn default_service_stream_pool_size_from_device_facts(facts: &DeviceFacts) -> usize {
    default_service_stream_pool_size_from_sm_count(facts.sm_count.and_then(nonzero_sm_count))
}

fn detect_device_multiprocessor_count(device: cudf::DeviceId) -> Option<usize> {
    device.multiprocessor_count().ok().and_then(nonzero_sm_count)
}

pub(super) fn nonzero_sm_count(count: u32) -> Option<usize> {
    (count > 0).then(|| usize::try_from(count).ok()).flatten()
}

#[derive(Debug)]
struct ServiceStreamPoolInner {
    id: u64,
    device: cudf::DeviceId,
    pool_size: usize,
    state: Mutex<ServiceStreamPoolState>,
    available: Condvar,
    state_changed: Condvar,
    next_lease_id: AtomicU64,
    cleanup_supervisor: StreamCleanupSupervisor,
    closed: AtomicBool,
}

#[derive(Debug, Default)]
struct ServiceStreamPoolState {
    idle: Vec<Arc<cudf::Runtime>>,
    created: usize,
    waiting_acquires: usize,
    pending_cleanups: usize,
    active_cleanups: usize,
    synchronization_failures: u64,
    #[cfg(test)]
    fail_next_synchronization: bool,
    #[cfg(test)]
    force_next_cleanup_enqueue_fallback: bool,
}

type StreamCleanupCallback = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct StreamCleanupCompletion {
    before_quiescence: Option<StreamCleanupCallback>,
    after_quiescence: Option<StreamCleanupCallback>,
}

impl StreamCleanupCompletion {
    fn new(
        before_quiescence: impl FnOnce() + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            before_quiescence: Some(Box::new(before_quiescence)),
            after_quiescence: Some(Box::new(after_quiescence)),
        }
    }

    fn run_before_quiescence(&mut self) {
        if let Some(completion) = self.before_quiescence.take() {
            completion();
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
            let _ = catch_unwind(AssertUnwindSafe(completion));
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
struct StreamCleanupSupervisor {
    device: cudf::DeviceId,
    pool_id: u64,
    sender: mpsc::Sender<StreamCleanupJob>,
    submission_gate: Mutex<()>,
    control: Arc<StreamCleanupSupervisorControl>,
    join: Mutex<Option<thread::JoinHandle<StreamCleanupWorkerOutcome>>>,
    join_disposition: AtomicU64,
}

const STREAM_CLEANUP_SUPERVISOR_NOT_JOINED: u64 = 0;
const STREAM_CLEANUP_SUPERVISOR_JOINED: u64 = 1;
const STREAM_CLEANUP_SUPERVISOR_JOIN_FAILED: u64 = 2;

#[derive(Debug, Default)]
struct StreamCleanupSupervisorControl {
    state: Mutex<StreamCleanupSupervisorState>,
    changed: Condvar,
    #[cfg(test)]
    panic_after_outcome: AtomicBool,
    #[cfg(test)]
    worker_outcome_pause: Mutex<Option<StreamCleanupWorkerOutcomePause>>,
    #[cfg(test)]
    shutdown_send_failed: Mutex<Option<mpsc::Sender<()>>>,
    #[cfg(test)]
    join_waiter: Mutex<Option<mpsc::Sender<()>>>,
}

#[derive(Debug, Default)]
struct StreamCleanupSupervisorState {
    shutdown_requested: bool,
    joiner_active: bool,
    terminal: Option<StreamCleanupSupervisorTerminal>,
}

#[derive(Debug, Clone)]
enum StreamCleanupWorkerOutcome {
    Clean,
    Panicked { panic_message: String },
    Unexpected { detail: &'static str },
}

#[derive(Debug, Clone)]
enum StreamCleanupSupervisorTerminal {
    Clean,
    Panicked { stage: &'static str, panic_message: String },
    Unexpected { detail: &'static str },
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct StreamCleanupWorkerOutcomePause {
    arrived: Arc<Barrier>,
    release: Arc<Barrier>,
}

enum StreamCleanupExecutorExit {
    Shutdown,
    ReceiverDisconnected,
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
    fn start(device: cudf::DeviceId, pool_id: u64) -> Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let control = Arc::new(StreamCleanupSupervisorControl::default());
        let thread_control = Arc::clone(&control);
        #[cfg(test)]
        let inject_spawn_failure = FAIL_NEXT_CLEANUP_SUPERVISOR_SPAWN.with(|failure| failure.replace(false));
        #[cfg(not(test))]
        let inject_spawn_failure = false;
        let join = if inject_spawn_failure {
            Err(std::io::Error::other("injected service stream cleanup supervisor spawn failure"))
        } else {
            thread::Builder::new()
                .name(format!("nexus-stream-cleanup-d{}-p{pool_id}", device.ordinal()))
                .spawn(move || run_stream_cleanup_worker(receiver, &thread_control))
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

        Ok(Self {
            device,
            pool_id,
            sender,
            submission_gate: Mutex::new(()),
            control,
            join: Mutex::new(Some(join)),
            join_disposition: AtomicU64::new(STREAM_CLEANUP_SUPERVISOR_NOT_JOINED),
        })
    }

    fn enqueue(&self, job: StreamCleanupJob) -> std::result::Result<(), mpsc::SendError<StreamCleanupJob>> {
        let _submission = self.submission_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let state = self.control.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.shutdown_requested || state.terminal.is_some() {
            return Err(mpsc::SendError(job));
        }
        // A sender failure only says that the receiver has gone away. The
        // worker's joined outcome remains the sole source of terminal state.
        self.sender.send(job)
    }

    fn request_shutdown(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let _submission = self.lock_submission_until(deadline, operation)?;
        let mut state = self.lock_control_until(deadline, operation)?;
        if state.shutdown_requested || state.terminal.is_some() {
            return Ok(());
        }
        state.shutdown_requested = true;
        let send_failed = self.sender.send(StreamCleanupJob::Shutdown).is_err();
        drop(state);
        if send_failed {
            #[cfg(test)]
            self.control.notify_shutdown_send_failed_for_test();
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

    fn ensure_not_failed(&self, operation: &'static str) -> Result<()> {
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
    fn finalize_if_finished_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
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
    fn wait_for_exit_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
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

            #[cfg(test)]
            self.control.notify_join_waiter_for_test();
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
        Ok(joined)
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

impl StreamCleanupSupervisorControl {
    #[cfg(test)]
    fn pause_before_worker_return_for_test(&self) {
        let pause = self.worker_outcome_pause.lock().unwrap().clone();
        if let Some(pause) = pause {
            pause.arrived.wait();
            pause.release.wait();
        }
    }

    #[cfg(test)]
    fn notify_shutdown_send_failed_for_test(&self) {
        if let Some(sender) = self.shutdown_send_failed.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }

    #[cfg(test)]
    fn notify_join_waiter_for_test(&self) {
        if let Some(sender) = self.join_waiter.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }
}

/// Service-level CUDA stream pool for short-lived native worker tasks.
///
/// The pool owns a bounded set of same-device cuDF runtimes. Each runtime
/// carries one CUDA stream and the process-local device memory manager. Streams
/// are created lazily up to `pool_size`; dropping a [`StreamLease`] returns the
/// runtime to the pool for reuse. This avoids creating a CUDA stream for every
/// bounded source-read worker while preserving the existing reservation-aware
/// allocator contract.
#[derive(Debug, Clone)]
pub(crate) struct ServiceStreamPool {
    inner: Arc<ServiceStreamPoolInner>,
}

impl ServiceStreamPool {
    /// Create a stream pool for `device`.
    ///
    /// `pool_size` is the maximum number of simultaneously leased streams.
    /// Runtime/stream creation is lazy: construction records the limit but does
    /// not allocate CUDA streams immediately.
    pub fn new(device: cudf::DeviceId, pool_size: usize) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        let id = NEXT_STREAM_POOL_ID.fetch_add(1, Ordering::Relaxed);
        // The cleanup receiver is a construction resource. Do not publish a
        // usable pool unless its owner thread was created successfully.
        let cleanup_supervisor = StreamCleanupSupervisor::start(device, id)?;
        let inner = Arc::new(ServiceStreamPoolInner {
            id,
            device,
            pool_size,
            state: Mutex::new(ServiceStreamPoolState::default()),
            available: Condvar::new(),
            state_changed: Condvar::new(),
            next_lease_id: AtomicU64::new(1),
            cleanup_supervisor,
            closed: AtomicBool::new(false),
        });
        Ok(Self { inner })
    }

    /// Return the process-global default pool for `device` and `pool_size`.
    ///
    /// This is the fallback used by native execution when a service has not
    /// injected its own pool yet. Pools are keyed by `(device ordinal,
    /// pool_size)` so differently sized service tests do not silently share a
    /// capacity contract.
    pub fn for_device(device: cudf::DeviceId, pool_size: usize) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        static POOLS: OnceLock<Mutex<HashMap<(usize, usize), ServiceStreamPool>>> = OnceLock::new();
        let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut pools = pools.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.service_stream_pool.registry",
                "service stream pool registry mutex was poisoned",
            )
        })?;
        let key = (device.ordinal(), pool_size);
        if let Some(pool) = pools.get(&key) {
            return Ok(pool.clone());
        }
        let pool = Self::new(device, pool_size)?;
        pools.insert(key, pool.clone());
        Ok(pool)
    }

    /// Return the process-global default-sized pool for `device`.
    pub fn default_for_device(device: cudf::DeviceId) -> Result<Self> {
        Self::for_device(device, default_service_stream_pool_size_for_device(device))
    }

    /// CUDA device served by this pool.
    pub fn device(&self) -> cudf::DeviceId {
        self.inner.device
    }

    /// Maximum number of simultaneously leased streams.
    pub fn pool_size(&self) -> usize {
        self.inner.pool_size
    }

    /// Wait until every runtime created by this pool has completed cleanup and
    /// returned to the idle set.
    ///
    /// This is a quiescence boundary for benchmark warmup and graceful service
    /// drain. It does not create runtimes that were never needed.
    pub(crate) fn wait_until_idle(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        self.wait_until_quiescent(0, deadline, operation)
    }

    /// Prevent new acquisitions and destroy every currently idle runtime.
    /// Leased runtimes are discarded after their deferred synchronization.
    pub(crate) fn close(&self, operation: &'static str) -> Result<()> {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.available.notify_all();
        let idle = match self.inner.state.try_lock() {
            Ok(mut state) => {
                let idle = std::mem::take(&mut state.idle);
                state.created = state.created.saturating_sub(idle.len());
                self.inner.state_changed.notify_all();
                idle
            },
            Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "service stream pool mutex was poisoned",
                ));
            },
        };
        drop(idle);
        Ok(())
    }

    /// Close this pool and wait until every lease and deferred cleanup has
    /// completed, then explicitly stop and join the cleanup supervisor, under
    /// the caller's absolute shutdown deadline.
    pub(crate) fn close_and_wait_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.available.notify_all();
        let mut state = self.lock_state_until(deadline, operation)?;
        let idle = std::mem::take(&mut state.idle);
        state.created = state.created.saturating_sub(idle.len());
        drop(state);
        drop(idle);
        self.inner.state_changed.notify_all();
        let mut state = self.lock_state_until(deadline, operation)?;
        loop {
            if let Err(error) = self.inner.cleanup_supervisor.ensure_not_failed(operation) {
                return Err(error);
            }
            if state.created == 0
                && state.idle.is_empty()
                && state.pending_cleanups == 0
                && state.active_cleanups == 0
                && state.waiting_acquires == 0
            {
                if Instant::now() >= deadline {
                    return Err(stream_pool_cleanup_timeout(operation, self, &state, 0));
                }
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(stream_pool_cleanup_timeout(operation, self, &state, 0));
            }
            let wait = remaining.min(STREAM_CLEANUP_SUPERVISOR_POLL_INTERVAL);
            let (next, timeout) = self.inner.state_changed.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "service stream pool mutex was poisoned while waiting for shutdown cleanup",
                )
            })?;
            state = next;
            if timeout.timed_out()
                && Instant::now() >= deadline
                && (state.created != 0
                    || !state.idle.is_empty()
                    || state.pending_cleanups != 0
                    || state.active_cleanups != 0
                    || state.waiting_acquires != 0)
            {
                return Err(stream_pool_cleanup_timeout(operation, self, &state, 0));
            }
        }
        drop(state);

        // A zero counter snapshot only proves that no work is currently
        // accounted. The supervisor is an owned resource in its own right, so
        // close must signal it and elect a joiner to commit its outcome.
        self.inner.cleanup_supervisor.request_shutdown(deadline, operation)?;
        self.inner.cleanup_supervisor.wait_for_exit_until(deadline, operation)
    }

    /// Wait until exactly `expected_leased_runtimes` created runtimes remain
    /// checked out and every other runtime has completed cleanup.
    pub(crate) fn wait_until_quiescent(
        &self,
        expected_leased_runtimes: usize,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<()> {
        if expected_leased_runtimes > self.pool_size() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "expected leased runtime count exceeds the service stream pool size",
            )
            .with_fact("device_ordinal", self.device().ordinal())
            .with_fact("stream_pool_size", self.pool_size())
            .with_fact("expected_leased_runtimes", expected_leased_runtimes)
            .with_fact("reason", "expected_leases_exceed_pool_size"));
        }

        let mut state = self.lock_state(operation)?;
        loop {
            self.inner.cleanup_supervisor.ensure_not_failed(operation)?;
            let active_leased_runtimes = state
                .created
                .saturating_sub(state.idle.len())
                .saturating_sub(state.pending_cleanups)
                .saturating_sub(state.active_cleanups);
            if active_leased_runtimes == expected_leased_runtimes
                && state.pending_cleanups == 0
                && state.active_cleanups == 0
            {
                drop(state);
                self.inner.cleanup_supervisor.finalize_if_finished_until(deadline, operation)?;
                return Ok(());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(stream_pool_cleanup_timeout(operation, self, &state, expected_leased_runtimes));
            }
            let wait = remaining.min(STREAM_CLEANUP_SUPERVISOR_POLL_INTERVAL);
            let (next, _timeout) = self.inner.state_changed.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "service stream pool mutex was poisoned while waiting for cleanup",
                )
            })?;
            state = next;
        }
    }

    #[cfg(test)]
    pub(crate) fn wait_until_acquire_queued_for_test(&self, deadline: Instant) {
        let mut state = self
            .lock_state("nexus_query_engine.service_stream_pool.test_wait_for_queued_acquire")
            .expect("stream pool test state");
        while state.waiting_acquires == 0 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "stream acquire did not enter the contended wait");
            let (next, timeout) =
                self.inner.state_changed.wait_timeout(state, remaining).expect("stream pool test wait");
            state = next;
            assert!(!timeout.timed_out() || state.waiting_acquires > 0, "stream acquire wait timed out");
        }
    }

    #[cfg(test)]
    fn fail_next_synchronization_for_test(&self) {
        self.inner.state.lock().unwrap().fail_next_synchronization = true;
    }

    #[cfg(test)]
    fn force_next_cleanup_enqueue_fallback_for_test(&self) {
        self.inner.state.lock().unwrap().force_next_cleanup_enqueue_fallback = true;
    }

    /// Acquire a CUDA stream/runtime lease, blocking when all pool slots are busy.
    pub fn acquire(&self) -> Result<StreamLease> {
        self.acquire_inner(
            None,
            None,
            "nexus_query_engine.service_stream_pool.acquire",
            RuntimeObservationHandle::disabled(),
            RuntimeStreamLeaseScope::default(),
        )
    }

    /// Acquire a CUDA stream/runtime lease while observing cooperative cancellation.
    pub fn acquire_with_cancellation(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<StreamLease> {
        self.acquire_inner(
            Some(cancellation),
            None,
            operation,
            RuntimeObservationHandle::disabled(),
            RuntimeStreamLeaseScope::default(),
        )
    }

    /// Acquire a CUDA stream/runtime lease while observing cancellation and an
    /// absolute wait deadline.
    pub fn acquire_with_cancellation_deadline(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<StreamLease> {
        self.acquire_inner(
            Some(cancellation),
            Some(deadline),
            operation,
            RuntimeObservationHandle::disabled(),
            RuntimeStreamLeaseScope::default(),
        )
    }

    /// Acquires a lease while recording host-side wait and lifecycle evidence.
    pub(crate) fn acquire_observed_with_cancellation_deadline(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
    ) -> Result<StreamLease> {
        self.acquire_inner(Some(cancellation), Some(deadline), operation, observation, scope)
    }

    pub(crate) fn acquire_observed_with_cancellation(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
    ) -> Result<StreamLease> {
        self.acquire_inner(Some(cancellation), None, operation, observation, scope)
    }

    /// Attempts to reserve an idle runtime for which `reserve` can also
    /// acquire a caller-owned companion reservation.
    ///
    /// The callback runs while the stream-pool mutex is held. It must remain
    /// host-only and must not call back into this stream pool. This is used by
    /// query admission to reserve an exact-runtime resident graph cache entry
    /// and its stream as one lifecycle; runtimes created after this call
    /// cannot own an existing cache entry and are intentionally not created.
    #[cfg(feature = "cugraph")]
    pub(crate) fn try_acquire_matching<T>(
        &self,
        operation: &'static str,
        mut reserve: impl FnMut(&Arc<cudf::Runtime>) -> Result<Option<T>>,
    ) -> Result<Option<(StreamLease, T)>> {
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(stream_pool_closed(operation, self));
        }
        let mut state = self.lock_state(operation)?;
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(stream_pool_closed(operation, self));
        }
        let health = cudf::runtime::device_health_snapshot(self.device());
        if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
            return Err(unhealthy_stream_device_error(operation, self.device(), health));
        }
        for index in (0..state.idle.len()).rev() {
            let runtime = &state.idle[index];
            let Some(reservation) = reserve(runtime)? else {
                continue;
            };
            let health = cudf::runtime::device_health_snapshot(self.device());
            if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                return Err(unhealthy_stream_device_error(operation, self.device(), health));
            }
            let runtime = state.idle.swap_remove(index);
            self.inner.state_changed.notify_all();
            return Ok(Some((
                StreamLease {
                    pool: self.clone(),
                    runtime: Some(runtime),
                    observation: None,
                    resource_epoch: None,
                    cleanup_completion: Mutex::new(None),
                },
                reservation,
            )));
        }
        Ok(None)
    }

    fn acquire_inner(
        &self,
        cancellation: Option<&ExecutionCancellation>,
        deadline: Option<Instant>,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
    ) -> Result<StreamLease> {
        enum AcquireAction {
            Reuse(Arc<cudf::Runtime>),
            Create,
        }

        let lease_observation = Arc::new(StreamLeaseObservation::new(self, observation, scope));
        lease_observation.define_resource();
        lease_observation.record(StreamLeaseLifecycleState::Enqueued);
        let enqueued_at = Instant::now();
        let action = loop {
            if self.inner.closed.load(Ordering::Acquire) {
                let error = stream_pool_closed(operation, self);
                lease_observation.terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                return Err(error);
            }
            if let Some(cancellation) = cancellation
                && let Err(error) = cancellation.check(operation)
            {
                lease_observation.record(StreamLeaseLifecycleState::CancellationRequested);
                lease_observation.terminal(StreamLeaseLifecycleState::Cancelled);
                return Err(error);
            }
            if let Some(deadline) = deadline
                && Instant::now() >= deadline
            {
                let error = stream_acquire_timeout_error(operation, self.device(), self.pool_size());
                lease_observation.terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                return Err(error);
            }
            let mut state = self.lock_state(operation)?;
            if self.inner.closed.load(Ordering::Acquire) {
                let error = stream_pool_closed(operation, self);
                lease_observation.terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                return Err(error);
            }
            let health = cudf::runtime::device_health_snapshot(self.device());
            if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                let error = unhealthy_stream_device_error(operation, self.device(), health);
                lease_observation.terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                return Err(error);
            }
            if let Some(runtime) = state.idle.pop() {
                self.inner.state_changed.notify_all();
                break AcquireAction::Reuse(runtime);
            }
            if state.created < self.pool_size() {
                state.created += 1;
                self.inner.state_changed.notify_all();
                break AcquireAction::Create;
            }
            if cancellation.is_some() || deadline.is_some() {
                let wait = stream_acquire_wait_interval(deadline);
                state.waiting_acquires = state.waiting_acquires.saturating_add(1);
                self.inner.state_changed.notify_all();
                let (mut state, _timeout) = self.inner.available.wait_timeout(state, wait).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        operation,
                        "service stream pool mutex was poisoned while waiting for a stream",
                    )
                })?;
                state.waiting_acquires = state.waiting_acquires.saturating_sub(1);
                self.inner.state_changed.notify_all();
            } else {
                state.waiting_acquires = state.waiting_acquires.saturating_add(1);
                self.inner.state_changed.notify_all();
                let mut state = self.inner.available.wait(state).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        operation,
                        "service stream pool mutex was poisoned while waiting for a stream",
                    )
                })?;
                state.waiting_acquires = state.waiting_acquires.saturating_sub(1);
                self.inner.state_changed.notify_all();
            }
        };

        let (runtime, acquisition) = match action {
            AcquireAction::Reuse(runtime) => (runtime, StreamLeaseAcquisition::Reused),
            AcquireAction::Create => match self.build_runtime() {
                Ok(runtime) if !self.inner.closed.load(Ordering::Acquire) => {
                    (Arc::new(runtime), StreamLeaseAcquisition::Created)
                },
                Ok(runtime) => {
                    self.discard_created_slot();
                    drop(runtime);
                    let error = stream_pool_closed(operation, self);
                    lease_observation
                        .terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                    return Err(error);
                },
                Err(error) => {
                    self.discard_created_slot();
                    lease_observation
                        .terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                    return Err(error);
                },
            },
        };
        lease_observation.record(StreamLeaseLifecycleState::Acquired {
            acquisition,
            host_wait_ns: u64::try_from(enqueued_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
        });
        Ok(StreamLease {
            pool: self.clone(),
            runtime: Some(runtime),
            observation: Some(lease_observation),
            resource_epoch: None,
            cleanup_completion: Mutex::new(None),
        })
    }

    fn build_runtime(&self) -> Result<cudf::Runtime> {
        cudf::Runtime::builder().with_device(self.device()).with_device_memory_manager().build().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.service_stream_pool.runtime",
                "failed to initialize service stream pool runtime",
                error,
            )
        })
    }

    fn lock_state(&self, operation: &'static str) -> Result<std::sync::MutexGuard<'_, ServiceStreamPoolState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "service stream pool mutex was poisoned")
        })
    }

    fn lock_state_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, ServiceStreamPoolState>> {
        loop {
            match self.inner.state.try_lock() {
                Ok(state) => return Ok(state),
                Err(std::sync::TryLockError::Poisoned(_)) => return self.lock_state(operation),
                Err(std::sync::TryLockError::WouldBlock) if Instant::now() >= deadline => {
                    return Err(QueryError::new(
                        QueryErrorCode::ServiceOverloaded,
                        operation,
                        "timed out acquiring service stream pool state during shutdown",
                    )
                    .with_fact("reason", "stream_pool_idle_timeout")
                    .with_fact("device_ordinal", self.device().ordinal()));
                },
                Err(std::sync::TryLockError::WouldBlock) => thread::yield_now(),
            }
        }
    }

    fn discard_created_slot(&self) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.created = state.created.saturating_sub(1);
                self.inner.available.notify_one();
                self.inner.state_changed.notify_all();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to update poisoned service stream pool state");
            },
        }
    }

    fn enqueue_cleanup(
        &self,
        runtime: Arc<cudf::Runtime>,
        resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
        mut completion: Option<StreamCleanupCompletion>,
    ) {
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
            let _ = runtime.synchronize();
            drop(runtime);
            drop(resource_epoch);
            if let Some(completion) = completion.as_mut() {
                completion.run_before_quiescence();
                completion.run_after_quiescence();
            }
            return;
        }
        let job =
            StreamCleanupJob::Runtime(DeferredStreamCleanup::new(self.clone(), runtime, resource_epoch, completion));
        #[cfg(test)]
        let force_enqueue_fallback = match self.inner.state.lock() {
            Ok(mut state) => std::mem::take(&mut state.force_next_cleanup_enqueue_fallback),
            Err(_) => false,
        };
        #[cfg(test)]
        if force_enqueue_fallback {
            match job {
                StreamCleanupJob::Runtime(mut cleanup) => cleanup.run(),
                StreamCleanupJob::Shutdown => unreachable!("runtime cleanup cannot enqueue supervisor shutdown"),
                StreamCleanupJob::Probe { .. }
                | StreamCleanupJob::Block { .. }
                | StreamCleanupJob::ExitUnexpectedly => {
                    unreachable!("new cleanup job is always a runtime cleanup")
                },
            }
            return;
        }
        if let Err(error) = self.inner.cleanup_supervisor.enqueue(job) {
            // The only production send failure is a terminal cleanup
            // supervisor. The synchronous fallback still releases the job's
            // permits, but the runtime cannot be returned to a pool whose
            // owner has disappeared.
            self.inner.closed.store(true, Ordering::Release);
            match error.0 {
                StreamCleanupJob::Runtime(mut cleanup) => cleanup.run(),
                StreamCleanupJob::Shutdown => unreachable!("runtime cleanup cannot enqueue supervisor shutdown"),
                #[cfg(test)]
                StreamCleanupJob::Probe { .. }
                | StreamCleanupJob::Block { .. }
                | StreamCleanupJob::ExitUnexpectedly => {},
            }
        }
    }
}

/// Owns one queued stream cleanup until both resource accounting and the
/// external completion handoff have finished. If a prior cleanup panics, the
/// receiver drops queued jobs; this guard turns that drop into fail-closed
/// resource cleanup rather than silently losing permits or completion work.
struct DeferredStreamCleanup {
    pool: ServiceStreamPool,
    runtime: Option<Arc<cudf::Runtime>>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
    completion: Option<StreamCleanupCompletion>,
    phase: DeferredStreamCleanupPhase,
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
    ) -> Self {
        Self { pool, runtime: Some(runtime), resource_epoch, completion, phase: DeferredStreamCleanupPhase::Pending }
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
            completion.run_before_quiescence();
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
        #[cfg(test)]
        let force_failure = match self.pool.inner.state.lock() {
            Ok(mut state) => std::mem::take(&mut state.fail_next_synchronization),
            Err(_) => false,
        };
        match self.runtime.as_ref().expect("deferred cleanup runtime is present").synchronize() {
            #[cfg(test)]
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
            completion.run_before_quiescence();
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

fn stream_acquire_wait_interval(deadline: Option<Instant>) -> Duration {
    let cancellation_poll = STREAM_ACQUIRE_CANCELLATION_POLL_INTERVAL;
    let Some(deadline) = deadline else {
        return cancellation_poll;
    };
    deadline.saturating_duration_since(Instant::now()).min(cancellation_poll)
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

fn stream_acquire_timeout_error(operation: &'static str, device: cudf::DeviceId, pool_size: usize) -> QueryError {
    QueryError::new(QueryErrorCode::ServiceOverloaded, operation, "service stream pool acquire timed out")
        .with_fact("reason", "stream_pool_wait_timeout")
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("stream_pool_size", pool_size)
}

fn unhealthy_stream_device_error(
    operation: &'static str,
    device: cudf::DeviceId,
    health: cudf::runtime::DeviceHealthSnapshot,
) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::CudfPersistentFailure,
        operation,
        "service stream pool device is unhealthy; runtime creation and reuse are rejected",
    )
    .with_fact("reason", "device_health_rejected_stream_acquire")
    .with_fact("device_ordinal", device.ordinal())
    .with_fact("device_fault_count", health.fault_count());
    if let Some(first_fault) = health.first_fault() {
        error = error
            .with_fact("first_device_fault_kind", format!("{:?}", first_fault.kind()))
            .with_fact("first_device_fault_operation", first_fault.operation());
    }
    error
}

fn stream_pool_closed(operation: &'static str, pool: &ServiceStreamPool) -> QueryError {
    let synchronization_failures = pool.inner.state.try_lock().ok().map(|state| state.synchronization_failures);
    let error = QueryError::new(QueryErrorCode::Cancelled, operation, "service stream pool is closed")
        .with_fact("reason", "stream_pool_closed")
        .with_fact("device_ordinal", pool.device().ordinal())
        .with_fact("stream_pool_size", pool.pool_size());
    match synchronization_failures {
        Some(failures) => error.with_fact("synchronization_failures", failures),
        None => error,
    }
}

fn stream_pool_cleanup_timeout(
    operation: &'static str,
    pool: &ServiceStreamPool,
    state: &ServiceStreamPoolState,
    expected_leased_runtimes: usize,
) -> QueryError {
    let active_leased_runtimes = state
        .created
        .saturating_sub(state.idle.len())
        .saturating_sub(state.pending_cleanups)
        .saturating_sub(state.active_cleanups);
    let reason =
        if expected_leased_runtimes == 0 { "stream_pool_idle_timeout" } else { "stream_pool_quiescence_timeout" };
    QueryError::new(QueryErrorCode::ServiceOverloaded, operation, "timed out waiting for service stream pool cleanup")
        .with_fact("device_ordinal", pool.device().ordinal())
        .with_fact("stream_pool_size", pool.pool_size())
        .with_fact("created_runtimes", state.created)
        .with_fact("idle_runtimes", state.idle.len())
        .with_fact("active_leased_runtimes", active_leased_runtimes)
        .with_fact("pending_cleanups", state.pending_cleanups)
        .with_fact("active_cleanups", state.active_cleanups)
        .with_fact("waiting_acquires", state.waiting_acquires)
        .with_fact("synchronization_failures", state.synchronization_failures)
        .with_fact("expected_leased_runtimes", expected_leased_runtimes)
        .with_fact("reason", reason)
        .with_cleanup_deadline_elapsed()
}

#[derive(Debug)]
struct StreamLeaseObservation {
    observation: RuntimeObservationHandle,
    lease_id: RuntimeResourceId,
    device_ordinal: u32,
    scope: RuntimeStreamLeaseScope,
    terminal_recorded: std::sync::atomic::AtomicBool,
}

impl StreamLeaseObservation {
    fn new(pool: &ServiceStreamPool, observation: RuntimeObservationHandle, scope: RuntimeStreamLeaseScope) -> Self {
        let sequence = pool.inner.next_lease_id.fetch_add(1, Ordering::Relaxed);
        Self {
            observation,
            lease_id: RuntimeResourceId::new(format!("stream-pool-{}-lease-{sequence}", pool.inner.id)),
            device_ordinal: u32::try_from(pool.device().ordinal()).unwrap_or(u32::MAX),
            scope,
            terminal_recorded: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn record(&self, lifecycle: StreamLeaseLifecycleState) {
        let context = self
            .scope
            .plan_execution_id
            .clone()
            .map_or_else(RuntimeEventContext::default, RuntimeEventContext::for_plan);
        self.observation.record_with_context(context, || {
            RuntimeObservationPayload::StreamLease(Box::new(RuntimeStreamLeaseObservation {
                lease_id: self.lease_id.clone(),
                device_ordinal: self.device_ordinal,
                scope: self.scope.clone(),
                lifecycle,
            }))
        });
    }

    fn define_resource(&self) {
        self.observation.record_topology(crate::observability::RuntimeObservationRecordClass::Resource, || {
            RuntimeTopologyRecord::Resource(RuntimeResourceRow {
                plan_execution_id: self.scope.plan_execution_id.clone(),
                resource_id: self.lease_id.clone(),
                resource_kind: "stream_lease".to_string(),
                domain: RuntimeResourceDomain::Device { device_ordinal: self.device_ordinal },
                producer_task_id: None,
                consumer_task_ids: Vec::new(),
                lifetime: "lease".to_string(),
            })
        });
    }

    fn terminal(&self, lifecycle: StreamLeaseLifecycleState) {
        if !self.terminal_recorded.swap(true, Ordering::AcqRel) {
            self.record(lifecycle);
        }
    }
}

/// RAII handle for a runtime borrowed from the backend-owned stream pool.
///
/// Dropping the lease synchronously records that query ownership was released,
/// then schedules runtime stream synchronization on the service cleanup thread.
/// The runtime returns to the pool only after synchronization completes. This
/// keeps source-read output safe to consume on another runtime before the stream
/// is reused by an unrelated worker without running CUDA synchronization on the
/// dropping thread.
pub struct StreamLease {
    pool: ServiceStreamPool,
    runtime: Option<Arc<cudf::Runtime>>,
    observation: Option<Arc<StreamLeaseObservation>>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
    cleanup_completion: Mutex<Option<StreamCleanupCompletion>>,
}

impl std::fmt::Debug for StreamLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamLease")
            .field("device_ordinal", &self.pool.device().ordinal())
            .field("has_runtime", &self.runtime.is_some())
            .field("has_resource_epoch", &self.resource_epoch.is_some())
            .field(
                "has_cleanup_completion",
                &self.cleanup_completion.lock().map(|completion| completion.is_some()).unwrap_or(true),
            )
            .finish()
    }
}

impl StreamLease {
    pub(super) fn pool(&self) -> &ServiceStreamPool {
        &self.pool
    }

    pub(crate) fn with_resource_epoch(mut self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        self.resource_epoch = Some(resource_epoch);
        self
    }

    pub(crate) fn install_cleanup_completion(
        &self,
        before_quiescence: impl FnOnce() + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> bool {
        let mut slot = match self.cleanup_completion.lock() {
            Ok(slot) => slot,
            Err(error) => error.into_inner(),
        };
        if slot.is_some() {
            return false;
        }
        *slot = Some(StreamCleanupCompletion::new(before_quiescence, after_quiescence));
        true
    }

    pub(crate) fn device(&self) -> cudf::DeviceId {
        self.runtime.as_ref().expect("stream lease runtime is present").device()
    }

    /// Clone a counted cuDF runtime derived from this lease.
    pub fn runtime(&self) -> Result<cudf::Runtime> {
        let runtime = self.runtime.as_ref().expect("stream lease runtime is present").as_ref().clone();
        let Some(resource_epoch) = &self.resource_epoch else {
            return Ok(runtime);
        };
        resource_epoch.check_open("nexus_query_engine.service_stream_pool.lease.runtime")?;
        let retained: Arc<dyn Send + Sync> = resource_epoch.clone();
        runtime.with_retained_lifetime(retained).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.service_stream_pool.lease.runtime",
                "failed to retain query-service lifetime for leased runtime",
                error,
            )
        })
    }

    /// Clone the shared runtime handle for a short-lived execution context.
    pub fn shared_runtime(&self) -> Result<Arc<cudf::Runtime>> {
        self.runtime().map(Arc::new)
    }
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        if let Some(observation) = self.observation.take() {
            observation.terminal(StreamLeaseLifecycleState::Released);
        }
        let cleanup_completion = match self.cleanup_completion.get_mut() {
            Ok(completion) => completion.take(),
            Err(error) => error.into_inner().take(),
        };
        self.pool.enqueue_cleanup(runtime, self.resource_epoch.take(), cleanup_completion);
    }
}

enum StreamCleanupJob {
    Runtime(DeferredStreamCleanup),
    Shutdown,
    #[cfg(test)]
    Probe {
        done: mpsc::Sender<thread::ThreadId>,
    },
    #[cfg(test)]
    Block {
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    },
    #[cfg(test)]
    ExitUnexpectedly,
}

fn run_stream_cleanup_worker(
    receiver: mpsc::Receiver<StreamCleanupJob>,
    _control: &StreamCleanupSupervisorControl,
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

    #[cfg(test)]
    _control.pause_before_worker_return_for_test();
    #[cfg(test)]
    if _control.panic_after_outcome.load(Ordering::Acquire) {
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
            StreamCleanupJob::Probe { done } => {
                let _ = done.send(thread::current().id());
            },
            #[cfg(test)]
            StreamCleanupJob::Block { entered, release } => {
                entered.wait();
                release.wait();
            },
            #[cfg(test)]
            StreamCleanupJob::ExitUnexpectedly => {
                return StreamCleanupExecutorExit::Unexpected("test requested unexpected cleanup supervisor exit");
            },
        }
    }
}

#[cfg(test)]
pub(crate) struct StreamCleanupBlocker {
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
pub(crate) fn block_stream_cleanup_thread_for_test(pool: &ServiceStreamPool) -> StreamCleanupBlocker {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    pool.inner
        .cleanup_supervisor
        .enqueue(StreamCleanupJob::Block { entered: Arc::clone(&entered), release: Arc::clone(&release) })
        .expect("cleanup thread accepts blocker");
    entered.wait();
    StreamCleanupBlocker { release: Some(release) }
}

#[cfg(test)]
fn stream_cleanup_thread_id_for_test(pool: &ServiceStreamPool) -> thread::ThreadId {
    let (sender, receiver) = mpsc::channel();
    pool.inner
        .cleanup_supervisor
        .enqueue(StreamCleanupJob::Probe { done: sender })
        .expect("cleanup thread accepts probe");
    receiver.recv().expect("cleanup thread returns probe")
}

#[cfg(test)]
fn fail_next_cleanup_supervisor_spawn_for_test() {
    FAIL_NEXT_CLEANUP_SUPERVISOR_SPAWN.with(|failure| failure.set(true));
}

#[cfg(test)]
fn panic_cleanup_supervisor_after_outcome_for_test(pool: &ServiceStreamPool) {
    pool.inner.cleanup_supervisor.control.panic_after_outcome.store(true, Ordering::Release);
}

#[cfg(test)]
fn pause_cleanup_supervisor_before_worker_return_for_test(pool: &ServiceStreamPool) -> StreamCleanupWorkerOutcomePause {
    let pause =
        StreamCleanupWorkerOutcomePause { arrived: Arc::new(Barrier::new(2)), release: Arc::new(Barrier::new(2)) };
    *pool.inner.cleanup_supervisor.control.worker_outcome_pause.lock().unwrap() = Some(pause.clone());
    pause
}

#[cfg(test)]
fn wait_for_cleanup_supervisor_worker_finished_for_test(pool: &ServiceStreamPool, deadline: Instant) {
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
fn observe_cleanup_supervisor_shutdown_send_failure_for_test(pool: &ServiceStreamPool) -> mpsc::Receiver<()> {
    let (sender, receiver) = mpsc::channel();
    *pool.inner.cleanup_supervisor.control.shutdown_send_failed.lock().unwrap() = Some(sender);
    receiver
}

#[cfg(test)]
fn observe_cleanup_supervisor_join_waiter_for_test(pool: &ServiceStreamPool) -> mpsc::Receiver<()> {
    let (sender, receiver) = mpsc::channel();
    *pool.inner.cleanup_supervisor.control.join_waiter.lock().unwrap() = Some(sender);
    receiver
}

#[cfg(test)]
fn force_cleanup_supervisor_unexpected_exit_for_test(pool: &ServiceStreamPool) {
    pool.inner
        .cleanup_supervisor
        .enqueue(StreamCleanupJob::ExitUnexpectedly)
        .expect("cleanup supervisor accepts unexpected-exit test job");
}

pub(crate) struct SourceReadStreamLease {
    context: ExecutionContext,
    _stream_lease: StreamLease,
}

impl SourceReadStreamLease {
    pub(super) fn new(context: ExecutionContext, stream_lease: StreamLease) -> Self {
        Self { context, _stream_lease: stream_lease }
    }

    pub(crate) fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

pub(super) fn validate_service_stream_pool_size(pool_size: usize) -> Result<()> {
    if pool_size == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.service_stream_pool.validate",
            "service stream pool size must be greater than zero",
        )
        .with_fact("field", "service_stream_pool_size")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_capacity_rejects_before_runtime_initialization() {
        let error = validate_service_stream_pool_size(0).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.operation(), "nexus_query_engine.service_stream_pool.validate");
        assert_eq!(error.fact_value("field"), Some("service_stream_pool_size"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    fn observed_stream_states(owner: &crate::observability::RuntimeObservationOwner) -> Vec<StreamLeaseLifecycleState> {
        owner
            .live_snapshot()
            .expect("observation remains live")
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::StreamLease(observation) => Some(observation.lifecycle.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn stream_cleanup_jobs_run_on_background_thread() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        assert_ne!(stream_cleanup_thread_id_for_test(&pool), thread::current().id());
    }

    #[test]
    fn cleanup_supervisor_spawn_failure_rejects_pool_construction() {
        fail_next_cleanup_supervisor_spawn_for_test();

        let error = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.status(), crate::error::QueryErrorStatus::Permanent);
        assert_eq!(error.operation(), STREAM_CLEANUP_SUPERVISOR_BUILD_OPERATION);
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_spawn_failed"));
        assert_eq!(error.fact_value("device_ordinal"), Some("0"));
    }

    #[test]
    fn close_and_wait_explicitly_joins_cleanup_supervisor_for_all_clones() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let clone = pool.clone();
        let deadline = Instant::now() + Duration::from_secs(5);

        pool.close_and_wait_until(deadline, "nexus_query_engine.service_stream_pool.test_join").unwrap();
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );

        clone.close_and_wait_until(deadline, "nexus_query_engine.service_stream_pool.test_join").unwrap();
        let error = clone.acquire().unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("stream_pool_closed"));
    }

    #[test]
    fn busy_cleanup_supervisor_times_out_then_joins_after_late_cleanup() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let blocker = block_stream_cleanup_thread_for_test(&pool);
        let lease = pool.acquire().unwrap();
        drop(lease);

        let timeout = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_millis(10),
                "nexus_query_engine.service_stream_pool.test_busy_cleanup",
            )
            .unwrap_err();
        assert!(timeout.is_cleanup_deadline_elapsed());
        assert_eq!(timeout.fact_value("reason"), Some("stream_pool_idle_timeout"));
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_NOT_JOINED,
        );

        drop(blocker);
        pool.close_and_wait_until(
            Instant::now() + Duration::from_secs(5),
            "nexus_query_engine.service_stream_pool.test_busy_cleanup",
        )
        .unwrap();
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn unexpected_cleanup_supervisor_exit_is_permanent_and_joined() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        force_cleanup_supervisor_unexpected_exit_for_test(&pool);
        let publication = pool
            .inner
            .cleanup_supervisor
            .wait_for_exit_until(
                Instant::now() + Duration::from_secs(5),
                "nexus_query_engine.service_stream_pool.test_unexpected_exit_publication",
            )
            .unwrap_err();
        assert_eq!(publication.fact_value("reason"), Some("stream_cleanup_supervisor_unexpected_exit"));

        let error = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "nexus_query_engine.service_stream_pool.test_unexpected_exit",
            )
            .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.status(), crate::error::QueryErrorStatus::Permanent);
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_unexpected_exit"));
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn unexpected_cleanup_supervisor_exit_recovers_queued_completion_payloads() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 2).unwrap();
        let cleanup_thread = stream_cleanup_thread_id_for_test(&pool);
        let blocker = block_stream_cleanup_thread_for_test(&pool);
        // Queue the forced exit before the leases so the receiver's teardown
        // drops both queued jobs and exercises their Drop recovery protocol.
        force_cleanup_supervisor_unexpected_exit_for_test(&pool);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        let (before_sender, before_receiver) = mpsc::channel();
        let (after_sender, after_receiver) = mpsc::channel();

        assert!(first.install_cleanup_completion(
            {
                let before_sender = before_sender.clone();
                move || before_sender.send(("first", thread::current().id())).unwrap()
            },
            {
                let after_sender = after_sender.clone();
                move || after_sender.send(("first", thread::current().id())).unwrap()
            },
        ));
        assert!(second.install_cleanup_completion(
            move || before_sender.send(("second", thread::current().id())).unwrap(),
            move || after_sender.send(("second", thread::current().id())).unwrap(),
        ));
        drop(first);
        drop(second);
        drop(blocker);

        let error = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "nexus_query_engine.service_stream_pool.test_exit_with_payload",
            )
            .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_unexpected_exit"));
        let mut before = vec![
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("first recovery before callback"),
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("second recovery before callback"),
        ];
        before.sort_by_key(|(label, _)| *label);
        let mut after = vec![
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("first recovery after callback"),
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("second recovery after callback"),
        ];
        after.sort_by_key(|(label, _)| *label);
        assert_eq!(before.iter().map(|(label, _)| *label).collect::<Vec<_>>(), vec!["first", "second"]);
        assert_eq!(after.iter().map(|(label, _)| *label).collect::<Vec<_>>(), vec!["first", "second"]);
        assert!(before.iter().all(|(_, thread_id)| *thread_id == cleanup_thread));
        assert!(after.iter().all(|(_, thread_id)| *thread_id == cleanup_thread));
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn cleanup_supervisor_panic_recovers_inflight_completion_payload() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let cleanup_thread = stream_cleanup_thread_id_for_test(&pool);
        let lease = pool.acquire().unwrap();
        let (before_sender, before_receiver) = mpsc::channel();
        let (after_sender, after_receiver) = mpsc::channel();
        assert!(lease.install_cleanup_completion(
            move || {
                before_sender.send(thread::current().id()).unwrap();
                panic!("injected in-flight cleanup completion panic");
            },
            move || after_sender.send(thread::current().id()).unwrap(),
        ));
        drop(lease);

        let error = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "nexus_query_engine.service_stream_pool.test_inflight_payload",
            )
            .unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_panicked"));
        assert_eq!(error.fact_value("supervisor_stage"), Some("cleanup_executor"));
        assert_eq!(
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("in-flight before callback"),
            cleanup_thread
        );
        assert_eq!(
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("in-flight after callback"),
            cleanup_thread
        );
        let state = pool.inner.state.lock().unwrap();
        assert_eq!(state.pending_cleanups, 0);
        assert_eq!(state.active_cleanups, 0);
        assert_eq!(state.created, 0);
    }

    #[test]
    fn wait_until_idle_commits_an_already_finished_cleanup_panic() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let pause = pause_cleanup_supervisor_before_worker_return_for_test(&pool);
        let lease = pool.acquire().unwrap();
        assert!(lease.install_cleanup_completion(|| panic!("injected cleanup panic before idle boundary"), || {},));
        drop(lease);

        pause.arrived.wait();
        pause.release.wait();
        let deadline = Instant::now() + Duration::from_secs(5);
        wait_for_cleanup_supervisor_worker_finished_for_test(&pool, deadline);

        let error = pool
            .wait_until_idle(deadline, "nexus_query_engine.service_stream_pool.test_idle_finished_panic")
            .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_panicked"));
        assert_eq!(error.fact_value("supervisor_stage"), Some("cleanup_executor"));
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn wait_until_quiescent_commits_an_already_finished_cleanup_panic() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let pause = pause_cleanup_supervisor_before_worker_return_for_test(&pool);
        let lease = pool.acquire().unwrap();
        assert!(
            lease.install_cleanup_completion(|| panic!("injected cleanup panic before quiescence boundary"), || {},)
        );
        drop(lease);

        pause.arrived.wait();
        pause.release.wait();
        let deadline = Instant::now() + Duration::from_secs(5);
        wait_for_cleanup_supervisor_worker_finished_for_test(&pool, deadline);

        let error = pool
            .wait_until_quiescent(0, deadline, "nexus_query_engine.service_stream_pool.test_quiescent_finished_panic")
            .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_panicked"));
        assert_eq!(error.fact_value("supervisor_stage"), Some("cleanup_executor"));
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn cleanup_supervisor_send_failure_does_not_mask_panicked_worker_outcome() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let cleanup_thread = stream_cleanup_thread_id_for_test(&pool);
        let pause = pause_cleanup_supervisor_before_worker_return_for_test(&pool);
        let shutdown_send_failed = observe_cleanup_supervisor_shutdown_send_failure_for_test(&pool);
        let lease = pool.acquire().unwrap();
        let (before_sender, before_receiver) = mpsc::channel();
        let (after_sender, after_receiver) = mpsc::channel();
        assert!(lease.install_cleanup_completion(
            move || {
                before_sender.send(thread::current().id()).unwrap();
                panic!("injected cleanup panic before shutdown send failure");
            },
            move || after_sender.send(thread::current().id()).unwrap(),
        ));
        drop(lease);

        // The worker has caught the callback panic and dropped its receiver,
        // including all queued-job recovery, but has not returned the outcome
        // to the join handle yet.
        pause.arrived.wait();
        let closer_pool = pool.clone();
        let closer = thread::spawn(move || {
            closer_pool.close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "nexus_query_engine.service_stream_pool.test_send_failure_panic",
            )
        });
        shutdown_send_failed
            .recv_timeout(Duration::from_secs(5))
            .expect("close observes the disconnected cleanup receiver");
        {
            let state = pool.inner.cleanup_supervisor.control.state.lock().unwrap();
            assert!(state.shutdown_requested);
            assert!(state.terminal.is_none(), "send failure cannot commit a terminal outcome");
        }

        pause.release.wait();
        let error = closer.join().expect("close thread joins").unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_panicked"));
        assert_eq!(error.fact_value("supervisor_stage"), Some("cleanup_executor"));
        assert_eq!(
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("in-flight before callback"),
            cleanup_thread
        );
        assert_eq!(
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("in-flight after callback"),
            cleanup_thread
        );
        let state = pool.inner.state.lock().unwrap();
        assert_eq!(state.pending_cleanups, 0);
        assert_eq!(state.active_cleanups, 0);
        assert_eq!(state.created, 0);
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn cleanup_supervisor_elects_one_joiner_for_concurrent_close_callers() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let blocker = block_stream_cleanup_thread_for_test(&pool);
        let join_waiter = observe_cleanup_supervisor_join_waiter_for_test(&pool);
        let first_pool = pool.clone();
        let first = thread::spawn(move || {
            first_pool.close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "nexus_query_engine.service_stream_pool.test_concurrent_joiner_first",
            )
        });
        let second_pool = pool.clone();
        let second = thread::spawn(move || {
            second_pool.close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "nexus_query_engine.service_stream_pool.test_concurrent_joiner_second",
            )
        });

        join_waiter.recv_timeout(Duration::from_secs(5)).expect("second close caller waits for the elected joiner");
        drop(blocker);
        first.join().expect("first close thread joins").expect("first close succeeds");
        second.join().expect("second close thread joins").expect("second close succeeds");
        let state = pool.inner.cleanup_supervisor.control.state.lock().unwrap();
        assert!(matches!(state.terminal, Some(StreamCleanupSupervisorTerminal::Clean)));
        assert!(!state.joiner_active);
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn cleanup_supervisor_precommit_join_timeout_is_retryable() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let blocker = block_stream_cleanup_thread_for_test(&pool);

        let timeout = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_millis(10),
                "nexus_query_engine.service_stream_pool.test_precommit_join_timeout",
            )
            .unwrap_err();
        assert!(timeout.is_cleanup_deadline_elapsed());
        assert_eq!(timeout.fact_value("reason"), Some("stream_cleanup_supervisor_join_timeout"));
        {
            let state = pool.inner.cleanup_supervisor.control.state.lock().unwrap();
            assert!(state.terminal.is_none(), "a deadline before join cannot commit a terminal outcome");
            assert!(!state.joiner_active, "a later close caller must be able to retry the join");
        }
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_NOT_JOINED,
        );

        drop(blocker);
        pool.close_and_wait_until(
            Instant::now() + Duration::from_secs(5),
            "nexus_query_engine.service_stream_pool.test_precommit_join_timeout",
        )
        .unwrap();
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );
    }

    #[test]
    fn cleanup_supervisor_join_panic_supersedes_provisional_clean_outcome() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        panic_cleanup_supervisor_after_outcome_for_test(&pool);
        let deadline = Instant::now() + Duration::from_secs(5);

        let error =
            pool.close_and_wait_until(deadline, "nexus_query_engine.service_stream_pool.test_join_panic").unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.status(), crate::error::QueryErrorStatus::Permanent);
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_panicked"));
        assert_eq!(error.fact_value("supervisor_stage"), Some("thread_join"));

        let repeated =
            pool.close_and_wait_until(deadline, "nexus_query_engine.service_stream_pool.test_join_panic").unwrap_err();
        assert_eq!(repeated.fact_value("reason"), Some("stream_cleanup_supervisor_panicked"));
        assert_eq!(repeated.fact_value("supervisor_stage"), Some("thread_join"));
    }

    #[test]
    fn cleanup_phase_callbacks_straddle_quiescence_accounting() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let lease = pool.acquire().unwrap();
        let before_pool = pool.clone();
        let after_pool = pool.clone();
        let (before_sender, before_receiver) = mpsc::channel();
        let (after_sender, after_receiver) = mpsc::channel();

        assert!(lease.install_cleanup_completion(
            move || {
                let active_cleanups = before_pool.inner.state.lock().unwrap().active_cleanups;
                before_sender.send((active_cleanups, thread::current().id())).unwrap();
            },
            move || {
                let active_cleanups = after_pool.inner.state.lock().unwrap().active_cleanups;
                after_sender.send((active_cleanups, thread::current().id())).unwrap();
            },
        ));
        drop(lease);

        let (before_active, before_thread) =
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("internal cleanup callback runs");
        let (after_active, after_thread) =
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("external completion callback runs");
        assert_eq!(before_active, 1, "resource teardown remains inside cleanup accounting");
        assert_eq!(after_active, 0, "external completion runs after cleanup quiescence is published");
        assert_eq!(before_thread, after_thread, "both phases run on the pre-created cleanup supervisor");
    }

    #[test]
    fn synchronous_cleanup_enqueue_fallback_dispatches_external_completion() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let lease = pool.acquire().unwrap();
        let dropping_thread = thread::current().id();
        let before_pool = pool.clone();
        let after_pool = pool.clone();
        let (before_sender, before_receiver) = mpsc::channel();
        let (after_sender, after_receiver) = mpsc::channel();

        assert!(lease.install_cleanup_completion(
            move || {
                let active_cleanups = before_pool.inner.state.lock().unwrap().active_cleanups;
                before_sender.send((active_cleanups, thread::current().id())).unwrap();
            },
            move || {
                let active_cleanups = after_pool.inner.state.lock().unwrap().active_cleanups;
                after_sender.send((active_cleanups, thread::current().id())).unwrap();
            },
        ));
        pool.force_next_cleanup_enqueue_fallback_for_test();
        drop(lease);

        let (before_active, before_thread) =
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("fallback internal cleanup callback runs");
        let (after_active, after_thread) =
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("fallback external completion callback runs");
        assert_eq!(before_active, 1, "fallback resource teardown remains inside cleanup accounting");
        assert_eq!(before_thread, dropping_thread, "fallback synchronizes resources on the dropping thread");
        assert_eq!(after_active, 0, "fallback external completion runs after cleanup quiescence is published");
        assert_eq!(after_thread, dropping_thread, "fallback transfers completion on its synchronous cleanup path");
    }

    #[test]
    fn synchronization_failure_discards_runtime_and_quarantines_pool() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let lease = pool.acquire().unwrap();
        pool.fail_next_synchronization_for_test();
        drop(lease);
        pool.wait_until_idle(
            Instant::now() + Duration::from_secs(5),
            "nexus_query_engine.service_stream_pool.test_sync_failure",
        )
        .unwrap();

        let error = pool.acquire().expect_err("synchronization failure closes the affected pool");
        assert_eq!(error.fact_value("reason"), Some("stream_pool_closed"));
        assert_eq!(error.fact_value("synchronization_failures"), Some("1"));
        let state = pool.inner.state.lock().unwrap();
        assert_eq!(state.created, 0);
        assert!(state.idle.is_empty());
    }

    #[test]
    fn unhealthy_device_rejects_idle_runtime_reuse() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let lease = pool.acquire().unwrap();
        drop(lease);
        let _ = stream_cleanup_thread_id_for_test(&pool);
        let _scope = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
        cudf::runtime::report_device_fault(
            cudf::DeviceId::DEFAULT,
            cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
            "injected_stream_reuse_fault",
            Some(cudf::runtime::DeviceFaultNativeStatus::Cudf(cudf_sys::CUDF_ERROR_FATAL_CUDA)),
        );

        let error = pool.acquire().expect_err("unhealthy device must reject idle runtime reuse");
        assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
        assert_eq!(error.fact_value("reason"), Some("device_health_rejected_stream_acquire"));
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn unhealthy_device_rejects_matching_before_warm_reservation_callback() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let lease = pool.acquire().unwrap();
        drop(lease);
        let _ = stream_cleanup_thread_id_for_test(&pool);
        let _scope = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
        cudf::runtime::report_device_fault(
            cudf::DeviceId::DEFAULT,
            cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
            "injected_pre_reservation_fault",
            Some(cudf::runtime::DeviceFaultNativeStatus::Cudf(cudf_sys::CUDF_ERROR_FATAL_CUDA)),
        );
        let mut reserve_calls = 0;

        let error = pool
            .try_acquire_matching("nexus_query_engine.service_stream_pool.test_pre_reservation_fault", |_runtime| {
                reserve_calls += 1;
                Ok(Some(()))
            })
            .expect_err("unhealthy device must reject before invoking warm reservation");

        assert_eq!(reserve_calls, 0);
        assert_eq!(error.fact_value("reason"), Some("device_health_rejected_stream_acquire"));
        let state = pool.inner.state.lock().unwrap();
        assert_eq!(state.created, 1);
        assert_eq!(state.idle.len(), 1);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn warm_resident_graph_reservation_fault_race_keeps_idle_runtime_in_pool() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let lease = pool.acquire().unwrap();
        drop(lease);
        let _ = stream_cleanup_thread_id_for_test(&pool);
        let scope = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
        let mut reserve_calls = 0;

        let error = pool
            .try_acquire_matching("nexus_query_engine.service_stream_pool.test_warm_fault_race", |_runtime| {
                reserve_calls += 1;
                cudf::runtime::report_device_fault(
                    cudf::DeviceId::DEFAULT,
                    cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
                    "injected_warm_resident_graph_reservation_fault",
                    Some(cudf::runtime::DeviceFaultNativeStatus::Cudf(cudf_sys::CUDF_ERROR_FATAL_CUDA)),
                );
                Ok(Some(()))
            })
            .expect_err("fault racing a warm reservation must reject idle runtime reuse");

        assert_eq!(reserve_calls, 1);
        assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
        assert_eq!(error.fact_value("reason"), Some("device_health_rejected_stream_acquire"));
        let state = pool.inner.state.lock().unwrap();
        assert_eq!(state.created, 1);
        assert_eq!(state.idle.len(), 1);
        drop(state);
        assert_eq!(scope.health_snapshot(cudf::DeviceId::DEFAULT).fault_count(), 1);
    }

    #[test]
    fn observed_stream_acquire_distinguishes_create_reuse_and_release() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "stream-lifecycle",
            None,
            crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
        );
        let cancellation = ExecutionCancellation::new();

        let created = pool
            .acquire_observed_with_cancellation(
                &cancellation,
                "nexus_query_engine.service_stream_pool.test_observed_create",
                owner.handle(),
                RuntimeStreamLeaseScope::default(),
            )
            .unwrap();
        drop(created);
        let _ = stream_cleanup_thread_id_for_test(&pool);
        let reused = pool
            .acquire_observed_with_cancellation(
                &cancellation,
                "nexus_query_engine.service_stream_pool.test_observed_reuse",
                owner.handle(),
                RuntimeStreamLeaseScope::default(),
            )
            .unwrap();
        drop(reused);
        let _ = stream_cleanup_thread_id_for_test(&pool);

        let states = observed_stream_states(&owner);
        crate::observability::assert_live_topology_references_resolve(&owner.live_snapshot().unwrap());
        assert_eq!(states.len(), 6);
        assert!(matches!(states[0], StreamLeaseLifecycleState::Enqueued));
        assert!(matches!(
            states[1],
            StreamLeaseLifecycleState::Acquired { acquisition: StreamLeaseAcquisition::Created, .. }
        ));
        assert!(matches!(states[2], StreamLeaseLifecycleState::Released));
        assert!(matches!(states[3], StreamLeaseLifecycleState::Enqueued));
        assert!(matches!(
            states[4],
            StreamLeaseLifecycleState::Acquired { acquisition: StreamLeaseAcquisition::Reused, .. }
        ));
        assert!(matches!(states[5], StreamLeaseLifecycleState::Released));
    }

    #[test]
    fn observed_stream_wait_cancellation_has_one_terminal_without_sleep() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let held = pool.acquire().unwrap();
        let cancellation = ExecutionCancellation::new();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "stream-cancel",
            None,
            crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
        );
        let waiter_pool = pool.clone();
        let waiter_cancellation = cancellation.clone();
        let observation = owner.handle();
        let (started_sender, started_receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            started_sender.send(()).unwrap();
            waiter_pool.acquire_observed_with_cancellation_deadline(
                &waiter_cancellation,
                Instant::now() + Duration::from_secs(1),
                "nexus_query_engine.service_stream_pool.test_observed_cancel",
                observation,
                RuntimeStreamLeaseScope::for_plan(crate::observability::PlanExecutionId::new("stream-plan"))
                    .with_source_worker(
                        crate::observability::RuntimeTaskId::new("source-task"),
                        crate::observability::RuntimeWorkerId::new("source-worker"),
                    )
                    .with_gpu_worker(
                        crate::observability::RuntimeJobId::new("gpu-job"),
                        crate::observability::RuntimeWorkerId::new("gpu-worker"),
                    ),
            )
        });
        started_receiver.recv().unwrap();
        pool.wait_until_acquire_queued_for_test(Instant::now() + Duration::from_secs(5));
        cancellation.cancel();

        let error = waiter.join().unwrap().unwrap_err();
        drop(held);
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        let snapshot = owner.live_snapshot().unwrap();
        let event = snapshot
            .events()
            .iter()
            .find(|event| matches!(event.payload, RuntimeObservationPayload::StreamLease(_)))
            .unwrap();
        assert_eq!(event.plan_execution_id.as_ref().map(|id| id.as_str()), Some("stream-plan"));
        let RuntimeObservationPayload::StreamLease(observation) = &event.payload else { unreachable!() };
        assert_eq!(observation.scope.task_id.as_ref().map(|id| id.as_str()), Some("source-task"));
        assert_eq!(observation.scope.source_worker_id.as_ref().map(|id| id.as_str()), Some("source-worker"));
        assert_eq!(observation.scope.gpu_job_id.as_ref().map(|id| id.as_str()), Some("gpu-job"));
        assert_eq!(observation.scope.gpu_worker_id.as_ref().map(|id| id.as_str()), Some("gpu-worker"));
        assert_eq!(
            observed_stream_states(&owner),
            vec![
                StreamLeaseLifecycleState::Enqueued,
                StreamLeaseLifecycleState::CancellationRequested,
                StreamLeaseLifecycleState::Cancelled,
            ]
        );
    }

    #[test]
    fn stream_acquire_observes_cancellation_while_pool_is_saturated() {
        let device = cudf::DeviceId::DEFAULT;
        let pool = ServiceStreamPool::new(device, 1).unwrap();
        let held = pool.acquire().unwrap();
        let cancellation = ExecutionCancellation::new();
        let waiter_cancellation = cancellation.clone();
        let waiter_pool = pool.clone();
        let (started_sender, started_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();

        let waiter = thread::spawn(move || {
            started_sender.send(()).unwrap();
            let result = waiter_pool.acquire_with_cancellation(
                &waiter_cancellation,
                "nexus_query_engine.service_stream_pool.test_cancelled_acquire",
            );
            result_sender.send(result.map(|_lease| ())).unwrap();
        });
        started_receiver.recv().unwrap();
        thread::sleep(Duration::from_millis(25));
        cancellation.cancel();
        let result = result_receiver.recv_timeout(Duration::from_secs(1));
        drop(held);
        waiter.join().unwrap();
        let error = result.expect("cancelled stream acquisition should finish").unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.operation(), "nexus_query_engine.service_stream_pool.test_cancelled_acquire");
    }

    #[test]
    fn stream_acquire_deadline_times_out_while_pool_is_saturated() {
        let device = cudf::DeviceId::DEFAULT;
        let pool = ServiceStreamPool::new(device, 1).unwrap();
        let held = pool.acquire().unwrap();
        let cancellation = ExecutionCancellation::new();
        let deadline = Instant::now() + Duration::from_millis(10);

        let error = pool
            .acquire_with_cancellation_deadline(
                &cancellation,
                deadline,
                "nexus_query_engine.service_stream_pool.test_timed_acquire",
            )
            .unwrap_err();

        drop(held);
        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.operation(), "nexus_query_engine.service_stream_pool.test_timed_acquire");
        assert_eq!(error.fact_value("reason"), Some("stream_pool_wait_timeout"));
        assert_eq!(error.fact_value("stream_pool_size"), Some("1"));
    }
}
