//! Service-level CUDA stream pooling for native worker tasks.

#[cfg(test)]
use std::sync::{Barrier, mpsc};
use std::{
    collections::HashMap,
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

#[cfg(test)]
use super::DeviceFacts;
#[doc(hidden)]
pub use super::stream_cleanup::StreamCleanupOutcome;
#[cfg(test)]
#[doc(hidden)]
pub use super::stream_cleanup::block_stream_cleanup_thread_for_test;
#[cfg(test)]
use super::stream_cleanup::{
    STREAM_CLEANUP_SUPERVISOR_BUILD_OPERATION, STREAM_CLEANUP_SUPERVISOR_JOINED, STREAM_CLEANUP_SUPERVISOR_NOT_JOINED,
    StreamCleanupEnqueueError, StreamCleanupSupervisorTerminal, fail_next_cleanup_supervisor_spawn_for_test,
    force_cleanup_supervisor_unexpected_exit_for_test, observe_cleanup_supervisor_join_waiter_for_test,
    observe_cleanup_supervisor_shutdown_send_failure_for_test, panic_cleanup_supervisor_after_outcome_for_test,
    pause_cleanup_supervisor_before_worker_return_for_test, probe_cleanup_job_for_test,
    stream_cleanup_thread_id_for_test, wait_for_cleanup_supervisor_worker_finished_for_test,
};
use super::{
    ExecutionCancellation, ExecutionContext,
    context::{AttemptAuxiliaryPermit, QueryServiceResourceEpoch},
    stream_cleanup::{ServiceStreamTestProbeSlot, StreamCleanupCompletion, StreamCleanupSupervisor},
};
use crate::{
    native_error::cudf_query_error,
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
static NEXT_STREAM_POOL_ID: AtomicU64 = AtomicU64::new(1);

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
pub(super) struct ServiceStreamPoolInner {
    pub(super) id: u64,
    pub(super) device: cudf::DeviceId,
    pub(super) pool_size: usize,
    pub(super) root_capacity: usize,
    pub(super) aux_capacity: usize,
    pub(super) state: Mutex<ServiceStreamPoolState>,
    pub(super) available: Condvar,
    pub(super) state_changed: Condvar,
    pub(super) next_lease_id: AtomicU64,
    pub(super) cleanup_probe: ServiceStreamTestProbeSlot,
    pub(super) cleanup_supervisor: Arc<StreamCleanupSupervisor>,
    pub(super) closed: AtomicBool,
    pub(super) process_exit_started: AtomicBool,
}

type ServiceStreamPoolRegistry = HashMap<(usize, usize), Weak<ServiceStreamPoolInner>>;

/// Resource class for bounded stream-pool admission.
///
/// For active-attempt limit `A_d` and per-attempt source-work bound `S_d`,
/// each device reserves `A_d` root stream slots and `A_d * S_d` auxiliary
/// stream slots. Auxiliary (child/source-read) acquisition can never consume
/// a future root slot.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamPermitClass {
    /// Per-attempt root execution stream.
    Root,
    /// Fragment and source-read streams.
    #[default]
    Auxiliary,
}

impl StreamPermitClass {
    /// Stable diagnostic label for this permit class.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Auxiliary => "auxiliary",
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct ServiceStreamPoolState {
    pub(super) idle: Vec<Arc<cudf::Runtime>>,
    pub(super) created: usize,
    pub(super) root_outstanding: usize,
    pub(super) aux_outstanding: usize,
    pub(super) waiting_acquires: usize,
    pub(super) pending_cleanups: usize,
    pub(super) active_cleanups: usize,
    pub(super) synchronization_failures: u64,
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
#[doc(hidden)]
pub struct ServiceStreamPool {
    pub(super) inner: Arc<ServiceStreamPoolInner>,
}

impl ServiceStreamPool {
    /// Create a stream pool for `device`.
    ///
    /// `pool_size` is the maximum number of simultaneously leased streams.
    /// Runtime/stream creation is lazy: construction records the limit but does
    /// not allocate CUDA streams immediately.
    pub fn new(device: cudf::DeviceId, pool_size: usize) -> Result<Self> {
        // Direct constructors keep pre-partition semantics: one root slot for
        // lifecycle work and an auxiliary class bounded only by the physical
        // pool size. Backend-owned pools pass explicit `A_d`/`A_d * S_d`
        // capacities from the device resource profile via
        // [`Self::new_with_classes`].
        Self::new_with_classes(device, pool_size, 1, pool_size)
    }

    /// Creates a pool with explicit root/auxiliary stream-slot capacities.
    pub fn new_with_classes(
        device: cudf::DeviceId,
        pool_size: usize,
        root_capacity: usize,
        aux_capacity: usize,
    ) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        for (field, capacity) in [("root_capacity", root_capacity), ("aux_capacity", aux_capacity)] {
            if capacity == 0 {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.service_stream_pool",
                    "service stream pool permit-class capacity must be greater than zero",
                )
                .with_fact("device_ordinal", device.ordinal())
                .with_fact("field", field)
                .with_fact("reason", "zero"));
            }
        }
        let id = NEXT_STREAM_POOL_ID.fetch_add(1, Ordering::Relaxed);
        let cleanup_probe = ServiceStreamTestProbeSlot::default();
        #[cfg(test)]
        cleanup_probe.install_next_pool_probe();
        // The cleanup receiver is a construction resource. Do not publish a
        // usable pool unless its owner thread was created successfully.
        let cleanup_supervisor = StreamCleanupSupervisor::start(device, id, cleanup_probe.clone())?;
        let inner = Arc::new(ServiceStreamPoolInner {
            id,
            device,
            pool_size,
            root_capacity,
            aux_capacity,
            state: Mutex::new(ServiceStreamPoolState::default()),
            available: Condvar::new(),
            state_changed: Condvar::new(),
            next_lease_id: AtomicU64::new(1),
            cleanup_probe,
            cleanup_supervisor: Arc::clone(&cleanup_supervisor),
            closed: AtomicBool::new(false),
            process_exit_started: AtomicBool::new(false),
        });
        cleanup_supervisor.register_process_exit(Arc::downgrade(&inner));
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
        static POOLS: OnceLock<Mutex<ServiceStreamPoolRegistry>> = OnceLock::new();
        let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut pools = pools.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.service_stream_pool.registry",
                "service stream pool registry mutex was poisoned",
            )
        })?;
        let key = (device.ordinal(), pool_size);
        if let Some(inner) = pools.get(&key).and_then(Weak::upgrade) {
            return Ok(Self { inner });
        }
        let pool = Self::new(device, pool_size)?;
        pools.insert(key, Arc::downgrade(&pool.inner));
        Ok(pool)
    }

    /// CUDA device served by this pool.
    pub fn device(&self) -> cudf::DeviceId {
        self.inner.device
    }

    /// Maximum number of simultaneously leased streams.
    pub fn pool_size(&self) -> usize {
        self.inner.pool_size
    }

    /// Permit-class counters for partition assertions.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn class_counters(&self) -> (usize, usize, usize, usize) {
        let state = self.inner.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.root_outstanding, self.inner.root_capacity, state.aux_outstanding, self.inner.aux_capacity)
    }

    /// Wait until every runtime created by this pool has completed cleanup and
    /// returned to the idle set.
    ///
    /// This is a quiescence boundary for benchmark warmup and graceful service
    /// drain. It does not create runtimes that were never needed.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn wait_until_idle(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        self.wait_until_quiescent(0, deadline, operation)
    }

    /// Prevent new acquisitions and destroy every currently idle runtime.
    /// Leased runtimes are discarded after their deferred synchronization.
    #[doc(hidden)]
    pub fn close(&self, operation: &'static str) -> Result<()> {
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
    #[doc(hidden)]
    pub fn close_and_wait_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
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
            self.inner.cleanup_supervisor.ensure_not_failed(operation)?;
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

    /// Terminal process-exit path. Unlike normal API shutdown it has no
    /// caller-selected timeout: returning while this worker can still destroy
    /// CUDA state would race libcudart finalization.
    pub(super) fn close_and_join_for_process_exit(&self) {
        // Ordinary shutdown waits for every lease because its destructor is
        // part of the reusable-capacity contract. `std::process::exit` skips
        // those destructors, so waiting for externally held leases here would
        // deadlock the atexit hook. Mark the terminal boundary first: a lease
        // racing this hook is leaked instead of running CUDA destruction after
        // the cleanup supervisor has been joined.
        self.inner.process_exit_started.store(true, Ordering::Release);
        self.inner.closed.store(true, Ordering::Release);
        self.inner.available.notify_all();
        let idle = {
            let mut state = self.inner.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let idle = std::mem::take(&mut state.idle);
            state.created = state.created.saturating_sub(idle.len());
            self.inner.state_changed.notify_all();
            idle
        };
        drop(idle);
        self.inner.cleanup_supervisor.join_for_process_exit();
    }

    /// Wait until exactly `expected_leased_runtimes` created runtimes remain
    /// checked out and every other runtime has completed cleanup.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn wait_until_quiescent(
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

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn wait_until_acquire_queued_for_test(&self, deadline: Instant) {
        let mut state = self
            .lock_state("query_engine.service_stream_pool.test_wait_for_queued_acquire")
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
    #[doc(hidden)]
    pub fn fail_next_synchronization_for_test(&self) {
        self.inner.cleanup_probe.with_test_probe(|probe| {
            probe.force_next_synchronization_failure.store(true, Ordering::Release);
        });
    }

    #[cfg(test)]
    fn force_next_cleanup_enqueue_fallback_for_test(&self) {
        self.inner.cleanup_probe.with_test_probe(|probe| {
            probe.force_next_cleanup_enqueue_fallback.store(true, Ordering::Release);
        });
    }

    /// Acquire a CUDA stream/runtime lease, blocking when all pool slots are busy.
    pub fn acquire(&self) -> Result<StreamLease> {
        self.acquire_with_class(StreamPermitClass::Auxiliary)
    }

    /// Acquire a lease of the given permit class, blocking when the class is full.
    pub fn acquire_with_class(&self, permit_class: StreamPermitClass) -> Result<StreamLease> {
        self.acquire_inner(
            None,
            None,
            "query_engine.service_stream_pool.acquire",
            RuntimeObservationHandle::disabled(),
            RuntimeStreamLeaseScope::default(),
            permit_class,
        )
    }

    /// Acquire a CUDA stream/runtime lease while observing cooperative cancellation.
    #[cfg(test)]
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
            StreamPermitClass::Auxiliary,
        )
    }

    /// Acquire a CUDA stream/runtime lease while observing cancellation and an
    /// absolute wait deadline.
    #[cfg(test)]
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
            StreamPermitClass::Auxiliary,
        )
    }

    /// Acquires a lease while recording host-side wait and lifecycle evidence.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn acquire_observed_with_cancellation_deadline(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
    ) -> Result<StreamLease> {
        self.acquire_observed_with_cancellation_deadline_and_class(
            cancellation,
            deadline,
            operation,
            observation,
            scope,
            StreamPermitClass::Auxiliary,
        )
    }

    #[doc(hidden)]
    pub fn acquire_observed_with_cancellation_deadline_and_class(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
        permit_class: StreamPermitClass,
    ) -> Result<StreamLease> {
        self.acquire_inner(Some(cancellation), Some(deadline), operation, observation, scope, permit_class)
    }

    #[doc(hidden)]
    pub fn acquire_observed_with_cancellation(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
    ) -> Result<StreamLease> {
        self.acquire_observed_with_cancellation_and_class(
            cancellation,
            operation,
            observation,
            scope,
            StreamPermitClass::Auxiliary,
        )
    }

    #[doc(hidden)]
    pub fn acquire_observed_with_cancellation_and_class(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
        permit_class: StreamPermitClass,
    ) -> Result<StreamLease> {
        self.acquire_inner(Some(cancellation), None, operation, observation, scope, permit_class)
    }

    fn acquire_inner(
        &self,
        cancellation: Option<&ExecutionCancellation>,
        deadline: Option<Instant>,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
        permit_class: StreamPermitClass,
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
            let class_open = match permit_class {
                StreamPermitClass::Root => state.root_outstanding < self.inner.root_capacity,
                StreamPermitClass::Auxiliary => state.aux_outstanding < self.inner.aux_capacity,
            };
            if class_open {
                if let Some(runtime) = state.idle.pop() {
                    match permit_class {
                        StreamPermitClass::Root => state.root_outstanding += 1,
                        StreamPermitClass::Auxiliary => state.aux_outstanding += 1,
                    }
                    self.inner.state_changed.notify_all();
                    break AcquireAction::Reuse(runtime);
                }
                if state.created < self.pool_size() {
                    state.created += 1;
                    match permit_class {
                        StreamPermitClass::Root => state.root_outstanding += 1,
                        StreamPermitClass::Auxiliary => state.aux_outstanding += 1,
                    }
                    self.inner.state_changed.notify_all();
                    break AcquireAction::Create;
                }
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
                    self.discard_created_slot(permit_class);
                    drop(runtime);
                    let error = stream_pool_closed(operation, self);
                    lease_observation
                        .terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                    return Err(error);
                },
                Err(error) => {
                    self.discard_created_slot(permit_class);
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
            permit_class,
        })
    }

    fn build_runtime(&self) -> Result<cudf::Runtime> {
        cudf::Runtime::builder().with_device(self.device()).with_device_memory_manager().build().map_err(|error| {
            cudf_query_error(
                "query_engine.service_stream_pool.runtime",
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

    fn discard_created_slot(&self, permit_class: StreamPermitClass) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.created = state.created.saturating_sub(1);
                match permit_class {
                    StreamPermitClass::Root => state.root_outstanding = state.root_outstanding.saturating_sub(1),
                    StreamPermitClass::Auxiliary => state.aux_outstanding = state.aux_outstanding.saturating_sub(1),
                }
                self.inner.available.notify_one();
                self.inner.state_changed.notify_all();
            },
            Err(_error) => {
                eprintln!("[query-engine] warning: failed to update poisoned service stream pool state");
            },
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
    permit_class: StreamPermitClass,
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

    #[doc(hidden)]
    pub fn with_resource_epoch(mut self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        self.resource_epoch = Some(resource_epoch);
        self
    }

    #[doc(hidden)]
    pub fn install_cleanup_completion(
        &self,
        before_quiescence: impl FnOnce(StreamCleanupOutcome) + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> bool {
        // Materialize the handoff before inspecting the slot.  A duplicate
        // installation is an ownership invariant violation: dropping this
        // completion deliberately reports `CallbackAbandoned` and runs its
        // tail rather than silently losing report/health work and releasing
        // the query's capacity without a terminal commit.
        let completion = StreamCleanupCompletion::new(before_quiescence, after_quiescence);
        let mut slot = match self.cleanup_completion.lock() {
            Ok(slot) => slot,
            Err(error) => error.into_inner(),
        };
        if slot.is_some() {
            return false;
        }
        *slot = Some(completion);
        true
    }

    #[doc(hidden)]
    pub fn device(&self) -> cudf::DeviceId {
        self.runtime.as_ref().expect("stream lease runtime is present").device()
    }

    /// Clone a counted cuDF runtime derived from this lease.
    pub fn runtime(&self) -> Result<cudf::Runtime> {
        let runtime = self.runtime.as_ref().expect("stream lease runtime is present").as_ref().clone();
        let Some(resource_epoch) = &self.resource_epoch else {
            return Ok(runtime);
        };
        resource_epoch.check_open("query_engine.service_stream_pool.lease.runtime")?;
        let retained: Arc<dyn Send + Sync> = resource_epoch.clone();
        runtime.with_retained_lifetime(retained).map_err(|error| {
            cudf_query_error(
                "query_engine.service_stream_pool.lease.runtime",
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
        self.pool.enqueue_cleanup(runtime, self.resource_epoch.take(), cleanup_completion, self.permit_class);
    }
}

#[doc(hidden)]
pub struct SourceReadStreamLease {
    context: ExecutionContext,
    _stream_lease: StreamLease,
    // Drop after the derived context and global stream lease so the
    // attempt-local envelope remains charged for every actual auxiliary
    // stream lifetime.
    _attempt_auxiliary_permit: Option<AttemptAuxiliaryPermit>,
}

impl SourceReadStreamLease {
    pub(super) fn new(
        context: ExecutionContext,
        stream_lease: StreamLease,
        attempt_auxiliary_permit: Option<AttemptAuxiliaryPermit>,
    ) -> Self {
        Self { context, _stream_lease: stream_lease, _attempt_auxiliary_permit: attempt_auxiliary_permit }
    }

    #[doc(hidden)]
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

pub(super) fn validate_service_stream_pool_size(pool_size: usize) -> Result<()> {
    if pool_size == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.service_stream_pool.validate",
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
        assert_eq!(error.operation(), "query_engine.service_stream_pool.validate");
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
    fn process_exit_joins_an_orphaned_cleanup_supervisor() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let supervisor = Arc::clone(&pool.inner.cleanup_supervisor);

        drop(pool);
        supervisor.join_for_process_exit();

        assert_eq!(supervisor.join_disposition.load(Ordering::Acquire), STREAM_CLEANUP_SUPERVISOR_JOINED,);
    }

    #[test]
    fn cleanup_submission_racing_process_exit_is_retained_after_terminal_boundary() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let supervisor = Arc::clone(&pool.inner.cleanup_supervisor);
        let submission_gate = supervisor.submission_gate.lock().unwrap();
        let submitter_supervisor = Arc::clone(&supervisor);
        let submitter_pool = pool.clone();
        let rendezvous = Arc::new(Barrier::new(2));
        let submitter_rendezvous = Arc::clone(&rendezvous);
        let (done, receiver) = mpsc::channel();
        let submitter = thread::spawn(move || {
            submitter_rendezvous.wait();
            submitter_supervisor.enqueue_before_process_exit(
                probe_cleanup_job_for_test(done),
                &submitter_pool.inner.process_exit_started,
            )
        });

        rendezvous.wait();
        pool.inner.process_exit_started.store(true, Ordering::Release);
        drop(submission_gate);
        let retained_job = match submitter.join().expect("cleanup submitter must not panic") {
            Err(StreamCleanupEnqueueError::ProcessExit(job)) => job,
            Ok(()) | Err(StreamCleanupEnqueueError::SupervisorStopped(_)) => {
                panic!("cleanup crossing the process-exit boundary must be retained")
            },
        };
        assert!(matches!(receiver.try_recv(), Err(mpsc::TryRecvError::Empty)));

        pool.close_and_join_for_process_exit();
        drop(retained_job);
        assert!(matches!(receiver.try_recv(), Err(mpsc::TryRecvError::Disconnected)));
    }

    #[test]
    fn cleanup_supervisor_spawn_failure_rejects_pool_construction() {
        fail_next_cleanup_supervisor_spawn_for_test();

        let error = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.status(), query_planning::error::QueryErrorStatus::Permanent);
        assert_eq!(error.operation(), STREAM_CLEANUP_SUPERVISOR_BUILD_OPERATION);
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_spawn_failed"));
        assert_eq!(error.fact_value("device_ordinal"), Some("0"));
    }

    #[test]
    fn close_and_wait_explicitly_joins_cleanup_supervisor_for_all_clones() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let clone = pool.clone();
        let deadline = Instant::now() + Duration::from_secs(5);

        pool.close_and_wait_until(deadline, "query_engine.service_stream_pool.test_join").unwrap();
        assert_eq!(
            pool.inner.cleanup_supervisor.join_disposition.load(Ordering::Acquire),
            STREAM_CLEANUP_SUPERVISOR_JOINED,
        );

        clone.close_and_wait_until(deadline, "query_engine.service_stream_pool.test_join").unwrap();
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
                "query_engine.service_stream_pool.test_busy_cleanup",
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
            "query_engine.service_stream_pool.test_busy_cleanup",
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
                "query_engine.service_stream_pool.test_unexpected_exit_publication",
            )
            .unwrap_err();
        assert_eq!(publication.fact_value("reason"), Some("stream_cleanup_supervisor_unexpected_exit"));

        let error = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "query_engine.service_stream_pool.test_unexpected_exit",
            )
            .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.status(), query_planning::error::QueryErrorStatus::Permanent);
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
                move |outcome| before_sender.send(("first", outcome, thread::current().id())).unwrap()
            },
            {
                let after_sender = after_sender.clone();
                move || after_sender.send(("first", thread::current().id())).unwrap()
            },
        ));
        assert!(second.install_cleanup_completion(
            move |outcome| before_sender.send(("second", outcome, thread::current().id())).unwrap(),
            move || after_sender.send(("second", thread::current().id())).unwrap(),
        ));
        drop(first);
        drop(second);
        drop(blocker);

        let error = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "query_engine.service_stream_pool.test_exit_with_payload",
            )
            .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_unexpected_exit"));
        let mut before = [
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("first recovery before callback"),
            before_receiver.recv_timeout(Duration::from_secs(5)).expect("second recovery before callback"),
        ];
        before.sort_by_key(|(label, _, _)| *label);
        let mut after = [
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("first recovery after callback"),
            after_receiver.recv_timeout(Duration::from_secs(5)).expect("second recovery after callback"),
        ];
        after.sort_by_key(|(label, _)| *label);
        assert_eq!(before.iter().map(|(label, _, _)| *label).collect::<Vec<_>>(), vec!["first", "second"]);
        assert!(
            before.iter().all(|(_, outcome, _)| *outcome == StreamCleanupOutcome::CallbackAbandoned),
            "queued cleanup recovery must preserve abandonment rather than misclassifying it as a sync failure"
        );
        assert_eq!(after.iter().map(|(label, _)| *label).collect::<Vec<_>>(), vec!["first", "second"]);
        assert!(before.iter().all(|(_, _, thread_id)| *thread_id == cleanup_thread));
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
            move |_| {
                before_sender.send(thread::current().id()).unwrap();
                panic!("injected in-flight cleanup completion panic");
            },
            move || after_sender.send(thread::current().id()).unwrap(),
        ));
        drop(lease);

        let error = pool
            .close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "query_engine.service_stream_pool.test_inflight_payload",
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
        assert!(lease.install_cleanup_completion(|_| panic!("injected cleanup panic before idle boundary"), || {},));
        drop(lease);

        pause.arrived.wait();
        pause.release.wait();
        let deadline = Instant::now() + Duration::from_secs(5);
        wait_for_cleanup_supervisor_worker_finished_for_test(&pool, deadline);

        let error =
            pool.wait_until_idle(deadline, "query_engine.service_stream_pool.test_idle_finished_panic").unwrap_err();
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
            lease.install_cleanup_completion(|_| panic!("injected cleanup panic before quiescence boundary"), || {},)
        );
        drop(lease);

        pause.arrived.wait();
        pause.release.wait();
        let deadline = Instant::now() + Duration::from_secs(5);
        wait_for_cleanup_supervisor_worker_finished_for_test(&pool, deadline);

        let error = pool
            .wait_until_quiescent(0, deadline, "query_engine.service_stream_pool.test_quiescent_finished_panic")
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
            move |_| {
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
                "query_engine.service_stream_pool.test_send_failure_panic",
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
                "query_engine.service_stream_pool.test_concurrent_joiner_first",
            )
        });
        let second_pool = pool.clone();
        let second = thread::spawn(move || {
            second_pool.close_and_wait_until(
                Instant::now() + Duration::from_secs(5),
                "query_engine.service_stream_pool.test_concurrent_joiner_second",
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
                "query_engine.service_stream_pool.test_precommit_join_timeout",
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
            "query_engine.service_stream_pool.test_precommit_join_timeout",
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
            pool.close_and_wait_until(deadline, "query_engine.service_stream_pool.test_join_panic").unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.status(), query_planning::error::QueryErrorStatus::Permanent);
        assert_eq!(error.fact_value("reason"), Some("stream_cleanup_supervisor_panicked"));
        assert_eq!(error.fact_value("supervisor_stage"), Some("thread_join"));

        let repeated =
            pool.close_and_wait_until(deadline, "query_engine.service_stream_pool.test_join_panic").unwrap_err();
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
            move |_| {
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
            move |_| {
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
            "query_engine.service_stream_pool.test_sync_failure",
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
            Some(cudf::runtime::DeviceFaultNativeStatus::CUDF_FATAL_CUDA),
        );

        let error = pool.acquire().expect_err("unhealthy device must reject idle runtime reuse");
        assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
        assert_eq!(error.fact_value("reason"), Some("device_health_rejected_stream_acquire"));
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
                "query_engine.service_stream_pool.test_observed_create",
                owner.handle(),
                RuntimeStreamLeaseScope::default(),
            )
            .unwrap();
        drop(created);
        let _ = stream_cleanup_thread_id_for_test(&pool);
        let reused = pool
            .acquire_observed_with_cancellation(
                &cancellation,
                "query_engine.service_stream_pool.test_observed_reuse",
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
                "query_engine.service_stream_pool.test_observed_cancel",
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
                "query_engine.service_stream_pool.test_cancelled_acquire",
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
        assert_eq!(error.operation(), "query_engine.service_stream_pool.test_cancelled_acquire");
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
                "query_engine.service_stream_pool.test_timed_acquire",
            )
            .unwrap_err();

        drop(held);
        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.operation(), "query_engine.service_stream_pool.test_timed_acquire");
        assert_eq!(error.fact_value("reason"), Some("stream_pool_wait_timeout"));
        assert_eq!(error.fact_value("stream_pool_size"), Some("1"));
    }
}

#[cfg(test)]
mod permit_class_tests {
    use super::*;

    #[test]
    fn root_class_waits_while_auxiliary_slots_are_free() {
        let pool = ServiceStreamPool::new_with_classes(cudf::DeviceId::DEFAULT, 3, 1, 2).unwrap();
        let root = pool.acquire_with_class(StreamPermitClass::Root).unwrap();
        let aux_one = pool.acquire_with_class(StreamPermitClass::Auxiliary).unwrap();
        let aux_two = pool.acquire_with_class(StreamPermitClass::Auxiliary).unwrap();
        let (root_out, root_cap, aux_out, aux_cap) = pool.class_counters();
        assert_eq!((root_out, root_cap, aux_out, aux_cap), (1, 1, 2, 2));

        let (sender, receiver) = std::sync::mpsc::channel();
        let second_pool = pool.clone();
        let waiter = std::thread::spawn(move || {
            sender.send(()).unwrap();
            second_pool.acquire_observed_with_cancellation_deadline_and_class(
                &ExecutionCancellation::new(),
                Instant::now() + std::time::Duration::from_millis(200),
                "query_engine.service_stream_pool.test_root_wait",
                RuntimeObservationHandle::disabled(),
                RuntimeStreamLeaseScope::default(),
                StreamPermitClass::Root,
            )
        });
        receiver.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!waiter.is_finished(), "root class is full even though every physical slot is busy anyway");

        drop(aux_two);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            !waiter.is_finished(),
            "freeing an auxiliary slot must not release a root waiter: classes are partitioned"
        );
        drop(root);
        let acquired = waiter.join().unwrap().expect("root waiter acquires after the root lease returns");
        drop(acquired);
        drop(aux_one);
    }

    #[test]
    fn auxiliary_class_waits_while_root_slots_are_free() {
        let pool = ServiceStreamPool::new_with_classes(cudf::DeviceId::DEFAULT, 3, 1, 1).unwrap();
        let _root = pool.acquire_with_class(StreamPermitClass::Root).unwrap();
        let aux = pool.acquire_with_class(StreamPermitClass::Auxiliary).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let second_pool = pool.clone();
        let waiter = std::thread::spawn(move || {
            sender.send(()).unwrap();
            second_pool.acquire_observed_with_cancellation_deadline_and_class(
                &ExecutionCancellation::new(),
                Instant::now() + std::time::Duration::from_millis(200),
                "query_engine.service_stream_pool.test_aux_wait",
                RuntimeObservationHandle::disabled(),
                RuntimeStreamLeaseScope::default(),
                StreamPermitClass::Auxiliary,
            )
        });
        receiver.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!waiter.is_finished(), "auxiliary class is full while the root slot stays free");
        drop(aux);
        let acquired = waiter.join().unwrap().expect("auxiliary waiter acquires after an auxiliary return");
        drop(acquired);
    }

    #[test]
    fn zero_class_capacity_is_rejected() {
        let error = ServiceStreamPool::new_with_classes(cudf::DeviceId::DEFAULT, 2, 0, 1).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("root_capacity"));
        let error = ServiceStreamPool::new_with_classes(cudf::DeviceId::DEFAULT, 2, 1, 0).unwrap_err();
        assert_eq!(error.fact_value("field"), Some("aux_capacity"));
    }
}
