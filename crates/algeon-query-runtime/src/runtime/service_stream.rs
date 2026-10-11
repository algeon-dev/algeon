//! Service-level CUDA stream pooling for native worker tasks.
//!
//! One `ServiceStreamPool` exists per installed device. The backend builds it in
//! `DeviceRegistry::install_resolved` with the capacities from the device
//! resource profile: `A_d` root slots, `A_d * S_d` auxiliary slots, and a
//! physical `pool_size` that bounds how many cuDF runtimes (each owning one CUDA
//! stream) exist at once. The capacity table in
//! `website/docs/design/admission.mdx#device-profiles-and-grants` owns that
//! contract; the lease and lane vocabulary comes from
//! `website/docs/design/gpu-execution.mdx#cuda-streams-and-dependencies`.
//!
//! A lease is produced in two steps. `SlotRequest::take_or_register` takes a
//! class permit and either an idle runtime or the right to create one, all
//! under the pool mutex and without CUDA calls. `StreamSlot::into_lease` then
//! builds the runtime on a thread where CUDA work is allowed; that build is
//! the only CUDA call on the acquire path. Dropping a `StreamLease` runs no
//! CUDA either: the runtime goes to the cleanup supervisor in
//! `stream_cleanup`, which synchronizes the stream and only then returns the
//! runtime to `idle` and releases the class permit.
//!
//! Accounting fails closed. `closed` is set once and never cleared, by an
//! explicit close, by a stream synchronization failure, or by an abandoned
//! cleanup. A closed pool rejects acquisition and destroys every returning
//! runtime; none is idled. Acquisition errors carry a `reason` fact:
//! `stream_pool_closed`, `stream_pool_wait_timeout`, or
//! `device_health_rejected_stream_acquire`; shutdown timeouts report
//! `stream_pool_idle_timeout` or `stream_pool_quiescence_timeout` together
//! with the cleanup-deadline marker.

#[cfg(test)]
use std::sync::{Barrier, mpsc};
use std::{
    cell::Cell,
    fmt,
    future::pending,
    mem,
    pin::{Pin, pin},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError, TryLockError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Instant,
};
#[cfg(any(test, feature = "test-hooks"))]
use std::{
    collections::HashMap,
    sync::{OnceLock, Weak},
};

use cudf::{
    DeviceId, Runtime,
    runtime::{DeviceHealthSnapshot, DeviceHealthStatus, device_health_snapshot},
};
use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::{
    sync::{Notify, futures::Notified},
    time::sleep_until,
};

#[cfg(test)]
use super::DeviceFacts;
pub(crate) use super::stream_cleanup::StreamCleanupOutcome;
#[cfg(test)]
pub(crate) use super::stream_cleanup::block_stream_cleanup_thread_for_test;
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
    deadline::block_until,
    stream_cleanup::{ServiceStreamTestProbeSlot, StreamCleanupCompletion, StreamCleanupSupervisor},
};
use crate::{
    native_error::cudf_query_error,
    observability::{
        RuntimeEventContext, RuntimeObservationHandle, RuntimeObservationPayload, RuntimeObservationRecordClass,
        RuntimeResourceDomain, RuntimeResourceId, RuntimeResourceRow, RuntimeStreamLeaseObservation,
        RuntimeStreamLeaseScope, RuntimeTopologyRecord, StreamLeaseAcquisition, StreamLeaseLifecycleState,
    },
};

/// Fallback and upper bound for the default service stream pool size.
///
/// The pool creates streams lazily, so this is a concurrency cap rather than a
/// startup allocation count. `default_service_stream_pool_size_for_device`
/// uses this value as the fallback when device SM count detection is
/// unavailable and as the upper bound when detection succeeds.
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) const DEFAULT_SERVICE_STREAM_POOL_SIZE: usize = 64;
static NEXT_STREAM_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Returns the default service stream pool size for a CUDA device.
///
/// The default is `min(device_multiprocessor_count * 2, 64)`. If the
/// multiprocessor count cannot be detected, this returns
/// [`DEFAULT_SERVICE_STREAM_POOL_SIZE`]. The detected value is cached per
/// device ordinal for the process lifetime.
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn default_service_stream_pool_size_for_device(device: DeviceId) -> usize {
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

#[cfg(any(test, feature = "test-hooks"))]
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

#[cfg(any(test, feature = "test-hooks"))]
fn detect_device_multiprocessor_count(device: DeviceId) -> Option<usize> {
    device.multiprocessor_count().ok().and_then(nonzero_sm_count)
}

#[cfg(any(test, feature = "test-hooks"))]
pub(super) fn nonzero_sm_count(count: u32) -> Option<usize> {
    (count > 0).then(|| usize::try_from(count).ok()).flatten()
}

/// State shared by every clone of one device pool and by its leases.
///
/// A `StreamLease` holds a pool clone, so this struct outlives every lease.
/// Its `Drop` (in `stream_cleanup`) requests cleanup-supervisor shutdown,
/// sound because no lease, and therefore no pending enqueue, can exist at
/// that point.
#[derive(Debug)]
pub(super) struct ServiceStreamPoolInner {
    pub(super) id: u64,
    pub(super) device: DeviceId,
    /// Bound on runtimes that exist at once, independent of the class
    /// capacities. A request must pass its class check and find either an
    /// idle runtime or `created < pool_size`.
    pub(super) pool_size: usize,
    pub(super) root_capacity: usize,
    pub(super) aux_capacity: usize,
    /// Protects every counter in the pool. `closed` is read again under this
    /// lock before a slot is taken.
    pub(super) state: Mutex<ServiceStreamPoolState>,
    /// Wakes acquire waiters. A waiter calls `Notified::enable` while holding
    /// `state`, so a release that follows its check cannot be missed.
    pub(super) available: Notify,
    /// Wakes shutdown and quiescence waiters; paired with `state`.
    pub(super) state_changed: Condvar,
    pub(super) next_lease_id: AtomicU64,
    pub(super) cleanup_probe: ServiceStreamTestProbeSlot,
    pub(super) cleanup_supervisor: Arc<StreamCleanupSupervisor>,
    /// Set once, never cleared. Set by `close`, by a failed stream
    /// synchronization, by an abandoned cleanup job, and by a cleanup enqueue
    /// that finds the supervisor gone. Unsynchronized readers may race a
    /// concurrent close; `take_or_register` rechecks it under `state`.
    pub(super) closed: AtomicBool,
    /// Set once by the process-exit hook before it joins the cleanup thread.
    /// After that, `enqueue_cleanup` leaks its payload; CUDA destruction may
    /// neither run nor be queued behind the joined thread.
    pub(super) process_exit_started: AtomicBool,
}

#[cfg(any(test, feature = "test-hooks"))]
type ServiceStreamPoolRegistry = HashMap<(usize, usize), Weak<ServiceStreamPoolInner>>;

/// Resource class for bounded stream-pool admission.
///
/// For active-attempt limit `A_d` and per-attempt source-work bound `S_d`,
/// each device reserves `A_d` root stream slots and `A_d * S_d` auxiliary
/// stream slots. Auxiliary (child/source-read) acquisition can never consume
/// a future root slot.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StreamPermitClass {
    /// Per-attempt root execution stream.
    Root,
    /// Fragment and source-read streams.
    #[default]
    Auxiliary,
}

/// Counters behind the pool mutex.
///
/// `created` counts runtimes that exist: idle, leased, or in cleanup. The
/// quiescence waiters and the timeout error derive `active_leased_runtimes`
/// as `created - idle.len() - pending_cleanups - active_cleanups`.
#[derive(Debug, Default)]
pub(super) struct ServiceStreamPoolState {
    /// Runtimes whose stream was synchronized by the cleanup thread and that
    /// may be leased again. Always empty once `closed` is set.
    pub(super) idle: Vec<Arc<Runtime>>,
    /// Incremented when a slot reserves the right to build a runtime, before
    /// the build; decremented when a runtime is destroyed or the build fails.
    pub(super) created: usize,
    /// Root class permits held. A permit is taken with the slot and released by
    /// the cleanup job after the stream is synchronized, so a new lease of the
    /// class cannot start before the previous holder's work has drained.
    pub(super) root_outstanding: usize,
    /// Auxiliary class permits held, with the same lifetime as `root_outstanding`.
    pub(super) aux_outstanding: usize,
    /// Threads or tasks parked in an acquire. Shutdown waits for zero so a
    /// late waiter cannot observe a destroyed pool.
    pub(super) waiting_acquires: usize,
    /// Cleanup jobs enqueued but not yet started by the cleanup thread.
    pub(super) pending_cleanups: usize,
    /// Cleanup jobs between `begin_accounting` and `finish_active_accounting`,
    /// which includes the `before_quiescence` completion callback.
    pub(super) active_cleanups: usize,
    /// Streams whose synchronization failed. Nonzero implies `closed`; the
    /// value is reported on `stream_pool_closed` errors.
    pub(super) synchronization_failures: u64,
}

/// Service-level CUDA stream pool for short-lived native worker tasks.
///
/// The pool owns a bounded set of same-device cuDF runtimes. Each runtime
/// carries one CUDA stream and the process-local device memory manager. Streams
/// are created lazily up to `pool_size`; dropping a `StreamLease` returns the
/// runtime to the pool for reuse. This avoids creating a CUDA stream for every
/// bounded source-read worker while preserving the existing reservation-aware
/// allocator contract.
#[derive(Debug, Clone)]
pub struct ServiceStreamPool {
    pub(super) inner: Arc<ServiceStreamPoolInner>,
}

impl ServiceStreamPool {
    /// Create a stream pool for `device`.
    ///
    /// `pool_size` is the maximum number of simultaneously leased streams.
    /// Runtime/stream creation is lazy: construction records the limit but does
    /// not allocate CUDA streams immediately.
    pub fn new(device: DeviceId, pool_size: usize) -> Result<Self> {
        // Direct constructors keep pre-partition semantics: one root slot for
        // lifecycle work and an auxiliary class bounded only by the physical
        // pool size. Backend-owned pools pass explicit `A_d`/`A_d * S_d`
        // capacities from the device resource profile via
        // [`Self::new_with_classes`].
        Self::new_with_classes(device, pool_size, 1, pool_size)
    }

    /// Creates a pool with explicit root/auxiliary stream-slot capacities.
    ///
    /// `root_capacity` is `A_d` and `aux_capacity` is `A_d * S_d` when the
    /// backend constructs the pool; `pool_size` bounds physical runtimes
    /// separately and the profile resolver guarantees it covers both classes.
    /// Construction spawns the cleanup thread but creates no CUDA stream. A
    /// cleanup-thread spawn failure fails construction, because a pool
    /// without that thread could not return any lease.
    pub(crate) fn new_with_classes(
        device: DeviceId,
        pool_size: usize,
        root_capacity: usize,
        aux_capacity: usize,
    ) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        for (field, capacity) in [("root_capacity", root_capacity), ("aux_capacity", aux_capacity)] {
            if capacity == 0 {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidConfiguration,
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
            available: Notify::new(),
            state_changed: Condvar::new(),
            next_lease_id: AtomicU64::new(1),
            cleanup_probe,
            cleanup_supervisor: Arc::clone(&cleanup_supervisor),
            closed: AtomicBool::new(false),
            process_exit_started: AtomicBool::new(false),
        });
        cleanup_supervisor.attach_pool(Arc::downgrade(&inner));
        Ok(Self { inner })
    }

    /// Return the process-global default pool for `device` and `pool_size`.
    ///
    /// This is the fallback used by native execution when a service has not
    /// injected its own pool yet. Pools are keyed by `(device ordinal,
    /// pool_size)` so differently sized service tests do not silently share a
    /// capacity contract.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn for_device(device: DeviceId, pool_size: usize) -> Result<Self> {
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
    pub fn device(&self) -> DeviceId {
        self.inner.device
    }

    /// Maximum number of simultaneously leased streams.
    pub(crate) fn pool_size(&self) -> usize {
        self.inner.pool_size
    }

    /// Permit-class counters for partition assertions.
    #[cfg(test)]
    pub(crate) fn class_counters(&self) -> (usize, usize, usize, usize) {
        let state = self.inner.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.root_outstanding, self.inner.root_capacity, state.aux_outstanding, self.inner.aux_capacity)
    }

    /// Wait until every runtime created by this pool has completed cleanup and
    /// returned to the idle set.
    ///
    /// This is a quiescence boundary for benchmark warmup and graceful service
    /// drain. It does not create runtimes that were never needed.
    #[cfg(test)]
    pub(crate) fn wait_until_idle(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        self.wait_until_quiescent(0, deadline, operation)
    }

    /// Prevent new acquisitions and destroy every currently idle runtime.
    /// Leased runtimes are discarded after their deferred synchronization.
    ///
    /// This is the non-blocking close used by `QueryService::close`, which
    /// must not wait behind an admission transaction. The flag store is what
    /// closes the pool; destroying the idle set is opportunistic. When the
    /// mutex is contended the idle runtimes stay in place and are destroyed by
    /// a later `close_and_wait_until` or by the final pool drop.
    pub(crate) fn close(&self, operation: &'static str) -> Result<()> {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.available.notify_waiters();
        let idle = match self.inner.state.try_lock() {
            Ok(mut state) => {
                let idle = mem::take(&mut state.idle);
                state.created = state.created.saturating_sub(idle.len());
                self.inner.state_changed.notify_all();
                idle
            },
            Err(TryLockError::WouldBlock) => return Ok(()),
            Err(TryLockError::Poisoned(_)) => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "service stream pool mutex was poisoned",
                ));
            },
        };
        // Stream destruction runs here, after the match arm released the mutex.
        drop(idle);
        Ok(())
    }

    /// Close this pool and wait until every lease and deferred cleanup has
    /// completed, then explicitly stop and join the cleanup supervisor, under
    /// the caller's absolute shutdown deadline.
    ///
    /// The wait ends when `created` is zero: a closed pool never idles a
    /// runtime, so every lease returning through cleanup destroys one, and
    /// zero means no runtime exists in any state. `waiting_acquires` must
    /// also be zero so no parked acquirer wakes against a joined supervisor.
    /// A supervisor that has already failed ends the wait early with its
    /// terminal error. Repeated calls are allowed; a timed-out
    /// call leaves the supervisor joinable for the next one.
    pub(crate) fn close_and_wait_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.available.notify_waiters();
        let mut state = self.lock_state_until(deadline, operation)?;
        let idle = mem::take(&mut state.idle);
        state.created = state.created.saturating_sub(idle.len());
        // Runtime destruction is a CUDA call; the cleanup thread and any
        // acquirer need the pool mutex meanwhile.
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
                // Same rule as `deadline::mutex_lock_until`: a result reached
                // after the deadline is not accepted as success, so a
                // coordinator cannot overrun its budget on a lucky late wake.
                if Instant::now() >= deadline {
                    return Err(stream_pool_cleanup_timeout(operation, self, &state, 0));
                }
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(stream_pool_cleanup_timeout(operation, self, &state, 0));
            }
            let (next, timeout) = self.inner.state_changed.wait_timeout(state, remaining).map_err(|_error| {
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
        // The elected joiner takes this mutex in `notify_pool_waiters` after
        // committing the terminal state; holding it here would deadlock.
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
        self.inner.available.notify_waiters();
        let idle = {
            let mut state = self.inner.state.lock().unwrap_or_else(PoisonError::into_inner);
            let idle = mem::take(&mut state.idle);
            state.created = state.created.saturating_sub(idle.len());
            self.inner.state_changed.notify_all();
            idle
        };
        // Destroyed outside the mutex and before the join: the cleanup thread
        // may still need the mutex for a job it is finishing.
        drop(idle);
        self.inner.cleanup_supervisor.join_for_process_exit();
    }

    /// Wait until exactly `expected_leased_runtimes` created runtimes remain
    /// checked out and every other runtime has completed cleanup.
    #[cfg(test)]
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
            let (next, _timeout) = self.inner.state_changed.wait_timeout(state, remaining).map_err(|_error| {
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
    pub(crate) fn fail_next_synchronization_for_test(&self) {
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
    pub(crate) fn acquire_with_class(&self, permit_class: StreamPermitClass) -> Result<StreamLease> {
        self.acquire_inner(
            None,
            None,
            "query_engine.service_stream_pool.acquire",
            RuntimeObservationHandle::disabled(),
            RuntimeStreamLeaseScope::default(),
            permit_class,
        )
    }

    /// Acquire a CUDA stream/runtime lease while observing cancellation and an
    /// absolute wait deadline.
    #[cfg(test)]
    pub(crate) fn acquire_with_cancellation_deadline(
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
    pub(crate) fn acquire_observed_with_cancellation_deadline(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Option<Instant>,
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

    pub(crate) fn acquire_observed_with_cancellation_deadline_and_class(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Option<Instant>,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
        permit_class: StreamPermitClass,
    ) -> Result<StreamLease> {
        self.acquire_inner(Some(cancellation), deadline, operation, observation, scope, permit_class)
    }

    pub(crate) fn acquire_observed_with_cancellation(
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

    pub(crate) fn acquire_observed_with_cancellation_and_class(
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
        self.reserve_blocking(cancellation, deadline, operation, observation, scope, permit_class)?
            .into_lease(operation)
    }

    /// Reserves one stream slot of `permit_class`, blocking the calling thread
    /// until a slot frees, cancellation is requested, or a deadline passes.
    /// Each of them wakes the thread directly.
    ///
    /// Callers are GPU pool workers and other threads that may sit inside a
    /// Tokio task, so the wait goes through `deadline::block_until`, which
    /// polls under `unconstrained`. A plain `block_on` of `Notified` would
    /// spend the enclosing task's cooperative budget, and a `Notified` that
    /// returns `Pending` on an exhausted budget defers its wake until the
    /// worker yields to the scheduler, which a parked thread never does.
    ///
    /// `block_until` forbids Tokio timers in the future it polls, so the
    /// cancellation's own deadline is folded into `wait_deadline` and
    /// `cancelled_or_expired` is not awaited here. The loop re-runs the full check in
    /// `take_or_register` after every wake, so which event woke the thread
    /// does not decide the outcome.
    fn reserve_blocking(
        &self,
        cancellation: Option<&ExecutionCancellation>,
        deadline: Option<Instant>,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
        permit_class: StreamPermitClass,
    ) -> Result<StreamSlot> {
        let request = SlotRequest::new(self, cancellation, deadline, operation, observation, scope, permit_class);
        let wait_deadline = match (deadline, cancellation.and_then(ExecutionCancellation::deadline)) {
            (Some(deadline), Some(cancellation_deadline)) => Some(deadline.min(cancellation_deadline)),
            (deadline, cancellation_deadline) => deadline.or(cancellation_deadline),
        };
        loop {
            let available = self.inner.available.notified();
            let mut available = pin!(available);
            // `_waiting` keeps this thread counted in `waiting_acquires` for
            // the whole park, so shutdown cannot complete underneath it.
            let _waiting = match request.take_or_register(available.as_mut())? {
                SlotOrWait::Slot(slot) => return Ok(slot),
                SlotOrWait::Wait(waiting) => waiting,
            };
            let _ = block_until(
                async {
                    match cancellation {
                        Some(cancellation) => tokio::select! {
                            biased;
                            () = available => {},
                            () = cancellation.cancel_requested() => {},
                        },
                        None => available.await,
                    }
                },
                wait_deadline,
            );
        }
    }

    /// Reserves one stream slot of `permit_class` without blocking the
    /// caller's thread. The reservation creates no CUDA stream; the lease is
    /// materialized from it where CUDA work is allowed.
    ///
    /// This runs on the host runtime (the query handle reserves auxiliary
    /// slots before submitting a pool job), so Tokio timers are available and
    /// the cancellation deadline is awaited through `cancelled_or_expired`.
    /// Dropping the future mid-wait releases the waiter count through
    /// `WaitingAcquire` and records the lease as cancelled through
    /// `SlotRequest::drop`.
    pub(crate) async fn reserve_async(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Option<Instant>,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
        permit_class: StreamPermitClass,
    ) -> Result<StreamSlot> {
        let request = SlotRequest::new(self, Some(cancellation), deadline, operation, observation, scope, permit_class);
        loop {
            let available = self.inner.available.notified();
            let mut available = pin!(available);
            let _waiting = match request.take_or_register(available.as_mut())? {
                SlotOrWait::Slot(slot) => return Ok(slot),
                SlotOrWait::Wait(waiting) => waiting,
            };
            let deadline_elapsed = async {
                match deadline {
                    Some(deadline) => sleep_until(deadline.into()).await,
                    None => pending().await,
                }
            };
            tokio::select! {
                biased;
                () = available => {},
                () = cancellation.cancelled_or_expired() => {},
                () = deadline_elapsed => {},
            }
        }
    }

    fn build_runtime(&self) -> Result<Runtime> {
        Runtime::builder().with_device(self.device()).with_device_memory_manager().build().map_err(|error| {
            cudf_query_error(
                "query_engine.service_stream_pool.runtime",
                "failed to initialize service stream pool runtime",
                error,
            )
        })
    }

    fn lock_state(&self, operation: &'static str) -> Result<MutexGuard<'_, ServiceStreamPoolState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "service stream pool mutex was poisoned")
        })
    }

    /// Takes the pool mutex without letting contention extend past
    /// `deadline`; on the shutdown paths a stuck holder then produces a typed
    /// timeout where a plain lock would block without bound. A poisoned mutex
    /// is reported as the ordinary poison error whatever the deadline.
    fn lock_state_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, ServiceStreamPoolState>> {
        loop {
            match self.inner.state.try_lock() {
                Ok(state) => return Ok(state),
                Err(TryLockError::Poisoned(_)) => return self.lock_state(operation),
                Err(TryLockError::WouldBlock) if Instant::now() >= deadline => {
                    return Err(QueryError::new(
                        QueryErrorCode::ServiceOverloaded,
                        operation,
                        "timed out acquiring service stream pool state during shutdown",
                    )
                    .with_fact("reason", "stream_pool_idle_timeout")
                    .with_fact("device_ordinal", self.device().ordinal()));
                },
                Err(TryLockError::WouldBlock) => thread::yield_now(),
            }
        }
    }

    /// Returns a `SlotAction::Create` reservation whose runtime was never
    /// leased: the physical count and the class permit both go back, and
    /// waiters are woken because either may have been the limiting resource.
    fn discard_created_slot(&self, permit_class: StreamPermitClass) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.created = state.created.saturating_sub(1);
                match permit_class {
                    StreamPermitClass::Root => state.root_outstanding = state.root_outstanding.saturating_sub(1),
                    StreamPermitClass::Auxiliary => state.aux_outstanding = state.aux_outstanding.saturating_sub(1),
                }
                self.inner.available.notify_waiters();
                self.inner.state_changed.notify_all();
            },
            Err(_error) => {
                eprintln!("[query-engine] warning: failed to update poisoned service stream pool state");
            },
        }
    }
}

/// One reservation attempt against a pool. A request dropped before it hands
/// out a slot, such as an abandoned async reservation, records its lease as
/// cancelled.
struct SlotRequest<'a> {
    pool: &'a ServiceStreamPool,
    cancellation: Option<&'a ExecutionCancellation>,
    deadline: Option<Instant>,
    operation: &'static str,
    permit_class: StreamPermitClass,
    observation: Arc<StreamLeaseObservation>,
    enqueued_at: Instant,
    handed_out: Cell<bool>,
}

enum SlotOrWait<'a> {
    Slot(StreamSlot),
    Wait(WaitingAcquire<'a>),
}

/// Counts one waiting acquire in the pool state until it drops, including
/// when an async reservation is abandoned mid-wait; shutdown waits for the
/// count to reach zero.
struct WaitingAcquire<'a> {
    pool: &'a ServiceStreamPool,
}

impl Drop for WaitingAcquire<'_> {
    fn drop(&mut self) {
        let mut state = self.pool.inner.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.waiting_acquires = state.waiting_acquires.saturating_sub(1);
        self.pool.inner.state_changed.notify_all();
    }
}

impl<'a> SlotRequest<'a> {
    fn new(
        pool: &'a ServiceStreamPool,
        cancellation: Option<&'a ExecutionCancellation>,
        deadline: Option<Instant>,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
        permit_class: StreamPermitClass,
    ) -> Self {
        let observation = Arc::new(StreamLeaseObservation::new(pool, observation, scope));
        observation.define_resource();
        observation.record(StreamLeaseLifecycleState::Enqueued);
        Self {
            pool,
            cancellation,
            deadline,
            operation,
            permit_class,
            observation,
            enqueued_at: Instant::now(),
            handed_out: Cell::new(false),
        }
    }

    /// Takes a slot if one is free. Otherwise enables `available` while the
    /// pool lock is held, so a release after this check wakes the waiter,
    /// and counts the waiter while the returned registration lives.
    fn take_or_register(&self, available: Pin<&mut Notified<'_>>) -> Result<SlotOrWait<'a>> {
        let pool = self.pool;
        let operation = self.operation;
        if pool.inner.closed.load(Ordering::Acquire) {
            return Err(self.fail(stream_pool_closed(operation, pool)));
        }
        if let Some(cancellation) = self.cancellation
            && let Err(error) = cancellation.check(operation)
        {
            self.observation.record(StreamLeaseLifecycleState::CancellationRequested);
            self.observation.terminal(StreamLeaseLifecycleState::Cancelled);
            return Err(error);
        }
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return Err(self.fail(stream_acquire_timeout_error(operation, pool.device(), pool.pool_size())));
        }
        let mut state = pool.lock_state(operation)?;
        // Rechecked under the mutex. A request that passes here holds its
        // slot before `close` can take the idle set, and its lease is then
        // destroyed by cleanup, never idled; a request that arrives after
        // the flag store fails even if idle runtimes are still present.
        if pool.inner.closed.load(Ordering::Acquire) {
            return Err(self.fail(stream_pool_closed(operation, pool)));
        }
        // Reusing an idle runtime on a faulted device is as unsafe as
        // creating one, so the health check precedes both actions.
        let health = device_health_snapshot(pool.device());
        if health.status() == DeviceHealthStatus::Unhealthy {
            return Err(self.fail(unhealthy_stream_device_error(operation, pool.device(), health)));
        }
        // The class check comes first: a full class waits even while idle
        // runtimes or physical capacity are available to the other class.
        let class_open = match self.permit_class {
            StreamPermitClass::Root => state.root_outstanding < pool.inner.root_capacity,
            StreamPermitClass::Auxiliary => state.aux_outstanding < pool.inner.aux_capacity,
        };
        let action = if !class_open {
            None
        } else if let Some(runtime) = state.idle.pop() {
            Some(SlotAction::Reuse(runtime))
        } else if state.created < pool.pool_size() {
            // The physical slot is reserved here; the CUDA stream is built
            // outside the mutex in `StreamSlot::into_lease`.
            state.created += 1;
            Some(SlotAction::Create)
        } else {
            None
        };
        if let Some(action) = action {
            match self.permit_class {
                StreamPermitClass::Root => state.root_outstanding += 1,
                StreamPermitClass::Auxiliary => state.aux_outstanding += 1,
            }
            pool.inner.state_changed.notify_all();
            self.handed_out.set(true);
            return Ok(SlotOrWait::Slot(StreamSlot {
                pool: pool.clone(),
                action: Some(action),
                permit_class: self.permit_class,
                observation: Arc::clone(&self.observation),
                enqueued_at: self.enqueued_at,
            }));
        }
        available.enable();
        state.waiting_acquires = state.waiting_acquires.saturating_add(1);
        pool.inner.state_changed.notify_all();
        Ok(SlotOrWait::Wait(WaitingAcquire { pool }))
    }

    fn fail(&self, error: QueryError) -> QueryError {
        self.observation.terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
        error
    }
}

impl Drop for SlotRequest<'_> {
    fn drop(&mut self) {
        if !self.handed_out.get() {
            self.observation.terminal(StreamLeaseLifecycleState::Cancelled);
        }
    }
}

enum SlotAction {
    Reuse(Arc<Runtime>),
    Create,
}

/// A reserved stream slot: an idle runtime to reuse, or the right to create
/// one. [`Self::into_lease`] turns it into a lease; dropping it unused returns
/// the slot to the pool.
pub(crate) struct StreamSlot {
    pool: ServiceStreamPool,
    action: Option<SlotAction>,
    permit_class: StreamPermitClass,
    observation: Arc<StreamLeaseObservation>,
    enqueued_at: Instant,
}

impl fmt::Debug for StreamSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("StreamSlot").field("permit_class", &self.permit_class).finish_non_exhaustive()
    }
}

impl StreamSlot {
    /// Leases the reserved runtime, creating its CUDA stream when the slot
    /// carries the right to create one.
    ///
    /// This is the one CUDA call on the acquire path, so it must run on a
    /// thread where CUDA work is allowed. On any failure the reservation is
    /// returned to the pool and the lease observation records `Failed`.
    pub(crate) fn into_lease(mut self, operation: &'static str) -> Result<StreamLease> {
        let pool = self.pool.clone();
        let (runtime, acquisition) = match self.action.take().expect("an unconsumed slot holds its action") {
            SlotAction::Reuse(runtime) => (runtime, StreamLeaseAcquisition::Reused),
            SlotAction::Create => match pool.build_runtime() {
                Ok(runtime) if !pool.inner.closed.load(Ordering::Acquire) => {
                    (Arc::new(runtime), StreamLeaseAcquisition::Created)
                },
                // The pool closed while the stream was being built. The
                // runtime never carried work, so it is destroyed here rather
                // than routed through cleanup.
                Ok(runtime) => {
                    pool.discard_created_slot(self.permit_class);
                    // Return the slot before the stream destroy so a waiter
                    // is not delayed by this CUDA call.
                    drop(runtime);
                    let error = stream_pool_closed(operation, &pool);
                    self.observation
                        .terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                    return Err(error);
                },
                Err(error) => {
                    pool.discard_created_slot(self.permit_class);
                    self.observation
                        .terminal(StreamLeaseLifecycleState::Failed { error_code: error.code().to_string() });
                    return Err(error);
                },
            },
        };
        self.observation.record(StreamLeaseLifecycleState::Acquired {
            acquisition,
            host_wait_ns: u64::try_from(self.enqueued_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
        });
        Ok(StreamLease {
            pool,
            runtime: Some(runtime),
            observation: Some(Arc::clone(&self.observation)),
            resource_epoch: None,
            cleanup_completion: Mutex::new(None),
            permit_class: self.permit_class,
        })
    }
}

impl Drop for StreamSlot {
    /// Returns an unconsumed reservation. Even a reused runtime returns through
    /// cleanup, so the class permit and the idle return follow the single
    /// accounting path in `stream_cleanup`.
    fn drop(&mut self) {
        let Some(action) = self.action.take() else { return };
        self.observation.terminal(StreamLeaseLifecycleState::Cancelled);
        match action {
            SlotAction::Reuse(runtime) => self.pool.enqueue_cleanup(runtime, None, None, self.permit_class),
            SlotAction::Create => self.pool.discard_created_slot(self.permit_class),
        }
    }
}

fn stream_acquire_timeout_error(operation: &'static str, device: DeviceId, pool_size: usize) -> QueryError {
    QueryError::new(QueryErrorCode::ServiceOverloaded, operation, "service stream pool acquire timed out")
        .with_fact("reason", "stream_pool_wait_timeout")
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("stream_pool_size", pool_size)
}

fn unhealthy_stream_device_error(
    operation: &'static str,
    device: DeviceId,
    health: DeviceHealthSnapshot,
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
    // Built on error paths that may already hold the pool mutex, so the
    // failure count is attached only when the lock is free.
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

/// Observation identity of one lease attempt, from `Enqueued` to its terminal
/// state. Shared by the `SlotRequest`, the `StreamSlot`, and the `StreamLease`
/// so that whichever of them ends the attempt records one terminal event.
#[derive(Debug)]
struct StreamLeaseObservation {
    observation: RuntimeObservationHandle,
    lease_id: RuntimeResourceId,
    device_ordinal: u32,
    scope: RuntimeStreamLeaseScope,
    /// Set by the first `terminal` call; later calls are no-ops.
    terminal_recorded: AtomicBool,
}

impl StreamLeaseObservation {
    fn new(pool: &ServiceStreamPool, observation: RuntimeObservationHandle, scope: RuntimeStreamLeaseScope) -> Self {
        let sequence = pool.inner.next_lease_id.fetch_add(1, Ordering::Relaxed);
        Self {
            observation,
            lease_id: RuntimeResourceId::new(format!("stream-pool-{}-lease-{sequence}", pool.inner.id)),
            device_ordinal: u32::try_from(pool.device().ordinal()).unwrap_or(u32::MAX),
            scope,
            terminal_recorded: AtomicBool::new(false),
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
        self.observation.record_topology(RuntimeObservationRecordClass::Resource, || {
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
    /// `None` only after `Drop` has moved the runtime into cleanup.
    runtime: Option<Arc<Runtime>>,
    observation: Option<Arc<StreamLeaseObservation>>,
    /// Query-service epoch this lease is bound to, when it was obtained
    /// through an admitted context. `runtime()` refuses to derive runtimes
    /// once the epoch closes, and the cleanup job retains the epoch until the
    /// runtime is destroyed.
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
    /// At most one completion, installed through `&self` because the lease
    /// is already shared by its execution contexts when the release path
    /// installs it.
    cleanup_completion: Mutex<Option<StreamCleanupCompletion>>,
    permit_class: StreamPermitClass,
}

impl fmt::Debug for StreamLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    /// Installs the callbacks that the cleanup thread runs after this lease's
    /// stream has been synchronized. Returns `false`, and runs nothing, when a
    /// completion is already installed.
    ///
    /// `before_quiescence` receives the synchronization outcome and runs while
    /// the cleanup still counts as active, so a quiescence waiter cannot pass
    /// until it returns; the attempt's report commit and capacity release
    /// happen there. `after_quiescence` runs once the counters are released.
    /// Both run on the cleanup thread, or on the dropping thread when the
    /// enqueue falls back to synchronous cleanup. If the cleanup thread dies
    /// first, `before_quiescence` still runs with `CallbackAbandoned`.
    pub(crate) fn install_cleanup_completion(
        &self,
        before_quiescence: impl FnOnce(StreamCleanupOutcome) + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> bool {
        // Materialize the handoff before inspecting the slot. A duplicate
        // installation is an ownership invariant violation; the rejected
        // completion is dropped, which reports `CallbackAbandoned` and runs
        // its tail, so the report and health work it carried is not lost and
        // the query's capacity is not released without a terminal commit.
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

    pub(crate) fn device(&self) -> DeviceId {
        self.runtime.as_ref().expect("stream lease runtime is present").device()
    }

    /// Clone a counted cuDF runtime derived from this lease.
    ///
    /// Fails once the bound query-service epoch has closed. The returned
    /// runtime retains that epoch, and so does every cuDF value produced from
    /// it, which keeps the service generation alive while GPU state derived
    /// from this stream exists.
    pub fn runtime(&self) -> Result<Runtime> {
        let runtime = self.runtime.as_ref().expect("stream lease runtime is present").as_ref().clone();
        let Some(resource_epoch) = &self.resource_epoch else {
            return Ok(runtime);
        };
        resource_epoch.check_open("query_engine.service_stream_pool.lease.runtime")?;
        let retained: Arc<dyn Send + Sync> = Arc::clone(resource_epoch) as _;
        runtime.with_retained_lifetime(retained).map_err(|error| {
            cudf_query_error(
                "query_engine.service_stream_pool.lease.runtime",
                "failed to retain query-service lifetime for leased runtime",
                error,
            )
        })
    }

    /// Clone the shared runtime handle for a short-lived execution context.
    pub fn shared_runtime(&self) -> Result<Arc<Runtime>> {
        self.runtime().map(Arc::new)
    }
}

impl Drop for StreamLease {
    /// Runs no CUDA call. The lease may be dropped on a host-runtime task, a
    /// pool worker, or inside another `Drop`; the stream synchronization that
    /// must precede reuse happens on the cleanup thread.
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

/// One unit of auxiliary attempt work: its attempt auxiliary permit, its
/// service stream lease, and the execution context derived from both.
///
/// Execution lanes and scheduled source reads hold the same lease; a lane's
/// context additionally carries that lane's fork subset.
pub struct AttemptWorkLease {
    context: ExecutionContext,
    _stream_lease: StreamLease,
    // Drop after the derived context and global stream lease so the
    // attempt-local envelope remains charged for every actual auxiliary
    // stream lifetime.
    _attempt_auxiliary_permit: Option<AttemptAuxiliaryPermit>,
}

impl AttemptWorkLease {
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
            QueryErrorCode::InvalidConfiguration,
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
    use std::time::Duration;

    use cudf::{
        runtime::{DeviceFaultKind, DeviceFaultNativeStatus, report_device_fault},
        testing::device_safety::DeviceSafetyTestScope,
    };
    use query_planning::error::QueryErrorStatus;

    use super::*;
    use crate::observability::{
        PlanExecutionId, RuntimeJobId, RuntimeObservationConfig, RuntimeObservationOwner, RuntimeTaskId,
        RuntimeWorkerId, assert_live_topology_references_resolve,
    };

    #[test]
    fn zero_capacity_rejects_before_runtime_initialization() {
        let error = validate_service_stream_pool_size(0).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(error.operation(), "query_engine.service_stream_pool.validate");
        assert_eq!(error.fact_value("field"), Some("service_stream_pool_size"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    fn observed_stream_states(owner: &RuntimeObservationOwner) -> Vec<StreamLeaseLifecycleState> {
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
        assert_eq!(error.status(), QueryErrorStatus::Permanent);
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
        assert_eq!(error.status(), QueryErrorStatus::Permanent);
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

    fn wait_until_idle_commits_an_already_finished_cleanup_panic_case() {
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
    fn wait_until_idle_commits_an_already_finished_cleanup_panic() {
        wait_until_idle_commits_an_already_finished_cleanup_panic_case();
        wait_until_quiescent_commits_an_already_finished_cleanup_panic();
    }

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
        assert_eq!(error.status(), QueryErrorStatus::Permanent);
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
        let after_pool = pool;
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
        let _scope = DeviceSafetyTestScope::install([]);
        report_device_fault(
            cudf::DeviceId::DEFAULT,
            DeviceFaultKind::FatalDeviceOperation,
            "injected_stream_reuse_fault",
            Some(DeviceFaultNativeStatus::CUDF_FATAL_CUDA),
        );

        let error = pool.acquire().expect_err("unhealthy device must reject idle runtime reuse");
        assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
        assert_eq!(error.fact_value("reason"), Some("device_health_rejected_stream_acquire"));
    }

    #[test]
    fn observed_stream_acquire_distinguishes_create_reuse_and_release() {
        let pool = ServiceStreamPool::new(cudf::DeviceId::DEFAULT, 1).unwrap();
        let owner =
            RuntimeObservationOwner::new("stream-lifecycle", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
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
        assert_live_topology_references_resolve(&owner.live_snapshot().unwrap());
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
        let owner =
            RuntimeObservationOwner::new("stream-cancel", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
        let waiter_pool = pool.clone();
        let waiter_cancellation = cancellation.clone();
        let observation = owner.handle();
        let (started_sender, started_receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            started_sender.send(()).unwrap();
            waiter_pool.acquire_observed_with_cancellation_deadline(
                &waiter_cancellation,
                Some(Instant::now() + Duration::from_secs(1)),
                "query_engine.service_stream_pool.test_observed_cancel",
                observation,
                RuntimeStreamLeaseScope::for_plan(PlanExecutionId::new("stream-plan"))
                    .with_source_worker(RuntimeTaskId::new("source-task"), RuntimeWorkerId::new("source-worker"))
                    .with_gpu_worker(RuntimeJobId::new("gpu-job"), RuntimeWorkerId::new("gpu-worker")),
            )
        });
        started_receiver.recv().unwrap();
        pool.wait_until_acquire_queued_for_test(Instant::now() + Duration::from_secs(5));
        cancellation.cancel();

        let error = waiter.join().unwrap().unwrap_err();
        drop(held);
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.operation(), "query_engine.service_stream_pool.test_observed_cancel");
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

    fn root_class_waits_while_auxiliary_slots_are_free_case() {
        let pool = ServiceStreamPool::new_with_classes(cudf::DeviceId::DEFAULT, 3, 1, 2).unwrap();
        let root = pool.acquire_with_class(StreamPermitClass::Root).unwrap();
        let aux_one = pool.acquire_with_class(StreamPermitClass::Auxiliary).unwrap();
        let aux_two = pool.acquire_with_class(StreamPermitClass::Auxiliary).unwrap();
        let (root_out, root_cap, aux_out, aux_cap) = pool.class_counters();
        assert_eq!((root_out, root_cap, aux_out, aux_cap), (1, 1, 2, 2));

        let (sender, receiver) = std::sync::mpsc::channel();
        let second_pool = pool;
        let waiter = std::thread::spawn(move || {
            sender.send(()).unwrap();
            second_pool.acquire_observed_with_cancellation_deadline_and_class(
                &ExecutionCancellation::new(),
                Some(Instant::now() + std::time::Duration::from_millis(200)),
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
    fn root_class_waits_while_auxiliary_slots_are_free() {
        root_class_waits_while_auxiliary_slots_are_free_case();
        auxiliary_class_waits_while_root_slots_are_free();
    }

    fn auxiliary_class_waits_while_root_slots_are_free() {
        let pool = ServiceStreamPool::new_with_classes(cudf::DeviceId::DEFAULT, 3, 1, 1).unwrap();
        let _root = pool.acquire_with_class(StreamPermitClass::Root).unwrap();
        let aux = pool.acquire_with_class(StreamPermitClass::Auxiliary).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let second_pool = pool;
        let waiter = std::thread::spawn(move || {
            sender.send(()).unwrap();
            second_pool.acquire_observed_with_cancellation_deadline_and_class(
                &ExecutionCancellation::new(),
                Some(Instant::now() + std::time::Duration::from_millis(200)),
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
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(error.fact_value("field"), Some("root_capacity"));
        let error = ServiceStreamPool::new_with_classes(cudf::DeviceId::DEFAULT, 2, 1, 0).unwrap_err();
        assert_eq!(error.fact_value("field"), Some("aux_capacity"));
    }
}
