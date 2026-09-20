//! Native execution context state and query-local runtime controls.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use ::object_store::ObjectStoreExt;
#[cfg(any(test, feature = "test-hooks"))]
use bytes::Bytes;
use query_planning::{
    admission::AttemptCapabilityEvidence,
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    mode::StatisticalAggregateExecutionMode,
    plan::QueryPlan,
};

use super::{
    AttemptWorkLease, CacheFillByteBudget, CacheFillBytePermit, DeploymentIoCapability, DevicePlacement,
    ExecutionBoundaryPolicy, GpuExecutionPool, GpuExecutorLease, LakehouseCacheFillConfig, MemoryControlPolicy,
    MemoryGrant, MemoryResourcePolicy, MemorySpaceId, ServiceStreamPool, SourceChunkPlanningMode,
    SourceSchedulerPolicy, SourceStorageProfile, StreamLease,
    allocator_correlation::AttemptCacheScopeAllocatorCorrelation,
    attempt_domain::{AttemptAllocationDomain, AttemptStreamProviderRoute},
    cache_events::{LakehouseCacheFillEventSink, LakehouseCacheFillEventSnapshot},
    cache_fill::{CacheFillConcurrencyGate, CacheFillOperationPermit},
    default_service_stream_pool_size_for_device,
    gpu::ledger::ResidentOwnerLease,
    io_lifecycle::{IoLifecycle, IoLifecycleLease},
    object_store::{
        ObjectStoreCustomFetchResult, ObjectStoreFetchResult, S3ObjectStoreKey, SourceObjectStoreRegistry,
        default_object_store_control_runtime,
    },
    resident::{GpuResidentInputMap, GpuResultHandle},
};
use crate::{
    metrics::{QueryId, TraceId},
    native_error::cudf_query_error,
    observability::{
        AttemptComposedScheduleProjection, PlanExecutionId, RuntimeExecutionObservationScope, RuntimeObservationHandle,
        RuntimePlanExecutionLifecycle, RuntimeQueryLifecycle, RuntimeStreamLeaseScope, RuntimeTaskId,
        RuntimeTerminalOutcome, RuntimeTraceId, RuntimeWorkerId, SealedRuntimeObservationSnapshot,
    },
};

mod builder;

#[doc(hidden)]
pub use builder::ExecutionContextBuilder;
pub(super) use builder::validate_service_stream_pool_device;
pub use builder::{RuntimeObservationBuildError, RuntimeObservedExecutionContextBuilder};

/// Shared cooperative cancellation flag for native query execution.
///
/// Cancellation is observed only at native-engine boundaries. It does not
/// preempt a libcudf/CUDA call already in flight, but it prevents the next
/// source chunk, pipeline task, operator, or materialization step from starting
/// once the flag is set.
#[derive(Debug, Clone)]
pub struct ExecutionCancellation {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl ExecutionCancellation {
    /// Creates a non-cancelled execution flag.
    pub fn new() -> Self {
        Self::with_deadline(None)
    }

    /// Creates a cancellation token with an optional absolute deadline.
    ///
    /// The deadline is checked at the same cooperative native-engine
    /// boundaries as caller cancellation. It cannot preempt a CUDA call
    /// already in flight, but it prevents the next safe boundary from
    /// starting more native work after the deadline has elapsed.
    pub fn with_deadline(deadline: Option<Instant>) -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)), deadline }
    }

    /// Marks the execution as cancelled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether this execution must stop at its next cooperative safe
    /// boundary.
    ///
    /// An elapsed deadline is intentionally an effective cancellation here.
    /// Callers that need to distinguish an explicit caller action from expiry
    /// must use [`Self::check`], which preserves explicit cancellation as the
    /// primary reason.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_requested() || self.is_expired()
    }

    /// Returns the optional absolute deadline shared by every clone.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Returns whether the optional deadline has elapsed.
    pub fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Returns a structured native error if cancellation has been requested.
    pub fn check(&self, operation: &'static str) -> Result<()> {
        if self.cancellation_requested() {
            return Err(QueryError::new(
                query_planning::error::QueryErrorCode::Cancelled,
                operation,
                "native query execution was cancelled",
            )
            .with_fact("cancellation_requested", true));
        }
        if self.is_expired() {
            return Err(QueryError::new(
                query_planning::error::QueryErrorCode::Cancelled,
                operation,
                "native query execution deadline expired",
            )
            .with_fact("cancellation_requested", true)
            .with_fact("reason", "deadline_expired"));
        }
        Ok(())
    }

    fn cancellation_requested(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Default for ExecutionCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::observability::RuntimeExecutionStageContext for ExecutionContext {
    fn runtime_operator_stage_context(
        &self,
    ) -> (
        crate::observability::RuntimeObservationHandle,
        Option<crate::observability::PlanExecutionId>,
        Option<crate::observability::RuntimeSpanId>,
    ) {
        let observation = self.runtime_observation().clone();
        let (plan_execution_id, parent_span_id) = self
            .runtime_execution_scope()
            .map(|scope| (Some(scope.plan_execution_id().clone()), Some(scope.plan_span_id())))
            .unwrap_or((None, None));
        (observation, plan_execution_id, parent_span_id)
    }
}

const ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Attempt-owned capacity for streams outside the admission envelope's root
/// binding.
///
/// The device ledger reserves `source_work_per_attempt` auxiliary slots for
/// every admitted attempt. This gate materializes that reservation locally so
/// one attempt cannot retain another attempt's share of the global auxiliary
/// stream pool.
#[derive(Debug)]
#[doc(hidden)]
pub struct AttemptAuxiliarySlots {
    device_ordinal: usize,
    capacity: usize,
    state: Mutex<AttemptAuxiliarySlotState>,
    available: Condvar,
    async_available: tokio::sync::Notify,
}

#[derive(Debug)]
struct AttemptAuxiliarySlotState {
    in_use: usize,
}

/// RAII ownership of one admitted attempt's auxiliary stream allowance.
#[derive(Debug)]
#[doc(hidden)]
pub struct AttemptAuxiliaryPermit {
    slots: Arc<AttemptAuxiliarySlots>,
}

impl AttemptAuxiliarySlots {
    #[doc(hidden)]
    pub fn new(device_ordinal: usize, capacity: usize) -> Arc<Self> {
        debug_assert!(capacity > 0, "admission profiles normalize auxiliary slot capacity to at least one");
        Arc::new(Self {
            device_ordinal,
            capacity: capacity.max(1),
            state: Mutex::new(AttemptAuxiliarySlotState { in_use: 0 }),
            available: Condvar::new(),
            async_available: tokio::sync::Notify::new(),
        })
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    #[doc(hidden)]
    pub fn acquire_with_cancellation(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        self.acquire_inner(Some(cancellation), None, operation)
    }

    #[doc(hidden)]
    pub fn acquire_with_cancellation_deadline(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        self.acquire_inner(Some(cancellation), Some(deadline), operation)
    }

    /// Takes one free slot, or reports that none is free right now.
    ///
    /// A holder of an attempt auxiliary permit must never block on another
    /// permit of the same attempt, so work that already owns one takes what
    /// is left over instead of waiting for it.
    #[doc(hidden)]
    pub fn try_acquire(self: &Arc<Self>, operation: &'static str) -> Result<Option<AttemptAuxiliaryPermit>> {
        let mut state = self.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "attempt auxiliary-slot mutex was poisoned")
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "attempt_auxiliary_slots_poisoned")
        })?;
        if state.in_use < self.capacity {
            state.in_use += 1;
            return Ok(Some(AttemptAuxiliaryPermit { slots: Arc::clone(self) }));
        }
        Ok(None)
    }

    /// Asynchronously acquires one attempt-local auxiliary slot without
    /// parking the caller's Tokio worker while another fragment owns it.
    #[doc(hidden)]
    pub async fn acquire_with_cancellation_deadline_async(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        loop {
            cancellation.check(operation)?;
            // Register before observing availability so a permit return in
            // this interval cannot be lost.
            let available = self.async_available.notified();
            let in_use = {
                let mut state = self.state.lock().map_err(|_error| {
                    QueryError::new(QueryErrorCode::Internal, operation, "attempt auxiliary-slot mutex was poisoned")
                        .with_fact("device_ordinal", self.device_ordinal)
                        .with_fact("reason", "attempt_auxiliary_slots_poisoned")
                })?;
                if state.in_use < self.capacity {
                    state.in_use += 1;
                    return Ok(AttemptAuxiliaryPermit { slots: Arc::clone(self) });
                }
                state.in_use
            };
            if Instant::now() >= deadline {
                return Err(self.exhausted_error(operation, in_use, true));
            }
            let wait =
                deadline.saturating_duration_since(Instant::now()).min(ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL);
            let _ = tokio::time::timeout(wait, available).await;
        }
    }

    fn acquire_inner(
        self: &Arc<Self>,
        cancellation: Option<&ExecutionCancellation>,
        deadline: Option<Instant>,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        loop {
            if let Some(cancellation) = cancellation {
                cancellation.check(operation)?;
            }
            let mut state = self.state.lock().map_err(|_error| {
                QueryError::new(QueryErrorCode::Internal, operation, "attempt auxiliary-slot mutex was poisoned")
                    .with_fact("device_ordinal", self.device_ordinal)
                    .with_fact("reason", "attempt_auxiliary_slots_poisoned")
            })?;
            if state.in_use < self.capacity {
                state.in_use += 1;
                return Ok(AttemptAuxiliaryPermit { slots: Arc::clone(self) });
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(self.exhausted_error(operation, state.in_use, true));
            }
            let wait = deadline.map_or(ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL, |deadline| {
                deadline.saturating_duration_since(Instant::now()).min(ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL)
            });
            let (_state, _timeout) = self.available.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "attempt auxiliary-slot mutex was poisoned while waiting for capacity",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "attempt_auxiliary_slots_poisoned")
            })?;
        }
    }

    fn exhausted_error(&self, operation: &'static str, in_use: usize, deadline_elapsed: bool) -> QueryError {
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            operation,
            "admitted attempt auxiliary-stream capacity is exhausted",
        )
        .with_fact("device_ordinal", self.device_ordinal)
        .with_fact("attempt_auxiliary_capacity", self.capacity)
        .with_fact("attempt_auxiliary_in_use", in_use)
        .with_fact("deadline_elapsed", deadline_elapsed)
        .with_fact(
            "reason",
            if deadline_elapsed { "attempt_auxiliary_slot_wait_timeout" } else { "attempt_auxiliary_slots_exhausted" },
        )
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[cfg_attr(not(test), expect(dead_code, reason = "used by in-crate auxiliary-slot tests"))]
    fn in_use_for_test(&self) -> usize {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).in_use
    }
}

impl Drop for AttemptAuxiliaryPermit {
    fn drop(&mut self) {
        let mut state = self.slots.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.in_use > 0, "attempt auxiliary-slot permit underflow");
        state.in_use = state.in_use.saturating_sub(1);
        self.slots.available.notify_one();
        self.slots.async_available.notify_one();
    }
}

#[cfg(test)]
mod attempt_auxiliary_slots_tests {
    use super::*;

    #[test]
    fn execution_cancellation_preserves_deadline_across_clones_and_keeps_caller_cancel_first() {
        let deadline = Instant::now() - Duration::from_millis(1);
        let cancellation = ExecutionCancellation::with_deadline(Some(deadline));
        let clone = cancellation.clone();
        assert_eq!(clone.deadline(), Some(deadline));
        assert!(clone.is_expired());
        assert!(clone.is_cancelled(), "an elapsed deadline must stop native safe boundaries");
        let deadline_error = clone
            .check("query_engine.test.execution_cancellation_deadline")
            .expect_err("expired execution deadline must stop a safe boundary");
        assert_eq!(deadline_error.fact_value("reason"), Some("deadline_expired"));

        cancellation.cancel();
        let caller_error = clone
            .check("query_engine.test.execution_cancellation_caller_cancel")
            .expect_err("explicit caller cancellation remains terminal");
        assert_eq!(caller_error.fact_value("reason"), None, "caller cancellation takes precedence over expiry");
    }

    #[test]
    fn held_slot_rejects_cancelled_and_expired_waiters_then_releases() {
        let slots = AttemptAuxiliarySlots::new(7, 1);
        let active = ExecutionCancellation::new();
        let held = slots
            .acquire_with_cancellation(&active, "query_engine.test.attempt_auxiliary_slots")
            .expect("first auxiliary permit fits the attempt envelope");
        assert_eq!(slots.in_use_for_test(), 1);

        let cancelled = ExecutionCancellation::new();
        cancelled.cancel();
        let cancelled_error = slots
            .acquire_with_cancellation(&cancelled, "query_engine.test.attempt_auxiliary_slots")
            .expect_err("cancelled waiter must not exceed the attempt envelope");
        assert_eq!(cancelled_error.code(), QueryErrorCode::Cancelled);

        let deadline_error = slots
            .acquire_with_cancellation_deadline(&active, Instant::now(), "query_engine.test.attempt_auxiliary_slots")
            .expect_err("expired waiter must not exceed the attempt envelope");
        assert_eq!(deadline_error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(deadline_error.fact_value("reason"), Some("attempt_auxiliary_slot_wait_timeout"));
        assert_eq!(deadline_error.fact_value("attempt_auxiliary_capacity"), Some("1"));
        assert_eq!(deadline_error.fact_value("attempt_auxiliary_in_use"), Some("1"));
        assert_eq!(deadline_error.fact_value("deadline_elapsed"), Some("true"));

        drop(held);
        let released = slots
            .acquire_with_cancellation(&active, "query_engine.test.attempt_auxiliary_slots")
            .expect("dropping the permit returns this attempt's capacity");
        assert_eq!(slots.in_use_for_test(), 1);
        drop(released);
        assert_eq!(slots.in_use_for_test(), 0);
    }

    #[test]
    fn each_attempt_keeps_its_own_auxiliary_share() {
        let first_attempt = AttemptAuxiliarySlots::new(3, 1);
        let second_attempt = AttemptAuxiliarySlots::new(3, 1);
        let cancellation = ExecutionCancellation::new();
        let first = first_attempt
            .acquire_with_cancellation(&cancellation, "query_engine.test.attempt_auxiliary_slots")
            .expect("first attempt reserves its only slot");

        let first_exhausted = first_attempt
            .acquire_with_cancellation_deadline(
                &cancellation,
                Instant::now(),
                "query_engine.test.attempt_auxiliary_slots",
            )
            .expect_err("one attempt cannot borrow another attempt's auxiliary share");
        assert_eq!(first_exhausted.fact_value("reason"), Some("attempt_auxiliary_slot_wait_timeout"));

        let second = second_attempt
            .acquire_with_cancellation(&cancellation, "query_engine.test.attempt_auxiliary_slots")
            .expect("second attempt retains its independently admitted share");
        assert_eq!(first_attempt.in_use_for_test(), 1);
        assert_eq!(second_attempt.in_use_for_test(), 1);
        drop(first);
        drop(second);
    }
}

type QueryServiceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;
type QueryResourceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;

/// Revocable lifetime shared by every resource derived from one query service.
#[derive(Clone)]
#[doc(hidden)]
pub struct QueryServiceEpoch {
    inner: Arc<QueryServiceEpochInner>,
}

struct QueryServiceEpochInner {
    open: AtomicBool,
    active_users: AtomicUsize,
    idle_state: Mutex<QueryServiceEpochIdleState>,
    idle_changed: Condvar,
}

struct QueryServiceEpochIdleState {
    on_idle: Option<QueryServiceEpochIdleCallback>,
    complete: bool,
}

impl std::fmt::Debug for QueryServiceEpoch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QueryServiceEpoch")
            .field("open", &self.inner.open.load(Ordering::Acquire))
            .field("active_users", &self.inner.active_users.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl QueryServiceEpoch {
    #[doc(hidden)]
    pub fn new(on_idle: Option<QueryServiceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryServiceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                idle_state: Mutex::new(QueryServiceEpochIdleState { on_idle, complete: false }),
                idle_changed: Condvar::new(),
            }),
        }
    }

    #[doc(hidden)]
    pub fn retain_existing_user(&self) -> QueryServiceEpochUser {
        self.inner.active_users.fetch_add(1, Ordering::AcqRel);
        QueryServiceEpochUser { epoch: self.clone() }
    }

    fn acquire(&self, operation: &'static str) -> Result<QueryServiceEpochUser> {
        if !self.inner.open.load(Ordering::Acquire) {
            return Err(query_service_epoch_closed(operation));
        }
        let user = self.retain_existing_user();
        if !self.inner.open.load(Ordering::Acquire) {
            drop(user);
            return Err(query_service_epoch_closed(operation));
        }
        Ok(user)
    }

    #[doc(hidden)]
    pub fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_service_epoch_closed(operation)) }
    }

    #[doc(hidden)]
    pub fn invalidate(&self) {
        self.inner.open.store(false, Ordering::Release);
        self.inner.idle_changed.notify_all();
    }

    fn release_user(&self) {
        let previous = self.inner.active_users.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "query service epoch user count underflow");
        if previous == 1 {
            self.release_if_idle();
        }
    }

    fn release_if_idle(&self) {
        if self.inner.open.load(Ordering::Acquire) || self.inner.active_users.load(Ordering::Acquire) != 0 {
            return;
        }
        let mut state = self.inner.idle_state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.complete_idle_locked(&mut state);
    }

    fn complete_idle_locked(&self, state: &mut QueryServiceEpochIdleState) {
        if state.complete
            || self.inner.open.load(Ordering::Acquire)
            || self.inner.active_users.load(Ordering::Acquire) != 0
        {
            return;
        }
        let callback = state.on_idle.take();
        if let Some(callback) = callback {
            callback();
        }
        state.complete = true;
        self.inner.idle_changed.notify_all();
    }

    #[doc(hidden)]
    pub fn wait_for_idle_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let mut state = match crate::runtime::deadline::mutex_lock_until(&self.inner.idle_state, deadline) {
            Ok(state) => state,
            Err(crate::runtime::deadline::DeadlineMutexLockError::Poisoned) => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service resource epoch idle mutex was poisoned during shutdown",
                ));
            },
            Err(crate::runtime::deadline::DeadlineMutexLockError::TimedOut) => {
                return Err(query_service_epoch_idle_timeout(
                    operation,
                    self.inner.active_users.load(Ordering::Acquire),
                ));
            },
        };
        let mut waited_for_idle = false;
        let mut completed_here = false;
        loop {
            if !state.complete
                && !self.inner.open.load(Ordering::Acquire)
                && self.inner.active_users.load(Ordering::Acquire) == 0
            {
                self.complete_idle_locked(&mut state);
                completed_here = true;
            }
            if state.complete {
                if (waited_for_idle || completed_here) && crate::runtime::deadline::deadline_expired(deadline) {
                    return Err(query_service_epoch_idle_timeout(
                        operation,
                        self.inner.active_users.load(Ordering::Acquire),
                    ));
                }
                return Ok(());
            }
            if crate::runtime::deadline::deadline_expired(deadline) {
                return Err(query_service_epoch_idle_timeout(
                    operation,
                    self.inner.active_users.load(Ordering::Acquire),
                ));
            }
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return Err(query_service_epoch_idle_timeout(
                    operation,
                    self.inner.active_users.load(Ordering::Acquire),
                ));
            };
            waited_for_idle = true;
            let (next, _timeout) = self.inner.idle_changed.wait_timeout(state, remaining).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service resource epoch condition variable was poisoned during shutdown",
                )
            })?;
            state = next;
        }
    }
}

fn query_service_epoch_closed(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "query service execution epoch is closed")
        .with_fact("reason", "query_service_epoch_closed")
}

fn query_service_epoch_idle_timeout(operation: &'static str, active_users: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for query service resource epoch to become idle",
    )
    .with_fact("reason", "query_service_resource_epoch_timeout")
    .with_fact("active_resource_users", active_users)
    .with_cleanup_deadline_elapsed()
}

#[cfg(test)]
mod query_service_epoch_tests;

#[derive(Debug)]
#[doc(hidden)]
pub struct QueryServiceEpochUser {
    epoch: QueryServiceEpoch,
}

/// Revocable lifetime shared by resources derived from one admitted query.
///
/// The epoch owns only revocation and drain accounting. Each counted anchor
/// owns the query lifetime separately, so the epoch cannot form a cycle with
/// the query's uncounted root-context template.
#[derive(Clone)]
#[doc(hidden)]
pub struct QueryResourceEpoch {
    inner: Arc<QueryResourceEpochInner>,
}

struct QueryResourceEpochInner {
    open: AtomicBool,
    active_users: AtomicUsize,
    on_idle: Mutex<Option<QueryResourceEpochIdleCallback>>,
}

impl std::fmt::Debug for QueryResourceEpoch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QueryResourceEpoch")
            .field("open", &self.inner.open.load(Ordering::Acquire))
            .field("active_users", &self.inner.active_users.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl QueryResourceEpoch {
    #[doc(hidden)]
    pub fn new(on_idle: Option<QueryResourceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryResourceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                on_idle: Mutex::new(on_idle),
            }),
        }
    }

    #[doc(hidden)]
    pub fn acquire(&self, owner: Arc<dyn Send + Sync>, operation: &'static str) -> Result<Arc<QueryResourceAnchor>> {
        self.check_open(operation)?;
        self.inner.active_users.fetch_add(1, Ordering::AcqRel);
        let anchor = Arc::new(QueryResourceAnchor { epoch: self.clone(), _owner: owner });
        if let Err(error) = self.check_open(operation) {
            drop(anchor);
            return Err(error);
        }
        Ok(anchor)
    }

    #[doc(hidden)]
    pub fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_resource_epoch_closed(operation)) }
    }

    #[doc(hidden)]
    pub fn invalidate(&self) {
        self.inner.open.store(false, Ordering::Release);
        self.notify_if_idle();
    }

    #[doc(hidden)]
    pub fn is_idle(&self) -> bool {
        self.inner.active_users.load(Ordering::Acquire) == 0
    }

    fn release_user(&self) {
        let previous = self.inner.active_users.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "query resource epoch user count underflow");
        if previous == 1 {
            self.notify_if_idle();
        }
    }

    fn notify_if_idle(&self) {
        if self.inner.open.load(Ordering::Acquire) || !self.is_idle() {
            return;
        }
        let callback = match self.inner.on_idle.lock() {
            Ok(mut callback) => callback.take(),
            Err(error) => error.into_inner().take(),
        };
        if let Some(callback) = callback {
            callback();
        }
    }
}

/// One counted outward ownership root for an admitted query.
///
/// Clones of the surrounding context/resource share this anchor. The owner is
/// deliberately held by the anchor rather than by [`QueryResourceEpoch`].
#[doc(hidden)]
pub struct QueryResourceAnchor {
    epoch: QueryResourceEpoch,
    _owner: Arc<dyn Send + Sync>,
}

impl QueryResourceAnchor {
    fn check_open(&self, operation: &'static str) -> Result<()> {
        self.epoch.check_open(operation)
    }

    fn belongs_to(&self, epoch: &QueryResourceEpoch) -> bool {
        Arc::ptr_eq(&self.epoch.inner, &epoch.inner)
    }
}

impl std::fmt::Debug for QueryResourceAnchor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("QueryResourceAnchor").field("epoch", &self.epoch).finish_non_exhaustive()
    }
}

impl Drop for QueryResourceAnchor {
    fn drop(&mut self) {
        // Drop runs while `_owner` is still live, so the idle callback can
        // always upgrade its weak query-lifetime reference.
        self.epoch.release_user();
    }
}

fn query_resource_epoch_closed(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "query resource epoch is closed")
        .with_fact("reason", "query_handle_released")
}

/// Shared counted owner attached to cloneable service-derived resources.
#[derive(Debug)]
#[doc(hidden)]
pub struct QueryServiceResourceEpoch {
    epoch: QueryServiceEpoch,
    _user: QueryServiceEpochUser,
    query_anchor: Option<Arc<QueryResourceAnchor>>,
}

impl QueryServiceResourceEpoch {
    #[doc(hidden)]
    pub fn acquire(
        epoch: &QueryServiceEpoch,
        query_anchor: Option<&Arc<QueryResourceAnchor>>,
        operation: &'static str,
    ) -> Result<Arc<Self>> {
        let user = epoch.acquire(operation)?;
        let query_anchor = query_anchor.cloned();
        if let Some(anchor) = &query_anchor
            && let Err(error) = anchor.check_open(operation)
        {
            drop(user);
            return Err(error);
        }
        Ok(Arc::new(Self { epoch: epoch.clone(), _user: user, query_anchor }))
    }

    #[doc(hidden)]
    pub fn check_open(&self, operation: &'static str) -> Result<()> {
        self.epoch.check_open(operation)?;
        if let Some(anchor) = &self.query_anchor {
            anchor.check_open(operation)?;
        }
        Ok(())
    }

    /// Whether this lease also retains its query anchor.
    #[doc(hidden)]
    pub fn has_query_anchor(&self) -> bool {
        self.query_anchor.is_some()
    }

    /// Shares only the backend service generation with a resident owner.
    ///
    /// Frozen residents outlive their attempt, so they must not retain the
    /// query anchor: keeping the query epoch alive would stall report
    /// publication and the frozen-resident handoff. The returned lease keeps
    /// the device generation alive until the retained GPU bytes release.
    #[doc(hidden)]
    pub fn service_generation_only(&self, operation: &'static str) -> Result<Arc<Self>> {
        self.epoch.check_open(operation)?;
        let user = self.epoch.retain_existing_user();
        // Re-check after retaining: a close that began between the two checks
        // must not hand back a lease that would keep the generation alive
        // past its own teardown. Failing here drops `user` and releases it.
        self.epoch.check_open(operation)?;
        Ok(Arc::new(Self { epoch: self.epoch.clone(), _user: user, query_anchor: None }))
    }
}

impl Drop for QueryServiceEpochUser {
    fn drop(&mut self) {
        self.epoch.release_user();
    }
}

/// Revocable context-scoped access to a service stream pool.
///
/// Query-service handles retain their query admission anchor even before a
/// stream is acquired. Each acquisition atomically checks both the query and
/// service epochs and gives the returned stream lease a counted owner that
/// survives derived runtime clones.
#[derive(Debug, Clone)]
pub struct QueryServiceStreamPool {
    pool: ServiceStreamPool,
    service_epoch: Option<QueryServiceEpoch>,
    query_anchor: Option<Arc<QueryResourceAnchor>>,
}

impl QueryServiceStreamPool {
    pub fn device(&self) -> cudf::DeviceId {
        self.pool.device()
    }

    pub fn pool_size(&self) -> usize {
        self.pool.pool_size()
    }

    pub fn acquire(&self) -> Result<StreamLease> {
        let operation = "query_engine.runtime.context.service_stream_pool.acquire";
        let resource_epoch = self.acquire_resource_epoch(operation)?;
        let lease = self.pool.acquire()?;
        if let Some(epoch) = &resource_epoch {
            epoch.check_open(operation)?;
        }
        Ok(self.attach_resource_epoch(lease, resource_epoch))
    }

    fn acquire_observed_with_cancellation(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
    ) -> Result<StreamLease> {
        let resource_epoch = self.acquire_resource_epoch(operation)?;
        let lease = self.pool.acquire_observed_with_cancellation(cancellation, operation, observation, scope)?;
        if let Some(epoch) = &resource_epoch {
            epoch.check_open(operation)?;
        }
        Ok(self.attach_resource_epoch(lease, resource_epoch))
    }

    fn acquire_resource_epoch(&self, operation: &'static str) -> Result<Option<Arc<QueryServiceResourceEpoch>>> {
        self.service_epoch
            .as_ref()
            .map(|epoch| QueryServiceResourceEpoch::acquire(epoch, self.query_anchor.as_ref(), operation))
            .transpose()
    }

    fn attach_resource_epoch(
        &self,
        lease: StreamLease,
        resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
    ) -> StreamLease {
        match resource_epoch {
            Some(resource_epoch) => lease.with_resource_epoch(resource_epoch),
            None => lease,
        }
    }
}

/// Immutable execution configuration shared by native query execution.
///
/// The context owns or shares the [`cudf::Runtime`] used for all device work in
/// a query. The runtime determines the CUDA device, stream, and memory resource;
/// the query engine layers its own chunk-planning, reservation, retry, and
/// boundary policies on top of that runtime. Cloning the context is cheap and
/// keeps the same runtime/cache handles.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Revocable service lifetime for runtime/resource acquisition.
    service_epoch: Option<QueryServiceEpoch>,
    /// Revocable lifetime for this admitted query's outward resources.
    query_epoch: Option<QueryResourceEpoch>,
    /// Counted query owner attached only to outward contexts, never the root template.
    query_anchor: Option<Arc<QueryResourceAnchor>>,
    /// Optional server-generation gate for file-backed native I/O.
    io_lifecycle: Option<IoLifecycle>,
    /// Optional caller-provided query-wide diagnostic correlation id.
    query_id: Option<QueryId>,
    /// Optional parent trace id used to group related query executions.
    trace_id: Option<TraceId>,
    /// Engine-minted identity for the physical attempt that admitted this context.
    ///
    /// Direct contexts are not attempt-bound and therefore leave this absent.
    attempt_id: Option<crate::observability::AttemptId>,
    /// Attempt-local carrier for exact resident-cache allocator failures.
    /// Direct contexts carry an inert value because they have no sealed
    /// attempt report that could truthfully own the evidence.
    cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation,
    /// Shared query-local bounded runtime observation session.
    runtime_observation: RuntimeObservationHandle,
    /// Shared terminal owner carried by query-service contexts.
    runtime_observation_owner: Option<crate::observability::RuntimeObservationOwner>,
    /// Current plan-execution scope installed while native execution runs.
    runtime_execution_scope: Option<crate::observability::RuntimeExecutionObservationScope>,
    /// Explicit plan/GPU correlation inherited by service-stream acquisitions.
    runtime_stream_lease_scope: RuntimeStreamLeaseScope,
    /// Shared cudf runtime used for GPU work.
    ///
    /// Runtime ownership is explicit: callers may inject an `Arc` to share a
    /// stream, allocator, and device across query plans, or allow the builder to
    /// create a fresh runtime from the configured device/memory-resource policy.
    runtime: Arc<cudf::Runtime>,
    /// Dedicated executor used for synchronous GPU work on this context's device.
    gpu_execution_pool: GpuExecutionPool,
    /// Runtime allocator construction policy recorded for diagnostics.
    ///
    /// This does not replace the allocator owned by an injected runtime. When an
    /// external runtime is supplied, the runtime's existing memory resource is
    /// authoritative and this field records [`MemoryResourcePolicy::ExternalRuntime`].
    memory_policy: MemoryResourcePolicy,
    /// Engine-level memory control contract for bounded execution.
    ///
    /// Bounded fields drive chunk planning, reservation admission, and
    /// retry/fan-in decisions. Source chunk byte and row-group settings are
    /// planning thresholds where row-group metadata permits exact splitting.
    memory_control_policy: MemoryControlPolicy,
    /// Optional native query memory domain that mirrors the current grant budget.
    grant_memory_domain: Option<Arc<cudf::QueryMemoryDomain>>,
    /// Authoritative domain for this service-admitted attempt.
    attempt_allocation_domain: Option<AttemptAllocationDomain>,
    /// Optional soft per-query memory grant backing bounded execution.
    ///
    /// This field is declared after native grant-backed resources so those
    /// resources are dropped before the final grant clone returns capacity to
    /// the ledger.
    memory_grant: Option<MemoryGrant>,
    /// Attempt-owned auxiliary stream allowance installed by query-service admission.
    attempt_auxiliary_slots: Option<Arc<AttemptAuxiliarySlots>>,
    /// Immutable post-grant evidence shared by this context and its query handle.
    /// Direct contexts intentionally carry no evidence.
    admitted_capability_evidence: Option<Arc<AttemptCapabilityEvidence>>,
    activated_deferred_resident_phase: Option<Arc<ActivatedDeferredResidentPhase>>,
    /// Alias removal must happen before the member lease can detach the raw
    /// stream from the attempt domain, and before the shared route can
    /// unregister its native provider mapping.
    _attempt_stream_provider_alias: Option<Arc<cudf::AttemptStreamProviderAlias>>,
    /// This context's shared member lease on its execution stream. One
    /// member per stream for the attempt; per-stage tokens clone it.
    attempt_member_lease: Option<cudf::StreamReservationLease>,
    /// Shared attempt fork-stream routing owner. Aliases are retained by
    /// derived contexts until their corresponding stream lease drains.
    attempt_stream_provider_route: Option<Arc<AttemptStreamProviderRoute>>,
    /// Execution boundary contract used by source reads, build/probe reuse, and
    /// result materialization.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Source scheduling policy for bounded local Parquet reads.
    source_scheduler_policy: SourceSchedulerPolicy,
    /// Source chunk planning mode for bounded local Parquet reads.
    source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Backend-owned source cache for this context's device, when the
    /// ordinal profile opted into resident caching.
    resident_source_cache: Option<crate::runtime::cache::local_parquet::GpuLocalParquetSourceCache>,
    /// Backend-owned graph cache for this context's device, when the ordinal
    /// profile opted into resident caching.
    #[cfg(feature = "cugraph")]
    resident_graph_cache: Option<crate::runtime::cache::resident_graph::GpuResidentGraphCache>,
    /// Service-level lakehouse cache-fill resource configuration.
    lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    /// Service-level pool used by bounded source-read workers.
    service_stream_pool: ServiceStreamPool,
    /// Service-wide byte budget for complete-object lakehouse cache fills.
    lakehouse_cache_fill_budget: CacheFillByteBudget,
    /// Service-wide operation-count gate for complete-object lakehouse cache fills.
    lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    /// Structured completion/failure events from asynchronous lakehouse cache fills.
    lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    /// Service-scoped async runtime used for object-store control-plane work and cache maintenance.
    object_store_control_runtime: Arc<tokio::runtime::Runtime>,
    /// Cached object-store clients used by metadata and cache-maintenance work.
    source_object_stores: Arc<SourceObjectStoreRegistry>,
    /// Deployment-level IO capability probes used for readiness/reporting.
    deployment_io_capability: DeploymentIoCapability,
    /// Cooperative cancellation flag observed at native execution boundaries.
    cancellation: ExecutionCancellation,
    /// Selected execution contract for floating statistical aggregates.
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    /// Per-query DataFusion exact-order scope for floating statistical aggregates.
    ///
    /// Source-read workers run on spawned OS threads that cannot observe the
    /// caller thread's tracking state, so this flag travels with the context
    /// clone handed to the worker instead of living in a thread-local.
    datafusion_order_tracking: bool,
    /// Runtime threshold policy for grouped aggregate strategy selection.
    aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert policy for local-throughput source chunk planning.
    source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Optional owned stream lease kept alive by contexts derived from it.
    stream_lease_guard: Option<Arc<StreamLease>>,
    /// Auxiliary envelope ownership paired with an owned auxiliary stream.
    auxiliary_stream_permit_guard: Option<Arc<AttemptAuxiliaryPermit>>,
    /// Whether this context was created from a QueryService admission handle.
    query_service_admitted: bool,
}

#[derive(Debug)]
struct ActivatedDeferredResidentPhase {
    consumer_execution_scope_fingerprint: String,
    inputs: Vec<ActivatedDeferredResidentInput>,
    capability_evidence: AttemptCapabilityEvidence,
    grant_identity: crate::runtime::gpu::GrantIdentity,
}

#[derive(Debug)]
struct ActivatedDeferredResidentInput {
    producer_execution_scope_fingerprint: String,
    input_name: String,
    input_schema: arrow::datatypes::SchemaRef,
    input_memory_identity: cudf::DataFrameMemoryIdentity,
    input_row_count: u64,
}

impl ActivatedDeferredResidentPhase {
    fn validate_inputs(&self, inputs: &GpuResidentInputMap, grant: &MemoryGrant) -> Result<()> {
        for activated in &self.inputs {
            let input = inputs.get(&activated.input_name).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "query_engine.runtime.context",
                    "the activated deferred resident input is missing",
                )
                .with_fact("reason", "deferred_resident_input_missing")
                .with_fact("input_name", &activated.input_name)
            })?;
            activated.validate(input, grant, self)?;
        }
        Ok(())
    }
}

impl ActivatedDeferredResidentInput {
    fn validate(
        &self,
        input: &GpuResultHandle,
        grant: &MemoryGrant,
        phase: &ActivatedDeferredResidentPhase,
    ) -> Result<()> {
        let row_count = u64::try_from(input.frame().num_rows()).map_err(|_| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the materialized graph input row count does not fit the admission contract",
            )
            .with_fact("reason", "deferred_graph_input_row_count_overflow")
        })?;
        let physical_bytes = input.device_allocated_size_bytes()?;
        let retained_by_grant = input.retained_owner_identity_for(grant).is_some();
        if phase.grant_identity != grant.identity()
            || input.schema() != &self.input_schema
            || input.frame().memory_identity() != self.input_memory_identity
            || input.producer_execution_scope() != Some(&self.producer_execution_scope_fingerprint)
            || row_count != self.input_row_count
            || (physical_bytes > 0 && !retained_by_grant)
        {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the GPU-resident graph input differs from the frame that activated its admitted phase",
            )
            .with_fact("reason", "deferred_graph_input_identity_mismatch")
            .with_fact("input_name", &self.input_name)
            .with_fact("input_row_count", row_count)
            .with_fact("activated_row_count", self.input_row_count)
            .with_fact("input_device_bytes", physical_bytes));
        }
        Ok(())
    }
}

/// Execution context tied to an owned CUDA stream lease.
///
/// This is the preferred API when a caller needs GPU work to run on a leased
/// service stream. Holding this value keeps the underlying stream lease alive
/// for at least as long as the borrowed context can be used.
#[derive(Debug)]
pub struct LeasedExecutionContext {
    context: ExecutionContext,
}

/// Root execution context paired with its runtime observation owner.
///
/// The context retains the owner internally so native execution can select the
/// correct query or plan completion boundary. Callers that need to inspect or
/// finish the standalone observation retain the same shared owner here.
#[derive(Debug)]
pub struct RuntimeObservedExecutionContext {
    context: ExecutionContext,
    observation: crate::observability::RuntimeObservationOwner,
}

impl RuntimeObservedExecutionContext {
    pub(super) fn new(context: ExecutionContext, observation: crate::observability::RuntimeObservationOwner) -> Self {
        Self { context, observation }
    }

    /// Returns the execution context carrying the observation lifecycle state.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }

    /// Returns the owner-only terminal-seal capability.
    pub fn observation(&self) -> &crate::observability::RuntimeObservationOwner {
        &self.observation
    }

    /// Returns the execution context together with its shared owner.
    pub fn into_parts(self) -> (ExecutionContext, crate::observability::RuntimeObservationOwner) {
        (self.context, self.observation)
    }
}

/// One native execution observation boundary selected from context ownership.
///
/// This type exposes plan recording and terminal completion without exposing
/// the observation owner stored by the execution context.
#[doc(hidden)]
#[derive(Debug)]
pub struct RuntimeExecutionObservationLifecycle {
    inner: RuntimeExecutionObservationLifecycleInner,
}

#[derive(Debug)]
enum RuntimeExecutionObservationLifecycleInner {
    Query(RuntimeQueryLifecycle),
    Plan(RuntimePlanExecutionLifecycle),
}

impl RuntimeExecutionObservationLifecycle {
    fn for_context(context: &ExecutionContext, plan_execution_id: PlanExecutionId) -> Self {
        let inner = match &context.runtime_observation_owner {
            Some(owner) if context.query_service_admitted => RuntimeExecutionObservationLifecycleInner::Plan(
                RuntimePlanExecutionLifecycle::from_owner(owner.clone(), plan_execution_id),
            ),
            Some(owner) => RuntimeExecutionObservationLifecycleInner::Query(RuntimeQueryLifecycle::from_owner(
                owner.clone(),
                plan_execution_id,
            )),
            None => RuntimeExecutionObservationLifecycleInner::Plan(RuntimePlanExecutionLifecycle::from_handle(
                context.runtime_observation.clone(),
                plan_execution_id,
            )),
        };
        Self { inner }
    }

    /// Returns the record-only scope for one native plan invocation.
    pub fn execution_scope(&self) -> RuntimeExecutionObservationScope {
        match &self.inner {
            RuntimeExecutionObservationLifecycleInner::Query(lifecycle) => lifecycle.execution_scope(),
            RuntimeExecutionObservationLifecycleInner::Plan(lifecycle) => lifecycle.execution_scope(),
        }
    }

    /// Completes this plan or standalone query according to context ownership.
    pub fn finish(self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        match self.inner {
            RuntimeExecutionObservationLifecycleInner::Query(lifecycle) => lifecycle.finish(outcome),
            RuntimeExecutionObservationLifecycleInner::Plan(lifecycle) => {
                lifecycle.finish(outcome);
                None
            },
        }
    }
}

impl LeasedExecutionContext {
    fn new(context: ExecutionContext) -> Self {
        Self { context }
    }

    /// Returns the context backed by the owned stream lease.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }

    /// Consumes the wrapper and keeps the owned stream lease on the context.
    pub fn into_context(self) -> ExecutionContext {
        self.context
    }
}

#[derive(Debug)]
pub(super) struct ExecutionContextParts {
    pub(super) service_epoch: Option<QueryServiceEpoch>,
    pub(super) query_epoch: Option<QueryResourceEpoch>,
    pub(super) query_anchor: Option<Arc<QueryResourceAnchor>>,
    pub(super) io_lifecycle: Option<IoLifecycle>,
    pub(super) query_id: Option<QueryId>,
    pub(super) trace_id: Option<TraceId>,
    pub(super) attempt_id: Option<crate::observability::AttemptId>,
    pub(super) cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation,
    pub(super) runtime_observation: RuntimeObservationHandle,
    pub(super) runtime_observation_owner: Option<crate::observability::RuntimeObservationOwner>,
    pub(super) runtime_execution_scope: Option<crate::observability::RuntimeExecutionObservationScope>,
    pub(super) runtime_stream_lease_scope: RuntimeStreamLeaseScope,
    pub(super) runtime: Arc<cudf::Runtime>,
    pub(super) gpu_execution_pool: GpuExecutionPool,
    pub(super) memory_policy: MemoryResourcePolicy,
    pub(super) memory_control_policy: MemoryControlPolicy,
    pub(super) memory_grant: Option<MemoryGrant>,
    pub(super) grant_memory_domain: Option<Arc<cudf::QueryMemoryDomain>>,
    pub(super) attempt_allocation_domain: Option<AttemptAllocationDomain>,
    pub(super) attempt_auxiliary_slots: Option<Arc<AttemptAuxiliarySlots>>,
    pub(super) admitted_capability_evidence: Option<Arc<AttemptCapabilityEvidence>>,
    pub(super) attempt_stream_provider_alias: Option<Arc<cudf::AttemptStreamProviderAlias>>,
    pub(super) attempt_member_lease: Option<cudf::StreamReservationLease>,
    pub(super) attempt_stream_provider_route: Option<Arc<AttemptStreamProviderRoute>>,
    pub(super) boundary_policy: ExecutionBoundaryPolicy,
    pub(super) source_scheduler_policy: SourceSchedulerPolicy,
    pub(super) source_chunk_planning_mode: SourceChunkPlanningMode,
    pub(super) resident_source_cache: Option<crate::runtime::cache::local_parquet::GpuLocalParquetSourceCache>,
    #[cfg(feature = "cugraph")]
    pub(super) resident_graph_cache: Option<crate::runtime::cache::resident_graph::GpuResidentGraphCache>,
    pub(super) lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    pub(super) service_stream_pool: ServiceStreamPool,
    pub(super) lakehouse_cache_fill_budget: CacheFillByteBudget,
    pub(super) lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    pub(super) lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    pub(super) object_store_control_runtime: Arc<tokio::runtime::Runtime>,
    pub(super) source_object_stores: Arc<SourceObjectStoreRegistry>,
    pub(super) deployment_io_capability: DeploymentIoCapability,
    pub(super) cancellation: ExecutionCancellation,
    pub(super) statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    pub(super) datafusion_order_tracking: bool,
    pub(super) aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    pub(super) source_chunk_planning_options: SourceChunkPlanningOptions,
    pub(super) stream_lease_guard: Option<Arc<StreamLease>>,
    pub(super) auxiliary_stream_permit_guard: Option<Arc<AttemptAuxiliaryPermit>>,
    pub(super) query_service_admitted: bool,
}

impl ExecutionContext {
    /// Starts a builder for an [`ExecutionContext`].
    // No production callers; kept private-but-present so tests/ui/public_api
    // pins that this factory stays unreachable outside the crate.
    pub fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    /// Builds a context around an owned cudf runtime.
    ///
    /// The resulting context records [`MemoryResourcePolicy::ExternalRuntime`]
    /// because the runtime, including its allocator and stream, has already been
    /// constructed by the caller.
    // No production callers; kept private-but-present so tests/ui/public_api
    // pins that this constructor stays unreachable outside the crate.
    pub fn from_runtime(runtime: cudf::Runtime) -> Result<Self> {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    /// Builds a context around a shared cudf runtime.
    ///
    /// Use this when multiple plans or adapter layers must share the same CUDA
    /// stream, device placement, allocator policy, or boundary observer state.
    // No production callers; kept private-but-present so tests/ui/public_api
    // pins that this constructor stays unreachable outside the crate.
    pub fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Result<Self> {
        let service_stream_pool =
            ServiceStreamPool::new(runtime.device(), default_service_stream_pool_size_for_device(runtime.device()))
                .expect("default service stream pool size is non-zero");
        let lakehouse_cache_fill_config = LakehouseCacheFillConfig::default();
        let gpu_execution_pool =
            GpuExecutionPool::shared_for_device(runtime.device().ordinal(), service_stream_pool.pool_size())?;
        Ok(Self::from_parts(ExecutionContextParts {
            service_epoch: None,
            query_epoch: None,
            query_anchor: None,
            io_lifecycle: None,
            query_id: None,
            trace_id: None,
            attempt_id: None,
            cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation::default(),
            runtime_observation: RuntimeObservationHandle::disabled(),
            runtime_observation_owner: None,
            runtime_execution_scope: None,
            runtime_stream_lease_scope: RuntimeStreamLeaseScope::default(),
            runtime,
            gpu_execution_pool,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            memory_grant: None,
            grant_memory_domain: None,
            attempt_allocation_domain: None,
            attempt_auxiliary_slots: None,
            admitted_capability_evidence: None,
            attempt_member_lease: None,
            attempt_stream_provider_route: None,
            attempt_stream_provider_alias: None,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            resident_source_cache: None,
            #[cfg(feature = "cugraph")]
            resident_graph_cache: None,
            lakehouse_cache_fill_config,
            service_stream_pool,
            lakehouse_cache_fill_budget: CacheFillByteBudget::default(),
            lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate::new(
                LakehouseCacheFillConfig::DEFAULT_MAX_CONCURRENT_OPS,
            )
            .expect("default lakehouse cache fill concurrency is non-zero"),
            lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            object_store_control_runtime: default_object_store_control_runtime()?,
            source_object_stores: Arc::new(SourceObjectStoreRegistry::new()),
            deployment_io_capability: DeploymentIoCapability::cached_current_host(),
            cancellation: ExecutionCancellation::new(),
            statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::default(),
            datafusion_order_tracking: false,
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            stream_lease_guard: None,
            auxiliary_stream_permit_guard: None,
            query_service_admitted: false,
        }))
    }

    pub(super) fn from_parts(parts: ExecutionContextParts) -> Self {
        Self {
            service_epoch: parts.service_epoch,
            query_epoch: parts.query_epoch,
            query_anchor: parts.query_anchor,
            io_lifecycle: parts.io_lifecycle,
            query_id: parts.query_id,
            trace_id: parts.trace_id,
            attempt_id: parts.attempt_id,
            cache_scope_allocator_correlation: parts.cache_scope_allocator_correlation,
            runtime_observation: parts.runtime_observation,
            runtime_observation_owner: parts.runtime_observation_owner,
            runtime_execution_scope: parts.runtime_execution_scope,
            runtime_stream_lease_scope: parts.runtime_stream_lease_scope,
            runtime: parts.runtime,
            gpu_execution_pool: parts.gpu_execution_pool,
            stream_lease_guard: parts.stream_lease_guard,
            memory_policy: parts.memory_policy,
            memory_control_policy: parts.memory_control_policy,
            memory_grant: parts.memory_grant,
            grant_memory_domain: parts.grant_memory_domain,
            attempt_allocation_domain: parts.attempt_allocation_domain,
            attempt_auxiliary_slots: parts.attempt_auxiliary_slots,
            admitted_capability_evidence: parts.admitted_capability_evidence,
            activated_deferred_resident_phase: None,
            attempt_member_lease: parts.attempt_member_lease,
            attempt_stream_provider_route: parts.attempt_stream_provider_route,
            _attempt_stream_provider_alias: parts.attempt_stream_provider_alias,
            boundary_policy: parts.boundary_policy,
            source_scheduler_policy: parts.source_scheduler_policy,
            source_chunk_planning_mode: parts.source_chunk_planning_mode,
            resident_source_cache: parts.resident_source_cache,
            #[cfg(feature = "cugraph")]
            resident_graph_cache: parts.resident_graph_cache,
            lakehouse_cache_fill_config: parts.lakehouse_cache_fill_config,
            service_stream_pool: parts.service_stream_pool,
            lakehouse_cache_fill_budget: parts.lakehouse_cache_fill_budget,
            lakehouse_cache_fill_concurrency: parts.lakehouse_cache_fill_concurrency,
            lakehouse_cache_fill_events: parts.lakehouse_cache_fill_events,
            object_store_control_runtime: parts.object_store_control_runtime,
            source_object_stores: parts.source_object_stores,
            deployment_io_capability: parts.deployment_io_capability,
            cancellation: parts.cancellation,
            statistical_aggregate_execution_mode: parts.statistical_aggregate_execution_mode,
            datafusion_order_tracking: parts.datafusion_order_tracking,
            aggregate_strategy_execution_options: parts.aggregate_strategy_execution_options,
            source_chunk_planning_options: parts.source_chunk_planning_options,
            auxiliary_stream_permit_guard: parts.auxiliary_stream_permit_guard,
            query_service_admitted: parts.query_service_admitted,
        }
    }

    /// Attaches one counted query owner to a clone of the internal root template.
    pub(super) fn with_query_resource_anchor(&self, anchor: Arc<QueryResourceAnchor>) -> Self {
        debug_assert!(
            self.query_epoch.as_ref().is_some_and(|epoch| anchor.belongs_to(epoch)),
            "query resource anchor must belong to the context query epoch"
        );
        let mut context = self.clone();
        context.query_anchor = Some(anchor);
        context
    }

    fn acquire_resource_epoch(&self, operation: &'static str) -> Result<Option<Arc<QueryServiceResourceEpoch>>> {
        let Some(service_epoch) = &self.service_epoch else {
            if let Some(query_epoch) = &self.query_epoch {
                query_epoch.check_open(operation)?;
            }
            return Ok(None);
        };
        QueryServiceResourceEpoch::acquire(service_epoch, self.query_anchor.as_ref(), operation).map(Some)
    }

    /// Returns the cudf runtime used for all GPU operations.
    pub fn runtime(&self) -> Result<cudf::Runtime> {
        let operation = "query_engine.runtime.context.runtime";
        // Validate the query/service epoch at acquisition time, but do not
        // embed it in GPU allocations. Escaping allocations are owned by a
        // ResidentOwnerLease; unowned residual allocations must reach final
        // validation so cleanup can fail closed instead of keeping the query
        // epoch alive forever and preventing B-to-R handoff.
        drop(self.acquire_resource_epoch(operation)?);
        Ok(self.runtime.as_ref().clone())
    }

    /// Acquire permission for one file-backed native I/O call.
    ///
    /// Standalone library contexts without a server lifecycle remain
    /// unrestricted. Server contexts carry one generation gate and reject
    /// acquisitions after graceful shutdown begins.
    pub fn acquire_io_lease(&self, operation: &'static str) -> Result<Option<IoLifecycleLease>> {
        self.io_lifecycle.as_ref().map(|lifecycle| lifecycle.acquire(operation)).transpose()
    }

    /// Attach a server-generation I/O lifecycle to this context clone.
    pub fn with_io_lifecycle(mut self, lifecycle: IoLifecycle) -> Self {
        self.io_lifecycle = Some(lifecycle);
        self
    }

    /// Returns a shared handle to the runtime.
    ///
    /// The caller receives the same runtime object, not a new runtime with
    /// copied configuration.
    pub fn shared_runtime(&self) -> Result<Arc<cudf::Runtime>> {
        self.runtime().map(Arc::new)
    }

    /// Returns an opaque lifecycle-bound executor for synchronous GPU work on
    /// this context's device.
    ///
    /// The lease's query-execution job follows this context's own claim on
    /// the attempt: a root context runs it as root-class work, while an
    /// auxiliary fragment context -- a second native island of the same
    /// attempt -- runs it as auxiliary work backed by the attempt auxiliary
    /// permit that context already holds. Root capacity is one job per
    /// admitted attempt, so a second island charging it would fail where it
    /// has a reserved auxiliary slot of its own.
    pub fn gpu_executor(&self) -> Result<GpuExecutorLease> {
        self.gpu_execution_pool().map(if self.auxiliary_stream_permit_guard.is_some() {
            GpuExecutorLease::from_pool_for_auxiliary_fragment
        } else {
            GpuExecutorLease::from_pool
        })
    }

    #[doc(hidden)]
    pub fn gpu_execution_pool(&self) -> Result<GpuExecutionPool> {
        let resource_epoch = self.acquire_resource_epoch("query_engine.runtime.context.gpu_execution_pool")?;
        Ok(match resource_epoch {
            Some(resource_epoch) => self.gpu_execution_pool.with_resource_epoch(resource_epoch),
            None => self.gpu_execution_pool.clone(),
        })
    }

    /// Returns the admitted attempt's fork-stream route, when this context
    /// carries one. Each lease derived from it holds one auxiliary fork of the
    /// route alone, up to [`AttemptStreamProviderRoute::lane_capacity`];
    /// direct contexts have no route and run serially.
    #[doc(hidden)]
    pub fn attempt_stream_provider_route(&self) -> Option<Arc<AttemptStreamProviderRoute>> {
        self.attempt_stream_provider_route.as_ref().map(Arc::clone)
    }

    /// Returns the query-wide diagnostic correlation id.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Returns the engine-minted attempt identity carried by an admitted context.
    ///
    /// This remains absent for direct contexts because they are not part of a
    /// public execution-attempt lifecycle.
    #[doc(hidden)]
    pub fn attempt_id(&self) -> Option<crate::observability::AttemptId> {
        self.attempt_id
    }

    /// Opens one exact allocator-event window for an exclusive resident graph
    /// checkout. The caller must finish the returned guard after cache-runtime
    /// temporaries are destroyed and before returning the checkout.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn begin_cache_scope_allocator_correlation(
        &self,
        scope: Arc<cudf::CacheAllocationScope>,
    ) -> Option<crate::runtime::allocator_correlation::CacheScopeCorrelationGuard> {
        self.cache_scope_allocator_correlation.begin(scope)
    }

    /// Returns the shared recording handle without terminal-seal authority.
    pub fn runtime_observation(&self) -> &RuntimeObservationHandle {
        &self.runtime_observation
    }

    /// Opens one native execution boundary without exposing the context's
    /// observation owner.
    #[doc(hidden)]
    pub fn begin_runtime_execution_observation(
        &self,
        plan_execution_id: PlanExecutionId,
    ) -> RuntimeExecutionObservationLifecycle {
        RuntimeExecutionObservationLifecycle::for_context(self, plan_execution_id)
    }

    /// Returns the active native plan-execution scope.
    pub fn runtime_execution_scope(&self) -> Option<&crate::observability::RuntimeExecutionObservationScope> {
        self.runtime_execution_scope.as_ref()
    }

    /// Returns a clone carrying one invocation's recording handle.
    ///
    /// Seal authority remains with the execution lifecycle that supplied the
    /// handle. Identity is validated before the clone can reach workers.
    #[doc(hidden)]
    pub fn with_runtime_observation_handle(&self, observation: RuntimeObservationHandle) -> Result<Self> {
        if let Some(query_id) = &self.query_id {
            validate_runtime_observation_query_id(&observation, query_id)?;
        }
        if let Some(trace_id) = &self.trace_id {
            validate_runtime_observation_trace_id(&observation, trace_id)?;
        }
        let mut context = self.clone();
        context.runtime_observation = observation;
        Ok(context)
    }

    /// Returns a clone scoped to one native plan invocation.
    #[doc(hidden)]
    pub fn with_runtime_execution_scope(
        &self,
        scope: crate::observability::RuntimeExecutionObservationScope,
    ) -> Result<Self> {
        let mut context = self.with_runtime_observation_handle(scope.observation().clone())?;
        context.runtime_stream_lease_scope.plan_execution_id = Some(scope.plan_execution_id().clone());
        context.runtime_execution_scope = Some(scope);
        Ok(context)
    }

    /// Returns a clone carrying explicit GPU job/worker correlation.
    pub fn with_runtime_gpu_stream_scope(&self, scope: RuntimeStreamLeaseScope) -> Self {
        let mut context = self.clone();
        context.runtime_stream_lease_scope.gpu_job_id = scope.gpu_job_id;
        context.runtime_stream_lease_scope.gpu_worker_id = scope.gpu_worker_id;
        context
    }

    /// Returns a cloned context with a diagnostic query id consistent with observation identity.
    pub fn with_query_id(&self, query_id: impl Into<QueryId>) -> Result<Self> {
        let mut context = self.clone();
        let query_id = query_id.into();
        validate_runtime_observation_query_id(&context.runtime_observation, &query_id)?;
        context.query_id = Some(query_id);
        Ok(context)
    }

    /// Returns a cloned context with a trace id consistent with observation identity.
    pub fn with_trace_id(&self, trace_id: impl Into<TraceId>) -> Result<Self> {
        let mut context = self.clone();
        let trace_id = trace_id.into();
        validate_runtime_observation_trace_id(&context.runtime_observation, &trace_id)?;
        context.trace_id = Some(trace_id);
        Ok(context)
    }

    /// Acquires one attempt work lease: an attempt auxiliary permit, a
    /// service stream lease, and the derived context that owns them.
    ///
    /// This is the single bundle behind every piece of auxiliary attempt work
    /// -- execution lanes and scheduled source reads alike. The derived
    /// context's alias holds one auxiliary fork of the attempt pool alone for
    /// the lease's lifetime, so its native forks never reach a stream another
    /// live holder is launching on. The permit it takes is what bounds those
    /// claims to the pool's width.
    #[doc(hidden)]
    pub fn acquire_attempt_work_lease(
        &self,
        source_worker: Option<(RuntimeTaskId, RuntimeWorkerId)>,
    ) -> Result<AttemptWorkLease> {
        // Admission reserves this bounded auxiliary share per attempt. Hold
        // the local permit for the exact lifetime of the derived stream lease,
        // so another attempt's reservation remains physically reachable.
        let attempt_auxiliary_permit = self
            .attempt_auxiliary_slots
            .as_ref()
            .map(|slots| {
                slots.acquire_with_cancellation(
                    &self.cancellation,
                    "query_engine.runtime.attempt_work_lease.attempt_auxiliary_acquire",
                )
            })
            .transpose()?;
        self.derive_attempt_work_lease(source_worker, attempt_auxiliary_permit)
    }

    /// Acquires one attempt work lease only if this attempt's auxiliary
    /// envelope has a free slot right now.
    ///
    /// Callers that already hold an attempt auxiliary permit -- the root
    /// execution job dispatching lanes among them -- must never block on a
    /// second one, because the holder they would wait for may be waiting on
    /// them. Returns `Ok(None)` when the envelope is fully claimed, so the
    /// caller narrows its own width instead of deadlocking.
    #[doc(hidden)]
    pub fn try_acquire_attempt_work_lease(&self) -> Result<Option<AttemptWorkLease>> {
        let attempt_auxiliary_permit = match self.attempt_auxiliary_slots.as_ref() {
            Some(slots) => {
                match slots.try_acquire("query_engine.runtime.attempt_work_lease.attempt_auxiliary_try_acquire")? {
                    Some(permit) => Some(permit),
                    None => return Ok(None),
                }
            },
            None => None,
        };
        self.derive_attempt_work_lease(None, attempt_auxiliary_permit).map(Some)
    }

    fn derive_attempt_work_lease(
        &self,
        source_worker: Option<(RuntimeTaskId, RuntimeWorkerId)>,
        attempt_auxiliary_permit: Option<AttemptAuxiliaryPermit>,
    ) -> Result<AttemptWorkLease> {
        let mut stream_scope = self.runtime_stream_lease_scope.clone();
        if let Some((task_id, worker_id)) = source_worker {
            stream_scope = stream_scope.with_source_worker(task_id, worker_id);
        }
        let stream_lease = self.service_stream_pool().acquire_observed_with_cancellation(
            &self.cancellation,
            "query_engine.runtime.attempt_work_lease.acquire",
            self.runtime_observation.clone(),
            stream_scope,
        )?;
        let context = self.with_stream_lease(&stream_lease)?;
        Ok(AttemptWorkLease::new(context, stream_lease, attempt_auxiliary_permit))
    }

    /// Whether this context runs on a stream leased for attempt work rather
    /// than on the attempt's own root stream.
    ///
    /// Work issued from such a context already holds an attempt auxiliary
    /// permit, so it must run inline instead of leasing a second one.
    #[doc(hidden)]
    pub fn is_attempt_work_lease_context(&self) -> bool {
        self._attempt_stream_provider_alias.is_some()
    }

    /// Which auxiliary fork of the attempt pool this context's alias holds
    /// alone. A context with no alias -- the attempt root, or a direct context
    /// -- reports `None`; the root runs on the route's reserved fork instead.
    #[doc(hidden)]
    pub fn attempt_claimed_fork_index(&self) -> Option<usize> {
        self._attempt_stream_provider_alias.as_ref().map(|alias| alias.claimed_fork_index())
    }

    /// Builds a short-lived execution context from a borrowed stream lease.
    ///
    /// The returned context shares this context's query policies, caches, and
    /// cancellation flag, but executes GPU work on the leased runtime. The
    /// caller must keep `stream_lease` alive until all work submitted through
    /// the returned context has completed.
    pub fn with_stream_lease(&self, stream_lease: &StreamLease) -> Result<Self> {
        self.check_service_epoch("query_engine.runtime.context.with_stream_lease")?;
        validate_service_stream_pool_device(stream_lease.pool(), self.runtime.device())?;
        let runtime = stream_lease.runtime()?;
        let attempt_member_lease = match self.attempt_allocation_domain.as_ref() {
            Some(domain) => Some(domain.create_member(runtime.stream())?),
            None => None,
        };
        let attempt_stream_provider_alias = self
            .attempt_stream_provider_route
            .as_ref()
            .map(|route| route.attach(runtime.stream()).map(Arc::new))
            .transpose()?;
        Ok(Self::from_parts(ExecutionContextParts {
            service_epoch: self.service_epoch.clone(),
            query_epoch: self.query_epoch.clone(),
            query_anchor: self.query_anchor.clone(),
            io_lifecycle: self.io_lifecycle.clone(),
            query_id: self.query_id.clone(),
            trace_id: self.trace_id.clone(),
            attempt_id: self.attempt_id,
            cache_scope_allocator_correlation: self.cache_scope_allocator_correlation.clone(),
            runtime_observation: self.runtime_observation.clone(),
            runtime_observation_owner: self.runtime_observation_owner.clone(),
            runtime_execution_scope: self.runtime_execution_scope.clone(),
            runtime_stream_lease_scope: self.runtime_stream_lease_scope.clone(),
            runtime: Arc::new(runtime),
            gpu_execution_pool: self.gpu_execution_pool.clone(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy,
            memory_grant: self.memory_grant.clone(),
            grant_memory_domain: self.grant_memory_domain.clone(),
            attempt_allocation_domain: self.attempt_allocation_domain.clone(),
            attempt_auxiliary_slots: self.attempt_auxiliary_slots.as_ref().map(Arc::clone),
            admitted_capability_evidence: self.admitted_capability_evidence.clone(),
            attempt_member_lease,
            attempt_stream_provider_route: self.attempt_stream_provider_route.as_ref().map(Arc::clone),
            attempt_stream_provider_alias,
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_chunk_planning_mode: self.source_chunk_planning_mode,
            resident_source_cache: self.resident_source_cache.clone(),
            #[cfg(feature = "cugraph")]
            resident_graph_cache: self.resident_graph_cache.clone(),
            lakehouse_cache_fill_config: self.lakehouse_cache_fill_config.clone(),
            service_stream_pool: self.service_stream_pool.clone(),
            lakehouse_cache_fill_budget: self.lakehouse_cache_fill_budget.clone(),
            lakehouse_cache_fill_concurrency: self.lakehouse_cache_fill_concurrency.clone(),
            lakehouse_cache_fill_events: self.lakehouse_cache_fill_events.clone(),
            object_store_control_runtime: Arc::clone(&self.object_store_control_runtime),
            source_object_stores: Arc::clone(&self.source_object_stores),
            deployment_io_capability: self.deployment_io_capability.clone(),
            cancellation: self.cancellation.clone(),
            statistical_aggregate_execution_mode: self.statistical_aggregate_execution_mode,
            datafusion_order_tracking: self.datafusion_order_tracking,
            aggregate_strategy_execution_options: self.aggregate_strategy_execution_options,
            source_chunk_planning_options: self.source_chunk_planning_options,
            stream_lease_guard: None,
            auxiliary_stream_permit_guard: None,
            query_service_admitted: self.query_service_admitted,
        }))
    }

    /// Builds a context and keeps the stream lease alive in the returned wrapper.
    pub fn with_owned_stream_lease(&self, stream_lease: StreamLease) -> Result<LeasedExecutionContext> {
        let stream_lease_guard = Arc::new(stream_lease);
        let mut context = self.with_stream_lease(stream_lease_guard.as_ref())?;
        context.stream_lease_guard = Some(stream_lease_guard);
        Ok(LeasedExecutionContext::new(context))
    }

    /// Pairs one admitted auxiliary permit with this context's owned stream.
    ///
    /// The permit is reference-counted with the stream guard, so cloning an
    /// outward context cannot retain the stream while releasing its attempt
    /// envelope share early.
    #[doc(hidden)]
    pub fn with_auxiliary_stream_permit_guard(mut self, permit: AttemptAuxiliaryPermit) -> Self {
        debug_assert!(self.stream_lease_guard.is_some(), "auxiliary permit must be paired with an owned stream lease");
        debug_assert!(
            self.auxiliary_stream_permit_guard.is_none(),
            "an owned auxiliary stream has exactly one attempt permit"
        );
        self.auxiliary_stream_permit_guard = Some(Arc::new(permit));
        self
    }

    /// Returns a context for an internal fragment that must bypass bounded admission.
    ///
    /// Query-local and service-scoped state is preserved; only the engine-level
    /// memory-control policy is relaxed. This is intentionally narrower than
    /// rebuilding from the shared cudf runtime, which would drop cancellation,
    /// source IO profiles, cache handles, and event sinks.
    // No production callers; kept private-but-present so tests/ui/public_api
    // pins that this downgrade stays unreachable outside the crate.
    #[doc(hidden)]
    pub fn for_unbounded_fragment(&self) -> Self {
        let mut context = self.clone();
        context.memory_control_policy = MemoryControlPolicy::Unbounded;
        context
    }

    /// Returns whether this context is backed by a QueryService admission slot.
    pub fn is_query_service_admitted(&self) -> bool {
        self.query_service_admitted
    }

    /// Returns the backend-owned local-Parquet cache for this context's
    /// ordinal, if its device profile enabled resident caching.
    #[doc(hidden)]
    pub fn resident_source_cache(&self) -> Option<crate::runtime::cache::local_parquet::GpuLocalParquetSourceCache> {
        self.resident_source_cache.clone()
    }

    /// Returns the backend-owned resident graph cache for this context's
    /// ordinal, if its device profile enabled resident caching.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn resident_graph_cache(&self) -> Option<crate::runtime::cache::resident_graph::GpuResidentGraphCache> {
        self.resident_graph_cache.clone()
    }

    fn validated_admitted_identity(
        &self,
        plan: &QueryPlan,
    ) -> Result<Option<(&AttemptCapabilityEvidence, MemoryGrant)>> {
        if !self.query_service_admitted {
            return Ok(None);
        }
        let evidence = self.admitted_capability_evidence.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "a query-service context requires its admitted capability evidence",
            )
            .with_fact("reason", "admitted_capability_evidence_missing")
            .with_fact("plan_scope", query_planning::admission::canonical_execution_scope_fingerprint(plan))
        })?;
        let runtime_device_ordinal = self.runtime()?.device().ordinal();
        let immutable_grant = self.memory_grant()?.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "an admitted context requires its immutable grant",
            )
            .with_fact("reason", "admitted_grant_missing")
        })?;
        if immutable_grant.device_ordinal() != runtime_device_ordinal {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the immutable grant device differs from the runtime device",
            )
            .with_fact("reason", "admitted_grant_device_mismatch")
            .with_fact("grant_device_ordinal", immutable_grant.device_ordinal())
            .with_fact("runtime_device_ordinal", runtime_device_ordinal));
        }
        if self.attempt_id != Some(immutable_grant.identity().attempt_id) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the context attempt identity differs from the immutable grant",
            )
            .with_fact("reason", "admitted_grant_attempt_mismatch")
            .with_fact(
                "context_attempt_id",
                self.attempt_id.map(|attempt_id| attempt_id.to_string()).unwrap_or_default(),
            )
            .with_fact("grant_attempt_id", immutable_grant.identity().attempt_id));
        }
        let domain = self.attempt_allocation_domain.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "an admitted context requires its allocation domain",
            )
            .with_fact("reason", "admitted_allocation_domain_missing")
        })?;
        if domain.device_ordinal() != runtime_device_ordinal || domain.grant_identity() != immutable_grant.identity() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the allocation domain identity differs from the immutable grant",
            )
            .with_fact("reason", "admitted_allocation_domain_identity_mismatch")
            .with_fact("domain_device_ordinal", domain.device_ordinal())
            .with_fact("grant_device_ordinal", immutable_grant.device_ordinal())
            .with_fact("domain_generation", domain.grant_identity().generation)
            .with_fact("grant_generation", immutable_grant.identity().generation));
        }
        Ok(Some((evidence, immutable_grant)))
    }

    fn validate_admitted_capability_evidence(
        &self,
        evidence: &AttemptCapabilityEvidence,
        plan: &QueryPlan,
    ) -> Result<()> {
        evidence.validate_covers_plan(plan)?;
        let execution_scope = query_planning::admission::canonical_execution_scope_fingerprint(plan);
        let caller_visible_plan = plan.clone_with_caller_visible_sink_contract();
        let expected_schema =
            query_planning::admission::log_safe_fingerprint(&format!("{:?}", caller_visible_plan.output_schema()));
        let scope_matches = evidence.execution_scopes().iter().filter(|shape| shape.matches(plan)).collect::<Vec<_>>();
        let matching_shapes = scope_matches
            .iter()
            .copied()
            .filter(|shape| shape.output_schema_fingerprint() == expected_schema)
            .collect::<Vec<_>>();
        if matching_shapes.is_empty() {
            if evidence.producer_occurrence_scope_count_for_plan(plan) > 0 {
                return Ok(());
            }
            if evidence.execution_manifest().and_then(|manifest| manifest.occurrence_for_plan(plan)).is_some() {
                return Ok(());
            }
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the admitted capability evidence output schema differs from the native plan",
            )
            .with_fact("reason", "admitted_capability_schema_mismatch")
            .with_fact("plan_scope", execution_scope)
            .with_fact("matching_shape_count", scope_matches.len()));
        }
        for shape in matching_shapes {
            shape.validate_manifest_core()?;
        }
        Ok(())
    }

    fn validate_admitted_occurrences(
        &self,
        evidence: &AttemptCapabilityEvidence,
        plan: &QueryPlan,
        occurrence_ids: &[String],
    ) -> Result<()> {
        evidence.validate_composed_occurrences(plan, occurrence_ids)
    }

    fn validate_admitted_evidence(
        &self,
        plan: &QueryPlan,
        occurrence_ids: Option<&[String]>,
    ) -> Result<Option<AttemptComposedScheduleProjection>> {
        let Some((submitted_evidence, immutable_grant)) = self.validated_admitted_identity(plan)? else {
            return Ok(None);
        };
        // An activated deferred-resident phase for this exact plan scope
        // replaces the submitted evidence with the scope actually proven at
        // activation (`with_admitted_gpu_inputs`); every other plan validates
        // against the evidence submitted at admission.
        let evidence = match &self.activated_deferred_resident_phase {
            Some(activated)
                if activated.consumer_execution_scope_fingerprint
                    == query_planning::admission::canonical_execution_scope_fingerprint(plan) =>
            {
                if activated.grant_identity != immutable_grant.identity() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "the activated deferred phase differs from the admitted plan or immutable grant",
                    )
                    .with_fact("reason", "deferred_resident_phase_identity_mismatch"));
                }
                &activated.capability_evidence
            },
            _ => submitted_evidence,
        };
        self.validate_admitted_capability_evidence(evidence, plan)?;
        if let Some(occurrence_ids) = occurrence_ids {
            self.validate_admitted_occurrences(evidence, plan, occurrence_ids)?;
        }
        Ok(None)
    }

    /// Universal pre-GPU gate for service-created execution contexts. Direct
    /// contexts remain unrestricted and return success without a evidence.
    pub fn validate_admitted_plan_capabilities(&self, plan: &QueryPlan) -> Result<()> {
        let Some((evidence, _)) = self.validated_admitted_identity(plan)? else {
            return Ok(());
        };
        self.validate_admitted_capability_evidence(evidence, plan)
    }

    /// Activates the deferred-resident phase for `plan` from its exact
    /// materialized resident inputs.
    ///
    /// Expected producer, occurrence, and schema identities come from the
    /// admitted compiled binding. Runtime frame identities are recorded only
    /// after those independent facts have matched.
    pub fn with_admitted_gpu_inputs(&self, plan: &QueryPlan, gpu_inputs: &GpuResidentInputMap) -> Result<Self> {
        let Some((submitted_evidence, immutable_grant)) = self.validated_admitted_identity(plan)? else {
            return Ok(self.clone());
        };
        if gpu_inputs.is_empty() {
            return Ok(self.clone());
        }
        // A phase activated for this same scope under a different grant is a
        // real admission violation. Otherwise the phase is re-derived from the
        // frames actually presented: one execution context legitimately runs a
        // plan more than once, and a later round may carry a different frame.
        if let Some(activated) = &self.activated_deferred_resident_phase
            && activated.consumer_execution_scope_fingerprint
                == query_planning::admission::canonical_execution_scope_fingerprint(plan)
            && activated.grant_identity != immutable_grant.identity()
        {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the activated deferred phase differs from the admitted immutable grant",
            )
            .with_fact("reason", "deferred_resident_phase_identity_mismatch"));
        }
        submitted_evidence.validate_covers_plan(plan)?;
        let expected_scopes = submitted_evidence
            .execution_scopes()
            .iter()
            .filter(|scope| scope.matches(plan) && !scope.producer_occurrences().is_empty())
            .collect::<Vec<_>>();
        if expected_scopes.is_empty() {
            return Ok(self.clone());
        }
        for scope in &expected_scopes {
            scope.validate_manifest_core()?;
        }
        let expected_scope = expected_scopes
            .iter()
            .copied()
            .find(|scope| {
                scope.producer_occurrences().iter().all(|expected| {
                    let Some(input) = gpu_inputs.get(expected.occurrence_id()) else {
                        return false;
                    };
                    let Some(actual_scope) = input.producer_execution_scope() else {
                        return false;
                    };
                    let actual_schema =
                        query_planning::admission::log_safe_fingerprint(&format!("{:?}", input.schema()));
                    (actual_scope == expected.execution_scope_fingerprint()
                        || input.producer_execution_plan().is_some_and(|plan| expected.matches_execution_plan(plan)))
                        && actual_schema == expected.output_schema_fingerprint()
                })
            })
            .unwrap_or(expected_scopes[0]);
        let mut capability_evidence = submitted_evidence.clone();
        capability_evidence.attest_cugraph_graph_execution_scope(plan);
        let inputs = expected_scope
            .producer_occurrences()
            .iter()
            .map(|expected| {
                let input_name = expected.occurrence_id();
                let input = gpu_inputs.get(input_name).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "the admitted deferred resident occurrence is missing",
                    )
                    .with_fact("reason", "deferred_resident_input_missing")
                    .with_fact("input_name", input_name)
                })?;
                let actual_scope = input.producer_execution_scope().ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "an admitted deferred resident occurrence has no producer scope",
                    )
                    .with_fact("reason", "deferred_resident_input_scope_missing")
                    .with_fact("input_name", input_name)
                })?;
                let actual_schema = query_planning::admission::log_safe_fingerprint(&format!("{:?}", input.schema()));
                if actual_scope != expected.execution_scope_fingerprint()
                    && !input.producer_execution_plan().is_some_and(|plan| expected.matches_execution_plan(plan))
                    || actual_schema != expected.output_schema_fingerprint()
                {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "a deferred resident input differs from its admitted producer binding",
                    )
                    .with_fact("reason", "deferred_resident_producer_binding_mismatch")
                    .with_fact("input_name", input_name)
                    .with_fact("expected_producer_scope", expected.execution_scope_fingerprint())
                    .with_fact("actual_producer_scope", actual_scope)
                    .with_fact("expected_schema", expected.output_schema_fingerprint())
                    .with_fact("actual_schema", actual_schema));
                }
                let physical_bytes = input.device_allocated_size_bytes()?;
                if physical_bytes > 0 && input.retained_owner_identity_for(&immutable_grant).is_none() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "a deferred resident input is not retained by the admitted grant",
                    )
                    .with_fact("reason", "deferred_resident_grant_owner_mismatch")
                    .with_fact("input_name", input_name)
                    .with_fact("input_device_bytes", physical_bytes));
                }
                let input_row_count = u64::try_from(input.frame().num_rows()).map_err(|_| {
                    QueryError::new(
                        QueryErrorCode::SizeOverflow,
                        "query_engine.runtime.context",
                        "the resident input row count exceeds the admission integer range",
                    )
                    .with_fact("reason", "deferred_resident_input_row_count_overflow")
                    .with_fact("input_name", input_name)
                })?;
                Ok(ActivatedDeferredResidentInput {
                    producer_execution_scope_fingerprint: actual_scope.to_owned(),
                    input_name: input_name.to_owned(),
                    input_schema: Arc::clone(input.schema()),
                    input_memory_identity: input.frame().memory_identity(),
                    input_row_count,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let activated = ActivatedDeferredResidentPhase {
            consumer_execution_scope_fingerprint: query_planning::admission::canonical_execution_scope_fingerprint(
                plan,
            ),
            inputs,
            capability_evidence,
            grant_identity: immutable_grant.identity(),
        };
        activated.validate_inputs(gpu_inputs, &immutable_grant)?;
        let mut context = self.clone();
        context.activated_deferred_resident_phase = Some(Arc::new(activated));
        Ok(context)
    }

    #[doc(hidden)]
    pub fn validate_admitted_plan_capabilities_with_gpu_inputs(
        &self,
        plan: &QueryPlan,
        gpu_inputs: &GpuResidentInputMap,
    ) -> Result<()> {
        self.validate_admitted_plan_capabilities(plan)?;
        let Some(activated) = self.activated_deferred_resident_phase.as_ref().filter(|activated| {
            activated.consumer_execution_scope_fingerprint
                == query_planning::admission::canonical_execution_scope_fingerprint(plan)
        }) else {
            return Ok(());
        };
        let immutable_grant = self.memory_grant()?.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the activated deferred phase lost its immutable grant",
            )
            .with_fact("reason", "deferred_resident_grant_missing")
        })?;
        activated.validate_inputs(gpu_inputs, &immutable_grant)
    }

    /// Validates one composed attempt against its admitted capability and
    /// resident-input identity contracts. Physical allocation remains owned
    /// by the immutable grant and live resident handles.
    pub fn validate_admitted_composed_attempt(
        &self,
        consumer_plan: &QueryPlan,
        occurrence_ids: &[String],
    ) -> Result<Option<AttemptComposedScheduleProjection>> {
        self.validate_admitted_evidence(consumer_plan, Some(occurrence_ids))
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn replace_admitted_capability_evidence_for_test(&mut self, evidence: Option<Arc<AttemptCapabilityEvidence>>) {
        self.admitted_capability_evidence = evidence;
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn admitted_capability_evidence_for_test(&self) -> Option<&AttemptCapabilityEvidence> {
        self.admitted_capability_evidence.as_deref()
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn replace_attempt_allocation_domain_for_test(&mut self, domain: Option<AttemptAllocationDomain>) {
        self.attempt_allocation_domain = domain;
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn attempt_allocation_domain_for_test(&self) -> Option<AttemptAllocationDomain> {
        self.attempt_allocation_domain.clone()
    }

    // Plain `#[cfg(test)]` would keep this invisible to downstream test
    // targets, which need it to cover the adapter fail-closed admission guard.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn replace_query_service_admitted_for_test(&mut self, admitted: bool) {
        self.query_service_admitted = admitted;
    }

    pub(super) fn install_root_stream_cleanup_completion(
        &self,
        before_quiescence: impl FnOnce(crate::runtime::service_stream::StreamCleanupOutcome) + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> bool {
        let Some(stream_lease) = &self.stream_lease_guard else {
            return false;
        };
        stream_lease.install_cleanup_completion(before_quiescence, after_quiescence)
    }

    pub(super) fn has_root_stream_lease(&self) -> bool {
        self.stream_lease_guard.is_some()
    }

    /// Returns the cooperative cancellation flag for this execution.
    pub fn cancellation(&self) -> &ExecutionCancellation {
        &self.cancellation
    }

    /// Fails with a structured cancellation error if the execution was cancelled.
    pub fn check_cancellation(&self, operation: &'static str) -> Result<()> {
        self.check_service_epoch(operation)?;
        self.cancellation.check(operation)
    }

    fn check_service_epoch(&self, operation: &'static str) -> Result<()> {
        if let Some(epoch) = &self.service_epoch {
            epoch.check_open(operation)?;
        }
        self.query_epoch.as_ref().map_or(Ok(()), |epoch| epoch.check_open(operation))
    }

    /// Reports the effective device placement derived from the runtime.
    pub fn device_placement(&self) -> DevicePlacement {
        DevicePlacement::SingleDevice { device_ordinal: self.runtime.device().ordinal() }
    }

    /// Reports the default device memory space used for bounded reservation accounting.
    pub fn device_memory_space(&self) -> MemorySpaceId {
        MemorySpaceId::Device { device_ordinal: self.runtime.device().ordinal() }
    }

    /// Returns the memory-resource construction policy recorded for this context.
    pub fn memory_policy(&self) -> MemoryResourcePolicy {
        self.memory_policy
    }

    /// Returns the engine-level memory control policy.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        match (self.memory_control_policy, self.memory_grant.as_ref()) {
            (
                MemoryControlPolicy::Bounded {
                    max_source_chunk_bytes,
                    max_row_groups_per_chunk,
                    max_retry_attempts,
                    min_retry_chunk_row_groups,
                    ..
                },
                Some(grant),
            ) => MemoryControlPolicy::Bounded {
                device_budget_bytes: grant.budget_bytes(),
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            },
            (policy, _) => policy,
        }
    }

    /// Returns the soft governor grant carried by this context, when present.
    pub fn memory_grant(&self) -> Result<Option<MemoryGrant>> {
        let Some(grant) = self.memory_grant.as_ref() else {
            return Ok(None);
        };
        let resource_epoch = self.acquire_resource_epoch("query_engine.runtime.context.memory_grant")?;
        Ok(Some(match resource_epoch {
            Some(resource_epoch) => grant.with_resource_epoch(resource_epoch),
            None => grant.clone(),
        }))
    }

    #[doc(hidden)]
    pub fn retain_resident_output_owner(&self, bytes: u64) -> Result<Option<ResidentOwnerLease>> {
        match self.memory_grant()? {
            Some(grant) => grant.retain_resident_owner(bytes),
            None => Ok(None),
        }
    }

    /// Creates a hard execution-budget token for this context's runtime stream.
    ///
    /// Service-admitted contexts attach a member of their immutable attempt
    /// domain; contexts without one use the runtime's standalone reservation
    /// path.
    pub fn execution_budget_token(&self, request: cudf::ExecutionBudgetRequest) -> Result<cudf::ExecutionBudgetToken> {
        let runtime = self.runtime()?;
        if let Some(lease) = self.attempt_member_lease.as_ref() {
            // One member per stream for the attempt's lifetime; per-stage
            // tokens share it rather than attaching a second member.
            return Ok(cudf::ExecutionBudgetToken::from_shared_lease(
                &runtime,
                request,
                lease.clone(),
                cudf::ExecutionBudgetEnforcement::QueryDomainMember,
            ));
        }
        if let Some(domain) = self.grant_memory_domain.as_ref() {
            return cudf::ExecutionBudgetToken::from_runtime_in_query_domain(&runtime, request, domain.as_ref())
                .map_err(|error| {
                    cudf_query_error(
                        "query_engine.runtime.context.execution_budget_token",
                        "failed to create query-domain execution budget token",
                        error,
                    )
                });
        }
        runtime.execution_budget_token(request).map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.context.execution_budget_token",
                "failed to create execution budget token",
                error,
            )
        })
    }

    /// Returns the immutable domain id for this admitted attempt.
    pub fn attempt_allocation_domain_id(&self) -> Option<u64> {
        self.attempt_allocation_domain.as_ref().map(AttemptAllocationDomain::domain_id)
    }

    /// Immutable grant identity owned by this admitted attempt domain.
    pub fn attempt_allocation_grant_identity(&self) -> Option<crate::runtime::gpu::GrantIdentity> {
        self.attempt_allocation_domain.as_ref().map(AttemptAllocationDomain::grant_identity)
    }

    /// Samples the allocation-domain current and peak usage.
    pub fn attempt_allocation_domain_usage(&self) -> Result<Option<cudf::QueryMemoryDomainUsage>> {
        self.attempt_allocation_domain.as_ref().map(AttemptAllocationDomain::usage).transpose()
    }

    /// Returns the effective bounded device budget used by reservation tracking.
    pub fn bounded_device_budget_bytes(&self) -> Option<u64> {
        match self.memory_control_policy() {
            MemoryControlPolicy::Unbounded => None,
            MemoryControlPolicy::Bounded { device_budget_bytes, .. } => Some(device_budget_bytes),
        }
    }

    /// Returns the execution boundary policy.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    /// Returns source scheduling controls for bounded source execution.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.source_scheduler_policy
    }

    /// Returns the source chunk planning mode.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.source_chunk_planning_mode
    }

    /// Returns the service-level lakehouse cache-fill resource configuration.
    pub fn lakehouse_cache_fill_config(&self) -> &LakehouseCacheFillConfig {
        &self.lakehouse_cache_fill_config
    }

    /// Returns the service-level CUDA stream pool used by source-read workers.
    pub fn service_stream_pool(&self) -> QueryServiceStreamPool {
        QueryServiceStreamPool {
            pool: self.service_stream_pool.clone(),
            service_epoch: self.service_epoch.clone(),
            query_anchor: self.query_anchor.clone(),
        }
    }

    /// Timeout used by blocking lakehouse cache maintenance resource acquisition.
    pub fn lakehouse_cache_fill_acquire_timeout(&self) -> Duration {
        self.lakehouse_cache_fill_config.acquire_timeout
    }

    /// Returns the service-wide background lakehouse cache-fill byte budget.
    pub fn lakehouse_cache_fill_budget(&self) -> CacheFillByteBudget {
        self.lakehouse_cache_fill_budget.clone()
    }

    /// Maximum background lakehouse cache-fill operations admitted concurrently.
    pub fn lakehouse_cache_fill_max_concurrent_ops(&self) -> usize {
        self.lakehouse_cache_fill_concurrency.max_concurrent_ops()
    }

    /// Structured asynchronous lakehouse cache-fill completion/failure events.
    #[doc(hidden)]
    pub fn lakehouse_cache_fill_events(&self) -> LakehouseCacheFillEventSink {
        self.lakehouse_cache_fill_events.clone()
    }

    /// Returns credential-free background cache-fill events observed by this context so far.
    pub fn lakehouse_cache_fill_event_snapshots(&self) -> Vec<LakehouseCacheFillEventSnapshot> {
        self.lakehouse_cache_fill_events.events().iter().map(LakehouseCacheFillEventSnapshot::from).collect()
    }

    /// Drains credential-free background cache-fill events observed by this context so far.
    pub fn drain_lakehouse_cache_fill_event_snapshots(&self) -> Vec<LakehouseCacheFillEventSnapshot> {
        self.lakehouse_cache_fill_events.drain().iter().map(LakehouseCacheFillEventSnapshot::from).collect()
    }

    #[doc(hidden)]
    pub fn acquire_lakehouse_cache_fill_operation_with_timeout(
        &self,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillOperationPermit>> {
        self.lakehouse_cache_fill_concurrency.acquire_with_timeout(&self.cancellation, timeout, operation)
    }

    #[doc(hidden)]
    pub fn acquire_lakehouse_cache_fill_budget_with_timeout(
        &self,
        bytes: u64,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillBytePermit>> {
        self.lakehouse_cache_fill_budget.acquire_with_timeout(&self.cancellation, bytes, timeout, operation)
    }

    #[doc(hidden)]
    pub fn s3_object_store(&self, key: S3ObjectStoreKey) -> Result<Arc<dyn ::object_store::ObjectStore>> {
        self.source_object_stores.s3_store(key)
    }

    #[doc(hidden)]
    pub fn source_storage_profile(&self, id: &str) -> Result<&SourceStorageProfile> {
        self.source_object_stores.profile(id)
    }

    /// Resolves default-chain AWS credentials from the object-store provider.
    ///
    /// Remote KvikIO reads use this to pass resolved identity into lower layers
    /// that lack a default chain. Recent results are reused briefly; on a cache
    /// miss, resolution runs on the object-store control runtime while this call
    /// blocks for the result.
    #[doc(hidden)]
    pub fn resolve_s3_default_chain_credential(
        &self,
        key: S3ObjectStoreKey,
        operation: &'static str,
    ) -> Result<Arc<::object_store::aws::AwsCredential>> {
        if let Some(credential) = self.source_object_stores.recently_resolved_s3_credential(&key) {
            return Ok(credential);
        }
        let storage_profile_id = key.storage_profile_id.clone();
        let provider = self.source_object_stores.s3_credential_provider(key.clone())?;
        let (sender, receiver) = std::sync::mpsc::channel();
        self.object_store_control_runtime.spawn(async move {
            let credential = provider.get_credential().await;
            let _ = sender.send(credential);
        });
        let credential_unavailable = |query_error: QueryError| {
            let query_error = query_error
                .with_fact("resource", "remote_parquet_source")
                .with_fact("credential_source", "default_chain")
                .with_fact("reason", "default_chain_credentials_unavailable");
            match storage_profile_id.as_deref() {
                Some(profile_id) => query_error.with_fact("storage_profile_id", profile_id),
                None => query_error,
            }
        };
        let credential = receiver.recv().map_err(|_error| {
            credential_unavailable(QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "source credential resolution worker stopped before returning a result",
            ))
        })?;
        let credential = credential.map_err(|error| {
            credential_unavailable(
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    operation,
                    "failed to resolve AWS default-chain credentials for native remote-direct read",
                )
                .with_source(error),
            )
        })?;
        self.source_object_stores.record_resolved_s3_credential(key, Arc::clone(&credential));
        Ok(credential)
    }

    #[doc(hidden)]
    pub fn spawn_cache_maintenance_fetch(
        &self,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        Self::spawn_cache_maintenance_fetch_on_runtime(
            self.object_store_control_runtime.as_ref(),
            store,
            location,
            on_result,
        )
    }

    /// Spawns caller-provided object-store control-plane work.
    ///
    /// The caller composes any combination of `get_range` / `get_ranges` calls
    /// inside the future. Elapsed time covers the entire future, including any
    /// rewrite work the caller performs inside it.
    #[doc(hidden)]
    pub fn spawn_object_store_control_task<F, Fut, T>(
        &self,
        future_factory: F,
        on_result: impl FnOnce(ObjectStoreCustomFetchResult<T>) + Send + 'static,
    ) -> tokio::task::JoinHandle<()>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.object_store_control_runtime.spawn(async move {
            let timer = Instant::now();
            let value = future_factory().await;
            on_result(ObjectStoreCustomFetchResult { value, elapsed: timer.elapsed() });
        })
    }

    /// Spawns caller-provided asynchronous lakehouse cache maintenance.
    #[doc(hidden)]
    pub fn spawn_lakehouse_cache_maintenance<F, Fut, T>(
        &self,
        future_factory: F,
        on_result: impl FnOnce(ObjectStoreCustomFetchResult<T>) + Send + 'static,
    ) -> tokio::task::JoinHandle<()>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.object_store_control_runtime.spawn(async move {
            let timer = Instant::now();
            let value = future_factory().await;
            on_result(ObjectStoreCustomFetchResult { value, elapsed: timer.elapsed() });
        })
    }

    /// Returns deployment-level IO capability diagnostics.
    pub fn deployment_io_capability(&self) -> &DeploymentIoCapability {
        &self.deployment_io_capability
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[cfg_attr(not(test), expect(dead_code, reason = "used by in-crate runtime source tests"))]
    pub(super) fn fetch_object_store_bytes_on_runtime(
        object_store_control_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
    ) -> Result<Bytes> {
        let (sender, receiver) = std::sync::mpsc::channel();
        Self::spawn_cache_maintenance_fetch_on_runtime(object_store_control_runtime, store, location, move |result| {
            let _ = sender.send(result.bytes);
        });
        receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.lakehouse_cache.maintenance.receive",
                "object-store cache maintenance worker stopped before returning a result",
            )
            .with_source(error)
        })?
    }

    fn spawn_cache_maintenance_fetch_on_runtime(
        object_store_control_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        object_store_control_runtime.spawn(async move {
            let path_fact = query_planning::source::redacted_object_store_path_diagnostic(location.as_ref());
            let bytes = async move {
                let result = store.get(&location).await?;
                result.bytes().await
            }
            .await
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "query_engine.lakehouse_cache.maintenance.object_store",
                    "failed to fetch lakehouse cache object through object_store",
                )
                .with_fact("path", path_fact)
            });
            on_result(ObjectStoreFetchResult { bytes });
        });
    }

    /// Returns the selected execution contract for floating statistical aggregates.
    pub fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.statistical_aggregate_execution_mode
    }

    /// Returns whether this query preserves DataFusion input order for floating
    /// statistical aggregates.
    ///
    /// Native execution derives this per query from the aggregate execution
    /// mode and the plan shape, then carries it on every context clone so
    /// source-read workers on spawned threads observe the same scope as the
    /// caller thread.
    pub fn datafusion_order_tracking(&self) -> bool {
        self.datafusion_order_tracking
    }

    /// Attaches the per-query DataFusion exact-order scope to this context clone.
    pub fn with_datafusion_order_tracking(mut self, enabled: bool) -> Self {
        self.datafusion_order_tracking = enabled;
        self
    }

    /// Returns grouped aggregate strategy execution policy.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.aggregate_strategy_execution_options
    }

    /// Returns expert local-throughput source chunk planning policy.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.source_chunk_planning_options
    }
}

fn validate_runtime_observation_query_id(observation: &RuntimeObservationHandle, query_id: &QueryId) -> Result<()> {
    let Some(observation_query_id) = observation.query_id() else {
        return Ok(());
    };
    if observation_query_id.as_str() == query_id.as_str() {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.runtime_observation.identity",
        "execution context query id conflicts with its runtime observation session",
    )
    .with_fact("reason", "runtime_observation_query_id_mismatch")
    .with_fact("context_query_id", query_id)
    .with_fact("observation_query_id", observation_query_id))
}

fn validate_runtime_observation_trace_id(observation: &RuntimeObservationHandle, trace_id: &TraceId) -> Result<()> {
    if !observation.is_enabled() {
        return Ok(());
    }
    let observation_trace_id = observation.trace_id().map(RuntimeTraceId::as_str);
    if observation_trace_id == Some(trace_id.as_str()) {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.runtime_observation.identity",
        "execution context trace id conflicts with its runtime observation session",
    )
    .with_fact("reason", "runtime_observation_trace_id_mismatch")
    .with_fact("context_trace_id", trace_id)
    .with_fact("observation_trace_id", observation_trace_id.unwrap_or("none")))
}
