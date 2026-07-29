//! Native execution context state and query-local runtime controls.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use ::object_store::ObjectStoreExt;
#[cfg(test)]
use bytes::Bytes;

use super::{
    CacheFillByteBudget, CacheFillBytePermit, DeploymentIoCapability, DevicePlacement, ExecutionBoundaryPolicy,
    GpuExecutionPool, GpuExecutorLease, LakehouseCacheFillConfig, MemoryControlPolicy, MemoryGrant,
    MemoryResourcePolicy, MemorySpaceId, ServiceStreamPool, SourceChunkPlanningMode, SourceReadStreamLease,
    SourceStorageProfile, StreamLease, cache,
    cache_events::{LakehouseCacheFillEventSink, LakehouseCacheFillEventSnapshot},
    cache_fill::{CacheFillConcurrencyGate, CacheFillOperationPermit},
    default_service_stream_pool_size_for_device,
    io_lifecycle::{IoLifecycle, IoLifecycleLease},
    object_store::{
        ObjectStoreCustomFetchResult, ObjectStoreFetchResult, S3ObjectStoreKey, SourceObjectStoreRegistry,
        default_object_store_control_runtime,
    },
};
use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    graph_memory::GraphMemoryCalibrationOverride,
    metrics::{QueryId, TraceId},
    mode::{NativePipelineExecutionMode, StddevSampExecutionMode},
    observability::{
        RuntimeObservationHandle, RuntimeObservationSession, RuntimeStreamLeaseScope, RuntimeTaskId, RuntimeTraceId,
        RuntimeWorkerId,
    },
};

mod builder;

pub(crate) use builder::ExecutionContextBuilder;
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
}

impl ExecutionCancellation {
    /// Creates a non-cancelled execution flag.
    pub fn new() -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)) }
    }

    /// Marks the execution as cancelled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Returns a structured native error if cancellation has been requested.
    pub fn check(&self, operation: &'static str) -> Result<()> {
        if self.is_cancelled() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::Cancelled,
                operation,
                "native query execution was cancelled",
            )
            .with_fact("cancellation_requested", true));
        }
        Ok(())
    }
}

impl Default for ExecutionCancellation {
    fn default() -> Self {
        Self::new()
    }
}

type QueryServiceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;
type QueryResourceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;

/// Revocable lifetime shared by every resource derived from one query service.
#[derive(Clone)]
pub(crate) struct QueryServiceEpoch {
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
    pub(crate) fn new(on_idle: Option<QueryServiceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryServiceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                idle_state: Mutex::new(QueryServiceEpochIdleState { on_idle, complete: false }),
                idle_changed: Condvar::new(),
            }),
        }
    }

    pub(crate) fn retain_existing_user(&self) -> QueryServiceEpochUser {
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

    pub(crate) fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_service_epoch_closed(operation)) }
    }

    pub(crate) fn invalidate(&self) {
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

    pub(crate) fn wait_for_idle_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
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
pub(crate) struct QueryServiceEpochUser {
    epoch: QueryServiceEpoch,
}

/// Revocable lifetime shared by resources derived from one admitted query.
///
/// The epoch owns only revocation and drain accounting. Each counted anchor
/// owns the query lifetime separately, so the epoch cannot form a cycle with
/// the query's uncounted root-context template.
#[derive(Clone)]
pub(crate) struct QueryResourceEpoch {
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
    pub(crate) fn new(on_idle: Option<QueryResourceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryResourceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                on_idle: Mutex::new(on_idle),
            }),
        }
    }

    pub(crate) fn acquire(
        &self,
        owner: Arc<dyn Send + Sync>,
        operation: &'static str,
    ) -> Result<Arc<QueryResourceAnchor>> {
        self.check_open(operation)?;
        self.inner.active_users.fetch_add(1, Ordering::AcqRel);
        let anchor = Arc::new(QueryResourceAnchor { epoch: self.clone(), _owner: owner });
        if let Err(error) = self.check_open(operation) {
            drop(anchor);
            return Err(error);
        }
        Ok(anchor)
    }

    pub(crate) fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_resource_epoch_closed(operation)) }
    }

    pub(crate) fn invalidate(&self) {
        self.inner.open.store(false, Ordering::Release);
        self.notify_if_idle();
    }

    pub(crate) fn is_idle(&self) -> bool {
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
pub(crate) struct QueryResourceAnchor {
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
pub(crate) struct QueryServiceResourceEpoch {
    epoch: QueryServiceEpoch,
    _user: QueryServiceEpochUser,
    query_anchor: Option<Arc<QueryResourceAnchor>>,
}

impl QueryServiceResourceEpoch {
    pub(crate) fn acquire(
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

    pub(crate) fn check_open(&self, operation: &'static str) -> Result<()> {
        self.epoch.check_open(operation)?;
        if let Some(anchor) = &self.query_anchor {
            anchor.check_open(operation)?;
        }
        Ok(())
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
        let operation = "nexus_query_engine.runtime.context.service_stream_pool.acquire";
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

/// Native source scheduling controls.
///
/// The scheduler preserves source order by default and uses `max_source_reads_in_flight`
/// as an upper bound on active source reads. A value of `1` keeps the historical
/// serialized source path. Values greater than `1` allow bounded execution to
/// prefetch the next source chunk while chunk-local operators consume the
/// current chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSchedulerPolicy {
    max_source_reads_in_flight: usize,
}

impl SourceSchedulerPolicy {
    /// Serialized source execution.
    pub const fn serialized() -> Self {
        Self { max_source_reads_in_flight: 1 }
    }

    /// Creates a source scheduler policy.
    pub const fn new(max_source_reads_in_flight: usize) -> Self {
        Self { max_source_reads_in_flight }
    }

    /// Maximum active source reads allowed by the scheduler.
    pub const fn max_source_reads_in_flight(self) -> usize {
        self.max_source_reads_in_flight
    }

    /// Returns whether source prefetch is enabled.
    pub const fn enables_prefetch(self) -> bool {
        self.max_source_reads_in_flight > 1
    }

    /// Validates the scheduler bound without constructing runtime resources.
    pub fn validate(self) -> Result<()> {
        if self.max_source_reads_in_flight == 0 {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_scheduler_policy.validate",
                "source scheduler max_source_reads_in_flight must be greater than zero",
            )
            .with_fact("field", "max_source_reads_in_flight")
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for SourceSchedulerPolicy {
    fn default() -> Self {
        Self { max_source_reads_in_flight: 2 }
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
    /// Optional soft per-query memory grant backing bounded execution.
    memory_grant: Option<MemoryGrant>,
    /// Optional native arena that mirrors the current grant budget.
    grant_reservation_arena: Option<Arc<cudf::ReservationArena>>,
    /// Optional native query memory domain that mirrors the current grant budget.
    grant_memory_domain: Option<Arc<cudf::QueryMemoryDomain>>,
    /// Execution boundary contract used by source reads, build/probe reuse, and
    /// result materialization.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Source scheduling policy for bounded local Parquet reads.
    source_scheduler_policy: SourceSchedulerPolicy,
    /// Source chunk planning mode for bounded local Parquet reads.
    source_chunk_planning_mode: SourceChunkPlanningMode,
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
    /// Selected native bounded-pipeline execution mode.
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    /// Selected execution contract for floating statistical aggregates.
    stddev_samp_execution_mode: StddevSampExecutionMode,
    /// Runtime threshold policy for grouped aggregate strategy selection.
    aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert policy for local-throughput source chunk planning.
    source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Optional graph calibration metadata supplied by fixture/config.
    graph_memory_calibration_override: GraphMemoryCalibrationOverride,
    /// Optional adaptive-only memory history store.
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    /// Optional durable GPU hash-join build cache.
    ///
    /// Required when the boundary policy asks for durable build
    /// materialization. Query-scoped build materialization does not use this
    /// cache.
    hash_join_build_cache: Option<Arc<cache::hash_join::GpuHashJoinBuildCache>>,
    /// Optional durable GPU local-Parquet source cache.
    ///
    /// Bounded execution charges retained cache bytes to the reservation ledger
    /// before admitting source or operator work.
    local_parquet_source_cache: Option<Arc<cache::local_parquet::GpuLocalParquetSourceCache>>,
    /// Optional durable GPU resident graph cache.
    #[cfg(feature = "cugraph")]
    resident_graph_cache: Option<Arc<cache::resident_graph::GpuResidentGraphCache>>,
    /// Exact-runtime graph entry reserved during warm-priced admission.
    #[cfg(feature = "cugraph")]
    reserved_resident_graph_hit: Option<Arc<ReservedResidentGraphHit>>,
    /// Optional owned stream lease kept alive by contexts derived from it.
    stream_lease_guard: Option<Arc<StreamLease>>,
    /// Whether this context was created from a QueryService admission handle.
    query_service_admitted: bool,
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

/// Root execution context paired with the unique runtime observation owner.
///
/// Clones obtained from [`Self::context`] contain only the recording handle.
/// The owner remains here until query-owned cleanup can seal it.
#[derive(Debug)]
pub struct RuntimeObservedExecutionContext {
    context: ExecutionContext,
    observation: RuntimeObservationSession,
}

impl RuntimeObservedExecutionContext {
    pub(super) fn new(context: ExecutionContext, observation: RuntimeObservationSession) -> Self {
        Self { context, observation }
    }

    /// Returns the cloneable execution context carrying recording access only.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }

    /// Returns the owner-only terminal-seal capability.
    pub fn observation(&self) -> &RuntimeObservationSession {
        &self.observation
    }

    /// Separates the cloneable execution context from its owner capability.
    pub fn into_parts(self) -> (ExecutionContext, RuntimeObservationSession) {
        (self.context, self.observation)
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
    pub(super) runtime_observation: RuntimeObservationHandle,
    pub(super) runtime_observation_owner: Option<crate::observability::RuntimeObservationOwner>,
    pub(super) runtime_execution_scope: Option<crate::observability::RuntimeExecutionObservationScope>,
    pub(super) runtime_stream_lease_scope: RuntimeStreamLeaseScope,
    pub(super) runtime: Arc<cudf::Runtime>,
    pub(super) gpu_execution_pool: GpuExecutionPool,
    pub(super) memory_policy: MemoryResourcePolicy,
    pub(super) memory_control_policy: MemoryControlPolicy,
    pub(super) memory_grant: Option<MemoryGrant>,
    pub(super) grant_reservation_arena: Option<Arc<cudf::ReservationArena>>,
    pub(super) grant_memory_domain: Option<Arc<cudf::QueryMemoryDomain>>,
    pub(super) boundary_policy: ExecutionBoundaryPolicy,
    pub(super) source_scheduler_policy: SourceSchedulerPolicy,
    pub(super) source_chunk_planning_mode: SourceChunkPlanningMode,
    pub(super) lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    pub(super) service_stream_pool: ServiceStreamPool,
    pub(super) lakehouse_cache_fill_budget: CacheFillByteBudget,
    pub(super) lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    pub(super) lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    pub(super) object_store_control_runtime: Arc<tokio::runtime::Runtime>,
    pub(super) source_object_stores: Arc<SourceObjectStoreRegistry>,
    pub(super) deployment_io_capability: DeploymentIoCapability,
    pub(super) cancellation: ExecutionCancellation,
    pub(super) native_pipeline_execution_mode: NativePipelineExecutionMode,
    pub(super) stddev_samp_execution_mode: StddevSampExecutionMode,
    pub(super) aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    pub(super) source_chunk_planning_options: SourceChunkPlanningOptions,
    pub(super) graph_memory_calibration_override: GraphMemoryCalibrationOverride,
    pub(super) adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    pub(super) hash_join_build_cache: Option<Arc<cache::hash_join::GpuHashJoinBuildCache>>,
    pub(super) local_parquet_source_cache: Option<Arc<cache::local_parquet::GpuLocalParquetSourceCache>>,
    #[cfg(feature = "cugraph")]
    pub(super) resident_graph_cache: Option<Arc<cache::resident_graph::GpuResidentGraphCache>>,
    #[cfg(feature = "cugraph")]
    pub(super) reserved_resident_graph_hit: Option<Arc<ReservedResidentGraphHit>>,
    pub(super) stream_lease_guard: Option<Arc<StreamLease>>,
    pub(super) query_service_admitted: bool,
}

#[cfg(feature = "cugraph")]
pub(super) struct ReservedResidentGraphHit {
    resident_key: crate::plan::ResidentGraphKey,
    preparation_pin: Mutex<Option<cache::resident_graph::GpuResidentGraphPreparationPin>>,
}

#[cfg(feature = "cugraph")]
impl std::fmt::Debug for ReservedResidentGraphHit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReservedResidentGraphHit").field("resident_key", &self.resident_key).finish_non_exhaustive()
    }
}

#[cfg(feature = "cugraph")]
impl ReservedResidentGraphHit {
    pub(super) fn new(preparation_pin: cache::resident_graph::GpuResidentGraphPreparationPin) -> Self {
        Self {
            resident_key: preparation_pin.resident_key().clone(),
            preparation_pin: Mutex::new(Some(preparation_pin)),
        }
    }
}

impl ExecutionContext {
    /// Starts a builder for an [`ExecutionContext`].
    pub(crate) fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    /// Builds a context around an owned cudf runtime.
    ///
    /// The resulting context records [`MemoryResourcePolicy::ExternalRuntime`]
    /// because the runtime, including its allocator and stream, has already been
    /// constructed by the caller.
    pub(crate) fn from_runtime(runtime: cudf::Runtime) -> Result<Self> {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    /// Builds a context around a shared cudf runtime.
    ///
    /// Use this when multiple plans or adapter layers must share the same CUDA
    /// stream, device placement, allocator policy, or boundary observer state.
    pub(crate) fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Result<Self> {
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
            runtime_observation: RuntimeObservationHandle::disabled(),
            runtime_observation_owner: None,
            runtime_execution_scope: None,
            runtime_stream_lease_scope: RuntimeStreamLeaseScope::default(),
            runtime,
            gpu_execution_pool,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            memory_grant: None,
            grant_reservation_arena: None,
            grant_memory_domain: None,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
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
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            graph_memory_calibration_override: GraphMemoryCalibrationOverride::default(),
            adaptive_memory_history: None,
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
            #[cfg(feature = "cugraph")]
            resident_graph_cache: None,
            #[cfg(feature = "cugraph")]
            reserved_resident_graph_hit: None,
            stream_lease_guard: None,
            query_service_admitted: false,
        }))
    }

    pub(super) fn from_parts(parts: ExecutionContextParts) -> Self {
        debug_assert!(
            parts.grant_reservation_arena.is_none() || parts.grant_memory_domain.is_none(),
            "execution context cannot carry both grant reservation arena and query memory domain"
        );
        Self {
            service_epoch: parts.service_epoch,
            query_epoch: parts.query_epoch,
            query_anchor: parts.query_anchor,
            io_lifecycle: parts.io_lifecycle,
            query_id: parts.query_id,
            trace_id: parts.trace_id,
            runtime_observation: parts.runtime_observation,
            runtime_observation_owner: parts.runtime_observation_owner,
            runtime_execution_scope: parts.runtime_execution_scope,
            runtime_stream_lease_scope: parts.runtime_stream_lease_scope,
            runtime: parts.runtime,
            gpu_execution_pool: parts.gpu_execution_pool,
            memory_policy: parts.memory_policy,
            memory_control_policy: parts.memory_control_policy,
            memory_grant: parts.memory_grant,
            grant_reservation_arena: parts.grant_reservation_arena,
            grant_memory_domain: parts.grant_memory_domain,
            boundary_policy: parts.boundary_policy,
            source_scheduler_policy: parts.source_scheduler_policy,
            source_chunk_planning_mode: parts.source_chunk_planning_mode,
            lakehouse_cache_fill_config: parts.lakehouse_cache_fill_config,
            service_stream_pool: parts.service_stream_pool,
            lakehouse_cache_fill_budget: parts.lakehouse_cache_fill_budget,
            lakehouse_cache_fill_concurrency: parts.lakehouse_cache_fill_concurrency,
            lakehouse_cache_fill_events: parts.lakehouse_cache_fill_events,
            object_store_control_runtime: parts.object_store_control_runtime,
            source_object_stores: parts.source_object_stores,
            deployment_io_capability: parts.deployment_io_capability,
            cancellation: parts.cancellation,
            native_pipeline_execution_mode: parts.native_pipeline_execution_mode,
            stddev_samp_execution_mode: parts.stddev_samp_execution_mode,
            aggregate_strategy_execution_options: parts.aggregate_strategy_execution_options,
            source_chunk_planning_options: parts.source_chunk_planning_options,
            graph_memory_calibration_override: parts.graph_memory_calibration_override,
            adaptive_memory_history: parts.adaptive_memory_history,
            hash_join_build_cache: parts.hash_join_build_cache,
            local_parquet_source_cache: parts.local_parquet_source_cache,
            #[cfg(feature = "cugraph")]
            resident_graph_cache: parts.resident_graph_cache,
            #[cfg(feature = "cugraph")]
            reserved_resident_graph_hit: parts.reserved_resident_graph_hit,
            stream_lease_guard: parts.stream_lease_guard,
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

    /// Derives the service-only base runtime used to establish local-Parquet
    /// cache affinity without retaining the current query admission anchor.
    pub(super) fn local_parquet_cache_owner_runtime(&self, operation: &'static str) -> Result<Arc<cudf::Runtime>> {
        self.check_service_epoch(operation)?;
        let Some(service_epoch) = &self.service_epoch else {
            return Ok(Arc::clone(&self.runtime));
        };
        let resource_epoch = QueryServiceResourceEpoch::acquire(service_epoch, None, operation)?;
        let retained: Arc<dyn Send + Sync> = resource_epoch;
        self.runtime.with_retained_lifetime(retained).map(Arc::new).map_err(|error| {
            QueryError::from_cudf(operation, "failed to derive local Parquet cache owner runtime", error)
        })
    }

    /// Attaches the current query anchor to a query-neutral local-Parquet
    /// cache runtime while preserving its exact stream and allocator binding.
    pub(super) fn local_parquet_cache_query_runtime(
        &self,
        cache_runtime: &Arc<cudf::Runtime>,
        operation: &'static str,
    ) -> Result<Arc<cudf::Runtime>> {
        self.check_service_epoch(operation)?;
        if !cudf::Runtime::shares_affinity(&self.runtime, cache_runtime) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "local Parquet cache runtime does not share the current query runtime affinity",
            )
            .with_fact("reason", "local_parquet_cache_runtime_affinity_mismatch")
            .with_fact("query_runtime_affinity_id", self.runtime.affinity_id().get())
            .with_fact("cache_runtime_affinity_id", cache_runtime.affinity_id().get()));
        }
        let Some(resource_epoch) = self.acquire_resource_epoch(operation)? else {
            return Ok(Arc::clone(cache_runtime));
        };
        let retained: Arc<dyn Send + Sync> = resource_epoch;
        cache_runtime.with_retained_lifetime(retained).map(Arc::new).map_err(|error| {
            QueryError::from_cudf(operation, "failed to derive query-scoped local Parquet cache runtime", error)
        })
    }

    /// Derives the service-only runtime used to own resident graph cache state.
    ///
    /// This is intentionally visible only inside `runtime`: cache-owned native
    /// state must retain the service execution domain without inheriting the
    /// admitted query anchor carried by the outward context.
    #[cfg(feature = "cugraph")]
    pub(super) fn resident_graph_cache_owner_runtime(&self, operation: &'static str) -> Result<Arc<cudf::Runtime>> {
        self.check_service_epoch(operation)?;
        let Some(service_epoch) = &self.service_epoch else {
            return Ok(Arc::clone(&self.runtime));
        };
        let resource_epoch = QueryServiceResourceEpoch::acquire(service_epoch, None, operation)?;
        let retained: Arc<dyn Send + Sync> = resource_epoch;
        self.runtime.with_retained_lifetime(retained).map(Arc::new).map_err(|error| {
            QueryError::from_cudf(operation, "failed to derive resident graph cache owner runtime", error)
        })
    }

    /// Attaches the current query anchor to a query-neutral resident-cache runtime.
    #[cfg(feature = "cugraph")]
    pub(super) fn resident_graph_query_runtime(
        &self,
        cache_runtime: &Arc<cudf::Runtime>,
        operation: &'static str,
    ) -> Result<Arc<cudf::Runtime>> {
        self.check_service_epoch(operation)?;
        if !cudf::Runtime::shares_affinity(&self.runtime, cache_runtime) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "resident graph cache runtime does not share the current query runtime affinity",
            )
            .with_fact("reason", "resident_graph_cache_runtime_affinity_mismatch")
            .with_fact("query_runtime_affinity_id", self.runtime.affinity_id().get())
            .with_fact("cache_runtime_affinity_id", cache_runtime.affinity_id().get()));
        }
        let Some(resource_epoch) = self.acquire_resource_epoch(operation)? else {
            return Ok(Arc::clone(cache_runtime));
        };
        let retained: Arc<dyn Send + Sync> = resource_epoch;
        cache_runtime.with_retained_lifetime(retained).map(Arc::new).map_err(|error| {
            QueryError::from_cudf(operation, "failed to derive query-scoped resident graph execution runtime", error)
        })
    }

    /// Returns the cudf runtime used for all GPU operations.
    pub fn runtime(&self) -> Result<cudf::Runtime> {
        let operation = "nexus_query_engine.runtime.context.runtime";
        let Some(resource_epoch) = self.acquire_resource_epoch(operation)? else {
            return Ok(self.runtime.as_ref().clone());
        };
        let retained: Arc<dyn Send + Sync> = resource_epoch;
        self.runtime
            .with_retained_lifetime(retained)
            .map_err(|error| QueryError::from_cudf(operation, "failed to derive a counted query runtime", error))
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
    pub fn gpu_executor(&self) -> Result<GpuExecutorLease> {
        self.gpu_execution_pool().map(GpuExecutorLease::from_pool)
    }

    pub(crate) fn gpu_execution_pool(&self) -> Result<GpuExecutionPool> {
        let resource_epoch = self.acquire_resource_epoch("nexus_query_engine.runtime.context.gpu_execution_pool")?;
        Ok(match resource_epoch {
            Some(resource_epoch) => self.gpu_execution_pool.with_resource_epoch(resource_epoch),
            None => self.gpu_execution_pool.clone(),
        })
    }

    /// Returns the query-wide diagnostic correlation id.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Returns the shared recording handle without terminal-seal authority.
    pub fn runtime_observation(&self) -> &RuntimeObservationHandle {
        &self.runtime_observation
    }

    /// Returns the shared observation owner carried from query admission.
    pub fn runtime_observation_owner(&self) -> Option<crate::observability::RuntimeObservationOwner> {
        self.runtime_observation_owner.clone()
    }

    /// Returns the active native plan-execution scope.
    pub fn runtime_execution_scope(&self) -> Option<&crate::observability::RuntimeExecutionObservationScope> {
        self.runtime_execution_scope.as_ref()
    }

    /// Returns a clone carrying one invocation's recording handle.
    ///
    /// Seal authority remains with the execution lifecycle that supplied the
    /// handle. Identity is validated before the clone can reach workers.
    pub(crate) fn with_runtime_observation_handle(&self, observation: RuntimeObservationHandle) -> Result<Self> {
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

    /// Returns a clone carrying the shared owner created before admission.
    pub(crate) fn with_runtime_observation_owner(
        &self,
        owner: crate::observability::RuntimeObservationOwner,
    ) -> Result<Self> {
        let mut context = self.with_runtime_observation_handle(owner.handle())?;
        context.runtime_observation_owner = Some(owner);
        Ok(context)
    }

    /// Returns a clone scoped to one native plan invocation.
    pub(crate) fn with_runtime_execution_scope(
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

    /// Acquires a same-device leased context for a scheduled source-read worker.
    pub(crate) fn acquire_source_read_lease(
        &self,
        source_worker: Option<(RuntimeTaskId, RuntimeWorkerId)>,
    ) -> Result<SourceReadStreamLease> {
        let mut stream_scope = self.runtime_stream_lease_scope.clone();
        if let Some((task_id, worker_id)) = source_worker {
            stream_scope = stream_scope.with_source_worker(task_id, worker_id);
        }
        let stream_lease = self.service_stream_pool().acquire_observed_with_cancellation(
            &self.cancellation,
            "nexus_query_engine.runtime.source_read_stream.acquire",
            self.runtime_observation.clone(),
            stream_scope,
        )?;
        let context = self.with_stream_lease(&stream_lease)?;
        Ok(SourceReadStreamLease::new(context, stream_lease))
    }

    /// Builds a short-lived execution context from a borrowed stream lease.
    ///
    /// The returned context shares this context's query policies, caches, and
    /// cancellation flag, but executes GPU work on the leased runtime. The
    /// caller must keep `stream_lease` alive until all work submitted through
    /// the returned context has completed.
    pub fn with_stream_lease(&self, stream_lease: &StreamLease) -> Result<Self> {
        self.check_service_epoch("nexus_query_engine.runtime.context.with_stream_lease")?;
        validate_service_stream_pool_device(stream_lease.pool(), self.runtime.device())?;
        Ok(Self::from_parts(ExecutionContextParts {
            service_epoch: self.service_epoch.clone(),
            query_epoch: self.query_epoch.clone(),
            query_anchor: self.query_anchor.clone(),
            io_lifecycle: self.io_lifecycle.clone(),
            query_id: self.query_id.clone(),
            trace_id: self.trace_id.clone(),
            runtime_observation: self.runtime_observation.clone(),
            runtime_observation_owner: self.runtime_observation_owner.clone(),
            runtime_execution_scope: self.runtime_execution_scope.clone(),
            runtime_stream_lease_scope: self.runtime_stream_lease_scope.clone(),
            runtime: stream_lease.shared_runtime()?,
            gpu_execution_pool: self.gpu_execution_pool.clone(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy,
            memory_grant: self.memory_grant.clone(),
            grant_reservation_arena: self.grant_reservation_arena.clone(),
            grant_memory_domain: self.grant_memory_domain.clone(),
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_chunk_planning_mode: self.source_chunk_planning_mode,
            lakehouse_cache_fill_config: self.lakehouse_cache_fill_config.clone(),
            service_stream_pool: self.service_stream_pool.clone(),
            lakehouse_cache_fill_budget: self.lakehouse_cache_fill_budget.clone(),
            lakehouse_cache_fill_concurrency: self.lakehouse_cache_fill_concurrency.clone(),
            lakehouse_cache_fill_events: self.lakehouse_cache_fill_events.clone(),
            object_store_control_runtime: Arc::clone(&self.object_store_control_runtime),
            source_object_stores: Arc::clone(&self.source_object_stores),
            deployment_io_capability: self.deployment_io_capability.clone(),
            cancellation: self.cancellation.clone(),
            native_pipeline_execution_mode: self.native_pipeline_execution_mode,
            stddev_samp_execution_mode: self.stddev_samp_execution_mode,
            aggregate_strategy_execution_options: self.aggregate_strategy_execution_options,
            source_chunk_planning_options: self.source_chunk_planning_options,
            graph_memory_calibration_override: self.graph_memory_calibration_override,
            adaptive_memory_history: self.adaptive_memory_history.as_ref().map(Arc::clone),
            hash_join_build_cache: self.hash_join_build_cache.as_ref().map(Arc::clone),
            local_parquet_source_cache: self.local_parquet_source_cache.as_ref().map(Arc::clone),
            #[cfg(feature = "cugraph")]
            resident_graph_cache: self.resident_graph_cache.as_ref().map(Arc::clone),
            // Source-read worker runtimes cannot consume a root-runtime graph
            // reservation. Keep it only on the admitted root context.
            #[cfg(feature = "cugraph")]
            reserved_resident_graph_hit: None,
            stream_lease_guard: None,
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

    /// Returns a context for an internal fragment that must bypass bounded admission.
    ///
    /// Query-local and service-scoped state is preserved; only the engine-level
    /// memory-control policy is relaxed. This is intentionally narrower than
    /// rebuilding from the shared cudf runtime, which would drop cancellation,
    /// source IO profiles, cache handles, and event sinks.
    pub fn for_unbounded_fragment(&self) -> Self {
        let mut context = self.clone();
        context.memory_control_policy = MemoryControlPolicy::Unbounded;
        context
    }

    /// Returns whether this context is backed by a QueryService admission slot.
    pub fn is_query_service_admitted(&self) -> bool {
        self.query_service_admitted
    }

    pub(super) fn install_root_stream_cleanup_completion(
        &self,
        before_quiescence: impl FnOnce() + Send + 'static,
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
        let resource_epoch = self.acquire_resource_epoch("nexus_query_engine.runtime.context.memory_grant")?;
        Ok(Some(match resource_epoch {
            Some(resource_epoch) => grant.with_resource_epoch(resource_epoch),
            None => grant.clone(),
        }))
    }

    /// Creates a hard execution-budget token for this context's runtime stream.
    ///
    /// Contexts carrying a memory grant attach the token as a member of the
    /// grant-backed arena. Other contexts use the runtime's standalone stream
    /// reservation path.
    pub fn execution_budget_token(&self, request: cudf::ExecutionBudgetRequest) -> Result<cudf::ExecutionBudgetToken> {
        let runtime = self.runtime()?;
        match (self.grant_memory_domain.as_ref(), self.grant_reservation_arena.as_ref()) {
            (Some(domain), None) | (Some(domain), Some(_)) => {
                cudf::ExecutionBudgetToken::from_runtime_in_query_domain(&runtime, request, domain.as_ref()).map_err(
                    |error| {
                        QueryError::from_cudf(
                            "nexus_query_engine.runtime.context.execution_budget_token",
                            "failed to create query-domain execution budget token",
                            error,
                        )
                    },
                )
            },
            (None, Some(arena)) => cudf::ExecutionBudgetToken::from_runtime_in_arena(&runtime, request, arena.as_ref())
                .map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.runtime.context.execution_budget_token",
                        "failed to create arena execution budget token",
                        error,
                    )
                }),
            _ => runtime.execution_budget_token(request).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.runtime.context.execution_budget_token",
                    "failed to create execution budget token",
                    error,
                )
            }),
        }
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
    pub(crate) fn lakehouse_cache_fill_events(&self) -> LakehouseCacheFillEventSink {
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

    pub(crate) fn acquire_lakehouse_cache_fill_operation_with_timeout(
        &self,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillOperationPermit>> {
        self.lakehouse_cache_fill_concurrency.acquire_with_timeout(&self.cancellation, timeout, operation)
    }

    pub(crate) fn acquire_lakehouse_cache_fill_budget_with_timeout(
        &self,
        bytes: u64,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillBytePermit>> {
        self.lakehouse_cache_fill_budget.acquire_with_timeout(&self.cancellation, bytes, timeout, operation)
    }

    pub(crate) fn s3_object_store(&self, key: S3ObjectStoreKey) -> Result<Arc<dyn ::object_store::ObjectStore>> {
        self.source_object_stores.s3_store(key)
    }

    pub(crate) fn source_storage_profile(&self, id: &str) -> Result<&SourceStorageProfile> {
        self.source_object_stores.profile(id)
    }

    /// Resolves default-chain AWS credentials from the object-store provider.
    ///
    /// Remote KvikIO reads use this to pass resolved identity into lower layers
    /// that lack a default chain. Recent results are reused briefly; on a cache
    /// miss, resolution runs on the object-store control runtime while this call
    /// blocks for the result.
    pub(crate) fn resolve_s3_default_chain_credential(
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

    pub(crate) fn spawn_cache_maintenance_fetch(
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
    pub(crate) fn spawn_object_store_control_task<F, Fut, T>(
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
    pub(crate) fn spawn_lakehouse_cache_maintenance<F, Fut, T>(
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

    #[cfg(test)]
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
                "nexus_query_engine.lakehouse_cache.maintenance.receive",
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
            let path_fact = crate::source::redacted_object_store_path_diagnostic(location.as_ref());
            let bytes = async move {
                let result = store.get(&location).await?;
                result.bytes().await
            }
            .await
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.lakehouse_cache.maintenance.object_store",
                    "failed to fetch lakehouse cache object through object_store",
                )
                .with_fact("path", path_fact)
            });
            on_result(ObjectStoreFetchResult { bytes });
        });
    }

    /// Returns the selected bounded-pipeline execution mode.
    pub fn native_pipeline_execution_mode(&self) -> NativePipelineExecutionMode {
        self.native_pipeline_execution_mode
    }

    /// Returns the selected execution contract for floating statistical aggregates.
    pub fn statistical_aggregate_execution_mode(&self) -> StddevSampExecutionMode {
        self.stddev_samp_execution_mode
    }

    /// Returns the selected execution contract through the legacy STDDEV_SAMP
    /// accessor.
    pub fn stddev_samp_execution_mode(&self) -> StddevSampExecutionMode {
        self.statistical_aggregate_execution_mode()
    }

    /// Returns grouped aggregate strategy execution policy.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.aggregate_strategy_execution_options
    }

    /// Returns expert local-throughput source chunk planning policy.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.source_chunk_planning_options
    }

    /// Returns fixture/config-supplied graph calibration metadata.
    pub fn graph_memory_calibration_override(&self) -> GraphMemoryCalibrationOverride {
        self.graph_memory_calibration_override
    }

    /// Returns a cloned context with different graph calibration metadata.
    pub fn with_graph_memory_calibration_override(&self, override_: GraphMemoryCalibrationOverride) -> Self {
        let mut context = self.clone();
        context.graph_memory_calibration_override = override_;
        context
    }

    /// Returns the adaptive-only memory history store, when configured.
    pub fn adaptive_memory_history(&self) -> Option<Arc<AdaptiveMemoryHistory>> {
        self.adaptive_memory_history.as_ref().map(Arc::clone)
    }

    /// Returns the durable GPU hash-join build cache, when configured.
    ///
    /// A clone of the `Arc` is returned so callers can retain the cache without
    /// borrowing the context.
    pub fn hash_join_build_cache(&self) -> Option<Arc<cache::hash_join::GpuHashJoinBuildCache>> {
        self.hash_join_build_cache.as_ref().map(Arc::clone)
    }

    /// Returns the durable GPU local-Parquet source cache, when configured.
    pub fn local_parquet_source_cache(&self) -> Option<Arc<cache::local_parquet::GpuLocalParquetSourceCache>> {
        self.local_parquet_source_cache.as_ref().map(Arc::clone)
    }

    /// Returns the durable GPU resident graph cache, when configured.
    #[cfg(feature = "cugraph")]
    pub fn resident_graph_cache(&self) -> Option<Arc<cache::resident_graph::GpuResidentGraphCache>> {
        self.resident_graph_cache.as_ref().map(Arc::clone)
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn has_reserved_resident_graph_price(&self) -> bool {
        self.reserved_resident_graph_hit.is_some()
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn reject_unmatched_reserved_resident_graph_price(&self, reason: &'static str) -> Result<()> {
        if self.has_reserved_resident_graph_price() {
            return Err(warm_graph_reservation_error(reason));
        }
        Ok(())
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn take_reserved_resident_graph_pin(
        &self,
        resident_key: &crate::plan::ResidentGraphKey,
    ) -> Result<Option<cache::resident_graph::GpuResidentGraphPreparationPin>> {
        let Some(reservation) = &self.reserved_resident_graph_hit else {
            return Ok(None);
        };
        if &reservation.resident_key != resident_key {
            return Err(warm_graph_reservation_error("resident_graph_key_mismatch")
                .with_fact("reserved_resident_graph_key", format!("{:?}", reservation.resident_key))
                .with_fact("requested_resident_graph_key", format!("{resident_key:?}")));
        }
        let mut pin = reservation
            .preparation_pin
            .lock()
            .map_err(|_error| warm_graph_reservation_error("preparation_pin_lock_poisoned"))?;
        pin.take().map(Some).ok_or_else(|| warm_graph_reservation_error("preparation_pin_already_consumed"))
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
        "nexus_query_engine.runtime_observation.identity",
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
        "nexus_query_engine.runtime_observation.identity",
        "execution context trace id conflicts with its runtime observation session",
    )
    .with_fact("reason", "runtime_observation_trace_id_mismatch")
    .with_fact("context_trace_id", trace_id)
    .with_fact("observation_trace_id", observation_trace_id.unwrap_or("none")))
}

#[cfg(feature = "cugraph")]
pub(super) fn warm_graph_reservation_error(reason: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "nexus_query_engine.query_service.warm_graph_reservation",
        "warm-priced resident graph reservation is no longer available",
    )
    .with_fact("reason", reason)
    .with_fact("retry_action", "readmit_query")
}
