//! Native execution context state and query-local runtime controls.

#[cfg(feature = "cugraph")]
use std::sync::Mutex;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use ::object_store::ObjectStoreExt;
#[cfg(test)]
use bytes::Bytes;

use super::{
    DeploymentIoCapability, DevicePlacement, ExecutionBoundaryPolicy, GpuExecutionPool, HostFetchBudget,
    HostFetchPermit, MemoryControlPolicy, MemoryGrant, MemoryResourcePolicy, MemorySpaceId, ServiceStreamPool,
    SourceChunkPlanningMode, SourceFetchConfig, SourceFetchStrategyDecision, SourceFetchSubstrate,
    SourceReadStreamLease, SourceStorageProfile, StreamLease, cache,
    cache_events::{LakehouseCacheFillEventSink, LakehouseCacheFillEventSnapshot},
    default_service_stream_pool_size_for_device,
    object_store::{
        ObjectStoreCustomFetchResult, ObjectStoreFetchResult, S3ObjectStoreKey, SourceObjectStoreRegistry,
        default_lakehouse_cache_fill_runtime, default_source_fetch_runtime,
    },
    source_fetch::{GdsConcurrencyGate, SourceFetchConcurrencyGate, SourceFetchOperationPermit},
};
use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    graph_memory::GraphMemoryCalibrationOverride,
    metrics::{QueryId, TraceId},
    mode::{NativePipelineExecutionMode, StddevSampExecutionMode},
};

mod builder;

pub use builder::ExecutionContextBuilder;
pub(super) use builder::validate_service_stream_pool_device;

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

    pub(super) fn validate(self) -> Result<()> {
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
    /// Optional caller-provided query-wide diagnostic correlation id.
    query_id: Option<QueryId>,
    /// Optional parent trace id used to group related query executions.
    trace_id: Option<TraceId>,
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
    /// Service-level source IO substrate and gate configuration.
    source_fetch_config: SourceFetchConfig,
    /// Service-level pool used by bounded source-read workers.
    service_stream_pool: ServiceStreamPool,
    /// Service-wide host-memory budget for object-store fetch buffers.
    host_fetch_budget: HostFetchBudget,
    /// Service-wide operation-count gate for object-store fetches.
    source_fetch_concurrency: SourceFetchConcurrencyGate,
    /// Service-wide byte budget for background lakehouse complete-object cache fills.
    lakehouse_cache_fill_budget: HostFetchBudget,
    /// Service-wide operation-count gate for background lakehouse complete-object cache fills.
    lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate,
    /// Structured completion/failure events from asynchronous lakehouse cache fills.
    lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    /// Service-wide operation-count gate for future GDS direct-device reads.
    gds_concurrency: GdsConcurrencyGate,
    /// Service-scoped async runtime used for object-store source fetches.
    source_fetch_runtime: Arc<tokio::runtime::Runtime>,
    /// Service-scoped async runtime used for background lakehouse cache fills.
    lakehouse_cache_fill_runtime: Arc<tokio::runtime::Runtime>,
    /// Cached object-store clients used by source fetch workers.
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

impl LeasedExecutionContext {
    fn new(context: ExecutionContext) -> Self {
        Self { context }
    }

    /// Returns the context backed by the owned stream lease.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

#[derive(Debug)]
pub(super) struct ExecutionContextParts {
    pub(super) query_id: Option<QueryId>,
    pub(super) trace_id: Option<TraceId>,
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
    pub(super) source_fetch_config: SourceFetchConfig,
    pub(super) service_stream_pool: ServiceStreamPool,
    pub(super) host_fetch_budget: HostFetchBudget,
    pub(super) source_fetch_concurrency: SourceFetchConcurrencyGate,
    pub(super) lakehouse_cache_fill_budget: HostFetchBudget,
    pub(super) lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate,
    pub(super) lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    pub(super) gds_concurrency: GdsConcurrencyGate,
    pub(super) source_fetch_runtime: Arc<tokio::runtime::Runtime>,
    pub(super) lakehouse_cache_fill_runtime: Arc<tokio::runtime::Runtime>,
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
    pub fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    /// Builds a context around an owned cudf runtime.
    ///
    /// The resulting context records [`MemoryResourcePolicy::ExternalRuntime`]
    /// because the runtime, including its allocator and stream, has already been
    /// constructed by the caller.
    pub fn from_runtime(runtime: cudf::Runtime) -> Result<Self> {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    /// Builds a context around a shared cudf runtime.
    ///
    /// Use this when multiple plans or adapter layers must share the same CUDA
    /// stream, device placement, allocator policy, or boundary observer state.
    pub fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Result<Self> {
        let service_stream_pool =
            ServiceStreamPool::new(runtime.device(), default_service_stream_pool_size_for_device(runtime.device()))
                .expect("default service stream pool size is non-zero");
        let source_fetch_config = SourceFetchConfig::default();
        let gpu_execution_pool =
            GpuExecutionPool::shared_for_device(runtime.device().ordinal(), service_stream_pool.pool_size())?;
        Ok(Self::from_parts(ExecutionContextParts {
            query_id: None,
            trace_id: None,
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
            source_fetch_config,
            service_stream_pool,
            host_fetch_budget: HostFetchBudget::default(),
            source_fetch_concurrency: SourceFetchConcurrencyGate::default(),
            lakehouse_cache_fill_budget: HostFetchBudget::default(),
            lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate::new(
                SourceFetchConfig::DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS,
            )
            .expect("default lakehouse cache fill concurrency is non-zero"),
            lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            gds_concurrency: GdsConcurrencyGate::default(),
            source_fetch_runtime: default_source_fetch_runtime()?,
            lakehouse_cache_fill_runtime: default_lakehouse_cache_fill_runtime()?,
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
            query_id: parts.query_id,
            trace_id: parts.trace_id,
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
            source_fetch_config: parts.source_fetch_config,
            service_stream_pool: parts.service_stream_pool,
            host_fetch_budget: parts.host_fetch_budget,
            source_fetch_concurrency: parts.source_fetch_concurrency,
            lakehouse_cache_fill_budget: parts.lakehouse_cache_fill_budget,
            lakehouse_cache_fill_concurrency: parts.lakehouse_cache_fill_concurrency,
            lakehouse_cache_fill_events: parts.lakehouse_cache_fill_events,
            gds_concurrency: parts.gds_concurrency,
            source_fetch_runtime: parts.source_fetch_runtime,
            lakehouse_cache_fill_runtime: parts.lakehouse_cache_fill_runtime,
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

    /// Returns the cudf runtime used for all GPU operations.
    pub fn runtime(&self) -> &cudf::Runtime {
        &self.runtime
    }

    /// Returns the CUDA stream owned by the runtime.
    pub fn stream(&self) -> &cudf::runtime::CudaStream {
        self.runtime.stream()
    }

    /// Returns the device memory resource owned by the runtime.
    pub fn memory_resource(&self) -> &cudf::core::memory::DeviceMemoryResource {
        self.runtime.memory_resource()
    }

    /// Returns a shared handle to the runtime.
    ///
    /// The caller receives the same runtime object, not a new runtime with
    /// copied configuration.
    pub fn shared_runtime(&self) -> Arc<cudf::Runtime> {
        Arc::clone(&self.runtime)
    }

    /// Returns the dedicated executor for synchronous GPU work on this context's device.
    pub fn gpu_execution_pool(&self) -> GpuExecutionPool {
        self.gpu_execution_pool.clone()
    }

    /// Returns the query-wide diagnostic correlation id.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Returns a cloned context with a different diagnostic query id.
    pub fn with_query_id(&self, query_id: impl Into<QueryId>) -> Self {
        let mut context = self.clone();
        context.query_id = Some(query_id.into());
        context
    }

    /// Returns a cloned context with a different diagnostic trace id.
    pub fn with_trace_id(&self, trace_id: impl Into<TraceId>) -> Self {
        let mut context = self.clone();
        context.trace_id = Some(trace_id.into());
        context
    }

    /// Acquires a same-device leased context for a scheduled source-read worker.
    pub(crate) fn acquire_source_read_lease(&self) -> Result<SourceReadStreamLease> {
        let stream_lease = self
            .service_stream_pool
            .acquire_with_cancellation(&self.cancellation, "nexus_query_engine.runtime.source_read_stream.acquire")?;
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
        validate_service_stream_pool_device(stream_lease.pool(), self.runtime.device())?;
        Ok(Self::from_parts(ExecutionContextParts {
            query_id: self.query_id.clone(),
            trace_id: self.trace_id.clone(),
            runtime: stream_lease.shared_runtime(),
            gpu_execution_pool: self.gpu_execution_pool.clone(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy,
            memory_grant: self.memory_grant.clone(),
            grant_reservation_arena: self.grant_reservation_arena.clone(),
            grant_memory_domain: self.grant_memory_domain.clone(),
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_chunk_planning_mode: self.source_chunk_planning_mode,
            source_fetch_config: self.source_fetch_config.clone(),
            service_stream_pool: self.service_stream_pool.clone(),
            host_fetch_budget: self.host_fetch_budget.clone(),
            source_fetch_concurrency: self.source_fetch_concurrency.clone(),
            lakehouse_cache_fill_budget: self.lakehouse_cache_fill_budget.clone(),
            lakehouse_cache_fill_concurrency: self.lakehouse_cache_fill_concurrency.clone(),
            lakehouse_cache_fill_events: self.lakehouse_cache_fill_events.clone(),
            gds_concurrency: self.gds_concurrency.clone(),
            source_fetch_runtime: Arc::clone(&self.source_fetch_runtime),
            lakehouse_cache_fill_runtime: Arc::clone(&self.lakehouse_cache_fill_runtime),
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

    /// Returns the cooperative cancellation flag for this execution.
    pub fn cancellation(&self) -> &ExecutionCancellation {
        &self.cancellation
    }

    /// Fails with a structured cancellation error if the execution was cancelled.
    pub fn check_cancellation(&self, operation: &'static str) -> Result<()> {
        self.cancellation.check(operation)
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
    pub fn memory_grant(&self) -> Option<&MemoryGrant> {
        self.memory_grant.as_ref()
    }

    /// Creates a hard execution-budget token for this context's runtime stream.
    ///
    /// Contexts carrying a memory grant attach the token as a member of the
    /// grant-backed arena. Other contexts use the runtime's standalone stream
    /// reservation path.
    pub fn execution_budget_token(
        &self,
        request: cudf::ExecutionBudgetRequest,
    ) -> cudf::Result<cudf::ExecutionBudgetToken> {
        match (self.grant_memory_domain.as_ref(), self.grant_reservation_arena.as_ref()) {
            (Some(domain), None) | (Some(domain), Some(_)) => {
                cudf::ExecutionBudgetToken::from_runtime_in_query_domain(self.runtime(), request, domain.as_ref())
            },
            (None, Some(arena)) => {
                cudf::ExecutionBudgetToken::from_runtime_in_arena(self.runtime(), request, arena.as_ref())
            },
            _ => self.runtime().execution_budget_token(request),
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

    /// Returns the service-level source IO configuration.
    pub fn source_fetch_config(&self) -> &SourceFetchConfig {
        &self.source_fetch_config
    }

    /// Returns the effective source IO substrate preference.
    pub fn source_fetch_substrate(&self) -> SourceFetchSubstrate {
        self.source_fetch_config.substrate
    }

    /// Returns the service-level CUDA stream pool used by source-read workers.
    pub fn service_stream_pool(&self) -> ServiceStreamPool {
        self.service_stream_pool.clone()
    }

    /// Returns the service-wide host-memory budget for object-store source fetches.
    pub fn host_fetch_budget(&self) -> HostFetchBudget {
        self.host_fetch_budget.clone()
    }

    /// Maximum object-store SourceFetch operations admitted concurrently.
    pub fn host_fetch_max_concurrent_ops(&self) -> usize {
        self.source_fetch_concurrency.max_concurrent_ops()
    }

    /// Timeout used by blocking object-store SourceFetch resource acquisition.
    pub fn host_fetch_acquire_timeout(&self) -> Duration {
        self.source_fetch_config.host_fetch_acquire_timeout
    }

    /// Returns the service-wide background lakehouse cache-fill byte budget.
    pub fn lakehouse_cache_fill_budget(&self) -> HostFetchBudget {
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

    /// Maximum future GDS DMA operations admitted concurrently.
    pub fn gds_max_concurrent_ops(&self) -> usize {
        self.gds_concurrency.max_concurrent_ops()
    }

    /// Per-chunk threshold below which future GDS-eligible reads use host-buffer IO.
    pub fn gds_min_chunk_bytes(&self) -> u64 {
        self.source_fetch_config.gds_min_chunk_bytes
    }

    /// Selects the source IO strategy for a location and optional chunk-size estimate.
    pub fn source_fetch_strategy_for_location(
        &self,
        source_location: &str,
        chunk_bytes: Option<u64>,
    ) -> SourceFetchStrategyDecision {
        self.source_fetch_config.decide_source_fetch_strategy(
            &self.deployment_io_capability,
            source_location,
            chunk_bytes,
        )
    }

    pub(crate) fn acquire_source_fetch_operation(&self) -> Result<SourceFetchOperationPermit> {
        self.source_fetch_concurrency.acquire()
    }

    pub(crate) fn try_acquire_source_fetch_operation(&self) -> Result<Option<SourceFetchOperationPermit>> {
        self.source_fetch_concurrency.try_acquire()
    }

    pub(crate) fn acquire_source_fetch_operation_with_timeout(
        &self,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<SourceFetchOperationPermit>> {
        self.source_fetch_concurrency.acquire_with_timeout(&self.cancellation, timeout, operation)
    }

    pub(crate) fn acquire_lakehouse_cache_fill_operation_with_timeout(
        &self,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<SourceFetchOperationPermit>> {
        self.lakehouse_cache_fill_concurrency.acquire_with_timeout(&self.cancellation, timeout, operation)
    }

    pub(crate) fn acquire_lakehouse_cache_fill_budget_with_timeout(
        &self,
        bytes: u64,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<HostFetchPermit>> {
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
    /// Remote-direct reads use this to pass the host-buffer identity into lower
    /// layers that lack a default chain. Recent results are reused briefly; on a
    /// cache miss, resolution runs on the source-fetch runtime while this call
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
        self.source_fetch_runtime.spawn(async move {
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

    pub(crate) fn spawn_object_store_fetch(
        &self,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        Self::spawn_object_store_fetch_on_runtime(self.source_fetch_runtime.as_ref(), store, location, on_result)
    }

    /// Spawns a caller-provided async fetch task on the source-fetch runtime.
    ///
    /// The caller composes any combination of `get_range` / `get_ranges` calls
    /// inside the future. Elapsed time covers the entire future, including any
    /// rewrite work the caller performs inside it.
    pub(crate) fn spawn_object_store_custom_fetch<F, Fut, T>(
        &self,
        future_factory: F,
        on_result: impl FnOnce(ObjectStoreCustomFetchResult<T>) + Send + 'static,
    ) -> tokio::task::JoinHandle<()>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.source_fetch_runtime.spawn(async move {
            let timer = Instant::now();
            let value = future_factory().await;
            on_result(ObjectStoreCustomFetchResult { value, elapsed: timer.elapsed() });
        })
    }

    /// Spawns a caller-provided async background lakehouse cache-fill task.
    pub(crate) fn spawn_lakehouse_cache_fill<F, Fut, T>(
        &self,
        future_factory: F,
        on_result: impl FnOnce(ObjectStoreCustomFetchResult<T>) + Send + 'static,
    ) -> tokio::task::JoinHandle<()>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.lakehouse_cache_fill_runtime.spawn(async move {
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
        source_fetch_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
    ) -> Result<Bytes> {
        let (sender, receiver) = std::sync::mpsc::channel();
        Self::spawn_object_store_fetch_on_runtime(source_fetch_runtime, store, location, move |result| {
            let _ = sender.send(result.bytes);
        });
        receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.source_fetch.receive",
                "object-store source fetch worker stopped before returning a result",
            )
            .with_source(error)
        })?
    }

    fn spawn_object_store_fetch_on_runtime(
        source_fetch_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        source_fetch_runtime.spawn(async move {
            let timer = Instant::now();
            let path_fact = crate::source::redacted_object_store_path_diagnostic(location.as_ref());
            let bytes = async move {
                let result = store.get(&location).await?;
                result.bytes().await
            }
            .await
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.source_fetch.object_store",
                    "failed to fetch native Parquet source bytes through object_store",
                )
                .with_fact("path", path_fact)
            });
            on_result(ObjectStoreFetchResult { bytes, elapsed: timer.elapsed() });
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
