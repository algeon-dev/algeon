//! Runtime, memory, boundary, and reservation contracts for native execution.
//!
//! This module is intentionally close to the public API because these values
//! control more than local execution choices. Policy enum string values,
//! reservation fact keys, and retry fact keys flow into tests, reports, and
//! adapter diagnostics. Treat those strings as stable diagnostic contracts:
//! change them only with the same care as a report schema change.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use ::object_store::ObjectStoreExt;
#[cfg(test)]
use bytes::Bytes;

use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    graph_memory::GraphMemoryCalibrationOverride,
    metrics::{QueryId, TraceId},
    mode::{NativePipelineExecutionMode, StddevSampExecutionMode},
};

mod boundary;
mod bounded_join;
pub mod gpu;
mod io_capability;
mod object_store;
mod policy;
mod query_service;
mod reservation;
mod service_stream;
mod source_fetch;

pub use boundary::{
    BuildMaterializationBoundary, ExecutionBoundaryPolicy, ProbeExecutionBoundary, ResultMaterializationBoundary,
    SourceReadBoundary,
};
pub use bounded_join::{
    BoundedJoinReservationContract, BoundedJoinReservationRequirement, BoundedJoinReservationStatus,
};
pub use gpu::{
    BestFitPlacement, CudfInventory, DeviceContext, DeviceFacts, DeviceMemoryFailureReport, DeviceRegistry,
    DeviceRegistryConfig, EvictableMemoryObserver, EvictableMemoryPressure, GpuComms, GpuInventory, GrantRequest,
    LeastLoadedPlacement, MemoryGovernor, MemoryGovernorGrantRow, MemoryGovernorReport, MemoryGrant,
    PinnedDevicePlacement, Placement, PlacementPolicy, PlacementRequest, Priority, QueryPlacementHint,
    QueryPlacementPolicy, QueryQos, SingleDeviceComms, device_budget_bytes_from_facts,
};
pub use io_capability::{
    DEFAULT_GDS_LOCAL_MOUNT_ROOT, DeploymentIoCapability, GdsAvailabilityStatus, GdsDriverProbeStatus,
    KVIKIO_COMPAT_MODE_ENV, KvikioCompatModeStatus, NVIDIA_FS_PROC_PATH, NVIDIA_FS_SYSFS_PATH, NvidiaFsProbeStatus,
    SourceGdsEligibility, source_gds_eligibility, source_gds_eligibility_with_default_mounts,
};
pub(crate) use object_store::{ObjectStoreCustomFetchResult, ObjectStoreFetchResult, S3ObjectStoreKey};
pub use object_store::{S3SourceStorageCredential, S3SourceStorageProfile, SourceStorageProfile};
use object_store::{SourceObjectStoreRegistry, default_lakehouse_cache_fill_runtime, default_source_fetch_runtime};
pub use policy::{
    DevicePlacement, MemoryControlPolicy, MemoryResourcePolicy, MemorySpaceId, ReservationKind, SourceChunkPlanningMode,
};
pub use query_service::{
    QueryAdmissionRequest, QueryAdmissionWaiter, QueryGrantBand, QueryGrantRequest, QueryHandle,
    QueryMemoryDomainPolicy, QueryService, QueryServiceAdmissionSnapshot, QueryServiceConfig, QueryServiceTenantConfig,
    SlotGuarantee, TenantId, TenantQueryContext, TenantQueueDepth,
};
pub use reservation::{MemoryReservationRequest, RetryFacts};
pub(crate) use service_stream::SourceReadStreamLease;
use service_stream::validate_service_stream_pool_size;
pub use service_stream::{
    DEFAULT_SERVICE_STREAM_POOL_SIZE, ServiceStreamPool, StreamLease, default_service_stream_pool_size_for_device,
};
pub use source_fetch::{
    DEFAULT_GDS_MAX_CONCURRENT_OPS, DEFAULT_GDS_MIN_CHUNK_BYTES, DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT,
    DEFAULT_HOST_FETCH_BUDGET_BYTES, DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES, DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS,
    DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS, HostFetchBudget, HostFetchPermit, SourceFetchConfig,
    SourceFetchStrategy, SourceFetchStrategyDecision, SourceFetchStrategyReason, SourceFetchSubstrate,
    default_host_fetch_budget_bytes,
};
use source_fetch::{GdsConcurrencyGate, SourceFetchConcurrencyGate, SourceFetchOperationPermit};

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

/// Structured outcome for an asynchronous lakehouse cache fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LakehouseCacheFillOutcome {
    /// The complete object was written to the local lakehouse cache.
    Completed,
    /// The background fill failed after it was accepted for execution.
    Failed,
}

/// Error facts captured for a failed asynchronous lakehouse cache fill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LakehouseCacheFillErrorEvent {
    pub(crate) code: String,
    pub(crate) kind: String,
    pub(crate) status: String,
    pub(crate) operation: String,
}

/// Structured asynchronous lakehouse cache fill event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LakehouseCacheFillEvent {
    pub(crate) outcome: LakehouseCacheFillOutcome,
    pub(crate) bytes: Option<u64>,
    pub(crate) elapsed: Duration,
    pub(crate) cache_path_hash: String,
    pub(crate) stable_id_hash: String,
    pub(crate) file_index: Option<usize>,
    pub(crate) chunk_index: Option<usize>,
    pub(crate) error: Option<LakehouseCacheFillErrorEvent>,
}

/// Credential-free snapshot of a background lakehouse cache-fill completion event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LakehouseCacheFillEventSnapshot {
    /// Stable event outcome.
    pub outcome: &'static str,
    /// Complete-object bytes written for successful fills, when known.
    pub bytes: Option<u64>,
    /// Elapsed time from accepted background job start to completion/failure.
    pub elapsed: Duration,
    /// Stable hash of the local cache path.
    pub cache_path_hash: String,
    /// Stable hash of the source stable id.
    pub stable_id_hash: String,
    /// Source file index associated with the fill, when known.
    pub file_index: Option<usize>,
    /// Source chunk index associated with the fill, when known.
    pub chunk_index: Option<usize>,
    /// Stable native error code for failed fills, when present.
    pub error_code: Option<String>,
    /// Stable native error kind for failed fills, when present.
    pub error_kind: Option<String>,
    /// Stable native error status for failed fills, when present.
    pub error_status: Option<String>,
    /// Stable native error operation for failed fills, when present.
    pub error_operation: Option<String>,
}

impl LakehouseCacheFillOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

impl From<&LakehouseCacheFillEvent> for LakehouseCacheFillEventSnapshot {
    fn from(event: &LakehouseCacheFillEvent) -> Self {
        Self {
            outcome: event.outcome.as_str(),
            bytes: event.bytes,
            elapsed: event.elapsed,
            cache_path_hash: event.cache_path_hash.clone(),
            stable_id_hash: event.stable_id_hash.clone(),
            file_index: event.file_index,
            chunk_index: event.chunk_index,
            error_code: event.error.as_ref().map(|error| error.code.clone()),
            error_kind: event.error.as_ref().map(|error| error.kind.clone()),
            error_status: event.error.as_ref().map(|error| error.status.clone()),
            error_operation: event.error.as_ref().map(|error| error.operation.clone()),
        }
    }
}

/// Query-context-local event sink for asynchronous lakehouse cache fill completion.
#[derive(Debug, Clone, Default)]
pub(crate) struct LakehouseCacheFillEventSink {
    events: Arc<Mutex<Vec<LakehouseCacheFillEvent>>>,
}

impl LakehouseCacheFillEventSink {
    pub(crate) fn record(&self, event: LakehouseCacheFillEvent) {
        match self.events.lock() {
            Ok(mut events) => events.push(event),
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to record poisoned lakehouse cache fill event sink");
            },
        }
    }

    pub(crate) fn events(&self) -> Vec<LakehouseCacheFillEvent> {
        match self.events.lock() {
            Ok(events) => events.clone(),
            Err(_error) => Vec::new(),
        }
    }

    pub(crate) fn drain(&self) -> Vec<LakehouseCacheFillEvent> {
        match self.events.lock() {
            Ok(mut events) => events.drain(..).collect(),
            Err(_error) => Vec::new(),
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

    fn validate(self) -> Result<()> {
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
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
    /// Optional durable GPU local-Parquet source cache.
    ///
    /// Bounded execution charges retained cache bytes to the reservation ledger
    /// before admitting source or operator work.
    local_parquet_source_cache: Option<Arc<crate::exec::GpuLocalParquetSourceCache>>,
    /// Optional durable GPU resident graph cache.
    #[cfg(feature = "cugraph")]
    resident_graph_cache: Option<Arc<crate::exec::GpuResidentGraphCache>>,
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
struct ExecutionContextParts {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    runtime: Arc<cudf::Runtime>,
    memory_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
    memory_grant: Option<MemoryGrant>,
    grant_reservation_arena: Option<Arc<cudf::ReservationArena>>,
    grant_memory_domain: Option<Arc<cudf::QueryMemoryDomain>>,
    boundary_policy: ExecutionBoundaryPolicy,
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    source_fetch_config: SourceFetchConfig,
    service_stream_pool: ServiceStreamPool,
    host_fetch_budget: HostFetchBudget,
    source_fetch_concurrency: SourceFetchConcurrencyGate,
    lakehouse_cache_fill_budget: HostFetchBudget,
    lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate,
    lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    gds_concurrency: GdsConcurrencyGate,
    source_fetch_runtime: Arc<tokio::runtime::Runtime>,
    lakehouse_cache_fill_runtime: Arc<tokio::runtime::Runtime>,
    source_object_stores: Arc<SourceObjectStoreRegistry>,
    deployment_io_capability: DeploymentIoCapability,
    cancellation: ExecutionCancellation,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    stddev_samp_execution_mode: StddevSampExecutionMode,
    aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    source_chunk_planning_options: SourceChunkPlanningOptions,
    graph_memory_calibration_override: GraphMemoryCalibrationOverride,
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
    local_parquet_source_cache: Option<Arc<crate::exec::GpuLocalParquetSourceCache>>,
    #[cfg(feature = "cugraph")]
    resident_graph_cache: Option<Arc<crate::exec::GpuResidentGraphCache>>,
    stream_lease_guard: Option<Arc<StreamLease>>,
    query_service_admitted: bool,
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
        Ok(Self::from_parts(ExecutionContextParts {
            query_id: None,
            trace_id: None,
            runtime,
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
            stream_lease_guard: None,
            query_service_admitted: false,
        }))
    }

    fn from_parts(parts: ExecutionContextParts) -> Self {
        debug_assert!(
            parts.grant_reservation_arena.is_none() || parts.grant_memory_domain.is_none(),
            "execution context cannot carry both grant reservation arena and query memory domain"
        );
        Self {
            query_id: parts.query_id,
            trace_id: parts.trace_id,
            runtime: parts.runtime,
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
    fn fetch_object_store_bytes_on_runtime(
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
    pub fn hash_join_build_cache(&self) -> Option<Arc<crate::exec::GpuHashJoinBuildCache>> {
        self.hash_join_build_cache.as_ref().map(Arc::clone)
    }

    /// Returns the durable GPU local-Parquet source cache, when configured.
    pub fn local_parquet_source_cache(&self) -> Option<Arc<crate::exec::GpuLocalParquetSourceCache>> {
        self.local_parquet_source_cache.as_ref().map(Arc::clone)
    }

    /// Returns the durable GPU resident graph cache, when configured.
    #[cfg(feature = "cugraph")]
    pub fn resident_graph_cache(&self) -> Option<Arc<crate::exec::GpuResidentGraphCache>> {
        self.resident_graph_cache.as_ref().map(Arc::clone)
    }
}

/// Builder for [`ExecutionContext`].
///
/// The builder separates runtime construction policy from execution policy. If
/// a runtime is supplied, it is used as-is and `device_ordinal` /
/// `memory_policy` no longer construct CUDA state. If no runtime is supplied,
/// the builder creates a cudf runtime from the requested device ordinal and
/// memory-resource policy.
#[derive(Debug, Clone, Default)]
pub struct ExecutionContextBuilder {
    /// Optional caller-provided diagnostic query id.
    query_id: Option<QueryId>,
    /// Optional caller-provided parent trace id.
    trace_id: Option<TraceId>,
    /// Prebuilt runtime to use instead of constructing one.
    runtime: Option<Arc<cudf::Runtime>>,
    /// CUDA device ordinal used only when this builder constructs the runtime.
    device_ordinal: Option<usize>,
    /// Memory resource used only when this builder constructs the runtime.
    memory_policy: MemoryResourcePolicy,
    /// Engine-level bounded/unbounded memory contract.
    memory_control_policy: MemoryControlPolicy,
    /// Optional soft governor grant backing bounded execution.
    memory_grant: Option<MemoryGrant>,
    /// Source/build/probe/result boundary contract.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Source read scheduling contract.
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Optional service-level source IO configuration.
    source_fetch_config: Option<SourceFetchConfig>,
    /// Source object-store profiles available to execution-time fetches.
    source_storage_profiles: Vec<SourceStorageProfile>,
    /// Optional caller-owned service stream pool.
    service_stream_pool: Option<ServiceStreamPool>,
    /// Pool size for the default process-global service stream pool.
    service_stream_pool_size: Option<usize>,
    /// Optional caller-owned host fetch budget.
    host_fetch_budget: Option<HostFetchBudget>,
    /// Host fetch budget size used when constructing the default budget.
    host_fetch_budget_bytes: Option<u64>,
    /// Timeout used by blocking object-store SourceFetch resource acquisition.
    host_fetch_acquire_timeout: Option<Duration>,
    /// Max object-store SourceFetch operations allowed concurrently.
    host_fetch_max_concurrent_ops: Option<usize>,
    /// Background lakehouse cache-fill budget size used when constructing the default fill budget.
    lakehouse_cache_fill_budget_bytes: Option<u64>,
    /// Max background lakehouse complete-object cache-fill operations allowed concurrently.
    lakehouse_cache_fill_max_concurrent_ops: Option<usize>,
    /// Optional caller-supplied deployment IO capability diagnostic.
    deployment_io_capability: Option<DeploymentIoCapability>,
    /// Cooperative cancellation flag observed by native execution.
    cancellation: ExecutionCancellation,
    /// Native bounded-pipeline execution mode.
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    /// Execution contract for floating statistical aggregates.
    stddev_samp_execution_mode: StddevSampExecutionMode,
    /// Runtime threshold policy for grouped aggregate strategy selection.
    aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert policy for local-throughput source chunk planning.
    source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Optional graph calibration metadata supplied by fixture/config.
    graph_memory_calibration_override: GraphMemoryCalibrationOverride,
    /// Optional adaptive-only memory history store.
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    /// Optional durable hash-join build cache.
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
    /// Optional durable local-Parquet source cache.
    local_parquet_source_cache: Option<Arc<crate::exec::GpuLocalParquetSourceCache>>,
    /// Optional durable resident graph cache.
    #[cfg(feature = "cugraph")]
    resident_graph_cache: Option<Arc<crate::exec::GpuResidentGraphCache>>,
}

impl ExecutionContextBuilder {
    /// Creates a builder with default runtime and execution policies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses an already-constructed cudf runtime.
    ///
    /// This marks the memory-resource policy as
    /// [`MemoryResourcePolicy::ExternalRuntime`] because allocator/device setup
    /// has happened outside the query engine.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Uses an already-shared cudf runtime.
    ///
    /// Prefer this over building separate runtimes when an adapter needs one
    /// CUDA stream, allocator, or device policy across multiple executions.
    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Supplies the query-wide diagnostic correlation id.
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    /// Supplies the optional parent trace id.
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Selects the CUDA device ordinal for runtime construction.
    ///
    /// Ignored when a runtime is supplied with [`Self::with_runtime`] or
    /// [`Self::with_shared_runtime`].
    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    /// Selects the memory resource used when constructing a runtime.
    ///
    /// Ignored for an injected runtime except for the recorded
    /// `ExternalRuntime` diagnostic state.
    pub fn with_memory_policy(mut self, memory_policy: MemoryResourcePolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    /// Sets the engine-level memory control contract.
    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    /// Attaches a soft governor grant to bounded execution contexts built here.
    pub fn with_memory_grant(mut self, memory_grant: MemoryGrant) -> Self {
        self.memory_grant = Some(memory_grant);
        self
    }

    /// Sets the execution boundary policy.
    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    /// Sets bounded source scheduling controls.
    pub fn with_source_scheduler_policy(mut self, source_scheduler_policy: SourceSchedulerPolicy) -> Self {
        self.source_scheduler_policy = source_scheduler_policy;
        self
    }

    /// Set bounded source chunk planning mode.
    pub fn with_source_chunk_planning_mode(mut self, mode: SourceChunkPlanningMode) -> Self {
        self.source_chunk_planning_mode = mode;
        self
    }

    /// Sets the service-level source IO configuration.
    ///
    /// Existing per-field builder setters for host-buffer budget and
    /// object-store operation count override the corresponding values in this
    /// config so older call sites can keep using their narrower knobs.
    pub fn with_source_fetch_config(mut self, source_fetch_config: SourceFetchConfig) -> Self {
        self.source_fetch_config = Some(source_fetch_config);
        self
    }

    /// Registers source object-store profiles used by remote host-buffer reads
    /// and local-cache miss fills.
    pub fn with_source_storage_profiles(mut self, profiles: impl IntoIterator<Item = SourceStorageProfile>) -> Self {
        self.source_storage_profiles = profiles.into_iter().collect();
        self
    }

    /// Uses a caller-owned service stream pool for bounded source-read workers.
    pub fn with_service_stream_pool(mut self, service_stream_pool: ServiceStreamPool) -> Self {
        self.service_stream_pool = Some(service_stream_pool);
        self
    }

    /// Sets the size of the default process-global service stream pool.
    ///
    /// Ignored when a pool is supplied with [`Self::with_service_stream_pool`].
    pub fn with_service_stream_pool_size(mut self, pool_size: usize) -> Self {
        self.service_stream_pool_size = Some(pool_size);
        self
    }

    /// Uses a caller-owned host fetch budget for object-store source reads.
    pub fn with_host_fetch_budget(mut self, host_fetch_budget: HostFetchBudget) -> Self {
        self.host_fetch_budget = Some(host_fetch_budget);
        self
    }

    /// Sets the size of the default host fetch budget.
    ///
    /// Ignored when a budget is supplied with [`Self::with_host_fetch_budget`].
    pub fn with_host_fetch_budget_bytes(mut self, budget_bytes: u64) -> Self {
        self.host_fetch_budget_bytes = Some(budget_bytes);
        self
    }

    /// Sets the timeout used by blocking object-store SourceFetch resource acquisition.
    pub fn with_host_fetch_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.host_fetch_acquire_timeout = Some(timeout);
        self
    }

    /// Sets the service-wide object-store SourceFetch operation cap.
    pub fn with_host_fetch_max_concurrent_ops(mut self, max_concurrent_ops: usize) -> Self {
        self.host_fetch_max_concurrent_ops = Some(max_concurrent_ops);
        self
    }

    /// Sets the service-wide background lakehouse cache-fill byte budget.
    pub fn with_lakehouse_cache_fill_budget_bytes(mut self, budget_bytes: u64) -> Self {
        self.lakehouse_cache_fill_budget_bytes = Some(budget_bytes);
        self
    }

    /// Sets the service-wide background lakehouse cache-fill operation cap.
    pub fn with_lakehouse_cache_fill_max_concurrent_ops(mut self, max_concurrent_ops: usize) -> Self {
        self.lakehouse_cache_fill_max_concurrent_ops = Some(max_concurrent_ops);
        self
    }

    /// Supplies deployment IO capability diagnostics instead of probing the current host.
    pub fn with_deployment_io_capability(mut self, capability: DeploymentIoCapability) -> Self {
        self.deployment_io_capability = Some(capability);
        self
    }

    /// Sets the cooperative cancellation flag for native execution.
    pub fn with_cancellation(mut self, cancellation: ExecutionCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Selects the native bounded-pipeline execution mode.
    pub fn with_native_pipeline_execution_mode(mut self, mode: NativePipelineExecutionMode) -> Self {
        self.native_pipeline_execution_mode = mode;
        self
    }

    /// Selects the execution contract for floating statistical aggregates.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StddevSampExecutionMode) -> Self {
        self.stddev_samp_execution_mode = mode;
        self
    }

    /// Selects the execution contract through the legacy STDDEV_SAMP builder.
    pub fn with_stddev_samp_execution_mode(self, mode: StddevSampExecutionMode) -> Self {
        self.with_statistical_aggregate_execution_mode(mode)
    }

    /// Sets grouped aggregate strategy execution policy.
    pub fn with_aggregate_strategy_execution_options(mut self, options: AggregateStrategyExecutionOptions) -> Self {
        self.aggregate_strategy_execution_options = options;
        self
    }

    /// Sets expert local-throughput source chunk planning policy.
    pub fn with_source_chunk_planning_options(mut self, options: SourceChunkPlanningOptions) -> Self {
        self.source_chunk_planning_options = options;
        self
    }

    /// Supplies fixture/config graph calibration metadata.
    pub fn with_graph_memory_calibration_override(mut self, override_: GraphMemoryCalibrationOverride) -> Self {
        self.graph_memory_calibration_override = override_;
        self
    }

    /// Provides the adaptive-only memory history store.
    pub fn with_adaptive_memory_history(mut self, memory_history: Arc<AdaptiveMemoryHistory>) -> Self {
        self.adaptive_memory_history = Some(memory_history);
        self
    }

    /// Provides the cache required by durable GPU hash-join materialization.
    pub fn with_hash_join_build_cache(mut self, cache: Arc<crate::exec::GpuHashJoinBuildCache>) -> Self {
        self.hash_join_build_cache = Some(cache);
        self
    }

    /// Provides the durable GPU local-Parquet source cache.
    ///
    /// Bounded execution charges retained source frames through the reservation
    /// ledger before using cached frames or retaining newly read frames.
    pub fn with_local_parquet_source_cache(mut self, cache: Arc<crate::exec::GpuLocalParquetSourceCache>) -> Self {
        self.local_parquet_source_cache = Some(cache);
        self
    }

    /// Provides the durable GPU resident graph cache.
    #[cfg(feature = "cugraph")]
    pub fn with_resident_graph_cache(mut self, cache: Arc<crate::exec::GpuResidentGraphCache>) -> Self {
        self.resident_graph_cache = Some(cache);
        self
    }

    /// Validates policies, constructs or records the runtime, and returns a context.
    ///
    /// Validation failures carry report-facing facts such as
    /// `missing_hash_join_build_cache`; do not churn those fact keys casually.
    pub fn build(self) -> Result<ExecutionContext> {
        self.memory_control_policy.validate()?;
        self.boundary_policy.validate(self.hash_join_build_cache.is_some())?;
        self.source_scheduler_policy.validate()?;
        self.aggregate_strategy_execution_options.validate()?;
        self.source_chunk_planning_options.validate()?;
        if let Some(service_stream_pool_size) = self.service_stream_pool_size {
            validate_service_stream_pool_size(service_stream_pool_size)?;
        }
        let deployment_io_capability =
            self.deployment_io_capability.unwrap_or_else(DeploymentIoCapability::cached_current_host);
        let mut source_fetch_config = self.source_fetch_config.unwrap_or_default();
        if let Some(host_fetch_budget_bytes) = self.host_fetch_budget_bytes {
            source_fetch_config.host_fetch_budget_bytes = Some(host_fetch_budget_bytes);
        }
        if let Some(host_fetch_acquire_timeout) = self.host_fetch_acquire_timeout {
            source_fetch_config.host_fetch_acquire_timeout = host_fetch_acquire_timeout;
        }
        if let Some(max_concurrent_ops) = self.host_fetch_max_concurrent_ops {
            source_fetch_config.host_fetch_max_concurrent_ops = max_concurrent_ops;
        }
        if let Some(lakehouse_cache_fill_budget_bytes) = self.lakehouse_cache_fill_budget_bytes {
            source_fetch_config.lakehouse_cache_fill_budget_bytes = Some(lakehouse_cache_fill_budget_bytes);
        }
        if let Some(lakehouse_cache_fill_max_concurrent_ops) = self.lakehouse_cache_fill_max_concurrent_ops {
            source_fetch_config.lakehouse_cache_fill_max_concurrent_ops = lakehouse_cache_fill_max_concurrent_ops;
        }
        source_fetch_config.validate_for_deployment(&deployment_io_capability)?;
        let host_fetch_budget = match self.host_fetch_budget {
            Some(budget) => budget,
            None => HostFetchBudget::new(
                source_fetch_config.host_fetch_budget_bytes.unwrap_or_else(default_host_fetch_budget_bytes),
            )?,
        };
        let source_fetch_concurrency =
            SourceFetchConcurrencyGate::new(source_fetch_config.host_fetch_max_concurrent_ops)?;
        let lakehouse_cache_fill_budget = HostFetchBudget::new(
            source_fetch_config.lakehouse_cache_fill_budget_bytes.unwrap_or_else(default_host_fetch_budget_bytes),
        )?;
        let lakehouse_cache_fill_concurrency =
            SourceFetchConcurrencyGate::new(source_fetch_config.lakehouse_cache_fill_max_concurrent_ops)?;
        let gds_concurrency = GdsConcurrencyGate::new(source_fetch_config.gds_max_concurrent_ops)?;
        let source_fetch_runtime = default_source_fetch_runtime()?;
        let lakehouse_cache_fill_runtime = default_lakehouse_cache_fill_runtime()?;
        let source_object_stores = Arc::new(SourceObjectStoreRegistry::with_profiles(self.source_storage_profiles)?);
        let query_id = self.query_id;
        let trace_id = self.trace_id;
        if let Some(runtime) = self.runtime {
            let service_stream_pool_size = self
                .service_stream_pool_size
                .unwrap_or_else(|| default_service_stream_pool_size_for_device(runtime.device()));
            let service_stream_pool = match self.service_stream_pool {
                Some(pool) => {
                    validate_service_stream_pool_device(&pool, runtime.device())?;
                    pool
                },
                None => ServiceStreamPool::for_device(runtime.device(), service_stream_pool_size)?,
            };
            let grant_reservation_arena = create_grant_reservation_arena(runtime.as_ref(), self.memory_grant.as_ref())?;
            return Ok(ExecutionContext::from_parts(ExecutionContextParts {
                query_id,
                trace_id,
                runtime,
                memory_policy: MemoryResourcePolicy::ExternalRuntime,
                memory_control_policy: self.memory_control_policy,
                memory_grant: self.memory_grant.clone(),
                grant_reservation_arena,
                grant_memory_domain: None,
                boundary_policy: self.boundary_policy,
                source_scheduler_policy: self.source_scheduler_policy,
                source_chunk_planning_mode: self.source_chunk_planning_mode,
                source_fetch_config,
                service_stream_pool,
                host_fetch_budget,
                source_fetch_concurrency,
                lakehouse_cache_fill_budget,
                lakehouse_cache_fill_concurrency,
                lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
                gds_concurrency,
                source_fetch_runtime,
                lakehouse_cache_fill_runtime,
                source_object_stores,
                deployment_io_capability,
                cancellation: self.cancellation,
                native_pipeline_execution_mode: self.native_pipeline_execution_mode,
                stddev_samp_execution_mode: self.stddev_samp_execution_mode,
                aggregate_strategy_execution_options: self.aggregate_strategy_execution_options,
                source_chunk_planning_options: self.source_chunk_planning_options,
                graph_memory_calibration_override: self.graph_memory_calibration_override,
                adaptive_memory_history: self.adaptive_memory_history,
                hash_join_build_cache: self.hash_join_build_cache,
                local_parquet_source_cache: self.local_parquet_source_cache,
                #[cfg(feature = "cugraph")]
                resident_graph_cache: self.resident_graph_cache,
                stream_lease_guard: None,
                query_service_admitted: false,
            }));
        }

        let mut builder = cudf::Runtime::builder();
        if let Some(device_ordinal) = self.device_ordinal {
            let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
                QueryError::from_cudf("nexus_query_engine.runtime.device", "failed to resolve CUDA device", error)
            })?;
            builder = builder.with_device(device);
        }

        let memory_policy = match self.memory_policy {
            MemoryResourcePolicy::ExternalRuntime => MemoryResourcePolicy::DefaultCuda,
            other => other,
        };

        match memory_policy {
            MemoryResourcePolicy::DefaultCuda | MemoryResourcePolicy::ExternalRuntime => {},
            MemoryResourcePolicy::Pool { initial_pool_size, maximum_pool_size } => {
                builder = builder.with_pool_memory_resource(initial_pool_size, maximum_pool_size);
            },
        }

        let runtime = builder.build().map_err(|error| {
            QueryError::from_cudf("nexus_query_engine.runtime.build", "failed to initialize cudf runtime", error)
        })?;
        let service_stream_pool_size = self
            .service_stream_pool_size
            .unwrap_or_else(|| default_service_stream_pool_size_for_device(runtime.device()));
        let service_stream_pool = match self.service_stream_pool {
            Some(pool) => {
                validate_service_stream_pool_device(&pool, runtime.device())?;
                pool
            },
            None => ServiceStreamPool::for_device(runtime.device(), service_stream_pool_size)?,
        };
        let runtime = Arc::new(runtime);
        let grant_reservation_arena = create_grant_reservation_arena(runtime.as_ref(), self.memory_grant.as_ref())?;
        Ok(ExecutionContext::from_parts(ExecutionContextParts {
            query_id,
            trace_id,
            runtime,
            memory_policy,
            memory_control_policy: self.memory_control_policy,
            memory_grant: self.memory_grant,
            grant_reservation_arena,
            grant_memory_domain: None,
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_chunk_planning_mode: self.source_chunk_planning_mode,
            source_fetch_config,
            service_stream_pool,
            host_fetch_budget,
            source_fetch_concurrency,
            lakehouse_cache_fill_budget,
            lakehouse_cache_fill_concurrency,
            lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            gds_concurrency,
            source_fetch_runtime,
            lakehouse_cache_fill_runtime,
            source_object_stores,
            deployment_io_capability,
            cancellation: self.cancellation,
            native_pipeline_execution_mode: self.native_pipeline_execution_mode,
            stddev_samp_execution_mode: self.stddev_samp_execution_mode,
            aggregate_strategy_execution_options: self.aggregate_strategy_execution_options,
            source_chunk_planning_options: self.source_chunk_planning_options,
            graph_memory_calibration_override: self.graph_memory_calibration_override,
            adaptive_memory_history: self.adaptive_memory_history,
            hash_join_build_cache: self.hash_join_build_cache,
            local_parquet_source_cache: self.local_parquet_source_cache,
            #[cfg(feature = "cugraph")]
            resident_graph_cache: self.resident_graph_cache,
            stream_lease_guard: None,
            query_service_admitted: false,
        }))
    }
}

fn validate_service_stream_pool_device(pool: &ServiceStreamPool, runtime_device: cudf::DeviceId) -> Result<()> {
    if pool.device() != runtime_device {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.service_stream_pool.device",
            "service stream pool device must match the execution runtime device",
        )
        .with_fact("pool_device_ordinal", pool.device().ordinal())
        .with_fact("runtime_device_ordinal", runtime_device.ordinal()));
    }
    Ok(())
}

fn create_grant_reservation_arena(
    runtime: &cudf::Runtime,
    memory_grant: Option<&MemoryGrant>,
) -> Result<Option<Arc<cudf::ReservationArena>>> {
    let Some(grant) = memory_grant else {
        return Ok(None);
    };
    let grant_key = grant.identity_key();
    let mut registry = grant_reservation_arena_registry().lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.runtime.grant_reservation_arena",
            "grant reservation arena registry mutex was poisoned",
        )
    })?;
    if let Some(arena) = registry.get(&grant_key).and_then(Weak::upgrade) {
        if arena.device() != runtime.device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.grant_reservation_arena",
                "memory grant reservation arena device does not match execution runtime device",
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("query_id", grant.query_id())
            .with_fact("arena_device_ordinal", arena.device().ordinal())
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        return Ok(Some(arena));
    }
    registry.remove(&grant_key);

    let limit_bytes = usize::try_from(grant.budget_bytes()).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.runtime.grant_reservation_arena",
            "memory grant budget exceeds platform usize range",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("query_id", grant.query_id())
        .with_fact("budget_bytes", grant.budget_bytes())
    })?;
    let manager = runtime.device_memory_manager().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.runtime.grant_reservation_arena",
            "failed to resolve device memory manager for grant-backed reservation arena",
            error,
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("query_id", grant.query_id())
    })?;
    let arena = manager.reservation_arena(grant.grant_id(), limit_bytes).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.runtime.grant_reservation_arena",
            "failed to create grant-backed reservation arena",
            error,
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("query_id", grant.query_id())
        .with_fact("budget_bytes", grant.budget_bytes())
    })?;
    let arena = Arc::new(arena);
    let observer_arena = Arc::clone(&arena);
    let resize_observer = sequenced_budget_resize_observer("grant reservation arena", move |budget_bytes| {
        let limit_bytes = usize::try_from(budget_bytes)
            .map_err(|_error| "memory grant budget exceeded platform usize range".to_owned())?;
        observer_arena.resize(limit_bytes).map_err(|error| error.to_string())
    });
    let initial_budget = grant.notify_on_budget_change(Arc::clone(&resize_observer))?;
    registry.insert(grant_key, Arc::downgrade(&arena));
    resize_observer(initial_budget);
    Ok(Some(arena))
}

fn create_grant_memory_domain(
    runtime: &cudf::Runtime,
    memory_grant: Option<&MemoryGrant>,
    policy: Option<QueryMemoryDomainPolicy>,
    device_capacity_bytes: u64,
) -> Result<Option<Arc<cudf::QueryMemoryDomain>>> {
    let (Some(grant), Some(policy)) = (memory_grant, policy) else {
        return Ok(None);
    };
    let grant_key = grant.identity_key();
    let mut registry = grant_memory_domain_registry().lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.runtime.grant_memory_domain",
            "grant query memory domain registry mutex was poisoned",
        )
    })?;
    if let Some(domain) = registry.get(&grant_key).and_then(Weak::upgrade) {
        if domain.device() != runtime.device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.grant_memory_domain",
                "memory grant query domain device does not match execution runtime device",
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("query_id", grant.query_id())
            .with_fact("domain_device_ordinal", domain.device().ordinal())
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        return Ok(Some(domain));
    }
    registry.remove(&grant_key);

    let (grant_bytes_u64, ceiling_bytes_u64) = grant.install_hard_cap_bytes_for_current_budget(|grant_bytes| {
        policy.ceiling_bytes(grant_bytes, device_capacity_bytes)
    })?;

    let grant_bytes = usize::try_from(grant_bytes_u64).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.runtime.grant_memory_domain",
            "memory grant budget exceeds platform usize range",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("query_id", grant.query_id())
        .with_fact("budget_bytes", grant_bytes_u64)
    })?;
    let ceiling_bytes = usize::try_from(ceiling_bytes_u64).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.runtime.grant_memory_domain",
            "memory grant query domain ceiling exceeds platform usize range",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("query_id", grant.query_id())
        .with_fact("ceiling_bytes", ceiling_bytes_u64)
    })?;
    let manager = runtime.device_memory_manager().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.runtime.grant_memory_domain",
            "failed to resolve device memory manager for grant query memory domain",
            error,
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("query_id", grant.query_id())
    })?;
    let domain = manager
        .query_memory_domain(
            grant.grant_id(),
            cudf::QueryMemoryDomainOptions::new(grant_bytes).with_ceiling_bytes(ceiling_bytes),
        )
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.grant_memory_domain",
                "failed to create grant query memory domain",
                error,
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("query_id", grant.query_id())
            .with_fact("budget_bytes", grant_bytes_u64)
            .with_fact("ceiling_bytes", ceiling_bytes_u64)
        })?;
    let domain = Arc::new(domain);
    let observer_domain = Arc::clone(&domain);
    let resize_observer = sequenced_budget_resize_observer("grant query memory domain", move |budget_bytes| {
        let grant_bytes = usize::try_from(budget_bytes)
            .map_err(|_error| "memory grant budget exceeded platform usize range".to_owned())?;
        observer_domain.resize_grant(grant_bytes).map_err(|error| error.to_string())
    });
    let initial_budget = grant.notify_on_budget_change(Arc::clone(&resize_observer))?;
    registry.insert(grant_key, Arc::downgrade(&domain));
    resize_observer(initial_budget);
    Ok(Some(domain))
}

fn sequenced_budget_resize_observer(
    label: &'static str,
    apply: impl Fn(u64) -> std::result::Result<(), String> + Send + Sync + 'static,
) -> gpu::BudgetChangeObserver {
    let observer_sequence = Arc::new(Mutex::new(0_u64));
    Arc::new(move |change| {
        let mut applied_sequence = match observer_sequence.lock() {
            Ok(applied_sequence) => applied_sequence,
            Err(_error) => {
                eprintln!(
                    "[nexus-query-engine] warning: failed to apply memory grant {label} resize after sequence mutex poison"
                );
                return;
            },
        };
        if change.sequence < *applied_sequence {
            return;
        }
        if let Err(error) = apply(change.budget_bytes) {
            eprintln!("[nexus-query-engine] warning: failed to resize memory grant {label}: {error}");
        }
        *applied_sequence = change.sequence;
    })
}

fn grant_reservation_arena_registry() -> &'static Mutex<HashMap<usize, Weak<cudf::ReservationArena>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<cudf::ReservationArena>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn grant_memory_domain_registry() -> &'static Mutex<HashMap<usize, Weak<cudf::QueryMemoryDomain>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<cudf::QueryMemoryDomain>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
mod tests {
    use super::{
        service_stream::{
            default_service_stream_pool_size_from_device_facts, default_service_stream_pool_size_from_sm_count,
            nonzero_sm_count,
        },
        source_fetch::default_host_fetch_budget_bytes_from_physical_memory,
        *,
    };

    #[test]
    fn object_store_fetch_returns_bytes_from_full_get() {
        let runtime = default_source_fetch_runtime().unwrap();
        let store = Arc::new(::object_store::memory::InMemory::new());
        let location = ::object_store::path::Path::from("part-0.parquet");
        runtime.block_on(store.put(&location, b"abcdef".as_ref().into())).unwrap();

        let bytes = ExecutionContext::fetch_object_store_bytes_on_runtime(runtime.as_ref(), store, location).unwrap();

        assert_eq!(bytes.as_ref(), b"abcdef");
    }

    #[test]
    fn service_stream_pool_default_size_is_sm_aware_with_cap_and_fallback() {
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(1)), 2);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(16)), 32);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(32)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(128)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(0)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
        assert_eq!(default_service_stream_pool_size_from_sm_count(None), DEFAULT_SERVICE_STREAM_POOL_SIZE);
    }

    #[derive(Debug)]
    struct FakeGpuInventory {
        facts: Vec<DeviceFacts>,
    }

    impl GpuInventory for FakeGpuInventory {
        fn device_count(&self) -> Result<usize> {
            Ok(self.facts.len())
        }

        fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
            self.facts.get(device_ordinal).cloned().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.test.gpu_inventory",
                    "fake device ordinal out of range",
                )
                .with_fact("device_ordinal", device_ordinal)
            })
        }
    }

    fn fake_device_facts(device: usize, total_bytes: u64, sm_count: Option<u32>) -> DeviceFacts {
        DeviceFacts { device, total_bytes, free_bytes: total_bytes / 2, sm_count, utilization_pct: None }
    }

    #[test]
    fn gpu_inventory_fake_drives_pool_size_and_budget_math_without_gpu() {
        let inventory = FakeGpuInventory { facts: vec![fake_device_facts(0, 16 * 1024 * 1024 * 1024, Some(12))] };
        let facts = inventory.facts(0).unwrap();

        assert_eq!(default_service_stream_pool_size_from_device_facts(&facts), 24);
        assert_eq!(device_budget_bytes_from_facts(&facts, 512 * 1024 * 1024, 256 * 1024 * 1024), 16_642_998_272);
    }

    #[test]
    fn gpu_inventory_all_facts_uses_fake_ordinals_without_cuda_device_resolution() {
        let inventory = FakeGpuInventory {
            facts: vec![
                fake_device_facts(0, 8 * 1024 * 1024 * 1024, Some(10)),
                fake_device_facts(1, 12 * 1024 * 1024 * 1024, Some(20)),
            ],
        };

        let facts = inventory.all_facts().unwrap();

        assert_eq!(facts.iter().map(|facts| facts.device).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(facts.iter().filter_map(|facts| facts.sm_count).collect::<Vec<_>>(), vec![10, 20]);
    }

    #[test]
    fn service_stream_pool_default_size_uses_sm_times_two_for_small_sm_device_fact() {
        let facts = fake_device_facts(0, 4 * 1024 * 1024 * 1024, Some(31));

        assert_eq!(default_service_stream_pool_size_from_device_facts(&facts), 62);
    }

    #[test]
    fn service_stream_pool_sm_count_filters_zero_counts() {
        assert_eq!(nonzero_sm_count(142), Some(142));
        assert_eq!(nonzero_sm_count(1), Some(1));
        assert_eq!(nonzero_sm_count(0), None);
    }

    #[test]
    fn default_host_fetch_budget_is_capped() {
        assert_eq!(
            default_host_fetch_budget_bytes_from_physical_memory(Some(128 * 1024 * 1024 * 1024)),
            SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES
        );
        assert_eq!(
            default_host_fetch_budget_bytes_from_physical_memory(Some(4 * 1024 * 1024 * 1024)),
            1024 * 1024 * 1024
        );
        assert_eq!(
            default_host_fetch_budget_bytes_from_physical_memory(None),
            SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_BYTES
        );
    }

    #[test]
    fn host_fetch_budget_tracks_acquire_grow_and_release() {
        let budget = HostFetchBudget::new(1024).unwrap();
        let mut first = budget.acquire(256).unwrap();
        assert_eq!(first.bytes(), 256);
        first.grow_to(512).unwrap();
        assert_eq!(first.bytes(), 512);
        let second = budget.acquire(512).unwrap();
        drop(first);
        let third = budget.acquire(512).unwrap();
        drop(second);
        drop(third);
    }

    #[test]
    fn host_fetch_budget_rejects_oversized_fetches() {
        let budget = HostFetchBudget::new(1024).unwrap();
        let error = budget.acquire(2048).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
        assert_eq!(error.fact_value("requested_bytes"), Some("2048"));
        assert_eq!(error.fact_value("budget_bytes"), Some("1024"));
    }

    #[test]
    fn execution_context_records_host_fetch_concurrency_limit() {
        let context = ExecutionContext::builder().with_host_fetch_max_concurrent_ops(7).build().unwrap();
        assert_eq!(context.host_fetch_max_concurrent_ops(), 7);
    }

    #[test]
    fn execution_context_records_host_fetch_acquire_timeout() {
        let context =
            ExecutionContext::builder().with_host_fetch_acquire_timeout(Duration::from_secs(11)).build().unwrap();
        assert_eq!(context.host_fetch_acquire_timeout(), Duration::from_secs(11));
    }

    #[test]
    fn execution_context_records_lakehouse_cache_fill_resource_limits() {
        let context = ExecutionContext::builder()
            .with_lakehouse_cache_fill_budget_bytes(2048)
            .with_lakehouse_cache_fill_max_concurrent_ops(3)
            .build()
            .unwrap();

        assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
        assert_eq!(context.lakehouse_cache_fill_max_concurrent_ops(), 3);
    }

    #[test]
    fn lakehouse_cache_fill_event_sink_records_structured_events() {
        let sink = LakehouseCacheFillEventSink::default();

        sink.record(LakehouseCacheFillEvent {
            outcome: LakehouseCacheFillOutcome::Completed,
            bytes: Some(4096),
            elapsed: Duration::from_millis(7),
            cache_path_hash: "cache-path-hash".to_string(),
            stable_id_hash: "stable-id-hash".to_string(),
            file_index: Some(2),
            chunk_index: None,
            error: None,
        });
        sink.record(LakehouseCacheFillEvent {
            outcome: LakehouseCacheFillOutcome::Failed,
            bytes: None,
            elapsed: Duration::from_millis(3),
            cache_path_hash: "failed-path-hash".to_string(),
            stable_id_hash: "failed-stable-hash".to_string(),
            file_index: Some(3),
            chunk_index: Some(4),
            error: Some(LakehouseCacheFillErrorEvent {
                code: "cudf_dependency_failure".to_string(),
                kind: "external".to_string(),
                status: "temporary".to_string(),
                operation: "nexus_query_engine.lakehouse_cache.background_fill.fetch".to_string(),
            }),
        });

        let events = sink.events();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].outcome, LakehouseCacheFillOutcome::Completed);
        assert_eq!(events[0].bytes, Some(4096));
        assert_eq!(events[1].outcome, LakehouseCacheFillOutcome::Failed);
        assert_eq!(
            events[1].error.as_ref().map(|error| error.operation.as_str()),
            Some("nexus_query_engine.lakehouse_cache.background_fill.fetch")
        );
    }

    #[test]
    fn stream_lease_context_shares_lakehouse_cache_fill_event_sink() {
        let context = ExecutionContext::builder().build().unwrap();
        let stream_lease = context.service_stream_pool().acquire().unwrap();
        let leased_context = context.with_stream_lease(&stream_lease).unwrap();

        leased_context.lakehouse_cache_fill_events().record(LakehouseCacheFillEvent {
            outcome: LakehouseCacheFillOutcome::Completed,
            bytes: Some(8192),
            elapsed: Duration::from_millis(5),
            cache_path_hash: "leased-cache-path-hash".to_string(),
            stable_id_hash: "leased-stable-id-hash".to_string(),
            file_index: Some(1),
            chunk_index: Some(2),
            error: None,
        });

        let snapshots = context.lakehouse_cache_fill_event_snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].outcome, "completed");
        assert_eq!(snapshots[0].bytes, Some(8192));
        assert_eq!(snapshots[0].cache_path_hash, "leased-cache-path-hash");
    }

    #[test]
    fn lakehouse_cache_fill_event_snapshots_can_be_drained_once() {
        let context = ExecutionContext::builder().build().unwrap();
        context.lakehouse_cache_fill_events().record(LakehouseCacheFillEvent {
            outcome: LakehouseCacheFillOutcome::Completed,
            bytes: Some(1024),
            elapsed: Duration::from_millis(2),
            cache_path_hash: "cache-path-hash".to_string(),
            stable_id_hash: "stable-id-hash".to_string(),
            file_index: None,
            chunk_index: None,
            error: None,
        });

        let first = context.drain_lakehouse_cache_fill_event_snapshots();
        let second = context.drain_lakehouse_cache_fill_event_snapshots();

        assert_eq!(first.len(), 1);
        assert_eq!(first[0].outcome, "completed");
        assert!(second.is_empty());
        assert!(context.lakehouse_cache_fill_event_snapshots().is_empty());
    }

    #[test]
    fn unbounded_fragment_context_preserves_query_state() {
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::builder()
            .with_cancellation(cancellation.clone())
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 1024 * 1024,
                max_source_chunk_bytes: 64 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_host_fetch_max_concurrent_ops(7)
            .build()
            .unwrap();
        context.lakehouse_cache_fill_events().record(LakehouseCacheFillEvent {
            outcome: LakehouseCacheFillOutcome::Completed,
            bytes: Some(2048),
            elapsed: Duration::from_millis(3),
            cache_path_hash: "fragment-cache-path-hash".to_string(),
            stable_id_hash: "fragment-stable-id-hash".to_string(),
            file_index: Some(1),
            chunk_index: Some(2),
            error: None,
        });
        cancellation.cancel();

        let fragment_context = context.for_unbounded_fragment();

        assert_eq!(fragment_context.memory_control_policy(), MemoryControlPolicy::Unbounded);
        assert!(fragment_context.cancellation().is_cancelled());
        assert_eq!(fragment_context.host_fetch_max_concurrent_ops(), 7);
        assert_eq!(fragment_context.lakehouse_cache_fill_event_snapshots().len(), 1);
        assert_eq!(context.lakehouse_cache_fill_event_snapshots().len(), 1);
    }

    #[test]
    fn execution_context_records_source_chunk_planning_mode() {
        let options =
            SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(2048);
        let context = ExecutionContext::builder()
            .with_source_chunk_planning_mode(SourceChunkPlanningMode::LocalParquetThroughput)
            .with_source_chunk_planning_options(options)
            .build()
            .unwrap();

        assert_eq!(context.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
        assert_eq!(context.source_chunk_planning_options(), options);
    }

    #[test]
    fn execution_context_rejects_invalid_source_chunk_planning_options() {
        let error = ExecutionContext::builder()
            .with_source_chunk_planning_options(
                SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(0),
            )
            .build()
            .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_max_row_groups_per_chunk"));
    }

    #[test]
    fn execution_context_rejects_zero_host_fetch_concurrency_limit() {
        let error = ExecutionContext::builder().with_host_fetch_max_concurrent_ops(0).build().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("host_fetch_max_concurrent_ops"));
    }

    #[test]
    fn execution_context_rejects_zero_host_fetch_acquire_timeout() {
        let error = ExecutionContext::builder().with_host_fetch_acquire_timeout(Duration::ZERO).build().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("host_fetch_acquire_timeout"));
    }

    fn available_gds_capability() -> DeploymentIoCapability {
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::Present { path: std::path::PathBuf::from(NVIDIA_FS_PROC_PATH) },
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::Available,
        )
    }

    #[test]
    fn source_fetch_config_defaults_match_source_fetch_contract() {
        let config = SourceFetchConfig::default();
        assert_eq!(config.substrate, SourceFetchSubstrate::Auto);
        assert_eq!(config.host_fetch_budget_bytes, None);
        assert_eq!(config.host_fetch_acquire_timeout, SourceFetchConfig::DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT);
        assert_eq!(config.host_fetch_max_concurrent_ops, SourceFetchConfig::DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS);
        assert_eq!(config.lakehouse_cache_fill_budget_bytes, None);
        assert_eq!(
            config.lakehouse_cache_fill_max_concurrent_ops,
            SourceFetchConfig::DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS
        );
        assert_eq!(config.gds_max_concurrent_ops, SourceFetchConfig::DEFAULT_GDS_MAX_CONCURRENT_OPS);
        assert_eq!(config.gds_min_chunk_bytes, SourceFetchConfig::DEFAULT_GDS_MIN_CHUNK_BYTES);
        assert_eq!(SourceFetchConfig::DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT, Duration::from_secs(30));
        assert_eq!(DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT, SourceFetchConfig::DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT);
        assert_eq!(DEFAULT_HOST_FETCH_BUDGET_BYTES, SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_BYTES);
        assert_eq!(DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES, SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES);
        assert_eq!(DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS, SourceFetchConfig::DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS);
        assert_eq!(
            DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS,
            SourceFetchConfig::DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS
        );
        assert_eq!(DEFAULT_GDS_MAX_CONCURRENT_OPS, SourceFetchConfig::DEFAULT_GDS_MAX_CONCURRENT_OPS);
        assert_eq!(DEFAULT_GDS_MIN_CHUNK_BYTES, SourceFetchConfig::DEFAULT_GDS_MIN_CHUNK_BYTES);
    }

    #[test]
    fn source_fetch_strategy_auto_keeps_remote_object_store_on_host_buffer() {
        let config = SourceFetchConfig::default();
        let decision = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "s3://bucket/path/file.parquet",
            Some(64 * 1024 * 1024),
        );

        assert_eq!(decision.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(decision.reason, SourceFetchStrategyReason::RemoteObjectStore);
        assert_eq!(decision.gds_eligibility, SourceGdsEligibility::RemoteObjectStore);
        assert_eq!(
            decision.diagnostic_facts(),
            vec![
                ("source_fetch_strategy", "host_buffer".to_string()),
                ("source_fetch_strategy_reason", "remote_object_store".to_string()),
                ("source_fetch_substrate", "auto".to_string()),
                ("source_gds_eligibility", "remote_object_store".to_string()),
            ]
        );
    }

    #[test]
    fn source_fetch_strategy_auto_selects_direct_device_only_for_gds_eligible_local_chunks() {
        let config = SourceFetchConfig::default();
        let decision = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(64 * 1024 * 1024),
        );

        assert_eq!(decision.strategy, SourceFetchStrategy::DirectDevice);
        assert_eq!(decision.reason, SourceFetchStrategyReason::GdsEligible);
        assert_eq!(decision.gds_eligibility, SourceGdsEligibility::Eligible);
    }

    #[test]
    fn source_fetch_strategy_host_buffer_substrate_overrides_gds_eligible_sources() {
        let config = SourceFetchConfig { substrate: SourceFetchSubstrate::HostBuffer, ..SourceFetchConfig::default() };
        let decision = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(64 * 1024 * 1024),
        );

        assert_eq!(decision.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(decision.reason, SourceFetchStrategyReason::SubstrateHostBuffer);
        assert_eq!(decision.gds_eligibility, SourceGdsEligibility::Eligible);
    }

    #[test]
    fn source_fetch_strategy_direct_device_still_falls_back_for_ineligible_sources() {
        let config =
            SourceFetchConfig { substrate: SourceFetchSubstrate::DirectDevice, ..SourceFetchConfig::default() };

        let remote = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "https://example.test/file.parquet",
            Some(64 * 1024 * 1024),
        );
        assert_eq!(remote.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(remote.reason, SourceFetchStrategyReason::RemoteObjectStore);

        let small = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(1024),
        );
        assert_eq!(small.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(small.reason, SourceFetchStrategyReason::BelowMinChunkBytes);

        let eligible = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(64 * 1024 * 1024),
        );
        assert_eq!(eligible.strategy, SourceFetchStrategy::DirectDevice);
        assert_eq!(eligible.reason, SourceFetchStrategyReason::GdsEligible);
    }

    #[test]
    fn execution_context_exposes_effective_source_fetch_strategy() {
        let context = ExecutionContext::builder()
            .with_deployment_io_capability(available_gds_capability())
            .with_source_fetch_config(SourceFetchConfig::default())
            .build()
            .unwrap();

        let decision = context
            .source_fetch_strategy_for_location("/opt/dlami/nvme/tpch/sf10/lineitem.parquet", Some(64 * 1024 * 1024));

        assert_eq!(decision.strategy, SourceFetchStrategy::DirectDevice);
        assert_eq!(decision.substrate, SourceFetchSubstrate::Auto);
    }

    #[test]
    fn source_fetch_config_rejects_zero_gds_controls() {
        let deployment = available_gds_capability();
        let zero_ops = SourceFetchConfig { gds_max_concurrent_ops: 0, ..SourceFetchConfig::default() };
        let error = zero_ops.validate_for_deployment(&deployment).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("gds_max_concurrent_ops"));

        let zero_cache_fill_ops =
            SourceFetchConfig { lakehouse_cache_fill_max_concurrent_ops: 0, ..SourceFetchConfig::default() };
        let error = zero_cache_fill_ops.validate_for_deployment(&deployment).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("lakehouse_cache_fill_max_concurrent_ops"));

        let zero_cache_fill_budget =
            SourceFetchConfig { lakehouse_cache_fill_budget_bytes: Some(0), ..SourceFetchConfig::default() };
        let error = zero_cache_fill_budget.validate_for_deployment(&deployment).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("lakehouse_cache_fill_budget_bytes"));

        let zero_host_fetch_acquire_timeout =
            SourceFetchConfig { host_fetch_acquire_timeout: Duration::ZERO, ..SourceFetchConfig::default() };
        let error = zero_host_fetch_acquire_timeout.validate_for_deployment(&deployment).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("host_fetch_acquire_timeout"));

        let zero_min_chunk = SourceFetchConfig { gds_min_chunk_bytes: 0, ..SourceFetchConfig::default() };
        let error = zero_min_chunk.validate_for_deployment(&deployment).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("gds_min_chunk_bytes"));
    }

    #[test]
    fn direct_device_source_fetch_config_requires_gds_available() {
        let unavailable = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::Available,
        );
        let config =
            SourceFetchConfig { substrate: SourceFetchSubstrate::DirectDevice, ..SourceFetchConfig::default() };
        let error = config.validate_for_deployment(&unavailable).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("source_fetch_substrate"), Some("direct_device"));
        assert_eq!(error.fact_value("gds_available"), Some("false"));
        assert_eq!(error.fact_value("gds_status"), Some("driver_missing"));

        config.validate_for_deployment(&available_gds_capability()).unwrap();
    }

    #[test]
    fn gds_concurrency_gate_limits_in_flight_ops() {
        let gate = GdsConcurrencyGate::new(1).unwrap();
        let permit = gate.acquire().unwrap();
        let gate_for_thread = gate.clone();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let (acquired_sender, acquired_receiver) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            let permit = gate_for_thread.acquire().unwrap();
            acquired_sender.send(()).unwrap();
            drop(permit);
        });
        started_receiver.recv().unwrap();
        assert!(acquired_receiver.try_recv().is_err());
        drop(permit);
        acquired_receiver.recv().unwrap();
        handle.join().unwrap();
    }
}
