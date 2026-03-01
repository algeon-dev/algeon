//! Runtime, memory, boundary, and reservation contracts for native execution.
//!
//! This module is intentionally close to the public API because these values
//! control more than local execution choices. Policy enum string values,
//! reservation fact keys, and retry fact keys flow into tests, reports, and
//! adapter diagnostics. Treat those strings as stable diagnostic contracts:
//! change them only with the same care as a report schema change.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use ::object_store::ObjectStoreExt;
#[cfg(test)]
use bytes::Bytes;

use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    error::{QueryError, QueryErrorCode, Result},
    mode::{NativePipelineExecutionMode, StddevSampExecutionMode},
};

mod boundary;
mod bounded_join;
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
    QueryHandle, QueryService, QueryServiceAdmissionSnapshot, QueryServiceConfig, QueryServiceTenantConfig, TenantId,
    TenantQueryContext, TenantQueueDepth,
};
pub use reservation::{MemoryReservationRequest, RetryFacts};
pub(crate) use service_stream::SourceReadStreamLease;
use service_stream::validate_service_stream_pool_size;
pub use service_stream::{
    DEFAULT_SERVICE_STREAM_POOL_SIZE, ServiceStreamPool, StreamLease, default_service_stream_pool_size_for_device,
};
pub use source_fetch::{
    DEFAULT_GDS_MAX_CONCURRENT_OPS, DEFAULT_GDS_MIN_CHUNK_BYTES, DEFAULT_HOST_FETCH_BUDGET_BYTES,
    DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES, DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS,
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
    pub fn from_runtime(runtime: cudf::Runtime) -> Self {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    /// Builds a context around a shared cudf runtime.
    ///
    /// Use this when multiple plans or adapter layers must share the same CUDA
    /// stream, device placement, allocator policy, or boundary observer state.
    pub fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Self {
        let service_stream_pool =
            ServiceStreamPool::new(runtime.device(), default_service_stream_pool_size_for_device(runtime.device()))
                .expect("default service stream pool size is non-zero");
        let source_fetch_config = SourceFetchConfig::default();
        Self {
            runtime,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            source_fetch_config,
            service_stream_pool,
            host_fetch_budget: HostFetchBudget::default(),
            source_fetch_concurrency: SourceFetchConcurrencyGate::default(),
            lakehouse_cache_fill_budget: HostFetchBudget::default(),
            lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate::new(
                DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS,
            )
            .expect("default lakehouse cache fill concurrency is non-zero"),
            lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            gds_concurrency: GdsConcurrencyGate::default(),
            source_fetch_runtime: default_source_fetch_runtime(),
            lakehouse_cache_fill_runtime: default_lakehouse_cache_fill_runtime(),
            source_object_stores: Arc::new(SourceObjectStoreRegistry::new()),
            deployment_io_capability: DeploymentIoCapability::probe_current_host(),
            cancellation: ExecutionCancellation::new(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            adaptive_memory_history: None,
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
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

    /// Acquires a same-device leased context for a scheduled source-read worker.
    pub(crate) fn acquire_source_read_lease(&self) -> Result<SourceReadStreamLease> {
        let stream_lease = self.service_stream_pool.acquire()?;
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
        Ok(Self {
            runtime: stream_lease.shared_runtime(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy,
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
            adaptive_memory_history: self.adaptive_memory_history.as_ref().map(Arc::clone),
            hash_join_build_cache: self.hash_join_build_cache.as_ref().map(Arc::clone),
            local_parquet_source_cache: self.local_parquet_source_cache.as_ref().map(Arc::clone),
        })
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
        self.memory_control_policy
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
    /// Prebuilt runtime to use instead of constructing one.
    runtime: Option<Arc<cudf::Runtime>>,
    /// CUDA device ordinal used only when this builder constructs the runtime.
    device_ordinal: Option<usize>,
    /// Memory resource used only when this builder constructs the runtime.
    memory_policy: MemoryResourcePolicy,
    /// Engine-level bounded/unbounded memory contract.
    memory_control_policy: MemoryControlPolicy,
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
    /// Optional adaptive-only memory history store.
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    /// Optional durable hash-join build cache.
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
    /// Optional durable local-Parquet source cache.
    local_parquet_source_cache: Option<Arc<crate::exec::GpuLocalParquetSourceCache>>,
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

    /// Validates policies, constructs or records the runtime, and returns a context.
    ///
    /// Validation failures carry report-facing facts such as
    /// `missing_hash_join_build_cache`; do not churn those fact keys casually.
    pub fn build(self) -> Result<ExecutionContext> {
        self.memory_control_policy.validate()?;
        self.boundary_policy.validate(self.hash_join_build_cache.is_some())?;
        self.source_scheduler_policy.validate()?;
        if let Some(service_stream_pool_size) = self.service_stream_pool_size {
            validate_service_stream_pool_size(service_stream_pool_size)?;
        }
        let deployment_io_capability =
            self.deployment_io_capability.unwrap_or_else(DeploymentIoCapability::probe_current_host);
        let mut source_fetch_config = self.source_fetch_config.unwrap_or_default();
        if let Some(host_fetch_budget_bytes) = self.host_fetch_budget_bytes {
            source_fetch_config.host_fetch_budget_bytes = Some(host_fetch_budget_bytes);
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
        let source_fetch_runtime = default_source_fetch_runtime();
        let lakehouse_cache_fill_runtime = default_lakehouse_cache_fill_runtime();
        let source_object_stores = Arc::new(SourceObjectStoreRegistry::with_profiles(self.source_storage_profiles)?);
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
            return Ok(ExecutionContext {
                runtime,
                memory_policy: MemoryResourcePolicy::ExternalRuntime,
                memory_control_policy: self.memory_control_policy,
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
                adaptive_memory_history: self.adaptive_memory_history,
                hash_join_build_cache: self.hash_join_build_cache,
                local_parquet_source_cache: self.local_parquet_source_cache,
            });
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
        Ok(ExecutionContext {
            runtime: Arc::new(runtime),
            memory_policy,
            memory_control_policy: self.memory_control_policy,
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
            adaptive_memory_history: self.adaptive_memory_history,
            hash_join_build_cache: self.hash_join_build_cache,
            local_parquet_source_cache: self.local_parquet_source_cache,
        })
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

#[cfg(test)]
mod tests {
    use super::{
        service_stream::{default_service_stream_pool_size_from_sm_count, nonzero_sm_count},
        source_fetch::default_host_fetch_budget_bytes_from_physical_memory,
        *,
    };

    #[test]
    fn object_store_fetch_returns_bytes_from_full_get() {
        let runtime = default_source_fetch_runtime();
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
            DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES
        );
        assert_eq!(
            default_host_fetch_budget_bytes_from_physical_memory(Some(4 * 1024 * 1024 * 1024)),
            1024 * 1024 * 1024
        );
        assert_eq!(default_host_fetch_budget_bytes_from_physical_memory(None), DEFAULT_HOST_FETCH_BUDGET_BYTES);
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
    fn execution_context_records_source_chunk_planning_mode() {
        let context = ExecutionContext::builder()
            .with_source_chunk_planning_mode(SourceChunkPlanningMode::LocalParquetThroughput)
            .build()
            .unwrap();

        assert_eq!(context.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    }

    #[test]
    fn execution_context_rejects_zero_host_fetch_concurrency_limit() {
        let error = ExecutionContext::builder().with_host_fetch_max_concurrent_ops(0).build().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("host_fetch_max_concurrent_ops"));
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
        assert_eq!(config.host_fetch_max_concurrent_ops, DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS);
        assert_eq!(config.lakehouse_cache_fill_budget_bytes, None);
        assert_eq!(config.lakehouse_cache_fill_max_concurrent_ops, DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS);
        assert_eq!(config.gds_max_concurrent_ops, DEFAULT_GDS_MAX_CONCURRENT_OPS);
        assert_eq!(config.gds_min_chunk_bytes, DEFAULT_GDS_MIN_CHUNK_BYTES);
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
