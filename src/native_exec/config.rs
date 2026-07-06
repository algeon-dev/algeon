//! Native execution settings copied into rewritten DataFusion plan nodes.
//!
//! The config is cloneable and cheap to pass through planning. cuDF runtime
//! construction, allocator setup, memory-control validation, and cache wiring
//! happen later when execution builds the native query context.

use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use datafusion::execution::context::TaskContext;
use nexus_query_engine::{
    AdaptiveMemoryHistory, AggregateStrategyExecutionOptions, CudfInventory, DynamicFilterExecutionOptions,
    ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext, GpuHashJoinBuildCache, GpuInventory,
    GpuLocalParquetSourceCache, KeyedHashJoinStrategyOptions, MemoryControlPolicy, MemoryResourcePolicy,
    NativePipelineExecutionMode, QueryHandle, QueryId, ServiceStreamPool, SourceChunkPlanningMode,
    SourceChunkPlanningOptions, SourceFetchConfig, SourceSchedulerPolicy, SourceStorageProfile,
    StddevSampExecutionMode, TenantQueryContext, TraceId, device_budget_bytes_from_facts,
};

use super::{
    admission::{QueryAdmissionSource, QueryAdmissionTicket, TicketTenantQueryContext},
    metrics::NativeMetricsCapture,
};
use crate::error::Error;

/// Adapter-side native execution configuration.
///
/// This type holds the DataFusion-facing knobs for native cuDF execution. It does not eagerly create
/// cuDF runtime state; `execution_context` builds the native `ExecutionContext` at execution time so
/// runtime injection, allocator setup, memory-control validation, boundary policy, and durable build
/// cache wiring happen at the DataFusion/native boundary.
#[derive(Debug, Clone)]
pub struct NativeCudfExecutionConfig {
    runtime: Option<Arc<cudf::Runtime>>,
    device_ordinal: Option<usize>,
    memory_resource_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
    boundary_policy: ExecutionBoundaryPolicy,
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    source_fetch_config: SourceFetchConfig,
    host_output_memory_pool_accounting: bool,
    service_stream_pool: Option<ServiceStreamPool>,
    service_stream_pool_size: Option<usize>,
    cancellation: ExecutionCancellation,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    stddev_samp_execution_mode: StddevSampExecutionMode,
    dynamic_filter_execution_options: DynamicFilterExecutionOptions,
    keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions,
    aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    source_chunk_planning_options: SourceChunkPlanningOptions,
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    hash_join_build_cache: Option<Arc<GpuHashJoinBuildCache>>,
    local_parquet_source_cache: Option<Arc<GpuLocalParquetSourceCache>>,
    source_storage_profiles: Vec<SourceStorageProfile>,
    query_service_handle: Option<Arc<QueryHandle>>,
    query_admission_source: Option<QueryAdmissionSource>,
    query_admission_ticket: Option<Arc<QueryAdmissionTicket>>,
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    plan_task_mapping_datafusion_shape: Option<String>,
    metrics_capture: Option<NativeMetricsCapture>,
}

/// Native execution context plus any service leases that must outlive execution.
#[derive(Debug)]
pub(crate) struct NativeCudfExecutionContext {
    context: ExecutionContext,
    _tenant_context: Option<TenantQueryContext>,
    _ticket_context: Option<TicketTenantQueryContext>,
}

impl NativeCudfExecutionContext {
    fn standalone(context: ExecutionContext) -> Self {
        Self { context, _tenant_context: None, _ticket_context: None }
    }

    fn tenant_bound(tenant_context: TenantQueryContext) -> Self {
        let context = tenant_context.context().clone();
        Self { context, _tenant_context: Some(tenant_context), _ticket_context: None }
    }

    fn ticket_bound(ticket_context: TicketTenantQueryContext) -> Self {
        let context = ticket_context.context().context().clone();
        Self { context, _tenant_context: None, _ticket_context: Some(ticket_context) }
    }

    fn with_query_id(mut self, query_id: QueryId) -> Self {
        self.context = self.context.with_query_id(query_id);
        self
    }

    fn with_trace_id(mut self, trace_id: TraceId) -> Self {
        self.context = self.context.with_trace_id(trace_id);
        self
    }

    pub(crate) fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

impl NativeCudfExecutionConfig {
    /// Fallback bounded device budget, in bytes, used when auto-sizing cannot
    /// read the primary GPU's total memory.
    pub const AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES: u64 = 256 * 1024 * 1024;
    /// Device-memory headroom, in bytes, reserved for libcudf and driver
    /// overhead when deriving a bounded budget from CUDA device inventory.
    pub const AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES: u64 = 500 * 1024 * 1024;
    /// Default bounded source chunk byte target used by
    /// [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES: u64 = 64 * 1024 * 1024;
    /// Default bounded source row-group target used by
    /// [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
    /// Default bounded retry-attempt cap used by [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS: usize = 3;
    /// Default bounded retry floor, in row groups, used by
    /// [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

    /// Use the default allocator policy with no native bounded-memory execution controls.
    pub fn unbounded() -> Self {
        Self {
            runtime: None,
            device_ordinal: None,
            memory_resource_policy: MemoryResourcePolicy::DefaultCuda,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            source_fetch_config: SourceFetchConfig::default(),
            host_output_memory_pool_accounting: false,
            service_stream_pool: None,
            service_stream_pool_size: None,
            cancellation: ExecutionCancellation::new(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            dynamic_filter_execution_options: DynamicFilterExecutionOptions::default(),
            keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            adaptive_memory_history: None,
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
            source_storage_profiles: Vec::new(),
            query_service_handle: None,
            query_admission_source: None,
            query_admission_ticket: None,
            query_id: None,
            trace_id: None,
            plan_task_mapping_datafusion_shape: None,
            metrics_capture: None,
        }
    }

    /// Use default runtime allocator construction with an auto-sized bounded
    /// memory-control profile.
    ///
    /// The device budget is CUDA device 0's total memory minus
    /// [`Self::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES`]. If CUDA inventory is
    /// unavailable or fails, the budget
    /// falls back to [`Self::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES`]. Source
    /// chunking uses a 64 MiB byte target, one row group per chunk, three retry
    /// attempts, and one row group as the retry floor.
    pub fn bounded_auto_sized() -> Self {
        Self::bounded(
            Self::auto_sized_device_budget_bytes(),
            Self::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            Self::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            Self::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            Self::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
        )
    }

    /// Returns the auto-sized bounded device budget in bytes.
    ///
    /// The value is derived from the primary GPU's reported total memory minus
    /// 500 MiB of libcudf/driver headroom. A 256 MiB fallback keeps bounded
    /// report and service defaults constructible on hosts without CUDA
    /// inventory or without a visible GPU.
    pub fn auto_sized_device_budget_bytes() -> u64 {
        static AUTO_SIZED_DEVICE_BUDGET_BYTES: OnceLock<u64> = OnceLock::new();
        *AUTO_SIZED_DEVICE_BUDGET_BYTES.get_or_init(detect_auto_sized_device_budget_bytes)
    }

    /// Returns the auto-sized bounded device budget for `device_ordinal`.
    ///
    /// Unlike [`Self::auto_sized_device_budget_bytes`], this helper is not
    /// process-cached because server placement can choose a non-default ordinal.
    pub fn auto_sized_device_budget_bytes_for_device(device_ordinal: usize) -> u64 {
        auto_sized_device_budget_bytes_for_device_with_inventory(&CudfInventory, device_ordinal)
    }

    /// Returns the auto-sized bounded device budget for `device_ordinal` after
    /// applying a total-memory budget fraction.
    ///
    /// The fraction is an upper bound on usable device memory. The fixed
    /// headroom remains a floor, so `1.0` preserves the historical
    /// `total - 500 MiB` behavior instead of granting the full device.
    pub fn auto_sized_device_budget_bytes_for_device_with_fraction(device_ordinal: usize, budget_fraction: f64) -> u64 {
        auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(
            &CudfInventory,
            device_ordinal,
            budget_fraction,
        )
    }

    /// Use default runtime allocator construction with bounded native memory-control settings.
    ///
    /// Byte fields are device-byte budgets. `max_source_chunk_bytes` and
    /// `max_row_groups_per_chunk` are planning targets for source chunking, not
    /// absolute promises that every emitted chunk will be smaller. Positive
    /// value validation is performed when the native `ExecutionContext` is
    /// built, which keeps this builder lightweight and reports native
    /// validation failures as adapter execution errors.
    pub fn bounded(
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    ) -> Self {
        Self {
            runtime: None,
            device_ordinal: None,
            memory_resource_policy: MemoryResourcePolicy::DefaultCuda,
            memory_control_policy: MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            },
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            source_fetch_config: SourceFetchConfig::default(),
            host_output_memory_pool_accounting: false,
            service_stream_pool: None,
            service_stream_pool_size: None,
            cancellation: ExecutionCancellation::new(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            dynamic_filter_execution_options: DynamicFilterExecutionOptions::default(),
            keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            adaptive_memory_history: None,
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
            source_storage_profiles: Vec::new(),
            query_service_handle: None,
            query_admission_source: None,
            query_admission_ticket: None,
            query_id: None,
            trace_id: None,
            plan_task_mapping_datafusion_shape: None,
            metrics_capture: None,
        }
    }

    /// Own `runtime` and share it with native execution through an `Arc`.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self
    }

    /// Use a caller-shared runtime, preserving its device, stream, allocator, and observer state.
    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Select the device for lazily created runtimes.
    ///
    /// This setting only matters when no runtime is supplied with `with_runtime` or
    /// `with_shared_runtime`; injected runtimes keep their own device selection.
    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    /// Set the allocator construction policy used when native execution builds a runtime.
    pub fn with_memory_resource_policy(mut self, memory_resource_policy: MemoryResourcePolicy) -> Self {
        self.memory_resource_policy = memory_resource_policy;
        self
    }

    /// Set bounded execution controls such as chunk sizing, reservations, and retry limits.
    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    /// Set source, build, probe, and result boundary choices for native execution.
    ///
    /// `ExecutionBoundaryPolicy` controls where data may remain device-resident across source reads,
    /// hash-join build/probe work, and result materialization. Durable hash-join build boundaries
    /// require `with_hash_join_build_cache` so the native context has a cache to write to and probe.
    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    /// Set bounded source scheduling controls.
    pub fn with_source_scheduler_policy(mut self, source_scheduler_policy: SourceSchedulerPolicy) -> Self {
        self.source_scheduler_policy = source_scheduler_policy;
        self
    }

    /// Set bounded source chunk planning mode.
    pub fn with_source_chunk_planning_mode(mut self, mode: SourceChunkPlanningMode) -> Self {
        self.source_chunk_planning_mode = mode;
        self
    }

    /// Set service-level source IO resource policy for standalone native execution.
    ///
    /// Query-service execution gets this policy from the admitted service handle
    /// instead; this setter covers direct `NativeCudfExec`/optimizer usage.
    pub fn with_source_fetch_config(mut self, source_fetch_config: SourceFetchConfig) -> Self {
        self.source_fetch_config = source_fetch_config;
        self
    }

    /// Set the timeout used by blocking object-store SourceFetch resource acquisition.
    pub fn with_host_fetch_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.source_fetch_config.host_fetch_acquire_timeout = timeout;
        self
    }

    /// Account native host Arrow output batches against DataFusion's host
    /// [`MemoryPool`](datafusion::execution::memory_pool::MemoryPool).
    ///
    /// This is intentionally host-only. Device memory admission and grants stay
    /// in the native query service/governor ledger.
    pub fn with_host_output_memory_pool_accounting(mut self, enabled: bool) -> Self {
        self.host_output_memory_pool_accounting = enabled;
        self
    }

    /// Use a caller-owned service stream pool for bounded source-read workers.
    pub fn with_service_stream_pool(mut self, service_stream_pool: ServiceStreamPool) -> Self {
        self.service_stream_pool = Some(service_stream_pool);
        self
    }

    /// Set the size of the default process-global service stream pool.
    ///
    /// Ignored when a pool is supplied with [`Self::with_service_stream_pool`].
    pub fn with_service_stream_pool_size(mut self, pool_size: usize) -> Self {
        self.service_stream_pool_size = Some(pool_size);
        self
    }

    /// Attach a cooperative cancellation flag observed by native execution.
    pub fn with_cancellation(mut self, cancellation: ExecutionCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Attach a server-side query metrics capture sink.
    #[cfg(feature = "server")]
    pub(crate) fn with_metrics_capture(mut self, capture: NativeMetricsCapture) -> Self {
        self.query_id = Some(QueryId::new(capture.query_id()));
        self.metrics_capture = Some(capture);
        self
    }

    /// Attach a diagnostic query id to native execution metrics.
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    /// Attach a query id only when the caller has not supplied one.
    pub(crate) fn with_query_id_if_absent(mut self, query_id: impl Into<QueryId>) -> Self {
        if self.query_id.is_none() {
            self.query_id = Some(query_id.into());
        }
        self
    }

    /// Attach a parent trace id to native diagnostics.
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Attach the DataFusion candidate shape used to annotate plan/task mapping rows.
    pub(crate) fn with_plan_task_mapping_datafusion_shape(mut self, shape: impl Into<String>) -> Self {
        let shape = shape.into();
        if !shape.is_empty() {
            self.plan_task_mapping_datafusion_shape = Some(shape);
        }
        self
    }

    pub(crate) fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    pub(crate) fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    pub(crate) fn plan_task_mapping_datafusion_shape(&self) -> Option<&str> {
        self.plan_task_mapping_datafusion_shape.as_deref()
    }

    pub(crate) fn metrics_capture(&self) -> Option<NativeMetricsCapture> {
        self.metrics_capture.clone()
    }

    /// Select strict bounded or adaptive native bounded-pipeline execution.
    pub fn with_native_pipeline_execution_mode(mut self, mode: NativePipelineExecutionMode) -> Self {
        self.native_pipeline_execution_mode = mode;
        self
    }

    /// Select the execution contract for floating statistical aggregates.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StddevSampExecutionMode) -> Self {
        self.stddev_samp_execution_mode = mode;
        self
    }

    /// Select the execution contract through the legacy STDDEV_SAMP builder.
    pub fn with_stddev_samp_execution_mode(self, mode: StddevSampExecutionMode) -> Self {
        self.with_statistical_aggregate_execution_mode(mode)
    }

    /// Set native dynamic-filter execution policy.
    ///
    /// This policy is copied into native optimizer options so route discovery
    /// and runtime payload handling share one threshold set. It is intentionally
    /// not exposed as a session/server knob until the expert surface is reviewed.
    pub fn with_dynamic_filter_execution_options(mut self, options: DynamicFilterExecutionOptions) -> Self {
        self.dynamic_filter_execution_options = options;
        self
    }

    /// Set native keyed-hash join strategy policy.
    ///
    /// This policy is copied into optimized native plans so capability
    /// admission, strict reports, and runtime observed-build strategy selection
    /// share one threshold set. It is intentionally not exposed as a
    /// session/server knob until the expert surface is reviewed.
    pub fn with_keyed_hash_join_strategy_options(mut self, options: KeyedHashJoinStrategyOptions) -> Self {
        self.keyed_hash_join_strategy_options = options;
        self
    }

    /// Set native grouped aggregate strategy execution policy.
    ///
    /// This policy is copied into native execution contexts so runtime
    /// aggregate strategy selection, exact-cardinality probes, and strategy
    /// metric rows share one threshold set. It is intentionally not exposed as
    /// a session/server knob until the expert surface is reviewed.
    pub fn with_aggregate_strategy_execution_options(mut self, options: AggregateStrategyExecutionOptions) -> Self {
        self.aggregate_strategy_execution_options = options;
        self
    }

    /// Set expert local-throughput source chunk planning policy.
    ///
    /// This policy is intentionally Rust-only; public source chunk knobs remain
    /// the bounded memory policy and source chunk planning mode.
    pub fn with_source_chunk_planning_options(mut self, options: SourceChunkPlanningOptions) -> Self {
        self.source_chunk_planning_options = options;
        self
    }

    /// Attach an adaptive-only memory-history store for adaptive admission/execution.
    ///
    /// Strict bounded admission does not read this store. Supplying it while strict mode is selected
    /// leaves strict reports deterministic and only makes the store available if the mode is later
    /// changed to adaptive.
    pub fn with_adaptive_memory_history(mut self, memory_history: Arc<AdaptiveMemoryHistory>) -> Self {
        self.adaptive_memory_history = Some(memory_history);
        self
    }

    /// Attach the durable GPU hash-join build cache required by durable build-boundary policy.
    pub fn with_hash_join_build_cache(mut self, cache: Arc<GpuHashJoinBuildCache>) -> Self {
        self.hash_join_build_cache = Some(cache);
        self
    }

    /// Attach a durable GPU local-Parquet source cache for repeated source scans.
    ///
    /// Bounded native execution charges cache-resident source bytes through the
    /// query-engine reservation ledger before source/operator admission.
    pub fn with_local_parquet_source_cache(mut self, cache: Arc<GpuLocalParquetSourceCache>) -> Self {
        self.local_parquet_source_cache = Some(cache);
        self
    }

    /// Source object-store profiles registered for native remote source
    /// fetches and local-cache miss fills.
    pub fn source_storage_profiles(&self) -> &[SourceStorageProfile] {
        &self.source_storage_profiles
    }

    /// Register source object-store profiles used by native remote source
    /// fetches and local-cache miss fills.
    pub fn with_source_storage_profiles(mut self, profiles: impl IntoIterator<Item = SourceStorageProfile>) -> Self {
        self.source_storage_profiles = profiles.into_iter().collect();
        self
    }

    /// Register one source object-store profile.
    pub fn with_source_storage_profile(mut self, profile: SourceStorageProfile) -> Self {
        self.source_storage_profiles.push(profile);
        self
    }

    /// Execute through an admitted multi-tenant query-service handle.
    ///
    /// The handle supplies the service-owned CUDA stream pool, source-fetch
    /// gates, deployment IO capability, and tenant device-budget slice used by
    /// native execution. The handle must remain unfinished until every plan
    /// node built from this config has completed execution; dropping the final
    /// handle releases the service admission slot.
    ///
    /// Source storage profiles are service-owned on this path. Register them
    /// on `QueryServiceConfig::source_storage_profiles`; profiles attached to
    /// this per-plan config are used only for standalone execution contexts.
    pub fn with_query_service_handle(mut self, handle: Arc<QueryHandle>) -> Self {
        self.memory_resource_policy = MemoryResourcePolicy::ExternalRuntime;
        self.memory_control_policy = handle.memory_control_policy();
        self.boundary_policy = handle.boundary_policy();
        self.source_scheduler_policy = handle.source_scheduler_policy();
        self.source_chunk_planning_mode = handle.source_chunk_planning_mode();
        self.source_fetch_config = handle.source_fetch_config();
        self.native_pipeline_execution_mode = handle.native_pipeline_execution_mode();
        self.stddev_samp_execution_mode = handle.stddev_samp_execution_mode();
        self.dynamic_filter_execution_options = handle.dynamic_filter_execution_options();
        self.keyed_hash_join_strategy_options = handle.keyed_hash_join_strategy_options();
        self.aggregate_strategy_execution_options = handle.aggregate_strategy_execution_options();
        self.source_chunk_planning_options = handle.source_chunk_planning_options();
        self.query_service_handle = Some(handle);
        self.query_admission_source = None;
        self.query_admission_ticket = None;
        self
    }

    pub(crate) fn with_query_admission_source(mut self, source: QueryAdmissionSource) -> Self {
        if self.query_service_handle.is_none() {
            self.query_admission_source = Some(source);
            self.query_admission_ticket = None;
        }
        self
    }

    pub(crate) fn with_plan_query_admission_ticket(mut self) -> Self {
        if self.query_service_handle.is_none()
            && self.query_admission_ticket.is_none()
            && let Some(source) = self.query_admission_source.clone()
        {
            self.query_admission_ticket = Some(QueryAdmissionTicket::new(source));
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn query_admission_ticket_id(&self) -> Option<u64> {
        self.query_admission_ticket.as_ref().map(|ticket| ticket.id())
    }

    pub(crate) fn query_admission_ticket(&self) -> Option<Arc<QueryAdmissionTicket>> {
        self.query_admission_ticket.as_ref().map(Arc::clone)
    }

    /// Returns whether execution is bound to a QueryService admission handle.
    pub(crate) fn has_query_service_handle(&self) -> bool {
        self.query_service_handle.is_some()
    }

    pub(crate) fn has_query_admission_authority(&self) -> bool {
        self.query_service_handle.is_some()
            || self.query_admission_source.is_some()
            || self.query_admission_ticket.is_some()
    }

    /// Memory-control policy copied into the lazily built native context.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    /// Boundary policy controlling source/build/probe/result residency.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    /// Source scheduling policy used by bounded native source reads.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.source_scheduler_policy
    }

    /// Source chunk planning mode used by bounded native source reads.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.source_chunk_planning_mode
    }

    /// Source IO resource policy used by standalone native execution.
    pub fn source_fetch_config(&self) -> &SourceFetchConfig {
        &self.source_fetch_config
    }

    /// Whether host Arrow output from native execution is charged to
    /// DataFusion's host-side memory pool.
    pub fn host_output_memory_pool_accounting(&self) -> bool {
        self.host_output_memory_pool_accounting
    }

    /// Strict bounded or adaptive pipeline execution mode.
    pub fn native_pipeline_execution_mode(&self) -> NativePipelineExecutionMode {
        self.native_pipeline_execution_mode
    }

    /// Execution contract for floating statistical aggregates.
    pub fn statistical_aggregate_execution_mode(&self) -> StddevSampExecutionMode {
        self.stddev_samp_execution_mode
    }

    /// Floating statistical aggregate contract returned through the legacy
    /// STDDEV_SAMP accessor.
    pub fn stddev_samp_execution_mode(&self) -> StddevSampExecutionMode {
        self.statistical_aggregate_execution_mode()
    }

    /// Dynamic-filter execution policy copied into optimized native plans.
    pub fn dynamic_filter_execution_options(&self) -> DynamicFilterExecutionOptions {
        self.dynamic_filter_execution_options
    }

    /// Keyed-hash join strategy policy copied into optimized native plans.
    pub fn keyed_hash_join_strategy_options(&self) -> KeyedHashJoinStrategyOptions {
        self.keyed_hash_join_strategy_options
    }

    /// Grouped aggregate strategy execution policy copied into native contexts.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.aggregate_strategy_execution_options
    }

    /// Expert local-throughput source chunk planning policy copied into native contexts.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.source_chunk_planning_options
    }

    /// Validate internal expert query policies before planning or execution.
    pub(crate) fn validate_query_policies(&self) -> crate::error::Result<()> {
        self.dynamic_filter_execution_options.validate().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.native_exec.config.validate",
                "invalid native dynamic-filter execution policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        self.keyed_hash_join_strategy_options.validate().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.native_exec.config.validate",
                "invalid native keyed hash-join strategy policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        self.aggregate_strategy_execution_options.validate().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.native_exec.config.validate",
                "invalid native aggregate strategy execution policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        self.source_chunk_planning_options.validate().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.native_exec.config.validate",
                "invalid native source chunk planning policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        Ok(())
    }

    /// Adaptive-only memory history store, if configured.
    pub fn adaptive_memory_history(&self) -> Option<Arc<AdaptiveMemoryHistory>> {
        self.adaptive_memory_history.as_ref().map(Arc::clone)
    }

    // DataFusion-to-native config boundary. Native context construction and validation errors are
    // converted here into adapter errors before any query execution starts.
    pub(crate) fn execution_context(&self) -> crate::error::Result<NativeCudfExecutionContext> {
        self.validate_query_policies()?;
        if let Some(handle) = &self.query_service_handle {
            return handle
                .acquire_context_with_cancellation(self.cancellation.clone())
                .map(NativeCudfExecutionContext::tenant_bound)
                .map(|mut context| {
                    if let Some(query_id) = self.query_id.clone() {
                        context = context.with_query_id(query_id);
                    }
                    if let Some(trace_id) = self.trace_id.clone() {
                        context = context.with_trace_id(trace_id);
                    }
                    context
                })
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.native_exec.context",
                        "failed to acquire query service native cudf execution context",
                        self.stamp_native_error_correlation(error),
                    )
                });
        }

        let mut builder = ExecutionContext::builder()
            .with_memory_policy(self.memory_resource_policy)
            .with_memory_control_policy(self.memory_control_policy)
            .with_boundary_policy(self.boundary_policy)
            .with_source_scheduler_policy(self.source_scheduler_policy)
            .with_source_chunk_planning_mode(self.source_chunk_planning_mode)
            .with_source_fetch_config(self.source_fetch_config.clone())
            .with_cancellation(self.cancellation.clone())
            .with_native_pipeline_execution_mode(self.native_pipeline_execution_mode)
            .with_statistical_aggregate_execution_mode(self.stddev_samp_execution_mode)
            .with_aggregate_strategy_execution_options(self.aggregate_strategy_execution_options)
            .with_source_chunk_planning_options(self.source_chunk_planning_options)
            .with_source_storage_profiles(self.source_storage_profiles.clone());
        if let Some(pool) = &self.service_stream_pool {
            builder = builder.with_service_stream_pool(pool.clone());
        }
        if let Some(pool_size) = self.service_stream_pool_size {
            builder = builder.with_service_stream_pool_size(pool_size);
        }
        if let Some(runtime) = &self.runtime {
            builder = builder.with_shared_runtime(Arc::clone(runtime));
        }
        if let Some(device_ordinal) = self.device_ordinal {
            builder = builder.with_device_ordinal(device_ordinal);
        }
        if let Some(cache) = &self.hash_join_build_cache {
            builder = builder.with_hash_join_build_cache(Arc::clone(cache));
        }
        if let Some(cache) = &self.local_parquet_source_cache {
            builder = builder.with_local_parquet_source_cache(Arc::clone(cache));
        }
        if let Some(memory_history) = &self.adaptive_memory_history {
            builder = builder.with_adaptive_memory_history(Arc::clone(memory_history));
        }
        if let Some(query_id) = &self.query_id {
            builder = builder.with_query_id(query_id.clone());
        }
        if let Some(trace_id) = &self.trace_id {
            builder = builder.with_trace_id(trace_id.clone());
        }
        builder.build().map(NativeCudfExecutionContext::standalone).map_err(|error| {
            Error::from_native(
                "datafusion_nexus.native_exec.context",
                "failed to build native cudf execution context",
                self.stamp_native_error_correlation(error),
            )
        })
    }

    pub(crate) async fn execution_context_async(
        &self,
        task_context: Arc<TaskContext>,
    ) -> crate::error::Result<NativeCudfExecutionContext> {
        self.validate_query_policies()?;
        if let Some(ticket) = &self.query_admission_ticket {
            return ticket
                .acquire_context(task_context, self.cancellation.clone())
                .await
                .map(NativeCudfExecutionContext::ticket_bound)
                .map(|mut context| {
                    if let Some(query_id) = self.query_id.clone() {
                        context = context.with_query_id(query_id);
                    }
                    if let Some(trace_id) = self.trace_id.clone() {
                        context = context.with_trace_id(trace_id);
                    }
                    context
                })
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.native_exec.context",
                        "failed to acquire backend query-service native cudf execution context",
                        self.stamp_native_error_correlation(error),
                    )
                });
        }
        self.execution_context()
    }

    fn stamp_native_error_correlation(
        &self,
        mut error: nexus_query_engine::QueryError,
    ) -> nexus_query_engine::QueryError {
        if let Some(query_id) = &self.query_id {
            error = error.with_query_id(query_id.clone());
        }
        if let Some(trace_id) = &self.trace_id {
            error = error.with_trace_id(trace_id.clone());
        }
        error
    }
}

fn detect_auto_sized_device_budget_bytes() -> u64 {
    auto_sized_device_budget_bytes_with_inventory(&CudfInventory)
}

fn auto_sized_device_budget_bytes_with_inventory(inventory: &impl GpuInventory) -> u64 {
    auto_sized_device_budget_bytes_for_device_with_inventory(inventory, cudf::DeviceId::DEFAULT.ordinal())
}

fn auto_sized_device_budget_bytes_for_device_with_inventory(
    inventory: &impl GpuInventory,
    device_ordinal: usize,
) -> u64 {
    inventory
        .facts(device_ordinal)
        .map(|facts| {
            device_budget_bytes_from_facts(
                &facts,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            )
        })
        .unwrap_or(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES)
}

fn auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(
    inventory: &impl GpuInventory,
    device_ordinal: usize,
    budget_fraction: f64,
) -> u64 {
    inventory
        .facts(device_ordinal)
        .map(|facts| {
            let headroom_budget = device_budget_bytes_from_facts(
                &facts,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            );
            let fraction_budget = fraction_budget_bytes(facts.total_bytes, budget_fraction)
                .filter(|budget| *budget > 0)
                .unwrap_or(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES);
            headroom_budget.min(fraction_budget)
        })
        .unwrap_or(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES)
}

fn fraction_budget_bytes(total_bytes: u64, budget_fraction: f64) -> Option<u64> {
    if !budget_fraction.is_finite() || !(0.0..=1.0).contains(&budget_fraction) || budget_fraction == 0.0 {
        return None;
    }
    let budget = (total_bytes as f64 * budget_fraction).floor();
    (budget > 0.0 && budget <= u64::MAX as f64).then_some(budget as u64)
}

#[cfg(test)]
mod tests {
    use nexus_query_engine::{
        DeploymentIoCapability, DeviceFacts, GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus,
        QueryError, QueryErrorCode, QueryService, QueryServiceConfig, QueryServiceTenantConfig, Result, TenantId,
    };

    use super::*;

    enum FakeInventory {
        Facts(DeviceFacts),
        Failure,
    }

    impl GpuInventory for FakeInventory {
        fn device_count(&self) -> Result<usize> {
            Ok(1)
        }

        fn facts(&self, _device_ordinal: usize) -> Result<DeviceFacts> {
            match self {
                Self::Facts(facts) => Ok(facts.clone()),
                Self::Failure => Err(QueryError::new(
                    QueryErrorCode::CudfTemporaryFailure,
                    "datafusion_nexus.test.gpu_inventory",
                    "simulated inventory failure",
                )),
            }
        }
    }

    fn device_facts(total_bytes: u64) -> DeviceFacts {
        DeviceFacts { device: 0, total_bytes, free_bytes: total_bytes / 2, sm_count: Some(16), utilization_pct: None }
    }

    #[test]
    fn auto_sized_budget_uses_total_memory_minus_headroom() {
        let total = 8 * 1024 * 1024 * 1024;
        let inventory = FakeInventory::Facts(device_facts(total));

        assert_eq!(
            auto_sized_device_budget_bytes_with_inventory(&inventory),
            total - NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES
        );
    }

    #[test]
    fn auto_sized_budget_falls_back_when_inventory_fails() {
        let inventory = FakeInventory::Failure;

        assert_eq!(
            auto_sized_device_budget_bytes_with_inventory(&inventory),
            NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES
        );
    }

    #[test]
    fn auto_sized_budget_for_device_uses_requested_ordinal() {
        struct OrdinalInventory;

        impl GpuInventory for OrdinalInventory {
            fn device_count(&self) -> Result<usize> {
                Ok(2)
            }

            fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
                Ok(device_facts(match device_ordinal {
                    0 => 8 * 1024 * 1024 * 1024,
                    1 => 16 * 1024 * 1024 * 1024,
                    other => panic!("unexpected device ordinal {other}"),
                }))
            }
        }

        assert_eq!(
            auto_sized_device_budget_bytes_for_device_with_inventory(&OrdinalInventory, 1),
            16 * 1024 * 1024 * 1024 - NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES
        );
    }

    #[test]
    fn auto_sized_budget_fraction_preserves_headroom_floor() {
        let total = 8 * 1024 * 1024 * 1024;
        let inventory = FakeInventory::Facts(device_facts(total));

        assert_eq!(
            auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(&inventory, 0, 1.0),
            total - NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES
        );
        assert_eq!(
            auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(&inventory, 0, 0.5),
            total / 2
        );
    }

    #[test]
    fn execution_context_carries_configured_correlation_ids() {
        let context = NativeCudfExecutionConfig::unbounded()
            .with_query_id("query-config")
            .with_trace_id("trace-config")
            .execution_context()
            .unwrap();

        assert_eq!(context.context().query_id().map(QueryId::as_str), Some("query-config"));
        assert_eq!(context.context().trace_id().map(TraceId::as_str), Some("trace-config"));
    }

    #[test]
    fn plan_query_admission_ticket_is_shared_by_config_clones() {
        let tenant_id = TenantId::new("tenant");
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let service = QueryService::new_admission_only(
            QueryServiceConfig::default(),
            vec![QueryServiceTenantConfig::new(tenant_id.clone(), 1024 * 1024)],
            deployment,
        )
        .unwrap();
        let source = QueryAdmissionSource::new(service, tenant_id);
        let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let config = base.clone().with_plan_query_admission_ticket();

        let cloned = config.clone().with_plan_query_admission_ticket();
        let separate_plan = base.with_plan_query_admission_ticket();

        assert_eq!(config.query_admission_ticket_id(), cloned.query_admission_ticket_id());
        assert_ne!(config.query_admission_ticket_id(), separate_plan.query_admission_ticket_id());
    }

    #[test]
    fn config_validation_errors_carry_configured_correlation_ids() {
        let error = NativeCudfExecutionConfig::unbounded()
            .with_query_id("query-config-error")
            .with_trace_id("trace-config-error")
            .with_source_chunk_planning_options(
                SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(0),
            )
            .validate_query_policies()
            .unwrap_err();

        assert_eq!(error.fact_value("query_id"), Some("query-config-error"));
        assert_eq!(error.fact_value("trace_id"), Some("trace-config-error"));
        let native = error
            .source_error()
            .and_then(|source| source.downcast_ref::<QueryError>())
            .expect("adapter error should retain native source error");
        assert_eq!(native.query_id().map(QueryId::as_str), Some("query-config-error"));
        assert_eq!(native.trace_id().map(TraceId::as_str), Some("trace-config-error"));
    }
}
