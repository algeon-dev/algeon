//! Native execution settings copied into rewritten DataFusion plan nodes.
//!
//! The config is cloneable and cheap to pass through planning. Backend
//! admission supplies the installed runtime, allocator, streams, and worker
//! executor when execution acquires the native query context.

use std::sync::Arc;

use datafusion::execution::context::TaskContext;
use query_engine::{
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    metrics::{QueryId, TraceId},
    mode::StatisticalAggregateExecutionMode,
    observability::{RuntimeObservationConfig, RuntimeObservationOwner, RuntimeTraceId},
    runtime::{
        AdmittedQueryContext, ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext, IoLifecycle,
        LakehouseCacheFillConfig, MemoryControlPolicy, SourceChunkPlanningMode, SourceSchedulerPolicy,
        SourceStorageProfile,
    },
};

use super::{
    admission::{
        QueryAdmissionFacts, QueryAdmissionSource, QueryAdmissionTicket, TicketAdmissionContext,
        query_admission_ticket_from_task_context,
    },
    metrics::{NativeExecutionMetricsCapture, NativeMetricsCapture},
};
use crate::error::Error;
#[cfg(feature = "cugraph")]
use crate::native_exec::graph::GraphExecutionServices;

/// Adapter-side native execution configuration.
///
/// This type holds the DataFusion-facing knobs for native cuDF execution. It does not eagerly create
/// cuDF runtime state. Production `ExecutionContext` values are acquired only
/// through backend admission when an accepted native plan begins execution.
#[derive(Debug, Clone)]
pub struct AlgeonNativeExecutionConfig {
    pub(super) memory_control_policy: MemoryControlPolicy,
    pub(super) boundary_policy: ExecutionBoundaryPolicy,
    pub(super) source_scheduler_policy: SourceSchedulerPolicy,
    pub(super) source_chunk_planning_mode: SourceChunkPlanningMode,
    pub(super) lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    pub(super) host_output_memory_pool_accounting: bool,
    pub(super) cancellation: ExecutionCancellation,
    pub(super) io_lifecycle: Option<IoLifecycle>,
    pub(super) statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    pub(super) dynamic_filter_execution_options: DynamicFilterExecutionOptions,
    pub(super) keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions,
    pub(super) aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    pub(super) source_chunk_planning_options: SourceChunkPlanningOptions,
    pub(super) source_storage_profiles: Vec<SourceStorageProfile>,
    pub(super) query_admission_source: Option<QueryAdmissionSource>,
    pub(super) query_admission_ticket: Option<Arc<QueryAdmissionTicket>>,
    pub(super) query_id: Option<QueryId>,
    pub(super) trace_id: Option<TraceId>,
    pub(super) plan_task_mapping_datafusion_shape: Option<String>,
    pub(super) metrics_capture: Option<NativeMetricsCapture>,
    pub(super) lakehouse_footer_pruning: bool,
    #[cfg(feature = "cugraph")]
    pub(super) graph_execution_services: GraphExecutionServices,
}

/// Native execution context plus any service leases that must outlive execution.
#[derive(Debug)]
pub(crate) struct AlgeonNativeExecutionContext {
    pub(super) context: ExecutionContext,
    pub(super) admission_facts: Option<QueryAdmissionFacts>,
    pub(super) record_admission_facts: bool,
    pub(super) _admission_context: Option<AdmittedQueryContext>,
    pub(super) ticket_context: Option<TicketAdmissionContext>,
}

mod admission;
mod context;
#[cfg(test)]
mod tests;

impl AlgeonNativeExecutionConfig {
    /// Fallback bounded device budget, in bytes, used only as an unadmitted
    /// planning template. A real execution context always replaces this with
    /// the immutable grant selected by the per-device ledger.
    pub const AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES: u64 = 256 * 1024 * 1024;
    /// Device-memory headroom, in bytes, reserved by automatic per-device
    /// backend profiles for libcudf and driver overhead.
    pub const AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES: u64 = 500 * 1024 * 1024;
    /// Default bounded source chunk byte target used by [`Self::bounded_default`].
    pub const DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES: u64 = 64 * 1024 * 1024;
    /// Default bounded source row-group target used by [`Self::bounded_default`].
    pub const DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
    /// Default bounded retry-attempt cap used by [`Self::bounded_default`].
    pub const DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS: usize = 3;
    /// Default bounded retry floor, in row groups, used by [`Self::bounded_default`].
    pub const DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

    /// Use the bounded planning defaults before a backend has admitted an
    /// attempt. This carries no selected device or capacity decision: admitted
    /// execution always receives the selected device's immutable grant.
    pub fn bounded_default() -> Self {
        Self::bounded(
            Self::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            Self::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            Self::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            Self::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            Self::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
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
            lakehouse_cache_fill_config: LakehouseCacheFillConfig::default(),
            host_output_memory_pool_accounting: false,
            cancellation: ExecutionCancellation::new(),
            io_lifecycle: None,
            statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::default(),
            dynamic_filter_execution_options: DynamicFilterExecutionOptions::default(),
            keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            source_storage_profiles: Vec::new(),
            query_admission_source: None,
            query_admission_ticket: None,
            query_id: None,
            trace_id: None,
            plan_task_mapping_datafusion_shape: None,
            metrics_capture: Some(NativeMetricsCapture::standalone()),
            lakehouse_footer_pruning: false,
            #[cfg(feature = "cugraph")]
            graph_execution_services: GraphExecutionServices::default(),
        }
    }

    /// Set bounded execution controls such as chunk sizing, reservations, and retry limits.
    ///
    /// Deployment resource profiles own capacity; backend admission replaces
    /// this template's budget with the selected immutable grant.
    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    /// Set source, build, probe, and result boundary choices for native execution.
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

    /// Set the lakehouse cache-fill resource policy projected into backend execution.
    pub fn with_lakehouse_cache_fill_config(mut self, config: LakehouseCacheFillConfig) -> Self {
        self.lakehouse_cache_fill_config = config;
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

    /// Attach a cooperative cancellation flag observed by native execution.
    #[cfg(any(test, feature = "server"))]
    pub(crate) fn with_cancellation(mut self, cancellation: ExecutionCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    #[cfg(feature = "server")]
    #[doc(hidden)]
    pub fn with_cancellation_for_internal_consumer(self, cancellation: ExecutionCancellation) -> Self {
        self.with_cancellation(cancellation)
    }

    /// Cooperative cancellation observed while an adapter-owned host boundary
    /// is staged before native context acquisition.
    pub(crate) fn cancellation(&self) -> &ExecutionCancellation {
        &self.cancellation
    }

    /// Attaches the server-generation gate used by file-backed native I/O.
    #[cfg(feature = "server")]
    pub(crate) fn with_io_lifecycle(mut self, lifecycle: IoLifecycle) -> Self {
        self.io_lifecycle = Some(lifecycle);
        self
    }

    #[cfg(feature = "server")]
    #[doc(hidden)]
    pub fn with_io_lifecycle_for_internal_consumer(self, lifecycle: IoLifecycle) -> Self {
        self.with_io_lifecycle(lifecycle)
    }

    /// Attach a server-side query metrics capture sink.
    #[cfg(feature = "server")]
    pub(crate) fn with_metrics_capture(mut self, capture: NativeMetricsCapture) -> Self {
        self.query_id = capture.query_id().map(QueryId::new);
        self.metrics_capture = Some(capture);
        self
    }

    #[cfg(feature = "server")]
    #[doc(hidden)]
    pub fn with_metrics_capture_for_internal_consumer(
        self,
        capture: crate::native_exec::ServerInternalNativeMetricsCapture,
    ) -> Self {
        self.with_metrics_capture(capture)
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

    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    pub(crate) fn plan_task_mapping_datafusion_shape(&self) -> Option<&str> {
        self.plan_task_mapping_datafusion_shape.as_deref()
    }

    pub(crate) fn new_metrics_capture(&self) -> Option<NativeExecutionMetricsCapture> {
        self.query_admission_ticket
            .as_ref()
            .and_then(|ticket| ticket.metrics_capture())
            .or_else(|| self.metrics_capture.as_ref().map(NativeMetricsCapture::new_execution))
    }

    pub fn runtime_observation_config(&self) -> RuntimeObservationConfig {
        self.metrics_capture
            .as_ref()
            .map_or_else(RuntimeObservationConfig::native_execution, NativeMetricsCapture::runtime_observation_config)
    }

    pub(crate) fn with_lakehouse_footer_pruning(mut self, enabled: bool) -> Self {
        self.lakehouse_footer_pruning = enabled;
        self
    }

    pub(crate) fn lakehouse_footer_pruning(&self) -> bool {
        self.lakehouse_footer_pruning
    }

    /// Select the execution contract for floating statistical aggregates.
    ///
    /// Defaults to [`StatisticalAggregateExecutionMode::GpuNativeTolerant`],
    /// whose results can differ from DataFusion in floating-point bits. Select
    /// [`StatisticalAggregateExecutionMode::DataFusionExactHostWelford`] when
    /// results must match DataFusion exactly.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StatisticalAggregateExecutionMode) -> Self {
        self.statistical_aggregate_execution_mode = mode;
        self
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
    /// admission, admission reports, and runtime observed-build strategy selection
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

    /// Source object-store profiles registered for native remote source
    /// fetches and local-cache miss fills.
    pub fn source_storage_profiles(&self) -> &[SourceStorageProfile] {
        &self.source_storage_profiles
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn graph_execution_services(&self) -> &GraphExecutionServices {
        &self.graph_execution_services
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn graph_bfs_singleflight_snapshot(&self) -> super::graph::GraphBfsSingleflightSnapshot {
        self.graph_execution_services.bfs_singleflight_snapshot()
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

    /// Memory-control policy copied into the lazily built native context.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    /// Boundary policy controlling source/build/probe/result residency.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    /// Maximum source bytes planned into one native bounded source chunk.
    pub fn max_source_chunk_bytes(&self) -> u64 {
        match self.memory_control_policy {
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
            MemoryControlPolicy::Unbounded => Self::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
        }
    }

    /// Maximum source row groups planned into one native bounded source chunk.
    pub fn max_row_groups_per_chunk(&self) -> usize {
        match self.memory_control_policy {
            MemoryControlPolicy::Bounded { max_row_groups_per_chunk, .. } => max_row_groups_per_chunk,
            MemoryControlPolicy::Unbounded => Self::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
        }
    }

    /// Maximum bounded retry attempts.
    pub fn max_retry_attempts(&self) -> usize {
        match self.memory_control_policy {
            MemoryControlPolicy::Bounded { max_retry_attempts, .. } => max_retry_attempts,
            MemoryControlPolicy::Unbounded => Self::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
        }
    }

    /// Maximum active native source reads allowed per bounded query.
    pub fn max_source_reads_in_flight(&self) -> usize {
        self.source_scheduler_policy.max_source_reads_in_flight()
    }

    /// Source scheduling policy used by bounded native source reads.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.source_scheduler_policy
    }

    /// Source chunk planning mode used by bounded native source reads.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.source_chunk_planning_mode
    }

    /// Lakehouse cache-fill resource policy projected into backend execution.
    pub fn lakehouse_cache_fill_config(&self) -> &LakehouseCacheFillConfig {
        &self.lakehouse_cache_fill_config
    }

    /// Whether host Arrow output from native execution is charged to
    /// DataFusion's host-side memory pool.
    pub fn host_output_memory_pool_accounting(&self) -> bool {
        self.host_output_memory_pool_accounting
    }

    /// Execution contract for floating statistical aggregates.
    pub fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.statistical_aggregate_execution_mode
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
                "algeon_datafusion.native_exec.config.validate",
                "invalid native dynamic-filter execution policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        self.keyed_hash_join_strategy_options.validate().map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_exec.config.validate",
                "invalid native keyed hash-join strategy policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        self.aggregate_strategy_execution_options.validate().map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_exec.config.validate",
                "invalid native aggregate strategy execution policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        self.source_chunk_planning_options.validate().map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_exec.config.validate",
                "invalid native source chunk planning policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        Ok(())
    }
}
