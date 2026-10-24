//! Native execution settings copied into rewritten DataFusion plan nodes.
//!
//! The config is cloneable and cheap to pass through planning. Backend
//! admission supplies the installed runtime, allocator, streams, and worker
//! executor when execution acquires the native query context.

use std::sync::Arc;

use datafusion::execution::context::TaskContext;
use query_planning::{
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    correlation::{QueryId, TraceId},
    dynamic_filter::DynamicFilterExecutionOptions,
    execution_contracts::SourceChunkPlanningMode,
    mode::StatisticalAggregateExecutionMode,
    policy::PlanningPolicy,
};
use query_runtime::{
    observability::RuntimeObservationConfig,
    runtime::{
        AdmittedQueryContext, ExecutionContext, IoLifecycle, LakehouseCacheFillConfig, S3SourceStorageProfile,
        SourceSchedulerPolicy, SourceStorageProfile,
    },
};

pub use self::{
    chunking::SourceChunking,
    source_reads::{LakehouseByteCache, LakehouseByteCacheFill},
};
use super::{
    admission::{
        QueryAdmissionFacts, QueryAdmissionSource, QueryAdmissionTicket, TicketAdmissionContext,
        invocation_admission_ticket,
    },
    metrics::{NativeExecutionMetricsCapture, NativeMetricsCapture},
};
#[cfg(feature = "cugraph")]
use crate::cugraph_sql::GraphBfsSingleflight;
use crate::{
    error::{Error, Result},
    native::{ParquetReadStrategyConfig, S3_STORAGE_PROFILE_ID, ambient_s3_endpoint},
};

/// Adapter-side native execution configuration.
///
/// This type holds the DataFusion-facing settings for native cuDF execution.
/// It does not eagerly create cuDF runtime state. Production `ExecutionContext`
/// values are acquired only through backend admission when an accepted native
/// plan begins execution.
#[derive(Debug, Clone)]
pub struct AlgeonNativeExecutionConfig {
    pub(super) source_chunking: SourceChunking,
    pub(super) source_scheduler_policy: SourceSchedulerPolicy,
    pub(super) source_chunk_planning_mode: SourceChunkPlanningMode,
    pub(super) lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    pub(super) host_output_memory_pool_accounting: bool,
    pub(super) io_lifecycle: Option<IoLifecycle>,
    pub(super) statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    pub(super) planning_policy: PlanningPolicy,
    pub(super) aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    pub(super) source_chunk_planning_options: SourceChunkPlanningOptions,
    pub(super) s3_storage: Option<S3SourceStorageProfile>,
    pub(super) lakehouse_byte_cache: Option<LakehouseByteCache>,
    /// The ambient AWS endpoint, read once here, that scopes byte-cache
    /// entries when no S3 storage profile names an endpoint.
    pub(super) ambient_s3_endpoint: Option<String>,
    pub(super) query_admission_source: Option<QueryAdmissionSource>,
    pub(super) query_admission_ticket: Option<Arc<QueryAdmissionTicket>>,
    pub(super) query_id: Option<QueryId>,
    pub(super) trace_id: Option<TraceId>,
    pub(super) plan_task_mapping_datafusion_shape: Option<String>,
    pub(super) metrics_capture: Option<NativeMetricsCapture>,
    pub(super) lakehouse_footer_pruning: bool,
    /// Coalesces identical BFS requests across the queries of one backend.
    #[cfg(feature = "cugraph")]
    pub(super) bfs_singleflight: GraphBfsSingleflight,
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
mod chunking;
mod context;
mod source_reads;
#[cfg(test)]
mod tests;

impl Default for AlgeonNativeExecutionConfig {
    fn default() -> Self {
        Self::new(SourceChunking::default())
    }
}

impl AlgeonNativeExecutionConfig {
    /// Device-memory headroom, in bytes, reserved by automatic per-device
    /// backend profiles for libcudf and driver overhead.
    pub const AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES: u64 = 500 * 1024 * 1024;

    /// A configuration with `source_chunking` and default settings otherwise.
    /// It carries no device capacity: admitted execution always runs under the
    /// selected device's immutable grant.
    pub fn new(source_chunking: SourceChunking) -> Self {
        Self {
            source_chunking,
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            lakehouse_cache_fill_config: LakehouseCacheFillConfig::default(),
            host_output_memory_pool_accounting: false,
            io_lifecycle: None,
            statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::default(),
            planning_policy: PlanningPolicy::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            s3_storage: None,
            lakehouse_byte_cache: None,
            ambient_s3_endpoint: ambient_s3_endpoint(),
            query_admission_source: None,
            query_admission_ticket: None,
            query_id: None,
            trace_id: None,
            plan_task_mapping_datafusion_shape: None,
            metrics_capture: Some(NativeMetricsCapture::standalone()),
            lakehouse_footer_pruning: false,
            #[cfg(feature = "cugraph")]
            bfs_singleflight: GraphBfsSingleflight::default(),
        }
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

    /// Attaches the server generation coordinator used by file-backed native I/O.
    #[cfg(feature = "server")]
    pub(crate) fn with_io_lifecycle(mut self, lifecycle: IoLifecycle) -> Self {
        self.io_lifecycle = Some(lifecycle);
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

    /// Borrows the optional diagnostic query correlation id carried by this execution template.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Borrows the optional parent trace correlation id carried by this execution template.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    pub(crate) fn plan_task_mapping_datafusion_shape(&self) -> Option<&str> {
        self.plan_task_mapping_datafusion_shape.as_deref()
    }

    pub(crate) fn new_metrics_capture(&self) -> Option<NativeExecutionMetricsCapture> {
        self.query_admission_ticket
            .as_ref()
            .map(|ticket| ticket.metrics_capture())
            .or_else(|| self.metrics_capture.as_ref().map(NativeMetricsCapture::new_execution))
    }

    /// Returns the bounded runtime observation retention policy copied into native execution.
    pub fn runtime_observation_config(&self) -> RuntimeObservationConfig {
        self.metrics_capture
            .as_ref()
            .map_or_else(RuntimeObservationConfig::native_execution, NativeMetricsCapture::runtime_observation_config)
    }

    /// Prunes `s3://` Parquet row groups from their footers before native
    /// execution admits the plan. Off by default.
    pub fn with_lakehouse_footer_pruning(mut self, enabled: bool) -> Self {
        self.lakehouse_footer_pruning = enabled;
        self
    }

    /// Whether footer pruning runs before admission.
    pub fn lakehouse_footer_pruning(&self) -> bool {
        self.lakehouse_footer_pruning
    }

    /// Select the execution contract for floating statistical aggregates.
    ///
    /// Defaults to [`StatisticalAggregateExecutionMode::GpuNativeTolerant`],
    /// whose results can differ from DataFusion in floating-point bits. Select
    /// [`StatisticalAggregateExecutionMode::DataFusionExactHostWelford`] when
    /// results must be identical to DataFusion's. In that mode a candidate whose
    /// statistical aggregate input DataFusion reads through more than one
    /// partition stays on DataFusion, whose own result then depends on the
    /// order partitions deliver rows.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StatisticalAggregateExecutionMode) -> Self {
        self.statistical_aggregate_execution_mode = mode;
        self
    }

    /// Set native dynamic-filter execution policy.
    ///
    /// Route discovery, admission, and runtime payload handling read one
    /// threshold set through [`Self::planning_policy`]. It is intentionally
    /// not exposed as a session/server setting until the expert surface is reviewed.
    pub fn with_dynamic_filter_execution_options(mut self, options: DynamicFilterExecutionOptions) -> Self {
        self.planning_policy.dynamic_filter = options;
        self
    }

    /// Set native keyed-hash join strategy policy.
    ///
    /// Capability admission, admission reports, and runtime observed-build
    /// strategy selection read one threshold set through
    /// [`Self::planning_policy`]. It is intentionally not exposed as a
    /// session/server setting until the expert surface is reviewed.
    pub fn with_keyed_hash_join_strategy_options(mut self, options: KeyedHashJoinStrategyOptions) -> Self {
        self.planning_policy.keyed_hash_join = options;
        self
    }

    /// Set native grouped aggregate strategy execution policy.
    ///
    /// This policy is copied into native execution contexts so runtime
    /// aggregate strategy selection, exact-cardinality probes, and strategy
    /// metric rows share one threshold set. It is intentionally not exposed as
    /// a session/server setting until the expert surface is reviewed.
    pub fn with_aggregate_strategy_execution_options(mut self, options: AggregateStrategyExecutionOptions) -> Self {
        self.aggregate_strategy_execution_options = options;
        self
    }

    /// Set expert throughput source chunk planning policy.
    ///
    /// This policy is intentionally Rust-only; public source chunk settings remain
    /// the bounded memory policy and source chunk planning mode.
    pub fn with_source_chunk_planning_options(mut self, options: SourceChunkPlanningOptions) -> Self {
        self.source_chunk_planning_options = options;
        self
    }

    /// The S3 storage every `s3://` source reads through, when configured.
    pub fn s3_storage(&self) -> Option<&S3SourceStorageProfile> {
        self.s3_storage.as_ref()
    }

    /// The runtime storage profiles native remote reads and cache fills
    /// resolve: the S3 storage under [`S3_STORAGE_PROFILE_ID`], when set.
    pub(crate) fn source_storage_profiles(&self) -> Vec<SourceStorageProfile> {
        self.s3_storage.iter().map(|profile| SourceStorageProfile::s3(S3_STORAGE_PROFILE_ID, profile.clone())).collect()
    }

    /// How lowering resolves an `s3://` Parquet file.
    pub(crate) fn source_read_strategy(&self) -> ParquetReadStrategyConfig {
        ParquetReadStrategyConfig::new(
            self.lakehouse_byte_cache.as_ref(),
            self.s3_storage.as_ref(),
            self.ambient_s3_endpoint.as_deref(),
        )
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn bfs_singleflight(&self) -> &GraphBfsSingleflight {
        &self.bfs_singleflight
    }

    /// Reads every `s3://` source through `profile`: native reads, cache
    /// fills, footer pruning, and the DataFusion object store that sessions
    /// installed with this config resolve for `s3://` URLs.
    pub fn with_s3_storage(mut self, profile: S3SourceStorageProfile) -> Self {
        self.s3_storage = Some(profile);
        self
    }

    /// Keeps a local byte cache of `s3://` Parquet objects.
    pub fn with_lakehouse_byte_cache(mut self, cache: LakehouseByteCache) -> Self {
        self.lakehouse_byte_cache = Some(cache);
        self
    }

    /// Source chunking of native reads.
    pub fn source_chunking(&self) -> SourceChunking {
        self.source_chunking
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

    /// Join and dynamic-filter thresholds native plans are optimized,
    /// admitted, and executed under.
    pub fn planning_policy(&self) -> PlanningPolicy {
        self.planning_policy
    }

    /// Grouped aggregate strategy execution policy copied into native contexts.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.aggregate_strategy_execution_options
    }

    /// Expert throughput source chunk planning policy copied into native contexts.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.source_chunk_planning_options
    }

    /// Validate internal expert query policies before planning or execution.
    pub(crate) fn validate_query_policies(&self) -> Result<()> {
        self.planning_policy.dynamic_filter.validate().map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_exec.config.validate",
                "invalid native dynamic-filter execution policy",
                self.stamp_native_error_correlation(error),
            )
        })?;
        self.planning_policy.keyed_hash_join.validate().map_err(|error| {
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
