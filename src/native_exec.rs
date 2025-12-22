use std::{any::Any, sync::Arc};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use cudf_query_engine::{
    ExecutionBoundaryPolicy, ExecutionContext, MemoryControlPolicy, MemoryResourcePolicy, QueryMetrics, QueryOutput,
    QueryPlan, execute_query_plan,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        metrics::{
            BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricsSet, RecordOutput, Time,
        },
        stream::RecordBatchStreamAdapter,
    },
};
use futures::TryStreamExt;

use crate::error::{Error, ErrorCode};

pub const NATIVE_CUDF_DATAFUSION_METRIC_NAMES: &[&str] = &[
    "input_batches",
    "input_rows",
    "host_to_device_uploads",
    "host_to_device_uploaded_rows",
    "host_to_device_uploaded_bytes",
    "device_to_host_materializations",
    "device_to_host_materialized_rows",
    "device_to_host_materialized_bytes",
    "device_to_device_copies",
    "device_to_device_copied_bytes",
    "source_read_boundary_direct_device",
    "build_materialization_boundary_query_scoped",
    "probe_execution_boundary_reusable",
    "result_materialization_boundary_explicit",
    "reservation_requested_bytes",
    "reservation_granted_bytes",
    "reservation_denied_count",
    "peak_device_memory_bytes",
    "resident_dataframe_peak_bytes",
    "allocator_high_water_bytes",
    "source_chunks_planned",
    "source_chunks_attempted",
    "source_chunks_committed",
    "source_chunks_retried",
    "row_groups_planned",
    "row_groups_committed",
    "partial_aggregate_count",
    "partial_aggregate_rows",
    "final_merge_attempts",
    "final_merge_levels",
    "parquet_files",
    "parquet_bytes_known",
    "row_groups_selected",
    "read_columns",
    "retry_count",
    "retry_resource_exhausted_count",
    "retry_reservation_failure_count",
    "hash_join_build_count",
    "hash_join_cache_hits",
    "hash_join_cache_misses",
    "hash_join_hash_table_bytes",
    "parquet_read_time",
    "source_read_time",
    "operator_time",
    "chunk_operator_time",
    "final_merge_time",
    "final_sort_time",
    "materialization_time",
];

#[derive(Debug, Clone)]
pub struct NativeCudfExecutionConfig {
    runtime: Option<Arc<cudf::Runtime>>,
    device_ordinal: Option<usize>,
    memory_resource_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
    boundary_policy: ExecutionBoundaryPolicy,
}

impl NativeCudfExecutionConfig {
    pub fn unbounded() -> Self {
        Self {
            runtime: None,
            device_ordinal: None,
            memory_resource_policy: MemoryResourcePolicy::DefaultCuda,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
        }
    }

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
        }
    }

    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self
    }

    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    pub fn with_memory_resource_policy(mut self, memory_resource_policy: MemoryResourcePolicy) -> Self {
        self.memory_resource_policy = memory_resource_policy;
        self
    }

    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    fn execution_context(&self) -> crate::error::Result<ExecutionContext> {
        let mut builder = ExecutionContext::builder()
            .with_memory_policy(self.memory_resource_policy)
            .with_memory_control_policy(self.memory_control_policy)
            .with_boundary_policy(self.boundary_policy);
        if let Some(runtime) = &self.runtime {
            builder = builder.with_shared_runtime(Arc::clone(runtime));
        }
        if let Some(device_ordinal) = self.device_ordinal {
            builder = builder.with_device_ordinal(device_ordinal);
        }
        builder.build().map_err(|error| {
            Error::from_native(
                "datafusion_cudf.native_exec.context",
                "failed to build native cudf execution context",
                error,
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct NativeCudfExec {
    plan: QueryPlan,
    execution_config: NativeCudfExecutionConfig,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NativeCudfExec {
    pub fn try_new(plan: QueryPlan, execution_config: NativeCudfExecutionConfig) -> crate::error::Result<Self> {
        if !plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_exec.try_new",
                "native DataFusion execution requires a host-arrow result sink",
            )
            .with_fact("result_sink", "gpu_resident"));
        }

        Ok(Self {
            cache: conservative_properties(Arc::clone(plan.output_schema())),
            plan,
            execution_config,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }
}

impl DisplayAs for NativeCudfExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "NativeCudfExec: native_nodes={}, sink=host_arrow, memory_control_policy={}, boundary_policy={}, output_boundary=device_to_host",
                self.plan.node_count(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "sink=host_arrow")?;
                writeln!(f, "memory_control_policy={}", self.execution_config.memory_control_policy())?;
                writeln!(f, "boundary_policy={}", self.execution_config.boundary_policy())?;
                writeln!(f, "output_boundary=device_to_host")
            },
        }
    }
}

impl ExecutionPlan for NativeCudfExec {
    fn name(&self) -> &str {
        "NativeCudfExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "datafusion_cudf.native_exec.with_new_children",
                "native cudf execution expected no children",
            )
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(self)
    }

    fn execute(&self, partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_exec.execute",
                "native cudf execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let schema = self.schema();
        let plan = self.plan.clone();
        let execution_config = self.execution_config.clone();
        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let output = futures::stream::once(async move {
            let context = execution_config
                .execution_context()
                .map_err(|error| DataFusionError::from(mark_native_runtime_error(error, "runtime_context")))?;
            let result = execute_query_plan(&context, &plan).map_err(|error| {
                DataFusionError::from(native_query_runtime_error(
                    "datafusion_cudf.native_exec.execute",
                    "native cudf query execution failed",
                    error,
                ))
            })?;
            let (output, query_metrics) = result.into_parts();
            metrics.record_query_metrics(&query_metrics);
            let QueryOutput::HostArrow(batch) = output else {
                return Err(DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_cudf.native_exec.execute",
                        "native cudf execution produced a non-host result for DataFusion",
                    )
                    .with_fact("result_sink", "gpu_resident"),
                ));
            };
            Ok::<Option<RecordBatch>, DataFusionError>(Some(batch.record_output(&metrics.baseline)))
        })
        .try_filter_map(|batch| async move { Ok(batch) });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

fn conservative_properties(schema: SchemaRef) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        Partitioning::UnknownPartitioning(1),
        EmissionType::Final,
        Boundedness::Bounded,
    ))
}

fn native_query_runtime_error(
    operation: &'static str,
    message: impl Into<String>,
    source: cudf_query_engine::QueryError,
) -> Error {
    mark_native_runtime_error(Error::from_native(operation, message, source), "runtime_execution")
}

fn mark_native_runtime_error(error: Error, phase: &'static str) -> Error {
    error.with_fact("native_datafusion_phase", phase).with_fact("cpu_fallback_after_native_start", false)
}

#[derive(Debug, Clone)]
struct NativeCudfMetrics {
    baseline: BaselineMetrics,
    input_batches: Count,
    input_rows: Count,
    host_to_device_uploads: Count,
    host_to_device_uploaded_rows: Count,
    host_to_device_uploaded_bytes: Count,
    device_to_host_materializations: Count,
    device_to_host_materialized_rows: Count,
    device_to_host_materialized_bytes: Count,
    device_to_device_copies: Count,
    device_to_device_copied_bytes: Count,
    source_read_boundary_direct_device: Count,
    build_materialization_boundary_query_scoped: Count,
    probe_execution_boundary_reusable: Count,
    result_materialization_boundary_explicit: Count,
    reservation_requested_bytes: Count,
    reservation_granted_bytes: Count,
    reservation_denied_count: Count,
    peak_device_memory_bytes: Count,
    resident_dataframe_peak_bytes: Count,
    allocator_high_water_bytes: Count,
    source_chunks_planned: Count,
    source_chunks_attempted: Count,
    source_chunks_committed: Count,
    source_chunks_retried: Count,
    row_groups_planned: Count,
    row_groups_committed: Count,
    partial_aggregate_count: Count,
    partial_aggregate_rows: Count,
    final_merge_attempts: Count,
    final_merge_levels: Count,
    parquet_files: Count,
    parquet_bytes_known: Count,
    row_groups_selected: Count,
    read_columns: Count,
    retry_count: Count,
    retry_resource_exhausted_count: Count,
    retry_reservation_failure_count: Count,
    hash_join_build_count: Count,
    hash_join_cache_hits: Count,
    hash_join_cache_misses: Count,
    hash_join_hash_table_bytes: Count,
    parquet_read_time: Time,
    source_read_time: Time,
    operator_time: Time,
    chunk_operator_time: Time,
    final_merge_time: Time,
    final_sort_time: Time,
    materialization_time: Time,
}

impl NativeCudfMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            input_batches: summary_counter(metrics, "input_batches", partition),
            input_rows: summary_counter(metrics, "input_rows", partition),
            host_to_device_uploads: summary_counter(metrics, "host_to_device_uploads", partition),
            host_to_device_uploaded_rows: summary_counter(metrics, "host_to_device_uploaded_rows", partition),
            host_to_device_uploaded_bytes: summary_counter(metrics, "host_to_device_uploaded_bytes", partition),
            device_to_host_materializations: summary_counter(metrics, "device_to_host_materializations", partition),
            device_to_host_materialized_rows: summary_counter(metrics, "device_to_host_materialized_rows", partition),
            device_to_host_materialized_bytes: summary_counter(metrics, "device_to_host_materialized_bytes", partition),
            device_to_device_copies: summary_counter(metrics, "device_to_device_copies", partition),
            device_to_device_copied_bytes: summary_counter(metrics, "device_to_device_copied_bytes", partition),
            source_read_boundary_direct_device: summary_counter(
                metrics,
                "source_read_boundary_direct_device",
                partition,
            ),
            build_materialization_boundary_query_scoped: summary_counter(
                metrics,
                "build_materialization_boundary_query_scoped",
                partition,
            ),
            probe_execution_boundary_reusable: summary_counter(metrics, "probe_execution_boundary_reusable", partition),
            result_materialization_boundary_explicit: summary_counter(
                metrics,
                "result_materialization_boundary_explicit",
                partition,
            ),
            reservation_requested_bytes: summary_counter(metrics, "reservation_requested_bytes", partition),
            reservation_granted_bytes: summary_counter(metrics, "reservation_granted_bytes", partition),
            reservation_denied_count: summary_counter(metrics, "reservation_denied_count", partition),
            peak_device_memory_bytes: summary_counter(metrics, "peak_device_memory_bytes", partition),
            resident_dataframe_peak_bytes: summary_counter(metrics, "resident_dataframe_peak_bytes", partition),
            allocator_high_water_bytes: summary_counter(metrics, "allocator_high_water_bytes", partition),
            source_chunks_planned: summary_counter(metrics, "source_chunks_planned", partition),
            source_chunks_attempted: summary_counter(metrics, "source_chunks_attempted", partition),
            source_chunks_committed: summary_counter(metrics, "source_chunks_committed", partition),
            source_chunks_retried: summary_counter(metrics, "source_chunks_retried", partition),
            row_groups_planned: summary_counter(metrics, "row_groups_planned", partition),
            row_groups_committed: summary_counter(metrics, "row_groups_committed", partition),
            partial_aggregate_count: summary_counter(metrics, "partial_aggregate_count", partition),
            partial_aggregate_rows: summary_counter(metrics, "partial_aggregate_rows", partition),
            final_merge_attempts: summary_counter(metrics, "final_merge_attempts", partition),
            final_merge_levels: summary_counter(metrics, "final_merge_levels", partition),
            parquet_files: summary_counter(metrics, "parquet_files", partition),
            parquet_bytes_known: summary_counter(metrics, "parquet_bytes_known", partition),
            row_groups_selected: summary_counter(metrics, "row_groups_selected", partition),
            read_columns: summary_counter(metrics, "read_columns", partition),
            retry_count: summary_counter(metrics, "retry_count", partition),
            retry_resource_exhausted_count: summary_counter(metrics, "retry_resource_exhausted_count", partition),
            retry_reservation_failure_count: summary_counter(metrics, "retry_reservation_failure_count", partition),
            hash_join_build_count: summary_counter(metrics, "hash_join_build_count", partition),
            hash_join_cache_hits: summary_counter(metrics, "hash_join_cache_hits", partition),
            hash_join_cache_misses: summary_counter(metrics, "hash_join_cache_misses", partition),
            hash_join_hash_table_bytes: summary_counter(metrics, "hash_join_hash_table_bytes", partition),
            parquet_read_time: summary_timer(metrics, "parquet_read_time", partition),
            source_read_time: summary_timer(metrics, "source_read_time", partition),
            operator_time: summary_timer(metrics, "operator_time", partition),
            chunk_operator_time: summary_timer(metrics, "chunk_operator_time", partition),
            final_merge_time: summary_timer(metrics, "final_merge_time", partition),
            final_sort_time: summary_timer(metrics, "final_sort_time", partition),
            materialization_time: summary_timer(metrics, "materialization_time", partition),
        }
    }

    fn record_query_metrics(&self, metrics: &QueryMetrics) {
        self.input_batches.add(count(metrics.input_batches));
        self.input_rows.add(count(metrics.input_rows));
        self.host_to_device_uploads.add(count(metrics.boundary.host_to_device_uploads));
        self.host_to_device_uploaded_rows.add(count(metrics.boundary.host_to_device_uploaded_rows));
        self.host_to_device_uploaded_bytes.add(count(metrics.boundary.host_to_device_uploaded_bytes));
        self.device_to_host_materializations.add(count(metrics.boundary.device_to_host_materializations));
        self.device_to_host_materialized_rows.add(count(metrics.boundary.device_to_host_materialized_rows));
        self.device_to_host_materialized_bytes.add(count(metrics.boundary.device_to_host_materialized_bytes));
        self.device_to_device_copies.add(count(metrics.boundary.device_to_device_copies));
        self.device_to_device_copied_bytes.add(count(metrics.boundary.device_to_device_copied_bytes));
        self.record_boundary_policy(metrics.execution_boundary_policy);
        self.reservation_requested_bytes.add(count(metrics.reservation_requested_bytes));
        self.reservation_granted_bytes.add(count(metrics.reservation_granted_bytes));
        self.reservation_denied_count.add(count(metrics.reservation_denied_count));
        add_optional_count(&self.peak_device_memory_bytes, metrics.peak_device_memory_bytes);
        add_optional_count(&self.resident_dataframe_peak_bytes, metrics.resident_dataframe_peak_bytes);
        add_optional_count(&self.allocator_high_water_bytes, metrics.allocator_high_water_bytes);
        self.source_chunks_planned.add(count(metrics.source_chunks_planned));
        self.source_chunks_attempted.add(count(metrics.source_chunks_attempted));
        self.source_chunks_committed.add(count(metrics.source_chunks_committed));
        self.source_chunks_retried.add(count(metrics.source_chunks_retried));
        self.row_groups_planned.add(count(metrics.row_groups_planned));
        self.row_groups_committed.add(count(metrics.row_groups_committed));
        self.partial_aggregate_count.add(count(metrics.partial_aggregate_count));
        self.partial_aggregate_rows.add(count(metrics.partial_aggregate_rows));
        self.final_merge_attempts.add(count(metrics.final_merge_attempts));
        self.final_merge_levels.add(count(metrics.final_merge_levels));
        self.parquet_files.add(count(metrics.parquet_files));
        self.parquet_bytes_known.add(count(metrics.parquet_bytes_known));
        self.row_groups_selected.add(count(metrics.row_groups_selected));
        self.read_columns.add(count(metrics.read_columns));
        self.retry_count.add(count(metrics.retry_count));
        self.retry_resource_exhausted_count.add(count(metrics.retry_resource_exhausted_count));
        self.retry_reservation_failure_count.add(count(metrics.retry_reservation_failure_count));
        self.hash_join_build_count.add(count(metrics.hash_join_build_count));
        self.hash_join_cache_hits.add(count(metrics.hash_join_cache_hits));
        self.hash_join_cache_misses.add(count(metrics.hash_join_cache_misses));
        self.hash_join_hash_table_bytes.add(count(hash_join_hash_table_bytes(metrics)));
        self.parquet_read_time.add_duration(metrics.parquet_read_time);
        self.source_read_time.add_duration(metrics.source_read_time);
        self.operator_time.add_duration(metrics.operator_time);
        self.chunk_operator_time.add_duration(metrics.chunk_operator_time);
        self.final_merge_time.add_duration(metrics.final_merge_time);
        self.final_sort_time.add_duration(metrics.final_sort_time);
        self.materialization_time.add_duration(metrics.materialization_time);
    }

    fn record_boundary_policy(&self, policy: ExecutionBoundaryPolicy) {
        if matches!(policy.source_read(), cudf_query_engine::SourceReadBoundary::DirectDeviceLocalParquet) {
            self.source_read_boundary_direct_device.add(1);
        }
        if matches!(
            policy.build_materialization(),
            cudf_query_engine::BuildMaterializationBoundary::QueryScopedGpuHashJoin
        ) {
            self.build_materialization_boundary_query_scoped.add(1);
        }
        if matches!(policy.probe_execution(), cudf_query_engine::ProbeExecutionBoundary::ReusableGpuHashJoin) {
            self.probe_execution_boundary_reusable.add(1);
        }
        if matches!(policy.result_materialization(), cudf_query_engine::ResultMaterializationBoundary::ExplicitSink) {
            self.result_materialization_boundary_explicit.add(1);
        }
    }
}

fn count(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn hash_join_hash_table_bytes(metrics: &QueryMetrics) -> u64 {
    metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).map(|join| join.hash_table_bytes).sum()
}

fn add_optional_count(metric: &Count, value: Option<u64>) {
    if let Some(value) = value {
        metric.add(count(value));
    }
}

fn summary_counter(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Count {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).counter(name, partition)
}

fn summary_timer(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Time {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).subset_time(name, partition)
}
