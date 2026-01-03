use cudf_query_engine::{ExecutionBoundaryPolicy, QueryMetrics};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, Time,
};

/// Stable public DataFusion metric-name contract exposed by native cuDF execution plans.
///
/// Every name in this list is registered as a DataFusion `MetricType::SUMMARY` metric by
/// `NativeCudfMetrics`. Timer names are DataFusion `Time` metrics stored in nanoseconds and
/// populated from native query `Duration` values. Keep this list, the `NativeCudfMetrics` fields,
/// and `NativeCudfMetrics::new` in exact sync because adapter consumers assert and parse these
/// names through DataFusion metrics.
pub const NATIVE_CUDF_DATAFUSION_METRIC_NAMES: &[&str] = &[
    "input_batches",
    "input_rows",
    "gpu_resident_input_handoffs",
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
    "build_materialization_boundary_durable",
    "probe_execution_boundary_reusable",
    "result_materialization_boundary_explicit",
    "reservation_requested_bytes",
    "reservation_granted_bytes",
    "reservation_denied_count",
    "peak_device_memory_bytes",
    "resident_dataframe_peak_bytes",
    "resident_dataframe_peak_bytes_observed",
    "allocator_high_water_bytes",
    "allocator_high_water_bytes_observed",
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
    "filtered_join_build_count",
    "hash_join_cache_hits",
    "hash_join_cache_misses",
    "hash_join_durable_cache_hits",
    "hash_join_durable_cache_misses",
    "hash_join_durable_cache_inserts",
    "hash_join_hash_table_bytes",
    "filtered_join_hash_table_bytes",
    "filtered_join_probe_state_bytes",
    "mixed_join_size_data_bytes",
    "join_gather_map_bytes",
    "parquet_read_time",
    "source_read_time",
    "operator_time",
    "chunk_operator_time",
    "final_merge_time",
    "final_sort_time",
    "materialization_time",
];

// Private translation layer from native `QueryMetrics` into DataFusion metrics. Field names must
// match `NATIVE_CUDF_DATAFUSION_METRIC_NAMES` and the registration names in `new`.
#[derive(Debug, Clone)]
pub(super) struct NativeCudfMetrics {
    pub(super) baseline: BaselineMetrics,
    input_batches: Count,
    input_rows: Count,
    gpu_resident_input_handoffs: Count,
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
    build_materialization_boundary_durable: Count,
    probe_execution_boundary_reusable: Count,
    result_materialization_boundary_explicit: Count,
    reservation_requested_bytes: Count,
    reservation_granted_bytes: Count,
    reservation_denied_count: Count,
    peak_device_memory_bytes: Count,
    resident_dataframe_peak_bytes: Count,
    resident_dataframe_peak_bytes_observed: Count,
    allocator_high_water_bytes: Count,
    allocator_high_water_bytes_observed: Count,
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
    filtered_join_build_count: Count,
    hash_join_cache_hits: Count,
    hash_join_cache_misses: Count,
    hash_join_durable_cache_hits: Count,
    hash_join_durable_cache_misses: Count,
    hash_join_durable_cache_inserts: Count,
    hash_join_hash_table_bytes: Count,
    filtered_join_hash_table_bytes: Count,
    filtered_join_probe_state_bytes: Count,
    mixed_join_size_data_bytes: Count,
    join_gather_map_bytes: Count,
    parquet_read_time: Time,
    source_read_time: Time,
    operator_time: Time,
    chunk_operator_time: Time,
    final_merge_time: Time,
    final_sort_time: Time,
    materialization_time: Time,
}

impl NativeCudfMetrics {
    pub(super) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            input_batches: summary_counter(metrics, "input_batches", partition),
            input_rows: summary_counter(metrics, "input_rows", partition),
            gpu_resident_input_handoffs: summary_counter(metrics, "gpu_resident_input_handoffs", partition),
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
            build_materialization_boundary_durable: summary_counter(
                metrics,
                "build_materialization_boundary_durable",
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
            resident_dataframe_peak_bytes_observed: summary_counter(
                metrics,
                "resident_dataframe_peak_bytes_observed",
                partition,
            ),
            allocator_high_water_bytes: summary_counter(metrics, "allocator_high_water_bytes", partition),
            allocator_high_water_bytes_observed: summary_counter(
                metrics,
                "allocator_high_water_bytes_observed",
                partition,
            ),
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
            filtered_join_build_count: summary_counter(metrics, "filtered_join_build_count", partition),
            hash_join_cache_hits: summary_counter(metrics, "hash_join_cache_hits", partition),
            hash_join_cache_misses: summary_counter(metrics, "hash_join_cache_misses", partition),
            hash_join_durable_cache_hits: summary_counter(metrics, "hash_join_durable_cache_hits", partition),
            hash_join_durable_cache_misses: summary_counter(metrics, "hash_join_durable_cache_misses", partition),
            hash_join_durable_cache_inserts: summary_counter(metrics, "hash_join_durable_cache_inserts", partition),
            hash_join_hash_table_bytes: summary_counter(metrics, "hash_join_hash_table_bytes", partition),
            filtered_join_hash_table_bytes: summary_counter(metrics, "filtered_join_hash_table_bytes", partition),
            filtered_join_probe_state_bytes: summary_counter(metrics, "filtered_join_probe_state_bytes", partition),
            mixed_join_size_data_bytes: summary_counter(metrics, "mixed_join_size_data_bytes", partition),
            join_gather_map_bytes: summary_counter(metrics, "join_gather_map_bytes", partition),
            parquet_read_time: summary_timer(metrics, "parquet_read_time", partition),
            source_read_time: summary_timer(metrics, "source_read_time", partition),
            operator_time: summary_timer(metrics, "operator_time", partition),
            chunk_operator_time: summary_timer(metrics, "chunk_operator_time", partition),
            final_merge_time: summary_timer(metrics, "final_merge_time", partition),
            final_sort_time: summary_timer(metrics, "final_sort_time", partition),
            materialization_time: summary_timer(metrics, "materialization_time", partition),
        }
    }

    pub(super) fn record_query_metrics(&self, metrics: &QueryMetrics) {
        self.input_batches.add(count(metrics.input_batches));
        self.input_rows.add(count(metrics.input_rows));
        self.gpu_resident_input_handoffs.add(count(metrics.boundary.gpu_resident_input_handoffs));
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
        add_optional_count(&self.peak_device_memory_bytes, None, metrics.peak_device_memory_bytes);
        add_optional_count(
            &self.resident_dataframe_peak_bytes,
            Some(&self.resident_dataframe_peak_bytes_observed),
            metrics.resident_dataframe_peak_bytes,
        );
        add_optional_count(
            &self.allocator_high_water_bytes,
            Some(&self.allocator_high_water_bytes_observed),
            metrics.allocator_high_water_bytes,
        );
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
        self.filtered_join_build_count.add(count(metrics.filtered_join_build_count));
        self.hash_join_cache_hits.add(count(metrics.hash_join_cache_hits));
        self.hash_join_cache_misses.add(count(metrics.hash_join_cache_misses));
        self.hash_join_durable_cache_hits.add(count(metrics.hash_join_durable_cache_hits));
        self.hash_join_durable_cache_misses.add(count(metrics.hash_join_durable_cache_misses));
        self.hash_join_durable_cache_inserts.add(count(metrics.hash_join_durable_cache_inserts));
        self.hash_join_hash_table_bytes.add(count(metrics.hash_join_hash_table_bytes));
        self.filtered_join_hash_table_bytes.add(count(metrics.filtered_join_hash_table_bytes));
        self.filtered_join_probe_state_bytes.add(count(metrics.filtered_join_probe_state_bytes));
        self.mixed_join_size_data_bytes.add(count(metrics.mixed_join_size_data_bytes));
        self.join_gather_map_bytes.add(count(join_gather_map_bytes(metrics)));
        self.parquet_read_time.add_duration(metrics.parquet_read_time);
        self.source_read_time.add_duration(metrics.source_read_time);
        self.operator_time.add_duration(metrics.operator_time);
        self.chunk_operator_time.add_duration(metrics.chunk_operator_time);
        self.final_merge_time.add_duration(metrics.final_merge_time);
        self.final_sort_time.add_duration(metrics.final_sort_time);
        self.materialization_time.add_duration(metrics.materialization_time);
    }

    fn record_boundary_policy(&self, policy: ExecutionBoundaryPolicy) {
        // These counters record which boundary policy was selected for this native query execution,
        // not rows or bytes crossing that boundary.
        if matches!(policy.source_read(), cudf_query_engine::SourceReadBoundary::DirectDeviceLocalParquet) {
            self.source_read_boundary_direct_device.add(1);
        }
        if matches!(
            policy.build_materialization(),
            cudf_query_engine::BuildMaterializationBoundary::QueryScopedGpuHashJoin
        ) {
            self.build_materialization_boundary_query_scoped.add(1);
        }
        if matches!(policy.build_materialization(), cudf_query_engine::BuildMaterializationBoundary::DurableGpuHashJoin)
        {
            self.build_materialization_boundary_durable.add(1);
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

fn join_gather_map_bytes(metrics: &QueryMetrics) -> u64 {
    metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).map(|join| join.gather_map_bytes).sum()
}

fn add_optional_count(metric: &Count, observed: Option<&Count>, value: Option<u64>) {
    if let Some(value) = value {
        // `*_observed` distinguishes a real `Some(0)` sample from no observation at all.
        if let Some(observed) = observed {
            observed.add(1);
        }
        metric.add(count(value));
    }
}

fn summary_counter(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Count {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).counter(name, partition)
}

fn summary_timer(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Time {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).subset_time(name, partition)
}
