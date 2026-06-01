//! Stable DataFusion metric names populated by native execution wrappers.
//!
//! The public name list is the contract consumed by tests, reports, and users
//! who inspect DataFusion metrics after native execution.

use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, Time,
};
use nexus_query_engine::{ExecutionBoundaryPolicy, JoinOperatorMetrics, QueryMetrics};

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
    "reservation_live_peak_bytes",
    "reservation_denied_count",
    "max_task_reservation_bytes",
    "max_task_observed_peak_bytes",
    "max_task_observed_peak_bytes_observed",
    "max_task_retained_bytes",
    "max_task_retained_bytes_observed",
    "native_graph_runtime_rows",
    "native_task_operator_rows",
    "native_task_runtime_rows",
    "native_dynamic_filter_runtime_rows",
    "native_source_read_runtime_rows",
    "peak_device_memory_bytes",
    "resident_dataframe_peak_bytes",
    "resident_dataframe_peak_bytes_observed",
    "allocator_high_water_bytes",
    "allocator_high_water_bytes_observed",
    "source_chunks_planned",
    "source_chunks_attempted",
    "source_chunks_committed",
    "source_chunks_retried",
    "source_read_prefetches_started",
    "source_read_prefetches_completed",
    "source_read_prefetches_skipped",
    "source_read_prefetches_abandoned",
    "source_cache_hits",
    "source_cache_misses",
    "source_cache_inserts",
    "source_cache_insert_skipped",
    "source_cache_retained_bytes",
    "row_groups_planned",
    "row_groups_committed",
    "partial_aggregate_count",
    "partial_aggregate_input_rows",
    "partial_aggregate_rows",
    "partial_aggregate_retained_bytes",
    "final_merge_attempts",
    "final_merge_levels",
    "final_merge_max_fan_in",
    "compiled_pipeline_attempts",
    "compiled_pipeline_tasks_executed",
    "compiled_pipeline_host_arrow_sinks",
    "parquet_files",
    "parquet_bytes_known",
    "row_groups_selected",
    "read_columns",
    "retry_count",
    "retry_resource_exhausted_count",
    "retry_reservation_failure_count",
    "adaptive_retry_count",
    "adaptive_spill_count",
    "adaptive_downgrade_count",
    "adaptive_memory_history_lookup_count",
    "adaptive_memory_history_hit_count",
    "adaptive_memory_history_miss_count",
    "dynamic_filter_produced_count",
    "dynamic_filter_skipped_count",
    "dynamic_filter_applied_count",
    "dynamic_filter_row_groups_pruned",
    "dynamic_filter_source_bytes_pruned",
    "dynamic_filter_reader_predicate_rows_before",
    "dynamic_filter_reader_predicate_rows_after",
    "dynamic_filter_reader_predicate_rows_filtered",
    "dynamic_filter_post_decode_rows_before",
    "dynamic_filter_post_decode_rows_after",
    "dynamic_filter_post_decode_rows_filtered",
    "dynamic_filter_source_post_decode_rows_filtered",
    "dynamic_filter_filter_post_decode_rows_filtered",
    "dynamic_filter_join_probe_post_decode_rows_filtered",
    "hash_join_build_count",
    "filtered_join_build_count",
    "one_shot_join_count",
    "hash_join_cache_hits",
    "hash_join_cache_misses",
    "hash_join_durable_cache_hits",
    "hash_join_durable_cache_misses",
    "hash_join_durable_cache_inserts",
    "hash_join_hash_table_bytes",
    "filtered_join_hash_table_bytes",
    "filtered_join_probe_state_bytes",
    "mixed_join_size_data_bytes",
    "one_shot_join_index_bytes",
    "one_shot_join_materialization_bytes",
    "one_shot_join_output_bytes",
    "join_gather_map_bytes",
    "native_aggregate_strategy_rows",
    "native_join_strategy_rows",
    "source_metadata_load_time",
    "source_read_queue_wait_time",
    "source_cache_lookup_time",
    "source_fetch_bytes",
    "source_fetch_count",
    "host_fetch_budget_wait_time",
    "source_fetch_concurrency_wait_time",
    "source_fetch_time",
    "source_decode_time",
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
pub(crate) struct NativeCudfMetrics {
    metrics_set: ExecutionPlanMetricsSet,
    pub(crate) baseline: BaselineMetrics,
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
    reservation_live_peak_bytes: Count,
    reservation_denied_count: Count,
    max_task_reservation_bytes: Count,
    max_task_observed_peak_bytes: Count,
    max_task_observed_peak_bytes_observed: Count,
    max_task_retained_bytes: Count,
    max_task_retained_bytes_observed: Count,
    peak_device_memory_bytes: Count,
    resident_dataframe_peak_bytes: Count,
    resident_dataframe_peak_bytes_observed: Count,
    allocator_high_water_bytes: Count,
    allocator_high_water_bytes_observed: Count,
    source_chunks_planned: Count,
    source_chunks_attempted: Count,
    source_chunks_committed: Count,
    source_chunks_retried: Count,
    source_read_prefetches_started: Count,
    source_read_prefetches_completed: Count,
    source_read_prefetches_skipped: Count,
    source_read_prefetches_abandoned: Count,
    source_cache_hits: Count,
    source_cache_misses: Count,
    source_cache_inserts: Count,
    source_cache_insert_skipped: Count,
    source_cache_retained_bytes: Count,
    row_groups_planned: Count,
    row_groups_committed: Count,
    partial_aggregate_count: Count,
    partial_aggregate_input_rows: Count,
    partial_aggregate_rows: Count,
    partial_aggregate_retained_bytes: Count,
    final_merge_attempts: Count,
    final_merge_levels: Count,
    final_merge_max_fan_in: Count,
    compiled_pipeline_attempts: Count,
    compiled_pipeline_tasks_executed: Count,
    compiled_pipeline_host_arrow_sinks: Count,
    parquet_files: Count,
    parquet_bytes_known: Count,
    row_groups_selected: Count,
    read_columns: Count,
    retry_count: Count,
    retry_resource_exhausted_count: Count,
    retry_reservation_failure_count: Count,
    adaptive_retry_count: Count,
    adaptive_spill_count: Count,
    adaptive_downgrade_count: Count,
    adaptive_memory_history_lookup_count: Count,
    adaptive_memory_history_hit_count: Count,
    adaptive_memory_history_miss_count: Count,
    dynamic_filter_produced_count: Count,
    dynamic_filter_skipped_count: Count,
    dynamic_filter_applied_count: Count,
    dynamic_filter_row_groups_pruned: Count,
    dynamic_filter_source_bytes_pruned: Count,
    dynamic_filter_reader_predicate_rows_before: Count,
    dynamic_filter_reader_predicate_rows_after: Count,
    dynamic_filter_reader_predicate_rows_filtered: Count,
    dynamic_filter_post_decode_rows_before: Count,
    dynamic_filter_post_decode_rows_after: Count,
    dynamic_filter_post_decode_rows_filtered: Count,
    dynamic_filter_source_post_decode_rows_filtered: Count,
    dynamic_filter_filter_post_decode_rows_filtered: Count,
    dynamic_filter_join_probe_post_decode_rows_filtered: Count,
    hash_join_build_count: Count,
    filtered_join_build_count: Count,
    one_shot_join_count: Count,
    hash_join_cache_hits: Count,
    hash_join_cache_misses: Count,
    hash_join_durable_cache_hits: Count,
    hash_join_durable_cache_misses: Count,
    hash_join_durable_cache_inserts: Count,
    hash_join_hash_table_bytes: Count,
    filtered_join_hash_table_bytes: Count,
    filtered_join_probe_state_bytes: Count,
    mixed_join_size_data_bytes: Count,
    one_shot_join_index_bytes: Count,
    one_shot_join_materialization_bytes: Count,
    one_shot_join_output_bytes: Count,
    join_gather_map_bytes: Count,
    native_aggregate_strategy_rows: Count,
    source_metadata_load_time: Time,
    source_read_queue_wait_time: Time,
    source_cache_lookup_time: Time,
    source_fetch_bytes: Count,
    source_fetch_count: Count,
    host_fetch_budget_wait_time: Time,
    source_fetch_concurrency_wait_time: Time,
    source_fetch_time: Time,
    source_decode_time: Time,
    parquet_read_time: Time,
    source_read_time: Time,
    operator_time: Time,
    chunk_operator_time: Time,
    final_merge_time: Time,
    final_sort_time: Time,
    materialization_time: Time,
}

impl NativeCudfMetrics {
    pub(crate) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let native_aggregate_strategy_rows = summary_counter(metrics, "native_aggregate_strategy_rows", partition);
        let _ = summary_counter(metrics, "native_join_strategy_rows", partition);
        let _ = summary_counter(metrics, "native_graph_runtime_rows", partition);
        let _ = summary_counter(metrics, "native_task_operator_rows", partition);
        let _ = summary_counter(metrics, "native_task_runtime_rows", partition);
        let _ = summary_counter(metrics, "native_dynamic_filter_runtime_rows", partition);
        let _ = summary_counter(metrics, "native_source_read_runtime_rows", partition);
        Self {
            metrics_set: metrics.clone(),
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
            reservation_live_peak_bytes: summary_counter(metrics, "reservation_live_peak_bytes", partition),
            reservation_denied_count: summary_counter(metrics, "reservation_denied_count", partition),
            max_task_reservation_bytes: summary_counter(metrics, "max_task_reservation_bytes", partition),
            max_task_observed_peak_bytes: summary_counter(metrics, "max_task_observed_peak_bytes", partition),
            max_task_observed_peak_bytes_observed: summary_counter(
                metrics,
                "max_task_observed_peak_bytes_observed",
                partition,
            ),
            max_task_retained_bytes: summary_counter(metrics, "max_task_retained_bytes", partition),
            max_task_retained_bytes_observed: summary_counter(metrics, "max_task_retained_bytes_observed", partition),
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
            source_read_prefetches_started: summary_counter(metrics, "source_read_prefetches_started", partition),
            source_read_prefetches_completed: summary_counter(metrics, "source_read_prefetches_completed", partition),
            source_read_prefetches_skipped: summary_counter(metrics, "source_read_prefetches_skipped", partition),
            source_read_prefetches_abandoned: summary_counter(metrics, "source_read_prefetches_abandoned", partition),
            source_cache_hits: summary_counter(metrics, "source_cache_hits", partition),
            source_cache_misses: summary_counter(metrics, "source_cache_misses", partition),
            source_cache_inserts: summary_counter(metrics, "source_cache_inserts", partition),
            source_cache_insert_skipped: summary_counter(metrics, "source_cache_insert_skipped", partition),
            source_cache_retained_bytes: summary_counter(metrics, "source_cache_retained_bytes", partition),
            row_groups_planned: summary_counter(metrics, "row_groups_planned", partition),
            row_groups_committed: summary_counter(metrics, "row_groups_committed", partition),
            partial_aggregate_count: summary_counter(metrics, "partial_aggregate_count", partition),
            partial_aggregate_input_rows: summary_counter(metrics, "partial_aggregate_input_rows", partition),
            partial_aggregate_rows: summary_counter(metrics, "partial_aggregate_rows", partition),
            partial_aggregate_retained_bytes: summary_counter(metrics, "partial_aggregate_retained_bytes", partition),
            final_merge_attempts: summary_counter(metrics, "final_merge_attempts", partition),
            final_merge_levels: summary_counter(metrics, "final_merge_levels", partition),
            final_merge_max_fan_in: summary_counter(metrics, "final_merge_max_fan_in", partition),
            compiled_pipeline_attempts: summary_counter(metrics, "compiled_pipeline_attempts", partition),
            compiled_pipeline_tasks_executed: summary_counter(metrics, "compiled_pipeline_tasks_executed", partition),
            compiled_pipeline_host_arrow_sinks: summary_counter(
                metrics,
                "compiled_pipeline_host_arrow_sinks",
                partition,
            ),
            parquet_files: summary_counter(metrics, "parquet_files", partition),
            parquet_bytes_known: summary_counter(metrics, "parquet_bytes_known", partition),
            row_groups_selected: summary_counter(metrics, "row_groups_selected", partition),
            read_columns: summary_counter(metrics, "read_columns", partition),
            retry_count: summary_counter(metrics, "retry_count", partition),
            retry_resource_exhausted_count: summary_counter(metrics, "retry_resource_exhausted_count", partition),
            retry_reservation_failure_count: summary_counter(metrics, "retry_reservation_failure_count", partition),
            adaptive_retry_count: summary_counter(metrics, "adaptive_retry_count", partition),
            adaptive_spill_count: summary_counter(metrics, "adaptive_spill_count", partition),
            adaptive_downgrade_count: summary_counter(metrics, "adaptive_downgrade_count", partition),
            adaptive_memory_history_lookup_count: summary_counter(
                metrics,
                "adaptive_memory_history_lookup_count",
                partition,
            ),
            adaptive_memory_history_hit_count: summary_counter(metrics, "adaptive_memory_history_hit_count", partition),
            adaptive_memory_history_miss_count: summary_counter(
                metrics,
                "adaptive_memory_history_miss_count",
                partition,
            ),
            dynamic_filter_produced_count: summary_counter(metrics, "dynamic_filter_produced_count", partition),
            dynamic_filter_skipped_count: summary_counter(metrics, "dynamic_filter_skipped_count", partition),
            dynamic_filter_applied_count: summary_counter(metrics, "dynamic_filter_applied_count", partition),
            dynamic_filter_row_groups_pruned: summary_counter(metrics, "dynamic_filter_row_groups_pruned", partition),
            dynamic_filter_source_bytes_pruned: summary_counter(
                metrics,
                "dynamic_filter_source_bytes_pruned",
                partition,
            ),
            dynamic_filter_reader_predicate_rows_before: summary_counter(
                metrics,
                "dynamic_filter_reader_predicate_rows_before",
                partition,
            ),
            dynamic_filter_reader_predicate_rows_after: summary_counter(
                metrics,
                "dynamic_filter_reader_predicate_rows_after",
                partition,
            ),
            dynamic_filter_reader_predicate_rows_filtered: summary_counter(
                metrics,
                "dynamic_filter_reader_predicate_rows_filtered",
                partition,
            ),
            dynamic_filter_post_decode_rows_before: summary_counter(
                metrics,
                "dynamic_filter_post_decode_rows_before",
                partition,
            ),
            dynamic_filter_post_decode_rows_after: summary_counter(
                metrics,
                "dynamic_filter_post_decode_rows_after",
                partition,
            ),
            dynamic_filter_post_decode_rows_filtered: summary_counter(
                metrics,
                "dynamic_filter_post_decode_rows_filtered",
                partition,
            ),
            dynamic_filter_source_post_decode_rows_filtered: summary_counter(
                metrics,
                "dynamic_filter_source_post_decode_rows_filtered",
                partition,
            ),
            dynamic_filter_filter_post_decode_rows_filtered: summary_counter(
                metrics,
                "dynamic_filter_filter_post_decode_rows_filtered",
                partition,
            ),
            dynamic_filter_join_probe_post_decode_rows_filtered: summary_counter(
                metrics,
                "dynamic_filter_join_probe_post_decode_rows_filtered",
                partition,
            ),
            hash_join_build_count: summary_counter(metrics, "hash_join_build_count", partition),
            filtered_join_build_count: summary_counter(metrics, "filtered_join_build_count", partition),
            one_shot_join_count: summary_counter(metrics, "one_shot_join_count", partition),
            hash_join_cache_hits: summary_counter(metrics, "hash_join_cache_hits", partition),
            hash_join_cache_misses: summary_counter(metrics, "hash_join_cache_misses", partition),
            hash_join_durable_cache_hits: summary_counter(metrics, "hash_join_durable_cache_hits", partition),
            hash_join_durable_cache_misses: summary_counter(metrics, "hash_join_durable_cache_misses", partition),
            hash_join_durable_cache_inserts: summary_counter(metrics, "hash_join_durable_cache_inserts", partition),
            hash_join_hash_table_bytes: summary_counter(metrics, "hash_join_hash_table_bytes", partition),
            filtered_join_hash_table_bytes: summary_counter(metrics, "filtered_join_hash_table_bytes", partition),
            filtered_join_probe_state_bytes: summary_counter(metrics, "filtered_join_probe_state_bytes", partition),
            mixed_join_size_data_bytes: summary_counter(metrics, "mixed_join_size_data_bytes", partition),
            one_shot_join_index_bytes: summary_counter(metrics, "one_shot_join_index_bytes", partition),
            one_shot_join_materialization_bytes: summary_counter(
                metrics,
                "one_shot_join_materialization_bytes",
                partition,
            ),
            one_shot_join_output_bytes: summary_counter(metrics, "one_shot_join_output_bytes", partition),
            join_gather_map_bytes: summary_counter(metrics, "join_gather_map_bytes", partition),
            native_aggregate_strategy_rows,
            source_metadata_load_time: summary_timer(metrics, "source_metadata_load_time", partition),
            source_read_queue_wait_time: summary_timer(metrics, "source_read_queue_wait_time", partition),
            source_cache_lookup_time: summary_timer(metrics, "source_cache_lookup_time", partition),
            source_fetch_bytes: summary_counter(metrics, "source_fetch_bytes", partition),
            source_fetch_count: summary_counter(metrics, "source_fetch_count", partition),
            host_fetch_budget_wait_time: summary_timer(metrics, "host_fetch_budget_wait_time", partition),
            source_fetch_concurrency_wait_time: summary_timer(metrics, "source_fetch_concurrency_wait_time", partition),
            source_fetch_time: summary_timer(metrics, "source_fetch_time", partition),
            source_decode_time: summary_timer(metrics, "source_decode_time", partition),
            parquet_read_time: summary_timer(metrics, "parquet_read_time", partition),
            source_read_time: summary_timer(metrics, "source_read_time", partition),
            operator_time: summary_timer(metrics, "operator_time", partition),
            chunk_operator_time: summary_timer(metrics, "chunk_operator_time", partition),
            final_merge_time: summary_timer(metrics, "final_merge_time", partition),
            final_sort_time: summary_timer(metrics, "final_sort_time", partition),
            materialization_time: summary_timer(metrics, "materialization_time", partition),
        }
    }

    pub(crate) fn record_query_metrics(&self, metrics: &QueryMetrics) {
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
        self.reservation_live_peak_bytes.add(count(metrics.reservation_live_peak_bytes));
        self.reservation_denied_count.add(count(metrics.reservation_denied_count));
        self.max_task_reservation_bytes.add(count(metrics.max_task_reservation_bytes));
        add_optional_count(
            &self.max_task_observed_peak_bytes,
            Some(&self.max_task_observed_peak_bytes_observed),
            metrics.max_task_observed_peak_bytes,
        );
        add_optional_count(
            &self.max_task_retained_bytes,
            Some(&self.max_task_retained_bytes_observed),
            metrics.max_task_retained_bytes,
        );
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
        self.source_read_prefetches_started.add(count(metrics.source_read_prefetches_started));
        self.source_read_prefetches_completed.add(count(metrics.source_read_prefetches_completed));
        self.source_read_prefetches_skipped.add(count(metrics.source_read_prefetches_skipped));
        self.source_read_prefetches_abandoned.add(count(metrics.source_read_prefetches_abandoned));
        self.source_cache_hits.add(count(metrics.source_cache_hits));
        self.source_cache_misses.add(count(metrics.source_cache_misses));
        self.source_cache_inserts.add(count(metrics.source_cache_inserts));
        self.source_cache_insert_skipped.add(count(metrics.source_cache_insert_skipped));
        self.source_cache_retained_bytes.add(count(metrics.source_cache_retained_bytes));
        self.row_groups_planned.add(count(metrics.row_groups_planned));
        self.row_groups_committed.add(count(metrics.row_groups_committed));
        self.partial_aggregate_count.add(count(metrics.partial_aggregate_count));
        self.partial_aggregate_input_rows.add(count(metrics.partial_aggregate_input_rows));
        self.partial_aggregate_rows.add(count(metrics.partial_aggregate_rows));
        self.partial_aggregate_retained_bytes.add(count(metrics.partial_aggregate_retained_bytes));
        self.final_merge_attempts.add(count(metrics.final_merge_attempts));
        self.final_merge_levels.add(count(metrics.final_merge_levels));
        self.final_merge_max_fan_in.add(count(metrics.final_merge_max_fan_in));
        self.compiled_pipeline_attempts.add(count(metrics.compiled_pipeline_attempts));
        self.compiled_pipeline_tasks_executed.add(count(metrics.compiled_pipeline_tasks_executed));
        self.compiled_pipeline_host_arrow_sinks.add(count(metrics.compiled_pipeline_host_arrow_sinks));
        self.parquet_files.add(count(metrics.parquet_files));
        self.parquet_bytes_known.add(count(metrics.parquet_bytes_known));
        self.row_groups_selected.add(count(metrics.row_groups_selected));
        self.read_columns.add(count(metrics.read_columns));
        self.retry_count.add(count(metrics.retry_count));
        self.retry_resource_exhausted_count.add(count(metrics.retry_resource_exhausted_count));
        self.retry_reservation_failure_count.add(count(metrics.retry_reservation_failure_count));
        self.adaptive_retry_count.add(count(metrics.adaptive_retry_count));
        self.adaptive_spill_count.add(count(metrics.adaptive_spill_count));
        self.adaptive_downgrade_count.add(count(metrics.adaptive_downgrade_count));
        self.adaptive_memory_history_lookup_count.add(count(metrics.adaptive_memory_history_lookup_count));
        self.adaptive_memory_history_hit_count.add(count(metrics.adaptive_memory_history_hit_count));
        self.adaptive_memory_history_miss_count.add(count(metrics.adaptive_memory_history_miss_count));
        self.dynamic_filter_produced_count.add(count(metrics.dynamic_filter_produced_count));
        self.dynamic_filter_skipped_count.add(count(metrics.dynamic_filter_skipped_count));
        self.dynamic_filter_applied_count.add(count(metrics.dynamic_filter_applied_count));
        self.dynamic_filter_row_groups_pruned.add(count(metrics.dynamic_filter_row_groups_pruned));
        self.dynamic_filter_source_bytes_pruned.add(count(metrics.dynamic_filter_source_bytes_pruned));
        self.dynamic_filter_reader_predicate_rows_before
            .add(count(metrics.dynamic_filter_reader_predicate_rows_before));
        self.dynamic_filter_reader_predicate_rows_after.add(count(metrics.dynamic_filter_reader_predicate_rows_after));
        self.dynamic_filter_reader_predicate_rows_filtered
            .add(count(metrics.dynamic_filter_reader_predicate_rows_filtered));
        self.dynamic_filter_post_decode_rows_before.add(count(metrics.dynamic_filter_post_decode_rows_before));
        self.dynamic_filter_post_decode_rows_after.add(count(metrics.dynamic_filter_post_decode_rows_after));
        self.dynamic_filter_post_decode_rows_filtered.add(count(metrics.dynamic_filter_post_decode_rows_filtered));
        self.dynamic_filter_source_post_decode_rows_filtered
            .add(count(metrics.dynamic_filter_source_post_decode_rows_filtered));
        self.dynamic_filter_filter_post_decode_rows_filtered
            .add(count(metrics.dynamic_filter_filter_post_decode_rows_filtered));
        self.dynamic_filter_join_probe_post_decode_rows_filtered
            .add(count(metrics.dynamic_filter_join_probe_post_decode_rows_filtered));
        self.hash_join_build_count.add(count(metrics.hash_join_build_count));
        self.filtered_join_build_count.add(count(metrics.filtered_join_build_count));
        self.one_shot_join_count.add(count(metrics.one_shot_join_count));
        self.hash_join_cache_hits.add(count(metrics.hash_join_cache_hits));
        self.hash_join_cache_misses.add(count(metrics.hash_join_cache_misses));
        self.hash_join_durable_cache_hits.add(count(metrics.hash_join_durable_cache_hits));
        self.hash_join_durable_cache_misses.add(count(metrics.hash_join_durable_cache_misses));
        self.hash_join_durable_cache_inserts.add(count(metrics.hash_join_durable_cache_inserts));
        self.hash_join_hash_table_bytes.add(count(metrics.hash_join_hash_table_bytes));
        self.filtered_join_hash_table_bytes.add(count(metrics.filtered_join_hash_table_bytes));
        self.filtered_join_probe_state_bytes.add(count(metrics.filtered_join_probe_state_bytes));
        self.mixed_join_size_data_bytes.add(count(metrics.mixed_join_size_data_bytes));
        self.one_shot_join_index_bytes.add(count(metrics.one_shot_join_index_bytes));
        self.one_shot_join_materialization_bytes.add(count(metrics.one_shot_join_materialization_bytes));
        self.one_shot_join_output_bytes.add(count(metrics.one_shot_join_output_bytes));
        self.join_gather_map_bytes.add(count(join_gather_map_bytes(metrics)));
        self.record_graph_runtime_rows(metrics);
        self.record_task_operator_rows(metrics);
        self.record_task_runtime_rows(metrics);
        self.record_dynamic_filter_runtime_rows(metrics);
        self.record_source_read_runtime_rows(metrics);
        self.record_aggregate_strategy_rows(metrics);
        self.record_join_strategy_rows(metrics);
        self.source_metadata_load_time.add_duration(metrics.source_metadata_load_time);
        self.source_read_queue_wait_time.add_duration(metrics.source_read_queue_wait_time);
        self.source_cache_lookup_time.add_duration(metrics.source_cache_lookup_time);
        self.source_fetch_bytes.add(count(metrics.source_fetch_bytes));
        self.source_fetch_count.add(count(metrics.source_fetch_count));
        self.host_fetch_budget_wait_time.add_duration(metrics.host_fetch_budget_wait_time);
        self.source_fetch_concurrency_wait_time.add_duration(metrics.source_fetch_concurrency_wait_time);
        self.source_fetch_time.add_duration(metrics.source_fetch_time);
        self.source_decode_time.add_duration(metrics.source_decode_time);
        self.parquet_read_time.add_duration(metrics.parquet_read_time);
        self.source_read_time.add_duration(metrics.source_read_time);
        self.operator_time.add_duration(metrics.operator_time);
        self.chunk_operator_time.add_duration(metrics.chunk_operator_time);
        self.final_merge_time.add_duration(metrics.final_merge_time);
        self.final_sort_time.add_duration(metrics.final_sort_time);
        self.materialization_time.add_duration(metrics.materialization_time);
    }

    fn record_graph_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in &metrics.graph_algorithm_runtime_rows {
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("node_id", row.node_id.clone())
                .with_new_label("algorithm", row.algorithm.clone())
                .with_new_label("runtime_outcome", row.runtime_outcome.clone())
                .with_new_label("input_boundary", row.input_boundary.clone())
                .with_new_label("output_boundary", row.output_boundary.clone())
                .with_new_label("construction_policy", row.construction_policy.clone())
                .with_new_label("source_column", row.source_column.clone())
                .with_new_label("destination_column", row.destination_column.clone())
                .with_new_label(
                    "weight_column",
                    row.weight_column.as_deref().map_or_else(|| "none".to_string(), ToOwned::to_owned),
                )
                .with_new_label("directed", row.directed.to_string())
                .with_new_label("renumber", row.renumber.to_string())
                .with_new_label("algorithm_parameters", row.algorithm_parameters.clone())
                .with_new_label("algorithm_metadata", row.algorithm_metadata.clone())
                .with_new_label("input_rows", row.input_rows.to_string())
                .with_new_label("input_columns", row.input_columns.to_string())
                .with_new_label("output_rows", row.output_rows.to_string())
                .with_new_label("output_columns", row.output_columns.to_string())
                .with_new_label(
                    "output_device_bytes",
                    row.output_device_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("elapsed_ns", row.elapsed_ns.to_string())
                .with_new_label("gpu_resident_input_handoffs", row.gpu_resident_input_handoffs.to_string())
                .with_new_label("host_to_device_uploads", row.host_to_device_uploads.to_string())
                .with_new_label("device_to_host_materializations", row.device_to_host_materializations.to_string())
                .global_counter("native_graph_runtime_rows")
                .add(1);
        }
    }

    fn record_task_operator_rows(&self, metrics: &QueryMetrics) {
        for row in metrics.task_operator_metrics.values() {
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("operator_kind", row.operator_kind.clone())
                .with_new_label("task_count", row.task_count.to_string())
                .with_new_label("elapsed_ns", row.elapsed_time.as_nanos().min(u128::from(u64::MAX)).to_string())
                .with_new_label("reservation_requested_bytes", row.reservation_requested_bytes.to_string())
                .with_new_label("reservation_granted_bytes", row.reservation_granted_bytes.to_string())
                .with_new_label("reservation_released_bytes", row.reservation_released_bytes.to_string())
                .with_new_label("reservation_live_peak_bytes", row.reservation_live_peak_bytes.to_string())
                .with_new_label(
                    "max_observed_peak_bytes",
                    row.max_observed_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "max_retained_bytes",
                    row.max_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .global_counter("native_task_operator_rows")
                .add(1);
        }
    }

    fn record_task_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in metrics.task_runtime_metrics.values() {
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("compiled_task_id", row.compiled_task_id.clone())
                .with_new_label("operator_kind", row.operator_kind.clone())
                .with_new_label("elapsed_ns", row.elapsed_time.as_nanos().min(u128::from(u64::MAX)).to_string())
                .with_new_label("reservation_requested_bytes", row.reservation_requested_bytes.to_string())
                .with_new_label("reservation_granted_bytes", row.reservation_granted_bytes.to_string())
                .with_new_label("reservation_released_bytes", row.reservation_released_bytes.to_string())
                .with_new_label("reservation_live_peak_bytes", row.reservation_live_peak_bytes.to_string())
                .with_new_label(
                    "max_observed_peak_bytes",
                    row.max_observed_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "max_retained_bytes",
                    row.max_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .global_counter("native_task_runtime_rows")
                .add(1);
        }
    }

    fn record_dynamic_filter_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in &metrics.dynamic_filter_runtime_rows {
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("dynamic_filter_id", row.dynamic_filter_id.clone())
                .with_new_label("event", row.event.clone())
                .with_new_label("producer_kind", row.producer_kind.clone())
                .with_new_label("producer_node_id", row.producer_node_id.clone())
                .with_new_label("consumer_kind", row.consumer_kind.clone())
                .with_new_label("consumer_node_id", row.consumer_node_id.clone())
                .with_new_label("consumer_lowering", row.consumer_lowering.clone())
                .with_new_label("consumer_reduction_stage", row.consumer_reduction_stage.clone())
                .with_new_label("filter_kind", row.filter_kind.clone())
                .with_new_label(
                    "payload_value_count",
                    row.payload_value_count.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "payload_min_i64",
                    row.payload_min_i64.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "payload_max_i64",
                    row.payload_max_i64.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "producer_rows",
                    row.producer_rows.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "skipped_reason",
                    row.skipped_reason.as_deref().map_or_else(|| "none".to_string(), ToOwned::to_owned),
                )
                .with_new_label(
                    "selected_row_groups_before",
                    row.selected_row_groups_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "selected_row_groups_after",
                    row.selected_row_groups_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "selected_source_bytes_before",
                    row.selected_source_bytes_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "selected_source_bytes_after",
                    row.selected_source_bytes_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "source_file_index",
                    row.source_file_index.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "source_chunk_index",
                    row.source_chunk_index.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "reader_predicate_rows_before",
                    row.reader_predicate_rows_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "reader_predicate_rows_after",
                    row.reader_predicate_rows_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("reader_predicate_rows_filtered", row.reader_predicate_rows_filtered.to_string())
                .with_new_label(
                    "post_decode_rows_before",
                    row.post_decode_rows_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "post_decode_rows_after",
                    row.post_decode_rows_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("post_decode_rows_filtered", row.post_decode_rows_filtered.to_string())
                .with_new_label("row_groups_pruned", row.row_groups_pruned.to_string())
                .with_new_label("source_bytes_pruned", row.source_bytes_pruned.to_string())
                .global_counter("native_dynamic_filter_runtime_rows")
                .add(1);
        }
    }

    fn record_source_read_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in &metrics.source_read_runtime_rows {
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("source_node_id", row.source_node_id.clone())
                .with_new_label("read_kind", row.read_kind.clone())
                .with_new_label("read_strategy", row.read_strategy.clone())
                .with_new_label("batch_index", row.batch_index.to_string())
                .with_new_label("chunk_count", row.chunk_count.to_string())
                .with_new_label("first_chunk_index", row.first_chunk_index.to_string())
                .with_new_label("last_chunk_index", row.last_chunk_index.to_string())
                .with_new_label("first_file_index", row.first_file_index.to_string())
                .with_new_label("last_file_index", row.last_file_index.to_string())
                .with_new_label("file_count", row.file_count.to_string())
                .with_new_label("row_group_count", row.row_group_count.to_string())
                .with_new_label("read_columns_per_file", row.read_columns_per_file.to_string())
                .with_new_label("read_columns_total", row.read_columns_total.to_string())
                .with_new_label("known_file_bytes", row.known_file_bytes.to_string())
                .with_new_label("estimated_source_read_bytes", row.estimated_source_read_bytes.to_string())
                .with_new_label("known_row_group_bytes", row.known_row_group_bytes.to_string())
                .with_new_label("known_projected_compressed_bytes", row.known_projected_compressed_bytes.to_string())
                .with_new_label(
                    "known_projected_uncompressed_bytes",
                    row.known_projected_uncompressed_bytes.to_string(),
                )
                .with_new_label(
                    "known_pure_filter_compressed_bytes",
                    row.known_pure_filter_compressed_bytes.to_string(),
                )
                .with_new_label(
                    "known_pure_filter_uncompressed_bytes",
                    row.known_pure_filter_uncompressed_bytes.to_string(),
                )
                .with_new_label("output_rows", row.output_rows.to_string())
                .with_new_label("reader_predicate", row.reader_predicate.to_string())
                .with_new_label("reader_dynamic_filter_count", row.reader_dynamic_filter_count.to_string())
                .with_new_label("source_batching_reason", row.source_batching_reason.clone())
                .with_new_label("elapsed_ns", row.elapsed.as_nanos().min(u128::from(u64::MAX)).to_string())
                .global_counter("native_source_read_runtime_rows")
                .add(1);
        }
    }

    fn record_join_strategy_rows(&self, metrics: &QueryMetrics) {
        let mut decision_facts_by_node: BTreeMap<&str, BTreeMap<String, String>> = BTreeMap::new();
        for decision in &metrics.join_strategy_decisions {
            decision_facts_by_node
                .entry(decision.node_id.as_str())
                .or_default()
                .extend(decision.facts.iter().map(|(key, value)| (key.clone(), value.clone())));
        }
        for operator in metrics.operators.iter().filter(|operator| operator.operator_kind == "join") {
            let Some(join) = operator.join.as_ref() else {
                continue;
            };
            let facts = decision_facts_by_node.get(operator.node_id.as_str());
            let join_strategy = facts
                .and_then(|facts| facts.get("join_strategy").map(String::as_str))
                .unwrap_or_else(|| join_strategy_label(join));
            let runtime = join_runtime_counters(join_strategy, join);
            let builder = MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("node_id", operator.node_id.clone())
                .with_new_label("join_strategy", join_strategy.to_string())
                .with_new_label("execution_strategy", join.execution_strategy.clone())
                .with_new_label("join_type", join.join_type.clone())
                .with_new_label("build_side", join.build_side.clone())
                .with_new_label("key_count", join.key_count.to_string())
                .with_new_label("estimated_build_rows", optional_fact(facts, "estimated_build_rows"))
                .with_new_label(
                    "observed_build_rows",
                    facts
                        .and_then(|facts| facts.get("observed_build_rows").cloned())
                        .unwrap_or_else(|| join.build_rows.to_string()),
                )
                .with_new_label(
                    "build_side_bytes",
                    facts
                        .and_then(|facts| facts.get("build_side_bytes").cloned())
                        .unwrap_or_else(|| join.build_bytes.to_string()),
                )
                .with_new_label("probe_reuse_count", optional_fact(facts, "probe_reuse_count"))
                .with_new_label("probe_side_bytes", optional_fact(facts, "probe_side_bytes"))
                .with_new_label("partition_count", optional_fact(facts, "partition_count"))
                .with_new_label("partition_target_bytes", optional_fact(facts, "partition_target_bytes"))
                .with_new_label("build_partition_batch_count", optional_fact(facts, "build_partition_batch_count"))
                .with_new_label("probe_partition_batch_count", optional_fact(facts, "probe_partition_batch_count"))
                .with_new_label("build_partitioned_bytes", optional_fact(facts, "build_partitioned_bytes"))
                .with_new_label("probe_partitioned_bytes", optional_fact(facts, "probe_partitioned_bytes"))
                .with_new_label(
                    "reusable_hash_join_rejected_reason",
                    optional_fact(facts, "reusable_hash_join_rejected_reason"),
                )
                .with_new_label(
                    "partitioned_hash_join_rejected_reason",
                    optional_fact(facts, "partitioned_hash_join_rejected_reason"),
                )
                .with_new_label("one_shot_equi_rejected_reason", optional_fact(facts, "one_shot_equi_rejected_reason"))
                .with_new_label("partitioned_hash_join_count", runtime.partitioned_hash_join_count.to_string())
                .with_new_label("hash_join_build_count", runtime.hash_join_build_count.to_string())
                .with_new_label("hash_join_hash_table_bytes", runtime.hash_join_hash_table_bytes.to_string())
                .with_new_label("filtered_join_build_count", runtime.filtered_join_build_count.to_string())
                .with_new_label("filtered_join_hash_table_bytes", runtime.filtered_join_hash_table_bytes.to_string())
                .with_new_label("filtered_join_probe_state_bytes", runtime.filtered_join_probe_state_bytes.to_string())
                .with_new_label("mixed_join_size_data_bytes", runtime.mixed_join_size_data_bytes.to_string())
                .with_new_label("one_shot_join_count", runtime.one_shot_join_count.to_string())
                .with_new_label("one_shot_join_index_bytes", runtime.one_shot_join_index_bytes.to_string())
                .with_new_label(
                    "one_shot_join_materialization_bytes",
                    runtime.one_shot_join_materialization_bytes.to_string(),
                )
                .with_new_label("one_shot_join_output_bytes", runtime.one_shot_join_output_bytes.to_string())
                .with_new_label("join_gather_map_bytes", join.gather_map_bytes.to_string())
                .with_new_label("join_output_reservation_bytes", join.output_reservation_bytes.to_string())
                .with_new_label("join_output_rows", join.output_rows.to_string())
                .with_new_label("join_output_bytes", join.output_bytes.to_string());
            builder.global_counter("native_join_strategy_rows").add(1);
        }
    }

    fn record_aggregate_strategy_rows(&self, metrics: &QueryMetrics) {
        for decision in &metrics.aggregate_strategy_decisions {
            let facts = &decision.facts;
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("node_id", decision.node_id.clone())
                .with_new_label("selected_strategy", aggregate_fact(facts, "selected_strategy"))
                .with_new_label("execution_stage", aggregate_fact(facts, "execution_stage"))
                .with_new_label("memory_mode", aggregate_fact(facts, "memory_mode"))
                .with_new_label("group_key_count", aggregate_fact(facts, "group_key_count"))
                .with_new_label("group_key_exprs", aggregate_fact(facts, "group_key_exprs"))
                .with_new_label("group_key_dtypes", aggregate_fact(facts, "group_key_dtypes"))
                .with_new_label("aggregate_functions", aggregate_fact(facts, "aggregate_functions"))
                .with_new_label(
                    "floating_statistical_aggregate",
                    aggregate_fact(facts, "floating_statistical_aggregate"),
                )
                .with_new_label(
                    "statistical_aggregate_execution_mode",
                    aggregate_fact(facts, "statistical_aggregate_execution_mode"),
                )
                .with_new_label(
                    "statistical_aggregate_tolerance_contract",
                    aggregate_fact(facts, "statistical_aggregate_tolerance_contract"),
                )
                .with_new_label(
                    "statistical_aggregate_functions",
                    aggregate_fact(facts, "statistical_aggregate_functions"),
                )
                .with_new_label("statistical_aggregate_ddof", aggregate_fact(facts, "statistical_aggregate_ddof"))
                .with_new_label("stddev_samp_execution_mode", aggregate_fact(facts, "stddev_samp_execution_mode"))
                .with_new_label("stddev_samp_ddof", aggregate_fact(facts, "stddev_samp_ddof"))
                .with_new_label("aggregate_filter_count", aggregate_fact(facts, "aggregate_filter_count"))
                .with_new_label("distinct_aggregate_count", aggregate_fact(facts, "distinct_aggregate_count"))
                .with_new_label("input_rows", aggregate_fact(facts, "input_rows"))
                .with_new_label("output_group_count", aggregate_fact(facts, "output_group_count"))
                .with_new_label("reduction_ratio_ppm", aggregate_fact(facts, "reduction_ratio_ppm"))
                .with_new_label("chunk_row_count", aggregate_fact(facts, "chunk_row_count"))
                .with_new_label("chunk_count", aggregate_fact(facts, "chunk_count"))
                .with_new_label("partial_aggregate_count", aggregate_fact(facts, "partial_aggregate_count"))
                .with_new_label("source_adjacent_partial", aggregate_fact(facts, "source_adjacent_partial"))
                .with_new_label("source_chunk_row_group_count", aggregate_fact(facts, "source_chunk_row_group_count"))
                .with_new_label(
                    "source_chunk_reads_all_row_groups",
                    aggregate_fact(facts, "source_chunk_reads_all_row_groups"),
                )
                .with_new_label("source_chunk_known_row_count", aggregate_fact(facts, "source_chunk_known_row_count"))
                .with_new_label(
                    "source_chunk_estimated_source_read_bytes",
                    aggregate_fact(facts, "source_chunk_estimated_source_read_bytes"),
                )
                .with_new_label(
                    "source_chunk_size_estimate_based",
                    aggregate_fact(facts, "source_chunk_size_estimate_based"),
                )
                .with_new_label(
                    "source_adjacent_partition_strategy",
                    aggregate_fact(facts, "source_adjacent_partition_strategy"),
                )
                .with_new_label(
                    "source_adjacent_partition_count",
                    aggregate_fact(facts, "source_adjacent_partition_count"),
                )
                .with_new_label(
                    "source_adjacent_non_empty_partition_count",
                    aggregate_fact(facts, "source_adjacent_non_empty_partition_count"),
                )
                .with_new_label(
                    "source_adjacent_partition_key_count",
                    aggregate_fact(facts, "source_adjacent_partition_key_count"),
                )
                .with_new_label(
                    "source_adjacent_max_partition_fan_in",
                    aggregate_fact(facts, "source_adjacent_max_partition_fan_in"),
                )
                .with_new_label(
                    "partitioned_merge_partition_count",
                    aggregate_fact(facts, "partitioned_merge_partition_count"),
                )
                .with_new_label(
                    "partitioned_merge_non_empty_partition_count",
                    aggregate_fact(facts, "partitioned_merge_non_empty_partition_count"),
                )
                .with_new_label(
                    "partitioned_merge_max_partition_fan_in",
                    aggregate_fact(facts, "partitioned_merge_max_partition_fan_in"),
                )
                .with_new_label("selection_elapsed_ns", aggregate_fact(facts, "selection_elapsed_ns"))
                .with_new_label("encode_elapsed_ns", aggregate_fact(facts, "encode_elapsed_ns"))
                .with_new_label("groupby_elapsed_ns", aggregate_fact(facts, "groupby_elapsed_ns"))
                .with_new_label("decode_elapsed_ns", aggregate_fact(facts, "decode_elapsed_ns"))
                .with_new_label("strategy_operator_elapsed_ns", aggregate_fact(facts, "strategy_operator_elapsed_ns"))
                .with_new_label("aggregate_structural_key", aggregate_fact(facts, "aggregate_structural_key"))
                .with_new_label(
                    "prior_cardinality_evidence_scope",
                    aggregate_fact(facts, "prior_cardinality_evidence_scope"),
                )
                .with_new_label(
                    "encoded_key_partial_cost_model",
                    aggregate_fact(facts, "encoded_key_partial_cost_model"),
                )
                .with_new_label(
                    "encoded_key_partial_cost_model_status",
                    aggregate_fact(facts, "encoded_key_partial_cost_model_status"),
                )
                .with_new_label(
                    "encoded_key_partial_selection_reason",
                    aggregate_fact(facts, "encoded_key_partial_selection_reason"),
                )
                .with_new_label(
                    "encoded_key_partial_reduction_ratio_threshold_ppm",
                    aggregate_fact(facts, "encoded_key_partial_reduction_ratio_threshold_ppm"),
                )
                .with_new_label(
                    "encoded_key_partial_prior_reduction_ratio_ppm",
                    aggregate_fact(facts, "encoded_key_partial_prior_reduction_ratio_ppm"),
                )
                .with_new_label(
                    "encoded_key_partial_prior_strategy",
                    aggregate_fact(facts, "encoded_key_partial_prior_strategy"),
                )
                .with_new_label(
                    "exact_cardinality_probe_status",
                    aggregate_fact(facts, "exact_cardinality_probe_status"),
                )
                .with_new_label("exact_cardinality_probe_key", aggregate_fact(facts, "exact_cardinality_probe_key"))
                .with_new_label(
                    "exact_cardinality_probe_input_rows",
                    aggregate_fact(facts, "exact_cardinality_probe_input_rows"),
                )
                .with_new_label(
                    "exact_cardinality_probe_non_null_distinct_count",
                    aggregate_fact(facts, "exact_cardinality_probe_non_null_distinct_count"),
                )
                .with_new_label(
                    "exact_cardinality_probe_null_count",
                    aggregate_fact(facts, "exact_cardinality_probe_null_count"),
                )
                .with_new_label(
                    "exact_cardinality_probe_num_groups",
                    aggregate_fact(facts, "exact_cardinality_probe_num_groups"),
                )
                .with_new_label(
                    "exact_cardinality_probe_reduction_ratio_ppm",
                    aggregate_fact(facts, "exact_cardinality_probe_reduction_ratio_ppm"),
                )
                .with_new_label(
                    "exact_cardinality_probe_elapsed_ns",
                    aggregate_fact(facts, "exact_cardinality_probe_elapsed_ns"),
                )
                .with_new_label(
                    "prepared_sort_groupby_rejected_reason",
                    aggregate_fact(facts, "prepared_sort_groupby_rejected_reason"),
                )
                .with_new_label(
                    "prepared_sort_groupby_contract_status",
                    aggregate_fact(facts, "prepared_sort_groupby_contract_status"),
                )
                .with_new_label(
                    "prepared_sort_groupby_required_contract",
                    aggregate_fact(facts, "prepared_sort_groupby_required_contract"),
                )
                .with_new_label(
                    "prepared_sort_groupby_cost_model",
                    aggregate_fact(facts, "prepared_sort_groupby_cost_model"),
                )
                .with_new_label(
                    "prepared_sort_groupby_cost_model_status",
                    aggregate_fact(facts, "prepared_sort_groupby_cost_model_status"),
                )
                .with_new_label(
                    "prepared_sort_groupby_selection_reason",
                    aggregate_fact(facts, "prepared_sort_groupby_selection_reason"),
                )
                .with_new_label(
                    "prepared_sort_groupby_num_groups",
                    aggregate_fact(facts, "prepared_sort_groupby_num_groups"),
                )
                .with_new_label(
                    "prepared_sort_groupby_grouped_key_output_bytes",
                    aggregate_fact(facts, "prepared_sort_groupby_grouped_key_output_bytes"),
                )
                .with_new_label(
                    "prepared_sort_groupby_max_group_size",
                    aggregate_fact(facts, "prepared_sort_groupby_max_group_size"),
                )
                .with_new_label("rejected_candidate_strategies", aggregate_fact(facts, "rejected_candidate_strategies"))
                .global_counter("native_aggregate_strategy_rows")
                .add(1);
            self.native_aggregate_strategy_rows.add(1);
        }
    }

    fn record_boundary_policy(&self, policy: ExecutionBoundaryPolicy) {
        // These counters record which boundary policy was selected for this native query execution,
        // not rows or bytes crossing that boundary.
        if matches!(policy.source_read(), nexus_query_engine::SourceReadBoundary::DirectDeviceLocalParquet) {
            self.source_read_boundary_direct_device.add(1);
        }
        if matches!(
            policy.build_materialization(),
            nexus_query_engine::BuildMaterializationBoundary::QueryScopedGpuHashJoin
        ) {
            self.build_materialization_boundary_query_scoped.add(1);
        }
        if matches!(
            policy.build_materialization(),
            nexus_query_engine::BuildMaterializationBoundary::DurableGpuHashJoin
        ) {
            self.build_materialization_boundary_durable.add(1);
        }
        if matches!(policy.probe_execution(), nexus_query_engine::ProbeExecutionBoundary::ReusableGpuHashJoin) {
            self.probe_execution_boundary_reusable.add(1);
        }
        if matches!(policy.result_materialization(), nexus_query_engine::ResultMaterializationBoundary::ExplicitSink) {
            self.result_materialization_boundary_explicit.add(1);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct JoinRuntimeCounters {
    partitioned_hash_join_count: u64,
    hash_join_build_count: u64,
    hash_join_hash_table_bytes: u64,
    filtered_join_build_count: u64,
    filtered_join_hash_table_bytes: u64,
    filtered_join_probe_state_bytes: u64,
    mixed_join_size_data_bytes: u64,
    one_shot_join_count: u64,
    one_shot_join_index_bytes: u64,
    one_shot_join_materialization_bytes: u64,
    one_shot_join_output_bytes: u64,
}

fn optional_fact(facts: Option<&BTreeMap<String, String>>, key: &str) -> String {
    facts.and_then(|facts| facts.get(key).cloned()).unwrap_or_else(|| "none".to_string())
}

fn aggregate_fact(facts: &BTreeMap<String, String>, key: &str) -> String {
    facts.get(key).cloned().unwrap_or_else(|| "none".to_string())
}

fn join_strategy_label(join: &JoinOperatorMetrics) -> &'static str {
    match join.execution_strategy.as_str() {
        "bounded_one_shot_equi_join" => "one_shot_equi",
        "bounded_partitioned_hash_equi_join" => "partitioned_hash_equi",
        "bounded_partitioned_mixed_keyed_residual_join" => "partitioned_mixed_keyed_residual",
        "bounded_reusable_hash_join" | "hash_join" => "reusable_hash_equi",
        "bounded_reusable_filtered_join" => "filtered_join",
        "bounded_mixed_keyed_residual_join" | "mixed_join" => "mixed_keyed_residual",
        "bounded_keyless_conditional_join" | "conditional_join" => "keyless_conditional",
        "bounded_projected_cross_join" | "cross_join" => "projected_cross",
        _ => "unknown",
    }
}

fn join_runtime_counters(join_strategy: &str, join: &JoinOperatorMetrics) -> JoinRuntimeCounters {
    match join_strategy {
        "reusable_hash_equi" => JoinRuntimeCounters {
            hash_join_build_count: u64::from(join.hash_table_bytes > 0),
            hash_join_hash_table_bytes: join.hash_table_bytes,
            ..Default::default()
        },
        "filtered_join" => JoinRuntimeCounters {
            filtered_join_build_count: u64::from(join.hash_table_bytes > 0),
            filtered_join_hash_table_bytes: join.hash_table_bytes,
            filtered_join_probe_state_bytes: join.filtered_probe_state_bytes,
            ..Default::default()
        },
        "one_shot_equi" => JoinRuntimeCounters {
            one_shot_join_count: 1,
            one_shot_join_index_bytes: join.gather_map_bytes,
            one_shot_join_materialization_bytes: join.output_reservation_bytes,
            one_shot_join_output_bytes: join.output_bytes,
            ..Default::default()
        },
        "partitioned_hash_equi" => JoinRuntimeCounters { partitioned_hash_join_count: 1, ..Default::default() },
        "partitioned_mixed_keyed_residual" => JoinRuntimeCounters {
            partitioned_hash_join_count: 1,
            mixed_join_size_data_bytes: join.mixed_size_data_bytes,
            ..Default::default()
        },
        "mixed_keyed_residual" => {
            JoinRuntimeCounters { mixed_join_size_data_bytes: join.mixed_size_data_bytes, ..Default::default() }
        },
        _ => JoinRuntimeCounters::default(),
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

#[cfg(test)]
mod tests {
    use nexus_query_engine::SourceReadRuntimeMetrics;

    use super::*;
    use crate::native_report::NativeDataFusionExecutionMetrics;

    #[test]
    fn aggregate_strategy_rows_bridge_statistical_contract_labels() {
        let metrics_set = ExecutionPlanMetricsSet::new();
        let native_metrics = NativeCudfMetrics::new(&metrics_set, 0);
        let mut query_metrics = QueryMetrics::default();
        let mut facts = BTreeMap::new();
        facts.insert("selected_strategy", "stddev_samp_gpu_native_tolerant".to_string());
        facts.insert("execution_stage", "single".to_string());
        facts.insert("memory_mode", "bounded".to_string());
        facts.insert("group_key_count", "4".to_string());
        facts.insert("group_key_exprs", "w_warehouse_name,w_warehouse_sk,i_item_sk,d_moy".to_string());
        facts.insert("group_key_dtypes", "Utf8View,Int64,Int64,Int64".to_string());
        facts.insert("aggregate_functions", "stddev_samp,avg".to_string());
        facts.insert("floating_statistical_aggregate", "true".to_string());
        facts.insert("statistical_aggregate_execution_mode", "gpu_native_tolerant".to_string());
        facts.insert("statistical_aggregate_tolerance_contract", "float_tolerant".to_string());
        facts.insert("statistical_aggregate_functions", "stddev_samp".to_string());
        facts.insert("statistical_aggregate_ddof", "stddev_samp:1".to_string());
        facts.insert("stddev_samp_execution_mode", "gpu_native_tolerant".to_string());
        facts.insert("stddev_samp_ddof", "1".to_string());

        query_metrics.record_aggregate_strategy_decision("n12", &facts);
        native_metrics.record_query_metrics(&query_metrics);

        let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
        assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
        let row = &metrics.aggregate_strategy_rows[0];
        assert_eq!(row.node_id, "n12");
        assert_eq!(row.selected_strategy, "stddev_samp_gpu_native_tolerant");
        assert_eq!(row.floating_statistical_aggregate, "true");
        assert_eq!(row.statistical_aggregate_execution_mode, "gpu_native_tolerant");
        assert_eq!(row.statistical_aggregate_tolerance_contract, "float_tolerant");
        assert_eq!(row.statistical_aggregate_functions, "stddev_samp");
        assert_eq!(row.statistical_aggregate_ddof, "stddev_samp:1");
        assert_eq!(row.stddev_samp_execution_mode, "gpu_native_tolerant");
        assert_eq!(row.stddev_samp_ddof, "1");
    }

    #[test]
    fn source_read_runtime_rows_bridge_runtime_labels() {
        let metrics_set = ExecutionPlanMetricsSet::new();
        let native_metrics = NativeCudfMetrics::new(&metrics_set, 0);
        let mut query_metrics = QueryMetrics::default();

        query_metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
            source_node_id: "n7".to_string(),
            read_kind: "chunk_batch".to_string(),
            read_strategy: "local_cache_kvikio".to_string(),
            batch_index: 2,
            chunk_count: 3,
            first_chunk_index: 4,
            last_chunk_index: 6,
            first_file_index: 10,
            last_file_index: 12,
            file_count: 3,
            row_group_count: 9,
            read_columns_per_file: 2,
            read_columns_total: 6,
            known_file_bytes: 900,
            estimated_source_read_bytes: 700,
            known_row_group_bytes: 600,
            known_projected_compressed_bytes: 500,
            known_projected_uncompressed_bytes: 800,
            known_pure_filter_compressed_bytes: 100,
            known_pure_filter_uncompressed_bytes: 200,
            output_rows: 42,
            reader_predicate: true,
            reader_dynamic_filter_count: 1,
            source_batching_reason: "batched".to_string(),
            elapsed: std::time::Duration::from_nanos(1234),
        });
        native_metrics.record_query_metrics(&query_metrics);

        let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
        assert_eq!(metrics.source_read_runtime_rows.len(), 1);
        let row = &metrics.source_read_runtime_rows[0];
        assert_eq!(row.source_node_id, "n7");
        assert_eq!(row.read_kind, "chunk_batch");
        assert_eq!(row.read_strategy, "local_cache_kvikio");
        assert_eq!(row.batch_index, 2);
        assert_eq!(row.chunk_count, 3);
        assert_eq!(row.first_file_index, 10);
        assert_eq!(row.file_count, 3);
        assert_eq!(row.row_group_count, 9);
        assert_eq!(row.read_columns_total, 6);
        assert_eq!(row.known_projected_uncompressed_bytes, 800);
        assert_eq!(row.known_pure_filter_uncompressed_bytes, 200);
        assert_eq!(row.output_rows, 42);
        assert!(row.reader_predicate);
        assert_eq!(row.reader_dynamic_filter_count, 1);
        assert_eq!(row.source_batching_reason, "batched");
        assert_eq!(row.elapsed_ns, 1234);
    }
}
