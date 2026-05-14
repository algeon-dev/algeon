use std::collections::BTreeMap;

use cudf_query_engine::{ExecutionBoundaryPolicy, JoinOperatorMetrics, QueryMetrics};
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
    "reservation_live_peak_bytes",
    "reservation_denied_count",
    "max_task_reservation_bytes",
    "max_task_observed_peak_bytes",
    "max_task_observed_peak_bytes_observed",
    "max_task_retained_bytes",
    "max_task_retained_bytes_observed",
    "native_task_operator_rows",
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
    "partial_aggregate_rows",
    "final_merge_attempts",
    "final_merge_levels",
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
    "native_join_strategy_rows",
    "source_metadata_load_time",
    "source_read_queue_wait_time",
    "source_cache_lookup_time",
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
    metrics_set: ExecutionPlanMetricsSet,
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
    partial_aggregate_rows: Count,
    final_merge_attempts: Count,
    final_merge_levels: Count,
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
    source_metadata_load_time: Time,
    source_read_queue_wait_time: Time,
    source_cache_lookup_time: Time,
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
        let _ = summary_counter(metrics, "native_join_strategy_rows", partition);
        let _ = summary_counter(metrics, "native_task_operator_rows", partition);
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
            partial_aggregate_rows: summary_counter(metrics, "partial_aggregate_rows", partition),
            final_merge_attempts: summary_counter(metrics, "final_merge_attempts", partition),
            final_merge_levels: summary_counter(metrics, "final_merge_levels", partition),
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
            source_metadata_load_time: summary_timer(metrics, "source_metadata_load_time", partition),
            source_read_queue_wait_time: summary_timer(metrics, "source_read_queue_wait_time", partition),
            source_cache_lookup_time: summary_timer(metrics, "source_cache_lookup_time", partition),
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
        self.partial_aggregate_rows.add(count(metrics.partial_aggregate_rows));
        self.final_merge_attempts.add(count(metrics.final_merge_attempts));
        self.final_merge_levels.add(count(metrics.final_merge_levels));
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
        self.record_task_operator_rows(metrics);
        self.record_join_strategy_rows(metrics);
        self.source_metadata_load_time.add_duration(metrics.source_metadata_load_time);
        self.source_read_queue_wait_time.add_duration(metrics.source_read_queue_wait_time);
        self.source_cache_lookup_time.add_duration(metrics.source_cache_lookup_time);
        self.parquet_read_time.add_duration(metrics.parquet_read_time);
        self.source_read_time.add_duration(metrics.source_read_time);
        self.operator_time.add_duration(metrics.operator_time);
        self.chunk_operator_time.add_duration(metrics.chunk_operator_time);
        self.final_merge_time.add_duration(metrics.final_merge_time);
        self.final_sort_time.add_duration(metrics.final_sort_time);
        self.materialization_time.add_duration(metrics.materialization_time);
    }

    fn record_task_operator_rows(&self, metrics: &QueryMetrics) {
        for row in metrics.task_operator_metrics.values() {
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::SUMMARY)
                .with_new_label("operator_kind", row.operator_kind.clone())
                .with_new_label("task_count", row.task_count.to_string())
                .with_new_label("reservation_live_peak_bytes", row.reservation_live_peak_bytes.to_string())
                .global_counter("native_task_operator_rows")
                .add(1);
        }
    }

    fn record_join_strategy_rows(&self, metrics: &QueryMetrics) {
        let decision_facts_by_node = metrics
            .join_strategy_decisions
            .iter()
            .map(|decision| (decision.node_id.as_str(), &decision.facts))
            .collect::<BTreeMap<_, _>>();
        for operator in metrics.operators.iter().filter(|operator| operator.operator_kind == "join") {
            let Some(join) = operator.join.as_ref() else {
                continue;
            };
            let facts = decision_facts_by_node.get(operator.node_id.as_str()).copied();
            let join_strategy = facts
                .and_then(|facts| facts.get("join_strategy").map(String::as_str))
                .unwrap_or_else(|| join_strategy_label(join));
            let runtime = join_runtime_counters(join_strategy, join);
            let mut builder = MetricBuilder::new(&self.metrics_set)
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
                .with_new_label(
                    "reusable_hash_join_rejected_reason",
                    optional_fact(facts, "reusable_hash_join_rejected_reason"),
                )
                .with_new_label("one_shot_equi_rejected_reason", optional_fact(facts, "one_shot_equi_rejected_reason"))
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
                .with_new_label("join_gather_map_bytes", join.gather_map_bytes.to_string());
            if join.output_reservation_bytes > 0 {
                builder =
                    builder.with_new_label("join_output_reservation_bytes", join.output_reservation_bytes.to_string());
            }
            builder.global_counter("native_join_strategy_rows").add(1);
        }
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

#[derive(Debug, Clone, Copy, Default)]
struct JoinRuntimeCounters {
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

fn join_strategy_label(join: &JoinOperatorMetrics) -> &'static str {
    match join.execution_strategy.as_str() {
        "bounded_one_shot_equi_join" => "one_shot_equi",
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
