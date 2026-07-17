//! Machine-readable descriptors for native DataFusion report fields.

use std::collections::BTreeSet;

use nexus_query_engine::metrics::{MetricAggregation, MetricDescriptor, MetricDirection, MetricUnit};

const NATIVE_REPORT_DESCRIPTOR_NAMES: &[&str] = &[
    "adaptive_downgrade_count",
    "adaptive_memory_history_hit_count",
    "adaptive_memory_history_lookup_count",
    "adaptive_memory_history_miss_count",
    "adaptive_pipeline_rows",
    "adaptive_retry_count",
    "adaptive_spill_count",
    "admission_sequence",
    "admission_sequence_observed",
    "admission_wait_ms",
    "admission_wait_time",
    "admission_wait_time_observed",
    "aggregate_encoded_key_partial_elapsed_ns",
    "aggregate_exact_cardinality_probe_count",
    "aggregate_exact_cardinality_probe_elapsed_ns",
    "aggregate_exact_cardinality_probe_input_rows",
    "aggregate_libcudf_runtime_probe_count",
    "aggregate_libcudf_runtime_probe_elapsed_ns",
    "aggregate_libcudf_runtime_probe_input_rows",
    "aggregate_prepared_sort_groupby_elapsed_ns",
    "aggregate_strategy_rows",
    "allocator_high_water_bytes",
    "allocator_high_water_bytes_observed",
    "build_materialization_boundary_durable",
    "build_materialization_boundary_query_scoped",
    "chunk_operator_time",
    "comparison_mode",
    "compiled_pipeline_attempts",
    "compiled_pipeline_host_arrow_sinks",
    "compiled_pipeline_tasks_executed",
    "cpu_output_batches",
    "cpu_output_fingerprint",
    "cpu_output_rows",
    "cpu_planning_status",
    "cpu_runtime_status",
    "device_ordinal",
    "device_ordinal_observed",
    "device_to_device_copied_bytes",
    "device_to_device_copies",
    "device_to_host_materializations",
    "device_to_host_materialized_bytes",
    "device_to_host_materialized_rows",
    "duckdb_cpu_parity_detail",
    "duckdb_cpu_parity_status",
    "duckdb_output_batches",
    "duckdb_output_fingerprint",
    "duckdb_output_rows",
    "duckdb_output_status",
    "dynamic_filter_applied_count",
    "dynamic_filter_filter_post_decode_rows_filtered",
    "dynamic_filter_join_probe_post_decode_rows_filtered",
    "dynamic_filter_measured_reduction_status",
    "dynamic_filter_post_decode_reduction_stage",
    "dynamic_filter_post_decode_rows_after",
    "dynamic_filter_post_decode_rows_before",
    "dynamic_filter_post_decode_rows_filtered",
    "dynamic_filter_produced_count",
    "dynamic_filter_reader_predicate_rows_after",
    "dynamic_filter_reader_predicate_rows_before",
    "dynamic_filter_reader_predicate_rows_filtered",
    "dynamic_filter_row_groups_pruned",
    "dynamic_filter_runtime_rows",
    "dynamic_filter_skipped_count",
    "dynamic_filter_source_bytes_pruned",
    "dynamic_filter_source_post_decode_rows_filtered",
    "dynamic_filter_work_avoidance_status",
    "edge_normalize_inserted",
    "filtered_join_build_count",
    "filtered_join_hash_table_bytes",
    "filtered_join_probe_state_bytes",
    "final_merge_attempts",
    "final_merge_levels",
    "final_merge_max_fan_in",
    "final_merge_time",
    "final_sort_time",
    "gpu_resident_input_handoffs",
    "graph_reservation_rows",
    "graph_runtime_rows",
    "hash_join_build_count",
    "hash_join_cache_hits",
    "hash_join_cache_misses",
    "hash_join_durable_cache_hits",
    "hash_join_durable_cache_inserts",
    "hash_join_durable_cache_misses",
    "hash_join_hash_table_bytes",
    "host_datafusion_boundary_count",
    "host_to_device_uploaded_bytes",
    "host_to_device_uploaded_rows",
    "host_to_device_uploads",
    "input_batches",
    "input_rows",
    "join_diagnostics",
    "join_gather_map_bytes",
    "join_strategy_rows",
    "lakehouse_cache_fill_event_snapshot_rows",
    "materialization_time",
    "max_task_observed_peak_bytes",
    "max_task_observed_peak_bytes_observed",
    "max_task_reservation_bytes",
    "max_task_retained_bytes",
    "max_task_retained_bytes_observed",
    "mixed_join_size_data_bytes",
    "native_aggregate_strategy_rows",
    "native_candidate_shape",
    "native_candidate_shape_family",
    "native_dynamic_filter_runtime_rows",
    "native_graph_reservation_rows",
    "native_graph_runtime_rows",
    "native_join_strategy_rows",
    "native_lakehouse_cache_fill_event_snapshot_rows",
    "native_optimizer",
    "native_output_batches",
    "native_output_fingerprint",
    "native_output_rows",
    "native_plan_fragment_count",
    "native_plan_task_mapping_rows",
    "native_planning_outcome",
    "native_reason_code",
    "native_reservation_denied_count",
    "native_reservation_live_peak_bytes",
    "native_retry_count",
    "native_runtime_failure",
    "native_runtime_outcome",
    "native_runtime_started",
    "native_source_read_runtime_rows",
    "native_task_operator_rows",
    "native_task_runtime_rows",
    "native_unsupported_category",
    "one_shot_join_count",
    "one_shot_join_index_bytes",
    "one_shot_join_materialization_bytes",
    "one_shot_join_output_bytes",
    "operator_time",
    "outcome",
    "output_batches",
    "output_bytes",
    "output_rows",
    "parity_detail",
    "parity_status",
    "parquet_bytes_known",
    "parquet_files",
    "parquet_read_time",
    "partial_aggregate_count",
    "partial_aggregate_input_rows",
    "partial_aggregate_retained_bytes",
    "partial_aggregate_rows",
    "peak_device_memory_bytes",
    "physical_root",
    "physical_shape",
    "plan_task_mapping_rows",
    "planning_outcome",
    "post_native_execution_path",
    "post_native_plan_root",
    "probe_execution_boundary_reusable",
    "query_id",
    "query_grant_bytes",
    "query_grant_bytes_observed",
    "read_columns",
    "reason_code",
    "rejected_before_execution",
    "reservation_denied_count",
    "reservation_granted_bytes",
    "reservation_live_peak_bytes",
    "reservation_requested_bytes",
    "resident_dataframe_peak_bytes",
    "resident_dataframe_peak_bytes_observed",
    "result_materialization_boundary_explicit",
    "retry_count",
    "retry_reservation_failure_count",
    "retry_resource_exhausted_count",
    "row_groups_committed",
    "row_groups_planned",
    "row_groups_selected",
    "runtime_failure",
    "runtime_outcome",
    "source_cache_hits",
    "source_cache_insert_skipped",
    "source_cache_inserts",
    "source_cache_lookup_time",
    "source_cache_misses",
    "source_cache_retained_bytes",
    "source_chunks_attempted",
    "source_chunks_committed",
    "source_chunks_planned",
    "source_chunks_retried",
    "source_diagnostics",
    "source_metadata_load_time",
    "source_read_boundary_direct_device",
    "source_read_prefetches_abandoned",
    "source_read_prefetches_completed",
    "source_read_prefetches_skipped",
    "source_read_prefetches_started",
    "source_read_queue_wait_time",
    "source_read_time",
    "strict_bounded_compiled_graphs",
    "strict_bounded_contract_rows",
    "task_operator_rows",
    "task_runtime_rows",
    "unsupported_category",
];

/// Returns descriptors for every native report TSV column and public DataFusion metric name.
pub fn native_report_metric_descriptors() -> Vec<MetricDescriptor> {
    native_report_metric_names().into_iter().map(descriptor_for_canonical_name).collect()
}

/// Returns the descriptor for a native report TSV column or public DataFusion metric name.
pub fn native_report_metric_descriptor(name: &str) -> Option<MetricDescriptor> {
    canonical_metric_name(name).map(descriptor_for_canonical_name)
}

/// Returns every TSV/DataFusion metric name covered by the adapter descriptor registry.
pub fn native_report_metric_names() -> Vec<&'static str> {
    let mut names = BTreeSet::new();
    names.extend(NATIVE_REPORT_DESCRIPTOR_NAMES.iter().copied());
    names.into_iter().collect()
}

fn canonical_metric_name(name: &str) -> Option<&'static str> {
    native_report_metric_names().into_iter().find(|candidate| *candidate == name)
}

fn descriptor_for_canonical_name(name: &'static str) -> MetricDescriptor {
    MetricDescriptor {
        name,
        unit: unit_for_name(name),
        aggregation: aggregation_for_name(name),
        semantics: semantics_for_name(name),
        direction: direction_for_name(name),
        healthy: healthy_for_name(name),
    }
}

fn unit_for_name(name: &str) -> MetricUnit {
    if is_label_field(name) || is_nested_row_cell(name) {
        MetricUnit::Label
    } else if is_nested_row_count_metric(name) {
        MetricUnit::Count
    } else if name.ends_with("_ms") {
        MetricUnit::Millis
    } else if name.ends_with("_time")
        || name.ends_with("_elapsed_ns")
        || name.ends_with("_time_ns")
        || name.ends_with("_elapsed_nanos")
    {
        MetricUnit::Nanos
    } else if name.ends_with("_rows") || name.contains("_output_rows") || name.contains("_input_rows") {
        MetricUnit::Rows
    } else if name.ends_with("_bytes") {
        if name.contains("host") {
            MetricUnit::HostBytes
        } else if name.contains("parquet")
            || name.contains("file")
            || name.contains("row_group")
            || name.contains("compressed")
            || name.contains("uncompressed")
        {
            MetricUnit::ParquetBytes
        } else {
            MetricUnit::DeviceBytes
        }
    } else if name == "native_runtime_started"
        || name == "rejected_before_execution"
        || name == "edge_normalize_inserted"
        || name.ends_with("_observed")
    {
        MetricUnit::Boolean
    } else {
        MetricUnit::Count
    }
}

fn aggregation_for_name(name: &str) -> MetricAggregation {
    if is_label_field(name) || is_nested_row_cell(name) {
        MetricAggregation::Last
    } else if is_nested_row_count_metric(name) {
        MetricAggregation::RowCount
    } else if name.contains("peak")
        || name.contains("high_water")
        || name.contains("retained")
        || name.contains("max_")
        || name.ends_with("_live_peak_bytes")
    {
        MetricAggregation::Peak
    } else if name.ends_with("_observed")
        || name == "native_runtime_started"
        || name == "rejected_before_execution"
        || name == "edge_normalize_inserted"
        || matches!(name, "admission_sequence" | "device_ordinal" | "query_grant_bytes")
    {
        MetricAggregation::Last
    } else {
        MetricAggregation::Sum
    }
}

fn direction_for_name(name: &str) -> Option<MetricDirection> {
    if name.contains("hit")
        || name.contains("pruned")
        || name.contains("filtered")
        || name == "gpu_resident_input_handoffs"
        || name == "native_output_rows"
    {
        Some(MetricDirection::HigherIsBetter)
    } else if name.contains("failure")
        || name.contains("denied")
        || name.contains("miss")
        || name.contains("retry")
        || name.contains("fallback")
        || name.contains("spill")
        || name.contains("downgrade")
        || name.contains("wait")
    {
        Some(MetricDirection::LowerIsBetter)
    } else {
        None
    }
}

fn healthy_for_name(name: &str) -> Option<&'static str> {
    if name.contains("failure")
        || name.contains("denied")
        || name.contains("retry")
        || name.contains("spill")
        || name.contains("downgrade")
    {
        Some("0")
    } else {
        None
    }
}

fn semantics_for_name(name: &str) -> &'static str {
    match name {
        "query_id" => "Query-wide diagnostic correlation id.",
        "reason_code" | "native_reason_code" => "Stable planning or runtime reason code.",
        "runtime_failure" | "native_runtime_failure" => "Stable runtime failure summary.",
        "physical_root" => "Root DataFusion physical operator label.",
        "physical_shape" => "Stable DataFusion physical-plan shape summary.",
        "native_candidate_shape" => "Stable native candidate shape summary.",
        "native_candidate_shape_family" => "Stable native candidate shape family.",
        "planning_outcome" | "native_planning_outcome" => "Native planning outcome label.",
        "runtime_outcome" | "native_runtime_outcome" => "Native runtime outcome label.",
        "parity_status" | "duckdb_cpu_parity_status" => "Output parity status label.",
        _ if is_nested_row_cell(name) => "Stable nested diagnostic rows serialized in a TSV cell.",
        _ if is_nested_row_count_metric(name) => "Count of nested diagnostic rows emitted as DataFusion metrics.",
        _ if is_label_field(name) => "Stable diagnostic label exported by native reports.",
        _ if name.ends_with("_ms") => "Elapsed time exported in milliseconds by the TSV report.",
        _ if name.ends_with("_time") => "Elapsed time exported in nanoseconds by DataFusion metrics.",
        _ if name.ends_with("_bytes") => "Byte counter or peak exported by native diagnostics.",
        _ if name.ends_with("_rows") => "Row counter exported by native diagnostics.",
        _ => "Native report counter or diagnostic value.",
    }
}

fn is_nested_row_cell(name: &str) -> bool {
    matches!(
        name,
        "strict_bounded_contract_rows"
            | "strict_bounded_compiled_graphs"
            | "adaptive_pipeline_rows"
            | "join_diagnostics"
            | "source_diagnostics"
            | "graph_reservation_rows"
            | "graph_runtime_rows"
            | "task_operator_rows"
            | "task_runtime_rows"
            | "dynamic_filter_runtime_rows"
            | "aggregate_strategy_rows"
            | "join_strategy_rows"
            | "lakehouse_cache_fill_event_snapshot_rows"
            | "plan_task_mapping_rows"
    )
}

fn is_nested_row_count_metric(name: &str) -> bool {
    matches!(
        name,
        "native_graph_reservation_rows"
            | "native_graph_runtime_rows"
            | "native_task_operator_rows"
            | "native_task_runtime_rows"
            | "native_dynamic_filter_runtime_rows"
            | "native_source_read_runtime_rows"
            | "native_lakehouse_cache_fill_event_snapshot_rows"
            | "native_aggregate_strategy_rows"
            | "native_join_strategy_rows"
            | "native_plan_task_mapping_rows"
    )
}

fn is_label_field(name: &str) -> bool {
    name.ends_with("_shape")
        || name.ends_with("_shape_family")
        || name.ends_with("_outcome")
        || name.ends_with("_status")
        || name.ends_with("_detail")
        || name.ends_with("_category")
        || name.ends_with("_path")
        || name.ends_with("_root")
        || name.ends_with("_policy")
        || name.ends_with("_mode")
        || name.ends_with("_fingerprint")
        || name.ends_with("_failure")
        || name.ends_with("_diagnostics")
        || name.ends_with("_reason_code")
        || matches!(name, "query_id" | "outcome" | "reason_code")
        || name == "comparison_mode"
        || name == "unsupported_category"
        || name == "native_optimizer"
        || name == "post_native_plan_root"
        || name == "post_native_execution_path"
        || name == "dynamic_filter_post_decode_reduction_stage"
        || name == "dynamic_filter_measured_reduction_status"
        || name == "dynamic_filter_work_avoidance_status"
        || name == "cpu_planning_status"
        || name == "cpu_runtime_status"
        || name == "duckdb_output_status"
}
