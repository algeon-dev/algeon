//! TSV column-order constants for native planning, execution, and fallback reports.

/// Fallback-pressure TSV column names in wire/report order.
pub(crate) const NATIVE_FALLBACK_PRESSURE_TSV_COLUMNS: &[&str] = &[
    "query_id",
    "physical_root",
    "physical_shape",
    "native_candidate_shape",
    "native_candidate_shape_family",
    "outcome",
    "reason_code",
    "unsupported_category",
    "adaptive_pipeline_rows",
    "join_diagnostics",
    "source_diagnostics",
    "rejected_before_execution",
];
/// Fallback-pressure TSV header.
pub const NATIVE_FALLBACK_PRESSURE_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tadaptive_pipeline_rows\tjoin_diagnostics\tsource_diagnostics\trejected_before_execution";

/// Planning-report TSV column names in wire/report order.
pub(crate) const NATIVE_DATAFUSION_PLANNING_TSV_COLUMNS: &[&str] = &[
    "query_id",
    "physical_root",
    "physical_shape",
    "native_candidate_shape",
    "native_candidate_shape_family",
    "outcome",
    "reason_code",
    "unsupported_category",
    "strict_bounded_contract_rows",
    "strict_bounded_compiled_graphs",
    "adaptive_pipeline_rows",
    "native_optimizer",
    "edge_normalize_inserted",
    "join_diagnostics",
    "source_diagnostics",
    "post_native_plan_root",
    "post_native_execution_path",
    "native_plan_fragment_count",
    "host_datafusion_boundary_count",
    "rejected_before_execution",
];
/// Planning-report TSV header. Nested row columns use stable key/value escaping.
pub const NATIVE_DATAFUSION_PLANNING_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tstrict_bounded_contract_rows\tstrict_bounded_compiled_graphs\tadaptive_pipeline_rows\tnative_optimizer\tedge_normalize_inserted\tjoin_diagnostics\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\tnative_plan_fragment_count\thost_datafusion_boundary_count\trejected_before_execution";

/// Execution-report TSV column names in wire/report order.
pub(crate) const NATIVE_DATAFUSION_EXECUTION_TSV_COLUMNS: &[&str] = &[
    "query_id",
    "physical_root",
    "physical_shape",
    "native_candidate_shape",
    "native_candidate_shape_family",
    "planning_outcome",
    "reason_code",
    "unsupported_category",
    "strict_bounded_contract_rows",
    "strict_bounded_compiled_graphs",
    "adaptive_pipeline_rows",
    "post_native_plan_root",
    "post_native_execution_path",
    "native_plan_fragment_count",
    "host_datafusion_boundary_count",
    "rejected_before_execution",
    "admission_sequence",
    "admission_wait_ms",
    "device_ordinal",
    "query_grant_bytes",
    "native_runtime_started",
    "runtime_outcome",
    "runtime_failure",
    "output_batches",
    "output_rows",
    "output_bytes",
    "gpu_resident_input_handoffs",
    "host_to_device_uploads",
    "host_to_device_uploaded_rows",
    "host_to_device_uploaded_bytes",
    "device_to_host_materializations",
    "device_to_host_materialized_rows",
    "device_to_host_materialized_bytes",
    "device_to_device_copies",
    "device_to_device_copied_bytes",
    "parquet_files",
    "parquet_bytes_known",
    "row_groups_selected",
    "read_columns",
    "resident_dataframe_peak_bytes",
    "allocator_high_water_bytes",
    "reservation_requested_bytes",
    "reservation_granted_bytes",
    "reservation_live_peak_bytes",
    "reservation_denied_count",
    "max_task_reservation_bytes",
    "max_task_observed_peak_bytes",
    "max_task_retained_bytes",
    "graph_reservation_rows",
    "graph_runtime_rows",
    "task_operator_rows",
    "task_runtime_rows",
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
    "dynamic_filter_post_decode_reduction_stage",
    "dynamic_filter_measured_reduction_status",
    "dynamic_filter_work_avoidance_status",
    "dynamic_filter_runtime_rows",
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
    "aggregate_libcudf_runtime_probe_count",
    "aggregate_libcudf_runtime_probe_elapsed_ns",
    "aggregate_libcudf_runtime_probe_input_rows",
    "aggregate_exact_cardinality_probe_count",
    "aggregate_exact_cardinality_probe_elapsed_ns",
    "aggregate_exact_cardinality_probe_input_rows",
    "aggregate_prepared_sort_groupby_elapsed_ns",
    "aggregate_encoded_key_partial_elapsed_ns",
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
    "aggregate_strategy_rows",
    "join_strategy_rows",
    "source_fetch_bytes",
    "source_fetch_count",
    "host_fetch_budget_wait_time_ms",
    "source_fetch_concurrency_wait_time_ms",
    "source_fetch_time_ms",
    "source_decode_time_ms",
    "source_diagnostics",
    "lakehouse_cache_fill_event_snapshot_rows",
    "plan_task_mapping_rows",
];
/// Execution-report TSV header. Byte columns are bytes unless a name says otherwise.
pub const NATIVE_DATAFUSION_EXECUTION_TSV_HEADER: &str = concat!(
    "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\t",
    "planning_outcome\treason_code\tunsupported_category\tstrict_bounded_contract_rows\t",
    "strict_bounded_compiled_graphs\tadaptive_pipeline_rows\tpost_native_plan_root\tpost_native_execution_path\t",
    "native_plan_fragment_count\thost_datafusion_boundary_count\trejected_before_execution\t",
    "admission_sequence\tadmission_wait_ms\tdevice_ordinal\tquery_grant_bytes\t",
    "native_runtime_started\truntime_outcome\truntime_failure\toutput_batches\toutput_rows\toutput_bytes\t",
    "gpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\t",
    "host_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\t",
    "device_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\t",
    "parquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\t",
    "allocator_high_water_bytes\treservation_requested_bytes\treservation_granted_bytes\t",
    "reservation_live_peak_bytes\treservation_denied_count\tmax_task_reservation_bytes\t",
    "max_task_observed_peak_bytes\tmax_task_retained_bytes\tgraph_reservation_rows\tgraph_runtime_rows\t",
    "task_operator_rows\ttask_runtime_rows\tdynamic_filter_produced_count\tdynamic_filter_skipped_count\t",
    "dynamic_filter_applied_count\tdynamic_filter_row_groups_pruned\tdynamic_filter_source_bytes_pruned\t",
    "dynamic_filter_reader_predicate_rows_before\tdynamic_filter_reader_predicate_rows_after\t",
    "dynamic_filter_reader_predicate_rows_filtered\tdynamic_filter_post_decode_rows_before\t",
    "dynamic_filter_post_decode_rows_after\tdynamic_filter_post_decode_rows_filtered\t",
    "dynamic_filter_source_post_decode_rows_filtered\tdynamic_filter_filter_post_decode_rows_filtered\t",
    "dynamic_filter_join_probe_post_decode_rows_filtered\tdynamic_filter_post_decode_reduction_stage\t",
    "dynamic_filter_measured_reduction_status\tdynamic_filter_work_avoidance_status\tdynamic_filter_runtime_rows\t",
    "source_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\t",
    "source_read_prefetches_started\tsource_read_prefetches_completed\tsource_read_prefetches_skipped\t",
    "source_read_prefetches_abandoned\tsource_cache_hits\tsource_cache_misses\tsource_cache_inserts\t",
    "source_cache_insert_skipped\tsource_cache_retained_bytes\trow_groups_planned\trow_groups_committed\t",
    "partial_aggregate_count\tpartial_aggregate_input_rows\tpartial_aggregate_rows\t",
    "partial_aggregate_retained_bytes\tfinal_merge_attempts\tfinal_merge_levels\tfinal_merge_max_fan_in\t",
    "aggregate_libcudf_runtime_probe_count\taggregate_libcudf_runtime_probe_elapsed_ns\t",
    "aggregate_libcudf_runtime_probe_input_rows\taggregate_exact_cardinality_probe_count\t",
    "aggregate_exact_cardinality_probe_elapsed_ns\taggregate_exact_cardinality_probe_input_rows\t",
    "aggregate_prepared_sort_groupby_elapsed_ns\taggregate_encoded_key_partial_elapsed_ns\t",
    "retry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\tadaptive_retry_count\t",
    "adaptive_spill_count\tadaptive_downgrade_count\tadaptive_memory_history_lookup_count\t",
    "adaptive_memory_history_hit_count\tadaptive_memory_history_miss_count\thash_join_build_count\t",
    "filtered_join_build_count\tone_shot_join_count\thash_join_cache_hits\thash_join_cache_misses\t",
    "hash_join_durable_cache_hits\thash_join_durable_cache_misses\thash_join_durable_cache_inserts\t",
    "hash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\t",
    "mixed_join_size_data_bytes\tone_shot_join_index_bytes\tone_shot_join_materialization_bytes\t",
    "one_shot_join_output_bytes\tjoin_gather_map_bytes\taggregate_strategy_rows\tjoin_strategy_rows\t",
    "source_fetch_bytes\tsource_fetch_count\thost_fetch_budget_wait_time_ms\t",
    "source_fetch_concurrency_wait_time_ms\tsource_fetch_time_ms\tsource_decode_time_ms\t",
    "source_diagnostics\tlakehouse_cache_fill_event_snapshot_rows\tplan_task_mapping_rows",
);

/// Shape-family sentinel used when no supported native plan family was found.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN: &str = "unsupported_native_physical_plan";
/// Shape family for sort over a local Parquet aggregate.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_LOCAL_PARQUET_AGGREGATE: &str = "sort_local_parquet_aggregate";
/// Shape family for sort/fetch over a local Parquet aggregate.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_LOCAL_PARQUET_AGGREGATE: &str = "sort_fetch_local_parquet_aggregate";
/// Shape family for sort-preserving merge/fetch over distributed local Parquet aggregates.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_PRESERVING_MERGE_FETCH_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE: &str =
    "sort_preserving_merge_fetch_local_parquet_distributed_aggregate";
/// Shape family for projected global aggregate over local Parquet input.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_AGGREGATE: &str =
    "projected_global_local_parquet_aggregate";
/// Shape family for projected global aggregate over distributed local Parquet input.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE: &str =
    "projected_global_local_parquet_distributed_aggregate";
/// Shape family for sort/fetch over hash-join aggregate input.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_AGGREGATE: &str = "sort_fetch_hash_join_aggregate";
/// Shape family for sort over hash-join aggregate input.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_AGGREGATE: &str = "sort_hash_join_aggregate";
/// Shape family for projected global aggregate over hash-join input.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_HASH_JOIN_AGGREGATE: &str =
    "projected_global_hash_join_aggregate";
/// Shape family for sort over mixed hash/nested-loop join aggregate input.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_AGGREGATE: &str =
    "sort_mixed_hash_nested_loop_join_aggregate";
/// Shape family for sort/fetch over hash-join input without aggregate.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_INPUT: &str = "sort_fetch_hash_join_input";
/// Shape family for sort over hash-join input without aggregate.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_INPUT: &str = "sort_hash_join_input";
/// Shape family for sort over mixed hash/nested-loop join input without aggregate.
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_INPUT: &str =
    "sort_mixed_hash_nested_loop_join_input";
