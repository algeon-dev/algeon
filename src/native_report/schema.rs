//! TSV schema constants for native planning, execution, and fallback reports.
//!
//! Headers define column order exactly. Schema versions should increase
//! whenever a consumer would need to change parsing logic or column meaning.

#![deny(missing_docs)]

/// Fallback-pressure TSV schema version.
pub const NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION: u32 = 6;
/// Fallback-pressure TSV header. Column order is stable for schema version 6.
pub const NATIVE_FALLBACK_PRESSURE_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tadaptive_pipeline_rows\tjoin_diagnostics\tsource_diagnostics\trejected_before_execution";

/// Planning-report TSV schema version.
pub const NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION: u32 = 11;
/// Planning-report TSV header. Nested row columns use stable key/value escaping.
pub const NATIVE_DATAFUSION_PLANNING_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tstrict_bounded_contract_rows\tstrict_bounded_compiled_graphs\tadaptive_pipeline_rows\tnative_optimizer\tedge_normalize_inserted\tjoin_diagnostics\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\tnative_plan_fragment_count\thost_datafusion_boundary_count\trejected_before_execution";

/// Execution-report TSV schema version.
pub const NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION: u32 = 26;
/// Execution-report TSV header. Byte columns are bytes unless a name says otherwise.
pub const NATIVE_DATAFUSION_EXECUTION_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\tplanning_outcome\treason_code\tunsupported_category\tstrict_bounded_contract_rows\tstrict_bounded_compiled_graphs\tadaptive_pipeline_rows\tpost_native_plan_root\tpost_native_execution_path\tnative_plan_fragment_count\thost_datafusion_boundary_count\trejected_before_execution\tnative_runtime_started\truntime_outcome\truntime_failure\toutput_batches\toutput_rows\toutput_bytes\tgpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\thost_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\tdevice_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\tparquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\tallocator_high_water_bytes\treservation_requested_bytes\treservation_granted_bytes\treservation_live_peak_bytes\treservation_denied_count\tmax_task_reservation_bytes\tmax_task_observed_peak_bytes\tmax_task_retained_bytes\tgraph_runtime_rows\ttask_operator_rows\ttask_runtime_rows\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\tsource_read_prefetches_started\tsource_read_prefetches_completed\tsource_read_prefetches_skipped\tsource_read_prefetches_abandoned\tsource_cache_hits\tsource_cache_misses\tsource_cache_inserts\tsource_cache_insert_skipped\tsource_cache_retained_bytes\trow_groups_planned\trow_groups_committed\tpartial_aggregate_count\tpartial_aggregate_input_rows\tpartial_aggregate_rows\tpartial_aggregate_retained_bytes\tfinal_merge_attempts\tfinal_merge_levels\tfinal_merge_max_fan_in\tretry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\tadaptive_retry_count\tadaptive_spill_count\tadaptive_downgrade_count\tadaptive_memory_history_lookup_count\tadaptive_memory_history_hit_count\tadaptive_memory_history_miss_count\thash_join_build_count\tfiltered_join_build_count\tone_shot_join_count\thash_join_cache_hits\thash_join_cache_misses\thash_join_durable_cache_hits\thash_join_durable_cache_misses\thash_join_durable_cache_inserts\thash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\tmixed_join_size_data_bytes\tone_shot_join_index_bytes\tone_shot_join_materialization_bytes\tone_shot_join_output_bytes\tjoin_gather_map_bytes\taggregate_strategy_rows\tjoin_strategy_rows\tsource_diagnostics";

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
