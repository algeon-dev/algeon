use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const CONCATENATE_RS: &str = include_str!("../../src/ops/concatenate.rs");
const BINARY_RS: &str = include_str!("../../src/ops/binary.rs");
const COPYING_RS: &str = include_str!("../../src/ops/copying.rs");
const DATETIME_RS: &str = include_str!("../../src/ops/datetime.rs");
const DICTIONARY_RS: &str = include_str!("../../src/ops/dictionary.rs");
const FILTER_RS: &str = include_str!("../../src/ops/filter.rs");
const GROUPBY_AGG_RS: &str = include_str!("../../src/ops/groupby/agg.rs");
const GROUPBY_SHIFT_RS: &str = include_str!("../../src/ops/groupby/shift.rs");
const HASHING_RS: &str = include_str!("../../src/ops/hashing.rs");
const IO_CSV_RS: &str = include_str!("../../src/io/csv.rs");
const IO_JSON_RS: &str = include_str!("../../src/io/json.rs");
const IO_PARQUET_RS: &str = include_str!("../../src/io/parquet.rs");
const JOIN_CONDITIONAL_RS: &str = include_str!("../../src/ops/join/conditional.rs");
const JOIN_FILTER_RS: &str = include_str!("../../src/ops/join/filter.rs");
const JOIN_HASH_RS: &str = include_str!("../../src/ops/join/hash.rs");
const JOIN_MIXED_RS: &str = include_str!("../../src/ops/join/mixed.rs");
const JOIN_MOD_RS: &str = include_str!("../../src/ops/join/mod.rs");
const LOOKUP_RS: &str = include_str!("../../src/ops/lookup.rs");
const PARTITIONING_RS: &str = include_str!("../../src/ops/partitioning.rs");
const REPLACE_RS: &str = include_str!("../../src/ops/replace.rs");
const REDUCTION_RS: &str = include_str!("../../src/ops/reduction.rs");
const SEARCH_RS: &str = include_str!("../../src/ops/search.rs");
const DATAFRAME_ROW_OPS_RS: &str = include_str!("../../src/frame/dataframe/row_ops.rs");
const EXECUTION_DOMAIN_RS: &str = include_str!("../../src/runtime/execution_domain.rs");
const EXPR_RS: &str = include_str!("../../src/ops/expr.rs");
const SERIES_CONTEXT_RS: &str = include_str!("../../src/frame/series/context.rs");
const RUNTIME_BUDGET_RS: &str = include_str!("../../src/runtime/budget.rs");
const SORTING_RS: &str = include_str!("../../src/ops/sorting/mod.rs");
const STREAM_COMPACTION_RS: &str = include_str!("../../src/ops/stream_compaction/mod.rs");
const STRINGS_RS: &str = include_str!("../../src/ops/strings.rs");
const UNARY_RS: &str = include_str!("../../src/ops/unary.rs");

const SAFE_LAYER_EXPLICIT_FFI_WITHOUT_EX_SUFFIX: &[(&str, &str)] = &[
    (
        "cudf_arrow_device_column_export",
        "execution-sensitive Arrow Device ABI export; accepts explicit cudf_exec_options_t despite legacy name",
    ),
    (
        "cudf_arrow_device_column_import",
        "execution-sensitive Arrow Device ABI import; accepts explicit cudf_exec_options_t despite legacy name",
    ),
    (
        "cudf_arrow_device_table_export",
        "execution-sensitive Arrow Device ABI export; accepts explicit cudf_exec_options_t despite legacy name",
    ),
    (
        "cudf_arrow_device_table_import",
        "execution-sensitive Arrow Device ABI import; accepts explicit cudf_exec_options_t despite legacy name",
    ),
];

const SAFE_LAYER_NON_EXECUTION_FFI_ALLOWLIST: &[(&str, &str)] = &[
    ("cudf_aggregation_create", "descriptor-only aggregation factory"),
    ("cudf_aggregation_create_count", "descriptor-only aggregation factory"),
    ("cudf_aggregation_create_std", "descriptor-only aggregation factory"),
    ("cudf_aggregation_create_variance", "descriptor-only aggregation factory"),
    ("cudf_aggregation_destroy", "pure lifecycle"),
    ("cudf_arrow_device_array_destroy", "pure lifecycle"),
    ("cudf_arrow_device_array_device_id", "metadata-only Arrow Device ABI accessor"),
    ("cudf_arrow_device_array_device_type", "metadata-only Arrow Device ABI accessor"),
    ("cudf_arrow_device_array_has_sync_event", "metadata-only Arrow Device ABI accessor"),
    ("cudf_arrow_device_array_sync_event_raw", "raw-pointer-only Arrow Device ABI accessor"),
    ("cudf_arrow_device_schema_destroy", "pure lifecycle"),
    ("cudf_arrow_device_table_column_name", "metadata-only Arrow Device ABI accessor"),
    ("cudf_arrow_device_table_fixed_width_values_ptr", "raw-pointer-only Arrow Device ABI accessor"),
    ("cudf_arrow_device_table_num_columns", "metadata-only Arrow Device ABI accessor"),
    ("cudf_ast_column_reference_create", "descriptor-only AST construction"),
    ("cudf_ast_expression_destroy", "pure lifecycle"),
    ("cudf_ast_literal_create", "descriptor-only AST construction"),
    ("cudf_ast_operation_binary_create", "descriptor-only AST construction"),
    ("cudf_ast_operation_unary_create", "descriptor-only AST construction"),
    ("cudf_column_adopt_device_buffer", "ownership transfer from provenance-bound device buffer"),
    ("cudf_column_adopt_device_buffer_nullable", "ownership transfer from provenance-bound device buffers"),
    ("cudf_column_create_from_owned_device_transfer", "ownership transfer from external device export"),
    ("cudf_column_destroy", "provenance-bound lifecycle"),
    ("cudf_column_device_allocated_size", "metadata-only allocation accounting"),
    ("cudf_column_view", "descriptor-only borrowed view construction"),
    ("cudf_column_view_child", "descriptor-only borrowed child view construction"),
    ("cudf_column_view_data_ptr", "raw-pointer-only accessor"),
    ("cudf_column_view_data_type", "metadata-only accessor"),
    ("cudf_column_view_destroy", "pure lifecycle"),
    ("cudf_column_view_nullable", "metadata-only accessor"),
    ("cudf_column_view_null_count", "metadata-only stored null-count accessor"),
    ("cudf_column_view_null_mask_ptr", "raw-pointer-only accessor"),
    ("cudf_column_view_num_children", "metadata-only accessor"),
    ("cudf_column_view_offset", "metadata-only accessor"),
    ("cudf_column_view_size", "metadata-only accessor"),
    ("cudf_conditional_full_join_size_data_destroy", "provenance-bound lifecycle"),
    ("cudf_conditional_full_join_size_data_device_allocated_size", "metadata-only cached join planning accessor"),
    ("cudf_conditional_full_join_size_data_gather_map_size", "metadata-only cached join planning accessor"),
    ("cudf_conditional_full_join_size_data_output_size", "metadata-only cached join planning accessor"),
    ("cudf_cuda_get_device", "runtime device metadata/guard support"),
    ("cudf_cuda_get_device_count", "runtime device metadata"),
    ("cudf_cuda_device_pci_bus_id", "runtime device metadata"),
    ("cudf_cuda_pointer_get_device", "runtime pointer metadata"),
    ("cudf_cuda_set_device", "runtime device guard support"),
    ("cudf_data_type_create", "descriptor-only data type construction"),
    ("cudf_data_type_create_fixed_point", "descriptor-only data type construction"),
    ("cudf_data_type_destroy", "pure lifecycle"),
    ("cudf_device_attribute", "runtime device metadata"),
    ("cudf_device_buffer_data", "raw-pointer-only accessor"),
    ("cudf_device_buffer_destroy", "provenance-bound lifecycle"),
    ("cudf_device_buffer_size", "metadata-only accessor"),
    ("cudf_device_memory_info", "runtime device memory metadata"),
    ("cudf_dictionary_indices", "descriptor-only dictionary child view accessor"),
    ("cudf_dictionary_keys", "descriptor-only dictionary child view accessor"),
    ("cudf_distinct_hash_join_destroy", "provenance-bound lifecycle"),
    ("cudf_event_create", "runtime event lifecycle"),
    ("cudf_event_destroy", "runtime event lifecycle"),
    ("cudf_event_is_ready", "runtime event state query with explicit event handle"),
    ("cudf_event_raw", "raw-pointer-only runtime event accessor"),
    ("cudf_event_record", "runtime event operation with explicit event and stream handles"),
    ("cudf_event_synchronize", "runtime synchronization with explicit event handle"),
    ("cudf_filtered_join_destroy", "provenance-bound lifecycle"),
    ("cudf_filtered_join_device_allocated_size", "metadata-only cached join planning accessor"),
    ("cudf_filtered_join_probe_destroy", "provenance-bound lifecycle"),
    ("cudf_filtered_join_probe_device_allocated_size", "metadata-only cached join planning accessor"),
    ("cudf_filtered_join_probe_output_size", "metadata-only cached join planning accessor"),
    ("cudf_free_column_array", "pure lifecycle for returned handle arrays"),
    ("cudf_gds_driver_available", "process-local GDS capability probe with no stream or allocator inputs"),
    ("cudf_groupby_scan_aggregation_create", "descriptor-only aggregation factory"),
    ("cudf_groupby_scan_aggregation_destroy", "pure lifecycle"),
    ("cudf_prepared_groupby_destroy", "pure lifecycle"),
    ("cudf_hash_join_destroy", "provenance-bound lifecycle"),
    ("cudf_hash_join_device_allocated_size", "metadata-only cached join planning accessor"),
    ("cudf_io_free_column_names", "pure lifecycle for IO metadata"),
    ("cudf_io_probe_remote_parquet_datasource", "metadata-only remote datasource capability probe"),
    ("cudf_io_remote_parquet_kvikio_remote_io_enabled", "metadata-only remote Parquet build capability probe"),
    ("cudf_is_supported_cast", "metadata-only cast support query"),
    ("cudf_last_error_message", "error metadata accessor"),
    ("cudf_memory_resource_create_budget_tracking", "runtime memory-resource lifecycle"),
    ("cudf_memory_resource_create_cuda", "runtime memory-resource lifecycle"),
    ("cudf_memory_resource_create_failure_callback", "runtime memory-resource lifecycle"),
    ("cudf_memory_resource_create_limiting", "runtime memory-resource lifecycle"),
    ("cudf_memory_resource_create_pool", "runtime memory-resource lifecycle"),
    ("cudf_memory_resource_destroy", "runtime memory-resource lifecycle"),
    ("cudf_memory_resource_raw", "raw-pointer-only memory-resource accessor"),
    ("cudf_memory_resource_usage", "metadata-only memory-resource accounting"),
    ("cudf_memory_manager_attach_reservation", "runtime stream-reservation lifecycle"),
    ("cudf_memory_manager_failure_callback_state", "runtime device memory-manager state query"),
    ("cudf_memory_manager_for_device", "runtime device memory-manager lifecycle"),
    ("cudf_memory_manager_for_device_with_failure_callback", "runtime device memory-manager lifecycle"),
    ("cudf_memory_manager_resource", "runtime device memory-manager lifecycle"),
    ("cudf_reservation_detach", "runtime stream-reservation lifecycle"),
    ("cudf_reservation_resize", "runtime stream-reservation lifecycle"),
    ("cudf_reservation_usage", "metadata-only stream-reservation accounting"),
    ("cudf_mixed_join_size_data_destroy", "provenance-bound lifecycle"),
    ("cudf_mixed_join_size_data_device_allocated_size", "metadata-only cached join planning accessor"),
    ("cudf_mixed_join_size_data_gather_map_size", "metadata-only cached join planning accessor"),
    ("cudf_mixed_join_size_data_output_size", "metadata-only cached join planning accessor"),
    ("cudf_raw_cuda_event_record", "runtime event operation with explicit raw event and stream handles"),
    ("cudf_raw_cuda_stream_wait_event", "runtime stream operation with explicit raw event and stream handles"),
    ("cudf_reduce_aggregation_create", "descriptor-only aggregation factory"),
    ("cudf_reduce_aggregation_create_nunique", "descriptor-only aggregation factory"),
    ("cudf_reduce_aggregation_create_std", "descriptor-only aggregation factory"),
    ("cudf_reduce_aggregation_create_variance", "descriptor-only aggregation factory"),
    ("cudf_reduce_aggregation_destroy", "pure lifecycle"),
    ("cudf_scalar_data_type", "metadata-only scalar accessor"),
    ("cudf_scalar_destroy", "provenance-bound lifecycle"),
    ("cudf_scan_aggregation_create", "descriptor-only aggregation factory"),
    ("cudf_scan_aggregation_destroy", "pure lifecycle"),
    ("cudf_stream_create", "runtime stream lifecycle"),
    ("cudf_stream_destroy", "runtime stream lifecycle"),
    ("cudf_stream_is_ready", "runtime stream state query with explicit stream handle"),
    ("cudf_stream_raw", "raw-pointer-only runtime stream accessor"),
    ("cudf_stream_synchronize", "runtime synchronization with explicit stream handle"),
    ("cudf_stream_wait_event", "runtime stream operation with explicit event and stream handles"),
    ("cudf_string_free", "pure lifecycle for returned host string"),
    ("cudf_table_create", "ownership transfer from existing columns"),
    ("cudf_table_create_from_owned_device_transfer", "ownership transfer from external device export"),
    ("cudf_table_destroy", "provenance-bound lifecycle"),
    ("cudf_table_release_columns", "ownership transfer from table to columns"),
    ("cudf_table_view", "descriptor-only borrowed view construction"),
    ("cudf_table_view_column", "descriptor-only borrowed column view construction"),
    ("cudf_table_view_destroy", "pure lifecycle"),
    ("cudf_table_view_from_column_views", "descriptor-only borrowed view construction"),
    ("cudf_table_view_num_columns", "metadata-only accessor"),
    ("cudf_table_view_num_rows", "metadata-only accessor"),
    ("cudf_table_view_select_columns", "descriptor-only borrowed view construction"),
    ("cudf_test_arrow_device_array_set_device_type", "test hook"),
    ("cudf_test_column_view_create_synthetic_string_without_children", "test hook"),
    ("cudf_test_event_reset_observation", "test hook"),
    ("cudf_test_event_stream_wait_call_count_for_stream_raw", "test hook"),
    ("cudf_test_memory_resource_allocation_count", "test hook"),
    ("cudf_test_memory_resource_create_stats", "test hook"),
    ("cudf_test_memory_resource_raw", "test hook"),
    ("cudf_test_stream_handle_value", "test hook"),
    ("cudf_type_size", "metadata-only type descriptor query"),
];

#[test]
fn concatenate_helpers_require_explicit_execution_options() {
    for (function, end_marker) in [
        ("concatenate_columns_with_exec", "/// Concatenate multiple tables"),
        ("concatenate_tables_with_exec", "pub(crate) fn trigger_concatenate_column_count_limit_error_for_test"),
    ] {
        let body = function_region(CONCATENATE_RS, &format!("pub(crate) fn {function}"), end_marker);
        assert!(
            body.contains("exec_options: &cudf_sys::cudf_exec_options_t"),
            "{function} must require an explicit options reference"
        );
        assert!(!body.contains("exec_options: Option"), "{function} must not carry optional execution options");
        assert!(
            !body.contains("cudf_sys::cudf_concatenate_columns(")
                && !body.contains("cudf_sys::cudf_concatenate_tables("),
            "{function} must not call ambient non-_ex concatenate FFI"
        );
    }
}

#[test]
fn binary_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_binary_operation_vv_ex",
        "cudf_sys::cudf_binary_operation_vs_ex",
        "cudf_sys::cudf_binary_operation_sv_ex",
    ] {
        assert!(BINARY_RS.contains(required), "binary helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_binary_operation_vv(",
        "cudf_sys::cudf_binary_operation_vs(",
        "cudf_sys::cudf_binary_operation_sv(",
        "execution_context_provided",
    ] {
        assert!(!BINARY_RS.contains(forbidden), "binary helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn execution_budget_token_uses_stream_bound_reservation_not_per_token_memory_resource() {
    for required in [
        "DeviceMemoryManager::for_device(runtime.device())",
        "manager.reserve_stream(runtime.stream()",
        "StreamBoundReservation",
        "reservation_usage",
    ] {
        assert!(RUNTIME_BUDGET_RS.contains(required), "budget token must retain `{required}`");
    }

    for forbidden in [
        "budget_tracking_limit",
        "ExecutionBudgetToken::new(upstream",
        "HardMemoryResourceLimit",
        "hard_memory_resource_limit",
    ] {
        assert!(
            !RUNTIME_BUDGET_RS.contains(forbidden),
            "budget token must not return to per-token memory-resource budgeting via `{forbidden}`"
        );
    }
}

#[test]
fn copying_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_gather_ex",
        "cudf_sys::cudf_gather_device_allocated_size_ex",
        "cudf_sys::cudf_scatter_ex",
        "cudf_sys::cudf_copy_if_else_vv_ex",
        "cudf_sys::cudf_shift_ex",
    ] {
        assert!(COPYING_RS.contains(required), "copying helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_gather(",
        "cudf_sys::cudf_scatter(",
        "cudf_sys::cudf_copy_if_else_vv(",
        "cudf_sys::cudf_shift(",
        "execution_context_provided",
    ] {
        assert!(!COPYING_RS.contains(forbidden), "copying helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn unary_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_unary_operation_ex",
        "cudf_sys::cudf_round_half_up_ex",
        "cudf_sys::cudf_is_null_ex",
        "cudf_sys::cudf_is_valid_ex",
        "cudf_sys::cudf_cast_ex",
        "cudf_sys::cudf_is_nan_ex",
        "cudf_sys::cudf_is_not_nan_ex",
    ] {
        assert!(UNARY_RS.contains(required), "unary helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_unary_operation(",
        "cudf_sys::cudf_round_half_up(",
        "cudf_sys::cudf_is_null(",
        "cudf_sys::cudf_is_valid(",
        "cudf_sys::cudf_cast(",
        "cudf_sys::cudf_is_nan(",
        "cudf_sys::cudf_is_not_nan(",
        "execution_context_provided",
    ] {
        assert!(!UNARY_RS.contains(forbidden), "unary helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn strings_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_strings_to_lower_ex",
        "cudf_sys::cudf_strings_to_upper_ex",
        "cudf_sys::cudf_strings_contains_ex",
        "cudf_sys::cudf_strings_like_ex",
        "cudf_sys::cudf_strings_replace_with_backrefs_ex",
        "cudf_sys::cudf_strings_starts_with_ex",
        "cudf_sys::cudf_strings_ends_with_ex",
        "cudf_sys::cudf_strings_slice_ex",
        "cudf_sys::cudf_strings_concatenate_ex",
        "cudf_sys::cudf_strings_strip_ex",
        "cudf_sys::cudf_strings_count_characters_ex",
        "cudf_sys::cudf_strings_count_bytes_ex",
        "cudf_sys::cudf_strings_try_to_integers_ex",
    ] {
        assert!(STRINGS_RS.contains(required), "strings helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_strings_to_lower(",
        "cudf_sys::cudf_strings_to_upper(",
        "cudf_sys::cudf_strings_contains(",
        "cudf_sys::cudf_strings_like(",
        "cudf_sys::cudf_strings_replace_with_backrefs(",
        "cudf_sys::cudf_strings_starts_with(",
        "cudf_sys::cudf_strings_ends_with(",
        "cudf_sys::cudf_strings_slice(",
        "cudf_sys::cudf_strings_concatenate(",
        "cudf_sys::cudf_strings_strip(",
        "cudf_sys::cudf_strings_count_characters(",
        "cudf_sys::cudf_strings_count_bytes(",
        "cudf_sys::cudf_strings_try_to_integers(",
        "execution_context_provided",
    ] {
        assert!(!STRINGS_RS.contains(forbidden), "strings helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn groupby_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_groupby_aggregate_with_keys_are_sorted_ex",
        "cudf_sys::cudf_prepared_groupby_create_ex",
        "cudf_sys::cudf_prepared_groupby_prepare_groups_ex",
        "cudf_sys::cudf_prepared_groupby_aggregate_ex",
        "cudf_sys::cudf_groupby_scan_ex",
    ] {
        assert!(GROUPBY_AGG_RS.contains(required), "groupby aggregation helpers must retain `{required}`");
    }

    for required in ["exec_options: &cudf_sys::cudf_exec_options_t", "cudf_sys::cudf_groupby_shift_ex"] {
        assert!(GROUPBY_SHIFT_RS.contains(required), "groupby shift helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_groupby_aggregate(",
        "cudf_sys::cudf_prepared_groupby_create(",
        "cudf_sys::cudf_prepared_groupby_prepare_groups(",
        "cudf_sys::cudf_prepared_groupby_aggregate(",
        "cudf_sys::cudf_groupby_scan(",
        "execution_context_provided",
    ] {
        assert!(
            !GROUPBY_AGG_RS.contains(forbidden),
            "groupby aggregation helpers must not retain `{forbidden}` fallback behavior"
        );
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_groupby_shift(",
        "execution_context_provided",
    ] {
        assert!(
            !GROUPBY_SHIFT_RS.contains(forbidden),
            "groupby shift helpers must not retain `{forbidden}` fallback behavior"
        );
    }
}

#[test]
fn hash_join_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_inner_join_ex",
        "cudf_sys::cudf_left_join_ex",
        "cudf_sys::cudf_full_join_ex",
        "cudf_sys::cudf_cross_join_ex",
        "cudf_sys::cudf_projected_cross_join_ex",
        "cudf_sys::cudf_projected_cross_join_device_allocated_size_ex",
        "cudf_sys::cudf_inner_join_indices_ex",
        "cudf_sys::cudf_left_join_indices_ex",
        "cudf_sys::cudf_full_join_indices_ex",
        "cudf_sys::cudf_left_semi_join_indices_ex",
        "cudf_sys::cudf_left_anti_join_indices_ex",
        "cudf_sys::cudf_hash_join_create_ex",
        "cudf_sys::cudf_hash_join_pre_build_reservation_size_ex",
        "cudf_sys::cudf_hash_join_inner_join_ex",
        "cudf_sys::cudf_hash_join_inner_join_with_output_size_ex",
        "cudf_sys::cudf_hash_join_inner_join_size_ex",
        "cudf_sys::cudf_hash_join_left_join_ex",
        "cudf_sys::cudf_hash_join_left_join_with_output_size_ex",
        "cudf_sys::cudf_hash_join_left_join_size_ex",
        "cudf_sys::cudf_hash_join_full_join_ex",
        "cudf_sys::cudf_hash_join_full_join_with_output_size_ex",
        "cudf_sys::cudf_hash_join_full_join_size_ex",
    ] {
        assert!(JOIN_HASH_RS.contains(required), "hash join helpers must retain `{required}`");
    }

    for forbidden in [
        "exec_options: Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "match (join_type, exec_options)",
        "match (semi_join, exec_options)",
        "match (join_type, output_size, exec_options)",
        "cudf_sys::cudf_inner_join(",
        "cudf_sys::cudf_left_join(",
        "cudf_sys::cudf_full_join(",
        "cudf_sys::cudf_cross_join(",
        "cudf_sys::cudf_projected_cross_join(",
        "cudf_sys::cudf_inner_join_indices(",
        "cudf_sys::cudf_left_join_indices(",
        "cudf_sys::cudf_full_join_indices(",
        "cudf_sys::cudf_left_semi_join_indices(",
        "cudf_sys::cudf_left_anti_join_indices(",
        "cudf_sys::cudf_hash_join_create(",
        "cudf_sys::cudf_hash_join_pre_build_reservation_size(",
        "cudf_sys::cudf_hash_join_inner_join(",
        "cudf_sys::cudf_hash_join_inner_join_with_output_size(",
        "cudf_sys::cudf_hash_join_inner_join_size(",
        "cudf_sys::cudf_hash_join_left_join(",
        "cudf_sys::cudf_hash_join_left_join_with_output_size(",
        "cudf_sys::cudf_hash_join_left_join_size(",
        "cudf_sys::cudf_hash_join_full_join(",
        "cudf_sys::cudf_hash_join_full_join_with_output_size(",
        "cudf_sys::cudf_hash_join_full_join_size(",
        "execution_context_provided",
    ] {
        assert!(!JOIN_HASH_RS.contains(forbidden), "hash join helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn conditional_mixed_and_filter_join_helpers_require_explicit_execution_options() {
    for (name, body, required) in [
        (
            "conditional join",
            JOIN_CONDITIONAL_RS,
            [
                "exec_options: &cudf_sys::cudf_exec_options_t",
                "cudf_sys::cudf_conditional_inner_join_indices_ex",
                "cudf_sys::cudf_conditional_left_join_indices_ex",
                "cudf_sys::cudf_conditional_full_join_indices_ex",
                "cudf_sys::cudf_conditional_inner_join_indices_with_output_size_ex",
                "cudf_sys::cudf_conditional_left_join_indices_with_output_size_ex",
                "cudf_sys::cudf_conditional_left_semi_join_indices_ex",
                "cudf_sys::cudf_conditional_left_anti_join_indices_ex",
                "cudf_sys::cudf_conditional_left_semi_join_indices_with_output_size_ex",
                "cudf_sys::cudf_conditional_left_anti_join_indices_with_output_size_ex",
                "cudf_sys::cudf_conditional_inner_join_size_ex",
                "cudf_sys::cudf_conditional_left_join_size_ex",
                "cudf_sys::cudf_conditional_full_join_begin_size_data_ex",
                "cudf_sys::cudf_conditional_left_semi_join_size_ex",
                "cudf_sys::cudf_conditional_left_anti_join_size_ex",
            ]
            .as_slice(),
        ),
        (
            "mixed join",
            JOIN_MIXED_RS,
            [
                "exec_options: &cudf_sys::cudf_exec_options_t",
                "cudf_sys::cudf_mixed_inner_join_indices_ex",
                "cudf_sys::cudf_mixed_left_join_indices_ex",
                "cudf_sys::cudf_mixed_full_join_indices_ex",
                "cudf_sys::cudf_mixed_inner_join_begin_size_data_ex",
                "cudf_sys::cudf_mixed_left_join_begin_size_data_ex",
                "cudf_sys::cudf_mixed_full_join_begin_size_data_ex",
                "cudf_sys::cudf_mixed_left_semi_join_begin_size_data_ex",
                "cudf_sys::cudf_mixed_left_anti_join_begin_size_data_ex",
                "cudf_sys::cudf_mixed_left_semi_join_indices_ex",
                "cudf_sys::cudf_mixed_left_anti_join_indices_ex",
            ]
            .as_slice(),
        ),
        (
            "filter join",
            JOIN_FILTER_RS,
            [
                "cudf_sys::cudf_filter_join_indices_ex",
                "cudf_sys::cudf_filtered_join_create_ex",
                "cudf_sys::cudf_filtered_join_left_semi_join_size_ex",
                "cudf_sys::cudf_filtered_join_left_anti_join_size_ex",
                "cudf_sys::cudf_filtered_join_left_semi_join_with_output_size_ex",
                "cudf_sys::cudf_filtered_join_left_anti_join_with_output_size_ex",
                "cudf_sys::cudf_filtered_join_begin_left_semi_probe_ex",
                "cudf_sys::cudf_filtered_join_begin_left_anti_probe_ex",
                "cudf_sys::cudf_filtered_join_probe_materialize_indices_ex",
            ]
            .as_slice(),
        ),
    ] {
        for required in required {
            assert!(body.contains(required), "{name} helpers must retain `{required}`");
        }
    }

    for (name, body, forbidden) in [
        (
            "conditional join",
            JOIN_CONDITIONAL_RS,
            [
                "exec_options: Option<&cudf_sys::cudf_exec_options_t>",
                "match (join_type, exec_options)",
                "match (semi_join, output_size, exec_options)",
                "match (semi_join, exec_options)",
                "match exec_options",
                "cudf_sys::cudf_conditional_inner_join_indices(",
                "cudf_sys::cudf_conditional_left_join_indices(",
                "cudf_sys::cudf_conditional_full_join_indices(",
                "cudf_sys::cudf_conditional_inner_join_indices_with_output_size(",
                "cudf_sys::cudf_conditional_left_join_indices_with_output_size(",
                "cudf_sys::cudf_conditional_left_semi_join_indices(",
                "cudf_sys::cudf_conditional_left_anti_join_indices(",
                "cudf_sys::cudf_conditional_left_semi_join_indices_with_output_size(",
                "cudf_sys::cudf_conditional_left_anti_join_indices_with_output_size(",
                "cudf_sys::cudf_conditional_inner_join_size(",
                "cudf_sys::cudf_conditional_left_join_size(",
                "cudf_sys::cudf_conditional_full_join_begin_size_data(",
                "cudf_sys::cudf_conditional_left_semi_join_size(",
                "cudf_sys::cudf_conditional_left_anti_join_size(",
            ]
            .as_slice(),
        ),
        (
            "mixed join",
            JOIN_MIXED_RS,
            [
                "exec_options: Option<&cudf_sys::cudf_exec_options_t>",
                "match (join_type, exec_options)",
                "match (semi_join, exec_options)",
                "cudf_sys::cudf_mixed_inner_join_indices(",
                "cudf_sys::cudf_mixed_left_join_indices(",
                "cudf_sys::cudf_mixed_full_join_indices(",
                "cudf_sys::cudf_mixed_inner_join_begin_size_data(",
                "cudf_sys::cudf_mixed_left_join_begin_size_data(",
                "cudf_sys::cudf_mixed_full_join_begin_size_data(",
                "cudf_sys::cudf_mixed_left_semi_join_begin_size_data(",
                "cudf_sys::cudf_mixed_left_anti_join_begin_size_data(",
                "cudf_sys::cudf_mixed_left_semi_join_indices(",
                "cudf_sys::cudf_mixed_left_anti_join_indices(",
            ]
            .as_slice(),
        ),
        (
            "filter join",
            JOIN_FILTER_RS,
            ["cudf_sys::cudf_filter_join_indices(", "cudf_sys::cudf_filtered_join_probe_materialize_indices("]
                .as_slice(),
        ),
    ] {
        for forbidden in forbidden {
            assert!(!body.contains(forbidden), "{name} helpers must not retain `{forbidden}` fallback behavior");
        }
    }
}

#[test]
fn sorting_helpers_require_explicit_execution_options() {
    let body = function_region(SORTING_RS, "pub(crate) struct SortExecution", "/// Sort the rows of a table");

    assert!(
        body.contains("exec_options: &'a cudf_sys::cudf_exec_options_t"),
        "SortExecution must require an explicit options reference"
    );
    assert!(
        !body.contains("Option<&'a cudf_sys::cudf_exec_options_t>"),
        "SortExecution must not carry optional execution options"
    );

    for forbidden in [
        "cudf_sys::cudf_sort(",
        "cudf_sys::cudf_stable_sort(",
        "cudf_sys::cudf_sort_by_key(",
        "cudf_sys::cudf_stable_sort_by_key(",
        "cudf_sys::cudf_sorted_order(",
        "execution_context_provided",
    ] {
        assert!(!body.contains(forbidden), "sorting helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn safe_layer_helpers_do_not_reintroduce_optional_exec_options() {
    for (name, source) in [
        ("dataframe row ops", DATAFRAME_ROW_OPS_RS),
        ("expression evaluation", EXPR_RS),
        ("series execution context", SERIES_CONTEXT_RS),
        ("lookup", LOOKUP_RS),
        ("execution domain", EXECUTION_DOMAIN_RS),
    ] {
        for forbidden in [
            "explicit_exec_options_raw",
            "Option<cudf_sys::cudf_exec_options_t>",
            "Option<&cudf_sys::cudf_exec_options_t>",
            "exec_options.as_ref().map_or(ptr::null()",
            "exec_options.is_some()",
            "unwrap_or(execution.exec_options())",
        ] {
            assert!(!source.contains(forbidden), "{name} must not retain `{forbidden}` fallback plumbing");
        }
    }
}

#[test]
fn safe_layer_diagnostics_do_not_use_ambiguous_execution_context_provided_label() {
    for (name, source) in [
        ("binary", BINARY_RS),
        ("copying", COPYING_RS),
        ("datetime", DATETIME_RS),
        ("dictionary", DICTIONARY_RS),
        ("filter", FILTER_RS),
        ("groupby aggregate", GROUPBY_AGG_RS),
        ("groupby shift", GROUPBY_SHIFT_RS),
        ("hashing", HASHING_RS),
        ("join conditional", JOIN_CONDITIONAL_RS),
        ("join filter", JOIN_FILTER_RS),
        ("join hash", JOIN_HASH_RS),
        ("join mixed", JOIN_MIXED_RS),
        ("join mod", JOIN_MOD_RS),
        ("lookup", LOOKUP_RS),
        ("partitioning", PARTITIONING_RS),
        ("reduction", REDUCTION_RS),
        ("replace", REPLACE_RS),
        ("search", SEARCH_RS),
        ("sorting", SORTING_RS),
        ("stream compaction", STREAM_COMPACTION_RS),
        ("strings", STRINGS_RS),
        ("unary", UNARY_RS),
    ] {
        assert!(
            !source.contains("execution_context_provided"),
            "{name} diagnostics must distinguish retained execution authority from user-provided overrides"
        );
    }
}

#[test]
fn stream_compaction_helpers_require_explicit_execution_options() {
    let body = function_region(STREAM_COMPACTION_RS, "pub(crate) struct StreamCompactionExecution", "/// Drop rows");

    assert!(
        body.contains("exec_options: &'a cudf_sys::cudf_exec_options_t"),
        "StreamCompactionExecution must require an explicit options reference"
    );
    assert!(
        body.contains("exec_options: &cudf_sys::cudf_exec_options_t"),
        "drop_nulls_impl must require an explicit options reference"
    );

    for forbidden in [
        "Option<&'a cudf_sys::cudf_exec_options_t>",
        "Option<&cudf_sys::cudf_exec_options_t>",
        "cudf_sys::cudf_drop_nulls(",
        "cudf_sys::cudf_distinct(",
        "execution_context_provided",
    ] {
        assert!(!body.contains(forbidden), "stream compaction helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn filter_helpers_require_explicit_execution_options() {
    let body = FILTER_RS;

    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_apply_boolean_mask_ex",
        "cudf_sys::cudf_filter_rows_by_predicate_ex",
    ] {
        assert!(body.contains(required), "filter helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_apply_boolean_mask(",
        "cudf_sys::cudf_filter_rows_by_predicate(",
        "execution_context_provided",
    ] {
        assert!(!body.contains(forbidden), "filter helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn hashing_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_hashing_murmurhash3_x86_32_ex",
        "cudf_sys::cudf_hashing_xxhash_64_ex",
    ] {
        assert!(HASHING_RS.contains(required), "hashing helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_hashing_murmurhash3_x86_32(",
        "cudf_sys::cudf_hashing_xxhash_64(",
        "execution_context_provided",
    ] {
        assert!(!HASHING_RS.contains(forbidden), "hashing helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn search_helpers_require_explicit_execution_options() {
    for required in ["exec_options: &cudf_sys::cudf_exec_options_t", "cudf_sys::cudf_contains_ex"] {
        assert!(SEARCH_RS.contains(required), "search helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_contains(",
        "execution_context_provided",
    ] {
        assert!(!SEARCH_RS.contains(forbidden), "search helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn replace_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_replace_nulls_scalar_ex",
        "cudf_sys::cudf_replace_nulls_column_ex",
        "cudf_sys::cudf_replace_nulls_policy_ex",
    ] {
        assert!(REPLACE_RS.contains(required), "replace helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_replace_nulls_scalar(",
        "cudf_sys::cudf_replace_nulls_column(",
        "cudf_sys::cudf_replace_nulls_policy(",
        "execution_context_provided",
    ] {
        assert!(!REPLACE_RS.contains(forbidden), "replace helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn partitioning_helpers_require_explicit_execution_options() {
    for required in ["exec_options: &cudf_sys::cudf_exec_options_t", "cudf_sys::cudf_hash_partition_ex"] {
        assert!(PARTITIONING_RS.contains(required), "partitioning helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_hash_partition(",
        "execution_context_provided",
    ] {
        assert!(
            !PARTITIONING_RS.contains(forbidden),
            "partitioning helpers must not retain `{forbidden}` fallback behavior"
        );
    }
}

#[test]
fn datetime_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_datetime_floor_ex",
        "cudf_sys::cudf_datetime_extract_component_ex",
        "cudf_sys::cudf_datetime_add_calendrical_months_scalar_ex",
    ] {
        assert!(DATETIME_RS.contains(required), "datetime helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_datetime_floor(",
        "cudf_sys::cudf_datetime_extract_component(",
        "cudf_sys::cudf_datetime_add_calendrical_months_scalar(",
        "execution_context_provided",
    ] {
        assert!(!DATETIME_RS.contains(forbidden), "datetime helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn reduction_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_reduce_ex",
        "cudf_sys::cudf_scan_ex",
        "cudf_sys::cudf_minmax_ex",
    ] {
        assert!(REDUCTION_RS.contains(required), "reduction helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_reduce(",
        "cudf_sys::cudf_scan(",
        "cudf_sys::cudf_minmax(",
        "execution_context_provided",
    ] {
        assert!(!REDUCTION_RS.contains(forbidden), "reduction helpers must not retain `{forbidden}` fallback behavior");
    }
}

#[test]
fn dictionary_helpers_require_explicit_execution_options() {
    for required in [
        "exec_options: &cudf_sys::cudf_exec_options_t",
        "cudf_sys::cudf_dictionary_encode_ex",
        "cudf_sys::cudf_dictionary_decode_ex",
        "cudf_sys::cudf_dictionary_set_keys_ex",
        "cudf_sys::cudf_dictionary_match_dictionaries_ex",
    ] {
        assert!(DICTIONARY_RS.contains(required), "dictionary helpers must retain `{required}`");
    }

    for forbidden in [
        "Option<&cudf_sys::cudf_exec_options_t>",
        "match exec_options",
        "cudf_sys::cudf_dictionary_encode(",
        "cudf_sys::cudf_dictionary_decode(",
        "cudf_sys::cudf_dictionary_set_keys(",
        "cudf_sys::cudf_dictionary_match_dictionaries(",
        "execution_context_provided",
    ] {
        assert!(
            !DICTIONARY_RS.contains(forbidden),
            "dictionary helpers must not retain `{forbidden}` fallback behavior"
        );
    }
}

#[test]
fn sorting_is_sorted_uses_explicit_retained_execution_options() {
    let body = function_region(
        SORTING_RS,
        "pub fn is_sorted",
        "// ---------------------------------------------------------------------------",
    );

    assert!(
        body.contains("ResolvedExecution::unary(input.execution_domain(), &ctx, operation)"),
        "sorting::is_sorted must derive execution options from the retained table-view domain"
    );
    assert!(body.contains("cudf_sys::cudf_is_sorted_ex"), "sorting::is_sorted must call the explicit FFI entry point");
    assert!(
        !body.contains("cudf_sys::cudf_is_sorted("),
        "sorting::is_sorted must not call the ambient non-_ex FFI entry point"
    );
}

#[test]
fn io_write_helpers_require_explicit_execution_options() {
    for (source, function, ex_call, ambient_call) in [
        (IO_CSV_RS, "write_csv_with_exec", "cudf_sys::cudf_io_write_csv_ex", "cudf_sys::cudf_io_write_csv("),
        (IO_JSON_RS, "write_json_with_exec", "cudf_sys::cudf_io_write_json_ex", "cudf_sys::cudf_io_write_json("),
        (
            IO_PARQUET_RS,
            "write_parquet_with_exec",
            "cudf_sys::cudf_io_write_parquet_ex",
            "cudf_sys::cudf_io_write_parquet(",
        ),
    ] {
        let body = function_region(source, &format!("pub(crate) fn {function}"), "\n}\n");
        assert!(
            body.contains("exec_options: &cudf_sys::cudf_exec_options_t"),
            "{function} must require explicit execution options"
        );
        assert!(!body.contains("exec_options: Option"), "{function} must not accept optional execution options");
        assert!(body.contains(ex_call), "{function} must call the explicit FFI entry point");
        assert!(!body.contains(ambient_call), "{function} must not call the ambient non-_ex FFI entry point");
        assert!(!body.contains("match exec_options"), "{function} must not branch on optional execution options");
    }
}

#[test]
fn safe_layer_non_ex_ffi_calls_stay_classified() {
    let explicit_without_suffix =
        classified_names(SAFE_LAYER_EXPLICIT_FFI_WITHOUT_EX_SUFFIX, "SAFE_LAYER_EXPLICIT_FFI_WITHOUT_EX_SUFFIX");
    let non_execution =
        classified_names(SAFE_LAYER_NON_EXECUTION_FFI_ALLOWLIST, "SAFE_LAYER_NON_EXECUTION_FFI_ALLOWLIST");
    let calls = safe_layer_cudf_sys_calls();

    let mut violations = Vec::new();
    for (name, locations) in &calls {
        if name.ends_with("_ex")
            || explicit_without_suffix.contains_key(name.as_str())
            || non_execution.contains_key(name.as_str())
        {
            continue;
        }

        violations.push(format!("{name}: {}", locations.iter().cloned().collect::<Vec<_>>().join(", ")));
    }

    let mut stale = Vec::new();
    for (name, classification) in explicit_without_suffix.iter().chain(non_execution.iter()) {
        if !calls.contains_key(*name) {
            stale.push(format!("{name}: {classification}"));
        }
    }

    if !violations.is_empty() || !stale.is_empty() {
        panic!(
            "safe-layer non-_ex cudf-sys calls must be explicitly classified as either \
             execution-sensitive calls with explicit options despite legacy naming, or as \
             non-execution-sensitive metadata/descriptor/lifecycle/runtime/test helpers.\n\
             unclassified calls:\n{}\n\nstale classifications:\n{}",
            violations.join("\n"),
            stale.join("\n")
        );
    }
}

fn function_region<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source.find(start_marker).expect("start marker must exist");
    let tail = &source[start..];
    let end = tail.find(end_marker).expect("end marker must exist");
    &tail[..end]
}

fn classified_names(
    entries: &'static [(&'static str, &'static str)],
    list_name: &str,
) -> BTreeMap<&'static str, &'static str> {
    let mut names = BTreeMap::new();
    for (name, classification) in entries {
        assert!(names.insert(*name, *classification).is_none(), "{list_name} has duplicate entry `{name}`");
    }
    names
}

fn safe_layer_cudf_sys_calls() -> BTreeMap<String, BTreeSet<String>> {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut calls = BTreeMap::new();

    for path in rust_source_files(&src_dir) {
        let source = fs::read_to_string(&path).expect("Rust source file must be readable");
        let relative = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).expect("source file must be under crate root");
        for (line_index, line) in source.lines().enumerate() {
            for name in cudf_sys_call_names(line) {
                calls.entry(name).or_insert_with(BTreeSet::new).insert(format!(
                    "{}:{}",
                    relative.display(),
                    line_index + 1
                ));
            }
        }
    }

    calls
}

fn rust_source_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).expect("source directory must be readable") {
        let path = entry.expect("source directory entry must be readable").path();
        if path.is_dir() {
            out.extend(rust_source_files(&path));
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn cudf_sys_call_names(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("cudf_sys::cudf_") {
        let after_prefix = &rest[start + "cudf_sys::".len()..];
        let name_len = after_prefix
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .map(char::len_utf8)
            .sum::<usize>();
        let name = &after_prefix[..name_len];
        if after_prefix[name_len..].starts_with('(') {
            out.push(name.to_owned());
        }
        rest = &after_prefix[name_len..];
    }
    out
}
