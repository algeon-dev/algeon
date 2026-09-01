#include "cudf_rust/cudf_rust.h"

#define SMOKE_CHECK_FN(result, function, ...)                                  \
  do {                                                                         \
    result (*const checked)(__VA_ARGS__) = &(function);                        \
    (void)checked;                                                             \
  } while (0)

#define SMOKE_CHECK_VAL(type, value)                                           \
  do {                                                                         \
    type checked = value;                                                      \
    (void)checked;                                                             \
  } while (0)

static int cudf_rust_c_header_smoke(void) {
  // clang-format off

  // Device and execution context
  SMOKE_CHECK_VAL(cudf_device_attribute_t, CUDF_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  SMOKE_CHECK_VAL(cudf_exec_options_t, CUDF_EXEC_OPTIONS_INITIALIZER);
  SMOKE_CHECK_FN(cudf_status_t, cudf_cuda_get_device, int32_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_cuda_set_device, int32_t);
  SMOKE_CHECK_FN(cudf_status_t, cudf_cuda_get_device_count, int32_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_cuda_pointer_get_device, void const*, int32_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_cuda_device_pci_bus_id, int32_t, char*, size_t);
  SMOKE_CHECK_FN(cudf_status_t, cudf_device_memory_info, int32_t, size_t*, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_device_attribute, int32_t, int32_t, int32_t*);

  // Stream and event management
  SMOKE_CHECK_FN(cudf_status_t, cudf_stream_create, cudf_stream_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_stream_destroy, cudf_stream_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_stream_raw, cudf_stream_handle, void**);
  SMOKE_CHECK_FN(cudf_status_t, cudf_raw_cuda_stream_synchronize, void*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_stream_wait_event, cudf_stream_handle, cudf_event_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_raw_cuda_stream_wait_event, void*, void*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_event_create, cudf_event_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_event_destroy, cudf_event_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_event_record, cudf_event_handle, cudf_stream_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_event_synchronize, cudf_event_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_event_is_ready, cudf_event_handle, int32_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_raw_cuda_event_record, void*, void*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_event_raw, cudf_event_handle, void**);

  // Memory management and resources
  SMOKE_CHECK_FN(cudf_status_t, cudf_memory_resource_create_cuda, cudf_memory_resource_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_memory_resource_destroy, cudf_memory_resource_handle, int32_t);
  SMOKE_CHECK_FN(cudf_status_t, cudf_memory_resource_raw, cudf_memory_resource_handle, void**);
  SMOKE_CHECK_FN(cudf_status_t, cudf_memory_resource_create_failure_callback, cudf_memory_resource_handle, cudf_memory_failure_callback_t, void*, cudf_memory_resource_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_memory_manager_for_device_with_failure_callback, int32_t, cudf_memory_failure_callback_t, void*, cudf_memory_manager_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_memory_manager_failure_callback_state, int32_t, int32_t*, int32_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_memory_manager_route_current_resource, cudf_memory_manager_handle, cudf_stream_handle, cudf_memory_resource_handle, cudf_current_resource_route_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_current_resource_route_destroy, cudf_current_resource_route_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_reservation_resize, cudf_reservation_handle, size_t);

  // Types, views, and buffer adoption
  SMOKE_CHECK_FN(cudf_status_t, cudf_data_type_create, cudf_type_id_t, cudf_data_type_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_data_type_destroy, cudf_data_type_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_view, cudf_column_handle, cudf_column_view_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_view_destroy, cudf_column_view_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_table_view, cudf_table_handle, cudf_table_view_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_table_view_destroy, cudf_table_view_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_device_buffer_destroy, cudf_device_buffer_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_device_buffer_data, cudf_device_buffer_handle, void**);
  SMOKE_CHECK_FN(cudf_status_t, cudf_device_buffer_size, cudf_device_buffer_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_adopt_device_buffer, cudf_type_id_t, cudf_device_buffer_handle, int32_t, cudf_column_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_adopt_device_buffer_nullable, cudf_type_id_t, cudf_device_buffer_handle, cudf_device_buffer_handle, int32_t, int32_t, cudf_column_handle*);

  // Column and table creation, copying, and inspection
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_create_numeric_ex, cudf_type_id_t, void const*, int32_t, uint8_t const*, int32_t, cudf_exec_options_t const*, cudf_column_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_create_fixed_width_from_host_ex, cudf_type_id_t, int32_t, void const*, int32_t, uint8_t const*, int32_t, cudf_exec_options_t const*, cudf_column_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_create_sequence_ex, cudf_scalar_handle, cudf_scalar_handle, int32_t, cudf_exec_options_t const*, cudf_column_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_create_from_device_data_ex, cudf_type_id_t, void const*, int32_t, cudf_exec_options_t const*, cudf_column_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_copy_ex, cudf_column_handle, cudf_exec_options_t const*, cudf_column_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_copy_from_view_ex, cudf_column_view_handle, cudf_exec_options_t const*, cudf_column_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_payload_size_ex, cudf_column_handle, cudf_exec_options_t const*, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_device_allocated_size, cudf_column_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_view_to_host_ex, cudf_column_view_handle, void*, size_t, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_column_view_strings_to_host_ex, cudf_column_view_handle, int32_t*, size_t, char*, size_t, int32_t*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_table_copy_ex, cudf_table_handle, cudf_exec_options_t const*, cudf_table_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_table_payload_size_ex, cudf_table_handle, cudf_exec_options_t const*, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_table_device_allocated_size, cudf_table_handle, size_t*);

  // Relational operations, filter, compaction, and hashing
  SMOKE_CHECK_FN(cudf_status_t, cudf_dictionary_encode_ex, cudf_column_view_handle, cudf_type_id_t, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_hashing_murmurhash3_x86_32_ex, cudf_table_view_handle, uint32_t, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_replace_nulls_scalar_ex, cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_replace_nulls_column_ex, cudf_column_view_handle, cudf_column_view_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_apply_boolean_mask_ex, cudf_table_view_handle, cudf_column_view_handle, cudf_table_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_filter_rows_by_predicate_ex, cudf_table_view_handle, cudf_row_predicate_node_t const*, int32_t, int32_t, cudf_table_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_gather_ex, cudf_table_view_handle, cudf_column_view_handle, cudf_out_of_bounds_policy_t, cudf_table_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_gather_fixed_width_dont_check_preflight, int64_t, uint64_t, cudf_gather_fixed_width_column_metadata_t const*, int32_t, int32_t, cudf_gather_fixed_width_dont_check_preflight_result_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_sorted_order_lexicographic_preflight, int64_t, cudf_type_id_t const*, int64_t const*, cudf_order_t const*, cudf_null_order_t const*, int32_t, int32_t, int32_t, cudf_sorted_order_lexicographic_preflight_result_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scatter_ex, cudf_table_view_handle, cudf_column_view_handle, cudf_table_view_handle, cudf_table_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_unique_ex, cudf_table_view_handle, int const*, int, cudf_duplicate_keep_option_t, cudf_stream_compaction_null_equality_t, cudf_table_handle*, cudf_exec_options_t const*);

  // AST expressions
  SMOKE_CHECK_FN(cudf_status_t, cudf_ast_column_reference_create, int32_t, cudf_ast_table_reference_t, cudf_ast_expression_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_ast_literal_create, cudf_scalar_handle, cudf_ast_expression_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_ast_operation_unary_create, cudf_ast_operator_t, cudf_ast_expression_handle, cudf_ast_expression_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_ast_operation_binary_create, cudf_ast_operator_t, cudf_ast_expression_handle, cudf_ast_expression_handle, cudf_ast_expression_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_ast_expression_destroy, cudf_ast_expression_handle);

  // Join operations
  SMOKE_CHECK_VAL(cudf_join_kind_t, CUDF_JOIN_KIND_INNER_JOIN);
  SMOKE_CHECK_VAL(cudf_nullable_join_t, CUDF_NULLABLE_JOIN_YES);
  SMOKE_CHECK_FN(cudf_status_t, cudf_conditional_full_join_size_data_destroy, cudf_conditional_full_join_size_data_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_conditional_full_join_size_data_output_size, cudf_conditional_full_join_size_data_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_conditional_full_join_size_data_device_allocated_size, cudf_conditional_full_join_size_data_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_conditional_full_join_size_data_gather_map_size, cudf_conditional_full_join_size_data_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_mixed_join_size_data_destroy, cudf_mixed_join_size_data_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_mixed_join_size_data_output_size, cudf_mixed_join_size_data_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_mixed_join_size_data_device_allocated_size, cudf_mixed_join_size_data_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_mixed_join_size_data_gather_map_size, cudf_mixed_join_size_data_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_hash_join_destroy, cudf_hash_join_handle);
  SMOKE_CHECK_FN(cudf_status_t, cudf_hash_join_device_allocated_size, cudf_hash_join_handle, size_t*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_distinct_hash_join_destroy, cudf_distinct_hash_join_handle);

  // Scalars and reduction
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_create_numeric_ex, cudf_type_id_t, void const*, int32_t, cudf_exec_options_t const*, cudf_scalar_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_create_string_ex, char const*, int32_t, cudf_exec_options_t const*, cudf_scalar_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_create_decimal_raw_ex, cudf_type_id_t, void const*, int32_t, int32_t, cudf_exec_options_t const*, cudf_scalar_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_create_timestamp_raw_ex, cudf_type_id_t, int64_t, int32_t, cudf_exec_options_t const*, cudf_scalar_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_create_duration_raw_ex, cudf_type_id_t, int64_t, int32_t, cudf_exec_options_t const*, cudf_scalar_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_is_valid_ex, cudf_scalar_handle, int32_t*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_value_ex, cudf_scalar_handle, void*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_string_value_ex, cudf_scalar_handle, char**, int32_t*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_scalar_copy_ex, cudf_scalar_handle, cudf_exec_options_t const*, cudf_scalar_handle*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_reduce_aggregation_create_nunique, int32_t, cudf_reduce_aggregation_handle*);

  // Strings
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_to_lower_ex, cudf_column_view_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_to_upper_ex, cudf_column_view_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_contains_ex, cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_starts_with_ex, cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_ends_with_ex, cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_like_ex, cudf_column_view_handle, char const*, int32_t, char const*, int32_t, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_slice_ex, cudf_column_view_handle, int32_t, int32_t, int32_t, int32_t, int32_t, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_strip_ex, cudf_column_view_handle, cudf_strip_side_t, cudf_scalar_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_count_characters_ex, cudf_column_view_handle, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_strings_count_bytes_ex, cudf_column_view_handle, cudf_column_handle*, cudf_exec_options_t const*);

  // Datetime
  SMOKE_CHECK_VAL(cudf_datetime_component_t, CUDF_DATETIME_COMPONENT_YEAR);
  SMOKE_CHECK_FN(cudf_status_t, cudf_datetime_extract_component_ex, cudf_column_view_handle, cudf_datetime_component_t, cudf_column_handle*, cudf_exec_options_t const*);
  SMOKE_CHECK_FN(cudf_status_t, cudf_datetime_add_calendrical_months_scalar_ex, cudf_column_view_handle, int32_t, cudf_column_handle*, cudf_exec_options_t const*);

  // GDS
  SMOKE_CHECK_FN(cudf_status_t, cudf_gds_driver_available, void);

  // clang-format on
  return 0;
}

#undef SMOKE_CHECK_FN
#undef SMOKE_CHECK_VAL
