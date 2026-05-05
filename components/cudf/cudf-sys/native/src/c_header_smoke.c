#include "cudf_rust.h"

int cudf_rust_c_header_smoke(void)
{
  cudf_status_t (*f1)(cudf_stream_handle*)                    = &cudf_stream_create;
  void (*f2)(cudf_stream_handle)                              = &cudf_stream_destroy;
  cudf_status_t (*f2b)(cudf_stream_handle, void**)            = &cudf_stream_raw;
  cudf_status_t (*f2b2)(void*)                                = &cudf_raw_cuda_stream_synchronize;
  cudf_status_t (*f2c)(cudf_event_handle*)                    = &cudf_event_create;
  void (*f2d)(cudf_event_handle)                              = &cudf_event_destroy;
  cudf_status_t (*f2e)(cudf_event_handle, cudf_stream_handle) = &cudf_event_record;
  cudf_status_t (*f2f)(cudf_event_handle)                     = &cudf_event_synchronize;
  cudf_status_t (*f2g)(cudf_event_handle, int32_t*)           = &cudf_event_is_ready;
  cudf_status_t (*f2h)(cudf_stream_handle, cudf_event_handle) = &cudf_stream_wait_event;
  cudf_status_t (*f2i)(void*, void*)                          = &cudf_raw_cuda_event_record;
  cudf_status_t (*f2j)(void*, void*)                          = &cudf_raw_cuda_stream_wait_event;
  cudf_status_t (*f2k)(cudf_event_handle, void**)             = &cudf_event_raw;
  cudf_status_t (*f3)(cudf_memory_resource_handle*)           = &cudf_memory_resource_create_cuda;
  cudf_status_t (*f4)(cudf_memory_resource_handle, int32_t)   = &cudf_memory_resource_destroy;
  cudf_status_t (*f4b)(cudf_memory_resource_handle, void**)   = &cudf_memory_resource_raw;

  cudf_status_t (*f5)(cudf_type_id_t, cudf_data_type_handle*) = &cudf_data_type_create;
  void (*f6)(cudf_data_type_handle)                           = &cudf_data_type_destroy;

  cudf_status_t (*f7)(cudf_column_handle, cudf_column_view_handle*) = &cudf_column_view;
  void (*f8)(cudf_column_view_handle)                               = &cudf_column_view_destroy;

  cudf_status_t (*f9)(cudf_table_handle, cudf_table_view_handle*) = &cudf_table_view;
  void (*f10)(cudf_table_view_handle)                             = &cudf_table_view_destroy;

  cudf_status_t (*f11)(cudf_column_view_handle, cudf_type_id_t, cudf_column_handle*) =
    &cudf_dictionary_encode;
  cudf_status_t (*f12)(cudf_table_view_handle, uint32_t, cudf_column_handle*) =
    &cudf_hashing_murmurhash3_x86_32;
  cudf_status_t (*f13)(cudf_column_handle, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_copy_ex;
  cudf_status_t (*f14)(cudf_column_view_handle, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_copy_from_view_ex;
  cudf_status_t (*f14b)(cudf_column_handle, size_t*) = &cudf_column_payload_size;
  cudf_status_t (*f14c)(cudf_column_handle, const cudf_exec_options_t*, size_t*) =
    &cudf_column_payload_size_ex;
  cudf_status_t (*f14d)(cudf_column_handle, size_t*) = &cudf_column_device_allocated_size;
  cudf_status_t (*f15)(cudf_type_id_t,
                       const void*,
                       int32_t,

                       const uint8_t*,
                       int32_t,

                       const cudf_exec_options_t*,
                       cudf_column_handle*)          = &cudf_column_create_numeric_ex;
  cudf_status_t (*f15a)(cudf_type_id_t,
                        int32_t,
                        const void*,
                        int32_t,

                        const uint8_t*,
                        int32_t,

                        const cudf_exec_options_t*,
                        cudf_column_handle*)         = &cudf_column_create_fixed_width_from_host_ex;
  cudf_status_t (*f15b)(cudf_scalar_handle,
                        cudf_scalar_handle,
                        int32_t,

                        const cudf_exec_options_t*,
                        cudf_column_handle*)         = &cudf_column_create_sequence_ex;
  cudf_status_t (*f16)(cudf_table_handle, const cudf_exec_options_t*, cudf_table_handle*) =
    &cudf_table_copy_ex;
  cudf_status_t (*f16b)(cudf_table_handle, size_t*) = &cudf_table_payload_size;
  cudf_status_t (*f16c)(cudf_table_handle, const cudf_exec_options_t*, size_t*) =
    &cudf_table_payload_size_ex;
  cudf_status_t (*f16d)(cudf_table_handle, size_t*) = &cudf_table_device_allocated_size;
  cudf_status_t (*f17)(cudf_column_view_handle, void*, size_t, const cudf_exec_options_t*) =
    &cudf_column_view_to_host_ex;
  cudf_status_t (*f17b)(cudf_column_view_handle,
                        int32_t*,
                        size_t,
                        char*,
                        size_t,
                        int32_t*,

                        const cudf_exec_options_t*) = &cudf_column_view_strings_to_host_ex;
  cudf_status_t (*f18)(
    cudf_type_id_t, const void*, int32_t, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_create_from_device_data_ex;
  cudf_status_t (*f19)(
    cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_replace_nulls_scalar_ex;
  cudf_status_t (*f20)(cudf_column_view_handle,
                       cudf_column_view_handle,
                       cudf_column_handle*,

                       const cudf_exec_options_t*)  = &cudf_replace_nulls_column_ex;
  cudf_status_t (*f21)(cudf_table_view_handle,
                       cudf_column_view_handle,
                       cudf_table_handle*,

                       const cudf_exec_options_t*)  = &cudf_apply_boolean_mask_ex;
  cudf_status_t (*f21b)(cudf_table_view_handle,

                        const cudf_row_predicate_node_t*,
                        int32_t,
                        int32_t,
                        cudf_table_handle*,

                        const cudf_exec_options_t*) = &cudf_filter_rows_by_predicate_ex;
  cudf_status_t (*f22)(cudf_table_view_handle,
                       cudf_column_view_handle,
                       cudf_out_of_bounds_policy_t,
                       cudf_table_handle*,

                       const cudf_exec_options_t*)  = &cudf_gather_ex;
  cudf_status_t (*f22_scatter)(
    cudf_table_view_handle, cudf_column_view_handle, cudf_table_view_handle, cudf_table_handle*) =
    &cudf_scatter;
  cudf_status_t (*f22_scatter_ex)(cudf_table_view_handle,
                                  cudf_column_view_handle,
                                  cudf_table_view_handle,
                                  cudf_table_handle*,

                                  const cudf_exec_options_t*) = &cudf_scatter_ex;
  cudf_status_t (*f22_cross)(cudf_table_view_handle, cudf_table_view_handle, cudf_table_handle*) =
    &cudf_cross_join;
  cudf_status_t (*f22_cross_ex)(cudf_table_view_handle,
                                cudf_table_view_handle,
                                cudf_table_handle*,

                                const cudf_exec_options_t*) = &cudf_cross_join_ex;
  cudf_status_t (*f22a0)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,
                         cudf_column_handle*)               = &cudf_inner_join_indices;
  cudf_status_t (*f22a1)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,
                         cudf_column_handle*,

                         const cudf_exec_options_t*)        = &cudf_inner_join_indices_ex;
  cudf_status_t (*f22a2)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,
                         cudf_column_handle*)               = &cudf_left_join_indices;
  cudf_status_t (*f22a3)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,
                         cudf_column_handle*,

                         const cudf_exec_options_t*)        = &cudf_left_join_indices_ex;
  cudf_status_t (*f22a4)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,
                         cudf_column_handle*)               = &cudf_full_join_indices;
  cudf_status_t (*f22a5)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,
                         cudf_column_handle*,

                         const cudf_exec_options_t*)        = &cudf_full_join_indices_ex;
  cudf_status_t (*f22a6)(
    cudf_table_view_handle, cudf_table_view_handle, cudf_null_equality_t, cudf_column_handle*) =
    &cudf_left_semi_join_indices;
  cudf_status_t (*f22a7)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,

                         const cudf_exec_options_t*) = &cudf_left_semi_join_indices_ex;
  cudf_status_t (*f22a8)(
    cudf_table_view_handle, cudf_table_view_handle, cudf_null_equality_t, cudf_column_handle*) =
    &cudf_left_anti_join_indices;
  cudf_status_t (*f22a9)(cudf_table_view_handle,
                         cudf_table_view_handle,
                         cudf_null_equality_t,
                         cudf_column_handle*,

                         const cudf_exec_options_t*) = &cudf_left_anti_join_indices_ex;
  cudf_status_t (*f22ast0)(int32_t, cudf_ast_table_reference_t, cudf_ast_expression_handle*) =
    &cudf_ast_column_reference_create;
  cudf_status_t (*f22ast1)(cudf_scalar_handle, cudf_ast_expression_handle*) =
    &cudf_ast_literal_create;
  cudf_status_t (*f22ast2)(
    cudf_ast_operator_t, cudf_ast_expression_handle, cudf_ast_expression_handle*) =
    &cudf_ast_operation_unary_create;
  cudf_status_t (*f22ast3)(cudf_ast_operator_t,
                           cudf_ast_expression_handle,
                           cudf_ast_expression_handle,
                           cudf_ast_expression_handle*) = &cudf_ast_operation_binary_create;
  void (*f22ast4)(cudf_ast_expression_handle)           = &cudf_ast_expression_destroy;
  cudf_status_t (*f22ci0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,
                          cudf_column_handle*)          = &cudf_conditional_inner_join_indices;
  cudf_status_t (*f22ci1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,
                          cudf_column_handle*,

                          const cudf_exec_options_t*)   = &cudf_conditional_inner_join_indices_ex;
  cudf_status_t (*f22cl0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,
                          cudf_column_handle*)          = &cudf_conditional_left_join_indices;
  cudf_status_t (*f22cl1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,
                          cudf_column_handle*,

                          const cudf_exec_options_t*)   = &cudf_conditional_left_join_indices_ex;
  cudf_status_t (*f22cf0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,
                          cudf_column_handle*)          = &cudf_conditional_full_join_indices;
  cudf_status_t (*f22cf1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,
                          cudf_column_handle*,

                          const cudf_exec_options_t*)   = &cudf_conditional_full_join_indices_ex;
  cudf_status_t (*f22cs0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*)          = &cudf_conditional_left_semi_join_indices;
  cudf_status_t (*f22cs1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_conditional_left_semi_join_indices_ex;
  cudf_status_t (*f22ca0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*)        = &cudf_conditional_left_anti_join_indices;
  cudf_status_t (*f22ca1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_conditional_left_anti_join_indices_ex;
  cudf_join_kind_t f22fj_kind                         = CUDF_JOIN_KIND_INNER_JOIN;
  cudf_status_t (*f22fj0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_column_view_handle,
                          cudf_column_view_handle,
                          cudf_ast_expression_handle,
                          cudf_join_kind_t,
                          cudf_column_handle*,
                          cudf_column_handle*)        = &cudf_filter_join_indices;
  cudf_status_t (*f22fj1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_column_view_handle,
                          cudf_column_view_handle,
                          cudf_ast_expression_handle,
                          cudf_join_kind_t,
                          cudf_column_handle*,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_filter_join_indices_ex;
  cudf_status_t (*f22mi0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,
                          cudf_column_handle*)        = &cudf_mixed_inner_join_indices;
  cudf_status_t (*f22mi1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_mixed_inner_join_indices_ex;
  cudf_status_t (*f22ml0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,
                          cudf_column_handle*)        = &cudf_mixed_left_join_indices;
  cudf_status_t (*f22ml1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_mixed_left_join_indices_ex;
  cudf_status_t (*f22mf0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,
                          cudf_column_handle*)        = &cudf_mixed_full_join_indices;
  cudf_status_t (*f22mf1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_mixed_full_join_indices_ex;
  cudf_status_t (*f22ms0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*)        = &cudf_mixed_left_semi_join_indices;
  cudf_status_t (*f22ms1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_mixed_left_semi_join_indices_ex;
  cudf_status_t (*f22ma0)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*)        = &cudf_mixed_left_anti_join_indices;
  cudf_status_t (*f22ma1)(cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_table_view_handle,
                          cudf_ast_expression_handle,
                          cudf_null_equality_t,
                          cudf_column_handle*,

                          const cudf_exec_options_t*) = &cudf_mixed_left_anti_join_indices_ex;
  cudf_status_t (*f22aa)(
    cudf_table_view_handle, cudf_null_equality_t, cudf_distinct_hash_join_handle*) =
    &cudf_distinct_hash_join_create;
  cudf_status_t (*f22aa_ex)(cudf_table_view_handle,
                            cudf_null_equality_t,
                            cudf_distinct_hash_join_handle*,

                            const cudf_exec_options_t*) = &cudf_distinct_hash_join_create_ex;
  cudf_status_t (*f22ab)(
    cudf_distinct_hash_join_handle, cudf_table_view_handle, cudf_column_handle*) =
    &cudf_distinct_hash_join_left_join;
  cudf_status_t (*f22ab_ex)(cudf_distinct_hash_join_handle,
                            cudf_table_view_handle,
                            cudf_column_handle*,

                            const cudf_exec_options_t*) = &cudf_distinct_hash_join_left_join_ex;
  void (*f22ac)(cudf_distinct_hash_join_handle)         = &cudf_distinct_hash_join_destroy;
  cudf_status_t (*f22ad)(cudf_table_view_handle,
                         cudf_nullable_join_t,
                         cudf_null_equality_t,
                         double,
                         cudf_hash_join_handle*)        = &cudf_hash_join_create;
  cudf_status_t (*f22ae)(cudf_table_view_handle,
                         cudf_nullable_join_t,
                         cudf_null_equality_t,
                         double,
                         cudf_hash_join_handle*,

                         const cudf_exec_options_t*)    = &cudf_hash_join_create_ex;
  void (*f22af)(cudf_hash_join_handle)                  = &cudf_hash_join_destroy;
  cudf_status_t (*f22af_size)(cudf_hash_join_handle, size_t*) =
    &cudf_hash_join_device_allocated_size;
  cudf_status_t (*f22ag)(
    cudf_hash_join_handle, cudf_table_view_handle, cudf_column_handle*, cudf_column_handle*) =
    &cudf_hash_join_inner_join;
  cudf_status_t (*f22ah)(cudf_hash_join_handle,
                         cudf_table_view_handle,
                         cudf_column_handle*,
                         cudf_column_handle*,

                         const cudf_exec_options_t*) = &cudf_hash_join_inner_join_ex;
  cudf_status_t (*f22ai)(
    cudf_hash_join_handle, cudf_table_view_handle, cudf_column_handle*, cudf_column_handle*) =
    &cudf_hash_join_left_join;
  cudf_status_t (*f22aj)(cudf_hash_join_handle,
                         cudf_table_view_handle,
                         cudf_column_handle*,
                         cudf_column_handle*,

                         const cudf_exec_options_t*) = &cudf_hash_join_left_join_ex;
  cudf_status_t (*f22ak)(
    cudf_hash_join_handle, cudf_table_view_handle, cudf_column_handle*, cudf_column_handle*) =
    &cudf_hash_join_full_join;
  cudf_status_t (*f22al)(cudf_hash_join_handle,
                         cudf_table_view_handle,
                         cudf_column_handle*,
                         cudf_column_handle*,

                         const cudf_exec_options_t*) = &cudf_hash_join_full_join_ex;
  cudf_status_t (*f22b)(
    cudf_type_id_t, const void*, int32_t, const cudf_exec_options_t*, cudf_scalar_handle*) =
    &cudf_scalar_create_numeric_ex;
  cudf_status_t (*f22c)(const char*, int32_t, const cudf_exec_options_t*, cudf_scalar_handle*) =
    &cudf_scalar_create_string_ex;
  cudf_status_t (*f22c0)(cudf_type_id_t, const void*, int32_t, int32_t, cudf_scalar_handle*) =
    &cudf_scalar_create_decimal_raw;
  cudf_status_t (*f22c00)(cudf_type_id_t, int64_t, int32_t, cudf_scalar_handle*) =
    &cudf_scalar_create_timestamp_raw;
  cudf_status_t (*f22c01)(cudf_type_id_t, int64_t, int32_t, cudf_scalar_handle*) =
    &cudf_scalar_create_duration_raw;
  cudf_status_t (*f22c1)(cudf_type_id_t,
                         const void*,
                         int32_t,
                         int32_t,

                         const cudf_exec_options_t*,
                         cudf_scalar_handle*) = &cudf_scalar_create_decimal_raw_ex;
  cudf_status_t (*f22c2)(
    cudf_type_id_t, int64_t, int32_t, const cudf_exec_options_t*, cudf_scalar_handle*) =
    &cudf_scalar_create_timestamp_raw_ex;
  cudf_status_t (*f22c3)(
    cudf_type_id_t, int64_t, int32_t, const cudf_exec_options_t*, cudf_scalar_handle*) =
    &cudf_scalar_create_duration_raw_ex;
  cudf_status_t (*f22d)(cudf_scalar_handle, int32_t*, const cudf_exec_options_t*) =
    &cudf_scalar_is_valid_ex;
  cudf_status_t (*f22e)(cudf_scalar_handle, void*, const cudf_exec_options_t*) =
    &cudf_scalar_value_ex;
  cudf_status_t (*f22f)(cudf_scalar_handle, char**, int32_t*, const cudf_exec_options_t*) =
    &cudf_scalar_string_value_ex;
  cudf_status_t (*f22g)(cudf_scalar_handle, cudf_scalar_handle*) = &cudf_scalar_copy;
  cudf_status_t (*f22h)(cudf_scalar_handle, const cudf_exec_options_t*, cudf_scalar_handle*) =
    &cudf_scalar_copy_ex;
  cudf_status_t (*f22i)(int32_t, cudf_reduce_aggregation_handle*) =
    &cudf_reduce_aggregation_create_nunique;
  cudf_exec_options_t opts = CUDF_EXEC_OPTIONS_INITIALIZER;
  cudf_nullable_join_t j1  = CUDF_NULLABLE_JOIN_YES;

  cudf_status_t (*f23)(size_t, cudf_device_buffer_handle*) = &cudf_device_buffer_create;
  void (*f24)(cudf_device_buffer_handle)                   = &cudf_device_buffer_destroy;
  cudf_status_t (*f25)(cudf_device_buffer_handle, void**)  = &cudf_device_buffer_data;
  cudf_status_t (*f26)(cudf_device_buffer_handle, size_t*) = &cudf_device_buffer_size;
  cudf_status_t (*f27)(cudf_device_buffer_handle, const void*, size_t) =
    &cudf_device_buffer_copy_from_host;
  cudf_status_t (*f28)(cudf_type_id_t, cudf_device_buffer_handle, int32_t, cudf_column_handle*) =
    &cudf_column_adopt_device_buffer;
  cudf_status_t (*f29)(cudf_type_id_t,
                       cudf_device_buffer_handle,
                       cudf_device_buffer_handle,
                       int32_t,
                       int32_t,
                       cudf_column_handle*) = &cudf_column_adopt_device_buffer_nullable;

  /* strings.h */
  cudf_status_t (*f30)(cudf_column_view_handle, cudf_column_handle*) = &cudf_strings_to_lower;
  cudf_status_t (*f31)(cudf_column_view_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_to_lower_ex;
  cudf_status_t (*f32)(cudf_column_view_handle, cudf_column_handle*) = &cudf_strings_to_upper;
  cudf_status_t (*f33)(cudf_column_view_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_to_upper_ex;
  cudf_status_t (*f34)(cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*) =
    &cudf_strings_contains;
  cudf_status_t (*f35)(
    cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_contains_ex;
  cudf_status_t (*f36)(cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*) =
    &cudf_strings_starts_with;
  cudf_status_t (*f37)(
    cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_starts_with_ex;
  cudf_status_t (*f38)(cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*) =
    &cudf_strings_ends_with;
  cudf_status_t (*f39)(
    cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_ends_with_ex;
  cudf_status_t (*f39a)(
    cudf_column_view_handle, const char*, int32_t, const char*, int32_t, cudf_column_handle*) =
    &cudf_strings_like;
  cudf_status_t (*f39b)(cudf_column_view_handle,
                        const char*,
                        int32_t,
                        const char*,
                        int32_t,
                        cudf_column_handle*,

                        const cudf_exec_options_t*) = &cudf_strings_like_ex;
  cudf_status_t (*f39c)(
    cudf_column_view_handle, int32_t, int32_t, int32_t, int32_t, int32_t, cudf_column_handle*) =
    &cudf_strings_slice;
  cudf_status_t (*f39d)(cudf_column_view_handle,
                        int32_t,
                        int32_t,
                        int32_t,
                        int32_t,
                        int32_t,
                        cudf_column_handle*,

                        const cudf_exec_options_t*) = &cudf_strings_slice_ex;
  cudf_status_t (*f40)(
    cudf_column_view_handle, cudf_strip_side_t, cudf_scalar_handle, cudf_column_handle*) =
    &cudf_strings_strip;
  cudf_status_t (*f41)(cudf_column_view_handle,
                       cudf_strip_side_t,
                       cudf_scalar_handle,
                       cudf_column_handle*,

                       const cudf_exec_options_t*) = &cudf_strings_strip_ex;
  cudf_status_t (*f42)(cudf_column_view_handle, cudf_column_handle*) =
    &cudf_strings_count_characters;
  cudf_status_t (*f43)(cudf_column_view_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_count_characters_ex;
  cudf_status_t (*f44)(cudf_column_view_handle, cudf_column_handle*) = &cudf_strings_count_bytes;
  cudf_status_t (*f45)(cudf_column_view_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_count_bytes_ex;

  /* device.h */
  cudf_status_t (*f46)(int32_t*)              = &cudf_cuda_get_device;
  cudf_status_t (*f47)(int32_t)               = &cudf_cuda_set_device;
  cudf_status_t (*f48)(int32_t*)              = &cudf_cuda_get_device_count;
  cudf_status_t (*f49)(const void*, int32_t*) = &cudf_cuda_pointer_get_device;

  /* datetime.h */
  cudf_status_t (*f50)(cudf_column_view_handle, cudf_datetime_component_t, cudf_column_handle*) =
    &cudf_datetime_extract_component;
  cudf_status_t (*f51)(cudf_column_view_handle,
                       cudf_datetime_component_t,
                       cudf_column_handle*,

                       const cudf_exec_options_t*) = &cudf_datetime_extract_component_ex;
  cudf_status_t (*f52)(cudf_column_view_handle, int32_t, cudf_column_handle*) =
    &cudf_datetime_add_calendrical_months_scalar;
  cudf_status_t (*f53)(cudf_column_view_handle, int32_t, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_datetime_add_calendrical_months_scalar_ex;
  cudf_datetime_component_t c1 = CUDF_DATETIME_COMPONENT_YEAR;

  (void)f1;
  (void)f2;
  (void)f2b;
  (void)f2b2;
  (void)f2c;
  (void)f2d;
  (void)f2e;
  (void)f2f;
  (void)f2g;
  (void)f2h;
  (void)f2i;
  (void)f2j;
  (void)f2k;
  (void)f3;
  (void)f4;
  (void)f4b;
  (void)f5;
  (void)f6;
  (void)f7;
  (void)f8;
  (void)f9;
  (void)f10;
  (void)f11;
  (void)f12;
  (void)f13;
  (void)f14;
  (void)f14b;
  (void)f14c;
  (void)f14d;
  (void)f15;
  (void)f15a;
  (void)f15b;
  (void)f16;
  (void)f16b;
  (void)f16c;
  (void)f16d;
  (void)f17;
  (void)f17b;
  (void)f18;
  (void)f19;
  (void)f20;
  (void)f21;
  (void)f22;
  (void)f22_scatter;
  (void)f22_scatter_ex;
  (void)f22_cross;
  (void)f22_cross_ex;
  (void)f22a0;
  (void)f22a1;
  (void)f22a2;
  (void)f22a3;
  (void)f22a4;
  (void)f22a5;
  (void)f22a6;
  (void)f22a7;
  (void)f22a8;
  (void)f22a9;
  (void)f22ast0;
  (void)f22ast1;
  (void)f22ast2;
  (void)f22ast3;
  (void)f22ast4;
  (void)f22ci0;
  (void)f22ci1;
  (void)f22cl0;
  (void)f22cl1;
  (void)f22cf0;
  (void)f22cf1;
  (void)f22cs0;
  (void)f22cs1;
  (void)f22ca0;
  (void)f22ca1;
  (void)f22fj_kind;
  (void)f22fj0;
  (void)f22fj1;
  (void)f22mi0;
  (void)f22mi1;
  (void)f22ml0;
  (void)f22ml1;
  (void)f22mf0;
  (void)f22mf1;
  (void)f22ms0;
  (void)f22ms1;
  (void)f22ma0;
  (void)f22ma1;
  (void)f22aa;
  (void)f22ab;
  (void)f22ac;
  (void)f22ad;
  (void)f22ae;
  (void)f22af;
  (void)f22ag;
  (void)f22ah;
  (void)f22ai;
  (void)f22aj;
  (void)f22ak;
  (void)f22al;
  (void)f22b;
  (void)f22c;
  (void)f22c0;
  (void)f22c00;
  (void)f22c01;
  (void)f22c1;
  (void)f22c2;
  (void)f22c3;
  (void)f22d;
  (void)f22e;
  (void)f22f;
  (void)f22g;
  (void)f22h;
  (void)f22i;
  (void)f23;
  (void)f24;
  (void)f25;
  (void)f26;
  (void)f27;
  (void)f28;
  (void)f29;
  (void)f30;
  (void)f31;
  (void)f32;
  (void)f33;
  (void)f34;
  (void)f35;
  (void)f36;
  (void)f37;
  (void)f38;
  (void)f39;
  (void)f39a;
  (void)f39b;
  (void)f39c;
  (void)f39d;
  (void)f40;
  (void)f41;
  (void)f42;
  (void)f43;
  (void)f44;
  (void)f45;
  (void)f46;
  (void)f47;
  (void)f48;
  (void)f49;
  (void)f50;
  (void)f51;
  (void)f52;
  (void)f53;
  (void)c1;
  (void)j1;
  (void)opts;
  return 0;
}
