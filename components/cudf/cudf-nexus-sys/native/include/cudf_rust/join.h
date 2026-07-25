#ifndef CUDF_RUST_JOIN_H
#define CUDF_RUST_JOIN_H

#include "ast.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// How to compare nulls in join keys.
typedef enum {
  CUDF_NULL_EQUALITY_EQUAL   = 0,
  CUDF_NULL_EQUALITY_UNEQUAL = 1,
} cudf_null_equality_t;

/// Whether all hash-join build/probe inputs may contain nulls.
typedef enum {
  CUDF_NULLABLE_JOIN_YES = 0,
  CUDF_NULLABLE_JOIN_NO  = 1,
} cudf_nullable_join_t;

/// Join kind for APIs that operate on already-computed join index pairs.
typedef enum {
  CUDF_JOIN_KIND_INNER_JOIN     = 0,
  CUDF_JOIN_KIND_LEFT_JOIN      = 1,
  CUDF_JOIN_KIND_FULL_JOIN      = 2,
  CUDF_JOIN_KIND_LEFT_SEMI_JOIN = 3,
  CUDF_JOIN_KIND_LEFT_ANTI_JOIN = 4,
} cudf_join_kind_t;

/// Opaque handle to a reusable libcudf distinct-hash-join build artifact.
typedef struct cudf_distinct_hash_join_handle_state* cudf_distinct_hash_join_handle;

/// Opaque handle to a reusable libcudf filtered-join build artifact.
typedef struct cudf_filtered_join_handle_state* cudf_filtered_join_handle;

/// Opaque handle to a retained libcudf filtered-join probe state.
typedef struct cudf_filtered_join_probe_handle_state* cudf_filtered_join_probe_handle;

/// Opaque handle to a reusable libcudf hash-join build artifact.
typedef struct cudf_hash_join_handle_state* cudf_hash_join_handle;

/// Opaque handle to retained libcudf conditional full join output-size data.
typedef struct cudf_conditional_full_join_size_data_handle_state*
  cudf_conditional_full_join_size_data_handle;

/// Opaque handle to retained libcudf mixed-join output-size data.
typedef struct cudf_mixed_join_size_data_handle_state* cudf_mixed_join_size_data_handle;

/// Inner join with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned tables.
cudf_status_t cudf_inner_join_ex(cudf_table_view_handle left_keys,
                                 cudf_table_view_handle right_keys,
                                 cudf_table_view_handle left_payload,
                                 cudf_table_view_handle right_payload,
                                 cudf_null_equality_t null_equality,
                                 cudf_table_handle* out_left,
                                 cudf_table_handle* out_right,
                                 cudf_exec_options_t const* options);

/// Left join with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned tables.
cudf_status_t cudf_left_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                cudf_exec_options_t const* options);

/// Full join with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned tables.
cudf_status_t cudf_full_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                cudf_exec_options_t const* options);

/// Cross join with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned table.
cudf_status_t cudf_cross_join_ex(cudf_table_view_handle left,
                                 cudf_table_view_handle right,
                                 cudf_table_handle* out,
                                 cudf_exec_options_t const* options);

/// Projected cross join with explicit input row counts and execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned table.
cudf_status_t cudf_projected_cross_join_ex(cudf_table_view_handle left_selected,
                                           cudf_table_view_handle right_selected,
                                           size_t left_num_rows,
                                           size_t right_num_rows,
                                           cudf_table_handle* out,
                                           cudf_exec_options_t const* options);

/// Return exact device bytes retained by a projected cross join result.
///
/// This materializes the projected result under the supplied execution context,
/// reports `table::alloc_size()`, and releases the temporary result.
/// `options` must name an explicit valid device and non-NULL stream/MR.
cudf_status_t cudf_projected_cross_join_device_allocated_size_ex(
  cudf_table_view_handle left_selected,
  cudf_table_view_handle right_selected,
  size_t left_num_rows,
  size_t right_num_rows,
  size_t* out,
  cudf_exec_options_t const* options);

/// Inner join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_inner_join_indices_ex(cudf_table_view_handle left_keys,
                                         cudf_table_view_handle right_keys,
                                         cudf_null_equality_t null_equality,
                                         cudf_column_handle* out_left_indices,
                                         cudf_column_handle* out_right_indices,
                                         cudf_exec_options_t const* options);

/// Left join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_left_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        cudf_exec_options_t const* options);

/// Full join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_full_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        cudf_exec_options_t const* options);

/// Left semi join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned index column.
cudf_status_t cudf_left_semi_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             cudf_exec_options_t const* options);

/// Left anti join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned index column.
cudf_status_t cudf_left_anti_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             cudf_exec_options_t const* options);

/// Conditional inner join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_conditional_inner_join_indices_ex(cudf_table_view_handle left,
                                                     cudf_table_view_handle right,
                                                     cudf_ast_expression_handle predicate,
                                                     cudf_column_handle* out_left_indices,
                                                     cudf_column_handle* out_right_indices,
                                                     cudf_exec_options_t const* options);

/// Conditional inner join indices with an exact output row count and explicit
/// execution context.
cudf_status_t cudf_conditional_inner_join_indices_with_output_size_ex(
  cudf_table_view_handle left,
  cudf_table_view_handle right,
  cudf_ast_expression_handle predicate,
  size_t output_size,
  cudf_column_handle* out_left_indices,
  cudf_column_handle* out_right_indices,
  cudf_exec_options_t const* options);

/// Return the exact output row count for a conditional inner join with
/// explicit execution context.
cudf_status_t cudf_conditional_inner_join_size_ex(cudf_table_view_handle left,
                                                  cudf_table_view_handle right,
                                                  cudf_ast_expression_handle predicate,
                                                  size_t* out,
                                                  cudf_exec_options_t const* options);

/// Conditional left join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_conditional_left_join_indices_ex(cudf_table_view_handle left,
                                                    cudf_table_view_handle right,
                                                    cudf_ast_expression_handle predicate,
                                                    cudf_column_handle* out_left_indices,
                                                    cudf_column_handle* out_right_indices,
                                                    cudf_exec_options_t const* options);

/// Conditional left join indices with an exact output row count and explicit
/// execution context.
cudf_status_t cudf_conditional_left_join_indices_with_output_size_ex(
  cudf_table_view_handle left,
  cudf_table_view_handle right,
  cudf_ast_expression_handle predicate,
  size_t output_size,
  cudf_column_handle* out_left_indices,
  cudf_column_handle* out_right_indices,
  cudf_exec_options_t const* options);

/// Return the exact output row count for a conditional left join with explicit
/// execution context.
cudf_status_t cudf_conditional_left_join_size_ex(cudf_table_view_handle left,
                                                 cudf_table_view_handle right,
                                                 cudf_ast_expression_handle predicate,
                                                 size_t* out,
                                                 cudf_exec_options_t const* options);

/// Conditional full join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_conditional_full_join_indices_ex(cudf_table_view_handle left,
                                                    cudf_table_view_handle right,
                                                    cudf_ast_expression_handle predicate,
                                                    cudf_column_handle* out_left_indices,
                                                    cudf_column_handle* out_right_indices,
                                                    cudf_exec_options_t const* options);

/// Compute and retain exact output-size data for a conditional full join with
/// explicit execution context.
cudf_status_t cudf_conditional_full_join_begin_size_data_ex(
  cudf_table_view_handle left,
  cudf_table_view_handle right,
  cudf_ast_expression_handle predicate,
  cudf_conditional_full_join_size_data_handle* out,
  cudf_exec_options_t const* options);

/// Destroy retained conditional full join output-size data. Ownership is
/// consumed on call; failure quarantines the handle and must not be retried.
cudf_status_t cudf_conditional_full_join_size_data_destroy(
  cudf_conditional_full_join_size_data_handle handle);

/// Return the exact output row count retained in conditional full join size data.
cudf_status_t cudf_conditional_full_join_size_data_output_size(
  cudf_conditional_full_join_size_data_handle handle, size_t* out);

/// Return device bytes retained by conditional full join size data.
cudf_status_t cudf_conditional_full_join_size_data_device_allocated_size(
  cudf_conditional_full_join_size_data_handle handle, size_t* out);

/// Return exact materialized gather-map bytes for retained conditional full join size data.
///
/// Conditional full joins report two INT32 gather maps.
cudf_status_t cudf_conditional_full_join_size_data_gather_map_size(
  cudf_conditional_full_join_size_data_handle handle, size_t* out);

/// Materialize conditional full join indices using retained exact output-size
/// data and explicit execution context.
cudf_status_t cudf_conditional_full_join_size_data_materialize_indices_ex(
  cudf_conditional_full_join_size_data_handle size_data,
  cudf_column_handle* out_left_indices,
  cudf_column_handle* out_right_indices,
  cudf_exec_options_t const* options);

/// Conditional left semi join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned index column.
cudf_status_t cudf_conditional_left_semi_join_indices_ex(cudf_table_view_handle left,
                                                         cudf_table_view_handle right,
                                                         cudf_ast_expression_handle predicate,
                                                         cudf_column_handle* out_indices,
                                                         cudf_exec_options_t const* options);

/// Conditional left semi join indices with an exact output row count and
/// explicit execution context.
cudf_status_t cudf_conditional_left_semi_join_indices_with_output_size_ex(
  cudf_table_view_handle left,
  cudf_table_view_handle right,
  cudf_ast_expression_handle predicate,
  size_t output_size,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options);

/// Return the exact output row count for a conditional left semi join with
/// explicit execution context.
cudf_status_t cudf_conditional_left_semi_join_size_ex(cudf_table_view_handle left,
                                                      cudf_table_view_handle right,
                                                      cudf_ast_expression_handle predicate,
                                                      size_t* out,
                                                      cudf_exec_options_t const* options);

/// Conditional left anti join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned index column.
cudf_status_t cudf_conditional_left_anti_join_indices_ex(cudf_table_view_handle left,
                                                         cudf_table_view_handle right,
                                                         cudf_ast_expression_handle predicate,
                                                         cudf_column_handle* out_indices,
                                                         cudf_exec_options_t const* options);

/// Conditional left anti join indices with an exact output row count and
/// explicit execution context.
cudf_status_t cudf_conditional_left_anti_join_indices_with_output_size_ex(
  cudf_table_view_handle left,
  cudf_table_view_handle right,
  cudf_ast_expression_handle predicate,
  size_t output_size,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options);

/// Return the exact output row count for a conditional left anti join with
/// explicit execution context.
cudf_status_t cudf_conditional_left_anti_join_size_ex(cudf_table_view_handle left,
                                                      cudf_table_view_handle right,
                                                      cudf_ast_expression_handle predicate,
                                                      size_t* out,
                                                      cudf_exec_options_t const* options);

/// Filter already-computed join-index pairs with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_filter_join_indices_ex(cudf_table_view_handle left,
                                          cudf_table_view_handle right,
                                          cudf_column_view_handle left_indices,
                                          cudf_column_view_handle right_indices,
                                          cudf_ast_expression_handle predicate,
                                          cudf_join_kind_t join_kind,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options);

/// Mixed inner join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_mixed_inner_join_indices_ex(cudf_table_view_handle left_equality,
                                               cudf_table_view_handle right_equality,
                                               cudf_table_view_handle left_conditional,
                                               cudf_table_view_handle right_conditional,
                                               cudf_ast_expression_handle predicate,
                                               cudf_null_equality_t null_equality,
                                               cudf_column_handle* out_left_indices,
                                               cudf_column_handle* out_right_indices,
                                               cudf_exec_options_t const* options);

/// Mixed left join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_mixed_left_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              cudf_exec_options_t const* options);

/// Mixed full join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_mixed_full_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              cudf_exec_options_t const* options);

/// Mixed left semi join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned index column.
cudf_status_t cudf_mixed_left_semi_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   cudf_exec_options_t const* options);

/// Mixed left anti join indices with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned index column.
cudf_status_t cudf_mixed_left_anti_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   cudf_exec_options_t const* options);

/// Compute and retain exact output-size data for a mixed inner join with
/// explicit execution context.
cudf_status_t cudf_mixed_inner_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                       cudf_table_view_handle right_equality,
                                                       cudf_table_view_handle left_conditional,
                                                       cudf_table_view_handle right_conditional,
                                                       cudf_ast_expression_handle predicate,
                                                       cudf_null_equality_t null_equality,
                                                       cudf_mixed_join_size_data_handle* out,
                                                       cudf_exec_options_t const* options);

/// Compute and retain exact output-size data for a mixed left join with
/// explicit execution context.
cudf_status_t cudf_mixed_left_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                      cudf_table_view_handle right_equality,
                                                      cudf_table_view_handle left_conditional,
                                                      cudf_table_view_handle right_conditional,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_null_equality_t null_equality,
                                                      cudf_mixed_join_size_data_handle* out,
                                                      cudf_exec_options_t const* options);

/// Compute and retain exact output-size data for a mixed full join with
/// explicit execution context.
cudf_status_t cudf_mixed_full_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                      cudf_table_view_handle right_equality,
                                                      cudf_table_view_handle left_conditional,
                                                      cudf_table_view_handle right_conditional,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_null_equality_t null_equality,
                                                      cudf_mixed_join_size_data_handle* out,
                                                      cudf_exec_options_t const* options);

/// Compute and retain exact output-size data for a mixed left semi join with
/// explicit execution context.
cudf_status_t cudf_mixed_left_semi_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                           cudf_table_view_handle right_equality,
                                                           cudf_table_view_handle left_conditional,
                                                           cudf_table_view_handle right_conditional,
                                                           cudf_ast_expression_handle predicate,
                                                           cudf_null_equality_t null_equality,
                                                           cudf_mixed_join_size_data_handle* out,
                                                           cudf_exec_options_t const* options);

/// Compute and retain exact output-size data for a mixed left anti join with
/// explicit execution context.
cudf_status_t cudf_mixed_left_anti_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                           cudf_table_view_handle right_equality,
                                                           cudf_table_view_handle left_conditional,
                                                           cudf_table_view_handle right_conditional,
                                                           cudf_ast_expression_handle predicate,
                                                           cudf_null_equality_t null_equality,
                                                           cudf_mixed_join_size_data_handle* out,
                                                           cudf_exec_options_t const* options);

/// Destroy retained mixed-join output-size data. Ownership is consumed on
/// call; failure quarantines the handle and must not be retried.
cudf_status_t cudf_mixed_join_size_data_destroy(cudf_mixed_join_size_data_handle handle);

/// Return the exact output row count retained in mixed-join size data.
cudf_status_t cudf_mixed_join_size_data_output_size(cudf_mixed_join_size_data_handle handle,
                                                    size_t* out);

/// Return device bytes retained by mixed-join size data.
cudf_status_t cudf_mixed_join_size_data_device_allocated_size(
  cudf_mixed_join_size_data_handle handle, size_t* out);

/// Return exact materialized gather-map bytes for retained mixed-join size data.
///
/// Pair joins report two INT32 gather maps. Left semi/anti joins report one
/// INT32 gather map.
cudf_status_t cudf_mixed_join_size_data_gather_map_size(cudf_mixed_join_size_data_handle handle,
                                                        size_t* out);

/// Materialize mixed join indices using retained exact output-size data and
/// explicit execution context.
cudf_status_t cudf_mixed_join_size_data_materialize_indices_ex(
  cudf_mixed_join_size_data_handle size_data,
  cudf_column_handle* out_left_indices,
  cudf_column_handle* out_right_indices,
  cudf_exec_options_t const* options);

/// Materialize a single left gather map from retained mixed semi/anti size data
/// with explicit execution context.
cudf_status_t cudf_mixed_join_size_data_materialize_single_indices_ex(
  cudf_mixed_join_size_data_handle size_data,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options);

/// Build a reusable distinct-hash-join artifact with explicit execution
/// context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned join handle because libcudf
/// retains build-time hash-table allocations in the handle.
cudf_status_t cudf_distinct_hash_join_create_ex(cudf_table_view_handle build,
                                                cudf_null_equality_t null_equality,
                                                cudf_distinct_hash_join_handle* out,
                                                cudf_exec_options_t const* options);

/// Destroy a reusable distinct-hash-join handle under its build-time device
/// and memory resource. Ownership is consumed on call; success means cleanup
/// completed and failure means the handle remains quarantined.
cudf_status_t cudf_distinct_hash_join_destroy(cudf_distinct_hash_join_handle handle);

/// Probe a reusable distinct-hash-join artifact with explicit execution
/// context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned index column.
cudf_status_t cudf_distinct_hash_join_left_join_ex(cudf_distinct_hash_join_handle join,
                                                   cudf_table_view_handle probe,
                                                   cudf_column_handle* out,
                                                   cudf_exec_options_t const* options);

/// Build a reusable filtered-join artifact with explicit execution context.
cudf_status_t cudf_filtered_join_create_ex(cudf_table_view_handle build,
                                           cudf_null_equality_t null_equality,
                                           double load_factor,
                                           cudf_filtered_join_handle* out,
                                           cudf_exec_options_t const* options);

/// Return exact retained filtered-join build bytes with explicit execution context.
cudf_status_t cudf_filtered_join_pre_build_reservation_size_ex(cudf_table_view_handle build,
                                                               cudf_null_equality_t null_equality,
                                                               double load_factor,
                                                               size_t* out,
                                                               cudf_exec_options_t const* options);

/// Destroy a reusable filtered-join handle under its build-time device and
/// memory resource. Ownership is consumed on call; failure quarantines it.
cudf_status_t cudf_filtered_join_destroy(cudf_filtered_join_handle handle);

/// Return device bytes retained by a reusable filtered-join build artifact.
cudf_status_t cudf_filtered_join_device_allocated_size(cudf_filtered_join_handle handle,
                                                       size_t* out);

/// Begin a retained left semi probe with explicit execution context.
cudf_status_t cudf_filtered_join_begin_left_semi_probe_ex(cudf_filtered_join_handle join,
                                                          cudf_table_view_handle probe,
                                                          cudf_filtered_join_probe_handle* out,
                                                          cudf_exec_options_t const* options);

/// Begin a retained left anti probe with explicit execution context.
cudf_status_t cudf_filtered_join_begin_left_anti_probe_ex(cudf_filtered_join_handle join,
                                                          cudf_table_view_handle probe,
                                                          cudf_filtered_join_probe_handle* out,
                                                          cudf_exec_options_t const* options);

/// Destroy a retained filtered-join probe state. Ownership is consumed on
/// call; failure quarantines the state and must not be retried.
cudf_status_t cudf_filtered_join_probe_destroy(cudf_filtered_join_probe_handle handle);

/// Return the exact output row count retained by a filtered-join probe state.
cudf_status_t cudf_filtered_join_probe_output_size(cudf_filtered_join_probe_handle handle,
                                                   size_t* out);

/// Return device bytes retained by a filtered-join probe state.
cudf_status_t cudf_filtered_join_probe_device_allocated_size(cudf_filtered_join_probe_handle handle,
                                                             size_t* out);

/// Materialize selected probe indices with explicit execution context.
cudf_status_t cudf_filtered_join_probe_materialize_indices_ex(
  cudf_filtered_join_probe_handle handle,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options);

/// Return the exact row count for a left semi probe with explicit execution context.
cudf_status_t cudf_filtered_join_left_semi_join_size_ex(cudf_filtered_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t* out,
                                                        cudf_exec_options_t const* options);

/// Return the exact row count for a left anti probe with explicit execution context.
cudf_status_t cudf_filtered_join_left_anti_join_size_ex(cudf_filtered_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t* out,
                                                        cudf_exec_options_t const* options);

/// Probe a reusable filtered join for left semi gather indices with explicit execution context.
cudf_status_t cudf_filtered_join_left_semi_join_with_output_size_ex(
  cudf_filtered_join_handle join,
  cudf_table_view_handle probe,
  size_t output_size,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options);

/// Probe a reusable filtered join for left anti gather indices with explicit execution context.
cudf_status_t cudf_filtered_join_left_anti_join_with_output_size_ex(
  cudf_filtered_join_handle join,
  cudf_table_view_handle probe,
  size_t output_size,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options);

/// Build a reusable hash-join artifact with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR. The
/// resolved memory resource is retained by the returned join handle for
/// build-time hash-table allocations and destruction.
cudf_status_t cudf_hash_join_create_ex(cudf_table_view_handle build,
                                       cudf_nullable_join_t nullable_join,
                                       cudf_null_equality_t null_equality,
                                       double load_factor,
                                       cudf_hash_join_handle* out,
                                       cudf_exec_options_t const* options);

/// Return exact retained device bytes that must be reserved before building a
/// reusable hash-join artifact with explicit execution context.
///
/// The execution context is part of the contract surface so callers can use the
/// same stream and memory-resource policy as the eventual build call. The call
/// does not allocate device memory; exact flat nullable-string preprocessing is
/// sized by inspecting existing offsets and null masks. Nested build-key tables
/// fail with CUDF_ERROR_INVALID_ARGUMENT.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
cudf_status_t cudf_hash_join_pre_build_reservation_size_ex(cudf_table_view_handle build,
                                                           double load_factor,
                                                           size_t* out,
                                                           cudf_exec_options_t const* options);

/// Return checked hash-table and retained key-view storage bytes from build
/// metadata only. This owner preflight does not construct a table or allocate
/// device memory. `rows` must fit `cudf::size_type`, `load_factor` must be in
/// `(0, 1]`, and `flat_device_view_count` is the number of flat key views.
cudf_status_t cudf_hash_join_metadata_only_owner_preflight_ex(
  size_t rows,
  double load_factor,
  size_t flat_device_view_count,
  size_t* hash_table_storage_bytes,
  size_t* retained_key_view_storage_bytes);

/// Destroy a reusable hash-join handle under its build-time device and memory
/// resource. Ownership is consumed on call; failure quarantines it.
cudf_status_t cudf_hash_join_destroy(cudf_hash_join_handle handle);

/// Return device bytes retained by a reusable hash-join build artifact.
///
/// This excludes the build table viewed by the handle and any per-probe
/// output/index allocations.
cudf_status_t cudf_hash_join_device_allocated_size(cudf_hash_join_handle handle, size_t* out);

/// Probe a reusable hash-join artifact for inner-join gather indices with
/// explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_hash_join_inner_join_ex(cudf_hash_join_handle join,
                                           cudf_table_view_handle probe,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices,
                                           cudf_exec_options_t const* options);

/// Probe a reusable hash-join artifact for inner-join gather indices with an
/// exact output row count and explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_hash_join_inner_join_with_output_size_ex(cudf_hash_join_handle join,
                                                            cudf_table_view_handle probe,
                                                            size_t output_size,
                                                            cudf_column_handle* out_left_indices,
                                                            cudf_column_handle* out_right_indices,
                                                            cudf_exec_options_t const* options);

/// Return the exact row count for an inner probe with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
cudf_status_t cudf_hash_join_inner_join_size_ex(cudf_hash_join_handle join,
                                                cudf_table_view_handle probe,
                                                size_t* out,
                                                cudf_exec_options_t const* options);

/// Probe a reusable hash-join artifact for left-join gather indices with
/// explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_hash_join_left_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options);

/// Probe a reusable hash-join artifact for left-join gather indices with an
/// exact output row count and explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_hash_join_left_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           cudf_exec_options_t const* options);

/// Return the exact row count for a left probe with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
cudf_status_t cudf_hash_join_left_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
                                               cudf_exec_options_t const* options);

/// Probe a reusable hash-join artifact for full-join gather indices with
/// explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_hash_join_full_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options);

/// Probe a reusable hash-join artifact for full-join gather indices with an
/// exact output row count and explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive both returned index columns.
cudf_status_t cudf_hash_join_full_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           cudf_exec_options_t const* options);

/// Return the exact row count for a full probe with explicit execution context.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
cudf_status_t cudf_hash_join_full_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
                                               cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_JOIN_H
