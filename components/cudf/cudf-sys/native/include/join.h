#ifndef CUDF_RUST_JOIN_H
#define CUDF_RUST_JOIN_H

#include "ast.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"

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

/// Opaque handle to a reusable libcudf distinct-hash-join build artifact.
typedef void* cudf_distinct_hash_join_handle;

/// Opaque handle to a reusable libcudf hash-join build artifact.
typedef void* cudf_hash_join_handle;

/// Inner join. Returns materialized left and right result tables.
/// The left result contains left_keys columns + left_payload columns for
/// matching rows. Similarly for the right result.
cudf_status_t cudf_inner_join(cudf_table_view_handle left_keys,
                              cudf_table_view_handle right_keys,
                              cudf_table_view_handle left_payload,
                              cudf_table_view_handle right_payload,
                              cudf_null_equality_t null_equality,
                              cudf_table_handle* out_left,
                              cudf_table_handle* out_right);

/// Inner join with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned tables.
cudf_status_t cudf_inner_join_ex(cudf_table_view_handle left_keys,
                                 cudf_table_view_handle right_keys,
                                 cudf_table_view_handle left_payload,
                                 cudf_table_view_handle right_payload,
                                 cudf_null_equality_t null_equality,
                                 cudf_table_handle* out_left,
                                 cudf_table_handle* out_right,
                                 const cudf_exec_options_t* options);

/// Left join. All left rows are preserved; right side is null-padded
/// for non-matching rows.
cudf_status_t cudf_left_join(cudf_table_view_handle left_keys,
                             cudf_table_view_handle right_keys,
                             cudf_table_view_handle left_payload,
                             cudf_table_view_handle right_payload,
                             cudf_null_equality_t null_equality,
                             cudf_table_handle* out_left,
                             cudf_table_handle* out_right);

/// Left join with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned tables.
cudf_status_t cudf_left_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                const cudf_exec_options_t* options);

/// Full (outer) join. All rows from both sides are preserved.
cudf_status_t cudf_full_join(cudf_table_view_handle left_keys,
                             cudf_table_view_handle right_keys,
                             cudf_table_view_handle left_payload,
                             cudf_table_view_handle right_payload,
                             cudf_null_equality_t null_equality,
                             cudf_table_handle* out_left,
                             cudf_table_handle* out_right);

/// Full join with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned tables.
cudf_status_t cudf_full_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                const cudf_exec_options_t* options);

/// Cross join. Returns a materialized table containing left columns followed by
/// right columns.
cudf_status_t cudf_cross_join(cudf_table_view_handle left,
                              cudf_table_view_handle right,
                              cudf_table_handle* out);

/// Cross join with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned table.
cudf_status_t cudf_cross_join_ex(cudf_table_view_handle left,
                                 cudf_table_view_handle right,
                                 cudf_table_handle* out,
                                 const cudf_exec_options_t* options);

/// Inner join. Returns INT32 left and right gather-index columns.
cudf_status_t cudf_inner_join_indices(cudf_table_view_handle left_keys,
                                      cudf_table_view_handle right_keys,
                                      cudf_null_equality_t null_equality,
                                      cudf_column_handle* out_left_indices,
                                      cudf_column_handle* out_right_indices);

/// Inner join indices with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
cudf_status_t cudf_inner_join_indices_ex(cudf_table_view_handle left_keys,
                                         cudf_table_view_handle right_keys,
                                         cudf_null_equality_t null_equality,
                                         cudf_column_handle* out_left_indices,
                                         cudf_column_handle* out_right_indices,
                                         const cudf_exec_options_t* options);

/// Left join. Returns INT32 left and right gather-index columns.
cudf_status_t cudf_left_join_indices(cudf_table_view_handle left_keys,
                                     cudf_table_view_handle right_keys,
                                     cudf_null_equality_t null_equality,
                                     cudf_column_handle* out_left_indices,
                                     cudf_column_handle* out_right_indices);

/// Left join indices with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
cudf_status_t cudf_left_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        const cudf_exec_options_t* options);

/// Full join. Returns INT32 left and right gather-index columns.
cudf_status_t cudf_full_join_indices(cudf_table_view_handle left_keys,
                                     cudf_table_view_handle right_keys,
                                     cudf_null_equality_t null_equality,
                                     cudf_column_handle* out_left_indices,
                                     cudf_column_handle* out_right_indices);

/// Full join indices with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
cudf_status_t cudf_full_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        const cudf_exec_options_t* options);

/// Left semi join. Returns an INT32 column of left/probe row indices that have
/// at least one match in the right/filter table.
cudf_status_t cudf_left_semi_join_indices(cudf_table_view_handle left_keys,
                                          cudf_table_view_handle right_keys,
                                          cudf_null_equality_t null_equality,
                                          cudf_column_handle* out_indices);

/// Left semi join indices with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
cudf_status_t cudf_left_semi_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             const cudf_exec_options_t* options);

/// Left anti join. Returns an INT32 column of left/probe row indices that have
/// no match in the right/filter table.
cudf_status_t cudf_left_anti_join_indices(cudf_table_view_handle left_keys,
                                          cudf_table_view_handle right_keys,
                                          cudf_null_equality_t null_equality,
                                          cudf_column_handle* out_indices);

/// Left anti join indices with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
cudf_status_t cudf_left_anti_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             const cudf_exec_options_t* options);

/// Conditional inner join. Returns INT32 left and right gather-index columns.
cudf_status_t cudf_conditional_inner_join_indices(cudf_table_view_handle left,
                                                  cudf_table_view_handle right,
                                                  cudf_ast_expression_handle predicate,
                                                  cudf_column_handle* out_left_indices,
                                                  cudf_column_handle* out_right_indices);

/// Conditional inner join indices with explicit execution context.
cudf_status_t cudf_conditional_inner_join_indices_ex(cudf_table_view_handle left,
                                                     cudf_table_view_handle right,
                                                     cudf_ast_expression_handle predicate,
                                                     cudf_column_handle* out_left_indices,
                                                     cudf_column_handle* out_right_indices,
                                                     const cudf_exec_options_t* options);

/// Conditional left join. Returns INT32 left and right gather-index columns.
cudf_status_t cudf_conditional_left_join_indices(cudf_table_view_handle left,
                                                 cudf_table_view_handle right,
                                                 cudf_ast_expression_handle predicate,
                                                 cudf_column_handle* out_left_indices,
                                                 cudf_column_handle* out_right_indices);

/// Conditional left join indices with explicit execution context.
cudf_status_t cudf_conditional_left_join_indices_ex(cudf_table_view_handle left,
                                                    cudf_table_view_handle right,
                                                    cudf_ast_expression_handle predicate,
                                                    cudf_column_handle* out_left_indices,
                                                    cudf_column_handle* out_right_indices,
                                                    const cudf_exec_options_t* options);

/// Conditional full join. Returns INT32 left and right gather-index columns.
cudf_status_t cudf_conditional_full_join_indices(cudf_table_view_handle left,
                                                 cudf_table_view_handle right,
                                                 cudf_ast_expression_handle predicate,
                                                 cudf_column_handle* out_left_indices,
                                                 cudf_column_handle* out_right_indices);

/// Conditional full join indices with explicit execution context.
cudf_status_t cudf_conditional_full_join_indices_ex(cudf_table_view_handle left,
                                                    cudf_table_view_handle right,
                                                    cudf_ast_expression_handle predicate,
                                                    cudf_column_handle* out_left_indices,
                                                    cudf_column_handle* out_right_indices,
                                                    const cudf_exec_options_t* options);

/// Conditional left semi join. Returns an INT32 column of left row indices.
cudf_status_t cudf_conditional_left_semi_join_indices(cudf_table_view_handle left,
                                                      cudf_table_view_handle right,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_column_handle* out_indices);

/// Conditional left semi join indices with explicit execution context.
cudf_status_t cudf_conditional_left_semi_join_indices_ex(cudf_table_view_handle left,
                                                         cudf_table_view_handle right,
                                                         cudf_ast_expression_handle predicate,
                                                         cudf_column_handle* out_indices,
                                                         const cudf_exec_options_t* options);

/// Conditional left anti join. Returns an INT32 column of left row indices.
cudf_status_t cudf_conditional_left_anti_join_indices(cudf_table_view_handle left,
                                                      cudf_table_view_handle right,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_column_handle* out_indices);

/// Conditional left anti join indices with explicit execution context.
cudf_status_t cudf_conditional_left_anti_join_indices_ex(cudf_table_view_handle left,
                                                         cudf_table_view_handle right,
                                                         cudf_ast_expression_handle predicate,
                                                         cudf_column_handle* out_indices,
                                                         const cudf_exec_options_t* options);

/// Mixed inner join. Equality tables are hash-joined first and `predicate` is
/// evaluated against the conditional tables. Returns INT32 gather-index
/// columns.
cudf_status_t cudf_mixed_inner_join_indices(cudf_table_view_handle left_equality,
                                            cudf_table_view_handle right_equality,
                                            cudf_table_view_handle left_conditional,
                                            cudf_table_view_handle right_conditional,
                                            cudf_ast_expression_handle predicate,
                                            cudf_null_equality_t null_equality,
                                            cudf_column_handle* out_left_indices,
                                            cudf_column_handle* out_right_indices);

/// Mixed inner join indices with explicit execution context.
cudf_status_t cudf_mixed_inner_join_indices_ex(cudf_table_view_handle left_equality,
                                               cudf_table_view_handle right_equality,
                                               cudf_table_view_handle left_conditional,
                                               cudf_table_view_handle right_conditional,
                                               cudf_ast_expression_handle predicate,
                                               cudf_null_equality_t null_equality,
                                               cudf_column_handle* out_left_indices,
                                               cudf_column_handle* out_right_indices,
                                               const cudf_exec_options_t* options);

/// Mixed left join. Returns INT32 gather-index columns.
cudf_status_t cudf_mixed_left_join_indices(cudf_table_view_handle left_equality,
                                           cudf_table_view_handle right_equality,
                                           cudf_table_view_handle left_conditional,
                                           cudf_table_view_handle right_conditional,
                                           cudf_ast_expression_handle predicate,
                                           cudf_null_equality_t null_equality,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices);

/// Mixed left join indices with explicit execution context.
cudf_status_t cudf_mixed_left_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              const cudf_exec_options_t* options);

/// Mixed full join. Returns INT32 gather-index columns.
cudf_status_t cudf_mixed_full_join_indices(cudf_table_view_handle left_equality,
                                           cudf_table_view_handle right_equality,
                                           cudf_table_view_handle left_conditional,
                                           cudf_table_view_handle right_conditional,
                                           cudf_ast_expression_handle predicate,
                                           cudf_null_equality_t null_equality,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices);

/// Mixed full join indices with explicit execution context.
cudf_status_t cudf_mixed_full_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              const cudf_exec_options_t* options);

/// Mixed left semi join. Returns an INT32 column of left row indices.
cudf_status_t cudf_mixed_left_semi_join_indices(cudf_table_view_handle left_equality,
                                                cudf_table_view_handle right_equality,
                                                cudf_table_view_handle left_conditional,
                                                cudf_table_view_handle right_conditional,
                                                cudf_ast_expression_handle predicate,
                                                cudf_null_equality_t null_equality,
                                                cudf_column_handle* out_indices);

/// Mixed left semi join indices with explicit execution context.
cudf_status_t cudf_mixed_left_semi_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   const cudf_exec_options_t* options);

/// Mixed left anti join. Returns an INT32 column of left row indices.
cudf_status_t cudf_mixed_left_anti_join_indices(cudf_table_view_handle left_equality,
                                                cudf_table_view_handle right_equality,
                                                cudf_table_view_handle left_conditional,
                                                cudf_table_view_handle right_conditional,
                                                cudf_ast_expression_handle predicate,
                                                cudf_null_equality_t null_equality,
                                                cudf_column_handle* out_indices);

/// Mixed left anti join indices with explicit execution context.
cudf_status_t cudf_mixed_left_anti_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   const cudf_exec_options_t* options);

/// Build a reusable distinct-hash-join artifact from a build-key table.
///
/// The build table must contain distinct rows. The returned handle borrows the
/// underlying build-column device memory, so the owning columns must outlive
/// the join handle.
cudf_status_t cudf_distinct_hash_join_create(cudf_table_view_handle build,
                                             cudf_null_equality_t null_equality,
                                             cudf_distinct_hash_join_handle* out);

/// Build a reusable distinct-hash-join artifact with explicit execution
/// context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource. When
/// `options->mr` is non-NULL, it must outlive the returned join handle because
/// libcudf retains build-time hash-table allocations in the handle.
cudf_status_t cudf_distinct_hash_join_create_ex(cudf_table_view_handle build,
                                                cudf_null_equality_t null_equality,
                                                cudf_distinct_hash_join_handle* out,
                                                const cudf_exec_options_t* options);

/// Destroy a reusable distinct-hash-join handle under its build-time device
/// and memory resource.
void cudf_distinct_hash_join_destroy(cudf_distinct_hash_join_handle handle);

/// Probe a reusable distinct-hash-join artifact and return build-row indices.
///
/// The returned INT32 column is row-aligned with `probe`. Matching rows contain
/// the matched build-row index; misses contain libcudf's `JoinNoMatch`
/// sentinel. Callers typically feed this column into gather with NULLIFY.
cudf_status_t cudf_distinct_hash_join_left_join(cudf_distinct_hash_join_handle join,
                                                cudf_table_view_handle probe,
                                                cudf_column_handle* out);

/// Probe a reusable distinct-hash-join artifact with explicit execution
/// context. The returned indices column is allocated using `options->mr` when
/// provided.
cudf_status_t cudf_distinct_hash_join_left_join_ex(cudf_distinct_hash_join_handle join,
                                                   cudf_table_view_handle probe,
                                                   cudf_column_handle* out,
                                                   const cudf_exec_options_t* options);

/// Build a reusable hash-join artifact from a build-key table.
///
/// The returned handle borrows the underlying build-column device memory, so
/// the owning columns must outlive the join handle.
cudf_status_t cudf_hash_join_create(cudf_table_view_handle build,
                                    cudf_nullable_join_t nullable_join,
                                    cudf_null_equality_t null_equality,
                                    double load_factor,
                                    cudf_hash_join_handle* out);

/// Build a reusable hash-join artifact with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource. When
/// `options->mr` is non-NULL, it must outlive the returned join handle because
/// libcudf retains build-time hash-table allocations in the handle.
cudf_status_t cudf_hash_join_create_ex(cudf_table_view_handle build,
                                       cudf_nullable_join_t nullable_join,
                                       cudf_null_equality_t null_equality,
                                       double load_factor,
                                       cudf_hash_join_handle* out,
                                       const cudf_exec_options_t* options);

/// Destroy a reusable hash-join handle under its build-time device and memory
/// resource.
void cudf_hash_join_destroy(cudf_hash_join_handle handle);

/// Probe a reusable hash-join artifact for inner-join gather indices.
///
/// The probe table is the logical left side and the build table is the logical
/// right side. Returned INT32 columns are left/probe indices and right/build
/// indices.
cudf_status_t cudf_hash_join_inner_join(cudf_hash_join_handle join,
                                        cudf_table_view_handle probe,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices);

/// Probe a reusable hash-join artifact for inner-join gather indices with
/// explicit execution context.
cudf_status_t cudf_hash_join_inner_join_ex(cudf_hash_join_handle join,
                                           cudf_table_view_handle probe,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices,
                                           const cudf_exec_options_t* options);

/// Probe a reusable hash-join artifact for left-join gather indices.
cudf_status_t cudf_hash_join_left_join(cudf_hash_join_handle join,
                                       cudf_table_view_handle probe,
                                       cudf_column_handle* out_left_indices,
                                       cudf_column_handle* out_right_indices);

/// Probe a reusable hash-join artifact for left-join gather indices with
/// explicit execution context.
cudf_status_t cudf_hash_join_left_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          const cudf_exec_options_t* options);

/// Probe a reusable hash-join artifact for full-join gather indices.
cudf_status_t cudf_hash_join_full_join(cudf_hash_join_handle join,
                                       cudf_table_view_handle probe,
                                       cudf_column_handle* out_left_indices,
                                       cudf_column_handle* out_right_indices);

/// Probe a reusable hash-join artifact for full-join gather indices with
/// explicit execution context.
cudf_status_t cudf_hash_join_full_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_JOIN_H
