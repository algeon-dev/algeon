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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned index
/// columns.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned index
/// columns.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned index
/// columns.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned index column.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned index column.
cudf_status_t cudf_conditional_left_anti_join_indices_ex(cudf_table_view_handle left,
                                                         cudf_table_view_handle right,
                                                         cudf_ast_expression_handle predicate,
                                                         cudf_column_handle* out_indices,
                                                         const cudf_exec_options_t* options);

/// Filter already-computed join-index pairs with an AST predicate.
///
/// `left_indices` and `right_indices` must be non-nullable INT32 column views
/// of equal length. Only inner, left, and full join kinds are supported. The
/// returned INT32 gather-index columns encode unmatched rows with libcudf's
/// no-match sentinel, not with nulls.
cudf_status_t cudf_filter_join_indices(cudf_table_view_handle left,
                                       cudf_table_view_handle right,
                                       cudf_column_view_handle left_indices,
                                       cudf_column_view_handle right_indices,
                                       cudf_ast_expression_handle predicate,
                                       cudf_join_kind_t join_kind,
                                       cudf_column_handle* out_left_indices,
                                       cudf_column_handle* out_right_indices);

/// Filter already-computed join-index pairs with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource. When
/// `options->mr` is non-NULL, it must outlive both returned index columns.
cudf_status_t cudf_filter_join_indices_ex(cudf_table_view_handle left,
                                          cudf_table_view_handle right,
                                          cudf_column_view_handle left_indices,
                                          cudf_column_view_handle right_indices,
                                          cudf_ast_expression_handle predicate,
                                          cudf_join_kind_t join_kind,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned index
/// columns.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned index
/// columns.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive both returned index
/// columns.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned index column.
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
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned index column.
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

/// Return exact retained device bytes that must be reserved before building a
/// reusable hash-join artifact.
///
/// This is a planning contract, not a build operation. It must not allocate a
/// hash table or retain the build table. The result covers the device memory
/// retained by the reusable hash-table artifact for supported build-key tables.
/// Build-key tables that require unsupported retained preprocessing buffers
/// (nested keys, or keys that may have non-empty nulls such as nullable string
/// keys) fail with CUDF_ERROR_INVALID_ARGUMENT.
cudf_status_t cudf_hash_join_pre_build_reservation_size(cudf_table_view_handle build,
                                                        double load_factor,
                                                        size_t* out);

/// Return exact retained device bytes that must be reserved before building a
/// reusable hash-join artifact with explicit execution context.
///
/// The execution context is part of the contract surface so callers can use the
/// same stream and memory-resource policy as the eventual build call. Build-key
/// tables that require unsupported retained preprocessing buffers (nested keys,
/// or keys that may have non-empty nulls such as nullable string keys) fail
/// with CUDF_ERROR_INVALID_ARGUMENT.
cudf_status_t cudf_hash_join_pre_build_reservation_size_ex(cudf_table_view_handle build,
                                                           double load_factor,
                                                           size_t* out,
                                                           const cudf_exec_options_t* options);

/// Destroy a reusable hash-join handle under its build-time device and memory
/// resource.
void cudf_hash_join_destroy(cudf_hash_join_handle handle);

/// Return device bytes retained by a reusable hash-join build artifact.
///
/// This excludes the build table viewed by the handle and any per-probe
/// output/index allocations.
cudf_status_t cudf_hash_join_device_allocated_size(cudf_hash_join_handle handle, size_t* out);

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

/// Probe a reusable hash-join artifact for inner-join gather indices with an
/// exact output row count supplied by the caller.
cudf_status_t cudf_hash_join_inner_join_with_output_size(cudf_hash_join_handle join,
                                                         cudf_table_view_handle probe,
                                                         size_t output_size,
                                                         cudf_column_handle* out_left_indices,
                                                         cudf_column_handle* out_right_indices);

/// Probe a reusable hash-join artifact for inner-join gather indices with an
/// exact output row count and explicit execution context.
cudf_status_t cudf_hash_join_inner_join_with_output_size_ex(cudf_hash_join_handle join,
                                                            cudf_table_view_handle probe,
                                                            size_t output_size,
                                                            cudf_column_handle* out_left_indices,
                                                            cudf_column_handle* out_right_indices,
                                                            const cudf_exec_options_t* options);

/// Return the exact row count for an inner probe of a reusable hash-join artifact.
cudf_status_t cudf_hash_join_inner_join_size(cudf_hash_join_handle join,
                                             cudf_table_view_handle probe,
                                             size_t* out);

/// Return the exact row count for an inner probe with explicit execution context.
cudf_status_t cudf_hash_join_inner_join_size_ex(cudf_hash_join_handle join,
                                                cudf_table_view_handle probe,
                                                size_t* out,
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

/// Probe a reusable hash-join artifact for left-join gather indices with an
/// exact output row count supplied by the caller.
cudf_status_t cudf_hash_join_left_join_with_output_size(cudf_hash_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t output_size,
                                                        cudf_column_handle* out_left_indices,
                                                        cudf_column_handle* out_right_indices);

/// Probe a reusable hash-join artifact for left-join gather indices with an
/// exact output row count and explicit execution context.
cudf_status_t cudf_hash_join_left_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           const cudf_exec_options_t* options);

/// Return the exact row count for a left probe of a reusable hash-join artifact.
cudf_status_t cudf_hash_join_left_join_size(cudf_hash_join_handle join,
                                            cudf_table_view_handle probe,
                                            size_t* out);

/// Return the exact row count for a left probe with explicit execution context.
cudf_status_t cudf_hash_join_left_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
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

/// Probe a reusable hash-join artifact for full-join gather indices with an
/// exact output row count supplied by the caller.
cudf_status_t cudf_hash_join_full_join_with_output_size(cudf_hash_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t output_size,
                                                        cudf_column_handle* out_left_indices,
                                                        cudf_column_handle* out_right_indices);

/// Probe a reusable hash-join artifact for full-join gather indices with an
/// exact output row count and explicit execution context.
cudf_status_t cudf_hash_join_full_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           const cudf_exec_options_t* options);

/// Return the exact row count for a full probe of a reusable hash-join artifact.
cudf_status_t cudf_hash_join_full_join_size(cudf_hash_join_handle join,
                                            cudf_table_view_handle probe,
                                            size_t* out);

/// Return the exact row count for a full probe with explicit execution context.
cudf_status_t cudf_hash_join_full_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
                                               const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_JOIN_H
