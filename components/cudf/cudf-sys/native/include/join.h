#ifndef CUDF_RUST_JOIN_H
#define CUDF_RUST_JOIN_H

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

/// Opaque handle to a reusable libcudf distinct-hash-join build artifact.
typedef void* cudf_distinct_hash_join_handle;

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
/// Passing NULL for `options`, or leaving `options->stream` NULL, uses cuDF's
/// default stream. `options->mr` is accepted for ABI consistency but the
/// distinct-hash-join build path does not expose an allocator override in the
/// underlying libcudf constructor.
cudf_status_t cudf_distinct_hash_join_create_ex(cudf_table_view_handle build,
                                                cudf_null_equality_t null_equality,
                                                cudf_distinct_hash_join_handle* out,
                                                const cudf_exec_options_t* options);

/// Destroy a reusable distinct-hash-join handle.
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

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_JOIN_H
