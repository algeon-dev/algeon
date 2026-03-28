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
  CUDF_NULL_EQUALITY_EQUAL = 0,
  CUDF_NULL_EQUALITY_UNEQUAL = 1,
} cudf_null_equality_t;

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
cudf_status_t cudf_full_join_ex(cudf_table_view_handle left_keys,
                                 cudf_table_view_handle right_keys,
                                 cudf_table_view_handle left_payload,
                                 cudf_table_view_handle right_payload,
                                 cudf_null_equality_t null_equality,
                                 cudf_table_handle* out_left,
                                 cudf_table_handle* out_right,
                                 const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_JOIN_H
