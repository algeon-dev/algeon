#ifndef CUDF_RUST_FILTER_H
#define CUDF_RUST_FILTER_H

#include "binary.h"
#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "scalar.h"
#include "table.h"
#include "unary.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Apply a boolean mask to a table. Rows where the mask is `true` are kept;
/// rows where the mask is `false` or null are dropped.
/// The mask must be a BOOL8 column with the same number of rows as the table.
cudf_status_t cudf_apply_boolean_mask(cudf_table_view_handle input,
                                      cudf_column_view_handle mask,
                                      cudf_table_handle* out);

/// Apply a boolean mask with explicit execution options.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned table.
cudf_status_t cudf_apply_boolean_mask_ex(cudf_table_view_handle input,
                                         cudf_column_view_handle mask,
                                         cudf_table_handle* out,
                                         const cudf_exec_options_t* options);

/// Node kinds for a validated row-predicate AST.
typedef enum {
  CUDF_ROW_PREDICATE_NODE_COLUMN_REF = 0,
  CUDF_ROW_PREDICATE_NODE_LITERAL    = 1,
  CUDF_ROW_PREDICATE_NODE_BINARY_OP  = 2,
  CUDF_ROW_PREDICATE_NODE_UNARY_OP   = 3,
  CUDF_ROW_PREDICATE_NODE_IS_NULL    = 4,
} cudf_row_predicate_node_kind_t;

/// One lowered row-predicate AST node.
typedef struct {
  cudf_row_predicate_node_kind_t kind;
  int32_t lhs_index;
  int32_t rhs_index;
  int32_t column_index;
  cudf_binary_operator_t binary_op;
  cudf_unary_operator_t unary_op;
  cudf_scalar_handle literal;
} cudf_row_predicate_node_t;

/// Evaluate a validated row predicate against a table and filter rows once.
cudf_status_t cudf_filter_rows_by_predicate(cudf_table_view_handle input,
                                            cudf_row_predicate_node_t const* nodes,
                                            int32_t num_nodes,
                                            int32_t root_index,
                                            cudf_table_handle* out);

/// Evaluate a validated row predicate with explicit execution options.
cudf_status_t cudf_filter_rows_by_predicate_ex(cudf_table_view_handle input,
                                               cudf_row_predicate_node_t const* nodes,
                                               int32_t num_nodes,
                                               int32_t root_index,
                                               cudf_table_handle* out,
                                               const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_FILTER_H
