#ifndef CUDF_RUST_AST_H
#define CUDF_RUST_AST_H

#include "error.h"
#include "scalar.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a libcudf AST expression node.
typedef void* cudf_ast_expression_handle;

/// Which input table a column reference reads from when evaluating a binary
/// predicate.
typedef enum {
  CUDF_AST_TABLE_REFERENCE_LEFT   = 0,
  CUDF_AST_TABLE_REFERENCE_RIGHT  = 1,
  CUDF_AST_TABLE_REFERENCE_OUTPUT = 2,
} cudf_ast_table_reference_t;

/// Operators supported by libcudf AST expressions.
typedef enum {
  CUDF_AST_ADD              = 0,
  CUDF_AST_SUB              = 1,
  CUDF_AST_MUL              = 2,
  CUDF_AST_DIV              = 3,
  CUDF_AST_TRUE_DIV         = 4,
  CUDF_AST_FLOOR_DIV        = 5,
  CUDF_AST_MOD              = 6,
  CUDF_AST_PYMOD            = 7,
  CUDF_AST_POW              = 8,
  CUDF_AST_EQUAL            = 9,
  CUDF_AST_NULL_EQUAL       = 10,
  CUDF_AST_NOT_EQUAL        = 11,
  CUDF_AST_LESS             = 12,
  CUDF_AST_GREATER          = 13,
  CUDF_AST_LESS_EQUAL       = 14,
  CUDF_AST_GREATER_EQUAL    = 15,
  CUDF_AST_BITWISE_AND      = 16,
  CUDF_AST_BITWISE_OR       = 17,
  CUDF_AST_BITWISE_XOR      = 18,
  CUDF_AST_LOGICAL_AND      = 19,
  CUDF_AST_NULL_LOGICAL_AND = 20,
  CUDF_AST_LOGICAL_OR       = 21,
  CUDF_AST_NULL_LOGICAL_OR  = 22,
  CUDF_AST_IDENTITY         = 23,
  CUDF_AST_IS_NULL          = 24,
  CUDF_AST_SIN              = 25,
  CUDF_AST_COS              = 26,
  CUDF_AST_TAN              = 27,
  CUDF_AST_ARCSIN           = 28,
  CUDF_AST_ARCCOS           = 29,
  CUDF_AST_ARCTAN           = 30,
  CUDF_AST_SINH             = 31,
  CUDF_AST_COSH             = 32,
  CUDF_AST_TANH             = 33,
  CUDF_AST_ARCSINH          = 34,
  CUDF_AST_ARCCOSH          = 35,
  CUDF_AST_ARCTANH          = 36,
  CUDF_AST_EXP              = 37,
  CUDF_AST_LOG              = 38,
  CUDF_AST_SQRT             = 39,
  CUDF_AST_CBRT             = 40,
  CUDF_AST_CEIL             = 41,
  CUDF_AST_FLOOR            = 42,
  CUDF_AST_ABS              = 43,
  CUDF_AST_RINT             = 44,
  CUDF_AST_BIT_INVERT       = 45,
  CUDF_AST_NOT              = 46,
  CUDF_AST_CAST_TO_INT64    = 47,
  CUDF_AST_CAST_TO_UINT64   = 48,
  CUDF_AST_CAST_TO_FLOAT64  = 49,
} cudf_ast_operator_t;

/// Create a column-reference expression.
cudf_status_t cudf_ast_column_reference_create(int32_t column_index,
                                               cudf_ast_table_reference_t table_source,
                                               cudf_ast_expression_handle* out);

/// Create a literal expression borrowing `scalar`.
///
/// The scalar handle must outlive every join or transform call that evaluates
/// the returned AST expression.
cudf_status_t cudf_ast_literal_create(cudf_scalar_handle scalar, cudf_ast_expression_handle* out);

/// Create a unary operation expression. The returned expression retains the
/// input AST handle internally, so destroying `input` after this call does not
/// invalidate the returned operation expression.
cudf_status_t cudf_ast_operation_unary_create(cudf_ast_operator_t op,
                                              cudf_ast_expression_handle input,
                                              cudf_ast_expression_handle* out);

/// Create a binary operation expression. The returned expression retains both
/// child AST handles internally, so destroying either child after this call
/// does not invalidate the returned operation expression.
cudf_status_t cudf_ast_operation_binary_create(cudf_ast_operator_t op,
                                               cudf_ast_expression_handle left,
                                               cudf_ast_expression_handle right,
                                               cudf_ast_expression_handle* out);

/// Destroy an AST expression handle. Ownership is consumed on call; success
/// means cleanup completed and failure means the handle is quarantined.
cudf_status_t cudf_ast_expression_destroy(cudf_ast_expression_handle handle);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_AST_H
