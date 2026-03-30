#ifndef CUDF_RUST_BINARY_H
#define CUDF_RUST_BINARY_H

#include "error.h"
#include "column.h"
#include "execution_context.h"
#include "scalar.h"
#include "types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Binary operators matching cudf::binary_operator (same integer values).
typedef enum {
  CUDF_BINARY_ADD = 0,         // 0
  CUDF_BINARY_SUB,             // 1
  CUDF_BINARY_MUL,             // 2
  CUDF_BINARY_DIV,             // 3
  CUDF_BINARY_TRUE_DIV,        // 4
  CUDF_BINARY_FLOOR_DIV,       // 5
  CUDF_BINARY_MOD,             // 6
  CUDF_BINARY_PMOD,            // 7
  CUDF_BINARY_PYMOD,           // 8
  CUDF_BINARY_POW,             // 9
  CUDF_BINARY_INT_POW,         // 10
  CUDF_BINARY_LOG_BASE,        // 11
  CUDF_BINARY_ATAN2,           // 12
  CUDF_BINARY_SHIFT_LEFT,      // 13
  CUDF_BINARY_SHIFT_RIGHT,     // 14
  CUDF_BINARY_SHIFT_RIGHT_UNSIGNED, // 15
  CUDF_BINARY_BITWISE_AND,     // 16
  CUDF_BINARY_BITWISE_OR,      // 17
  CUDF_BINARY_BITWISE_XOR,     // 18
  CUDF_BINARY_LOGICAL_AND,     // 19
  CUDF_BINARY_LOGICAL_OR,      // 20
  CUDF_BINARY_EQUAL,           // 21
  CUDF_BINARY_NOT_EQUAL,       // 22
  CUDF_BINARY_LESS,            // 23
  CUDF_BINARY_GREATER,         // 24
  CUDF_BINARY_LESS_EQUAL,      // 25
  CUDF_BINARY_GREATER_EQUAL,   // 26
  CUDF_BINARY_NULL_EQUALS,     // 27
  CUDF_BINARY_NULL_NOT_EQUALS, // 28
  CUDF_BINARY_NULL_MAX,        // 29
  CUDF_BINARY_NULL_MIN,        // 30
} cudf_binary_operator_t;

/// Binary operation: column op column.
cudf_status_t cudf_binary_operation_vv(cudf_column_view_handle lhs,
                                       cudf_column_view_handle rhs,
                                       cudf_binary_operator_t op,
                                       cudf_data_type_handle output_type,
                                       cudf_column_handle* out);

/// Binary operation: column op column with explicit execution context.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_binary_operation_vv_ex(cudf_column_view_handle lhs,
                                          cudf_column_view_handle rhs,
                                          cudf_binary_operator_t op,
                                          cudf_data_type_handle output_type,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

/// Binary operation: column op scalar.
cudf_status_t cudf_binary_operation_vs(cudf_column_view_handle lhs,
                                       cudf_scalar_handle rhs,
                                       cudf_binary_operator_t op,
                                       cudf_data_type_handle output_type,
                                       cudf_column_handle* out);

/// Binary operation: column op scalar with explicit execution context.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_binary_operation_vs_ex(cudf_column_view_handle lhs,
                                          cudf_scalar_handle rhs,
                                          cudf_binary_operator_t op,
                                          cudf_data_type_handle output_type,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

/// Binary operation: scalar op column.
cudf_status_t cudf_binary_operation_sv(cudf_scalar_handle lhs,
                                       cudf_column_view_handle rhs,
                                       cudf_binary_operator_t op,
                                       cudf_data_type_handle output_type,
                                       cudf_column_handle* out);

/// Binary operation: scalar op column with explicit execution context.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_binary_operation_sv_ex(cudf_scalar_handle lhs,
                                          cudf_column_view_handle rhs,
                                          cudf_binary_operator_t op,
                                          cudf_data_type_handle output_type,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_BINARY_H
