#ifndef CUDF_RUST_UNARY_H
#define CUDF_RUST_UNARY_H

#include "error.h"
#include "column.h"
#include "types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Unary operators matching cudf::unary_operator.
typedef enum {
  CUDF_UNARY_SIN = 0,
  CUDF_UNARY_COS,
  CUDF_UNARY_TAN,
  CUDF_UNARY_ARCSIN,
  CUDF_UNARY_ARCCOS,
  CUDF_UNARY_ARCTAN,
  CUDF_UNARY_SINH,
  CUDF_UNARY_COSH,
  CUDF_UNARY_TANH,
  CUDF_UNARY_ARCSINH,
  CUDF_UNARY_ARCCOSH,
  CUDF_UNARY_ARCTANH,
  CUDF_UNARY_EXP,
  CUDF_UNARY_LOG,
  CUDF_UNARY_SQRT,
  CUDF_UNARY_CBRT,
  CUDF_UNARY_CEIL,
  CUDF_UNARY_FLOOR,
  CUDF_UNARY_ABS,
  CUDF_UNARY_RINT,
  CUDF_UNARY_BIT_COUNT,
  CUDF_UNARY_BIT_INVERT,
  CUDF_UNARY_NOT,
  CUDF_UNARY_NEGATE,
} cudf_unary_operator_t;

/// Apply a unary operation element-wise.
cudf_status_t cudf_unary_operation(cudf_column_view_handle input,
                                   cudf_unary_operator_t op,
                                   cudf_column_handle* out);

/// Return a boolean column: true where elements are null.
cudf_status_t cudf_is_null(cudf_column_view_handle input,
                           cudf_column_handle* out);

/// Return a boolean column: true where elements are valid (non-null).
cudf_status_t cudf_is_valid(cudf_column_view_handle input,
                            cudf_column_handle* out);

/// Cast a column to the specified data type.
cudf_status_t cudf_cast(cudf_column_view_handle input,
                        cudf_data_type_handle output_type,
                        cudf_column_handle* out);

/// Check whether a cast between two data types is supported.
cudf_status_t cudf_is_supported_cast(cudf_data_type_handle from_type,
                                     cudf_data_type_handle to_type,
                                     int32_t* out);

/// Return a boolean column: true where elements are NaN.
cudf_status_t cudf_is_nan(cudf_column_view_handle input,
                          cudf_column_handle* out);

/// Return a boolean column: true where elements are not NaN.
cudf_status_t cudf_is_not_nan(cudf_column_view_handle input,
                              cudf_column_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_UNARY_H
