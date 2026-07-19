#ifndef CUDF_RUST_DATETIME_H
#define CUDF_RUST_DATETIME_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Datetime components wrapped by cudf.
typedef enum {
  CUDF_DATETIME_COMPONENT_YEAR  = 0,
  CUDF_DATETIME_COMPONENT_MONTH = 1,
  CUDF_DATETIME_COMPONENT_DAY   = 2,
} cudf_datetime_component_t;

/// Datetime rounding frequencies wrapped by cudf.
typedef enum {
  CUDF_DATETIME_ROUNDING_DAY         = 0,
  CUDF_DATETIME_ROUNDING_HOUR        = 1,
  CUDF_DATETIME_ROUNDING_MINUTE      = 2,
  CUDF_DATETIME_ROUNDING_SECOND      = 3,
  CUDF_DATETIME_ROUNDING_MILLISECOND = 4,
  CUDF_DATETIME_ROUNDING_MICROSECOND = 5,
  CUDF_DATETIME_ROUNDING_NANOSECOND  = 6,
} cudf_datetime_rounding_frequency_t;

/// Extract a datetime component as an INT16 column with explicit execution
/// context.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_datetime_extract_component_ex(cudf_column_view_handle input,
                                                 cudf_datetime_component_t component,
                                                 cudf_column_handle* out,
                                                 const cudf_exec_options_t* options);

/// Floor datetimes to the requested frequency with explicit execution context.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_datetime_floor_ex(cudf_column_view_handle input,
                                     cudf_datetime_rounding_frequency_t frequency,
                                     cudf_column_handle* out,
                                     const cudf_exec_options_t* options);

/// Add calendrical months to a timestamp column with explicit execution
/// context.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_datetime_add_calendrical_months_scalar_ex(cudf_column_view_handle input,
                                                             int32_t months,
                                                             cudf_column_handle* out,
                                                             const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_DATETIME_H
