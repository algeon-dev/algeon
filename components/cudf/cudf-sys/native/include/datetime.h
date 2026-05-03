#ifndef CUDF_RUST_DATETIME_H
#define CUDF_RUST_DATETIME_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Datetime components wrapped by cudf-rs.
typedef enum {
  CUDF_DATETIME_COMPONENT_YEAR  = 0,
  CUDF_DATETIME_COMPONENT_MONTH = 1,
  CUDF_DATETIME_COMPONENT_DAY   = 2,
} cudf_datetime_component_t;

/// Extract a datetime component as an INT16 column.
cudf_status_t cudf_datetime_extract_component(cudf_column_view_handle input,
                                              cudf_datetime_component_t component,
                                              cudf_column_handle* out);

/// Extract a datetime component as an INT16 column with explicit execution
/// context. When `options->mr` is non-NULL, it must outlive the returned
/// column.
cudf_status_t cudf_datetime_extract_component_ex(cudf_column_view_handle input,
                                                 cudf_datetime_component_t component,
                                                 cudf_column_handle* out,
                                                 const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_DATETIME_H
