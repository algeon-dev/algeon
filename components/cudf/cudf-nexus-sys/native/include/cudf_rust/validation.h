#ifndef CUDF_RUST_VALIDATION_H
#define CUDF_RUST_VALIDATION_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Count null, NaN, and infinite values in a FLOAT32 column on the explicit
/// execution stream and return only the bounded host-visible count scalar.
cudf_status_t cudf_float32_non_finite_count_ex(cudf_column_view_handle input,
                                               int32_t* out_non_finite_count,
                                               cudf_exec_options_t const* options);

/// Count NaN and infinite values in one exact raw FLOAT32 device-buffer range
/// on the explicit execution stream.
cudf_status_t cudf_float32_buffer_non_finite_count_ex(float const* input,
                                                      size_t element_count,
                                                      int32_t* out_non_finite_count,
                                                      cudf_exec_options_t const* options);

/// Count values outside [min_inclusive, max_exclusive) in one exact raw INT64
/// device-buffer range on the explicit execution stream.
cudf_status_t cudf_int64_buffer_out_of_range_count_ex(int64_t const* input,
                                                      size_t element_count,
                                                      int64_t min_inclusive,
                                                      int64_t max_exclusive,
                                                      int32_t* out_of_range_count,
                                                      cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_VALIDATION_H
