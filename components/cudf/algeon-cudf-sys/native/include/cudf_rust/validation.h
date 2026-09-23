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

/// Add a finite scalar in place to every Float32 value on the execution stream.
/// IEEE infinity and NaN retain their non-finite status.
cudf_status_t cudf_float32_buffer_add_scalar_ex(float* values,
                                                size_t element_count,
                                                float scalar,
                                                cudf_exec_options_t const* options);

/// Replace Int64 values with fill_value where the paired Float32 value is infinite.
cudf_status_t cudf_int64_buffer_fill_where_float32_infinite_ex(int64_t* values,
                                                                float const* paired_values,
                                                                size_t element_count,
                                                                int64_t fill_value,
                                                                cudf_exec_options_t const* options);

/// Replace each Int64 equal to from with to and set its paired Float32 to positive infinity.
cudf_status_t cudf_int64_float32_buffer_replace_value_and_set_paired_infinite_ex(int64_t* values,
                                                                                 float* paired_values,
                                                                                 size_t element_count,
                                                                                 int64_t from,
                                                                                 int64_t to,
                                                                                 cudf_exec_options_t const* options);

/// Count non-finite Float32 values whose paired Int64 value differs from excluded_value.
cudf_status_t cudf_float32_buffer_non_finite_unless_int64_equals_count_ex(float const* values,
                                                                          int64_t const* paired_values,
                                                                          size_t element_count,
                                                                          int64_t excluded_value,
                                                                          int32_t* out_non_finite_count,
                                                                          cudf_exec_options_t const* options);

/// Count values outside [min_inclusive, max_exclusive) except the exact sentinel.
cudf_status_t cudf_int64_buffer_out_of_range_or_sentinel_count_ex(int64_t const* input,
                                                                   size_t element_count,
                                                                   int64_t min_inclusive,
                                                                   int64_t max_exclusive,
                                                                   int64_t sentinel,
                                                                   int32_t* out_of_range_count,
                                                                   cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_VALIDATION_H
