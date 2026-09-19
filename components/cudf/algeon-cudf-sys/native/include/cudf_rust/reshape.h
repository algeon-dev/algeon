#ifndef CUDF_RUST_RESHAPE_H
#define CUDF_RUST_RESHAPE_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Interleave equal-length columns into one row-major column with explicit
/// execution context.
///
/// `options` must be non-NULL and carry an explicit device, stream, and memory
/// resource. `options->mr` must outlive the returned column.
cudf_status_t cudf_interleave_columns_ex(cudf_column_view_handle const* columns,
                                         int32_t num_columns,
                                         cudf_exec_options_t const* options,
                                         cudf_column_handle* out);

/// Copy a flat, non-null Float32 column into a contiguous dense matrix order.
///
/// The input holds exactly `rows * columns` row-major values. `column_major`
/// selects whether the returned flat column keeps that row-major order (`0`)
/// or rearranges it into contiguous column-major order (`1`). All work uses
/// the explicit stream and memory resource carried by `options`.
cudf_status_t cudf_pack_flat_float32_matrix_ex(cudf_column_view_handle input,
                                               int32_t rows,
                                               int32_t columns,
                                               int32_t column_major,
                                               cudf_exec_options_t const* options,
                                               cudf_column_handle* out);

/// Copy a borrowed device-resident Float32 span into a dense matrix order.
///
/// `input` points at exactly `rows * columns` row-major Float32 values on the
/// CUDA device selected by `options`. The caller retains the source allocation
/// and must keep it alive until work enqueued on `options->stream` completes.
/// The returned column performs exactly one output allocation and one payload
/// copy/reorder on the explicit stream and memory resource.
cudf_status_t cudf_pack_borrowed_float32_matrix_ex(void const* input,
                                                   int32_t rows,
                                                   int32_t columns,
                                                   int32_t column_major,
                                                   cudf_exec_options_t const* options,
                                                   cudf_column_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_RESHAPE_H
