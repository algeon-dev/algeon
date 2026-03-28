#ifndef CUDF_RUST_SCALAR_H
#define CUDF_RUST_SCALAR_H

#include "error.h"
#include "types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::scalar (owning).
typedef void* cudf_scalar_handle;

/// Create a numeric scalar. `value` is reinterpreted according to `type_id`.
///
/// This helper enqueues any required device work on cuDF's default stream and
/// returns without an extra stream synchronization barrier.
cudf_status_t cudf_scalar_create_numeric(cudf_type_id_t type_id,
                                         const void* value, int32_t is_valid,
                                         cudf_scalar_handle* out);

/// Create a string scalar from a null-terminated UTF-8 string.
///
/// This helper enqueues any required device work on cuDF's default stream and
/// returns without an extra stream synchronization barrier.
cudf_status_t cudf_scalar_create_string(const char* value, int32_t is_valid,
                                        cudf_scalar_handle* out);

/// Destroy a scalar handle.
void cudf_scalar_destroy(cudf_scalar_handle handle);

/// Get the type_id of a scalar.
cudf_status_t cudf_scalar_type(cudf_scalar_handle handle, cudf_type_id_t* out);

/// Get the full logical type of a scalar, including decimal scale.
cudf_status_t cudf_scalar_data_type(cudf_scalar_handle handle,
                                    cudf_type_id_t* out_type_id,
                                    int32_t* out_scale);

/// Check if a scalar is valid (non-null).
cudf_status_t cudf_scalar_is_valid(cudf_scalar_handle handle, int32_t* out);

/// Get the value of a numeric scalar. `out` must point to memory large enough
/// for the scalar's type.
cudf_status_t cudf_scalar_value(cudf_scalar_handle handle, void* out);

/// Get the string value of a string scalar as an owned host copy.
///
/// On success, `out` receives a heap-allocated, null-terminated UTF-8 string
/// and `len_out` receives its byte length (not including the null terminator).
/// The caller owns `out` and must free it with `cudf_string_free`.
cudf_status_t cudf_scalar_string_value(cudf_scalar_handle handle,
                                       char** out, int32_t* len_out);

/// Free an owned host string returned by this shim.
void cudf_string_free(char* ptr);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SCALAR_H
