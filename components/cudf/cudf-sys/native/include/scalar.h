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
cudf_status_t cudf_scalar_create_numeric(cudf_type_id_t type_id,
                                         const void* value, int32_t is_valid,
                                         cudf_scalar_handle* out);

/// Create a string scalar from a null-terminated UTF-8 string.
cudf_status_t cudf_scalar_create_string(const char* value, int32_t is_valid,
                                        cudf_scalar_handle* out);

/// Destroy a scalar handle.
void cudf_scalar_destroy(cudf_scalar_handle handle);

/// Get the type_id of a scalar.
cudf_status_t cudf_scalar_type(cudf_scalar_handle handle, cudf_type_id_t* out);

/// Check if a scalar is valid (non-null).
cudf_status_t cudf_scalar_is_valid(cudf_scalar_handle handle, int32_t* out);

/// Get the value of a numeric scalar. `out` must point to memory large enough
/// for the scalar's type.
cudf_status_t cudf_scalar_value(cudf_scalar_handle handle, void* out);

/// Get the string value of a string scalar. Returns a pointer to thread-local
/// host storage; the caller must NOT free it.
///
/// The pointer remains valid until the next shim call on the same thread.
/// `len_out` receives the byte length (not including null terminator).
cudf_status_t cudf_scalar_string_value(cudf_scalar_handle handle,
                                       const char** out, int32_t* len_out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SCALAR_H
