#ifndef CUDF_RUST_SCALAR_H
#define CUDF_RUST_SCALAR_H

#include "error.h"
#include "execution_context.h"
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

/// Create a numeric scalar with explicit execution options.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// This helper does not add an extra stream synchronization barrier before
/// returning.
/// When `options->mr` is non-NULL, it must outlive the returned scalar.
cudf_status_t cudf_scalar_create_numeric_ex(cudf_type_id_t type_id,
                                            const void* value,
                                            int32_t is_valid,
                                            const cudf_exec_options_t* options,
                                            cudf_scalar_handle* out);

/// Create a string scalar from a null-terminated UTF-8 string.
///
/// This helper synchronizes cuDF's default stream before return on success so
/// the borrowed host string is no longer in use.
cudf_status_t cudf_scalar_create_string(const char* value, int32_t is_valid,
                                        cudf_scalar_handle* out);

/// Create a string scalar with explicit execution options.
///
/// The selected stream is synchronized before return on success so the
/// borrowed host string is no longer in use.
/// When `options->mr` is non-NULL, it must outlive the returned scalar.
cudf_status_t cudf_scalar_create_string_ex(const char* value,
                                           int32_t is_valid,
                                           const cudf_exec_options_t* options,
                                           cudf_scalar_handle* out);

/// Deep-copy a scalar on cuDF's default stream.
///
/// Supports all scalar subtypes: numeric, boolean, string, timestamp,
/// duration, fixed-point decimal, list, and struct.  Returns
/// CUDF_ERROR_TYPE for DICTIONARY32 (no corresponding scalar class).
///
/// This is a device-only operation: no stream synchronization is added
/// before returning.
cudf_status_t cudf_scalar_copy(cudf_scalar_handle src,
                                cudf_scalar_handle* out);

/// Deep-copy a scalar with explicit execution options.
///
/// When `options` is NULL, or `options->stream` / `options->mr` are NULL,
/// cuDF's default stream and current device memory resource are used.
/// This is a device-only operation: no stream synchronization is added
/// before returning.
/// When `options->mr` is non-NULL, it must outlive the returned scalar.
cudf_status_t cudf_scalar_copy_ex(cudf_scalar_handle src,
                                   const cudf_exec_options_t* options,
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

/// Check if a scalar is valid (non-null) with explicit execution options.
///
/// On success, `out` is ready to read when this call returns because libcudf
/// performs synchronous host materialization internally. `options->mr` is
/// ignored because this helper does not allocate device outputs, and this shim
/// does not add an extra synchronization barrier beyond that materialization.
cudf_status_t cudf_scalar_is_valid_ex(cudf_scalar_handle handle,
                                      int32_t* out,
                                      const cudf_exec_options_t* options);

/// Get the value of a numeric scalar. `out` must point to memory large enough
/// for the scalar's type.
cudf_status_t cudf_scalar_value(cudf_scalar_handle handle, void* out);

/// Get the value of a numeric scalar with explicit execution options.
///
/// On success, `out` is ready to read when this call returns because libcudf
/// performs synchronous host materialization internally. `options->mr` is
/// ignored because this helper does not allocate device outputs, and this shim
/// does not add an extra synchronization barrier beyond that materialization.
cudf_status_t cudf_scalar_value_ex(cudf_scalar_handle handle,
                                   void* out,
                                   const cudf_exec_options_t* options);

/// Get the string value of a string scalar as an owned host copy.
///
/// On success, `out` receives a heap-allocated, null-terminated UTF-8 string
/// and `len_out` receives its byte length (not including the null terminator).
/// The caller owns `out` and must free it with `cudf_string_free`.
cudf_status_t cudf_scalar_string_value(cudf_scalar_handle handle,
                                       char** out, int32_t* len_out);

/// Get the string value of a string scalar with explicit execution options.
///
/// On success, `out` receives a heap-allocated, null-terminated UTF-8 string
/// and `len_out` receives its byte length (not including the null terminator).
/// The caller owns `out` and must free it with `cudf_string_free`. The owned
/// host copy is ready to read when this call returns because libcudf performs
/// synchronous host materialization internally. `options->mr` is ignored
/// because this helper does not allocate device outputs, and this shim does
/// not add an extra synchronization barrier beyond that materialization.
cudf_status_t cudf_scalar_string_value_ex(
    cudf_scalar_handle handle,
    char** out,
    int32_t* len_out,
    const cudf_exec_options_t* options);

/// Free an owned host string returned by this shim.
void cudf_string_free(char* ptr);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SCALAR_H
