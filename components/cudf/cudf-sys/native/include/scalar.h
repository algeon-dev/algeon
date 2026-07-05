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
typedef void *cudf_scalar_handle;

/// Create a numeric scalar with explicit execution options.
///
/// `value` is reinterpreted according to `type_id`. `options` must name an
/// explicit valid device and non-NULL stream/MR.
/// This helper does not add an extra stream synchronization barrier before
/// returning. `options->mr` must outlive the returned scalar.
cudf_status_t cudf_scalar_create_numeric_ex(cudf_type_id_t type_id,
                                            const void *value, int32_t is_valid,
                                            const cudf_exec_options_t *options,
                                            cudf_scalar_handle *out);

/// Create a fixed-point decimal scalar from its raw integer representation
/// with explicit execution options.
///
/// `type_id` must be DECIMAL32, DECIMAL64, or DECIMAL128. `value` points to
/// an `int32_t`, `int64_t`, or 16-byte signed integer raw representation,
/// respectively. When `is_valid` is zero, `value` may be NULL; the returned
/// scalar still has the requested decimal type and scale.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// This helper does not add an extra stream synchronization barrier before
/// returning. `options->mr` must outlive the returned scalar.
cudf_status_t cudf_scalar_create_decimal_raw_ex(
    cudf_type_id_t type_id, const void *value, int32_t scale, int32_t is_valid,
    const cudf_exec_options_t *options, cudf_scalar_handle *out);

/// Create a timestamp scalar from a raw integer count with explicit execution
/// options.
///
/// `type_id` must be one of TIMESTAMP_DAYS, TIMESTAMP_SECONDS,
/// TIMESTAMP_MILLISECONDS, TIMESTAMP_MICROSECONDS, or TIMESTAMP_NANOSECONDS.
/// Days are stored by libcudf as int32 and reject values outside that range;
/// all other resolutions use int64. The raw value is interpreted directly as
/// units since the Unix epoch. No parsing or timezone conversion is performed.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// This helper does not add an extra stream synchronization barrier before
/// returning. `options->mr` must outlive the returned scalar.
cudf_status_t cudf_scalar_create_timestamp_raw_ex(
    cudf_type_id_t type_id, int64_t value, int32_t is_valid,
    const cudf_exec_options_t *options, cudf_scalar_handle *out);

/// Create a duration scalar from a raw integer count with explicit execution
/// options.
///
/// `type_id` must be one of DURATION_DAYS, DURATION_SECONDS,
/// DURATION_MILLISECONDS, DURATION_MICROSECONDS, or DURATION_NANOSECONDS.
/// Days are stored by libcudf as int32 and reject values outside that range;
/// all other resolutions use int64. `options` must name an explicit valid
/// device and non-NULL stream/MR. This helper does not add an extra stream
/// synchronization barrier before returning. `options->mr` must outlive the
/// returned scalar.
cudf_status_t cudf_scalar_create_duration_raw_ex(
    cudf_type_id_t type_id, int64_t value, int32_t is_valid,
    const cudf_exec_options_t *options, cudf_scalar_handle *out);

/// Create a string scalar with explicit execution options.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// The selected stream is synchronized before return on success so the
/// borrowed host string is no longer in use. `options->mr` must outlive the
/// returned scalar.
cudf_status_t cudf_scalar_create_string_ex(const char *value, int32_t is_valid,
                                           const cudf_exec_options_t *options,
                                           cudf_scalar_handle *out);

/// Deep-copy a scalar with explicit execution options.
///
/// Supports all scalar subtypes: numeric, boolean, string, timestamp,
/// duration, fixed-point decimal, list, and struct.  Returns
/// CUDF_ERROR_TYPE for DICTIONARY32 (no corresponding scalar class).
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// This is a device-only operation: no stream synchronization is added
/// before returning. `options->mr` must outlive the returned scalar.
cudf_status_t cudf_scalar_copy_ex(cudf_scalar_handle src,
                                  const cudf_exec_options_t *options,
                                  cudf_scalar_handle *out);

/// Destroy a scalar handle.
void cudf_scalar_destroy(cudf_scalar_handle handle);

/// Get the type_id of a scalar.
cudf_status_t cudf_scalar_type(cudf_scalar_handle handle, cudf_type_id_t *out);

/// Get the full logical type of a scalar, including decimal scale.
cudf_status_t cudf_scalar_data_type(cudf_scalar_handle handle,
                                    cudf_type_id_t *out_type_id,
                                    int32_t *out_scale);

/// Check if a scalar is valid (non-null) with explicit execution options.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// On success, `out` is ready to read when this call returns because libcudf
/// performs synchronous host materialization internally. `options->mr` is
/// ignored because this helper does not allocate device outputs, and this shim
/// does not add an extra synchronization barrier beyond that materialization.
cudf_status_t cudf_scalar_is_valid_ex(cudf_scalar_handle handle, int32_t *out,
                                      const cudf_exec_options_t *options);

/// Get the raw value of a fixed-width scalar with explicit execution options.
///
/// `out` must point to memory large enough for the scalar's physical
/// representation. `options` must name an explicit valid device and non-NULL
/// stream/MR.
/// On success, `out` is ready to read when this call returns because libcudf
/// performs synchronous host materialization internally. `options->mr` is
/// ignored because this helper does not allocate device outputs, and this shim
/// does not add an extra synchronization barrier beyond that materialization.
cudf_status_t cudf_scalar_value_ex(cudf_scalar_handle handle, void *out,
                                   const cudf_exec_options_t *options);

/// Get the string value of a string scalar with explicit execution options.
///
/// On success, `out` receives a heap-allocated, null-terminated UTF-8 string
/// and `len_out` receives its byte length (not including the null terminator).
/// The caller owns `out` and must free it with `cudf_string_free`.
/// `options` must name an explicit valid device and non-NULL stream/MR. The
/// owned host copy is ready to read when this call returns because libcudf
/// performs synchronous host materialization internally. `options->mr` is
/// ignored because this helper does not allocate device outputs, and this shim
/// does not add an extra synchronization barrier beyond that materialization.
cudf_status_t cudf_scalar_string_value_ex(cudf_scalar_handle handle, char **out,
                                          int32_t *len_out,
                                          const cudf_exec_options_t *options);

/// Free an owned host string returned by this shim.
void cudf_string_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif // CUDF_RUST_SCALAR_H
