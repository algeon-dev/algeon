#ifndef CUDF_RUST_DEVICE_BUFFER_H
#define CUDF_RUST_DEVICE_BUFFER_H

#include "column.h"
#include "error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an rmm::device_buffer (owning GPU memory).
typedef void* cudf_device_buffer_handle;

/// Allocate a new device buffer of `size` bytes using the current
/// per-device RMM memory resource.
///
/// On success, `*out` is a valid buffer handle. The buffer contents
/// are uninitialized.
///
/// Returns CUDF_ERROR_OUT_OF_MEMORY if allocation fails.
cudf_status_t cudf_device_buffer_create(size_t size, cudf_device_buffer_handle* out);

/// Release a device buffer. Safe to call with NULL (no-op).
///
/// After this call, the handle is invalid. If the buffer was previously
/// adopted by a column (via cudf_column_adopt_device_buffer), the
/// handle is already invalid and must NOT be destroyed again.
void cudf_device_buffer_destroy(cudf_device_buffer_handle handle);

/// Get the device pointer from a buffer handle.
///
/// The returned pointer is valid until the buffer is destroyed or
/// adopted into a column. It refers to GPU memory and must not be
/// dereferenced from host code.
///
/// For zero-size buffers this call still succeeds, but the returned pointer
/// may be NULL because there is no addressable device range.
cudf_status_t cudf_device_buffer_data(cudf_device_buffer_handle handle, void** out);

/// Get the size in bytes of a buffer.
cudf_status_t cudf_device_buffer_size(cudf_device_buffer_handle handle, size_t* out);

/// Copy host data into the device buffer.
///
/// `host_data` must point to at least `size` bytes. `size` must not
/// exceed the buffer's size. The copy is issued on the default stream
/// and synchronized before return.
///
/// Returns CUDF_ERROR_INVALID_OPERATION if `size > buffer_size`.
cudf_status_t cudf_device_buffer_copy_from_host(cudf_device_buffer_handle handle,
                                                const void* host_data,
                                                size_t size);

/// Create a column by adopting (moving) an existing device buffer.
/// Zero-copy: no device-to-device memcpy occurs.
///
/// The buffer handle is consumed: after a successful call, the handle
/// is no longer valid and must NOT be passed to
/// cudf_device_buffer_destroy.
///
/// On failure, the buffer handle remains valid (ownership is not
/// transferred).
///
/// `type_id` must be a fixed-width type.
/// `row_count` is the number of elements (not bytes).
/// Buffer byte size must equal `row_count * element_size_of(type_id)`.
///
/// The resulting column has no null mask (null_count = 0).
cudf_status_t cudf_column_adopt_device_buffer(cudf_type_id_t type_id,
                                              cudf_device_buffer_handle buffer,
                                              int32_t row_count,
                                              cudf_column_handle* out);

/// Create a nullable column by adopting data and null-mask buffers.
/// Zero-copy: no device-to-device memcpy occurs.
///
/// Both `data_buffer` and `null_mask_buffer` are consumed on success.
/// On failure, both remain valid.
///
/// `null_mask_buffer` must be at least
/// bitmask_allocation_size_bytes(row_count) bytes.
///
/// `null_count` must be in [0, row_count].
cudf_status_t cudf_column_adopt_device_buffer_nullable(cudf_type_id_t type_id,
                                                       cudf_device_buffer_handle data_buffer,
                                                       cudf_device_buffer_handle null_mask_buffer,
                                                       int32_t null_count,
                                                       int32_t row_count,
                                                       cudf_column_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_DEVICE_BUFFER_H
