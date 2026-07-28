#ifndef CUDF_RUST_DEVICE_BUFFER_H
#define CUDF_RUST_DEVICE_BUFFER_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an rmm::device_buffer (owning GPU memory).
typedef struct cudf_device_buffer_handle_state* cudf_device_buffer_handle;

/// Allocate a new device buffer of `size` bytes using the explicit execution
/// options.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned buffer and any column that later
/// adopts it.
///
/// On success, `*out` is a valid buffer handle. The buffer contents are
/// uninitialized.
///
/// Returns CUDF_ERROR_OUT_OF_MEMORY if allocation fails.
cudf_status_t cudf_device_buffer_create_ex(size_t size,
                                           const cudf_exec_options_t* options,
                                           cudf_device_buffer_handle* out);

/// Release a device buffer. Safe to call with NULL (no-op).
///
/// Ownership is consumed on call, including on failure. CUDF_SUCCESS means
/// cleanup completed; any error means the allocation was quarantined and the
/// handle must not be reused or destroyed again. If the buffer was previously
/// adopted by a column (via cudf_column_adopt_device_buffer), the
/// handle is already invalid and must NOT be destroyed again.
cudf_status_t cudf_device_buffer_destroy(cudf_device_buffer_handle handle);

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

/// Copy host data into the device buffer with explicit execution options.
///
/// `host_data` must point to at least `size` bytes. `size` must not
/// exceed the buffer's size.
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. The explicit device must
/// match the buffer's retained provenance. The copy is issued on the selected
/// stream and synchronized before return so the borrowed host data is no
/// longer in use.
///
/// Returns CUDF_ERROR_INVALID_OPERATION if `size > buffer_size`.
cudf_status_t cudf_device_buffer_copy_from_host_ex(cudf_device_buffer_handle handle,
                                                   const void* host_data,
                                                   size_t size,
                                                   const cudf_exec_options_t* options);

/// Copy host data into a range of the device buffer without synchronizing.
///
/// Issues a `cudaMemcpyAsync` of `size` bytes from `host_data` into the
/// buffer at byte offset `dst_offset` on the selected stream and returns
/// immediately. `dst_offset + size` must not exceed the buffer's size.
///
/// Unlike `cudf_device_buffer_copy_from_host_ex`, the stream is NOT
/// synchronized before return. The caller must guarantee that `host_data`
/// stays valid and unmodified until the copy completes in stream order
/// (e.g. by recording a CUDA event on the same stream after this call and
/// waiting on it before reusing the host region). For true asynchronous
/// overlap `host_data` should point into page-locked (pinned) host memory
/// such as an allocation from `cudf_pinned_host_alloc`; pageable sources
/// fall back to the driver's staged copy path.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. The explicit device must
/// match the buffer's retained provenance.
///
/// Returns CUDF_ERROR_INVALID_OPERATION if the range is out of bounds.
cudf_status_t cudf_device_buffer_copy_from_host_range_async_ex(cudf_device_buffer_handle handle,
                                                               size_t dst_offset,
                                                               const void* host_data,
                                                               size_t size,
                                                               const cudf_exec_options_t* options);

/// Allocate `size` bytes of page-locked (pinned) host memory.
///
/// The allocation is portable across CUDA devices (`cudaHostAllocPortable`).
/// Pinned host memory is a scarce global resource: callers should retain and
/// reuse allocations (e.g. as staging slabs) instead of allocating per
/// operation.
///
/// On success `*out` is a valid host pointer that must be released with
/// `cudf_pinned_host_free`. `size` must be greater than zero.
cudf_status_t cudf_pinned_host_alloc(size_t size, void** out);

/// Release pinned host memory allocated by `cudf_pinned_host_alloc`.
/// Safe to call with NULL (no-op).
cudf_status_t cudf_pinned_host_free(void* ptr);

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
