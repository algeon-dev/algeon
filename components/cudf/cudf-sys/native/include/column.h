#ifndef CUDF_RUST_COLUMN_H
#define CUDF_RUST_COLUMN_H

#include "error.h"
#include "execution_context.h"
#include "scalar.h"
#include "types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::column (owning).
typedef void* cudf_column_handle;

/// Opaque handle to a cudf::column_view (non-owning).
typedef void* cudf_column_view_handle;

/// Create an empty column of the given type.
cudf_status_t cudf_column_create_empty(cudf_type_id_t type_id,
                                       cudf_column_handle* out);

/// Create a numeric column from host data.
/// `data` points to host memory of `size` elements of type `type_id`.
/// `null_mask` is optional (pass NULL for no nulls); if provided it must
/// contain ceil(size/8) bytes of validity bitmask (LSB order).
cudf_status_t cudf_column_create_numeric(cudf_type_id_t type_id,
                                         const void* data, int32_t size,
                                         const uint8_t* null_mask,
                                         int32_t null_count,
                                         cudf_column_handle* out);

/// Create a numeric column from host data with explicit execution options.
cudf_status_t cudf_column_create_numeric_ex(cudf_type_id_t type_id,
                                            const void* data, int32_t size,
                                            const uint8_t* null_mask,
                                            int32_t null_count,
                                            const cudf_exec_options_t* options,
                                            cudf_column_handle* out);

/// Deep-copy a column.
cudf_status_t cudf_column_copy(cudf_column_handle src,
                               cudf_column_handle* out);

/// Deep-copy a column with explicit execution options.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_column_copy_ex(cudf_column_handle src,
                                  const cudf_exec_options_t* options,
                                  cudf_column_handle* out);

/// Deep-copy a column from a column_view. Works for all types including
/// strings and nested types.
cudf_status_t cudf_column_copy_from_view(cudf_column_view_handle src,
                                         cudf_column_handle* out);

/// Deep-copy a column from a column_view with explicit execution options.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_column_copy_from_view_ex(cudf_column_view_handle src,
                                            const cudf_exec_options_t* options,
                                            cudf_column_handle* out);

/// Destroy an owning column handle.
void cudf_column_destroy(cudf_column_handle handle);

/// Get an immutable view of a column. The view is heap-allocated and must
/// be freed with cudf_column_view_destroy. The view borrows from `handle`
/// and must not outlive it.
cudf_status_t cudf_column_view(cudf_column_handle handle,
                               cudf_column_view_handle* out);

/// Destroy a column_view handle.
void cudf_column_view_destroy(cudf_column_view_handle handle);

/// Get the type_id of a column_view.
cudf_status_t cudf_column_view_type(cudf_column_view_handle handle,
                                    cudf_type_id_t* out);

/// Get the full logical type of a column_view, including decimal scale.
cudf_status_t cudf_column_view_data_type(cudf_column_view_handle handle,
                                         cudf_type_id_t* out_type_id,
                                         int32_t* out_scale);

/// Get the number of elements in a column_view.
cudf_status_t cudf_column_view_size(cudf_column_view_handle handle,
                                    int32_t* out);

/// Get the null count of a column_view.
cudf_status_t cudf_column_view_null_count(cudf_column_view_handle handle,
                                          int32_t* out);

/// Check if the column_view has a null mask allocated.
cudf_status_t cudf_column_view_nullable(cudf_column_view_handle handle,
                                        int32_t* out);

/// Get the number of child columns in a column_view.
cudf_status_t cudf_column_view_num_children(cudf_column_view_handle handle,
                                            int32_t* out);

/// Get a child column_view by index. The returned view is heap-allocated
/// and must be freed with cudf_column_view_destroy.
cudf_status_t cudf_column_view_child(cudf_column_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle* out);

/// Copy column data to host buffer. `host_buf` must be large enough.
/// Only valid for fixed-width types.
cudf_status_t cudf_column_view_to_host(cudf_column_view_handle handle,
                                       void* host_buf, size_t buf_size);

/// Copy column data to host buffer with explicit execution options.
///
/// The provided stream is synchronized before return so `host_buf` is fully
/// populated on success. `options->mr` is ignored because this helper does not
/// allocate output device storage.
cudf_status_t cudf_column_view_to_host_ex(cudf_column_view_handle handle,
                                          void* host_buf,
                                          size_t buf_size,
                                          const cudf_exec_options_t* options);

/// Get the raw device (GPU) data pointer of a column_view.
///
/// Returns the pointer to the device memory that backs this column's data.
/// The pointer remains valid as long as the underlying column/table is alive.
///
/// For fixed-width types the pointer addresses `size` contiguous elements.
/// For STRING columns the data pointer is NULL — use child column views
/// (child 0 = offsets, child 1 = chars) to access device data instead.
///
/// The offset of the column view is NOT applied — the pointer is to the
/// base of the device buffer. Callers working with sliced views must add
/// `offset * element_size` manually.
cudf_status_t cudf_column_view_data_ptr(cudf_column_view_handle handle,
                                        const void** out);

/// Get the element offset of a column_view into its parent buffer.
///
/// Returns 0 for non-sliced columns. For sliced views, the offset indicates
/// how many elements from the start of the device buffer the view begins.
cudf_status_t cudf_column_view_offset(cudf_column_view_handle handle,
                                      int32_t* out);

/// Copy null mask to host buffer. `host_buf` must be at least
/// bitmask_allocation_size_bytes(size) bytes.
///
/// Returns CUDF_ERROR_NULL if column has no null mask.
/// Returns CUDF_ERROR_INVALID_OPERATION for sliced views (offset != 0).
/// This helper synchronizes cuDF's default stream before return so `host_buf`
/// is fully populated on success.
cudf_status_t cudf_column_view_null_mask_to_host(cudf_column_view_handle handle,
                                                 uint8_t* host_buf,
                                                 size_t buf_size);

/// Create a column by copying data from an existing device (GPU) pointer.
///
/// `device_data` must point to GPU memory with `size` elements of `type_id`.
/// Performs a device-to-device copy; the new column owns its own memory.
///
/// For `size == 0`, `device_data` may be NULL.
/// For `size > 0`, `device_data` must be non-NULL.
///
/// This legacy helper synchronizes cuDF's default stream before return so the
/// borrowed `device_data` pointer is no longer in use when the call succeeds.
cudf_status_t cudf_column_create_from_device_data(
    cudf_type_id_t type_id,
    const void* device_data,
    int32_t size,
    cudf_column_handle* out);

/// Create a column by copying data from an existing device (GPU) pointer with
/// explicit execution options.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned column.
///
/// Unlike `cudf_column_create_from_device_data`, this `_ex` variant does not
/// synchronize before returning. The caller must keep `device_data` valid until
/// work on the selected stream has completed, and must establish any required
/// cross-stream ordering for later consumers of the returned column.
cudf_status_t cudf_column_create_from_device_data_ex(
    cudf_type_id_t type_id,
    const void* device_data,
    int32_t size,
    const cudf_exec_options_t* options,
    cudf_column_handle* out);

/// Create a STRING type column from host string data.
///
/// `strings` is an array of pointers to UTF-8 string data (not necessarily
/// null-terminated). `lengths` holds the byte length of each string.
/// `num_strings` is the number of strings.
///
/// If `null_mask` is non-NULL and `null_count` > 0, the null_mask is a
/// packed bitmask (LSB order) indicating which rows are valid.
///
/// For num_strings == 0, an empty STRING column is returned.
cudf_status_t cudf_column_create_strings(
    const char** strings,
    const int32_t* lengths,
    int32_t num_strings,
    const uint8_t* null_mask,
    int32_t null_count,
    cudf_column_handle* out);

/// Create a STRING type column from host string data with explicit execution options.
cudf_status_t cudf_column_create_strings_ex(
    const char** strings,
    const int32_t* lengths,
    int32_t num_strings,
    const uint8_t* null_mask,
    int32_t null_count,
    const cudf_exec_options_t* options,
    cudf_column_handle* out);

/// Create a column by repeating a scalar value `size` times.
///
/// Uses `cudf::make_column_from_scalar` under the hood.
/// For `size == 0`, an empty column of the scalar's type is returned.
cudf_status_t cudf_column_from_scalar(cudf_scalar_handle scalar,
                                      int32_t size,
                                      cudf_column_handle* out);

/// Copy string column data to host buffers.
///
/// For a STRING column with `n` rows, this copies:
///   - `offsets_buf`: n+1 int32 values (byte offsets into chars).
///   - `chars_buf`: the concatenated character bytes.
///
/// `offsets_buf_size` must be >= (n+1)*sizeof(int32_t).
/// `chars_buf_size` must be >= total chars bytes (= offsets[n] - offsets[0]).
///
/// Returns CUDF_ERROR_TYPE if the column is not STRING type.
/// This helper synchronizes cuDF's default stream before return whenever it
/// must materialize offsets or chars into the caller's host buffers.
cudf_status_t cudf_column_view_strings_to_host(
    cudf_column_view_handle handle,
    int32_t* offsets_buf,
    size_t offsets_buf_size,
    char* chars_buf,
    size_t chars_buf_size,
    int32_t* total_chars_bytes);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_COLUMN_H
