#ifndef CUDF_RUST_COLUMN_H
#define CUDF_RUST_COLUMN_H

#include "error.h"
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

/// Deep-copy a column.
cudf_status_t cudf_column_copy(cudf_column_handle src,
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

/// Copy null mask to host buffer. `host_buf` must be at least ceil(size/8) bytes.
/// Returns CUDF_ERROR_NULL if column has no null mask.
cudf_status_t cudf_column_view_null_mask_to_host(cudf_column_view_handle handle,
                                                 uint8_t* host_buf,
                                                 size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_COLUMN_H
