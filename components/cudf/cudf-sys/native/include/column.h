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
typedef void *cudf_column_handle;

/// Opaque handle to a cudf::column_view (non-owning).
typedef void *cudf_column_view_handle;

typedef enum cudf_interop_placement_kind_t {
  CUDF_INTEROP_PLACEMENT_HOST = 0,
  CUDF_INTEROP_PLACEMENT_CUDA_DEVICE = 1,
} cudf_interop_placement_kind_t;

typedef enum cudf_interop_validity_kind_t {
  CUDF_INTEROP_VALIDITY_NONE = 0,
  CUDF_INTEROP_VALIDITY_MASK = 1,
} cudf_interop_validity_kind_t;

typedef enum cudf_interop_layout_kind_t {
  CUDF_INTEROP_LAYOUT_FIXED_WIDTH = 0,
  CUDF_INTEROP_LAYOUT_BINARY_LIKE = 1,
  CUDF_INTEROP_LAYOUT_LIST = 2,
  CUDF_INTEROP_LAYOUT_STRUCT = 3,
  CUDF_INTEROP_LAYOUT_DICTIONARY = 4,
} cudf_interop_layout_kind_t;

typedef struct cudf_interop_buffer_binding_t {
  uint32_t buffer_id;
  void const *data;
  size_t byte_len;
  cudf_interop_placement_kind_t placement_kind;
  int32_t device;
} cudf_interop_buffer_binding_t;

typedef struct cudf_interop_semantic_column_t {
  uint32_t column_id;
  cudf_type_id_t type_id;
  int32_t scale;
  int32_t len;
  int32_t offset;
  cudf_interop_validity_kind_t validity_kind;
  uint32_t validity_buffer_id;
  int32_t null_count;
  cudf_interop_layout_kind_t layout_kind;
  uint32_t values_buffer_id;
  uint32_t offsets_buffer_id;
  uint32_t data_buffer_id;
  uint32_t indices_buffer_id;
  uint32_t child_column_id;
  uint32_t const *child_column_ids;
  int32_t num_child_column_ids;
  uint32_t keys_column_id;
} cudf_interop_semantic_column_t;

/// Create an empty column of the given type with explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
cudf_status_t cudf_column_create_empty_ex(cudf_type_id_t type_id,
                                          const cudf_exec_options_t *options,
                                          cudf_column_handle *out);

/// Create a numeric column from host data with explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
///
/// If host-source copies are queued, the selected stream is synchronized
/// before return on success or failure so the borrowed host buffers are no
/// longer in use.
cudf_status_t cudf_column_create_numeric_ex(cudf_type_id_t type_id,
                                            const void *data, int32_t size,
                                            const uint8_t *null_mask,
                                            int32_t null_count,
                                            const cudf_exec_options_t *options,
                                            cudf_column_handle *out);

/// Create a fixed-width column from host data with explicit logical dtype and
/// execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
///
/// If host-source copies are queued, the selected stream is synchronized
/// before return on success or failure so the borrowed host buffers are no
/// longer in use.
cudf_status_t cudf_column_create_fixed_width_from_host_ex(
    cudf_type_id_t type_id, int32_t scale, const void *data, int32_t size,
    const uint8_t *null_mask, int32_t null_count,
    const cudf_exec_options_t *options, cudf_column_handle *out);

/// Create a fixed-width sequence column with explicit execution options.
///
/// The output is `[init, init + step, init + 2*step, ...]` with `size` rows.
/// `init` and `step` must be valid fixed-width scalars of the same type.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
cudf_status_t cudf_column_create_sequence_ex(cudf_scalar_handle init,
                                             cudf_scalar_handle step,
                                             int32_t size,
                                             const cudf_exec_options_t *options,
                                             cudf_column_handle *out);

/// Deep-copy a column with explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
cudf_status_t cudf_column_copy_ex(cudf_column_handle src,
                                  const cudf_exec_options_t *options,
                                  cudf_column_handle *out);

/// Deep-copy a column from a column_view with explicit execution options.
///
/// Works for all types including strings and nested types. `options` must be
/// non-NULL and must name an explicit valid device, a non-NULL stream, and a
/// non-NULL memory resource. `options->mr` must outlive the returned column.
cudf_status_t cudf_column_copy_from_view_ex(cudf_column_view_handle src,
                                            const cudf_exec_options_t *options,
                                            cudf_column_handle *out);

/// Deep-copy a fixed-width column from raw borrowed interop-view facts with
/// explicit execution options.
///
/// This reconstructs a temporary `cudf::column_view` from the borrowed device
/// pointer, optional null mask, and logical offset, then deep-copies the
/// active logical range into a new owning column.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
cudf_status_t cudf_column_copy_from_borrowed_view_ex(
    cudf_type_id_t type_id, int32_t scale, int32_t size, int32_t offset,
    const void *data, const void *null_mask, int32_t null_count,
    const cudf_exec_options_t *options, cudf_column_handle *out);

/// Deep-copy a column from a prepared-device export semantic graph with
/// explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
cudf_status_t cudf_column_copy_from_prepared_device_export_ex(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_exec_options_t const *options,
    cudf_column_handle *out);

/// Deep-copy a column from a prepared-host export semantic graph with explicit
/// execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
cudf_status_t cudf_column_copy_from_prepared_host_export_ex(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_exec_options_t const *options,
    cudf_column_handle *out);

/// Create a zero-copy owning column handle from an owned device transfer
/// semantic graph.
///
/// The resulting handle retains only shim-owned semantic metadata. Higher
/// layers remain responsible for retaining the transfer root owner for the
/// lifetime of the returned handle.
cudf_status_t cudf_column_create_from_owned_device_transfer(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_column_handle *out);

/// Return payload bytes for this owning column with explicit execution options.
///
/// Compound parent columns contribute through their children; STRING also
/// accounts for its chars payload using the resolved execution stream.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. Payload accounting excludes
/// allocator padding.
cudf_status_t cudf_column_payload_size_ex(cudf_column_handle handle,
                                          const cudf_exec_options_t *options,
                                          size_t *out);

/// Return the total device allocation footprint of this owning column.
///
/// This forwards to libcudf `column::alloc_size()` and includes padding.
cudf_status_t cudf_column_device_allocated_size(cudf_column_handle handle,
                                                size_t *out);

/// Destroy an owning column handle.
void cudf_column_destroy(cudf_column_handle handle);

/// Get an immutable view of a column. The view is heap-allocated and must
/// be freed with cudf_column_view_destroy. The view borrows from `handle`
/// and must not outlive it.
cudf_status_t cudf_column_view(cudf_column_handle handle,
                               cudf_column_view_handle *out);

/// Destroy a column_view handle.
void cudf_column_view_destroy(cudf_column_view_handle handle);

/// Get the type_id of a column_view.
cudf_status_t cudf_column_view_type(cudf_column_view_handle handle,
                                    cudf_type_id_t *out);

/// Get the full logical type of a column_view, including decimal scale.
cudf_status_t cudf_column_view_data_type(cudf_column_view_handle handle,
                                         cudf_type_id_t *out_type_id,
                                         int32_t *out_scale);

/// Get the number of elements in a column_view.
cudf_status_t cudf_column_view_size(cudf_column_view_handle handle,
                                    int32_t *out);

/// Get the stored null count metadata of a column_view.
cudf_status_t cudf_column_view_null_count(cudf_column_view_handle handle,
                                          int32_t *out);

/// Resolve the null count of a column_view with explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. If the view carries an
/// unknown null count, the selected stream is used to compute it.
cudf_status_t cudf_column_view_null_count_ex(
    cudf_column_view_handle handle, const cudf_exec_options_t *options,
    int32_t *out);

/// Force the next column-view metadata read (`size` or `null_count`) to fail.
void cudf_test_fail_next_column_view_metadata_read();

/// Force a future owning-column handle publication to fail after
/// `successful_releases` successful publications on this thread. Intended only
/// for ownership/cleanup regression tests. Pass a negative value to disable.
void cudf_test_fail_column_handle_release_after(int32_t successful_releases);

/// Force the next fixed-width host import to fail after the data copy is
/// queued. Intended only for synchronization regression tests.
void cudf_test_fail_next_fixed_width_host_import_after_data_copy();

/// Force the next fixed-width host import to fail after the null-mask copy is
/// queued. Intended only for synchronization regression tests.
void cudf_test_fail_next_fixed_width_host_import_after_null_mask_copy();

/// Force the next strings host import to fail after the offsets copy is
/// queued. Intended only for synchronization regression tests.
void cudf_test_fail_next_strings_host_import_after_offsets_copy();

/// Force the next strings host import to fail after the chars copy is queued.
/// Intended only for synchronization regression tests.
void cudf_test_fail_next_strings_host_import_after_chars_copy();

/// Force the next strings host import to fail after the null-mask copy is
/// queued. Intended only for synchronization regression tests.
void cudf_test_fail_next_strings_host_import_after_null_mask_copy();

/// Force the next strings host import to fail after host-source copies are
/// queued and before the libcudf strings column is assembled.
void cudf_test_fail_next_strings_host_import_before_make_strings_column();

/// Force the next strings host import to fail after the libcudf strings column
/// is assembled but before the shim returns success.
void cudf_test_fail_next_strings_host_import_after_make_strings_column();

/// Test-only helper that constructs a fixed-width column_view with a caller-
/// controlled logical offset. The returned view is synthetic and must only be
/// used for metadata/slicing regression tests.
cudf_status_t cudf_test_column_view_create_synthetic_fixed_width(
    cudf_type_id_t type_id, int32_t size, int32_t offset,
    cudf_column_view_handle *out);

/// Test-only helper that constructs a non-empty STRING column_view without an
/// offsets child. The returned view is intentionally invalid and must only be
/// used for layout-validation regression tests.
cudf_status_t cudf_test_column_view_create_synthetic_string_without_children(
    int32_t size, cudf_column_view_handle *out);

/// Check if the column_view has a null mask allocated.
cudf_status_t cudf_column_view_nullable(cudf_column_view_handle handle,
                                        int32_t *out);

/// Get the number of child columns in a column_view.
cudf_status_t cudf_column_view_num_children(cudf_column_view_handle handle,
                                            int32_t *out);

/// Get a child column_view by index. The returned view is heap-allocated
/// and must be freed with cudf_column_view_destroy.
cudf_status_t cudf_column_view_child(cudf_column_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle *out);

/// Get the logical child column view of a LIST column with the root offset
/// applied to the child range, using explicit execution options.
///
/// Returns `CUDF_ERROR_TYPE` if the input column is not LIST type.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. The memory resource is
/// retained only for provenance; this helper does not allocate output storage.
cudf_status_t
cudf_column_view_list_sliced_child_ex(cudf_column_view_handle handle,
                                      const cudf_exec_options_t *options,
                                      cudf_column_view_handle *out);

/// Copy column data to host buffer with explicit execution options.
///
/// `host_buf` must be large enough. Only valid for fixed-width types.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. The provided stream is
/// synchronized before return so `host_buf` is fully populated on success.
cudf_status_t cudf_column_view_to_host_ex(cudf_column_view_handle handle,
                                          void *host_buf, size_t buf_size,
                                          const cudf_exec_options_t *options);

/// Get the raw device (GPU) data pointer of a column_view.
///
/// Returns the pointer to the device memory that backs this column's data.
/// The pointer remains valid as long as the underlying column/table is alive.
///
/// For fixed-width types the pointer addresses `size` contiguous elements.
/// For STRING columns the data pointer is NULL. Use child column views
/// for offsets and a string-specific chars helper for the character buffer.
///
/// The offset of the column view is NOT applied — the pointer is to the
/// base of the device buffer. Callers working with sliced views must add
/// `offset * element_size` manually.
cudf_status_t cudf_column_view_data_ptr(cudf_column_view_handle handle,
                                        const void **out);

/// Get the device pointer and active byte length for a STRING column's chars
/// buffer with explicit execution options.
///
/// Returns CUDF_ERROR_TYPE if the column is not STRING type.
/// The returned pointer is the base of the chars child buffer. For sliced
/// parent views, `*out_size` is trimmed so the reported byte span ends at the
/// last byte needed by the logical slice.
/// The returned pointer is borrowed producer-owned device memory.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. The memory resource is
/// retained only for provenance; this helper does not allocate output storage.
cudf_status_t cudf_column_view_string_chars_ptr_and_size_ex(
    cudf_column_view_handle handle, const cudf_exec_options_t *options,
    const void **out_ptr, size_t *out_size);

/// Get the raw device null-mask pointer of a column_view.
///
/// Returns `CUDF_ERROR_NULL` when the column has no null mask.
cudf_status_t cudf_column_view_null_mask_ptr(cudf_column_view_handle handle,
                                             const void **out);

/// Get the element offset of a column_view into its parent buffer.
///
/// Returns 0 for non-sliced columns. For sliced views, the offset indicates
/// how many elements from the start of the device buffer the view begins.
cudf_status_t cudf_column_view_offset(cudf_column_view_handle handle,
                                      int32_t *out);

/// Create a sliced borrowed column view over an existing column view with
/// explicit execution options.
///
/// This is zero-copy with respect to device data: it creates only a new
/// `cudf::column_view` handle. `options` must be non-NULL and must name an
/// explicit valid device, a non-NULL stream, and a non-NULL memory resource.
/// Nullable slices resolve their active-range null count on the selected
/// stream before returning. The memory resource is retained only for
/// provenance; this helper does not allocate output storage.
cudf_status_t cudf_column_view_slice_ex(cudf_column_view_handle handle,
                                        int32_t offset, int32_t length,
                                        const cudf_exec_options_t *options,
                                        cudf_column_view_handle *out);

/// Copy null mask to host buffer with explicit execution options.
///
/// `host_buf` must be at least bitmask_allocation_size_bytes(size) bytes.
/// Returns CUDF_ERROR_NULL if column has no null mask.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. The selected stream is
/// synchronized before return so `host_buf` is fully populated on success.
/// For sliced views, the exported host bitmask is normalized to the logical
/// slice so the view's first row maps to bit 0.
cudf_status_t
cudf_column_view_null_mask_to_host_ex(cudf_column_view_handle handle,
                                      uint8_t *host_buf, size_t buf_size,
                                      const cudf_exec_options_t *options);

/// Create a column by copying data from an existing device (GPU) pointer.
///
/// `device_data` must point to GPU memory with `size` elements of `type_id`.
/// Performs a device-to-device copy; the new column owns its own memory.
///
/// For `size == 0`, `device_data` may be NULL.
/// For `size > 0`, `device_data` must be non-NULL.
///
/// Execution-sensitive production callers must use
/// `cudf_column_create_from_device_data_ex`; this wrapper rejects calls because
/// explicit execution options are required.
cudf_status_t cudf_column_create_from_device_data(cudf_type_id_t type_id,
                                                  const void *device_data,
                                                  int32_t size,
                                                  cudf_column_handle *out);

/// Create a column by copying data from an existing device (GPU) pointer with
/// explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
///
/// Unlike `cudf_column_create_from_device_data`, this `_ex` variant does not
/// synchronize before returning. The caller must keep `device_data` valid until
/// work on the selected stream has completed, and must establish any required
/// cross-stream ordering for later consumers of the returned column.
cudf_status_t cudf_column_create_from_device_data_ex(
    cudf_type_id_t type_id, const void *device_data, int32_t size,
    const cudf_exec_options_t *options, cudf_column_handle *out);

/// Create a STRING type column from host string data with explicit execution
/// options.
///
/// `strings` is an array of pointers to UTF-8 string data (not necessarily
/// null-terminated). `lengths` holds the byte length of each string.
/// `num_strings` is the number of strings.
///
/// If `null_mask` is non-NULL and `null_count` > 0, the null_mask is a
/// packed bitmask (LSB order) indicating which rows are valid.
///
/// For num_strings == 0, an empty STRING column is returned.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
///
/// If host-source copies are queued, the selected stream is synchronized
/// before return on success or failure so the borrowed host buffers are no
/// longer in use.
cudf_status_t cudf_column_create_strings_ex(
    const char **strings, const int32_t *lengths, int32_t num_strings,
    const uint8_t *null_mask, int32_t null_count,
    const cudf_exec_options_t *options, cudf_column_handle *out);

/// Create a column by repeating a scalar value `size` times with explicit
/// execution options.
///
/// Uses `cudf::make_column_from_scalar` under the hood.
/// For `size == 0`, an empty column of the scalar's type is returned.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column.
cudf_status_t cudf_column_from_scalar_ex(cudf_scalar_handle scalar,
                                         int32_t size, cudf_column_handle *out,
                                         const cudf_exec_options_t *options);

/// Copy string column data to host buffers with explicit execution options.
///
/// For a STRING column with `n` rows, this copies:
///   - `offsets_buf`: n+1 int32 values (byte offsets into chars).
///   - `chars_buf`: the concatenated character bytes.
///
/// Empty STRING columns still require `offsets_buf`; they write exactly one
/// zero offset (`offsets_buf[0] = 0`) and report `*total_chars_bytes = 0`.
/// In that case `chars_buf` is ignored because there are no chars to copy.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. The selected stream is
/// synchronized before return whenever offsets or chars are materialized into
/// the caller's host buffers.
cudf_status_t cudf_column_view_strings_to_host_ex(
    cudf_column_view_handle handle, int32_t *offsets_buf,
    size_t offsets_buf_size, char *chars_buf, size_t chars_buf_size,
    int32_t *total_chars_bytes, const cudf_exec_options_t *options);

#ifdef __cplusplus
}
#endif

#endif // CUDF_RUST_COLUMN_H
