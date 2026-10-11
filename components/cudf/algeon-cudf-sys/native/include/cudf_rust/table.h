#ifndef CUDF_RUST_TABLE_H
#define CUDF_RUST_TABLE_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::table (owning).
typedef struct cudf_table_handle_state* cudf_table_handle;

/// Opaque handle to a cudf::table_view (non-owning).
typedef struct cudf_table_view_handle_state* cudf_table_view_handle;

/// Create a table from an array of column handles.
///
/// On success, all input column handles are consumed and moved into the
/// returned table. The caller must NOT destroy any of the consumed column
/// handles after a successful call.
///
/// On failure, no input columns are consumed. The caller retains ownership of
/// every input column handle and remains responsible for destroying them.
cudf_status_t cudf_table_create(cudf_column_handle* columns,
                                int32_t num_columns,
                                cudf_table_handle* out);

/// Create a table from an array of column handles with an explicit row count.
///
/// The explicit row count is required to preserve non-zero rows when
/// `num_columns == 0`. When columns are present, it must match every column's
/// size. Ownership and failure behavior match cudf_table_create.
cudf_status_t cudf_table_create_with_num_rows(cudf_column_handle* columns,
                                              int32_t num_columns,
                                              int32_t num_rows,
                                              cudf_table_handle* out);

/// Deep-copy a table with explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned table.
cudf_status_t cudf_table_copy_ex(cudf_table_handle src,
                                 cudf_exec_options_t const* options,
                                 cudf_table_handle* out);

typedef struct cudf_interop_semantic_table_t {
  int32_t num_rows;
  uint32_t const* column_ids;
  int32_t num_columns;
} cudf_interop_semantic_table_t;

/// Deep-copy a table from a prepared-device export semantic graph with
/// explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned table.
cudf_status_t cudf_table_copy_from_prepared_device_export_ex(
  cudf_interop_semantic_table_t const* table,
  cudf_interop_semantic_column_t const* semantic_columns,
  int32_t num_semantic_columns,
  cudf_interop_buffer_binding_t const* bindings,
  int32_t num_bindings,
  cudf_exec_options_t const* options,
  cudf_table_handle* out);

/// Deep-copy a table from a prepared-host export semantic graph with explicit
/// execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned table.
cudf_status_t cudf_table_copy_from_prepared_host_export_ex(
  cudf_interop_semantic_table_t const* table,
  cudf_interop_semantic_column_t const* semantic_columns,
  int32_t num_semantic_columns,
  cudf_interop_buffer_binding_t const* bindings,
  int32_t num_bindings,
  cudf_exec_options_t const* options,
  cudf_table_handle* out);

/// Return payload bytes for this owning table with explicit execution options.
///
/// This sums payload accounting for each owning column.
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource.
cudf_status_t cudf_table_payload_size_ex(cudf_table_handle handle,
                                         cudf_exec_options_t const* options,
                                         size_t* out);

/// Return the total device allocation size of this owning table.
///
/// This forwards to libcudf `table::alloc_size()` and includes padding.
cudf_status_t cudf_table_device_allocated_size(cudf_table_handle handle, size_t* out);

/// Destroy an owning table handle.
///
/// Ownership is consumed on call, including on failure. CUDF_SUCCESS means
/// cleanup completed; any error means the allocation was quarantined and the
/// handle must not be reused or destroyed again.
cudf_status_t cudf_table_destroy(cudf_table_handle handle);

/// Get an immutable view of a table. The view is heap-allocated and must
/// be freed with cudf_table_view_destroy. Borrows from `handle`.
cudf_status_t cudf_table_view(cudf_table_handle handle, cudf_table_view_handle* out);

/// Destroy a table_view handle.
cudf_status_t cudf_table_view_destroy(cudf_table_view_handle handle);

/// Get the number of columns in a table_view.
cudf_status_t cudf_table_view_num_columns(cudf_table_view_handle handle, int32_t* out);

/// Get the number of rows in a table_view.
cudf_status_t cudf_table_view_num_rows(cudf_table_view_handle handle, int32_t* out);

/// Get a column_view from a table_view by index. The returned view is
/// heap-allocated and must be freed with cudf_column_view_destroy.
cudf_status_t cudf_table_view_column(cudf_table_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle* out);

/// Construct a table_view from an array of column_view handles.
/// The returned table_view is heap-allocated and must be freed with
/// cudf_table_view_destroy. It borrows from the column_views, which
/// must remain alive for the lifetime of the returned table_view.
cudf_status_t cudf_table_view_from_column_views(cudf_column_view_handle* column_views,
                                                int32_t num_columns,
                                                cudf_table_view_handle* out);

/// Construct a table_view from column_view handles with an explicit row count.
///
/// The explicit row count is required to preserve non-zero rows when
/// `num_columns == 0`. When column views are present, it must match every
/// column view's size. The returned table_view borrows from the column views.
cudf_status_t cudf_table_view_from_column_views_with_num_rows(cudf_column_view_handle* column_views,
                                                              int32_t num_columns,
                                                              int32_t num_rows,
                                                              cudf_table_view_handle* out);

/// Construct a projected table_view by selecting columns from an existing
/// table_view. The returned table_view is heap-allocated and must be freed
/// with cudf_table_view_destroy.
cudf_status_t cudf_table_view_select_columns(cudf_table_view_handle handle,
                                             int32_t const* indices,
                                             int32_t num_indices,
                                             cudf_table_view_handle* out);

/// Release all columns from a table, destroying the table in the process.
///
/// The table is consumed: after a successful call, `table` is destroyed and
/// must NOT be used again. The caller receives ownership of each column via
/// `out_columns`, which is a heap-allocated array of `out_count` column
/// handles. When `out_count == 0`, `*out_columns` is NULL. Each column must be
/// individually destroyed with `cudf_column_destroy`. The array itself must be
/// freed with `cudf_free_column_array` declared below.
///
/// On failure, including host-side allocation failure while preparing
/// `out_columns`, `table` is NOT consumed and remains owned by the caller,
/// who must still destroy it.
cudf_status_t cudf_table_release_columns(cudf_table_handle table,
                                         cudf_column_handle** out_columns,
                                         int32_t* out_count);

/// Free a heap-allocated array of column handles returned by APIs such as
/// cudf_table_release_columns, cudf_groupby_aggregate, and
/// cudf_dictionary_match_dictionaries.
/// Does NOT destroy the individual columns; caller must do that first.
void cudf_free_column_array(cudf_column_handle* array);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_TABLE_H
