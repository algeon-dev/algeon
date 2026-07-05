#include "table.h"

#include "execution_context_internal.hpp"
#include "owning_handle.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <vector>

namespace {
// Internal fault injection for ffi_regressions.rs. Thread-local so parallel
// tests on other threads do not see the failpoint.
thread_local bool g_fail_next_table_release_columns_handle_array_alloc = false;
thread_local bool g_fail_next_table_view_negative_num_columns = false;
thread_local bool g_fail_next_table_view_negative_num_rows = false;

bool should_fail_table_release_columns_handle_array_alloc() {
  if (!g_fail_next_table_release_columns_handle_array_alloc) {
    return false;
  }
  g_fail_next_table_release_columns_handle_array_alloc = false;
  return true;
}

bool should_fail_next_table_view_negative_num_columns() {
  if (!g_fail_next_table_view_negative_num_columns) {
    return false;
  }
  g_fail_next_table_view_negative_num_columns = false;
  return true;
}

bool should_fail_next_table_view_negative_num_rows() {
  if (!g_fail_next_table_view_negative_num_rows) {
    return false;
  }
  g_fail_next_table_view_negative_num_rows = false;
  return true;
}

cudf_status_t validate_table_create_columns(cudf_column_handle *columns,
                                            int32_t num_columns) {
  for (int32_t i = 0; i < num_columns; ++i) {
    if (columns[i] == nullptr) {
      set_error("column handle is null");
      return CUDF_ERROR_NULL;
    }
  }

  if (num_columns == 0) {
    return CUDF_SUCCESS;
  }

  auto const expected_size = cudf_column_state_from_handle(columns[0]).size();
  for (int32_t i = 1; i < num_columns; ++i) {
    auto const actual_size = cudf_column_state_from_handle(columns[i]).size();
    if (actual_size != expected_size) {
      set_error("Column size mismatch: " + std::to_string(actual_size) +
                " != " + std::to_string(expected_size));
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }

  return CUDF_SUCCESS;
}
} // namespace

extern "C" {
void cudf_test_fail_next_table_release_columns_handle_array_alloc() {
  g_fail_next_table_release_columns_handle_array_alloc = true;
}

void cudf_test_table_view_fail_next_negative_num_columns() {
  g_fail_next_table_view_negative_num_columns = true;
}

void cudf_test_table_view_fail_next_negative_num_rows() {
  g_fail_next_table_view_negative_num_rows = true;
}

cudf_status_t cudf_table_create(cudf_column_handle *columns,
                                int32_t num_columns, cudf_table_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(columns, "columns is null");
  }
  CUDF_RUST_TRY {
    cudf_status_t const validation_status =
        validate_table_create_columns(columns, num_columns);
    if (validation_status != CUDF_SUCCESS) {
      return validation_status;
    }

    std::vector<std::unique_ptr<cudf_column_handle_state>> states;
    states.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      states.push_back(cudf_take_column_state(columns[i]));
    }

    for (int32_t i = 0; i < num_columns; ++i) {
      columns[i] = nullptr;
    }

    auto const expected_rows = num_columns == 0
                                   ? static_cast<cudf::size_type>(0)
                                   : states.front()->size();
    *out = cudf_release_table_handle(cudf_make_table_handle_state_from_columns(
        expected_rows, std::move(states)));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_copy_ex(cudf_table_handle src,
                                 cudf_exec_options_t const *options,
                                 cudf_table_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto copy =
          cudf_table_state_from_handle(src).deep_copy(exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(copy),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_create_from_owned_device_transfer(
    cudf_interop_semantic_table_t const *table,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_table_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto state = cudf_make_foreign_table_handle_state(
        table, semantic_columns, num_semantic_columns, bindings, num_bindings);
    *out = cudf_release_table_handle(std::move(state));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_payload_size_ex(cudf_table_handle handle,
                                         cudf_exec_options_t const *options,
                                         size_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    *out = cudf_table_state_from_handle(handle).payload_size(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_device_allocated_size(cudf_table_handle handle,
                                               size_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    *out = cudf_table_state_from_handle(handle).device_allocated_size();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_table_destroy(cudf_table_handle handle) {
  if (handle == nullptr) {
    return;
  }
  auto *state = static_cast<cudf_table_handle_state *>(handle);
  if (auto provenance = state->destroy_provenance()) {
    auto const cleanup =
        cudf_destroy_under_provenance(*provenance, [state]() { delete state; });
    if (cleanup != cudf_provenance_cleanup_result::completed) {
      std::terminate();
    }
  } else {
    delete state;
  }
}

cudf_status_t cudf_table_view(cudf_table_handle handle,
                              cudf_table_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view =
        new cudf::table_view(cudf_table_state_from_handle(handle).view());
    *out = static_cast<cudf_table_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_table_view_destroy(cudf_table_view_handle handle) {
  delete static_cast<cudf::table_view *>(handle);
}

cudf_status_t cudf_table_view_num_columns(cudf_table_view_handle handle,
                                          int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  if (should_fail_next_table_view_negative_num_columns()) {
    *out = -1;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::table_view *>(handle);
    *out = view->num_columns();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_view_num_rows(cudf_table_view_handle handle,
                                       int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  if (should_fail_next_table_view_negative_num_rows()) {
    *out = -1;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::table_view *>(handle);
    *out = view->num_rows();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_view_column(cudf_table_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::table_view *>(handle);
    auto *col_view = new cudf::column_view(view->column(index));
    *out = static_cast<cudf_column_view_handle>(col_view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t
cudf_table_view_from_column_views(cudf_column_view_handle *column_views,
                                  int32_t num_columns,
                                  cudf_table_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(column_views, "column_views is null");
  }
  CUDF_RUST_TRY {
    std::vector<cudf::column_view> views;
    views.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      CUDF_RUST_NULL_CHECK(column_views[i], "column_view handle is null");
      views.push_back(*static_cast<cudf::column_view *>(column_views[i]));
    }
    auto *tv = new cudf::table_view(views);
    *out = static_cast<cudf_table_view_handle>(tv);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_view_select_columns(cudf_table_view_handle handle,
                                             const int32_t *indices,
                                             int32_t num_indices,
                                             cudf_table_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  if (num_indices < 0) {
    set_error("num_indices must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_indices > 0) {
    CUDF_RUST_NULL_CHECK(indices, "indices is null");
  }
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::table_view *>(handle);
    std::vector<cudf::column_view> selected;
    selected.reserve(num_indices);

    for (int32_t i = 0; i < num_indices; ++i) {
      auto const index = indices[i];
      if (index < 0 || index >= view->num_columns()) {
        set_error("table column index out of bounds");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      selected.push_back(view->column(index));
    }

    auto *projected = new cudf::table_view(selected);
    *out = static_cast<cudf_table_view_handle>(projected);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_release_columns(cudf_table_handle table,
                                         cudf_column_handle **out_columns,
                                         int32_t *out_count) {
  CUDF_RUST_NULL_CHECK(out_columns, "out_columns is null");
  *out_columns = nullptr;
  CUDF_RUST_NULL_CHECK(out_count, "out_count is null");
  *out_count = 0;
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_TRY {
    auto &table_state = cudf_table_state_from_handle(table);
    auto num_cols = table_state.num_columns();

    if (num_cols == 0) {
      auto owned_table = cudf_take_table_state(table);
      *out_count = 0;
      return CUDF_SUCCESS;
    }

    // Prepare the output buffer before consuming `table` so allocation
    // failures leave ownership unchanged for the caller.
    if (should_fail_table_release_columns_handle_array_alloc()) {
      set_error("failed to allocate column handle array");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }

    std::unique_ptr<cudf_column_handle[]> arr{new (std::nothrow)
                                                  cudf_column_handle[num_cols]};
    if (arr == nullptr) {
      set_error("failed to allocate column handle array");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < num_cols; ++i) {
      arr[i] = nullptr;
    }

    auto const num_rows = table_state.num_rows();
    std::vector<std::unique_ptr<cudf_column_handle_state>> cols;
    try {
      cols = table_state.release_columns();
      for (int32_t i = 0; i < num_cols; ++i) {
        arr[i] = cudf_publish_column_handle(cols[i]);
      }
    } catch (...) {
      std::vector<std::unique_ptr<cudf_column_handle_state>> restored_cols;
      restored_cols.reserve(static_cast<std::size_t>(num_cols));
      for (int32_t i = 0; i < num_cols; ++i) {
        if (arr[i] != nullptr) {
          restored_cols.push_back(cudf_take_column_state(arr[i]));
          arr[i] = nullptr;
        } else if (static_cast<std::size_t>(i) < cols.size() &&
                   cols[i] != nullptr) {
          restored_cols.push_back(std::move(cols[i]));
        }
      }
      table_state.restore_released_columns(num_rows, std::move(restored_cols));
      throw;
    }

    auto owned_table = cudf_take_table_state(table);
    owned_table.reset();
    *out_columns = arr.release();
    *out_count = num_cols;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_free_column_array(cudf_column_handle *array) { delete[] array; }
} // extern "C"
