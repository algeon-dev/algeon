#include "table.h"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

namespace {

// Internal fault injection for ffi_regressions.rs. Thread-local so parallel
// tests on other threads do not see the failpoint.
thread_local bool g_fail_next_table_release_columns_handle_array_alloc = false;

bool should_fail_table_release_columns_handle_array_alloc() {
  if (!g_fail_next_table_release_columns_handle_array_alloc) { return false; }
  g_fail_next_table_release_columns_handle_array_alloc = false;
  return true;
}

cudf_status_t validate_table_create_columns(cudf_column_handle* columns,
                                            int32_t num_columns) {
  for (int32_t i = 0; i < num_columns; ++i) {
    if (columns[i] == nullptr) {
      set_error("column handle is null");
      return CUDF_ERROR_NULL;
    }
  }

  if (num_columns == 0) { return CUDF_SUCCESS; }

  auto const expected_size = static_cast<cudf::column*>(columns[0])->size();
  for (int32_t i = 1; i < num_columns; ++i) {
    auto const actual_size = static_cast<cudf::column*>(columns[i])->size();
    if (actual_size != expected_size) {
      set_error("Column size mismatch: " + std::to_string(actual_size) +
                " != " + std::to_string(expected_size));
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }

  return CUDF_SUCCESS;
}

}  // namespace

extern "C" {

void cudf_test_fail_next_table_release_columns_handle_array_alloc() {
  g_fail_next_table_release_columns_handle_array_alloc = true;
}

cudf_status_t cudf_table_create(cudf_column_handle* columns,
                                int32_t num_columns,
                                cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(columns, "columns is null");
  }
  *out = nullptr;
  CUDF_RUST_TRY {
    auto const validation_status =
        validate_table_create_columns(columns, num_columns);
    if (validation_status != CUDF_SUCCESS) { return validation_status; }

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(num_columns);
    auto* table_storage = ::operator new(sizeof(cudf::table), std::nothrow);
    if (table_storage == nullptr) {
      set_error("failed to allocate table handle");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < num_columns; ++i) {
      cols.push_back(
          std::unique_ptr<cudf::column>(static_cast<cudf::column*>(columns[i])));
    }

    // Remaining operations are non-fallible after validation, reserve, and
    // raw storage allocation complete: transfer ownership atomically by only
    // nulling caller handles once table construction succeeds.
    auto* tbl = new (table_storage) cudf::table(std::move(cols));
    for (int32_t i = 0; i < num_columns; ++i) {
      columns[i] = nullptr;
    }

    *out = static_cast<cudf_table_handle>(tbl);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_copy(cudf_table_handle src,
                               cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* tbl = static_cast<cudf::table*>(src);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto copy = std::make_unique<cudf::table>(tbl->view(), stream, mr);
    *out = static_cast<cudf_table_handle>(copy.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_table_destroy(cudf_table_handle handle) {
  delete static_cast<cudf::table*>(handle);
}

cudf_status_t cudf_table_view(cudf_table_handle handle,
                              cudf_table_view_handle* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* tbl = static_cast<cudf::table*>(handle);
    auto* view = new cudf::table_view(tbl->view());
    *out = static_cast<cudf_table_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_table_view_destroy(cudf_table_view_handle handle) {
  delete static_cast<cudf::table_view*>(handle);
}

cudf_status_t cudf_table_view_num_columns(cudf_table_view_handle handle,
                                          int32_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(handle);
    *out = view->num_columns();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_view_num_rows(cudf_table_view_handle handle,
                                       int32_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(handle);
    *out = view->num_rows();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_view_column(cudf_table_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(handle);
    auto* col_view = new cudf::column_view(view->column(index));
    *out = static_cast<cudf_column_view_handle>(col_view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_view_from_column_views(
    cudf_column_view_handle* column_views, int32_t num_columns,
    cudf_table_view_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(column_views, "column_views is null");
  }
  *out = nullptr;
  CUDF_RUST_TRY {
    std::vector<cudf::column_view> views;
    views.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      CUDF_RUST_NULL_CHECK(column_views[i], "column_view handle is null");
      views.push_back(
          *static_cast<cudf::column_view*>(column_views[i]));
    }
    auto* tv = new cudf::table_view(views);
    *out = static_cast<cudf_table_view_handle>(tv);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_release_columns(cudf_table_handle table,
                                         cudf_column_handle** out_columns,
                                         int32_t* out_count) {
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_NULL_CHECK(out_columns, "out_columns is null");
  CUDF_RUST_NULL_CHECK(out_count, "out_count is null");
  *out_columns = nullptr;
  *out_count = 0;
  CUDF_RUST_TRY {
    auto* tbl = static_cast<cudf::table*>(table);
    auto num_cols = static_cast<int32_t>(tbl->num_columns());

    if (num_cols == 0) {
      auto owned_table = std::unique_ptr<cudf::table>(tbl);
      owned_table->release();
      *out_count = 0;
      return CUDF_SUCCESS;
    }

    // Prepare the output buffer before consuming `table` so allocation
    // failures leave ownership unchanged for the caller.
    if (should_fail_table_release_columns_handle_array_alloc()) {
      set_error("failed to allocate column handle array");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }

    auto* arr = new (std::nothrow) cudf_column_handle[num_cols];
    if (arr == nullptr) {
      set_error("failed to allocate column handle array");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }

    // From this point onward, remaining steps are non-fallible: release the
    // columns, transfer their handles, and destroy the now-empty table.
    auto owned_table = std::unique_ptr<cudf::table>(tbl);
    auto cols = owned_table->release();
    for (int32_t i = 0; i < num_cols; ++i) {
      arr[i] = static_cast<cudf_column_handle>(cols[i].release());
    }

    *out_columns = arr;
    *out_count = num_cols;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_free_column_array(cudf_column_handle* array) {
  delete[] array;
}

}  // extern "C"
