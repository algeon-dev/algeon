#include "table.h"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <vector>

extern "C" {

cudf_status_t cudf_table_create(cudf_column_handle* columns,
                                int32_t num_columns,
                                cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(columns, "columns is null");
  }
  CUDF_RUST_TRY {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      CUDF_RUST_NULL_CHECK(columns[i], "column handle is null");
      // Take ownership of each column.
      cols.push_back(
          std::unique_ptr<cudf::column>(static_cast<cudf::column*>(columns[i])));
      // Null out the caller's handle to prevent double-free.
      columns[i] = nullptr;
    }
    auto tbl = std::make_unique<cudf::table>(std::move(cols));
    *out = static_cast<cudf_table_handle>(tbl.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_copy(cudf_table_handle src,
                               cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
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

}  // extern "C"
