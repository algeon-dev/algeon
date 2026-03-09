#include "concatenate.h"
#include "shim_common.h"

#include <cudf/concatenate.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <vector>

extern "C" {

cudf_status_t cudf_concatenate_columns(
    const cudf_column_view_handle* columns,
    int32_t num_columns,
    cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(columns, "columns is null");
  }
  CUDF_RUST_TRY {
    std::vector<cudf::column_view> views;
    views.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      auto* cv = static_cast<cudf::column_view*>(columns[i]);
      views.push_back(*cv);
    }

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::concatenate(views, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_concatenate_tables(
    const cudf_table_view_handle* tables,
    int32_t num_tables,
    cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_tables > 0) {
    CUDF_RUST_NULL_CHECK(tables, "tables is null");
  }
  CUDF_RUST_TRY {
    std::vector<cudf::table_view> views;
    views.reserve(num_tables);
    for (int32_t i = 0; i < num_tables; ++i) {
      auto* tv = static_cast<cudf::table_view*>(tables[i]);
      views.push_back(*tv);
    }

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::concatenate(views, stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
