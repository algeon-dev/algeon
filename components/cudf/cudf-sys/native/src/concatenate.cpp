#include "concatenate.h"
#include "execution_context_internal.hpp"
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
  return cudf_concatenate_columns_ex(columns, num_columns, nullptr, out);
}

cudf_status_t cudf_concatenate_columns_ex(
    const cudf_column_view_handle* columns,
    int32_t num_columns,
    cudf_exec_options_t const* options,
    cudf_column_handle* out) {
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
    std::vector<cudf::column_view> views;
    views.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      CUDF_RUST_NULL_CHECK(columns[i], "column view handle is null");
      auto* cv = static_cast<cudf::column_view*>(columns[i]);
      views.push_back(*cv);
    }

    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto result = cudf::concatenate(views, exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_concatenate_tables(
    const cudf_table_view_handle* tables,
    int32_t num_tables,
    cudf_table_handle* out) {
  return cudf_concatenate_tables_ex(tables, num_tables, nullptr, out);
}

cudf_status_t cudf_concatenate_tables_ex(
    const cudf_table_view_handle* tables,
    int32_t num_tables,
    cudf_exec_options_t const* options,
    cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (num_tables < 0) {
    set_error("num_tables must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_tables > 0) {
    CUDF_RUST_NULL_CHECK(tables, "tables is null");
  }
  CUDF_RUST_TRY {
    std::vector<cudf::table_view> views;
    views.reserve(num_tables);
    for (int32_t i = 0; i < num_tables; ++i) {
      CUDF_RUST_NULL_CHECK(tables[i], "table view handle is null");
      auto* tv = static_cast<cudf::table_view*>(tables[i]);
      views.push_back(*tv);
    }

    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto result = cudf::concatenate(views, exec.stream, exec.mr);
    *out = cudf_release_table_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
