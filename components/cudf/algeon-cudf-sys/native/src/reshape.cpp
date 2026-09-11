#include "cudf_rust/reshape.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

extern "C" {
cudf_status_t cudf_interleave_columns_ex(cudf_column_view_handle const* columns,
                                         int32_t num_columns,
                                         cudf_exec_options_t const* options,
                                         cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (num_columns <= 0) {
    set_error("num_columns must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(columns, "columns is null");
  CUDF_RUST_TRY
  {
    std::vector<cudf::column_view> views;
    views.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      CUDF_RUST_NULL_CHECK(columns[i], "column view handle is null");
      auto* cv = reinterpret_cast<cudf::column_view*>(columns[i]);
      views.push_back(*cv);
    }

    auto const rows = static_cast<int64_t>(views.front().size());
    for (auto const& view : views) {
      if (static_cast<int64_t>(view.size()) != rows) {
        set_error("interleave input columns must have the same row count");
        return CUDF_ERROR_INVALID_OPERATION;
      }
    }
    auto const output_size = rows * static_cast<int64_t>(num_columns);
    if (output_size > static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max())) {
      set_error("interleave output size exceeds cudf size_type maximum");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto result = cudf::interleave_columns(cudf::table_view{views}, exec->stream, *exec->mr);
    *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
