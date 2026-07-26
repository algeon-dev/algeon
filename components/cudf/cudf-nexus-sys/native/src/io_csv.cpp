#include "execution_context_internal.hpp"
#include "io.h"
#include "shim_common.h"

#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <string>
#include <vector>

extern "C" {
cudf_status_t cudf_io_read_csv_ex(const char* filepath,
                                  char delimiter,
                                  int32_t has_header,
                                  cudf_table_handle* out,
                                  cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::csv_reader_options::builder(source).delimiter(delimiter).header(
      has_header ? 0 : -1);

    {
      auto options         = builder.build();
      auto [tbl, metadata] = cudf::io::read_csv(options, exec->stream, *exec->mr);

      *out = cudf_release_table_handle(std::move(tbl), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_write_csv_ex(cudf_table_view_handle table,
                                   const char* filepath,
                                   const char** column_names,
                                   int32_t num_columns,
                                   char delimiter,
                                   int32_t include_header,
                                   cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(column_names, "column_names is null"); }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = static_cast<cudf::table_view*>(table);
    auto sink  = cudf::io::sink_info(std::string(filepath));

    auto builder = cudf::io::csv_writer_options::builder(sink, *view)
                     .inter_column_delimiter(delimiter)
                     .include_header(include_header != 0);

    if (column_names != nullptr && num_columns > 0) {
      cudf_status_t status = validate_c_string_array(column_names, num_columns, "column_names");
      if (status != CUDF_SUCCESS) { return status; }
      std::vector<std::string> names;
      names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        names.emplace_back(column_names[i]);
      }
      builder.names(std::move(names));
    }

    auto options = builder.build();
    {
      cudf::io::write_csv(options, exec->stream);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}
}  // extern "C"
