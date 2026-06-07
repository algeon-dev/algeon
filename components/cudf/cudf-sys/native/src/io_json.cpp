#include "io.h"
#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" {

cudf_status_t cudf_io_read_json(const char* filepath, int32_t lines,
                                cudf_table_handle* out) {
  return cudf_io_read_json_ex(filepath, lines, out, nullptr);
}

cudf_status_t cudf_io_read_json_ex(const char* filepath, int32_t lines,
                                   cudf_table_handle* out,
                                   cudf_exec_options_t const* exec_options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::json_reader_options::builder(source)
                       .lines(lines != 0);

    auto options = builder.build();
    auto result = cudf::io::read_json(options, exec.stream, exec.mr);

    *out = cudf_release_table_handle(std::move(result.tbl));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_json_with_metadata(
    const char* filepath, int32_t lines,
    cudf_table_handle* out_table,
    char*** out_names, int32_t* out_num_names) {
  return cudf_io_read_json_with_metadata_ex(filepath, lines, out_table,
                                            out_names, out_num_names, nullptr);
}

cudf_status_t cudf_io_read_json_with_metadata_ex(
    const char* filepath, int32_t lines,
    cudf_table_handle* out_table,
    char*** out_names, int32_t* out_num_names,
    cudf_exec_options_t const* exec_options) {
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::json_reader_options::builder(source)
                       .lines(lines != 0);

    auto options = builder.build();
    auto result = cudf::io::read_json(options, exec.stream, exec.mr);

    // Extract column names from metadata.
    auto const& schema = result.metadata.schema_info;
    auto const n = static_cast<int32_t>(schema.size());
    if (n > 0) {
      char** names = new (std::nothrow) char*[n];
      if (!names) {
        set_error("failed to allocate column name array");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
      for (int32_t i = 0; i < n; ++i) {
        names[i] = nullptr;
      }
      for (int32_t i = 0; i < n; ++i) {
        auto const& name = schema[i].name;
        auto const size = name.size() + 1;
        names[i] = new (std::nothrow) char[size];
        if (!names[i]) {
          // Clean up on allocation failure.
          for (int32_t j = 0; j < i; ++j) delete[] names[j];
          delete[] names;
          set_error("failed to allocate column name string");
          return CUDF_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(names[i], name.c_str(), size);
      }
      *out_names = names;
      *out_num_names = n;
    }

    *out_table = cudf_release_table_handle(std::move(result.tbl));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_write_json(cudf_table_view_handle table,
                                 const char* filepath,
                                 const char** column_names,
                                 int32_t num_columns,
                                 int32_t lines, int32_t include_nulls) {
  return cudf_io_write_json_ex(table, filepath, column_names, num_columns,
                               lines, include_nulls, nullptr);
}

cudf_status_t cudf_io_write_json_ex(cudf_table_view_handle table,
                                    const char* filepath,
                                    const char** column_names,
                                    int32_t num_columns,
                                    int32_t lines, int32_t include_nulls,
                                    cudf_exec_options_t const* exec_options) {
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(column_names, "column_names is null");
  }
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = static_cast<cudf::table_view*>(table);

    auto sink = cudf::io::sink_info(std::string(filepath));
    auto builder = cudf::io::json_writer_options::builder(sink, *view)
                       .lines(lines != 0)
                       .include_nulls(include_nulls != 0);

    // Set column names via metadata.
    if (num_columns > 0) {
      cudf::io::table_metadata meta;
      meta.schema_info.resize(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        meta.schema_info[i].name = column_names[i];
      }
      builder.metadata(meta);
    }

    auto options = builder.build();
    cudf::io::write_json(options, exec.stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

}  // extern "C"
