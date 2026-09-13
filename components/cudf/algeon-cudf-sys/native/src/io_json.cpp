#include "cudf_rust/io.h"
#include "execution_context_internal.hpp"
#include "io_test_hooks.hpp"
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
cudf_status_t cudf_io_read_json_ex(char const* filepath,
                                   int32_t lines,
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
    auto builder = cudf::io::json_reader_options::builder(source).lines(lines != 0);

    {
      auto options         = builder.build();
      auto [tbl, metadata] = cudf::io::read_json(options, exec->stream, *exec->mr);

      *out = cudf_release_table_handle(std::move(tbl), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_json_with_metadata_ex(char const* filepath,
                                                 int32_t lines,
                                                 cudf_table_handle* out_table,
                                                 char*** out_names,
                                                 int32_t* out_num_names,
                                                 cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::json_reader_options::builder(source).lines(lines != 0);

    cudf::io::table_with_metadata result;
    {
      auto options = builder.build();
      result       = cudf::io::read_json(options, exec->stream, *exec->mr);
    }

    auto const fault = cudf::rust::test_support::take_io_metadata_fault();
    if (fault != 0) {
      cudf_status_t const fault_status =
        cudf::rust::test_support::apply_io_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table =
        cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    auto const& schema = result.metadata.schema_info;
    auto const n       = static_cast<int32_t>(schema.size());
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
        auto const size  = name.size() + 1;
        names[i]         = new (std::nothrow) char[size];
        if (!names[i]) {
          for (int32_t j = 0; j < i; ++j)
            delete[] names[j];
          delete[] names;
          set_error("failed to allocate column name string");
          return CUDF_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(names[i], name.c_str(), size);
      }
      *out_names     = names;
      *out_num_names = n;
    }

    *out_table =
      cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_write_json_ex(cudf_table_view_handle table,
                                    char const* filepath,
                                    char const** column_names,
                                    int32_t num_columns,
                                    int32_t lines,
                                    int32_t include_nulls,
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

    auto* view = reinterpret_cast<cudf::table_view*>(table);

    auto sink    = cudf::io::sink_info(std::string(filepath));
    auto builder = cudf::io::json_writer_options::builder(sink, *view)
                     .lines(lines != 0)
                     .include_nulls(include_nulls != 0);

    if (num_columns > 0) {
      cudf::io::table_metadata meta;
      meta.schema_info.resize(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        meta.schema_info[i].name = column_names[i];
      }
      builder.metadata(meta);
    }

    auto options = builder.build();
    {
      cudf::io::write_json(options, exec->stream);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}
}  // extern "C"
