#include "execution_context_internal.hpp"
#include "ast_handle.hpp"
#include "io.h"
#include "io_test_hooks.hpp"
#include "shim_common.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

extern "C" {
cudf_status_t cudf_io_read_parquet(const char* filepath,
                                   const char** columns,
                                   int32_t num_columns,
                                   int64_t skip_rows,
                                   int64_t num_rows,
                                   const int32_t* row_groups,
                                   int32_t num_row_groups,
                                   cudf_table_handle* out)
{
  return cudf_io_read_parquet_ex(
    filepath, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups, out, nullptr);
}

cudf_status_t cudf_io_read_parquet_ex(const char* filepath,
                                      const char** columns,
                                      int32_t num_columns,
                                      int64_t skip_rows,
                                      int64_t num_rows,
                                      const int32_t* row_groups,
                                      int32_t num_row_groups,
                                      cudf_table_handle* out,
                                      cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (skip_rows < 0) {
    set_error("skip_rows must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_rows < -1) {
    set_error("num_rows must be >= -1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups < 0) {
    set_error("num_row_groups must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups > 0 && row_groups == nullptr) {
    set_error("row_groups is null but num_row_groups > 0");
    return CUDF_ERROR_NULL;
  }
  if (num_row_groups > 0 && (skip_rows > 0 || num_rows >= 0)) {
    set_error("row_groups cannot be combined with skip_rows or num_rows");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(columns, "columns is null"); }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (columns != nullptr && num_columns > 0) {
      cudf_status_t status = validate_c_string_array(columns, num_columns, "columns");
      if (status != CUDF_SUCCESS) { return status; }
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.column_names(col_names);
    }

    if (num_row_groups > 0) {
      std::vector<cudf::size_type> rg_indices(row_groups, row_groups + num_row_groups);
      builder.row_groups({std::move(rg_indices)});

    } else {
      if (skip_rows > 0) { builder.skip_rows(skip_rows); }
      if (num_rows >= 0) { builder.num_rows(num_rows); }
    }

    {
      auto options         = builder.build();
      auto [tbl, metadata] = cudf::io::read_parquet(options, exec.stream, exec.mr);

      *out = cudf_release_table_handle(std::move(tbl));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_write_parquet(cudf_table_view_handle table,
                                    const char* filepath,
                                    const char** column_names,
                                    int32_t num_columns,
                                    int32_t row_group_size_rows)
{
  return cudf_io_write_parquet_ex(
    table, filepath, column_names, num_columns, row_group_size_rows, nullptr);
}

cudf_status_t cudf_io_write_parquet_ex(cudf_table_view_handle table,
                                       const char* filepath,
                                       const char** column_names,
                                       int32_t num_columns,
                                       int32_t row_group_size_rows,
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
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = static_cast<cudf::table_view*>(table);
    auto sink  = cudf::io::sink_info(std::string(filepath));

    auto builder = cudf::io::parquet_writer_options::builder(sink, *view);

    if (column_names != nullptr && num_columns > 0) {
      cudf_status_t status = validate_c_string_array(column_names, num_columns, "column_names");
      if (status != CUDF_SUCCESS) { return status; }
      cudf::io::table_input_metadata table_meta(*view);
      auto const names_to_apply =
        std::min(static_cast<size_t>(num_columns), table_meta.column_metadata.size());
      for (size_t i = 0; i < names_to_apply; ++i) {
        table_meta.column_metadata[i].set_name(std::string(column_names[i]));
      }
      builder.metadata(std::move(table_meta));
    }

    auto options = builder.build();

    if (row_group_size_rows > 0) { options.set_row_group_size_rows(row_group_size_rows); }
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      cudf::io::write_parquet(options, exec.stream);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_with_metadata(const char* filepath,
                                                 const char** columns,
                                                 int32_t num_columns,
                                                 int64_t skip_rows,
                                                 int64_t num_rows,
                                                 const int32_t* row_groups,
                                                 int32_t num_row_groups,
                                                 cudf_table_handle* out_table,
                                                 char*** out_names,
                                                 int32_t* out_num_names)
{
  return cudf_io_read_parquet_with_metadata_ex(filepath,
                                               columns,
                                               num_columns,
                                               skip_rows,
                                               num_rows,
                                               row_groups,
                                               num_row_groups,
                                               out_table,
                                               out_names,
                                               out_num_names,
                                               nullptr);
}

cudf_status_t cudf_io_read_parquet_with_metadata_ex(const char* filepath,
                                                    const char** columns,
                                                    int32_t num_columns,
                                                    int64_t skip_rows,
                                                    int64_t num_rows,
                                                    const int32_t* row_groups,
                                                    int32_t num_row_groups,
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
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (skip_rows < 0) {
    set_error("skip_rows must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_rows < -1) {
    set_error("num_rows must be >= -1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups < 0) {
    set_error("num_row_groups must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups > 0 && row_groups == nullptr) {
    set_error("row_groups is null but num_row_groups > 0");
    return CUDF_ERROR_NULL;
  }
  if (num_row_groups > 0 && (skip_rows > 0 || num_rows >= 0)) {
    set_error("row_groups cannot be combined with skip_rows or num_rows");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(columns, "columns is null"); }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (columns != nullptr && num_columns > 0) {
      cudf_status_t status = validate_c_string_array(columns, num_columns, "columns");
      if (status != CUDF_SUCCESS) { return status; }
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.column_names(col_names);
    }

    if (num_row_groups > 0) {
      std::vector<cudf::size_type> rg_indices(row_groups, row_groups + num_row_groups);
      builder.row_groups({std::move(rg_indices)});

    } else {
      if (skip_rows > 0) { builder.skip_rows(skip_rows); }
      if (num_rows >= 0) { builder.num_rows(num_rows); }
    }

    cudf::io::table_with_metadata result;
    {
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec.stream, exec.mr);
    }

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl));
      return CUDF_SUCCESS;
    }

    // Extract column names from metadata
    auto const& schema_info = result.metadata.schema_info;
    auto const n            = static_cast<int32_t>(schema_info.size());

    char** names = nullptr;
    if (n > 0) {
      names = new (std::nothrow) char*[n];
      if (names == nullptr) {
        set_error("failed to allocate column name array");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
      for (int32_t i = 0; i < n; ++i) {
        names[i] = nullptr;
      }
      for (int32_t i = 0; i < n; ++i) {
        auto const& name = schema_info[i].name;
        auto const size  = name.size() + 1;
        names[i]         = new (std::nothrow) char[size];
        if (names[i] == nullptr) {
          for (int32_t j = 0; j < i; ++j) {
            delete[] names[j];
          }
          delete[] names;
          set_error("failed to allocate column name string");
          return CUDF_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(names[i], name.c_str(), size);
      }
    }

    *out_table     = cudf_release_table_handle(std::move(result.tbl));
    *out_names     = names;
    *out_num_names = n;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_with_metadata_filter_ex(const char* filepath,
                                                           const char** columns,
                                                           int32_t num_columns,
                                                           int64_t skip_rows,
                                                           int64_t num_rows,
                                                           const int32_t* row_groups,
                                                           int32_t num_row_groups,
                                                           cudf_ast_expression_handle predicate,
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
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (skip_rows < 0) {
    set_error("skip_rows must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_rows < -1) {
    set_error("num_rows must be >= -1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups < 0) {
    set_error("num_row_groups must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups > 0 && row_groups == nullptr) {
    set_error("row_groups is null but num_row_groups > 0");
    return CUDF_ERROR_NULL;
  }
  if (num_row_groups > 0 && (skip_rows > 0 || num_rows >= 0)) {
    set_error("row_groups cannot be combined with skip_rows or num_rows");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(columns, "columns is null"); }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (columns != nullptr && num_columns > 0) {
      cudf_status_t status = validate_c_string_array(columns, num_columns, "columns");
      if (status != CUDF_SUCCESS) { return status; }
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.column_names(col_names);
    }

    if (num_row_groups > 0) {
      std::vector<cudf::size_type> rg_indices(row_groups, row_groups + num_row_groups);
      builder.row_groups({std::move(rg_indices)});

    } else {
      if (skip_rows > 0) { builder.skip_rows(skip_rows); }
      if (num_rows >= 0) { builder.num_rows(num_rows); }
    }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec.stream, exec.mr);
    }

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl));
      return CUDF_SUCCESS;
    }

    // Extract column names from metadata
    auto const& schema_info = result.metadata.schema_info;
    auto const n            = static_cast<int32_t>(schema_info.size());

    char** names = nullptr;
    if (n > 0) {
      names = new (std::nothrow) char*[n];
      if (names == nullptr) {
        set_error("failed to allocate column name array");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
      for (int32_t i = 0; i < n; ++i) {
        names[i] = nullptr;
      }
      for (int32_t i = 0; i < n; ++i) {
        auto const& name = schema_info[i].name;
        auto const size  = name.size() + 1;
        names[i]         = new (std::nothrow) char[size];
        if (names[i] == nullptr) {
          for (int32_t j = 0; j < i; ++j) {
            delete[] names[j];
          }
          delete[] names;
          set_error("failed to allocate column name string");
          return CUDF_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(names[i], name.c_str(), size);
      }
    }

    *out_table     = cudf_release_table_handle(std::move(result.tbl));
    *out_names     = names;
    *out_num_names = n;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

void cudf_io_free_column_names(char** names, int32_t num_names)
{
  if (names == nullptr) return;
  for (int32_t i = 0; i < num_names; ++i) {
    delete[] names[i];
  }
  delete[] names;
}
}  // extern "C"
