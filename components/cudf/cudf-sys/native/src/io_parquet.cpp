#include "execution_context_internal.hpp"
#include "ast_handle.hpp"
#include "io.h"
#include "io_test_hooks.hpp"
#include "shim_common.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>
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

namespace {
template <typename Builder>
cudf_status_t configure_parquet_reader_builder(Builder& builder,
                                               const char** columns,
                                               int32_t num_columns,
                                               int64_t skip_rows,
                                               int64_t num_rows,
                                               const int32_t* row_groups,
                                               int32_t num_row_groups)
{
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

  return CUDF_SUCCESS;
}

cudf_status_t validate_parquet_reader_args(const char** columns,
                                           int32_t num_columns,
                                           int64_t skip_rows,
                                           int64_t num_rows,
                                           const int32_t* row_groups,
                                           int32_t num_row_groups)
{
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
  return CUDF_SUCCESS;
}

cudf_status_t validate_parquet_multi_reader_args(const char** filepaths,
                                                 int32_t num_files,
                                                 const char** columns,
                                                 int32_t num_columns,
                                                 const int32_t* row_group_offsets,
                                                 int32_t num_row_group_offsets,
                                                 const int32_t* row_groups,
                                                 int32_t num_row_groups)
{
  if (num_files <= 0) {
    set_error("num_files must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(filepaths, "filepaths is null");
  cudf_status_t const filepaths_status = validate_c_string_array(filepaths, num_files, "filepaths");
  if (filepaths_status != CUDF_SUCCESS) { return filepaths_status; }

  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(columns, "columns is null"); }
  if (num_columns > 0) {
    cudf_status_t const column_status = validate_c_string_array(columns, num_columns, "columns");
    if (column_status != CUDF_SUCCESS) { return column_status; }
  }

  if (num_row_group_offsets < 0) {
    set_error("num_row_group_offsets must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups < 0) {
    set_error("num_row_groups must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_group_offsets == 0) {
    if (row_group_offsets != nullptr) {
      set_error("row_group_offsets must be null when num_row_group_offsets is 0");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (num_row_groups != 0 || row_groups != nullptr) {
      set_error("row_groups must be null and num_row_groups must be 0 when row_group_offsets are absent");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    return CUDF_SUCCESS;
  }

  CUDF_RUST_NULL_CHECK(row_group_offsets, "row_group_offsets is null");
  if (num_row_group_offsets != num_files + 1) {
    set_error("num_row_group_offsets must equal num_files + 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups > 0) {
    CUDF_RUST_NULL_CHECK(row_groups, "row_groups is null");
  } else if (row_groups != nullptr) {
    set_error("row_groups must be null when num_row_groups is 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (row_group_offsets[0] != 0) {
    set_error("row_group_offsets[0] must be 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (int32_t i = 0; i < num_files; ++i) {
    if (row_group_offsets[i] > row_group_offsets[i + 1]) {
      set_error("row_group_offsets must be monotonically non-decreasing");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }
  if (row_group_offsets[num_files] != num_row_groups) {
    set_error("last row_group_offsets entry must equal num_row_groups");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (int32_t i = 0; i < num_row_groups; ++i) {
    if (row_groups[i] < 0) {
      set_error("row group index must be >= 0");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }
  return CUDF_SUCCESS;
}

cudf_status_t validate_host_parquet_buffer(const void* host_buffer, size_t host_buffer_len)
{
  if (host_buffer_len == 0) {
    set_error("host_buffer_len must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(host_buffer, "host_buffer is null");
  return CUDF_SUCCESS;
}

char** copy_schema_names(cudf::io::table_metadata const& metadata, int32_t* out_num_names)
{
  auto const& schema_info = metadata.schema_info;
  auto const n            = static_cast<int32_t>(schema_info.size());
  *out_num_names          = n;

  char** names = nullptr;
  if (n > 0) {
    names = new (std::nothrow) char*[n];
    if (names == nullptr) {
      set_error("failed to allocate column name array");
      return nullptr;
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
        return nullptr;
      }
      std::memcpy(names[i], name.c_str(), size);
    }
  }

  return names;
}
}  // namespace

extern "C" {
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
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options         = builder.build();
      auto [tbl, metadata] = cudf::io::read_parquet(options, exec->stream, *exec->mr);

      *out = cudf_release_table_handle(std::move(tbl), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_from_host_buffer_ex(const void* host_buffer,
                                                       size_t host_buffer_len,
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
  cudf_status_t const buffer_status = validate_host_parquet_buffer(host_buffer, host_buffer_len);
  if (buffer_status != CUDF_SUCCESS) { return buffer_status; }
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto bytes = cudf::host_span<std::byte const>{
      static_cast<std::byte const*>(host_buffer), host_buffer_len};
    auto source  = cudf::io::source_info(bytes);
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options         = builder.build();
      auto [tbl, metadata] = cudf::io::read_parquet(options, exec->stream, *exec->mr);

      *out = cudf_release_table_handle(std::move(tbl), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_write_parquet_ex(cudf_table_view_handle table,
                                       const char* filepath,
                                       const char** column_names,
                                       int32_t num_columns,
                                       int32_t row_group_size_rows,
                                       int32_t dictionary_policy,
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
    switch (dictionary_policy) {
      case -1: break;
      case 0: options.set_dictionary_policy(cudf::io::dictionary_policy::ADAPTIVE); break;
      case 1: options.set_dictionary_policy(cudf::io::dictionary_policy::NEVER); break;
      case 2: options.set_dictionary_policy(cudf::io::dictionary_policy::ALWAYS); break;
      default:
        set_error("dictionary_policy must be -1 (default), 0 (adaptive), 1 (never), or 2 (always)");
        return CUDF_ERROR_INVALID_OPERATION;
    }
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      cudf::io::write_parquet(options, exec->stream);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
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
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table     = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names     = names;
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
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table     = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names     = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_multi_with_metadata_filter_ex(
  const char** filepaths,
  int32_t num_files,
  const char** columns,
  int32_t num_columns,
  const int32_t* row_group_offsets,
  int32_t num_row_group_offsets,
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
  cudf_status_t const args_status = validate_parquet_multi_reader_args(
    filepaths,
    num_files,
    columns,
    num_columns,
    row_group_offsets,
    num_row_group_offsets,
    row_groups,
    num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<std::string> paths;
    paths.reserve(num_files);
    for (int32_t i = 0; i < num_files; ++i) {
      paths.emplace_back(filepaths[i]);
    }
    auto source  = cudf::io::source_info(std::move(paths));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (columns != nullptr && num_columns > 0) {
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.column_names(col_names);
    }

    if (num_row_group_offsets > 0) {
      std::vector<std::vector<cudf::size_type>> grouped_row_groups;
      grouped_row_groups.reserve(num_files);
      for (int32_t file_index = 0; file_index < num_files; ++file_index) {
        auto const begin = row_group_offsets[file_index];
        auto const end   = row_group_offsets[file_index + 1];
        std::vector<cudf::size_type> file_row_groups;
        file_row_groups.reserve(end - begin);
        for (int32_t row_group_index = begin; row_group_index < end; ++row_group_index) {
          file_row_groups.push_back(static_cast<cudf::size_type>(row_groups[row_group_index]));
        }
        grouped_row_groups.push_back(std::move(file_row_groups));
      }
      builder.row_groups(std::move(grouped_row_groups));
    }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
  const void* host_buffer,
  size_t host_buffer_len,
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
  cudf_status_t const buffer_status = validate_host_parquet_buffer(host_buffer, host_buffer_len);
  if (buffer_status != CUDF_SUCCESS) { return buffer_status; }
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto bytes = cudf::host_span<std::byte const>{
      static_cast<std::byte const*>(host_buffer), host_buffer_len};
    auto source  = cudf::io::source_info(bytes);
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names = names;
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
