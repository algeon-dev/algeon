#include "io.h"
#include "shim_common.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" {

cudf_status_t cudf_io_read_parquet(const char* filepath,
                                   const char** columns,
                                   int32_t num_columns,
                                   int64_t skip_rows, int64_t num_rows,
                                   cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
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
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(columns, "columns is null");
  }
  CUDF_RUST_TRY {
    auto source = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (columns != nullptr && num_columns > 0) {
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.columns(col_names);
    }

    if (skip_rows > 0) {
      builder.skip_rows(skip_rows);
    }
    if (num_rows >= 0) {
      builder.num_rows(num_rows);
    }

    auto options = builder.build();
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::io::read_parquet(options, stream, mr);

    *out = static_cast<cudf_table_handle>(result.tbl.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_io_write_parquet(cudf_table_view_handle table,
                                    const char* filepath,
                                    const char** column_names,
                                    int32_t num_columns) {
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
    auto* view = static_cast<cudf::table_view*>(table);
    auto sink = cudf::io::sink_info(std::string(filepath));

    auto builder =
        cudf::io::parquet_writer_options::builder(sink, *view);

    if (column_names != nullptr && num_columns > 0) {
      cudf::io::table_input_metadata table_meta(*view);
      for (int32_t i = 0; i < num_columns && i < static_cast<int32_t>(table_meta.column_metadata.size()); ++i) {
        table_meta.column_metadata[i].set_name(std::string(column_names[i]));
      }
      builder.metadata(std::move(table_meta));
    }

    auto options = builder.build();
    cudf::io::write_parquet(options);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_io_read_parquet_with_metadata(const char* filepath,
                                                  const char** columns,
                                                  int32_t num_columns,
                                                  int64_t skip_rows,
                                                  int64_t num_rows,
                                                  cudf_table_handle* out_table,
                                                  char*** out_names,
                                                  int32_t* out_num_names) {
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
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
  if (num_columns > 0) {
    CUDF_RUST_NULL_CHECK(columns, "columns is null");
  }
  CUDF_RUST_TRY {
    auto source = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (columns != nullptr && num_columns > 0) {
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.columns(col_names);
    }

    if (skip_rows > 0) {
      builder.skip_rows(skip_rows);
    }
    if (num_rows >= 0) {
      builder.num_rows(num_rows);
    }

    auto options = builder.build();
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::io::read_parquet(options, stream, mr);

    // Extract column names from metadata
    auto const& schema_info = result.metadata.schema_info;
    auto const n = static_cast<int32_t>(schema_info.size());

    char** names = static_cast<char**>(std::malloc(
        static_cast<size_t>(n) * sizeof(char*)));
    if (names == nullptr && n > 0) {
      set_error("failed to allocate column name array");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }
    for (int32_t i = 0; i < n; ++i) {
      names[i] = strdup(schema_info[i].name.c_str());
      if (names[i] == nullptr) {
        // Clean up already allocated strings
        for (int32_t j = 0; j < i; ++j) {
          std::free(names[j]);
        }
        std::free(names);
        set_error("failed to allocate column name string");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
    }

    *out_table = static_cast<cudf_table_handle>(result.tbl.release());
    *out_names = names;
    *out_num_names = n;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_io_free_column_names(char** names, int32_t num_names) {
  if (names == nullptr) return;
  for (int32_t i = 0; i < num_names; ++i) {
    std::free(names[i]);
  }
  std::free(names);
}

}  // extern "C"
