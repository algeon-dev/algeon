#include "io.h"
#include "shim_common.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

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

}  // extern "C"
