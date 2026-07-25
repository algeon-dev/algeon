#include "cudf_rust/partitioning.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace {
cudf_status_t copy_column_indices(int32_t const* column_indices,
                                  int32_t num_columns,
                                  std::vector<cudf::size_type>* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_columns <= 0) {
    set_error("num_columns must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(column_indices, "column_indices is null");
  out->reserve(static_cast<std::size_t>(num_columns));
  for (int32_t i = 0; i < num_columns; ++i) {
    if (column_indices[i] < 0) {
      set_error("column index must be >= 0");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    out->push_back(static_cast<cudf::size_type>(column_indices[i]));
  }
  return CUDF_SUCCESS;
}
}  // namespace

extern "C" {
cudf_status_t cudf_hash_partition_ex(cudf_table_view_handle input,
                                     int32_t const* column_indices,
                                     int32_t num_columns,
                                     int32_t num_partitions,
                                     cudf_table_handle* out,
                                     int32_t* partition_offsets,
                                     size_t partition_offsets_len,
                                     cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(partition_offsets, "partition_offsets is null");
  if (num_partitions <= 0) {
    set_error("num_partitions must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  auto const required_offsets = static_cast<size_t>(num_partitions) + 1;
  if (partition_offsets_len < required_offsets) {
    set_error("partition_offsets_len must be at least num_partitions + 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  std::fill(partition_offsets, partition_offsets + partition_offsets_len, 0);

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<cudf::size_type> columns;
    cudf_status_t const columns_status = copy_column_indices(column_indices, num_columns, &columns);
    if (columns_status != CUDF_SUCCESS) { return columns_status; }

    auto* view       = reinterpret_cast<cudf::table_view*>(input);
    auto partitioned = [&] {
      // libcudf hash_partition passes `mr` to the output table, but some
      // temporaries still allocate through RMM's current resource.
      return cudf::hash_partition(*view,
                                  columns,
                                  static_cast<int>(num_partitions),
                                  cudf::hash_id::HASH_MURMUR3,
                                  cudf::DEFAULT_HASH_SEED,
                                  exec->stream,
                                  *exec->mr);
    }();
    auto const& offsets = partitioned.second;
    if (offsets.size() != required_offsets) {
      set_error("libcudf returned an unexpected partition offset count");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    for (size_t i = 0; i < offsets.size(); ++i) {
      partition_offsets[i] = static_cast<int32_t>(offsets[i]);
    }
    *out = cudf_release_table_handle(std::move(partitioned.first));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
