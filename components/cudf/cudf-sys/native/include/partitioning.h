#ifndef CUDF_RUST_PARTITIONING_H
#define CUDF_RUST_PARTITIONING_H

#include "error.h"
#include "execution_context.h"
#include "table.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Hash-partition table rows by selected column indices.
///
/// `column_indices` must contain `num_columns` valid input column indices.
/// `partition_offsets` must have space for `num_partitions + 1` int32 values.
cudf_status_t cudf_hash_partition(cudf_table_view_handle input,
                                  const int32_t* column_indices,
                                  int32_t num_columns,
                                  int32_t num_partitions,
                                  cudf_table_handle* out,
                                  int32_t* partition_offsets,
                                  size_t partition_offsets_len);

/// Hash-partition with explicit execution context.
cudf_status_t cudf_hash_partition_ex(cudf_table_view_handle input,
                                     const int32_t* column_indices,
                                     int32_t num_columns,
                                     int32_t num_partitions,
                                     cudf_table_handle* out,
                                     int32_t* partition_offsets,
                                     size_t partition_offsets_len,
                                     const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_PARTITIONING_H
