#pragma once

#include <cugraph_c/array.h>
#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Crate-private stack payload. No upstream error owner or thread-local state is retained.
typedef struct {
  cugraph_allocation_source_t allocation_source;
  cugraph_error_failure_source_t failure_source;
  uint64_t domain_id;
  uint64_t member_id;
  uint64_t generation;
  uint64_t requested_bytes;
  uint64_t alignment;
  int32_t cuda_error;
  char message[512];
} cugraph_rs_edge_count_error_t;

typedef struct {
  size_t count;
  int64_t source;
  int64_t destination;
  size_t multiplicity;
} cugraph_rs_edge_count_result_t;

cugraph_error_code_t cugraph_rs_count_logical_edges(
  void* cuda_stream,
  void* device_memory_resource,
  cugraph_type_erased_device_array_view_t const* sources,
  cugraph_type_erased_device_array_view_t const* destinations,
  cugraph_graph_properties_t properties,
  cugraph_rs_edge_count_result_t* result,
  cugraph_rs_edge_count_error_t* error);

#ifdef __cplusplus
}
#endif
