#pragma once

// Crate-local C++ shim: cugraph-sys stays limited to upstream C API bindings,
// while cugraph needs to build a raft::handle_t from a cudf Runtime stream
// and RMM memory resource before handing it to cugraph_create_resource_handle().

#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cugraph_rs_runtime_resource_handle
    cugraph_rs_runtime_resource_handle_t;

typedef enum cugraph_rs_cuda_failure_classification {
  CUGRAPH_RS_CUDA_FAILURE_NONE = 0,
  CUGRAPH_RS_CUDA_FAILURE_RECOVERABLE = 1,
  CUGRAPH_RS_CUDA_FAILURE_FATAL = 2
} cugraph_rs_cuda_failure_classification_t;

typedef enum cugraph_rs_device_resource_kind {
  CUGRAPH_RS_DEVICE_RESOURCE_SIMILARITY_RESULT = 1,
  CUGRAPH_RS_DEVICE_RESOURCE_VERTEX_PAIRS = 2,
  CUGRAPH_RS_DEVICE_RESOURCE_RNG_STATE = 3,
  CUGRAPH_RS_DEVICE_RESOURCE_CORE_RESULT = 4,
  CUGRAPH_RS_DEVICE_RESOURCE_K_CORE_RESULT = 5,
  CUGRAPH_RS_DEVICE_RESOURCE_DEGREES_RESULT = 6,
  CUGRAPH_RS_DEVICE_RESOURCE_INDUCED_SUBGRAPH_RESULT = 7,
  CUGRAPH_RS_DEVICE_RESOURCE_LAYOUT_RESULT = 8,
  CUGRAPH_RS_DEVICE_RESOURCE_HIERARCHICAL_CLUSTERING_RESULT = 9,
  CUGRAPH_RS_DEVICE_RESOURCE_TRIANGLE_COUNT_RESULT = 10,
  CUGRAPH_RS_DEVICE_RESOURCE_RANDOM_WALK_RESULT = 11,
  CUGRAPH_RS_DEVICE_RESOURCE_SAMPLE_RESULT = 12,
  CUGRAPH_RS_DEVICE_RESOURCE_CLUSTERING_RESULT = 13,
  CUGRAPH_RS_DEVICE_RESOURCE_EDGELIST = 14,
  CUGRAPH_RS_DEVICE_RESOURCE_GRAPH = 15,
  CUGRAPH_RS_DEVICE_RESOURCE_BFS_PREDICATE_RESULT = 16,
  CUGRAPH_RS_DEVICE_RESOURCE_PATHS_RESULT = 17,
  CUGRAPH_RS_DEVICE_RESOURCE_EXTRACT_PATHS_RESULT = 18,
  CUGRAPH_RS_DEVICE_RESOURCE_RUNTIME_RESOURCE_HANDLE = 19,
  CUGRAPH_RS_DEVICE_RESOURCE_RESOURCE_HANDLE = 20,
  CUGRAPH_RS_DEVICE_RESOURCE_CENTRALITY_RESULT = 21,
  CUGRAPH_RS_DEVICE_RESOURCE_EDGE_CENTRALITY_RESULT = 22,
  CUGRAPH_RS_DEVICE_RESOURCE_HITS_RESULT = 23,
  CUGRAPH_RS_DEVICE_RESOURCE_LABELING_RESULT = 24,
  CUGRAPH_RS_DEVICE_RESOURCE_DEVICE_ARRAY = 26
} cugraph_rs_device_resource_kind_t;

typedef enum cugraph_rs_destroy_status {
  CUGRAPH_RS_DESTROY_SUCCESS = 0,
  CUGRAPH_RS_DESTROY_CUDA_ERROR = 1,
  CUGRAPH_RS_DESTROY_FATAL_CUDA = 2,
  CUGRAPH_RS_DESTROY_UNKNOWN_ERROR = 3,
  CUGRAPH_RS_DESTROY_INVALID_RESOURCE_KIND = 4
} cugraph_rs_destroy_status_t;

cugraph_rs_runtime_resource_handle_t *
cugraph_rs_runtime_resource_handle_create(void *cuda_stream,
                                          void *device_memory_resource);

cugraph_resource_handle_t *cugraph_rs_runtime_resource_handle_get(
    cugraph_rs_runtime_resource_handle_t *handle);

// Inspect CUDA's typed thread/context status after a failed cuGraph C API call.
// This deliberately avoids classifying failures from human-readable messages.
cugraph_rs_cuda_failure_classification_t
cugraph_rs_classify_cuda_failure_after_error(int32_t *cuda_status);

// Consume one device-bound cuGraph owner. Every upstream void destructor is
// contained behind this typed, exception-catching boundary.
cugraph_rs_destroy_status_t cugraph_rs_destroy_device_resource(
    void *resource, cugraph_rs_device_resource_kind_t kind,
    int32_t *cuda_status);

// Test-only fault injection. The next destroy on this host thread consumes the
// owner without crossing the upstream void destructor boundary.
void cugraph_rs_test_fail_next_device_resource_destroy(
    cugraph_rs_destroy_status_t status, int32_t cuda_status);

#ifdef __cplusplus
}
#endif
