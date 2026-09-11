#include "runtime_resource_handle.h"

// cudf passes its runtime memory resource through a project-local compat
// base. Keep memory_resource_compat.hpp byte-identical to cudf, then wrap the
// borrowed pointer as the RAFT device_resource expected by handle_t.
#include "memory_resource_compat.hpp"

#include <cudf_rust/memory_resource.h>
#include <cugraph/detail/effective_device_memory.hpp>
#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/graph_functions.h>
#include <cugraph_c/random.h>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <new>
#include <stdexcept>

// algorithms.h provides the opaque layout-result type through the tree API,
// but intentionally omits the layout header because the upstream headers
// define the same opaque typedef twice.
extern "C" void cugraph_layout_result_free(cugraph_layout_result_t *result);

namespace {
thread_local cugraph_rs_destroy_status_t g_next_destroy_status =
    CUGRAPH_RS_DESTROY_SUCCESS;
thread_local int32_t g_next_destroy_cuda_status = 0;

cugraph_rs_destroy_status_t typed_cuda_destroy_status(int32_t *cuda_status) {
  if (cudaPeekAtLastError() == cudaSuccess) {
    *cuda_status = static_cast<int32_t>(cudaSuccess);
    return CUGRAPH_RS_DESTROY_SUCCESS;
  }
  auto const classification =
      cugraph_rs_classify_cuda_failure_after_error(cuda_status);
  return classification == CUGRAPH_RS_CUDA_FAILURE_FATAL
             ? CUGRAPH_RS_DESTROY_FATAL_CUDA
             : CUGRAPH_RS_DESTROY_CUDA_ERROR;
}

} // namespace

struct cugraph_rs_runtime_resource_handle {
  std::unique_ptr<raft::handle_t> raft_handle;
  cugraph_resource_handle_t *cugraph_handle{nullptr};
  bool effective_memory_provider_registered{false};

  cugraph_rs_runtime_resource_handle(void *cuda_stream,
                                     void *device_memory_resource,
                                     void *memory_resource_owner) {
    auto stream =
        rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(cuda_stream)};
    auto *mr = reinterpret_cast<rmm::mr::device_memory_resource *>(
        device_memory_resource);
    auto owner = reinterpret_cast<cudf_memory_resource_handle>(
        memory_resource_owner);
    // Borrow the cudf-owned resource; this wrapper does not take ownership.
    raft_handle = std::make_unique<raft::handle_t>(
        stream, std::shared_ptr<rmm::cuda_stream_pool>{},
        raft::mr::device_resource{rmm::mr::any_device_resource_from_raw(mr)});
    size_t available = 0;
    auto const effective_memory_status =
        cudf_memory_resource_effective_device_memory(owner, cuda_stream,
                                                     &available);
    if (effective_memory_status == CUDF_SUCCESS) {
      cugraph::detail::register_effective_device_memory_provider(
          *raft_handle, [owner, cuda_stream]() {
            size_t current_available = 0;
            auto const status = cudf_memory_resource_effective_device_memory(
                owner, cuda_stream, &current_available);
            if (status != CUDF_SUCCESS) {
              throw std::runtime_error{
                  "invalid query-domain usage while sizing cuGraph work"};
            }
            return current_available;
          });
      effective_memory_provider_registered = true;
    } else if (effective_memory_status != CUDF_ERROR_UNSUPPORTED) {
      throw std::runtime_error{
          "invalid query-domain usage while creating a cuGraph handle"};
    }
    try {
      cugraph_handle = cugraph_create_resource_handle(raft_handle.get());
      if (cugraph_handle == nullptr) { throw std::bad_alloc{}; }
    } catch (...) {
      cugraph::detail::unregister_effective_device_memory_provider(
          *raft_handle);
      cugraph::detail::release_effective_device_memory_registration(
          *raft_handle);
      effective_memory_provider_registered = false;
      throw;
    }
  }

  void release() {
    if (cugraph_handle != nullptr) {
      cugraph_free_resource_handle(cugraph_handle);
      cugraph_handle = nullptr;
    }
    if (effective_memory_provider_registered) {
      if (!cugraph::detail::unregister_effective_device_memory_provider(
              *raft_handle)) {
        throw std::runtime_error{
            "cuGraph effective-memory provider registration was lost"};
      }
      if (!cugraph::detail::release_effective_device_memory_registration(
              *raft_handle)) {
        throw std::runtime_error{
            "cuGraph effective-memory provider registration could not be released"};
      }
      effective_memory_provider_registered = false;
    }
    raft_handle.reset();
  }

  ~cugraph_rs_runtime_resource_handle() noexcept = default;
};

extern "C" cugraph_rs_runtime_resource_handle_t *
cugraph_rs_runtime_resource_handle_create(void *cuda_stream,
                                          void *device_memory_resource,
                                          void *memory_resource_owner) {
  if ((cuda_stream == nullptr) || (device_memory_resource == nullptr) ||
      (memory_resource_owner == nullptr)) {
    return nullptr;
  }

  try {
    return new cugraph_rs_runtime_resource_handle(
        cuda_stream, device_memory_resource, memory_resource_owner);
  } catch (...) {
    return nullptr;
  }
}

extern "C" int32_t cugraph_rs_runtime_resource_handle_effective_device_memory(
    cugraph_rs_runtime_resource_handle_t *handle, size_t *out) {
  if (handle == nullptr || out == nullptr || handle->raft_handle == nullptr) {
    return 1;
  }
  try {
    *out = cugraph::detail::effective_device_memory(*handle->raft_handle);
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" cugraph_resource_handle_t *cugraph_rs_runtime_resource_handle_get(
    cugraph_rs_runtime_resource_handle_t *handle) {
  if (handle == nullptr) {
    return nullptr;
  }
  return handle->cugraph_handle;
}

extern "C" cugraph_rs_cuda_failure_classification_t
cugraph_rs_classify_cuda_failure_after_error(int32_t *cuda_status) {
  if (cuda_status == nullptr) {
    return CUGRAPH_RS_CUDA_FAILURE_RECOVERABLE;
  }

  // Match libcudf's fatal-CUDA confirmation policy: clear the thread's last
  // status, then prove that the same typed CUDA failure survives both a benign
  // runtime call and a device synchronization. Persistent equality indicates
  // that the current device/context is no longer safe for later work.
  auto const reported = cudaGetLastError();
  auto const probe = cudaFree(nullptr);
  if (probe != cudaSuccess) {
    auto const synchronized = cudaDeviceSynchronize();
    *cuda_status = static_cast<int32_t>(probe);
    if (probe == synchronized) {
      return CUGRAPH_RS_CUDA_FAILURE_FATAL;
    }
    return CUGRAPH_RS_CUDA_FAILURE_RECOVERABLE;
  }

  *cuda_status = static_cast<int32_t>(reported);
  return reported == cudaSuccess ? CUGRAPH_RS_CUDA_FAILURE_NONE
                                 : CUGRAPH_RS_CUDA_FAILURE_RECOVERABLE;
}

extern "C" cugraph_rs_destroy_status_t
cugraph_rs_destroy_device_resource(
    void *resource, cugraph_rs_device_resource_kind_t kind,
    int32_t *cuda_status) {
  if (resource == nullptr || cuda_status == nullptr) {
    return CUGRAPH_RS_DESTROY_UNKNOWN_ERROR;
  }
  *cuda_status = static_cast<int32_t>(cudaSuccess);

  if (g_next_destroy_status != CUGRAPH_RS_DESTROY_SUCCESS) {
    auto const injected_status = g_next_destroy_status;
    *cuda_status = g_next_destroy_cuda_status;
    g_next_destroy_status = CUGRAPH_RS_DESTROY_SUCCESS;
    g_next_destroy_cuda_status = 0;
    return injected_status;
  }

  try {
    switch (kind) {
    case CUGRAPH_RS_DEVICE_RESOURCE_SIMILARITY_RESULT:
      cugraph_similarity_result_free(
          static_cast<cugraph_similarity_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_VERTEX_PAIRS:
      cugraph_vertex_pairs_free(static_cast<cugraph_vertex_pairs_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_RNG_STATE:
      cugraph_rng_state_free(static_cast<cugraph_rng_state_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_CORE_RESULT:
      cugraph_core_result_free(static_cast<cugraph_core_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_K_CORE_RESULT:
      cugraph_k_core_result_free(static_cast<cugraph_k_core_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_DEGREES_RESULT:
      cugraph_degrees_result_free(
          static_cast<cugraph_degrees_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_INDUCED_SUBGRAPH_RESULT:
      cugraph_induced_subgraph_result_free(
          static_cast<cugraph_induced_subgraph_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_LAYOUT_RESULT:
      cugraph_layout_result_free(static_cast<cugraph_layout_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_HIERARCHICAL_CLUSTERING_RESULT:
      cugraph_hierarchical_clustering_result_free(
          static_cast<cugraph_hierarchical_clustering_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_TRIANGLE_COUNT_RESULT:
      cugraph_triangle_count_result_free(
          static_cast<cugraph_triangle_count_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_RANDOM_WALK_RESULT:
      cugraph_random_walk_result_free(
          static_cast<cugraph_random_walk_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_SAMPLE_RESULT:
      cugraph_sample_result_free(static_cast<cugraph_sample_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_CLUSTERING_RESULT:
      cugraph_clustering_result_free(
          static_cast<cugraph_clustering_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_EDGELIST:
      cugraph_edgelist_free(static_cast<cugraph_edgelist_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_GRAPH:
      cugraph_graph_free(static_cast<cugraph_graph_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_BFS_PREDICATE_RESULT:
      cugraph_bfs_predicate_result_free(
          static_cast<cugraph_bfs_predicate_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_PATHS_RESULT:
      cugraph_paths_result_free(static_cast<cugraph_paths_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_EXTRACT_PATHS_RESULT:
      cugraph_extract_paths_result_free(
          static_cast<cugraph_extract_paths_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_RUNTIME_RESOURCE_HANDLE:
      static_cast<cugraph_rs_runtime_resource_handle_t *>(resource)->release();
      delete static_cast<cugraph_rs_runtime_resource_handle_t *>(resource);
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_RESOURCE_HANDLE:
      cugraph_free_resource_handle(
          static_cast<cugraph_resource_handle_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_CENTRALITY_RESULT:
      cugraph_centrality_result_free(
          static_cast<cugraph_centrality_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_EDGE_CENTRALITY_RESULT:
      cugraph_edge_centrality_result_free(
          static_cast<cugraph_edge_centrality_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_HITS_RESULT:
      cugraph_hits_result_free(static_cast<cugraph_hits_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_LABELING_RESULT:
      cugraph_labeling_result_free(
          static_cast<cugraph_labeling_result_t *>(resource));
      break;
    case CUGRAPH_RS_DEVICE_RESOURCE_DEVICE_ARRAY:
      cugraph_type_erased_device_array_free(
          static_cast<cugraph_type_erased_device_array_t *>(resource));
      break;
    default:
      return CUGRAPH_RS_DESTROY_INVALID_RESOURCE_KIND;
    }
  } catch (...) {
    auto const status = typed_cuda_destroy_status(cuda_status);
    return status == CUGRAPH_RS_DESTROY_SUCCESS
               ? CUGRAPH_RS_DESTROY_UNKNOWN_ERROR
               : status;
  }

  return typed_cuda_destroy_status(cuda_status);
}

extern "C" void cugraph_rs_test_fail_next_device_resource_destroy(
    cugraph_rs_destroy_status_t status, int32_t cuda_status) {
  g_next_destroy_status = status;
  g_next_destroy_cuda_status = cuda_status;
}
