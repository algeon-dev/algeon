#include "runtime_resource_handle.h"

// cudf-rs passes its runtime memory resource through a project-local compat
// base. Keep memory_resource_compat.hpp byte-identical to cudf-rs, then wrap the
// borrowed pointer as the RAFT 26.06 device_resource expected by handle_t.
#include "memory_resource_compat.hpp"

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <new>

struct cugraph_rs_runtime_resource_handle {
  std::unique_ptr<raft::handle_t> raft_handle;
  cugraph_resource_handle_t *cugraph_handle{nullptr};

  cugraph_rs_runtime_resource_handle(void *cuda_stream,
                                     void *device_memory_resource) {
    auto stream =
        rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(cuda_stream)};
    auto *mr = reinterpret_cast<rmm::mr::device_memory_resource *>(
        device_memory_resource);
    // Borrow the cudf-owned resource; this wrapper does not take ownership.
    raft_handle = std::make_unique<raft::handle_t>(
        stream, std::shared_ptr<rmm::cuda_stream_pool>{},
        raft::mr::device_resource{rmm::mr::any_device_resource_from_raw(mr)});
    cugraph_handle = cugraph_create_resource_handle(raft_handle.get());
    if (cugraph_handle == nullptr) {
      throw std::bad_alloc{};
    }
  }

  ~cugraph_rs_runtime_resource_handle() noexcept {
    if (cugraph_handle != nullptr) {
      cugraph_free_resource_handle(cugraph_handle);
      cugraph_handle = nullptr;
    }
  }
};

extern "C" cugraph_rs_runtime_resource_handle_t *
cugraph_rs_runtime_resource_handle_create(void *cuda_stream,
                                          void *device_memory_resource) {
  if ((cuda_stream == nullptr) || (device_memory_resource == nullptr)) {
    return nullptr;
  }

  try {
    return new cugraph_rs_runtime_resource_handle(cuda_stream,
                                                  device_memory_resource);
  } catch (...) {
    return nullptr;
  }
}

extern "C" cugraph_resource_handle_t *cugraph_rs_runtime_resource_handle_get(
    cugraph_rs_runtime_resource_handle_t *handle) {
  if (handle == nullptr) {
    return nullptr;
  }
  return handle->cugraph_handle;
}

extern "C" void cugraph_rs_runtime_resource_handle_destroy(
    cugraph_rs_runtime_resource_handle_t *handle) {
  delete handle;
}
