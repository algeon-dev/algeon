#include "runtime_resource_handle.h"

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <new>

namespace {

struct unmanaged_mr_deleter {
  void operator()(rmm::mr::device_memory_resource *) const noexcept {}
};

} // namespace

struct cugraph_rs_runtime_resource_handle {
  std::unique_ptr<raft::handle_t> raft_handle;
  cugraph_resource_handle_t *cugraph_handle{nullptr};

  cugraph_rs_runtime_resource_handle(void *cuda_stream,
                                     void *device_memory_resource) {
    auto stream =
        rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(cuda_stream)};
    auto *mr = reinterpret_cast<rmm::mr::device_memory_resource *>(
        device_memory_resource);
    auto mr_ref = std::shared_ptr<rmm::mr::device_memory_resource>(
        mr, unmanaged_mr_deleter{});

    raft_handle = std::make_unique<raft::handle_t>(
        stream, std::shared_ptr<rmm::cuda_stream_pool>{}, std::move(mr_ref));
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
