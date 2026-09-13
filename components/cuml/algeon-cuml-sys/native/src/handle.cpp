#include "cuml_rust/handle.h"
#include "shim_common.hpp"
#include "memory_resource_compat.hpp"

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <new>
#include <stdexcept>

extern "C" int32_t cudf_rust_rmm_current_device_resource_is(void* resource);
cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;

struct cuml_handle {
  std::unique_ptr<raft::handle_t> raft;
  rmm::mr::device_memory_resource* mr;
  cuml_handle(int32_t device, void* stream, void* memory_resource, void* memory_resource_owner)
  {
    auto const device_status = cudaSetDevice(device);
    if (device_status != cudaSuccess) {
      throw rmm::cuda_error(cudaGetErrorString(device_status));
    }
    mr = reinterpret_cast<rmm::mr::device_memory_resource*>(memory_resource);
    (void)memory_resource_owner;
    if (!cudf_rust_rmm_current_device_resource_is(mr)) {
      throw std::logic_error("cuML handle memory resource is not the current RMM resource");
    }
    auto cuda_stream = rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(stream)};
    raft = std::make_unique<raft::handle_t>(cuda_stream,
      std::shared_ptr<rmm::cuda_stream_pool>{},
      raft::mr::device_resource{rmm::mr::any_device_resource_from_raw(mr)});
  }
};

rmm::mr::device_memory_resource* cuml_handle_mr(cuml_handle_t handle)
{
  if (handle == nullptr) throw std::invalid_argument("cuML handle is null");
  return handle->mr;
}

raft::handle_t& cuml_handle_raft(cuml_handle_t handle)
{
  if (handle == nullptr || handle->raft == nullptr) {
    throw std::invalid_argument("cuML handle is null");
  }
  return *handle->raft;
}

raft::handle_t* cuml_internal_raft_handle(cuml_handle_t handle) noexcept
{
  return handle == nullptr ? nullptr : handle->raft.get();
}

extern "C" cuml_status_t cuml_handle_create(int32_t device, void* cuda_stream,
    void* device_memory_resource, void* memory_resource_owner, cuml_handle_t* out_handle)
{
  cuml_clear_error();
  if (out_handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "handle output is null");
  *out_handle = nullptr;
  if (cuda_stream == nullptr || device_memory_resource == nullptr || memory_resource_owner == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "handle creation argument is null");
  }
  CUML_RUST_TRY {
    *out_handle = new cuml_handle(device, cuda_stream, device_memory_resource, memory_resource_owner);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_handle_destroy(cuml_handle_t handle)
{
  cuml_clear_error();
  if (handle == nullptr) return CUML_SUCCESS;
  try {
    delete handle;
    return CUML_SUCCESS;
  } catch (...) {
    return cuml_translate_cleanup_exception_noexcept();
  }
}
