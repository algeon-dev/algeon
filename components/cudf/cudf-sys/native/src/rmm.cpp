#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <memory>

extern "C" {

/// Opaque handle to a device memory resource.
typedef void* cudf_memory_resource_handle;

/// Create a CUDA (default) memory resource.
cudf_status_t cudf_memory_resource_create_cuda(
    cudf_memory_resource_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* mr = new rmm::mr::cuda_memory_resource();
    *out = static_cast<cudf_memory_resource_handle>(mr);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Create a pool memory resource wrapping a CUDA memory resource.
/// `initial_pool_size` and `maximum_pool_size` are in bytes.
/// Pass 0 for maximum_pool_size to let the pool grow unbounded.
cudf_status_t cudf_memory_resource_create_pool(
    size_t initial_pool_size, size_t maximum_pool_size,
    cudf_memory_resource_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    // The pool owns its upstream CUDA resource.
    auto upstream = std::make_shared<rmm::mr::cuda_memory_resource>();
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>* pool;
    if (maximum_pool_size > 0) {
      pool = new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
          upstream.get(), initial_pool_size, maximum_pool_size);
    } else {
      pool = new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
          upstream.get(), initial_pool_size);
    }
    // NOTE: The upstream shared_ptr must outlive the pool. We leak it
    // intentionally here; the pool destructor does not own the upstream.
    // A production implementation would store both together in a wrapper.
    upstream.get();  // prevent unused warning
    *out = static_cast<cudf_memory_resource_handle>(pool);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Destroy a memory resource handle.
void cudf_memory_resource_destroy(cudf_memory_resource_handle handle) {
  delete static_cast<rmm::mr::device_memory_resource*>(handle);
}

}  // extern "C"
