#include "memory_resource.h"
#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>

namespace {

struct memory_resource_owner {
  std::shared_ptr<rmm::mr::cuda_memory_resource> upstream;
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource>* stats =
      nullptr;
};

}  // namespace

rmm::device_async_resource_ref cudf_memory_resource_ref_from_handle(
    cudf_memory_resource_handle handle) {
  return rmm::device_async_resource_ref{
      *cudf_memory_resource_raw_from_handle(handle)};
}

rmm::mr::device_memory_resource* cudf_memory_resource_raw_from_handle(
    cudf_memory_resource_handle handle) {
  return static_cast<memory_resource_owner*>(handle)->mr.get();
}

extern "C" {

/// Create a CUDA (default) memory resource.
cudf_status_t cudf_memory_resource_create_cuda(
    cudf_memory_resource_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto owner = std::make_unique<memory_resource_owner>();
    owner->mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    *out = static_cast<cudf_memory_resource_handle>(owner.release());
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
  *out = nullptr;
  CUDF_RUST_TRY {
    auto owner = std::make_unique<memory_resource_owner>();
    owner->upstream = std::make_shared<rmm::mr::cuda_memory_resource>();
    if (maximum_pool_size > 0) {
      owner->mr =
          std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
              owner->upstream.get(), initial_pool_size, maximum_pool_size);
    } else {
      owner->mr =
          std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
              owner->upstream.get(), initial_pool_size);
    }
    *out = static_cast<cudf_memory_resource_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Destroy a memory resource handle.
void cudf_memory_resource_destroy(cudf_memory_resource_handle handle) {
  delete static_cast<memory_resource_owner*>(handle);
}

cudf_status_t cudf_test_memory_resource_create_stats(
    cudf_memory_resource_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto owner = std::make_unique<memory_resource_owner>();
    owner->upstream = std::make_shared<rmm::mr::cuda_memory_resource>();
    auto stats =
        std::make_unique<rmm::mr::statistics_resource_adaptor<
            rmm::mr::cuda_memory_resource>>(owner->upstream.get());
    owner->stats = stats.get();
    owner->mr = std::move(stats);
    *out = static_cast<cudf_memory_resource_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_memory_resource_allocation_count(
    cudf_memory_resource_handle handle, int64_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* owner = static_cast<memory_resource_owner*>(handle);
    if (owner->stats == nullptr) {
      set_error("memory resource does not expose allocation statistics");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = owner->stats->get_allocations_counter().total;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
