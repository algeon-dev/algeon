#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory_resource.h"
#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
struct memory_resource_owner {
  std::shared_ptr<rmm::mr::cuda_memory_resource> upstream;
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource>* stats = nullptr;
};

struct memory_resource_handle_state {
  std::shared_ptr<memory_resource_owner> owner;
};

std::mutex& memory_resource_registry_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<rmm::mr::device_memory_resource*, std::weak_ptr<void>>&
memory_resource_registry()
{
  static std::unordered_map<rmm::mr::device_memory_resource*, std::weak_ptr<void>> registry;
  return registry;
}

memory_resource_handle_state& memory_resource_state_from_handle(cudf_memory_resource_handle handle)
{
  return *static_cast<memory_resource_handle_state*>(handle);
}

memory_resource_owner& memory_resource_owner_from_handle(cudf_memory_resource_handle handle)
{
  return *memory_resource_state_from_handle(handle).owner;
}

void register_memory_resource_lifetime(std::shared_ptr<memory_resource_owner> const& owner)
{
  std::scoped_lock lock{memory_resource_registry_mutex()};
  memory_resource_registry()[owner->mr.get()] = owner;
}
}  // namespace

rmm::device_async_resource_ref cudf_memory_resource_ref_from_handle(
  cudf_memory_resource_handle handle)
{
  return rmm::device_async_resource_ref{*cudf_memory_resource_raw_from_handle(handle)};
}

rmm::mr::device_memory_resource* cudf_memory_resource_raw_from_handle(
  cudf_memory_resource_handle handle)
{
  return memory_resource_owner_from_handle(handle).mr.get();
}

std::shared_ptr<void> cudf_memory_resource_lifetime_from_handle(cudf_memory_resource_handle handle)
{
  return memory_resource_state_from_handle(handle).owner;
}

std::shared_ptr<void> cudf_memory_resource_lifetime_from_raw(
  rmm::mr::device_memory_resource* raw_mr)
{
  std::scoped_lock lock{memory_resource_registry_mutex()};
  auto& registry = memory_resource_registry();
  auto const it  = registry.find(raw_mr);
  if (it == registry.end()) { return {}; }

  auto lifetime = it->second.lock();
  if (lifetime == nullptr) { registry.erase(it); }
  return lifetime;
}

extern "C" {
/// Create a CUDA (default) memory resource.
cudf_status_t cudf_memory_resource_create_cuda(cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto state       = std::make_unique<memory_resource_handle_state>();
    state->owner     = std::make_shared<memory_resource_owner>();
    state->owner->mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Create a pool memory resource wrapping a CUDA memory resource.
/// `initial_pool_size` and `maximum_pool_size` are in bytes.
/// Pass 0 for maximum_pool_size to let the pool grow unbounded.
cudf_status_t cudf_memory_resource_create_pool(size_t initial_pool_size,
                                               size_t maximum_pool_size,
                                               cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto state     = std::make_unique<memory_resource_handle_state>();
    state->owner   = std::make_shared<memory_resource_owner>();
    auto& owner    = *state->owner;
    owner.upstream = std::make_shared<rmm::mr::cuda_memory_resource>();
    if (maximum_pool_size > 0) {
      owner.mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
        owner.upstream.get(), initial_pool_size, maximum_pool_size);

    } else {
      owner.mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
        owner.upstream.get(), initial_pool_size);
    }
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Destroy a memory resource handle.
cudf_status_t cudf_memory_resource_destroy(cudf_memory_resource_handle handle, int32_t device)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    cudf_device_guard guard;
    cudf_status_t const status = guard.activate(device);
    if (status != CUDF_SUCCESS) { return status; }
    std::scoped_lock lock{cudf_current_device_resource_mutex()};
    auto& owner = memory_resource_owner_from_handle(handle);
    if (rmm::mr::get_current_device_resource() == owner.mr.get()) {
      set_error(
        "cannot destroy a memory resource while it is installed as the current device resource");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    delete static_cast<memory_resource_handle_state*>(handle);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_raw(cudf_memory_resource_handle handle, void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    *out = memory_resource_owner_from_handle(handle).mr.get();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_memory_resource_create_stats(cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto state     = std::make_unique<memory_resource_handle_state>();
    state->owner   = std::make_shared<memory_resource_owner>();
    auto& owner    = *state->owner;
    owner.upstream = std::make_shared<rmm::mr::cuda_memory_resource>();
    auto stats =
      std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource>>(
        owner.upstream.get());
    owner.stats = stats.get();
    owner.mr    = std::move(stats);
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_memory_resource_allocation_count(cudf_memory_resource_handle handle,
                                                         int64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    if (owner.stats == nullptr) {
      set_error("memory resource does not expose allocation statistics");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = owner.stats->get_allocations_counter().total;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_memory_resource_active_allocation_count(cudf_memory_resource_handle handle,
                                                                int64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    if (owner.stats == nullptr) {
      set_error("memory resource does not expose allocation statistics");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = owner.stats->get_allocations_counter().value;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_memory_resource_raw(cudf_memory_resource_handle handle, void** out)
{
  return cudf_memory_resource_raw(handle, out);
}

cudf_status_t cudf_test_memory_resource_set_current(cudf_memory_resource_handle handle)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    std::scoped_lock lock{cudf_current_device_resource_mutex()};
    rmm::mr::set_current_device_resource(owner.mr.get());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_memory_resource_set_current_raw(void* raw_mr)
{
  CUDF_RUST_NULL_CHECK(raw_mr, "raw_mr is null");
  CUDF_RUST_TRY
  {
    std::scoped_lock lock{cudf_current_device_resource_mutex()};
    rmm::mr::set_current_device_resource(static_cast<rmm::mr::device_memory_resource*>(raw_mr));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
