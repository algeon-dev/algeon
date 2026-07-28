#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory/internal.hpp"
#include "rmm_interop_bridge.hpp"
#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <new>
#include <stdexcept>
#include <string>

namespace cudf::rust::memory {

cudf_memory_manager_options_t default_memory_manager_options()
{
  cudf_memory_manager_options_t options{};
  options.struct_size             = sizeof(cudf_memory_manager_options_t);
  options.base_kind               = CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC;
  options.flags                   = 0;
  options.initial_pool_size_bytes = 0;
  options.maximum_pool_size_bytes = 0;
  options.release_threshold_bytes = k_default_async_release_threshold;
  return options;
}

std::unordered_map<int32_t, std::shared_ptr<rmm::mr::device_memory_resource>>&
default_memory_resources()
{
  static std::unordered_map<int32_t, std::shared_ptr<rmm::mr::device_memory_resource>> resources;
  return resources;
}

std::unordered_map<int32_t, rmm::mr::device_memory_resource*>& current_memory_resource_identities()
{
  static std::unordered_map<int32_t, rmm::mr::device_memory_resource*> identities;
  return identities;
}

// Protects only the identity/default maps above. The critical section is a
// map lookup or insert (tens of nanoseconds) and must never span a libcudf
// call. rmm's own set_current_device_resource carries per-device locking.
std::mutex& current_memory_resource_identity_mutex()
{
  static std::mutex mutex;
  return mutex;
}

// Caller must hold current_memory_resource_identity_mutex().
rmm::mr::device_memory_resource* set_current_device_resource_raw_locked(
  int32_t device, rmm::mr::device_memory_resource* target)
{
  auto& identities        = current_memory_resource_identities();
  auto const existing     = identities.find(device);
  bool const had_existing = existing != identities.end();
  auto* previous          = had_existing ? existing->second : nullptr;
  if (had_existing) {
    existing->second = target;
  } else {
    identities.emplace(device, target);
  }
  try {
    cudf_rust_rmm_set_current_device_resource(target);
  } catch (...) {
    if (had_existing) {
      identities.find(device)->second = previous;
    } else {
      identities.erase(device);
    }
    throw;
  }
  return target;
}

int32_t current_cuda_device()
{
  int32_t device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  return device;
}

rmm::mr::device_memory_resource* default_memory_resource_for_device(int32_t device)
{
  auto& resources = default_memory_resources();
  auto it         = resources.find(device);
  if (it == resources.end()) {
    it = resources
           .emplace(
             device,
             std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>())
           .first;
  }
  return it->second.get();
}

std::mutex& memory_manager_registry_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<int32_t, std::shared_ptr<device_memory_manager_state>>& memory_manager_registry()
{
  static std::unordered_map<int32_t, std::shared_ptr<device_memory_manager_state>> registry;
  return registry;
}

std::shared_ptr<device_memory_manager_state> memory_manager_from_handle(
  cudf_memory_manager_handle handle)
{
  std::scoped_lock lock{memory_manager_registry_mutex()};
  for (auto const& [_, manager] : memory_manager_registry()) {
    if (reinterpret_cast<cudf_memory_manager_handle>(manager.get()) == handle) { return manager; }
  }
  return {};
}

cudf_status_t normalize_memory_manager_options(cudf_memory_manager_options_t const* options,
                                               cudf_memory_manager_options_t& out)
{
  out = default_memory_manager_options();
  if (options == nullptr) { return CUDF_SUCCESS; }
  if (options->struct_size != sizeof(cudf_memory_manager_options_t)) {
    set_error("memory manager options struct_size is invalid");
    return CUDF_ERROR_INVALID_ARGUMENT;
  }
  if (options->flags != 0) {
    set_error("memory manager options flags must be zero");
    return CUDF_ERROR_INVALID_ARGUMENT;
  }
  switch (options->base_kind) {
    case CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC:
    case CUDF_MEMORY_MANAGER_BASE_POOL: out = *options; return CUDF_SUCCESS;
    default:
      set_error("memory manager options base_kind is invalid");
      return CUDF_ERROR_INVALID_ARGUMENT;
  }
}

bool memory_manager_options_equal(cudf_memory_manager_options_t const& left,
                                  cudf_memory_manager_options_t const& right)
{
  if (left.base_kind != right.base_kind) { return false; }
  if (left.base_kind == CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC) {
    return left.initial_pool_size_bytes == right.initial_pool_size_bytes &&
           left.release_threshold_bytes == right.release_threshold_bytes;
  }
  return left.initial_pool_size_bytes == right.initial_pool_size_bytes &&
         left.maximum_pool_size_bytes == right.maximum_pool_size_bytes;
}

std::shared_ptr<device_memory_manager_state> make_memory_manager_for_current_device(
  int32_t device,
  cudf_memory_failure_callback_t failure_callback,
  cudf_memory_failure_callback_v2_t failure_callback_v2,
  void* failure_callback_arg,
  cudf_memory_manager_options_t const& options)
{
  auto manager                  = std::make_shared<device_memory_manager_state>();
  manager->device               = device;
  manager->options              = options;
  manager->failure_callback     = failure_callback;
  manager->failure_callback_v2  = failure_callback_v2;
  manager->failure_callback_arg = failure_callback_arg;
  manager->failure_callback_installed =
    failure_callback != nullptr || failure_callback_v2 != nullptr;
  manager->owner = std::make_shared<memory_resource_owner>();

  auto& owner  = *manager->owner;
  owner.device = device;
  if (options.base_kind == CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC) {
    auto async_pool  = make_shared_cuda_async_resource_box(options.initial_pool_size_bytes,
                                                          options.release_threshold_bytes);
    owner.async_pool = async_pool.get();
    owner.pool       = std::move(async_pool);
  } else {
    owner.upstream =
      std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>();
    auto upstream = rmm::mr::any_device_resource_from_raw(owner.upstream.get());
    if (options.maximum_pool_size_bytes > 0) {
      owner.pool =
        std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::pool_memory_resource>>(
          std::move(upstream),
          static_cast<std::size_t>(options.initial_pool_size_bytes),
          static_cast<std::size_t>(options.maximum_pool_size_bytes));
    } else {
      owner.pool =
        std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::pool_memory_resource>>(
          std::move(upstream), static_cast<std::size_t>(options.initial_pool_size_bytes));
    }
  }
  auto adaptor = std::make_unique<reservation_aware_resource_adaptor>(owner.pool.get());
  owner.reservation_adaptor = adaptor.get();
  if (failure_callback_v2 != nullptr) {
    owner.reservation_mr = std::move(adaptor);
    owner.mr             = make_bounded_failure_callback_resource(owner.reservation_mr.get(),
                                                      failure_callback_v2,
                                                      failure_callback_arg,
                                                      async_pool_handle(owner.async_pool),
                                                      device);
  } else if (failure_callback != nullptr) {
    owner.reservation_mr = std::move(adaptor);
    owner.mr             = make_bounded_failure_callback_resource(owner.reservation_mr.get(),
                                                      failure_callback,
                                                      failure_callback_arg,
                                                      async_pool_handle(owner.async_pool),
                                                      device);
  } else {
    owner.mr = std::move(adaptor);
  }
  register_memory_resource_lifetime(manager->owner);

  return manager;
}

cudf_status_t validate_memory_manager_device(int32_t device)
{
  if (device < 0) {
    set_error("device must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  int32_t device_count           = 0;
  cudaError_t const count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess) {
    auto msg = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(count_status);
    set_error(msg);
    return CUDF_ERROR_CUDA;
  }
  if (device >= device_count) {
    set_error("device is outside the available CUDA device range");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  return CUDF_SUCCESS;
}

cudf_status_t validate_stream_device_for_manager(cudf_stream_handle stream,
                                                 int32_t manager_device,
                                                 char const* error_message)
{
  if (cudf_stream_device_from_handle(stream) != manager_device) {
    set_error(error_message);
    return CUDF_ERROR_INVALID_OPERATION;
  }
  return CUDF_SUCCESS;
}

cudf_status_t memory_manager_for_device_impl(int32_t device,
                                             cudf_memory_failure_callback_t failure_callback,
                                             cudf_memory_failure_callback_v2_t failure_callback_v2,
                                             void* failure_callback_arg,
                                             cudf_memory_manager_options_t const* options,
                                             cudf_memory_manager_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    cudf_status_t const validation_status = validate_memory_manager_device(device);
    if (validation_status != CUDF_SUCCESS) { return validation_status; }
    if (failure_callback == nullptr && failure_callback_v2 == nullptr &&
        failure_callback_arg != nullptr) {
      set_error("failure callback argument requires a failure callback");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    cudf_memory_manager_options_t resolved_options{};
    cudf_status_t const options_status =
      normalize_memory_manager_options(options, resolved_options);
    if (options_status != CUDF_SUCCESS) { return options_status; }

    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }

    std::scoped_lock lock{memory_manager_registry_mutex()};
    auto& registry = memory_manager_registry();
    auto it        = registry.find(device);
    if (it == registry.end()) {
      it = registry
             .emplace(device,
                      make_memory_manager_for_current_device(device,
                                                             failure_callback,
                                                             failure_callback_v2,
                                                             failure_callback_arg,
                                                             resolved_options))
             .first;
    } else if (options != nullptr &&
               !memory_manager_options_equal(it->second->options, resolved_options)) {
      set_error(
        "device memory manager already exists with different physical-base "
        "options; "
        "the process-global allocator cannot be replaced after first use");
      return CUDF_ERROR_INVALID_OPERATION;
    } else if ((failure_callback != nullptr || failure_callback_v2 != nullptr) &&
               !it->second->failure_callback_installed) {
      set_error(
        "device memory manager already exists without a failure callback; "
        "install the callback before first use");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (!it->second->current_resource_installed) {
      // Device-domain invariant: this is the one production install point. The
      // adaptor stays the device's current resource for the process lifetime.
      // The manager is already registry-owned before the current resource is
      // switched, so any exception during the switch cannot leave RMM pointing
      // at freed memory.
      try {
        cudf_set_current_device_resource_raw(it->second->owner->mr.get());
      } catch (...) {
        set_error(
          "device memory manager current resource installation failed; "
          "manager remains registered");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      it->second->current_resource_installed = true;
    }
    *out = reinterpret_cast<cudf_memory_manager_handle>(it->second.get());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // namespace cudf::rust::memory

using namespace cudf::rust::memory;

namespace rmm {
const char* bad_alloc::what() const noexcept { return _what.c_str(); }
}  // namespace rmm

rmm::mr::device_memory_resource* cudf_tracked_current_device_resource_raw()
{
  auto const device = current_cuda_device();
  std::scoped_lock lock{current_memory_resource_identity_mutex()};
  auto& identities = current_memory_resource_identities();
  auto const found = identities.find(device);
  if (found != identities.end()) { return found->second; }

  return set_current_device_resource_raw_locked(device, default_memory_resource_for_device(device));
}

void cudf_set_current_device_resource_raw(rmm::mr::device_memory_resource* target)
{
  auto const device = current_cuda_device();
  std::scoped_lock lock{current_memory_resource_identity_mutex()};
  set_current_device_resource_raw_locked(device, target);
}

extern "C" {
cudf_status_t cudf_memory_manager_for_device(int32_t device, cudf_memory_manager_handle* out)
{
  return memory_manager_for_device_impl(device, nullptr, nullptr, nullptr, nullptr, out);
}

cudf_status_t cudf_memory_manager_for_device_with_options(
  int32_t device, cudf_memory_manager_options_t const* options, cudf_memory_manager_handle* out)
{
  CUDF_RUST_NULL_CHECK(options, "options is null");
  return memory_manager_for_device_impl(device, nullptr, nullptr, nullptr, options, out);
}

cudf_status_t cudf_memory_manager_for_device_with_failure_callback(
  int32_t device,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(callback, "callback is null");
  return memory_manager_for_device_impl(device, callback, nullptr, callback_arg, nullptr, out);
}

cudf_status_t cudf_memory_manager_for_device_with_failure_callback_v2(
  int32_t device,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudf_memory_manager_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(callback, "callback is null");
  return memory_manager_for_device_impl(device, nullptr, callback, callback_arg, nullptr, out);
}

cudf_status_t cudf_memory_manager_for_device_with_failure_callback_and_options(
  int32_t device,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(callback, "callback is null");
  CUDF_RUST_NULL_CHECK(options, "options is null");
  return memory_manager_for_device_impl(device, callback, nullptr, callback_arg, options, out);
}

cudf_status_t cudf_memory_manager_for_device_with_failure_callback_v2_and_options(
  int32_t device,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(callback, "callback is null");
  CUDF_RUST_NULL_CHECK(options, "options is null");
  return memory_manager_for_device_impl(device, nullptr, callback, callback_arg, options, out);
}

cudf_status_t cudf_memory_manager_failure_callback_state(int32_t device,
                                                         int32_t* exists,
                                                         int32_t* failure_callback_installed)
{
  CUDF_RUST_NULL_CHECK(exists, "exists is null");
  *exists = 0;
  CUDF_RUST_NULL_CHECK(failure_callback_installed, "failure_callback_installed is null");
  *failure_callback_installed = 0;
  CUDF_RUST_TRY
  {
    cudf_status_t const validation_status = validate_memory_manager_device(device);
    if (validation_status != CUDF_SUCCESS) { return validation_status; }
    std::scoped_lock lock{memory_manager_registry_mutex()};
    auto const it = memory_manager_registry().find(device);
    if (it == memory_manager_registry().end()) { return CUDF_SUCCESS; }
    *exists                     = 1;
    *failure_callback_installed = it->second->failure_callback_installed ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_options_for_device(int32_t device,
                                                     int32_t* exists,
                                                     cudf_memory_manager_options_t* out)
{
  CUDF_RUST_NULL_CHECK(exists, "exists is null");
  *exists = 0;
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = default_memory_manager_options();
  CUDF_RUST_TRY
  {
    cudf_status_t const validation_status = validate_memory_manager_device(device);
    if (validation_status != CUDF_SUCCESS) { return validation_status; }
    std::scoped_lock lock{memory_manager_registry_mutex()};
    auto const it = memory_manager_registry().find(device);
    if (it == memory_manager_registry().end()) { return CUDF_SUCCESS; }
    *exists = 1;
    *out    = it->second->options;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_resource(cudf_memory_manager_handle manager,
                                           cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager handle is not registered");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state              = std::make_unique<memory_resource_handle_state>();
    state->owner            = manager_state->owner;
    state->non_owning_alias = true;
    *out                    = reinterpret_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_pool_usage(cudf_memory_manager_handle manager,
                                             cudf_memory_pool_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager handle is not registered");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (manager_state->owner->async_pool == nullptr) {
      set_error("memory manager is not backed by a CUDA async pool");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    read_async_pool_usage(manager_state->owner->async_pool, *out);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_trim_to(cudf_memory_manager_handle manager, uint64_t keep_bytes)
{
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager handle is not registered");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (manager_state->owner->async_pool == nullptr) {
      set_error("memory manager is not backed by a CUDA async pool");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    trim_async_pool_to(manager_state->owner->async_pool, keep_bytes);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
