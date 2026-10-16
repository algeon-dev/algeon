#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory/internal.hpp"
#include "rmm_interop_bridge.hpp"
#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <atomic>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace cudf::rust::memory {

namespace {

/// Enforces the manager-wide allocation limit through the same accounting
/// state used by private query-domain pools.
class manager_allocation_limit_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  manager_allocation_limit_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                            std::shared_ptr<manager_allocation_limit_state> state)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)}, state_{std::move(state)}
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    if (bytes == 0) { return upstream_.allocate(stream, bytes, alignment); }
    auto const aligned_bytes = cudf::rust::reservation::allocation_charge(bytes, alignment);
    if (!state_->try_reserve(aligned_bytes)) {
      throw rmm::out_of_memory{"memory manager allocation limit exceeded"};
    }
    try {
      return upstream_.allocate(stream, bytes, alignment);
    } catch (...) {
      state_->release(aligned_bytes);
      throw;
    }
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    if (bytes != 0) {
      state_->release(cudf::rust::reservation::allocation_charge(bytes, alignment));
    }
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::shared_ptr<manager_allocation_limit_state> state_;
};

}  // namespace

#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
std::atomic<bool>& fail_next_current_resource_install()
{
  static std::atomic<bool> value{false};
  return value;
}

std::atomic<bool>& invoke_failure_callback_before_install_returns()
{
  static std::atomic<bool> value{false};
  return value;
}
#endif

cudf_memory_manager_options_t default_memory_manager_options()
{
  cudf_memory_manager_options_t options{};
  options.struct_size             = sizeof(cudf_memory_manager_options_t);
  options.base_kind               = CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC;
  options.flags                   = 0;
  options.initial_pool_size_bytes = 0;
  options.maximum_pool_size_bytes = 0;
  options.release_threshold_bytes = k_default_async_release_threshold;
  options.allocation_limit_bytes  = 0;
  return options;
}

std::unordered_map<int32_t, std::shared_ptr<rmm::mr::device_memory_resource>>&
default_memory_resources()
{
  // Deliberately leaked: device-resource destruction can still run while a
  // process unwinds worker threads at exit; these registries must outlive
  // static destruction.
  static auto& resources =
    *new std::unordered_map<int32_t, std::shared_ptr<rmm::mr::device_memory_resource>>;
  return resources;
}

std::unordered_map<int32_t, rmm::mr::device_memory_resource*>& current_memory_resource_identities()
{
  static auto& identities = *new std::unordered_map<int32_t, rmm::mr::device_memory_resource*>;
  return identities;
}

// Protects only the identity/default maps above. The critical section is a
// map lookup or insert (tens of nanoseconds) and must never span a libcudf
// call. rmm's own set_current_device_resource carries per-device locking.
std::mutex& current_memory_resource_identity_mutex()
{
  static auto& mutex = *new std::mutex;
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
  static auto& mutex = *new std::mutex;
  return mutex;
}

std::unordered_map<int32_t, std::shared_ptr<device_memory_manager_state>>& memory_manager_registry()
{
  static auto& registry =
    *new std::unordered_map<int32_t, std::shared_ptr<device_memory_manager_state>>;
  return registry;
}

std::atomic<std::uint64_t>& next_memory_manager_installation_generation()
{
  static std::atomic<std::uint64_t> next{1};
  return next;
}

bool installation_identity_equal(cudf_memory_manager_installation_identity_t const& left,
                                 cudf_memory_manager_installation_identity_t const& right)
{
  return left.callback_version == right.callback_version &&
         std::memcmp(
           left.owner_fingerprint, right.owner_fingerprint, sizeof(left.owner_fingerprint)) == 0 &&
         std::memcmp(left.partition_fingerprint,
                     right.partition_fingerprint,
                     sizeof(left.partition_fingerprint)) == 0;
}

bool manager_is_actual_current_resource(device_memory_manager_state const& manager)
{
  return cudf_rust_rmm_current_device_resource_is(manager.owner->mr.get()) != 0;
}

void fill_installation_record(device_memory_manager_state* manager,
                              int32_t device,
                              cudf_memory_manager_installation_record_t& out,
                              bool created_by_call = false,
                              cudf_memory_manager_installation_cause_t requested_cause =
                                CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_NONE)
{
  out                   = {};
  out.struct_size       = sizeof(cudf_memory_manager_installation_record_t);
  out.device_ordinal    = device;
  out.effective_options = default_memory_manager_options();
  if (manager == nullptr) {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_UNINSTALLED;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_INSTALLABLE;
    return;
  }

  out.flags |= CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_EXISTS;
  if (created_by_call) { out.flags |= CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CREATED_BY_CALL; }
  out.effective_options       = manager->options;
  out.installation_generation = manager->installation_generation;
  if (manager->failure_callback_installed) {
    out.flags |= CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_FAILURE_CALLBACK_INSTALLED;
  }
  if (manager->failure_callback_panicked.load(std::memory_order_acquire)) {
    out.flags |= CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CALLBACK_PANICKED;
  }
  if (manager->has_installation_identity) {
    out.flags |= CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_HAS_IDENTITY;
    out.identity = manager->installation_identity;
  }

  bool const actual_current = manager_is_actual_current_resource(*manager);
  if (manager->current_resource_installed && !actual_current) {
    manager->current_resource_integrity_lost = true;
  }
  if (manager->current_resource_installed && actual_current) {
    out.flags |= CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CURRENT_RESOURCE_INSTALLED;
  }

  if (manager->current_resource_integrity_lost) {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_INCOMPLETE;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_PERMANENTLY_UNUSABLE;
    out.cause  = CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CURRENT_RESOURCE_REPLACED;
  } else if (manager->failure_callback_panicked.load(std::memory_order_acquire)) {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_CALLBACK_PANICKED;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_PERMANENTLY_UNUSABLE;
    out.cause  = CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_FAILURE_CALLBACK_PANICKED;
  } else if (!manager->failure_callback_installed) {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_UNIDENTIFIED;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_PERMANENTLY_UNUSABLE;
    out.cause  = CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_FAILURE_CALLBACK_MISSING;
  } else if (!manager->has_installation_identity) {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_UNIDENTIFIED;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_PERMANENTLY_UNUSABLE;
    out.cause  = CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_IDENTITY_MISSING;
  } else if (!manager->current_resource_installed && manager->current_resource_install_retryable) {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_INCOMPLETE;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_RETRYABLE;
    out.cause  = manager->current_resource_install_cause;
  } else if (!manager->current_resource_installed || !actual_current) {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_INCOMPLETE;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_PERMANENTLY_UNUSABLE;
    out.cause  = manager->current_resource_install_cause;
  } else {
    out.status = CUDF_MEMORY_MANAGER_INSTALLATION_OWNED;
    out.retry  = CUDF_MEMORY_MANAGER_INSTALLATION_REUSABLE;
  }
  // A rejected installation reports the reason this request was rejected, even
  // when a pre-existing health issue coexists with it. In particular, immutable
  // option conflicts must not be masked by an
  // allocator that was also installed without a failure callback.
  if (requested_cause != CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_NONE) {
    out.cause = requested_cause;
  }
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
      out                         = *options;
      out.maximum_pool_size_bytes = 0;
      break;
    case CUDF_MEMORY_MANAGER_BASE_POOL:
      out                         = *options;
      out.release_threshold_bytes = k_default_async_release_threshold;
      break;
    default:
      set_error("memory manager options base_kind is invalid");
      return CUDF_ERROR_INVALID_ARGUMENT;
  }
  if (out.allocation_limit_bytes > 0 && out.initial_pool_size_bytes > out.allocation_limit_bytes) {
    set_error("memory manager initial pool size exceeds allocation limit");
    return CUDF_ERROR_INVALID_ARGUMENT;
  }
  if (out.base_kind == CUDF_MEMORY_MANAGER_BASE_POOL && out.maximum_pool_size_bytes > 0) {
    if (out.maximum_pool_size_bytes < out.initial_pool_size_bytes) {
      set_error("memory manager maximum pool size is below initial pool size");
      return CUDF_ERROR_INVALID_ARGUMENT;
    }
    if (out.allocation_limit_bytes > 0 &&
        out.maximum_pool_size_bytes > out.allocation_limit_bytes) {
      set_error("memory manager maximum pool size exceeds allocation limit");
      return CUDF_ERROR_INVALID_ARGUMENT;
    }
  }
  return CUDF_SUCCESS;
}

bool memory_manager_options_equal(cudf_memory_manager_options_t const& left,
                                  cudf_memory_manager_options_t const& right)
{
  if (left.base_kind != right.base_kind ||
      left.allocation_limit_bytes != right.allocation_limit_bytes) {
    return false;
  }
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
  void* failure_callback_arg,
  cudf_memory_manager_options_t const& options,
  cudf_memory_manager_installation_identity_t const* installation_identity)
{
  auto manager                        = std::make_shared<device_memory_manager_state>();
  manager->device                     = device;
  manager->allocator_events           = std::make_unique<allocator_event_ring>();
  manager->options                    = options;
  manager->failure_callback           = failure_callback;
  manager->failure_callback_arg       = failure_callback_arg;
  manager->failure_callback_installed = failure_callback != nullptr;
  if (installation_identity != nullptr) {
    manager->has_installation_identity = true;
    manager->installation_identity     = *installation_identity;
  }
  if (manager->failure_callback_installed) {
    manager->installation_generation =
      next_memory_manager_installation_generation().fetch_add(1, std::memory_order_relaxed);
  }
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
  auto* physical_base = owner.pool.get();
  if (options.allocation_limit_bytes > 0) {
    owner.manager_allocation_limit = std::make_shared<manager_allocation_limit_state>(
      static_cast<std::size_t>(options.allocation_limit_bytes));
    owner.limiting_mr = std::make_unique<manager_allocation_limit_resource_adaptor>(
      physical_base, owner.manager_allocation_limit);
    physical_base = owner.limiting_mr.get();
  }
  auto adaptor              = std::make_unique<reservation_aware_resource_adaptor>(physical_base);
  owner.reservation_adaptor = adaptor.get();
  if (failure_callback != nullptr) {
    owner.reservation_mr = std::move(adaptor);
    owner.mr             = make_bounded_failure_callback_resource(owner.reservation_mr.get(),
                                                      failure_callback,
                                                      failure_callback_arg,
                                                      async_pool_handle(owner.async_pool),
                                                      device,
                                                      manager->allocator_events.get(),
                                                      &manager->failure_callback_panicked);
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
  int32_t device_count             = 0;
  cudf_status_t const count_status = cudf_read_cuda_device_count(&device_count);
  if (count_status != CUDF_SUCCESS) { return count_status; }
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

cudf_status_t memory_manager_for_device_impl(
  int32_t device,
  cudf_memory_failure_callback_t failure_callback,
  void* failure_callback_arg,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out,
  cudf_memory_manager_installation_identity_t const* installation_identity,
  cudf_memory_manager_installation_record_t* installation)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    cudf_status_t const validation_status = validate_memory_manager_device(device);
    if (validation_status != CUDF_SUCCESS) { return validation_status; }
    if (failure_callback == nullptr && failure_callback_arg != nullptr) {
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
    auto& registry       = memory_manager_registry();
    auto it              = registry.find(device);
    bool created_by_call = false;
    if (installation != nullptr) {
      fill_installation_record(
        it == registry.end() ? nullptr : it->second.get(), device, *installation);
    }
    if (it == registry.end()) {
      it = registry
             .emplace(device,
                      make_memory_manager_for_current_device(device,
                                                             failure_callback,
                                                             failure_callback_arg,
                                                             resolved_options,
                                                             installation_identity))
             .first;
      created_by_call = true;
      // Publish callback retention before later installation work can fail.
      if (installation != nullptr) {
        fill_installation_record(it->second.get(), device, *installation, created_by_call);
      }
    } else if (options != nullptr &&
               !memory_manager_options_equal(it->second->options, resolved_options)) {
      if (installation != nullptr) {
        fill_installation_record(it->second.get(),
                                 device,
                                 *installation,
                                 false,
                                 CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_OPTIONS_MISMATCH);
      }
      set_error(CUDF_ERROR_RUNTIME_BINDING_MISMATCH,
                "device memory manager already exists with different physical-base "
                "options; "
                "the process-global allocator cannot be replaced after first use");
      return CUDF_ERROR_RUNTIME_BINDING_MISMATCH;
    } else if (failure_callback != nullptr && !it->second->failure_callback_installed) {
      if (installation != nullptr) {
        fill_installation_record(it->second.get(), device, *installation);
      }
      set_error(
        "device memory manager already exists without a failure callback; "
        "install the callback before first use");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (installation_identity != nullptr) {
      if (!it->second->has_installation_identity ||
          !installation_identity_equal(it->second->installation_identity, *installation_identity)) {
        if (installation != nullptr) {
          fill_installation_record(it->second.get(),
                                   device,
                                   *installation,
                                   false,
                                   CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_IDENTITY_MISMATCH);
        }
        set_error("device memory manager installation identity does not match");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      if (it->second->failure_callback != failure_callback ||
          it->second->failure_callback_arg != failure_callback_arg) {
        if (installation != nullptr) {
          fill_installation_record(
            it->second.get(),
            device,
            *installation,
            false,
            CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CALLBACK_BINDING_MISMATCH);
        }
        set_error("device memory manager failure callback binding does not match");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      if (it->second->failure_callback_panicked.load(std::memory_order_acquire)) {
        if (installation != nullptr) {
          fill_installation_record(it->second.get(), device, *installation, created_by_call);
        }
        set_error("device memory manager failure callback previously panicked");
        return CUDF_ERROR_INVALID_OPERATION;
      }
    }
    if (!it->second->current_resource_installed) {
      // Device-domain invariant: this is the one production install point. The
      // adaptor stays the device's current resource for the process lifetime.
      // The manager is already registry-owned before the current resource is
      // switched, so any exception during the switch cannot leave RMM pointing
      // at freed memory.
#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
      if (fail_next_current_resource_install().exchange(false, std::memory_order_acq_rel)) {
        it->second->current_resource_install_retryable = true;
        it->second->current_resource_install_cause =
          CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CURRENT_RESOURCE_INSTALL_FAILED;
        if (installation != nullptr) {
          fill_installation_record(it->second.get(), device, *installation, created_by_call);
        }
        set_error(
          "device memory manager current resource installation failed by injected test fault; "
          "manager remains registered");
        return CUDF_ERROR_INVALID_OPERATION;
      }
#endif
      try {
        cudf_set_current_device_resource_raw(it->second->owner->mr.get());
      } catch (...) {
        it->second->current_resource_install_retryable = false;
        it->second->current_resource_install_cause =
          CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CURRENT_RESOURCE_INSTALL_FAILED;
        if (installation != nullptr) {
          fill_installation_record(it->second.get(), device, *installation, created_by_call);
        }
        set_error(
          "device memory manager current resource installation failed; "
          "manager remains registered");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      it->second->current_resource_installed         = true;
      it->second->current_resource_install_retryable = false;
      it->second->current_resource_install_cause     = CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_NONE;
    }
#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
    if (invoke_failure_callback_before_install_returns().exchange(false,
                                                                  std::memory_order_acq_rel) &&
        it->second->options.allocation_limit_bytes > 0) {
      auto const request = static_cast<std::size_t>(it->second->options.allocation_limit_bytes) + 1;
      std::string route_error;
      auto route = it->second->owner->reservation_adaptor->attach_current_resource_route(
        cudf::get_default_stream(), it->second->owner, true, route_error);
      if (route == nullptr) { throw std::runtime_error{route_error}; }
      try {
        auto stream      = cuda::stream_ref{cudaStream_t{nullptr}};
        auto* allocation = it->second->owner->mr->allocate(stream, request);
        it->second->owner->mr->deallocate(stream, allocation, request);
      } catch (rmm::out_of_memory const&) {
      } catch (...) {
        it->second->owner->reservation_adaptor->detach_current_resource_route(route);
        throw;
      }
      it->second->owner->reservation_adaptor->detach_current_resource_route(route);
    }
#endif
    if (installation != nullptr) {
      fill_installation_record(it->second.get(), device, *installation, created_by_call);
      if (installation_identity != nullptr &&
          installation->retry != CUDF_MEMORY_MANAGER_INSTALLATION_REUSABLE) {
        set_error("device memory manager installation is not reusable");
        return CUDF_ERROR_INVALID_OPERATION;
      }
    }
    *out = reinterpret_cast<cudf_memory_manager_handle>(it->second.get());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // namespace cudf::rust::memory

using namespace cudf::rust::memory;

rmm::mr::device_memory_resource* cudf_tracked_current_device_resource_raw()
{
  auto const device = current_cuda_device();
  std::scoped_lock lock{current_memory_resource_identity_mutex()};
  auto& identities = current_memory_resource_identities();
  auto const found = identities.find(device);
  if (found != identities.end()) { return found->second; }

  return set_current_device_resource_raw_locked(device, default_memory_resource_for_device(device));
}

rmm::mr::device_memory_resource* cudf_tracked_current_device_resource_if_installed_raw()
{
  auto const device = current_cuda_device();
  std::scoped_lock lock{current_memory_resource_identity_mutex()};
  auto& identities = current_memory_resource_identities();
  auto const found = identities.find(device);
  return found != identities.end() ? found->second : nullptr;
}

void cudf_set_current_device_resource_raw(rmm::mr::device_memory_resource* target)
{
  auto const device = current_cuda_device();
  std::scoped_lock lock{current_memory_resource_identity_mutex()};
  set_current_device_resource_raw_locked(device, target);
}

extern "C" {
#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
void cudf_test_memory_manager_fail_next_current_resource_install(void)
{
  fail_next_current_resource_install().store(true, std::memory_order_release);
}

void cudf_test_memory_manager_invoke_failure_callback_before_install_returns(void)
{
  invoke_failure_callback_before_install_returns().store(true, std::memory_order_release);
}
#endif
cudf_status_t cudf_memory_manager_for_device(int32_t device, cudf_memory_manager_handle* out)
{
  return memory_manager_for_device_impl(device, nullptr, nullptr, nullptr, out, nullptr, nullptr);
}

cudf_status_t cudf_memory_manager_install(
  int32_t device,
  cudf_memory_manager_options_t const* options,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_installation_identity_t const* identity,
  cudf_memory_manager_handle* out,
  cudf_memory_manager_installation_record_t* installation)
{
  CUDF_RUST_NULL_CHECK(installation, "installation is null");
  *installation = {};
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(options, "options is null");
  if (identity != nullptr && callback == nullptr) {
    set_error("installation identity requires a failure callback");
    return CUDF_ERROR_INVALID_ARGUMENT;
  }
  return memory_manager_for_device_impl(
    device, callback, callback_arg, options, out, identity, installation);
}

cudf_status_t cudf_memory_manager_installation_record(
  int32_t device, cudf_memory_manager_installation_record_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out                = {};
  out->struct_size    = sizeof(cudf_memory_manager_installation_record_t);
  out->device_ordinal = device;
  CUDF_RUST_TRY
  {
    cudf_status_t const validation_status = validate_memory_manager_device(device);
    if (validation_status != CUDF_SUCCESS) { return validation_status; }
    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }
    std::scoped_lock lock{memory_manager_registry_mutex()};
    auto const found = memory_manager_registry().find(device);
    fill_installation_record(
      found == memory_manager_registry().end() ? nullptr : found->second.get(), device, *out);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
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

cudf_status_t cudf_memory_manager_installation_state(int32_t device,
                                                     int32_t* exists,
                                                     int32_t* failure_callback_installed,
                                                     int32_t* current_resource_installed)
{
  CUDF_RUST_NULL_CHECK(exists, "exists is null");
  *exists = 0;
  CUDF_RUST_NULL_CHECK(failure_callback_installed, "failure_callback_installed is null");
  *failure_callback_installed = 0;
  CUDF_RUST_NULL_CHECK(current_resource_installed, "current_resource_installed is null");
  *current_resource_installed = 0;
  CUDF_RUST_TRY
  {
    cudf_status_t const validation_status = validate_memory_manager_device(device);
    if (validation_status != CUDF_SUCCESS) { return validation_status; }
    std::scoped_lock lock{memory_manager_registry_mutex()};
    auto const it = memory_manager_registry().find(device);
    if (it == memory_manager_registry().end()) { return CUDF_SUCCESS; }
    *exists                     = 1;
    *failure_callback_installed = it->second->failure_callback_installed ? 1 : 0;
    *current_resource_installed =
      it->second->current_resource_installed && manager_is_actual_current_resource(*it->second) ? 1
                                                                                                : 0;
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
    state->current_resource_route_uses_manager_upstream = true;
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

cudf_status_t cudf_memory_manager_allocator_event_ring_bounds(
  cudf_memory_manager_handle manager, cudf_allocator_event_ring_bounds_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager allocator event ring is unavailable");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = manager_state->allocator_events->bounds();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_read_allocator_event(cudf_memory_manager_handle manager,
                                                       uint64_t sequence,
                                                       cudf_allocator_event_t* out,
                                                       cudf_allocator_event_read_status_t* status)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(status, "status is null");
  *status = CUDF_ALLOCATOR_EVENT_READ_NOT_YET_RECORDED;
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager allocator event ring is unavailable");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *status = manager_state->allocator_events->read(sequence, *out);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
