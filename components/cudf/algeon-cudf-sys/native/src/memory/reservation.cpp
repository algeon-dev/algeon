#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory/internal.hpp"
#include "rmm_interop_bridge.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <stdexcept>

namespace cudf::rust::memory {

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  rmm::mr::device_memory_resource* upstream)
  : upstream_{rmm::mr::any_device_resource_from_raw(upstream)}
{
}

std::shared_ptr<current_resource_route_state>
reservation_aware_resource_adaptor::attach_current_resource_route(
  rmm::cuda_stream_view stream,
  std::shared_ptr<memory_resource_owner> target_owner,
  bool routes_to_manager_upstream,
  std::string& error)
{
  if (target_owner == nullptr || target_owner->mr == nullptr) {
    error = "current-resource route target is not a live memory resource";
    return {};
  }

  auto const key = stream_key(stream);
  std::scoped_lock lock{current_resource_route_mutex_};
  auto const existing = current_resource_routes_by_stream_.find(key);
  if (existing != current_resource_routes_by_stream_.end()) {
    auto const& state = existing->second;
    if (state != nullptr && state->attached) {
      if (state->target_owner.get() != target_owner.get() ||
          state->routes_to_manager_upstream != routes_to_manager_upstream) {
        error = "stream already has a different current-resource allocation route";
        return {};
      }
      state->attachment_count += 1;
      return state;
    }
    current_resource_routes_by_stream_.erase(existing);
  }

  auto state                              = std::make_shared<current_resource_route_state>();
  state->stream_key                       = key;
  state->target_owner                     = std::move(target_owner);
  state->routes_to_manager_upstream       = routes_to_manager_upstream;
  current_resource_routes_by_stream_[key] = state;
  return state;
}

void reservation_aware_resource_adaptor::detach_current_resource_route(
  std::shared_ptr<current_resource_route_state> const& state)
{
  if (state == nullptr) { return; }
  std::scoped_lock lock{current_resource_route_mutex_};
  if (state->attachment_count > 1) {
    state->attachment_count -= 1;
    return;
  }
  state->attachment_count = 0;
  state->attached     = false;
  auto const existing = current_resource_routes_by_stream_.find(state->stream_key);
  if (existing != current_resource_routes_by_stream_.end() &&
      existing->second.get() == state.get()) {
    current_resource_routes_by_stream_.erase(existing);
  }
}

void* reservation_aware_resource_adaptor::stream_key(rmm::cuda_stream_view stream)
{
  return reinterpret_cast<void*>(stream.value());
}

std::shared_ptr<current_resource_route_state>
reservation_aware_resource_adaptor::current_resource_route_for_stream(rmm::cuda_stream_view stream)
{
  auto const key = stream_key(stream);
  std::scoped_lock lock{current_resource_route_mutex_};
  auto const it = current_resource_routes_by_stream_.find(key);
  if (it == current_resource_routes_by_stream_.end()) { return {}; }
  auto const& state = it->second;
  if (state == nullptr || !state->attached || state->target_owner == nullptr ||
      state->target_owner->mr == nullptr) {
    return {};
  }
  return state;
}

std::shared_ptr<current_resource_allocation_record>
reservation_aware_resource_adaptor::current_resource_route_for_deallocation(void* ptr)
{
  if (ptr == nullptr) { return {}; }
  std::scoped_lock lock{current_resource_route_mutex_};
  auto const it = current_resource_routes_by_pointer_.find(ptr);
  if (it == current_resource_routes_by_pointer_.end()) { return {}; }
  auto record = std::move(it->second);
  current_resource_routes_by_pointer_.erase(it);
  return record;
}

void* reservation_aware_resource_adaptor::do_allocate(std::size_t bytes,
                                                      rmm::cuda_stream_view stream,
                                                      std::size_t alignment)
{
  query_domain_allocations_->throw_if_invariant_violated();
  cudf::rust::test_support::record_allocation_attempt(bytes);
  auto query_route = query_domain_member_for_stream(stream, bytes);
  if (query_route.has_value()) {
    auto domain = query_route->member->domain;
    try {
      cudf::rust::test_support::pause_query_domain_route_if_armed(domain->id);
      auto* result = query_route->member->resource->allocate_routed(stream, bytes, alignment);
      if (bytes != 0) {
        query_route->member->resource->commit_routed_allocation();
        cleanup_retired_query_domain_member(query_route->member);
      }
      query_route->operation.finish();
      cleanup_query_domain_if_idle(domain);
      return result;
    } catch (...) {
      query_route->operation.finish();
      cleanup_query_domain_if_idle(domain);
      throw;
    }
  }

  auto current_resource_route = current_resource_route_for_stream(stream);
  if (current_resource_route != nullptr) {
    auto* target = current_resource_route->target_owner->mr.get();
    void* result = current_resource_route->routes_to_manager_upstream
                     ? upstream_.allocate(stream, bytes, alignment)
                     : target->allocate(stream, bytes, alignment);
    if (bytes != 0) {
      try {
        std::scoped_lock lock{current_resource_route_mutex_};
        auto record = std::make_shared<current_resource_allocation_record>(
          current_resource_allocation_record{current_resource_route, bytes, alignment});
        auto const insertion = current_resource_routes_by_pointer_.emplace(result, std::move(record));
        if (!insertion.second) {
          throw std::runtime_error{"duplicate current-resource routed pointer"};
        }
      } catch (...) {
        if (current_resource_route->routes_to_manager_upstream) {
          upstream_.deallocate(stream, result, bytes, alignment);
        } else {
          target->deallocate(stream, result, bytes, alignment);
        }
        throw;
      }
    }
    return result;
  }
  if (bytes != 0) {
    throw cudf::rust::invariant_error{
      "device allocation stream has no query domain or explicit backend owner"};
  }
  return upstream_.allocate(stream, bytes, alignment);
}

void reservation_aware_resource_adaptor::do_deallocate(void* ptr,
                                                       std::size_t bytes,
                                                       rmm::cuda_stream_view stream,
                                                       std::size_t alignment) noexcept
{
  auto current_resource_route = current_resource_route_for_deallocation(ptr);
  if (current_resource_route != nullptr) {
    if (current_resource_route->route->routes_to_manager_upstream) {
      upstream_.deallocate(stream, ptr, current_resource_route->bytes, current_resource_route->alignment);
    } else {
      current_resource_route->route->target_owner->mr->deallocate(
        stream, ptr, current_resource_route->bytes, current_resource_route->alignment);
    }
    return;
  }

  auto query_record = query_domain_allocations_->take(ptr);
  if (query_record.has_value()) {
    auto operation = query_record->domain->resource->begin_operation();
    query_record->domain->resource->deallocate_from_adaptor(
      ptr, query_record->member, stream, query_record->bytes, query_record->alignment);
    if (query_record->routed) {
      query_record->member->routed_current_allocations.fetch_sub(1, std::memory_order_acq_rel);
    }
    operation.finish();
    cleanup_query_domain_if_idle(query_record->domain);
    return;
  }

  if (bytes != 0) {
    query_domain_allocations_->record_invariant_violation(
      query_domain_allocation_registry::invariant_violation::unknown_device_pointer);
    set_error(CUDF_ERROR_INVARIANT_VIOLATION, "unknown or already-freed device allocation pointer");
    return;
  }

  upstream_.deallocate(stream, ptr, bytes, alignment);
}

[[nodiscard]] bool reservation_aware_resource_adaptor::do_is_equal(
  rmm::mr::device_memory_resource const& other) const noexcept
{
  return this == std::addressof(other);
}

}  // namespace cudf::rust::memory

using namespace cudf::rust::memory;

void cudf_current_resource_route_destroy_preserving_error(
  cudf_current_resource_route_handle route) noexcept
{
  if (route == nullptr) { return; }
  try {
    auto* state = reinterpret_cast<current_resource_route_handle_state*>(route);
    state->manager->owner->reservation_adaptor->detach_current_resource_route(state->route);
    delete state;
  } catch (...) {
  }
}

extern "C" {
cudf_status_t cudf_memory_manager_route_current_resource(cudf_memory_manager_handle manager,
                                                         cudf_stream_handle stream,
                                                         cudf_memory_resource_handle resource,
                                                         cudf_current_resource_route_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_NULL_CHECK(resource, "memory resource is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager handle is not registered");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    cudf_status_t const stream_status = validate_stream_device_for_manager(
      stream,
      manager_state->device,
      "current-resource route requires stream and memory manager to be on the same CUDA device");
    if (stream_status != CUDF_SUCCESS) { return stream_status; }

    auto& resource_state = memory_resource_state_from_handle(resource);
    if (resource_state.owner == nullptr || resource_state.owner->mr == nullptr) {
      set_error("current-resource route target memory resource is not live");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (resource_state.owner->device != manager_state->device) {
      set_error(
        "current-resource route requires target resource and memory manager to be on the same CUDA "
        "device");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(manager_state->device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }
    if (!manager_state->current_resource_installed ||
        cudf_rust_rmm_current_device_resource_is(manager_state->owner->mr.get()) == 0) {
      manager_state->current_resource_integrity_lost = true;
      set_error("memory manager's permanent current resource is no longer installed");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto* adaptor = manager_state->owner->reservation_adaptor;
    std::string attach_error;
    bool const routes_to_manager_upstream =
      resource_state.current_resource_route_uses_manager_upstream;
    if (routes_to_manager_upstream &&
        resource_state.owner.get() != manager_state->owner.get()) {
      set_error("current-resource route declares manager-upstream ownership for a different resource owner");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto route = adaptor->attach_current_resource_route(
      cudf_stream_view_from_handle(stream),
      resource_state.owner,
      routes_to_manager_upstream,
      attach_error);
    if (route == nullptr) {
      set_error(attach_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state     = std::make_unique<current_resource_route_handle_state>();
    state->manager = std::move(manager_state);
    state->route   = std::move(route);
    *out           = reinterpret_cast<cudf_current_resource_route_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_route_current_resource_raw(
  cudf_memory_manager_handle manager,
  void* raw_stream,
  cudf_memory_resource_handle resource,
  cudf_current_resource_route_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_NULL_CHECK(raw_stream, "raw stream is null");
  CUDF_RUST_NULL_CHECK(resource, "memory resource is null");
  if (reinterpret_cast<uintptr_t>(raw_stream) <= 2) {
    set_error("raw stream must not be a CUDA default stream");
    return CUDF_ERROR_INVALID_ARGUMENT;
  }
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager handle is not registered");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    int stream_device = -1;
    auto const cuda_status = cudaStreamGetDevice(reinterpret_cast<cudaStream_t>(raw_stream), &stream_device);
    if (cuda_status != cudaSuccess) {
      set_error(cudaGetErrorString(cuda_status));
      return CUDF_ERROR_CUDA;
    }
    if (stream_device != manager_state->device) {
      set_error("raw current-resource route requires stream and memory manager to be on the same CUDA device");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto& resource_state = memory_resource_state_from_handle(resource);
    if (resource_state.owner == nullptr || resource_state.owner->mr == nullptr) {
      set_error("current-resource route target memory resource is not live");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (resource_state.owner->device != manager_state->device) {
      set_error("current-resource route requires target resource and memory manager to be on the same CUDA device");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(manager_state->device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }
    if (!manager_state->current_resource_installed ||
        cudf_rust_rmm_current_device_resource_is(manager_state->owner->mr.get()) == 0) {
      manager_state->current_resource_integrity_lost = true;
      set_error("memory manager's permanent current resource is no longer installed");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto* adaptor = manager_state->owner->reservation_adaptor;
    std::string attach_error;
    bool const routes_to_manager_upstream = resource_state.current_resource_route_uses_manager_upstream;
    if (routes_to_manager_upstream && resource_state.owner.get() != manager_state->owner.get()) {
      set_error("current-resource route declares manager-upstream ownership for a different resource owner");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto route = adaptor->attach_current_resource_route(
      rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(raw_stream)},
      resource_state.owner,
      routes_to_manager_upstream,
      attach_error);
    if (route == nullptr) {
      set_error(attach_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state     = std::make_unique<current_resource_route_handle_state>();
    state->manager = std::move(manager_state);
    state->route   = std::move(route);
    *out           = reinterpret_cast<cudf_current_resource_route_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_current_resource_route_destroy(cudf_current_resource_route_handle route)
{
  CUDF_RUST_NULL_CHECK(route, "current-resource route is null");
  cudf_clear_error();
  cudf_current_resource_route_destroy_preserving_error(route);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_reservation_usage(cudf_reservation_handle reservation,
                                     cudf_memory_resource_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(reservation, "reservation is null");
  CUDF_RUST_TRY
  {
    auto& state   = *reinterpret_cast<reservation_handle_state*>(reservation);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    *out = adaptor->query_domain_member_usage(state.query_domain_member);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reservation_detach(cudf_reservation_handle reservation)
{
  CUDF_RUST_NULL_CHECK(reservation, "reservation is null");
  CUDF_RUST_TRY
  {
    auto* state   = reinterpret_cast<reservation_handle_state*>(reservation);
    auto* adaptor = state->manager->owner->reservation_adaptor;
    adaptor->detach_query_domain_member(state->query_domain_member);
    delete state;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
