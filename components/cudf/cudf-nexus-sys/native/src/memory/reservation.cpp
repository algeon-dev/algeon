#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory/internal.hpp"
#include "rmm_interop_bridge.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <rmm/error.hpp>

#include <stdexcept>

namespace cudf::rust::memory {

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  rmm::mr::device_memory_resource* upstream)
  : upstream_{rmm::mr::any_device_resource_from_raw(upstream)}
{
}

std::shared_ptr<reservation_state> reservation_aware_resource_adaptor::attach(
  rmm::cuda_stream_view stream,
  std::uint64_t reservation_id,
  std::size_t limit_bytes,
  std::shared_ptr<reservation_arena_state> arena,
  std::string& error)
{
  auto const key = stream_key(stream);
  std::unique_lock query_lock{query_domain_mutex_};
  auto const existing_domain = query_domain_members_by_stream_.find(key);
  if (existing_domain != query_domain_members_by_stream_.end()) {
    auto state = existing_domain->second;
    if (state != nullptr &&
        (state->attached ||
         (state->resource != nullptr && state->resource->routed_pending_allocations() != 0))) {
      error =
        "stream still has an attached or retiring query memory domain "
        "member";
      return {};
    }
    query_domain_members_by_stream_.erase(existing_domain);
    query_domain_member_count_.fetch_sub(1, std::memory_order_relaxed);
  }
  std::scoped_lock lock{mutex_};
  auto const existing = reservations_by_stream_.find(key);
  if (existing != reservations_by_stream_.end()) {
    auto state = existing->second.lock();
    if (state != nullptr && state->attached) {
      error = "stream already has an attached memory reservation";
      return {};
    }
    reservations_by_stream_.erase(existing);
  }
  if (arena != nullptr && !arena->active) {
    error = "reservation arena is not active";
    return {};
  }

  auto state              = std::make_shared<reservation_state>();
  state->id               = reservation_id;
  state->allocation_limit = limit_bytes;
  state->stream_key       = key;
  state->arena            = std::move(arena);
  if (state->arena != nullptr) { state->arena->active_members += 1; }
  reservations_by_stream_[key] = state;
  return state;
}

std::shared_ptr<reservation_arena_state> reservation_aware_resource_adaptor::create_arena(
  std::uint64_t arena_id, std::size_t limit_bytes)
{
  std::scoped_lock lock{mutex_};
  auto state              = std::make_shared<reservation_arena_state>();
  state->id               = arena_id;
  state->allocation_limit = limit_bytes;
  return state;
}

void reservation_aware_resource_adaptor::detach(std::shared_ptr<reservation_state> const& state)
{
  if (state == nullptr) { return; }
  std::scoped_lock lock{mutex_};
  bool const was_attached = state->attached;
  state->attached         = false;
  auto const existing     = reservations_by_stream_.find(state->stream_key);
  if (existing != reservations_by_stream_.end()) {
    auto existing_state = existing->second.lock();
    if (existing_state == nullptr || existing_state.get() == state.get()) {
      reservations_by_stream_.erase(existing);
    }
  }
  if (was_attached && state->arena != nullptr && state->arena->active_members > 0) {
    state->arena->active_members -= 1;
  }
}

void reservation_aware_resource_adaptor::destroy_arena(
  std::shared_ptr<reservation_arena_state> const& state)
{
  if (state == nullptr) { return; }
  std::scoped_lock lock{mutex_};
  state->active = false;
}

std::shared_ptr<current_resource_route_state>
reservation_aware_resource_adaptor::attach_current_resource_route(
  rmm::cuda_stream_view stream,
  std::shared_ptr<memory_resource_owner> target_owner,
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
      error = "stream already has a current-resource allocation route";
      return {};
    }
    current_resource_routes_by_stream_.erase(existing);
  }

  auto state          = std::make_shared<current_resource_route_state>();
  state->stream_key   = key;
  state->target_owner = std::move(target_owner);
  current_resource_routes_by_stream_[key] = state;
  return state;
}

void reservation_aware_resource_adaptor::detach_current_resource_route(
  std::shared_ptr<current_resource_route_state> const& state)
{
  if (state == nullptr) { return; }
  std::scoped_lock lock{current_resource_route_mutex_};
  state->attached = false;
  auto const existing = current_resource_routes_by_stream_.find(state->stream_key);
  if (existing != current_resource_routes_by_stream_.end() && existing->second.get() == state.get()) {
    current_resource_routes_by_stream_.erase(existing);
  }
}

cudf_memory_resource_usage_t reservation_aware_resource_adaptor::usage(
  std::shared_ptr<reservation_state> const& state)
{
  cudf_memory_resource_usage_t out{};
  if (state == nullptr) { return out; }
  std::scoped_lock lock{mutex_};
  fill_usage(out,
             state->allocation_limit,
             state->current_bytes,
             state->peak_bytes,
             state->total_bytes,
             state->current_aligned_bytes,
             state->peak_aligned_bytes,
             state->total_allocations,
             state->current_allocations,
             state->peak_allocations,
             state->failed_allocation_count,
             state->largest_failed_allocation_bytes);
  out.reserved[0] = state->id;
  out.reserved[1] = state->attached ? 1 : 0;
  return out;
}

cudf_memory_resource_usage_t reservation_aware_resource_adaptor::arena_usage(
  std::shared_ptr<reservation_arena_state> const& state)
{
  cudf_memory_resource_usage_t out{};
  if (state == nullptr) { return out; }
  std::scoped_lock lock{mutex_};
  fill_usage(out,
             state->allocation_limit,
             state->current_bytes,
             state->peak_bytes,
             state->total_bytes,
             state->current_aligned_bytes,
             state->peak_aligned_bytes,
             state->total_allocations,
             state->current_allocations,
             state->peak_allocations,
             state->failed_allocation_count,
             state->largest_failed_allocation_bytes);
  out.reserved[0] = state->id;
  out.reserved[1] = state->active ? 1 : 0;
  out.reserved[2] = state->active_members;
  return out;
}

bool reservation_aware_resource_adaptor::resize(std::shared_ptr<reservation_state> const& state,
                                                std::size_t limit_bytes,
                                                std::string& error)
{
  if (state == nullptr) {
    error = "reservation is null";
    return false;
  }
  std::scoped_lock lock{mutex_};
  if (!state->attached) {
    error = "reservation is not attached";
    return false;
  }
  state->allocation_limit = limit_bytes;
  return true;
}

bool reservation_aware_resource_adaptor::resize_arena(
  std::shared_ptr<reservation_arena_state> const& state,
  std::size_t limit_bytes,
  std::string& error)
{
  if (state == nullptr) {
    error = "reservation arena is null";
    return false;
  }
  std::scoped_lock lock{mutex_};
  if (!state->active) {
    error = "reservation arena is not active";
    return false;
  }
  state->allocation_limit = limit_bytes;
  return true;
}

void* reservation_aware_resource_adaptor::stream_key(rmm::cuda_stream_view stream)
{
  return reinterpret_cast<void*>(stream.value());
}

template <typename State>
void reservation_aware_resource_adaptor::record_failed_allocation(State& state, std::size_t bytes)
{
  state.failed_allocation_count += 1;
  update_peak(state.largest_failed_allocation_bytes, bytes);
}

std::shared_ptr<reservation_state> reservation_aware_resource_adaptor::reservation_for_stream(
  rmm::cuda_stream_view stream)
{
  auto const key = stream_key(stream);
  auto const it  = reservations_by_stream_.find(key);
  if (it == reservations_by_stream_.end()) { return {}; }
  auto state = it->second.lock();
  if (state == nullptr || !state->attached) {
    reservations_by_stream_.erase(it);
    return {};
  }
  return state;
}

template <typename State>
void reservation_aware_resource_adaptor::reserve_allocation(State& state,
                                                            std::size_t bytes,
                                                            std::size_t aligned_bytes,
                                                            char const* limit_message,
                                                            memory_failure_domain_identity identity)
{
  if (state.current_aligned_bytes > state.allocation_limit ||
      aligned_bytes > state.allocation_limit - state.current_aligned_bytes) {
    record_failed_allocation(state, bytes);
    throw attributed_out_of_memory{
      limit_message, CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT, identity};
  }
  state.current_aligned_bytes += aligned_bytes;
}

template <typename State>
void reservation_aware_resource_adaptor::commit_allocation(State& state,
                                                           std::size_t bytes,
                                                           std::size_t aligned_bytes)
{
  state.current_bytes += bytes;
  state.total_bytes += bytes;
  state.current_allocations += 1;
  state.total_allocations += 1;
  update_peak(state.peak_bytes, state.current_bytes);
  update_peak(state.peak_aligned_bytes, state.current_aligned_bytes);
  update_peak(state.peak_allocations, state.current_allocations);
  (void)aligned_bytes;
}

template <typename State>
void reservation_aware_resource_adaptor::release_reserved_alignment(State& state,
                                                                    std::size_t aligned_bytes)
{
  if (state.current_aligned_bytes >= aligned_bytes) {
    state.current_aligned_bytes -= aligned_bytes;
  } else {
    state.current_aligned_bytes = 0;
  }
}

template <typename State>
void reservation_aware_resource_adaptor::rollback_reserved_allocation(State& state,
                                                                      std::size_t bytes,
                                                                      std::size_t aligned_bytes)
{
  release_reserved_alignment(state, aligned_bytes);
  record_failed_allocation(state, bytes);
}

template <typename State>
void reservation_aware_resource_adaptor::release_allocation(State& state,
                                                            std::size_t bytes,
                                                            std::size_t aligned_bytes)
{
  state.current_bytes = state.current_bytes >= bytes ? state.current_bytes - bytes : 0;
  state.current_aligned_bytes =
    state.current_aligned_bytes >= aligned_bytes ? state.current_aligned_bytes - aligned_bytes : 0;
  state.current_allocations = state.current_allocations > 0 ? state.current_allocations - 1 : 0;
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

std::shared_ptr<current_resource_route_state>
reservation_aware_resource_adaptor::current_resource_route_for_deallocation(void* ptr)
{
  if (ptr == nullptr) { return {}; }
  std::scoped_lock lock{current_resource_route_mutex_};
  auto const it = current_resource_routes_by_pointer_.find(ptr);
  if (it == current_resource_routes_by_pointer_.end()) { return {}; }
  auto state = std::move(it->second);
  current_resource_routes_by_pointer_.erase(it);
  return state;
}

void* reservation_aware_resource_adaptor::do_allocate(std::size_t bytes,
                                                      rmm::cuda_stream_view stream,
                                                      std::size_t alignment)
{
  auto query_route = query_domain_member_for_stream(stream, bytes);
  if (query_route.has_value()) {
    auto domain = query_route->member->domain;
    try {
      cudf::rust::test_support::pause_query_domain_route_if_armed(domain->id);
      auto* result = query_route->member->resource->allocate_routed(stream, bytes, alignment);
      if (bytes != 0) {
        try {
          std::unique_lock lock{query_domain_mutex_};
          // Committed routed allocations retain their originating member by
          // pointer, independently of whichever member is attached to the
          // stream when the allocation is eventually freed. Direct member
          // allocations bypass this map and retain their resource directly.
          auto const insertion =
            query_domain_members_by_pointer_.emplace(result, query_route->member);
          if (!insertion.second) {
            throw std::runtime_error{"duplicate query memory domain routed pointer"};
          }
          query_domain_routed_pointer_count_.fetch_add(1, std::memory_order_relaxed);
        } catch (...) {
          query_route->member->resource->rollback_routed_allocation(
            result, stream, bytes, alignment);
          throw;
        }
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
    void* result = target->allocate(stream, bytes, alignment);
    if (bytes != 0) {
      try {
        std::scoped_lock lock{current_resource_route_mutex_};
        auto const insertion = current_resource_routes_by_pointer_.emplace(result, current_resource_route);
        if (!insertion.second) {
          throw std::runtime_error{"duplicate current-resource routed pointer"};
        }
      } catch (...) {
        target->deallocate(stream, result, bytes, alignment);
        throw;
      }
    }
    return result;
  }

  auto const aligned_bytes = cudf::rust::reservation::allocation_charge(bytes, alignment);
  std::shared_ptr<reservation_state> reservation;
  std::shared_ptr<reservation_arena_state> arena;
  {
    std::scoped_lock lock{mutex_};
    reservation = reservation_for_stream(stream);
    if (reservation != nullptr) {
      arena = reservation->arena;
      if (arena != nullptr) {
        try {
          reserve_allocation(*arena,
                             bytes,
                             aligned_bytes,
                             "reservation arena memory resource allocation limit exceeded",
                             memory_failure_domain_identity{
                               CUDF_MEMORY_FAILURE_DOMAIN_RESERVATION_ARENA,
                               arena->id,
                               reservation->id,
                             });
        } catch (...) {
          record_failed_allocation(*reservation, bytes);
          throw;
        }
      }
      try {
        reserve_allocation(*reservation,
                           bytes,
                           aligned_bytes,
                           "stream reservation memory resource allocation limit exceeded",
                           memory_failure_domain_identity{
                             CUDF_MEMORY_FAILURE_DOMAIN_STREAM_RESERVATION,
                             reservation->id,
                             0,
                           });
      } catch (...) {
        if (arena != nullptr) { release_reserved_alignment(*arena, aligned_bytes); }
        throw;
      }
    }
  }

  void* ptr = nullptr;
  try {
    ptr = upstream_.allocate(stream, bytes, alignment);
  } catch (...) {
    if (reservation != nullptr) {
      std::scoped_lock lock{mutex_};
      if (arena != nullptr) { rollback_reserved_allocation(*arena, bytes, aligned_bytes); }
      rollback_reserved_allocation(*reservation, bytes, aligned_bytes);
    }
    throw;
  }

  if (reservation != nullptr) {
    try {
      std::scoped_lock lock{mutex_};
      // Insert before commit: the map insert is the only throwing step, and
      // commit_allocation is pure arithmetic, so a throw here always means
      // "reserved but never committed".
      allocations_[ptr] = allocation_record{bytes, aligned_bytes, reservation, arena};
      if (arena != nullptr) { commit_allocation(*arena, bytes, aligned_bytes); }
      commit_allocation(*reservation, bytes, aligned_bytes);
    } catch (...) {
      // Bookkeeping failed (e.g. bad_alloc on rehash): release the upstream
      // allocation and roll the reservation back so neither the GPU block
      // nor the accounted bytes leak.
      upstream_.deallocate(stream, ptr, bytes, alignment);
      {
        std::scoped_lock lock{mutex_};
        allocations_.erase(ptr);
        if (arena != nullptr) { rollback_reserved_allocation(*arena, bytes, aligned_bytes); }
        rollback_reserved_allocation(*reservation, bytes, aligned_bytes);
      }
      throw;
    }
  }
  return ptr;
}

void reservation_aware_resource_adaptor::do_deallocate(void* ptr,
                                                       std::size_t bytes,
                                                       rmm::cuda_stream_view stream,
                                                       std::size_t alignment) noexcept
{
  auto member_route = query_domain_member_for_deallocation(ptr);
  if (member_route.has_value()) {
    auto domain = member_route->member->domain;
    member_route->member->resource->deallocate_routed(ptr, stream, bytes, alignment);
    member_route->operation.finish();
    cleanup_retired_query_domain_member(member_route->member);
    cleanup_query_domain_if_idle(domain);
    return;
  }

  auto query_route = query_domain_for_pointer(ptr);
  if (query_route.has_value()) {
    auto domain = query_route->domain;
    domain->resource->deallocate_from_adaptor(ptr, {}, stream, bytes, alignment);
    query_route->operation.finish();
    cleanup_query_domain_if_idle(domain);
    return;
  }

  auto current_resource_route = current_resource_route_for_deallocation(ptr);
  if (current_resource_route != nullptr) {
    current_resource_route->target_owner->mr->deallocate(stream, ptr, bytes, alignment);
    return;
  }

  allocation_record record{};
  bool found = false;
  {
    std::scoped_lock lock{mutex_};
    auto const it = allocations_.find(ptr);
    if (it != allocations_.end()) {
      record = std::move(it->second);
      allocations_.erase(it);
      found = true;
    }
  }

  upstream_.deallocate(stream, ptr, bytes, alignment);

  if (found && record.reservation != nullptr) {
    std::scoped_lock lock{mutex_};
    if (record.arena != nullptr) {
      release_allocation(*record.arena, record.bytes, record.aligned_bytes);
    }
    release_allocation(*record.reservation, record.bytes, record.aligned_bytes);
  }
}

[[nodiscard]] bool reservation_aware_resource_adaptor::do_is_equal(
  rmm::mr::device_memory_resource const& other) const noexcept
{
  return this == std::addressof(other);
}

}  // namespace cudf::rust::memory

using namespace cudf::rust::memory;

extern "C" {
cudf_status_t cudf_memory_manager_route_current_resource(
  cudf_memory_manager_handle manager,
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

    // Routing the manager resource back through its own permanent adaptor
    // would recurse. It is already the current-resource allocation path.
    if (resource_state.owner->mr.get() == manager_state->owner->mr.get()) {
      return CUDF_SUCCESS;
    }

    auto* adaptor = manager_state->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("memory manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    std::string attach_error;
    auto route = adaptor->attach_current_resource_route(
      cudf_stream_view_from_handle(stream), resource_state.owner, attach_error);
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
  CUDF_RUST_TRY
  {
    auto* state = reinterpret_cast<current_resource_route_handle_state*>(route);
    if (state->manager != nullptr && state->manager->owner != nullptr) {
      auto* adaptor = state->manager->owner->reservation_adaptor;
      if (adaptor != nullptr) { adaptor->detach_current_resource_route(state->route); }
    }
    delete state;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_attach_reservation(cudf_memory_manager_handle manager,
                                                     cudf_stream_handle stream,
                                                     uint64_t reservation_id,
                                                     size_t limit_bytes,
                                                     cudf_reservation_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
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
      "stream reservation requires stream and memory manager to be on the "
      "same CUDA device");
    if (stream_status != CUDF_SUCCESS) { return stream_status; }
    auto* adaptor = manager_state->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("memory manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    std::string attach_error;
    auto reservation = adaptor->attach(
      cudf_stream_view_from_handle(stream), reservation_id, limit_bytes, nullptr, attach_error);
    if (reservation == nullptr) {
      set_error(attach_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state         = std::make_unique<reservation_handle_state>();
    state->manager     = std::move(manager_state);
    state->reservation = std::move(reservation);
    *out               = reinterpret_cast<cudf_reservation_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_create_reservation_arena(cudf_memory_manager_handle manager,
                                                           uint64_t arena_id,
                                                           size_t limit_bytes,
                                                           cudf_reservation_arena_handle* out)
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
    auto* adaptor = manager_state->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("memory manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state     = std::make_unique<reservation_arena_handle_state>();
    state->manager = std::move(manager_state);
    state->arena   = adaptor->create_arena(arena_id, limit_bytes);
    *out           = reinterpret_cast<cudf_reservation_arena_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_attach_reservation_to_arena(cudf_memory_manager_handle manager,
                                                              cudf_stream_handle stream,
                                                              cudf_reservation_arena_handle arena,
                                                              uint64_t reservation_id,
                                                              size_t limit_bytes,
                                                              cudf_reservation_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_NULL_CHECK(arena, "arena is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager handle is not registered");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto& arena_state = *reinterpret_cast<reservation_arena_handle_state*>(arena);
    if (arena_state.manager.get() != manager_state.get()) {
      set_error("reservation arena belongs to a different memory manager");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    cudf_status_t const stream_status = validate_stream_device_for_manager(
      stream,
      manager_state->device,
      "arena stream reservation requires stream and memory manager to be on "
      "the same CUDA device");
    if (stream_status != CUDF_SUCCESS) { return stream_status; }
    auto* adaptor = manager_state->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("memory manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    std::string attach_error;
    auto reservation = adaptor->attach(cudf_stream_view_from_handle(stream),
                                       reservation_id,
                                       limit_bytes,
                                       arena_state.arena,
                                       attach_error);
    if (reservation == nullptr) {
      set_error(attach_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state         = std::make_unique<reservation_handle_state>();
    state->manager     = std::move(manager_state);
    state->reservation = std::move(reservation);
    *out               = reinterpret_cast<cudf_reservation_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
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
    if (adaptor == nullptr) {
      set_error("reservation manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (state.query_domain_member != nullptr) {
      *out = adaptor->query_domain_member_usage(state.query_domain_member);
    } else {
      *out = adaptor->usage(state.reservation);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reservation_resize(cudf_reservation_handle reservation, size_t limit_bytes)
{
  CUDF_RUST_NULL_CHECK(reservation, "reservation is null");
  CUDF_RUST_TRY
  {
    auto& state   = *reinterpret_cast<reservation_handle_state*>(reservation);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("reservation manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    std::string resize_error;
    bool resized = false;
    if (state.query_domain_member != nullptr) {
      resized =
        adaptor->resize_query_domain_member(state.query_domain_member, limit_bytes, resize_error);
    } else {
      resized = adaptor->resize(state.reservation, limit_bytes, resize_error);
    }
    if (!resized) {
      set_error(resize_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }
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
    if (adaptor != nullptr) {
      if (state->query_domain_member != nullptr) {
        adaptor->detach_query_domain_member(state->query_domain_member);
      } else {
        adaptor->detach(state->reservation);
      }
    }
    delete state;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reservation_arena_usage(cudf_reservation_arena_handle arena,
                                           cudf_memory_resource_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(arena, "arena is null");
  CUDF_RUST_TRY
  {
    auto& state   = *reinterpret_cast<reservation_arena_handle_state*>(arena);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("reservation arena manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = adaptor->arena_usage(state.arena);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reservation_arena_resize(cudf_reservation_arena_handle arena, size_t limit_bytes)
{
  CUDF_RUST_NULL_CHECK(arena, "arena is null");
  CUDF_RUST_TRY
  {
    auto& state   = *reinterpret_cast<reservation_arena_handle_state*>(arena);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("reservation arena manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    std::string resize_error;
    if (!adaptor->resize_arena(state.arena, limit_bytes, resize_error)) {
      set_error(resize_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reservation_arena_destroy(cudf_reservation_arena_handle arena)
{
  CUDF_RUST_NULL_CHECK(arena, "arena is null");
  CUDF_RUST_TRY
  {
    auto* state   = reinterpret_cast<reservation_arena_handle_state*>(arena);
    auto* adaptor = state->manager->owner->reservation_adaptor;
    if (adaptor != nullptr) { adaptor->destroy_arena(state->arena); }
    delete state;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
