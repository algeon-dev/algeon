#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory_resource.h"
#include "memory_resource_compat.hpp"
#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <new>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace {
void update_atomic_peak(std::atomic<std::size_t>& peak, std::size_t value)
{
  auto observed = peak.load(std::memory_order_relaxed);
  while (observed < value &&
         !peak.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
  }
}

void update_peak(std::size_t& peak, std::size_t value) { peak = std::max(peak, value); }

void fill_usage(cudf_memory_resource_usage_t& out,
                std::size_t allocation_limit,
                std::size_t current_bytes,
                std::size_t peak_bytes,
                std::size_t total_bytes,
                std::size_t current_aligned_bytes,
                std::size_t peak_aligned_bytes,
                std::size_t total_allocations,
                std::size_t current_allocations,
                std::size_t peak_allocations,
                std::size_t failed_allocation_count,
                std::size_t largest_failed_allocation_bytes)
{
  out                                = {};
  out.struct_size                    = sizeof(cudf_memory_resource_usage_t);
  out.allocation_limit_bytes         = allocation_limit;
  out.current_bytes                  = current_bytes;
  out.peak_bytes                     = peak_bytes;
  out.total_bytes                    = total_bytes;
  out.current_aligned_bytes          = current_aligned_bytes;
  out.peak_aligned_bytes             = peak_aligned_bytes;
  out.total_allocations              = total_allocations;
  out.current_allocations            = current_allocations;
  out.peak_allocations               = peak_allocations;
  out.failed_allocation_count        = failed_allocation_count;
  out.largest_failed_allocation_bytes = largest_failed_allocation_bytes;
}

class budget_tracking_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  budget_tracking_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                   std::size_t allocation_limit)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)},
      allocation_limit_{allocation_limit}
  {
  }

  [[nodiscard]] cudf_memory_resource_usage_t usage() const
  {
    cudf_memory_resource_usage_t out{};
    out.struct_size                     = sizeof(cudf_memory_resource_usage_t);
    out.allocation_limit_bytes          = allocation_limit_;
    out.current_bytes                   = current_bytes_.load(std::memory_order_relaxed);
    out.peak_bytes                      = peak_bytes_.load(std::memory_order_relaxed);
    out.total_bytes                     = total_bytes_.load(std::memory_order_relaxed);
    out.current_aligned_bytes           = current_aligned_bytes_.load(std::memory_order_relaxed);
    out.peak_aligned_bytes              = peak_aligned_bytes_.load(std::memory_order_relaxed);
    out.total_allocations               = total_allocations_.load(std::memory_order_relaxed);
    out.current_allocations             = current_allocations_.load(std::memory_order_relaxed);
    out.peak_allocations                = peak_allocations_.load(std::memory_order_relaxed);
    out.failed_allocation_count         = failed_allocation_count_.load(std::memory_order_relaxed);
    out.largest_failed_allocation_bytes =
      largest_failed_allocation_bytes_.load(std::memory_order_relaxed);
    return out;
  }

 private:
  void record_failed_allocation(std::size_t bytes) noexcept
  {
    failed_allocation_count_.fetch_add(1, std::memory_order_relaxed);
    update_atomic_peak(largest_failed_allocation_bytes_, bytes);
  }

  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream,
                    std::size_t alignment) override
  {
    auto const aligned_bytes = rmm::align_up(bytes, alignment);
    auto old_aligned         = current_aligned_bytes_.load(std::memory_order_relaxed);
    std::size_t new_aligned  = 0;
    while (true) {
      if (old_aligned > allocation_limit_ || aligned_bytes > allocation_limit_ - old_aligned) {
        record_failed_allocation(bytes);
        throw std::bad_alloc{};
      }
      new_aligned = old_aligned + aligned_bytes;
      if (current_aligned_bytes_.compare_exchange_weak(
            old_aligned, new_aligned, std::memory_order_relaxed)) {
        break;
      }
    }

    void* ptr = nullptr;
    try {
      ptr = upstream_.allocate(stream, bytes, alignment);
    } catch (...) {
      current_aligned_bytes_ -= aligned_bytes;
      record_failed_allocation(bytes);
      throw;
    }

    auto const old_requested = current_bytes_.fetch_add(bytes);
    auto const new_requested = old_requested + bytes;
    auto const old_allocs    = current_allocations_.fetch_add(1);
    auto const new_allocs    = old_allocs + 1;

    update_atomic_peak(peak_aligned_bytes_, new_aligned);
    update_atomic_peak(peak_bytes_, new_requested);
    update_atomic_peak(peak_allocations_, new_allocs);
    total_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    auto const aligned_bytes = rmm::align_up(bytes, alignment);
    upstream_.deallocate(stream, ptr, bytes, alignment);
    current_aligned_bytes_ -= aligned_bytes;
    current_bytes_ -= bytes;
    current_allocations_ -= 1;
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::size_t allocation_limit_;
  std::atomic<std::size_t> current_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
  std::atomic<std::size_t> total_bytes_{0};
  std::atomic<std::size_t> current_aligned_bytes_{0};
  std::atomic<std::size_t> peak_aligned_bytes_{0};
  std::atomic<std::size_t> total_allocations_{0};
  std::atomic<std::size_t> current_allocations_{0};
  std::atomic<std::size_t> peak_allocations_{0};
  std::atomic<std::size_t> failed_allocation_count_{0};
  std::atomic<std::size_t> largest_failed_allocation_bytes_{0};
};

struct reservation_state {
  std::uint64_t id              = 0;
  std::size_t allocation_limit  = 0;
  void* stream_key              = nullptr;
  bool attached                 = true;
  std::size_t current_bytes     = 0;
  std::size_t peak_bytes        = 0;
  std::size_t total_bytes       = 0;
  std::size_t current_aligned_bytes = 0;
  std::size_t peak_aligned_bytes    = 0;
  std::size_t total_allocations     = 0;
  std::size_t current_allocations   = 0;
  std::size_t peak_allocations      = 0;
  std::size_t failed_allocation_count = 0;
  std::size_t largest_failed_allocation_bytes = 0;
};

class reservation_aware_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  explicit reservation_aware_resource_adaptor(rmm::mr::device_memory_resource* upstream)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)}
  {
  }

  std::shared_ptr<reservation_state> attach(rmm::cuda_stream_view stream,
                                            std::uint64_t reservation_id,
                                            std::size_t limit_bytes,
                                            std::string& error)
  {
    auto const key = stream_key(stream);
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

    auto state              = std::make_shared<reservation_state>();
    state->id              = reservation_id;
    state->allocation_limit = limit_bytes;
    state->stream_key      = key;
    reservations_by_stream_[key] = state;
    return state;
  }

  void detach(std::shared_ptr<reservation_state> const& state)
  {
    if (state == nullptr) { return; }
    std::scoped_lock lock{mutex_};
    state->attached = false;
    auto const existing = reservations_by_stream_.find(state->stream_key);
    if (existing != reservations_by_stream_.end()) {
      auto existing_state = existing->second.lock();
      if (existing_state == nullptr || existing_state.get() == state.get()) {
        reservations_by_stream_.erase(existing);
      }
    }
  }

  cudf_memory_resource_usage_t usage(std::shared_ptr<reservation_state> const& state)
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

 private:
  struct allocation_record {
    std::size_t bytes = 0;
    std::size_t aligned_bytes = 0;
    std::shared_ptr<reservation_state> reservation;
  };

  static void* stream_key(rmm::cuda_stream_view stream)
  {
    return reinterpret_cast<void*>(stream.value());
  }

  void record_failed_allocation(reservation_state& state, std::size_t bytes)
  {
    state.failed_allocation_count += 1;
    update_peak(state.largest_failed_allocation_bytes, bytes);
  }

  std::shared_ptr<reservation_state> reservation_for_stream(rmm::cuda_stream_view stream)
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

  void reserve_allocation(reservation_state& state, std::size_t bytes, std::size_t aligned_bytes)
  {
    if (state.current_aligned_bytes > state.allocation_limit ||
        aligned_bytes > state.allocation_limit - state.current_aligned_bytes) {
      record_failed_allocation(state, bytes);
      throw std::bad_alloc{};
    }
    state.current_aligned_bytes += aligned_bytes;
  }

  void commit_allocation(reservation_state& state, std::size_t bytes, std::size_t aligned_bytes)
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

  void rollback_reserved_allocation(reservation_state& state,
                                    std::size_t bytes,
                                    std::size_t aligned_bytes)
  {
    if (state.current_aligned_bytes >= aligned_bytes) {
      state.current_aligned_bytes -= aligned_bytes;
    } else {
      state.current_aligned_bytes = 0;
    }
    record_failed_allocation(state, bytes);
  }

  void release_allocation(reservation_state& state, std::size_t bytes, std::size_t aligned_bytes)
  {
    state.current_bytes = state.current_bytes >= bytes ? state.current_bytes - bytes : 0;
    state.current_aligned_bytes =
      state.current_aligned_bytes >= aligned_bytes ? state.current_aligned_bytes - aligned_bytes : 0;
    state.current_allocations =
      state.current_allocations > 0 ? state.current_allocations - 1 : 0;
  }

  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream,
                    std::size_t alignment) override
  {
    auto const aligned_bytes = rmm::align_up(bytes, alignment);
    std::shared_ptr<reservation_state> reservation;
    {
      std::scoped_lock lock{mutex_};
      reservation = reservation_for_stream(stream);
      if (reservation != nullptr) { reserve_allocation(*reservation, bytes, aligned_bytes); }
    }

    void* ptr = nullptr;
    try {
      ptr = upstream_.allocate(stream, bytes, alignment);
    } catch (...) {
      if (reservation != nullptr) {
        std::scoped_lock lock{mutex_};
        rollback_reserved_allocation(*reservation, bytes, aligned_bytes);
      }
      throw;
    }

    if (reservation != nullptr) {
      std::scoped_lock lock{mutex_};
      commit_allocation(*reservation, bytes, aligned_bytes);
      allocations_[ptr] = allocation_record{bytes, aligned_bytes, reservation};
    }
    return ptr;
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
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
      release_allocation(*record.reservation, record.bytes, record.aligned_bytes);
    }
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::mutex mutex_;
  std::unordered_map<void*, std::weak_ptr<reservation_state>> reservations_by_stream_;
  std::unordered_map<void*, allocation_record> allocations_;
};

struct memory_resource_owner {
  std::shared_ptr<rmm::mr::device_memory_resource> upstream;
  std::shared_ptr<rmm::mr::device_memory_resource> pool;
  std::shared_ptr<void> upstream_lifetime;
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  rmm::mr::statistics_resource_adaptor* stats = nullptr;
  budget_tracking_resource_adaptor* budget_tracker                         = nullptr;
  reservation_aware_resource_adaptor* reservation_adaptor                   = nullptr;
};

struct memory_resource_handle_state {
  std::shared_ptr<memory_resource_owner> owner;
  bool non_owning_alias = false;
};

struct device_memory_manager_state {
  int32_t device = -1;
  std::shared_ptr<memory_resource_owner> owner;
};

struct reservation_handle_state {
  std::shared_ptr<device_memory_manager_state> manager;
  std::shared_ptr<reservation_state> reservation;
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
           .emplace(device,
                    std::make_shared<
                      rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>())
           .first;
  }
  return it->second.get();
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

std::mutex& memory_manager_registry_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<int32_t, std::shared_ptr<device_memory_manager_state>>&
memory_manager_registry()
{
  static std::unordered_map<int32_t, std::shared_ptr<device_memory_manager_state>> registry;
  return registry;
}

std::shared_ptr<device_memory_manager_state> memory_manager_from_handle(
  cudf_memory_manager_handle handle)
{
  std::scoped_lock lock{memory_manager_registry_mutex()};
  for (auto const& [_, manager] : memory_manager_registry()) {
    if (manager.get() == handle) { return manager; }
  }
  return {};
}

std::shared_ptr<device_memory_manager_state> make_memory_manager_for_current_device(
  int32_t device)
{
  auto manager    = std::make_shared<device_memory_manager_state>();
  manager->device = device;
  manager->owner  = std::make_shared<memory_resource_owner>();

  auto& owner    = *manager->owner;
  owner.upstream =
    std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>();
  owner.pool = std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::pool_memory_resource>>(
    rmm::mr::any_device_resource_from_raw(owner.upstream.get()), 0);
  auto adaptor = std::make_unique<reservation_aware_resource_adaptor>(owner.pool.get());
  owner.reservation_adaptor = adaptor.get();
  owner.mr                  = std::move(adaptor);
  register_memory_resource_lifetime(manager->owner);

  std::scoped_lock current_lock{cudf_current_device_resource_mutex()};
  cudf_set_current_device_resource_raw(owner.mr.get());
  return manager;
}
}  // namespace

namespace rmm {
const char* bad_alloc::what() const noexcept { return _what.c_str(); }
}  // namespace rmm

rmm::mr::device_memory_resource* cudf_tracked_current_device_resource_raw()
{
  auto const device = current_cuda_device();
  auto& identities  = current_memory_resource_identities();
  auto const found  = identities.find(device);
  if (found != identities.end()) { return found->second; }

  auto* initial = default_memory_resource_for_device(device);
  cudf_set_current_device_resource_raw(initial);
  return initial;
}

void cudf_set_current_device_resource_raw(rmm::mr::device_memory_resource* target)
{
  auto const device = current_cuda_device();
  rmm::mr::set_current_device_resource(
    rmm::mr::any_device_resource_from_raw(target));
  current_memory_resource_identities()[device] = target;
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
    state->owner->mr =
      std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>();
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
    owner.upstream =
      std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>();
    auto upstream = rmm::mr::any_device_resource_from_raw(owner.upstream.get());
    if (maximum_pool_size > 0) {
      owner.mr =
        std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::pool_memory_resource>>(
          std::move(upstream), initial_pool_size, maximum_pool_size);

    } else {
      owner.mr =
        std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::pool_memory_resource>>(
          std::move(upstream), initial_pool_size);
    }
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_create_limiting(cudf_memory_resource_handle upstream,
                                                   size_t allocation_limit,
                                                   cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(upstream, "upstream is null");
  CUDF_RUST_TRY
  {
    auto state                    = std::make_unique<memory_resource_handle_state>();
    state->owner                  = std::make_shared<memory_resource_owner>();
    auto& owner                   = *state->owner;
    owner.upstream_lifetime       = cudf_memory_resource_lifetime_from_handle(upstream);
    auto* upstream_mr             = cudf_memory_resource_raw_from_handle(upstream);
    owner.mr =
      std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::limiting_resource_adaptor>>(
        rmm::mr::any_device_resource_from_raw(upstream_mr),
        allocation_limit);
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_create_budget_tracking(cudf_memory_resource_handle upstream,
                                                          size_t allocation_limit,
                                                          cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(upstream, "upstream is null");
  CUDF_RUST_TRY
  {
    auto state              = std::make_unique<memory_resource_handle_state>();
    state->owner            = std::make_shared<memory_resource_owner>();
    auto& owner             = *state->owner;
    owner.upstream_lifetime = cudf_memory_resource_lifetime_from_handle(upstream);
    auto* upstream_mr       = cudf_memory_resource_raw_from_handle(upstream);
    auto tracker = std::make_unique<budget_tracking_resource_adaptor>(upstream_mr,
                                                                      allocation_limit);
    owner.budget_tracker = tracker.get();
    owner.mr             = std::move(tracker);
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_usage(cudf_memory_resource_handle handle,
                                         cudf_memory_resource_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    if (owner.budget_tracker == nullptr) {
      set_error("memory resource does not expose budget tracking usage");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = owner.budget_tracker->usage();
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
    auto* state = static_cast<memory_resource_handle_state*>(handle);
    auto& owner = *state->owner;
    if (!state->non_owning_alias && cudf_tracked_current_device_resource_raw() == owner.mr.get()) {
      set_error(
        "cannot destroy a memory resource while it is installed as the current device resource");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    delete state;
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

cudf_status_t cudf_memory_manager_for_device(int32_t device, cudf_memory_manager_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    if (device < 0) {
      set_error("device must be >= 0");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    int32_t device_count = 0;
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

    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }

    std::scoped_lock lock{memory_manager_registry_mutex()};
    auto& registry = memory_manager_registry();
    auto it        = registry.find(device);
    if (it == registry.end()) {
      it = registry.emplace(device, make_memory_manager_for_current_device(device)).first;
    }
    *out = static_cast<cudf_memory_manager_handle>(it->second.get());
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

    auto state             = std::make_unique<memory_resource_handle_state>();
    state->owner           = manager_state->owner;
    state->non_owning_alias = true;
    *out = static_cast<cudf_memory_resource_handle>(state.release());
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
    auto* adaptor = manager_state->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("memory manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    std::string attach_error;
    auto reservation =
      adaptor->attach(cudf_stream_view_from_handle(stream), reservation_id, limit_bytes, attach_error);
    if (reservation == nullptr) {
      set_error(attach_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state         = std::make_unique<reservation_handle_state>();
    state->manager     = std::move(manager_state);
    state->reservation = std::move(reservation);
    *out = static_cast<cudf_reservation_handle>(state.release());
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
    auto& state = *static_cast<reservation_handle_state*>(reservation);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("reservation manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = adaptor->usage(state.reservation);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reservation_detach(cudf_reservation_handle reservation)
{
  CUDF_RUST_NULL_CHECK(reservation, "reservation is null");
  CUDF_RUST_TRY
  {
    auto* state = static_cast<reservation_handle_state*>(reservation);
    auto* adaptor = state->manager->owner->reservation_adaptor;
    if (adaptor != nullptr) { adaptor->detach(state->reservation); }
    delete state;
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
    owner.upstream =
      std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>();
    auto stats =
      std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::statistics_resource_adaptor>>(
        rmm::mr::any_device_resource_from_raw(owner.upstream.get()));
    owner.stats = &stats->resource();
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
    cudf_set_current_device_resource_raw(owner.mr.get());
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
    cudf_set_current_device_resource_raw(static_cast<rmm::mr::device_memory_resource*>(raw_mr));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
