#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory_resource.h"
#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
void update_atomic_peak(std::atomic<std::size_t>& peak, std::size_t value)
{
  auto observed = peak.load(std::memory_order_relaxed);
  while (observed < value &&
         !peak.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
  }
}

class budget_tracking_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  budget_tracking_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                   std::size_t allocation_limit)
    : upstream_{rmm::device_async_resource_ref{*upstream}},
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

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    auto const aligned_bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto old_aligned         = current_aligned_bytes_.load(std::memory_order_relaxed);
    std::size_t new_aligned  = 0;
    while (true) {
      if (old_aligned > allocation_limit_ || aligned_bytes > allocation_limit_ - old_aligned) {
        record_failed_allocation(bytes);
        auto const msg = std::string("Exceeded memory limit (failed to allocate ") +
                         rmm::detail::format_bytes(bytes) + ")";
        RMM_FAIL(msg.c_str(), rmm::out_of_memory);
      }
      new_aligned = old_aligned + aligned_bytes;
      if (current_aligned_bytes_.compare_exchange_weak(
            old_aligned, new_aligned, std::memory_order_relaxed)) {
        break;
      }
    }

    void* ptr = nullptr;
    try {
      ptr = upstream_.allocate(stream, bytes);
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

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    auto const aligned_bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    upstream_.deallocate(stream, ptr, bytes);
    current_aligned_bytes_ -= aligned_bytes;
    current_bytes_ -= bytes;
    current_allocations_ -= 1;
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  rmm::device_async_resource_ref upstream_;
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

struct memory_resource_owner {
  std::shared_ptr<rmm::mr::cuda_memory_resource> upstream;
  std::shared_ptr<void> upstream_lifetime;
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource>* stats = nullptr;
  budget_tracking_resource_adaptor* budget_tracker                         = nullptr;
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
    owner.mr = std::make_unique<rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>>(
      upstream_mr, allocation_limit);
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
