#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory/internal.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

namespace cudf::rust::memory {

std::optional<std::size_t> optional_nonzero_bytes(std::uint64_t value)
{
  if (value == 0) { return std::nullopt; }
  return static_cast<std::size_t>(value);
}

std::optional<std::size_t> optional_threshold_bytes(std::uint64_t value)
{
  return static_cast<std::size_t>(value);
}

std::unique_ptr<cuda_async_resource_box> make_cuda_async_resource_box(
  std::uint64_t initial_pool_size, std::uint64_t release_threshold)
{
  return std::make_unique<cuda_async_resource_box>(optional_nonzero_bytes(initial_pool_size),
                                                   optional_threshold_bytes(release_threshold));
}

std::shared_ptr<cuda_async_resource_box> make_shared_cuda_async_resource_box(
  std::uint64_t initial_pool_size, std::uint64_t release_threshold)
{
  return std::make_shared<cuda_async_resource_box>(optional_nonzero_bytes(initial_pool_size),
                                                   optional_threshold_bytes(release_threshold));
}

cudaMemPool_t async_pool_handle(cuda_async_resource_box* async_pool) noexcept
{
  return async_pool == nullptr ? nullptr : async_pool->resource().pool_handle();
}

void fill_pool_usage(cudf_memory_pool_usage_t& out,
                     std::uint64_t reserved_current,
                     std::uint64_t reserved_high,
                     std::uint64_t used_current,
                     std::uint64_t used_high)
{
  out                        = {};
  out.struct_size            = sizeof(cudf_memory_pool_usage_t);
  out.reserved_current_bytes = reserved_current;
  out.reserved_high_bytes    = reserved_high;
  out.used_current_bytes     = used_current;
  out.used_high_bytes        = used_high;
}

bool read_cuda_pool_usage_noexcept(cudaMemPool_t pool, cudf_memory_pool_usage_t& out) noexcept;

bool read_async_pool_usage_noexcept(cuda_async_resource_box* async_pool,
                                    cudf_memory_pool_usage_t& out) noexcept
{
  if (async_pool == nullptr) { return false; }
  return read_cuda_pool_usage_noexcept(async_pool->resource().pool_handle(), out);
}

bool read_cuda_pool_usage_noexcept(cudaMemPool_t pool, cudf_memory_pool_usage_t& out) noexcept
{
  if (pool == nullptr) { return false; }
  std::uint64_t reserved_current = 0;
  std::uint64_t reserved_high    = 0;
  std::uint64_t used_current     = 0;
  std::uint64_t used_high        = 0;
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_current) !=
      cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved_high) !=
      cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_current) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  fill_pool_usage(out, reserved_current, reserved_high, used_current, used_high);
  return true;
}

void read_async_pool_usage(cuda_async_resource_box* async_pool, cudf_memory_pool_usage_t& out)
{
  if (async_pool == nullptr) {
    throw std::runtime_error{"memory resource is not backed by a CUDA async pool"};
  }
  if (!read_async_pool_usage_noexcept(async_pool, out)) {
    throw std::runtime_error{"failed to read CUDA async pool usage"};
  }
}

void trim_async_pool_to(cuda_async_resource_box* async_pool, std::uint64_t keep_bytes)
{
  if (async_pool == nullptr) {
    throw std::runtime_error{"memory resource is not backed by a CUDA async pool"};
  }
  CUDF_CUDA_TRY(cudaMemPoolTrimTo(async_pool->resource().pool_handle(), keep_bytes));
}

void fill_failure_info(cudf_memory_failure_info_t& out,
                       int32_t device_ordinal,
                       std::size_t request_bytes,
                       cudaMemPool_t pool,
                       cudf_memory_failure_source_t failure_source,
                       memory_failure_domain_identity identity,
                       std::uint32_t callback_attempt,
                       std::uint32_t trim_retry_count)
{
  out                  = {};
  out.struct_size      = sizeof(cudf_memory_failure_info_t);
  out.device_ordinal   = device_ordinal;
  out.callback_attempt = callback_attempt;
  out.trim_retry_count = trim_retry_count;
  out.requested_bytes  = static_cast<std::uint64_t>(request_bytes);
  out.failure_source   = static_cast<std::uint32_t>(failure_source);
  if (identity.kind != CUDF_MEMORY_FAILURE_DOMAIN_NONE) {
    out.flags |= CUDF_MEMORY_FAILURE_INFO_HAS_ALLOCATION_DOMAIN;
    out.reserved[0] = static_cast<std::uint64_t>(identity.kind);
    out.reserved[1] = identity.primary_id;
    out.reserved[2] = identity.secondary_id;
  }
  cudf_memory_pool_usage_t usage{};
  if (read_cuda_pool_usage_noexcept(pool, usage)) {
    out.flags |= CUDF_MEMORY_FAILURE_INFO_HAS_POOL_USAGE;
    out.pool_reserved_current_bytes = usage.reserved_current_bytes;
    out.pool_reserved_high_bytes    = usage.reserved_high_bytes;
    out.pool_used_current_bytes     = usage.used_current_bytes;
    out.pool_used_high_bytes        = usage.used_high_bytes;
  }
}

void update_atomic_peak(std::atomic<std::size_t>& peak, std::size_t value)
{
  auto observed = peak.load(std::memory_order_relaxed);
  while (observed < value &&
         !peak.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {}
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
  out                                 = {};
  out.struct_size                     = sizeof(cudf_memory_resource_usage_t);
  out.allocation_limit_bytes          = allocation_limit;
  out.current_bytes                   = current_bytes;
  out.peak_bytes                      = peak_bytes;
  out.total_bytes                     = total_bytes;
  out.current_aligned_bytes           = current_aligned_bytes;
  out.peak_aligned_bytes              = peak_aligned_bytes;
  out.total_allocations               = total_allocations;
  out.current_allocations             = current_allocations;
  out.peak_allocations                = peak_allocations;
  out.failed_allocation_count         = failed_allocation_count;
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
    out.struct_size             = sizeof(cudf_memory_resource_usage_t);
    out.allocation_limit_bytes  = allocation_limit_;
    out.current_bytes           = current_bytes_.load(std::memory_order_relaxed);
    out.peak_bytes              = peak_bytes_.load(std::memory_order_relaxed);
    out.total_bytes             = total_bytes_.load(std::memory_order_relaxed);
    out.current_aligned_bytes   = current_aligned_bytes_.load(std::memory_order_relaxed);
    out.peak_aligned_bytes      = peak_aligned_bytes_.load(std::memory_order_relaxed);
    out.total_allocations       = total_allocations_.load(std::memory_order_relaxed);
    out.current_allocations     = current_allocations_.load(std::memory_order_relaxed);
    out.peak_allocations        = peak_allocations_.load(std::memory_order_relaxed);
    out.failed_allocation_count = failed_allocation_count_.load(std::memory_order_relaxed);
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

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    auto const aligned_bytes = cudf::rust::reservation::allocation_charge(bytes, alignment);
    auto old_aligned         = current_aligned_bytes_.load(std::memory_order_relaxed);
    std::size_t new_aligned  = 0;
    while (true) {
      if (old_aligned > allocation_limit_ || aligned_bytes > allocation_limit_ - old_aligned) {
        record_failed_allocation(bytes);
        throw rmm::out_of_memory{"budget-tracking memory resource allocation limit exceeded"};
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
    auto const aligned_bytes = cudf::rust::reservation::allocation_charge(bytes, alignment);
    upstream_.deallocate(stream, ptr, bytes, alignment);
    current_aligned_bytes_ -= aligned_bytes;
    current_bytes_ -= bytes;
    current_allocations_ -= 1;
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
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

class bounded_failure_callback_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  bounded_failure_callback_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                            cudf_memory_failure_callback_t callback,
                                            void* callback_arg,
                                            cudaMemPool_t pool,
                                            int32_t device_ordinal,
                                            allocator_event_ring* allocator_events,
                                            std::atomic<bool>* callback_panicked)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)},
      callback_{callback},
      callback_arg_{callback_arg},
      pool_{pool},
      device_ordinal_{device_ordinal},
      allocator_events_{allocator_events},
      callback_panicked_{callback_panicked}
  {
  }

  bounded_failure_callback_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                            cudf_memory_failure_callback_v2_t callback,
                                            void* callback_arg,
                                            cudaMemPool_t pool,
                                            int32_t device_ordinal,
                                            allocator_event_ring* allocator_events,
                                            std::atomic<bool>* callback_panicked)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)},
      callback_v2_{callback},
      callback_arg_{callback_arg},
      pool_{pool},
      device_ordinal_{device_ordinal},
      allocator_events_{allocator_events},
      callback_panicked_{callback_panicked}
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    try {
      return upstream_.allocate(stream, bytes, alignment);
    } catch (rmm::out_of_memory const& error) {
      auto const* attributed_error = dynamic_cast<attributed_out_of_memory const*>(&error);
      auto const failure_source    = attributed_error == nullptr
                                       ? CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL
                                       : attributed_error->source();
      auto const failure_identity = attributed_error == nullptr ? memory_failure_domain_identity{}
                                                                : attributed_error->identity();
      auto const failure_pool = attributed_error != nullptr && attributed_error->pool() != nullptr
                                  ? attributed_error->pool()
                                  : pool_;
      cudf_memory_failure_info_t info{};
      fill_failure_info(
        info, device_ordinal_, bytes, failure_pool, failure_source, failure_identity, 0, 0);
      if (callback_v2_ != nullptr) {
        static_cast<void>(callback_v2_(&info, callback_arg_));
      } else if (callback_ != nullptr) {
        static_cast<void>(callback_(bytes, callback_arg_));
      }
      if (allocator_events_ != nullptr) {
        allocator_events_->record(info, failure_identity, CUDF_MEMORY_FAILURE_ACTION_DECLINE);
      }
      throw;
    }
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  cudf_memory_failure_callback_t callback_       = nullptr;
  cudf_memory_failure_callback_v2_t callback_v2_ = nullptr;
  void* callback_arg_                            = nullptr;
  cudaMemPool_t pool_                            = nullptr;
  int32_t device_ordinal_                        = -1;
  allocator_event_ring* allocator_events_        = nullptr;
  std::atomic<bool>* callback_panicked_          = nullptr;
};

std::unique_ptr<rmm::mr::device_memory_resource> make_bounded_failure_callback_resource(
  rmm::mr::device_memory_resource* upstream,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudaMemPool_t pool,
  int32_t device_ordinal,
  allocator_event_ring* allocator_events,
  std::atomic<bool>* callback_panicked)
{
  return std::make_unique<bounded_failure_callback_resource_adaptor>(
    upstream, callback, callback_arg, pool, device_ordinal, allocator_events, callback_panicked);
}

std::unique_ptr<rmm::mr::device_memory_resource> make_bounded_failure_callback_resource(
  rmm::mr::device_memory_resource* upstream,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudaMemPool_t pool,
  int32_t device_ordinal,
  allocator_event_ring* allocator_events,
  std::atomic<bool>* callback_panicked)
{
  return std::make_unique<bounded_failure_callback_resource_adaptor>(
    upstream, callback, callback_arg, pool, device_ordinal, allocator_events, callback_panicked);
}

std::mutex& memory_resource_registry_mutex()
{
  static auto& mutex = *new std::mutex;
  return mutex;
}

std::unordered_map<rmm::mr::device_memory_resource*, std::weak_ptr<void>>&
memory_resource_registry()
{
  static auto& registry =
    *new std::unordered_map<rmm::mr::device_memory_resource*, std::weak_ptr<void>>;
  return registry;
}

memory_resource_handle_state& memory_resource_state_from_handle(cudf_memory_resource_handle handle)
{
  return *reinterpret_cast<memory_resource_handle_state*>(handle);
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

}  // namespace cudf::rust::memory

using namespace cudf::rust::memory;

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
    auto state           = std::make_unique<memory_resource_handle_state>();
    state->owner         = std::make_shared<memory_resource_owner>();
    state->owner->device = current_cuda_device();
    state->owner->mr =
      std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>();
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
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
    auto state   = std::make_unique<memory_resource_handle_state>();
    state->owner = std::make_shared<memory_resource_owner>();
    auto& owner  = *state->owner;
    owner.device = current_cuda_device();
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
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_create_cuda_async(size_t initial_pool_size,
                                                     uint64_t release_threshold,
                                                     cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto state               = std::make_unique<memory_resource_handle_state>();
    state->owner             = std::make_shared<memory_resource_owner>();
    auto async_pool          = make_cuda_async_resource_box(initial_pool_size, release_threshold);
    auto* async_pool_ptr     = async_pool.get();
    state->owner->device     = current_cuda_device();
    state->owner->async_pool = async_pool_ptr;
    state->owner->mr         = std::move(async_pool);
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
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
    auto state              = std::make_unique<memory_resource_handle_state>();
    state->owner            = std::make_shared<memory_resource_owner>();
    auto& owner             = *state->owner;
    auto& upstream_owner    = memory_resource_owner_from_handle(upstream);
    owner.upstream_lifetime = cudf_memory_resource_lifetime_from_handle(upstream);
    owner.async_pool        = upstream_owner.async_pool;
    owner.device            = upstream_owner.device;
    auto* upstream_mr       = cudf_memory_resource_raw_from_handle(upstream);
    owner.mr =
      std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::limiting_resource_adaptor>>(
        rmm::mr::any_device_resource_from_raw(upstream_mr), allocation_limit);
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_create_failure_callback(cudf_memory_resource_handle upstream,
                                                           cudf_memory_failure_callback_t callback,
                                                           void* callback_arg,
                                                           cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(upstream, "upstream is null");
  CUDF_RUST_NULL_CHECK(callback, "callback is null");
  CUDF_RUST_TRY
  {
    auto state              = std::make_unique<memory_resource_handle_state>();
    state->owner            = std::make_shared<memory_resource_owner>();
    auto& owner             = *state->owner;
    auto& upstream_owner    = memory_resource_owner_from_handle(upstream);
    owner.upstream_lifetime = cudf_memory_resource_lifetime_from_handle(upstream);
    owner.async_pool        = upstream_owner.async_pool;
    owner.device            = upstream_owner.device;
    auto* upstream_mr       = cudf_memory_resource_raw_from_handle(upstream);
    owner.mr                = make_bounded_failure_callback_resource(upstream_mr,
                                                      callback,
                                                      callback_arg,
                                                      async_pool_handle(owner.async_pool),
                                                      owner.device,
                                                      nullptr);
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_create_failure_callback_v2(
  cudf_memory_resource_handle upstream,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(upstream, "upstream is null");
  CUDF_RUST_NULL_CHECK(callback, "callback is null");
  CUDF_RUST_TRY
  {
    auto state              = std::make_unique<memory_resource_handle_state>();
    state->owner            = std::make_shared<memory_resource_owner>();
    auto& owner             = *state->owner;
    auto& upstream_owner    = memory_resource_owner_from_handle(upstream);
    owner.upstream_lifetime = cudf_memory_resource_lifetime_from_handle(upstream);
    owner.async_pool        = upstream_owner.async_pool;
    owner.device            = upstream_owner.device;
    auto* upstream_mr       = cudf_memory_resource_raw_from_handle(upstream);
    owner.mr                = make_bounded_failure_callback_resource(upstream_mr,
                                                      callback,
                                                      callback_arg,
                                                      async_pool_handle(owner.async_pool),
                                                      owner.device,
                                                      nullptr);
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
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
    auto& upstream_owner    = memory_resource_owner_from_handle(upstream);
    owner.upstream_lifetime = cudf_memory_resource_lifetime_from_handle(upstream);
    owner.async_pool        = upstream_owner.async_pool;
    owner.device            = upstream_owner.device;
    auto* upstream_mr       = cudf_memory_resource_raw_from_handle(upstream);
    auto tracker =
      std::make_unique<budget_tracking_resource_adaptor>(upstream_mr, allocation_limit);
    owner.budget_tracker = tracker.get();
    owner.mr             = std::move(tracker);
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
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

cudf_status_t cudf_memory_resource_pool_usage(cudf_memory_resource_handle handle,
                                              cudf_memory_pool_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    if (owner.async_pool == nullptr) {
      set_error("memory resource is not backed by a CUDA async pool");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    read_async_pool_usage(owner.async_pool, *out);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_resource_trim_to(cudf_memory_resource_handle handle, uint64_t keep_bytes)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    if (owner.async_pool == nullptr) {
      set_error("memory resource is not backed by a CUDA async pool");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    trim_async_pool_to(owner.async_pool, keep_bytes);
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
    auto* state = reinterpret_cast<memory_resource_handle_state*>(handle);
    auto& owner = *state->owner;
    if (!state->non_owning_alias &&
        cudf_tracked_current_device_resource_if_installed_raw() == owner.mr.get()) {
      set_error(
        "cannot destroy a memory resource while it is installed as the "
        "current device resource");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    cudf::rust::test_support::record_memory_resource_destroy(owner.mr.get());
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

}  // extern "C"
