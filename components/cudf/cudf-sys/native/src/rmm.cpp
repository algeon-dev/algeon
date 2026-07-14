#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory_resource.h"
#include "memory_resource_compat.hpp"
#include "shim_common.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda.h>
#include <cuda/memory_resource>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace {

constexpr std::uint32_t k_max_failure_callback_retries = 8;
constexpr std::uint64_t k_default_async_release_threshold =
  std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t k_trim_retry_idle_to_request_ratio = 10;

using cuda_async_resource_box =
  rmm::mr::device_memory_resource_box<rmm::mr::cuda_async_memory_resource>;

struct memory_failure_domain_identity {
  cudf_memory_failure_domain_t kind = CUDF_MEMORY_FAILURE_DOMAIN_NONE;
  std::uint64_t primary_id          = 0;
  std::uint64_t secondary_id        = 0;
};

#ifndef NDEBUG
struct query_domain_route_pause_state {
  std::atomic<bool> armed{false};
  std::atomic<bool> reached{false};
  std::atomic<bool> released{false};
  std::atomic<std::uint64_t> target_domain_id{0};
};

query_domain_route_pause_state& query_domain_route_pause()
{
  static query_domain_route_pause_state state;
  return state;
}

void pause_query_domain_route_if_armed(std::uint64_t domain_id) noexcept
{
  auto& state = query_domain_route_pause();
  if (!state.armed.load(std::memory_order_acquire)) { return; }
  if (state.target_domain_id.load(std::memory_order_relaxed) != domain_id) { return; }
  state.reached.store(true, std::memory_order_release);
  while (!state.released.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  state.armed.store(false, std::memory_order_release);
}
#endif

class attributed_out_of_memory final : public rmm::out_of_memory {
 public:
  attributed_out_of_memory(char const* message,
                           cudf_memory_failure_source_t source,
                           memory_failure_domain_identity identity,
                           cudaMemPool_t pool = nullptr)
    : rmm::out_of_memory{message}, source_{source}, identity_{identity}, pool_{pool}
  {
  }

  [[nodiscard]] cudf_memory_failure_source_t source() const noexcept { return source_; }
  [[nodiscard]] memory_failure_domain_identity identity() const noexcept { return identity_; }
  [[nodiscard]] cudaMemPool_t pool() const noexcept { return pool_; }

 private:
  cudf_memory_failure_source_t source_;
  memory_failure_domain_identity identity_;
  cudaMemPool_t pool_;
};

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

bool async_pool_has_trim_retry_headroom(cudf_memory_pool_usage_t const& usage,
                                        std::size_t request_bytes);
bool trim_cuda_pool_to_zero_noexcept(cudaMemPool_t pool) noexcept;

bool trim_cuda_pool_before_retry(cudaMemPool_t pool, std::size_t request_bytes) noexcept
{
  if (pool == nullptr || request_bytes == 0) { return false; }
  cudf_memory_pool_usage_t usage{};
  if (!read_cuda_pool_usage_noexcept(pool, usage)) { return false; }
  if (!async_pool_has_trim_retry_headroom(usage, request_bytes)) { return false; }
  return trim_cuda_pool_to_zero_noexcept(pool);
}

bool trim_cuda_pool_to_zero_noexcept(cudaMemPool_t pool) noexcept
{
  if (pool == nullptr) { return false; }
  if (cudaMemPoolTrimTo(pool, 0) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

bool async_pool_has_trim_retry_headroom(cudf_memory_pool_usage_t const& usage,
                                        std::size_t request_bytes)
{
  if (request_bytes == 0 || usage.reserved_current_bytes <= usage.used_current_bytes) {
    return false;
  }
  auto const idle_bytes = usage.reserved_current_bytes - usage.used_current_bytes;
  return idle_bytes / static_cast<std::uint64_t>(request_bytes) >=
         k_trim_retry_idle_to_request_ratio;
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
  out                   = {};
  out.struct_size       = sizeof(cudf_memory_failure_info_t);
  out.device_ordinal    = device_ordinal;
  out.callback_attempt  = callback_attempt;
  out.trim_retry_count  = trim_retry_count;
  out.requested_bytes   = static_cast<std::uint64_t>(request_bytes);
  out.failure_source    = static_cast<std::uint32_t>(failure_source);
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

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    auto const aligned_bytes = rmm::align_up(bytes, alignment);
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
    auto const aligned_bytes = rmm::align_up(bytes, alignment);
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

struct query_domain_member_counters {
  std::uint64_t id = 0;
  std::atomic<bool> attached{true};
  std::atomic<std::size_t> allocation_limit{0};
  std::atomic<std::size_t> current_bytes{0};
  std::atomic<std::size_t> peak_bytes{0};
  std::atomic<std::size_t> total_bytes{0};
  std::atomic<std::size_t> current_aligned_bytes{0};
  std::atomic<std::size_t> peak_aligned_bytes{0};
  std::atomic<std::size_t> total_allocations{0};
  std::atomic<std::size_t> current_allocations{0};
  std::atomic<std::size_t> peak_allocations{0};
  std::atomic<std::size_t> failed_allocation_count{0};
  std::atomic<std::size_t> largest_failed_allocation_bytes{0};
  std::atomic<std::size_t> routed_pending_allocations{0};
  std::atomic<std::size_t> routed_current_allocations{0};
  std::atomic<std::size_t> routed_total_allocations{0};
  std::atomic<std::size_t> direct_total_allocations{0};
};

class query_domain_resource final : public rmm::mr::device_memory_resource {
 public:
  class operation_guard {
   public:
    operation_guard(operation_guard const&)            = delete;
    operation_guard& operator=(operation_guard const&) = delete;

    operation_guard(operation_guard&& other) noexcept
      : resource_{std::exchange(other.resource_, nullptr)}
    {
    }

    operation_guard& operator=(operation_guard&& other) noexcept
    {
      if (this != &other) {
        finish();
        resource_ = std::exchange(other.resource_, nullptr);
      }
      return *this;
    }

    ~operation_guard() { finish(); }

    void finish() noexcept
    {
      if (resource_ == nullptr) { return; }
      resource_->in_flight_operations_.fetch_sub(1, std::memory_order_acq_rel);
      resource_ = nullptr;
    }

   private:
    friend class query_domain_resource;

    explicit operation_guard(query_domain_resource* resource) noexcept : resource_{resource} {}

    query_domain_resource* resource_ = nullptr;
  };

  query_domain_resource(cudaMemPool_t pool,
                        std::uint64_t domain_id,
                        std::size_t grant_bytes,
                        std::size_t requested_ceiling_bytes,
                        std::size_t effective_ceiling_bytes)
    : view_{pool},
      domain_id_{domain_id},
      grant_bytes_{grant_bytes},
      requested_ceiling_bytes_{requested_ceiling_bytes},
      effective_ceiling_bytes_{effective_ceiling_bytes}
  {
  }

  [[nodiscard]] cudf_memory_resource_usage_t usage(bool active, std::size_t active_members) const
  {
    cudf_memory_resource_usage_t out{};
    fill_usage(out,
               grant_bytes_.load(std::memory_order_relaxed),
               current_bytes_.load(std::memory_order_relaxed),
               peak_bytes_.load(std::memory_order_relaxed),
               total_bytes_.load(std::memory_order_relaxed),
               current_aligned_bytes_.load(std::memory_order_relaxed),
               peak_aligned_bytes_.load(std::memory_order_relaxed),
               total_allocations_.load(std::memory_order_relaxed),
               current_allocations_.load(std::memory_order_relaxed),
               peak_allocations_.load(std::memory_order_relaxed),
               failed_allocation_count_.load(std::memory_order_relaxed),
               largest_failed_allocation_bytes_.load(std::memory_order_relaxed));
    out.reserved[0] = domain_id_;
    out.reserved[1] = active ? 1 : 0;
    out.reserved[2] = active_members;
    out.reserved[3] = requested_ceiling_bytes_;
    out.reserved[4] = effective_ceiling_bytes_;
    out.reserved[5] = 1;
    return out;
  }

  [[nodiscard]] cudf_memory_resource_usage_t member_usage(
    std::shared_ptr<query_domain_member_counters> const& member, bool attached) const
  {
    cudf_memory_resource_usage_t out{};
    if (member == nullptr) { return out; }
    fill_usage(out,
               member->allocation_limit.load(std::memory_order_relaxed),
               member->current_bytes.load(std::memory_order_relaxed),
               member->peak_bytes.load(std::memory_order_relaxed),
               member->total_bytes.load(std::memory_order_relaxed),
               member->current_aligned_bytes.load(std::memory_order_relaxed),
               member->peak_aligned_bytes.load(std::memory_order_relaxed),
               member->total_allocations.load(std::memory_order_relaxed),
               member->current_allocations.load(std::memory_order_relaxed),
               member->peak_allocations.load(std::memory_order_relaxed),
               member->failed_allocation_count.load(std::memory_order_relaxed),
               member->largest_failed_allocation_bytes.load(std::memory_order_relaxed));
    out.reserved[0] = member->id;
    out.reserved[1] = attached ? 1 : 0;
    out.reserved[2] = member->direct_total_allocations.load(std::memory_order_relaxed);
    out.reserved[3] = member->routed_total_allocations.load(std::memory_order_relaxed);
    return out;
  }

  [[nodiscard]] std::uint64_t domain_id() const noexcept { return domain_id_; }

  [[nodiscard]] std::size_t current_allocations() const noexcept
  {
    return current_allocations_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t in_flight_operations() const noexcept
  {
    return in_flight_operations_.load(std::memory_order_acquire);
  }

  [[nodiscard]] operation_guard begin_operation() noexcept
  {
    in_flight_operations_.fetch_add(1, std::memory_order_acq_rel);
    return operation_guard{this};
  }

  bool resize_grant(std::size_t grant_bytes, std::string& error)
  {
    if (grant_bytes > requested_ceiling_bytes_) {
      error = "query memory domain grant exceeds immutable pool ceiling";
      return false;
    }
    grant_bytes_.store(grant_bytes, std::memory_order_relaxed);
    return true;
  }

  void trim_to(std::uint64_t keep_bytes)
  {
    CUDF_CUDA_TRY(cudaMemPoolTrimTo(view_.pool_handle(), keep_bytes));
  }

  bool trim_to_noexcept(std::uint64_t keep_bytes) noexcept
  {
    if (cudaMemPoolTrimTo(view_.pool_handle(), keep_bytes) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }

  bool pool_usage(cudf_memory_pool_usage_t& out) const noexcept
  {
    return read_cuda_pool_usage_noexcept(view_.pool_handle(), out);
  }

  void* allocate_for_member(std::shared_ptr<query_domain_member_counters> const& member,
                            rmm::cuda_stream_view stream,
                            std::size_t bytes,
                            std::size_t alignment,
                            bool require_attached = true)
  {
    return allocate_impl(member, stream, bytes, alignment, require_attached);
  }

  void deallocate_from_adaptor(void* ptr,
                               std::shared_ptr<query_domain_member_counters> const& member,
                               rmm::cuda_stream_view stream,
                               std::size_t bytes,
                               std::size_t alignment) noexcept
  {
    deallocate_impl(ptr, member, stream, bytes, alignment);
  }

 private:
  void record_failed_allocation(std::size_t bytes) noexcept
  {
    failed_allocation_count_.fetch_add(1, std::memory_order_relaxed);
    update_atomic_peak(largest_failed_allocation_bytes_, bytes);
  }

  void record_member_failed_allocation(std::shared_ptr<query_domain_member_counters> const& member,
                                       std::size_t bytes) noexcept
  {
    if (member == nullptr) { return; }
    member->failed_allocation_count.fetch_add(1, std::memory_order_relaxed);
    update_atomic_peak(member->largest_failed_allocation_bytes, bytes);
  }

  static bool reserve_aligned_bytes(std::atomic<std::size_t>& current,
                                    std::atomic<std::size_t> const& limit,
                                    std::size_t aligned_bytes,
                                    std::size_t& reserved_total) noexcept
  {
    auto observed = current.load(std::memory_order_relaxed);
    while (true) {
      auto const current_limit = limit.load(std::memory_order_relaxed);
      if (observed > current_limit || aligned_bytes > current_limit - observed) { return false; }
      reserved_total = observed + aligned_bytes;
      if (current.compare_exchange_weak(observed, reserved_total, std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  bool reserve_domain_aligned_bytes(std::size_t aligned_bytes, std::size_t& reserved_total) noexcept
  {
    return reserve_aligned_bytes(
      current_aligned_bytes_, grant_bytes_, aligned_bytes, reserved_total);
  }

  bool would_exceed_pool_ceiling(std::size_t aligned_bytes) const noexcept
  {
    cudf_memory_pool_usage_t usage{};
    if (!pool_usage(usage)) { return false; }
    if (usage.reserved_current_bytes > effective_ceiling_bytes_) { return true; }
    return aligned_bytes >
           effective_ceiling_bytes_ - static_cast<std::size_t>(usage.reserved_current_bytes);
  }

  [[nodiscard]] memory_failure_domain_identity failure_identity(
    std::shared_ptr<query_domain_member_counters> const& member) const noexcept
  {
    return memory_failure_domain_identity{
      CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN,
      domain_id_,
      member == nullptr ? 0 : member->id,
    };
  }

  void* allocate_impl(std::shared_ptr<query_domain_member_counters> const& member,
                      rmm::cuda_stream_view stream,
                      std::size_t bytes,
                      std::size_t alignment,
                      bool require_attached = true)
  {
    if (bytes == 0) { return view_.allocate(stream, bytes, alignment); }
    auto const aligned_bytes = rmm::align_up(bytes, alignment);
    std::size_t member_reserved_total = 0;
    if (member != nullptr && require_attached &&
        !member->attached.load(std::memory_order_acquire)) {
      throw std::invalid_argument{"query memory domain member is detached"};
    }
    if (member != nullptr &&
        !reserve_aligned_bytes(member->current_aligned_bytes,
                               member->allocation_limit,
                               aligned_bytes,
                               member_reserved_total)) {
      record_member_failed_allocation(member, bytes);
      throw attributed_out_of_memory{"query memory domain member limit exceeded",
                                     CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
                                     failure_identity(member),
                                     view_.pool_handle()};
    }

    std::size_t domain_reserved_total = 0;
    if (!reserve_domain_aligned_bytes(aligned_bytes, domain_reserved_total)) {
      if (member != nullptr) { member->current_aligned_bytes.fetch_sub(aligned_bytes); }
      record_failed_allocation(bytes);
      record_member_failed_allocation(member, bytes);
      throw attributed_out_of_memory{"query memory domain grant exceeded",
                                     CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
                                     failure_identity(member),
                                     view_.pool_handle()};
    }

    void* ptr = nullptr;
    try {
      ptr = view_.allocate(stream, bytes, alignment);
    } catch (rmm::out_of_memory const&) {
      current_aligned_bytes_.fetch_sub(aligned_bytes);
      if (member != nullptr) { member->current_aligned_bytes.fetch_sub(aligned_bytes); }
      record_failed_allocation(bytes);
      record_member_failed_allocation(member, bytes);
      if (would_exceed_pool_ceiling(aligned_bytes)) {
        throw attributed_out_of_memory{"query memory domain pool ceiling exceeded",
                                       CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
                                       failure_identity(member),
                                       view_.pool_handle()};
      }
      throw attributed_out_of_memory{"query memory domain physical allocation failed",
                                     CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL,
                                     failure_identity(member),
                                     view_.pool_handle()};
    } catch (...) {
      current_aligned_bytes_.fetch_sub(aligned_bytes);
      if (member != nullptr) { member->current_aligned_bytes.fetch_sub(aligned_bytes); }
      record_failed_allocation(bytes);
      record_member_failed_allocation(member, bytes);
      throw;
    }

      if (member != nullptr) {
        auto const member_requested   = member->current_bytes.fetch_add(bytes) + bytes;
        auto const member_allocations = member->current_allocations.fetch_add(1) + 1;
        member->total_bytes.fetch_add(bytes, std::memory_order_relaxed);
        member->total_allocations.fetch_add(1, std::memory_order_relaxed);
        update_atomic_peak(member->peak_bytes, member_requested);
        update_atomic_peak(member->peak_aligned_bytes, member_reserved_total);
        update_atomic_peak(member->peak_allocations, member_allocations);
    }

    auto const old_requested = current_bytes_.fetch_add(bytes);
    auto const new_requested = old_requested + bytes;
    auto const old_allocs    = current_allocations_.fetch_add(1);
    auto const new_allocs    = old_allocs + 1;

    update_atomic_peak(peak_aligned_bytes_, domain_reserved_total);
    update_atomic_peak(peak_bytes_, new_requested);
    update_atomic_peak(peak_allocations_, new_allocs);
    total_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    return ptr;
  }

  void deallocate_impl(void* ptr,
                       std::shared_ptr<query_domain_member_counters> const& member,
                       rmm::cuda_stream_view stream,
                       std::size_t bytes,
                       std::size_t alignment) noexcept
  {
    auto const aligned_bytes = rmm::align_up(bytes, alignment);
    view_.deallocate(stream, ptr, bytes, alignment);
    current_aligned_bytes_.fetch_sub(aligned_bytes);
    current_bytes_.fetch_sub(bytes);
    current_allocations_.fetch_sub(1);

    if (member != nullptr) {
      member->current_aligned_bytes.fetch_sub(aligned_bytes);
      member->current_bytes.fetch_sub(bytes);
      member->current_allocations.fetch_sub(1);
    }
  }

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    return allocate_impl({}, stream, bytes, alignment);
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    deallocate_impl(ptr, {}, stream, bytes, alignment);
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  rmm::mr::cuda_async_view_memory_resource view_;
  std::uint64_t domain_id_;
  std::atomic<std::size_t> grant_bytes_{0};
  std::size_t requested_ceiling_bytes_;
  std::size_t effective_ceiling_bytes_;
  std::atomic<std::size_t> current_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
  std::atomic<std::size_t> total_bytes_{0};
  std::atomic<std::size_t> current_aligned_bytes_{0};
  std::atomic<std::size_t> peak_aligned_bytes_{0};
  std::atomic<std::size_t> total_allocations_{0};
  std::atomic<std::size_t> current_allocations_{0};
  std::atomic<std::size_t> peak_allocations_{0};
  std::atomic<std::size_t> in_flight_operations_{0};
  std::atomic<std::size_t> failed_allocation_count_{0};
  std::atomic<std::size_t> largest_failed_allocation_bytes_{0};
};

struct query_domain_state {
  std::uint64_t id = 0;
  int32_t device  = -1;
  cudaMemPool_t pool{};
  std::size_t ceiling_bytes = 0;
  std::size_t effective_ceiling_bytes = 0;
  bool active                         = true;
  std::atomic<bool> trim_pending{false};
  std::size_t active_members = 0;
  std::shared_ptr<query_domain_resource> resource;

  ~query_domain_state()
  {
    resource.reset();
    if (pool != nullptr) {
      cudaMemPoolDestroy(pool);
      pool = nullptr;
    }
  }
};

struct query_domain_member_state {
  std::uint64_t id = 0;
  void* stream_key = nullptr;
  bool attached    = true;
  std::shared_ptr<query_domain_state> domain;
  std::shared_ptr<query_domain_member_counters> counters;
  std::shared_ptr<class query_domain_member_resource> resource;
};

class query_domain_member_resource final : public rmm::mr::device_memory_resource {
 public:
  using cleanup_callback = std::function<void(std::shared_ptr<query_domain_state> const&)>;

  query_domain_member_resource(std::shared_ptr<query_domain_state> domain,
                               std::shared_ptr<query_domain_member_counters> counters,
                               cleanup_callback cleanup)
    : domain_{std::move(domain)}, counters_{std::move(counters)}, cleanup_{std::move(cleanup)}
  {
  }

  void* allocate_routed(rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment)
  {
    try {
      return domain_->resource->allocate_for_member(counters_, stream, bytes, alignment, false);
    } catch (...) {
      if (bytes != 0) { cancel_routed_allocation(); }
      throw;
    }
  }

  void reserve_routed_allocation() noexcept
  {
    counters_->routed_pending_allocations.fetch_add(1, std::memory_order_acq_rel);
    counters_->routed_current_allocations.fetch_add(1, std::memory_order_acq_rel);
  }

  void commit_routed_allocation() noexcept
  {
    counters_->routed_pending_allocations.fetch_sub(1, std::memory_order_acq_rel);
    counters_->routed_total_allocations.fetch_add(1, std::memory_order_relaxed);
    cleanup_(domain_);
  }

  void cancel_routed_allocation() noexcept
  {
    counters_->routed_pending_allocations.fetch_sub(1, std::memory_order_acq_rel);
    counters_->routed_current_allocations.fetch_sub(1, std::memory_order_acq_rel);
    cleanup_(domain_);
  }

  void rollback_routed_allocation(void* ptr,
                                  rmm::cuda_stream_view stream,
                                  std::size_t bytes,
                                  std::size_t alignment) noexcept
  {
    domain_->resource->deallocate_from_adaptor(ptr, counters_, stream, bytes, alignment);
    cancel_routed_allocation();
  }

  void deallocate_routed(void* ptr,
                         rmm::cuda_stream_view stream,
                         std::size_t bytes,
                         std::size_t alignment) noexcept
  {
    domain_->resource->deallocate_from_adaptor(ptr, counters_, stream, bytes, alignment);
    if (bytes != 0) {
      counters_->routed_current_allocations.fetch_sub(1, std::memory_order_acq_rel);
    }
    cleanup_(domain_);
  }

  [[nodiscard]] std::size_t routed_pending_allocations() const noexcept
  {
    return counters_->routed_pending_allocations.load(std::memory_order_acquire);
  }

  void detach() noexcept { counters_->attached.store(false, std::memory_order_release); }

  bool resize(std::size_t limit_bytes) noexcept
  {
    counters_->allocation_limit.store(limit_bytes, std::memory_order_release);
    return true;
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    auto* ptr = domain_->resource->allocate_for_member(counters_, stream, bytes, alignment);
    counters_->direct_total_allocations.fetch_add(1, std::memory_order_relaxed);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    domain_->resource->deallocate_from_adaptor(ptr, counters_, stream, bytes, alignment);
    cleanup_(domain_);
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  std::shared_ptr<query_domain_state> domain_;
  std::shared_ptr<query_domain_member_counters> counters_;
  cleanup_callback cleanup_;
};

class shared_query_domain_member_resource final : public rmm::mr::device_memory_resource {
 public:
  explicit shared_query_domain_member_resource(
    std::shared_ptr<query_domain_member_resource> resource)
    : resource_{std::move(resource)},
      upstream_{rmm::mr::any_device_resource_from_raw(resource_.get())}
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    return upstream_.allocate(stream, bytes, alignment);
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

  std::shared_ptr<query_domain_member_resource> resource_;
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
};

struct reservation_state {
  std::uint64_t id              = 0;
  std::size_t allocation_limit  = 0;
  void* stream_key              = nullptr;
  bool attached                 = true;
  std::shared_ptr<struct reservation_arena_state> arena;
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

struct reservation_arena_state {
  std::uint64_t id              = 0;
  std::size_t allocation_limit  = 0;
  bool active                   = true;
  std::size_t active_members    = 0;
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
    state->id              = reservation_id;
    state->allocation_limit = limit_bytes;
    state->stream_key      = key;
    state->arena           = std::move(arena);
    if (state->arena != nullptr) { state->arena->active_members += 1; }
    reservations_by_stream_[key] = state;
    return state;
  }

  std::shared_ptr<reservation_arena_state> create_arena(std::uint64_t arena_id,
                                                        std::size_t limit_bytes)
  {
    std::scoped_lock lock{mutex_};
    auto state              = std::make_shared<reservation_arena_state>();
    state->id              = arena_id;
    state->allocation_limit = limit_bytes;
    return state;
  }

  std::shared_ptr<query_domain_state> create_query_domain(std::uint64_t domain_id,
                                                          std::size_t grant_bytes,
                                                          std::size_t ceiling_bytes,
                                                          std::uint64_t release_threshold,
                                                          int32_t device,
                                                          std::string& error)
  {
    if (ceiling_bytes == 0) {
      error = "query memory domain ceiling must be nonzero";
      return {};
    }
    if (grant_bytes > ceiling_bytes) {
      error = "query memory domain grant exceeds ceiling";
      return {};
    }
#if !CUDF_RUST_HAS_CUDA_MEMPOOL_MAX_SIZE
    error = "query memory domains require CUDA private-pool hard ceiling support";
    return {};
#else

    cudaMemPoolProps props{};
    props.allocType      = cudaMemAllocationTypePinned;
    props.handleTypes    = cudaMemHandleTypeNone;
    props.location.type  = cudaMemLocationTypeDevice;
    props.location.id    = device;
    props.maxSize        = ceiling_bytes;
    cudaMemPool_t pool   = nullptr;
    CUDF_CUDA_TRY(cudaMemPoolCreate(&pool, &props));
    try {
      CUDF_CUDA_TRY(
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));
      std::uint64_t effective_ceiling = 0;
      CUDF_CUDA_TRY(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrMaxPoolSize, &effective_ceiling));
      if (effective_ceiling == 0 || effective_ceiling < ceiling_bytes) {
        error =
          "CUDA private-pool effective ceiling is below the requested "
          "ceiling";
        cudaMemPoolDestroy(pool);
        return {};
      }
      auto state            = std::make_shared<query_domain_state>();
      auto* domain_pool     = pool;
      state->id             = domain_id;
      state->device         = device;
      state->pool           = domain_pool;
      pool                  = nullptr;
      state->ceiling_bytes  = ceiling_bytes;
      state->effective_ceiling_bytes = static_cast<std::size_t>(effective_ceiling);
      state->resource                = std::make_shared<query_domain_resource>(
        domain_pool, domain_id, grant_bytes, state->ceiling_bytes, state->effective_ceiling_bytes);
      {
        std::unique_lock lock{query_domain_mutex_};
        query_domains_by_pool_[domain_pool] = state;
        query_domain_pool_count_.fetch_add(1, std::memory_order_relaxed);
      }
      return state;
    } catch (...) {
      if (pool != nullptr) { cudaMemPoolDestroy(pool); }
      throw;
    }
#endif
  }

  std::shared_ptr<query_domain_member_state> attach_query_domain(
    std::shared_ptr<query_domain_state> const& domain,
    rmm::cuda_stream_view stream,
    std::uint64_t reservation_id,
    std::size_t limit_bytes,
    std::string& error)
  {
    if (domain == nullptr) {
      error = "query memory domain is null";
      return {};
    }
    auto const key = stream_key(stream);
    std::unique_lock query_lock{query_domain_mutex_};
    if (!domain->active) {
      error = "query memory domain is not active";
      return {};
    }
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
    {
      std::scoped_lock legacy_lock{mutex_};
      auto const existing = reservations_by_stream_.find(key);
      if (existing != reservations_by_stream_.end()) {
        auto state = existing->second.lock();
        if (state != nullptr && state->attached) {
          error = "stream already has an attached memory reservation";
          return {};
        }
        reservations_by_stream_.erase(existing);
      }
    }

    auto state        = std::make_shared<query_domain_member_state>();
    state->id        = reservation_id;
    state->stream_key = key;
    state->domain    = domain;
    state->counters  = std::make_shared<query_domain_member_counters>();
    state->counters->id = reservation_id;
    state->counters->allocation_limit.store(limit_bytes, std::memory_order_relaxed);
    std::weak_ptr<query_domain_member_state> weak_state = state;
    state->resource = std::make_shared<query_domain_member_resource>(
      domain,
      state->counters,
      [this, weak_state](std::shared_ptr<query_domain_state> const& member_domain) {
        if (auto member = weak_state.lock()) { cleanup_retired_query_domain_member(member); }
        cleanup_query_domain_if_idle(member_domain);
      });
    domain->active_members += 1;
    query_domain_members_by_stream_[key] = state;
    query_domain_member_count_.fetch_add(1, std::memory_order_relaxed);
    return state;
  }

  void detach(std::shared_ptr<reservation_state> const& state)
  {
    if (state == nullptr) { return; }
    std::scoped_lock lock{mutex_};
    bool const was_attached = state->attached;
    state->attached = false;
    auto const existing = reservations_by_stream_.find(state->stream_key);
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

  void detach_query_domain_member(std::shared_ptr<query_domain_member_state> const& state)
  {
    if (state == nullptr) { return; }
    std::shared_ptr<query_domain_state> domain;
    {
      std::unique_lock lock{query_domain_mutex_};
      bool const was_attached = state->attached;
      state->attached         = false;
      if (state->resource != nullptr) { state->resource->detach(); }
      domain                  = state->domain;
      auto const existing     = query_domain_members_by_stream_.find(state->stream_key);
      if (existing != query_domain_members_by_stream_.end() &&
          (state->resource == nullptr || state->resource->routed_pending_allocations() == 0)) {
        auto existing_state = existing->second;
        if (existing_state == nullptr || existing_state.get() == state.get()) {
          query_domain_members_by_stream_.erase(existing);
          query_domain_member_count_.fetch_sub(1, std::memory_order_relaxed);
        }
      }
      if (was_attached) {
        if (domain != nullptr && domain->active_members > 0) { domain->active_members -= 1; }
      }
    }
    cleanup_query_domain_if_idle(domain);
  }

  void destroy_arena(std::shared_ptr<reservation_arena_state> const& state)
  {
    if (state == nullptr) { return; }
    std::scoped_lock lock{mutex_};
    state->active = false;
  }

  void destroy_query_domain(std::shared_ptr<query_domain_state> const& state)
  {
    if (state == nullptr) { return; }
    {
      std::unique_lock lock{query_domain_mutex_};
      state->active = false;
      state->trim_pending.store(true, std::memory_order_release);
    }
    cleanup_query_domain_if_idle(state);
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

  cudf_memory_resource_usage_t arena_usage(std::shared_ptr<reservation_arena_state> const& state)
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

  cudf_memory_resource_usage_t query_domain_usage(std::shared_ptr<query_domain_state> const& state)
  {
    if (state == nullptr || state->resource == nullptr) { return {}; }
    std::shared_lock lock{query_domain_mutex_};
    return state->resource->usage(state->active, state->active_members);
  }

  [[nodiscard]] std::size_t query_domain_pool_count_for_id(std::uint64_t domain_id)
  {
    std::shared_lock lock{query_domain_mutex_};
    return static_cast<std::size_t>(std::count_if(query_domains_by_pool_.cbegin(),
                                                  query_domains_by_pool_.cend(),
                                                  [domain_id](auto const& entry) {
                                                    return entry.second != nullptr &&
                                                           entry.second->id == domain_id;
                                                  }));
  }

  cudf_memory_resource_usage_t query_domain_member_usage(
    std::shared_ptr<query_domain_member_state> const& state)
  {
    if (state == nullptr || state->domain == nullptr || state->domain->resource == nullptr) {
      return {};
    }
    std::shared_lock lock{query_domain_mutex_};
    return state->domain->resource->member_usage(state->counters, state->attached);
  }

  bool resize(std::shared_ptr<reservation_state> const& state,
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

  bool resize_arena(std::shared_ptr<reservation_arena_state> const& state,
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

  bool resize_query_domain(std::shared_ptr<query_domain_state> const& state,
                           std::size_t grant_bytes,
                           std::string& error)
  {
    if (state == nullptr || state->resource == nullptr) {
      error = "query memory domain is null";
      return false;
    }
    std::shared_lock lock{query_domain_mutex_};
    if (!state->active) {
      error = "query memory domain is not active";
      return false;
    }
    return state->resource->resize_grant(grant_bytes, error);
  }

  bool resize_query_domain_member(std::shared_ptr<query_domain_member_state> const& state,
                                  std::size_t limit_bytes,
                                  std::string& error)
  {
    if (state == nullptr || state->resource == nullptr) {
      error = "query memory domain member is null";
      return false;
    }
    std::shared_lock lock{query_domain_mutex_};
    if (!state->attached) {
      error = "query memory domain member is not attached";
      return false;
    }
    return state->resource->resize(limit_bytes);
  }

  void trim_query_domain(std::shared_ptr<query_domain_state> const& state, std::uint64_t keep_bytes)
  {
    if (state == nullptr || state->resource == nullptr) {
      throw std::runtime_error{"query memory domain is null"};
    }
    state->resource->trim_to(keep_bytes);
  }

 private:
  struct query_domain_member_route {
    std::shared_ptr<query_domain_member_state> member;
    query_domain_resource::operation_guard operation;
  };

  struct query_domain_pointer_route {
    std::shared_ptr<query_domain_state> domain;
    query_domain_resource::operation_guard operation;
  };

  struct allocation_record {
    std::size_t bytes = 0;
    std::size_t aligned_bytes = 0;
    std::shared_ptr<reservation_state> reservation;
    std::shared_ptr<reservation_arena_state> arena;
  };

  static void* stream_key(rmm::cuda_stream_view stream)
  {
    return reinterpret_cast<void*>(stream.value());
  }

  std::optional<query_domain_member_route> query_domain_member_for_stream(
    rmm::cuda_stream_view stream, std::size_t bytes)
  {
    if (query_domain_member_count_.load(std::memory_order_relaxed) == 0) { return {}; }
    auto const key = stream_key(stream);
    std::shared_lock lock{query_domain_mutex_};
    auto const it = query_domain_members_by_stream_.find(key);
    if (it == query_domain_members_by_stream_.end()) { return {}; }
    auto state = it->second;
    if (state == nullptr || !state->attached || state->domain == nullptr ||
        state->domain->resource == nullptr || state->resource == nullptr) {
      return {};
    }
    auto operation = state->domain->resource->begin_operation();
    if (bytes != 0) { state->resource->reserve_routed_allocation(); }
    return query_domain_member_route{state, std::move(operation)};
  }

  std::optional<query_domain_member_route> query_domain_member_for_deallocation(void* ptr)
  {
    if (query_domain_routed_pointer_count_.load(std::memory_order_relaxed) == 0 ||
        ptr == nullptr) {
      return {};
    }
    std::unique_lock lock{query_domain_mutex_};
    auto const it = query_domain_members_by_pointer_.find(ptr);
    if (it == query_domain_members_by_pointer_.end()) { return {}; }
    auto state = std::move(it->second);
    query_domain_members_by_pointer_.erase(it);
    query_domain_routed_pointer_count_.fetch_sub(1, std::memory_order_relaxed);
    if (state == nullptr || state->domain == nullptr || state->domain->resource == nullptr ||
        state->resource == nullptr) {
      return {};
    }
    return query_domain_member_route{state, state->domain->resource->begin_operation()};
  }

  std::optional<query_domain_pointer_route> query_domain_for_pointer(void* ptr)
  {
    if (query_domain_pool_count_.load(std::memory_order_relaxed) == 0) { return {}; }
    if (ptr == nullptr) { return {}; }
    cudaMemPool_t pool = nullptr;
    auto const result  = cuPointerGetAttribute(
      &pool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, reinterpret_cast<CUdeviceptr>(ptr));
    if (result != CUDA_SUCCESS || pool == nullptr) { return {}; }
    std::shared_lock lock{query_domain_mutex_};
    auto const it = query_domains_by_pool_.find(pool);
    if (it == query_domains_by_pool_.end()) { return {}; }
    auto domain = it->second;
    if (domain == nullptr || domain->resource == nullptr) { return {}; }
    return query_domain_pointer_route{domain, domain->resource->begin_operation()};
  }

  void cleanup_query_domain_if_idle(std::shared_ptr<query_domain_state> const& domain)
  {
    if (domain == nullptr || domain->resource == nullptr) { return; }
    if (domain->resource->in_flight_operations() != 0) { return; }
    if (domain->resource->current_allocations() != 0 &&
        !domain->trim_pending.load(std::memory_order_acquire)) {
      return;
    }
    std::unique_lock lock{query_domain_mutex_};
    if (domain->resource->in_flight_operations() != 0) { return; }
    if (!domain->active && domain->trim_pending.load(std::memory_order_relaxed) &&
        domain->resource->trim_to_noexcept(0)) {
      domain->trim_pending.store(false, std::memory_order_release);
    }
    if (domain->active || domain->active_members != 0 ||
        domain->resource->current_allocations() != 0) {
      return;
    }
    auto const it = query_domains_by_pool_.find(domain->pool);
    if (it != query_domains_by_pool_.end() && it->second.get() == domain.get()) {
      query_domains_by_pool_.erase(it);
      query_domain_pool_count_.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  void cleanup_retired_query_domain_member(std::shared_ptr<query_domain_member_state> const& member)
  {
    if (member == nullptr || member->resource == nullptr ||
        member->resource->routed_pending_allocations() != 0) {
      return;
    }
    std::unique_lock lock{query_domain_mutex_};
    if (member->attached || member->resource->routed_pending_allocations() != 0) { return; }
    auto const it = query_domain_members_by_stream_.find(member->stream_key);
    if (it != query_domain_members_by_stream_.end() &&
        (it->second == nullptr || it->second.get() == member.get())) {
      query_domain_members_by_stream_.erase(it);
      query_domain_member_count_.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  template <typename State>
  void record_failed_allocation(State& state, std::size_t bytes)
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

  template <typename State>
  void reserve_allocation(State& state,
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
  void commit_allocation(State& state, std::size_t bytes, std::size_t aligned_bytes)
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
  void release_reserved_alignment(State& state, std::size_t aligned_bytes)
  {
    if (state.current_aligned_bytes >= aligned_bytes) {
      state.current_aligned_bytes -= aligned_bytes;
    } else {
      state.current_aligned_bytes = 0;
    }
  }

  template <typename State>
  void rollback_reserved_allocation(State& state, std::size_t bytes, std::size_t aligned_bytes)
  {
    release_reserved_alignment(state, aligned_bytes);
    record_failed_allocation(state, bytes);
  }

  template <typename State>
  void release_allocation(State& state, std::size_t bytes, std::size_t aligned_bytes)
  {
    state.current_bytes = state.current_bytes >= bytes ? state.current_bytes - bytes : 0;
    state.current_aligned_bytes = state.current_aligned_bytes >= aligned_bytes
                                    ? state.current_aligned_bytes - aligned_bytes
                                    : 0;
    state.current_allocations   = state.current_allocations > 0 ? state.current_allocations - 1 : 0;
  }

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    auto query_route = query_domain_member_for_stream(stream, bytes);
    if (query_route.has_value()) {
      auto domain = query_route->member->domain;
      try {
#ifndef NDEBUG
        pause_query_domain_route_if_armed(domain->id);
#endif
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

    auto const aligned_bytes = rmm::align_up(bytes, alignment);
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

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
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

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::mutex mutex_;
  std::shared_mutex query_domain_mutex_;
  std::unordered_map<void*, std::shared_ptr<query_domain_member_state>>
    query_domain_members_by_stream_;
  std::unordered_map<void*, std::shared_ptr<query_domain_member_state>>
    query_domain_members_by_pointer_;
  std::unordered_map<cudaMemPool_t, std::shared_ptr<query_domain_state>> query_domains_by_pool_;
  std::atomic<std::size_t> query_domain_member_count_{0};
  std::atomic<std::size_t> query_domain_routed_pointer_count_{0};
  std::atomic<std::size_t> query_domain_pool_count_{0};
  std::unordered_map<void*, std::weak_ptr<reservation_state>> reservations_by_stream_;
  std::unordered_map<void*, allocation_record> allocations_;
};

class bounded_failure_callback_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  bounded_failure_callback_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                            cudf_memory_failure_callback_t callback,
                                            void* callback_arg,
                                            cudaMemPool_t pool,
                                            int32_t device_ordinal)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)},
      callback_{callback},
      callback_arg_{callback_arg},
      pool_{pool},
      device_ordinal_{device_ordinal}
  {
  }

  bounded_failure_callback_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                            cudf_memory_failure_callback_v2_t callback,
                                            void* callback_arg,
                                            cudaMemPool_t pool,
                                            int32_t device_ordinal)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)},
      callback_v2_{callback},
      callback_arg_{callback_arg},
      pool_{pool},
      device_ordinal_{device_ordinal}
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    std::uint32_t callback_retries = 0;
    bool trim_retry_attempted      = false;
    std::uint32_t trim_retry_count = 0;
    while (true) {
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
        if (callback_v2_ != nullptr) {
          if (callback_retries >= k_max_failure_callback_retries) { throw; }
          cudf_memory_failure_info_t info{};
          fill_failure_info(info,
                            device_ordinal_,
                            bytes,
                            failure_pool,
                            failure_source,
                            failure_identity,
                            callback_retries,
                            trim_retry_count);
          auto const action = callback_v2_(&info, callback_arg_);
          if (action == CUDF_MEMORY_FAILURE_ACTION_RETRY) {
            ++callback_retries;
            continue;
          }
          if (failure_source == CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL &&
              action == CUDF_MEMORY_FAILURE_ACTION_TRIM_AND_RETRY &&
              trim_cuda_pool_to_zero_noexcept(failure_pool)) {
            ++callback_retries;
            ++trim_retry_count;
            continue;
          }
          throw;
        }
        if (failure_source == CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL && !trim_retry_attempted &&
            trim_cuda_pool_before_retry(failure_pool, bytes)) {
          trim_retry_attempted = true;
          continue;
        }
        if (callback_ == nullptr || callback_retries >= k_max_failure_callback_retries ||
            callback_(bytes, callback_arg_) == 0) {
          throw;
        }
        ++callback_retries;
      }
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
  cudf_memory_failure_callback_t callback_ = nullptr;
  cudf_memory_failure_callback_v2_t callback_v2_ = nullptr;
  void* callback_arg_                      = nullptr;
  cudaMemPool_t pool_                            = nullptr;
  int32_t device_ordinal_                   = -1;
};

class fail_first_allocation_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  explicit fail_first_allocation_resource_adaptor(rmm::mr::device_memory_resource* upstream)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)}
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    if (!failed_once_.exchange(true, std::memory_order_acq_rel)) {
      throw rmm::out_of_memory{"test memory resource forced first allocation failure"};
    }
    return upstream_.allocate(stream, bytes, alignment);
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
  std::atomic<bool> failed_once_{false};
};

struct memory_resource_owner {
  std::shared_ptr<rmm::mr::device_memory_resource> upstream;
  std::shared_ptr<rmm::mr::device_memory_resource> pool;
  std::shared_ptr<void> upstream_lifetime;
  std::unique_ptr<rmm::mr::device_memory_resource> reservation_mr;
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  rmm::mr::statistics_resource_adaptor* stats = nullptr;
  budget_tracking_resource_adaptor* budget_tracker                         = nullptr;
  reservation_aware_resource_adaptor* reservation_adaptor                   = nullptr;
  cuda_async_resource_box* async_pool                                       = nullptr;
  int32_t device                                                            = -1;
};

struct memory_resource_handle_state {
  std::shared_ptr<memory_resource_owner> owner;
  bool non_owning_alias = false;
};

struct device_memory_manager_state {
  int32_t device = -1;
  std::shared_ptr<memory_resource_owner> owner;
  cudf_memory_manager_options_t options = default_memory_manager_options();
  cudf_memory_failure_callback_t failure_callback       = nullptr;
  cudf_memory_failure_callback_v2_t failure_callback_v2 = nullptr;
  void* failure_callback_arg                            = nullptr;
  bool failure_callback_installed = false;
  bool current_resource_installed = false;
};

struct reservation_handle_state {
  std::shared_ptr<device_memory_manager_state> manager;
  std::shared_ptr<reservation_state> reservation;
  std::shared_ptr<query_domain_member_state> query_domain_member;
};

struct reservation_arena_handle_state {
  std::shared_ptr<device_memory_manager_state> manager;
  std::shared_ptr<reservation_arena_state> arena;
};

struct query_domain_handle_state {
  std::shared_ptr<device_memory_manager_state> manager;
  std::shared_ptr<query_domain_state> domain;
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
  auto& identities = current_memory_resource_identities();
  auto const existing = identities.find(device);
  bool const had_existing = existing != identities.end();
  auto* previous = had_existing ? existing->second : nullptr;
  if (had_existing) {
    existing->second = target;
  } else {
    identities.emplace(device, target);
  }
  try {
    rmm::mr::set_current_device_resource(rmm::mr::any_device_resource_from_raw(target));
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

std::unique_ptr<memory_resource_handle_state> make_query_domain_member_resource_handle(
  std::shared_ptr<device_memory_manager_state> const& manager,
  std::shared_ptr<query_domain_member_state> const& member)
{
  auto state              = std::make_unique<memory_resource_handle_state>();
  state->owner            = std::make_shared<memory_resource_owner>();
  auto& owner             = *state->owner;
  owner.device            = manager->device;
  owner.upstream_lifetime = member->resource;
  auto direct             = std::make_unique<shared_query_domain_member_resource>(member->resource);
  if (manager->failure_callback_v2 != nullptr) {
    owner.reservation_mr = std::move(direct);
    owner.mr =
      std::make_unique<bounded_failure_callback_resource_adaptor>(owner.reservation_mr.get(),
                                                                  manager->failure_callback_v2,
                                                                  manager->failure_callback_arg,
                                                                  member->domain->pool,
                                                                  manager->device);
  } else if (manager->failure_callback != nullptr) {
    owner.reservation_mr = std::move(direct);
    owner.mr =
      std::make_unique<bounded_failure_callback_resource_adaptor>(owner.reservation_mr.get(),
                                                                  manager->failure_callback,
                                                                  manager->failure_callback_arg,
                                                                  member->domain->pool,
                                                                  manager->device);
  } else {
    owner.mr = std::move(direct);
  }
  register_memory_resource_lifetime(state->owner);
  return state;
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
    if (manager.get() == handle) { return manager; }
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
  auto manager    = std::make_shared<device_memory_manager_state>();
  manager->device = device;
  manager->options = options;
  manager->failure_callback     = failure_callback;
  manager->failure_callback_v2  = failure_callback_v2;
  manager->failure_callback_arg = failure_callback_arg;
  manager->failure_callback_installed =
    failure_callback != nullptr || failure_callback_v2 != nullptr;
  manager->owner  = std::make_shared<memory_resource_owner>();

  auto& owner = *manager->owner;
  owner.device = device;
  if (options.base_kind == CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC) {
    auto async_pool = make_shared_cuda_async_resource_box(options.initial_pool_size_bytes,
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
    owner.mr             = std::make_unique<bounded_failure_callback_resource_adaptor>(
      owner.reservation_mr.get(),
      failure_callback_v2,
      failure_callback_arg,
      async_pool_handle(owner.async_pool),
      device);
  } else if (failure_callback != nullptr) {
    owner.reservation_mr = std::move(adaptor);
    owner.mr             = std::make_unique<bounded_failure_callback_resource_adaptor>(
      owner.reservation_mr.get(),
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
        "device memory manager already exists with different physical-base options; "
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
    *out = static_cast<cudf_memory_manager_handle>(it->second.get());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // namespace

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
    state->owner->device = current_cuda_device();
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
    owner.device   = current_cuda_device();
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

cudf_status_t cudf_memory_resource_create_cuda_async(size_t initial_pool_size,
                                                     uint64_t release_threshold,
                                                     cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto state        = std::make_unique<memory_resource_handle_state>();
    state->owner      = std::make_shared<memory_resource_owner>();
    auto async_pool   = make_cuda_async_resource_box(initial_pool_size, release_threshold);
    auto* async_pool_ptr = async_pool.get();
    state->owner->device = current_cuda_device();
    state->owner->async_pool = async_pool_ptr;
    state->owner->mr         = std::move(async_pool);
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
    auto& upstream_owner          = memory_resource_owner_from_handle(upstream);
    owner.upstream_lifetime       = cudf_memory_resource_lifetime_from_handle(upstream);
    owner.async_pool              = upstream_owner.async_pool;
    owner.device                  = upstream_owner.device;
    auto* upstream_mr             = cudf_memory_resource_raw_from_handle(upstream);
    owner.mr =
      std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::limiting_resource_adaptor>>(
        rmm::mr::any_device_resource_from_raw(upstream_mr), allocation_limit);
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
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
    owner.mr                = std::make_unique<bounded_failure_callback_resource_adaptor>(
      upstream_mr, callback, callback_arg, async_pool_handle(owner.async_pool), owner.device);
    register_memory_resource_lifetime(state->owner);
    *out = static_cast<cudf_memory_resource_handle>(state.release());
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
    owner.mr                = std::make_unique<bounded_failure_callback_resource_adaptor>(
      upstream_mr, callback, callback_arg, async_pool_handle(owner.async_pool), owner.device);
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
    auto* state = static_cast<memory_resource_handle_state*>(handle);
    auto& owner = *state->owner;
    if (!state->non_owning_alias && cudf_tracked_current_device_resource_raw() == owner.mr.get()) {
      set_error(
        "cannot destroy a memory resource while it is installed as the "
        "current device resource");
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
    *exists = 1;
    *failure_callback_installed = it->second->failure_callback_installed ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_options_for_device(
  int32_t device, int32_t* exists, cudf_memory_manager_options_t* out)
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

    auto state             = std::make_unique<memory_resource_handle_state>();
    state->owner           = manager_state->owner;
    state->non_owning_alias = true;
    *out = static_cast<cudf_memory_resource_handle>(state.release());
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
    *out = static_cast<cudf_reservation_handle>(state.release());
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
    *out = static_cast<cudf_reservation_arena_handle>(state.release());
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
    auto& arena_state = *static_cast<reservation_arena_handle_state*>(arena);
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
    auto& state = *static_cast<reservation_handle_state*>(reservation);
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
    auto* state = static_cast<reservation_handle_state*>(reservation);
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
    auto& state = *static_cast<reservation_arena_handle_state*>(arena);
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
    auto& state = *static_cast<reservation_arena_handle_state*>(arena);
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
    auto* state = static_cast<reservation_arena_handle_state*>(arena);
    auto* adaptor = state->manager->owner->reservation_adaptor;
    if (adaptor != nullptr) { adaptor->destroy_arena(state->arena); }
    delete state;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_memory_manager_create_query_memory_domain(cudf_memory_manager_handle manager,
                                                             uint64_t domain_id,
                                                             size_t grant_bytes,
                                                             size_t ceiling_bytes,
                                                             uint64_t release_threshold_bytes,
                                                             cudf_query_memory_domain_handle* out)
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
    if (manager_state->owner->async_pool == nullptr) {
      set_error("query memory domains require a CUDA async manager base");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(manager_state->device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }

    std::string create_error;
    auto domain = adaptor->create_query_domain(domain_id,
                                               grant_bytes,
                                               ceiling_bytes,
                                               release_threshold_bytes,
                                               manager_state->device,
                                               create_error);
    if (domain == nullptr) {
      set_error(create_error);
      return CUDF_ERROR_INVALID_ARGUMENT;
    }

    auto state     = std::make_unique<query_domain_handle_state>();
    state->manager = std::move(manager_state);
    state->domain  = std::move(domain);
    *out = static_cast<cudf_query_memory_domain_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_attach_stream(cudf_query_memory_domain_handle domain,
                                                     cudf_stream_handle stream,
                                                     uint64_t reservation_id,
                                                     size_t limit_bytes,
                                                     cudf_reservation_handle* out,
                                                     cudf_memory_resource_handle* out_resource)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_NULL_CHECK(out_resource, "out_resource is null");
  *out = nullptr;
  *out_resource = nullptr;
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_TRY
  {
    auto& domain_state = *static_cast<query_domain_handle_state*>(domain);
    cudf_status_t const stream_status = validate_stream_device_for_manager(
      stream,
      domain_state.manager->device,
      "query memory domain requires stream and domain manager to be on the "
      "same CUDA device");
    if (stream_status != CUDF_SUCCESS) { return stream_status; }
    auto* adaptor      = domain_state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("query memory domain manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    std::string attach_error;
    auto member = adaptor->attach_query_domain(domain_state.domain,
                                               cudf_stream_view_from_handle(stream),
                                               reservation_id,
                                               limit_bytes,
                                               attach_error);
    if (member == nullptr) {
      set_error(attach_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }

    try {
      auto resource_state = make_query_domain_member_resource_handle(domain_state.manager, member);
      auto state          = std::make_unique<reservation_handle_state>();
      state->manager      = domain_state.manager;
      state->query_domain_member = member;
      *out_resource = static_cast<cudf_memory_resource_handle>(resource_state.release());
      *out          = static_cast<cudf_reservation_handle>(state.release());
    } catch (...) {
      adaptor->detach_query_domain_member(member);
      throw;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

int32_t cudf_query_memory_domain_driver_ceiling_supported(void)
{
#if CUDF_RUST_HAS_CUDA_MEMPOOL_MAX_SIZE
  return 1;
#else
  return 0;
#endif
}

cudf_status_t cudf_query_memory_domain_usage(cudf_query_memory_domain_handle domain,
                                             cudf_memory_resource_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto& state = *static_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("query memory domain manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = adaptor->query_domain_usage(state.domain);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_pool_usage(cudf_query_memory_domain_handle domain,
                                                  cudf_memory_pool_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto& state = *static_cast<query_domain_handle_state*>(domain);
    if (state.domain == nullptr || state.domain->resource == nullptr ||
        !state.domain->resource->pool_usage(*out)) {
      set_error("failed to read query memory domain pool usage");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_resize_grant(cudf_query_memory_domain_handle domain,
                                                    size_t grant_bytes)
{
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto& state = *static_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("query memory domain manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    std::string resize_error;
    if (!adaptor->resize_query_domain(state.domain, grant_bytes, resize_error)) {
      set_error(resize_error);
      return CUDF_ERROR_INVALID_ARGUMENT;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_trim(cudf_query_memory_domain_handle domain,
                                            uint64_t keep_bytes)
{
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto& state = *static_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("query memory domain manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    adaptor->trim_query_domain(state.domain, keep_bytes);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_destroy(cudf_query_memory_domain_handle domain)
{
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto* state = static_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state->manager->owner->reservation_adaptor;
    if (adaptor != nullptr) { adaptor->destroy_query_domain(state->domain); }
    delete state;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_memory_manager_query_domain_pool_count_for_id(
  cudf_memory_manager_handle manager, uint64_t domain_id, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
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
    *out = adaptor->query_domain_pool_count_for_id(domain_id);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

int32_t cudf_test_query_domain_route_pause_arm(uint64_t domain_id)
{
#ifndef NDEBUG
  auto& state = query_domain_route_pause();
  state.reached.store(false, std::memory_order_relaxed);
  state.released.store(false, std::memory_order_relaxed);
  state.target_domain_id.store(domain_id, std::memory_order_relaxed);
  state.armed.store(true, std::memory_order_release);
  return 1;
#else
  (void)domain_id;
  return 0;
#endif
}

int32_t cudf_test_query_domain_route_pause_reached()
{
#ifndef NDEBUG
  return query_domain_route_pause().reached.load(std::memory_order_acquire) ? 1 : 0;
#else
  return 0;
#endif
}

void cudf_test_query_domain_route_pause_release()
{
#ifndef NDEBUG
  auto& state = query_domain_route_pause();
  state.released.store(true, std::memory_order_release);
  state.armed.store(false, std::memory_order_release);
#endif
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
    owner.device   = current_cuda_device();
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

cudf_status_t cudf_test_memory_resource_create_fail_first_allocation(
  cudf_memory_resource_handle upstream, cudf_memory_resource_handle* out)
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
    owner.mr                = std::make_unique<fail_first_allocation_resource_adaptor>(upstream_mr);
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
    cudf_set_current_device_resource_raw(static_cast<rmm::mr::device_memory_resource*>(raw_mr));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
