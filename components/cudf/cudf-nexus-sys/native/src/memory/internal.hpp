#ifndef CUDF_RUST_MEMORY_INTERNAL_HPP
#define CUDF_RUST_MEMORY_INTERNAL_HPP

#include "cudf_rust/memory_manager.h"
#include "cudf_rust/memory_resource.h"
#include "cudf_rust/query_memory_domain.h"
#include "cudf_rust/reservation.h"
#include "memory_resource_compat.hpp"
#include "reservation_accounting.hpp"

#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda.h>
#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace rmm::mr {
class statistics_resource_adaptor;
}

namespace cudf::rust::memory {

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
  /// Monotonically minted by the native query-domain owner. It is never a
  /// pointer identity and is required before the engine may correlate an
  /// event to an attempt.
  std::uint64_t generation          = 0;
};

/// Per-device, multi-producer fixed allocator-event ring.
///
/// Writers run inside allocation-failure handling. They use only atomics and
/// preallocated slots; consumers copy a sequence only after its publication
/// stamp is stable. A reader never infers a missing record from zero-valued
/// payload fields.
class allocator_event_ring final {
 public:
  allocator_event_ring() = default;

  allocator_event_ring(allocator_event_ring const&)            = delete;
  allocator_event_ring& operator=(allocator_event_ring const&) = delete;

  void record(cudf_memory_failure_info_t const& info,
              memory_failure_domain_identity identity,
              cudf_memory_failure_action_t action) noexcept;

  [[nodiscard]] cudf_allocator_event_ring_bounds_t bounds() const noexcept;

  [[nodiscard]] cudf_allocator_event_read_status_t read(
    std::uint64_t sequence, cudf_allocator_event_t& out) const noexcept;

 private:
  struct slot {
    // Writers never wait for this flag. A contending callback leaves its
    // sequence without a stable record; consumers must treat that as
    // unavailable rather than risking a mixed payload.
    std::atomic_flag writer_active = ATOMIC_FLAG_INIT;
    std::atomic<std::uint64_t> published_sequence{0};
    std::atomic<std::uint64_t> monotonic_time_ns{0};
    std::atomic<std::int32_t> device_ordinal{-1};
    std::atomic<std::uint32_t> completeness{0};
    std::atomic<std::uint32_t> domain_kind{CUDF_MEMORY_FAILURE_DOMAIN_NONE};
    std::atomic<std::uint32_t> failure_source{CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL};
    std::atomic<std::int32_t> action{CUDF_MEMORY_FAILURE_ACTION_DECLINE};
    std::atomic<std::uint32_t> callback_attempt{0};
    std::atomic<std::uint32_t> trim_retry_count{0};
    std::atomic<std::uint64_t> domain_id{0};
    std::atomic<std::uint64_t> domain_generation{0};
    std::atomic<std::uint64_t> domain_member_id{0};
    std::atomic<std::uint64_t> requested_bytes{0};
    std::atomic<std::uint64_t> pool_reserved_current_bytes{0};
    std::atomic<std::uint64_t> pool_reserved_high_bytes{0};
    std::atomic<std::uint64_t> pool_used_current_bytes{0};
    std::atomic<std::uint64_t> pool_used_high_bytes{0};
  };

  static constexpr std::size_t k_capacity = CUDF_ALLOCATOR_EVENT_RING_CAPACITY;

  std::atomic<std::uint64_t> next_sequence_{1};
  std::array<slot, k_capacity> slots_{};
};

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

cudf_memory_manager_options_t default_memory_manager_options();
std::unique_ptr<cuda_async_resource_box> make_cuda_async_resource_box(
  std::uint64_t initial_pool_size, std::uint64_t release_threshold);
std::shared_ptr<cuda_async_resource_box> make_shared_cuda_async_resource_box(
  std::uint64_t initial_pool_size, std::uint64_t release_threshold);
cudaMemPool_t async_pool_handle(cuda_async_resource_box* async_pool) noexcept;
void fill_pool_usage(cudf_memory_pool_usage_t& out,
                     std::uint64_t reserved_current,
                     std::uint64_t reserved_high,
                     std::uint64_t used_current,
                     std::uint64_t used_high);
bool read_cuda_pool_usage_noexcept(cudaMemPool_t pool, cudf_memory_pool_usage_t& out) noexcept;
bool read_async_pool_usage_noexcept(cuda_async_resource_box* async_pool,
                                    cudf_memory_pool_usage_t& out) noexcept;
void read_async_pool_usage(cuda_async_resource_box* async_pool, cudf_memory_pool_usage_t& out);
void trim_async_pool_to(cuda_async_resource_box* async_pool, std::uint64_t keep_bytes);
bool trim_cuda_pool_before_retry(cudaMemPool_t pool, std::size_t request_bytes) noexcept;
bool trim_cuda_pool_to_zero_noexcept(cudaMemPool_t pool) noexcept;
bool async_pool_has_trim_retry_headroom(cudf_memory_pool_usage_t const& usage,
                                        std::size_t request_bytes);
void fill_failure_info(cudf_memory_failure_info_t& out,
                       int32_t device_ordinal,
                       std::size_t request_bytes,
                       cudaMemPool_t pool,
                       cudf_memory_failure_source_t failure_source,
                       memory_failure_domain_identity identity,
                       std::uint32_t callback_attempt,
                       std::uint32_t trim_retry_count);
void update_atomic_peak(std::atomic<std::size_t>& peak, std::size_t value);
void update_peak(std::size_t& peak, std::size_t value);
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
                std::size_t largest_failed_allocation_bytes);

/// Shared live-allocation gate for one manager's base resource and every
/// private query-domain pool it creates.  A static partition is an ownership
/// boundary, so a query domain must consume the same limit rather than getting
/// an independent private-pool allowance.
class manager_allocation_limit_state final {
 public:
  explicit manager_allocation_limit_state(std::size_t limit_bytes) : limit_bytes_{limit_bytes} {}

  [[nodiscard]] bool try_reserve(std::size_t aligned_bytes) noexcept
  {
    auto observed = current_aligned_bytes_.load(std::memory_order_relaxed);
    while (true) {
      if (observed > limit_bytes_ || aligned_bytes > limit_bytes_ - observed) { return false; }
      if (current_aligned_bytes_.compare_exchange_weak(
            observed, observed + aligned_bytes, std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  void release(std::size_t aligned_bytes) noexcept
  {
    current_aligned_bytes_.fetch_sub(aligned_bytes, std::memory_order_relaxed);
  }

 private:
  std::size_t limit_bytes_;
  std::atomic<std::size_t> current_aligned_bytes_{0};
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
                        std::uint64_t domain_generation,
                        std::size_t grant_bytes,
                        std::size_t requested_ceiling_bytes,
                        std::size_t effective_ceiling_bytes,
                        std::shared_ptr<manager_allocation_limit_state> manager_allocation_limit);

  [[nodiscard]] cudf_memory_resource_usage_t usage(bool active, std::size_t active_members) const;

  [[nodiscard]] cudf_memory_resource_usage_t member_usage(
    std::shared_ptr<query_domain_member_counters> const& member, bool attached) const;

  [[nodiscard]] std::uint64_t domain_id() const noexcept;

  [[nodiscard]] std::size_t current_allocations() const noexcept;

  [[nodiscard]] std::size_t in_flight_operations() const noexcept;

  [[nodiscard]] operation_guard begin_operation() noexcept;

  bool resize_grant(std::size_t grant_bytes, std::string& error);

  void trim_to(std::uint64_t keep_bytes);

  bool trim_to_noexcept(std::uint64_t keep_bytes) noexcept;

  bool pool_usage(cudf_memory_pool_usage_t& out) const noexcept;

  void* allocate_for_member(std::shared_ptr<query_domain_member_counters> const& member,
                            rmm::cuda_stream_view stream,
                            std::size_t bytes,
                            std::size_t alignment,
                            bool require_attached = true);

  void deallocate_from_adaptor(void* ptr,
                               std::shared_ptr<query_domain_member_counters> const& member,
                               rmm::cuda_stream_view stream,
                               std::size_t bytes,
                               std::size_t alignment) noexcept;

 private:
  void record_failed_allocation(std::size_t bytes) noexcept;

  void record_member_failed_allocation(std::shared_ptr<query_domain_member_counters> const& member,
                                       std::size_t bytes) noexcept;

  static bool reserve_aligned_bytes(std::atomic<std::size_t>& current,
                                    std::atomic<std::size_t> const& limit,
                                    std::size_t aligned_bytes,
                                    std::size_t& reserved_total) noexcept;

  bool reserve_domain_aligned_bytes(std::size_t aligned_bytes,
                                    std::size_t& reserved_total) noexcept;

  [[nodiscard]] bool would_exceed_pool_ceiling(std::size_t aligned_bytes) const noexcept;

  [[nodiscard]] memory_failure_domain_identity failure_identity(
    std::shared_ptr<query_domain_member_counters> const& member) const noexcept;

  void* allocate_impl(std::shared_ptr<query_domain_member_counters> const& member,
                      rmm::cuda_stream_view stream,
                      std::size_t bytes,
                      std::size_t alignment,
                      bool require_attached = true);

  void deallocate_impl(void* ptr,
                       std::shared_ptr<query_domain_member_counters> const& member,
                       rmm::cuda_stream_view stream,
                       std::size_t bytes,
                       std::size_t alignment) noexcept;

  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream,
                    std::size_t alignment) override;

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override;

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override;

  rmm::mr::cuda_async_view_memory_resource view_;
  std::uint64_t domain_id_;
  std::uint64_t domain_generation_;
  std::atomic<std::size_t> grant_bytes_{0};
  std::size_t requested_ceiling_bytes_;
  std::size_t effective_ceiling_bytes_;
  std::shared_ptr<manager_allocation_limit_state> manager_allocation_limit_;
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
  std::uint64_t generation = 0;
  int32_t device   = -1;
  cudaMemPool_t pool{};
  std::size_t ceiling_bytes           = 0;
  std::size_t effective_ceiling_bytes = 0;
  bool active                         = true;
  std::atomic<bool> trim_pending{false};
  std::size_t active_members = 0;
  std::shared_ptr<query_domain_resource> resource;

  ~query_domain_state();
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
                               cleanup_callback cleanup);

  void* allocate_routed(rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment);

  void reserve_routed_allocation() noexcept;

  void commit_routed_allocation() noexcept;

  void cancel_routed_allocation() noexcept;

  void rollback_routed_allocation(void* ptr,
                                  rmm::cuda_stream_view stream,
                                  std::size_t bytes,
                                  std::size_t alignment) noexcept;

  void deallocate_routed(void* ptr,
                         rmm::cuda_stream_view stream,
                         std::size_t bytes,
                         std::size_t alignment) noexcept;

  [[nodiscard]] std::size_t routed_pending_allocations() const noexcept;

  void detach() noexcept;

  bool resize(std::size_t limit_bytes) noexcept;

 private:
  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream,
                    std::size_t alignment) override;

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override;

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override;

  std::shared_ptr<query_domain_state> domain_;
  std::shared_ptr<query_domain_member_counters> counters_;
  cleanup_callback cleanup_;
};

class shared_query_domain_member_resource final : public rmm::mr::device_memory_resource {
 public:
  explicit shared_query_domain_member_resource(
    std::shared_ptr<query_domain_member_resource> resource);

 private:
  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream,
                    std::size_t alignment) override;

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override;

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override;

  std::shared_ptr<query_domain_member_resource> resource_;
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
};

struct reservation_state {
  std::uint64_t id             = 0;
  std::size_t allocation_limit = 0;
  void* stream_key             = nullptr;
  bool attached                = true;
  std::shared_ptr<struct reservation_arena_state> arena;
  std::size_t current_bytes                   = 0;
  std::size_t peak_bytes                      = 0;
  std::size_t total_bytes                     = 0;
  std::size_t current_aligned_bytes           = 0;
  std::size_t peak_aligned_bytes              = 0;
  std::size_t total_allocations               = 0;
  std::size_t current_allocations             = 0;
  std::size_t peak_allocations                = 0;
  std::size_t failed_allocation_count         = 0;
  std::size_t largest_failed_allocation_bytes = 0;
};

struct reservation_arena_state {
  std::uint64_t id                            = 0;
  std::size_t allocation_limit                = 0;
  bool active                                 = true;
  std::size_t active_members                  = 0;
  std::size_t current_bytes                   = 0;
  std::size_t peak_bytes                      = 0;
  std::size_t total_bytes                     = 0;
  std::size_t current_aligned_bytes           = 0;
  std::size_t peak_aligned_bytes              = 0;
  std::size_t total_allocations               = 0;
  std::size_t current_allocations             = 0;
  std::size_t peak_allocations                = 0;
  std::size_t failed_allocation_count         = 0;
  std::size_t largest_failed_allocation_bytes = 0;
};

class reservation_aware_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  explicit reservation_aware_resource_adaptor(rmm::mr::device_memory_resource* upstream);

  std::shared_ptr<reservation_state> attach(rmm::cuda_stream_view stream,
                                            std::uint64_t reservation_id,
                                            std::size_t limit_bytes,
                                            std::shared_ptr<reservation_arena_state> arena,
                                            std::string& error);

  std::shared_ptr<reservation_arena_state> create_arena(std::uint64_t arena_id,
                                                        std::size_t limit_bytes);

  std::shared_ptr<query_domain_state> create_query_domain(std::uint64_t domain_id,
                                                          std::size_t grant_bytes,
                                                          std::size_t ceiling_bytes,
                                                          std::uint64_t release_threshold,
                                                          int32_t device,
                                                          std::shared_ptr<manager_allocation_limit_state> manager_allocation_limit,
                                                          std::string& error);

  std::shared_ptr<query_domain_member_state> attach_query_domain(
    std::shared_ptr<query_domain_state> const& domain,
    rmm::cuda_stream_view stream,
    std::uint64_t reservation_id,
    std::size_t limit_bytes,
    std::string& error);

  std::shared_ptr<struct current_resource_route_state> attach_current_resource_route(
    rmm::cuda_stream_view stream,
    std::shared_ptr<struct memory_resource_owner> target_owner,
    std::string& error);

  void detach_current_resource_route(
    std::shared_ptr<struct current_resource_route_state> const& state);

  void detach(std::shared_ptr<reservation_state> const& state);

  void detach_query_domain_member(std::shared_ptr<query_domain_member_state> const& state);

  void destroy_arena(std::shared_ptr<reservation_arena_state> const& state);

  void destroy_query_domain(std::shared_ptr<query_domain_state> const& state);

  cudf_memory_resource_usage_t usage(std::shared_ptr<reservation_state> const& state);

  cudf_memory_resource_usage_t arena_usage(std::shared_ptr<reservation_arena_state> const& state);

  cudf_memory_resource_usage_t query_domain_usage(std::shared_ptr<query_domain_state> const& state);

  [[nodiscard]] std::size_t query_domain_pool_count_for_id(std::uint64_t domain_id);

  cudf_memory_resource_usage_t query_domain_member_usage(
    std::shared_ptr<query_domain_member_state> const& state);

  bool resize(std::shared_ptr<reservation_state> const& state,
              std::size_t limit_bytes,
              std::string& error);

  bool resize_arena(std::shared_ptr<reservation_arena_state> const& state,
                    std::size_t limit_bytes,
                    std::string& error);

  bool resize_query_domain(std::shared_ptr<query_domain_state> const& state,
                           std::size_t grant_bytes,
                           std::string& error);

  bool resize_query_domain_member(std::shared_ptr<query_domain_member_state> const& state,
                                  std::size_t limit_bytes,
                                  std::string& error);

  void trim_query_domain(std::shared_ptr<query_domain_state> const& state,
                         std::uint64_t keep_bytes);

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
    std::size_t bytes         = 0;
    std::size_t aligned_bytes = 0;
    std::shared_ptr<reservation_state> reservation;
    std::shared_ptr<reservation_arena_state> arena;
  };

  static void* stream_key(rmm::cuda_stream_view stream);

  std::optional<query_domain_member_route> query_domain_member_for_stream(
    rmm::cuda_stream_view stream, std::size_t bytes);

  std::optional<query_domain_member_route> query_domain_member_for_deallocation(void* ptr);

  std::optional<query_domain_pointer_route> query_domain_for_pointer(void* ptr);

  std::shared_ptr<struct current_resource_route_state> current_resource_route_for_stream(
    rmm::cuda_stream_view stream);

  std::shared_ptr<struct current_resource_route_state> current_resource_route_for_deallocation(
    void* ptr);

  void cleanup_query_domain_if_idle(std::shared_ptr<query_domain_state> const& domain);

  void cleanup_retired_query_domain_member(
    std::shared_ptr<query_domain_member_state> const& member);

  template <typename State>
  void record_failed_allocation(State& state, std::size_t bytes);

  std::shared_ptr<reservation_state> reservation_for_stream(rmm::cuda_stream_view stream);

  template <typename State>
  void reserve_allocation(State& state,
                          std::size_t bytes,
                          std::size_t aligned_bytes,
                          char const* limit_message,
                          memory_failure_domain_identity identity);

  template <typename State>
  void commit_allocation(State& state, std::size_t bytes, std::size_t aligned_bytes);

  template <typename State>
  void release_reserved_alignment(State& state, std::size_t aligned_bytes);

  template <typename State>
  void rollback_reserved_allocation(State& state, std::size_t bytes, std::size_t aligned_bytes);

  template <typename State>
  void release_allocation(State& state, std::size_t bytes, std::size_t aligned_bytes);

  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream,
                    std::size_t alignment) override;

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override;

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override;

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
  std::mutex current_resource_route_mutex_;
  std::unordered_map<void*, std::shared_ptr<struct current_resource_route_state>>
    current_resource_routes_by_stream_;
  std::unordered_map<void*, std::shared_ptr<struct current_resource_route_state>>
    current_resource_routes_by_pointer_;
  std::unordered_map<void*, std::weak_ptr<reservation_state>> reservations_by_stream_;
  std::unordered_map<void*, allocation_record> allocations_;
};

class budget_tracking_resource_adaptor;

struct memory_resource_owner {
  std::shared_ptr<rmm::mr::device_memory_resource> upstream;
  std::shared_ptr<rmm::mr::device_memory_resource> pool;
  std::shared_ptr<void> upstream_lifetime;
  // Reverse member destruction tears this down before its physical base.
  std::unique_ptr<rmm::mr::device_memory_resource> limiting_mr;
  std::shared_ptr<manager_allocation_limit_state> manager_allocation_limit;
  std::unique_ptr<rmm::mr::device_memory_resource> reservation_mr;
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
  rmm::mr::statistics_resource_adaptor* stats = nullptr;
#endif
  budget_tracking_resource_adaptor* budget_tracker        = nullptr;
  reservation_aware_resource_adaptor* reservation_adaptor = nullptr;
  cuda_async_resource_box* async_pool                     = nullptr;
  int32_t device                                          = -1;
};

struct memory_resource_handle_state {
  std::shared_ptr<memory_resource_owner> owner;
  bool non_owning_alias = false;
};

struct current_resource_route_state {
  void* stream_key = nullptr;
  // Keep the explicit target resource, including its full native upstream
  // chain, alive until every routed allocation has been deallocated.
  std::shared_ptr<memory_resource_owner> target_owner;
  bool attached = true;
};

struct device_memory_manager_state {
  int32_t device = -1;
  std::shared_ptr<memory_resource_owner> owner;
  cudf_memory_manager_options_t options                 = default_memory_manager_options();
  cudf_memory_failure_callback_t failure_callback       = nullptr;
  cudf_memory_failure_callback_v2_t failure_callback_v2 = nullptr;
  void* failure_callback_arg                            = nullptr;
  bool failure_callback_installed                       = false;
  bool current_resource_installed                       = false;
  bool current_resource_install_retryable               = false;
  bool current_resource_integrity_lost                  = false;
  cudf_memory_manager_installation_cause_t current_resource_install_cause =
    CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_NONE;
  std::atomic<bool> failure_callback_panicked{false};
  bool has_installation_identity                        = false;
  cudf_memory_manager_installation_identity_t installation_identity{};
  std::uint64_t installation_generation                 = 0;
  /// Constructed before any allocator callback can be reached and retained for
  /// the process lifetime of this device manager.
  std::unique_ptr<allocator_event_ring> allocator_events;
};

struct current_resource_route_handle_state {
  std::shared_ptr<device_memory_manager_state> manager;
  std::shared_ptr<current_resource_route_state> route;
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

memory_resource_handle_state& memory_resource_state_from_handle(cudf_memory_resource_handle handle);
memory_resource_owner& memory_resource_owner_from_handle(cudf_memory_resource_handle handle);
void register_memory_resource_lifetime(std::shared_ptr<memory_resource_owner> const& owner);
int32_t current_cuda_device();
std::shared_ptr<device_memory_manager_state> memory_manager_from_handle(
  cudf_memory_manager_handle handle);
cudf_status_t validate_stream_device_for_manager(cudf_stream_handle stream,
                                                 int32_t manager_device,
                                                 char const* error_message);
std::unique_ptr<memory_resource_handle_state> make_query_domain_member_resource_handle(
  std::shared_ptr<device_memory_manager_state> const& manager,
  std::shared_ptr<query_domain_member_state> const& member);
std::unique_ptr<rmm::mr::device_memory_resource> make_bounded_failure_callback_resource(
  rmm::mr::device_memory_resource* upstream,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudaMemPool_t pool,
  int32_t device_ordinal,
  allocator_event_ring* allocator_events,
  std::atomic<bool>* callback_panicked = nullptr);
std::unique_ptr<rmm::mr::device_memory_resource> make_bounded_failure_callback_resource(
  rmm::mr::device_memory_resource* upstream,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudaMemPool_t pool,
  int32_t device_ordinal,
  allocator_event_ring* allocator_events,
  std::atomic<bool>* callback_panicked = nullptr);

}  // namespace cudf::rust::memory

#endif  // CUDF_RUST_MEMORY_INTERNAL_HPP
