#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory/internal.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <rmm/error.hpp>

#include <algorithm>
#include <stdexcept>

namespace cudf::rust::memory {

namespace {

std::atomic<std::uint64_t>& next_query_domain_generation()
{
  static std::atomic<std::uint64_t> next{1};
  return next;
}

}  // namespace

void query_domain_allocation_registry::insert(void* ptr, query_domain_allocation_record record)
{
  if (ptr == nullptr) { throw std::invalid_argument{"query memory domain allocation pointer is null"}; }
  std::scoped_lock lock{mutex_};
  auto const [_, inserted] = allocations_.emplace(ptr, std::move(record));
  if (!inserted) { throw std::runtime_error{"duplicate query memory domain allocation pointer"}; }
}

std::optional<query_domain_allocation_record> query_domain_allocation_registry::take(
  void* ptr) noexcept
{
  if (ptr == nullptr) { return {}; }
  std::scoped_lock lock{mutex_};
  auto const it = allocations_.find(ptr);
  if (it == allocations_.end()) { return {}; }
  auto record = std::move(it->second);
  allocations_.erase(it);
  return record;
}

std::vector<std::pair<void*, query_domain_allocation_record>>
query_domain_allocation_registry::live_for_domain(
  std::shared_ptr<query_domain_state> const& domain)
{
  std::vector<std::pair<void*, query_domain_allocation_record>> live;
  if (domain == nullptr) { return live; }
  {
    std::scoped_lock lock{mutex_};
    live.reserve(allocations_.size());
    for (auto const& [ptr, record] : allocations_) {
      if (record.domain == domain) { live.emplace_back(ptr, record); }
    }
  }
  std::sort(live.begin(), live.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.first < rhs.first;
  });
  return live;
}

void query_domain_allocation_registry::record_invariant_violation(
  invariant_violation violation) noexcept
{
  auto expected = invariant_violation::none;
  invariant_violation_.compare_exchange_strong(
    expected, violation, std::memory_order_release, std::memory_order_relaxed);
}

void query_domain_allocation_registry::throw_if_invariant_violated() const
{
  switch (invariant_violation_.load(std::memory_order_acquire)) {
    case invariant_violation::none: return;
    case invariant_violation::unknown_query_pointer:
      throw cudf::logic_error{
        "allocator invariant violation: unknown or already-freed query memory domain pointer"};
    case invariant_violation::unknown_device_pointer:
      throw cudf::logic_error{
        "allocator invariant violation: unknown or already-freed device allocation pointer"};
  }
  throw cudf::logic_error{"allocator invariant violation"};
}

query_domain_resource::query_domain_resource(
  cudaMemPool_t pool,
  std::uint64_t domain_id,
  std::uint64_t domain_generation,
  std::size_t grant_bytes,
  std::size_t requested_ceiling_bytes,
  std::shared_ptr<manager_allocation_limit_state> manager_allocation_limit)
  : view_{pool},
    domain_id_{domain_id},
    domain_generation_{domain_generation},
    grant_bytes_{grant_bytes},
    requested_ceiling_bytes_{requested_ceiling_bytes},
    manager_allocation_limit_{std::move(manager_allocation_limit)}
{
}

[[nodiscard]] cudf_memory_resource_usage_t query_domain_resource::usage(
  bool active, std::size_t active_members) const
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
  return out;
}

[[nodiscard]] cudf_memory_resource_usage_t query_domain_resource::member_usage(
  std::shared_ptr<query_domain_member_counters> const& member, bool attached) const
{
  cudf_memory_resource_usage_t out{};
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

[[nodiscard]] std::uint64_t query_domain_resource::domain_id() const noexcept { return domain_id_; }

[[nodiscard]] std::size_t query_domain_resource::current_allocations() const noexcept
{
  return current_allocations_.load(std::memory_order_relaxed);
}

[[nodiscard]] std::size_t query_domain_resource::in_flight_operations() const noexcept
{
  return in_flight_operations_.load(std::memory_order_acquire);
}

[[nodiscard]] query_domain_resource::operation_guard
query_domain_resource::begin_operation() noexcept
{
  in_flight_operations_.fetch_add(1, std::memory_order_acq_rel);
  return operation_guard{this};
}

bool query_domain_resource::resize_grant(std::size_t grant_bytes, std::string& error)
{
  if (grant_bytes > requested_ceiling_bytes_) {
    error = "query memory domain grant exceeds immutable pool ceiling";
    return false;
  }
  grant_bytes_.store(grant_bytes, std::memory_order_relaxed);
  return true;
}

bool query_domain_resource::freeze(std::string& error)
{
  bool expected = true;
  if (!allocation_open_.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
    return true;
  }
  if (in_flight_operations_.load(std::memory_order_acquire) != 0) {
    allocation_open_.store(true, std::memory_order_release);
    error = "query memory domain has in-flight allocation operations";
    return false;
  }
  return true;
}

bool query_domain_resource::pool_usage(cudf_memory_pool_usage_t& out) const noexcept
{
  return read_cuda_pool_usage_noexcept(view_.pool_handle(), out);
}

void* query_domain_resource::allocate_for_member(
  std::shared_ptr<query_domain_member_counters> const& member,
  rmm::cuda_stream_view stream,
  std::size_t bytes,
  std::size_t alignment,
  bool require_attached)
{
  return allocate_impl(member, stream, bytes, alignment, require_attached);
}

void query_domain_resource::deallocate_from_adaptor(
  void* ptr,
  std::shared_ptr<query_domain_member_counters> const& member,
  rmm::cuda_stream_view stream,
  std::size_t bytes,
  std::size_t alignment) noexcept
{
  deallocate_impl(ptr, member, stream, bytes, alignment);
}

void query_domain_resource::record_failed_allocation(std::size_t bytes) noexcept
{
  failed_allocation_count_.fetch_add(1, std::memory_order_relaxed);
  update_atomic_peak(largest_failed_allocation_bytes_, bytes);
}

void query_domain_resource::record_member_failed_allocation(
  std::shared_ptr<query_domain_member_counters> const& member, std::size_t bytes) noexcept
{
  if (member == nullptr) { return; }
  member->failed_allocation_count.fetch_add(1, std::memory_order_relaxed);
  update_atomic_peak(member->largest_failed_allocation_bytes, bytes);
}

bool query_domain_resource::reserve_aligned_bytes(std::atomic<std::size_t>& current,
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

bool query_domain_resource::reserve_domain_aligned_bytes(std::size_t aligned_bytes,
                                                         std::size_t& reserved_total) noexcept
{
  return reserve_aligned_bytes(current_aligned_bytes_, grant_bytes_, aligned_bytes, reserved_total);
}

[[nodiscard]] memory_failure_domain_identity query_domain_resource::failure_identity(
  std::shared_ptr<query_domain_member_counters> const& member) const noexcept
{
  return memory_failure_domain_identity{
    CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN,
    domain_id_,
    member == nullptr ? 0 : member->id,
    domain_generation_,
  };
}

void* query_domain_resource::allocate_impl(
  std::shared_ptr<query_domain_member_counters> const& member,
  rmm::cuda_stream_view stream,
  std::size_t bytes,
  std::size_t alignment,
  bool require_attached)
{
  if (bytes == 0) { return view_.allocate(stream, bytes, alignment); }
  if (!allocation_open_.load(std::memory_order_acquire)) {
    // A frozen domain rejecting new work is a routing/lifetime invariant
    // violation, not caller input: the attempt was supposed to stop
    // allocating before its output was published.
    throw cudf::logic_error{
      "allocator invariant violation: query memory domain is frozen to new allocations"};
  }
  auto const aligned_bytes          = cudf::rust::reservation::allocation_charge(bytes, alignment);
  std::size_t member_reserved_total = 0;
  if (member != nullptr && require_attached && !member->attached.load(std::memory_order_acquire)) {
    throw std::invalid_argument{"query memory domain member is detached"};
  }
  if (member != nullptr && !reserve_aligned_bytes(member->current_aligned_bytes,
                                                  member->allocation_limit,
                                                  aligned_bytes,
                                                  member_reserved_total)) {
    record_member_failed_allocation(member, bytes);
    throw attributed_out_of_memory{"query memory domain member limit exceeded",
                                   CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
                                   failure_identity(member),
                                   view_.pool_handle(),
                                   bytes,
                                   alignment};
  }

  std::size_t domain_reserved_total = 0;
  if (!reserve_domain_aligned_bytes(aligned_bytes, domain_reserved_total)) {
    if (member != nullptr) { member->current_aligned_bytes.fetch_sub(aligned_bytes); }
    record_failed_allocation(bytes);
    record_member_failed_allocation(member, bytes);
    throw attributed_out_of_memory{"query memory domain grant exceeded",
                                   CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
                                   failure_identity(member),
                                   view_.pool_handle(),
                                   bytes,
                                   alignment};
  }

  if (manager_allocation_limit_ != nullptr &&
      !manager_allocation_limit_->try_reserve(aligned_bytes)) {
    current_aligned_bytes_.fetch_sub(aligned_bytes);
    if (member != nullptr) { member->current_aligned_bytes.fetch_sub(aligned_bytes); }
    record_failed_allocation(bytes);
    record_member_failed_allocation(member, bytes);
    throw attributed_out_of_memory{"manager allocation limit exceeded",
                                   CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
                                   failure_identity(member),
                                   view_.pool_handle(),
                                   bytes,
                                   alignment};
  }

  void* ptr = nullptr;
  try {
    if (cudf::rust::test_support::take_query_domain_allocation_failure(domain_id_)) {
      throw rmm::out_of_memory{"injected query-domain physical allocation failure"};
    }
    ptr = view_.allocate(stream, bytes, alignment);
  } catch (rmm::out_of_memory const&) {
    if (manager_allocation_limit_ != nullptr) { manager_allocation_limit_->release(aligned_bytes); }
    current_aligned_bytes_.fetch_sub(aligned_bytes);
    if (member != nullptr) { member->current_aligned_bytes.fetch_sub(aligned_bytes); }
    record_failed_allocation(bytes);
    record_member_failed_allocation(member, bytes);
    throw attributed_out_of_memory{"query memory domain physical allocation failed",
                                   CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL,
                                   failure_identity(member),
                                   view_.pool_handle(),
                                   bytes,
                                   alignment};
  } catch (...) {
    if (manager_allocation_limit_ != nullptr) { manager_allocation_limit_->release(aligned_bytes); }
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

void query_domain_resource::deallocate_impl(
  void* ptr,
  std::shared_ptr<query_domain_member_counters> const& member,
  rmm::cuda_stream_view stream,
  std::size_t bytes,
  std::size_t alignment) noexcept
{
  if (bytes == 0) {
    view_.deallocate(stream, ptr, bytes, alignment);
    return;
  }
  auto const aligned_bytes = cudf::rust::reservation::allocation_charge(bytes, alignment);
  view_.deallocate(stream, ptr, bytes, alignment);
  if (manager_allocation_limit_ != nullptr) { manager_allocation_limit_->release(aligned_bytes); }
  current_aligned_bytes_.fetch_sub(aligned_bytes);
  current_bytes_.fetch_sub(bytes);
  current_allocations_.fetch_sub(1);

  if (member != nullptr) {
    member->current_aligned_bytes.fetch_sub(aligned_bytes);
    member->current_bytes.fetch_sub(bytes);
    member->current_allocations.fetch_sub(1);
  }
}

void* query_domain_resource::do_allocate(std::size_t bytes,
                                         rmm::cuda_stream_view stream,
                                         std::size_t alignment)
{
  return allocate_impl({}, stream, bytes, alignment);
}

void query_domain_resource::do_deallocate(void* ptr,
                                          std::size_t bytes,
                                          rmm::cuda_stream_view stream,
                                          std::size_t alignment) noexcept
{
  deallocate_impl(ptr, {}, stream, bytes, alignment);
}

[[nodiscard]] bool query_domain_resource::do_is_equal(
  rmm::mr::device_memory_resource const& other) const noexcept
{
  return this == std::addressof(other);
}

query_domain_member_resource::query_domain_member_resource(
  std::shared_ptr<query_domain_state> domain,
  std::shared_ptr<query_domain_member_counters> counters,
  std::shared_ptr<query_domain_allocation_registry> allocations,
  cleanup_callback cleanup)
  : domain_{std::move(domain)},
    counters_{std::move(counters)},
    allocations_{std::move(allocations)},
    cleanup_{std::move(cleanup)}
{
}

void* query_domain_member_resource::allocate_routed(rmm::cuda_stream_view stream,
                                                    std::size_t bytes,
                                                    std::size_t alignment)
{
  allocations_->throw_if_invariant_violated();
  void* ptr = nullptr;
  try {
    ptr = domain_->resource->allocate_for_member(counters_, stream, bytes, alignment, false);
    if (bytes != 0) {
      allocations_->insert(ptr,
                           query_domain_allocation_record{domain_,
                                                          counters_,
                                                          bytes,
                                                          cudf::rust::reservation::allocation_charge(
                                                            bytes, alignment),
                                                          alignment,
                                                          reinterpret_cast<void*>(stream.value()),
                                                          true});
    }
    return ptr;
  } catch (...) {
    if (ptr != nullptr && bytes != 0) {
      domain_->resource->deallocate_from_adaptor(ptr, counters_, stream, bytes, alignment);
    }
    if (bytes != 0) { cancel_routed_allocation(); }
    throw;
  }
}

void query_domain_member_resource::reserve_routed_allocation() noexcept
{
  counters_->routed_pending_allocations.fetch_add(1, std::memory_order_acq_rel);
  counters_->routed_current_allocations.fetch_add(1, std::memory_order_acq_rel);
}

void query_domain_member_resource::commit_routed_allocation() noexcept
{
  counters_->routed_pending_allocations.fetch_sub(1, std::memory_order_acq_rel);
  counters_->routed_total_allocations.fetch_add(1, std::memory_order_relaxed);
  cleanup_(domain_);
}

void query_domain_member_resource::cancel_routed_allocation() noexcept
{
  counters_->routed_pending_allocations.fetch_sub(1, std::memory_order_acq_rel);
  counters_->routed_current_allocations.fetch_sub(1, std::memory_order_acq_rel);
  cleanup_(domain_);
}

[[nodiscard]] std::size_t query_domain_member_resource::routed_pending_allocations() const noexcept
{
  return counters_->routed_pending_allocations.load(std::memory_order_acquire);
}

void query_domain_member_resource::detach() noexcept
{
  counters_->attached.store(false, std::memory_order_release);
}

bool query_domain_member_resource::resize(std::size_t limit_bytes) noexcept
{
  counters_->allocation_limit.store(limit_bytes, std::memory_order_release);
  return true;
}

void* query_domain_member_resource::do_allocate(std::size_t bytes,
                                                rmm::cuda_stream_view stream,
                                                std::size_t alignment)
{
  allocations_->throw_if_invariant_violated();
  auto* ptr = domain_->resource->allocate_for_member(counters_, stream, bytes, alignment);
  if (bytes != 0) {
    try {
      allocations_->insert(ptr,
                           query_domain_allocation_record{domain_,
                                                          counters_,
                                                          bytes,
                                                          cudf::rust::reservation::allocation_charge(
                                                            bytes, alignment),
                                                          alignment,
                                                          reinterpret_cast<void*>(stream.value()),
                                                          false});
    } catch (...) {
      domain_->resource->deallocate_from_adaptor(ptr, counters_, stream, bytes, alignment);
      throw;
    }
  }
  counters_->direct_total_allocations.fetch_add(1, std::memory_order_relaxed);
  return ptr;
}

void query_domain_member_resource::do_deallocate(void* ptr,
                                                 std::size_t bytes,
                                                 rmm::cuda_stream_view stream,
                                                 std::size_t alignment) noexcept
{
  // A zero-byte allocation is never registered, so it owns no counters and its
  // pointer is not a registry member. Releasing it must stay inert: a registry
  // lookup here would miss and latch the unknown-pointer invariant.
  if (bytes == 0) {
    domain_->resource->deallocate_from_adaptor(ptr, counters_, stream, bytes, alignment);
    cleanup_(domain_);
    return;
  }
  auto const record = allocations_->take(ptr);
  if (!record.has_value()) {
    allocations_->record_invariant_violation(
      query_domain_allocation_registry::invariant_violation::unknown_query_pointer);
    set_error("unknown or already-freed query memory domain pointer");
    return;
  }
  record->domain->resource->deallocate_from_adaptor(
    ptr, record->member, stream, record->bytes, record->alignment);
  if (record->routed) {
    record->member->routed_current_allocations.fetch_sub(1, std::memory_order_acq_rel);
  }
  cleanup_(record->domain);
}

[[nodiscard]] bool query_domain_member_resource::do_is_equal(
  rmm::mr::device_memory_resource const& other) const noexcept
{
  return this == std::addressof(other);
}

shared_query_domain_member_resource::shared_query_domain_member_resource(
  std::shared_ptr<query_domain_member_resource> resource)
  : resource_{std::move(resource)},
    upstream_{rmm::mr::any_device_resource_from_raw(resource_.get())}
{
}

void* shared_query_domain_member_resource::do_allocate(std::size_t bytes,
                                                       rmm::cuda_stream_view stream,
                                                       std::size_t alignment)
{
  return upstream_.allocate(stream, bytes, alignment);
}

void shared_query_domain_member_resource::do_deallocate(void* ptr,
                                                        std::size_t bytes,
                                                        rmm::cuda_stream_view stream,
                                                        std::size_t alignment) noexcept
{
  upstream_.deallocate(stream, ptr, bytes, alignment);
}

[[nodiscard]] bool shared_query_domain_member_resource::do_is_equal(
  rmm::mr::device_memory_resource const& other) const noexcept
{
  return this == std::addressof(other);
}

std::shared_ptr<query_domain_state> reservation_aware_resource_adaptor::create_query_domain(
  std::uint64_t domain_id,
  std::size_t grant_bytes,
  std::size_t ceiling_bytes,
  int32_t device,
  cudaMemPool_t pool,
  std::shared_ptr<manager_allocation_limit_state> manager_allocation_limit,
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
  if (pool == nullptr) {
    error = "query memory domains require a manager-owned CUDA async pool";
    return {};
  }
  auto state           = std::make_shared<query_domain_state>();
  state->id            = domain_id;
  state->generation    = next_query_domain_generation().fetch_add(1, std::memory_order_relaxed);
  state->device        = device;
  state->pool          = pool;
  state->ceiling_bytes = ceiling_bytes;
  state->resource      = std::make_shared<query_domain_resource>(pool,
                                                             domain_id,
                                                             state->generation,
                                                             grant_bytes,
                                                             state->ceiling_bytes,
                                                             std::move(manager_allocation_limit));
  {
    std::unique_lock lock{query_domain_mutex_};
    query_domains_by_generation_.emplace(state->generation, state);
  }
  return state;
}

std::shared_ptr<query_domain_member_state> reservation_aware_resource_adaptor::attach_query_domain(
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

  auto state          = std::make_shared<query_domain_member_state>();
  state->id           = reservation_id;
  state->stream_key   = key;
  state->domain       = domain;
  state->counters     = std::make_shared<query_domain_member_counters>();
  state->counters->id = reservation_id;
  state->counters->allocation_limit.store(limit_bytes, std::memory_order_relaxed);
  std::weak_ptr<query_domain_member_state> weak_state = state;
  state->resource = std::make_shared<query_domain_member_resource>(
    domain,
    state->counters,
    query_domain_allocations_,
    [this, weak_state](std::shared_ptr<query_domain_state> const& member_domain) {
      if (auto member = weak_state.lock()) { cleanup_retired_query_domain_member(member); }
      cleanup_query_domain_if_idle(member_domain);
    });
  domain->active_members += 1;
  query_domain_members_by_stream_[key] = state;
  query_domain_member_count_.fetch_add(1, std::memory_order_relaxed);
  return state;
}

void reservation_aware_resource_adaptor::detach_query_domain_member(
  std::shared_ptr<query_domain_member_state> const& state)
{
  if (state == nullptr) { return; }
  std::shared_ptr<query_domain_state> domain;
  {
    std::unique_lock lock{query_domain_mutex_};
    bool const was_attached = state->attached;
    state->attached         = false;
    if (state->resource != nullptr) { state->resource->detach(); }
    domain              = state->domain;
    auto const existing = query_domain_members_by_stream_.find(state->stream_key);
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

void reservation_aware_resource_adaptor::destroy_query_domain(
  std::shared_ptr<query_domain_state> const& state)
{
  if (state == nullptr) { return; }
  {
    std::unique_lock lock{query_domain_mutex_};
    state->active = false;
  }
  cleanup_query_domain_if_idle(state);
}

bool reservation_aware_resource_adaptor::freeze_query_domain(
  std::shared_ptr<query_domain_state> const& state, std::string& error)
{
  std::unique_lock lock{query_domain_mutex_};
  if (!state->active) {
    error = "query memory domain is not active";
    return false;
  }
  if (state->active_members != 0) {
    error = "query memory domain still has active stream members";
    return false;
  }
  return state->resource->freeze(error);
}

std::vector<std::pair<void*, query_domain_allocation_record>>
reservation_aware_resource_adaptor::live_query_domain_allocations(
  std::shared_ptr<query_domain_state> const& state)
{
  return query_domain_allocations_->live_for_domain(state);
}

cudf_memory_resource_usage_t reservation_aware_resource_adaptor::query_domain_usage(
  std::shared_ptr<query_domain_state> const& state)
{
  std::shared_lock lock{query_domain_mutex_};
  return state->resource->usage(state->active, state->active_members);
}

[[nodiscard]] std::size_t reservation_aware_resource_adaptor::query_domain_count_for_id(
  std::uint64_t domain_id)
{
  std::shared_lock lock{query_domain_mutex_};
  return static_cast<std::size_t>(std::count_if(
    query_domains_by_generation_.cbegin(),
    query_domains_by_generation_.cend(),
    [domain_id](auto const& entry) { return entry.second->id == domain_id; }));
}

cudf_memory_resource_usage_t reservation_aware_resource_adaptor::query_domain_member_usage(
  std::shared_ptr<query_domain_member_state> const& state)
{
  std::shared_lock lock{query_domain_mutex_};
  return state->domain->resource->member_usage(state->counters, state->attached);
}

bool reservation_aware_resource_adaptor::resize_query_domain(
  std::shared_ptr<query_domain_state> const& state, std::size_t grant_bytes, std::string& error)
{
  std::shared_lock lock{query_domain_mutex_};
  if (!state->active) {
    error = "query memory domain is not active";
    return false;
  }
  return state->resource->resize_grant(grant_bytes, error);
}

bool reservation_aware_resource_adaptor::resize_query_domain_member(
  std::shared_ptr<query_domain_member_state> const& state,
  std::size_t limit_bytes,
  std::string& error)
{
  std::shared_lock lock{query_domain_mutex_};
  if (!state->attached) {
    error = "query memory domain member is not attached";
    return false;
  }
  return state->resource->resize(limit_bytes);
}

std::optional<reservation_aware_resource_adaptor::query_domain_member_route>
reservation_aware_resource_adaptor::query_domain_member_for_stream(rmm::cuda_stream_view stream,
                                                                   std::size_t bytes)
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

void reservation_aware_resource_adaptor::cleanup_query_domain_if_idle(
  std::shared_ptr<query_domain_state> const& domain)
{
  if (domain == nullptr || domain->resource == nullptr) { return; }
  if (domain->resource->in_flight_operations() != 0) { return; }
  if (domain->resource->current_allocations() != 0) { return; }
  std::unique_lock lock{query_domain_mutex_};
  if (domain->resource->in_flight_operations() != 0) { return; }
  if (domain->active || domain->active_members != 0 ||
      domain->resource->current_allocations() != 0) {
    return;
  }
  auto const it = query_domains_by_generation_.find(domain->generation);
  if (it != query_domains_by_generation_.end() && it->second.get() == domain.get()) {
    query_domains_by_generation_.erase(it);
  }
}

void reservation_aware_resource_adaptor::cleanup_retired_query_domain_member(
  std::shared_ptr<query_domain_member_state> const& member)
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
    owner.mr             = make_bounded_failure_callback_resource(owner.reservation_mr.get(),
                                                      manager->failure_callback_v2,
                                                      manager->failure_callback_arg,
                                                      member->domain->pool,
                                                      manager->device,
                                                      manager->allocator_events.get(),
                                                      &manager->failure_callback_panicked);
  } else if (manager->failure_callback != nullptr) {
    owner.reservation_mr = std::move(direct);
    owner.mr             = make_bounded_failure_callback_resource(owner.reservation_mr.get(),
                                                      manager->failure_callback,
                                                      manager->failure_callback_arg,
                                                      member->domain->pool,
                                                      manager->device,
                                                      manager->allocator_events.get(),
                                                      &manager->failure_callback_panicked);
  } else {
    owner.mr = std::move(direct);
  }
  register_memory_resource_lifetime(state->owner);
  return state;
}

}  // namespace cudf::rust::memory

using namespace cudf::rust::memory;

extern "C" {
cudf_status_t cudf_memory_manager_create_query_memory_domain(cudf_memory_manager_handle manager,
                                                             uint64_t domain_id,
                                                             size_t grant_bytes,
                                                             size_t ceiling_bytes,
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
                                               manager_state->device,
                                               async_pool_handle(manager_state->owner->async_pool),
                                               manager_state->owner->manager_allocation_limit,
                                               create_error);
    if (domain == nullptr) {
      set_error(create_error);
      return CUDF_ERROR_INVALID_ARGUMENT;
    }

    auto state     = std::make_unique<query_domain_handle_state>();
    state->manager = std::move(manager_state);
    state->domain  = std::move(domain);
    *out           = reinterpret_cast<cudf_query_memory_domain_handle>(state.release());
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
  *out          = nullptr;
  *out_resource = nullptr;
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_TRY
  {
    auto& domain_state                = *reinterpret_cast<query_domain_handle_state*>(domain);
    cudf_status_t const stream_status = validate_stream_device_for_manager(
      stream,
      domain_state.manager->device,
      "query memory domain requires stream and domain manager to be on the "
      "same CUDA device");
    if (stream_status != CUDF_SUCCESS) { return stream_status; }
    auto* adaptor = domain_state.manager->owner->reservation_adaptor;
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
      *out_resource = reinterpret_cast<cudf_memory_resource_handle>(resource_state.release());
      *out          = reinterpret_cast<cudf_reservation_handle>(state.release());
    } catch (...) {
      adaptor->detach_query_domain_member(member);
      throw;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_usage(cudf_query_memory_domain_handle domain,
                                             cudf_memory_resource_usage_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto& state   = *reinterpret_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    *out          = adaptor->query_domain_usage(state.domain);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_identity(cudf_query_memory_domain_handle domain,
                                                cudf_query_memory_domain_identity_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto const& state   = *reinterpret_cast<query_domain_handle_state*>(domain);
    out->struct_size    = sizeof(cudf_query_memory_domain_identity_t);
    out->device_ordinal = state.manager->device;
    out->domain_id      = state.domain->id;
    out->generation     = state.domain->generation;
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
    auto& state = *reinterpret_cast<query_domain_handle_state*>(domain);
    if (!state.domain->resource->pool_usage(*out)) {
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
    auto& state   = *reinterpret_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    std::string resize_error;
    if (!adaptor->resize_query_domain(state.domain, grant_bytes, resize_error)) {
      set_error(resize_error);
      return CUDF_ERROR_INVALID_ARGUMENT;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_freeze(cudf_query_memory_domain_handle domain)
{
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto& state   = *reinterpret_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    std::string freeze_error;
    if (!adaptor->freeze_query_domain(state.domain, freeze_error)) {
      set_error(freeze_error);
      return CUDF_ERROR_INVALID_OPERATION;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_live_allocations(
  cudf_query_memory_domain_handle domain,
  cudf_query_memory_domain_allocation_t* out,
  size_t capacity,
  size_t* out_count)
{
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_NULL_CHECK(out_count, "query memory domain live allocation count is null");
  CUDF_RUST_TRY
  {
    auto& state   = *reinterpret_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state.manager->owner->reservation_adaptor;
    auto const live = adaptor->live_query_domain_allocations(state.domain);
    *out_count      = live.size();
    if (out == nullptr) { return CUDF_SUCCESS; }
    auto const copied = std::min(capacity, live.size());
    for (std::size_t index = 0; index < copied; ++index) {
      auto const& [ptr, record] = live[index];
      out[index]                = cudf_query_memory_domain_allocation_t{
                       sizeof(cudf_query_memory_domain_allocation_t),
                       ptr,
                       record.bytes,
                       record.aligned_bytes,
                       record.alignment,
                       {0, 0}};
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_query_memory_domain_destroy(cudf_query_memory_domain_handle domain)
{
  CUDF_RUST_NULL_CHECK(domain, "query memory domain is null");
  CUDF_RUST_TRY
  {
    auto* state   = reinterpret_cast<query_domain_handle_state*>(domain);
    auto* adaptor = state->manager->owner->reservation_adaptor;
    adaptor->destroy_query_domain(state->domain);
    delete state;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
