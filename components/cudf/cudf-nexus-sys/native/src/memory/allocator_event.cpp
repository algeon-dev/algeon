#include "memory/internal.hpp"

#include <ctime>
#include <limits>

namespace cudf::rust::memory {

namespace {

[[nodiscard]] std::uint64_t monotonic_time_ns(std::uint32_t& completeness) noexcept
{
  timespec timestamp{};
  if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0 || timestamp.tv_sec < 0 ||
      timestamp.tv_nsec < 0) {
    return 0;
  }
  constexpr auto k_nanoseconds_per_second = std::uint64_t{1'000'000'000};
  auto const seconds                      = static_cast<std::uint64_t>(timestamp.tv_sec);
  if (seconds > std::numeric_limits<std::uint64_t>::max() / k_nanoseconds_per_second) { return 0; }
  completeness |= CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_TIMESTAMP;
  return seconds * k_nanoseconds_per_second + static_cast<std::uint64_t>(timestamp.tv_nsec);
}

void initialize_event(cudf_allocator_event_t& out) noexcept
{
  out             = {};
  out.struct_size = sizeof(cudf_allocator_event_t);
}

}  // namespace

void allocator_event_ring::record(cudf_memory_failure_info_t const& info,
                                  memory_failure_domain_identity identity,
                                  cudf_memory_failure_action_t action) noexcept
{
  // Every writer reserves a distinct sequence without waiting for a reader.
  // Payload words are atomic because a producer can wrap to a slot while a
  // lagging reader is copying it. The release publication stamp makes the
  // matching payload visible to a successful reader.
  auto const sequence = next_sequence_.fetch_add(1, std::memory_order_relaxed);
  auto& target        = slots_[(sequence - 1) % k_capacity];

  // A slot admits at most one writer at a time. Waiting here would violate
  // the callback contract, so a contending sequence deliberately has no
  // stable record and is later read as unavailable. The stale-writer check
  // also prevents a delayed, lower sequence from clobbering a newer event
  // after the ring wraps.
  if (target.writer_active.test_and_set(std::memory_order_acquire)) { return; }
  auto const published_sequence = target.published_sequence.load(std::memory_order_acquire);
  if (published_sequence > sequence) {
    target.writer_active.clear(std::memory_order_release);
    return;
  }

  // Invalidate a previous publication before overwriting its payload. Without
  // this transition, a reader that observed the old sequence could copy the
  // new payload before this writer publishes its sequence and incorrectly
  // report an available old record. The final acquire check in `read` turns
  // that concurrent overwrite into Pending or Overwritten instead.
  target.published_sequence.store(0, std::memory_order_release);

  std::uint32_t completeness = CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_ACTION;
  auto const time_ns         = monotonic_time_ns(completeness);
  bool const has_domain =
    identity.kind != CUDF_MEMORY_FAILURE_DOMAIN_NONE && identity.generation != 0;
  if (has_domain) { completeness |= CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_DOMAIN_IDENTITY; }
  if ((info.flags & CUDF_MEMORY_FAILURE_INFO_HAS_POOL_USAGE) != 0) {
    completeness |= CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_POOL_USAGE;
  }

  // A reader that observes any replacement payload with acquire semantics
  // thereby observes the invalidation above before its final publication
  // check. That makes a concurrent replacement Pending/Overwritten rather
  // than an available record with a mixed payload.
  target.monotonic_time_ns.store(time_ns, std::memory_order_release);
  target.device_ordinal.store(info.device_ordinal, std::memory_order_release);
  target.completeness.store(completeness, std::memory_order_release);
  target.domain_kind.store(
    has_domain ? static_cast<std::uint32_t>(identity.kind) : CUDF_MEMORY_FAILURE_DOMAIN_NONE,
    std::memory_order_release);
  target.failure_source.store(info.failure_source, std::memory_order_release);
  target.action.store(static_cast<std::int32_t>(action), std::memory_order_release);
  target.callback_attempt.store(info.callback_attempt, std::memory_order_release);
  target.trim_retry_count.store(info.trim_retry_count, std::memory_order_release);
  target.domain_id.store(has_domain ? identity.primary_id : 0, std::memory_order_release);
  target.domain_generation.store(has_domain ? identity.generation : 0, std::memory_order_release);
  target.domain_member_id.store(has_domain ? identity.secondary_id : 0, std::memory_order_release);
  target.requested_bytes.store(info.requested_bytes, std::memory_order_release);
  target.pool_reserved_current_bytes.store(info.pool_reserved_current_bytes,
                                           std::memory_order_release);
  target.pool_reserved_high_bytes.store(info.pool_reserved_high_bytes, std::memory_order_release);
  target.pool_used_current_bytes.store(info.pool_used_current_bytes, std::memory_order_release);
  target.pool_used_high_bytes.store(info.pool_used_high_bytes, std::memory_order_release);
  target.published_sequence.store(sequence, std::memory_order_release);
  target.writer_active.clear(std::memory_order_release);
}

[[nodiscard]] cudf_allocator_event_ring_bounds_t allocator_event_ring::bounds() const noexcept
{
  cudf_allocator_event_ring_bounds_t out{};
  out.struct_size   = sizeof(cudf_allocator_event_ring_bounds_t);
  out.capacity      = k_capacity;
  out.next_sequence = next_sequence_.load(std::memory_order_acquire);
  if (out.next_sequence > 1) {
    out.first_available_sequence =
      out.next_sequence > k_capacity ? out.next_sequence - k_capacity : std::uint64_t{1};
  }
  return out;
}

[[nodiscard]] cudf_allocator_event_read_status_t allocator_event_ring::read(
  std::uint64_t sequence, cudf_allocator_event_t& out) const noexcept
{
  initialize_event(out);
  if (sequence == 0) { return CUDF_ALLOCATOR_EVENT_READ_NOT_YET_RECORDED; }

  auto current_bounds = bounds();
  if (sequence >= current_bounds.next_sequence) {
    return CUDF_ALLOCATOR_EVENT_READ_NOT_YET_RECORDED;
  }
  if (sequence < current_bounds.first_available_sequence) {
    return CUDF_ALLOCATOR_EVENT_READ_OVERWRITTEN;
  }

  auto const& source = slots_[(sequence - 1) % k_capacity];
  if (source.published_sequence.load(std::memory_order_acquire) != sequence) {
    // Recheck against a fresh producer cursor so a concurrent wrap is reported
    // as overwrite rather than a permanently pending record.
    current_bounds = bounds();
    return sequence < current_bounds.first_available_sequence
             ? CUDF_ALLOCATOR_EVENT_READ_OVERWRITTEN
             : CUDF_ALLOCATOR_EVENT_READ_PENDING;
  }

  out.sequence          = sequence;
  out.monotonic_time_ns = source.monotonic_time_ns.load(std::memory_order_acquire);
  out.device_ordinal    = source.device_ordinal.load(std::memory_order_acquire);
  out.completeness      = source.completeness.load(std::memory_order_acquire);
  out.domain_kind       = source.domain_kind.load(std::memory_order_acquire);
  out.failure_source    = source.failure_source.load(std::memory_order_acquire);
  out.action            = source.action.load(std::memory_order_acquire);
  out.callback_attempt  = source.callback_attempt.load(std::memory_order_acquire);
  out.trim_retry_count  = source.trim_retry_count.load(std::memory_order_acquire);
  out.domain_id         = source.domain_id.load(std::memory_order_acquire);
  out.domain_generation = source.domain_generation.load(std::memory_order_acquire);
  out.domain_member_id  = source.domain_member_id.load(std::memory_order_acquire);
  out.requested_bytes   = source.requested_bytes.load(std::memory_order_acquire);
  out.pool_reserved_current_bytes =
    source.pool_reserved_current_bytes.load(std::memory_order_acquire);
  out.pool_reserved_high_bytes = source.pool_reserved_high_bytes.load(std::memory_order_acquire);
  out.pool_used_current_bytes  = source.pool_used_current_bytes.load(std::memory_order_acquire);
  out.pool_used_high_bytes     = source.pool_used_high_bytes.load(std::memory_order_acquire);

  if (source.published_sequence.load(std::memory_order_acquire) != sequence) {
    initialize_event(out);
    current_bounds = bounds();
    return sequence < current_bounds.first_available_sequence
             ? CUDF_ALLOCATOR_EVENT_READ_OVERWRITTEN
             : CUDF_ALLOCATOR_EVENT_READ_PENDING;
  }
  return CUDF_ALLOCATOR_EVENT_READ_AVAILABLE;
}

}  // namespace cudf::rust::memory
