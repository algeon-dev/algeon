#ifndef CUDF_RUST_ALLOCATOR_EVENT_H
#define CUDF_RUST_ALLOCATOR_EVENT_H

#include "memory_resource.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Number of fixed slots retained by every device-memory-manager allocator
/// event ring. The producer path never allocates or waits for a reader.
enum { CUDF_ALLOCATOR_EVENT_RING_CAPACITY = 256 };
enum { CUDF_ALLOCATOR_EVENT_RESERVED_SLOTS = 4 };

enum {
  /// A monotonic timestamp was captured while handling the allocation failure.
  CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_TIMESTAMP = 1u << 0,
  /// The event carries a complete domain identity, including its generation.
  CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_DOMAIN_IDENTITY = 1u << 1,
  /// The event carries one bounded CUDA-pool usage sample.
  CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_POOL_USAGE = 1u << 2,
  /// The event carries the allocator's terminal action.
  CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_ACTION = 1u << 3,
};

/// Result of reading one exact sequence from a device-manager event ring.
typedef enum {
  /// The requested sequence was copied into the supplied event record.
  CUDF_ALLOCATOR_EVENT_READ_AVAILABLE = 0,
  /// The sequence has no stable record at the instant of this read.
  CUDF_ALLOCATOR_EVENT_READ_PENDING = 1,
  /// The requested sequence has fallen behind the fixed ring capacity.
  CUDF_ALLOCATOR_EVENT_READ_OVERWRITTEN = 2,
  /// The requested sequence has not been reserved by a producer yet.
  CUDF_ALLOCATOR_EVENT_READ_NOT_YET_RECORDED = 3,
} cudf_allocator_event_read_status_t;

/// Bounds of one fixed allocator-event ring at a point in time.
typedef struct {
  size_t struct_size;
  uint64_t capacity;
  /// Lowest sequence that can still be present in the fixed ring. Zero means
  /// no sequence has been produced yet.
  uint64_t first_available_sequence;
  /// First sequence not yet reserved by a producer. All produced sequences are
  /// strictly smaller than this value.
  uint64_t next_sequence;
  uint64_t reserved[CUDF_ALLOCATOR_EVENT_RESERVED_SLOTS];
} cudf_allocator_event_ring_bounds_t;

/// One bounded, allocation-free allocator callback record.
///
/// `domain_id` and `domain_generation` are meaningful only with
/// `CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_DOMAIN_IDENTITY`. A missing pool
/// sample is represented by clearing its completeness bit, never by treating a
/// zero-valued sample as observed data.
typedef struct {
  size_t struct_size;
  uint64_t sequence;
  uint64_t monotonic_time_ns;
  int32_t device_ordinal;
  uint32_t completeness;
  uint32_t domain_kind;
  uint32_t failure_source;
  int32_t action;
  uint32_t callback_attempt;
  uint32_t trim_retry_count;
  uint64_t domain_id;
  uint64_t domain_generation;
  uint64_t domain_member_id;
  uint64_t requested_bytes;
  uint64_t pool_reserved_current_bytes;
  uint64_t pool_reserved_high_bytes;
  uint64_t pool_used_current_bytes;
  uint64_t pool_used_high_bytes;
  uint64_t reserved[CUDF_ALLOCATOR_EVENT_RESERVED_SLOTS];
} cudf_allocator_event_t;

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_ALLOCATOR_EVENT_H
