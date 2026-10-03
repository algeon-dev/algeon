#ifndef CUDF_RUST_QUERY_MEMORY_DOMAIN_H
#define CUDF_RUST_QUERY_MEMORY_DOMAIN_H

#include "reservation.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to one query-scoped CUDA memory domain.
typedef struct cudf_query_memory_domain_handle_state* cudf_query_memory_domain_handle;

/// Stable native identity for one query-memory-domain instance. `generation`
/// changes whenever a new domain is created, even if a caller reuses the same
/// logical domain id after teardown.
typedef struct {
  size_t struct_size;
  int32_t device_ordinal;
  uint32_t reserved_padding;
  uint64_t domain_id;
  uint64_t generation;
  uint64_t reserved[4];
} cudf_query_memory_domain_identity_t;

/// Create a domain with an immutable grant and pool ceiling.
cudf_status_t cudf_memory_manager_create_query_memory_domain(cudf_memory_manager_handle manager,
                                                             uint64_t domain_id,
                                                             size_t grant_bytes,
                                                             size_t ceiling_bytes,
                                                             cudf_query_memory_domain_handle* out);
/// Attach a member whose allocation limit remains fixed until detach.
cudf_status_t cudf_query_memory_domain_attach_stream(cudf_query_memory_domain_handle domain,
                                                     cudf_stream_handle stream,
                                                     uint64_t reservation_id,
                                                     size_t limit_bytes,
                                                     cudf_reservation_handle* out,
                                                     cudf_memory_resource_handle* out_resource);
cudf_status_t cudf_query_memory_domain_attach_raw_stream(cudf_query_memory_domain_handle domain,
                                                         void* raw_stream,
                                                         uint64_t reservation_id,
                                                         size_t limit_bytes,
                                                         cudf_reservation_handle* out,
                                                         cudf_memory_resource_handle* out_resource);
cudf_status_t cudf_query_memory_domain_usage(cudf_query_memory_domain_handle domain,
                                             cudf_memory_resource_usage_t* out);
cudf_status_t cudf_query_memory_domain_identity(cudf_query_memory_domain_handle domain,
                                                cudf_query_memory_domain_identity_t* out);
cudf_status_t cudf_query_memory_domain_pool_usage(cudf_query_memory_domain_handle domain,
                                                  cudf_memory_pool_usage_t* out);
cudf_status_t cudf_query_memory_domain_freeze(cudf_query_memory_domain_handle domain);

/// One surviving allocation owned by a query memory domain.
///
/// `device_pointer` is the base pointer the allocator registered, not an
/// interior column pointer. Owner attestation compares published output
/// buffers against these base allocations.
typedef struct {
  size_t struct_size;
  void* device_pointer;
  size_t bytes;
  size_t aligned_bytes;
  size_t alignment;
  uint64_t reserved[2];
} cudf_query_memory_domain_allocation_t;

/// Copies this domain's live allocation registry into `out`.
///
/// Writes at most `capacity` records and always reports the full live count
/// in `out_count`, so a caller can detect truncation. Passing a null `out`
/// with zero `capacity` queries the count alone.
cudf_status_t cudf_query_memory_domain_live_allocations(
  cudf_query_memory_domain_handle domain,
  cudf_query_memory_domain_allocation_t* out,
  size_t capacity,
  size_t* out_count);
cudf_status_t cudf_query_memory_domain_destroy(cudf_query_memory_domain_handle domain);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_QUERY_MEMORY_DOMAIN_H
