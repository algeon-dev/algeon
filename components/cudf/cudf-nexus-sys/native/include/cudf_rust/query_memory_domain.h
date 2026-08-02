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

cudf_status_t cudf_memory_manager_create_query_memory_domain(cudf_memory_manager_handle manager,
                                                             uint64_t domain_id,
                                                             size_t grant_bytes,
                                                             size_t ceiling_bytes,
                                                             uint64_t release_threshold_bytes,
                                                             cudf_query_memory_domain_handle* out);
cudf_status_t cudf_query_memory_domain_attach_stream(cudf_query_memory_domain_handle domain,
                                                     cudf_stream_handle stream,
                                                     uint64_t reservation_id,
                                                     size_t limit_bytes,
                                                     cudf_reservation_handle* out,
                                                     cudf_memory_resource_handle* out_resource);
int32_t cudf_query_memory_domain_driver_ceiling_supported(void);
cudf_status_t cudf_query_memory_domain_usage(cudf_query_memory_domain_handle domain,
                                             cudf_memory_resource_usage_t* out);
cudf_status_t cudf_query_memory_domain_identity(cudf_query_memory_domain_handle domain,
                                                cudf_query_memory_domain_identity_t* out);
cudf_status_t cudf_query_memory_domain_pool_usage(cudf_query_memory_domain_handle domain,
                                                  cudf_memory_pool_usage_t* out);
cudf_status_t cudf_query_memory_domain_resize_grant(cudf_query_memory_domain_handle domain,
                                                    size_t grant_bytes);
cudf_status_t cudf_query_memory_domain_trim(cudf_query_memory_domain_handle domain,
                                            uint64_t keep_bytes);
cudf_status_t cudf_query_memory_domain_destroy(cudf_query_memory_domain_handle domain);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_QUERY_MEMORY_DOMAIN_H
