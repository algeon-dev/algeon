#ifndef CUDF_RUST_RESERVATION_H
#define CUDF_RUST_RESERVATION_H

#include "memory_manager.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to one stream-bound query-memory-domain member.
typedef struct cudf_reservation_handle_state* cudf_reservation_handle;

cudf_status_t cudf_memory_manager_route_current_resource_raw(cudf_memory_manager_handle manager,
                                                              void* raw_stream,
                                                              cudf_memory_resource_handle resource,
                                                              cudf_current_resource_route_handle* out);
cudf_status_t cudf_reservation_usage(cudf_reservation_handle reservation,
                                     cudf_memory_resource_usage_t* out);
cudf_status_t cudf_reservation_detach(cudf_reservation_handle reservation);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_RESERVATION_H
