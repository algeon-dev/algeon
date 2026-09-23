#ifndef CUDF_RUST_RESERVATION_H
#define CUDF_RUST_RESERVATION_H

#include "memory_manager.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to one stream-bound memory reservation.
typedef struct cudf_reservation_handle_state* cudf_reservation_handle;
/// Opaque handle to one query-scoped reservation arena.
typedef struct cudf_reservation_arena_handle_state* cudf_reservation_arena_handle;

cudf_status_t cudf_memory_manager_route_current_resource_raw(cudf_memory_manager_handle manager,
                                                              void* raw_stream,
                                                              cudf_memory_resource_handle resource,
                                                              cudf_current_resource_route_handle* out);
cudf_status_t cudf_memory_manager_attach_reservation(cudf_memory_manager_handle manager,
                                                     cudf_stream_handle stream,
                                                     uint64_t reservation_id,
                                                     size_t limit_bytes,
                                                     cudf_reservation_handle* out);
cudf_status_t cudf_memory_manager_create_reservation_arena(cudf_memory_manager_handle manager,
                                                           uint64_t arena_id,
                                                           size_t limit_bytes,
                                                           cudf_reservation_arena_handle* out);
cudf_status_t cudf_memory_manager_attach_reservation_to_arena(cudf_memory_manager_handle manager,
                                                              cudf_stream_handle stream,
                                                              cudf_reservation_arena_handle arena,
                                                              uint64_t reservation_id,
                                                              size_t limit_bytes,
                                                              cudf_reservation_handle* out);
cudf_status_t cudf_memory_manager_attach_reservation_to_arena_raw(cudf_memory_manager_handle manager,
                                                                  void* raw_stream,
                                                                  cudf_reservation_arena_handle arena,
                                                                  uint64_t reservation_id,
                                                                  size_t limit_bytes,
                                                                  cudf_reservation_handle* out);
cudf_status_t cudf_reservation_usage(cudf_reservation_handle reservation,
                                     cudf_memory_resource_usage_t* out);
cudf_status_t cudf_reservation_resize(cudf_reservation_handle reservation, size_t limit_bytes);
cudf_status_t cudf_reservation_detach(cudf_reservation_handle reservation);
cudf_status_t cudf_reservation_arena_usage(cudf_reservation_arena_handle arena,
                                           cudf_memory_resource_usage_t* out);
cudf_status_t cudf_reservation_arena_resize(cudf_reservation_arena_handle arena,
                                            size_t limit_bytes);
cudf_status_t cudf_reservation_arena_destroy(cudf_reservation_arena_handle arena);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_RESERVATION_H
