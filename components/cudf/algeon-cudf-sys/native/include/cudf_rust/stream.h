#ifndef CUDF_RUST_STREAM_H
#define CUDF_RUST_STREAM_H

#include "error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an owning CUDA stream.
typedef struct cudf_stream_handle_state* cudf_stream_handle;

/// Create a new CUDA stream.
///
/// Use this handle directly with `cudf_stream_synchronize` /
/// `cudf_stream_is_ready`, or attach it to a `cudf_exec_options_t` to direct
/// `_ex` APIs onto an explicit stream.
cudf_status_t cudf_stream_create(cudf_stream_handle* out);

/// Destroy a CUDA stream handle. Ownership is consumed on call, including on
/// failure; the handle must never be reused.
cudf_status_t cudf_stream_destroy(cudf_stream_handle handle);

/// Synchronize work already enqueued on a CUDA stream handle.
cudf_status_t cudf_stream_synchronize(cudf_stream_handle handle);

/// Synchronize work already enqueued on a raw borrowed CUDA stream pointer.
///
/// `raw_stream` must be a valid borrowed `cudaStream_t` value. This does not
/// take ownership and is intended for compatibility layers that carry raw CUDA
/// stream identity rather than a `cudf_stream_handle`.
cudf_status_t cudf_raw_cuda_stream_synchronize(void* raw_stream);

/// Check whether work already enqueued on a CUDA stream handle has completed
/// (1 ready, 0 not ready).
cudf_status_t cudf_stream_is_ready(cudf_stream_handle handle, int32_t* out);

/// Borrow the underlying raw CUDA stream pointer from an owning stream handle.
///
/// This does not transfer ownership. `*out` is valid only while `handle`
/// remains alive.
cudf_status_t cudf_stream_raw(cudf_stream_handle handle, void** out);

/// Install Algeon's attempt-owned fork-stream provider.
///
/// Installation is process-local and idempotent. Before installation,
/// libcudf retains its upstream global-pool behavior. After installation,
/// every parent passed to `fork_streams` must have an active attempt route.
cudf_status_t cudf_stream_provider_install(void);

/// Register the parent and bounded auxiliary stream set for one attempt.
///
/// The caller retains ownership of every raw stream. The registration remains
/// active until `cudf_stream_provider_unregister_attempt` returns, and the
/// caller must keep the streams alive through that point and through any
/// worker/stream drain that follows it.
cudf_status_t cudf_stream_provider_register_attempt(void* parent_stream,
                                                    void* const* auxiliary_streams,
                                                    size_t auxiliary_count);

/// Remove one attempt route after its streams have drained.
cudf_status_t cudf_stream_provider_unregister_attempt(void* parent_stream);

/// Attach another live attempt-member stream as a fork parent alias, with an
/// optional fork subset of its own.
///
/// When `fork_end > fork_begin`, forks requested from `alias_stream` are
/// served round-robin from `[fork_begin, fork_end)` of the attempt's
/// registered fork pool; the range must be non-empty and inside that pool.
/// Passing `fork_begin == fork_end == 0` attaches the alias with whole-pool
/// behavior. The subset is a property of the alias, not of the calling
/// thread, so forks requested from the alias on any thread stay inside it.
/// Detaching the alias removes its subset.
cudf_status_t cudf_stream_provider_attach_attempt_stream(void* parent_stream,
                                                         void* alias_stream,
                                                         size_t fork_begin,
                                                         size_t fork_end);

/// Remove a previously attached attempt-member stream alias after it drains.
cudf_status_t cudf_stream_provider_detach_attempt_stream(void* alias_stream);
#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_H
