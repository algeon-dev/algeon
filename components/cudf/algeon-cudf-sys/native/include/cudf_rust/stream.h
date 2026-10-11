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
/// stream identity without a `cudf_stream_handle`.
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

/// Register the parent and bounded fork stream set for one attempt.
///
/// The last element of `fork_streams` is the parent's reserved fork: the
/// parent stream is served only from it, and it is never handed to an alias.
/// The preceding `fork_count - 1` entries are the auxiliary forks, one per
/// attached alias. `fork_count` must therefore be at least two.
///
/// The caller retains ownership of every raw stream. The registration remains
/// active until `cudf_stream_provider_unregister_attempt` returns, and the
/// caller must keep the streams alive through that point and through any
/// worker/stream drain that follows it.
cudf_status_t cudf_stream_provider_register_attempt(void* parent_stream,
                                                    void* const* fork_streams,
                                                    size_t fork_count);

/// Remove one attempt route after its streams have drained.
cudf_status_t cudf_stream_provider_unregister_attempt(void* parent_stream);

/// Attach another live attempt-member stream as a fork parent alias holding
/// one auxiliary fork of the route for its lifetime.
///
/// The route claims the lowest auxiliary fork index no live alias holds and
/// confines every fork requested from `alias_stream` (and from the fork it
/// is served) to it; the chosen index is written to
/// `out_claimed_fork_index` when that pointer is non-null. The route chooses
/// the index, atomically with the attach, because its live-claim table is the
/// only record of which forks are taken. The claim is a property of the alias,
/// not of the calling thread, so forks requested from the alias on any thread
/// stay inside it, and detaching the alias releases it.
///
/// The parent's reserved fork is never claimable, so an alias can never be
/// served the stream the root runs on.
///
/// The attach fails with `CUDF_ERROR_INVALID_OPERATION` when every auxiliary
/// fork is already claimed. A caller holding a freshly obtained attempt
/// auxiliary permit never sees that: live claims never outnumber the
/// attempt's permit holders, which admission caps at the auxiliary fork
/// count.
cudf_status_t cudf_stream_provider_attach_attempt_stream(void* parent_stream,
                                                         void* alias_stream,
                                                         size_t* out_claimed_fork_index);

/// Remove a previously attached attempt-member stream alias after it drains.
cudf_status_t cudf_stream_provider_detach_attempt_stream(void* alias_stream);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_H
