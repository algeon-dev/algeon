#include "stream.h"

#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

namespace {
struct cudf_stream_owner {
  rmm::cuda_stream stream;
  int32_t device = -1;
};

struct cudf_stream_handle_state {
  std::shared_ptr<cudf_stream_owner> owner;
};

cudf_stream_handle_state &
cudf_stream_state_from_handle(cudf_stream_handle handle) {
  return *static_cast<cudf_stream_handle_state *>(handle);
}
} // namespace

rmm::cuda_stream_view cudf_stream_view_from_handle(cudf_stream_handle handle) {
  return cudf_stream_state_from_handle(handle).owner->stream.view();
}

int32_t cudf_stream_device_from_handle(cudf_stream_handle handle) {
  return cudf_stream_state_from_handle(handle).owner->device;
}

std::shared_ptr<void>
cudf_stream_lifetime_from_handle(cudf_stream_handle handle) {
  return cudf_stream_state_from_handle(handle).owner;
}

extern "C" {
/// Create a new CUDA stream.
cudf_status_t cudf_stream_create(cudf_stream_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto state = std::make_unique<cudf_stream_handle_state>();
    state->owner = std::make_shared<cudf_stream_owner>();
    int32_t device = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    state->owner->device = device;
    *out = static_cast<cudf_stream_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Destroy a CUDA stream handle.
void cudf_stream_destroy(cudf_stream_handle handle) {
  delete static_cast<cudf_stream_handle_state *>(handle);
}

/// Synchronize a CUDA stream (block until all operations complete).
cudf_status_t cudf_stream_synchronize(cudf_stream_handle handle) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto stream = cudf_stream_view_from_handle(handle);
    stream.synchronize();
    cudf_record_synchronized_stream(stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Synchronize a raw borrowed CUDA stream pointer.
cudf_status_t cudf_raw_cuda_stream_synchronize(void *raw_stream) {
  CUDF_RUST_NULL_CHECK(raw_stream, "raw_stream is null");
  CUDF_RUST_TRY {
    auto stream =
        rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(raw_stream)};
    stream.synchronize();
    cudf_record_synchronized_stream(stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Check if all operations on the stream have completed (non-blocking).
/// Sets *out to 1 if ready, 0 if not.
cudf_status_t cudf_stream_is_ready(cudf_stream_handle handle, int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto stream = cudf_stream_view_from_handle(handle);
    cudaError_t err = cudaStreamQuery(stream.value());
    if (err == cudaSuccess) {
      *out = 1;

    } else if (err == cudaErrorNotReady) {
      *out = 0;

    } else {
      auto msg =
          std::string("CUDA stream query failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stream_raw(cudf_stream_handle handle, void **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    *out =
        reinterpret_cast<void *>(cudf_stream_view_from_handle(handle).value());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // extern "C"
