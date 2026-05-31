#include "stream.h"
#include "shim_common.h"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <string>

extern "C" {

/// Create a new CUDA stream.
cudf_status_t cudf_stream_create(cudf_stream_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* stream = new rmm::cuda_stream();
    *out = static_cast<cudf_stream_handle>(stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Destroy a CUDA stream handle.
void cudf_stream_destroy(cudf_stream_handle handle) {
  delete static_cast<rmm::cuda_stream*>(handle);
}

/// Synchronize a CUDA stream (block until all operations complete).
cudf_status_t cudf_stream_synchronize(cudf_stream_handle handle) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* stream = static_cast<rmm::cuda_stream*>(handle);
    stream->synchronize();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Check if all operations on the stream have completed (non-blocking).
/// Sets *out to 1 if ready, 0 if not.
cudf_status_t cudf_stream_is_ready(cudf_stream_handle handle, int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* stream = static_cast<rmm::cuda_stream*>(handle);
    cudaError_t err = cudaStreamQuery(stream->view().value());
    if (err == cudaSuccess) {
      *out = 1;
    } else if (err == cudaErrorNotReady) {
      *out = 0;
    } else {
      auto msg = std::string("CUDA stream query failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
