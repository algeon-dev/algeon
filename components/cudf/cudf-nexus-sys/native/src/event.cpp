#include "cudf_rust/event.h"

#include "shim_common.h"
#include "test_support_internal.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <string>

rmm::cuda_stream_view cudf_stream_view_from_handle(cudf_stream_handle handle);

namespace {
cudaEvent_t event_from_handle(cudf_event_handle handle)
{
  return *reinterpret_cast<cudaEvent_t*>(handle);
}

cudaEvent_t event_from_raw(void* raw_event) { return reinterpret_cast<cudaEvent_t>(raw_event); }

cudaStream_t stream_from_raw(void* raw_stream) { return static_cast<cudaStream_t>(raw_stream); }

cudf_status_t cuda_error_status(cudaError_t err, char const* operation)
{
  if (err == cudaSuccess) { return CUDF_SUCCESS; }
  auto msg = std::string(operation) + " failed: " + cudaGetErrorString(err);
  set_error(msg);
  return CUDF_ERROR_CUDA;
}

}  // namespace

extern "C" {
cudf_status_t cudf_event_create(cudf_event_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto* event     = new cudaEvent_t{};
    cudaError_t err = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      delete event;
      return cuda_error_status(err, "cudaEventCreateWithFlags");
    }
    *out = reinterpret_cast<cudf_event_handle>(event);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_event_destroy(cudf_event_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* event       = reinterpret_cast<cudaEvent_t*>(handle);
  auto const status = cuda_error_status(cudaEventDestroy(*event), "cudaEventDestroy");
  // The host-side owner is consumed even if CUDA rejects device cleanup.
  delete event;
  return status;
}

cudf_status_t cudf_event_record(cudf_event_handle event, cudf_stream_handle stream)
{
  CUDF_RUST_NULL_CHECK(event, "event is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_TRY
  {
    auto err =
      cudaEventRecord(event_from_handle(event), cudf_stream_view_from_handle(stream).value());
    return cuda_error_status(err, "cudaEventRecord");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_event_synchronize(cudf_event_handle event)
{
  CUDF_RUST_NULL_CHECK(event, "event is null");
  CUDF_RUST_TRY
  {
    auto const event_handle_raw = reinterpret_cast<std::uintptr_t>(event);
    cudf::rust::test_support::record_event_synchronize(event_handle_raw);
    if (cudf::rust::test_support::event_synchronize_should_fail()) {
      set_error("forced cudaEventSynchronize failure for tests");
      return CUDF_ERROR_CUDA;
    }
    cudf::rust::test_support::wait_for_event_synchronize_release();
    auto err = cudaEventSynchronize(event_from_handle(event));
    return cuda_error_status(err, "cudaEventSynchronize");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_event_is_ready(cudf_event_handle event, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(event, "event is null");
  CUDF_RUST_TRY
  {
    auto err = cudaEventQuery(event_from_handle(event));
    if (err == cudaSuccess) {
      *out = 1;
      return CUDF_SUCCESS;
    }
    if (err == cudaErrorNotReady) {
      *out = 0;
      return CUDF_SUCCESS;
    }
    return cuda_error_status(err, "cudaEventQuery");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stream_wait_event(cudf_stream_handle stream, cudf_event_handle event)
{
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_NULL_CHECK(event, "event is null");
  CUDF_RUST_TRY
  {
    auto const stream_raw =
      reinterpret_cast<std::uintptr_t>(cudf_stream_view_from_handle(stream).value());
    auto const event_raw = reinterpret_cast<std::uintptr_t>(event_from_handle(event));
    cudf::rust::test_support::record_stream_wait(stream_raw, event_raw);
    auto err = cudaStreamWaitEvent(
      cudf_stream_view_from_handle(stream).value(), event_from_handle(event), 0);
    return cuda_error_status(err, "cudaStreamWaitEvent");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_raw_cuda_event_record(void* raw_event, void* raw_stream)
{
  CUDF_RUST_NULL_CHECK(raw_event, "raw_event is null");
  CUDF_RUST_NULL_CHECK(raw_stream, "raw_stream is null");
  CUDF_RUST_TRY
  {
    auto const stream_raw = reinterpret_cast<std::uintptr_t>(raw_stream);
    cudf::rust::test_support::record_raw_event_record(stream_raw,
                                                      reinterpret_cast<std::uintptr_t>(raw_event));
    auto err = cudaEventRecord(event_from_raw(raw_event), stream_from_raw(raw_stream));
    return cuda_error_status(err, "cudaEventRecord");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_raw_cuda_stream_wait_event(void* raw_stream, void* raw_event)
{
  CUDF_RUST_NULL_CHECK(raw_stream, "raw_stream is null");
  CUDF_RUST_NULL_CHECK(raw_event, "raw_event is null");
  CUDF_RUST_TRY
  {
    auto const stream_raw = reinterpret_cast<std::uintptr_t>(raw_stream);
    auto const event_raw  = reinterpret_cast<std::uintptr_t>(raw_event);
    cudf::rust::test_support::record_stream_wait(stream_raw, event_raw);
    auto err = cudaStreamWaitEvent(stream_from_raw(raw_stream), event_from_raw(raw_event), 0);
    return cuda_error_status(err, "cudaStreamWaitEvent");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_event_raw(cudf_event_handle handle, void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    *out = reinterpret_cast<void*>(event_from_handle(handle));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
