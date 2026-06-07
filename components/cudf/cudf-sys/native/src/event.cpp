#include "event.h"
#include "shim_common.h"

#include <atomic>
#include <mutex>
#include <rmm/cuda_stream_view.hpp>
#include <unordered_map>

#include <cuda_runtime_api.h>

#include <string>

rmm::cuda_stream_view cudf_stream_view_from_handle(cudf_stream_handle handle);

namespace {

cudaEvent_t event_from_handle(cudf_event_handle handle)
{
  return *static_cast<cudaEvent_t*>(handle);
}

cudaEvent_t event_from_raw(void* raw_event)
{
  return static_cast<cudaEvent_t>(raw_event);
}

cudaStream_t stream_from_raw(void* raw_stream)
{
  return static_cast<cudaStream_t>(raw_stream);
}

cudf_status_t cuda_error_status(cudaError_t err, char const* operation)
{
  if (err == cudaSuccess) { return CUDF_SUCCESS; }
  auto msg = std::string(operation) + " failed: " + cudaGetErrorString(err);
  set_error(msg);
  return CUDF_ERROR_CUDA;
}

std::atomic<int32_t> g_stream_wait_call_count{0};
std::atomic<std::uintptr_t> g_last_wait_stream_raw{0};
std::atomic<std::uintptr_t> g_last_wait_event_raw{0};
std::atomic<int32_t> g_raw_record_call_count{0};
std::atomic<std::uintptr_t> g_last_raw_record_stream_raw{0};
std::atomic<std::uintptr_t> g_last_raw_record_event_raw{0};
std::atomic<int32_t> g_event_synchronize_call_count{0};
std::mutex g_observation_mutex;
std::unordered_map<std::uintptr_t, int32_t> g_stream_wait_counts_by_stream_raw;
std::unordered_map<std::uintptr_t, int32_t> g_stream_wait_counts_by_event_raw;
std::unordered_map<std::uintptr_t, int32_t> g_raw_record_counts_by_stream_raw;

}  // namespace

extern "C" {

cudf_status_t cudf_event_create(cudf_event_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* event = new cudaEvent_t{};
    cudaError_t err = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      delete event;
      return cuda_error_status(err, "cudaEventCreateWithFlags");
    }
    *out = static_cast<cudf_event_handle>(event);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_event_destroy(cudf_event_handle handle)
{
  if (handle == nullptr) { return; }
  auto* event = static_cast<cudaEvent_t*>(handle);
  cudaEventDestroy(*event);
  delete event;
}

cudf_status_t cudf_event_record(cudf_event_handle event, cudf_stream_handle stream)
{
  CUDF_RUST_NULL_CHECK(event, "event is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_TRY {
    auto err = cudaEventRecord(event_from_handle(event), cudf_stream_view_from_handle(stream).value());
    return cuda_error_status(err, "cudaEventRecord");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_event_synchronize(cudf_event_handle event)
{
  CUDF_RUST_NULL_CHECK(event, "event is null");
  CUDF_RUST_TRY {
    g_event_synchronize_call_count.fetch_add(1, std::memory_order_relaxed);
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
  CUDF_RUST_TRY {
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
  CUDF_RUST_TRY {
    auto const stream_raw = reinterpret_cast<std::uintptr_t>(cudf_stream_view_from_handle(stream).value());
    g_stream_wait_call_count.fetch_add(1, std::memory_order_relaxed);
    g_last_wait_stream_raw.store(stream_raw, std::memory_order_relaxed);
    auto const event_raw = reinterpret_cast<std::uintptr_t>(event_from_handle(event));
    g_last_wait_event_raw.store(event_raw, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(g_observation_mutex);
      g_stream_wait_counts_by_stream_raw[stream_raw] += 1;
      g_stream_wait_counts_by_event_raw[event_raw] += 1;
    }
    auto err = cudaStreamWaitEvent(cudf_stream_view_from_handle(stream).value(), event_from_handle(event), 0);
    return cuda_error_status(err, "cudaStreamWaitEvent");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_raw_cuda_event_record(void* raw_event, void* raw_stream)
{
  CUDF_RUST_NULL_CHECK(raw_event, "raw_event is null");
  CUDF_RUST_NULL_CHECK(raw_stream, "raw_stream is null");
  CUDF_RUST_TRY {
    auto const stream_raw = reinterpret_cast<std::uintptr_t>(raw_stream);
    g_raw_record_call_count.fetch_add(1, std::memory_order_relaxed);
    g_last_raw_record_event_raw.store(reinterpret_cast<std::uintptr_t>(raw_event), std::memory_order_relaxed);
    g_last_raw_record_stream_raw.store(stream_raw, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(g_observation_mutex);
      g_raw_record_counts_by_stream_raw[stream_raw] += 1;
    }
    auto err = cudaEventRecord(event_from_raw(raw_event), stream_from_raw(raw_stream));
    return cuda_error_status(err, "cudaEventRecord");
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_raw_cuda_stream_wait_event(void* raw_stream, void* raw_event)
{
  CUDF_RUST_NULL_CHECK(raw_stream, "raw_stream is null");
  CUDF_RUST_NULL_CHECK(raw_event, "raw_event is null");
  CUDF_RUST_TRY {
    auto const stream_raw = reinterpret_cast<std::uintptr_t>(raw_stream);
    g_stream_wait_call_count.fetch_add(1, std::memory_order_relaxed);
    g_last_wait_stream_raw.store(stream_raw, std::memory_order_relaxed);
    auto const event_raw = reinterpret_cast<std::uintptr_t>(raw_event);
    g_last_wait_event_raw.store(event_raw, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(g_observation_mutex);
      g_stream_wait_counts_by_stream_raw[stream_raw] += 1;
      g_stream_wait_counts_by_event_raw[event_raw] += 1;
    }
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
  CUDF_RUST_TRY {
    *out = reinterpret_cast<void*>(event_from_handle(handle));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_test_event_reset_observation()
{
  g_stream_wait_call_count.store(0, std::memory_order_relaxed);
  g_last_wait_stream_raw.store(0, std::memory_order_relaxed);
  g_last_wait_event_raw.store(0, std::memory_order_relaxed);
  g_raw_record_call_count.store(0, std::memory_order_relaxed);
  g_last_raw_record_stream_raw.store(0, std::memory_order_relaxed);
  g_last_raw_record_event_raw.store(0, std::memory_order_relaxed);
  g_event_synchronize_call_count.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_observation_mutex);
  g_stream_wait_counts_by_stream_raw.clear();
  g_stream_wait_counts_by_event_raw.clear();
  g_raw_record_counts_by_stream_raw.clear();
}

cudf_status_t cudf_test_event_synchronize_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_event_synchronize_call_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_stream_wait_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_stream_wait_call_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_stream_wait_call_count_for_stream_raw(std::uintptr_t stream_raw, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::lock_guard<std::mutex> lock(g_observation_mutex);
  auto const it = g_stream_wait_counts_by_stream_raw.find(stream_raw);
  *out          = (it == g_stream_wait_counts_by_stream_raw.end()) ? 0 : it->second;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_stream_wait_call_count_for_event_raw(std::uintptr_t event_raw, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::lock_guard<std::mutex> lock(g_observation_mutex);
  auto const it = g_stream_wait_counts_by_event_raw.find(event_raw);
  *out          = (it == g_stream_wait_counts_by_event_raw.end()) ? 0 : it->second;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_wait_stream_raw(std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_wait_stream_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_wait_event_raw(std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_wait_event_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_raw_record_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_raw_record_call_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_raw_record_call_count_for_stream_raw(std::uintptr_t stream_raw, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::lock_guard<std::mutex> lock(g_observation_mutex);
  auto const it = g_raw_record_counts_by_stream_raw.find(stream_raw);
  *out          = (it == g_raw_record_counts_by_stream_raw.end()) ? 0 : it->second;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_raw_record_stream_raw(std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_raw_record_stream_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_raw_record_event_raw(std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_raw_record_event_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

}  // extern "C"
