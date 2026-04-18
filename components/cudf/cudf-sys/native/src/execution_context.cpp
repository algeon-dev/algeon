#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <string>

namespace {

thread_local std::uintptr_t g_last_exec_stream_value = std::numeric_limits<std::uintptr_t>::max();
thread_local std::uintptr_t g_last_exec_sync_stream_value =
  std::numeric_limits<std::uintptr_t>::max();
thread_local void* g_last_exec_mr_raw   = nullptr;
thread_local int32_t g_last_exec_device = -1;

void record_exec_context(rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr_raw)
{
  g_last_exec_stream_value = reinterpret_cast<std::uintptr_t>(stream.value());
  g_last_exec_mr_raw       = mr_raw;

  int32_t dev = -1;
  if (cudaGetDevice(&dev) == cudaSuccess) { g_last_exec_device = dev; }
}

cudf_status_t get_current_device_or_error(int32_t* out)
{
  cudaError_t err = cudaGetDevice(out);
  if (err != cudaSuccess) {
    auto msg = std::string("cudaGetDevice failed: ") + cudaGetErrorString(err);
    set_error(msg);
    return CUDF_ERROR_CUDA;
  }
  return CUDF_SUCCESS;
}

}  // namespace

cudf_status_t cudf_device_guard::activate(int32_t target)
{
  cudf_status_t const current_status = get_current_device_or_error(&previous_);
  if (current_status != CUDF_SUCCESS) { return current_status; }
  if (previous_ == target) {
    active_ = false;
    return CUDF_SUCCESS;
  }

  cudaError_t err = cudaSetDevice(target);
  if (err != cudaSuccess) {
    auto msg = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
    set_error(msg);
    return CUDF_ERROR_CUDA;
  }

  active_ = true;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_resolve_exec_options(cudf_exec_options_t const* options, cudf_exec_context* out)
{
  if (out == nullptr) {
    set_error("out is null");
    return CUDF_ERROR_NULL;
  }

  // Step 1: Null-options fast path — resolve the current device's defaults and return.
  // This must come before any struct-field access since there is no struct.
  if (options == nullptr) {
    int32_t dev                        = 0;
    cudf_status_t const current_status = get_current_device_or_error(&dev);
    if (current_status != CUDF_SUCCESS) { return current_status; }
    out->device = dev;
    out->stream = cudf::get_default_stream();
    out->mr     = cudf::get_current_device_resource_ref();
    out->mr_raw = rmm::mr::get_current_device_resource();
    record_exec_context(out->stream, out->mr_raw);
    return CUDF_SUCCESS;
  }

  // Step 2: Validate the options struct before reading any fields beyond
  // struct_size.  This prevents reading past a caller's valid prefix (ABI
  // versioning) and avoids side-effects (e.g. device switching) on invalid data.
  if (options->struct_size < sizeof(cudf_exec_options_t)) {
    set_error("exec options struct_size is smaller than cudf_exec_options_t");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  // Only CUDF_EXEC_FLAG_DEVICE_EXPLICIT is defined; reject unknown bits.
  if ((options->flags & ~static_cast<uint32_t>(CUDF_EXEC_FLAG_DEVICE_EXPLICIT)) != 0) {
    set_error("exec options flags contains unknown bits");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (auto const reserved : options->reserved) {
    if (reserved != 0) {
      set_error("exec options reserved fields must be zero");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }

  // Step 3: Resolve device.  If options carries CUDF_EXEC_FLAG_DEVICE_EXPLICIT,
  // activate that device via an RAII guard stored in `out` (restored when
  // `out` is destroyed at the caller's scope exit).  Otherwise use whatever
  // device is current on the calling thread.
  if ((options->flags & CUDF_EXEC_FLAG_DEVICE_EXPLICIT) != 0) {
    cudf_status_t const device_status = out->guard.activate(options->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    out->device = options->device;
  } else {
    int32_t dev                        = 0;
    cudf_status_t const current_status = get_current_device_or_error(&dev);
    if (current_status != CUDF_SUCCESS) { return current_status; }
    out->device = dev;
  }

  // Step 4: Resolve stream and MR defaults on the now-active device.
  out->stream = cudf::get_default_stream();
  out->mr     = cudf::get_current_device_resource_ref();
  out->mr_raw = rmm::mr::get_current_device_resource();

  // Step 5: Apply optional stream/MR overrides from the caller.
  if (options->stream != nullptr) { out->stream = cudf_stream_view_from_handle(options->stream); }
  if (options->mr != nullptr) {
    out->mr_raw = cudf_memory_resource_raw_from_handle(options->mr);
    out->mr     = cudf_memory_resource_ref_from_handle(options->mr);
  }

  record_exec_context(out->stream, out->mr_raw);
  return CUDF_SUCCESS;
}

void cudf_record_synchronized_stream(rmm::cuda_stream_view stream)
{
  g_last_exec_sync_stream_value = reinterpret_cast<std::uintptr_t>(stream.value());
}

extern "C" {

void cudf_test_exec_reset_observation()
{
  g_last_exec_stream_value      = std::numeric_limits<std::uintptr_t>::max();
  g_last_exec_sync_stream_value = std::numeric_limits<std::uintptr_t>::max();
  g_last_exec_mr_raw            = nullptr;
  g_last_exec_device            = -1;
}

cudf_status_t cudf_test_exec_last_stream_value(std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_sync_stream_value(std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_sync_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_mr_raw(void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_mr_raw;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_default_stream_value(std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = reinterpret_cast<std::uintptr_t>(cudf::get_default_stream().value());
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_stream_handle_value(cudf_stream_handle handle, std::uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out                       = 0;
  void* raw                  = nullptr;
  cudf_status_t const status = cudf_stream_raw(handle, &raw);
  if (status != CUDF_SUCCESS) { return status; }
  *out = reinterpret_cast<std::uintptr_t>(raw);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_current_device_resource_raw(void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = rmm::mr::get_current_device_resource();
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_device(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_device;
  return CUDF_SUCCESS;
}

}  // extern "C"
