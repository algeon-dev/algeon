#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace {
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

struct cudf_exec_mr_lifetime {
  std::shared_ptr<void> resource_owner_lifetime;
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> resource_ref_lifetime;
};

std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> make_exec_mr_ref(
  rmm::mr::device_memory_resource* raw)
{
  return std::make_shared<cuda::mr::any_resource<cuda::mr::device_accessible>>(
    rmm::mr::any_device_resource_from_raw(raw));
}

std::shared_ptr<void> make_exec_mr_lifetime(
  std::shared_ptr<void> resource_owner_lifetime,
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> resource_ref_lifetime)
{
  auto lifetime                     = std::make_shared<cudf_exec_mr_lifetime>();
  lifetime->resource_owner_lifetime = std::move(resource_owner_lifetime);
  lifetime->resource_ref_lifetime   = std::move(resource_ref_lifetime);
  return lifetime;
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

cudf_exec_context::cudf_exec_context(
  rmm::cuda_stream_view stream,
  std::shared_ptr<void> stream_lifetime,
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> mr,
  rmm::mr::device_memory_resource* mr_raw,
  std::shared_ptr<void> mr_lifetime,
  int32_t device,
  cudf_device_guard guard)
  : stream(stream),
    stream_lifetime(std::move(stream_lifetime)),
    mr(std::move(mr)),
    mr_raw(mr_raw),
    mr_lifetime(std::move(mr_lifetime)),
    device(device),
    guard(std::move(guard))
{
}

cudf_exec_context::cudf_exec_context(cudf_exec_context&& other) noexcept
  : stream(other.stream),
    stream_lifetime(std::move(other.stream_lifetime)),
    mr(std::move(other.mr)),
    mr_raw(other.mr_raw),
    mr_lifetime(std::move(other.mr_lifetime)),
    device(other.device),
    current_resource_route(std::exchange(other.current_resource_route, nullptr)),
    stream_provider_execution_scope(std::exchange(other.stream_provider_execution_scope, false)),
    guard(std::move(other.guard))
{
}

cudf_exec_context::~cudf_exec_context()
{
  if (stream_provider_execution_scope) { cudf_stream_provider_exit_execution(stream); }
  if (current_resource_route != nullptr) {
    cudf_current_resource_route_destroy_preserving_error(current_resource_route);
  }
}

cudf_exec_context cudf_make_exec_context()
{
  auto* mr_raw = cudf_tracked_current_device_resource_raw();
  auto mr_ref  = make_exec_mr_ref(mr_raw);
  return {cudf::get_default_stream(),
          {},
          mr_ref,
          mr_raw,
          make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_raw(mr_raw), mr_ref),
          0};
}

cudf_status_t cudf_resolve_exec_options(cudf_exec_options_t const* options, cudf_exec_context* out)
{
  if (out == nullptr) {
    set_error("out is null");
    return CUDF_ERROR_NULL;
  }

  // Step 1: Null-options fast path — resolve the current device's defaults and
  // return. This must come before any struct-field access since there is no
  // struct.
  if (options == nullptr) {
    int32_t dev                        = 0;
    cudf_status_t const current_status = get_current_device_or_error(&dev);
    if (current_status != CUDF_SUCCESS) { return current_status; }
    out->device          = dev;
    out->stream          = cudf::get_default_stream();
    out->stream_lifetime = {};
    out->mr_raw          = cudf_tracked_current_device_resource_raw();
    out->mr              = make_exec_mr_ref(out->mr_raw);
    out->mr_lifetime =
      make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_raw(out->mr_raw), out->mr);
    cudf::rust::test_support::record_exec_context(
      reinterpret_cast<std::uintptr_t>(out->stream.value()), out->mr_raw);
    return CUDF_SUCCESS;
  }

  // Step 2: Validate the options struct before reading any fields beyond
  // struct_size.  This prevents reading past a caller's valid prefix (ABI
  // versioning) and avoids side-effects (e.g. device switching) on invalid
  // data.
  if (options->struct_size < sizeof(cudf_exec_options_t)) {
    set_error("exec options struct_size is smaller than cudf_exec_options_t");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  // Only CUDF_EXEC_FLAG_DEVICE_EXPLICIT is defined; reject unknown bits.
  if ((options->flags & ~CUDF_EXEC_FLAG_DEVICE_EXPLICIT) != 0) {
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
  out->mr_raw = cudf_tracked_current_device_resource_raw();
  out->mr     = make_exec_mr_ref(out->mr_raw);
  out->mr_lifetime =
    make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_raw(out->mr_raw), out->mr);

  // Step 5: Apply optional stream/MR overrides from the caller.
  if (options->stream != nullptr) {
    out->stream          = cudf_stream_view_from_handle(options->stream);
    out->stream_lifetime = cudf_stream_lifetime_from_handle(options->stream);
  }
  if (options->mr != nullptr) {
    out->mr_raw = cudf_memory_resource_raw_from_handle(options->mr);
    out->mr     = make_exec_mr_ref(out->mr_raw);
    out->mr_lifetime =
      make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_handle(options->mr), out->mr);
  }

  cudf::rust::test_support::record_exec_context(
    reinterpret_cast<std::uintptr_t>(out->stream.value()), out->mr_raw);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_resolve_required_exec_options(cudf_exec_options_t const* options,
                                                 std::unique_ptr<cudf_exec_context>* out)
{
  if (out == nullptr) {
    set_error("out is null");
    return CUDF_ERROR_NULL;
  }
  if (options == nullptr) {
    set_error("exec options are required for this operation");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  if (options->struct_size < sizeof(cudf_exec_options_t)) {
    set_error("exec options struct_size is smaller than cudf_exec_options_t");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if ((options->flags & ~CUDF_EXEC_FLAG_DEVICE_EXPLICIT) != 0) {
    set_error("exec options flags contains unknown bits");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if ((options->flags & CUDF_EXEC_FLAG_DEVICE_EXPLICIT) == 0) {
    set_error("exec options must carry an explicit device");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (auto const reserved : options->reserved) {
    if (reserved != 0) {
      set_error("exec options reserved fields must be zero");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }
  if (options->device < 0) {
    set_error("exec options device must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  int32_t device_count           = 0;
  cudaError_t const count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess) {
    auto msg = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(count_status);
    set_error(msg);
    return CUDF_ERROR_CUDA;
  }
  if (options->device >= device_count) {
    set_error("exec options device is outside the available CUDA device range");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (options->stream == nullptr) {
    set_error("exec options stream is required for this operation");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (options->mr == nullptr) {
    set_error("exec options memory resource is required for this operation");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  cudf_device_guard guard;
  cudf_status_t const device_status = guard.activate(options->device);
  if (device_status != CUDF_SUCCESS) { return device_status; }

  auto stream          = cudf_stream_view_from_handle(options->stream);
  auto stream_lifetime = cudf_stream_lifetime_from_handle(options->stream);
  auto* mr_raw         = cudf_memory_resource_raw_from_handle(options->mr);
  auto mr              = make_exec_mr_ref(mr_raw);
  auto mr_lifetime =
    make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_handle(options->mr), mr);

  *out = std::make_unique<cudf_exec_context>(stream,
                                             std::move(stream_lifetime),
                                             std::move(mr),
                                             mr_raw,
                                             std::move(mr_lifetime),
                                             options->device,
                                             std::move(guard));

  cudf_memory_manager_handle manager = nullptr;
  cudf_status_t const manager_status =
    cudf_memory_manager_for_device(options->device, &manager);
  if (manager_status != CUDF_SUCCESS) {
    out->reset();
    return manager_status;
  }
  cudf_status_t const route_status = cudf_memory_manager_route_current_resource(
    manager, options->stream, options->mr, &(*out)->current_resource_route);
  if (route_status != CUDF_SUCCESS) {
    out->reset();
    return route_status;
  }
  try {
    cudf_stream_provider_enter_execution((*out)->stream);
    (*out)->stream_provider_execution_scope = true;
  } catch (...) {
    out->reset();
    return translate_exception();
  }

  cudf::rust::test_support::record_exec_context(
    reinterpret_cast<std::uintptr_t>((*out)->stream.value()), (*out)->mr_raw);
  return CUDF_SUCCESS;
}

void cudf_record_synchronized_stream(rmm::cuda_stream_view stream)
{
  cudf::rust::test_support::record_synchronized_stream(
    reinterpret_cast<std::uintptr_t>(stream.value()));
}
