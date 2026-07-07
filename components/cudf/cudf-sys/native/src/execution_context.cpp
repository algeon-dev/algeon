#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace {
constexpr std::size_t k_exec_mr_observation_capacity = 64;

thread_local std::uintptr_t g_last_exec_stream_value =
    std::numeric_limits<std::uintptr_t>::max();
thread_local std::uintptr_t g_last_exec_sync_stream_value =
    std::numeric_limits<std::uintptr_t>::max();
thread_local void *g_last_exec_mr_raw = nullptr;
thread_local int32_t g_last_exec_device = -1;
thread_local std::array<void *, k_exec_mr_observation_capacity>
    g_exec_mr_observations{};
thread_local std::size_t g_exec_mr_observation_start = 0;
thread_local std::size_t g_exec_mr_observation_count = 0;
thread_local bool g_exec_mr_observation_overflowed = false;

void record_exec_context(rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource *mr_raw) {
  g_last_exec_stream_value = reinterpret_cast<std::uintptr_t>(stream.value());
  g_last_exec_mr_raw = mr_raw;
  if (g_exec_mr_observation_count < g_exec_mr_observations.size()) {
    auto const physical =
        (g_exec_mr_observation_start + g_exec_mr_observation_count) %
        g_exec_mr_observations.size();
    g_exec_mr_observations[physical] = mr_raw;
    ++g_exec_mr_observation_count;
  } else {
    g_exec_mr_observations[g_exec_mr_observation_start] = mr_raw;
    g_exec_mr_observation_start =
        (g_exec_mr_observation_start + 1) % g_exec_mr_observations.size();
    g_exec_mr_observation_overflowed = true;
  }

  int32_t dev = -1;
  if (cudaGetDevice(&dev) == cudaSuccess) {
    g_last_exec_device = dev;
  }
}

void *exec_mr_observation_at(std::size_t index) {
  auto const physical =
      (g_exec_mr_observation_start + index) % g_exec_mr_observations.size();
  return g_exec_mr_observations[physical];
}

cudf_status_t get_current_device_or_error(int32_t *out) {
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
  std::shared_ptr<rmm::device_async_resource_ref> resource_ref_lifetime;
};

std::shared_ptr<rmm::device_async_resource_ref>
make_exec_mr_ref(rmm::mr::device_memory_resource *raw) {
  return std::make_shared<rmm::device_async_resource_ref>(raw);
}

std::shared_ptr<void> make_exec_mr_lifetime(
    std::shared_ptr<void> resource_owner_lifetime,
    std::shared_ptr<rmm::device_async_resource_ref> resource_ref_lifetime) {
  auto lifetime = std::make_shared<cudf_exec_mr_lifetime>();
  lifetime->resource_owner_lifetime = std::move(resource_owner_lifetime);
  lifetime->resource_ref_lifetime = std::move(resource_ref_lifetime);
  return lifetime;
}
} // namespace

std::recursive_mutex &cudf_current_device_resource_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

cudf_status_t cudf_device_guard::activate(int32_t target) {
  cudf_status_t const current_status = get_current_device_or_error(&previous_);
  if (current_status != CUDF_SUCCESS) {
    return current_status;
  }
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

cudf_current_device_resource_guard::cudf_current_device_resource_guard(
    rmm::mr::device_memory_resource *target)
    : lock_(cudf_current_device_resource_mutex()),
      previous_(rmm::mr::get_current_device_resource()) {
  if (previous_ != target) {
    rmm::mr::set_current_device_resource(target);
    restore_previous_ = true;
  }
}

cudf_current_device_resource_guard::~cudf_current_device_resource_guard() {
  if (restore_previous_) {
    rmm::mr::set_current_device_resource(previous_);
  }
}

cudf_exec_context::cudf_exec_context(rmm::cuda_stream_view stream,
                                     std::shared_ptr<void> stream_lifetime,
                                     std::shared_ptr<rmm::device_async_resource_ref> mr,
                                     rmm::mr::device_memory_resource *mr_raw,
                                     std::shared_ptr<void> mr_lifetime,
                                     int32_t device, cudf_device_guard guard)
    : stream(stream), stream_lifetime(std::move(stream_lifetime)),
      mr(std::move(mr)), mr_raw(mr_raw), mr_lifetime(std::move(mr_lifetime)),
      device(device), guard(std::move(guard)) {}

cudf_exec_context::cudf_exec_context(cudf_exec_context &&other) noexcept
    : stream(other.stream), stream_lifetime(std::move(other.stream_lifetime)),
      mr(other.mr), mr_raw(other.mr_raw),
      mr_lifetime(std::move(other.mr_lifetime)), device(other.device),
      guard(std::move(other.guard)),
      current_resource_lock(std::move(other.current_resource_lock)),
      previous_mr_raw(other.previous_mr_raw),
      current_resource_installed(other.current_resource_installed) {
  other.previous_mr_raw = nullptr;
  other.current_resource_installed = false;
}

cudf_exec_context::~cudf_exec_context() {
  if (current_resource_installed) {
    rmm::mr::set_current_device_resource(previous_mr_raw);
  }
}

void cudf_exec_context::install_current_device_resource() {
  if (current_resource_installed) {
    return;
  }
  current_resource_lock = std::unique_lock<std::recursive_mutex>{
      cudf_current_device_resource_mutex()};
  previous_mr_raw = rmm::mr::get_current_device_resource();
  if (previous_mr_raw != mr_raw) {
    rmm::mr::set_current_device_resource(mr_raw);
  }
  current_resource_installed = true;
}

cudf_exec_context cudf_make_exec_context() {
  std::scoped_lock lock{cudf_current_device_resource_mutex()};
  auto *mr_raw = rmm::mr::get_current_device_resource();
  auto mr_ref = make_exec_mr_ref(mr_raw);
  return {cudf::get_default_stream(),
          {},
          mr_ref,
          mr_raw,
          make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_raw(mr_raw), mr_ref),
          0};
}

cudf_status_t cudf_resolve_exec_options(cudf_exec_options_t const *options,
                                        cudf_exec_context *out) {
  if (out == nullptr) {
    set_error("out is null");
    return CUDF_ERROR_NULL;
  }

  // Step 1: Null-options fast path — resolve the current device's defaults and
  // return. This must come before any struct-field access since there is no
  // struct.
  if (options == nullptr) {
    int32_t dev = 0;
    cudf_status_t const current_status = get_current_device_or_error(&dev);
    if (current_status != CUDF_SUCCESS) {
      return current_status;
    }
    out->device = dev;
    out->stream = cudf::get_default_stream();
    out->stream_lifetime = {};
    out->mr_raw = rmm::mr::get_current_device_resource();
    out->mr = make_exec_mr_ref(out->mr_raw);
    out->mr_lifetime =
        make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_raw(out->mr_raw), out->mr);
    record_exec_context(out->stream, out->mr_raw);
    out->install_current_device_resource();
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
    if (device_status != CUDF_SUCCESS) {
      return device_status;
    }
    out->device = options->device;
  } else {
    int32_t dev = 0;
    cudf_status_t const current_status = get_current_device_or_error(&dev);
    if (current_status != CUDF_SUCCESS) {
      return current_status;
    }
    out->device = dev;
  }

  // Step 4: Resolve stream and MR defaults on the now-active device.
  out->stream = cudf::get_default_stream();
  out->mr_raw = rmm::mr::get_current_device_resource();
  out->mr = make_exec_mr_ref(out->mr_raw);
  out->mr_lifetime =
      make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_raw(out->mr_raw), out->mr);

  // Step 5: Apply optional stream/MR overrides from the caller.
  if (options->stream != nullptr) {
    out->stream = cudf_stream_view_from_handle(options->stream);
    out->stream_lifetime = cudf_stream_lifetime_from_handle(options->stream);
  }
  if (options->mr != nullptr) {
    out->mr_raw = cudf_memory_resource_raw_from_handle(options->mr);
    out->mr = make_exec_mr_ref(out->mr_raw);
    out->mr_lifetime =
        make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_handle(options->mr), out->mr);
  }

  record_exec_context(out->stream, out->mr_raw);
  out->install_current_device_resource();
  return CUDF_SUCCESS;
}

cudf_status_t
cudf_resolve_required_exec_options(cudf_exec_options_t const *options,
                                   std::unique_ptr<cudf_exec_context> *out) {
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
  int32_t device_count = 0;
  cudaError_t const count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess) {
    auto msg = std::string("cudaGetDeviceCount failed: ") +
               cudaGetErrorString(count_status);
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
  if (device_status != CUDF_SUCCESS) {
    return device_status;
  }

  auto stream = cudf_stream_view_from_handle(options->stream);
  auto stream_lifetime = cudf_stream_lifetime_from_handle(options->stream);
  auto *mr_raw = cudf_memory_resource_raw_from_handle(options->mr);
  auto mr = make_exec_mr_ref(mr_raw);
  auto mr_lifetime =
      make_exec_mr_lifetime(cudf_memory_resource_lifetime_from_handle(options->mr), mr);

  *out = std::make_unique<cudf_exec_context>(
      stream, std::move(stream_lifetime), std::move(mr), mr_raw,
      std::move(mr_lifetime), options->device, std::move(guard));

  record_exec_context((*out)->stream, (*out)->mr_raw);
  return CUDF_SUCCESS;
}

void cudf_record_synchronized_stream(rmm::cuda_stream_view stream) {
  g_last_exec_sync_stream_value =
      reinterpret_cast<std::uintptr_t>(stream.value());
}

extern "C" {
void cudf_test_exec_reset_observation() {
  g_last_exec_stream_value = std::numeric_limits<std::uintptr_t>::max();
  g_last_exec_sync_stream_value = std::numeric_limits<std::uintptr_t>::max();
  g_last_exec_mr_raw = nullptr;
  g_last_exec_device = -1;
  g_exec_mr_observations.fill(nullptr);
  g_exec_mr_observation_start = 0;
  g_exec_mr_observation_count = 0;
  g_exec_mr_observation_overflowed = false;
}

cudf_status_t cudf_test_exec_last_stream_value(std::uintptr_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_sync_stream_value(std::uintptr_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_sync_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_mr_raw(void **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_mr_raw;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_observed_mr_count(std::size_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_exec_mr_observation_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_observed_mr_overflowed(int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_exec_mr_observation_overflowed ? 1 : 0;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_observed_mr_raw_at(std::size_t index, void **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (index >= g_exec_mr_observation_count) {
    set_error("execution memory-resource observation index is out of range");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = exec_mr_observation_at(index);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_default_stream_value(std::uintptr_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = reinterpret_cast<std::uintptr_t>(cudf::get_default_stream().value());
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_stream_handle_value(cudf_stream_handle handle,
                                            std::uintptr_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  void *raw = nullptr;
  cudf_status_t const status = cudf_stream_raw(handle, &raw);
  if (status != CUDF_SUCCESS) {
    return status;
  }
  *out = reinterpret_cast<std::uintptr_t>(raw);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_current_device_resource_raw(void **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::scoped_lock lock{cudf_current_device_resource_mutex()};
  *out = rmm::mr::get_current_device_resource();
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_device(int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_device;
  return CUDF_SUCCESS;
}
} // extern "C"
