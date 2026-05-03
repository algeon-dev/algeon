#ifndef CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP
#define CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP

#include "execution_context.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace rmm::mr {
class device_memory_resource;
}

/// RAII guard that saves the calling thread's current CUDA device and restores
/// it on destruction. Used inside `cudf_resolve_exec_options()` to activate
/// the device requested by `cudf_exec_options_t::device`.
class cudf_device_guard {
 public:
  cudf_device_guard() = default;

  /// Save the current device and switch to `target`.  No-op when
  /// `target` matches the current device.
  cudf_status_t activate(int32_t target);

  ~cudf_device_guard()
  {
    if (active_) { cudaSetDevice(previous_); }
  }

  // non-copyable
  cudf_device_guard(cudf_device_guard const&)            = delete;
  cudf_device_guard& operator=(cudf_device_guard const&) = delete;

  // movable (transfer ownership of the restore obligation)
  cudf_device_guard(cudf_device_guard&& other) noexcept
    : previous_(other.previous_), active_(other.active_)
  {
    other.active_ = false;
  }
  cudf_device_guard& operator=(cudf_device_guard&& other) noexcept
  {
    if (this != &other) {
      if (active_) { cudaSetDevice(previous_); }
      previous_     = other.previous_;
      active_       = other.active_;
      other.active_ = false;
    }
    return *this;
  }

 private:
  int32_t previous_ = 0;
  bool active_      = false;
};

struct cudf_exec_context {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  rmm::mr::device_memory_resource* mr_raw;
  int32_t device;
  cudf_device_guard guard;
};

/// RAII guard for libcudf APIs whose public constructor surface still resolves
/// allocations through RMM's current device resource instead of an explicit MR
/// parameter. Keep this scoped to the smallest call that performs the
/// allocation and prefer explicit `mr` arguments whenever libcudf offers them.
class cudf_current_device_resource_guard {
 public:
  explicit cudf_current_device_resource_guard(rmm::mr::device_memory_resource* target);
  ~cudf_current_device_resource_guard();

  cudf_current_device_resource_guard(cudf_current_device_resource_guard const&)            = delete;
  cudf_current_device_resource_guard& operator=(cudf_current_device_resource_guard const&) = delete;

 private:
  rmm::mr::device_memory_resource* previous_;
};

cudf_status_t cudf_resolve_exec_options(cudf_exec_options_t const* options, cudf_exec_context* out);
void cudf_record_synchronized_stream(rmm::cuda_stream_view stream);

#endif  // CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP
