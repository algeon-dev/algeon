#ifndef CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP
#define CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP

#include "execution_context.h"
#include "memory_resource_compat.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

/// RAII guard that saves the calling thread's current CUDA device and restores
/// it on destruction. Used inside `cudf_resolve_exec_options()` to activate
/// the device requested by `cudf_exec_options_t::device`.
class cudf_device_guard {
public:
  cudf_device_guard() = default;

  /// Save the current device and switch to `target`.  No-op when
  /// `target` matches the current device.
  cudf_status_t activate(int32_t target);

  ~cudf_device_guard() {
    if (active_) {
      cudaSetDevice(previous_);
    }
  }

  // non-copyable
  cudf_device_guard(cudf_device_guard const &) = delete;

  cudf_device_guard &operator=(cudf_device_guard const &) = delete;

  // movable (transfer ownership of the restore obligation)
  cudf_device_guard(cudf_device_guard &&other) noexcept
      : previous_(other.previous_), active_(other.active_) {
    other.active_ = false;
  }

  cudf_device_guard &operator=(cudf_device_guard &&other) noexcept {
    if (this != &other) {
      if (active_) {
        cudaSetDevice(previous_);
      }
      previous_ = other.previous_;
      active_ = other.active_;
      other.active_ = false;
    }
    return *this;
  }

private:
  int32_t previous_ = 0;
  bool active_ = false;
};

// Device-domain invariant: each device's current RMM resource is installed
// once when its memory domain is created (manager creation or lazy default)
// and never changes afterwards. Operations must never swap it; libcudf
// temporaries deliberately resolve through the long-lived per-device adaptor
// and are accounted per stream by the reservation-aware adaptor. The
// identity map behind these helpers carries its own internal lock.

rmm::mr::device_memory_resource *cudf_tracked_current_device_resource_raw();

void cudf_set_current_device_resource_raw(
    rmm::mr::device_memory_resource *target);

struct cudf_exec_context {
  rmm::cuda_stream_view stream;
  std::shared_ptr<void> stream_lifetime;
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> mr;
  rmm::mr::device_memory_resource *mr_raw;
  std::shared_ptr<void> mr_lifetime;
  int32_t device;
  cudf_device_guard guard;

  cudf_exec_context(rmm::cuda_stream_view stream,
                    std::shared_ptr<void> stream_lifetime,
                    std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> mr,
                    rmm::mr::device_memory_resource *mr_raw,
                    std::shared_ptr<void> mr_lifetime, int32_t device,
                    cudf_device_guard guard = {});

  cudf_exec_context(cudf_exec_context const &) = delete;
  cudf_exec_context &operator=(cudf_exec_context const &) = delete;
  cudf_exec_context(cudf_exec_context &&other) noexcept = default;
  cudf_exec_context &operator=(cudf_exec_context &&) noexcept = delete;
};

cudf_exec_context cudf_make_exec_context();

cudf_status_t cudf_resolve_exec_options(cudf_exec_options_t const *options,
                                        cudf_exec_context *out);

cudf_status_t
cudf_resolve_required_exec_options(cudf_exec_options_t const *options,
                                   std::unique_ptr<cudf_exec_context> *out);

void cudf_record_synchronized_stream(rmm::cuda_stream_view stream);

#endif // CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP
