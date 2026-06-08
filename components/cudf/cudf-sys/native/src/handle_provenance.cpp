#include "handle_provenance.hpp"

#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/utilities/error.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <mutex>

cudf_handle_provenance
cudf_capture_exec_provenance(cudf_exec_context const &exec) noexcept {
  return cudf_handle_provenance{exec.device, exec.stream_lifetime, exec.mr_raw,
                                exec.mr_lifetime};
}

cudf_handle_provenance cudf_capture_current_handle_provenance() {
  std::scoped_lock lock{cudf_current_device_resource_mutex()};

  int32_t device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  auto *mr_raw = cudf_tracked_current_device_resource_raw();
  return cudf_handle_provenance{
      device, {}, mr_raw, cudf_memory_resource_lifetime_from_raw(mr_raw)};
}

void cudf_translate_cleanup_exception_noexcept() noexcept {
  try {
    static_cast<void>(translate_exception());
  } catch (...) {
  }
}

cudf_status_t cudf_translate_exception_for_provenance_call() {
  return translate_exception();
}
