#include "handle_provenance.hpp"

#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/utilities/error.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <mutex>

namespace {
thread_local cudf_status_t g_next_provenance_destroy_failure = CUDF_SUCCESS;
}

cudf_handle_provenance
cudf_capture_exec_provenance(cudf_exec_context const &exec) noexcept {
  return cudf_handle_provenance{exec.device, exec.stream_lifetime, exec.mr_raw,
                                exec.mr_lifetime};
}

cudf_handle_provenance cudf_capture_current_handle_provenance() {
  int32_t device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  auto *mr_raw = cudf_tracked_current_device_resource_raw();
  return cudf_handle_provenance{
      device, {}, mr_raw, cudf_memory_resource_lifetime_from_raw(mr_raw)};
}

cudf_status_t cudf_translate_cleanup_exception_noexcept() noexcept {
  try {
    return translate_exception();
  } catch (...) {
    return CUDF_ERROR_UNKNOWN;
  }
}

cudf_status_t cudf_take_test_provenance_destroy_failure() noexcept {
  auto const status = g_next_provenance_destroy_failure;
  g_next_provenance_destroy_failure = CUDF_SUCCESS;
  return status;
}

cudf_status_t cudf_translate_exception_for_provenance_call() {
  return translate_exception();
}

extern "C" void
cudf_test_fail_next_provenance_destroy(cudf_status_t status) {
  g_next_provenance_destroy_failure = status;
}
