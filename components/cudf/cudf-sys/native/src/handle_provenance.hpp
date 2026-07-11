#ifndef CUDF_RUST_HANDLE_PROVENANCE_HPP
#define CUDF_RUST_HANDLE_PROVENANCE_HPP

#include "execution_context_internal.hpp"

#include <cudf/utilities/error.hpp>

#include <cstdint>
#include <memory>
#include <utility>

struct cudf_handle_provenance {
  int32_t device;
  std::shared_ptr<void> stream_lifetime;
  rmm::mr::device_memory_resource *mr_raw;
  std::shared_ptr<void> mr_lifetime;
};

cudf_handle_provenance
cudf_capture_exec_provenance(cudf_exec_context const &exec) noexcept;

cudf_handle_provenance cudf_capture_current_handle_provenance();

void cudf_translate_cleanup_exception_noexcept() noexcept;

cudf_status_t cudf_translate_exception_for_provenance_call();

enum class cudf_provenance_cleanup_result {
  completed,
  not_run,
  failed,
};

template <typename Callable>
cudf_status_t
cudf_call_under_provenance(cudf_handle_provenance const &provenance,
                           Callable &&call) {
  try {
    cudf_device_guard device_guard;
    cudf_status_t const device_status =
        device_guard.activate(provenance.device);
    if (device_status != CUDF_SUCCESS) {
      return device_status;
    }

    std::forward<Callable>(call)();
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_exception_for_provenance_call();
  }
}

template <typename Callable>
decltype(auto)
cudf_invoke_under_provenance(cudf_handle_provenance const &provenance,
                             Callable &&call) {
  cudf_device_guard device_guard;
  cudf_status_t const device_status = device_guard.activate(provenance.device);
  if (device_status != CUDF_SUCCESS) {
    throw cudf::logic_error("failed to activate handle provenance device");
  }

  return std::forward<Callable>(call)();
}

template <typename Callable>
cudf_provenance_cleanup_result
cudf_destroy_under_provenance(cudf_handle_provenance const &provenance,
                              Callable &&cleanup) noexcept {
  try {
    cudf_device_guard device_guard;
    if (device_guard.activate(provenance.device) != CUDF_SUCCESS) {
      return cudf_provenance_cleanup_result::not_run;
    }

    std::forward<Callable>(cleanup)();
    return cudf_provenance_cleanup_result::completed;
  } catch (...) {
    cudf_translate_cleanup_exception_noexcept();
    return cudf_provenance_cleanup_result::failed;
  }
}

#endif // CUDF_RUST_HANDLE_PROVENANCE_HPP
