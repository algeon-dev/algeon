#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cstdint>
#include <limits>

namespace {

thread_local std::uintptr_t g_last_exec_stream_value =
    std::numeric_limits<std::uintptr_t>::max();
thread_local std::uintptr_t g_last_exec_sync_stream_value =
    std::numeric_limits<std::uintptr_t>::max();
thread_local void* g_last_exec_mr_raw = nullptr;

void record_exec_context(rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr_raw) {
  g_last_exec_stream_value =
      reinterpret_cast<std::uintptr_t>(stream.value());
  g_last_exec_mr_raw = mr_raw;
}

}  // namespace

cudf_status_t cudf_resolve_exec_options(cudf_exec_options_t const* options,
                                        cudf_exec_context* out) {
  if (out == nullptr) {
    set_error("out is null");
    return CUDF_ERROR_NULL;
  }

  out->stream = cudf::get_default_stream();
  out->mr = cudf::get_current_device_resource_ref();
  out->mr_raw = rmm::mr::get_current_device_resource();

  if (options == nullptr) {
    record_exec_context(out->stream, out->mr_raw);
    return CUDF_SUCCESS;
  }

  if (options->struct_size < sizeof(cudf_exec_options_t)) {
    set_error("exec options struct_size is smaller than cudf_exec_options_t");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (options->flags != 0) {
    set_error("exec options flags must be zero");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (auto const reserved : options->reserved) {
    if (reserved != 0) {
      set_error("exec options reserved fields must be zero");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }

  if (options->stream != nullptr) {
    out->stream = cudf_stream_view_from_handle(options->stream);
  }
  if (options->mr != nullptr) {
    out->mr_raw = cudf_memory_resource_raw_from_handle(options->mr);
    out->mr = cudf_memory_resource_ref_from_handle(options->mr);
  }

  record_exec_context(out->stream, out->mr_raw);
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
}

cudf_status_t cudf_test_exec_last_stream_value(std::uintptr_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_sync_stream_value(std::uintptr_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_sync_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_mr_raw(void** out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_mr_raw;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_default_stream_value(std::uintptr_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = reinterpret_cast<std::uintptr_t>(cudf::get_default_stream().value());
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_stream_handle_value(cudf_stream_handle handle,
                                            std::uintptr_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    *out = reinterpret_cast<std::uintptr_t>(
        cudf_stream_view_from_handle(handle).value());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_current_device_resource_raw(void** out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = rmm::mr::get_current_device_resource();
  return CUDF_SUCCESS;
}

}  // extern "C"
