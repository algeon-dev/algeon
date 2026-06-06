#include "execution_context_internal.hpp"
#include "replace.h"
#include "shim_common.h"

#include <cudf/replace.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

extern "C" {

cudf_status_t cudf_replace_nulls_scalar(cudf_column_view_handle input,
                                        cudf_scalar_handle replacement,
                                        cudf_column_handle* out) {
  return cudf_replace_nulls_scalar_ex(input, replacement, out, nullptr);
}

cudf_status_t cudf_replace_nulls_scalar_ex(
    cudf_column_view_handle input,
    cudf_scalar_handle replacement,
    cudf_column_handle* out,
    cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(replacement, "replacement is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto* scalar = static_cast<cudf::scalar*>(replacement);
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto result = cudf::replace_nulls(*view, *scalar, exec.stream, exec.mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_replace_nulls_column(cudf_column_view_handle input,
                                        cudf_column_view_handle replacement,
                                        cudf_column_handle* out) {
  return cudf_replace_nulls_column_ex(input, replacement, out, nullptr);
}

cudf_status_t cudf_replace_nulls_column_ex(
    cudf_column_view_handle input,
    cudf_column_view_handle replacement,
    cudf_column_handle* out,
    cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(replacement, "replacement is null");
  CUDF_RUST_TRY {
    auto* view_input = static_cast<cudf::column_view*>(input);
    auto* view_replacement = static_cast<cudf::column_view*>(replacement);
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto result = cudf::replace_nulls(*view_input, *view_replacement, exec.stream, exec.mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_replace_nulls_policy(cudf_column_view_handle input,
                                        int32_t policy,
                                        cudf_column_handle* out) {
  return cudf_replace_nulls_policy_ex(input, policy, out, nullptr);
}

cudf_status_t cudf_replace_nulls_policy_ex(
    cudf_column_view_handle input,
    int32_t policy,
    cudf_column_handle* out,
    cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto const rp = (policy == 0) ? cudf::replace_policy::PRECEDING
                                  : cudf::replace_policy::FOLLOWING;
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto result = cudf::replace_nulls(*view, rp, exec.stream, exec.mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
