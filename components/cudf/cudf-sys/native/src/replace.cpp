#include "replace.h"

#include "execution_context_internal.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

extern "C" {
cudf_status_t cudf_replace_nulls_scalar_ex(cudf_column_view_handle input,
                                           cudf_scalar_handle replacement,
                                           cudf_column_handle* out,
                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(replacement, "replacement is null");
  CUDF_RUST_TRY
  {
    auto* view                      = static_cast<cudf::column_view*>(input);
    auto& scalar                    = cudf_scalar_from_handle(replacement);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = cudf::replace_nulls(*view, scalar, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_replace_nulls_column_ex(cudf_column_view_handle input,
                                           cudf_column_view_handle replacement,
                                           cudf_column_handle* out,
                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(replacement, "replacement is null");
  CUDF_RUST_TRY
  {
    auto* view_input                = static_cast<cudf::column_view*>(input);
    auto* view_replacement          = static_cast<cudf::column_view*>(replacement);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = cudf::replace_nulls(*view_input, *view_replacement, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_replace_nulls_policy_ex(cudf_column_view_handle input,
                                           int32_t policy,
                                           cudf_column_handle* out,
                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto* view = static_cast<cudf::column_view*>(input);
    cudf::replace_policy rp;
    switch (policy) {
      case 0: rp = cudf::replace_policy::PRECEDING; break;
      case 1: rp = cudf::replace_policy::FOLLOWING; break;
      default:
        set_error("replace policy must be 0 (PRECEDING) or 1 (FOLLOWING)");
        return CUDF_ERROR_INVALID_OPERATION;
    }
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = cudf::replace_nulls(*view, rp, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
