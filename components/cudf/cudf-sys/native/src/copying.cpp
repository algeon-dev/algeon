#include "execution_context_internal.hpp"
#include "copying.h"
#include "shim_common.h"

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

extern "C" {

cudf_status_t cudf_gather(cudf_table_view_handle input,
                          cudf_column_view_handle gather_map,
                          cudf_out_of_bounds_policy_t bounds_policy,
                          cudf_table_handle* out) {
  return cudf_gather_ex(input, gather_map, bounds_policy, out, nullptr);
}

cudf_status_t cudf_gather_ex(cudf_table_view_handle input,
                             cudf_column_view_handle gather_map,
                             cudf_out_of_bounds_policy_t bounds_policy,
                             cudf_table_handle* out,
                             cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(gather_map, "gather_map is null");

  cudf::out_of_bounds_policy policy;
  switch (bounds_policy) {
    case CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY:
      policy = cudf::out_of_bounds_policy::NULLIFY;
      break;
    case CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK:
      policy = cudf::out_of_bounds_policy::DONT_CHECK;
      break;
    default:
      set_error("invalid bounds_policy");
      return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto* map_view = static_cast<cudf::column_view*>(gather_map);
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto result = cudf::gather(*view, *map_view, policy, exec.stream, exec.mr);
    *out = cudf_release_table_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_copy_if_else_vv(cudf_column_view_handle lhs,
                                    cudf_column_view_handle rhs,
                                    cudf_column_view_handle boolean_mask,
                                    cudf_column_handle* out) {
  return cudf_copy_if_else_vv_ex(lhs, rhs, boolean_mask, out, nullptr);
}

cudf_status_t cudf_copy_if_else_vv_ex(cudf_column_view_handle lhs,
                                       cudf_column_view_handle rhs,
                                       cudf_column_view_handle boolean_mask,
                                       cudf_column_handle* out,
                                       cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(lhs, "lhs is null");
  CUDF_RUST_NULL_CHECK(rhs, "rhs is null");
  CUDF_RUST_NULL_CHECK(boolean_mask, "boolean_mask is null");
  CUDF_RUST_TRY {
    auto* lhs_view = static_cast<cudf::column_view*>(lhs);
    auto* rhs_view = static_cast<cudf::column_view*>(rhs);
    auto* mask_view = static_cast<cudf::column_view*>(boolean_mask);
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto result = cudf::copy_if_else(*lhs_view, *rhs_view, *mask_view,
                                     exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
