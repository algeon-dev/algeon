#include "execution_context_internal.hpp"
#include "filter.h"
#include "shim_common.h"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

extern "C" {

cudf_status_t cudf_apply_boolean_mask(cudf_table_view_handle input,
                                      cudf_column_view_handle mask,
                                      cudf_table_handle* out) {
  return cudf_apply_boolean_mask_ex(input, mask, out, nullptr);
}

cudf_status_t cudf_apply_boolean_mask_ex(cudf_table_view_handle input,
                                         cudf_column_view_handle mask,
                                         cudf_table_handle* out,
                                         cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(mask, "mask is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto* mask_view = static_cast<cudf::column_view*>(mask);
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           rmm::mr::get_current_device_resource()};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto result = cudf::apply_boolean_mask(*view, *mask_view, exec.stream, exec.mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
