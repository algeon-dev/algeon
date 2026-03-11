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
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(gather_map, "gather_map is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;

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
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::gather(*view, *map_view, policy, stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
