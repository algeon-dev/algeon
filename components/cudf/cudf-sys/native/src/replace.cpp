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
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(replacement, "replacement is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto* scalar = static_cast<cudf::scalar*>(replacement);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::replace_nulls(*view, *scalar, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_replace_nulls_column(cudf_column_view_handle input,
                                        cudf_column_view_handle replacement,
                                        cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(replacement, "replacement is null");
  CUDF_RUST_TRY {
    auto* view_input = static_cast<cudf::column_view*>(input);
    auto* view_replacement = static_cast<cudf::column_view*>(replacement);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::replace_nulls(*view_input, *view_replacement, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
