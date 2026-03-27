#include "search.h"
#include "shim_common.h"

#include <cudf/search.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

extern "C" {

cudf_status_t cudf_contains(cudf_column_view_handle haystack,
                            cudf_column_view_handle needles,
                            cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(haystack, "haystack is null");
  CUDF_RUST_NULL_CHECK(needles, "needles is null");
  CUDF_RUST_TRY {
    auto* haystack_view = static_cast<cudf::column_view*>(haystack);
    auto* needles_view = static_cast<cudf::column_view*>(needles);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::contains(*haystack_view, *needles_view, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
