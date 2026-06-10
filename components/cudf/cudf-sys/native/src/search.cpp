#include "search.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/search.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

extern "C" {
cudf_status_t cudf_contains_ex(cudf_column_view_handle haystack,
                               cudf_column_view_handle needles,
                               cudf_column_handle* out,
                               cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(haystack, "haystack is null");
  CUDF_RUST_NULL_CHECK(needles, "needles is null");
  CUDF_RUST_TRY
  {
    auto* haystack_view             = static_cast<cudf::column_view*>(haystack);
    auto* needles_view              = static_cast<cudf::column_view*>(needles);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      auto result = cudf::contains(*haystack_view, *needles_view, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
