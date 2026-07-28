#include "cudf_rust/arrow_host.h"

#include "execution_context_internal.hpp"
#include "owning_handle.hpp"
#include "shim_common.h"

#include <cudf/interop.hpp>

#include <memory>

cudf_status_t cudf_arrow_host_column_import_ex(void const* schema,
                                               void const* array,
                                               cudf_exec_options_t const* options,
                                               cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(schema, "schema is null");
  CUDF_RUST_NULL_CHECK(array, "array is null");

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const* schema_ptr = reinterpret_cast<ArrowSchema const*>(schema);
    auto const* array_ptr  = static_cast<ArrowArray const*>(array);

    cudf_host_source_copy_sync_guard host_copy_guard{exec->stream};
    host_copy_guard.mark_copy_may_be_pending();
    auto column = cudf::from_arrow_column(schema_ptr, array_ptr, exec->stream, *exec->mr);
    host_copy_guard.synchronize_and_release();

    *out = cudf_release_column_handle(std::move(column), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
