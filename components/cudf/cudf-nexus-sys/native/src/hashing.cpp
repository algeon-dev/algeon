#include "hashing.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

extern "C" {
cudf_status_t cudf_hashing_murmurhash3_x86_32_ex(cudf_table_view_handle input,
                                                 uint32_t seed,
                                                 cudf_column_handle* out,
                                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto* view = static_cast<cudf::table_view*>(input);
    auto exec  = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    {
      auto result = cudf::hashing::murmurhash3_x86_32(*view, seed, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hashing_xxhash_64_ex(cudf_table_view_handle input,
                                        uint64_t seed,
                                        cudf_column_handle* out,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto* view = static_cast<cudf::table_view*>(input);
    auto exec  = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    {
      auto result = cudf::hashing::xxhash_64(*view, seed, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
