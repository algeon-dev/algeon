#include "hashing.h"
#include "shim_common.h"

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

extern "C" {

cudf_status_t cudf_hashing_murmurhash3_x86_32(cudf_table_view_handle input,
                                               uint32_t seed,
                                               cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::hashing::murmurhash3_x86_32(*view, seed, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hashing_xxhash_64(cudf_table_view_handle input,
                                     uint64_t seed,
                                     cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::hashing::xxhash_64(*view, seed, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
