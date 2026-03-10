#include "stream_compaction.h"
#include "shim_common.h"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <vector>

extern "C" {

cudf_status_t cudf_drop_nulls(cudf_table_view_handle input,
                              const int* keys,
                              int num_keys,
                              int keep_threshold,
                              cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::size_type> key_vec(keys, keys + num_keys);

    auto result = cudf::drop_nulls(*view, key_vec,
                                   static_cast<cudf::size_type>(keep_threshold),
                                   stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_distinct(cudf_table_view_handle input,
                            const int* keys,
                            int num_keys,
                            int keep,
                            int null_equality,
                            int nan_equality,
                            cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::size_type> key_vec(keys, keys + num_keys);

    auto keep_opt = static_cast<cudf::duplicate_keep_option>(keep);
    auto nulls_eq = static_cast<cudf::null_equality>(null_equality);
    auto nans_eq = static_cast<cudf::nan_equality>(nan_equality);

    auto result = cudf::distinct(*view, key_vec, keep_opt, nulls_eq, nans_eq,
                                 stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
