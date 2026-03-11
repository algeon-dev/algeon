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
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_keys > 0) {
    CUDF_RUST_NULL_CHECK(keys, "keys is null");
  }
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::size_type> key_vec;
    if (num_keys > 0) { key_vec.assign(keys, keys + num_keys); }

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
                            cudf_duplicate_keep_option_t keep,
                            cudf_stream_compaction_null_equality_t null_equality,
                            cudf_stream_compaction_nan_equality_t nan_equality,
                            cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_keys > 0) {
    CUDF_RUST_NULL_CHECK(keys, "keys is null");
  }
  *out = nullptr;

  cudf::duplicate_keep_option keep_opt;
  switch (keep) {
    case CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY:
      keep_opt = cudf::duplicate_keep_option::KEEP_ANY;
      break;
    case CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST:
      keep_opt = cudf::duplicate_keep_option::KEEP_FIRST;
      break;
    case CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST:
      keep_opt = cudf::duplicate_keep_option::KEEP_LAST;
      break;
    case CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE:
      keep_opt = cudf::duplicate_keep_option::KEEP_NONE;
      break;
    default:
      set_error("invalid keep option");
      return CUDF_ERROR_INVALID_OPERATION;
  }

  cudf::null_equality nulls_eq;
  switch (null_equality) {
    case CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL:
      nulls_eq = cudf::null_equality::EQUAL;
      break;
    case CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL:
      nulls_eq = cudf::null_equality::UNEQUAL;
      break;
    default:
      set_error("invalid null_equality");
      return CUDF_ERROR_INVALID_OPERATION;
  }

  cudf::nan_equality nans_eq;
  switch (nan_equality) {
    case CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL:
      nans_eq = cudf::nan_equality::ALL_EQUAL;
      break;
    case CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL:
      nans_eq = cudf::nan_equality::UNEQUAL;
      break;
    default:
      set_error("invalid nan_equality");
      return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::size_type> key_vec;
    if (num_keys > 0) { key_vec.assign(keys, keys + num_keys); }

    auto result = cudf::distinct(*view, key_vec, keep_opt, nulls_eq, nans_eq,
                                 stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
