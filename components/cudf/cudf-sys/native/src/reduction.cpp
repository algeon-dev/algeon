#include "reduction.h"
#include "shim_common.h"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

namespace {

cudf_status_t to_scan_type(cudf_scan_type_t input, cudf::scan_type* out) {
  switch (input) {
    case CUDF_SCAN_INCLUSIVE:
      *out = cudf::scan_type::INCLUSIVE;
      return CUDF_SUCCESS;
    case CUDF_SCAN_EXCLUSIVE:
      *out = cudf::scan_type::EXCLUSIVE;
      return CUDF_SUCCESS;
    default:
      set_error("invalid scan_type");
      return CUDF_ERROR_INVALID_OPERATION;
  }
}

cudf_status_t to_null_policy(int32_t input, cudf::null_policy* out) {
  switch (input) {
    case 0:
      *out = cudf::null_policy::EXCLUDE;
      return CUDF_SUCCESS;
    case 1:
      *out = cudf::null_policy::INCLUDE;
      return CUDF_SUCCESS;
    default:
      set_error("invalid null_policy");
      return CUDF_ERROR_INVALID_OPERATION;
  }
}

}  // namespace

extern "C" {

cudf_status_t cudf_reduce_aggregation_create(
    cudf_reduce_aggregation_kind_t kind,
    cudf_reduce_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    std::unique_ptr<cudf::reduce_aggregation> agg;
    switch (kind) {
      case CUDF_REDUCE_AGG_SUM:
        agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MIN:
        agg = cudf::make_min_aggregation<cudf::reduce_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MAX:
        agg = cudf::make_max_aggregation<cudf::reduce_aggregation>();
        break;
      case CUDF_REDUCE_AGG_PRODUCT:
        agg = cudf::make_product_aggregation<cudf::reduce_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MEAN:
        agg = cudf::make_mean_aggregation<cudf::reduce_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MEDIAN:
        agg = cudf::make_median_aggregation<cudf::reduce_aggregation>();
        break;
      case CUDF_REDUCE_AGG_ANY:
        agg = cudf::make_any_aggregation<cudf::reduce_aggregation>();
        break;
      case CUDF_REDUCE_AGG_ALL:
        agg = cudf::make_all_aggregation<cudf::reduce_aggregation>();
        break;
      default:
        set_error("unknown reduce aggregation kind");
        return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = static_cast<cudf_reduce_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reduce_aggregation_create_variance(
    int32_t ddof,
    cudf_reduce_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto agg = cudf::make_variance_aggregation<cudf::reduce_aggregation>(ddof);
    *out = static_cast<cudf_reduce_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_reduce_aggregation_create_std(
    int32_t ddof,
    cudf_reduce_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto agg = cudf::make_std_aggregation<cudf::reduce_aggregation>(ddof);
    *out = static_cast<cudf_reduce_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_reduce_aggregation_destroy(cudf_reduce_aggregation_handle handle) {
  delete static_cast<cudf::reduce_aggregation*>(handle);
}

cudf_status_t cudf_scan_aggregation_create(
    cudf_reduce_aggregation_kind_t kind,
    cudf_scan_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    std::unique_ptr<cudf::scan_aggregation> agg;
    switch (kind) {
      case CUDF_REDUCE_AGG_SUM:
        agg = cudf::make_sum_aggregation<cudf::scan_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MIN:
        agg = cudf::make_min_aggregation<cudf::scan_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MAX:
        agg = cudf::make_max_aggregation<cudf::scan_aggregation>();
        break;
      case CUDF_REDUCE_AGG_PRODUCT:
        agg = cudf::make_product_aggregation<cudf::scan_aggregation>();
        break;
      default:
        set_error("unsupported scan aggregation kind");
        return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = static_cast<cudf_scan_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_scan_aggregation_destroy(cudf_scan_aggregation_handle handle) {
  delete static_cast<cudf::scan_aggregation*>(handle);
}

cudf_status_t cudf_reduce(cudf_column_view_handle input,
                          cudf_reduce_aggregation_handle agg,
                          cudf_data_type_handle output_type,
                          cudf_scalar_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(agg, "agg is null");
  CUDF_RUST_NULL_CHECK(output_type, "output_type is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto* reduce_agg = static_cast<cudf::reduce_aggregation*>(agg);
    auto* dtype = static_cast<cudf::data_type*>(output_type);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::reduce(*view, *reduce_agg, *dtype, stream, mr);
    *out = static_cast<cudf_scalar_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scan(cudf_column_view_handle input,
                        cudf_scan_aggregation_handle agg,
                        cudf_scan_type_t scan_type,
                        int32_t null_policy,
                        cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(agg, "agg is null");
  cudf::scan_type cudf_scan_type;
  cudf_status_t status = to_scan_type(scan_type, &cudf_scan_type);
  if (status != CUDF_SUCCESS) {
    return status;
  }
  cudf::null_policy cudf_null_policy;
  status = to_null_policy(null_policy, &cudf_null_policy);
  if (status != CUDF_SUCCESS) {
    return status;
  }
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto* scan_agg = static_cast<cudf::scan_aggregation*>(agg);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::scan(*view, *scan_agg, cudf_scan_type,
                             cudf_null_policy, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_minmax(cudf_column_view_handle input,
                          cudf_scalar_handle* out_min,
                          cudf_scalar_handle* out_max) {
  CUDF_RUST_NULL_CHECK(out_min, "out_min is null");
  *out_min = nullptr;
  CUDF_RUST_NULL_CHECK(out_max, "out_max is null");
  *out_max = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto [min_scalar, max_scalar] = cudf::minmax(*view, stream, mr);
    *out_min = static_cast<cudf_scalar_handle>(min_scalar.release());
    *out_max = static_cast<cudf_scalar_handle>(max_scalar.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
