#include "groupby.h"
#include "shim_common.h"

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <vector>

extern "C" {

cudf_status_t cudf_aggregation_create(cudf_aggregation_kind_t kind,
                                      cudf_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    std::unique_ptr<cudf::groupby_aggregation> agg;
    switch (kind) {
      case CUDF_AGG_SUM:
        agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
        break;
      case CUDF_AGG_MIN:
        agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
        break;
      case CUDF_AGG_MAX:
        agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
        break;
      case CUDF_AGG_COUNT_VALID:
        agg = cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::EXCLUDE);
        break;
      case CUDF_AGG_COUNT_ALL:
        agg = cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::INCLUDE);
        break;
      case CUDF_AGG_MEAN:
        agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
        break;
      case CUDF_AGG_MEDIAN:
        agg = cudf::make_median_aggregation<cudf::groupby_aggregation>();
        break;
      case CUDF_AGG_NUNIQUE:
        agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::EXCLUDE);
        break;
      case CUDF_AGG_STD:
        agg = cudf::make_std_aggregation<cudf::groupby_aggregation>(1);
        break;
      case CUDF_AGG_VARIANCE:
        agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>(1);
        break;
      default:
        set_error("unknown aggregation kind");
        return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_aggregation_create_count(cudf_null_policy_t null_policy,
                                            cudf_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto policy = (null_policy == CUDF_NULL_POLICY_INCLUDE)
                      ? cudf::null_policy::INCLUDE
                      : cudf::null_policy::EXCLUDE;
    auto agg =
        cudf::make_count_aggregation<cudf::groupby_aggregation>(policy);
    *out = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_aggregation_create_std(int32_t ddof,
                                          cudf_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto agg =
        cudf::make_std_aggregation<cudf::groupby_aggregation>(ddof);
    *out = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_aggregation_create_variance(int32_t ddof,
                                               cudf_aggregation_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto agg =
        cudf::make_variance_aggregation<cudf::groupby_aggregation>(ddof);
    *out = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_aggregation_destroy(cudf_aggregation_handle handle) {
  delete static_cast<cudf::groupby_aggregation*>(handle);
}

cudf_status_t cudf_groupby_aggregate(
    cudf_table_view_handle keys,
    const cudf_column_view_handle* values,
    const cudf_aggregation_handle* aggregations,
    const int32_t* agg_counts,
    int32_t num_requests,
    cudf_null_policy_t keys_null_policy,
    cudf_table_handle* out_keys,
    cudf_column_handle** out_values,
    int32_t* out_total_columns) {
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  CUDF_RUST_NULL_CHECK(out_keys, "out_keys is null");
  CUDF_RUST_NULL_CHECK(out_values, "out_values is null");
  CUDF_RUST_NULL_CHECK(out_total_columns, "out_total_columns is null");
  *out_keys = nullptr;
  *out_values = nullptr;
  *out_total_columns = 0;
  if (num_requests < 0) {
    set_error("num_requests must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_requests > 0) {
    CUDF_RUST_NULL_CHECK(values, "values is null");
    CUDF_RUST_NULL_CHECK(aggregations, "aggregations is null");
    CUDF_RUST_NULL_CHECK(agg_counts, "agg_counts is null");
  }
  CUDF_RUST_TRY {
    auto* keys_view = static_cast<cudf::table_view*>(keys);
    auto null_policy = (keys_null_policy == CUDF_NULL_POLICY_INCLUDE)
                           ? cudf::null_policy::INCLUDE
                           : cudf::null_policy::EXCLUDE;

    cudf::groupby::groupby gb(*keys_view, null_policy);

    // Build aggregation requests.
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.reserve(num_requests);

    for (int32_t i = 0; i < num_requests; ++i) {
      if (agg_counts[i] < 0) {
        set_error("agg_counts entries must be >= 0");
        return CUDF_ERROR_INVALID_OPERATION;
      }
    }

    int32_t agg_offset = 0;
    for (int32_t i = 0; i < num_requests; ++i) {
      CUDF_RUST_NULL_CHECK(values[i], "value column view handle is null");
      cudf::groupby::aggregation_request req;
      req.values = *static_cast<cudf::column_view*>(values[i]);
      for (int32_t j = 0; j < agg_counts[i]; ++j) {
        CUDF_RUST_NULL_CHECK(aggregations[agg_offset + j], "aggregation handle is null");
        auto* agg = static_cast<cudf::groupby_aggregation*>(
            aggregations[agg_offset + j]);
        auto cloned = agg->clone();
        req.aggregations.push_back(
            std::unique_ptr<cudf::groupby_aggregation>(
                dynamic_cast<cudf::groupby_aggregation*>(cloned.release())));
      }
      agg_offset += agg_counts[i];
      requests.push_back(std::move(req));
    }

    auto [result_keys, result_values] = gb.aggregate(requests);

    // Flatten result columns into a single array.
    int32_t total = 0;
    for (auto& rv : result_values) {
      total += static_cast<int32_t>(rv.results.size());
    }

    auto* col_array = new cudf_column_handle[total];
    int32_t idx = 0;
    for (auto& rv : result_values) {
      for (auto& col : rv.results) {
        col_array[idx++] = static_cast<cudf_column_handle>(col.release());
      }
    }

    // Publish outputs only after all work succeeds.
    *out_keys = static_cast<cudf_table_handle>(result_keys.release());
    *out_total_columns = total;
    *out_values = col_array;

    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_free_column_array(cudf_column_handle* array) {
  delete[] array;
}

}  // extern "C"
