#include "groupby.h"

#include "execution_context_internal.hpp"
#include "handle_provenance.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace {
thread_local bool g_fail_next_groupby_negative_total_columns = false;

bool should_fail_next_groupby_negative_total_columns()
{
  if (!g_fail_next_groupby_negative_total_columns) { return false; }
  g_fail_next_groupby_negative_total_columns = false;
  return true;
}

cudf_status_t to_cudf_null_policy(cudf_null_policy_t input,
                                  cudf::null_policy* out,
                                  const char* param_name)
{
  switch (input) {
    case CUDF_NULL_POLICY_EXCLUDE: *out = cudf::null_policy::EXCLUDE; return CUDF_SUCCESS;
    case CUDF_NULL_POLICY_INCLUDE: *out = cudf::null_policy::INCLUDE; return CUDF_SUCCESS;
    default: set_error(std::string("invalid ") + param_name); return CUDF_ERROR_INVALID_OPERATION;
  }
}
}  // namespace

extern "C" {
void cudf_test_groupby_fail_next_negative_total_columns()
{
  g_fail_next_groupby_negative_total_columns = true;
}

cudf_status_t cudf_aggregation_create(cudf_aggregation_kind_t kind, cudf_aggregation_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    std::unique_ptr<cudf::groupby_aggregation> agg;
    switch (kind) {
      case CUDF_AGG_SUM: agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>(); break;
      case CUDF_AGG_MIN: agg = cudf::make_min_aggregation<cudf::groupby_aggregation>(); break;
      case CUDF_AGG_MAX: agg = cudf::make_max_aggregation<cudf::groupby_aggregation>(); break;
      case CUDF_AGG_COUNT_VALID:
        agg = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
        break;
      case CUDF_AGG_COUNT_ALL:
        agg = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
        break;
      case CUDF_AGG_MEAN: agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>(); break;
      case CUDF_AGG_MEDIAN: agg = cudf::make_median_aggregation<cudf::groupby_aggregation>(); break;
      case CUDF_AGG_NUNIQUE:
        agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
        break;
      case CUDF_AGG_STD: agg = cudf::make_std_aggregation<cudf::groupby_aggregation>(1); break;
      case CUDF_AGG_VARIANCE:
        agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>(1);
        break;
      default: set_error("unknown aggregation kind"); return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_aggregation_create_count(cudf_null_policy_t null_policy,
                                            cudf_aggregation_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  cudf::null_policy policy;
  cudf_status_t status = to_cudf_null_policy(null_policy, &policy, "null_policy");
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>(policy);
    *out     = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_aggregation_create_std(int32_t ddof, cudf_aggregation_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto agg = cudf::make_std_aggregation<cudf::groupby_aggregation>(ddof);
    *out     = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_aggregation_create_variance(int32_t ddof, cudf_aggregation_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>(ddof);
    *out     = static_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_aggregation_destroy(cudf_aggregation_handle handle)
{
  delete static_cast<cudf::groupby_aggregation*>(handle);
}

cudf_status_t cudf_groupby_scan_aggregation_create(cudf_reduce_aggregation_kind_t kind,
                                                   cudf_groupby_scan_aggregation_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    std::unique_ptr<cudf::groupby_scan_aggregation> agg;
    switch (kind) {
      case CUDF_REDUCE_AGG_SUM:
        agg = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MIN:
        agg = cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
        break;
      case CUDF_REDUCE_AGG_MAX:
        agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
        break;
      case CUDF_REDUCE_AGG_PRODUCT:
        agg = cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
        break;
      default: set_error("unknown grouped scan aggregation kind"); return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = static_cast<cudf_groupby_scan_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_groupby_scan_aggregation_destroy(cudf_groupby_scan_aggregation_handle handle)
{
  delete static_cast<cudf::groupby_scan_aggregation*>(handle);
}

cudf_status_t cudf_groupby_aggregate_ex(cudf_table_view_handle keys,
                                        const cudf_column_view_handle* values,
                                        const cudf_aggregation_handle* aggregations,
                                        const int32_t* agg_counts,
                                        int32_t num_requests,
                                        int32_t num_aggregations,
                                        cudf_null_policy_t keys_null_policy,
                                        cudf_table_handle* out_keys,
                                        cudf_column_handle** out_values,
                                        int32_t* out_total_columns,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_keys, "out_keys is null");
  *out_keys = nullptr;
  CUDF_RUST_NULL_CHECK(out_values, "out_values is null");
  *out_values = nullptr;
  CUDF_RUST_NULL_CHECK(out_total_columns, "out_total_columns is null");
  *out_total_columns = 0;
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  if (num_requests < 0) {
    set_error("num_requests must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_aggregations < 0) {
    set_error("num_aggregations must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_requests > 0) {
    CUDF_RUST_NULL_CHECK(values, "values is null");
    CUDF_RUST_NULL_CHECK(agg_counts, "agg_counts is null");
  }
  if (num_aggregations > 0) { CUDF_RUST_NULL_CHECK(aggregations, "aggregations is null"); }
  cudf::null_policy null_policy;
  cudf_status_t status = to_cudf_null_policy(keys_null_policy, &null_policy, "keys_null_policy");
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* keys_view = static_cast<cudf::table_view*>(keys);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      cudf::groupby::groupby gb(*keys_view, null_policy);

      // Build aggregation requests.
      std::vector<cudf::groupby::aggregation_request> requests;
      requests.reserve(num_requests);

      int64_t required_aggregations = 0;
      for (int32_t i = 0; i < num_requests; ++i) {
        if (agg_counts[i] < 0) {
          set_error("agg_counts entries must be >= 0");
          return CUDF_ERROR_INVALID_OPERATION;
        }
        required_aggregations += agg_counts[i];
      }
      if (required_aggregations != num_aggregations) {
        set_error("num_aggregations must equal sum(agg_counts)");
        return CUDF_ERROR_INVALID_OPERATION;
      }

      int32_t agg_offset = 0;
      for (int32_t i = 0; i < num_requests; ++i) {
        CUDF_RUST_NULL_CHECK(values[i], "value column view handle is null");
        cudf::groupby::aggregation_request req;
        req.values = *static_cast<cudf::column_view*>(values[i]);
        for (int32_t j = 0; j < agg_counts[i]; ++j) {
          CUDF_RUST_NULL_CHECK(aggregations[agg_offset + j], "aggregation handle is null");
          auto* agg   = static_cast<cudf::groupby_aggregation*>(aggregations[agg_offset + j]);
          auto cloned = agg->clone();
          req.aggregations.push_back(std::unique_ptr<cudf::groupby_aggregation>(
            dynamic_cast<cudf::groupby_aggregation*>(cloned.release())));
        }
        agg_offset += agg_counts[i];
        requests.push_back(std::move(req));
      }

      auto [result_keys, result_values] = gb.aggregate(requests, exec->stream, *exec->mr);

      if (should_fail_next_groupby_negative_total_columns()) {
        *out_keys          = cudf_release_table_handle(std::move(result_keys));
        *out_values        = nullptr;
        *out_total_columns = -1;
        return CUDF_SUCCESS;
      }

      // Flatten result columns into a single array.
      int32_t total = 0;
      for (auto& [results] : result_values) {
        total += static_cast<int32_t>(results.size());
      }

      cudf_column_handle* col_array = nullptr;
      if (total > 0) {
        col_array = new (std::nothrow) cudf_column_handle[total];
        if (col_array == nullptr) {
          set_error("failed to allocate groupby result column handle array");
          return CUDF_ERROR_OUT_OF_MEMORY;
        }
      }
      // invariant: idx only advances when results is non-empty, which implies
      // total > 0, so col_array was successfully allocated above.
      int32_t idx = 0;
      for (auto& [results] : result_values) {
        for (auto& col : results) {
          col_array[idx++] = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
        }
      }

      // Publish outputs only after all work succeeds.
      *out_keys          = cudf_release_table_handle(std::move(result_keys), cudf_capture_exec_provenance(*exec));
      *out_total_columns = total;
      *out_values        = col_array;
    }

    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_groupby_scan_ex(cudf_table_view_handle keys,
                                   const cudf_column_view_handle* values,
                                   const cudf_groupby_scan_aggregation_handle* aggregations,
                                   const int32_t* agg_counts,
                                   int32_t num_requests,
                                   int32_t num_aggregations,
                                   cudf_null_policy_t keys_null_policy,
                                   int32_t keys_are_sorted,
                                   cudf_table_handle* out_keys,
                                   cudf_column_handle** out_values,
                                   int32_t* out_total_columns,
                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_keys, "out_keys is null");
  *out_keys = nullptr;
  CUDF_RUST_NULL_CHECK(out_values, "out_values is null");
  *out_values = nullptr;
  CUDF_RUST_NULL_CHECK(out_total_columns, "out_total_columns is null");
  *out_total_columns = 0;
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  if (num_requests < 0) {
    set_error("num_requests must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_aggregations < 0) {
    set_error("num_aggregations must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_requests > 0) {
    CUDF_RUST_NULL_CHECK(values, "values is null");
    CUDF_RUST_NULL_CHECK(agg_counts, "agg_counts is null");
  }
  if (num_aggregations > 0) { CUDF_RUST_NULL_CHECK(aggregations, "aggregations is null"); }
  cudf::null_policy null_policy;
  cudf_status_t status = to_cudf_null_policy(keys_null_policy, &null_policy, "keys_null_policy");
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* keys_view = static_cast<cudf::table_view*>(keys);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      cudf::groupby::groupby gb(*keys_view,
                                null_policy,
                                keys_are_sorted != 0 ? cudf::sorted::YES : cudf::sorted::NO);

      std::vector<cudf::groupby::scan_request> requests;
      requests.reserve(num_requests);

      int64_t required_aggregations = 0;
      for (int32_t i = 0; i < num_requests; ++i) {
        if (agg_counts[i] < 0) {
          set_error("agg_counts entries must be >= 0");
          return CUDF_ERROR_INVALID_OPERATION;
        }
        required_aggregations += agg_counts[i];
      }
      if (required_aggregations != num_aggregations) {
        set_error("num_aggregations must equal sum(agg_counts)");
        return CUDF_ERROR_INVALID_OPERATION;
      }

      int32_t agg_offset = 0;
      for (int32_t i = 0; i < num_requests; ++i) {
        CUDF_RUST_NULL_CHECK(values[i], "value column view handle is null");
        cudf::groupby::scan_request req;
        req.values = *static_cast<cudf::column_view*>(values[i]);
        for (int32_t j = 0; j < agg_counts[i]; ++j) {
          CUDF_RUST_NULL_CHECK(aggregations[agg_offset + j], "aggregation handle is null");
          auto* agg   = static_cast<cudf::groupby_scan_aggregation*>(aggregations[agg_offset + j]);
          auto cloned = agg->clone();
          req.aggregations.push_back(std::unique_ptr<cudf::groupby_scan_aggregation>(
            dynamic_cast<cudf::groupby_scan_aggregation*>(cloned.release())));
        }
        agg_offset += agg_counts[i];
        requests.push_back(std::move(req));
      }

      auto [result_keys, result_values] = gb.scan(requests, exec->stream, *exec->mr);

      int32_t total = 0;
      for (auto& [results] : result_values) {
        total += static_cast<int32_t>(results.size());
      }

      cudf_column_handle* col_array = nullptr;
      if (total > 0) {
        col_array = new (std::nothrow) cudf_column_handle[total];
        if (col_array == nullptr) {
          set_error("failed to allocate groupby scan result column handle array");
          return CUDF_ERROR_OUT_OF_MEMORY;
        }
      }
      int32_t idx = 0;
      for (auto& [results] : result_values) {
        for (auto& col : results) {
          col_array[idx++] = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
        }
      }

      *out_keys          = cudf_release_table_handle(std::move(result_keys), cudf_capture_exec_provenance(*exec));
      *out_total_columns = total;
      *out_values        = col_array;
    }

    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_groupby_shift_ex(cudf_table_view_handle keys,
                                    cudf_column_view_handle const* values,
                                    int32_t const* offsets,
                                    cudf_scalar_handle const* fill_values,
                                    int32_t num_value_columns,
                                    cudf_null_policy_t keys_null_policy,
                                    int32_t keys_are_sorted,
                                    cudf_table_handle* out_keys,
                                    cudf_table_handle* out_values,
                                    cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_keys, "out_keys is null");
  *out_keys = nullptr;
  CUDF_RUST_NULL_CHECK(out_values, "out_values is null");
  *out_values = nullptr;
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  if (num_value_columns < 0) {
    set_error("num_value_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_value_columns > 0) {
    CUDF_RUST_NULL_CHECK(values, "values is null");
    CUDF_RUST_NULL_CHECK(offsets, "offsets is null");
    CUDF_RUST_NULL_CHECK(fill_values, "fill_values is null");
  }

  cudf::null_policy null_policy;
  cudf_status_t status = to_cudf_null_policy(keys_null_policy, &null_policy, "keys_null_policy");
  if (status != CUDF_SUCCESS) { return status; }

  CUDF_RUST_TRY
  {
    auto* keys_view                 = static_cast<cudf::table_view*>(keys);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<cudf::column_view> value_columns;
    value_columns.reserve(static_cast<size_t>(num_value_columns));
    std::vector<cudf::size_type> offsets_host;
    offsets_host.reserve(static_cast<size_t>(num_value_columns));
    std::vector<std::reference_wrapper<cudf::scalar const>> fill_refs;
    fill_refs.reserve(static_cast<size_t>(num_value_columns));
    for (int32_t i = 0; i < num_value_columns; ++i) {
      CUDF_RUST_NULL_CHECK(values[i], "value column view handle is null");
      CUDF_RUST_NULL_CHECK(fill_values[i], "fill_values entry is null");
      value_columns.push_back(*static_cast<cudf::column_view*>(values[i]));
      offsets_host.push_back(static_cast<cudf::size_type>(offsets[i]));
      fill_refs.emplace_back(cudf_scalar_from_handle_const(fill_values[i]));
    }
    cudf::table_view values_table_view{value_columns};

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      cudf::groupby::groupby gb(*keys_view,
                                null_policy,
                                keys_are_sorted != 0 ? cudf::sorted::YES : cudf::sorted::NO);
      auto [result_keys, result_values] = gb.shift(
        values_table_view,
        cudf::host_span<cudf::size_type const>(offsets_host.data(), offsets_host.size()),
        fill_refs,
        exec->stream,
        *exec->mr);

      // Publish both handles only after both releases succeed so a host
      // allocation failure on the second handle does not leave Rust without
      // ownership of the first.
      cudf_table_handle keys_handle   = cudf_release_table_handle(std::move(result_keys), cudf_capture_exec_provenance(*exec));
      cudf_table_handle values_handle = nullptr;
      try {
        values_handle = cudf_release_table_handle(std::move(result_values), cudf_capture_exec_provenance(*exec));
      } catch (...) {
        cudf_table_destroy(keys_handle);
        throw;
      }
      *out_keys   = keys_handle;
      *out_values = values_handle;
    }

    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
