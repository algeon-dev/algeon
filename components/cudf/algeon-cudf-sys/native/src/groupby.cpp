#include "cudf_rust/groupby.h"

#include "execution_context_internal.hpp"
#include "handle_provenance.hpp"
#include "owning_handle.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace {
struct prepared_groupby_owner {
  cudf_handle_provenance provenance;
  std::unique_ptr<cudf::groupby::groupby> groupby;
};

struct groupby_column_handle_deleter {
  void operator()(cudf_column_handle handle) const noexcept { cudf_column_destroy(handle); }
};

using unique_groupby_column_handle =
  std::unique_ptr<std::remove_pointer_t<cudf_column_handle>, groupby_column_handle_deleter>;

struct groupby_table_handle_deleter {
  void operator()(cudf_table_handle handle) const noexcept { cudf_table_destroy(handle); }
};

using unique_groupby_table_handle =
  std::unique_ptr<std::remove_pointer_t<cudf_table_handle>, groupby_table_handle_deleter>;

cudf_status_t to_cudf_null_policy(cudf_null_policy_t input,
                                  cudf::null_policy* out,
                                  char const* param_name)
{
  switch (input) {
    case CUDF_NULL_POLICY_EXCLUDE: *out = cudf::null_policy::EXCLUDE; return CUDF_SUCCESS;
    case CUDF_NULL_POLICY_INCLUDE: *out = cudf::null_policy::INCLUDE; return CUDF_SUCCESS;
    default: set_error(std::string("invalid ") + param_name); return CUDF_ERROR_INVALID_OPERATION;
  }
}
}  // namespace

extern "C" {
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
    *out = reinterpret_cast<cudf_aggregation_handle>(agg.release());
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
    *out     = reinterpret_cast<cudf_aggregation_handle>(agg.release());
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
    *out     = reinterpret_cast<cudf_aggregation_handle>(agg.release());
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
    *out     = reinterpret_cast<cudf_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_aggregation_destroy(cudf_aggregation_handle handle)
{
  try {
    delete reinterpret_cast<cudf::groupby_aggregation*>(handle);
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
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
      default:
        set_error("unknown grouped scan aggregation kind");
        return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = reinterpret_cast<cudf_groupby_scan_aggregation_handle>(agg.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_groupby_scan_aggregation_destroy(cudf_groupby_scan_aggregation_handle handle)
{
  try {
    delete reinterpret_cast<cudf::groupby_scan_aggregation*>(handle);
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}

cudf_status_t cudf_groupby_aggregate_ex(cudf_table_view_handle keys,
                                        cudf_column_view_handle const* values,
                                        cudf_aggregation_handle const* aggregations,
                                        int32_t const* agg_counts,
                                        int32_t num_requests,
                                        int32_t num_aggregations,
                                        cudf_null_policy_t keys_null_policy,
                                        cudf_table_handle* out_keys,
                                        cudf_column_handle** out_values,
                                        int32_t* out_total_columns,
                                        cudf_exec_options_t const* options)
{
  return cudf_groupby_aggregate_with_keys_are_sorted_ex(keys,
                                                        values,
                                                        aggregations,
                                                        agg_counts,
                                                        num_requests,
                                                        num_aggregations,
                                                        keys_null_policy,
                                                        0,
                                                        out_keys,
                                                        out_values,
                                                        out_total_columns,
                                                        options);
}

static cudf_status_t cudf_groupby_aggregate_from_groupby(
  cudf::groupby::groupby& gb,
  std::vector<cudf::groupby::aggregation_request>& requests,
  cudf_table_handle* out_keys,
  cudf_column_handle** out_values,
  int32_t* out_total_columns,
  cudf_exec_context const& exec)
{
  auto [result_keys, result_values] = gb.aggregate(requests, exec.stream, *exec.mr);

  if (cudf::rust::test_support::take_groupby_negative_total_columns_failure()) {
    *out_keys          = cudf_release_table_handle(std::move(result_keys));
    *out_values        = nullptr;
    *out_total_columns = -1;
    return CUDF_SUCCESS;
  }

  int32_t total = 0;
  for (auto& [results] : result_values) {
    total += static_cast<int32_t>(results.size());
  }

  std::vector<unique_groupby_column_handle> published_columns;
  published_columns.reserve(static_cast<std::size_t>(total));
  for (auto& [results] : result_values) {
    for (auto& col : results) {
      published_columns.emplace_back(
        cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(exec)));
    }
  }
  unique_groupby_table_handle keys_handle{
    cudf_release_table_handle(std::move(result_keys), cudf_capture_exec_provenance(exec))};

  auto col_array = cudf_allocate_column_handle_array(static_cast<std::size_t>(total));
  if (total > 0 && col_array == nullptr) {
    set_error("failed to allocate groupby result column handle array");
    return CUDF_ERROR_OUT_OF_MEMORY;
  }

  int32_t idx = 0;
  for (auto& col : published_columns) {
    col_array.get()[idx++] = col.release();
  }

  // Publish outputs only after all work succeeds.
  *out_keys          = keys_handle.release();
  *out_total_columns = total;
  *out_values        = col_array.release();
  return CUDF_SUCCESS;
}

static cudf_status_t build_groupby_aggregation_requests(
  cudf_column_view_handle const* values,
  cudf_aggregation_handle const* aggregations,
  int32_t const* agg_counts,
  int32_t num_requests,
  int32_t num_aggregations,
  std::vector<cudf::groupby::aggregation_request>* out)
{
  out->clear();
  out->reserve(num_requests);

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
    req.values = *reinterpret_cast<cudf::column_view*>(values[i]);
    for (int32_t j = 0; j < agg_counts[i]; ++j) {
      CUDF_RUST_NULL_CHECK(aggregations[agg_offset + j], "aggregation handle is null");
      auto* agg   = reinterpret_cast<cudf::groupby_aggregation*>(aggregations[agg_offset + j]);
      auto cloned = agg->clone();
      req.aggregations.push_back(std::unique_ptr<cudf::groupby_aggregation>(
        dynamic_cast<cudf::groupby_aggregation*>(cloned.release())));
    }
    agg_offset += agg_counts[i];
    out->push_back(std::move(req));
  }
  return CUDF_SUCCESS;
}

static cudf_status_t validate_groupby_aggregate_args(cudf_column_view_handle const* values,
                                                     cudf_aggregation_handle const* aggregations,
                                                     int32_t const* agg_counts,
                                                     int32_t num_requests,
                                                     int32_t num_aggregations,
                                                     cudf_table_handle* out_keys,
                                                     cudf_column_handle** out_values,
                                                     int32_t* out_total_columns)
{
  CUDF_RUST_NULL_CHECK(out_keys, "out_keys is null");
  *out_keys = nullptr;
  CUDF_RUST_NULL_CHECK(out_values, "out_values is null");
  *out_values = nullptr;
  CUDF_RUST_NULL_CHECK(out_total_columns, "out_total_columns is null");
  *out_total_columns = 0;
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
  return CUDF_SUCCESS;
}

cudf_status_t cudf_prepared_groupby_create_ex(cudf_table_view_handle keys,
                                              cudf_null_policy_t keys_null_policy,
                                              int32_t keys_are_sorted,
                                              cudf_prepared_groupby_handle* out,
                                              cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  cudf::null_policy null_policy;
  cudf_status_t status = to_cudf_null_policy(keys_null_policy, &null_policy, "keys_null_policy");
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* keys_view   = reinterpret_cast<cudf::table_view*>(keys);
    auto owner        = std::make_unique<prepared_groupby_owner>();
    owner->provenance = cudf_capture_exec_provenance(*exec);
    {
      owner->groupby = std::make_unique<cudf::groupby::groupby>(
        *keys_view, null_policy, keys_are_sorted == 0 ? cudf::sorted::NO : cudf::sorted::YES);
    }
    *out = reinterpret_cast<cudf_prepared_groupby_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_prepared_groupby_destroy(cudf_prepared_groupby_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* owner     = reinterpret_cast<prepared_groupby_owner*>(handle);
  auto provenance = owner->provenance;
  // Ownership is consumed even when cleanup is rejected.
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_prepared_groupby_prepare_groups_ex(cudf_prepared_groupby_handle groupby,
                                                      int64_t* out_num_groups,
                                                      size_t* out_grouped_key_output_bytes,
                                                      int64_t* out_max_group_size,
                                                      cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_num_groups, "out_num_groups is null");
  CUDF_RUST_NULL_CHECK(out_grouped_key_output_bytes, "out_grouped_key_output_bytes is null");
  CUDF_RUST_NULL_CHECK(out_max_group_size, "out_max_group_size is null");
  *out_num_groups               = 0;
  *out_grouped_key_output_bytes = 0;
  *out_max_group_size           = 0;
  CUDF_RUST_NULL_CHECK(groupby, "groupby is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* prepared = reinterpret_cast<prepared_groupby_owner*>(groupby);
    {
      auto groups = prepared->groupby->get_groups({}, exec->stream, *exec->mr);
      if (groups.keys != nullptr) { *out_grouped_key_output_bytes = groups.keys->alloc_size(); }
      if (groups.offsets.empty()) {
        *out_num_groups = 0;
      } else {
        *out_num_groups                = static_cast<int64_t>(groups.offsets.size() - 1);
        cudf::size_type max_group_size = 0;
        for (std::size_t i = 1; i < groups.offsets.size(); ++i) {
          max_group_size = std::max(max_group_size, groups.offsets[i] - groups.offsets[i - 1]);
        }
        *out_max_group_size = static_cast<int64_t>(max_group_size);
      }
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_prepared_groupby_aggregate_ex(cudf_prepared_groupby_handle groupby,
                                                 cudf_column_view_handle const* values,
                                                 cudf_aggregation_handle const* aggregations,
                                                 int32_t const* agg_counts,
                                                 int32_t num_requests,
                                                 int32_t num_aggregations,
                                                 cudf_table_handle* out_keys,
                                                 cudf_column_handle** out_values,
                                                 int32_t* out_total_columns,
                                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(groupby, "groupby is null");
  cudf_status_t status = validate_groupby_aggregate_args(values,
                                                         aggregations,
                                                         agg_counts,
                                                         num_requests,
                                                         num_aggregations,
                                                         out_keys,
                                                         out_values,
                                                         out_total_columns);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<cudf::groupby::aggregation_request> requests;
    status = build_groupby_aggregation_requests(
      values, aggregations, agg_counts, num_requests, num_aggregations, &requests);
    if (status != CUDF_SUCCESS) { return status; }

    auto* prepared = reinterpret_cast<prepared_groupby_owner*>(groupby);
    {
      return cudf_groupby_aggregate_from_groupby(
        *prepared->groupby, requests, out_keys, out_values, out_total_columns, *exec);
    }
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_groupby_aggregate_with_keys_are_sorted_ex(
  cudf_table_view_handle keys,
  cudf_column_view_handle const* values,
  cudf_aggregation_handle const* aggregations,
  int32_t const* agg_counts,
  int32_t num_requests,
  int32_t num_aggregations,
  cudf_null_policy_t keys_null_policy,
  int32_t keys_are_sorted,
  cudf_table_handle* out_keys,
  cudf_column_handle** out_values,
  int32_t* out_total_columns,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  cudf_status_t status = validate_groupby_aggregate_args(values,
                                                         aggregations,
                                                         agg_counts,
                                                         num_requests,
                                                         num_aggregations,
                                                         out_keys,
                                                         out_values,
                                                         out_total_columns);
  if (status != CUDF_SUCCESS) { return status; }
  cudf::null_policy null_policy;
  status = to_cudf_null_policy(keys_null_policy, &null_policy, "keys_null_policy");
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<cudf::groupby::aggregation_request> requests;
    status = build_groupby_aggregation_requests(
      values, aggregations, agg_counts, num_requests, num_aggregations, &requests);
    if (status != CUDF_SUCCESS) { return status; }

    auto* keys_view = reinterpret_cast<cudf::table_view*>(keys);
    {
      cudf::groupby::groupby gb(
        *keys_view, null_policy, keys_are_sorted == 0 ? cudf::sorted::NO : cudf::sorted::YES);
      return cudf_groupby_aggregate_from_groupby(
        gb, requests, out_keys, out_values, out_total_columns, *exec);
    }
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_groupby_scan_ex(cudf_table_view_handle keys,
                                   cudf_column_view_handle const* values,
                                   cudf_groupby_scan_aggregation_handle const* aggregations,
                                   int32_t const* agg_counts,
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

    auto* keys_view = reinterpret_cast<cudf::table_view*>(keys);

    {
      cudf::groupby::groupby gb(
        *keys_view, null_policy, keys_are_sorted != 0 ? cudf::sorted::YES : cudf::sorted::NO);

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
        req.values = *reinterpret_cast<cudf::column_view*>(values[i]);
        for (int32_t j = 0; j < agg_counts[i]; ++j) {
          CUDF_RUST_NULL_CHECK(aggregations[agg_offset + j], "aggregation handle is null");
          auto* agg =
            reinterpret_cast<cudf::groupby_scan_aggregation*>(aggregations[agg_offset + j]);
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

      std::vector<unique_groupby_column_handle> published_columns;
      published_columns.reserve(static_cast<std::size_t>(total));
      for (auto& [results] : result_values) {
        for (auto& col : results) {
          published_columns.emplace_back(
            cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec)));
        }
      }
      unique_groupby_table_handle keys_handle{
        cudf_release_table_handle(std::move(result_keys), cudf_capture_exec_provenance(*exec))};

      auto col_array = cudf_allocate_column_handle_array(static_cast<std::size_t>(total));
      if (total > 0 && col_array == nullptr) {
        set_error("failed to allocate groupby scan result column handle array");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
      int32_t idx = 0;
      for (auto& col : published_columns) {
        col_array.get()[idx++] = col.release();
      }

      *out_keys          = keys_handle.release();
      *out_total_columns = total;
      *out_values        = col_array.release();
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
    auto* keys_view                 = reinterpret_cast<cudf::table_view*>(keys);
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
      value_columns.push_back(*reinterpret_cast<cudf::column_view*>(values[i]));
      offsets_host.push_back(static_cast<cudf::size_type>(offsets[i]));
      fill_refs.emplace_back(cudf_scalar_from_handle_const(fill_values[i]));
    }
    cudf::table_view values_table_view{value_columns};

    {
      cudf::groupby::groupby gb(
        *keys_view, null_policy, keys_are_sorted != 0 ? cudf::sorted::YES : cudf::sorted::NO);
      auto [result_keys, result_values] =
        gb.shift(values_table_view,
                 std::span<cudf::size_type const>{offsets_host.data(), offsets_host.size()},
                 fill_refs,
                 exec->stream,
                 *exec->mr);

      // Publish both handles only after both releases succeed so a host
      // allocation failure on the second handle does not leave Rust without
      // ownership of the first.
      cudf_table_handle keys_handle =
        cudf_release_table_handle(std::move(result_keys), cudf_capture_exec_provenance(*exec));
      cudf_table_handle values_handle = nullptr;
      try {
        values_handle =
          cudf_release_table_handle(std::move(result_values), cudf_capture_exec_provenance(*exec));
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
