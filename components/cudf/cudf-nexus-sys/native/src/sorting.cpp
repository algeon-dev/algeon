#include "cudf_rust/sorting.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <vector>

static_assert(sizeof(uint64_t) >= sizeof(std::size_t));

namespace {
/// Build a vector of cudf::order from the C enum array.
cudf_status_t to_order_vec(cudf_order_t const *orders, int32_t n,
                           std::vector<cudf::order> *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  out->resize(n);
  for (int32_t i = 0; i < n; ++i) {
    switch (orders[i]) {
    case CUDF_ORDER_ASCENDING:
      (*out)[i] = cudf::order::ASCENDING;
      break;
    case CUDF_ORDER_DESCENDING:
      (*out)[i] = cudf::order::DESCENDING;
      break;
    default:
      set_error("invalid column_order value");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }
  return CUDF_SUCCESS;
}

/// Build a vector of cudf::null_order from the C enum array.
cudf_status_t to_null_order_vec(cudf_null_order_t const *orders, int32_t n,
                                std::vector<cudf::null_order> *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  out->resize(n);
  for (int32_t i = 0; i < n; ++i) {
    switch (orders[i]) {
    case CUDF_NULL_ORDER_BEFORE:
      (*out)[i] = cudf::null_order::BEFORE;
      break;
    case CUDF_NULL_ORDER_AFTER:
      (*out)[i] = cudf::null_order::AFTER;
      break;
    default:
      set_error("invalid null_precedence value");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }
  return CUDF_SUCCESS;
}

cudf_status_t is_sorted_impl(cudf_table_view_handle input,
                             cudf_order_t const *column_order,
                             cudf_null_order_t const *null_precedence,
                             int32_t num_keys, int32_t *out,
                             cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = reinterpret_cast<cudf::table_view *>(input);

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      cudf_status_t status = to_order_vec(column_order, num_keys, &orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }
    if (null_precedence != nullptr && num_keys > 0) {
      cudf_status_t status =
          to_null_order_vec(null_precedence, num_keys, &null_orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }

    {
      *out = cudf::is_sorted(*view, orders, null_orders, exec->stream) ? 1 : 0;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // namespace

extern "C" {
cudf_status_t cudf_sorted_order_radix_preflight(
    int64_t num_rows, cudf_type_id_t key_type, int64_t null_count,
    int32_t key_count, int32_t stable, cudf_order_t key_order, int32_t device,
    cudf_sorted_order_radix_preflight_result_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  if (stable != 0 && stable != 1) {
    set_error("stable must be 0 or 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  cudf::order const cudf_order = [&] {
    switch (key_order) {
    case CUDF_ORDER_ASCENDING:
      return cudf::order::ASCENDING;
    case CUDF_ORDER_DESCENDING:
      return cudf::order::DESCENDING;
    default:
      set_error("invalid key_order value");
      return cudf::order::ASCENDING;
    }
  }();
  if (key_order != CUDF_ORDER_ASCENDING && key_order != CUDF_ORDER_DESCENDING) {
    return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY {
    auto const result = cudf::sorted_order_radix_preflight(
        num_rows, cudf::data_type(to_cudf_type_id(key_type)), null_count,
        key_count, stable != 0, cudf_order, device);
    out->key_input_bytes = static_cast<uint64_t>(result.key_input_bytes);
    out->key_output_bytes = static_cast<uint64_t>(result.key_output_bytes);
    out->sequence_bytes = static_cast<uint64_t>(result.sequence_bytes);
    out->temporary_workspace_bytes =
        static_cast<uint64_t>(result.temporary_workspace_bytes);
    out->retained_order_bytes =
        static_cast<uint64_t>(result.retained_order_bytes);
    out->active_phase_peak_bytes =
        static_cast<uint64_t>(result.active_phase_peak_bytes);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sorted_order_lexicographic_preflight(
    int64_t num_rows, cudf_type_id_t const *key_types,
    int64_t const *null_counts, cudf_order_t const *key_orders,
    cudf_null_order_t const *null_precedence, int32_t key_count, int32_t stable,
    int32_t device, cudf_sorted_order_lexicographic_preflight_result_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  if (key_count < 1) {
    set_error("key_count must be at least one");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(key_types, "key_types is null");
  CUDF_RUST_NULL_CHECK(null_counts, "null_counts is null");
  CUDF_RUST_NULL_CHECK(key_orders, "key_orders is null");
  CUDF_RUST_NULL_CHECK(null_precedence, "null_precedence is null");
  if (stable != 0 && stable != 1) {
    set_error("stable must be 0 or 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  std::vector<cudf::data_type> types;
  types.reserve(static_cast<std::size_t>(key_count));
  for (int32_t index = 0; index < key_count; ++index) {
    types.emplace_back(to_cudf_type_id(key_types[index]));
  }
  std::vector<std::int64_t> nulls(null_counts, null_counts + key_count);
  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> null_orders;
  auto status = to_order_vec(key_orders, key_count, &orders);
  if (status != CUDF_SUCCESS) {
    return status;
  }
  status = to_null_order_vec(null_precedence, key_count, &null_orders);
  if (status != CUDF_SUCCESS) {
    return status;
  }

  CUDF_RUST_TRY {
    auto const result = cudf::sorted_order_lexicographic_preflight(
        num_rows, types, nulls, stable != 0, orders, null_orders, device);
    out->comparator_state_bytes =
        static_cast<uint64_t>(result.comparator_state_bytes);
    out->temporary_workspace_bytes =
        static_cast<uint64_t>(result.temporary_workspace_bytes);
    out->retained_order_bytes =
        static_cast<uint64_t>(result.retained_order_bytes);
    out->active_phase_peak_bytes =
        static_cast<uint64_t>(result.active_phase_peak_bytes);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sort_ex(cudf_table_view_handle input,
                           cudf_order_t const *column_order,
                           cudf_null_order_t const *null_precedence,
                           int32_t num_keys, cudf_table_handle *out,
                           cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = reinterpret_cast<cudf::table_view *>(input);

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      cudf_status_t status = to_order_vec(column_order, num_keys, &orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }
    if (null_precedence != nullptr && num_keys > 0) {
      cudf_status_t status =
          to_null_order_vec(null_precedence, num_keys, &null_orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }

    {
      auto result =
          cudf::sort(*view, orders, null_orders, exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stable_sort_ex(cudf_table_view_handle input,
                                  cudf_order_t const *column_order,
                                  cudf_null_order_t const *null_precedence,
                                  int32_t num_keys, cudf_table_handle *out,
                                  cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = reinterpret_cast<cudf::table_view *>(input);

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      cudf_status_t status = to_order_vec(column_order, num_keys, &orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }
    if (null_precedence != nullptr && num_keys > 0) {
      cudf_status_t status =
          to_null_order_vec(null_precedence, num_keys, &null_orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }

    {
      auto result = cudf::stable_sort(*view, orders, null_orders, exec->stream,
                                      *exec->mr);
      *out = cudf_release_table_handle(std::move(result),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sort_by_key_ex(cudf_table_view_handle values,
                                  cudf_table_view_handle keys,
                                  cudf_order_t const *column_order,
                                  cudf_null_order_t const *null_precedence,
                                  int32_t num_key_columns,
                                  cudf_table_handle *out,
                                  cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(values, "values is null");
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  if (num_key_columns < 0) {
    set_error("num_key_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *vals = reinterpret_cast<cudf::table_view *>(values);
    auto *ks = reinterpret_cast<cudf::table_view *>(keys);

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_key_columns > 0) {
      cudf_status_t status =
          to_order_vec(column_order, num_key_columns, &orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }
    if (null_precedence != nullptr && num_key_columns > 0) {
      cudf_status_t status =
          to_null_order_vec(null_precedence, num_key_columns, &null_orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }

    {
      auto result = cudf::sort_by_key(*vals, *ks, orders, null_orders,
                                      exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stable_sort_by_key_ex(
    cudf_table_view_handle values, cudf_table_view_handle keys,
    cudf_order_t const *column_order, cudf_null_order_t const *null_precedence,
    int32_t num_key_columns, cudf_table_handle *out,
    cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(values, "values is null");
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  if (num_key_columns < 0) {
    set_error("num_key_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *vals = reinterpret_cast<cudf::table_view *>(values);
    auto *ks = reinterpret_cast<cudf::table_view *>(keys);

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_key_columns > 0) {
      cudf_status_t status =
          to_order_vec(column_order, num_key_columns, &orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }
    if (null_precedence != nullptr && num_key_columns > 0) {
      cudf_status_t status =
          to_null_order_vec(null_precedence, num_key_columns, &null_orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }

    {
      auto result = cudf::stable_sort_by_key(*vals, *ks, orders, null_orders,
                                             exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sorted_order_ex(cudf_table_view_handle input,
                                   cudf_order_t const *column_order,
                                   cudf_null_order_t const *null_precedence,
                                   int32_t num_keys, cudf_column_handle *out,
                                   cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = reinterpret_cast<cudf::table_view *>(input);

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      cudf_status_t status = to_order_vec(column_order, num_keys, &orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }
    if (null_precedence != nullptr && num_keys > 0) {
      cudf_status_t status =
          to_null_order_vec(null_precedence, num_keys, &null_orders);
      if (status != CUDF_SUCCESS) {
        return status;
      }
    }

    {
      auto result = cudf::sorted_order(*view, orders, null_orders, exec->stream,
                                       *exec->mr);
      *out = cudf_release_column_handle(std::move(result),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_sorted_ex(cudf_table_view_handle input,
                                cudf_order_t const *column_order,
                                cudf_null_order_t const *null_precedence,
                                int32_t num_keys, int32_t *out,
                                cudf_exec_options_t const *options) {
  return is_sorted_impl(input, column_order, null_precedence, num_keys, out,
                        options);
}
} // extern "C"
