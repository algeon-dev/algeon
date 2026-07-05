#include "sorting.h"

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

namespace {
/// Build a vector of cudf::order from the C enum array.
cudf_status_t to_order_vec(const cudf_order_t *orders, int32_t n,
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
cudf_status_t to_null_order_vec(const cudf_null_order_t *orders, int32_t n,
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

    auto *view = static_cast<cudf::table_view *>(input);

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
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      *out = cudf::is_sorted(*view, orders, null_orders, exec->stream) ? 1 : 0;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // namespace

extern "C" {
cudf_status_t cudf_sort_ex(cudf_table_view_handle input,
                           const cudf_order_t *column_order,
                           const cudf_null_order_t *null_precedence,
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

    auto *view = static_cast<cudf::table_view *>(input);

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
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result =
          cudf::sort(*view, orders, null_orders, exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stable_sort_ex(cudf_table_view_handle input,
                                  const cudf_order_t *column_order,
                                  const cudf_null_order_t *null_precedence,
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

    auto *view = static_cast<cudf::table_view *>(input);

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
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result =
          cudf::stable_sort(*view, orders, null_orders, exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sort_by_key_ex(cudf_table_view_handle values,
                                  cudf_table_view_handle keys,
                                  const cudf_order_t *column_order,
                                  const cudf_null_order_t *null_precedence,
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

    auto *vals = static_cast<cudf::table_view *>(values);
    auto *ks = static_cast<cudf::table_view *>(keys);

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
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = cudf::sort_by_key(*vals, *ks, orders, null_orders,
                                      exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stable_sort_by_key_ex(
    cudf_table_view_handle values, cudf_table_view_handle keys,
    const cudf_order_t *column_order, const cudf_null_order_t *null_precedence,
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

    auto *vals = static_cast<cudf::table_view *>(values);
    auto *ks = static_cast<cudf::table_view *>(keys);

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
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = cudf::stable_sort_by_key(*vals, *ks, orders, null_orders,
                                             exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sorted_order_ex(cudf_table_view_handle input,
                                   const cudf_order_t *column_order,
                                   const cudf_null_order_t *null_precedence,
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

    auto *view = static_cast<cudf::table_view *>(input);

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
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = cudf::sorted_order(*view, orders, null_orders, exec->stream,
                                       *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_sorted_ex(cudf_table_view_handle input,
                                const cudf_order_t *column_order,
                                const cudf_null_order_t *null_precedence,
                                int32_t num_keys, int32_t *out,
                                cudf_exec_options_t const *options) {
  return is_sorted_impl(input, column_order, null_precedence, num_keys, out,
                        options);
}
} // extern "C"
