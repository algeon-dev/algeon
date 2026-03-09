#include "sorting.h"
#include "shim_common.h"

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <vector>

namespace {

/// Build a vector of cudf::order from the C enum array.
std::vector<cudf::order> to_order_vec(const cudf_order_t* orders, int32_t n) {
  std::vector<cudf::order> result(n);
  for (int32_t i = 0; i < n; ++i) {
    result[i] = (orders[i] == CUDF_ORDER_DESCENDING)
                    ? cudf::order::DESCENDING
                    : cudf::order::ASCENDING;
  }
  return result;
}

/// Build a vector of cudf::null_order from the C enum array.
std::vector<cudf::null_order> to_null_order_vec(const cudf_null_order_t* orders,
                                                int32_t n) {
  std::vector<cudf::null_order> result(n);
  for (int32_t i = 0; i < n; ++i) {
    result[i] = (orders[i] == CUDF_NULL_ORDER_BEFORE)
                    ? cudf::null_order::BEFORE
                    : cudf::null_order::AFTER;
  }
  return result;
}

}  // namespace

extern "C" {

cudf_status_t cudf_sort(cudf_table_view_handle input,
                        const cudf_order_t* column_order,
                        const cudf_null_order_t* null_precedence,
                        int32_t num_keys,
                        cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      orders = to_order_vec(column_order, num_keys);
    }
    if (null_precedence != nullptr && num_keys > 0) {
      null_orders = to_null_order_vec(null_precedence, num_keys);
    }

    auto result = cudf::sort(*view, orders, null_orders, stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stable_sort(cudf_table_view_handle input,
                               const cudf_order_t* column_order,
                               const cudf_null_order_t* null_precedence,
                               int32_t num_keys,
                               cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      orders = to_order_vec(column_order, num_keys);
    }
    if (null_precedence != nullptr && num_keys > 0) {
      null_orders = to_null_order_vec(null_precedence, num_keys);
    }

    auto result = cudf::stable_sort(*view, orders, null_orders, stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sort_by_key(cudf_table_view_handle values,
                               cudf_table_view_handle keys,
                               const cudf_order_t* column_order,
                               const cudf_null_order_t* null_precedence,
                               int32_t num_key_columns,
                               cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(values, "values is null");
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_key_columns < 0) {
    set_error("num_key_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* vals = static_cast<cudf::table_view*>(values);
    auto* ks = static_cast<cudf::table_view*>(keys);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_key_columns > 0) {
      orders = to_order_vec(column_order, num_key_columns);
    }
    if (null_precedence != nullptr && num_key_columns > 0) {
      null_orders = to_null_order_vec(null_precedence, num_key_columns);
    }

    auto result =
        cudf::sort_by_key(*vals, *ks, orders, null_orders, stream, mr);
    *out = static_cast<cudf_table_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_sorted_order(cudf_table_view_handle input,
                                const cudf_order_t* column_order,
                                const cudf_null_order_t* null_precedence,
                                int32_t num_keys,
                                cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      orders = to_order_vec(column_order, num_keys);
    }
    if (null_precedence != nullptr && num_keys > 0) {
      null_orders = to_null_order_vec(null_precedence, num_keys);
    }

    auto result = cudf::sorted_order(*view, orders, null_orders, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_sorted(cudf_table_view_handle input,
                             const cudf_order_t* column_order,
                             const cudf_null_order_t* null_precedence,
                             int32_t num_keys,
                             int32_t* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::table_view*>(input);

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    if (column_order != nullptr && num_keys > 0) {
      orders = to_order_vec(column_order, num_keys);
    }
    if (null_precedence != nullptr && num_keys > 0) {
      null_orders = to_null_order_vec(null_precedence, num_keys);
    }

    *out = cudf::is_sorted(*view, orders, null_orders) ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
