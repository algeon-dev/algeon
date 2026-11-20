#include "cudf_rust/sorting.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

static_assert(sizeof(uint64_t) >= sizeof(std::size_t));

namespace {
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

std::unique_ptr<cudf::column>
constant_bool_column(cudf::size_type rows, bool value,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) {
  auto const fill = cudf::numeric_scalar<bool>(value, true, stream);
  return cudf::make_column_from_scalar(fill, rows, stream, mr);
}

bool scalar_is_nan(cudf::scalar const &value, rmm::cuda_stream_view stream) {
  switch (value.type().id()) {
  case cudf::type_id::FLOAT32:
    return std::isnan(
        static_cast<cudf::numeric_scalar<float> const &>(value).value(stream));
  case cudf::type_id::FLOAT64:
    return std::isnan(
        static_cast<cudf::numeric_scalar<double> const &>(value).value(stream));
  default:
    return false;
  }
}

// Rows whose non-null value sorts no later than `boundary` under `order`, with
// cuDF's sort semantics for floating point: every NaN equals every other NaN
// and sorts after all other values, and -0.0 equals 0.0. Null rows are null or
// false.
std::unique_ptr<cudf::column>
not_after_boundary(cudf::column_view const &input,
                   cudf::scalar const &boundary, cudf::order order,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr) {
  auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};
  bool const descending = order == cudf::order::DESCENDING;
  auto const comparison = descending ? cudf::binary_operator::GREATER_EQUAL
                                     : cudf::binary_operator::LESS_EQUAL;
  if (!cudf::is_floating_point(input.type())) {
    return cudf::binary_operation(input, boundary, comparison, bool_type,
                                  stream, mr);
  }
  // IEEE comparisons are false against NaN, so the NaN rows are placed
  // explicitly: they lead a descending order and trail an ascending one.
  if (scalar_is_nan(boundary, stream)) {
    return descending ? cudf::is_nan(input, stream, mr)
                      : cudf::is_valid(input, stream, mr);
  }
  auto ranked = cudf::binary_operation(input, boundary, comparison, bool_type,
                                       stream, mr);
  if (!descending) {
    return ranked;
  }
  auto const nan_rows = cudf::is_nan(input, stream, mr);
  return cudf::binary_operation(ranked->view(), nan_rows->view(),
                                cudf::binary_operator::LOGICAL_OR, bool_type,
                                stream, mr);
}

// See `cudf_top_k_with_ties_mask_ex` in sorting.h for the contract.
// `null_order` is cuDF's: BEFORE means a null compares below every value.
std::unique_ptr<cudf::column>
top_k_with_ties_mask(cudf::column_view const &input, cudf::size_type k,
                     cudf::order order, cudf::null_order null_order,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  CUDF_EXPECTS(cudf::is_fixed_width(input.type()),
               "top_k_with_ties_mask requires a fixed-width column",
               cudf::data_type_error);
  cudf::size_type const rows = input.size();
  cudf::size_type const null_rows = input.null_count();
  cudf::size_type const valid_rows = rows - null_rows;
  bool const nulls_lead =
      (order == cudf::order::ASCENDING) == (null_order == cudf::null_order::BEFORE);
  if (k >= rows) {
    return constant_bool_column(rows, true, stream, mr);
  }
  if (k == 0) {
    return constant_bool_column(rows, false, stream, mr);
  }
  // Leading nulls tie with one another, so k of them keep every null row.
  if (nulls_lead && null_rows >= k) {
    return cudf::is_null(input, stream, mr);
  }
  cudf::size_type const valid_k = nulls_lead ? k - null_rows : k;
  // Trailing nulls are reached only after every valid row.
  if (valid_k > valid_rows) {
    return constant_bool_column(rows, true, stream, mr);
  }

  // Intermediates are allocated from the call's resource as well; shim
  // operations never read the ambient current resource.
  auto valid_storage = std::unique_ptr<cudf::table>{};
  auto valid = input;
  if (null_rows > 0) {
    valid_storage =
        cudf::drop_nulls(cudf::table_view{{input}}, {0}, stream, mr);
    valid = valid_storage->get_column(0).view();
  }
  // `top_k` returns its k values in no particular order; sorting those k
  // values with the full sort's comparator puts the boundary last.
  auto const best = cudf::top_k(valid, valid_k, order, stream, mr);
  auto const best_sorted =
      cudf::sort(cudf::table_view{{best->view()}}, {order}, {}, stream, mr);
  auto const boundary = cudf::get_element(best_sorted->get_column(0).view(),
                                          valid_k - 1, stream, mr);
  auto ranked = not_after_boundary(input, *boundary, order, stream, mr);
  if (null_rows == 0) {
    return ranked;
  }
  // Leading nulls all survive here (fewer than k of them) and trailing nulls
  // none (k valid rows exist). The null-aware operators turn a null `ranked`
  // row into the null-row answer: (null || true) and (null && false).
  auto const null_rows_mask = nulls_lead ? cudf::is_null(input, stream, mr)
                                         : cudf::is_valid(input, stream, mr);
  return cudf::binary_operation(
      ranked->view(), null_rows_mask->view(),
      nulls_lead ? cudf::binary_operator::NULL_LOGICAL_OR
                 : cudf::binary_operator::NULL_LOGICAL_AND,
      cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
}
} // namespace

extern "C" {
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
                                   int32_t num_keys, int32_t stable, cudf_column_handle *out,
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
      auto result = stable != 0
                        ? cudf::stable_sorted_order(*view, orders, null_orders,
                                                    exec->stream, *exec->mr)
                        : cudf::sorted_order(*view, orders, null_orders,
                                             exec->stream, *exec->mr);
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

cudf_status_t cudf_top_k_with_ties_mask_ex(cudf_column_view_handle input,
                                           int32_t k, cudf_order_t order,
                                           cudf_null_order_t null_order,
                                           cudf_column_handle *out,
                                           cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    std::vector<cudf::order> orders;
    cudf_status_t status = to_order_vec(&order, 1, &orders);
    if (status != CUDF_SUCCESS) {
      return status;
    }
    std::vector<cudf::null_order> null_orders;
    status = to_null_order_vec(&null_order, 1, &null_orders);
    if (status != CUDF_SUCCESS) {
      return status;
    }

    auto *view = reinterpret_cast<cudf::column_view *>(input);
    {
      auto result = top_k_with_ties_mask(*view, k, orders[0], null_orders[0],
                                         exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // extern "C"
