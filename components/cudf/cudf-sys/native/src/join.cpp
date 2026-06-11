#include "join.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>
#include <vector>

namespace {

/// Concatenate key and payload views into one table view.
cudf::table_view concat_views(const cudf::table_view& keys, const cudf::table_view& payload)
{
  std::vector<cudf::column_view> cols;
  cols.reserve(keys.num_columns() + payload.num_columns());
  for (cudf::size_type i = 0; i < keys.num_columns(); ++i) {
    cols.push_back(keys.column(i));
  }
  for (cudf::size_type i = 0; i < payload.num_columns(); ++i) {
    cols.push_back(payload.column(i));
  }
  return cudf::table_view(cols);
}

/// Materialize a table given a gather map (index column).
std::unique_ptr<cudf::table> gather_table(
  const cudf::table_view& source,
  const std::unique_ptr<rmm::device_uvector<cudf::size_type>>& map,
  cudf::out_of_bounds_policy oob_policy,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Wrap the device_uvector as a column_view for gather.
  cudf::column_view map_view(cudf::data_type{cudf::type_id::INT32},
                             static_cast<cudf::size_type>(map->size()),
                             map->data(),
                             nullptr,
                             0);

  return cudf::gather(source, map_view, oob_policy, stream, mr);
}

cudf_status_t to_null_equality(cudf_null_equality_t input, cudf::null_equality* out)
{
  switch (input) {
    case CUDF_NULL_EQUALITY_EQUAL: *out = cudf::null_equality::EQUAL; return CUDF_SUCCESS;
    case CUDF_NULL_EQUALITY_UNEQUAL: *out = cudf::null_equality::UNEQUAL; return CUDF_SUCCESS;
    default: set_error("invalid null_equality"); return CUDF_ERROR_INVALID_OPERATION;
  }
}

}  // namespace

extern "C" {

cudf_status_t cudf_inner_join(cudf_table_view_handle left_keys,
                              cudf_table_view_handle right_keys,
                              cudf_table_view_handle left_payload,
                              cudf_table_view_handle right_payload,
                              cudf_null_equality_t null_equality,
                              cudf_table_handle* out_left,
                              cudf_table_handle* out_right)
{
  return cudf_inner_join_ex(left_keys,
                            right_keys,
                            left_payload,
                            right_payload,
                            null_equality,
                            out_left,
                            out_right,
                            nullptr);
}

cudf_status_t cudf_inner_join_ex(cudf_table_view_handle left_keys,
                                 cudf_table_view_handle right_keys,
                                 cudf_table_view_handle left_payload,
                                 cudf_table_view_handle right_payload,
                                 cudf_null_equality_t null_equality,
                                 cudf_table_handle* out_left,
                                 cudf_table_handle* out_right,
                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left, "out_left is null");
  *out_left = nullptr;
  CUDF_RUST_NULL_CHECK(out_right, "out_right is null");
  *out_right = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  CUDF_RUST_NULL_CHECK(left_payload, "left_payload is null");
  CUDF_RUST_NULL_CHECK(right_payload, "right_payload is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    auto [left_indices, right_indices] = cudf::inner_join(*lk, *rk, null_eq, exec.stream, exec.mr);

    auto left_full  = concat_views(*lk, *lp);
    auto right_full = concat_views(*rk, *rp);

    auto left_result = gather_table(
      left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec.stream, exec.mr);
    auto right_result = gather_table(
      right_full, right_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec.stream, exec.mr);

    *out_left  = cudf_release_table_handle(std::move(left_result));
    *out_right = cudf_release_table_handle(std::move(right_result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_join(cudf_table_view_handle left_keys,
                             cudf_table_view_handle right_keys,
                             cudf_table_view_handle left_payload,
                             cudf_table_view_handle right_payload,
                             cudf_null_equality_t null_equality,
                             cudf_table_handle* out_left,
                             cudf_table_handle* out_right)
{
  return cudf_left_join_ex(left_keys,
                           right_keys,
                           left_payload,
                           right_payload,
                           null_equality,
                           out_left,
                           out_right,
                           nullptr);
}

cudf_status_t cudf_left_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left, "out_left is null");
  *out_left = nullptr;
  CUDF_RUST_NULL_CHECK(out_right, "out_right is null");
  *out_right = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  CUDF_RUST_NULL_CHECK(left_payload, "left_payload is null");
  CUDF_RUST_NULL_CHECK(right_payload, "right_payload is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    auto [left_indices, right_indices] = cudf::left_join(*lk, *rk, null_eq, exec.stream, exec.mr);

    auto left_full  = concat_views(*lk, *lp);
    auto right_full = concat_views(*rk, *rp);

    auto left_result = gather_table(
      left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec.stream, exec.mr);
    auto right_result = gather_table(
      right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec.stream, exec.mr);

    *out_left  = cudf_release_table_handle(std::move(left_result));
    *out_right = cudf_release_table_handle(std::move(right_result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_full_join(cudf_table_view_handle left_keys,
                             cudf_table_view_handle right_keys,
                             cudf_table_view_handle left_payload,
                             cudf_table_view_handle right_payload,
                             cudf_null_equality_t null_equality,
                             cudf_table_handle* out_left,
                             cudf_table_handle* out_right)
{
  return cudf_full_join_ex(left_keys,
                           right_keys,
                           left_payload,
                           right_payload,
                           null_equality,
                           out_left,
                           out_right,
                           nullptr);
}

cudf_status_t cudf_full_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left, "out_left is null");
  *out_left = nullptr;
  CUDF_RUST_NULL_CHECK(out_right, "out_right is null");
  *out_right = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  CUDF_RUST_NULL_CHECK(left_payload, "left_payload is null");
  CUDF_RUST_NULL_CHECK(right_payload, "right_payload is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    auto [left_indices, right_indices] = cudf::full_join(*lk, *rk, null_eq, exec.stream, exec.mr);

    auto left_full  = concat_views(*lk, *lp);
    auto right_full = concat_views(*rk, *rp);

    auto left_result = gather_table(
      left_full, left_indices, cudf::out_of_bounds_policy::NULLIFY, exec.stream, exec.mr);
    auto right_result = gather_table(
      right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec.stream, exec.mr);

    *out_left  = cudf_release_table_handle(std::move(left_result));
    *out_right = cudf_release_table_handle(std::move(right_result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
