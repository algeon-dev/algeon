#include "join.h"

#include "ast_handle.hpp"
#include "execution_context_internal.hpp"
#include "handle_provenance.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
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

cudf_status_t to_nullable_join(cudf_nullable_join_t input, cudf::nullable_join* out)
{
  switch (input) {
    case CUDF_NULLABLE_JOIN_YES: *out = cudf::nullable_join::YES; return CUDF_SUCCESS;
    case CUDF_NULLABLE_JOIN_NO: *out = cudf::nullable_join::NO; return CUDF_SUCCESS;
    default: set_error("invalid nullable_join"); return CUDF_ERROR_INVALID_OPERATION;
  }
}

cudf_status_t to_join_kind(cudf_join_kind_t input, cudf::join_kind* out)
{
  switch (input) {
    case CUDF_JOIN_KIND_INNER_JOIN: *out = cudf::join_kind::INNER_JOIN; return CUDF_SUCCESS;
    case CUDF_JOIN_KIND_LEFT_JOIN: *out = cudf::join_kind::LEFT_JOIN; return CUDF_SUCCESS;
    case CUDF_JOIN_KIND_FULL_JOIN: *out = cudf::join_kind::FULL_JOIN; return CUDF_SUCCESS;
    case CUDF_JOIN_KIND_LEFT_SEMI_JOIN: *out = cudf::join_kind::LEFT_SEMI_JOIN; return CUDF_SUCCESS;
    case CUDF_JOIN_KIND_LEFT_ANTI_JOIN: *out = cudf::join_kind::LEFT_ANTI_JOIN; return CUDF_SUCCESS;
    default: set_error("invalid join_kind"); return CUDF_ERROR_INVALID_OPERATION;
  }
}

std::unique_ptr<cudf::column> release_join_index_column(
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices)
{
  return std::make_unique<cudf::column>(std::move(*indices), rmm::device_buffer{}, 0);
}

using join_index_pair = std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                                  std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;

enum class pair_join_kind { inner, left, full };

enum class single_side_join_kind { semi, anti };

enum class hash_join_probe_kind { inner, left, full };

struct distinct_hash_join_owner {
  std::unique_ptr<cudf::distinct_hash_join> join;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  int32_t device                          = 0;
};

struct hash_join_owner {
  std::shared_ptr<void> upstream_mr_lifetime;
  std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>
    observed_mr;
  std::unique_ptr<cudf::hash_join> join;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  std::size_t device_allocated_size_bytes = 0;
  int32_t device                          = 0;
};

struct column_handle_deleter {
  void operator()(void* handle) const noexcept { cudf_column_destroy(handle); }
};

using unique_column_handle = std::unique_ptr<void, column_handle_deleter>;

unique_column_handle make_unique_column_handle(std::unique_ptr<cudf::column> column,
                                               cudf_handle_provenance const& provenance)
{
  return unique_column_handle{cudf_release_column_handle(std::move(column), provenance)};
}

void release_join_index_pair(join_index_pair indices,
                             cudf_column_handle* out_left_indices,
                             cudf_column_handle* out_right_indices,
                             cudf_handle_provenance const& provenance)
{
  auto left_column  = release_join_index_column(std::move(indices.first));
  auto right_column = release_join_index_column(std::move(indices.second));

  auto left_handle  = make_unique_column_handle(std::move(left_column), provenance);
  auto right_handle = make_unique_column_handle(std::move(right_column), provenance);

  *out_left_indices  = left_handle.release();
  *out_right_indices = right_handle.release();
}

cudf_status_t conditional_pair_join_indices_impl(cudf_table_view_handle left,
                                                 cudf_table_view_handle right,
                                                 cudf_ast_expression_handle predicate,
                                                 cudf_column_handle* out_left_indices,
                                                 cudf_column_handle* out_right_indices,
                                                 cudf_exec_options_t const* options,
                                                 pair_join_kind kind)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left, "left is null");
  CUDF_RUST_NULL_CHECK(right, "right is null");
  CUDF_RUST_NULL_CHECK(predicate, "predicate is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view  = static_cast<cudf::table_view*>(left);
    auto* right_view = static_cast<cudf::table_view*>(right);
    auto const& pred = cudf_ast_expression_from_handle(predicate);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      join_index_pair indices;
      switch (kind) {
        case pair_join_kind::inner:
          indices =
            cudf::conditional_inner_join(*left_view, *right_view, pred, {}, exec.stream, exec.mr);
          break;
        case pair_join_kind::left:
          indices =
            cudf::conditional_left_join(*left_view, *right_view, pred, {}, exec.stream, exec.mr);
          break;
        case pair_join_kind::full:
          indices =
            cudf::conditional_full_join(*left_view, *right_view, pred, exec.stream, exec.mr);
          break;
      }
      release_join_index_pair(std::move(indices),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t conditional_single_side_join_indices_impl(cudf_table_view_handle left,
                                                        cudf_table_view_handle right,
                                                        cudf_ast_expression_handle predicate,
                                                        cudf_column_handle* out_indices,
                                                        cudf_exec_options_t const* options,
                                                        single_side_join_kind kind)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left, "left is null");
  CUDF_RUST_NULL_CHECK(right, "right is null");
  CUDF_RUST_NULL_CHECK(predicate, "predicate is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view  = static_cast<cudf::table_view*>(left);
    auto* right_view = static_cast<cudf::table_view*>(right);
    auto const& pred = cudf_ast_expression_from_handle(predicate);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices;
      switch (kind) {
        case single_side_join_kind::semi:
          indices = cudf::conditional_left_semi_join(
            *left_view, *right_view, pred, {}, exec.stream, exec.mr);
          break;
        case single_side_join_kind::anti:
          indices = cudf::conditional_left_anti_join(
            *left_view, *right_view, pred, {}, exec.stream, exec.mr);
          break;
      }
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t mixed_pair_join_indices_impl(cudf_table_view_handle left_equality,
                                           cudf_table_view_handle right_equality,
                                           cudf_table_view_handle left_conditional,
                                           cudf_table_view_handle right_conditional,
                                           cudf_ast_expression_handle predicate,
                                           cudf_null_equality_t null_equality,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices,
                                           cudf_exec_options_t const* options,
                                           pair_join_kind kind)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_equality, "left_equality is null");
  CUDF_RUST_NULL_CHECK(right_equality, "right_equality is null");
  CUDF_RUST_NULL_CHECK(left_conditional, "left_conditional is null");
  CUDF_RUST_NULL_CHECK(right_conditional, "right_conditional is null");
  CUDF_RUST_NULL_CHECK(predicate, "predicate is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* le         = static_cast<cudf::table_view*>(left_equality);
    auto* re         = static_cast<cudf::table_view*>(right_equality);
    auto* lc         = static_cast<cudf::table_view*>(left_conditional);
    auto* rc         = static_cast<cudf::table_view*>(right_conditional);
    auto const& pred = cudf_ast_expression_from_handle(predicate);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      join_index_pair indices;
      switch (kind) {
        case pair_join_kind::inner:
          indices =
            cudf::mixed_inner_join(*le, *re, *lc, *rc, pred, null_eq, {}, exec.stream, exec.mr);
          break;
        case pair_join_kind::left:
          indices =
            cudf::mixed_left_join(*le, *re, *lc, *rc, pred, null_eq, {}, exec.stream, exec.mr);
          break;
        case pair_join_kind::full:
          indices =
            cudf::mixed_full_join(*le, *re, *lc, *rc, pred, null_eq, {}, exec.stream, exec.mr);
          break;
      }
      release_join_index_pair(std::move(indices),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t mixed_single_side_join_indices_impl(cudf_table_view_handle left_equality,
                                                  cudf_table_view_handle right_equality,
                                                  cudf_table_view_handle left_conditional,
                                                  cudf_table_view_handle right_conditional,
                                                  cudf_ast_expression_handle predicate,
                                                  cudf_null_equality_t null_equality,
                                                  cudf_column_handle* out_indices,
                                                  cudf_exec_options_t const* options,
                                                  single_side_join_kind kind)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_equality, "left_equality is null");
  CUDF_RUST_NULL_CHECK(right_equality, "right_equality is null");
  CUDF_RUST_NULL_CHECK(left_conditional, "left_conditional is null");
  CUDF_RUST_NULL_CHECK(right_conditional, "right_conditional is null");
  CUDF_RUST_NULL_CHECK(predicate, "predicate is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* le         = static_cast<cudf::table_view*>(left_equality);
    auto* re         = static_cast<cudf::table_view*>(right_equality);
    auto* lc         = static_cast<cudf::table_view*>(left_conditional);
    auto* rc         = static_cast<cudf::table_view*>(right_conditional);
    auto const& pred = cudf_ast_expression_from_handle(predicate);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices;
      switch (kind) {
        case single_side_join_kind::semi:
          indices =
            cudf::mixed_left_semi_join(*le, *re, *lc, *rc, pred, null_eq, exec.stream, exec.mr);
          break;
        case single_side_join_kind::anti:
          indices =
            cudf::mixed_left_anti_join(*le, *re, *lc, *rc, pred, null_eq, exec.stream, exec.mr);
          break;
      }
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t filter_join_indices_impl(cudf_table_view_handle left,
                                       cudf_table_view_handle right,
                                       cudf_column_view_handle left_indices,
                                       cudf_column_view_handle right_indices,
                                       cudf_ast_expression_handle predicate,
                                       cudf_join_kind_t join_kind,
                                       cudf_column_handle* out_left_indices,
                                       cudf_column_handle* out_right_indices,
                                       cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left, "left is null");
  CUDF_RUST_NULL_CHECK(right, "right is null");
  CUDF_RUST_NULL_CHECK(left_indices, "left_indices is null");
  CUDF_RUST_NULL_CHECK(right_indices, "right_indices is null");
  CUDF_RUST_NULL_CHECK(predicate, "predicate is null");
  cudf::join_kind kind;
  cudf_status_t status = to_join_kind(join_kind, &kind);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view          = static_cast<cudf::table_view*>(left);
    auto* right_view         = static_cast<cudf::table_view*>(right);
    auto* left_indices_view  = static_cast<cudf::column_view*>(left_indices);
    auto* right_indices_view = static_cast<cudf::column_view*>(right_indices);
    auto const& pred         = cudf_ast_expression_from_handle(predicate);

    CUDF_EXPECTS(left_indices_view->type().id() == cudf::type_id::INT32,
                 "left_indices must be INT32",
                 std::invalid_argument);
    CUDF_EXPECTS(right_indices_view->type().id() == cudf::type_id::INT32,
                 "right_indices must be INT32",
                 std::invalid_argument);
    CUDF_EXPECTS(
      !left_indices_view->nullable(), "left_indices must not be nullable", std::invalid_argument);
    CUDF_EXPECTS(
      !right_indices_view->nullable(), "right_indices must not be nullable", std::invalid_argument);
    CUDF_EXPECTS(left_indices_view->size() == right_indices_view->size(),
                 "left_indices and right_indices must have the same size",
                 std::invalid_argument);

    cudf::device_span<cudf::size_type const> left_span{
      left_indices_view->data<cudf::size_type>(),
      static_cast<std::size_t>(left_indices_view->size())};
    cudf::device_span<cudf::size_type const> right_span{
      right_indices_view->data<cudf::size_type>(),
      static_cast<std::size_t>(right_indices_view->size())};

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto indices = cudf::filter_join_indices(
        *left_view, *right_view, left_span, right_span, pred, kind, exec.stream, exec.mr);
      release_join_index_pair(std::move(indices),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t hash_join_size_impl(cudf_hash_join_handle join,
                                  cudf_table_view_handle probe,
                                  size_t* out,
                                  cudf_exec_options_t const* options,
                                  hash_join_probe_kind kind)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      switch (kind) {
        case hash_join_probe_kind::inner: *out = hash_join->inner_join_size(*probe_view, exec.stream); break;
        case hash_join_probe_kind::left: *out = hash_join->left_join_size(*probe_view, exec.stream); break;
        case hash_join_probe_kind::full:
          *out = hash_join->full_join_size(*probe_view, exec.stream, exec.mr);
          break;
      }
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
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
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    {
      auto [left_indices, right_indices] =
        cudf::inner_join(*lk, *rk, null_eq, exec.stream, exec.mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec.stream, exec.mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec.stream, exec.mr);

      *out_left  = cudf_release_table_handle(std::move(left_result));
      *out_right = cudf_release_table_handle(std::move(right_result));
    }
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
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    {
      auto [left_indices, right_indices] = cudf::left_join(*lk, *rk, null_eq, exec.stream, exec.mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec.stream, exec.mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec.stream, exec.mr);

      *out_left  = cudf_release_table_handle(std::move(left_result));
      *out_right = cudf_release_table_handle(std::move(right_result));
    }
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
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    {
      auto [left_indices, right_indices] = cudf::full_join(*lk, *rk, null_eq, exec.stream, exec.mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::NULLIFY, exec.stream, exec.mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec.stream, exec.mr);

      *out_left  = cudf_release_table_handle(std::move(left_result));
      *out_right = cudf_release_table_handle(std::move(right_result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_cross_join(cudf_table_view_handle left,
                              cudf_table_view_handle right,
                              cudf_table_handle* out)
{
  return cudf_cross_join_ex(left, right, out, nullptr);
}

cudf_status_t cudf_cross_join_ex(cudf_table_view_handle left,
                                 cudf_table_view_handle right,
                                 cudf_table_handle* out,
                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(left, "left is null");
  CUDF_RUST_NULL_CHECK(right, "right is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view  = static_cast<cudf::table_view*>(left);
    auto* right_view = static_cast<cudf::table_view*>(right);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto result = cudf::cross_join(*left_view, *right_view, exec.stream, exec.mr);
      *out = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_inner_join_indices(cudf_table_view_handle left_keys,
                                      cudf_table_view_handle right_keys,
                                      cudf_null_equality_t null_equality,
                                      cudf_column_handle* out_left_indices,
                                      cudf_column_handle* out_right_indices)
{
  return cudf_inner_join_indices_ex(
    left_keys, right_keys, null_equality, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_inner_join_indices_ex(cudf_table_view_handle left_keys,
                                         cudf_table_view_handle right_keys,
                                         cudf_null_equality_t null_equality,
                                         cudf_column_handle* out_left_indices,
                                         cudf_column_handle* out_right_indices,
                                         cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(cudf::inner_join(*lk, *rk, null_eq, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_join_indices(cudf_table_view_handle left_keys,
                                     cudf_table_view_handle right_keys,
                                     cudf_null_equality_t null_equality,
                                     cudf_column_handle* out_left_indices,
                                     cudf_column_handle* out_right_indices)
{
  return cudf_left_join_indices_ex(
    left_keys, right_keys, null_equality, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_left_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(cudf::left_join(*lk, *rk, null_eq, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_full_join_indices(cudf_table_view_handle left_keys,
                                     cudf_table_view_handle right_keys,
                                     cudf_null_equality_t null_equality,
                                     cudf_column_handle* out_left_indices,
                                     cudf_column_handle* out_right_indices)
{
  return cudf_full_join_indices_ex(
    left_keys, right_keys, null_equality, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_full_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(cudf::full_join(*lk, *rk, null_eq, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_semi_join_indices(cudf_table_view_handle left_keys,
                                          cudf_table_view_handle right_keys,
                                          cudf_null_equality_t null_equality,
                                          cudf_column_handle* out_indices)
{
  return cudf_left_semi_join_indices_ex(left_keys, right_keys, null_equality, out_indices, nullptr);
}

cudf_status_t cudf_left_semi_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      cudf::filtered_join join{*rk, null_eq, cudf::set_as_build_table::RIGHT, 0.5, exec.stream};
      auto indices = join.semi_join(*lk, exec.stream, exec.mr);
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_anti_join_indices(cudf_table_view_handle left_keys,
                                          cudf_table_view_handle right_keys,
                                          cudf_null_equality_t null_equality,
                                          cudf_column_handle* out_indices)
{
  return cudf_left_anti_join_indices_ex(left_keys, right_keys, null_equality, out_indices, nullptr);
}

cudf_status_t cudf_left_anti_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      cudf::filtered_join join{*rk, null_eq, cudf::set_as_build_table::RIGHT, 0.5, exec.stream};
      auto indices = join.anti_join(*lk, exec.stream, exec.mr);
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_conditional_inner_join_indices(cudf_table_view_handle left,
                                                  cudf_table_view_handle right,
                                                  cudf_ast_expression_handle predicate,
                                                  cudf_column_handle* out_left_indices,
                                                  cudf_column_handle* out_right_indices)
{
  return cudf_conditional_inner_join_indices_ex(
    left, right, predicate, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_conditional_inner_join_indices_ex(cudf_table_view_handle left,
                                                     cudf_table_view_handle right,
                                                     cudf_ast_expression_handle predicate,
                                                     cudf_column_handle* out_left_indices,
                                                     cudf_column_handle* out_right_indices,
                                                     cudf_exec_options_t const* options)
{
  return conditional_pair_join_indices_impl(
    left, right, predicate, out_left_indices, out_right_indices, options, pair_join_kind::inner);
}

cudf_status_t cudf_conditional_left_join_indices(cudf_table_view_handle left,
                                                 cudf_table_view_handle right,
                                                 cudf_ast_expression_handle predicate,
                                                 cudf_column_handle* out_left_indices,
                                                 cudf_column_handle* out_right_indices)
{
  return cudf_conditional_left_join_indices_ex(
    left, right, predicate, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_conditional_left_join_indices_ex(cudf_table_view_handle left,
                                                    cudf_table_view_handle right,
                                                    cudf_ast_expression_handle predicate,
                                                    cudf_column_handle* out_left_indices,
                                                    cudf_column_handle* out_right_indices,
                                                    cudf_exec_options_t const* options)
{
  return conditional_pair_join_indices_impl(
    left, right, predicate, out_left_indices, out_right_indices, options, pair_join_kind::left);
}

cudf_status_t cudf_conditional_full_join_indices(cudf_table_view_handle left,
                                                 cudf_table_view_handle right,
                                                 cudf_ast_expression_handle predicate,
                                                 cudf_column_handle* out_left_indices,
                                                 cudf_column_handle* out_right_indices)
{
  return cudf_conditional_full_join_indices_ex(
    left, right, predicate, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_conditional_full_join_indices_ex(cudf_table_view_handle left,
                                                    cudf_table_view_handle right,
                                                    cudf_ast_expression_handle predicate,
                                                    cudf_column_handle* out_left_indices,
                                                    cudf_column_handle* out_right_indices,
                                                    cudf_exec_options_t const* options)
{
  return conditional_pair_join_indices_impl(
    left, right, predicate, out_left_indices, out_right_indices, options, pair_join_kind::full);
}

cudf_status_t cudf_conditional_left_semi_join_indices(cudf_table_view_handle left,
                                                      cudf_table_view_handle right,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_column_handle* out_indices)
{
  return cudf_conditional_left_semi_join_indices_ex(left, right, predicate, out_indices, nullptr);
}

cudf_status_t cudf_conditional_left_semi_join_indices_ex(cudf_table_view_handle left,
                                                         cudf_table_view_handle right,
                                                         cudf_ast_expression_handle predicate,
                                                         cudf_column_handle* out_indices,
                                                         cudf_exec_options_t const* options)
{
  return conditional_single_side_join_indices_impl(
    left, right, predicate, out_indices, options, single_side_join_kind::semi);
}

cudf_status_t cudf_conditional_left_anti_join_indices(cudf_table_view_handle left,
                                                      cudf_table_view_handle right,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_column_handle* out_indices)
{
  return cudf_conditional_left_anti_join_indices_ex(left, right, predicate, out_indices, nullptr);
}

cudf_status_t cudf_conditional_left_anti_join_indices_ex(cudf_table_view_handle left,
                                                         cudf_table_view_handle right,
                                                         cudf_ast_expression_handle predicate,
                                                         cudf_column_handle* out_indices,
                                                         cudf_exec_options_t const* options)
{
  return conditional_single_side_join_indices_impl(
    left, right, predicate, out_indices, options, single_side_join_kind::anti);
}

cudf_status_t cudf_filter_join_indices(cudf_table_view_handle left,
                                       cudf_table_view_handle right,
                                       cudf_column_view_handle left_indices,
                                       cudf_column_view_handle right_indices,
                                       cudf_ast_expression_handle predicate,
                                       cudf_join_kind_t join_kind,
                                       cudf_column_handle* out_left_indices,
                                       cudf_column_handle* out_right_indices)
{
  return cudf_filter_join_indices_ex(left,
                                     right,
                                     left_indices,
                                     right_indices,
                                     predicate,
                                     join_kind,
                                     out_left_indices,
                                     out_right_indices,
                                     nullptr);
}

cudf_status_t cudf_filter_join_indices_ex(cudf_table_view_handle left,
                                          cudf_table_view_handle right,
                                          cudf_column_view_handle left_indices,
                                          cudf_column_view_handle right_indices,
                                          cudf_ast_expression_handle predicate,
                                          cudf_join_kind_t join_kind,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options)
{
  return filter_join_indices_impl(left,
                                  right,
                                  left_indices,
                                  right_indices,
                                  predicate,
                                  join_kind,
                                  out_left_indices,
                                  out_right_indices,
                                  options);
}

cudf_status_t cudf_mixed_inner_join_indices(cudf_table_view_handle left_equality,
                                            cudf_table_view_handle right_equality,
                                            cudf_table_view_handle left_conditional,
                                            cudf_table_view_handle right_conditional,
                                            cudf_ast_expression_handle predicate,
                                            cudf_null_equality_t null_equality,
                                            cudf_column_handle* out_left_indices,
                                            cudf_column_handle* out_right_indices)
{
  return cudf_mixed_inner_join_indices_ex(left_equality,
                                          right_equality,
                                          left_conditional,
                                          right_conditional,
                                          predicate,
                                          null_equality,
                                          out_left_indices,
                                          out_right_indices,
                                          nullptr);
}

cudf_status_t cudf_mixed_inner_join_indices_ex(cudf_table_view_handle left_equality,
                                               cudf_table_view_handle right_equality,
                                               cudf_table_view_handle left_conditional,
                                               cudf_table_view_handle right_conditional,
                                               cudf_ast_expression_handle predicate,
                                               cudf_null_equality_t null_equality,
                                               cudf_column_handle* out_left_indices,
                                               cudf_column_handle* out_right_indices,
                                               cudf_exec_options_t const* options)
{
  return mixed_pair_join_indices_impl(left_equality,
                                      right_equality,
                                      left_conditional,
                                      right_conditional,
                                      predicate,
                                      null_equality,
                                      out_left_indices,
                                      out_right_indices,
                                      options,
                                      pair_join_kind::inner);
}

cudf_status_t cudf_mixed_left_join_indices(cudf_table_view_handle left_equality,
                                           cudf_table_view_handle right_equality,
                                           cudf_table_view_handle left_conditional,
                                           cudf_table_view_handle right_conditional,
                                           cudf_ast_expression_handle predicate,
                                           cudf_null_equality_t null_equality,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices)
{
  return cudf_mixed_left_join_indices_ex(left_equality,
                                         right_equality,
                                         left_conditional,
                                         right_conditional,
                                         predicate,
                                         null_equality,
                                         out_left_indices,
                                         out_right_indices,
                                         nullptr);
}

cudf_status_t cudf_mixed_left_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              cudf_exec_options_t const* options)
{
  return mixed_pair_join_indices_impl(left_equality,
                                      right_equality,
                                      left_conditional,
                                      right_conditional,
                                      predicate,
                                      null_equality,
                                      out_left_indices,
                                      out_right_indices,
                                      options,
                                      pair_join_kind::left);
}

cudf_status_t cudf_mixed_full_join_indices(cudf_table_view_handle left_equality,
                                           cudf_table_view_handle right_equality,
                                           cudf_table_view_handle left_conditional,
                                           cudf_table_view_handle right_conditional,
                                           cudf_ast_expression_handle predicate,
                                           cudf_null_equality_t null_equality,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices)
{
  return cudf_mixed_full_join_indices_ex(left_equality,
                                         right_equality,
                                         left_conditional,
                                         right_conditional,
                                         predicate,
                                         null_equality,
                                         out_left_indices,
                                         out_right_indices,
                                         nullptr);
}

cudf_status_t cudf_mixed_full_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              cudf_exec_options_t const* options)
{
  return mixed_pair_join_indices_impl(left_equality,
                                      right_equality,
                                      left_conditional,
                                      right_conditional,
                                      predicate,
                                      null_equality,
                                      out_left_indices,
                                      out_right_indices,
                                      options,
                                      pair_join_kind::full);
}

cudf_status_t cudf_mixed_left_semi_join_indices(cudf_table_view_handle left_equality,
                                                cudf_table_view_handle right_equality,
                                                cudf_table_view_handle left_conditional,
                                                cudf_table_view_handle right_conditional,
                                                cudf_ast_expression_handle predicate,
                                                cudf_null_equality_t null_equality,
                                                cudf_column_handle* out_indices)
{
  return cudf_mixed_left_semi_join_indices_ex(left_equality,
                                              right_equality,
                                              left_conditional,
                                              right_conditional,
                                              predicate,
                                              null_equality,
                                              out_indices,
                                              nullptr);
}

cudf_status_t cudf_mixed_left_semi_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   cudf_exec_options_t const* options)
{
  return mixed_single_side_join_indices_impl(left_equality,
                                             right_equality,
                                             left_conditional,
                                             right_conditional,
                                             predicate,
                                             null_equality,
                                             out_indices,
                                             options,
                                             single_side_join_kind::semi);
}

cudf_status_t cudf_mixed_left_anti_join_indices(cudf_table_view_handle left_equality,
                                                cudf_table_view_handle right_equality,
                                                cudf_table_view_handle left_conditional,
                                                cudf_table_view_handle right_conditional,
                                                cudf_ast_expression_handle predicate,
                                                cudf_null_equality_t null_equality,
                                                cudf_column_handle* out_indices)
{
  return cudf_mixed_left_anti_join_indices_ex(left_equality,
                                              right_equality,
                                              left_conditional,
                                              right_conditional,
                                              predicate,
                                              null_equality,
                                              out_indices,
                                              nullptr);
}

cudf_status_t cudf_mixed_left_anti_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   cudf_exec_options_t const* options)
{
  return mixed_single_side_join_indices_impl(left_equality,
                                             right_equality,
                                             left_conditional,
                                             right_conditional,
                                             predicate,
                                             null_equality,
                                             out_indices,
                                             options,
                                             single_side_join_kind::anti);
}

cudf_status_t cudf_distinct_hash_join_create(cudf_table_view_handle build,
                                             cudf_null_equality_t null_equality,
                                             cudf_distinct_hash_join_handle* out)
{
  return cudf_distinct_hash_join_create_ex(build, null_equality, out, nullptr);
}

cudf_status_t cudf_distinct_hash_join_create_ex(cudf_table_view_handle build,
                                                cudf_null_equality_t null_equality,
                                                cudf_distinct_hash_join_handle* out,
                                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(build, "build is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* build_view = static_cast<cudf::table_view*>(build);
    auto owner       = std::make_unique<distinct_hash_join_owner>();
    owner->device    = exec.device;
    owner->mr_raw    = exec.mr_raw;
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      owner->join =
        std::make_unique<cudf::distinct_hash_join>(*build_view, null_eq, 0.5, exec.stream);
    }
    *out = static_cast<cudf_distinct_hash_join_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_distinct_hash_join_destroy(cudf_distinct_hash_join_handle handle)
{
  if (handle == nullptr) { return; }
  cudf_clear_error();
  try {
    auto* owner = static_cast<distinct_hash_join_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return; }
    cudf_current_device_resource_guard current_mr{owner->mr_raw};
    delete owner;
  } catch (...) {
    (void)translate_exception();
  }
}

cudf_status_t cudf_distinct_hash_join_left_join(cudf_distinct_hash_join_handle join,
                                                cudf_table_view_handle probe,
                                                cudf_column_handle* out)
{
  return cudf_distinct_hash_join_left_join_ex(join, probe, out, nullptr);
}

cudf_status_t cudf_distinct_hash_join_left_join_ex(cudf_distinct_hash_join_handle join,
                                                   cudf_table_view_handle probe,
                                                   cudf_column_handle* out,
                                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner         = static_cast<distinct_hash_join_owner*>(join);
    auto* distinct_join = owner->join.get();
    auto* probe_view    = static_cast<cudf::table_view*>(probe);
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto indices = distinct_join->left_join(*probe_view, exec.stream, exec.mr);
      *out         = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                        cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_create(cudf_table_view_handle build,
                                    cudf_nullable_join_t nullable_join,
                                    cudf_null_equality_t null_equality,
                                    double load_factor,
                                    cudf_hash_join_handle* out)
{
  return cudf_hash_join_create_ex(build, nullable_join, null_equality, load_factor, out, nullptr);
}

cudf_status_t cudf_hash_join_create_ex(cudf_table_view_handle build,
                                       cudf_nullable_join_t nullable_join,
                                       cudf_null_equality_t null_equality,
                                       double load_factor,
                                       cudf_hash_join_handle* out,
                                       cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(build, "build is null");
  cudf::nullable_join nullable;
  cudf_status_t status = to_nullable_join(nullable_join, &nullable);
  if (status != CUDF_SUCCESS) { return status; }
  cudf::null_equality null_eq;
  status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* build_view = static_cast<cudf::table_view*>(build);
    auto owner       = std::make_unique<hash_join_owner>();
    owner->device    = exec.device;
    owner->upstream_mr_lifetime = exec.mr_lifetime;
    owner->observed_mr =
      std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(
        exec.mr);
    owner->mr_raw = owner->observed_mr.get();
    {
      cudf_current_device_resource_guard current_mr{owner->mr_raw};
      owner->join =
        std::make_unique<cudf::hash_join>(*build_view, nullable, null_eq, load_factor, exec.stream);
      auto const bytes = owner->observed_mr->get_bytes_counter().value;
      owner->device_allocated_size_bytes = bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
    }
    *out = static_cast<cudf_hash_join_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_pre_build_reservation_size(cudf_table_view_handle build,
                                                        cudf_nullable_join_t nullable_join,
                                                        cudf_null_equality_t null_equality,
                                                        double load_factor,
                                                        size_t* out)
{
  return cudf_hash_join_pre_build_reservation_size_ex(
    build, nullable_join, null_equality, load_factor, out, nullptr);
}

cudf_status_t cudf_hash_join_pre_build_reservation_size_ex(cudf_table_view_handle build,
                                                           cudf_nullable_join_t nullable_join,
                                                           cudf_null_equality_t null_equality,
                                                           double load_factor,
                                                           size_t* out,
                                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(build, "build is null");
  cudf::nullable_join nullable;
  cudf_status_t status = to_nullable_join(nullable_join, &nullable);
  if (status != CUDF_SUCCESS) { return status; }
  cudf::null_equality null_eq;
  status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    (void)build;
    (void)nullable;
    (void)null_eq;
    (void)load_factor;
    (void)exec;
    set_error(
      "libcudf does not expose exact pre-build hash-join reservation bytes; "
      "callers must keep bounded hash-join execution disabled instead of "
      "estimating this allocation");
    return CUDF_ERROR_UNSUPPORTED;
  }
  CUDF_RUST_CATCH
}

void cudf_hash_join_destroy(cudf_hash_join_handle handle)
{
  if (handle == nullptr) { return; }
  cudf_clear_error();
  try {
    auto* owner = static_cast<hash_join_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return; }
    cudf_current_device_resource_guard current_mr{owner->mr_raw};
    delete owner;
  } catch (...) {
    (void)translate_exception();
  }
}

cudf_status_t cudf_hash_join_device_allocated_size(cudf_hash_join_handle handle, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "hash join is null");
  CUDF_RUST_TRY
  {
    auto* owner = static_cast<hash_join_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    *out = owner->device_allocated_size_bytes;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_inner_join(cudf_hash_join_handle join,
                                        cudf_table_view_handle probe,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices)
{
  return cudf_hash_join_inner_join_ex(join, probe, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_hash_join_inner_join_ex(cudf_hash_join_handle join,
                                           cudf_table_view_handle probe,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices,
                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(hash_join->inner_join(*probe_view, {}, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_inner_join_with_output_size(cudf_hash_join_handle join,
                                                         cudf_table_view_handle probe,
                                                         size_t output_size,
                                                         cudf_column_handle* out_left_indices,
                                                         cudf_column_handle* out_right_indices)
{
  return cudf_hash_join_inner_join_with_output_size_ex(
    join, probe, output_size, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_hash_join_inner_join_with_output_size_ex(cudf_hash_join_handle join,
                                                            cudf_table_view_handle probe,
                                                            size_t output_size,
                                                            cudf_column_handle* out_left_indices,
                                                            cudf_column_handle* out_right_indices,
                                                            cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(hash_join->inner_join(
                                *probe_view, std::optional<std::size_t>{output_size}, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_inner_join_size(cudf_hash_join_handle join,
                                             cudf_table_view_handle probe,
                                             size_t* out)
{
  return cudf_hash_join_inner_join_size_ex(join, probe, out, nullptr);
}

cudf_status_t cudf_hash_join_inner_join_size_ex(cudf_hash_join_handle join,
                                                cudf_table_view_handle probe,
                                                size_t* out,
                                                cudf_exec_options_t const* options)
{
  return hash_join_size_impl(join, probe, out, options, hash_join_probe_kind::inner);
}

cudf_status_t cudf_hash_join_left_join(cudf_hash_join_handle join,
                                       cudf_table_view_handle probe,
                                       cudf_column_handle* out_left_indices,
                                       cudf_column_handle* out_right_indices)
{
  return cudf_hash_join_left_join_ex(join, probe, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_hash_join_left_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(hash_join->left_join(*probe_view, {}, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_left_join_with_output_size(cudf_hash_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t output_size,
                                                        cudf_column_handle* out_left_indices,
                                                        cudf_column_handle* out_right_indices)
{
  return cudf_hash_join_left_join_with_output_size_ex(
    join, probe, output_size, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_hash_join_left_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(hash_join->left_join(
                                *probe_view, std::optional<std::size_t>{output_size}, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_left_join_size(cudf_hash_join_handle join,
                                            cudf_table_view_handle probe,
                                            size_t* out)
{
  return cudf_hash_join_left_join_size_ex(join, probe, out, nullptr);
}

cudf_status_t cudf_hash_join_left_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
                                               cudf_exec_options_t const* options)
{
  return hash_join_size_impl(join, probe, out, options, hash_join_probe_kind::left);
}

cudf_status_t cudf_hash_join_full_join(cudf_hash_join_handle join,
                                       cudf_table_view_handle probe,
                                       cudf_column_handle* out_left_indices,
                                       cudf_column_handle* out_right_indices)
{
  return cudf_hash_join_full_join_ex(join, probe, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_hash_join_full_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(hash_join->full_join(*probe_view, {}, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_full_join_with_output_size(cudf_hash_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t output_size,
                                                        cudf_column_handle* out_left_indices,
                                                        cudf_column_handle* out_right_indices)
{
  return cudf_hash_join_full_join_with_output_size_ex(
    join, probe, output_size, out_left_indices, out_right_indices, nullptr);
}

cudf_status_t cudf_hash_join_full_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      release_join_index_pair(hash_join->full_join(
                                *probe_view, std::optional<std::size_t>{output_size}, exec.stream, exec.mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_full_join_size(cudf_hash_join_handle join,
                                            cudf_table_view_handle probe,
                                            size_t* out)
{
  return cudf_hash_join_full_join_size_ex(join, probe, out, nullptr);
}

cudf_status_t cudf_hash_join_full_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
                                               cudf_exec_options_t const* options)
{
  return hash_join_size_impl(join, probe, out, options, hash_join_probe_kind::full);
}
}  // extern "C"
