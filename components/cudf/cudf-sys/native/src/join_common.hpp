#pragma once

#include "join.h"

#include "ast_handle.hpp"
#include "execution_context_internal.hpp"
#include "handle_provenance.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf_rust::join {
/// Concatenate key and payload views into one table view.
inline cudf::table_view concat_views(const cudf::table_view& keys, const cudf::table_view& payload)
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

inline std::size_t checked_join_gather_map_size_bytes(std::size_t output_size, std::size_t map_count)
{
  constexpr auto size_type_bytes = sizeof(cudf::size_type);
  auto constexpr max_size        = std::numeric_limits<std::size_t>::max();
  CUDF_EXPECTS(output_size <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "join gather-map row count exceeds cudf::size_type maximum");
  CUDF_EXPECTS(map_count == 0 || output_size <= max_size / map_count,
               "join gather-map reservation size overflow");
  auto const element_count = output_size * map_count;
  CUDF_EXPECTS(element_count <= max_size / size_type_bytes,
               "join gather-map reservation size overflow");
  return element_count * size_type_bytes;
}

inline cudf::size_type checked_projected_cross_join_rows(std::size_t rows, char const*)
{
  CUDF_EXPECTS(rows <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "projected cross join row count exceeds cudf::size_type maximum");
  return static_cast<cudf::size_type>(rows);
}

inline std::size_t checked_projected_cross_join_output_rows(std::size_t left_num_rows,
                                                     std::size_t right_num_rows)
{
  auto constexpr max_size = std::numeric_limits<std::size_t>::max();
  CUDF_EXPECTS(left_num_rows == 0 || right_num_rows <= max_size / left_num_rows,
               "projected cross join output row count overflow");
  auto const output_rows = left_num_rows * right_num_rows;
  checked_projected_cross_join_rows(output_rows, "projected cross join output row count");
  return output_rows;
}

inline std::unique_ptr<cudf::table> projected_cross_join_table(cudf::table_view const* left_selected,
                                                        cudf::table_view const* right_selected,
                                                        std::size_t left_num_rows,
                                                        std::size_t right_num_rows,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto const left_rows_i32 =
    checked_projected_cross_join_rows(left_num_rows, "left projected cross join row count");
  auto const right_rows_i32 =
    checked_projected_cross_join_rows(right_num_rows, "right projected cross join row count");
  auto const output_rows = checked_projected_cross_join_output_rows(left_num_rows, right_num_rows);

  auto const left_columns  = left_selected == nullptr ? 0 : left_selected->num_columns();
  auto const right_columns = right_selected == nullptr ? 0 : right_selected->num_columns();
  CUDF_EXPECTS(left_columns > 0 || right_columns > 0,
               "projected cross join requires at least one selected output column");

  if (left_selected != nullptr) {
    CUDF_EXPECTS(left_selected->num_rows() == left_rows_i32,
                 "left selected table row count does not match left_num_rows");
  }
  if (right_selected != nullptr) {
    CUDF_EXPECTS(right_selected->num_rows() == right_rows_i32,
                 "right selected table row count does not match right_num_rows");
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(static_cast<std::size_t>(left_columns + right_columns));

  auto append_columns = [&columns](std::unique_ptr<cudf::table> table) {
    auto released = table->release();
    std::move(released.begin(), released.end(), std::back_inserter(columns));
  };

  if (output_rows == 0) {
    if (left_columns > 0) { append_columns(cudf::empty_like(*left_selected)); }
    if (right_columns > 0) { append_columns(cudf::empty_like(*right_selected)); }
    return std::make_unique<cudf::table>(std::move(columns));
  }

  if (left_columns > 0) { append_columns(cudf::repeat(*left_selected, right_rows_i32, stream, mr)); }
  if (right_columns > 0) { append_columns(cudf::tile(*right_selected, left_rows_i32, stream, mr)); }
  return std::make_unique<cudf::table>(std::move(columns));
}

/// Materialize a table given a gather map (index column).
inline std::unique_ptr<cudf::table> gather_table(
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

inline cudf_status_t to_null_equality(cudf_null_equality_t input, cudf::null_equality* out)
{
  switch (input) {
    case CUDF_NULL_EQUALITY_EQUAL: *out = cudf::null_equality::EQUAL; return CUDF_SUCCESS;
    case CUDF_NULL_EQUALITY_UNEQUAL: *out = cudf::null_equality::UNEQUAL; return CUDF_SUCCESS;
    default: set_error("invalid null_equality"); return CUDF_ERROR_INVALID_OPERATION;
  }
}

inline cudf_status_t to_nullable_join(cudf_nullable_join_t input, cudf::nullable_join* out)
{
  switch (input) {
    case CUDF_NULLABLE_JOIN_YES: *out = cudf::nullable_join::YES; return CUDF_SUCCESS;
    case CUDF_NULLABLE_JOIN_NO: *out = cudf::nullable_join::NO; return CUDF_SUCCESS;
    default: set_error("invalid nullable_join"); return CUDF_ERROR_INVALID_OPERATION;
  }
}

inline cudf_status_t to_join_kind(cudf_join_kind_t input, cudf::join_kind* out)
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

inline std::unique_ptr<cudf::column> release_join_index_column(
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices)
{
  return std::make_unique<cudf::column>(std::move(*indices), rmm::device_buffer{}, 0);
}

using join_index_pair = std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                                  std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;

enum class pair_join_kind { inner, left, full };

enum class single_side_join_kind { semi, anti };

enum class conditional_join_size_kind { inner, left, semi, anti };

enum class mixed_join_size_kind { inner, left, full, semi, anti };

enum class hash_join_probe_kind { inner, left, full };

enum class filtered_join_probe_kind { semi, anti };

struct distinct_hash_join_owner {
  std::unique_ptr<cudf::distinct_hash_join> join;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  int32_t device                          = 0;
};

struct filtered_join_owner {
  std::shared_ptr<void> upstream_mr_lifetime;
  std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>
    observed_mr;
  std::unique_ptr<cudf::filtered_join> join;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  std::size_t device_allocated_size_bytes = 0;
  int32_t device                          = 0;
};

struct filtered_join_probe_owner {
  std::shared_ptr<void> upstream_mr_lifetime;
  std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>
    observed_mr;
  std::unique_ptr<cudf::filtered_join_probe_state> probe;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  std::size_t device_allocated_size_bytes = 0;
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

struct conditional_full_join_size_data_owner {
  std::shared_ptr<void> upstream_mr_lifetime;
  std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>
    observed_mr;
  std::unique_ptr<cudf::conditional_full_join_size_data> size_data;
  std::optional<cudf::table_view> left;
  std::optional<cudf::table_view> right;
  cudf_ast_expression_owner_ref predicate;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  std::size_t output_size                 = 0;
  std::size_t device_allocated_size_bytes = 0;
  int32_t device                          = 0;
};

struct mixed_join_size_data_owner {
  std::shared_ptr<void> upstream_mr_lifetime;
  std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>
    observed_mr;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> matches_per_row;
  std::unique_ptr<cudf::mixed_full_join_size_data> full_size_data;
  std::unique_ptr<cudf::mixed_left_semi_anti_join_size_data> single_size_data;
  std::optional<cudf::table_view> left_equality;
  std::optional<cudf::table_view> right_equality;
  std::optional<cudf::table_view> left_conditional;
  std::optional<cudf::table_view> right_conditional;
  cudf_ast_expression_owner_ref predicate;
  cudf::null_equality null_equality = cudf::null_equality::EQUAL;
  mixed_join_size_kind kind          = mixed_join_size_kind::inner;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  std::size_t output_size                 = 0;
  std::size_t device_allocated_size_bytes = 0;
  int32_t device                          = 0;
};

struct column_handle_deleter {
  void operator()(void* handle) const noexcept { cudf_column_destroy(handle); }
};

using unique_column_handle = std::unique_ptr<void, column_handle_deleter>;

inline unique_column_handle make_unique_column_handle(std::unique_ptr<cudf::column> column,
                                               cudf_handle_provenance const& provenance)
{
  return unique_column_handle{cudf_release_column_handle(std::move(column), provenance)};
}

inline void release_join_index_pair(join_index_pair indices,
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

inline cudf_status_t conditional_pair_join_indices_impl(cudf_table_view_handle left,
                                                 cudf_table_view_handle right,
                                                 cudf_ast_expression_handle predicate,
                                                 cudf_column_handle* out_left_indices,
                                                 cudf_column_handle* out_right_indices,
                                                 std::optional<std::size_t> output_size,
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
            cudf::conditional_inner_join(*left_view, *right_view, pred, output_size, exec.stream, exec.mr);
          break;
        case pair_join_kind::left:
          indices =
            cudf::conditional_left_join(*left_view, *right_view, pred, output_size, exec.stream, exec.mr);
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

inline cudf_status_t conditional_single_side_join_indices_impl(cudf_table_view_handle left,
                                                        cudf_table_view_handle right,
                                                        cudf_ast_expression_handle predicate,
                                                        cudf_column_handle* out_indices,
                                                        std::optional<std::size_t> output_size,
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
            *left_view, *right_view, pred, output_size, exec.stream, exec.mr);
          break;
        case single_side_join_kind::anti:
          indices = cudf::conditional_left_anti_join(
            *left_view, *right_view, pred, output_size, exec.stream, exec.mr);
          break;
      }
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t conditional_join_size_impl(cudf_table_view_handle left,
                                         cudf_table_view_handle right,
                                         cudf_ast_expression_handle predicate,
                                         size_t* out,
                                         cudf_exec_options_t const* options,
                                         conditional_join_size_kind kind)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
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
      switch (kind) {
        case conditional_join_size_kind::inner:
          *out = cudf::conditional_inner_join_size(*left_view, *right_view, pred, exec.stream, exec.mr);
          break;
        case conditional_join_size_kind::left:
          *out = cudf::conditional_left_join_size(*left_view, *right_view, pred, exec.stream, exec.mr);
          break;
        case conditional_join_size_kind::semi:
          *out = cudf::conditional_left_semi_join_size(*left_view, *right_view, pred, exec.stream, exec.mr);
          break;
        case conditional_join_size_kind::anti:
          *out = cudf::conditional_left_anti_join_size(*left_view, *right_view, pred, exec.stream, exec.mr);
          break;
      }
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t mixed_pair_join_indices_impl(cudf_table_view_handle left_equality,
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

inline cudf_status_t conditional_full_join_size_data_impl(
  cudf_table_view_handle left,
  cudf_table_view_handle right,
  cudf_ast_expression_handle predicate,
  cudf_conditional_full_join_size_data_handle* out,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(left, "left is null");
  CUDF_RUST_NULL_CHECK(right, "right is null");
  CUDF_RUST_NULL_CHECK(predicate, "predicate is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view    = static_cast<cudf::table_view*>(left);
    auto* right_view   = static_cast<cudf::table_view*>(right);
    auto const& pred   = cudf_ast_expression_from_handle(predicate);
    auto owner         = std::make_unique<conditional_full_join_size_data_owner>();
    owner->device      = exec.device;
    owner->upstream_mr_lifetime = exec.mr_lifetime;
    owner->left        = *left_view;
    owner->right       = *right_view;
    owner->predicate   = cudf_ast_expression_owner_ref_from_handle(predicate);
    owner->observed_mr =
      std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(
        exec.mr);
    owner->mr_raw = owner->observed_mr.get();
    {
      cudf_current_device_resource_guard current_mr{owner->mr_raw};
      auto const observed_mr_ref = cudf::get_current_device_resource_ref();
      owner->size_data =
        cudf::conditional_full_join_size(*left_view, *right_view, pred, exec.stream, observed_mr_ref);
      owner->output_size = owner->size_data->output_size();
      auto const bytes   = owner->observed_mr->get_bytes_counter().value;
      owner->device_allocated_size_bytes = bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
    }
    *out = static_cast<cudf_conditional_full_join_size_data_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t conditional_full_join_size_data_materialize_indices_impl(
  cudf_conditional_full_join_size_data_handle size_data,
  cudf_column_handle* out_left_indices,
  cudf_column_handle* out_right_indices,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(size_data, "conditional full join size data is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner = static_cast<conditional_full_join_size_data_owner*>(size_data);
    CUDF_EXPECTS(owner->device == exec.device,
                 "conditional full join size data must be materialized on its creation device");
    CUDF_EXPECTS(owner->size_data != nullptr, "conditional full join size data missing state");
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto indices = owner->size_data->materialize_indices(exec.stream, exec.mr);
      release_join_index_pair(std::move(indices),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t mixed_single_side_join_indices_impl(cudf_table_view_handle left_equality,
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

inline cudf_status_t mixed_join_size_data_impl(cudf_table_view_handle left_equality,
                                        cudf_table_view_handle right_equality,
                                        cudf_table_view_handle left_conditional,
                                        cudf_table_view_handle right_conditional,
                                        cudf_ast_expression_handle predicate,
                                        cudf_null_equality_t null_equality,
                                        cudf_mixed_join_size_data_handle* out,
                                        cudf_exec_options_t const* options,
                                        mixed_join_size_kind kind)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
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

    auto owner                  = std::make_unique<mixed_join_size_data_owner>();
    owner->device               = exec.device;
    owner->upstream_mr_lifetime = exec.mr_lifetime;
    owner->left_equality        = *le;
    owner->right_equality       = *re;
    owner->left_conditional     = *lc;
    owner->right_conditional    = *rc;
    owner->predicate            = cudf_ast_expression_owner_ref_from_handle(predicate);
    owner->null_equality        = null_eq;
    owner->kind                 = kind;
    owner->observed_mr =
      std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(
        exec.mr);
    owner->mr_raw = owner->observed_mr.get();
    {
      cudf_current_device_resource_guard current_mr{owner->mr_raw};
      auto const observed_mr_ref = cudf::get_current_device_resource_ref();
      switch (kind) {
        case mixed_join_size_kind::inner: {
          auto size_data = cudf::mixed_inner_join_size(
            *le, *re, *lc, *rc, pred, null_eq, exec.stream, observed_mr_ref);
          owner->output_size     = size_data.first;
          owner->matches_per_row = std::move(size_data.second);
          break;
        }
        case mixed_join_size_kind::left: {
          auto size_data = cudf::mixed_left_join_size(
            *le, *re, *lc, *rc, pred, null_eq, exec.stream, observed_mr_ref);
          owner->output_size     = size_data.first;
          owner->matches_per_row = std::move(size_data.second);
          break;
        }
        case mixed_join_size_kind::full:
          owner->full_size_data =
            cudf::mixed_full_join_size(*le, *re, *lc, *rc, pred, null_eq, exec.stream, observed_mr_ref);
          owner->output_size = owner->full_size_data->output_size();
          break;
        case mixed_join_size_kind::semi:
          owner->single_size_data =
            cudf::mixed_left_semi_join_size(*le, *re, *lc, *rc, pred, null_eq, exec.stream, observed_mr_ref);
          owner->output_size = owner->single_size_data->output_size();
          break;
        case mixed_join_size_kind::anti:
          owner->single_size_data =
            cudf::mixed_left_anti_join_size(*le, *re, *lc, *rc, pred, null_eq, exec.stream, observed_mr_ref);
          owner->output_size = owner->single_size_data->output_size();
          break;
      }
      auto const bytes = owner->observed_mr->get_bytes_counter().value;
      owner->device_allocated_size_bytes = bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
    }
    *out = static_cast<cudf_mixed_join_size_data_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t mixed_join_size_data_materialize_indices_impl(
  cudf_mixed_join_size_data_handle size_data,
  cudf_column_handle* out_left_indices,
  cudf_column_handle* out_right_indices,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(size_data, "size_data is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner = static_cast<mixed_join_size_data_owner*>(size_data);
    CUDF_EXPECTS(owner->device == exec.device,
                 "mixed join size data must be materialized on its creation device");
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      join_index_pair indices;
      switch (owner->kind) {
        case mixed_join_size_kind::inner:
        case mixed_join_size_kind::left: {
          CUDF_EXPECTS(owner->left_equality.has_value(), "mixed join size data missing left equality view");
          CUDF_EXPECTS(owner->right_equality.has_value(), "mixed join size data missing right equality view");
          CUDF_EXPECTS(owner->left_conditional.has_value(),
                       "mixed join size data missing left conditional view");
          CUDF_EXPECTS(owner->right_conditional.has_value(),
                       "mixed join size data missing right conditional view");
          CUDF_EXPECTS(owner->predicate != nullptr, "mixed join size data missing predicate");
          CUDF_EXPECTS(owner->matches_per_row != nullptr, "mixed join size data missing per-row matches");
          auto const& pred = *owner->predicate->expression;
          auto span = cudf::device_span<cudf::size_type const>{owner->matches_per_row->begin(),
                                                               owner->matches_per_row->size()};
          cudf::output_size_data_type output_size_data = std::make_pair(owner->output_size, span);
          indices = owner->kind == mixed_join_size_kind::inner
                      ? cudf::mixed_inner_join(*owner->left_equality,
                                               *owner->right_equality,
                                               *owner->left_conditional,
                                               *owner->right_conditional,
                                               pred,
                                               owner->null_equality,
                                               output_size_data,
                                               exec.stream,
                                               exec.mr)
                      : cudf::mixed_left_join(*owner->left_equality,
                                              *owner->right_equality,
                                              *owner->left_conditional,
                                              *owner->right_conditional,
                                              pred,
                                              owner->null_equality,
                                              output_size_data,
                                              exec.stream,
                                              exec.mr);
          break;
        }
        case mixed_join_size_kind::full:
          CUDF_EXPECTS(owner->full_size_data != nullptr, "mixed full join size data missing state");
          indices = owner->full_size_data->materialize_indices(exec.stream, exec.mr);
          break;
        case mixed_join_size_kind::semi:
        case mixed_join_size_kind::anti:
          CUDF_FAIL("mixed semi/anti join size data materializes a single index column");
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

inline cudf_status_t mixed_join_size_data_materialize_single_indices_impl(
  cudf_mixed_join_size_data_handle size_data,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(size_data, "size_data is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner = static_cast<mixed_join_size_data_owner*>(size_data);
    CUDF_EXPECTS(owner->device == exec.device,
                 "mixed join size data must be materialized on its creation device");
    CUDF_EXPECTS(owner->kind == mixed_join_size_kind::semi || owner->kind == mixed_join_size_kind::anti,
                 "mixed pair join size data materializes two index columns");
    CUDF_EXPECTS(owner->single_size_data != nullptr, "mixed semi/anti join size data missing state");
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto indices = owner->single_size_data->materialize_indices(exec.stream, exec.mr);
      *out_indices =
        cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                   cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t filter_join_indices_impl(cudf_table_view_handle left,
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

inline cudf_status_t hash_join_size_impl(cudf_hash_join_handle join,
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

inline cudf_status_t filtered_join_create_impl(cudf_table_view_handle build,
                                        cudf_null_equality_t null_equality,
                                        double load_factor,
                                        cudf_filtered_join_handle* out,
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
    auto owner       = std::make_unique<filtered_join_owner>();
    owner->device    = exec.device;
    owner->upstream_mr_lifetime = exec.mr_lifetime;
    owner->observed_mr =
      std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(
        exec.mr);
    owner->mr_raw = owner->observed_mr.get();
    {
      cudf_current_device_resource_guard current_mr{owner->mr_raw};
      owner->join = std::make_unique<cudf::filtered_join>(
        *build_view, null_eq, cudf::set_as_build_table::RIGHT, load_factor, exec.stream);
      auto const bytes = owner->observed_mr->get_bytes_counter().value;
      owner->device_allocated_size_bytes = bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
    }
    *out = static_cast<cudf_filtered_join_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t filtered_join_size_impl(cudf_filtered_join_handle join,
                                      cudf_table_view_handle probe,
                                      size_t* out,
                                      cudf_exec_options_t const* options,
                                      filtered_join_probe_kind kind)
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

    auto* owner      = static_cast<filtered_join_owner*>(join);
    auto* probe_view = static_cast<cudf::table_view*>(probe);
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto indices = kind == filtered_join_probe_kind::semi
                       ? owner->join->semi_join(*probe_view, exec.stream, exec.mr)
                       : owner->join->anti_join(*probe_view, exec.stream, exec.mr);
      *out = indices->size();
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t filtered_join_indices_with_output_size_impl(cudf_filtered_join_handle join,
                                                          cudf_table_view_handle probe,
                                                          size_t output_size,
                                                          cudf_column_handle* out_indices,
                                                          cudf_exec_options_t const* options,
                                                          filtered_join_probe_kind kind)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<filtered_join_owner*>(join);
    auto* probe_view = static_cast<cudf::table_view*>(probe);
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto indices = kind == filtered_join_probe_kind::semi
                       ? owner->join->semi_join(*probe_view, exec.stream, exec.mr)
                       : owner->join->anti_join(*probe_view, exec.stream, exec.mr);
      CUDF_EXPECTS(indices->size() == output_size,
                   "filtered join output_size does not match produced index count",
                   std::invalid_argument);
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t filtered_join_begin_probe_impl(cudf_filtered_join_handle join,
                                             cudf_table_view_handle probe,
                                             cudf_filtered_join_probe_handle* out,
                                             cudf_exec_options_t const* options,
                                             filtered_join_probe_kind kind)
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

    auto* owner      = static_cast<filtered_join_owner*>(join);
    auto* probe_view = static_cast<cudf::table_view*>(probe);
    auto probe_owner = std::make_unique<filtered_join_probe_owner>();
    probe_owner->device               = exec.device;
    probe_owner->upstream_mr_lifetime = exec.mr_lifetime;
    probe_owner->observed_mr =
      std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(
        exec.mr);
    probe_owner->mr_raw = probe_owner->observed_mr.get();
    {
      cudf_current_device_resource_guard current_mr{probe_owner->mr_raw};
      auto const observed_mr_ref = cudf::get_current_device_resource_ref();
      probe_owner->probe = kind == filtered_join_probe_kind::semi
                             ? owner->join->begin_left_semi_probe(*probe_view, exec.stream, observed_mr_ref)
                             : owner->join->begin_left_anti_probe(*probe_view, exec.stream, observed_mr_ref);
      auto const bytes = probe_owner->observed_mr->get_bytes_counter().value;
      probe_owner->device_allocated_size_bytes = bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
    }
    *out = static_cast<cudf_filtered_join_probe_handle>(probe_owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

inline cudf_status_t filtered_join_probe_materialize_indices_impl(
  cudf_filtered_join_probe_handle handle,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "filtered join probe is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner = static_cast<filtered_join_probe_owner*>(handle);
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto indices = owner->probe->materialize_indices(exec.stream, exec.mr);
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // namespace cudf_rust::join
