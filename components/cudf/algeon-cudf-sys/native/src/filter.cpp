#include "cudf_rust/filter.h"

#include "execution_context_internal.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
[[nodiscard]] cudf::ast::ast_operator to_ast_binary_operator(cudf_binary_operator_t op)
{
  switch (op) {
    case CUDF_BINARY_LOGICAL_AND: return cudf::ast::ast_operator::LOGICAL_AND;
    case CUDF_BINARY_LOGICAL_OR: return cudf::ast::ast_operator::LOGICAL_OR;
    case CUDF_BINARY_NULL_LOGICAL_AND: return cudf::ast::ast_operator::NULL_LOGICAL_AND;
    case CUDF_BINARY_NULL_LOGICAL_OR: return cudf::ast::ast_operator::NULL_LOGICAL_OR;
    case CUDF_BINARY_EQUAL: return cudf::ast::ast_operator::EQUAL;
    case CUDF_BINARY_NOT_EQUAL: return cudf::ast::ast_operator::NOT_EQUAL;
    case CUDF_BINARY_LESS: return cudf::ast::ast_operator::LESS;
    case CUDF_BINARY_GREATER: return cudf::ast::ast_operator::GREATER;
    case CUDF_BINARY_LESS_EQUAL: return cudf::ast::ast_operator::LESS_EQUAL;
    case CUDF_BINARY_GREATER_EQUAL: return cudf::ast::ast_operator::GREATER_EQUAL;
    default: throw std::invalid_argument("unsupported row-predicate binary operator");
  }
}

[[nodiscard]] cudf::ast::ast_operator to_ast_unary_operator(cudf_unary_operator_t op)
{
  switch (op) {
    case CUDF_UNARY_NOT: return cudf::ast::ast_operator::NOT;
    default: throw std::invalid_argument("unsupported row-predicate unary operator");
  }
}

[[nodiscard]] cudf::ast::expression const& emplace_literal(cudf::ast::tree& exprs,
                                                           cudf::scalar& scalar)
{
  switch (scalar.type().id()) {
    case cudf::type_id::INT8:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int8_t>&>(scalar));
    case cudf::type_id::INT16:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int16_t>&>(scalar));
    case cudf::type_id::INT32:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int32_t>&>(scalar));
    case cudf::type_id::INT64:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int64_t>&>(scalar));
    case cudf::type_id::UINT8:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint8_t>&>(scalar));
    case cudf::type_id::UINT16:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::numeric_scalar<uint16_t>&>(scalar));
    case cudf::type_id::UINT32:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::numeric_scalar<uint32_t>&>(scalar));
    case cudf::type_id::UINT64:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::numeric_scalar<uint64_t>&>(scalar));
    case cudf::type_id::FLOAT32:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<float>&>(scalar));
    case cudf::type_id::FLOAT64:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<double>&>(scalar));
    case cudf::type_id::BOOL8:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<bool>&>(scalar));
    case cudf::type_id::STRING:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::string_scalar&>(scalar));
    case cudf::type_id::DECIMAL32:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal32>&>(scalar));
    case cudf::type_id::DECIMAL64:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal64>&>(scalar));
    case cudf::type_id::DECIMAL128:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal128>&>(scalar));
    case cudf::type_id::TIMESTAMP_DAYS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_D>&>(scalar));
    case cudf::type_id::TIMESTAMP_SECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_s>&>(scalar));
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ms>&>(scalar));
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_us>&>(scalar));
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ns>&>(scalar));
    case cudf::type_id::DURATION_DAYS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::duration_scalar<cudf::duration_D>&>(scalar));
    case cudf::type_id::DURATION_SECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::duration_scalar<cudf::duration_s>&>(scalar));
    case cudf::type_id::DURATION_MILLISECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::duration_scalar<cudf::duration_ms>&>(scalar));
    case cudf::type_id::DURATION_MICROSECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::duration_scalar<cudf::duration_us>&>(scalar));
    case cudf::type_id::DURATION_NANOSECONDS:
      return exprs.emplace<cudf::ast::literal>(
        static_cast<cudf::duration_scalar<cudf::duration_ns>&>(scalar));
    default: throw std::invalid_argument("unsupported row-predicate literal scalar type");
  }
}

[[nodiscard]] cudf::ast::expression const& resolve_operand(
  std::vector<cudf::ast::expression const*> const& lowered, int32_t operand_index)
{
  auto const* expr = lowered.at(static_cast<std::size_t>(operand_index));
  if (expr == nullptr) {
    throw std::invalid_argument("row-predicate operand references an unresolved node");
  }
  return *expr;
}

// A mask whose rows are all valid and true selects every input row, so the
// caller can keep the input columns instead of receiving a copy of them.
[[nodiscard]] bool mask_keeps_every_row(cudf::column_view const& mask,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (mask.is_empty()) { return true; }
  if (mask.has_nulls()) { return false; }
  // A non-empty mask without nulls makes the ALL reduction valid.
  auto const all =
    cudf::reduce(mask, *cudf::make_all_aggregation<cudf::reduce_aggregation>(), mask.type(), stream, mr);
  return static_cast<cudf::numeric_scalar<bool> const&>(*all).value(stream);
}

// Filters `input` by `mask` unless the mask keeps every row; then `*out` stays
// null and `*out_all_rows_pass` is 1.
void filter_by_mask_unless_all_rows_pass(cudf::table_view const& input,
                                         cudf::column_view const& mask,
                                         cudf_exec_context const& exec,
                                         cudf_table_handle* out,
                                         int32_t* out_all_rows_pass)
{
  if (mask_keeps_every_row(mask, exec.stream, *exec.mr)) {
    *out_all_rows_pass = 1;
    return;
  }
  auto result = cudf::apply_boolean_mask(input, mask, exec.stream, *exec.mr);
  *out        = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(exec));
}
}  // namespace

extern "C" {
cudf_status_t cudf_apply_boolean_mask_ex(cudf_table_view_handle input,
                                         cudf_column_view_handle mask,
                                         cudf_table_handle* out,
                                         int32_t* out_all_rows_pass,
                                         cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(out_all_rows_pass, "out_all_rows_pass is null");
  *out_all_rows_pass = 0;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(mask, "mask is null");
  CUDF_RUST_TRY
  {
    auto* view                      = reinterpret_cast<cudf::table_view*>(input);
    auto* mask_view                 = reinterpret_cast<cudf::column_view*>(mask);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    filter_by_mask_unless_all_rows_pass(*view, *mask_view, *exec, out, out_all_rows_pass);
    if (*out_all_rows_pass == 0) { cudf::rust::test_support::record_apply_boolean_mask(); }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_filter_rows_by_predicate_ex(cudf_table_view_handle input,
                                               cudf_row_predicate_node_t const* nodes,
                                               int32_t num_nodes,
                                               int32_t root_index,
                                               cudf_table_handle* out,
                                               int32_t* out_all_rows_pass,
                                               cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(out_all_rows_pass, "out_all_rows_pass is null");
  *out_all_rows_pass = 0;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(nodes, "nodes is null");
  CUDF_RUST_TRY
  {
    if (num_nodes <= 0) {
      set_error("row-predicate node array must not be empty");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (root_index < 0 || root_index >= num_nodes) {
      set_error("row-predicate root_index is out of range");
      return CUDF_ERROR_INDEX;
    }

    auto* view                      = reinterpret_cast<cudf::table_view*>(input);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::ast::tree exprs;
    std::vector<cudf::ast::expression const*> lowered(static_cast<std::size_t>(num_nodes), nullptr);

    for (int32_t i = 0; i < num_nodes; ++i) {
      auto const& node = nodes[i];
      switch (node.kind) {
        case CUDF_ROW_PREDICATE_NODE_COLUMN_REF:
          if (node.column_index < 0 || node.column_index >= view->num_columns()) {
            set_error("row-predicate column_index is out of range");
            return CUDF_ERROR_INDEX;
          }
          lowered[static_cast<std::size_t>(i)] = &exprs.emplace<cudf::ast::column_reference>(
            static_cast<cudf::size_type>(node.column_index));
          break;
        case CUDF_ROW_PREDICATE_NODE_LITERAL:
          CUDF_RUST_NULL_CHECK(node.literal, "row-predicate literal handle is null");
          lowered[static_cast<std::size_t>(i)] =
            &emplace_literal(exprs, cudf_scalar_from_handle(node.literal));
          break;
        case CUDF_ROW_PREDICATE_NODE_BINARY_OP:
          if (node.lhs_index < 0 || node.lhs_index >= i || node.rhs_index < 0 ||
              node.rhs_index >= i) {
            set_error("row-predicate binary operands must reference earlier nodes");
            return CUDF_ERROR_INDEX;
          }
          lowered[static_cast<std::size_t>(i)] =
            &exprs.emplace<cudf::ast::operation>(to_ast_binary_operator(node.binary_op),
                                                 resolve_operand(lowered, node.lhs_index),
                                                 resolve_operand(lowered, node.rhs_index));
          break;
        case CUDF_ROW_PREDICATE_NODE_UNARY_OP:
          if (node.lhs_index < 0 || node.lhs_index >= i) {
            set_error("row-predicate unary operand must reference an earlier node");
            return CUDF_ERROR_INDEX;
          }
          lowered[static_cast<std::size_t>(i)] = &exprs.emplace<cudf::ast::operation>(
            to_ast_unary_operator(node.unary_op), resolve_operand(lowered, node.lhs_index));
          break;
        case CUDF_ROW_PREDICATE_NODE_IS_NULL:
          if (node.lhs_index < 0 || node.lhs_index >= i) {
            set_error("row-predicate is-null operand must reference an earlier node");
            return CUDF_ERROR_INDEX;
          }
          lowered[static_cast<std::size_t>(i)] = &exprs.emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::IS_NULL, resolve_operand(lowered, node.lhs_index));
          break;
        default:
          set_error("unsupported row-predicate node kind");
          return CUDF_ERROR_INVALID_OPERATION;
      }
    }

    {
      cudf::rust::test_support::record_row_predicate_filter();
      cudf::rust::test_support::record_row_predicate_compute_mask();
      auto mask =
        cudf::compute_column(*view, resolve_operand(lowered, root_index), exec->stream, *exec->mr);
      if (mask->type().id() != cudf::type_id::BOOL8) {
        set_error("row predicate must evaluate to Bool8");
        return CUDF_ERROR_TYPE;
      }

      filter_by_mask_unless_all_rows_pass(*view, mask->view(), *exec, out, out_all_rows_pass);
      if (*out_all_rows_pass == 0) { cudf::rust::test_support::record_row_predicate_final_filter(); }
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
