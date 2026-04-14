#include "execution_context_internal.hpp"
#include "filter.h"
#include "shim_common.h"

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

thread_local int32_t g_apply_boolean_mask_call_count          = 0;
thread_local int32_t g_row_predicate_filter_call_count       = 0;
thread_local int32_t g_row_predicate_compute_mask_call_count = 0;
thread_local int32_t g_row_predicate_final_filter_call_count = 0;

[[nodiscard]] cudf::ast::ast_operator to_ast_binary_operator(cudf_binary_operator_t op)
{
  switch (op) {
    case CUDF_BINARY_LOGICAL_AND: return cudf::ast::ast_operator::LOGICAL_AND;
    case CUDF_BINARY_LOGICAL_OR: return cudf::ast::ast_operator::LOGICAL_OR;
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

[[nodiscard]] cudf::ast::expression const& emplace_literal(cudf::ast::tree& exprs, cudf::scalar& scalar)
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
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint16_t>&>(scalar));
    case cudf::type_id::UINT32:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint32_t>&>(scalar));
    case cudf::type_id::UINT64:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint64_t>&>(scalar));
    case cudf::type_id::FLOAT32:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<float>&>(scalar));
    case cudf::type_id::FLOAT64:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<double>&>(scalar));
    case cudf::type_id::BOOL8:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<bool>&>(scalar));
    case cudf::type_id::STRING:
      return exprs.emplace<cudf::ast::literal>(static_cast<cudf::string_scalar&>(scalar));
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

}  // namespace

extern "C" {

cudf_status_t cudf_apply_boolean_mask(cudf_table_view_handle input,
                                      cudf_column_view_handle mask,
                                      cudf_table_handle* out) {
  return cudf_apply_boolean_mask_ex(input, mask, out, nullptr);
}

cudf_status_t cudf_apply_boolean_mask_ex(cudf_table_view_handle input,
                                         cudf_column_view_handle mask,
                                         cudf_table_handle* out,
                                         cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(mask, "mask is null");
  CUDF_RUST_TRY {
    ++g_apply_boolean_mask_call_count;
    auto* view      = static_cast<cudf::table_view*>(input);
    auto* mask_view = static_cast<cudf::column_view*>(mask);
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto result = cudf::apply_boolean_mask(*view, *mask_view, exec.stream, exec.mr);
    *out        = cudf_release_table_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_filter_rows_by_predicate(cudf_table_view_handle input,
                                            cudf_row_predicate_node_t const* nodes,
                                            int32_t num_nodes,
                                            int32_t root_index,
                                            cudf_table_handle* out) {
  return cudf_filter_rows_by_predicate_ex(input, nodes, num_nodes, root_index, out, nullptr);
}

cudf_status_t cudf_filter_rows_by_predicate_ex(cudf_table_view_handle input,
                                               cudf_row_predicate_node_t const* nodes,
                                               int32_t num_nodes,
                                               int32_t root_index,
                                               cudf_table_handle* out,
                                               cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(nodes, "nodes is null");
  CUDF_RUST_TRY {
    if (num_nodes <= 0) {
      set_error("row-predicate node array must not be empty");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (root_index < 0 || root_index >= num_nodes) {
      set_error("row-predicate root_index is out of range");
      return CUDF_ERROR_INDEX;
    }

    auto* view = static_cast<cudf::table_view*>(input);
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

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
          lowered[static_cast<std::size_t>(i)] =
            &exprs.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(node.column_index));
          break;
        case CUDF_ROW_PREDICATE_NODE_LITERAL:
          CUDF_RUST_NULL_CHECK(node.literal, "row-predicate literal handle is null");
          lowered[static_cast<std::size_t>(i)] = &emplace_literal(exprs, *static_cast<cudf::scalar*>(node.literal));
          break;
        case CUDF_ROW_PREDICATE_NODE_BINARY_OP:
          if (node.lhs_index < 0 || node.lhs_index >= i || node.rhs_index < 0 || node.rhs_index >= i) {
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
          lowered[static_cast<std::size_t>(i)] =
            &exprs.emplace<cudf::ast::operation>(to_ast_unary_operator(node.unary_op),
                                                 resolve_operand(lowered, node.lhs_index));
          break;
        case CUDF_ROW_PREDICATE_NODE_IS_NULL:
          if (node.lhs_index < 0 || node.lhs_index >= i) {
            set_error("row-predicate is-null operand must reference an earlier node");
            return CUDF_ERROR_INDEX;
          }
          lowered[static_cast<std::size_t>(i)] =
            &exprs.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL,
                                                 resolve_operand(lowered, node.lhs_index));
          break;
        default:
          set_error("unsupported row-predicate node kind");
          return CUDF_ERROR_INVALID_OPERATION;
      }
    }

    ++g_row_predicate_filter_call_count;
    ++g_row_predicate_compute_mask_call_count;
    auto mask = cudf::compute_column(*view, resolve_operand(lowered, root_index), exec.stream, exec.mr);
    if (mask->type().id() != cudf::type_id::BOOL8) {
      set_error("row predicate must evaluate to Bool8");
      return CUDF_ERROR_TYPE;
    }

    ++g_row_predicate_final_filter_call_count;
    auto result = cudf::apply_boolean_mask(*view, mask->view(), exec.stream, exec.mr);
    *out        = cudf_release_table_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_test_row_predicate_reset_observation() {
  g_apply_boolean_mask_call_count          = 0;
  g_row_predicate_filter_call_count       = 0;
  g_row_predicate_compute_mask_call_count = 0;
  g_row_predicate_final_filter_call_count = 0;
}

cudf_status_t cudf_test_row_predicate_filter_call_count(int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_row_predicate_filter_call_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_row_predicate_compute_mask_call_count(int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_row_predicate_compute_mask_call_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_row_predicate_final_filter_call_count(int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_row_predicate_final_filter_call_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_apply_boolean_mask_call_count(int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_apply_boolean_mask_call_count;
  return CUDF_SUCCESS;
}

}  // extern "C"
