#include "cudf_rust/ast.h"

#include "ast_handle.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
[[nodiscard]] cudf::ast::table_reference to_table_reference(cudf_ast_table_reference_t input)
{
  switch (input) {
    case CUDF_AST_TABLE_REFERENCE_LEFT: return cudf::ast::table_reference::LEFT;
    case CUDF_AST_TABLE_REFERENCE_RIGHT: return cudf::ast::table_reference::RIGHT;
    case CUDF_AST_TABLE_REFERENCE_OUTPUT: return cudf::ast::table_reference::OUTPUT;
    default: throw std::invalid_argument("invalid AST table reference");
  }
}

[[nodiscard]] cudf::ast::ast_operator to_ast_operator(cudf_ast_operator_t input)
{
  auto const raw = static_cast<int32_t>(input);
  if (raw < static_cast<int32_t>(CUDF_AST_ADD) ||
      raw > static_cast<int32_t>(CUDF_AST_CAST_TO_FLOAT64)) {
    throw std::invalid_argument("invalid AST operator");
  }
  return static_cast<cudf::ast::ast_operator>(raw);
}

template <typename Scalar>
std::unique_ptr<cudf::ast::expression> make_literal(cudf::scalar& scalar)
{
  return std::make_unique<cudf::ast::literal>(static_cast<Scalar&>(scalar));
}

[[nodiscard]] std::unique_ptr<cudf::ast::expression> make_literal_expression(cudf::scalar& scalar)
{
  switch (scalar.type().id()) {
    case cudf::type_id::INT8: return make_literal<cudf::numeric_scalar<int8_t>>(scalar);
    case cudf::type_id::INT16: return make_literal<cudf::numeric_scalar<int16_t>>(scalar);
    case cudf::type_id::INT32: return make_literal<cudf::numeric_scalar<int32_t>>(scalar);
    case cudf::type_id::INT64: return make_literal<cudf::numeric_scalar<int64_t>>(scalar);
    case cudf::type_id::UINT8: return make_literal<cudf::numeric_scalar<uint8_t>>(scalar);
    case cudf::type_id::UINT16: return make_literal<cudf::numeric_scalar<uint16_t>>(scalar);
    case cudf::type_id::UINT32: return make_literal<cudf::numeric_scalar<uint32_t>>(scalar);
    case cudf::type_id::UINT64: return make_literal<cudf::numeric_scalar<uint64_t>>(scalar);
    case cudf::type_id::FLOAT32: return make_literal<cudf::numeric_scalar<float>>(scalar);
    case cudf::type_id::FLOAT64: return make_literal<cudf::numeric_scalar<double>>(scalar);
    case cudf::type_id::BOOL8: return make_literal<cudf::numeric_scalar<bool>>(scalar);
    case cudf::type_id::STRING: return make_literal<cudf::string_scalar>(scalar);
    case cudf::type_id::DECIMAL32:
      return make_literal<cudf::fixed_point_scalar<numeric::decimal32>>(scalar);
    case cudf::type_id::DECIMAL64:
      return make_literal<cudf::fixed_point_scalar<numeric::decimal64>>(scalar);
    case cudf::type_id::DECIMAL128:
      return make_literal<cudf::fixed_point_scalar<numeric::decimal128>>(scalar);
    case cudf::type_id::TIMESTAMP_DAYS:
      return make_literal<cudf::timestamp_scalar<cudf::timestamp_D>>(scalar);
    case cudf::type_id::TIMESTAMP_SECONDS:
      return make_literal<cudf::timestamp_scalar<cudf::timestamp_s>>(scalar);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return make_literal<cudf::timestamp_scalar<cudf::timestamp_ms>>(scalar);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return make_literal<cudf::timestamp_scalar<cudf::timestamp_us>>(scalar);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return make_literal<cudf::timestamp_scalar<cudf::timestamp_ns>>(scalar);
    case cudf::type_id::DURATION_DAYS:
      return make_literal<cudf::duration_scalar<cudf::duration_D>>(scalar);
    case cudf::type_id::DURATION_SECONDS:
      return make_literal<cudf::duration_scalar<cudf::duration_s>>(scalar);
    case cudf::type_id::DURATION_MILLISECONDS:
      return make_literal<cudf::duration_scalar<cudf::duration_ms>>(scalar);
    case cudf::type_id::DURATION_MICROSECONDS:
      return make_literal<cudf::duration_scalar<cudf::duration_us>>(scalar);
    case cudf::type_id::DURATION_NANOSECONDS:
      return make_literal<cudf::duration_scalar<cudf::duration_ns>>(scalar);
    default: throw std::invalid_argument("unsupported AST literal scalar type");
  }
}
}  // namespace

extern "C" {
cudf_status_t cudf_ast_column_reference_create(int32_t column_index,
                                               cudf_ast_table_reference_t table_source,
                                               cudf_ast_expression_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    if (column_index < 0) {
      set_error("AST column_index must be non-negative");
      return CUDF_ERROR_INDEX;
    }
    auto expression = std::make_unique<cudf::ast::column_reference>(
      static_cast<cudf::size_type>(column_index), to_table_reference(table_source));
    *out = cudf_release_ast_expression_handle(std::move(expression));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_ast_literal_create(cudf_scalar_handle scalar, cudf_ast_expression_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(scalar, "scalar is null");
  CUDF_RUST_TRY
  {
    auto expression = make_literal_expression(cudf_scalar_from_handle(scalar));
    *out            = cudf_release_ast_expression_handle(std::move(expression));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_ast_operation_unary_create(cudf_ast_operator_t op,
                                              cudf_ast_expression_handle input,
                                              cudf_ast_expression_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto child      = cudf_ast_expression_owner_ref_from_handle(input);
    auto expression = std::make_unique<cudf::ast::operation>(
      to_ast_operator(op), cudf_ast_expression_from_handle(input));
    *out = cudf_release_ast_expression_handle(std::move(expression), {child});
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_ast_operation_binary_create(cudf_ast_operator_t op,
                                               cudf_ast_expression_handle left,
                                               cudf_ast_expression_handle right,
                                               cudf_ast_expression_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(left, "left is null");
  CUDF_RUST_NULL_CHECK(right, "right is null");
  CUDF_RUST_TRY
  {
    auto left_child  = cudf_ast_expression_owner_ref_from_handle(left);
    auto right_child = cudf_ast_expression_owner_ref_from_handle(right);
    auto expression =
      std::make_unique<cudf::ast::operation>(to_ast_operator(op),
                                             cudf_ast_expression_from_handle(left),
                                             cudf_ast_expression_from_handle(right));
    *out = cudf_release_ast_expression_handle(std::move(expression), {left_child, right_child});
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_ast_expression_destroy(cudf_ast_expression_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  try {
    delete reinterpret_cast<cudf_ast_expression_owner_ref*>(handle);
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}
}  // extern "C"
