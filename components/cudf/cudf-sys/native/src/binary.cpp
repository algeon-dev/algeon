#include "binary.h"

#include "execution_context_internal.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace {
cudf_status_t to_binary_operator(cudf_binary_operator_t input, cudf::binary_operator* out)
{
  if (input < CUDF_BINARY_ADD || input > CUDF_BINARY_NULL_MIN) {
    set_error("invalid binary operator");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = static_cast<cudf::binary_operator>(input);
  return CUDF_SUCCESS;
}
}  // namespace

extern "C" {
cudf_status_t cudf_binary_operation_vv(cudf_column_view_handle lhs,
                                       cudf_column_view_handle rhs,
                                       cudf_binary_operator_t op,
                                       cudf_data_type_handle output_type,
                                       cudf_column_handle* out)
{
  return cudf_binary_operation_vv_ex(lhs, rhs, op, output_type, out, nullptr);
}

cudf_status_t cudf_binary_operation_vv_ex(cudf_column_view_handle lhs,
                                          cudf_column_view_handle rhs,
                                          cudf_binary_operator_t op,
                                          cudf_data_type_handle output_type,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(lhs, "lhs is null");
  CUDF_RUST_NULL_CHECK(rhs, "rhs is null");
  CUDF_RUST_NULL_CHECK(output_type, "output_type is null");
  cudf::binary_operator cudf_op;
  cudf_status_t status = to_binary_operator(op, &cudf_op);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lhs_view = static_cast<cudf::column_view*>(lhs);
    auto* rhs_view = static_cast<cudf::column_view*>(rhs);
    auto* dtype    = static_cast<cudf::data_type*>(output_type);

    {
      auto result =
        cudf::binary_operation(*lhs_view, *rhs_view, cudf_op, *dtype, exec.stream, exec.mr);
      *out = cudf_release_column_handle(std::move(result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_binary_operation_vs(cudf_column_view_handle lhs,
                                       cudf_scalar_handle rhs,
                                       cudf_binary_operator_t op,
                                       cudf_data_type_handle output_type,
                                       cudf_column_handle* out)
{
  return cudf_binary_operation_vs_ex(lhs, rhs, op, output_type, out, nullptr);
}

cudf_status_t cudf_binary_operation_vs_ex(cudf_column_view_handle lhs,
                                          cudf_scalar_handle rhs,
                                          cudf_binary_operator_t op,
                                          cudf_data_type_handle output_type,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(lhs, "lhs is null");
  CUDF_RUST_NULL_CHECK(rhs, "rhs is null");
  CUDF_RUST_NULL_CHECK(output_type, "output_type is null");
  cudf::binary_operator cudf_op;
  cudf_status_t status = to_binary_operator(op, &cudf_op);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lhs_view   = static_cast<cudf::column_view*>(lhs);
    auto& rhs_scalar = cudf_scalar_from_handle(rhs);
    auto* dtype      = static_cast<cudf::data_type*>(output_type);

    {
      auto result =
        cudf::binary_operation(*lhs_view, rhs_scalar, cudf_op, *dtype, exec.stream, exec.mr);
      *out = cudf_release_column_handle(std::move(result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_binary_operation_sv(cudf_scalar_handle lhs,
                                       cudf_column_view_handle rhs,
                                       cudf_binary_operator_t op,
                                       cudf_data_type_handle output_type,
                                       cudf_column_handle* out)
{
  return cudf_binary_operation_sv_ex(lhs, rhs, op, output_type, out, nullptr);
}

cudf_status_t cudf_binary_operation_sv_ex(cudf_scalar_handle lhs,
                                          cudf_column_view_handle rhs,
                                          cudf_binary_operator_t op,
                                          cudf_data_type_handle output_type,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(lhs, "lhs is null");
  CUDF_RUST_NULL_CHECK(rhs, "rhs is null");
  CUDF_RUST_NULL_CHECK(output_type, "output_type is null");
  cudf::binary_operator cudf_op;
  cudf_status_t status = to_binary_operator(op, &cudf_op);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto& lhs_scalar = cudf_scalar_from_handle(lhs);
    auto* rhs_view   = static_cast<cudf::column_view*>(rhs);
    auto* dtype      = static_cast<cudf::data_type*>(output_type);

    {
      auto result =
        cudf::binary_operation(lhs_scalar, *rhs_view, cudf_op, *dtype, exec.stream, exec.mr);
      *out = cudf_release_column_handle(std::move(result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
