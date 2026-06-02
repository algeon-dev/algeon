#include "unary.h"
#include "shim_common.h"

#include <cudf/unary.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace {

cudf_status_t to_unary_operator(cudf_unary_operator_t input,
                                cudf::unary_operator* out) {
  if (input < CUDF_UNARY_SIN || input > CUDF_UNARY_NEGATE) {
    set_error("invalid unary operator");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = static_cast<cudf::unary_operator>(input);
  return CUDF_SUCCESS;
}

}  // namespace

extern "C" {

cudf_status_t cudf_unary_operation(cudf_column_view_handle input,
                                   cudf_unary_operator_t op,
                                   cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  cudf::unary_operator cudf_op;
  cudf_status_t status = to_unary_operator(op, &cudf_op);
  if (status != CUDF_SUCCESS) {
    return status;
  }
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::unary_operation(*view, cudf_op, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_null(cudf_column_view_handle input,
                           cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_null(*view, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_valid(cudf_column_view_handle input,
                            cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_valid(*view, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_cast(cudf_column_view_handle input,
                        cudf_data_type_handle output_type,
                        cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(output_type, "output_type is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto* dtype = static_cast<cudf::data_type*>(output_type);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::cast(*view, *dtype, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_supported_cast(cudf_data_type_handle from_type,
                                     cudf_data_type_handle to_type,
                                     int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(from_type, "from_type is null");
  CUDF_RUST_NULL_CHECK(to_type, "to_type is null");
  CUDF_RUST_TRY {
    auto* from_dt = static_cast<cudf::data_type*>(from_type);
    auto* to_dt = static_cast<cudf::data_type*>(to_type);

    *out = cudf::is_supported_cast(*from_dt, *to_dt) ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_nan(cudf_column_view_handle input,
                          cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_nan(*view, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_is_not_nan(cudf_column_view_handle input,
                              cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_not_nan(*view, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
