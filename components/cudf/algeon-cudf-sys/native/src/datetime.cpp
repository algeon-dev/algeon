#include "cudf_rust/datetime.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

namespace {
cudf_status_t to_datetime_component(cudf_datetime_component_t input,
                                    cudf::datetime::datetime_component* out)
{
  switch (input) {
    case CUDF_DATETIME_COMPONENT_YEAR:
      *out = cudf::datetime::datetime_component::YEAR;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_COMPONENT_MONTH:
      *out = cudf::datetime::datetime_component::MONTH;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_COMPONENT_DAY:
      *out = cudf::datetime::datetime_component::DAY;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_COMPONENT_HOUR:
      *out = cudf::datetime::datetime_component::HOUR;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_COMPONENT_MINUTE:
      *out = cudf::datetime::datetime_component::MINUTE;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_COMPONENT_SECOND:
      *out = cudf::datetime::datetime_component::SECOND;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_COMPONENT_MILLISECOND:
      *out = cudf::datetime::datetime_component::MILLISECOND;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_COMPONENT_MICROSECOND:
      *out = cudf::datetime::datetime_component::MICROSECOND;
      return CUDF_SUCCESS;
    default: set_error("invalid datetime component"); return CUDF_ERROR_INVALID_OPERATION;
  }
}

cudf_status_t to_datetime_rounding_frequency(cudf_datetime_rounding_frequency_t input,
                                             cudf::datetime::rounding_frequency* out)
{
  switch (input) {
    case CUDF_DATETIME_ROUNDING_DAY:
      *out = cudf::datetime::rounding_frequency::DAY;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_ROUNDING_HOUR:
      *out = cudf::datetime::rounding_frequency::HOUR;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_ROUNDING_MINUTE:
      *out = cudf::datetime::rounding_frequency::MINUTE;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_ROUNDING_SECOND:
      *out = cudf::datetime::rounding_frequency::SECOND;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_ROUNDING_MILLISECOND:
      *out = cudf::datetime::rounding_frequency::MILLISECOND;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_ROUNDING_MICROSECOND:
      *out = cudf::datetime::rounding_frequency::MICROSECOND;
      return CUDF_SUCCESS;
    case CUDF_DATETIME_ROUNDING_NANOSECOND:
      *out = cudf::datetime::rounding_frequency::NANOSECOND;
      return CUDF_SUCCESS;
    default: set_error("invalid datetime rounding frequency"); return CUDF_ERROR_INVALID_OPERATION;
  }
}
}  // namespace

extern "C" {
cudf_status_t cudf_datetime_extract_component_ex(cudf_column_view_handle input,
                                                 cudf_datetime_component_t component,
                                                 cudf_column_handle* out,
                                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  cudf::datetime::datetime_component cudf_component;
  cudf_status_t const component_status = to_datetime_component(component, &cudf_component);
  if (component_status != CUDF_SUCCESS) { return component_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = reinterpret_cast<cudf::column_view*>(input);

    {
      auto result =
        cudf::datetime::extract_datetime_component(*view, cudf_component, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_datetime_floor_ex(cudf_column_view_handle input,
                                     cudf_datetime_rounding_frequency_t frequency,
                                     cudf_column_handle* out,
                                     cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  cudf::datetime::rounding_frequency cudf_frequency;
  cudf_status_t const frequency_status = to_datetime_rounding_frequency(frequency, &cudf_frequency);
  if (frequency_status != CUDF_SUCCESS) { return frequency_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = reinterpret_cast<cudf::column_view*>(input);

    {
      auto result = cudf::datetime::floor_datetimes(*view, cudf_frequency, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_datetime_add_calendrical_months_scalar_ex(cudf_column_view_handle input,
                                                             int32_t months,
                                                             cudf_column_handle* out,
                                                             cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = reinterpret_cast<cudf::column_view*>(input);
    {
      cudf::numeric_scalar<int32_t> months_scalar(months, true, exec->stream, *exec->mr);
      auto result =
        cudf::datetime::add_calendrical_months(*view, months_scalar, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
