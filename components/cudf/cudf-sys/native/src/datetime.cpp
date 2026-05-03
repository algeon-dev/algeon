#include "datetime.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

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
    default: set_error("invalid datetime component"); return CUDF_ERROR_INVALID_OPERATION;
  }
}
}  // namespace

extern "C" {
cudf_status_t cudf_datetime_extract_component(cudf_column_view_handle input,
                                              cudf_datetime_component_t component,
                                              cudf_column_handle* out)
{
  return cudf_datetime_extract_component_ex(input, component, out, nullptr);
}

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
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = static_cast<cudf::column_view*>(input);

    {
      auto result =
        cudf::datetime::extract_datetime_component(*view, cudf_component, exec.stream, exec.mr);
      *out = cudf_release_column_handle(std::move(result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
