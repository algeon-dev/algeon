#include "strings.h"
#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// Helper: cast a column_view handle to strings_column_view.
// Caller must guarantee the column is of type STRING.
static cudf::strings_column_view to_strings_view(cudf_column_view_handle h) {
  auto* view = static_cast<cudf::column_view*>(h);
  return cudf::strings_column_view{*view};
}

// Helper: validate that a scalar handle is a string scalar.
// Returns CUDF_SUCCESS on success, CUDF_ERROR_TYPE on type mismatch.
static cudf_status_t check_string_scalar(cudf_scalar_handle handle,
                                         cudf::string_scalar** out) {
  auto* s = static_cast<cudf::scalar*>(handle);
  if (s->type().id() != cudf::type_id::STRING) {
    set_error("scalar is not a string type");
    return CUDF_ERROR_TYPE;
  }
  *out = static_cast<cudf::string_scalar*>(s);
  return CUDF_SUCCESS;
}

extern "C" {

// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_to_lower(cudf_column_view_handle input,
                                    cudf_column_handle* out) {
  return cudf_strings_to_lower_ex(input, out, nullptr);
}

cudf_status_t cudf_strings_to_lower_ex(cudf_column_view_handle input,
                                       cudf_column_handle* out,
                                       cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto result = cudf::strings::to_lower(to_strings_view(input),
                                          exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_to_upper(cudf_column_view_handle input,
                                    cudf_column_handle* out) {
  return cudf_strings_to_upper_ex(input, out, nullptr);
}

cudf_status_t cudf_strings_to_upper_ex(cudf_column_view_handle input,
                                       cudf_column_handle* out,
                                       cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto result = cudf::strings::to_upper(to_strings_view(input),
                                          exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Find / contains / starts_with / ends_with (scalar target)
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_contains(cudf_column_view_handle input,
                                    cudf_scalar_handle target,
                                    cudf_column_handle* out) {
  return cudf_strings_contains_ex(input, target, out, nullptr);
}

cudf_status_t cudf_strings_contains_ex(cudf_column_view_handle input,
                                       cudf_scalar_handle target,
                                       cudf_column_handle* out,
                                       cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(target, "target is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::string_scalar* scalar = nullptr;
    cudf_status_t const type_status = check_string_scalar(target, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    auto result = cudf::strings::contains(to_strings_view(input),
                                          *scalar, exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_starts_with(cudf_column_view_handle input,
                                       cudf_scalar_handle target,
                                       cudf_column_handle* out) {
  return cudf_strings_starts_with_ex(input, target, out, nullptr);
}

cudf_status_t cudf_strings_starts_with_ex(cudf_column_view_handle input,
                                          cudf_scalar_handle target,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(target, "target is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::string_scalar* scalar = nullptr;
    cudf_status_t const type_status = check_string_scalar(target, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    auto result = cudf::strings::starts_with(to_strings_view(input),
                                             *scalar, exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_ends_with(cudf_column_view_handle input,
                                     cudf_scalar_handle target,
                                     cudf_column_handle* out) {
  return cudf_strings_ends_with_ex(input, target, out, nullptr);
}

cudf_status_t cudf_strings_ends_with_ex(cudf_column_view_handle input,
                                        cudf_scalar_handle target,
                                        cudf_column_handle* out,
                                        cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(target, "target is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::string_scalar* scalar = nullptr;
    cudf_status_t const type_status = check_string_scalar(target, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    auto result = cudf::strings::ends_with(to_strings_view(input),
                                           *scalar, exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Strip / trim
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_strip(cudf_column_view_handle input,
                                 cudf_strip_side_t side,
                                 cudf_scalar_handle to_strip,
                                 cudf_column_handle* out) {
  return cudf_strings_strip_ex(input, side, to_strip, out, nullptr);
}

cudf_status_t cudf_strings_strip_ex(cudf_column_view_handle input,
                                    cudf_strip_side_t side,
                                    cudf_scalar_handle to_strip,
                                    cudf_column_handle* out,
                                    cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(to_strip, "to_strip is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::strings::side_type cudf_side;
    switch (side) {
      case CUDF_STRIP_LEFT:  cudf_side = cudf::strings::side_type::LEFT; break;
      case CUDF_STRIP_RIGHT: cudf_side = cudf::strings::side_type::RIGHT; break;
      case CUDF_STRIP_BOTH:  cudf_side = cudf::strings::side_type::BOTH; break;
      default:
        set_error("invalid strip side type");
        return CUDF_ERROR_INVALID_OPERATION;
    }

    cudf::string_scalar* scalar = nullptr;
    cudf_status_t const type_status = check_string_scalar(to_strip, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    auto result = cudf::strings::strip(to_strings_view(input),
                                       cudf_side, *scalar,
                                       exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_count_characters(cudf_column_view_handle input,
                                            cudf_column_handle* out) {
  return cudf_strings_count_characters_ex(input, out, nullptr);
}

cudf_status_t cudf_strings_count_characters_ex(cudf_column_view_handle input,
                                               cudf_column_handle* out,
                                               cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto result = cudf::strings::count_characters(to_strings_view(input),
                                                  exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_count_bytes(cudf_column_view_handle input,
                                       cudf_column_handle* out) {
  return cudf_strings_count_bytes_ex(input, out, nullptr);
}

cudf_status_t cudf_strings_count_bytes_ex(cudf_column_view_handle input,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto result = cudf::strings::count_bytes(to_strings_view(input),
                                             exec.stream, exec.mr);
    *out = cudf_release_column_handle(std::move(result));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
