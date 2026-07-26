#include "strings.h"

#include "execution_context_internal.hpp"
#include "handle_provenance.hpp"
#include "reservation_accounting.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Helper: cast a column_view handle to strings_column_view.
// Caller must guarantee the column is of type STRING.
static cudf::strings_column_view to_strings_view(cudf_column_view_handle h)
{
  auto* view = static_cast<cudf::column_view*>(h);
  return cudf::strings_column_view{*view};
}

// Helper: validate that a scalar handle is a string scalar.
// Returns CUDF_SUCCESS on success, CUDF_ERROR_TYPE on type mismatch.
static cudf_status_t check_string_scalar(cudf_scalar_handle handle, cudf::string_scalar** out)
{
  auto& s = cudf_scalar_from_handle(handle);
  if (s.type().id() != cudf::type_id::STRING) {
    set_error("scalar is not a string type");
    return CUDF_ERROR_TYPE;
  }
  *out = static_cast<cudf::string_scalar*>(&s);
  return CUDF_SUCCESS;
}

static cudf_status_t validate_like_arguments(char const* pattern,
                                             int32_t pattern_len,
                                             char const* escape_character,
                                             int32_t escape_character_len)
{
  if (pattern_len < 0) {
    set_error("pattern_len must be non-negative");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (escape_character_len < 0) {
    set_error("escape_character_len must be non-negative");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (escape_character_len > 1) {
    set_error("escape_character_len must be 0 or 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (pattern_len > 0 && pattern == nullptr) {
    set_error("pattern is null");
    return CUDF_ERROR_NULL;
  }
  if (escape_character_len > 0 && escape_character == nullptr) {
    set_error("escape_character is null");
    return CUDF_ERROR_NULL;
  }
  return CUDF_SUCCESS;
}

static cudf_status_t validate_string_range(char const* value, int32_t value_len, char const* value_name)
{
  if (value_len < 0) {
    set_error(std::string{value_name} + "_len must be non-negative");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (value_len > 0 && value == nullptr) {
    set_error(std::string{value_name} + " is null");
    return CUDF_ERROR_NULL;
  }
  return CUDF_SUCCESS;
}

static std::string_view make_host_string_view(char const* ptr, int32_t len)
{
  return len == 0 ? std::string_view{} : std::string_view{ptr, static_cast<std::size_t>(len)};
}

static std::size_t checked_add_reservation_bytes(std::size_t left, std::size_t right)
{
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error{"replace-with-backrefs reservation bound overflowed size_t"};
  }
  return left + right;
}

extern "C" {
// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_to_lower_ex(cudf_column_view_handle input,
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

    {
      auto result = cudf::strings::to_lower(to_strings_view(input), exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_to_upper_ex(cudf_column_view_handle input,
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

    {
      auto result = cudf::strings::to_upper(to_strings_view(input), exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Find / contains / starts_with / ends_with (scalar target)
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_contains_ex(cudf_column_view_handle input,
                                       cudf_scalar_handle target,
                                       cudf_column_handle* out,
                                       cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(target, "target is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::string_scalar* scalar     = nullptr;
    cudf_status_t const type_status = check_string_scalar(target, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    {
      auto result = cudf::strings::contains(to_strings_view(input), *scalar, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_starts_with_ex(cudf_column_view_handle input,
                                          cudf_scalar_handle target,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(target, "target is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::string_scalar* scalar     = nullptr;
    cudf_status_t const type_status = check_string_scalar(target, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    {
      auto result =
        cudf::strings::starts_with(to_strings_view(input), *scalar, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_ends_with_ex(cudf_column_view_handle input,
                                        cudf_scalar_handle target,
                                        cudf_column_handle* out,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(target, "target is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::string_scalar* scalar     = nullptr;
    cudf_status_t const type_status = check_string_scalar(target, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    {
      auto result = cudf::strings::ends_with(to_strings_view(input), *scalar, exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// LIKE
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_like_ex(cudf_column_view_handle input,
                                   char const* pattern,
                                   int32_t pattern_len,
                                   char const* escape_character,
                                   int32_t escape_character_len,
                                   cudf_column_handle* out,
                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  cudf_status_t const arg_status =
    validate_like_arguments(pattern, pattern_len, escape_character, escape_character_len);
  if (arg_status != CUDF_SUCCESS) { return arg_status; }

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const pattern_view = make_host_string_view(pattern, pattern_len);
    auto const escape_view  = make_host_string_view(escape_character, escape_character_len);
    {
      auto result = cudf::strings::like(
        to_strings_view(input), pattern_view, escape_view, exec->stream, *exec->mr);
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Regex replace
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_replace_with_backrefs_ex(cudf_column_view_handle input,
                                                    char const* pattern,
                                                    int32_t pattern_len,
                                                    char const* replacement,
                                                    int32_t replacement_len,
                                                    cudf_column_handle* out,
                                                    cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  cudf_status_t const pattern_status = validate_string_range(pattern, pattern_len, "pattern");
  if (pattern_status != CUDF_SUCCESS) { return pattern_status; }
  cudf_status_t const replacement_status =
    validate_string_range(replacement, replacement_len, "replacement");
  if (replacement_status != CUDF_SUCCESS) { return replacement_status; }

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const pattern_view     = make_host_string_view(pattern, pattern_len);
    auto const replacement_view = make_host_string_view(replacement, replacement_len);
    {
      auto prog   = cudf::strings::regex_program::create(pattern_view);
      auto result = cudf::strings::replace_with_backrefs(
        to_strings_view(input), *prog, replacement_view, exec->stream, *exec->mr);
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_replace_with_backrefs_output_size_ex(cudf_column_view_handle input,
                                                                char const* pattern,
                                                                int32_t pattern_len,
                                                                char const* replacement,
                                                                int32_t replacement_len,
                                                                size_t* retained_output_bytes,
                                                                size_t* temporary_workspace_bytes,
                                                                size_t* reservation_required_bytes,
                                                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(retained_output_bytes, "retained_output_bytes is null");
  *retained_output_bytes = 0;
  CUDF_RUST_NULL_CHECK(temporary_workspace_bytes, "temporary_workspace_bytes is null");
  *temporary_workspace_bytes = 0;
  CUDF_RUST_NULL_CHECK(reservation_required_bytes, "reservation_required_bytes is null");
  *reservation_required_bytes = 0;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  cudf_status_t const pattern_status = validate_string_range(pattern, pattern_len, "pattern");
  if (pattern_status != CUDF_SUCCESS) { return pattern_status; }
  cudf_status_t const replacement_status =
    validate_string_range(replacement, replacement_len, "replacement");
  if (replacement_status != CUDF_SUCCESS) { return replacement_status; }

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const pattern_view     = make_host_string_view(pattern, pattern_len);
    auto const replacement_view = make_host_string_view(replacement, replacement_len);
    {
      // Reservation admission happens at the shim's compatibility-resource
      // callback. Measuring inside libcudf can see a different default
      // alignment after resource type erasure, so observe the exact boundary
      // shared with the reservation-aware adaptor instead.
      cudf::rust::reservation::statistics_resource_adaptor measurement{exec->mr_raw};
      auto measurement_mr = rmm::mr::any_device_resource_from_raw(&measurement);
      auto prog           = cudf::strings::regex_program::create(pattern_view);
      auto prepared       = std::unique_ptr<cudf::strings::replace_with_backrefs_prepared,
                                            void (*)(cudf::strings::replace_with_backrefs_prepared*)>{
        cudf::strings::prepare_replace_with_backrefs(
          to_strings_view(input), *prog, replacement_view, exec->stream, measurement_mr),
        &cudf::strings::destroy_replace_with_backrefs_prepared};
      auto const facts =
        cudf::strings::get_replace_with_backrefs_prepared_output_size_facts(*prepared);

      *retained_output_bytes = facts.retained_output_bytes;
      *temporary_workspace_bytes =
        std::max(measurement.peak_bytes(), facts.peak_execute_workspace_bytes);
      *reservation_required_bytes =
        std::max(measurement.peak_aligned_bytes(),
                 checked_add_reservation_bytes(measurement.current_aligned_bytes(),
                                               facts.execute_reservation_required_bytes));

      // Every prepared allocation references measurement, so destroy the
      // handle before the observer leaves scope.
      prepared.reset();
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

namespace {

struct prepared_replace_with_backrefs_owner {
  cudf::strings::replace_with_backrefs_prepared* prepared{nullptr};
  cudf_handle_provenance provenance{};

  ~prepared_replace_with_backrefs_owner() noexcept
  {
    if (prepared != nullptr) {
      cudf::strings::destroy_replace_with_backrefs_prepared(prepared);
      prepared = nullptr;
    }
  }
};

cudf_status_t destroy_consumed_prepared_replace_owner(
  prepared_replace_with_backrefs_owner* owner) noexcept
{
  auto const provenance = owner->provenance;
  // A non-success status deliberately leaves the consumed owner unreachable:
  // its retained GPU allocations are quarantined rather than destroyed on an
  // unverified ambient device.
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cleanup_consumed_prepared_replace_owner_before_return(
  prepared_replace_with_backrefs_owner* owner, cudf_status_t operation_status) noexcept
{
  auto const cleanup_status = destroy_consumed_prepared_replace_owner(owner);
  return cleanup_status == CUDF_SUCCESS ? operation_status : cleanup_status;
}

}  // namespace

cudf_status_t cudf_strings_prepare_replace_with_backrefs_ex(
  cudf_column_view_handle input,
  char const* pattern,
  int32_t pattern_len,
  char const* replacement,
  int32_t replacement_len,
  cudf_prepared_replace_with_backrefs_handle* out,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  cudf_status_t const pattern_status = validate_string_range(pattern, pattern_len, "pattern");
  if (pattern_status != CUDF_SUCCESS) { return pattern_status; }
  cudf_status_t const replacement_status =
    validate_string_range(replacement, replacement_len, "replacement");
  if (replacement_status != CUDF_SUCCESS) { return replacement_status; }

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const pattern_view     = make_host_string_view(pattern, pattern_len);
    auto const replacement_view = make_host_string_view(replacement, replacement_len);
    {
      auto prog         = cudf::strings::regex_program::create(pattern_view);
      auto owner        = std::make_unique<prepared_replace_with_backrefs_owner>();
      owner->provenance = cudf_capture_exec_provenance(*exec);
      owner->prepared   = cudf::strings::prepare_replace_with_backrefs(
        to_strings_view(input), *prog, replacement_view, exec->stream, *exec->mr);
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
      *out = static_cast<cudf_prepared_replace_with_backrefs_handle>(owner.release());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_prepared_replace_with_backrefs_output_size_facts(
  cudf_prepared_replace_with_backrefs_handle handle,
  size_t* retained_output_bytes,
  size_t* retained_state_bytes,
  size_t* peak_execute_workspace_bytes,
  size_t* execute_reservation_required_bytes,
  uint64_t* size_pass_count)
{
  CUDF_RUST_NULL_CHECK(retained_output_bytes, "retained_output_bytes is null");
  *retained_output_bytes = 0;
  CUDF_RUST_NULL_CHECK(retained_state_bytes, "retained_state_bytes is null");
  *retained_state_bytes = 0;
  CUDF_RUST_NULL_CHECK(peak_execute_workspace_bytes, "peak_execute_workspace_bytes is null");
  *peak_execute_workspace_bytes = 0;
  CUDF_RUST_NULL_CHECK(execute_reservation_required_bytes,
                       "execute_reservation_required_bytes is null");
  *execute_reservation_required_bytes = 0;
  CUDF_RUST_NULL_CHECK(size_pass_count, "size_pass_count is null");
  *size_pass_count = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");

  CUDF_RUST_TRY
  {
    auto* owner = static_cast<prepared_replace_with_backrefs_owner*>(handle);
    CUDF_RUST_NULL_CHECK(owner->prepared, "prepared replace state is null");
    auto const facts =
      cudf::strings::get_replace_with_backrefs_prepared_output_size_facts(*owner->prepared);
    *retained_output_bytes              = facts.retained_output_bytes;
    *retained_state_bytes               = facts.retained_state_bytes;
    *peak_execute_workspace_bytes       = facts.peak_execute_workspace_bytes;
    *execute_reservation_required_bytes = facts.execute_reservation_required_bytes;
    *size_pass_count                    = facts.size_pass_count;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_execute_replace_with_backrefs_ex(
  cudf_column_view_handle input,
  cudf_prepared_replace_with_backrefs_handle handle,
  cudf_column_handle* out,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(handle, "handle is null");

  // Ownership of the prepared handle is always consumed on call. Keep the raw
  // owner until execution context validation completes: a default unique_ptr
  // would destroy retained GPU state on the ambient device when validation
  // returns early.
  auto* owner = static_cast<prepared_replace_with_backrefs_owner*>(handle);

  cudf_clear_error();
  try {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return cleanup_consumed_prepared_replace_owner_before_return(owner, exec_status);
    }

    if (exec->device != owner->provenance.device) {
      set_error("execute device must match prepared replace-with-backrefs provenance");
      return cleanup_consumed_prepared_replace_owner_before_return(
        owner, CUDF_ERROR_INVALID_OPERATION);
    }

    if (owner->prepared == nullptr) {
      set_error("prepared replace state is null");
      return cleanup_consumed_prepared_replace_owner_before_return(
        owner, CUDF_ERROR_NULL);
    }

    {
      auto* prepared  = owner->prepared;
      owner->prepared = nullptr;  // ownership transferred to execute
      auto result     = cudf::strings::execute_replace_with_backrefs(
        to_strings_view(input), prepared, exec->stream, *exec->mr);
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    // libcudf consumed and released the prepared GPU state under the validated
    // execution device. Only host provenance references remain in the owner.
    delete owner;
    return CUDF_SUCCESS;
  } catch (...) {
    auto const operation_status = cudf_translate_cleanup_exception_noexcept();
    if (owner->prepared == nullptr) {
      // Ownership was already transferred into libcudf, whose exception path
      // consumed it under the validated execution device.
      delete owner;
      return operation_status;
    }
    return cleanup_consumed_prepared_replace_owner_before_return(owner, operation_status);
  }
}

cudf_status_t cudf_strings_prepared_replace_with_backrefs_destroy(
  cudf_prepared_replace_with_backrefs_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* owner     = static_cast<prepared_replace_with_backrefs_owner*>(handle);
  auto provenance = owner->provenance;
  // Ownership is consumed even when cleanup is rejected.
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_strings_replace_with_backrefs_size_pass_count(uint64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = cudf::strings::replace_with_backrefs_size_pass_count();
  return CUDF_SUCCESS;
}

cudf_status_t cudf_strings_replace_with_backrefs_reset_size_pass_count()
{
  cudf::strings::replace_with_backrefs_reset_size_pass_count();
  return CUDF_SUCCESS;
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_slice_ex(cudf_column_view_handle input,
                                    int32_t start,
                                    int32_t start_valid,
                                    int32_t stop,
                                    int32_t stop_valid,
                                    int32_t step,
                                    cudf_column_handle* out,
                                    cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  if (step == 0) {
    set_error("step must not be 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      cudf::numeric_scalar<cudf::size_type> start_scalar{
        start, start_valid != 0, exec->stream, *exec->mr};
      cudf::numeric_scalar<cudf::size_type> stop_scalar{
        stop, stop_valid != 0, exec->stream, *exec->mr};
      cudf::numeric_scalar<cudf::size_type> step_scalar{step, true, exec->stream, *exec->mr};
      auto result = cudf::strings::slice_strings(
        to_strings_view(input), start_scalar, stop_scalar, step_scalar, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Concatenate
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_concatenate_ex(cudf_column_view_handle const* columns,
                                          int32_t num_columns,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (num_columns <= 1) {
    set_error("at least two string columns are required");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(columns, "columns is null");
  for (int32_t index = 0; index < num_columns; ++index) {
    CUDF_RUST_NULL_CHECK(columns[index], "column is null");
  }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<cudf::column_view> views;
    views.reserve(static_cast<std::size_t>(num_columns));
    for (int32_t index = 0; index < num_columns; ++index) {
      views.push_back(*static_cast<cudf::column_view*>(columns[index]));
    }

    {
      cudf::table_view strings_columns{views};
      cudf::string_scalar separator{"", true, exec->stream, *exec->mr};
      cudf::string_scalar null_replacement{"", true, exec->stream, *exec->mr};
      auto result = cudf::strings::concatenate(strings_columns,
                                               separator,
                                               null_replacement,
                                               cudf::strings::separator_on_nulls::NO,
                                               exec->stream,
                                               *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Strip / trim
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_strip_ex(cudf_column_view_handle input,
                                    cudf_strip_side_t side,
                                    cudf_scalar_handle to_strip,
                                    cudf_column_handle* out,
                                    cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(to_strip, "to_strip is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::strings::side_type cudf_side;
    switch (side) {
      case CUDF_STRIP_LEFT: cudf_side = cudf::strings::side_type::LEFT; break;
      case CUDF_STRIP_RIGHT: cudf_side = cudf::strings::side_type::RIGHT; break;
      case CUDF_STRIP_BOTH: cudf_side = cudf::strings::side_type::BOTH; break;
      default: set_error("invalid strip side type"); return CUDF_ERROR_INVALID_OPERATION;
    }

    cudf::string_scalar* scalar     = nullptr;
    cudf_status_t const type_status = check_string_scalar(to_strip, &scalar);
    if (type_status != CUDF_SUCCESS) { return type_status; }
    {
      auto result =
        cudf::strings::strip(to_strings_view(input), cudf_side, *scalar, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_count_characters_ex(cudf_column_view_handle input,
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

    {
      auto result = cudf::strings::count_characters(to_strings_view(input), exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_strings_count_bytes_ex(cudf_column_view_handle input,
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

    {
      auto result = cudf::strings::count_bytes(to_strings_view(input), exec->stream, *exec->mr);
      *out        = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

// ---------------------------------------------------------------------------
// Numeric conversion (null-on-failure / TRY_CAST contract)
// ---------------------------------------------------------------------------

cudf_status_t cudf_strings_try_to_integers_ex(cudf_column_view_handle input,
                                              cudf_data_type_handle output_type,
                                              cudf_column_handle* out,
                                              cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(output_type, "output_type is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto strings_view = to_strings_view(input);
    auto* dtype       = static_cast<cudf::data_type*>(output_type);
    if (!cudf::is_integral(*dtype)) {
      set_error("cudf_strings_try_to_integers: output_type must be an integral type");
      return CUDF_ERROR_TYPE;
    }

    {
      // `is_integer(input, output_type)` produces true only when the input
      // string parses cleanly AND fits the requested integer type. It returns
      // null for null input rows. `to_integers` performs the parse (returning
      // undefined values for rejected rows, null for null input). Masking the
      // parsed column against the validity mask with a default-constructed
      // (invalid) scalar gives TRY_CAST semantics: success → parsed value,
      // parse failure or overflow → null, null input → null.
      auto valid_mask = cudf::strings::is_integer(strings_view, *dtype, exec->stream, *exec->mr);
      auto integers   = cudf::strings::to_integers(strings_view, *dtype, exec->stream, *exec->mr);
      auto null_value = cudf::make_default_constructed_scalar(*dtype, exec->stream, *exec->mr);
      auto result =
        cudf::copy_if_else(integers->view(), *null_value, valid_mask->view(), exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
