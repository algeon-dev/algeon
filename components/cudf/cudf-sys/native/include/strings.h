#ifndef CUDF_RUST_STRINGS_H
#define CUDF_RUST_STRINGS_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "scalar.h"
#include "types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Strip side type matching cudf::strings::side_type.
typedef enum {
  CUDF_STRIP_LEFT  = 0,
  CUDF_STRIP_RIGHT = 1,
  CUDF_STRIP_BOTH  = 2,
} cudf_strip_side_t;

// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

/// Convert string column to lowercase with explicit execution context.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_to_lower_ex(cudf_column_view_handle input,
                                       cudf_column_handle* out,
                                       const cudf_exec_options_t* options);

/// Convert string column to uppercase with explicit execution context.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_to_upper_ex(cudf_column_view_handle input,
                                       cudf_column_handle* out,
                                       const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Find / contains / starts_with / ends_with (scalar target)
// ---------------------------------------------------------------------------

/// Element-wise literal substring containment with explicit execution context.
/// Returns BOOL8 column.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_contains_ex(cudf_column_view_handle input,
                                       cudf_scalar_handle target,
                                       cudf_column_handle* out,
                                       const cudf_exec_options_t* options);

/// Element-wise starts-with with explicit execution context.
/// Returns BOOL8 column.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_starts_with_ex(cudf_column_view_handle input,
                                          cudf_scalar_handle target,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

/// Element-wise ends-with with explicit execution context.
/// Returns BOOL8 column.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_ends_with_ex(cudf_column_view_handle input,
                                        cudf_scalar_handle target,
                                        cudf_column_handle* out,
                                        const cudf_exec_options_t* options);

/// Element-wise SQL LIKE pattern match with explicit execution context.
/// Returns BOOL8 column.
///
/// `pattern` and `escape_character` are borrowed host byte ranges. Null
/// pointers are accepted only when the corresponding length is zero. The
/// escape range must be empty or contain exactly one byte.
/// On success, synchronizes the selected stream before returning so borrowed
/// `pattern` and `escape_character` host memory is no longer in use.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_like_ex(cudf_column_view_handle input,
                                   const char* pattern,
                                   int32_t pattern_len,
                                   const char* escape_character,
                                   int32_t escape_character_len,
                                   cudf_column_handle* out,
                                   const cudf_exec_options_t* options);

/// Element-wise regex replacement with back-reference templates and explicit
/// execution context. Returns STRING column.
///
/// `pattern` and `replacement` are borrowed host byte ranges. Null pointers are
/// accepted only when the corresponding length is zero.
/// On success, synchronizes the selected stream before returning so borrowed
/// `pattern` and `replacement` host memory is no longer in use.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_replace_with_backrefs_ex(cudf_column_view_handle input,
                                                    const char* pattern,
                                                    int32_t pattern_len,
                                                    const char* replacement,
                                                    int32_t replacement_len,
                                                    cudf_column_handle* out,
                                                    const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

/// Slice each string with explicit execution context.
/// Slices by character positions `[start, stop)` and `step`.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_slice_ex(cudf_column_view_handle input,
                                    int32_t start,
                                    int32_t start_valid,
                                    int32_t stop,
                                    int32_t stop_valid,
                                    int32_t step,
                                    cudf_column_handle* out,
                                    const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Concatenate
// ---------------------------------------------------------------------------

/// Row-wise concatenate string columns with explicit execution context.
/// Uses an empty separator. Null inputs are replaced with empty strings.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_concatenate_ex(const cudf_column_view_handle* columns,
                                          int32_t num_columns,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Strip / trim
// ---------------------------------------------------------------------------

/// Strip characters from one or both sides of each string with explicit
/// execution context.
/// `to_strip` is a scalar string of characters to strip. If null scalar,
/// strips whitespace.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_strip_ex(cudf_column_view_handle input,
                                    cudf_strip_side_t side,
                                    cudf_scalar_handle to_strip,
                                    cudf_column_handle* out,
                                    const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// Character count with explicit execution context.
/// Counts characters, not bytes, and returns an INT32 column.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_count_characters_ex(cudf_column_view_handle input,
                                               cudf_column_handle* out,
                                               const cudf_exec_options_t* options);

/// Byte count with explicit execution context.
/// Returns INT32 column.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_count_bytes_ex(cudf_column_view_handle input,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Numeric conversion (null-on-failure / TRY_CAST contract)
// ---------------------------------------------------------------------------

/// Try-cast strings to integers with explicit execution context.
///
/// Converts a string column to integers with SQL `TRY_CAST` semantics: each
/// element that does not parse as an integer that fits `output_type` produces
/// a null row instead of an error. Null inputs propagate as null outputs.
///
/// `output_type` must be an integral data type. Builds on
/// `cudf::strings::is_integer(input, output_type)` to detect rows that parse
/// cleanly and fit the target type, `cudf::strings::to_integers(input,
/// output_type)` for the converted values, and `cudf::copy_if_else` to mask
/// rejected rows to null.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_strings_try_to_integers_ex(cudf_column_view_handle input,
                                              cudf_data_type_handle output_type,
                                              cudf_column_handle* out,
                                              const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STRINGS_H
