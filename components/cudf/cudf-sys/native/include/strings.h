#ifndef CUDF_RUST_STRINGS_H
#define CUDF_RUST_STRINGS_H

#include "error.h"
#include "column.h"
#include "scalar.h"
#include "execution_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Strip side type matching cudf::strings::side_type.
typedef enum {
  CUDF_STRIP_LEFT = 0,
  CUDF_STRIP_RIGHT = 1,
  CUDF_STRIP_BOTH = 2,
} cudf_strip_side_t;

// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

/// Convert string column to lowercase.
cudf_status_t cudf_strings_to_lower(cudf_column_view_handle input,
                                    cudf_column_handle* out);

/// Convert string column to lowercase with explicit execution context.
cudf_status_t cudf_strings_to_lower_ex(cudf_column_view_handle input,
                                       cudf_column_handle* out,
                                       const cudf_exec_options_t* options);

/// Convert string column to uppercase.
cudf_status_t cudf_strings_to_upper(cudf_column_view_handle input,
                                    cudf_column_handle* out);

/// Convert string column to uppercase with explicit execution context.
cudf_status_t cudf_strings_to_upper_ex(cudf_column_view_handle input,
                                       cudf_column_handle* out,
                                       const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Find / contains / starts_with / ends_with (scalar target)
// ---------------------------------------------------------------------------

/// Element-wise literal substring containment test. Returns BOOL8 column.
cudf_status_t cudf_strings_contains(cudf_column_view_handle input,
                                    cudf_scalar_handle target,
                                    cudf_column_handle* out);

/// Element-wise literal substring containment with explicit execution context.
cudf_status_t cudf_strings_contains_ex(cudf_column_view_handle input,
                                       cudf_scalar_handle target,
                                       cudf_column_handle* out,
                                       const cudf_exec_options_t* options);

/// Element-wise starts-with test against a scalar string. Returns BOOL8 column.
cudf_status_t cudf_strings_starts_with(cudf_column_view_handle input,
                                       cudf_scalar_handle target,
                                       cudf_column_handle* out);

/// Element-wise starts-with with explicit execution context.
cudf_status_t cudf_strings_starts_with_ex(cudf_column_view_handle input,
                                          cudf_scalar_handle target,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

/// Element-wise ends-with test against a scalar string. Returns BOOL8 column.
cudf_status_t cudf_strings_ends_with(cudf_column_view_handle input,
                                     cudf_scalar_handle target,
                                     cudf_column_handle* out);

/// Element-wise ends-with with explicit execution context.
cudf_status_t cudf_strings_ends_with_ex(cudf_column_view_handle input,
                                        cudf_scalar_handle target,
                                        cudf_column_handle* out,
                                        const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Strip / trim
// ---------------------------------------------------------------------------

/// Strip characters from one or both sides of each string.
/// `to_strip` is a scalar string of characters to strip. If null scalar,
/// strips whitespace.
cudf_status_t cudf_strings_strip(cudf_column_view_handle input,
                                 cudf_strip_side_t side,
                                 cudf_scalar_handle to_strip,
                                 cudf_column_handle* out);

/// Strip with explicit execution context.
cudf_status_t cudf_strings_strip_ex(cudf_column_view_handle input,
                                    cudf_strip_side_t side,
                                    cudf_scalar_handle to_strip,
                                    cudf_column_handle* out,
                                    const cudf_exec_options_t* options);

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// Character count (not byte count) per string. Returns INT32 column.
cudf_status_t cudf_strings_count_characters(cudf_column_view_handle input,
                                            cudf_column_handle* out);

/// Character count with explicit execution context.
cudf_status_t cudf_strings_count_characters_ex(cudf_column_view_handle input,
                                               cudf_column_handle* out,
                                               const cudf_exec_options_t* options);

/// Byte count per string. Returns INT32 column.
cudf_status_t cudf_strings_count_bytes(cudf_column_view_handle input,
                                       cudf_column_handle* out);

/// Byte count with explicit execution context.
cudf_status_t cudf_strings_count_bytes_ex(cudf_column_view_handle input,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STRINGS_H
