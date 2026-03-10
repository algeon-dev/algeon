#ifndef CUDF_RUST_DICTIONARY_H
#define CUDF_RUST_DICTIONARY_H

#include "column.h"
#include "error.h"
#include "types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Encode a column into a dictionary column.
///
/// The result is an owning column of type DICTIONARY32 whose children are:
///   - child 0: indices column (type `indices_type_id`, default INT32)
///   - child 1: keys column (unique, sorted values from the input)
///
/// Example: [429, 111, 213, 111, 213, 429, 213]
///   keys:    [111, 213, 429]
///   indices: [2, 0, 1, 0, 1, 2, 1]
///
/// The caller must free the result with `cudf_column_destroy`.
cudf_status_t cudf_dictionary_encode(cudf_column_view_handle input,
                                     cudf_type_id_t indices_type_id,
                                     cudf_column_handle* out);

/// Decode a dictionary column back to a regular column.
///
/// The input must be a DICTIONARY32 column. The result is a regular column
/// with the same type as the dictionary keys.
///
/// The caller must free the result with `cudf_column_destroy`.
cudf_status_t cudf_dictionary_decode(cudf_column_view_handle dict_column,
                                     cudf_column_handle* out);

/// Get the keys column view from a dictionary column.
///
/// The input must be a DICTIONARY32 column. The returned view borrows from
/// the input and must not outlive it. Free with `cudf_column_view_destroy`.
cudf_status_t cudf_dictionary_keys(cudf_column_view_handle dict_column,
                                   cudf_column_view_handle* out);

/// Get the indices column view from a dictionary column.
///
/// The input must be a DICTIONARY32 column. The returned view borrows from
/// the input and must not outlive it. Free with `cudf_column_view_destroy`.
cudf_status_t cudf_dictionary_indices(cudf_column_view_handle dict_column,
                                      cudf_column_view_handle* out);

/// Replace the keys of a dictionary column.
///
/// Any new elements found in `keys` are added; any existing keys not in
/// `keys` are removed (rows that referenced removed keys become null).
/// The indices are updated to reflect the new key positions.
///
/// The caller must free the result with `cudf_column_destroy`.
cudf_status_t cudf_dictionary_set_keys(cudf_column_view_handle dict_column,
                                       cudf_column_view_handle keys,
                                       cudf_column_handle* out);

/// Align multiple dictionary columns to a shared key set.
///
/// The keys from all input dictionary columns are concatenated, deduplicated,
/// and used to re-key each column via `set_keys`. After this call every
/// output column shares the same key set, so their indices are directly
/// comparable.
///
/// `dict_columns` is an array of `num_columns` dictionary column views.
/// `out` is a caller-allocated array of `num_columns` column handles that
/// will receive the matched columns. Each must be freed with
/// `cudf_column_destroy`.
cudf_status_t cudf_dictionary_match_dictionaries(
    const cudf_column_view_handle* dict_columns,
    int32_t num_columns,
    cudf_column_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_DICTIONARY_H
