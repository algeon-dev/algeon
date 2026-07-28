#ifndef CUDF_RUST_DICTIONARY_H
#define CUDF_RUST_DICTIONARY_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Encode a column into a dictionary column with explicit execution context.
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
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_dictionary_encode_ex(cudf_column_view_handle input,
                                        cudf_type_id_t indices_type_id,
                                        cudf_column_handle* out,
                                        const cudf_exec_options_t* options);

/// Decode a dictionary column back to a regular column with explicit execution
/// context.
///
/// The input must be a DICTIONARY32 column. The result is a regular column
/// with the same type as the dictionary keys.
///
/// The caller must free the result with `cudf_column_destroy`.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_dictionary_decode_ex(cudf_column_view_handle dict_column,
                                        cudf_column_handle* out,
                                        const cudf_exec_options_t* options);

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

/// Replace the keys of a dictionary column with explicit execution context.
///
/// Any new elements found in `keys` are added; any existing keys not in
/// `keys` are removed (rows that referenced removed keys become null).
/// The indices are updated to reflect the new key positions.
///
/// The caller must free the result with `cudf_column_destroy`.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_dictionary_set_keys_ex(cudf_column_view_handle dict_column,
                                          cudf_column_view_handle keys,
                                          cudf_column_handle* out,
                                          const cudf_exec_options_t* options);

/// Align multiple dictionary columns to a shared key set with explicit
/// execution context.
///
/// The keys from all input dictionary columns are concatenated, deduplicated,
/// and used to re-key each column via `set_keys`. After this call every
/// output column shares the same key set, so their indices are directly
/// comparable.
///
/// `dict_columns` is an array of `num_columns` dictionary column views.
/// `out_columns` receives a heap-allocated array of `out_count` column
/// handles. Each matched column must be freed with `cudf_column_destroy`,
/// then the array itself must be freed with `cudf_free_column_array`.
///
/// Contract:
/// - `num_columns` must be > 0
/// - `dict_columns` must point to at least `num_columns` entries
/// - `out_columns` must be non-NULL
/// - `out_count` must be non-NULL
/// - on failure, `*out_columns` is set to NULL and `*out_count` is set to 0
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive all returned columns.
cudf_status_t cudf_dictionary_match_dictionaries_ex(const cudf_column_view_handle* dict_columns,
                                                    int32_t num_columns,
                                                    cudf_column_handle** out_columns,
                                                    int32_t* out_count,
                                                    const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_DICTIONARY_H
