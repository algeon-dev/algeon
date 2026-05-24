use crate::{cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_status_t, cudf_type_id_t};

extern "C" {
    // -- Dictionary ----------------------------------------------------------

    /// Encode with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_dictionary_encode_ex(
        input: cudf_column_view_handle,
        indices_type_id: cudf_type_id_t,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Decode with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_dictionary_decode_ex(
        dict_column: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_dictionary_keys(
        dict_column: cudf_column_view_handle,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_dictionary_indices(
        dict_column: cudf_column_view_handle,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    /// Replace dictionary keys with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_dictionary_set_keys_ex(
        dict_column: cudf_column_view_handle,
        keys: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Align dictionary columns with explicit execution options.
    ///
    /// On success, `out_columns` receives a heap-allocated array of
    /// `out_count` owning column handles. Destroy each column individually,
    /// then free the array with [`crate::cudf_free_column_array`]. On failure,
    /// the shim initializes `out_columns` to null and `out_count` to zero.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned columns.
    pub fn cudf_dictionary_match_dictionaries_ex(
        dict_columns: *const cudf_column_view_handle,
        num_columns: i32,
        out_columns: *mut *mut cudf_column_handle,
        out_count: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
