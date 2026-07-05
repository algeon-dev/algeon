use crate::{
    cudf_duplicate_keep_option_t, cudf_exec_options_t, cudf_status_t, cudf_stream_compaction_nan_equality_t,
    cudf_stream_compaction_null_equality_t, cudf_table_handle, cudf_table_view_handle,
};

extern "C" {
    // -- Stream Compaction ---------------------------------------------------

    /// Drop rows with explicit execution options. `options` must be non-null
    /// and must carry an explicit device, stream, and memory resource.
    pub fn cudf_drop_nulls_ex(
        input: cudf_table_view_handle,
        keys: *const i32,
        num_keys: i32,
        keep_threshold: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Return distinct rows with explicit execution options. `options` must be
    /// non-null and must carry an explicit device, stream, and memory resource.
    pub fn cudf_distinct_ex(
        input: cudf_table_view_handle,
        keys: *const i32,
        num_keys: i32,
        keep: cudf_duplicate_keep_option_t,
        null_equality: cudf_stream_compaction_null_equality_t,
        nan_equality: cudf_stream_compaction_nan_equality_t,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
