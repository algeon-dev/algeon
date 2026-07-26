use crate::{
    cudf_column_view_handle, cudf_exec_options_t, cudf_row_predicate_node_t, cudf_status_t, cudf_table_handle,
    cudf_table_view_handle,
};

unsafe extern "C" {
    // -- Filter --------------------------------------------------------------

    /// Apply a boolean mask with explicit execution options. `options` must be
    /// non-null and must carry an explicit device, stream, and memory resource.
    pub fn cudf_apply_boolean_mask_ex(
        input: cudf_table_view_handle,
        mask: cudf_column_view_handle,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Evaluate a validated row predicate with explicit execution context.
    /// `options` must be non-null and must carry an explicit device, stream,
    /// and memory resource.
    pub fn cudf_filter_rows_by_predicate_ex(
        input: cudf_table_view_handle,
        nodes: *const cudf_row_predicate_node_t,
        num_nodes: i32,
        root_index: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
