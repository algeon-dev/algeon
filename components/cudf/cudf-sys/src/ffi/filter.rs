use crate::{
    cudf_column_view_handle, cudf_exec_options_t, cudf_row_predicate_node_t, cudf_status_t, cudf_table_handle,
    cudf_table_view_handle,
};

extern "C" {
    // -- Filter --------------------------------------------------------------

    pub fn cudf_apply_boolean_mask(
        input: cudf_table_view_handle,
        mask: cudf_column_view_handle,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Apply a boolean mask with explicit execution options.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_apply_boolean_mask_ex(
        input: cudf_table_view_handle,
        mask: cudf_column_view_handle,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Evaluate a validated row predicate and filter the table once.
    pub fn cudf_filter_rows_by_predicate(
        input: cudf_table_view_handle,
        nodes: *const cudf_row_predicate_node_t,
        num_nodes: i32,
        root_index: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Evaluate a validated row predicate with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_filter_rows_by_predicate_ex(
        input: cudf_table_view_handle,
        nodes: *const cudf_row_predicate_node_t,
        num_nodes: i32,
        root_index: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
