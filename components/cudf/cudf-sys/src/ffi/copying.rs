use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_out_of_bounds_policy_t, cudf_status_t,
    cudf_table_handle, cudf_table_view_handle,
};

extern "C" {
    // -- Copying (gather) ----------------------------------------------------

    pub fn cudf_gather(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: cudf_out_of_bounds_policy_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Gather rows with explicit execution options.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_gather_ex(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: cudf_out_of_bounds_policy_t,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Copying (copy_if_else) ----------------------------------------------

    pub fn cudf_copy_if_else_vv(
        lhs: cudf_column_view_handle,
        rhs: cudf_column_view_handle,
        boolean_mask: cudf_column_view_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_copy_if_else_vv_ex(
        lhs: cudf_column_view_handle,
        rhs: cudf_column_view_handle,
        boolean_mask: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
