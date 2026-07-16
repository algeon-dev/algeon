use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_out_of_bounds_policy_t, cudf_scalar_handle,
    cudf_status_t, cudf_table_handle, cudf_table_view_handle,
};

unsafe extern "C" {
    // -- Copying (gather) ----------------------------------------------------

    /// Gather rows with explicit execution options.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
    pub fn cudf_gather_ex(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: cudf_out_of_bounds_policy_t,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Return the exact retained device allocation footprint of a gather
    /// result using explicit execution options.
    /// `options` must name an explicit valid device and non-null stream/MR.
    pub fn cudf_gather_device_allocated_size_ex(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: cudf_out_of_bounds_policy_t,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Copying (scatter) ---------------------------------------------------

    /// Scatter rows with explicit execution options.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
    pub fn cudf_scatter_ex(
        source: cudf_table_view_handle,
        scatter_map: cudf_column_view_handle,
        target: cudf_table_view_handle,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Copying (copy_if_else) ----------------------------------------------

    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_copy_if_else_vv_ex(
        lhs: cudf_column_view_handle,
        rhs: cudf_column_view_handle,
        boolean_mask: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Copying (shift) -----------------------------------------------------

    /// Shift with explicit execution options.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_shift_ex(
        input: cudf_column_view_handle,
        offset: i32,
        fill_value: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
