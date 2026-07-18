use crate::{
    cudf_binary_operator_t, cudf_column_handle, cudf_column_view_handle, cudf_data_type_handle, cudf_exec_options_t,
    cudf_scalar_handle, cudf_status_t,
};

unsafe extern "C" {
    // -- Binary --------------------------------------------------------------

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_binary_operation_vv_ex(
        lhs: cudf_column_view_handle,
        rhs: cudf_column_view_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_binary_operation_vs_ex(
        lhs: cudf_column_view_handle,
        rhs: cudf_scalar_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_binary_operation_sv_ex(
        lhs: cudf_scalar_handle,
        rhs: cudf_column_view_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
