use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_data_type_handle, cudf_exec_options_t, cudf_status_t,
    cudf_unary_operator_t,
};

unsafe extern "C" {
    // -- Unary ---------------------------------------------------------------

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_unary_operation_ex(
        input: cudf_column_view_handle,
        op: cudf_unary_operator_t,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_round_half_up_ex(
        input: cudf_column_view_handle,
        decimal_places: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_is_null_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_is_valid_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_cast_ex(
        input: cudf_column_view_handle,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_supported_cast(
        from_type: cudf_data_type_handle,
        to_type: cudf_data_type_handle,
        out: *mut i32,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_is_nan_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_is_not_nan_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
