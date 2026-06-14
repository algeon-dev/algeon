use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_datetime_component_t, cudf_exec_options_t, cudf_status_t,
};

extern "C" {
    // -- Datetime ------------------------------------------------------------

    pub fn cudf_datetime_extract_component(
        input: cudf_column_view_handle,
        component: cudf_datetime_component_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_datetime_extract_component_ex(
        input: cudf_column_view_handle,
        component: cudf_datetime_component_t,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
