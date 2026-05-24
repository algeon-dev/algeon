use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_datetime_component_t, cudf_datetime_rounding_frequency_t,
    cudf_exec_options_t, cudf_status_t,
};

extern "C" {
    // -- Datetime ------------------------------------------------------------

    /// Extract a datetime component with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_datetime_extract_component_ex(
        input: cudf_column_view_handle,
        component: cudf_datetime_component_t,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Floor datetimes with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_datetime_floor_ex(
        input: cudf_column_view_handle,
        frequency: cudf_datetime_rounding_frequency_t,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Add calendrical months with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_datetime_add_calendrical_months_scalar_ex(
        input: cudf_column_view_handle,
        months: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
