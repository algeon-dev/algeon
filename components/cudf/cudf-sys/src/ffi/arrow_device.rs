use std::os::raw::{c_char, c_void};

use crate::{
    cudf_arrow_device_array_handle, cudf_arrow_device_export_materialization_t,
    cudf_arrow_device_import_materialization_t, cudf_arrow_schema_handle, cudf_column_handle, cudf_column_view_handle,
    cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle,
};

extern "C" {
    // -- Arrow Device Column Bridge -----------------------------------------

    /// Export a column view through Arrow C Device ABI using strict explicit
    /// execution options.
    ///
    /// `options` must name an explicit valid device, non-null stream, and
    /// non-null memory resource. The memory resource must outlive the returned
    /// device array when bridge-owned storage is materialized.
    pub fn cudf_arrow_device_column_export(
        view: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out_schema: *mut cudf_arrow_schema_handle,
        out_array: *mut cudf_arrow_device_array_handle,
        out_materialization: *mut cudf_arrow_device_export_materialization_t,
    ) -> cudf_status_t;

    /// Export a table view through Arrow C Device ABI using strict explicit
    /// execution options.
    ///
    /// `options` must name an explicit valid device, non-null stream, and
    /// non-null memory resource. The memory resource must outlive the returned
    /// device array when bridge-owned storage is materialized.
    pub fn cudf_arrow_device_table_export(
        view: cudf_table_view_handle,
        options: *const cudf_exec_options_t,
        column_names: *const *const c_char,
        num_column_names: i32,
        out_schema: *mut cudf_arrow_schema_handle,
        out_array: *mut cudf_arrow_device_array_handle,
        out_materialization: *mut cudf_arrow_device_export_materialization_t,
    ) -> cudf_status_t;

    /// Import an Arrow C Device ABI column using strict explicit execution
    /// options.
    ///
    /// `options` must name an explicit valid device, non-null stream, and
    /// non-null memory resource. The memory resource must outlive the returned
    /// column when bridge-owned storage is materialized.
    pub fn cudf_arrow_device_column_import(
        schema: cudf_arrow_schema_handle,
        array: cudf_arrow_device_array_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
        out_materialization: *mut cudf_arrow_device_import_materialization_t,
    ) -> cudf_status_t;

    /// Import an Arrow C Device ABI table using strict explicit execution
    /// options.
    ///
    /// `options` must name an explicit valid device, non-null stream, and
    /// non-null memory resource. The memory resource must outlive the returned
    /// table when bridge-owned storage is materialized.
    pub fn cudf_arrow_device_table_import(
        schema: cudf_arrow_schema_handle,
        array: cudf_arrow_device_array_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
        out_materialization: *mut cudf_arrow_device_import_materialization_t,
    ) -> cudf_status_t;

    pub fn cudf_arrow_device_schema_destroy(schema: cudf_arrow_schema_handle);

    pub fn cudf_arrow_device_array_destroy(array: cudf_arrow_device_array_handle);

    pub fn cudf_arrow_device_array_device_id(array: cudf_arrow_device_array_handle, out: *mut i64) -> cudf_status_t;

    pub fn cudf_arrow_device_array_device_type(array: cudf_arrow_device_array_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_arrow_device_array_has_sync_event(
        array: cudf_arrow_device_array_handle,
        out: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_arrow_device_array_sync_event_raw(
        array: cudf_arrow_device_array_handle,
        out: *mut *mut c_void,
    ) -> cudf_status_t;

    pub fn cudf_test_arrow_device_array_set_device_type(array: cudf_arrow_device_array_handle, device_type: i32);

    pub fn cudf_arrow_device_column_fixed_width_values_ptr(
        schema: cudf_arrow_schema_handle,
        array: cudf_arrow_device_array_handle,
        out: *mut *const c_void,
    ) -> cudf_status_t;

    pub fn cudf_arrow_device_table_num_columns(schema: cudf_arrow_schema_handle, out: *mut i64) -> cudf_status_t;

    pub fn cudf_arrow_device_table_column_name(
        schema: cudf_arrow_schema_handle,
        index: i64,
        out: *mut *const c_char,
    ) -> cudf_status_t;

    pub fn cudf_arrow_device_table_fixed_width_values_ptr(
        schema: cudf_arrow_schema_handle,
        array: cudf_arrow_device_array_handle,
        index: i64,
        out: *mut *const c_void,
    ) -> cudf_status_t;
}
