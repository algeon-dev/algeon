use std::os::raw::c_void;

use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_data_type_handle, cudf_exec_options_t,
    cudf_reduce_aggregation_kind_t, cudf_scalar_handle, cudf_scan_type_t, cudf_status_t,
};

extern "C" {
    // -- Reduction -----------------------------------------------------------

    pub fn cudf_reduce_aggregation_create(kind: cudf_reduce_aggregation_kind_t, out: *mut *mut c_void)
    -> cudf_status_t;

    pub fn cudf_reduce_aggregation_create_variance(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_create_std(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_create_nunique(null_policy: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_destroy(handle: *mut c_void);

    pub fn cudf_scan_aggregation_create(kind: cudf_reduce_aggregation_kind_t, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_scan_aggregation_destroy(handle: *mut c_void);

    /// Reduce with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned scalar.
    pub fn cudf_reduce_ex(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        output_type: cudf_data_type_handle,
        out: *mut cudf_scalar_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Scan with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_scan_ex(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        scan_type: cudf_scan_type_t,
        null_policy: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Compute min/max with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned scalars.
    pub fn cudf_minmax_ex(
        input: cudf_column_view_handle,
        out_min: *mut cudf_scalar_handle,
        out_max: *mut cudf_scalar_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
