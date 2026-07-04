use std::os::raw::c_void;

use crate::{
    cudf_aggregation_kind_t, cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_null_policy_t,
    cudf_reduce_aggregation_kind_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle,
};

extern "C" {
    // -- GroupBy --------------------------------------------------------------

    pub fn cudf_aggregation_create(kind: cudf_aggregation_kind_t, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_count(null_policy: cudf_null_policy_t, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_std(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_variance(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_destroy(handle: *mut c_void);

    pub fn cudf_groupby_scan_aggregation_create(
        kind: cudf_reduce_aggregation_kind_t,
        out: *mut *mut c_void,
    ) -> cudf_status_t;

    pub fn cudf_groupby_scan_aggregation_destroy(handle: *mut c_void);

    /// Perform a groupby aggregation and return the keys table plus a
    /// heap-allocated flat array of result columns.
    ///
    /// On failure, all outputs are initialized to null / zero. When the total
    /// number of result columns is zero, `out_values` is returned as null.
    pub fn cudf_groupby_aggregate(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: cudf_null_policy_t,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
    ) -> cudf_status_t;

    /// Groupby aggregate with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned keys
    /// table and every returned result column.
    pub fn cudf_groupby_aggregate_ex(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: cudf_null_policy_t,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_groupby_scan(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: cudf_null_policy_t,
        keys_are_sorted: i32,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_groupby_scan_ex(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: cudf_null_policy_t,
        keys_are_sorted: i32,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Free a heap-allocated array of owning column handles returned by this
    /// shim. Does not destroy the individual columns.
    pub fn cudf_free_column_array(array: *mut cudf_column_handle);
}
