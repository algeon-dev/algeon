use crate::{
    cudf_column_handle, cudf_exec_options_t, cudf_null_order_t, cudf_order_t, cudf_status_t, cudf_table_handle,
    cudf_table_view_handle,
};

extern "C" {
    // -- Sorting -------------------------------------------------------------

    pub fn cudf_sort(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Sort with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_sort_ex(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Stable sort with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_stable_sort_ex(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_sort_by_key(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Sort by key with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_sort_by_key_ex(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort_by_key(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Stable sort by key with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_stable_sort_by_key_ex(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_sorted_order(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Sorted order with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned column.
    pub fn cudf_sorted_order_ex(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_sorted(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut i32,
    ) -> cudf_status_t;
}
