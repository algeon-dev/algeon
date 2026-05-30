use std::{
    os::raw::{c_char, c_void},
    ptr,
};

fn make_i32_column(data: &[i32]) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_column_view(col: *mut c_void) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view(col, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_table_from_one_column(col: *mut c_void) -> *mut c_void {
    let mut col_arr = [col];
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(col_arr.as_mut_ptr(), 1, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    // ownership moved into table
    assert!(col_arr[0].is_null());
    out
}

fn read_i32_column_data(col: *mut c_void, expected_len: usize) -> Vec<i32> {
    let view = make_column_view(col);
    let mut buf = vec![0i32; expected_len];
    let buf_size = expected_len * std::mem::size_of::<i32>();
    let st = unsafe { cudf_sys::cudf_column_view_to_host(view, buf.as_mut_ptr() as *mut c_void, buf_size) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    buf
}

fn make_table_view(tbl: *mut c_void) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_view(tbl, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

#[test]
fn test_scalar_string_value_stability_until_next_call() {
    let s = std::ffi::CString::new("hello").unwrap();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(s.as_ptr(), 1, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut ptr1: *const c_char = ptr::null();
    let mut len1: i32 = 0;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut ptr1, &mut len1) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!ptr1.is_null());
    assert_eq!(len1, 5);

    // Next shim call may invalidate pointer according to contract.
    let mut ptr2: *const c_char = ptr::null();
    let mut len2: i32 = 0;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut ptr2, &mut len2) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!ptr2.is_null());
    assert_eq!(len2, 5);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_concatenate_columns_null_element_returns_null_error() {
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);

    let views = [view, ptr::null_mut()];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_concatenate_columns(views.as_ptr(), views.len() as i32, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_concatenate_tables_null_element_returns_null_error() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let tables = [tv, ptr::null_mut()];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_concatenate_tables(tables.as_ptr(), tables.len() as i32, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_sort_negative_num_keys_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_sort(tv, ptr::null(), ptr::null(), -1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_drop_nulls_negative_num_keys_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls(tv, keys.as_ptr(), -1, 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_distinct_negative_num_keys_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_distinct(tv, keys.as_ptr(), -1, 0, 0, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_gather_invalid_bounds_policy_rejected() {
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let map_col = make_i32_column(&[0, 2]);
    let map_view = make_column_view(map_col);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_gather(tv, map_view, 99, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(map_view);
        cudf_sys::cudf_column_destroy(map_col);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_distinct_invalid_policy_values_rejected() {
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();

    // Invalid keep policy
    let status = unsafe { cudf_sys::cudf_distinct(tv, keys.as_ptr(), 1, 99, 0, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    // Invalid null equality policy
    let status = unsafe { cudf_sys::cudf_distinct(tv, keys.as_ptr(), 1, 0, 99, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    // Invalid nan equality policy
    let status = unsafe { cudf_sys::cudf_distinct(tv, keys.as_ptr(), 1, 0, 0, 99, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_column_create_numeric_negative_size_rejected() {
    let data: [i32; 1] = [42];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            -1,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_column_create_numeric_null_count_out_of_range_rejected() {
    let data = [1i32, 2, 3];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            4,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_column_create_numeric_null_count_requires_mask() {
    let data = [1i32, 2, 3];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            2,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_drop_nulls_zero_num_keys_allows_null_keys_pointer() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls(tv, ptr::null(), 0, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_distinct_zero_num_keys_allows_null_keys_pointer() {
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_distinct(tv, ptr::null(), 0, 0, 0, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_groupby_negative_num_requests_rejected() {
    let key_col = make_i32_column(&[1, 1, 2]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = ptr::null_mut();
    let mut out_total: i32 = -1;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys_view,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            -1,
            0,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    unsafe {
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

#[test]
fn test_memory_resource_pool_create_destroy_smoke() {
    let mut mr: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_pool(1 << 20, 1 << 24, &mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());

    unsafe { cudf_sys::cudf_memory_resource_destroy(mr) };
}

// ===========================================================================
// cudf_table_release_columns tests
// ===========================================================================

#[test]
fn test_table_release_columns_single_column() {
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 1);
    assert!(!out_columns.is_null());

    // Read the released column and verify data.
    let released_col = unsafe { *out_columns };
    assert!(!released_col.is_null());

    let data = read_i32_column_data(released_col, 3);
    assert_eq!(data, vec![10, 20, 30]);

    // Cleanup: destroy the released column and the array.
    unsafe {
        cudf_sys::cudf_column_destroy(released_col);
        cudf_sys::cudf_free_column_array(out_columns);
    }
    // Note: tbl is consumed by release_columns, no need to destroy it.
}

#[test]
fn test_table_release_columns_multiple_columns() {
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let mut col_arr = [c1, c2];
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(col_arr.as_mut_ptr(), 2, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 2);
    assert!(!out_columns.is_null());

    let col_ptrs = unsafe { std::slice::from_raw_parts(out_columns, 2) };

    let data1 = read_i32_column_data(col_ptrs[0], 3);
    assert_eq!(data1, vec![1, 2, 3]);

    let data2 = read_i32_column_data(col_ptrs[1], 3);
    assert_eq!(data2, vec![4, 5, 6]);

    unsafe {
        cudf_sys::cudf_column_destroy(col_ptrs[0]);
        cudf_sys::cudf_column_destroy(col_ptrs[1]);
        cudf_sys::cudf_free_column_array(out_columns);
    }
}

#[test]
fn test_table_release_columns_empty_table() {
    // Create an empty table (0 columns).
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(ptr::null_mut(), 0, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 0);
    // out_columns may or may not be null for 0-column tables, but count should be 0.
}

#[test]
fn test_table_release_columns_null_table_rejected() {
    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(ptr::null_mut(), &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
}

#[test]
fn test_table_release_columns_null_out_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);

    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, ptr::null_mut(), &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    // Table was not consumed on error — must still be destroyed.
    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}

#[test]
fn test_table_release_columns_null_out_count_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    // Table was not consumed on error — must still be destroyed.
    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}
