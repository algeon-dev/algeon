use std::ptr;

use crate::{
    helpers::*,
    support::{last_error_message, lock_shim_state},
};

#[test]
fn test_table_create_failure_does_not_partially_consume_columns() {
    let _shim_state = lock_shim_state();
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let mut columns = [c1, ptr::null_mut(), c2];

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), columns.len() as i32, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
    assert_eq!(columns[0], c1);
    assert!(columns[1].is_null());
    assert_eq!(columns[2], c2);
    assert!(last_error_message().contains("column handle is null"));

    let data1 = read_i32_column_data(columns[0], 3);
    assert_eq!(data1, vec![1, 2, 3]);

    let data2 = read_i32_column_data(columns[2], 3);
    assert_eq!(data2, vec![4, 5, 6]);

    let mut valid_columns = [columns[0], columns[2]];
    let mut tbl: cudf_sys::cudf_table_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(valid_columns.as_mut_ptr(), valid_columns.len() as i32, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());
    assert!(valid_columns.iter().all(|col| col.is_null()));

    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}

#[test]
fn test_table_release_columns_single_column() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 1);
    assert!(!out_columns.is_null());

    let released_col = unsafe { *out_columns };
    assert!(!released_col.is_null());

    let data = read_i32_column_data(released_col, 3);
    assert_eq!(data, vec![10, 20, 30]);

    unsafe {
        cudf_sys::cudf_column_destroy(released_col);
        cudf_sys::cudf_free_column_array(out_columns);
    }
}

#[test]
fn test_table_release_columns_multiple_columns() {
    let _shim_state = lock_shim_state();
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let mut col_arr = [c1, c2];
    let mut tbl: cudf_sys::cudf_table_handle = ptr::null_mut();
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
    let _shim_state = lock_shim_state();
    let mut tbl: cudf_sys::cudf_table_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(ptr::null_mut(), 0, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 0);
    assert!(out_columns.is_null());
}

#[test]
fn test_table_create_with_num_rows_preserves_zero_column_row_count() {
    let _shim_state = lock_shim_state();
    let mut tbl: cudf_sys::cudf_table_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create_with_num_rows(ptr::null_mut(), 0, 3, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());

    let tv = make_table_view(tbl);
    let mut num_cols: i32 = -1;
    let mut num_rows: i32 = -1;
    assert_eq!(unsafe { cudf_sys::cudf_table_view_num_columns(tv, &mut num_cols) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_table_view_num_rows(tv, &mut num_rows) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 0);
    assert_eq!(num_rows, 3);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_table_release_columns_null_table_rejected() {
    let _shim_state = lock_shim_state();
    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(ptr::null_mut(), &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
}

#[test]
fn test_table_release_columns_null_out_columns_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);

    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, ptr::null_mut(), &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}

#[test]
fn test_table_release_columns_null_out_count_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}

#[test]
fn test_table_release_columns_failure_does_not_consume_table() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[7, 8, 9]);
    let tbl = make_table_from_one_column(col);

    let mut out_count: i32 = -1;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, ptr::null_mut(), &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert_eq!(out_count, -1);

    let tv = make_table_view(tbl);
    let mut num_cols: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(tv, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 1);

    let mut released_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut released_count: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut released_columns, &mut released_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(released_count, 1);
    assert!(!released_columns.is_null());

    let released_col = unsafe { *released_columns };
    let data = read_i32_column_data(released_col, 3);
    assert_eq!(data, vec![7, 8, 9]);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_column_destroy(released_col);
        cudf_sys::cudf_free_column_array(released_columns);
    }
}

#[test]
fn test_table_release_columns_allocation_failure_does_not_consume_table() {
    let _shim_state = lock_shim_state();
    const NUM_COLUMNS: usize = 2;

    let mut columns: Vec<cudf_sys::cudf_column_handle> = (0..NUM_COLUMNS).map(|_| make_empty_i32_column()).collect();
    let mut tbl: cudf_sys::cudf_table_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), NUM_COLUMNS as i32, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());
    assert!(columns.iter().all(|col| col.is_null()));

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    unsafe { cudf_sys::test_support::cudf_test_fail_next_table_release_columns_handle_array_alloc() };
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);
    assert!(last_error_message().contains("failed to allocate column handle array"));

    let tv = make_table_view(tbl);
    let mut num_cols: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(tv, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, NUM_COLUMNS as i32);

    let mut released_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut released_count: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut released_columns, &mut released_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(released_count, NUM_COLUMNS as i32);
    assert!(!released_columns.is_null());

    let released = unsafe { std::slice::from_raw_parts(released_columns, released_count as usize) };
    for &col in released {
        unsafe { cudf_sys::cudf_column_destroy(col) };
    }

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_free_column_array(released_columns);
    }
}

#[test]
fn test_table_release_columns_publish_failure_does_not_consume_table() {
    let _shim_state = lock_shim_state();
    let mut columns = vec![make_i32_column(&[1, 2, 3]), make_i32_column(&[4, 5, 6])];
    let mut tbl: cudf_sys::cudf_table_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), columns.len() as i32, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());
    assert!(columns.iter().all(|col| col.is_null()));

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(1) };
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(-1) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);

    let tv = make_table_view(tbl);
    let mut num_cols: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(tv, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 2);

    let mut released_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut released_count: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut released_columns, &mut released_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(released_count, 2);
    assert!(!released_columns.is_null());

    let released = unsafe { std::slice::from_raw_parts(released_columns, released_count as usize) };
    assert_eq!(read_i32_column_data(released[0], 3), vec![1, 2, 3]);
    assert_eq!(read_i32_column_data(released[1], 3), vec![4, 5, 6]);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        for &col in released {
            cudf_sys::cudf_column_destroy(col);
        }
        cudf_sys::cudf_free_column_array(released_columns);
    }
}

#[test]
fn test_table_view_from_column_views_two_columns() {
    let _shim_state = lock_shim_state();
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let cv1 = make_column_view(c1);
    let cv2 = make_column_view(c2);

    let mut views = [cv1, cv2];
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let mut num_cols: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 2);

    let mut num_rows: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_rows(out, &mut num_rows) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_rows, 3);

    unsafe {
        cudf_sys::cudf_table_view_destroy(out);
        cudf_sys::cudf_column_view_destroy(cv1);
        cudf_sys::cudf_column_view_destroy(cv2);
        cudf_sys::cudf_column_destroy(c1);
        cudf_sys::cudf_column_destroy(c2);
    }
}

#[test]
fn test_table_view_from_column_views_single_column() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[10, 20]);
    let cv = make_column_view(col);

    let mut views = [cv];
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let mut num_cols: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 1);

    let mut num_rows: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_rows(out, &mut num_rows) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_rows, 2);

    unsafe {
        cudf_sys::cudf_table_view_destroy(out);
        cudf_sys::cudf_column_view_destroy(cv);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_table_view_from_column_views_null_out_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1]);
    let cv = make_column_view(col);

    let mut views = [cv];
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 1, ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe {
        cudf_sys::cudf_column_view_destroy(cv);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_table_view_from_column_views_negative_count_rejected() {
    let _shim_state = lock_shim_state();
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), -1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_table_view_from_column_views_null_array_with_nonzero_count_rejected() {
    let _shim_state = lock_shim_state();
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_table_view_from_column_views_null_element_rejected() {
    let _shim_state = lock_shim_state();
    let c1 = make_i32_column(&[1, 2, 3]);
    let cv1 = make_column_view(c1);

    let mut views = [cv1, ptr::null_mut()];
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(cv1);
        cudf_sys::cudf_column_destroy(c1);
    }
}

#[test]
fn test_table_view_from_column_views_zero_columns_succeeds() {
    let _shim_state = lock_shim_state();
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let mut num_cols: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 0);

    let mut num_rows: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_rows(out, &mut num_rows) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_rows, 0);

    unsafe {
        cudf_sys::cudf_table_view_destroy(out);
    }
}

#[test]
fn test_table_view_from_column_views_with_num_rows_preserves_zero_column_row_count() {
    let _shim_state = lock_shim_state();
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views_with_num_rows(ptr::null_mut(), 0, 3, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let mut num_cols: i32 = -1;
    let mut num_rows: i32 = -1;
    assert_eq!(unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_table_view_num_rows(out, &mut num_rows) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 0);
    assert_eq!(num_rows, 3);

    unsafe { cudf_sys::cudf_table_view_destroy(out) };
}

#[test]
fn test_table_view_from_column_views_with_num_rows_rejects_mismatched_columns() {
    let _shim_state = lock_shim_state();
    let column = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(column);
    let mut views = [view];
    let mut out: cudf_sys::cudf_table_view_handle = ptr::null_mut();

    let status =
        unsafe { cudf_sys::cudf_table_view_from_column_views_with_num_rows(views.as_mut_ptr(), 1, 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(column);
    }
}

#[test]
fn test_table_view_select_zero_columns_preserves_source_row_count() {
    let _shim_state = lock_shim_state();
    let tbl = make_table_from_columns(vec![make_i32_column(&[1, 2, 3]), make_i32_column(&[4, 5, 6])]);
    let source = make_table_view(tbl);
    let mut projected: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_select_columns(source, ptr::null(), 0, &mut projected) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!projected.is_null());

    let mut num_cols: i32 = -1;
    let mut num_rows: i32 = -1;
    assert_eq!(unsafe { cudf_sys::cudf_table_view_num_columns(projected, &mut num_cols) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_table_view_num_rows(projected, &mut num_rows) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 0);
    assert_eq!(num_rows, 3);

    unsafe {
        cudf_sys::cudf_table_view_destroy(projected);
        cudf_sys::cudf_table_view_destroy(source);
        cudf_sys::cudf_table_destroy(tbl);
    }
}
