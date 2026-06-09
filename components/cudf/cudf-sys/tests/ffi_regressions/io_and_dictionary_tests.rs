use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::*,
    support::{last_error_message, lock_shim_state},
};

#[test]
fn test_table_view_null_handle_clears_out() {
    let mut out = 1usize as *mut c_void;
    let status = unsafe { cudf_sys::cudf_table_view(ptr::null_mut(), &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_io_read_csv_missing_file_clears_out() {
    let path = unique_missing_path("csv");
    let mut out = 1usize as *mut c_void;
    let status = unsafe { cudf_sys::cudf_io_read_csv(path.as_ptr(), b',' as c_char, 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_IO);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_missing_file_clears_out() {
    let path = unique_missing_path("parquet");
    let mut out = 1usize as *mut c_void;
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, 0, -1, ptr::null(), 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_IO);
    assert!(out.is_null());
}

#[test]
fn test_non_io_runtime_error_maps_to_unknown() {
    let _shim_state = lock_shim_state();
    let status = unsafe { cudf_sys::cudf_test_throw_runtime_error() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_UNKNOWN);
    assert!(last_error_message().contains("non-io runtime failure"));
}

#[test]
fn test_bad_alloc_maps_to_out_of_memory() {
    let _shim_state = lock_shim_state();
    let status = unsafe { cudf_sys::cudf_test_throw_bad_alloc() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(!last_error_message().is_empty());
}

#[test]
fn test_invalid_argument_maps_to_invalid_argument() {
    let _shim_state = lock_shim_state();
    let status = unsafe { cudf_sys::cudf_test_throw_invalid_argument() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(last_error_message().contains("invalid argument"));
}

#[test]
fn test_io_read_parquet_with_metadata_invalid_args_clear_outputs() {
    let path = unique_missing_path("parquet");
    let mut out_table = 1usize as *mut c_void;
    let mut out_names = 1usize as *mut *mut c_char;
    let mut out_num_names = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            -1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
}

#[test]
fn test_io_write_csv_negative_num_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("csv");

    let out_status = unsafe { cudf_sys::cudf_io_write_csv(tv, path.as_ptr(), ptr::null(), -1, b',' as i8, 1) };
    assert_eq!(out_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_write_csv_null_column_name_entry_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("csv");
    let names = [ptr::null()];

    let status = unsafe { cudf_sys::cudf_io_write_csv(tv, path.as_ptr(), names.as_ptr(), 1, b',' as i8, 1) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_read_parquet_negative_num_columns_rejected() {
    let path = unique_missing_path("parquet");
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), -1, 0, -1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_null_column_name_entry_rejected() {
    let path = unique_missing_path("parquet");
    let columns = [ptr::null()];
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), columns.as_ptr(), 1, 0, -1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_negative_skip_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, -1, -1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_invalid_num_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, 0, -2, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_write_parquet_negative_num_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");

    let status = unsafe { cudf_sys::cudf_io_write_parquet(tv, path.as_ptr(), ptr::null(), -1, 0) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_write_parquet_null_column_name_entry_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");
    let names = [ptr::null()];

    let status = unsafe { cudf_sys::cudf_io_write_parquet(tv, path.as_ptr(), names.as_ptr(), 1, 0) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_read_parquet_with_metadata_negative_num_columns_rejected() {
    let path = unique_missing_path("parquet");
    let mut out_table: *mut c_void = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            -1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_io_read_parquet_with_metadata_null_column_name_entry_rejected() {
    let path = unique_missing_path("parquet");
    let columns = [ptr::null()];
    let mut out_table: *mut c_void = 1usize as *mut c_void;
    let mut out_names: *mut *mut i8 = 1usize as *mut *mut i8;
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            columns.as_ptr(),
            1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
}

#[test]
fn test_io_read_parquet_with_metadata_negative_skip_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out_table: *mut c_void = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            0,
            -1,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_io_read_parquet_with_metadata_invalid_num_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out_table: *mut c_void = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            0,
            0,
            -2,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_dictionary_match_dictionaries_invalid_num_columns_clears_outputs() {
    let dict_columns = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_columns = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_count: i32 = -1;

    let status =
        unsafe { cudf_sys::cudf_dictionary_match_dictionaries(dict_columns, 0, &mut out_columns, &mut out_count) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);
}

#[test]
fn test_replace_nulls_policy_rejects_invalid_policy_value() {
    let col = make_i32_column_nullable(&[1, 0, 3], &[true, false, true]);
    let view = make_column_view(col);

    let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe { cudf_sys::cudf_replace_nulls_policy(view, 42, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null(), "invalid policy must not produce an output column");
    assert!(
        last_error_message().contains("policy"),
        "expected invalid policy error message, got: {}",
        last_error_message()
    );

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_dictionary_match_dictionaries_null_entry_clears_outputs() {
    let dict = make_dictionary_i32_column(&[1, 2, 1]);
    let dict_view = make_column_view(dict);
    let dict_columns = [dict_view, ptr::null_mut()];

    let mut out_columns = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_count: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(dict_view);
        cudf_sys::cudf_column_destroy(dict);
    }
}

#[test]
fn test_dictionary_match_dictionaries_returns_heap_array() {
    let left = make_dictionary_i32_column(&[10, 20, 10, 30]);
    let right = make_dictionary_i32_column(&[20, 30, 40, 20]);
    let left_view = make_column_view(left);
    let right_view = make_column_view(right);
    let dict_columns = [left_view, right_view];

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, dict_columns.len() as i32);
    assert!(!out_columns.is_null());

    let matched = unsafe { std::slice::from_raw_parts(out_columns, out_count as usize) };
    assert_eq!(decode_dictionary_i32_column(matched[0], 4), vec![10, 20, 10, 30]);
    assert_eq!(decode_dictionary_i32_column(matched[1], 4), vec![20, 30, 40, 20]);

    unsafe {
        for &col in matched {
            cudf_sys::cudf_column_destroy(col);
        }
        cudf_sys::cudf_free_column_array(out_columns);
        cudf_sys::cudf_column_view_destroy(right_view);
        cudf_sys::cudf_column_view_destroy(left_view);
        cudf_sys::cudf_column_destroy(right);
        cudf_sys::cudf_column_destroy(left);
    }
}
