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

fn make_empty_i32_column() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_create_empty(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out) };
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

fn last_error_message() -> String {
    let ptr = unsafe { cudf_sys::cudf_last_error_message() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

unsafe extern "C" {
    fn cudf_test_fail_next_table_release_columns_handle_array_alloc();
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
fn test_scalar_create_numeric_invalid_allows_null_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, ptr::null(), 0, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());

    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid(scalar, &mut is_valid) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_numeric_valid_requires_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, ptr::null(), 1, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

#[test]
fn test_stream_is_ready_null_out_rejected() {
    let mut stream: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_stream_create(&mut stream) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let st = unsafe { cudf_sys::cudf_stream_is_ready(stream, ptr::null_mut()) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);

    unsafe { cudf_sys::cudf_stream_destroy(stream) };
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
fn test_sort_invalid_order_enum_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let column_order = [99i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_sort(tv, column_order.as_ptr(), ptr::null(), 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_sort_invalid_null_order_enum_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let null_order = [99i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_sort(tv, ptr::null(), null_order.as_ptr(), 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_drop_nulls_negative_keep_threshold_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls(tv, ptr::null(), 0, -1, &mut out) };

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
fn test_column_view_to_host_string_type_rejected() {
    let strings = [std::ffi::CString::new("ab").unwrap(), std::ffi::CString::new("cd").unwrap()];
    let string_ptrs = [strings[0].as_ptr(), strings[1].as_ptr()];
    let lengths = [2i32, 2i32];

    let mut col: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(string_ptrs.as_ptr(), lengths.as_ptr(), 2, ptr::null(), 0, &mut col)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!col.is_null());

    let view = make_column_view(col);
    let mut buf = [0u8; 16];
    let status = unsafe { cudf_sys::cudf_column_view_to_host(view, buf.as_mut_ptr() as *mut c_void, buf.len()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_TYPE);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_create_strings_negative_length_rejected() {
    let strings = [std::ffi::CString::new("ab").unwrap()];
    let string_ptrs = [strings[0].as_ptr()];
    let lengths = [-1i32];

    let mut col: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(string_ptrs.as_ptr(), lengths.as_ptr(), 1, ptr::null(), 0, &mut col)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(col.is_null());
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
fn test_groupby_aggregation_count_mismatch_rejected() {
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);

    let values = [val_view];
    let agg_counts = [1i32];
    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(0, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());
    let aggregations = [agg];

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = ptr::null_mut();
    let mut out_total: i32 = -1;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys_view,
            values.as_ptr(),
            aggregations.as_ptr(),
            agg_counts.as_ptr(),
            1,
            0,
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
        cudf_sys::cudf_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(val_view);
        cudf_sys::cudf_column_destroy(val_col);
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

#[test]
fn test_io_write_csv_negative_num_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = std::ffi::CString::new("/tmp/cudf_sys_write_csv_negative_num_columns.csv").unwrap();

    let out_status = unsafe { cudf_sys::cudf_io_write_csv(tv, path.as_ptr(), ptr::null(), -1, b',' as i8, 1) };
    assert_eq!(out_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_read_parquet_negative_num_columns_rejected() {
    let path = std::ffi::CString::new("/tmp/does-not-matter.parquet").unwrap();
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), -1, 0, -1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_negative_skip_rows_rejected() {
    let path = std::ffi::CString::new("/tmp/does-not-matter.parquet").unwrap();
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, -1, -1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_invalid_num_rows_rejected() {
    let path = std::ffi::CString::new("/tmp/does-not-matter.parquet").unwrap();
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, 0, -2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_write_parquet_negative_num_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = std::ffi::CString::new("/tmp/cudf_sys_write_parquet_negative_num_columns.parquet").unwrap();

    let status = unsafe { cudf_sys::cudf_io_write_parquet(tv, path.as_ptr(), ptr::null(), -1) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_read_parquet_with_metadata_negative_num_columns_rejected() {
    let path = std::ffi::CString::new("/tmp/does-not-matter.parquet").unwrap();
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
fn test_io_read_parquet_with_metadata_negative_skip_rows_rejected() {
    let path = std::ffi::CString::new("/tmp/does-not-matter.parquet").unwrap();
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
    let path = std::ffi::CString::new("/tmp/does-not-matter.parquet").unwrap();
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
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

// ===========================================================================
// cudf_table_create tests
// ===========================================================================

#[test]
fn test_table_create_failure_does_not_partially_consume_columns() {
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let mut columns = [c1, ptr::null_mut(), c2];

    let mut out: *mut c_void = ptr::null_mut();
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
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(valid_columns.as_mut_ptr(), valid_columns.len() as i32, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());
    assert!(valid_columns.iter().all(|col| col.is_null()));

    unsafe { cudf_sys::cudf_table_destroy(tbl) };
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

#[test]
fn test_table_release_columns_failure_does_not_consume_table() {
    let col = make_i32_column(&[7, 8, 9]);
    let tbl = make_table_from_one_column(col);

    let mut out_count: i32 = -1;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, ptr::null_mut(), &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert_eq!(out_count, -1);

    // Regression check: on failure, the table must remain valid and usable.
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
    const NUM_COLUMNS: usize = 2;

    let mut columns: Vec<*mut c_void> = (0..NUM_COLUMNS).map(|_| make_empty_i32_column()).collect();
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), NUM_COLUMNS as i32, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());
    assert!(columns.iter().all(|col| col.is_null()));

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    unsafe { cudf_test_fail_next_table_release_columns_handle_array_alloc() };
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);
    assert!(last_error_message().contains("failed to allocate column handle array"));

    // Regression check: a failure after entering the function body must not
    // consume the table; the caller should still own and be able to use it.
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

// ===========================================================================
// cudf_table_view_from_column_views tests
// ===========================================================================

#[test]
fn test_table_view_from_column_views_two_columns() {
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let cv1 = make_column_view(c1);
    let cv2 = make_column_view(c2);

    let mut views = [cv1, cv2];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    // Verify num_columns == 2.
    let mut num_cols: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 2);

    // Verify num_rows == 3.
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
    let col = make_i32_column(&[10, 20]);
    let cv = make_column_view(col);

    let mut views = [cv];
    let mut out: *mut c_void = ptr::null_mut();
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
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), -1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_table_view_from_column_views_null_array_with_nonzero_count_rejected() {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_table_view_from_column_views_null_element_rejected() {
    let c1 = make_i32_column(&[1, 2, 3]);
    let cv1 = make_column_view(c1);

    let mut views = [cv1, ptr::null_mut()];
    let mut out: *mut c_void = ptr::null_mut();
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
    let mut out: *mut c_void = ptr::null_mut();
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
