use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::*,
    support::{
        CurrentDeviceResourceRestore, NO_SYNC_RECORDED, OwnedMemoryResource, OwnedStream, current_device_resource_raw,
        last_error_message, last_exec_mr_raw, last_exec_stream_value, last_exec_sync_stream_value, lock_shim_state,
        reset_exec_observation, stream_value,
    },
};

fn read_bool8_column_data(col: *mut c_void) -> Vec<u8> {
    let view = make_column_view(col);

    let mut type_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
    let status = unsafe { cudf_sys::cudf_column_view_type(view, &mut type_id) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, cudf_sys::cudf_type_id_t::CUDF_TYPE_BOOL8);

    let mut size = 0;
    let status = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(size >= 0);

    let mut values = vec![0u8; size as usize];
    let status = unsafe {
        cudf_sys::cudf_column_view_to_host(
            view,
            values.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(values.as_slice()),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    values
}

fn make_bool8_column(data: &[u8]) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_BOOL8,
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

fn column_len(col: *mut c_void) -> usize {
    let view = make_column_view(col);
    let mut size = 0;
    let status = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(size >= 0);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    size as usize
}

fn read_all_i32_column_data(col: *mut c_void) -> Vec<i32> {
    read_i32_column_data(col, column_len(col))
}

fn assert_readable_nonempty_i32_indices(col: *mut c_void, context: &str) {
    let indices = read_all_i32_column_data(col);
    assert!(!indices.is_empty(), "{context} must return at least one index");
}

fn make_ast_column_reference(table_reference: cudf_sys::cudf_ast_table_reference_t) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_ast_column_reference_create(0, table_reference, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_left_less_than_right_predicate() -> *mut c_void {
    let left = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_LEFT);
    let right = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_RIGHT);
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_ast_operation_binary_create(cudf_sys::cudf_ast_operator_t::CUDF_AST_LESS, left, right, &mut out)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    unsafe {
        cudf_sys::cudf_ast_expression_destroy(right);
        cudf_sys::cudf_ast_expression_destroy(left);
    }
    out
}

#[test]
fn test_column_view_strings_to_host_empty_rejects_undersized_offsets_buffer() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let mut offsets = vec![-1i32; 1];
    let mut total_chars = -1;

    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(view, offsets.as_mut_ptr(), 0, ptr::null_mut(), 0, &mut total_chars)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert_eq!(offsets, vec![-1]);
    assert_eq!(total_chars, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_strings_like_validates_pattern_and_escape_arguments() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["abc"]);
    let view = make_column_view(col);
    let pattern = b"a%";
    let escape = b"/";

    let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status =
        unsafe { cudf_sys::cudf_strings_like(view, pattern.as_ptr() as *const c_char, -1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe { cudf_sys::cudf_strings_like(view, ptr::null(), 1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_strings_like(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            ptr::null(),
            -1,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_strings_like(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            escape.as_ptr() as *const c_char,
            2,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_strings_like(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            ptr::null(),
            1,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_strings_like_matches_and_synchronizes_explicit_stream() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["abc", "adc", "ab", "axc"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let pattern = b"a_c";

    reset_exec_observation();
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_like_ex(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            ptr::null(),
            0,
            &mut out,
            exec.as_raw(),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
    assert_eq!(last_exec_sync_stream_value(), stream_value(stream.as_raw()));
    assert_eq!(read_bool8_column_data(out), vec![1, 1, 0, 1]);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_strings_slice_validates_step_and_initializes_output() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["abcdef"]);
    let view = make_column_view(col);
    let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();

    let status = unsafe { cudf_sys::cudf_strings_slice(view, 0, 1, 3, 1, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_strings_slice_uses_validity_flags_and_explicit_stream() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["abcdef", "xy"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_slice_ex(view, 1, 1, 4, 1, 2, &mut out, exec.as_raw()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_string_column_data(out), vec!["bd".to_string(), "y".to_string()]);
    unsafe { cudf_sys::cudf_column_destroy(out) };

    out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_slice(view, 2, 1, 0, 0, 1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(read_string_column_data(out), vec!["cdef".to_string(), "".to_string()]);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_concatenate_columns_null_element_returns_null_error() {
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let column_order = [99i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sort_raw(tv, column_order.as_ptr(), ptr::null(), 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_sort_invalid_null_order_enum_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let null_order = [99i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sort_raw(tv, ptr::null(), null_order.as_ptr(), 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_drop_nulls_negative_keep_threshold_rejected() {
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), -1, 0, 0, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_gather_invalid_bounds_policy_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let map_col = make_i32_column(&[0, 2]);
    let map_view = make_column_view(map_col);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_gather_raw(tv, map_view, 99, &mut out) };

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
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();

    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 99, 0, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 0, 99, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 0, 0, 99, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_column_create_numeric_negative_size_rejected() {
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
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
fn test_column_create_fixed_width_from_host_preserves_decimal_scale() {
    let _shim_state = lock_shim_state();
    let data = [12345i128, -67890, 5];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128,
            -2,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let view = make_column_view(out);
    let mut type_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
    let mut scale = 0;
    let status = unsafe { cudf_sys::cudf_column_view_data_type(view, &mut type_id, &mut scale) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128);
    assert_eq!(scale, -2);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(out);
    }
}

#[test]
fn test_column_create_fixed_width_from_host_rejects_nonzero_scale_for_non_decimal() {
    let _shim_state = lock_shim_state();
    let data = [1i32, 2, 3];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            -2,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("scale must be 0"));
}

#[test]
fn test_column_create_fixed_width_from_host_rejects_non_fixed_width_type() {
    let _shim_state = lock_shim_state();
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_STRING,
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_TYPE);
    assert!(out.is_null());
}

#[test]
fn test_column_create_fixed_width_from_host_null_count_requires_mask() {
    let _shim_state = lock_shim_state();
    let data = [1i32, 2, 3];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            0,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            1,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_column_view_to_host_string_type_rejected() {
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
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
fn test_column_create_strings_positive_length_requires_non_null_string_pointer() {
    let _shim_state = lock_shim_state();
    let string_ptrs = [ptr::null()];
    let lengths = [2i32];

    let mut col: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(string_ptrs.as_ptr(), lengths.as_ptr(), 1, ptr::null(), 0, &mut col)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(col.is_null());
}

#[test]
fn test_column_create_strings_rejects_count_that_would_overflow_offsets() {
    let _shim_state = lock_shim_state();
    let strings = [std::ffi::CString::new("").unwrap()];
    let string_ptrs = [strings[0].as_ptr()];
    let lengths = [0i32];

    let mut col: *mut c_void = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(string_ptrs.as_ptr(), lengths.as_ptr(), i32::MAX, ptr::null(), 0, &mut col)
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(col.is_null());
    assert!(last_error_message().contains("num_strings"));
}

#[test]
fn test_drop_nulls_zero_num_keys_allows_null_keys_pointer() {
    let _shim_state = lock_shim_state();
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
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct(
            tv,
            ptr::null(),
            0,
            cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            cudf_sys::cudf_stream_compaction_null_equality_t::CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL,
            cudf_sys::cudf_stream_compaction_nan_equality_t::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL,
            &mut out,
        )
    };

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
    let _shim_state = lock_shim_state();
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
            cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE,
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
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);

    let values = [val_view];
    let agg_counts = [1i32];
    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::cudf_aggregation_kind_t::CUDF_AGG_SUM, &mut agg) };
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
            cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE,
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
fn test_aggregation_create_count_invalid_null_policy_rejected() {
    let _shim_state = lock_shim_state();
    let mut agg: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_aggregation_create_count_raw(99, &mut agg) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(agg.is_null());
    assert!(last_error_message().contains("invalid null_policy"));
}

#[test]
fn test_groupby_invalid_keys_null_policy_rejected() {
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);
    let values = [val_view];
    let agg_counts = [1i32];

    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::cudf_aggregation_kind_t::CUDF_AGG_SUM, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());
    let aggregations = [agg];

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = ptr::null_mut();
    let mut out_total: i32 = -1;
    let status = unsafe {
        cudf_groupby_aggregate_raw(
            keys_view,
            values.as_ptr(),
            aggregations.as_ptr(),
            agg_counts.as_ptr(),
            1,
            1,
            99,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);
    assert!(last_error_message().contains("invalid keys_null_policy"));

    unsafe {
        cudf_sys::cudf_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(val_view);
        cudf_sys::cudf_column_destroy(val_col);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

#[test]
fn test_groupby_zero_aggregations_returns_null_output_array() {
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_total: i32 = -1;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys_view,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            0,
            cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    unsafe {
        cudf_sys::cudf_table_destroy(out_keys);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

fn assert_join_invalid_null_equality_rejected(
    join_fn: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        i32,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
) {
    let left_key_col = make_i32_column(&[1, 2]);
    let right_key_col = make_i32_column(&[2, 3]);
    let left_keys_tbl = make_table_from_one_column(left_key_col);
    let right_keys_tbl = make_table_from_one_column(right_key_col);
    let left_keys_view = make_table_view(left_keys_tbl);
    let right_keys_view = make_table_view(right_keys_tbl);
    let left_payload_tbl = make_empty_table();
    let right_payload_tbl = make_empty_table();
    let left_payload_view = make_table_view(left_payload_tbl);
    let right_payload_view = make_table_view(right_payload_tbl);

    let mut out_left: *mut c_void = ptr::null_mut();
    let mut out_right: *mut c_void = ptr::null_mut();
    let status = unsafe {
        join_fn(
            left_keys_view,
            right_keys_view,
            left_payload_view,
            right_payload_view,
            99,
            &mut out_left,
            &mut out_right,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_left.is_null());
    assert!(out_right.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(right_payload_view);
        cudf_sys::cudf_table_view_destroy(left_payload_view);
        cudf_sys::cudf_table_view_destroy(right_keys_view);
        cudf_sys::cudf_table_view_destroy(left_keys_view);
        cudf_sys::cudf_table_destroy(right_payload_tbl);
        cudf_sys::cudf_table_destroy(left_payload_tbl);
        cudf_sys::cudf_table_destroy(right_keys_tbl);
        cudf_sys::cudf_table_destroy(left_keys_tbl);
    }
}

fn assert_join_indices_invalid_null_equality_rejected(
    join_fn: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, *mut *mut c_void, *mut *mut c_void) -> i32,
) {
    let left_key_col = make_i32_column(&[1, 2]);
    let right_key_col = make_i32_column(&[2, 3]);
    let left_keys_tbl = make_table_from_one_column(left_key_col);
    let right_keys_tbl = make_table_from_one_column(right_key_col);
    let left_keys_view = make_table_view(left_keys_tbl);
    let right_keys_view = make_table_view(right_keys_tbl);

    let mut out_left: *mut c_void = ptr::null_mut();
    let mut out_right: *mut c_void = ptr::null_mut();
    let status = unsafe { join_fn(left_keys_view, right_keys_view, 99, &mut out_left, &mut out_right) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_left.is_null());
    assert!(out_right.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(right_keys_view);
        cudf_sys::cudf_table_view_destroy(left_keys_view);
        cudf_sys::cudf_table_destroy(right_keys_tbl);
        cudf_sys::cudf_table_destroy(left_keys_tbl);
    }
}

fn assert_filtered_join_indices_invalid_null_equality_rejected(
    join_fn: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, *mut *mut c_void) -> i32,
) {
    let left_key_col = make_i32_column(&[1, 2]);
    let right_key_col = make_i32_column(&[2, 3]);
    let left_keys_tbl = make_table_from_one_column(left_key_col);
    let right_keys_tbl = make_table_from_one_column(right_key_col);
    let left_keys_view = make_table_view(left_keys_tbl);
    let right_keys_view = make_table_view(right_keys_tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { join_fn(left_keys_view, right_keys_view, 99, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(right_keys_view);
        cudf_sys::cudf_table_view_destroy(left_keys_view);
        cudf_sys::cudf_table_destroy(right_keys_tbl);
        cudf_sys::cudf_table_destroy(left_keys_tbl);
    }
}

#[test]
fn test_inner_join_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_invalid_null_equality_rejected(cudf_inner_join_raw);
}

#[test]
fn test_left_join_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_invalid_null_equality_rejected(cudf_left_join_raw);
}

#[test]
fn test_full_join_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_invalid_null_equality_rejected(cudf_full_join_raw);
}

#[test]
fn test_inner_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_indices_invalid_null_equality_rejected(cudf_inner_join_indices_raw);
}

#[test]
fn test_left_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_indices_invalid_null_equality_rejected(cudf_left_join_indices_raw);
}

#[test]
fn test_full_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_indices_invalid_null_equality_rejected(cudf_full_join_indices_raw);
}

#[test]
fn test_left_semi_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_filtered_join_indices_invalid_null_equality_rejected(cudf_left_semi_join_indices_raw);
}

#[test]
fn test_left_anti_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_filtered_join_indices_invalid_null_equality_rejected(cudf_left_anti_join_indices_raw);
}

#[test]
fn test_distinct_hash_join_create_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_distinct_hash_join_create_raw(tv, 99, &mut join) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(join.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_hash_join_create_invalid_nullable_join_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_hash_join_create_raw(
            tv,
            99,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL as i32,
            0.5,
            &mut join,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(join.is_null());
    assert!(last_error_message().contains("invalid nullable_join"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_hash_join_create_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_hash_join_create_raw(tv, 0, 99, 0.5, &mut join) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(join.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_ast_invalid_table_reference_operator_and_null_handle_clear_output() {
    let _shim_state = lock_shim_state();

    let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe { cudf_ast_column_reference_create_raw(0, 99, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid AST table reference"));

    let left = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_LEFT);
    let right = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_RIGHT);

    out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe { cudf_ast_operation_binary_create_raw(99, left, right, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid AST operator"));

    out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_ast_operation_unary_create(
            cudf_sys::cudf_ast_operator_t::CUDF_AST_NOT,
            ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe { cudf_ast_operation_unary_create_raw(99, left, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid AST operator"));

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(right);
        cudf_sys::cudf_ast_expression_destroy(left);
    }
}

#[test]
fn test_conditional_join_default_forms_validate_null_handles_and_clear_outputs() {
    let _shim_state = lock_shim_state();
    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 4]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let predicate = make_left_less_than_right_predicate();

    for (name, join_indices) in [
        ("inner", cudf_conditional_inner_join_indices_raw as ConditionalPairJoinRaw),
        ("left", cudf_conditional_left_join_indices_raw),
        ("full", cudf_conditional_full_join_indices_raw),
    ] {
        let mut out_right = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        let status = unsafe { join_indices(left_tv, right_tv, predicate, ptr::null_mut(), &mut out_right) };
        assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null left output");
        assert!(last_error_message().contains("out_left_indices is null"));

        let mut out_left = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        let status = unsafe { join_indices(left_tv, right_tv, predicate, &mut out_left, ptr::null_mut()) };
        assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null right output");
        assert!(out_left.is_null(), "{name} conditional join must clear left output first");
        assert!(last_error_message().contains("out_right_indices is null"));

        for (case, left, right, pred, message) in [
            ("left", ptr::null_mut(), right_tv, predicate, "left is null"),
            ("right", left_tv, ptr::null_mut(), predicate, "right is null"),
            ("predicate", left_tv, right_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out_left = std::ptr::NonNull::<c_void>::dangling().as_ptr();
            let mut out_right = std::ptr::NonNull::<c_void>::dangling().as_ptr();
            let status = unsafe { join_indices(left, right, pred, &mut out_left, &mut out_right) };
            assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null {case} status");
            assert!(out_left.is_null(), "{name} conditional join must clear left output for null {case}");
            assert!(out_right.is_null(), "{name} conditional join must clear right output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    for (name, join_indices) in [
        ("semi", cudf_conditional_left_semi_join_indices_raw as ConditionalSingleJoinRaw),
        ("anti", cudf_conditional_left_anti_join_indices_raw),
    ] {
        let status = unsafe { join_indices(left_tv, right_tv, predicate, ptr::null_mut()) };
        assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null output");
        assert!(last_error_message().contains("out_indices is null"));

        for (case, left, right, pred, message) in [
            ("left", ptr::null_mut(), right_tv, predicate, "left is null"),
            ("right", left_tv, ptr::null_mut(), predicate, "right is null"),
            ("predicate", left_tv, right_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
            let status = unsafe { join_indices(left, right, pred, &mut out) };
            assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null {case} status");
            assert!(out.is_null(), "{name} conditional join must clear output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_hash_join_create_ex_uses_explicit_memory_resource_for_build_allocations() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);

    let target_before = target_mr.allocation_count();
    reset_exec_observation();

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_create_ex(
            tv,
            cudf_sys::cudf_nullable_join_t::CUDF_NULLABLE_JOIN_YES,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            0.5,
            &mut join,
            exec.as_raw(),
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        current_device_resource_raw(),
        previous_raw,
        "hash-join construction must restore the caller's current device resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "hash-join build allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_hash_join_probe_ex_uses_explicit_memory_resource_for_probe_allocations() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let build_col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_create(
            build_tv,
            cudf_sys::cudf_nullable_join_t::CUDF_NULLABLE_JOIN_YES,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            0.5,
            &mut join,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_one_column(make_i32_column(&[8, 7, 42, 1, 2, 99, 3, 4]));
    let probe_tv = make_table_view(probe_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);

    for (name, probe) in [
        (
            "inner",
            cudf_sys::cudf_hash_join_inner_join_ex
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut *mut c_void,
                    *mut *mut c_void,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_hash_join_left_join_ex),
        ("full", cudf_sys::cudf_hash_join_full_join_ex),
    ] {
        let target_before = target_mr.allocation_count();
        let current_before = current_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: *mut c_void = ptr::null_mut();
        let mut out_right: *mut c_void = ptr::null_mut();
        let current_borrowed = current_mr.borrowed();
        cudf_sys::test_set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");
        let status = unsafe { probe(join, probe_tv, &mut out_left, &mut out_right, exec.as_raw()) };
        let restored_current = current_device_resource_raw();
        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} hash-join probe failed");
        assert!(!out_left.is_null(), "{name} hash-join probe did not publish left indices");
        assert!(!out_right.is_null(), "{name} hash-join probe did not publish right indices");
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} hash-join probe must restore the caller's current device resource"
        );
        assert_eq!(
            current_mr.allocation_count(),
            current_before,
            "{name} hash-join probe temporaries must not allocate from the caller's current memory resource"
        );
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} hash-join probe allocations must be charged to the explicit execution memory resource"
        );

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

#[test]
fn test_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 3]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2, 2, 4]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);

    for (name, join_indices) in [
        (
            "inner",
            cudf_sys::cudf_inner_join_indices_ex
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    cudf_sys::cudf_null_equality_t,
                    *mut *mut c_void,
                    *mut *mut c_void,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_left_join_indices_ex),
        ("full", cudf_sys::cudf_full_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        let current_before = current_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: *mut c_void = ptr::null_mut();
        let mut out_right: *mut c_void = ptr::null_mut();
        let current_borrowed = current_mr.borrowed();
        cudf_sys::test_set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");
        let status = unsafe {
            join_indices(
                left_tv,
                right_tv,
                cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
                &mut out_left,
                &mut out_right,
                exec.as_raw(),
            )
        };
        let restored_current = current_device_resource_raw();
        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} join indices failed");
        assert!(!out_left.is_null(), "{name} join indices did not publish left indices");
        assert!(!out_right.is_null(), "{name} join indices did not publish right indices");
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} join indices must restore the caller's current memory resource"
        );
        assert_eq!(
            current_mr.allocation_count(),
            current_before,
            "{name} join-index temporaries must not allocate from the caller's current memory resource"
        );
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} join-index allocations must be charged to the explicit execution memory resource"
        );

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_filtered_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 3, 5, 8, 13, 21]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2, 3, 13]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);

    for (name, join_indices) in [
        (
            "semi",
            cudf_sys::cudf_left_semi_join_indices_ex
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    cudf_sys::cudf_null_equality_t,
                    *mut *mut c_void,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("anti", cudf_sys::cudf_left_anti_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        let current_before = current_mr.allocation_count();
        reset_exec_observation();

        let mut out: *mut c_void = ptr::null_mut();
        let current_borrowed = current_mr.borrowed();
        cudf_sys::test_set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");
        let status = unsafe {
            join_indices(
                left_tv,
                right_tv,
                cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
                &mut out,
                exec.as_raw(),
            )
        };
        let restored_current = current_device_resource_raw();
        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} filtered join indices failed");
        assert!(!out.is_null(), "{name} filtered join indices did not publish indices");
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} filtered join indices must restore the caller's current memory resource"
        );
        assert_eq!(
            current_mr.allocation_count(),
            current_before,
            "{name} filtered join temporaries must not allocate from the caller's current memory resource"
        );
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} filtered join allocations must be charged to the explicit execution memory resource"
        );

        unsafe { cudf_sys::cudf_column_destroy(out) };
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_conditional_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 4]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let predicate = make_left_less_than_right_predicate();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);

    for (name, join_indices) in [
        (
            "inner",
            cudf_sys::cudf_conditional_inner_join_indices_ex
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut *mut c_void,
                    *mut *mut c_void,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_conditional_left_join_indices_ex),
        ("full", cudf_sys::cudf_conditional_full_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        let current_before = current_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: *mut c_void = ptr::null_mut();
        let mut out_right: *mut c_void = ptr::null_mut();
        let current_borrowed = current_mr.borrowed();
        cudf_sys::test_set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");
        let status =
            unsafe { join_indices(left_tv, right_tv, predicate, &mut out_left, &mut out_right, exec.as_raw()) };
        let restored_current = current_device_resource_raw();
        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} conditional join failed");
        assert!(!out_left.is_null(), "{name} conditional join did not publish left indices");
        assert!(!out_right.is_null(), "{name} conditional join did not publish right indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} conditional join must restore the caller's current memory resource"
        );
        assert_eq!(
            current_mr.allocation_count(),
            current_before,
            "{name} conditional join temporaries must not allocate from the caller's current memory resource"
        );
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} conditional join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out_left, &format!("{name} conditional left indices"));
        assert_readable_nonempty_i32_indices(out_right, &format!("{name} conditional right indices"));

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    for (name, join_indices) in [
        (
            "semi",
            cudf_sys::cudf_conditional_left_semi_join_indices_ex
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut *mut c_void,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("anti", cudf_sys::cudf_conditional_left_anti_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        let current_before = current_mr.allocation_count();
        reset_exec_observation();

        let mut out: *mut c_void = ptr::null_mut();
        let current_borrowed = current_mr.borrowed();
        cudf_sys::test_set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");
        let status = unsafe { join_indices(left_tv, right_tv, predicate, &mut out, exec.as_raw()) };
        let restored_current = current_device_resource_raw();
        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} conditional join failed");
        assert!(!out.is_null(), "{name} conditional join did not publish indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} conditional join must restore the caller's current memory resource"
        );
        assert_eq!(
            current_mr.allocation_count(),
            current_before,
            "{name} conditional join temporaries must not allocate from the caller's current memory resource"
        );
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} conditional join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out, &format!("{name} conditional indices"));

        unsafe { cudf_sys::cudf_column_destroy(out) };
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_mixed_join_invalid_null_equality_rejected_and_clears_outputs() {
    let _shim_state = lock_shim_state();
    let left_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 1, 2, 3]));
    let right_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 4]));
    let left_cond_tbl = make_table_from_one_column(make_i32_column(&[10, 30, 5, 7]));
    let right_cond_tbl = make_table_from_one_column(make_i32_column(&[20, 10, 6, 8]));
    let left_eq_tv = make_table_view(left_eq_tbl);
    let right_eq_tv = make_table_view(right_eq_tbl);
    let left_cond_tv = make_table_view(left_cond_tbl);
    let right_cond_tv = make_table_view(right_cond_tbl);
    let predicate = make_left_less_than_right_predicate();

    for (name, join_indices) in [
        (
            "inner",
            cudf_mixed_inner_join_indices_raw
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    i32,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> i32,
        ),
        ("left", cudf_mixed_left_join_indices_raw),
        ("full", cudf_mixed_full_join_indices_raw),
    ] {
        let mut out_left = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        let mut out_right = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                99,
                &mut out_left,
                &mut out_right,
            )
        };

        assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION, "{name} mixed join status");
        assert!(out_left.is_null(), "{name} mixed join must clear left output");
        assert!(out_right.is_null(), "{name} mixed join must clear right output");
        assert!(last_error_message().contains("invalid null_equality"));
    }

    for (name, join_indices) in [
        (
            "semi",
            cudf_mixed_left_semi_join_indices_raw
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    i32,
                    *mut *mut c_void,
                ) -> i32,
        ),
        ("anti", cudf_mixed_left_anti_join_indices_raw),
    ] {
        let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        let status =
            unsafe { join_indices(left_eq_tv, right_eq_tv, left_cond_tv, right_cond_tv, predicate, 99, &mut out) };

        assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION, "{name} mixed join status");
        assert!(out.is_null(), "{name} mixed join must clear output");
        assert!(last_error_message().contains("invalid null_equality"));
    }

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_cond_tv);
        cudf_sys::cudf_table_view_destroy(left_cond_tv);
        cudf_sys::cudf_table_view_destroy(right_eq_tv);
        cudf_sys::cudf_table_view_destroy(left_eq_tv);
        cudf_sys::cudf_table_destroy(right_cond_tbl);
        cudf_sys::cudf_table_destroy(left_cond_tbl);
        cudf_sys::cudf_table_destroy(right_eq_tbl);
        cudf_sys::cudf_table_destroy(left_eq_tbl);
    }
}

#[test]
fn test_mixed_join_null_handles_clear_outputs() {
    let _shim_state = lock_shim_state();
    let left_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 1, 2, 3]));
    let right_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 4]));
    let left_cond_tbl = make_table_from_one_column(make_i32_column(&[10, 30, 5, 7]));
    let right_cond_tbl = make_table_from_one_column(make_i32_column(&[20, 10, 6, 8]));
    let left_eq_tv = make_table_view(left_eq_tbl);
    let right_eq_tv = make_table_view(right_eq_tbl);
    let left_cond_tv = make_table_view(left_cond_tbl);
    let right_cond_tv = make_table_view(right_cond_tbl);
    let predicate = make_left_less_than_right_predicate();
    let null_equality = cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL as i32;

    for (name, join_indices) in [
        ("inner", cudf_mixed_inner_join_indices_raw as MixedPairJoinRaw),
        ("left", cudf_mixed_left_join_indices_raw),
        ("full", cudf_mixed_full_join_indices_raw),
    ] {
        let mut out_right = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                null_equality,
                ptr::null_mut(),
                &mut out_right,
            )
        };
        assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null left output");
        assert!(last_error_message().contains("out_left_indices is null"));

        let mut out_left = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                null_equality,
                &mut out_left,
                ptr::null_mut(),
            )
        };
        assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null right output");
        assert!(out_left.is_null(), "{name} mixed join must clear left output first");
        assert!(last_error_message().contains("out_right_indices is null"));

        for (case, left_eq, right_eq, left_cond, right_cond, pred, message) in [
            (
                "left_equality",
                ptr::null_mut(),
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                "left_equality is null",
            ),
            (
                "right_equality",
                left_eq_tv,
                ptr::null_mut(),
                left_cond_tv,
                right_cond_tv,
                predicate,
                "right_equality is null",
            ),
            (
                "left_conditional",
                left_eq_tv,
                right_eq_tv,
                ptr::null_mut(),
                right_cond_tv,
                predicate,
                "left_conditional is null",
            ),
            (
                "right_conditional",
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                ptr::null_mut(),
                predicate,
                "right_conditional is null",
            ),
            ("predicate", left_eq_tv, right_eq_tv, left_cond_tv, right_cond_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out_left = std::ptr::NonNull::<c_void>::dangling().as_ptr();
            let mut out_right = std::ptr::NonNull::<c_void>::dangling().as_ptr();
            let status = unsafe {
                join_indices(
                    left_eq,
                    right_eq,
                    left_cond,
                    right_cond,
                    pred,
                    null_equality,
                    &mut out_left,
                    &mut out_right,
                )
            };
            assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null {case} status");
            assert!(out_left.is_null(), "{name} mixed join must clear left output for null {case}");
            assert!(out_right.is_null(), "{name} mixed join must clear right output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    for (name, join_indices) in [
        ("semi", cudf_mixed_left_semi_join_indices_raw as MixedSingleJoinRaw),
        ("anti", cudf_mixed_left_anti_join_indices_raw),
    ] {
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                null_equality,
                ptr::null_mut(),
            )
        };
        assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null output");
        assert!(last_error_message().contains("out_indices is null"));

        for (case, left_eq, right_eq, left_cond, right_cond, pred, message) in [
            (
                "left_equality",
                ptr::null_mut(),
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                "left_equality is null",
            ),
            (
                "right_equality",
                left_eq_tv,
                ptr::null_mut(),
                left_cond_tv,
                right_cond_tv,
                predicate,
                "right_equality is null",
            ),
            (
                "left_conditional",
                left_eq_tv,
                right_eq_tv,
                ptr::null_mut(),
                right_cond_tv,
                predicate,
                "left_conditional is null",
            ),
            (
                "right_conditional",
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                ptr::null_mut(),
                predicate,
                "right_conditional is null",
            ),
            ("predicate", left_eq_tv, right_eq_tv, left_cond_tv, right_cond_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
            let status =
                unsafe { join_indices(left_eq, right_eq, left_cond, right_cond, pred, null_equality, &mut out) };
            assert_eq!(status, cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null {case} status");
            assert!(out.is_null(), "{name} mixed join must clear output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_cond_tv);
        cudf_sys::cudf_table_view_destroy(left_cond_tv);
        cudf_sys::cudf_table_view_destroy(right_eq_tv);
        cudf_sys::cudf_table_view_destroy(left_eq_tv);
        cudf_sys::cudf_table_destroy(right_cond_tbl);
        cudf_sys::cudf_table_destroy(left_cond_tbl);
        cudf_sys::cudf_table_destroy(right_eq_tbl);
        cudf_sys::cudf_table_destroy(left_eq_tbl);
    }
}

#[test]
fn test_mixed_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 1, 2, 3]));
    let right_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 4]));
    let left_cond_tbl = make_table_from_one_column(make_i32_column(&[10, 30, 5, 7]));
    let right_cond_tbl = make_table_from_one_column(make_i32_column(&[20, 10, 6, 8]));
    let left_eq_tv = make_table_view(left_eq_tbl);
    let right_eq_tv = make_table_view(right_eq_tbl);
    let left_cond_tv = make_table_view(left_cond_tbl);
    let right_cond_tv = make_table_view(right_cond_tbl);
    let predicate = make_left_less_than_right_predicate();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);

    for (name, join_indices) in [
        (
            "inner",
            cudf_sys::cudf_mixed_inner_join_indices_ex
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    cudf_sys::cudf_null_equality_t,
                    *mut *mut c_void,
                    *mut *mut c_void,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_mixed_left_join_indices_ex),
        ("full", cudf_sys::cudf_mixed_full_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        let current_before = current_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: *mut c_void = ptr::null_mut();
        let mut out_right: *mut c_void = ptr::null_mut();
        let current_borrowed = current_mr.borrowed();
        cudf_sys::test_set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
                &mut out_left,
                &mut out_right,
                exec.as_raw(),
            )
        };
        let restored_current = current_device_resource_raw();
        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} mixed join failed");
        assert!(!out_left.is_null(), "{name} mixed join did not publish left indices");
        assert!(!out_right.is_null(), "{name} mixed join did not publish right indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} mixed join must restore the caller's current memory resource"
        );
        assert_eq!(
            current_mr.allocation_count(),
            current_before,
            "{name} mixed join temporaries must not allocate from the caller's current memory resource"
        );
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} mixed join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out_left, &format!("{name} mixed left indices"));
        assert_readable_nonempty_i32_indices(out_right, &format!("{name} mixed right indices"));

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    for (name, join_indices) in [
        (
            "semi",
            cudf_sys::cudf_mixed_left_semi_join_indices_ex
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    cudf_sys::cudf_null_equality_t,
                    *mut *mut c_void,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("anti", cudf_sys::cudf_mixed_left_anti_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        let current_before = current_mr.allocation_count();
        reset_exec_observation();

        let mut out: *mut c_void = ptr::null_mut();
        let current_borrowed = current_mr.borrowed();
        cudf_sys::test_set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
                &mut out,
                exec.as_raw(),
            )
        };
        let restored_current = current_device_resource_raw();
        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} mixed join failed");
        assert!(!out.is_null(), "{name} mixed join did not publish indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} mixed join must restore the caller's current memory resource"
        );
        assert_eq!(
            current_mr.allocation_count(),
            current_before,
            "{name} mixed join temporaries must not allocate from the caller's current memory resource"
        );
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} mixed join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out, &format!("{name} mixed indices"));

        unsafe { cudf_sys::cudf_column_destroy(out) };
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_cond_tv);
        cudf_sys::cudf_table_view_destroy(left_cond_tv);
        cudf_sys::cudf_table_view_destroy(right_eq_tv);
        cudf_sys::cudf_table_view_destroy(left_eq_tv);
        cudf_sys::cudf_table_destroy(right_cond_tbl);
        cudf_sys::cudf_table_destroy(left_cond_tbl);
        cudf_sys::cudf_table_destroy(right_eq_tbl);
        cudf_sys::cudf_table_destroy(left_eq_tbl);
    }
}

#[test]
fn test_distinct_hash_join_probe_ex_uses_explicit_memory_resource_for_probe_allocations() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let build_col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_hash_join_create(
            build_tv,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            &mut join,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_one_column(make_i32_column(&[8, 7, 42, 1, 2, 99, 3, 4]));
    let probe_tv = make_table_view(probe_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);

    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();
    reset_exec_observation();

    let mut out: *mut c_void = ptr::null_mut();
    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let status = unsafe { cudf_sys::cudf_distinct_hash_join_left_join_ex(join, probe_tv, &mut out, exec.as_raw()) };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "distinct hash-join probe must restore the caller's current memory resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "distinct hash-join probe temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "distinct hash-join probe allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_distinct_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

#[test]
fn test_gather_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input_col = make_i32_column(&[10, 20, 30, 40]);
    let input_tbl = make_table_from_one_column(input_col);
    let input_view = make_table_view(input_tbl);
    let gather_map = make_i32_column(&[3, 1, 0, 2]);
    let gather_map_view = make_column_view(gather_map);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_gather_ex(
            input_view,
            gather_map_view,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            &mut out,
            exec.as_raw(),
        )
    };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "gather must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "gather temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "gather allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_column_view_destroy(gather_map_view);
        cudf_sys::cudf_column_destroy(gather_map);
        cudf_sys::cudf_table_view_destroy(input_view);
        cudf_sys::cudf_table_destroy(input_tbl);
    }
}

#[test]
fn test_cross_join_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 3, 4]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[10, 20, 30]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_cross_join_ex(left_tv, right_tv, &mut out, exec.as_raw()) };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "cross_join must restore the caller's current memory resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "cross_join temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "cross_join allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_scatter_null_scatter_map_returns_libcudf_error() {
    let _shim_state = lock_shim_state();
    let source_tbl = make_table_from_one_column(make_i32_column(&[10, 20]));
    let target_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 3, 4]));
    let source_tv = make_table_view(source_tbl);
    let target_tv = make_table_view(target_tbl);
    let scatter_map = make_i32_column_nullable(&[0, 1], &[true, false]);
    let scatter_map_view = make_column_view(scatter_map);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_scatter_raw(source_tv, scatter_map_view, target_tv, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(scatter_map_view);
        cudf_sys::cudf_column_destroy(scatter_map);
        cudf_sys::cudf_table_view_destroy(target_tv);
        cudf_sys::cudf_table_view_destroy(source_tv);
        cudf_sys::cudf_table_destroy(target_tbl);
        cudf_sys::cudf_table_destroy(source_tbl);
    }
}

#[test]
fn test_scatter_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let source_tbl = make_table_from_one_column(make_i32_column(&[10, 20, 30]));
    let target_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 3, 4, 5]));
    let source_tv = make_table_view(source_tbl);
    let target_tv = make_table_view(target_tbl);
    let scatter_map = make_i32_column(&[4, 2, 0]);
    let scatter_map_view = make_column_view(scatter_map);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_scatter_ex(source_tv, scatter_map_view, target_tv, &mut out, exec.as_raw()) };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "scatter must restore the caller's current memory resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "scatter temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "scatter allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_column_view_destroy(scatter_map_view);
        cudf_sys::cudf_column_destroy(scatter_map);
        cudf_sys::cudf_table_view_destroy(target_tv);
        cudf_sys::cudf_table_view_destroy(source_tv);
        cudf_sys::cudf_table_destroy(target_tbl);
        cudf_sys::cudf_table_destroy(source_tbl);
    }
}

#[test]
fn test_apply_boolean_mask_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input_col = make_i32_column(&[10, 20, 30, 40]);
    let input_tbl = make_table_from_one_column(input_col);
    let input_view = make_table_view(input_tbl);
    let mask = make_bool8_column(&[1, 0, 1, 0]);
    let mask_view = make_column_view(mask);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_apply_boolean_mask_ex(input_view, mask_view, &mut out, exec.as_raw()) };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "apply_boolean_mask must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "apply_boolean_mask temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "apply_boolean_mask allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_column_view_destroy(mask_view);
        cudf_sys::cudf_column_destroy(mask);
        cudf_sys::cudf_table_view_destroy(input_view);
        cudf_sys::cudf_table_destroy(input_tbl);
    }
}

#[test]
fn test_replace_nulls_policy_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input = make_i32_column_nullable(&[10, 20, 30, 40], &[true, false, true, false]);
    let input_view = make_column_view(input);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_replace_nulls_policy_ex(input_view, 0, &mut out, exec.as_raw()) };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "replace_nulls policy must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "replace_nulls policy temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "replace_nulls policy allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(input_view);
        cudf_sys::cudf_column_destroy(input);
    }
}

#[test]
fn test_distinct_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input_col = make_i32_column(&[1, 2, 1, 3, 2, 4, 4, 5]);
    let input_tbl = make_table_from_one_column(input_col);
    let input_view = make_table_view(input_tbl);
    let keys = [0_i32];
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_ex(
            input_view,
            keys.as_ptr(),
            keys.len() as i32,
            cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            cudf_sys::cudf_stream_compaction_null_equality_t::CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL,
            cudf_sys::cudf_stream_compaction_nan_equality_t::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL,
            &mut out,
            exec.as_raw(),
        )
    };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "distinct must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "distinct temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "distinct allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(input_view);
        cudf_sys::cudf_table_destroy(input_tbl);
    }
}

#[test]
fn test_dictionary_encode_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let col = make_i32_column(&[3, 1, 3, 2, 4, 2, 1, 5]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_dictionary_encode_ex(view, cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out, exec.as_raw())
    };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary encode must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "dictionary encode temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary encode allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_dictionary_decode_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let dict = make_dictionary_i32_column(&[10, 20, 10, 30, 20, 40]);
    let dict_view = make_column_view(dict);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_dictionary_decode_ex(dict_view, &mut out, exec.as_raw()) };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary decode must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "dictionary decode temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary decode allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(dict_view);
        cudf_sys::cudf_column_destroy(dict);
    }
}

#[test]
fn test_dictionary_set_keys_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let dict = make_dictionary_i32_column(&[1, 2, 1, 3, 2, 4]);
    let dict_view = make_column_view(dict);
    let keys = make_i32_column(&[1, 2, 3, 4, 5]);
    let keys_view = make_column_view(keys);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_dictionary_set_keys_ex(dict_view, keys_view, &mut out, exec.as_raw()) };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary set_keys must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "dictionary set_keys temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary set_keys allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(keys_view);
        cudf_sys::cudf_column_destroy(keys);
        cudf_sys::cudf_column_view_destroy(dict_view);
        cudf_sys::cudf_column_destroy(dict);
    }
}

#[test]
fn test_dictionary_match_dictionaries_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left = make_dictionary_i32_column(&[10, 20, 10, 30]);
    let right = make_dictionary_i32_column(&[20, 30, 40, 20]);
    let left_view = make_column_view(left);
    let right_view = make_column_view(right);
    let dict_columns = [left_view, right_view];
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries_ex(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
            exec.as_raw(),
        )
    };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, dict_columns.len() as i32);
    assert!(!out_columns.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary match_dictionaries must restore the caller's current device resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "dictionary match_dictionaries temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary match_dictionaries allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    let matched = unsafe { std::slice::from_raw_parts(out_columns, out_count as usize) };
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

#[test]
fn test_join_indices_publish_no_outputs_when_second_handle_release_fails() {
    let _shim_state = lock_shim_state();
    let left_tbl = make_table_from_columns(vec![make_i32_column(&[1, 2, 3])]);
    let right_tbl = make_table_from_columns(vec![make_i32_column(&[2, 3, 4])]);
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);

    let mut left_indices = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let mut right_indices = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    unsafe { cudf_sys::cudf_test_fail_column_handle_release_after(1) };

    let status = unsafe {
        cudf_sys::cudf_inner_join_indices(
            left_tv,
            right_tv,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            &mut left_indices,
            &mut right_indices,
        )
    };
    unsafe { cudf_sys::cudf_test_fail_column_handle_release_after(-1) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(left_indices.is_null());
    assert!(right_indices.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_destroy(left_tbl);
        cudf_sys::cudf_table_destroy(right_tbl);
    }
}

#[test]
fn test_distinct_hash_join_left_join_returns_build_indices() {
    let _shim_state = lock_shim_state();
    let build_col = make_i32_column(&[10, 20, 30]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_hash_join_create(
            build_tv,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            &mut join,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_columns(vec![make_i32_column(&[20, 99, 10, 30])]);
    let probe_tv = make_table_view(probe_tbl);
    let mut out: *mut c_void = ptr::null_mut();

    let status = unsafe { cudf_distinct_hash_join_left_join_raw(join, probe_tv, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(read_i32_column_data(out, 4), vec![1, i32::MIN, 0, 2]);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_distinct_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

#[test]
fn test_scan_invalid_scan_type_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);
    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scan_aggregation_create(cudf_sys::cudf_reduce_aggregation_kind_t::CUDF_REDUCE_AGG_SUM, &mut agg)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_scan_raw(view, agg, 99, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid scan_type"));

    unsafe {
        cudf_sys::cudf_scan_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scan_invalid_null_policy_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);
    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scan_aggregation_create(cudf_sys::cudf_reduce_aggregation_kind_t::CUDF_REDUCE_AGG_SUM, &mut agg)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());

    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_scan(view, agg, cudf_sys::cudf_scan_type_t::CUDF_SCAN_INCLUSIVE, 99, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid null_policy"));

    unsafe {
        cudf_sys::cudf_scan_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_unary_operation_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_unary_operation_raw(view, 99, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid unary operator"));

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_binary_operation_vv_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let lhs_col = make_i32_column(&[1, 2, 3]);
    let rhs_col = make_i32_column(&[4, 5, 6]);
    let lhs_view = make_column_view(lhs_col);
    let rhs_view = make_column_view(rhs_col);
    let dtype = make_i32_data_type();

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_binary_operation_vv_raw(lhs_view, rhs_view, 99, dtype, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid binary operator"));

    unsafe {
        cudf_sys::cudf_data_type_destroy(dtype);
        cudf_sys::cudf_column_view_destroy(rhs_view);
        cudf_sys::cudf_column_view_destroy(lhs_view);
        cudf_sys::cudf_column_destroy(rhs_col);
        cudf_sys::cudf_column_destroy(lhs_col);
    }
}

#[test]
fn test_binary_operation_vs_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let lhs_col = make_i32_column(&[1, 2, 3]);
    let lhs_view = make_column_view(lhs_col);
    let rhs_scalar = make_i32_scalar(4);
    let dtype = make_i32_data_type();

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_binary_operation_vs_raw(lhs_view, rhs_scalar, 99, dtype, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid binary operator"));

    unsafe {
        cudf_sys::cudf_data_type_destroy(dtype);
        cudf_sys::cudf_scalar_destroy(rhs_scalar);
        cudf_sys::cudf_column_view_destroy(lhs_view);
        cudf_sys::cudf_column_destroy(lhs_col);
    }
}

#[test]
fn test_binary_operation_sv_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let lhs_scalar = make_i32_scalar(4);
    let rhs_col = make_i32_column(&[1, 2, 3]);
    let rhs_view = make_column_view(rhs_col);
    let dtype = make_i32_data_type();

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_binary_operation_sv_raw(lhs_scalar, rhs_view, 99, dtype, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid binary operator"));

    unsafe {
        cudf_sys::cudf_data_type_destroy(dtype);
        cudf_sys::cudf_column_view_destroy(rhs_view);
        cudf_sys::cudf_column_destroy(rhs_col);
        cudf_sys::cudf_scalar_destroy(lhs_scalar);
    }
}
