use std::{os::raw::c_void, ptr};

use crate::{
    helpers::make_column_view,
    support::{
        NO_SYNC_RECORDED, OwnedStream, current_device_resource_raw, last_error_message, last_exec_mr_raw,
        last_exec_stream_value, last_exec_sync_stream_value, lock_shim_state, make_stats_memory_resource,
        reset_exec_observation, stream_value,
    },
};

#[link(name = "cudf_rust")]
extern "C" {
    #[link_name = "cudf_datetime_extract_component"]
    fn cudf_datetime_extract_component_raw(input: *mut c_void, component: i32, out: *mut *mut c_void) -> i32;
}

fn make_timestamp_days_column(values: &[i32]) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS,
            0,
            values.as_ptr() as *const c_void,
            values.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn read_i16_column_data(col: *mut c_void, expected_len: usize) -> Vec<i16> {
    let view = make_column_view(col);
    let mut buf = vec![0i16; expected_len];
    let buf_size = expected_len * std::mem::size_of::<i16>();
    let status = unsafe { cudf_sys::cudf_column_view_to_host(view, buf.as_mut_ptr() as *mut c_void, buf_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    buf
}

fn assert_column_type(col: *mut c_void, expected_type: cudf_sys::cudf_type_id_t) {
    let view = make_column_view(col);
    let mut type_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
    let mut scale = -1;
    let status = unsafe { cudf_sys::cudf_column_view_data_type(view, &mut type_id, &mut scale) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, expected_type);
    assert_eq!(scale, 0);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
}

#[test]
fn test_datetime_extract_component_returns_int16_values() {
    let input = make_timestamp_days_column(&[0, 10_561, 18_321]);
    let view = make_column_view(input);
    let mut out: *mut c_void = ptr::null_mut();

    let status = unsafe {
        cudf_sys::cudf_datetime_extract_component(
            view,
            cudf_sys::cudf_datetime_component_t::CUDF_DATETIME_COMPONENT_YEAR,
            &mut out,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_column_type(out, cudf_sys::cudf_type_id_t::CUDF_TYPE_INT16);
    assert_eq!(read_i16_column_data(out, 3), vec![1970, 1998, 2020]);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(input);
    }
}

#[test]
fn test_datetime_extract_component_ex_uses_explicit_stream_and_mr() {
    let _shim_state = lock_shim_state();
    let input = make_timestamp_days_column(&[0, 10_561]);
    let view = make_column_view(input);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();

    reset_exec_observation();
    let status = unsafe {
        cudf_sys::cudf_datetime_extract_component_ex(
            view,
            cudf_sys::cudf_datetime_component_t::CUDF_DATETIME_COMPONENT_MONTH,
            &mut out,
            exec.as_raw(),
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_ne!(last_exec_mr_raw(), ptr::null_mut());
    assert_ne!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i16_column_data(out, 2), vec![1, 12]);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(input);
    }
}

#[test]
fn test_datetime_extract_component_validates_null_handles_and_invalid_component() {
    let input = make_timestamp_days_column(&[0]);
    let view = make_column_view(input);
    let mut out: *mut c_void = ptr::null_mut();

    let status = unsafe {
        cudf_sys::cudf_datetime_extract_component(
            ptr::null_mut(),
            cudf_sys::cudf_datetime_component_t::CUDF_DATETIME_COMPONENT_YEAR,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
    assert!(last_error_message().contains("input is null"));

    let status = unsafe { cudf_datetime_extract_component_raw(view, 99, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid datetime component"));

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(input);
    }
}
