use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::*,
    support::{
        NO_SYNC_RECORDED, OwnedStream, last_error_message, last_exec_mr_raw, last_exec_stream_value,
        last_exec_sync_stream_value, lock_shim_state, make_stats_memory_resource, owned_string_value,
        required_exec_options, reset_exec_observation, stream_value,
    },
};

fn create_device_buffer(size: usize, exec: &cudf_sys::cudf_exec_options_t) -> cudf_sys::cudf_device_buffer_handle {
    let mut buffer: cudf_sys::cudf_device_buffer_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_device_buffer_create_ex(size, exec, &mut buffer) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!buffer.is_null());
    buffer
}

fn device_buffer_exec_owners() -> (OwnedStream, crate::support::OwnedMemoryResource, cudf_sys::cudf_exec_options_t) {
    strict_exec_owners()
}

fn strict_exec_owners() -> (OwnedStream, crate::support::OwnedMemoryResource, cudf_sys::cudf_exec_options_t) {
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    (stream, mr, exec)
}

#[test]
fn test_scalar_string_value_round_trips_empty_and_non_empty_values() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    for expected in ["hello", ""] {
        let scalar = make_string_scalar(expected);
        assert_eq!(scalar_string_value(scalar, &exec), expected);

        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_nullable_device_buffer_adoption_rejects_same_handle_for_data_and_mask() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = device_buffer_exec_owners();
    let buffer = create_device_buffer(64, &exec);

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_adopt_device_buffer_nullable(cudf_sys::CUDF_TYPE_INT8, buffer, buffer, 0, 64, &mut out)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("distinct handles"));

    unsafe { cudf_sys::cudf_device_buffer_destroy(buffer) };
}

#[test]
fn test_device_buffer_adoption_publish_failure_leaves_buffer_handle_valid() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = device_buffer_exec_owners();
    let values = [10i32, 20, 30];
    let buffer = create_device_buffer(std::mem::size_of_val(&values), &exec);
    let st = unsafe {
        cudf_sys::cudf_device_buffer_copy_from_host_ex(
            buffer,
            values.as_ptr() as *const c_void,
            std::mem::size_of_val(&values),
            &exec,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(0) };
    let st = unsafe {
        cudf_sys::cudf_column_adopt_device_buffer(cudf_sys::CUDF_TYPE_INT32, buffer, values.len() as i32, &mut out)
    };
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(-1) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out.is_null());

    let mut size = 0usize;
    let st = unsafe { cudf_sys::cudf_device_buffer_size(buffer, &mut size) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(size, std::mem::size_of_val(&values));

    let st = unsafe {
        cudf_sys::cudf_column_adopt_device_buffer(cudf_sys::CUDF_TYPE_INT32, buffer, values.len() as i32, &mut out)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(read_i32_column_data(out, values.len()), values);

    unsafe { cudf_sys::cudf_column_destroy(out) };
}

#[test]
fn test_nullable_device_buffer_adoption_publish_failure_leaves_both_handles_valid() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = device_buffer_exec_owners();
    let values = [10i32, 20, 30];
    let data_buffer = create_device_buffer(std::mem::size_of_val(&values), &exec);
    let mask_buffer = create_device_buffer(64, &exec);
    let st = unsafe {
        cudf_sys::cudf_device_buffer_copy_from_host_ex(
            data_buffer,
            values.as_ptr() as *const c_void,
            std::mem::size_of_val(&values),
            &exec,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(0) };
    let st = unsafe {
        cudf_sys::cudf_column_adopt_device_buffer_nullable(
            cudf_sys::CUDF_TYPE_INT32,
            data_buffer,
            mask_buffer,
            0,
            values.len() as i32,
            &mut out,
        )
    };
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(-1) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out.is_null());

    let mut data_size = 0usize;
    let mut mask_size = 0usize;
    let st = unsafe { cudf_sys::cudf_device_buffer_size(data_buffer, &mut data_size) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let st = unsafe { cudf_sys::cudf_device_buffer_size(mask_buffer, &mut mask_size) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(data_size, std::mem::size_of_val(&values));
    assert_eq!(mask_size, 64);

    let st = unsafe {
        cudf_sys::cudf_column_adopt_device_buffer_nullable(
            cudf_sys::CUDF_TYPE_INT32,
            data_buffer,
            mask_buffer,
            0,
            values.len() as i32,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(read_i32_column_data(out, values.len()), values);

    unsafe { cudf_sys::cudf_column_destroy(out) };
}

#[test]
fn test_column_view_null_mask_to_host_ex_supports_sliced_nullable_views() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column_nullable(
        &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        &[true, false, true, true, false, true, false, true, true, false],
    );
    let sliced = make_sliced_column_view(col, 3, 6);
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut mask = vec![0xAAu8; 64];
    let st = unsafe { cudf_sys::cudf_column_view_null_mask_to_host_ex(sliced, mask.as_mut_ptr(), mask.len(), &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS, "sliced nullable null-mask export should succeed");

    assert_eq!(mask[0], 0b0011_0101, "mask bits should be normalized to the logical slice starting at bit 0");
    assert!(mask[1..].iter().all(|&byte| byte == 0), "no extra bits outside the logical slice should be set");

    unsafe {
        cudf_sys::cudf_column_view_destroy(sliced);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scalar_string_value_owned_output_survives_subsequent_shim_calls() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let scalar = make_string_scalar("hello");

    let mut ptr1: *mut c_char = ptr::null_mut();
    let mut len1: i32 = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut ptr1, &mut len1, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut type_id = cudf_sys::CUDF_TYPE_EMPTY;
    let st = unsafe { cudf_sys::cudf_scalar_type(scalar, &mut type_id) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, cudf_sys::CUDF_TYPE_STRING);

    let mut ptr2: *mut c_char = ptr::null_mut();
    let mut len2: i32 = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut ptr2, &mut len2, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    assert_eq!(owned_string_value(ptr1, len1), "hello");
    assert_eq!(owned_string_value(ptr2, len2), "hello");

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_string_value_null_handle_initializes_outputs_before_failure() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut len: i32 = 123;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(ptr::null_mut(), &mut out, &mut len, &exec) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
    assert_eq!(len, 0);
}

#[test]
fn test_scalar_string_value_wrong_type_initializes_outputs_before_failure() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let scalar = make_i32_scalar(42);

    let mut out = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut len: i32 = 123;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut out, &mut len, &exec) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(out.is_null());
    assert_eq!(len, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_numeric_invalid_allows_null_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric_ex(cudf_sys::CUDF_TYPE_INT32, ptr::null(), 0, &exec, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());

    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid_ex(scalar, &mut is_valid, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_numeric_valid_requires_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric_ex(cudf_sys::CUDF_TYPE_INT32, ptr::null(), 1, &exec, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

fn assert_scalar_data_type(
    scalar: cudf_sys::cudf_scalar_handle,
    expected_type: cudf_sys::cudf_type_id_t,
    expected_scale: i32,
) {
    let mut type_id = cudf_sys::CUDF_TYPE_EMPTY;
    let mut scale = i32::MIN;
    let st = unsafe { cudf_sys::cudf_scalar_data_type(scalar, &mut type_id, &mut scale) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, expected_type);
    assert_eq!(scale, expected_scale);
}

fn assert_scalar_validity(scalar: cudf_sys::cudf_scalar_handle, exec: &cudf_sys::cudf_exec_options_t, expected: i32) {
    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid_ex(scalar, &mut is_valid, exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, expected);
}

unsafe fn scalar_raw_value<T: Copy>(scalar: cudf_sys::cudf_scalar_handle, exec: &cudf_sys::cudf_exec_options_t) -> T {
    let mut out = std::mem::MaybeUninit::<T>::uninit();
    let st = unsafe { cudf_sys::cudf_scalar_value_ex(scalar, out.as_mut_ptr() as *mut c_void, exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    unsafe { out.assume_init() }
}

fn scalar_string_value(scalar: cudf_sys::cudf_scalar_handle, exec: &cudf_sys::cudf_exec_options_t) -> String {
    let mut out: *mut c_char = ptr::null_mut();
    let mut len: i32 = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut out, &mut len, exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    owned_string_value(out, len)
}

#[test]
fn test_scalar_create_decimal_raw_round_trips_type_scale_validity_and_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let scale = -2;

    let raw32 = -12345_i32;
    let mut decimal32: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::CUDF_TYPE_DECIMAL32,
            &raw32 as *const i32 as *const c_void,
            scale,
            1,
            &exec,
            &mut decimal32,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_scalar_data_type(decimal32, cudf_sys::CUDF_TYPE_DECIMAL32, scale);
    assert_scalar_validity(decimal32, &exec, 1);
    assert_eq!(unsafe { scalar_raw_value::<i32>(decimal32, &exec) }, raw32);

    let raw64 = -9_876_543_210_i64;
    let mut decimal64: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::CUDF_TYPE_DECIMAL64,
            &raw64 as *const i64 as *const c_void,
            scale,
            1,
            &exec,
            &mut decimal64,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_scalar_data_type(decimal64, cudf_sys::CUDF_TYPE_DECIMAL64, scale);
    assert_scalar_validity(decimal64, &exec, 1);
    assert_eq!(unsafe { scalar_raw_value::<i64>(decimal64, &exec) }, raw64);

    let raw128 = -123_456_789_012_345_678_901_234_567_i128;
    let mut decimal128: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::CUDF_TYPE_DECIMAL128,
            &raw128 as *const i128 as *const c_void,
            scale,
            1,
            &exec,
            &mut decimal128,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_scalar_data_type(decimal128, cudf_sys::CUDF_TYPE_DECIMAL128, scale);
    assert_scalar_validity(decimal128, &exec, 1);
    assert_eq!(unsafe { scalar_raw_value::<i128>(decimal128, &exec) }, raw128);

    unsafe {
        cudf_sys::cudf_scalar_destroy(decimal32);
        cudf_sys::cudf_scalar_destroy(decimal64);
        cudf_sys::cudf_scalar_destroy(decimal128);
    }
}

#[test]
fn test_scalar_create_timestamp_raw_round_trips_type_validity_and_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let cases = [
        (cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS, 19_118_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS, 1_651_795_200_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_MILLISECONDS, 1_651_795_200_123_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_MICROSECONDS, 1_651_795_200_123_456_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_NANOSECONDS, 1_651_795_200_123_456_789_i64),
    ];

    for (type_id, raw) in cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_scalar_create_timestamp_raw_ex(type_id, raw, 1, &exec, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, &exec, 1);
        if type_id == cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar, &exec) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar, &exec) }, raw);
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_duration_raw_round_trips_type_validity_and_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let cases = [
        (cudf_sys::CUDF_TYPE_DURATION_DAYS, -7_i64),
        (cudf_sys::CUDF_TYPE_DURATION_SECONDS, -123_i64),
        (cudf_sys::CUDF_TYPE_DURATION_MILLISECONDS, -123_456_i64),
        (cudf_sys::CUDF_TYPE_DURATION_MICROSECONDS, -123_456_789_i64),
        (cudf_sys::CUDF_TYPE_DURATION_NANOSECONDS, -123_456_789_012_i64),
    ];

    for (type_id, raw) in cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_scalar_create_duration_raw_ex(type_id, raw, 1, &exec, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, &exec, 1);
        if type_id == cudf_sys::CUDF_TYPE_DURATION_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar, &exec) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar, &exec) }, raw);
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_raw_ex_constructors_round_trip_with_explicit_exec() {
    let _shim_state = lock_shim_state();
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let mr = make_stats_memory_resource();
    let explicit_mr = mr.raw_device_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    let raw_decimal32 = -12_345_i32;
    let raw_decimal64 = -9_876_543_i64;
    let raw_decimal128 = -123_456_789_012_345_678_901_234_567_i128;
    let decimal_cases = [
        (cudf_sys::CUDF_TYPE_DECIMAL32, &raw_decimal32 as *const i32 as *const c_void, -2),
        (cudf_sys::CUDF_TYPE_DECIMAL64, &raw_decimal64 as *const i64 as *const c_void, -4),
        (cudf_sys::CUDF_TYPE_DECIMAL128, &raw_decimal128 as *const i128 as *const c_void, -6),
    ];
    for (type_id, raw_ptr, scale) in decimal_cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe { cudf_sys::cudf_scalar_create_decimal_raw_ex(type_id, raw_ptr, scale, 1, &exec, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), explicit_stream);
        assert_eq!(last_exec_mr_raw(), explicit_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
        assert_scalar_data_type(scalar, type_id, scale);
        assert_scalar_validity(scalar, &exec, 1);
        match type_id {
            cudf_sys::CUDF_TYPE_DECIMAL32 => {
                assert_eq!(unsafe { scalar_raw_value::<i32>(scalar, &exec) }, raw_decimal32)
            },
            cudf_sys::CUDF_TYPE_DECIMAL64 => {
                assert_eq!(unsafe { scalar_raw_value::<i64>(scalar, &exec) }, raw_decimal64)
            },
            cudf_sys::CUDF_TYPE_DECIMAL128 => {
                assert_eq!(unsafe { scalar_raw_value::<i128>(scalar, &exec) }, raw_decimal128)
            },
            _ => unreachable!(),
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }

    let timestamp_cases = [
        (cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS, -12_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS, 1_700_000_000_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_MILLISECONDS, 1_700_000_000_123_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_MICROSECONDS, 1_700_000_000_123_456_i64),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_NANOSECONDS, 1_700_000_000_123_456_789_i64),
    ];
    for (type_id, raw) in timestamp_cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe { cudf_sys::cudf_scalar_create_timestamp_raw_ex(type_id, raw, 1, &exec, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), explicit_stream);
        assert_eq!(last_exec_mr_raw(), explicit_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, &exec, 1);
        if type_id == cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar, &exec) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar, &exec) }, raw);
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }

    let duration_cases = [
        (cudf_sys::CUDF_TYPE_DURATION_DAYS, -7_i64),
        (cudf_sys::CUDF_TYPE_DURATION_SECONDS, -42_i64),
        (cudf_sys::CUDF_TYPE_DURATION_MILLISECONDS, -42_000_i64),
        (cudf_sys::CUDF_TYPE_DURATION_MICROSECONDS, -42_000_000_i64),
        (cudf_sys::CUDF_TYPE_DURATION_NANOSECONDS, -42_000_000_000_i64),
    ];
    for (type_id, raw) in duration_cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe { cudf_sys::cudf_scalar_create_duration_raw_ex(type_id, raw, 1, &exec, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), explicit_stream);
        assert_eq!(last_exec_mr_raw(), explicit_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, &exec, 1);
        if type_id == cudf_sys::CUDF_TYPE_DURATION_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar, &exec) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar, &exec) }, raw);
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_raw_ex_null_and_default_options_are_rejected() {
    let _shim_state = lock_shim_state();
    let default_exec = cudf_sys::ExecutionContext::new();

    for exec_options in [ptr::null(), default_exec.as_raw()] {
        let raw_decimal = 123_i64;
        let mut decimal: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe {
            cudf_sys::cudf_scalar_create_decimal_raw_ex(
                cudf_sys::CUDF_TYPE_DECIMAL64,
                &raw_decimal as *const i64 as *const c_void,
                -2,
                1,
                exec_options,
                &mut decimal,
            )
        };
        assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
        assert!(decimal.is_null());

        let mut timestamp: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        let st = unsafe {
            cudf_sys::cudf_scalar_create_timestamp_raw_ex(
                cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS,
                1_700_000_000,
                1,
                exec_options,
                &mut timestamp,
            )
        };
        assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
        assert!(timestamp.is_null());

        let mut duration: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        let st = unsafe {
            cudf_sys::cudf_scalar_create_duration_raw_ex(
                cudf_sys::CUDF_TYPE_DURATION_SECONDS,
                -42,
                1,
                exec_options,
                &mut duration,
            )
        };
        assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
        assert!(duration.is_null());
    }
}

#[test]
fn test_scalar_create_raw_ex_propagates_exec_option_validation_errors() {
    let _shim_state = lock_shim_state();
    let bad_exec = cudf_sys::cudf_exec_options_t { struct_size: 0, ..Default::default() };
    let raw_decimal = 123_i64;

    let mut decimal: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::CUDF_TYPE_DECIMAL64,
            &raw_decimal as *const i64 as *const c_void,
            -2,
            1,
            &bad_exec,
            &mut decimal,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(decimal.is_null());
    assert!(last_error_message().contains("struct_size"));

    let mut timestamp: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_timestamp_raw_ex(
            cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS,
            1_700_000_000,
            1,
            &bad_exec,
            &mut timestamp,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(timestamp.is_null());
    assert!(last_error_message().contains("struct_size"));

    let mut duration: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_duration_raw_ex(
            cudf_sys::CUDF_TYPE_DURATION_SECONDS,
            -42,
            1,
            &bad_exec,
            &mut duration,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(duration.is_null());
    assert!(last_error_message().contains("struct_size"));
}

#[test]
fn test_scalar_create_raw_rejects_wrong_family_type_ids() {
    let _shim_state = lock_shim_state();
    let raw_decimal = 1_i64;
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS,
            &raw_decimal as *const i64 as *const c_void,
            -2,
            1,
            &exec,
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(scalar.is_null());
    assert!(last_error_message().contains("Unsupported type for decimal scalar"));

    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_timestamp_raw_ex(cudf_sys::CUDF_TYPE_DURATION_SECONDS, 1, 1, &exec, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(scalar.is_null());
    assert!(last_error_message().contains("Unsupported type for timestamp scalar"));

    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_duration_raw_ex(cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS, 1, 1, &exec, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(scalar.is_null());
    assert!(last_error_message().contains("Unsupported type for duration scalar"));
}

#[test]
fn test_scalar_create_temporal_days_rejects_i64_values_outside_i32_range() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let cases = [
        (cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS, i64::from(i32::MAX) + 1, true),
        (cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS, i64::from(i32::MIN) - 1, true),
        (cudf_sys::CUDF_TYPE_DURATION_DAYS, i64::from(i32::MAX) + 1, false),
        (cudf_sys::CUDF_TYPE_DURATION_DAYS, i64::from(i32::MIN) - 1, false),
    ];

    for (type_id, raw, is_timestamp) in cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        let st = if is_timestamp {
            unsafe { cudf_sys::cudf_scalar_create_timestamp_raw_ex(type_id, raw, 1, &exec, &mut scalar) }
        } else {
            unsafe { cudf_sys::cudf_scalar_create_duration_raw_ex(type_id, raw, 1, &exec, &mut scalar) }
        };
        assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
        assert!(scalar.is_null());
        assert!(last_error_message().contains("outside the requested type range"));
    }
}

#[test]
fn test_scalar_create_raw_decimal_invalid_allows_null_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let cases = [cudf_sys::CUDF_TYPE_DECIMAL32, cudf_sys::CUDF_TYPE_DECIMAL64, cudf_sys::CUDF_TYPE_DECIMAL128];

    for type_id in cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        let st =
            unsafe { cudf_sys::cudf_scalar_create_decimal_raw_ex(type_id, ptr::null(), -3, 0, &exec, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, -3);
        assert_scalar_validity(scalar, &exec, 0);
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_raw_temporal_invalid_preserves_dtype() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let cases = [
        cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS,
        cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS,
        cudf_sys::CUDF_TYPE_TIMESTAMP_MILLISECONDS,
        cudf_sys::CUDF_TYPE_TIMESTAMP_MICROSECONDS,
        cudf_sys::CUDF_TYPE_TIMESTAMP_NANOSECONDS,
        cudf_sys::CUDF_TYPE_DURATION_DAYS,
        cudf_sys::CUDF_TYPE_DURATION_SECONDS,
        cudf_sys::CUDF_TYPE_DURATION_MILLISECONDS,
        cudf_sys::CUDF_TYPE_DURATION_MICROSECONDS,
        cudf_sys::CUDF_TYPE_DURATION_NANOSECONDS,
    ];

    for type_id in cases {
        let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
        let st = if matches!(
            type_id,
            cudf_sys::CUDF_TYPE_TIMESTAMP_DAYS
                | cudf_sys::CUDF_TYPE_TIMESTAMP_SECONDS
                | cudf_sys::CUDF_TYPE_TIMESTAMP_MILLISECONDS
                | cudf_sys::CUDF_TYPE_TIMESTAMP_MICROSECONDS
                | cudf_sys::CUDF_TYPE_TIMESTAMP_NANOSECONDS
        ) {
            unsafe { cudf_sys::cudf_scalar_create_timestamp_raw_ex(type_id, 0, 0, &exec, &mut scalar) }
        } else {
            unsafe { cudf_sys::cudf_scalar_create_duration_raw_ex(type_id, 0, 0, &exec, &mut scalar) }
        };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, &exec, 0);
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_raw_rejects_null_out_and_valid_decimal_null_value() {
    let _shim_state = lock_shim_state();
    let raw = 123_i32;
    let (_stream, _mr, exec) = strict_exec_owners();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::CUDF_TYPE_DECIMAL32,
            &raw as *const i32 as *const c_void,
            -2,
            1,
            &exec,
            ptr::null_mut(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);

    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::CUDF_TYPE_DECIMAL32,
            ptr::null(),
            -2,
            1,
            &exec,
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

#[test]
fn test_scalar_create_string_valid_requires_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string_ex(ptr::null(), 1, &exec, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

#[test]
fn test_scalar_create_string_invalid_allows_null_value() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut scalar: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string_ex(ptr::null(), 0, &exec, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());

    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid_ex(scalar, &mut is_valid, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_view_slice_nested_nullable_preserves_logical_offset_in_null_count() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column_nullable(&[10, 20, 30, 40, 50], &[true, false, true, false, true]);
    let base_view = make_column_view(col);
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut first_slice: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice_ex(base_view, 1, 3, &exec, &mut first_slice) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!first_slice.is_null());

    let mut first_offset = -1i32;
    let mut first_null_count = -1i32;
    let st = unsafe { cudf_sys::cudf_column_view_offset(first_slice, &mut first_offset) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let st = unsafe { cudf_sys::cudf_column_view_null_count(first_slice, &mut first_null_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(first_offset, 1);
    assert_eq!(first_null_count, 2);

    let mut second_slice: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice_ex(first_slice, 1, 1, &exec, &mut second_slice) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!second_slice.is_null());

    let mut second_offset = -1i32;
    let mut second_null_count = -1i32;
    let st = unsafe { cudf_sys::cudf_column_view_offset(second_slice, &mut second_offset) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let st = unsafe { cudf_sys::cudf_column_view_null_count(second_slice, &mut second_null_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(second_offset, 2);
    assert_eq!(second_null_count, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(second_slice);
        cudf_sys::cudf_column_view_destroy(first_slice);
        cudf_sys::cudf_column_view_destroy(base_view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_slice_rejects_active_offset_overflow() {
    let _shim_state = lock_shim_state();
    let mut view: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::test_support::cudf_test_column_view_create_synthetic_fixed_width(
            cudf_sys::CUDF_TYPE_INT32,
            2,
            i32::MAX - 1,
            &mut view,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!view.is_null());

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut slice: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice_ex(view, 1, 1, &exec, &mut slice) };

    if !slice.is_null() {
        unsafe { cudf_sys::cudf_column_view_destroy(slice) };
    }
    unsafe { cudf_sys::cudf_column_view_destroy(view) };

    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(slice.is_null());
    assert!(crate::support::last_error_message().contains("slice range exceeds active column offset range"));
}

#[test]
fn test_column_view_strings_to_host_ex_rejects_string_view_without_offsets_child() {
    let _shim_state = lock_shim_state();
    let mut view: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let st =
        unsafe { cudf_sys::test_support::cudf_test_column_view_create_synthetic_string_without_children(1, &mut view) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!view.is_null());

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut offsets = vec![-1_i32; 2];
    let mut total_chars = -1_i32;
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut total_chars,
            &exec,
        )
    };

    unsafe { cudf_sys::cudf_column_view_destroy(view) };

    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert_eq!(offsets, vec![-1, -1]);
    assert_eq!(total_chars, 0);
    assert!(last_error_message().contains("string column view must expose an offsets child"));
}

#[test]
fn test_column_view_string_chars_ptr_and_size_ex_reports_active_sliced_chars_span() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["zero", "one", "two", "three"]);
    let full_view = make_column_view(col);
    let sliced_view = make_sliced_column_view(col, 1, 2);
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut full_ptr: *const c_void = ptr::null();
    let mut full_size: usize = 0;
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(full_view, &exec, &mut full_ptr, &mut full_size)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!full_ptr.is_null());
    assert_eq!(full_size, "zeroonetwothree".len());

    let mut sliced_ptr: *const c_void = ptr::null();
    let mut sliced_size: usize = 0;
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(sliced_view, &exec, &mut sliced_ptr, &mut sliced_size)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!sliced_ptr.is_null());
    assert_eq!(sliced_ptr, full_ptr);
    // The pointer is the chars BASE and the slice's offsets index from that
    // base, so the span must cover everything the slice references:
    // "zero" + "one" + "two"; ending at the last byte of "two".
    assert_eq!(sliced_size, "zeroonetwo".len());

    unsafe {
        cudf_sys::cudf_column_view_destroy(sliced_view);
        cudf_sys::cudf_column_view_destroy(full_view);
        cudf_sys::cudf_column_destroy(col);
    }
}
