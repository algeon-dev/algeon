use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::*,
    support::{
        NO_SYNC_RECORDED, OwnedStream, current_device_resource_raw, default_stream_value, last_error_message,
        last_exec_mr_raw, last_exec_stream_value, last_exec_sync_stream_value, lock_shim_state,
        make_stats_memory_resource, owned_string_value, reset_exec_observation, stream_value,
    },
};

#[test]
fn test_scalar_string_value_round_trips_empty_and_non_empty_values() {
    for expected in ["hello", ""] {
        let s = std::ffi::CString::new(expected).unwrap();
        let mut scalar: *mut c_void = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_scalar_create_string(s.as_ptr(), 1, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);

        let mut out: *mut c_char = ptr::null_mut();
        let mut len: i32 = -1;
        let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut out, &mut len) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(len as usize, expected.len());
        assert_eq!(owned_string_value(out, len), expected);

        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_column_view_null_mask_to_host_ex_supports_sliced_nullable_views() {
    let col = make_i32_column_nullable(
        &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        &[true, false, true, true, false, true, false, true, true, false],
    );
    let sliced = make_sliced_column_view(col, 3, 6);

    let mut mask = vec![0xAAu8; 64];
    let st =
        unsafe { cudf_sys::cudf_column_view_null_mask_to_host_ex(sliced, mask.as_mut_ptr(), mask.len(), ptr::null()) };
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
    let s = std::ffi::CString::new("hello").unwrap();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(s.as_ptr(), 1, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut ptr1: *mut c_char = ptr::null_mut();
    let mut len1: i32 = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut ptr1, &mut len1) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut type_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
    let st = unsafe { cudf_sys::cudf_scalar_type(scalar, &mut type_id) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, cudf_sys::cudf_type_id_t::CUDF_TYPE_STRING);

    let mut ptr2: *mut c_char = ptr::null_mut();
    let mut len2: i32 = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut ptr2, &mut len2) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    assert_eq!(owned_string_value(ptr1, len1), "hello");
    assert_eq!(owned_string_value(ptr2, len2), "hello");

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_string_value_null_handle_initializes_outputs_before_failure() {
    let mut out = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut len: i32 = 123;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(ptr::null_mut(), &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
    assert_eq!(len, 0);
}

#[test]
fn test_scalar_string_value_wrong_type_initializes_outputs_before_failure() {
    let scalar = make_i32_scalar(42);

    let mut out = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut len: i32 = 123;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(out.is_null());
    assert_eq!(len, 0);

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

fn assert_scalar_data_type(scalar: *mut c_void, expected_type: cudf_sys::cudf_type_id_t, expected_scale: i32) {
    let mut type_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
    let mut scale = i32::MIN;
    let st = unsafe { cudf_sys::cudf_scalar_data_type(scalar, &mut type_id, &mut scale) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, expected_type);
    assert_eq!(scale, expected_scale);
}

fn assert_scalar_validity(scalar: *mut c_void, expected: i32) {
    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid(scalar, &mut is_valid) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, expected);
}

unsafe fn scalar_raw_value<T: Copy>(scalar: *mut c_void) -> T {
    let mut out = std::mem::MaybeUninit::<T>::uninit();
    let st = unsafe { cudf_sys::cudf_scalar_value(scalar, out.as_mut_ptr() as *mut c_void) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    unsafe { out.assume_init() }
}

#[test]
fn test_scalar_create_decimal_raw_round_trips_type_scale_validity_and_value() {
    let scale = -2;

    let raw32 = -12345_i32;
    let mut decimal32: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32,
            &raw32 as *const i32 as *const c_void,
            scale,
            1,
            &mut decimal32,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_scalar_data_type(decimal32, cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32, scale);
    assert_scalar_validity(decimal32, 1);
    assert_eq!(unsafe { scalar_raw_value::<i32>(decimal32) }, raw32);

    let raw64 = -9_876_543_210_i64;
    let mut decimal64: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64,
            &raw64 as *const i64 as *const c_void,
            scale,
            1,
            &mut decimal64,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_scalar_data_type(decimal64, cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64, scale);
    assert_scalar_validity(decimal64, 1);
    assert_eq!(unsafe { scalar_raw_value::<i64>(decimal64) }, raw64);

    let raw128 = -123_456_789_012_345_678_901_234_567_i128;
    let mut decimal128: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128,
            &raw128 as *const i128 as *const c_void,
            scale,
            1,
            &mut decimal128,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_scalar_data_type(decimal128, cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128, scale);
    assert_scalar_validity(decimal128, 1);
    assert_eq!(unsafe { scalar_raw_value::<i128>(decimal128) }, raw128);

    unsafe {
        cudf_sys::cudf_scalar_destroy(decimal32);
        cudf_sys::cudf_scalar_destroy(decimal64);
        cudf_sys::cudf_scalar_destroy(decimal128);
    }
}

#[test]
fn test_scalar_create_timestamp_raw_round_trips_type_validity_and_value() {
    let cases = [
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS, 19_118_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS, 1_651_795_200_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MILLISECONDS, 1_651_795_200_123_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MICROSECONDS, 1_651_795_200_123_456_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_NANOSECONDS, 1_651_795_200_123_456_789_i64),
    ];

    for (type_id, raw) in cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_scalar_create_timestamp_raw(type_id, raw, 1, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, 1);
        if type_id == cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar) }, raw);
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_duration_raw_round_trips_type_validity_and_value() {
    let cases = [
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS, -7_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_SECONDS, -123_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MILLISECONDS, -123_456_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MICROSECONDS, -123_456_789_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_NANOSECONDS, -123_456_789_012_i64),
    ];

    for (type_id, raw) in cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_scalar_create_duration_raw(type_id, raw, 1, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, 1);
        if type_id == cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar) }, raw);
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
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    let raw_decimal32 = -12_345_i32;
    let raw_decimal64 = -9_876_543_i64;
    let raw_decimal128 = -123_456_789_012_345_678_901_234_567_i128;
    let decimal_cases = [
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32, &raw_decimal32 as *const i32 as *const c_void, -2),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64, &raw_decimal64 as *const i64 as *const c_void, -4),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128, &raw_decimal128 as *const i128 as *const c_void, -6),
    ];
    for (type_id, raw_ptr, scale) in decimal_cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe {
            cudf_sys::cudf_scalar_create_decimal_raw_ex(type_id, raw_ptr, scale, 1, exec.as_raw(), &mut scalar)
        };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), explicit_stream);
        assert_eq!(last_exec_mr_raw(), explicit_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
        assert_scalar_data_type(scalar, type_id, scale);
        assert_scalar_validity(scalar, 1);
        match type_id {
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32 => {
                assert_eq!(unsafe { scalar_raw_value::<i32>(scalar) }, raw_decimal32)
            },
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64 => {
                assert_eq!(unsafe { scalar_raw_value::<i64>(scalar) }, raw_decimal64)
            },
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128 => {
                assert_eq!(unsafe { scalar_raw_value::<i128>(scalar) }, raw_decimal128)
            },
            _ => unreachable!(),
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }

    let timestamp_cases = [
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS, -12_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS, 1_700_000_000_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MILLISECONDS, 1_700_000_000_123_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MICROSECONDS, 1_700_000_000_123_456_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_NANOSECONDS, 1_700_000_000_123_456_789_i64),
    ];
    for (type_id, raw) in timestamp_cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe { cudf_sys::cudf_scalar_create_timestamp_raw_ex(type_id, raw, 1, exec.as_raw(), &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), explicit_stream);
        assert_eq!(last_exec_mr_raw(), explicit_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, 1);
        if type_id == cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar) }, raw);
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }

    let duration_cases = [
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS, -7_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_SECONDS, -42_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MILLISECONDS, -42_000_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MICROSECONDS, -42_000_000_i64),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_NANOSECONDS, -42_000_000_000_i64),
    ];
    for (type_id, raw) in duration_cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe { cudf_sys::cudf_scalar_create_duration_raw_ex(type_id, raw, 1, exec.as_raw(), &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), explicit_stream);
        assert_eq!(last_exec_mr_raw(), explicit_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, 1);
        if type_id == cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS {
            assert_eq!(unsafe { scalar_raw_value::<i32>(scalar) }, raw as i32);
        } else {
            assert_eq!(unsafe { scalar_raw_value::<i64>(scalar) }, raw);
        }
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_raw_default_constructors_use_default_exec_without_syncing() {
    let _shim_state = lock_shim_state();
    let default_stream = default_stream_value();
    let default_mr = current_device_resource_raw();

    let raw_decimal = -123_i64;
    let mut decimal: *mut c_void = ptr::null_mut();
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64,
            &raw_decimal as *const i64 as *const c_void,
            -2,
            1,
            &mut decimal,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut timestamp: *mut c_void = ptr::null_mut();
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_timestamp_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MICROSECONDS,
            1_700_000_000,
            1,
            &mut timestamp,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut duration: *mut c_void = ptr::null_mut();
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_duration_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_NANOSECONDS,
            -42,
            1,
            &mut duration,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe {
        cudf_sys::cudf_scalar_destroy(duration);
        cudf_sys::cudf_scalar_destroy(timestamp);
        cudf_sys::cudf_scalar_destroy(decimal);
    }
}

#[test]
fn test_scalar_create_raw_ex_null_and_default_options_use_default_exec_without_syncing() {
    let _shim_state = lock_shim_state();
    let default_stream = default_stream_value();
    let default_mr = current_device_resource_raw();
    let default_exec = cudf_sys::ExecutionContext::new();

    for exec_options in [ptr::null(), default_exec.as_raw()] {
        let raw_decimal = 123_i64;
        let mut decimal: *mut c_void = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe {
            cudf_sys::cudf_scalar_create_decimal_raw_ex(
                cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64,
                &raw_decimal as *const i64 as *const c_void,
                -2,
                1,
                exec_options,
                &mut decimal,
            )
        };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), default_stream);
        assert_eq!(last_exec_mr_raw(), default_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

        let mut timestamp: *mut c_void = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe {
            cudf_sys::cudf_scalar_create_timestamp_raw_ex(
                cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS,
                1_700_000_000,
                1,
                exec_options,
                &mut timestamp,
            )
        };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), default_stream);
        assert_eq!(last_exec_mr_raw(), default_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

        let mut duration: *mut c_void = ptr::null_mut();
        reset_exec_observation();
        let st = unsafe {
            cudf_sys::cudf_scalar_create_duration_raw_ex(
                cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_SECONDS,
                -42,
                1,
                exec_options,
                &mut duration,
            )
        };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(last_exec_stream_value(), default_stream);
        assert_eq!(last_exec_mr_raw(), default_mr);
        assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

        unsafe {
            cudf_sys::cudf_scalar_destroy(duration);
            cudf_sys::cudf_scalar_destroy(timestamp);
            cudf_sys::cudf_scalar_destroy(decimal);
        }
    }
}

#[test]
fn test_scalar_create_raw_ex_propagates_exec_option_validation_errors() {
    let bad_exec = cudf_sys::cudf_exec_options_t { struct_size: 0, ..Default::default() };
    let raw_decimal = 123_i64;

    let mut decimal: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64,
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

    let mut timestamp: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_timestamp_raw_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS,
            1_700_000_000,
            1,
            &bad_exec,
            &mut timestamp,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(timestamp.is_null());
    assert!(last_error_message().contains("struct_size"));

    let mut duration: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_duration_raw_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_SECONDS,
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
    let raw_decimal = 1_i64;
    let exec = cudf_sys::ExecutionContext::new();

    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS,
            &raw_decimal as *const i64 as *const c_void,
            -2,
            1,
            exec.as_raw(),
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(scalar.is_null());
    assert!(last_error_message().contains("Unsupported type for decimal scalar"));

    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_timestamp_raw_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_SECONDS,
            1,
            1,
            exec.as_raw(),
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(scalar.is_null());
    assert!(last_error_message().contains("Unsupported type for timestamp scalar"));

    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_duration_raw_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS,
            1,
            1,
            exec.as_raw(),
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(scalar.is_null());
    assert!(last_error_message().contains("Unsupported type for duration scalar"));
}

#[test]
fn test_scalar_create_temporal_days_rejects_i64_values_outside_i32_range() {
    let exec = cudf_sys::ExecutionContext::new();
    let cases = [
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS, i64::from(i32::MAX) + 1, true),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS, i64::from(i32::MIN) - 1, true),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS, i64::from(i32::MAX) + 1, false),
        (cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS, i64::from(i32::MIN) - 1, false),
    ];

    for (type_id, raw, is_timestamp) in cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        let st = if is_timestamp {
            unsafe { cudf_sys::cudf_scalar_create_timestamp_raw_ex(type_id, raw, 1, exec.as_raw(), &mut scalar) }
        } else {
            unsafe { cudf_sys::cudf_scalar_create_duration_raw_ex(type_id, raw, 1, exec.as_raw(), &mut scalar) }
        };
        assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
        assert!(scalar.is_null());
        assert!(last_error_message().contains("outside the requested type range"));
    }
}

#[test]
fn test_scalar_create_raw_decimal_invalid_allows_null_value() {
    let cases = [
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128,
    ];

    for type_id in cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_scalar_create_decimal_raw(type_id, ptr::null(), -3, 0, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, -3);
        assert_scalar_validity(scalar, 0);
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_raw_temporal_invalid_preserves_dtype() {
    let cases = [
        cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MILLISECONDS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MICROSECONDS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_NANOSECONDS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_SECONDS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MILLISECONDS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MICROSECONDS,
        cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_NANOSECONDS,
    ];

    for type_id in cases {
        let mut scalar: *mut c_void = ptr::null_mut();
        let st = if matches!(
            type_id,
            cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS
                | cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS
                | cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MILLISECONDS
                | cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MICROSECONDS
                | cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_NANOSECONDS
        ) {
            unsafe { cudf_sys::cudf_scalar_create_timestamp_raw(type_id, 0, 0, &mut scalar) }
        } else {
            unsafe { cudf_sys::cudf_scalar_create_duration_raw(type_id, 0, 0, &mut scalar) }
        };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_scalar_data_type(scalar, type_id, 0);
        assert_scalar_validity(scalar, 0);
        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_scalar_create_raw_rejects_null_out_and_valid_decimal_null_value() {
    let raw = 123_i32;
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32,
            &raw as *const i32 as *const c_void,
            -2,
            1,
            ptr::null_mut(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);

    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_decimal_raw(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32,
            ptr::null(),
            -2,
            1,
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

#[test]
fn test_scalar_create_string_valid_requires_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(ptr::null(), 1, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

#[test]
fn test_scalar_create_string_invalid_allows_null_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(ptr::null(), 0, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());

    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid(scalar, &mut is_valid) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_view_slice_nested_nullable_preserves_logical_offset_in_null_count() {
    let col = make_i32_column_nullable(&[10, 20, 30, 40, 50], &[true, false, true, false, true]);
    let base_view = make_column_view(col);

    let mut first_slice: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice(base_view, 1, 3, &mut first_slice) };
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

    let mut second_slice: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice(first_slice, 1, 1, &mut second_slice) };
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
    let mut view: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_test_column_view_create_synthetic_fixed_width(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            2,
            i32::MAX - 1,
            &mut view,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!view.is_null());

    let mut slice: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice(view, 1, 1, &mut slice) };

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
    let mut view: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_test_column_view_create_synthetic_string_without_children(1, &mut view) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!view.is_null());

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
            ptr::null(),
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
    let col = make_string_column(&["zero", "one", "two", "three"]);
    let full_view = make_column_view(col);
    let sliced_view = make_sliced_column_view(col, 1, 2);

    let mut full_ptr: *const c_void = ptr::null();
    let mut full_size: usize = 0;
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(full_view, ptr::null(), &mut full_ptr, &mut full_size)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!full_ptr.is_null());
    assert_eq!(full_size, "zeroonetwothree".len());

    let mut sliced_ptr: *const c_void = ptr::null();
    let mut sliced_size: usize = 0;
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(
            sliced_view,
            ptr::null(),
            &mut sliced_ptr,
            &mut sliced_size,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!sliced_ptr.is_null());
    assert_eq!(sliced_ptr, full_ptr);
    assert_eq!(sliced_size, "onetwo".len());

    unsafe {
        cudf_sys::cudf_column_view_destroy(sliced_view);
        cudf_sys::cudf_column_view_destroy(full_view);
        cudf_sys::cudf_column_destroy(col);
    }
}
