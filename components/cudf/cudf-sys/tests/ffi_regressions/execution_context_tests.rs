use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::*,
    support::{
        NO_SYNC_RECORDED, OwnedStream, current_device_resource_raw, default_stream_value, last_exec_mr_raw,
        last_exec_stream_value, last_exec_sync_stream_value, lock_shim_state, make_stats_memory_resource,
        owned_string_value, reset_exec_observation, stream_value,
    },
};

#[test]
fn test_execution_context_defaults_match_legacy_column_copy() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3, 4]);
    let default_stream = default_stream_value();
    let default_mr = current_device_resource_raw();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy(col, &mut legacy_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i32_column_data(legacy_out, 4), vec![1, 2, 3, 4]);

    reset_exec_observation();
    let mut null_opts_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, ptr::null(), &mut null_opts_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i32_column_data(null_opts_out, 4), vec![1, 2, 3, 4]);

    let exec = cudf_sys::ExecutionContext::new();
    reset_exec_observation();
    let mut explicit_defaults_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, exec.as_raw(), &mut explicit_defaults_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i32_column_data(explicit_defaults_out, 4), vec![1, 2, 3, 4]);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_defaults_out);
        cudf_sys::cudf_column_destroy(null_opts_out);
        cudf_sys::cudf_column_destroy(legacy_out);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scalar_create_numeric_remains_unsynchronized_before_return() {
    let _shim_state = lock_shim_state();
    reset_exec_observation();
    let scalar = make_i32_scalar(42);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut value = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value(scalar, &mut value as *mut i32 as *mut c_void) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(value, 42);
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_string_synchronizes_default_stream_before_return() {
    let _shim_state = lock_shim_state();
    let default_stream = default_stream_value();
    let hello = std::ffi::CString::new("hello").unwrap();
    reset_exec_observation();
    let mut string_scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(hello.as_ptr(), 1, &mut string_scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!string_scalar.is_null());
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(string_scalar, &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    unsafe { cudf_sys::cudf_scalar_destroy(string_scalar) };
}

#[test]
fn test_scalar_create_numeric_ex_uses_explicit_stream_and_mr_without_syncing_before_return() {
    let _shim_state = lock_shim_state();
    let value = 42i32;
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    reset_exec_observation();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            &value as *const i32 as *const c_void,
            1,
            exec.as_raw(),
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_ne!(last_exec_mr_raw(), ptr::null_mut());
    assert_ne!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut out = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value(scalar, &mut out as *mut i32 as *mut c_void) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, value);
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_string_ex_uses_explicit_stream_and_mr_and_synchronizes_before_return() {
    let _shim_state = lock_shim_state();
    let hello = std::ffi::CString::new("hello").unwrap();
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    reset_exec_observation();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string_ex(hello.as_ptr(), 1, exec.as_raw(), &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_ne!(last_exec_mr_raw(), ptr::null_mut());
    assert_ne!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_create_numeric_host_source_sync_contract() {
    let _shim_state = lock_shim_state();
    let data = [31i32, 32, 33];
    let validity = [true, false, true];
    let null_mask = validity_to_bitmask(&validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;
    let default_stream = default_stream_value();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            &mut legacy_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!legacy_out.is_null());
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);
    assert_eq!(read_i32_column_data(legacy_out, data.len()), data);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            exec.as_raw(),
            &mut explicit_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!explicit_out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
    assert_eq!(read_i32_column_data(explicit_out, data.len()), data);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
    }
}

#[test]
fn test_column_create_strings_host_source_sync_contract() {
    let _shim_state = lock_shim_state();
    let values = ["ab", "", "cde"];
    let c_strings: Vec<_> = values.iter().map(|value| std::ffi::CString::new(*value).unwrap()).collect();
    let string_ptrs: Vec<_> = c_strings.iter().map(|value| value.as_ptr()).collect();
    let lengths: Vec<_> = values.iter().map(|value| value.len() as i32).collect();
    let default_stream = default_stream_value();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            values.len() as i32,
            ptr::null(),
            0,
            &mut legacy_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!legacy_out.is_null());
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);
    assert_eq!(read_string_column_data(legacy_out), values);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            values.len() as i32,
            ptr::null(),
            0,
            exec.as_raw(),
            &mut explicit_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!explicit_out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
    assert_eq!(read_string_column_data(explicit_out), values);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
    }
}

#[test]
fn test_stream_handle_has_no_effect_until_attached_to_execution_context() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[7, 8, 9]);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy(col, &mut legacy_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_ne!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, exec.as_raw(), &mut explicit_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_to_host_execution_context_defaults_and_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[11, 12, 13]);
    let view = make_column_view(col);
    let default_stream = default_stream_value();

    let mut legacy_buf = vec![0i32; 3];
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_to_host(
            view,
            legacy_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(legacy_buf.as_slice()),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(legacy_buf, vec![11, 12, 13]);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let mut explicit_buf = vec![0i32; 3];
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_to_host_ex(
            view,
            explicit_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(explicit_buf.as_slice()),
            exec.as_raw(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(explicit_buf, vec![11, 12, 13]);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    let explicit_defaults = cudf_sys::ExecutionContext::new();
    let mut default_opts_buf = vec![0i32; 3];
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_to_host_ex(
            view,
            default_opts_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(default_opts_buf.as_slice()),
            explicit_defaults.as_raw(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(default_opts_buf, vec![11, 12, 13]);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_null_mask_to_host_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column_nullable(&[1, 2, 3, 4, 5], &[true, false, true, false, true]);
    let view = make_column_view(col);
    let default_stream = default_stream_value();
    let mut host_mask = [0u8; 64];

    reset_exec_observation();
    let st = unsafe { cudf_sys::cudf_column_view_null_mask_to_host(view, host_mask.as_mut_ptr(), host_mask.len()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(host_mask[0] & 0b0001_1111, 0b0001_0101);
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["ab", "", "cde"]);
    let view = make_column_view(col);
    let default_stream = default_stream_value();
    let mut offsets = vec![-1i32; 4];
    let mut chars = vec![0u8; 5];
    let mut total_chars = -1;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            chars.as_mut_ptr() as *mut c_char,
            chars.len(),
            &mut total_chars,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(offsets, vec![0, 2, 2, 5]);
    assert_eq!(total_chars, 5);
    assert_eq!(chars.as_slice(), b"abcde");
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_empty_writes_single_zero_offset() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let mut offsets = vec![-1i32; 1];
    let mut total_chars = -1;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut total_chars,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(offsets, vec![0]);
    assert_eq!(total_chars, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_empty_ex_matches_legacy_contract() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let mut legacy_offsets = vec![-1i32; 1];
    let mut legacy_total_chars = -1;
    let mut ex_offsets = vec![-1i32; 1];
    let mut ex_total_chars = -1;

    let legacy_st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            legacy_offsets.as_mut_ptr(),
            std::mem::size_of_val(legacy_offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut legacy_total_chars,
        )
    };
    assert_eq!(legacy_st, cudf_sys::CUDF_SUCCESS);

    reset_exec_observation();
    let ex_st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            ex_offsets.as_mut_ptr(),
            std::mem::size_of_val(ex_offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut ex_total_chars,
            exec.as_raw(),
        )
    };
    assert_eq!(ex_st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(ex_offsets, legacy_offsets);
    assert_eq!(ex_total_chars, legacy_total_chars);
    assert_eq!(ex_offsets, vec![0]);
    assert_eq!(ex_total_chars, 0);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_ex_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["ab", "", "cde"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let mut offsets = vec![-1i32; 4];
    let mut chars = vec![0u8; 5];
    let mut total_chars = -1;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            chars.as_mut_ptr() as *mut c_char,
            chars.len(),
            &mut total_chars,
            exec.as_raw(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(offsets, vec![0, 2, 2, 5]);
    assert_eq!(total_chars, 5);
    assert_eq!(chars.as_slice(), b"abcde");
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scalar_is_valid_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_i32_scalar(42);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid_ex(scalar, &mut is_valid, exec.as_raw()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 1);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_value_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_i32_scalar(42);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut out = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value_ex(scalar, &mut out as *mut i32 as *mut c_void, exec.as_raw()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 42);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_string_value_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_string_scalar("hello");
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut out, &mut len, exec.as_raw()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_create_from_device_data_sync_and_async_contracts() {
    let _shim_state = lock_shim_state();
    let source = make_i32_column(&[21, 22, 23]);
    let source_view = make_column_view(source);
    let default_stream = default_stream_value();

    let mut device_ptr: *const c_void = ptr::null();
    let st = unsafe { cudf_sys::cudf_column_view_data_ptr(source_view, &mut device_ptr) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!device_ptr.is_null());

    reset_exec_observation();
    let mut sync_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_from_device_data(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            device_ptr,
            3,
            &mut sync_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_sync_stream_value(), default_stream);
    assert_eq!(read_i32_column_data(sync_out, 3), vec![21, 22, 23]);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut async_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_from_device_data_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            device_ptr,
            3,
            exec.as_raw(),
            &mut async_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    stream.synchronize();
    assert_eq!(read_i32_column_data(async_out, 3), vec![21, 22, 23]);

    unsafe {
        cudf_sys::cudf_column_destroy(async_out);
        cudf_sys::cudf_column_destroy(sync_out);
        cudf_sys::cudf_column_view_destroy(source_view);
        cudf_sys::cudf_column_destroy(source);
    }
}

#[test]
fn test_memory_resource_has_no_effect_until_attached_to_execution_context() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[21, 22, 23]);
    let stats_mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy(col, &mut legacy_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(stats_mr.allocation_count(), 0);
    assert_eq!(last_exec_mr_raw(), default_mr);

    let borrowed_mr = stats_mr.borrowed();
    let exec_opts = cudf_sys::ExecutionContext::new().with_memory_resource(&borrowed_mr);
    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, exec_opts.as_raw(), &mut explicit_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(stats_mr.allocation_count() > 0);
    assert_ne!(last_exec_mr_raw(), default_mr);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
        cudf_sys::cudf_column_destroy(col);
    }
}
