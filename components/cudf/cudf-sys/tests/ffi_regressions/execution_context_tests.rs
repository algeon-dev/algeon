use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::*,
    support::{
        NO_SYNC_RECORDED, OwnedStream, current_device_resource_raw, last_exec_mr_raw, last_exec_stream_value,
        last_exec_sync_stream_value, lock_shim_state, make_stats_memory_resource, owned_string_value,
        required_exec_options, reset_exec_observation, stream_value,
    },
};

fn strict_exec_owners() -> (OwnedStream, crate::support::OwnedMemoryResource, cudf_sys::cudf_exec_options_t) {
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    (stream, mr, exec)
}

#[test]
fn test_is_sorted_requires_explicit_exec_options() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3, 4]);
    let tbl = make_table_from_one_column(col);
    let view = make_table_view(tbl);

    let mut null_opts_out = -1;
    let st = unsafe { cudf_sys::cudf_is_sorted_ex(view, ptr::null(), ptr::null(), 1, &mut null_opts_out, ptr::null()) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert_eq!(null_opts_out, 0);

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    let mut explicit_out = 0;
    reset_exec_observation();
    let st = unsafe { cudf_sys::cudf_is_sorted_ex(view, ptr::null(), ptr::null(), 1, &mut explicit_out, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(explicit_out, 1);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe {
        cudf_sys::cudf_table_view_destroy(view);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_device_buffer_copy_from_host_requires_explicit_exec_options_and_syncs() {
    let _shim_state = lock_shim_state();
    let values = [1i32, 2, 3];
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    let mut buffer: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_device_buffer_create_ex(std::mem::size_of_val(&values), &exec, &mut buffer) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!buffer.is_null());

    let st = unsafe {
        cudf_sys::cudf_device_buffer_copy_from_host_ex(
            buffer,
            values.as_ptr() as *const c_void,
            std::mem::size_of_val(&values),
            ptr::null(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_device_buffer_copy_from_host_ex(
            buffer,
            values.as_ptr() as *const c_void,
            std::mem::size_of_val(&values),
            &exec,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    unsafe { cudf_sys::cudf_device_buffer_destroy(buffer) };
}

#[test]
fn test_column_copy_requires_explicit_exec_options() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3, 4]);

    reset_exec_observation();
    let mut null_opts_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, ptr::null(), &mut null_opts_out) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(null_opts_out.is_null());

    reset_exec_observation();
    let default_exec = cudf_sys::ExecutionContext::new();
    let mut default_opts_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, default_exec.as_raw(), &mut default_opts_out) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(default_opts_out.is_null());

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, &exec, &mut explicit_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i32_column_data_with_exec(explicit_out, 4, &exec), vec![1, 2, 3, 4]);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scalar_create_numeric_remains_unsynchronized_before_return() {
    let _shim_state = lock_shim_state();
    let value = 42i32;
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    reset_exec_observation();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            &value as *const i32 as *const c_void,
            1,
            &exec,
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut value = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value_ex(scalar, &mut value as *mut i32 as *mut c_void, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(value, 42);
    stream.synchronize();
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
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
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    reset_exec_observation();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            &value as *const i32 as *const c_void,
            1,
            &exec,
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
    let st = unsafe { cudf_sys::cudf_scalar_value_ex(scalar, &mut out as *mut i32 as *mut c_void, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, value);
    stream.synchronize();
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
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    reset_exec_observation();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string_ex(hello.as_ptr(), 1, &exec, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_ne!(last_exec_mr_raw(), ptr::null_mut());
    assert_ne!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut out, &mut len, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    stream.synchronize();
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_create_numeric_host_source_sync_contract() {
    let _shim_state = lock_shim_state();
    let data = [31i32, 32, 33];
    let validity = [true, false, true];
    let null_mask = validity_to_bitmask(&validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            &exec,
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
    }
}

#[test]
fn test_column_create_fixed_width_from_host_ex_uses_explicit_stream_and_syncs() {
    let _shim_state = lock_shim_state();
    let data = [41i32, 42, 43];
    let validity = [true, false, true];
    let null_mask = validity_to_bitmask(&validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            0,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            &exec,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
    assert_eq!(read_i32_column_data(out, data.len()), data);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
    }
}

#[test]
fn test_column_create_fixed_width_from_host_ex_syncs_on_error_after_data_copy() {
    let _shim_state = lock_shim_state();
    let data = [51i32, 52, 53];
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    unsafe { cudf_sys::cudf_test_fail_next_fixed_width_host_import_after_data_copy() };
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            0,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &exec,
            &mut out,
        )
    };

    assert_ne!(st, cudf_sys::CUDF_SUCCESS);
    assert!(out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
}

#[test]
fn test_column_create_fixed_width_from_host_ex_syncs_on_error_after_null_mask_copy() {
    let _shim_state = lock_shim_state();
    let data = [61i32, 62, 63];
    let validity = [true, false, true];
    let null_mask = validity_to_bitmask(&validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    unsafe { cudf_sys::cudf_test_fail_next_fixed_width_host_import_after_null_mask_copy() };
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            0,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            &exec,
            &mut out,
        )
    };

    assert_ne!(st, cudf_sys::CUDF_SUCCESS);
    assert!(out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
}

#[test]
fn test_column_create_strings_host_source_sync_contract() {
    let _shim_state = lock_shim_state();
    let values = ["ab", "", "cde"];
    let c_strings: Vec<_> = values.iter().map(|value| std::ffi::CString::new(*value).unwrap()).collect();
    let string_ptrs: Vec<_> = c_strings.iter().map(|value| value.as_ptr()).collect();
    let lengths: Vec<_> = values.iter().map(|value| value.len() as i32).collect();

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            values.len() as i32,
            ptr::null(),
            0,
            &exec,
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
    }
}

fn assert_strings_import_error_syncs_selected_stream(
    arm_fault: unsafe extern "C" fn(),
    values: &[&str],
    validity: Option<&[bool]>,
) {
    let _shim_state = lock_shim_state();
    let c_strings: Vec<_> = values.iter().map(|value| std::ffi::CString::new(*value).unwrap()).collect();
    let string_ptrs: Vec<_> = c_strings.iter().map(|value| value.as_ptr()).collect();
    let lengths: Vec<_> = values.iter().map(|value| value.len() as i32).collect();
    let null_mask = validity.map(validity_to_bitmask).unwrap_or_default();
    let null_count =
        validity.map(|validity| validity.iter().filter(|&&is_valid| !is_valid).count() as i32).unwrap_or(0);
    let null_mask_ptr = if null_mask.is_empty() { ptr::null() } else { null_mask.as_ptr() };

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    unsafe { arm_fault() };
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            values.len() as i32,
            null_mask_ptr,
            null_count,
            &exec,
            &mut out,
        )
    };

    assert_ne!(st, cudf_sys::CUDF_SUCCESS);
    assert!(out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
}

#[test]
fn test_column_view_slice_ex_uses_explicit_stream_for_null_count() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column_nullable(&[10, 20, 30, 40], &[true, false, true, false]);
    let view = make_column_view(col);
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    let mut slice: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice_ex(view, 1, 2, &exec, &mut slice) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!slice.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut null_count = -1;
    let st = unsafe { cudf_sys::cudf_column_view_null_count(slice, &mut null_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(null_count, 1);

    unsafe {
        cudf_sys::cudf_column_view_destroy(slice);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_create_strings_ex_syncs_on_error_after_offsets_copy() {
    assert_strings_import_error_syncs_selected_stream(
        cudf_sys::cudf_test_fail_next_strings_host_import_after_offsets_copy,
        &["ab", "", "cde"],
        None,
    );
}

#[test]
fn test_column_create_strings_ex_syncs_on_error_after_chars_copy() {
    assert_strings_import_error_syncs_selected_stream(
        cudf_sys::cudf_test_fail_next_strings_host_import_after_chars_copy,
        &["ab", "", "cde"],
        None,
    );
}

#[test]
fn test_column_create_strings_ex_syncs_on_error_after_null_mask_copy() {
    let validity = [true, false, true];
    assert_strings_import_error_syncs_selected_stream(
        cudf_sys::cudf_test_fail_next_strings_host_import_after_null_mask_copy,
        &["ab", "", "cde"],
        Some(&validity),
    );
}

#[test]
fn test_column_create_strings_ex_syncs_on_error_before_make_strings_column() {
    assert_strings_import_error_syncs_selected_stream(
        cudf_sys::cudf_test_fail_next_strings_host_import_before_make_strings_column,
        &["ab", "", "cde"],
        None,
    );
}

#[test]
fn test_column_create_strings_ex_syncs_on_error_after_make_strings_column() {
    assert_strings_import_error_syncs_selected_stream(
        cudf_sys::cudf_test_fail_next_strings_host_import_after_make_strings_column,
        &["ab", "", "cde"],
        None,
    );
}

#[test]
fn test_stream_handle_has_no_effect_until_attached_to_execution_context() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[7, 8, 9]);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());

    let borrowed_stream = stream.borrowed();
    let mr = make_stats_memory_resource();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, &exec, &mut explicit_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_to_host_execution_context_defaults_and_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[11, 12, 13]);
    let view = make_column_view(col);

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    let mut explicit_buf = vec![0i32; 3];
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_to_host_ex(
            view,
            explicit_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(explicit_buf.as_slice()),
            &exec,
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
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

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
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    let mut host_mask = [0u8; 64];

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_null_mask_to_host_ex(view, host_mask.as_mut_ptr(), host_mask.len(), &exec)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(host_mask[0] & 0b0001_1111, 0b0001_0101);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

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
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
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
            &exec,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(offsets, vec![0, 2, 2, 5]);
    assert_eq!(total_chars, 5);
    assert_eq!(chars.as_slice(), b"abcde");
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

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
    let (_stream, _mr, exec) = strict_exec_owners();

    reset_exec_observation();
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
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    let mut ex_offsets = vec![-1i32; 1];
    let mut ex_total_chars = -1;

    reset_exec_observation();
    let ex_st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            ex_offsets.as_mut_ptr(),
            std::mem::size_of_val(ex_offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut ex_total_chars,
            &exec,
        )
    };
    assert_eq!(ex_st, cudf_sys::CUDF_SUCCESS);
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
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
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
            &exec,
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
fn test_column_view_string_chars_ptr_and_size_empty_reports_zero_bytes_without_shim_sync() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    let mut chars_ptr: *const c_void = ptr::null();
    let mut chars_size: usize = usize::MAX;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(view, &exec, &mut chars_ptr, &mut chars_size)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert!(chars_ptr.is_null());
    assert_eq!(chars_size, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_string_chars_ptr_and_size_ex_uses_explicit_stream_without_shim_sync() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["ab", "", "cde"]);
    let view = make_column_view(col);
    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());
    let mut chars_ptr: *const c_void = ptr::null();
    let mut chars_size: usize = 0;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(view, &exec, &mut chars_ptr, &mut chars_size)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert!(!chars_ptr.is_null());
    assert_eq!(chars_size, 5);

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
    let mr = make_stats_memory_resource();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    reset_exec_observation();
    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid_ex(scalar, &mut is_valid, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 1);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    stream.synchronize();
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_value_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_i32_scalar(42);
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    reset_exec_observation();
    let mut out = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value_ex(scalar, &mut out as *mut i32 as *mut c_void, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 42);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    stream.synchronize();
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_string_value_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_string_scalar("hello");
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    reset_exec_observation();
    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut out, &mut len, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    stream.synchronize();
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_create_from_device_data_sync_and_async_contracts() {
    let _shim_state = lock_shim_state();
    let source = make_i32_column(&[21, 22, 23]);
    let source_view = make_column_view(source);

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
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(sync_out.is_null());

    let (stream, _mr, exec) = strict_exec_owners();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    let mut async_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_from_device_data_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            device_ptr,
            3,
            &exec,
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

    assert_eq!(stats_mr.allocation_count(), 0);

    let stream = OwnedStream::new();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = stats_mr.borrowed();
    let exec_opts = required_exec_options(&borrowed_stream, &borrowed_mr);
    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, &exec_opts, &mut explicit_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(stats_mr.allocation_count() > 0);
    assert_ne!(last_exec_mr_raw(), default_mr);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(col);
    }
}
