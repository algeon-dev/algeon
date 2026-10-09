use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use cudf_sys::test_support;

use super::*;
use crate::support::run_named_cases;

fn read_bool8_column_data(col: cudf_sys::cudf_column_handle) -> Vec<u8> {
    let view = make_column_view(col);

    let mut type_id = cudf_sys::CUDF_TYPE_EMPTY;
    let status = unsafe { cudf_sys::cudf_column_view_type(view, &mut type_id) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, cudf_sys::CUDF_TYPE_BOOL8);

    let mut size = 0;
    let status = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(size >= 0);

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut values = vec![0u8; size as usize];
    let status = unsafe {
        cudf_sys::cudf_column_view_to_host_ex(
            view,
            values.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(values.as_slice()),
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    values
}

fn prepare_replace_with_backrefs(
    view: cudf_sys::cudf_column_view_handle,
    exec: &cudf_sys::cudf_exec_options_t,
) -> cudf_sys::cudf_prepared_replace_with_backrefs_handle {
    let pattern = r"([a-z]+)-(\d+)";
    let replacement = r"\2/\1";
    let mut prepared = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_prepare_replace_with_backrefs_ex(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            replacement.as_ptr() as *const c_char,
            replacement.len() as i32,
            &mut prepared,
            exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!prepared.is_null());
    prepared
}

fn test_column_view_strings_to_host_empty_rejects_undersized_offsets_buffer() {
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut offsets = vec![-1i32; 1];
    let mut total_chars = -1;

    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            offsets.as_mut_ptr(),
            0,
            ptr::null_mut(),
            0,
            &mut total_chars,
            &exec,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert_eq!(offsets, vec![-1]);
    assert_eq!(total_chars, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

fn test_strings_like_validates_pattern_and_escape_arguments() {
    let col = make_string_column(&["abc"]);
    let view = make_column_view(col);
    let pattern = b"a%";
    let escape = b"/";
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_strings_like_ex(view, pattern.as_ptr() as *const c_char, -1, ptr::null(), 0, &mut out, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe { cudf_sys::cudf_strings_like_ex(view, ptr::null(), 1, ptr::null(), 0, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_strings_like_ex(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            ptr::null(),
            -1,
            &mut out,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_strings_like_ex(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            escape.as_ptr() as *const c_char,
            2,
            &mut out,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_strings_like_ex(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            ptr::null(),
            1,
            &mut out,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

fn prepared_replace_execute_invalid_options_clean_up_under_prepare_provenance() {
    let col = make_string_column(&["abc-123", "def-456"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let baseline = mr.active_allocation_count();
    let prepared = prepare_replace_with_backrefs(view, &exec);
    assert!(mr.active_allocation_count() > baseline);

    let invalid_exec = cudf_sys::cudf_exec_options_t { struct_size: 0, ..Default::default() };
    let mut out = std::ptr::dangling_mut::<cudf_sys::cudf_column_handle_state>();
    let status =
        unsafe { cudf_sys::cudf_strings_execute_replace_with_backrefs_ex(view, prepared, &mut out, &invalid_exec) };
    assert_ne!(status, cudf_sys::CUDF_SUCCESS);
    assert!(out.is_null());
    assert_eq!(
        mr.active_allocation_count(),
        baseline,
        "failed option resolution must release prepared allocations through captured provenance"
    );

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn prepared_replace_execute_cleanup_failure_is_quarantined_without_crossing_abi() {
    let col = make_string_column(&["abc-123", "def-456"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let baseline = mr.active_allocation_count();
    let prepared = prepare_replace_with_backrefs(view, &exec);
    let prepared_allocations = mr.active_allocation_count();
    assert!(prepared_allocations > baseline);

    unsafe { test_support::cudf_test_fail_next_provenance_destroy(cudf_sys::CUDF_ERROR_FATAL_CUDA) };
    let invalid_exec = cudf_sys::cudf_exec_options_t { struct_size: 0, ..Default::default() };
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_strings_execute_replace_with_backrefs_ex(view, prepared, &mut out, &invalid_exec) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_FATAL_CUDA);
    assert!(out.is_null());
    assert_eq!(
        mr.active_allocation_count(),
        prepared_allocations,
        "injected cleanup failure must quarantine rather than destroy on an unverified device"
    );

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

fn test_strings_like_matches_and_synchronizes_explicit_stream() {
    let col = make_string_column(&["abc", "adc", "ab", "axc"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let pattern = b"a_c";

    reset_exec_observation();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_like_ex(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            ptr::null(),
            0,
            &mut out,
            &exec,
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

fn test_strings_slice_validates_step_and_initializes_output() {
    let col = make_string_column(&["abcdef"]);
    let view = make_column_view(col);
    let mut out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();

    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe { cudf_sys::cudf_strings_slice_ex(view, 0, 1, 3, 1, 0, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

fn test_strings_slice_uses_validity_flags_and_explicit_stream() {
    let col = make_string_column(&["abcdef", "xy"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());

    reset_exec_observation();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_slice_ex(view, 1, 1, 4, 1, 2, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_string_column_data(out), vec!["bd".to_string(), "y".to_string()]);
    unsafe { cudf_sys::cudf_column_destroy(out) };

    out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_slice_ex(view, 2, 1, 0, 0, 1, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(read_string_column_data(out), vec!["cdef".to_string(), "".to_string()]);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

fn test_column_create_strings_negative_length_rejected() {
    let strings = [std::ffi::CString::new("ab").unwrap()];
    let mut string_ptrs = [strings[0].as_ptr()];
    let lengths = [-1i32];

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut col: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_mut_ptr(),
            lengths.as_ptr(),
            1,
            ptr::null(),
            0,
            &exec,
            &mut col,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(col.is_null());
}

fn test_column_create_strings_positive_length_requires_non_null_string_pointer() {
    let mut string_ptrs = [ptr::null()];
    let lengths = [2i32];

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut col: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_mut_ptr(),
            lengths.as_ptr(),
            1,
            ptr::null(),
            0,
            &exec,
            &mut col,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(col.is_null());
}

fn test_column_create_strings_rejects_count_that_would_overflow_offsets() {
    let strings = [std::ffi::CString::new("").unwrap()];
    let mut string_ptrs = [strings[0].as_ptr()];
    let lengths = [0i32];
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut col = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_mut_ptr(),
            lengths.as_ptr(),
            i32::MAX,
            ptr::null(),
            0,
            &exec,
            &mut col,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(col.is_null());
    assert!(last_error_message().contains("num_strings"));
}

#[test]
fn strings_cases() {
    run_named_cases(&[
        (
            "test_strings_slice_uses_validity_flags_and_explicit_stream",
            test_strings_slice_uses_validity_flags_and_explicit_stream,
        ),
        (
            "test_strings_like_matches_and_synchronizes_explicit_stream",
            test_strings_like_matches_and_synchronizes_explicit_stream,
        ),
        (
            "prepared_replace_execute_invalid_options_clean_up_under_prepare_provenance",
            prepared_replace_execute_invalid_options_clean_up_under_prepare_provenance,
        ),
        (
            "test_strings_slice_validates_step_and_initializes_output",
            test_strings_slice_validates_step_and_initializes_output,
        ),
        (
            "test_column_view_strings_to_host_empty_rejects_undersized_offsets_buffer",
            test_column_view_strings_to_host_empty_rejects_undersized_offsets_buffer,
        ),
        (
            "test_column_create_strings_positive_length_requires_non_null_string_pointer",
            test_column_create_strings_positive_length_requires_non_null_string_pointer,
        ),
        (
            "test_strings_like_validates_pattern_and_escape_arguments",
            test_strings_like_validates_pattern_and_escape_arguments,
        ),
        ("test_column_create_strings_negative_length_rejected", test_column_create_strings_negative_length_rejected),
        (
            "test_column_create_strings_rejects_count_that_would_overflow_offsets",
            test_column_create_strings_rejects_count_that_would_overflow_offsets,
        ),
    ]);
}
