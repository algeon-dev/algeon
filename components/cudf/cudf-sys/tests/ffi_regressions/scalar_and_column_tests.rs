use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{helpers::*, support::owned_string_value};

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
