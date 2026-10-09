use std::{os::raw::c_void, ptr};

use cudf_sys::test_support;

use super::*;
use crate::support::run_named_cases;

fn test_interleave_columns_rejects_oversized_output_before_allocation() {
    let rows = i32::MAX / 2 + 1;
    let mut first: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let mut second: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    for out in [&mut first, &mut second] {
        let status = unsafe {
            test_support::cudf_test_column_view_create_synthetic_fixed_width(cudf_sys::CUDF_TYPE_INT32, rows, 0, out)
        };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert!(!out.is_null());
    }

    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let allocations_before = mr.allocation_count();
    let views = [first, second];
    let mut output = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status =
        unsafe { cudf_sys::cudf_interleave_columns_ex(views.as_ptr(), views.len() as i32, &exec, &mut output) };
    let error = last_error_message();
    let allocations_after = mr.allocation_count();

    unsafe {
        cudf_sys::cudf_column_view_destroy(second);
        cudf_sys::cudf_column_view_destroy(first);
    }

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(output.is_null());
    assert!(error.contains("interleave output size exceeds cudf size_type maximum"));
    assert_eq!(allocations_after, allocations_before, "oversized interleave must not launch or allocate output");
}

fn test_column_create_numeric_negative_size_rejected() {
    let data: [i32; 1] = [42];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            -1,
            ptr::null(),
            0,
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

fn test_column_create_numeric_null_count_out_of_range_rejected() {
    let data = [1i32, 2, 3];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            4,
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

fn test_column_create_numeric_null_count_requires_mask() {
    let data = [1i32, 2, 3];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            2,
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

fn test_column_create_fixed_width_from_host_preserves_decimal_scale() {
    let data = [12345i128, -67890, 5];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host_ex(
            cudf_sys::CUDF_TYPE_DECIMAL128,
            -2,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let view = make_column_view(out);
    let mut type_id = cudf_sys::CUDF_TYPE_EMPTY;
    let mut scale = 0;
    let status = unsafe { cudf_sys::cudf_column_view_data_type(view, &mut type_id, &mut scale) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, cudf_sys::CUDF_TYPE_DECIMAL128);
    assert_eq!(scale, -2);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(out);
    }
}

fn test_column_create_fixed_width_from_host_rejects_nonzero_scale_for_non_decimal() {
    let data = [1i32, 2, 3];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host_ex(
            cudf_sys::CUDF_TYPE_INT32,
            -2,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("scale must be 0"));
}

fn test_column_create_fixed_width_from_host_rejects_non_fixed_width_type() {
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host_ex(
            cudf_sys::CUDF_TYPE_STRING,
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_TYPE);
    assert!(out.is_null());
}

fn test_column_create_fixed_width_from_host_null_count_requires_mask() {
    let data = [1i32, 2, 3];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_fixed_width_from_host_ex(
            cudf_sys::CUDF_TYPE_INT32,
            0,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            1,
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

fn test_column_view_to_host_string_type_rejected() {
    let strings = [std::ffi::CString::new("ab").unwrap(), std::ffi::CString::new("cd").unwrap()];
    let mut string_ptrs = [strings[0].as_ptr(), strings[1].as_ptr()];
    let lengths = [2i32, 2i32];

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut col: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_mut_ptr(),
            lengths.as_ptr(),
            2,
            ptr::null(),
            0,
            &exec,
            &mut col,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!col.is_null());

    let view = make_column_view(col);
    let mut buf = [0u8; 16];
    let status =
        unsafe { cudf_sys::cudf_column_view_to_host_ex(view, buf.as_mut_ptr() as *mut c_void, buf.len(), &exec) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_TYPE);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn column_cases() {
    run_named_cases(&[
        ("test_column_create_numeric_null_count_requires_mask", test_column_create_numeric_null_count_requires_mask),
        (
            "test_column_create_fixed_width_from_host_rejects_nonzero_scale_for_non_decimal",
            test_column_create_fixed_width_from_host_rejects_nonzero_scale_for_non_decimal,
        ),
        (
            "test_column_create_fixed_width_from_host_preserves_decimal_scale",
            test_column_create_fixed_width_from_host_preserves_decimal_scale,
        ),
        ("test_column_create_numeric_negative_size_rejected", test_column_create_numeric_negative_size_rejected),
        (
            "test_column_create_numeric_null_count_out_of_range_rejected",
            test_column_create_numeric_null_count_out_of_range_rejected,
        ),
        (
            "test_interleave_columns_rejects_oversized_output_before_allocation",
            test_interleave_columns_rejects_oversized_output_before_allocation,
        ),
        ("test_column_view_to_host_string_type_rejected", test_column_view_to_host_string_type_rejected),
        (
            "test_column_create_fixed_width_from_host_null_count_requires_mask",
            test_column_create_fixed_width_from_host_null_count_requires_mask,
        ),
        (
            "test_column_create_fixed_width_from_host_rejects_non_fixed_width_type",
            test_column_create_fixed_width_from_host_rejects_non_fixed_width_type,
        ),
    ]);
}
