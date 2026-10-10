use std::ptr;

use super::*;

#[test]
fn test_scan_invalid_scan_type_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);
    let mut agg: cudf_sys::cudf_scan_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scan_aggregation_create(cudf_sys::CUDF_REDUCE_AGG_SUM, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_scan_raw(raw_handle(view), raw_handle(agg), 99, 0, raw_out_handle(&mut out), &exec) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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
    let mut agg: cudf_sys::cudf_scan_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scan_aggregation_create(cudf_sys::CUDF_REDUCE_AGG_SUM, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_scan_raw(
            raw_handle(view),
            raw_handle(agg),
            cudf_sys::CUDF_SCAN_INCLUSIVE as i32,
            99,
            raw_out_handle(&mut out),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_unary_operation_raw(raw_handle(view), 99, raw_out_handle(&mut out), &exec) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_binary_operation_vv_raw(
            raw_handle(lhs_view),
            raw_handle(rhs_view),
            99,
            raw_handle(dtype),
            raw_out_handle(&mut out),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_binary_operation_vs_raw(
            raw_handle(lhs_view),
            raw_handle(rhs_scalar),
            99,
            raw_handle(dtype),
            raw_out_handle(&mut out),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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
    let (_stream, _mr, exec) = strict_exec_owners();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_binary_operation_sv_raw(
            raw_handle(lhs_scalar),
            raw_handle(rhs_view),
            99,
            raw_handle(dtype),
            raw_out_handle(&mut out),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid binary operator"));

    unsafe {
        cudf_sys::cudf_data_type_destroy(dtype);
        cudf_sys::cudf_column_view_destroy(rhs_view);
        cudf_sys::cudf_column_destroy(rhs_col);
        cudf_sys::cudf_scalar_destroy(lhs_scalar);
    }
}
