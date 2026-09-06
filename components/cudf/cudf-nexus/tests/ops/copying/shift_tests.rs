//! Tests for `cudf::ops::copying::shift`.

use cudf::{ops::copying, testing::prelude::*};

use crate::common::{column_from_slice, null_scalar, scalar};

fn shift_forward_with_null_fill() {
    let col = column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let cv = col.testing_view().unwrap();
    let fill = null_scalar::<i32>().unwrap();

    let shifted = copying::shift(&cv, 1, &fill).unwrap();
    let sv = shifted.testing_view().unwrap();
    assert_eq!(sv.len().unwrap(), 4);
    assert_eq!(sv.null_count().unwrap(), 1);

    let host = sv.to_optional_vec::<i32>().unwrap();
    assert_eq!(host, vec![None, Some(10), Some(20), Some(30)]);
}

fn shift_backward_with_value_fill() {
    let col = column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let cv = col.testing_view().unwrap();
    let fill = scalar(-1i32).unwrap();

    let shifted = copying::shift(&cv, -2, &fill).unwrap();
    let sv = shifted.testing_view().unwrap();
    let host = sv.to_vec::<i32>().unwrap();
    assert_eq!(host, vec![30, 40, -1, -1]);
}

fn shift_zero_offset_is_identity() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let cv = col.testing_view().unwrap();
    let fill = null_scalar::<i32>().unwrap();

    let shifted = copying::shift(&cv, 0, &fill).unwrap();
    let sv = shifted.testing_view().unwrap();
    assert_eq!(sv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

fn shift_offset_exceeding_length_yields_all_fill() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let cv = col.testing_view().unwrap();
    let fill = scalar(99i32).unwrap();

    let shifted_forward = copying::shift(&cv, 5, &fill).unwrap();
    assert_eq!(shifted_forward.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![99, 99, 99]);

    let shifted_backward = copying::shift(&cv, -5, &fill).unwrap();
    assert_eq!(shifted_backward.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![99, 99, 99]);
}

fn shift_dtype_mismatch_rejects_with_structured_error() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let cv = col.testing_view().unwrap();
    let wrong_fill = scalar(1.5f64).unwrap();

    let err = copying::shift(&cv, 1, &wrong_fill).unwrap_err();
    assert_eq!(err.code(), cudf::core::error::ErrorCode::TypeMismatch);
    let dump = format!("{err:?}");
    assert!(dump.contains("shift_fill_value"), "missing resource fact: {dump}");
    assert!(dump.contains("dtype_mismatch"), "missing reason: {dump}");
}

#[test]
fn shift_tests_cases() {
    crate::common::run_named_cases(&[
        ("shift_forward_with_null_fill", shift_forward_with_null_fill),
        ("shift_backward_with_value_fill", shift_backward_with_value_fill),
        ("shift_zero_offset_is_identity", shift_zero_offset_is_identity),
        ("shift_offset_exceeding_length_yields_all_fill", shift_offset_exceeding_length_yields_all_fill),
        ("shift_dtype_mismatch_rejects_with_structured_error", shift_dtype_mismatch_rejects_with_structured_error),
    ]);
}
