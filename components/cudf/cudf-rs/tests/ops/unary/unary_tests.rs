//! Tests for unary operations.

use cudf::{
    DataType, TypeId,
    ops::unary::{self, UnaryOperator},
    testing::prelude::*,
};

use crate::common::{column_from_slice, column_from_slice_nullable};

#[test]
fn test_sqrt() {
    let col = column_from_slice(&[4.0f64, 9.0, 16.0]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Sqrt).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_abs() {
    let col = column_from_slice(&[-3i32, 0, 5]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Abs).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![3, 0, 5]);
}

#[test]
fn test_negate() {
    let col = column_from_slice(&[1i32, -2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Negate).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![-1, 2, -3]);
}

#[test]
fn test_is_null() {
    let col = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::is_null(&view).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![false, true, false]);
}

#[test]
fn test_is_valid() {
    let col = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::is_valid(&view).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, true]);
}

#[test]
fn test_cast() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let target = DataType::new(TypeId::Float64);
    let result = unary::cast(&view, &target).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Float64);
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_is_supported_cast() {
    let from = DataType::new(TypeId::Int32);
    let to = DataType::new(TypeId::Float64);
    assert!(unary::is_supported_cast(&from, &to).unwrap());
}

#[test]
fn test_is_nan() {
    let col = column_from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::is_nan(&view).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![false, true, false]);
}

#[test]
fn test_is_not_nan() {
    let col = column_from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::is_not_nan(&view).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, true]);
}

#[test]
fn test_cast_error_contains_operation_context() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let unsupported = DataType::new(TypeId::Struct);
    let err = unary::cast(&view, &unsupported).expect_err("expected error");

    assert_eq!(err.operation(), "unary::cast");
    assert!(err.context().iter().any(|(k, _)| *k == "output_type_id"));
}

#[test]
fn test_trig_sin() {
    let col = column_from_slice(&[0.0f64]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Sin).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<f64>().unwrap();
    assert!((values[0] - 0.0).abs() < 1e-10);
}

// --- logical_not tests ---

#[test]
fn test_logical_not_basic() {
    let col = column_from_slice(&[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::logical_not(&view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_logical_not_with_nulls() {
    let col = column_from_slice_nullable(&[true, false, true], &[true, true, false]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::logical_not(&view).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert!(!values[0]);
    assert!(values[1]);
    // Index 2 is null — value is undefined, just check null mask
    let validity = rv.to_host_validity().unwrap().unwrap();
    assert!(validity[0]);
    assert!(validity[1]);
    assert!(!validity[2]);
}

#[test]
fn test_logical_not_empty() {
    let col = column_from_slice::<bool>(&[]).unwrap();
    let view = col.testing_view().unwrap();

    let result = unary::logical_not(&view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.len().unwrap(), 0);
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
}
