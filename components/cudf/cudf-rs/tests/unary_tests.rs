//! Tests for unary operations.

use cudf::column::Column;
use cudf::types::{DataType, TypeId};
use cudf::unary::{self, UnaryOperator};

#[test]
fn test_sqrt() {
    let col = Column::from_slice(&[4.0f64, 9.0, 16.0]).unwrap();
    let view = col.view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Sqrt).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_abs() {
    let col = Column::from_slice(&[-3i32, 0, 5]).unwrap();
    let view = col.view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Abs).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![3, 0, 5]);
}

#[test]
fn test_negate() {
    let col = Column::from_slice(&[1i32, -2, 3]).unwrap();
    let view = col.view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Negate).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![-1, 2, -3]);
}

#[test]
fn test_is_null() {
    let col = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let view = col.view().unwrap();

    let result = unary::is_null(&view).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![false, true, false]);
}

#[test]
fn test_is_valid() {
    let col = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let view = col.view().unwrap();

    let result = unary::is_valid(&view).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, true]);
}

#[test]
fn test_cast() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();

    let target = DataType::new(TypeId::Float64);
    let result = unary::cast(&view, &target).unwrap();
    let rv = result.view().unwrap();
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
    let col = Column::from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap();
    let view = col.view().unwrap();

    let result = unary::is_nan(&view).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![false, true, false]);
}

#[test]
fn test_is_not_nan() {
    let col = Column::from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap();
    let view = col.view().unwrap();

    let result = unary::is_not_nan(&view).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, true]);
}

#[test]
fn test_trig_sin() {
    let col = Column::from_slice(&[0.0f64]).unwrap();
    let view = col.view().unwrap();

    let result = unary::unary_operation(&view, UnaryOperator::Sin).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<f64>().unwrap();
    assert!((values[0] - 0.0).abs() < 1e-10);
}
