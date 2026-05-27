//! Tests for binary operations.

use cudf::binary::{self, BinaryOperator};
use cudf::column::Column;
use cudf::scalar::Scalar;
use cudf::types::{DataType, TypeId};

#[test]
fn test_add_column_column() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32))
            .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![11, 22, 33]);
}

#[test]
fn test_mul_column_scalar() {
    let col = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let scalar = Scalar::from_value(10i32).unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::Mul,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![20, 30, 40]);
}

#[test]
fn test_div_scalar_column() {
    let scalar = Scalar::from_value(100.0f64).unwrap();
    let col = Column::from_slice(&[2.0f64, 4.0, 5.0]).unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_sv(
        &scalar,
        &cv,
        BinaryOperator::Div,
        &DataType::new(TypeId::Float64),
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![50.0, 25.0, 20.0]);
}

#[test]
fn test_comparison() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[2i32, 2, 1]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::Less,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, false]);
}

#[test]
fn test_binary_operation_error_contains_operation_context() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20]).unwrap(); // mismatched length
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let err =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32))
            .err()
            .expect("expected error");

    assert_eq!(err.operation(), "binary::binary_operation_vv");
    assert!(err.context().iter().any(|(k, _)| *k == "operator"));
}

#[test]
fn test_logical_and() {
    let a = Column::from_slice(&[true, true, false, false]).unwrap();
    let b = Column::from_slice(&[true, false, true, false]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::LogicalAnd,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, false, false]);
}
