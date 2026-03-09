//! Tests for reduction and scan operations.

use cudf::column::Column;
use cudf::reduction::{self, NullPolicy, ReduceAggregation, ScanAggregation, ScanType};
use cudf::types::{DataType, TypeId};

#[test]
fn test_reduce_sum() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(
        &view,
        &ReduceAggregation::Sum,
        &DataType::new(TypeId::Int64),
    )
    .unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i64>().unwrap(), 10);
}

#[test]
fn test_reduce_min() {
    let col = Column::from_slice(&[5i32, 1, 3]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(
        &view,
        &ReduceAggregation::Min,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i32>().unwrap(), 1);
}

#[test]
fn test_reduce_max() {
    let col = Column::from_slice(&[5i32, 1, 3]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(
        &view,
        &ReduceAggregation::Max,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i32>().unwrap(), 5);
}

#[test]
fn test_reduce_mean() {
    let col = Column::from_slice(&[2.0f64, 4.0, 6.0]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(
        &view,
        &ReduceAggregation::Mean,
        &DataType::new(TypeId::Float64),
    )
    .unwrap();
    assert!(result.is_valid().unwrap());
    assert!((result.value::<f64>().unwrap() - 4.0).abs() < 1e-10);
}

#[test]
fn test_reduce_product() {
    let col = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(
        &view,
        &ReduceAggregation::Product,
        &DataType::new(TypeId::Int64),
    )
    .unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i64>().unwrap(), 24);
}

#[test]
fn test_scan_inclusive_sum() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::scan(
        &view,
        ScanAggregation::Sum,
        ScanType::Inclusive,
        NullPolicy::Exclude,
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![1, 3, 6, 10]);
}

#[test]
fn test_scan_exclusive_sum() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::scan(
        &view,
        ScanAggregation::Sum,
        ScanType::Exclusive,
        NullPolicy::Exclude,
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![0, 1, 3, 6]);
}

#[test]
fn test_minmax() {
    let col = Column::from_slice(&[5i32, 1, 3, 7, 2]).unwrap();
    let view = col.view().unwrap();

    let (min_scalar, max_scalar) = reduction::minmax(&view).unwrap();
    assert!(min_scalar.is_valid().unwrap());
    assert!(max_scalar.is_valid().unwrap());
    assert_eq!(min_scalar.value::<i32>().unwrap(), 1);
    assert_eq!(max_scalar.value::<i32>().unwrap(), 7);
}
