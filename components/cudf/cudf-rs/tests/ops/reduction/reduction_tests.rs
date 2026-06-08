//! Tests for reduction and scan operations.

use cudf::{
    DataType, Series, TypeId,
    core::Column,
    ops::reduction::{self, NullPolicy, ReduceAgg, ScanAgg, ScanType},
};

#[test]
fn test_reduce_sum() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(&view, &ReduceAgg::Sum, &DataType::new(TypeId::Int64)).unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i64>().unwrap(), 10);
}

#[test]
fn test_reduce_min() {
    let col = Column::from_slice(&[5i32, 1, 3]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(&view, &ReduceAgg::Min, &DataType::new(TypeId::Int32)).unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i32>().unwrap(), 1);
}

#[test]
fn test_reduce_max() {
    let col = Column::from_slice(&[5i32, 1, 3]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(&view, &ReduceAgg::Max, &DataType::new(TypeId::Int32)).unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i32>().unwrap(), 5);
}

#[test]
fn test_reduce_mean() {
    let col = Column::from_slice(&[2.0f64, 4.0, 6.0]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(&view, &ReduceAgg::Mean, &DataType::new(TypeId::Float64)).unwrap();
    assert!(result.is_valid().unwrap());
    assert!((result.value::<f64>().unwrap() - 4.0).abs() < 1e-10);
}

#[test]
fn test_reduce_product() {
    let col = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::reduce(&view, &ReduceAgg::Product, &DataType::new(TypeId::Int64)).unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i64>().unwrap(), 24);
}

#[test]
fn test_scan_inclusive_sum() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::scan(&view, ScanAgg::Sum, ScanType::Inclusive, NullPolicy::Exclude).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![1, 3, 6, 10]);
}

#[test]
fn test_scan_exclusive_sum() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let view = col.view().unwrap();

    let result = reduction::scan(&view, ScanAgg::Sum, ScanType::Exclusive, NullPolicy::Exclude).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![0, 1, 3, 6]);
}

#[test]
fn test_reduce_ddof_above_i32_max_contains_context() {
    let col = Column::from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
    let view = col.view().unwrap();

    let err = reduction::reduce(&view, &ReduceAgg::Variance { ddof: u32::MAX }, &DataType::new(TypeId::Float64))
        .err()
        .expect("expected error");

    assert_eq!(err.operation(), "reduction::create_reduce_agg_handle(variance)");
    let ctx = err.context();
    assert!(ctx.iter().any(|(k, _)| *k == "ddof"));
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

#[test]
fn test_reduce_decimal_scalar_dtype_preserves_scale() {
    let dt = DataType::fixed_point(TypeId::Decimal128, -2);
    let series = Series::new::<i32>("x", &[100, 200, 300]).unwrap().astype(dt).unwrap();
    let view = series.as_column_view().unwrap();

    let result = reduction::reduce(&view, &ReduceAgg::Min, &dt).unwrap();

    assert_eq!(result.data_type().unwrap(), dt);
}
