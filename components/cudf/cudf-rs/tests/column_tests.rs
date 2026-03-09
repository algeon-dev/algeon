//! Tests for Column and ColumnView.

use cudf::column::Column;
use cudf::types::{TypeId, DataType};

#[test]
fn test_create_empty_column() {
    let col = Column::empty(TypeId::Int32).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 0);
    assert!(view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap(), DataType::new(TypeId::Int32));
}

#[test]
fn test_create_i32_column() {
    let data = vec![1i32, 2, 3, 4, 5];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 5);
    assert!(!view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(view.null_count().unwrap(), 0);
    assert!(!view.nullable().unwrap());

    let host_data = view.to_vec::<i32>().unwrap();
    assert_eq!(host_data, data);
}

#[test]
fn test_create_f64_column() {
    let data = vec![1.5f64, 2.5, 3.5];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.data_type().unwrap().id(), TypeId::Float64);

    let host_data = view.to_vec::<f64>().unwrap();
    assert_eq!(host_data, data);
}

#[test]
fn test_nullable_column() {
    let data = vec![10i32, 20, 30, 40];
    let validity = vec![true, false, true, false];
    let col = Column::from_slice_nullable(&data, &validity).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 2);

    let host_data = view.to_vec::<i32>().unwrap();
    assert_eq!(host_data, data);

    let mask = view.validity().unwrap().unwrap();
    assert_eq!(mask, validity);
}

#[test]
fn test_column_copy() {
    let data = vec![100i64, 200, 300];
    let col = Column::from_slice(&data).unwrap();
    let copy = col.copy().unwrap();
    let view = copy.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i64>().unwrap(), data);
}

#[test]
fn test_type_mismatch_error() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();

    let result = view.to_vec::<f64>();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), cudf::ErrorKind::TypeError);
}

#[test]
fn test_u8_column() {
    let data = vec![0u8, 128, 255];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.to_vec::<u8>().unwrap(), data);
}

#[test]
fn test_mismatched_validity_length() {
    let data = vec![1i32, 2, 3];
    let validity = vec![true, false]; // wrong length
    let result = Column::from_slice_nullable(&data, &validity);
    assert!(result.is_err());
}

#[test]
fn test_various_numeric_types() {
    // i8
    let col = Column::from_slice(&[1i8, -1, 127]).unwrap();
    assert_eq!(col.view().unwrap().to_vec::<i8>().unwrap(), vec![1i8, -1, 127]);

    // i16
    let col = Column::from_slice(&[1i16, -1, 32767]).unwrap();
    assert_eq!(col.view().unwrap().to_vec::<i16>().unwrap(), vec![1i16, -1, 32767]);

    // u16
    let col = Column::from_slice(&[0u16, 1000, 65535]).unwrap();
    assert_eq!(col.view().unwrap().to_vec::<u16>().unwrap(), vec![0u16, 1000, 65535]);

    // u32
    let col = Column::from_slice(&[0u32, 42, u32::MAX]).unwrap();
    assert_eq!(col.view().unwrap().to_vec::<u32>().unwrap(), vec![0u32, 42, u32::MAX]);

    // u64
    let col = Column::from_slice(&[0u64, 99]).unwrap();
    assert_eq!(col.view().unwrap().to_vec::<u64>().unwrap(), vec![0u64, 99]);

    // f32
    let col = Column::from_slice(&[1.0f32, -0.5, 3.125]).unwrap();
    let vals = col.view().unwrap().to_vec::<f32>().unwrap();
    assert!((vals[2] - 3.125).abs() < 0.001);
}
