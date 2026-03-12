//! Tests for DataFrame column-level convenience.

use cudf::column::Column;
use cudf::dataframe::DataFrame;
use cudf::error::ErrorKind;
use cudf::scalar::Scalar;
use cudf::types::{DataType, TypeId};

// ---------------------------------------------------------------------------
// Column-level convenience (with_column, replace_nulls, cast_column)
// ---------------------------------------------------------------------------

#[test]
fn test_with_column_adds_new_column() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    let new_col = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let result = df.with_column("b", new_col).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows().unwrap(), 3);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // Original column preserved.
    let a_vals: Vec<i32> = result.column("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    // New column present.
    let b_vals: Vec<f64> = result.column("b").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_with_column_replaces_existing_column() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![
            Column::from_slice(&[1i32, 2]).unwrap(),
            Column::from_slice(&[10i32, 20]).unwrap(),
        ],
    )
    .unwrap();

    let replacement = Column::from_slice(&[99i32, 100]).unwrap();
    let result = df.with_column("b", replacement).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // "a" unchanged.
    let a_vals: Vec<i32> = result.column("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);

    // "b" replaced.
    let b_vals: Vec<i32> = result.column("b").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(b_vals, vec![99, 100]);
}

#[test]
fn test_with_column_preserves_schema_order_on_replace() {
    let df = DataFrame::new(
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        vec![
            Column::from_slice(&[1i32]).unwrap(),
            Column::from_slice(&[2i32]).unwrap(),
            Column::from_slice(&[3i32]).unwrap(),
        ],
    )
    .unwrap();

    // Replace middle column.
    let replacement = Column::from_slice(&[42i32]).unwrap();
    let result = df.with_column("y", replacement).unwrap();

    assert_eq!(result.column_names(), vec!["x", "y", "z"]);

    let y_val: Vec<i32> = result.column("y").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(y_val, vec![42]);
}

#[test]
fn test_with_column_mismatched_row_count_returns_error() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    // New column has 2 rows, DataFrame has 3.
    let new_col = Column::from_slice(&[10i32, 20]).unwrap();
    let err = df
        .with_column("b", new_col)
        .err()
        .expect("expected row count mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_replace_nulls_fills_values() {
    let col_a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let col_b = Column::from_slice_nullable(&[10i32, 0, 30], &[true, false, true]).unwrap();

    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![col_a, col_b],
    )
    .unwrap();

    let fill = Scalar::from_value(99i32).unwrap();
    let result = df.replace_nulls("b", &fill).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // "a" unchanged.
    let a_vals: Vec<i32> = result.column("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    // "b" has nulls replaced.
    let b_vals: Vec<i32> = result.column("b").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(b_vals, vec![10, 99, 30]);
}

#[test]
fn test_replace_nulls_nonexistent_column_returns_error() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice(&[1i32]).unwrap()],
    )
    .unwrap();

    let fill = Scalar::from_value(0i32).unwrap();
    let err = df
        .replace_nulls("missing", &fill)
        .err()
        .expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_replace_nulls_preserves_other_columns() {
    let col_a = Column::from_slice(&[1i32, 2]).unwrap();
    let col_b = Column::from_slice_nullable(&[10i32, 0], &[true, false]).unwrap();
    let col_c = Column::from_slice(&[100i64, 200]).unwrap();

    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![col_a, col_b, col_c],
    )
    .unwrap();

    let fill = Scalar::from_value(99i32).unwrap();
    let result = df.replace_nulls("b", &fill).unwrap();

    assert_eq!(result.column_names(), vec!["a", "b", "c"]);

    let a_vals: Vec<i32> = result.column("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);

    let c_vals: Vec<i64> = result.column("c").unwrap().to_vec::<i64>().unwrap();
    assert_eq!(c_vals, vec![100, 200]);
}

#[test]
fn test_cast_column_changes_type() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![
            Column::from_slice(&[1i32, 2, 3]).unwrap(),
            Column::from_slice(&[10i32, 20, 30]).unwrap(),
        ],
    )
    .unwrap();

    let target = DataType::new(TypeId::Float64);
    let result = df.cast_column("b", &target).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // "a" unchanged, still i32.
    let a_vals: Vec<i32> = result.column("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    // "b" cast to f64.
    let b_type = result.column("b").unwrap().data_type().unwrap();
    assert_eq!(b_type.id(), TypeId::Float64);
    let b_vals: Vec<f64> = result.column("b").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_cast_column_nonexistent_returns_error() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice(&[1i32]).unwrap()],
    )
    .unwrap();

    let target = DataType::new(TypeId::Float64);
    let err = df
        .cast_column("missing", &target)
        .err()
        .expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_cast_column_preserves_schema_order() {
    let df = DataFrame::new(
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        vec![
            Column::from_slice(&[1i32]).unwrap(),
            Column::from_slice(&[2i32]).unwrap(),
            Column::from_slice(&[3i32]).unwrap(),
        ],
    )
    .unwrap();

    let target = DataType::new(TypeId::Int64);
    let result = df.cast_column("y", &target).unwrap();

    assert_eq!(result.column_names(), vec!["x", "y", "z"]);

    // y is now Int64.
    let y_type = result.column("y").unwrap().data_type().unwrap();
    assert_eq!(y_type.id(), TypeId::Int64);
    let y_vals: Vec<i64> = result.column("y").unwrap().to_vec::<i64>().unwrap();
    assert_eq!(y_vals, vec![2]);
}
