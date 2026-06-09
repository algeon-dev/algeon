//! Tests for DataFrame column-level convenience.

use cudf::{
    DataType, ErrorKind, TypeId,
    core::{Column, Scalar},
};

use crate::common::assert_missing_column_error;

// ---------------------------------------------------------------------------
// Column-level convenience (with_column, fillna_with_scalar, cast_column)
// ---------------------------------------------------------------------------

#[test]
fn test_with_column_adds_new_column() {
    let df =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()])
            .unwrap();

    let new_col = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let result = df.with_column("b", new_col).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // Original column preserved.
    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    // New column present.
    let b_vals: Vec<f64> = result.series("b").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_with_column_replaces_existing_column() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let replacement = Column::from_slice(&[99i32, 100]).unwrap();
    let result = df.with_column("b", replacement).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // "a" unchanged.
    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);

    // "b" replaced.
    let b_vals: Vec<i32> = result.series("b").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(b_vals, vec![99, 100]);
}

#[test]
fn test_with_column_preserves_schema_order_on_replace() {
    let df = crate::common::dataframe_from_columns(
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

    let y_val: Vec<i32> = result.series("y").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(y_val, vec![42]);
}

#[test]
fn test_with_column_mismatched_row_count_returns_error() {
    let df =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()])
            .unwrap();

    // New column has 2 rows, DataFrame has 3.
    let new_col = Column::from_slice(&[10i32, 20]).unwrap();
    let err = df.with_column("b", new_col).expect_err("expected row count mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_fillna_with_scalar_fills_values() {
    let col_a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let col_b = Column::from_slice_nullable(&[10i32, 0, 30], &[true, false, true]).unwrap();

    let df = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![col_a, col_b]).unwrap();

    let fill = Scalar::from_value(99i32).unwrap();
    let result = df.fillna_with_scalar("b", &fill).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // "a" unchanged.
    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    // "b" has nulls replaced.
    let b_vals: Vec<i32> = result.series("b").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(b_vals, vec![10, 99, 30]);
}

#[test]
fn test_fillna_with_scalar_nonexistent_column_returns_error() {
    let df = crate::common::dataframe_from_columns(vec!["a".to_string()], vec![Column::from_slice(&[1i32]).unwrap()])
        .unwrap();

    let fill = Scalar::from_value(0i32).unwrap();
    let err = df.fillna_with_scalar("missing", &fill).expect_err("expected error");
    assert_missing_column_error(&err, "missing");
}

#[test]
fn test_fillna_with_scalar_preserves_other_columns() {
    let col_a = Column::from_slice(&[1i32, 2]).unwrap();
    let col_b = Column::from_slice_nullable(&[10i32, 0], &[true, false]).unwrap();
    let col_c = Column::from_slice(&[100i64, 200]).unwrap();

    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![col_a, col_b, col_c],
    )
    .unwrap();

    let fill = Scalar::from_value(99i32).unwrap();
    let result = df.fillna_with_scalar("b", &fill).unwrap();

    assert_eq!(result.column_names(), vec!["a", "b", "c"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);

    let c_vals: Vec<i64> = result.series("c").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(c_vals, vec![100, 200]);
}

#[test]
fn test_cast_column_changes_type() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();

    let target = DataType::new(TypeId::Float64);
    let result = df.cast_column("b", target).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    // "a" unchanged, still i32.
    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    // "b" cast to f64.
    let b_type = result.series("b").unwrap().dtype().unwrap();
    assert_eq!(b_type.id(), TypeId::Float64);
    let b_vals: Vec<f64> = result.series("b").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_cast_column_nonexistent_returns_error() {
    let df = crate::common::dataframe_from_columns(vec!["a".to_string()], vec![Column::from_slice(&[1i32]).unwrap()])
        .unwrap();

    let target = DataType::new(TypeId::Float64);
    let err = df.cast_column("missing", target).expect_err("expected error");
    assert_missing_column_error(&err, "missing");
}

#[test]
fn test_cast_column_preserves_schema_order() {
    let df = crate::common::dataframe_from_columns(
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        vec![
            Column::from_slice(&[1i32]).unwrap(),
            Column::from_slice(&[2i32]).unwrap(),
            Column::from_slice(&[3i32]).unwrap(),
        ],
    )
    .unwrap();

    let target = DataType::new(TypeId::Int64);
    let result = df.cast_column("y", target).unwrap();

    assert_eq!(result.column_names(), vec!["x", "y", "z"]);

    // y is now Int64.
    let y_type = result.series("y").unwrap().dtype().unwrap();
    assert_eq!(y_type.id(), TypeId::Int64);
    let y_vals: Vec<i64> = result.series("y").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(y_vals, vec![2]);
}

#[test]
fn test_with_column_add_preserves_existing_string_column() {
    let df = crate::common::dataframe_from_columns(
        vec!["name".to_string(), "a".to_string()],
        vec![Column::from_strs(&["alice", "bob", "cara"]).unwrap(), Column::from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    let result = df.with_column("b", Column::from_slice(&[10i64, 20, 30]).unwrap()).unwrap();

    assert_eq!(result.column_names(), vec!["name", "a", "b"]);

    let names = result.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(names, vec!["alice", "bob", "cara"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    let b_vals: Vec<i64> = result.series("b").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(b_vals, vec![10, 20, 30]);
}

#[test]
fn test_fillna_with_scalar_with_string_sibling_column() {
    let df = crate::common::dataframe_from_columns(
        vec!["name".to_string(), "b".to_string()],
        vec![
            Column::from_strs(&["x", "y", "z"]).unwrap(),
            Column::from_slice_nullable(&[10i32, 0, 30], &[true, false, true]).unwrap(),
        ],
    )
    .unwrap();

    let fill = Scalar::from_value(99i32).unwrap();
    let result = df.fillna_with_scalar("b", &fill).unwrap();

    let names = result.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(names, vec!["x", "y", "z"]);

    let b_vals: Vec<i32> = result.series("b").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(b_vals, vec![10, 99, 30]);
}

#[test]
fn test_cast_column_with_string_sibling_column() {
    let df = crate::common::dataframe_from_columns(
        vec!["name".to_string(), "qty".to_string()],
        vec![Column::from_strs(&["a", "b", "c"]).unwrap(), Column::from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    let result = df.cast_column("qty", DataType::new(TypeId::Float64)).unwrap();

    let names = result.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(names, vec!["a", "b", "c"]);

    let qty_type = result.series("qty").unwrap().dtype().unwrap();
    assert_eq!(qty_type.id(), TypeId::Float64);

    let qty_vals: Vec<f64> = result.series("qty").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(qty_vals, vec![1.0, 2.0, 3.0]);
}
