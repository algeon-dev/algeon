//! Tests for DataFrame ↔ Arrow RecordBatch interop (feature-gated behind "arrow").

#![cfg(feature = "arrow")]

use arrow::array::*;
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cudf::arrow::dataframe_from_record_batch;
use cudf::column::Column;
use cudf::dataframe::DataFrame;
use cudf::types::TypeId;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Round-trip: DataFrame → RecordBatch → DataFrame
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_i32() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.schema().field(0).name(), "a");
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Int32);

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["a"]);
    let vals: Vec<i32> = df2.column("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_round_trip_i64() {
    let df = DataFrame::new(
        vec!["x".to_string()],
        vec![Column::from_slice(&[100i64, 200, 300]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Int64);

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let vals: Vec<i64> = df2.column("x").unwrap().to_vec::<i64>().unwrap();
    assert_eq!(vals, vec![100, 200, 300]);
}

#[test]
fn test_round_trip_f32_f64() {
    let df = DataFrame::new(
        vec!["f32_col".to_string(), "f64_col".to_string()],
        vec![
            Column::from_slice(&[1.5f32, 2.5, 3.5]).unwrap(),
            Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap(),
        ],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Float32);
    assert_eq!(*batch.schema().field(1).data_type(), ArrowDataType::Float64);

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let f32_vals: Vec<f32> = df2.column("f32_col").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(f32_vals, vec![1.5, 2.5, 3.5]);

    let f64_vals: Vec<f64> = df2.column("f64_col").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(f64_vals, vec![10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// Nullable columns: verify null mask round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_nullable_i32() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
    assert!(arr.is_valid(2));
    assert_eq!(arr.value(0), 1);
    assert_eq!(arr.value(2), 3);

    // Round-trip back.
    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let cv = df2.column("a").unwrap();
    let validity = cv.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    let vals: Vec<i32> = cv.to_vec::<i32>().unwrap();
    assert_eq!(vals[0], 1);
    assert_eq!(vals[2], 3);
}

#[test]
fn test_round_trip_nullable_f64() {
    let df = DataFrame::new(
        vec!["v".to_string()],
        vec![
            Column::from_slice_nullable(&[1.1f64, 0.0, 3.3, 0.0], &[true, false, true, false])
                .unwrap(),
        ],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
    assert!(arr.is_valid(2));
    assert!(!arr.is_valid(3));

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let cv = df2.column("v").unwrap();
    let validity = cv.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true, false]);

    let vals: Vec<f64> = cv.to_vec::<f64>().unwrap();
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[2] - 3.3).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Bool8 columns
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_bool8() {
    let df = DataFrame::new(
        vec!["flag".to_string()],
        vec![Column::from_slice(&[true, false, true]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(
        *batch.schema().field(0).data_type(),
        ArrowDataType::Boolean
    );

    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(arr.value(0), true);
    assert_eq!(arr.value(1), false);
    assert_eq!(arr.value(2), true);

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let vals: Vec<bool> = df2.column("flag").unwrap().to_vec::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

#[test]
fn test_round_trip_nullable_bool8() {
    let df = DataFrame::new(
        vec!["flag".to_string()],
        vec![
            Column::from_slice_nullable(&[true, false, true], &[true, false, true]).unwrap(),
        ],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
    assert!(arr.is_valid(2));
    assert_eq!(arr.value(0), true);
    assert_eq!(arr.value(2), true);

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let cv = df2.column("flag").unwrap();
    let validity = cv.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);
}

// ---------------------------------------------------------------------------
// Schema (column names) preserved
// ---------------------------------------------------------------------------

#[test]
fn test_schema_preserved_multi_column() {
    let df = DataFrame::new(
        vec![
            "id".to_string(),
            "score".to_string(),
            "active".to_string(),
        ],
        vec![
            Column::from_slice(&[1i32, 2]).unwrap(),
            Column::from_slice(&[99.5f64, 88.0]).unwrap(),
            Column::from_slice(&[true, false]).unwrap(),
        ],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.schema().fields().len(), 3);
    assert_eq!(batch.schema().field(0).name(), "id");
    assert_eq!(batch.schema().field(1).name(), "score");
    assert_eq!(batch.schema().field(2).name(), "active");

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["id", "score", "active"]);

    // Verify types survived.
    assert_eq!(df2.column("id").unwrap().data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(df2.column("score").unwrap().data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(df2.column("active").unwrap().data_type().unwrap().id(), TypeId::Bool8);
}

// ---------------------------------------------------------------------------
// From RecordBatch (Arrow → DataFrame)
// ---------------------------------------------------------------------------

#[test]
fn test_from_record_batch_creates_dataframe() {
    let schema = Schema::new(vec![
        Field::new("x", ArrowDataType::Int32, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
        ],
    )
    .unwrap();

    let df = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows().unwrap(), 3);
    assert_eq!(df.column_names(), vec!["x", "y"]);

    let x_vals: Vec<i32> = df.column("x").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(x_vals, vec![10, 20, 30]);

    let y_vals: Vec<f64> = df.column("y").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(y_vals, vec![1.1, 2.2, 3.3]);
}

#[test]
fn test_from_record_batch_with_nulls() {
    let schema = Schema::new(vec![Field::new("a", ArrowDataType::Int32, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
        ]))],
    )
    .unwrap();

    let df = dataframe_from_record_batch(&batch).unwrap();
    let cv = df.column("a").unwrap();
    let validity = cv.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    let vals: Vec<i32> = cv.to_vec::<i32>().unwrap();
    assert_eq!(vals[0], 1);
    assert_eq!(vals[2], 3);
}
