#![allow(deprecated)] // Tests exercise deprecated DataFrame::column()
//! Tests for DataFrame ↔ Arrow RecordBatch interop (feature-gated behind "arrow").

#![cfg(feature = "arrow")]

use std::sync::Arc;

use arrow::{
    array::*,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{arrow::dataframe_from_record_batch, column::Column, dataframe::DataFrame, types::TypeId};

// ---------------------------------------------------------------------------
// Round-trip: DataFrame → RecordBatch → DataFrame
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_i32() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

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
    let df = DataFrame::new(vec!["x".to_string()], vec![Column::from_slice(&[100i64, 200, 300]).unwrap()]).unwrap();

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
        vec![Column::from_slice(&[1.5f32, 2.5, 3.5]).unwrap(), Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap()],
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
    let arr = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
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
        vec![Column::from_slice_nullable(&[1.1f64, 0.0, 3.3, 0.0], &[true, false, true, false]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
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
    let df = DataFrame::new(vec!["flag".to_string()], vec![Column::from_slice(&[true, false, true]).unwrap()]).unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Boolean);

    let arr = batch.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
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
        vec![Column::from_slice_nullable(&[true, false, true], &[true, false, true]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
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
        vec!["id".to_string(), "score".to_string(), "active".to_string()],
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
    let schema =
        Schema::new(vec![Field::new("x", ArrowDataType::Int32, false), Field::new("y", ArrowDataType::Float64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30])), Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3]))],
    )
    .unwrap();

    let df = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["x", "y"]);

    let x_vals: Vec<i32> = df.column("x").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(x_vals, vec![10, 20, 30]);

    let y_vals: Vec<f64> = df.column("y").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(y_vals, vec![1.1, 2.2, 3.3]);
}

#[test]
fn test_from_record_batch_with_nulls() {
    let schema = Schema::new(vec![Field::new("a", ArrowDataType::Int32, true)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))]).unwrap();

    let df = dataframe_from_record_batch(&batch).unwrap();
    let cv = df.column("a").unwrap();
    let validity = cv.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    let vals: Vec<i32> = cv.to_vec::<i32>().unwrap();
    assert_eq!(vals[0], 1);
    assert_eq!(vals[2], 3);
}

// ---------------------------------------------------------------------------
// String columns: round-trip via Arrow StringArray
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_string_only() {
    let df =
        DataFrame::new(vec!["name".to_string()], vec![Column::from_strings(&["alice", "bob", "charlie"]).unwrap()])
            .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Utf8);

    let arr = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.value(0), "alice");
    assert_eq!(arr.value(1), "bob");
    assert_eq!(arr.value(2), "charlie");

    // Round-trip back to GPU DataFrame.
    let df2 = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["name"]);
    assert_eq!(df2.column("name").unwrap().data_type().unwrap().id(), TypeId::String);
    let strs = df2.column("name").unwrap().to_strings().unwrap();
    assert_eq!(strs, vec!["alice", "bob", "charlie"]);
}

#[test]
fn test_round_trip_mixed_string_and_numeric() {
    let df = DataFrame::new(
        vec!["city".to_string(), "pop".to_string()],
        vec![
            Column::from_strings(&["tokyo", "london", "paris"]).unwrap(),
            Column::from_slice(&[14_000_000i64, 9_000_000, 2_000_000]).unwrap(),
        ],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Utf8);
    assert_eq!(*batch.schema().field(1).data_type(), ArrowDataType::Int64);

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["city", "pop"]);

    let cities = df2.column("city").unwrap().to_strings().unwrap();
    assert_eq!(cities, vec!["tokyo", "london", "paris"]);

    let pops: Vec<i64> = df2.column("pop").unwrap().to_vec::<i64>().unwrap();
    assert_eq!(pops, vec![14_000_000, 9_000_000, 2_000_000]);
}

#[test]
fn test_round_trip_nullable_string() {
    let df = DataFrame::new(
        vec!["tag".to_string()],
        vec![Column::from_strings_nullable(&["hello", "", "world"], &[true, false, true]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
    assert!(arr.is_valid(2));
    assert_eq!(arr.value(0), "hello");
    assert_eq!(arr.value(2), "world");

    // Round-trip back.
    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let cv = df2.column("tag").unwrap();
    let validity = cv.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    let strs = cv.to_strings().unwrap();
    assert_eq!(strs[0], "hello");
    assert_eq!(strs[2], "world");
}

#[test]
fn test_round_trip_string_with_empty_values() {
    let df = DataFrame::new(vec!["s".to_string()], vec![Column::from_strings(&["", "data", ""]).unwrap()]).unwrap();

    let batch = df.to_record_batch().unwrap();
    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let strs = df2.column("s").unwrap().to_strings().unwrap();
    assert_eq!(strs, vec!["", "data", ""]);
}

#[test]
fn test_round_trip_string_unicode() {
    let df =
        DataFrame::new(vec!["text".to_string()], vec![Column::from_strings(&["héllo", "日本語", "🚀🎉"]).unwrap()])
            .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.value(0), "héllo");
    assert_eq!(arr.value(1), "日本語");
    assert_eq!(arr.value(2), "🚀🎉");

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    let strs = df2.column("text").unwrap().to_strings().unwrap();
    assert_eq!(strs, vec!["héllo", "日本語", "🚀🎉"]);
}

#[test]
fn test_from_record_batch_string_array() {
    let schema = Schema::new(vec![Field::new("name", ArrowDataType::Utf8, false)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"]))])
            .unwrap();

    let df = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column("name").unwrap().data_type().unwrap().id(), TypeId::String);
    let strs = df.column("name").unwrap().to_strings().unwrap();
    assert_eq!(strs, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn test_from_record_batch_nullable_string_array() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::Utf8, true)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]))])
            .unwrap();

    let df = dataframe_from_record_batch(&batch).unwrap();
    let cv = df.column("s").unwrap();
    let validity = cv.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    let strs = cv.to_strings().unwrap();
    assert_eq!(strs[0], "a");
    assert_eq!(strs[2], "c");
}

#[test]
fn test_from_record_batch_large_string_array() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::LargeUtf8, false)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(LargeStringArray::from(vec!["x", "y", "z"]))]).unwrap();

    let df = dataframe_from_record_batch(&batch).unwrap();
    let strs = df.column("s").unwrap().to_strings().unwrap();
    assert_eq!(strs, vec!["x", "y", "z"]);
}

#[test]
fn test_round_trip_empty_string_schema() {
    let df = DataFrame::new(vec!["s".to_string()], vec![Column::from_strings(&[]).unwrap()]).unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Utf8);

    let arr = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.len(), 0);

    let df2 = dataframe_from_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["s"]);
    assert_eq!(df2.num_rows(), 0);
    assert_eq!(df2.column("s").unwrap().data_type().unwrap().id(), TypeId::String);
    let strs = df2.column("s").unwrap().to_strings().unwrap();
    assert!(strs.is_empty());
}
