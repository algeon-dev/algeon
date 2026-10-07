//! Tests for DataFrame::gather().

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{DataFrame, Exec, testing::prelude::*};

use crate::common::{
    column_from_slice, column_from_strs, dataframe_from_columns, run_named_cases, sample_runtime, series,
};

fn test_gather_sequential_indices() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = series::<i32>("idx", &[0, 1, 2]).unwrap();
    let result = df.on(Exec::RETAINED).gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_optional_vec.
    let vals = result.series("x").unwrap().to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), Some(20), Some(30)]);
}

fn test_gather_reversed() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = series::<i32>("idx", &[2, 1, 0]).unwrap();
    let result = df.on(Exec::RETAINED).gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_optional_vec.
    let vals = result.series("x").unwrap().to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(30), Some(20), Some(10)]);
}

fn test_gather_out_of_bounds_nullify() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = series::<i32>("idx", &[0, 99, 2]).unwrap();
    let result = df.on(Exec::RETAINED).gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    let view = result.series("x").unwrap();
    let validity = view.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert!(validity[0]);
    assert!(!validity[1]); // OOB → null
    assert!(validity[2]);
}

fn test_gather_duplicate_indices() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = series::<i32>("idx", &[1, 1, 1]).unwrap();
    let result = df.on(Exec::RETAINED).gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_optional_vec.
    let vals = result.series("x").unwrap().to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(20), Some(20), Some(20)]);
}

fn test_gather_preserves_schema() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let df = dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![c1, c2]).unwrap();

    let indices = series::<i32>("idx", &[2, 0]).unwrap();
    let result = df.on(Exec::RETAINED).gather(&indices).unwrap();

    assert_eq!(result.column_names(), &["id", "score"]);
    assert_eq!(result.num_rows(), 2);
    // Gather produces nullable output; use to_optional_vec.
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(3), Some(1)]);
    assert_eq!(result.series("score").unwrap().to_optional_vec::<f64>().unwrap(), vec![Some(30.0), Some(10.0)]);
}

fn test_gather_empty_indices() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = series::<i32>("idx", &[]).unwrap();
    let result = df.on(Exec::RETAINED).gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), &["x"]);

    let item = Arc::new(Field::new("element", ArrowDataType::Float32, false));
    let vectors =
        FixedSizeListArray::new(Arc::clone(&item), 2, Arc::new(Float32Array::from(vec![1.0, 0.0, 0.0, 1.0])), None);
    let schema = Arc::new(Schema::new(vec![Field::new("vector", ArrowDataType::FixedSizeList(item, 2), false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(vectors) as ArrayRef]).unwrap();
    let nested = DataFrame::from_record_batch(&sample_runtime(), &batch).unwrap();
    let source_type = nested.to_record_batch().unwrap().column(0).data_type().clone();
    let vector = nested.series("vector").unwrap();
    for empty in [
        nested.on(Exec::RETAINED).gather(&indices).unwrap(),
        nested.on(Exec::RETAINED).head(0).unwrap(),
        DataFrame::from_series([("vector", vector.on(Exec::RETAINED).gather(&indices).unwrap())]).unwrap(),
        DataFrame::from_series([("vector", vector.on(Exec::RETAINED).head(0).unwrap())]).unwrap(),
    ] {
        let exported = empty.to_record_batch().unwrap();
        assert_eq!(exported.num_rows(), 0);
        assert_eq!(exported.column(0).data_type(), &source_type);
    }
}

fn test_gather_multi_type() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let c3 = column_from_strs(&["a", "b", "c"]).unwrap();
    let df =
        dataframe_from_columns(vec!["ints".to_string(), "floats".to_string(), "strs".to_string()], vec![c1, c2, c3])
            .unwrap();

    let indices = series::<i32>("idx", &[2, 0]).unwrap();
    let result = df.on(Exec::RETAINED).gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 2);
    // Gather produces nullable output; use to_optional_vec / to_optional_strings.
    assert_eq!(result.series("ints").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(3), Some(1)]);
    assert_eq!(result.series("floats").unwrap().to_optional_vec::<f64>().unwrap(), vec![Some(3.5), Some(1.5)]);
    assert_eq!(
        result.series("strs").unwrap().to_optional_strings().unwrap(),
        vec![Some("c".to_string()), Some("a".to_string())]
    );
}

#[test]
fn gather_tests_cases() {
    run_named_cases(&[
        ("test_gather_sequential_indices", test_gather_sequential_indices),
        ("test_gather_reversed", test_gather_reversed),
        ("test_gather_out_of_bounds_nullify", test_gather_out_of_bounds_nullify),
        ("test_gather_duplicate_indices", test_gather_duplicate_indices),
        ("test_gather_preserves_schema", test_gather_preserves_schema),
        ("test_gather_empty_indices", test_gather_empty_indices),
        ("test_gather_multi_type", test_gather_multi_type),
    ]);
}
