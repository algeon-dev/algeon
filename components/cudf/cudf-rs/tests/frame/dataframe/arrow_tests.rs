//! Tests for DataFrame ↔ Arrow RecordBatch interop.
use std::sync::Arc;

use arrow::{
    array::*,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{DataFrame, ErrorCode, ErrorKind, TypeId, testing::prelude::*};

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, column_from_strs_nullable, error_context_value,
};

fn import_record_batch(batch: &RecordBatch) -> cudf::Result<DataFrame> {
    let runtime = crate::common::sample_runtime();
    DataFrame::from_record_batch(&runtime, batch)
}

// ---------------------------------------------------------------------------
// Round-trip: DataFrame → RecordBatch → DataFrame
// ---------------------------------------------------------------------------

#[test]
fn test_arrow_datatype_try_from_type_id_supported_mappings() {
    assert_eq!(ArrowDataType::try_from(TypeId::Int32).unwrap(), ArrowDataType::Int32);
    assert_eq!(ArrowDataType::try_from(TypeId::Float64).unwrap(), ArrowDataType::Float64);
    assert_eq!(ArrowDataType::try_from(TypeId::Bool8).unwrap(), ArrowDataType::Boolean);
    assert_eq!(ArrowDataType::try_from(TypeId::String).unwrap(), ArrowDataType::Utf8);
}

#[test]
fn test_arrow_datatype_try_from_type_id_rejects_unsupported_types() {
    let err =
        ArrowDataType::try_from(TypeId::List).expect_err("list columns are not supported by Arrow host-copy interop");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow::TryFrom<TypeId> for ArrowDataType");
    assert_eq!(error_context_value(&err, "type_id"), Some("list"));
    assert!(err.message().contains("unsupported cudf type"));
}

#[test]
fn test_type_id_try_from_arrow_datatype_supported_mappings() {
    assert_eq!(TypeId::try_from(&ArrowDataType::Int32).unwrap(), TypeId::Int32);
    assert_eq!(TypeId::try_from(&ArrowDataType::Float64).unwrap(), TypeId::Float64);
    assert_eq!(TypeId::try_from(&ArrowDataType::Boolean).unwrap(), TypeId::Bool8);
    assert_eq!(TypeId::try_from(&ArrowDataType::Utf8).unwrap(), TypeId::String);
    assert_eq!(TypeId::try_from(&ArrowDataType::LargeUtf8).unwrap(), TypeId::String);
}

#[test]
fn test_type_id_try_from_arrow_datatype_rejects_unsupported_types() {
    let err = TypeId::try_from(&ArrowDataType::LargeBinary)
        .expect_err("large binary arrays are not supported by cudf Arrow import");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow::TryFrom<&ArrowDataType> for TypeId");
    assert_eq!(error_context_value(&err, "arrow_type"), Some("LargeBinary"));
    assert!(err.message().contains("unsupported Arrow type"));
}

#[test]
fn test_arrow_numeric_wrapper_preserves_source_chain() {
    let err = cudf::testing::trigger_numeric_to_arrow_error_for_test()
        .expect_err("expected numeric Arrow wrapper to surface a mapped error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "arrow::numeric_to_arrow");
    assert_eq!(error_context_value(&err, "resource"), Some("arrow_array"));
    assert_eq!(error_context_value(&err, "reason"), Some("invalid_arrow_primitive_array"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some("ensure Arrow primitive values and null buffers have matching logical lengths")
    );

    let source = std::error::Error::source(&err).expect("expected preserved Arrow source");
    assert_eq!(source.to_string(), err.message());
}

#[test]
fn test_to_record_batch_wrapper_preserves_source_chain() {
    let err = cudf::testing::trigger_record_batch_error_for_test()
        .expect_err("expected RecordBatch wrapper to surface a mapped error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "dataframe::to_record_batch");
    assert_eq!(error_context_value(&err, "resource"), Some("record_batch"));
    assert_eq!(error_context_value(&err, "reason"), Some("invalid_arrow_record_batch"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some("ensure Arrow schemas and column arrays have matching row counts and compatible field layouts")
    );

    let source = std::error::Error::source(&err).expect("expected preserved Arrow source");
    assert_eq!(source.to_string(), err.message());
}

#[test]
fn test_arrow_string_array_downcast_mismatch_uses_stable_foreign_contract() {
    let err = cudf::testing::trigger_arrow_string_array_downcast_mismatch_for_test()
        .expect_err("expected Utf8 downcast mismatch to surface a mapped foreign-contract error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "arrow::string_array_to_column");
    assert_eq!(error_context_value(&err, "resource"), Some("arrow_array"));
    assert_eq!(error_context_value(&err, "reason"), Some("unexpected_arrow_array_layout"));
    assert_eq!(error_context_value(&err, "arrow_type"), Some("Utf8"));
    assert_eq!(error_context_value(&err, "actual_arrow_type"), Some("Int32"));
    assert_eq!(error_context_value(&err, "remedy"), Some("ensure Utf8 fields are backed by Arrow StringArray values"));
}

#[test]
fn test_arrow_large_string_array_downcast_mismatch_uses_same_foreign_contract_family() {
    let err = cudf::testing::trigger_arrow_large_string_array_downcast_mismatch_for_test()
        .expect_err("expected LargeUtf8 downcast mismatch to surface a mapped foreign-contract error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "arrow::large_string_array_to_column");
    assert_eq!(error_context_value(&err, "resource"), Some("arrow_array"));
    assert_eq!(error_context_value(&err, "reason"), Some("unexpected_arrow_array_layout"));
    assert_eq!(error_context_value(&err, "arrow_type"), Some("LargeUtf8"));
    assert_eq!(error_context_value(&err, "actual_arrow_type"), Some("Int32"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some("ensure LargeUtf8 fields are backed by Arrow LargeStringArray values")
    );
}

#[test]
fn test_from_record_batch_rejects_unsupported_arrow_array_type_with_stable_code() {
    let schema = Schema::new(vec![Field::new("payload", ArrowDataType::LargeBinary, false)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(LargeBinaryArray::from(vec![b"a".as_slice(), b"b"]))])
            .unwrap();

    let err = import_record_batch(&batch).expect_err("LargeBinary import should be rejected with a stable contract");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow::arrow_array_to_column");
    assert_eq!(error_context_value(&err, "field_name"), Some("payload"));
    assert_eq!(error_context_value(&err, "arrow_type"), Some("LargeBinary"));
}

#[test]
fn test_round_trip_i32() {
    let df =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2, 3]).unwrap()])
            .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Int32);

    let df2 = import_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["a"]);
    let vals: Vec<i32> = df2.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_round_trip_i64() {
    let df = crate::common::dataframe_from_columns(
        vec!["x".to_string()],
        vec![column_from_slice(&[100i64, 200, 300]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Int64);

    let df2 = import_record_batch(&batch).unwrap();
    let vals: Vec<i64> = df2.series("x").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![100, 200, 300]);
}

#[test]
fn test_round_trip_f32_f64() {
    let df = crate::common::dataframe_from_columns(
        vec!["f32_col".to_string(), "f64_col".to_string()],
        vec![column_from_slice(&[1.5f32, 2.5, 3.5]).unwrap(), column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Float32);
    assert_eq!(*batch.schema().field(1).data_type(), ArrowDataType::Float64);

    let df2 = import_record_batch(&batch).unwrap();
    let f32_vals: Vec<f32> = df2.series("f32_col").unwrap().to_vec_strict::<f32>().unwrap();
    assert_eq!(f32_vals, vec![1.5, 2.5, 3.5]);

    let f64_vals: Vec<f64> = df2.series("f64_col").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(f64_vals, vec![10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// Nullable columns: verify null mask round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_nullable_i32() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string()],
        vec![column_from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap()],
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
    let df2 = import_record_batch(&batch).unwrap();
    let cv = df2.series("a").unwrap();
    let validity = cv.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    // Nullable column: use to_optional_vec to access values alongside nulls.
    let vals = cv.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals[0], Some(1));
    assert_eq!(vals[1], None);
    assert_eq!(vals[2], Some(3));
}

#[test]
fn test_round_trip_nullable_f64() {
    let df = crate::common::dataframe_from_columns(
        vec!["v".to_string()],
        vec![column_from_slice_nullable(&[1.1f64, 0.0, 3.3, 0.0], &[true, false, true, false]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
    assert!(arr.is_valid(2));
    assert!(!arr.is_valid(3));

    let df2 = import_record_batch(&batch).unwrap();
    let cv = df2.series("v").unwrap();
    let validity = cv.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true, false]);

    // Nullable column: use to_optional_vec to access values alongside nulls.
    let vals = cv.to_optional_vec::<f64>().unwrap();
    assert!((vals[0].unwrap() - 1.1).abs() < 1e-10);
    assert_eq!(vals[1], None);
    assert!((vals[2].unwrap() - 3.3).abs() < 1e-10);
    assert_eq!(vals[3], None);
}

// ---------------------------------------------------------------------------
// Bool8 columns
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_bool8() {
    let df = crate::common::dataframe_from_columns(
        vec!["flag".to_string()],
        vec![column_from_slice(&[true, false, true]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Boolean);

    let arr = batch.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(arr.value(0));
    assert!(!arr.value(1));
    assert!(arr.value(2));

    let df2 = import_record_batch(&batch).unwrap();
    let vals: Vec<bool> = df2.series("flag").unwrap().to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

#[test]
fn test_round_trip_nullable_bool8() {
    let df = crate::common::dataframe_from_columns(
        vec!["flag".to_string()],
        vec![column_from_slice_nullable(&[true, false, true], &[true, false, true]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
    assert!(arr.is_valid(2));
    assert!(arr.value(0));
    assert!(arr.value(2));

    let df2 = import_record_batch(&batch).unwrap();
    let cv = df2.series("flag").unwrap();
    let validity = cv.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);
}

#[test]
fn test_to_record_batch_nullable_bool_from_comparison_preserves_nulls() {
    let lhs = crate::common::nullable_series("lhs", &[1i32, 0, 3], &[true, false, true]).unwrap();
    let rhs = crate::common::nullable_series("rhs", &[1i32, 2, 0], &[true, true, true]).unwrap();
    let flag = lhs.eq(&rhs).unwrap();
    let df = DataFrame::from_columns([("flag", flag)]).unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();

    assert_eq!(arr.len(), 3);
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
    assert!(arr.is_valid(2));
    assert!(arr.value(0));
    assert!(!arr.value(2));
}

// ---------------------------------------------------------------------------
// Schema (column names) preserved
// ---------------------------------------------------------------------------

#[test]
fn test_schema_preserved_multi_column() {
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string(), "active".to_string()],
        vec![
            column_from_slice(&[1i32, 2]).unwrap(),
            column_from_slice(&[99.5f64, 88.0]).unwrap(),
            column_from_slice(&[true, false]).unwrap(),
        ],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.schema().fields().len(), 3);

    let df2 = import_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["id", "score", "active"]);

    // Verify types survived.
    assert_eq!(df2.series("id").unwrap().dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(df2.series("score").unwrap().dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(df2.series("active").unwrap().dtype().unwrap().id(), TypeId::Bool8);
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

    let df = import_record_batch(&batch).unwrap();
    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["x", "y"]);

    let x_vals: Vec<i32> = df.series("x").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(x_vals, vec![10, 20, 30]);

    let y_vals: Vec<f64> = df.series("y").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(y_vals, vec![1.1, 2.2, 3.3]);
}

#[test]
fn test_from_record_batch_binds_caller_runtime_provenance() {
    let runtime = crate::common::sample_runtime();
    let expected_id = runtime.testing_execution_domain_id();
    let expected_stream = runtime.stream().testing_raw_handle_value();
    let expected_mr = runtime.memory_resource().testing_raw_handle_value();

    let schema =
        Schema::new(vec![Field::new("x", ArrowDataType::Int32, false), Field::new("y", ArrowDataType::Float64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30])), Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3]))],
    )
    .unwrap();

    let df = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), expected_id);
    assert_eq!(df.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(df.testing_execution_domain_mr_value(), expected_mr);
}

#[test]
fn test_from_record_batch_with_nulls() {
    let schema = Schema::new(vec![Field::new("a", ArrowDataType::Int32, true)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))]).unwrap();

    let df = import_record_batch(&batch).unwrap();
    let cv = df.series("a").unwrap();
    let validity = cv.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    // Nullable column: use to_optional_vec to access values alongside nulls.
    let vals = cv.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals[0], Some(1));
    assert_eq!(vals[1], None);
    assert_eq!(vals[2], Some(3));
}

#[test]
fn test_from_record_batch_primitive_slice_import_uses_logical_values() {
    let schema = Schema::new(vec![Field::new("a", ArrowDataType::Int32, false)]);
    let base: Arc<dyn Array> = Arc::new(Int32Array::from(vec![10, 20, 30, 40]));
    let sliced = base.slice(1, 2);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![sliced]).unwrap();

    let df = import_record_batch(&batch).unwrap();
    let vals: Vec<i32> = df.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![20, 30]);
}

#[test]
fn test_from_record_batch_nullable_primitive_slice_import_preserves_offset_nulls() {
    let schema = Schema::new(vec![Field::new("a", ArrowDataType::Int32, true)]);
    let base: Arc<dyn Array> = Arc::new(Int32Array::from(vec![Some(10), None, Some(30), Some(40)]));
    let sliced = base.slice(1, 2);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![sliced]).unwrap();

    let df = import_record_batch(&batch).unwrap();
    let series = df.series("a").unwrap();
    let validity = series.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![false, true]);

    let vals = series.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, Some(30)]);
}

// ---------------------------------------------------------------------------
// String columns: round-trip via Arrow StringArray
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_string_only() {
    let df = crate::common::dataframe_from_columns(
        vec!["name".to_string()],
        vec![column_from_strs(&["alice", "bob", "charlie"]).unwrap()],
    )
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
    let df2 = import_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["name"]);
    assert_eq!(df2.series("name").unwrap().dtype().unwrap().id(), TypeId::String);
    let strs = df2.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(strs, vec!["alice", "bob", "charlie"]);
}

#[test]
fn test_round_trip_mixed_string_and_numeric() {
    let df = crate::common::dataframe_from_columns(
        vec!["city".to_string(), "pop".to_string()],
        vec![
            column_from_strs(&["tokyo", "london", "paris"]).unwrap(),
            column_from_slice(&[14_000_000i64, 9_000_000, 2_000_000]).unwrap(),
        ],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Utf8);
    assert_eq!(*batch.schema().field(1).data_type(), ArrowDataType::Int64);

    let df2 = import_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["city", "pop"]);

    let cities = df2.series("city").unwrap().to_strings_strict().unwrap();
    assert_eq!(cities, vec!["tokyo", "london", "paris"]);

    let pops: Vec<i64> = df2.series("pop").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(pops, vec![14_000_000, 9_000_000, 2_000_000]);
}

#[test]
fn test_round_trip_nullable_string() {
    let df = crate::common::dataframe_from_columns(
        vec!["tag".to_string()],
        vec![column_from_strs_nullable(&["hello", "", "world"], &[true, false, true]).unwrap()],
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
    let df2 = import_record_batch(&batch).unwrap();
    let cv = df2.series("tag").unwrap();
    let validity = cv.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    // Nullable column: use to_optional_strings to access values alongside nulls.
    let strs = cv.to_optional_strings().unwrap();
    assert_eq!(strs[0], Some("hello".to_string()));
    assert_eq!(strs[1], None);
    assert_eq!(strs[2], Some("world".to_string()));
}

#[test]
fn test_round_trip_string_with_empty_values() {
    let df = crate::common::dataframe_from_columns(
        vec!["s".to_string()],
        vec![column_from_strs(&["", "data", ""]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let df2 = import_record_batch(&batch).unwrap();
    let strs = df2.series("s").unwrap().to_strings_strict().unwrap();
    assert_eq!(strs, vec!["", "data", ""]);
}

#[test]
fn test_round_trip_string_unicode() {
    let df = crate::common::dataframe_from_columns(
        vec!["text".to_string()],
        vec![column_from_strs(&["héllo", "日本語", "🚀🎉"]).unwrap()],
    )
    .unwrap();

    let batch = df.to_record_batch().unwrap();
    let arr = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.value(0), "héllo");
    assert_eq!(arr.value(1), "日本語");
    assert_eq!(arr.value(2), "🚀🎉");

    let df2 = import_record_batch(&batch).unwrap();
    let strs = df2.series("text").unwrap().to_strings_strict().unwrap();
    assert_eq!(strs, vec!["héllo", "日本語", "🚀🎉"]);
}

#[test]
fn test_from_record_batch_string_array() {
    let schema = Schema::new(vec![Field::new("name", ArrowDataType::Utf8, false)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"]))])
            .unwrap();

    let df = import_record_batch(&batch).unwrap();
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.series("name").unwrap().dtype().unwrap().id(), TypeId::String);
    let strs = df.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(strs, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn test_from_record_batch_nullable_string_array() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::Utf8, true)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]))])
            .unwrap();

    let df = import_record_batch(&batch).unwrap();
    let cv = df.series("s").unwrap();
    let validity = cv.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);

    // Nullable column: use to_optional_strings to access values alongside nulls.
    let strs = cv.to_optional_strings().unwrap();
    assert_eq!(strs[0], Some("a".to_string()));
    assert_eq!(strs[1], None);
    assert_eq!(strs[2], Some("c".to_string()));
}

#[test]
fn test_from_record_batch_large_string_array() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::LargeUtf8, false)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(LargeStringArray::from(vec!["x", "y", "z"]))]).unwrap();

    let df = import_record_batch(&batch).unwrap();
    let strs = df.series("s").unwrap().to_strings_strict().unwrap();
    assert_eq!(strs, vec!["x", "y", "z"]);
}

#[test]
fn test_round_trip_empty_string_schema() {
    let df =
        crate::common::dataframe_from_columns(vec!["s".to_string()], vec![column_from_strs(&[]).unwrap()]).unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Utf8);

    let arr = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.len(), 0);

    let df2 = import_record_batch(&batch).unwrap();
    assert_eq!(df2.column_names(), vec!["s"]);
    assert_eq!(df2.num_rows(), 0);
    assert_eq!(df2.series("s").unwrap().dtype().unwrap().id(), TypeId::String);
    let strs = df2.series("s").unwrap().to_strings_strict().unwrap();
    assert!(strs.is_empty());
}
