//! Tests for DataFrame ↔ Arrow RecordBatch interop.
use std::sync::{Arc, Mutex};

use arrow::{
    array::*,
    datatypes::{DataType as ArrowDataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::{BoundaryKind, DataFrame, DataType, ErrorCode, ErrorKind, Runtime, TypeId, testing::prelude::*};

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, column_from_strs_nullable, error_context_value,
};

fn import_record_batch(batch: &RecordBatch) -> cudf::Result<DataFrame> {
    let runtime = crate::common::sample_runtime();
    DataFrame::from_record_batch(&runtime, batch)
}

fn decimal128_series(values: &[i64], scale: i32) -> cudf::Result<cudf::Series> {
    crate::common::series("amount", values)?.astype(DataType::fixed_point(TypeId::Decimal128, scale))
}

fn timestamp_series(values: &[i64], duration_type: TypeId, timestamp_type: TypeId) -> cudf::Result<cudf::Series> {
    crate::common::series("ts", values)?.astype(duration_type)?.astype(timestamp_type)
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
    assert_eq!(ArrowDataType::try_from(TypeId::TimestampDays).unwrap(), ArrowDataType::Date32);
    assert_eq!(
        ArrowDataType::try_from(TypeId::TimestampSeconds).unwrap(),
        ArrowDataType::Timestamp(TimeUnit::Second, None)
    );
    assert_eq!(
        ArrowDataType::try_from(TypeId::TimestampMilliseconds).unwrap(),
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None)
    );
    assert_eq!(
        ArrowDataType::try_from(TypeId::TimestampMicroseconds).unwrap(),
        ArrowDataType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert_eq!(
        ArrowDataType::try_from(TypeId::TimestampNanoseconds).unwrap(),
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)
    );
}

#[test]
fn test_arrow_datatype_try_from_type_id_rejects_unsupported_types() {
    let err =
        ArrowDataType::try_from(TypeId::List).expect_err("list columns are not supported by Arrow host-copy interop");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow::TryFrom<TypeId> for ArrowDataType");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("list"));
    assert!(err.message().contains("unsupported cudf type"));

    let decimal_err =
        ArrowDataType::try_from(TypeId::Decimal128).expect_err("TypeId alone must not fabricate a Decimal128 scale");
    assert_eq!(decimal_err.kind(), ErrorKind::Unsupported);
    assert_eq!(decimal_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(decimal_err.operation(), "arrow::TryFrom<TypeId> for ArrowDataType");
    assert_eq!(error_context_value(&decimal_err, "type_id").as_deref(), Some("decimal128"));
    assert_eq!(error_context_value(&decimal_err, "reason").as_deref(), Some("decimal_scale_required"));
}

#[test]
fn test_type_id_try_from_arrow_datatype_supported_mappings() {
    assert_eq!(TypeId::try_from(&ArrowDataType::Int32).unwrap(), TypeId::Int32);
    assert_eq!(TypeId::try_from(&ArrowDataType::Float64).unwrap(), TypeId::Float64);
    assert_eq!(TypeId::try_from(&ArrowDataType::Boolean).unwrap(), TypeId::Bool8);
    assert_eq!(TypeId::try_from(&ArrowDataType::Utf8).unwrap(), TypeId::String);
    assert_eq!(TypeId::try_from(&ArrowDataType::LargeUtf8).unwrap(), TypeId::String);
    assert_eq!(TypeId::try_from(&ArrowDataType::Date32).unwrap(), TypeId::TimestampDays);
    assert_eq!(TypeId::try_from(&ArrowDataType::Timestamp(TimeUnit::Second, None)).unwrap(), TypeId::TimestampSeconds);
    assert_eq!(
        TypeId::try_from(&ArrowDataType::Timestamp(TimeUnit::Millisecond, None)).unwrap(),
        TypeId::TimestampMilliseconds
    );
    assert_eq!(
        TypeId::try_from(&ArrowDataType::Timestamp(TimeUnit::Microsecond, None)).unwrap(),
        TypeId::TimestampMicroseconds
    );
    assert_eq!(
        TypeId::try_from(&ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)).unwrap(),
        TypeId::TimestampNanoseconds
    );
    assert_eq!(TypeId::try_from(&ArrowDataType::Decimal128(18, 2)).unwrap(), TypeId::Decimal128);
}

#[test]
fn test_type_id_try_from_arrow_datatype_rejects_unsupported_types() {
    let err = TypeId::try_from(&ArrowDataType::LargeBinary)
        .expect_err("large binary arrays are not supported by cudf Arrow import");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow::TryFrom<&ArrowDataType> for TypeId");
    assert_eq!(error_context_value(&err, "arrow_type").as_deref(), Some("LargeBinary"));
    assert!(err.message().contains("unsupported Arrow type"));
}

#[test]
fn test_arrow_host_dtype_mapping_preserves_logical_metadata() {
    use cudf::interop::arrow_host::{
        arrow_type_from_cudf_dtype, cudf_dtype_from_arrow_type, is_supported_arrow_numeric_type,
        is_supported_arrow_type,
    };

    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::Decimal128(18, 2)).unwrap(),
        DataType::fixed_point(TypeId::Decimal128, -2)
    );
    assert_eq!(
        arrow_type_from_cudf_dtype(DataType::fixed_point(TypeId::Decimal128, -6)).unwrap(),
        ArrowDataType::Decimal128(38, 6)
    );
    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)).unwrap(),
        DataType::new(TypeId::TimestampNanoseconds)
    );
    assert_eq!(
        arrow_type_from_cudf_dtype(DataType::new(TypeId::TimestampMilliseconds)).unwrap(),
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None)
    );
    assert_eq!(cudf_dtype_from_arrow_type(&ArrowDataType::LargeUtf8).unwrap(), DataType::new(TypeId::String));
    assert!(is_supported_arrow_type(&ArrowDataType::Boolean));
    assert!(is_supported_arrow_numeric_type(&ArrowDataType::Float64));
    assert!(!is_supported_arrow_numeric_type(&ArrowDataType::Boolean));
}

#[test]
fn test_arrow_host_dtype_mapping_rejects_decimal_scale_outside_arrow_range() {
    use cudf::interop::arrow_host::arrow_type_from_cudf_dtype;

    let err = arrow_type_from_cudf_dtype(DataType::fixed_point(TypeId::Decimal128, -39))
        .expect_err("Arrow Decimal128 scale must fit the exported precision");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow_host::arrow_type_from_cudf_dtype");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("decimal128"));
    assert_eq!(error_context_value(&err, "cudf_scale").as_deref(), Some("-39"));
    assert_eq!(error_context_value(&err, "arrow_scale").as_deref(), Some("39"));
    assert_eq!(error_context_value(&err, "arrow_precision").as_deref(), Some("38"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("decimal_scale_exceeds_decimal128_precision"));

    let range_err = arrow_type_from_cudf_dtype(DataType::fixed_point(TypeId::Decimal128, -129))
        .expect_err("large Arrow Decimal128 scale must fail before Arrow array construction");
    assert_eq!(range_err.kind(), ErrorKind::Unsupported);
    assert_eq!(range_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(range_err.operation(), "arrow_host::arrow_type_from_cudf_dtype");
    assert_eq!(error_context_value(&range_err, "type_id").as_deref(), Some("decimal128"));
    assert_eq!(error_context_value(&range_err, "cudf_scale").as_deref(), Some("-129"));
    assert_eq!(error_context_value(&range_err, "arrow_scale").as_deref(), Some("129"));
    assert_eq!(
        error_context_value(&range_err, "reason").as_deref(),
        Some("decimal_scale_exceeds_decimal128_precision")
    );
}

#[test]
fn test_arrow_host_schema_validation_exposes_planner_metadata() {
    use cudf::interop::arrow_host::validate_arrow_schema;

    let schema = Schema::new(vec![
        Field::new("id", ArrowDataType::Int32, false),
        Field::new("amount", ArrowDataType::Decimal128(12, 3), true),
        Field::new("label", ArrowDataType::LargeUtf8, true),
    ]);

    let mapped = validate_arrow_schema(&schema, 17).unwrap();

    assert_eq!(mapped.row_count(), 17);
    assert_eq!(mapped.len(), 3);
    assert_eq!(mapped.field(0).unwrap().name(), "id");
    assert_eq!(mapped.field(0).unwrap().cudf_data_type(), DataType::new(TypeId::Int32));
    assert_eq!(mapped.field(0).unwrap().arrow_data_type(), &ArrowDataType::Int32);
    assert!(!mapped.field(0).unwrap().nullable());
    assert_eq!(mapped.field(1).unwrap().name(), "amount");
    assert_eq!(mapped.field(1).unwrap().cudf_data_type(), DataType::fixed_point(TypeId::Decimal128, -3));
    assert_eq!(mapped.field_by_name("label").unwrap().cudf_data_type(), DataType::new(TypeId::String));
}

#[test]
fn test_arrow_host_schema_validation_accepts_empty_schema_without_copying() {
    use cudf::interop::arrow_host::validate_arrow_schema;

    let schema = Schema::empty();
    let mapped = validate_arrow_schema(&schema, 0).unwrap();

    assert!(mapped.is_empty());
    assert_eq!(mapped.len(), 0);
    assert_eq!(mapped.row_count(), 0);
}

#[test]
fn test_arrow_host_record_batch_schema_validation_uses_batch_row_count_without_copying() {
    use cudf::interop::arrow_host::validate_record_batch_schema;

    let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowDataType::Int32, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

    let mapped = validate_record_batch_schema(&batch).unwrap();

    assert_eq!(mapped.row_count(), 3);
    assert_eq!(mapped.fields()[0].name(), "id");
    assert_eq!(mapped.fields()[0].cudf_data_type(), DataType::new(TypeId::Int32));
}

#[test]
fn test_arrow_host_record_batch_schema_validation_reanchors_schema_errors() {
    use cudf::interop::arrow_host::validate_record_batch_schema;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int32, false),
        Field::new("id", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Int32Array::from(vec![4, 5, 6]))],
    )
    .unwrap();

    let err = validate_record_batch_schema(&batch).expect_err("duplicate names should be anchored at RecordBatch API");

    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(err.operation(), "arrow_host::validate_record_batch_schema");
    assert!(err.call_chain().contains(&"arrow_host::validate_arrow_schema"));
    assert_eq!(error_context_value(&err, "duplicate_name").as_deref(), Some("id"));
}

#[test]
fn test_arrow_host_schema_validation_rejects_timezone_and_nested_types_before_copy() {
    use cudf::interop::arrow_host::{cudf_dtype_from_arrow_type, validate_arrow_schema};

    let tz_err = cudf_dtype_from_arrow_type(&ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())))
        .expect_err("timezone-aware Arrow timestamps are not supported");
    assert_eq!(tz_err.kind(), ErrorKind::Unsupported);
    assert_eq!(tz_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(tz_err.operation(), "arrow_host::cudf_dtype_from_arrow_type");
    assert_eq!(error_context_value(&tz_err, "reason").as_deref(), Some("timezone_not_supported"));

    let nested_schema = Schema::new(vec![Field::new(
        "items",
        ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Int32, true))),
        true,
    )]);
    let nested_err = validate_arrow_schema(&nested_schema, 0).expect_err("nested Arrow types should be rejected");
    assert_eq!(nested_err.kind(), ErrorKind::Unsupported);
    assert_eq!(nested_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(nested_err.operation(), "arrow_host::validate_arrow_schema");
    assert_eq!(error_context_value(&nested_err, "field_name").as_deref(), Some("items"));
    assert_eq!(error_context_value(&nested_err, "reason").as_deref(), Some("unsupported_arrow_type"));
}

#[test]
fn test_arrow_host_schema_validation_rejects_duplicate_names_before_copy() {
    use cudf::interop::arrow_host::validate_arrow_schema;

    let schema = Schema::new(vec![
        Field::new("id", ArrowDataType::Int32, false),
        Field::new("value", ArrowDataType::Int64, false),
        Field::new("id", ArrowDataType::Float64, true),
    ]);

    let err = validate_arrow_schema(&schema, 0).expect_err("duplicate Arrow field names should be rejected");
    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(err.operation(), "arrow_host::validate_arrow_schema");
    assert_eq!(error_context_value(&err, "duplicate_name").as_deref(), Some("id"));
}

#[test]
fn test_arrow_numeric_wrapper_preserves_source_chain() {
    let err = cudf::testing::arrow::trigger_numeric_to_arrow_error_for_test()
        .expect_err("expected numeric Arrow wrapper to surface a mapped error");

    // Arrow producer contract violation classifies as ForeignInvariantViolation.
    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "arrow::numeric_to_arrow");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("arrow_array"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_arrow_primitive_array"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure Arrow primitive values and null buffers have matching logical lengths")
    );

    let source = std::error::Error::source(&err).expect("expected preserved Arrow source");
    assert_eq!(source.to_string(), err.message());
}

#[test]
fn test_to_record_batch_wrapper_preserves_source_chain() {
    let err = cudf::testing::arrow::trigger_record_batch_error_for_test()
        .expect_err("expected RecordBatch wrapper to surface a mapped error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dataframe::to_record_batch");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("record_batch"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_arrow_record_batch"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure Arrow schemas and column arrays have matching row counts and compatible field layouts")
    );

    let source = std::error::Error::source(&err).expect("expected preserved Arrow source");
    assert_eq!(source.to_string(), err.message());
}

#[test]
fn test_arrow_string_array_downcast_mismatch_uses_stable_foreign_contract() {
    let err = cudf::testing::arrow::trigger_arrow_string_array_downcast_mismatch_for_test()
        .expect_err("expected Utf8 downcast mismatch to surface a mapped foreign-contract error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "arrow::string_array_to_column");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("arrow_array"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("unexpected_arrow_array_layout"));
    assert_eq!(error_context_value(&err, "arrow_type").as_deref(), Some("Utf8"));
    assert_eq!(error_context_value(&err, "actual_arrow_type").as_deref(), Some("Int32"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure Utf8 fields are backed by Arrow StringArray values")
    );
}

#[test]
fn test_arrow_large_string_array_downcast_mismatch_uses_same_foreign_contract_family() {
    let err = cudf::testing::arrow::trigger_arrow_large_string_array_downcast_mismatch_for_test()
        .expect_err("expected LargeUtf8 downcast mismatch to surface a mapped foreign-contract error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "arrow::large_string_array_to_column");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("arrow_array"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("unexpected_arrow_array_layout"));
    assert_eq!(error_context_value(&err, "arrow_type").as_deref(), Some("LargeUtf8"));
    assert_eq!(error_context_value(&err, "actual_arrow_type").as_deref(), Some("Int32"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
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
    assert_eq!(error_context_value(&err, "field_name").as_deref(), Some("payload"));
    assert_eq!(error_context_value(&err, "arrow_type").as_deref(), Some("LargeBinary"));
}

#[test]
fn test_from_record_batch_rejects_unsupported_decimal_width_with_stable_code() {
    let array = Decimal64Array::from(vec![Some(12345i64), None, Some(-5)]).with_precision_and_scale(18, 2).unwrap();
    let schema = Schema::new(vec![Field::new("amount64", ArrowDataType::Decimal64(18, 2), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap();

    let err = import_record_batch(&batch).expect_err("Decimal64 import is intentionally unsupported in this slice");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow::arrow_array_to_column");
    assert_eq!(error_context_value(&err, "field_name").as_deref(), Some("amount64"));
    assert_eq!(error_context_value(&err, "arrow_type").as_deref(), Some("Decimal64(18, 2)"));
}

#[test]
fn test_from_record_batch_rejects_unsupported_decimal32_width_with_stable_code() {
    let array = Decimal32Array::from(vec![Some(12345i32), None, Some(-5)]).with_precision_and_scale(9, 2).unwrap();
    let schema = Schema::new(vec![Field::new("amount32", ArrowDataType::Decimal32(9, 2), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap();

    let err = import_record_batch(&batch).expect_err("Decimal32 import is intentionally unsupported in this slice");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "arrow::arrow_array_to_column");
    assert_eq!(error_context_value(&err, "field_name").as_deref(), Some("amount32"));
    assert_eq!(error_context_value(&err, "arrow_type").as_deref(), Some("Decimal32(9, 2)"));
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

#[test]
fn test_to_record_batch_exports_decimal128_date32_and_timestamp_units() {
    let decimal = decimal128_series(&[12345, -67890, 5], -2).unwrap();
    let date = timestamp_series(&[18_262, 18_263, 18_264], TypeId::DurationDays, TypeId::TimestampDays).unwrap();
    let ts_s = timestamp_series(&[1, 2, 3], TypeId::DurationSeconds, TypeId::TimestampSeconds).unwrap();
    let ts_ms =
        timestamp_series(&[1_000, 2_000, 3_000], TypeId::DurationMilliseconds, TypeId::TimestampMilliseconds).unwrap();
    let ts_us =
        timestamp_series(&[1_000, 2_000, 3_000], TypeId::DurationMicroseconds, TypeId::TimestampMicroseconds).unwrap();
    let ts_ns =
        timestamp_series(&[1_000, 2_000, 3_000], TypeId::DurationNanoseconds, TypeId::TimestampNanoseconds).unwrap();
    let df = DataFrame::from_series([
        ("amount", decimal),
        ("date", date),
        ("ts_s", ts_s),
        ("ts_ms", ts_ms),
        ("ts_us", ts_us),
        ("ts_ns", ts_ns),
    ])
    .unwrap();

    let batch = df.to_record_batch().unwrap();

    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Decimal128(38, 2));
    assert_eq!(*batch.schema().field(1).data_type(), ArrowDataType::Date32);
    assert_eq!(*batch.schema().field(2).data_type(), ArrowDataType::Timestamp(TimeUnit::Second, None));
    assert_eq!(*batch.schema().field(3).data_type(), ArrowDataType::Timestamp(TimeUnit::Millisecond, None));
    assert_eq!(*batch.schema().field(4).data_type(), ArrowDataType::Timestamp(TimeUnit::Microsecond, None));
    assert_eq!(*batch.schema().field(5).data_type(), ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));

    let amount = batch.column(0).as_primitive::<arrow::datatypes::Decimal128Type>();
    assert_eq!(amount.precision(), 38);
    assert_eq!(amount.scale(), 2);
    assert_eq!(amount.values().as_ref(), &[1234500i128, -6789000, 500]);
    assert_eq!(
        batch.column(1).as_primitive::<arrow::datatypes::Date32Type>().values().as_ref(),
        &[18_262, 18_263, 18_264]
    );
    assert_eq!(batch.column(2).as_primitive::<arrow::datatypes::TimestampSecondType>().values().as_ref(), &[1, 2, 3]);
    assert_eq!(
        batch.column(3).as_primitive::<arrow::datatypes::TimestampMillisecondType>().values().as_ref(),
        &[1_000, 2_000, 3_000]
    );
    assert_eq!(
        batch.column(4).as_primitive::<arrow::datatypes::TimestampMicrosecondType>().values().as_ref(),
        &[1_000, 2_000, 3_000]
    );
    assert_eq!(
        batch.column(5).as_primitive::<arrow::datatypes::TimestampNanosecondType>().values().as_ref(),
        &[1_000, 2_000, 3_000]
    );
}

#[test]
fn test_round_trip_nullable_decimal128_date32_and_timestamp_units() {
    let amount =
        Decimal128Array::from(vec![Some(12345i128), None, Some(-67890)]).with_precision_and_scale(20, 2).unwrap();
    let date = Date32Array::from(vec![Some(18_262), None, Some(18_264)]);
    let ts_s = TimestampSecondArray::from(vec![Some(1), None, Some(3)]);
    let ts_ms = TimestampMillisecondArray::from(vec![Some(1_000), None, Some(3_000)]);
    let ts_us = TimestampMicrosecondArray::from(vec![Some(1_000), None, Some(3_000)]);
    let ts_ns = TimestampNanosecondArray::from(vec![Some(1_000), None, Some(3_000)]);
    let schema = Schema::new(vec![
        Field::new("amount", ArrowDataType::Decimal128(20, 2), true),
        Field::new("date", ArrowDataType::Date32, true),
        Field::new("ts_s", ArrowDataType::Timestamp(TimeUnit::Second, None), true),
        Field::new("ts_ms", ArrowDataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("ts_us", ArrowDataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("ts_ns", ArrowDataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(amount), Arc::new(date), Arc::new(ts_s), Arc::new(ts_ms), Arc::new(ts_us), Arc::new(ts_ns)],
    )
    .unwrap();

    let df = import_record_batch(&batch).unwrap();

    assert_eq!(df.series("amount").unwrap().dtype().unwrap(), DataType::fixed_point(TypeId::Decimal128, -2));
    assert_eq!(df.series("date").unwrap().dtype().unwrap().id(), TypeId::TimestampDays);
    assert_eq!(df.series("ts_s").unwrap().dtype().unwrap().id(), TypeId::TimestampSeconds);
    assert_eq!(df.series("ts_ms").unwrap().dtype().unwrap().id(), TypeId::TimestampMilliseconds);
    assert_eq!(df.series("ts_us").unwrap().dtype().unwrap().id(), TypeId::TimestampMicroseconds);
    assert_eq!(df.series("ts_ns").unwrap().dtype().unwrap().id(), TypeId::TimestampNanoseconds);

    let round_trip = df.to_record_batch().unwrap();
    assert_eq!(*round_trip.schema().field(0).data_type(), ArrowDataType::Decimal128(38, 2));
    assert_eq!(*round_trip.schema().field(1).data_type(), ArrowDataType::Date32);
    assert_eq!(*round_trip.schema().field(2).data_type(), ArrowDataType::Timestamp(TimeUnit::Second, None));
    assert_eq!(*round_trip.schema().field(3).data_type(), ArrowDataType::Timestamp(TimeUnit::Millisecond, None));
    assert_eq!(*round_trip.schema().field(4).data_type(), ArrowDataType::Timestamp(TimeUnit::Microsecond, None));
    assert_eq!(*round_trip.schema().field(5).data_type(), ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));

    for column_index in 0..round_trip.num_columns() {
        let arr = round_trip.column(column_index);
        assert!(arr.is_valid(0));
        assert!(!arr.is_valid(1));
        assert!(arr.is_valid(2));
    }

    assert_eq!(
        round_trip.column(0).as_primitive::<arrow::datatypes::Decimal128Type>().values().as_ref(),
        &[12345i128, 0, -67890]
    );
    assert_eq!(
        round_trip.column(1).as_primitive::<arrow::datatypes::Date32Type>().values().as_ref(),
        &[18_262, 0, 18_264]
    );
    assert_eq!(
        round_trip.column(2).as_primitive::<arrow::datatypes::TimestampSecondType>().values().as_ref(),
        &[1, 0, 3]
    );
    assert_eq!(
        round_trip.column(3).as_primitive::<arrow::datatypes::TimestampMillisecondType>().values().as_ref(),
        &[1_000, 0, 3_000]
    );
    assert_eq!(
        round_trip.column(4).as_primitive::<arrow::datatypes::TimestampMicrosecondType>().values().as_ref(),
        &[1_000, 0, 3_000]
    );
    assert_eq!(
        round_trip.column(5).as_primitive::<arrow::datatypes::TimestampNanosecondType>().values().as_ref(),
        &[1_000, 0, 3_000]
    );
}

#[test]
fn test_from_record_batch_mixed_decimal_temporal_boundary_observation() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();
    let amount = Decimal128Array::from(vec![12345i128, 67890]).with_precision_and_scale(20, 2).unwrap();
    let date = Date32Array::from(vec![18_262, 18_263]);
    let schema = Schema::new(vec![
        Field::new("amount", ArrowDataType::Decimal128(20, 2), false),
        Field::new("date", ArrowDataType::Date32, false),
    ]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(amount), Arc::new(date)]).unwrap();

    let df = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    assert_eq!(df.num_rows(), 2);
    let observations = observed.lock().unwrap();
    assert_eq!(observations.len(), 1);
    let observation = observations[0];
    assert_eq!(observation.kind, BoundaryKind::HostToDevice);
    assert_eq!(observation.rows, 2);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
}

#[test]
fn test_to_record_batch_mixed_decimal_temporal_boundary_observation() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();

    let amount =
        runtime.series(&[12345i64, -67890]).unwrap().astype(DataType::fixed_point(TypeId::Decimal128, -2)).unwrap();
    let date = runtime
        .series(&[18_262i32, 18_263])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    let ts_s = runtime
        .series(&[1i64, 2])
        .unwrap()
        .astype(TypeId::DurationSeconds)
        .unwrap()
        .astype(TypeId::TimestampSeconds)
        .unwrap();
    let df = DataFrame::from_series([("amount", amount), ("date", date), ("ts_s", ts_s)]).unwrap();

    let batch = df.to_record_batch().unwrap();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 3);
    let observations = observed.lock().unwrap();
    assert_eq!(observations.len(), 1);
    let observation = observations[0];
    assert_eq!(observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(observation.rows, 2);
    assert_eq!(observation.columns, 3);
    assert!(observation.synchronized_for_host);
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
    let df = DataFrame::from_series([("flag", flag)]).unwrap();

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
