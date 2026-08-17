//! Tests for DataFrame ↔ Arrow RecordBatch interop.
use std::sync::{Arc, Mutex};

use arrow::{
    array::*,
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
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

fn integer_array(data_type: &ArrowDataType, values: &[Option<i64>]) -> ArrayRef {
    macro_rules! array {
        ($array:ty, $native:ty) => {
            Arc::new(<$array>::from(values.iter().map(|value| value.map(|value| value as $native)).collect::<Vec<_>>()))
        };
    }
    match data_type {
        ArrowDataType::Int8 => array!(Int8Array, i8),
        ArrowDataType::Int16 => array!(Int16Array, i16),
        ArrowDataType::Int32 => array!(Int32Array, i32),
        ArrowDataType::Int64 => array!(Int64Array, i64),
        ArrowDataType::UInt8 => array!(UInt8Array, u8),
        ArrowDataType::UInt16 => array!(UInt16Array, u16),
        ArrowDataType::UInt32 => array!(UInt32Array, u32),
        ArrowDataType::UInt64 => array!(UInt64Array, u64),
        data_type => panic!("expected an integer data type, got {data_type:?}"),
    }
}

fn assert_string_list_value(array: &ListArray, index: usize, expected: &[Option<&str>]) {
    let values = array.value(index);
    let strings = values.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(strings.len(), expected.len());
    for (row, expected) in expected.iter().enumerate() {
        match expected {
            Some(value) => {
                assert!(strings.is_valid(row));
                assert_eq!(strings.value(row), *value);
            },
            None => assert!(strings.is_null(row)),
        }
    }
}

fn assert_large_string_list_value(array: &LargeListArray, index: usize, expected: &[Option<&str>]) {
    let values = array.value(index);
    let strings = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
    assert_eq!(strings.len(), expected.len());
    for (row, expected) in expected.iter().enumerate() {
        match expected {
            Some(value) => {
                assert!(strings.is_valid(row));
                assert_eq!(strings.value(row), *value);
            },
            None => assert!(strings.is_null(row)),
        }
    }
}

fn assert_float32_list_value(array: &ListArray, index: usize, expected: &[Option<f32>]) {
    let values = array.value(index);
    let floats = values.as_primitive::<arrow::datatypes::Float32Type>();
    assert_eq!(floats.len(), expected.len());
    for (row, expected) in expected.iter().enumerate() {
        match expected {
            Some(value) => {
                assert!(floats.is_valid(row));
                assert_eq!(floats.value(row), *value);
            },
            None => assert!(floats.is_null(row)),
        }
    }
}

fn assert_float64_large_list_value(array: &LargeListArray, index: usize, expected: &[f64]) {
    let values = array.value(index);
    let floats = values.as_primitive::<arrow::datatypes::Float64Type>();
    assert_eq!(floats.len(), expected.len());
    for (actual, expected) in floats.values().iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

fn decimal128_series(values: &[i64], scale: i32) -> cudf::Result<cudf::Series> {
    crate::common::series("amount", values)?.astype(DataType::fixed_point(TypeId::Decimal128, scale))
}

fn nullable_decimal32_series(values: &[i32], validity: &[bool], scale: i32) -> cudf::Result<cudf::Series> {
    crate::common::nullable_series("amount32", values, validity)?
        .astype(DataType::fixed_point(TypeId::Decimal32, scale))
}

fn nullable_decimal64_series(values: &[i64], validity: &[bool], scale: i32) -> cudf::Result<cudf::Series> {
    crate::common::nullable_series("amount64", values, validity)?
        .astype(DataType::fixed_point(TypeId::Decimal64, scale))
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
    assert_eq!(ArrowDataType::try_from(TypeId::DurationSeconds).unwrap(), ArrowDataType::Duration(TimeUnit::Second));
    assert_eq!(
        ArrowDataType::try_from(TypeId::DurationMilliseconds).unwrap(),
        ArrowDataType::Duration(TimeUnit::Millisecond)
    );
    assert_eq!(
        ArrowDataType::try_from(TypeId::DurationMicroseconds).unwrap(),
        ArrowDataType::Duration(TimeUnit::Microsecond)
    );
    assert_eq!(
        ArrowDataType::try_from(TypeId::DurationNanoseconds).unwrap(),
        ArrowDataType::Duration(TimeUnit::Nanosecond)
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
    assert_eq!(TypeId::try_from(&ArrowDataType::Duration(TimeUnit::Second)).unwrap(), TypeId::DurationSeconds);
    assert_eq!(
        TypeId::try_from(&ArrowDataType::Duration(TimeUnit::Millisecond)).unwrap(),
        TypeId::DurationMilliseconds
    );
    assert_eq!(
        TypeId::try_from(&ArrowDataType::Duration(TimeUnit::Microsecond)).unwrap(),
        TypeId::DurationMicroseconds
    );
    assert_eq!(TypeId::try_from(&ArrowDataType::Duration(TimeUnit::Nanosecond)).unwrap(), TypeId::DurationNanoseconds);
    assert_eq!(TypeId::try_from(&ArrowDataType::Decimal128(18, 2)).unwrap(), TypeId::Decimal128);
    assert_eq!(
        TypeId::try_from(&ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Utf8, true)))).unwrap(),
        TypeId::List
    );
    assert_eq!(
        TypeId::try_from(&ArrowDataType::LargeList(Arc::new(Field::new("item", ArrowDataType::LargeUtf8, true))))
            .unwrap(),
        TypeId::List
    );
    assert_eq!(
        TypeId::try_from(&ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Float32, true)))).unwrap(),
        TypeId::List
    );
    assert_eq!(
        TypeId::try_from(&ArrowDataType::FixedSizeList(Arc::new(Field::new("item", ArrowDataType::Float32, true)), 3,))
            .unwrap(),
        TypeId::List
    );
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
        arrow_type_from_cudf_dtype, cudf_dtype_can_export_to_arrow_type, cudf_dtype_from_arrow_type,
        is_supported_arrow_numeric_type, is_supported_arrow_type,
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
        arrow_type_from_cudf_dtype(DataType::fixed_point(TypeId::Decimal32, -2)).unwrap(),
        ArrowDataType::Decimal128(38, 2)
    );
    assert_eq!(
        arrow_type_from_cudf_dtype(DataType::fixed_point(TypeId::Decimal64, -4)).unwrap(),
        ArrowDataType::Decimal128(38, 4)
    );
    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)).unwrap(),
        DataType::new(TypeId::TimestampNanoseconds)
    );
    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::Date64).unwrap(),
        DataType::new(TypeId::TimestampMilliseconds)
    );
    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::Time32(TimeUnit::Millisecond)).unwrap(),
        DataType::new(TypeId::DurationMilliseconds)
    );
    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::Time64(TimeUnit::Nanosecond)).unwrap(),
        DataType::new(TypeId::DurationNanoseconds)
    );
    assert_eq!(
        arrow_type_from_cudf_dtype(DataType::new(TypeId::TimestampMilliseconds)).unwrap(),
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None)
    );
    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::Duration(TimeUnit::Nanosecond)).unwrap(),
        DataType::new(TypeId::DurationNanoseconds)
    );
    assert_eq!(
        arrow_type_from_cudf_dtype(DataType::new(TypeId::DurationMilliseconds)).unwrap(),
        ArrowDataType::Duration(TimeUnit::Millisecond)
    );
    assert_eq!(cudf_dtype_from_arrow_type(&ArrowDataType::LargeUtf8).unwrap(), DataType::new(TypeId::String));
    assert_eq!(cudf_dtype_from_arrow_type(&ArrowDataType::Utf8View).unwrap(), DataType::new(TypeId::String));
    assert_eq!(
        cudf_dtype_from_arrow_type(&ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Utf8, true))))
            .unwrap(),
        DataType::new(TypeId::List)
    );
    assert!(
        !cudf_dtype_can_export_to_arrow_type(
            DataType::new(TypeId::List),
            &ArrowDataType::LargeList(Arc::new(Field::new("element", ArrowDataType::LargeUtf8, true)))
        ),
        "a dtype-only predicate cannot prove a nested child schema"
    );
    assert!(is_supported_arrow_type(&ArrowDataType::Boolean));
    assert!(is_supported_arrow_type(&ArrowDataType::Utf8View));
    assert!(is_supported_arrow_type(&ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Utf8View, true)))));
    assert!(is_supported_arrow_type(&ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Float32, true,)))));
    assert!(is_supported_arrow_type(&ArrowDataType::FixedSizeList(
        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
        2,
    )));
    assert!(is_supported_arrow_numeric_type(&ArrowDataType::Float64));
    assert!(!is_supported_arrow_numeric_type(&ArrowDataType::Boolean));
    assert!(cudf_dtype_can_export_to_arrow_type(
        DataType::fixed_point(TypeId::Decimal64, -2),
        &ArrowDataType::Decimal128(38, 2)
    ));
    assert!(!cudf_dtype_can_export_to_arrow_type(
        DataType::fixed_point(TypeId::Decimal64, -3),
        &ArrowDataType::Decimal128(38, 2)
    ));
}

#[test]
fn test_arrow_host_rejects_non_positive_fixed_size_list_dimension() {
    use cudf::interop::arrow_host::cudf_dtype_from_arrow_type;

    let error = cudf_dtype_from_arrow_type(&ArrowDataType::FixedSizeList(
        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
        0,
    ))
    .expect_err("zero-width fixed-size vectors must be rejected");

    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error_context_value(&error, "dimension").as_deref(), Some("0"));
    assert_eq!(error_context_value(&error, "reason").as_deref(), Some("non_positive_fixed_size_list_dimension"));
}

#[test]
fn test_arrow_host_fixed_size_list_policy_accepts_only_numeric_children() {
    use cudf::interop::arrow_host::cudf_dtype_from_arrow_type;

    for child in [
        ArrowDataType::Int8,
        ArrowDataType::Int16,
        ArrowDataType::Int32,
        ArrowDataType::Int64,
        ArrowDataType::UInt8,
        ArrowDataType::UInt16,
        ArrowDataType::UInt32,
        ArrowDataType::UInt64,
        ArrowDataType::Float32,
        ArrowDataType::Float64,
    ] {
        let data_type = ArrowDataType::FixedSizeList(Arc::new(Field::new("item", child, true)), 2);
        assert_eq!(cudf_dtype_from_arrow_type(&data_type).unwrap(), DataType::new(TypeId::List));
    }

    for child in [
        ArrowDataType::Boolean,
        ArrowDataType::Utf8,
        ArrowDataType::Decimal128(10, 2),
        ArrowDataType::Date32,
        ArrowDataType::Binary,
        ArrowDataType::List(Arc::new(Field::new("nested", ArrowDataType::Int32, true))),
    ] {
        let data_type = ArrowDataType::FixedSizeList(Arc::new(Field::new("item", child, true)), 2);
        assert!(cudf_dtype_from_arrow_type(&data_type).is_err(), "{data_type:?}");
    }
}

#[test]
fn test_all_null_fixed_size_list_rejects_child_offset_overflow_before_allocation() {
    let runtime = crate::common::sample_runtime();
    let data_type = ArrowDataType::FixedSizeList(Arc::new(Field::new("item", ArrowDataType::Int8, true)), i32::MAX);

    let error = cudf::interop::arrow_host::all_null_list_series(&runtime, &data_type, 2)
        .expect_err("rows times dimension beyond the cudf list offset domain must fail before allocation");

    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error_context_value(&error, "root_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&error, "child_rows").as_deref(), Some("4294967294"));
}

#[test]
fn test_arrow_host_duration_import_export_mappings() {
    use cudf::interop::arrow_host::{
        arrow_type_from_cudf_dtype, cudf_dtype_can_export_to_arrow_type, cudf_dtype_from_arrow_type,
    };

    let cases = [
        (TimeUnit::Second, TypeId::DurationSeconds),
        (TimeUnit::Millisecond, TypeId::DurationMilliseconds),
        (TimeUnit::Microsecond, TypeId::DurationMicroseconds),
        (TimeUnit::Nanosecond, TypeId::DurationNanoseconds),
    ];
    for (unit, type_id) in cases {
        let arrow_type = ArrowDataType::Duration(unit);
        let cudf_dtype = DataType::new(type_id);

        assert_eq!(cudf_dtype_from_arrow_type(&arrow_type).unwrap(), cudf_dtype);
        assert_eq!(arrow_type_from_cudf_dtype(cudf_dtype).unwrap(), arrow_type);
        assert!(cudf_dtype_can_export_to_arrow_type(cudf_dtype, &arrow_type));
        assert_eq!(TypeId::try_from(&arrow_type).unwrap(), type_id);
        assert_eq!(ArrowDataType::try_from(type_id).unwrap(), arrow_type);
    }

    let duration_days_err = arrow_type_from_cudf_dtype(DataType::new(TypeId::DurationDays))
        .expect_err("Arrow has no duration day unit for cudf DurationDays export");
    assert_eq!(duration_days_err.kind(), ErrorKind::Unsupported);
    assert_eq!(duration_days_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(duration_days_err.operation(), "arrow_host::arrow_type_from_cudf_dtype");
    assert_eq!(error_context_value(&duration_days_err, "type_id").as_deref(), Some("duration_days"));
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
fn test_arrow_host_schema_validation_accepts_timezone_and_fixed_width_lists_before_copy() {
    use cudf::interop::arrow_host::{cudf_dtype_from_arrow_type, validate_arrow_schema};

    let tz_dtype = cudf_dtype_from_arrow_type(&ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())))
        .expect("timezone-aware Arrow timestamps should import as physical cuDF timestamps");
    assert_eq!(tz_dtype, DataType::new(TypeId::TimestampMicroseconds));

    let numeric_list_schema = Schema::new(vec![Field::new(
        "items",
        ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Int32, true))),
        true,
    )]);
    let mapped = validate_arrow_schema(&numeric_list_schema, 0).expect("numeric Arrow lists should be admitted");
    assert_eq!(mapped.fields()[0].cudf_data_type(), DataType::new(TypeId::List));

    let boolean_list_schema = Schema::new(vec![Field::new(
        "items",
        ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Boolean, true))),
        true,
    )]);
    let mapped = validate_arrow_schema(&boolean_list_schema, 0).expect("Boolean Arrow lists should be admitted");
    assert_eq!(mapped.fields()[0].cudf_data_type(), DataType::new(TypeId::List));
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
fn test_zero_column_export_preserves_logical_row_count() {
    let df =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2, 3]).unwrap()])
            .unwrap()
            .select(Vec::<&str>::new())
            .unwrap();

    let batch = df.to_record_batch().unwrap();
    assert_eq!(batch.num_columns(), 0);
    assert_eq!(batch.num_rows(), 3);
}

#[test]
fn test_zero_column_gpu_record_batch_materialization_preserves_logical_row_count() {
    let frame =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2, 3]).unwrap()])
            .unwrap()
            .select(Vec::<&str>::new())
            .unwrap();
    let schema = Arc::new(Schema::empty());

    let gpu_batch = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema)).unwrap();
    let batch = gpu_batch.to_host_record_batch().unwrap();

    assert_eq!(batch.schema(), schema);
    assert_eq!(batch.num_columns(), 0);
    assert_eq!(batch.num_rows(), 3);
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
fn test_to_record_batch_exports_decimal32_and_decimal64_as_decimal128_with_nulls() {
    let decimal32 = nullable_decimal32_series(&[123, 0, -5], &[true, false, true], -2).unwrap();
    let decimal64 = nullable_decimal64_series(&[12345, 0, -678], &[true, false, true], -3).unwrap();
    let df = DataFrame::from_series([("amount32", decimal32), ("amount64", decimal64)]).unwrap();

    let batch = df.to_record_batch().unwrap();

    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Decimal128(38, 2));
    assert!(batch.schema().field(0).is_nullable());
    assert_eq!(*batch.schema().field(1).data_type(), ArrowDataType::Decimal128(38, 3));
    assert!(batch.schema().field(1).is_nullable());

    let amount32 = batch.column(0).as_primitive::<arrow::datatypes::Decimal128Type>();
    assert_eq!(amount32.precision(), 38);
    assert_eq!(amount32.scale(), 2);
    assert!(amount32.is_valid(0));
    assert!(!amount32.is_valid(1));
    assert!(amount32.is_valid(2));
    assert_eq!(amount32.values().as_ref(), &[12300i128, 0, -500]);

    let amount64 = batch.column(1).as_primitive::<arrow::datatypes::Decimal128Type>();
    assert_eq!(amount64.precision(), 38);
    assert_eq!(amount64.scale(), 3);
    assert!(amount64.is_valid(0));
    assert!(!amount64.is_valid(1));
    assert!(amount64.is_valid(2));
    assert_eq!(amount64.values().as_ref(), &[12345000i128, 0, -678000]);
}

#[test]
fn test_gpu_record_batch_accepts_decimal64_under_decimal128_schema_contract() {
    let decimal64 = nullable_decimal64_series(&[12345, 0, -678], &[true, false, true], -3).unwrap();
    let frame = DataFrame::from_series([("amount", decimal64)]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("amount", ArrowDataType::Decimal128(38, 3), true)]));

    let gpu_batch = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema)).unwrap();
    let materialized = gpu_batch.to_host_record_batch().unwrap();

    assert_eq!(materialized.schema(), schema);
    let amount = materialized.column(0).as_primitive::<arrow::datatypes::Decimal128Type>();
    assert_eq!(amount.precision(), 38);
    assert_eq!(amount.scale(), 3);
    assert!(amount.is_valid(0));
    assert!(!amount.is_valid(1));
    assert!(amount.is_valid(2));
    assert_eq!(amount.values().as_ref(), &[12345000i128, 0, -678000]);
}

#[test]
fn test_timezone_timestamp_import_preserves_schema_contract_on_gpu_record_batch_export() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "event_ts",
        ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        true,
    )]));
    let values = TimestampMicrosecondArray::from(vec![Some(1_700_000_000_000_000), None, Some(1_700_000_000_000_123)])
        .with_timezone("UTC");
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)]).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    assert_eq!(frame.series("event_ts").unwrap().dtype().unwrap(), DataType::new(TypeId::TimestampMicroseconds));
    let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
        .unwrap()
        .to_host_record_batch()
        .unwrap();

    assert_eq!(materialized.schema(), schema);
    let event_ts = materialized.column(0).as_primitive::<arrow::datatypes::TimestampMicrosecondType>();
    assert_eq!(event_ts.values().as_ref(), &[1_700_000_000_000_000, 0, 1_700_000_000_000_123]);
    assert!(event_ts.is_valid(0));
    assert!(!event_ts.is_valid(1));
    assert!(event_ts.is_valid(2));
}

#[test]
fn test_list_utf8_import_exports_arrow_list_with_null_lists_and_null_elements() {
    let item = Arc::new(Field::new("item", ArrowDataType::Utf8, true));
    let values = Arc::new(StringArray::from(vec![Some("alpha"), Some("beta"), None, Some("delta")]));
    let lists = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 2, 4])),
        values,
        Some(NullBuffer::from(vec![true, true, false, true])),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("items", ArrowDataType::List(item), true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(lists)]).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    assert_eq!(frame.series("items").unwrap().dtype().unwrap(), DataType::new(TypeId::List));
    let materialized = frame.to_record_batch().unwrap();

    let items = materialized.column(0).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(items.len(), 4);
    assert!(items.is_valid(0));
    assert!(items.is_valid(1));
    assert!(items.is_null(2));
    assert!(items.is_valid(3));
    assert_string_list_value(items, 0, &[Some("alpha"), Some("beta")]);
    assert_string_list_value(items, 1, &[]);
    assert_string_list_value(items, 3, &[None, Some("delta")]);
}

#[test]
fn test_large_list_utf8_import_preserves_schema_contract_on_gpu_record_batch_export() {
    let element = Arc::new(Field::new("element", ArrowDataType::LargeUtf8, true));
    let values = Arc::new(LargeStringArray::from(vec![Some("north"), None, Some("south")]));
    let lists = LargeListArray::new(
        Arc::clone(&element),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1, 3])),
        values,
        None,
    );
    let schema = Arc::new(Schema::new(vec![Field::new("items", ArrowDataType::LargeList(element), true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(lists)]).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
        .unwrap()
        .to_host_record_batch()
        .unwrap();

    assert_eq!(materialized.schema(), schema);
    let items = materialized.column(0).as_any().downcast_ref::<LargeListArray>().unwrap();
    assert_large_string_list_value(items, 0, &[Some("north")]);
    assert_large_string_list_value(items, 1, &[None, Some("south")]);
}

#[test]
fn test_sliced_list_utf8_import_normalizes_offsets() {
    let item = Arc::new(Field::new("item", ArrowDataType::Utf8, false));
    let values = Arc::new(StringArray::from(vec!["skip", "kept", "also_kept"]));
    let lists =
        ListArray::new(Arc::clone(&item), OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 3, 3])), values, None);
    let sliced = lists.slice(1, 2);
    let schema = Arc::new(Schema::new(vec![Field::new("items", ArrowDataType::List(item), false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(sliced)]).unwrap();

    let materialized = import_record_batch(&batch).unwrap().to_record_batch().unwrap();

    let items = materialized.column(0).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(items.value_offsets(), &[0, 2, 2]);
    assert_string_list_value(items, 0, &[Some("kept"), Some("also_kept")]);
    assert_string_list_value(items, 1, &[]);
}

#[test]
fn test_list_float32_import_exports_numeric_child_with_root_and_child_validity() {
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, true));
    let values = Arc::new(Float32Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)]));
    let lists = Arc::new(ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 3, 5])),
        values,
        Some(NullBuffer::from(vec![true, false, true, true])),
    ));
    let schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(item), true)]));
    let arrays = vec![Arc::clone(&lists) as ArrayRef];
    let estimated = cudf::interop::arrow_host::estimate_list_import_device_memory_bytes(&arrays).unwrap();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    assert_eq!(frame.series("features").unwrap().dtype().unwrap(), DataType::new(TypeId::List));
    assert!(estimated >= frame.series("features").unwrap().device_allocated_size_bytes().unwrap());
    let materialized = frame.to_record_batch().unwrap();
    let features = materialized.column(0).as_any().downcast_ref::<ListArray>().unwrap();

    assert_eq!(features.value_offsets(), &[0, 2, 2, 3, 5]);
    assert!(features.is_valid(0));
    assert!(features.is_null(1));
    assert_float32_list_value(features, 0, &[Some(1.0), None]);
    assert_float32_list_value(features, 2, &[Some(3.0)]);
    assert_float32_list_value(features, 3, &[Some(4.0), Some(5.0)]);

    let features_series = frame.series("features").unwrap();
    let export = features_series.try_prepare_device_column_export().unwrap();
    let child = match export.column().layout() {
        rapids_interop::ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected list layout, got {layout:?}"),
    };
    let child = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child.data_type().id(), rapids_interop::TypeId::Float32);
    assert!(matches!(child.layout(), rapids_interop::ColumnLayout::FixedWidth { .. }));
}

#[test]
fn test_sliced_large_list_float64_import_preserves_non_null_special_values_and_empty_rows() {
    let item = Arc::new(Field::new("element", ArrowDataType::Float64, false));
    let values = Arc::new(Float64Array::from(vec![123.0, f64::INFINITY, f64::NEG_INFINITY, -0.0, f64::NAN, 42.0, 7.0]));
    let full = LargeListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1, 3, 3, 3, 5, 7])),
        values,
        None,
    );
    let sliced = Arc::new(full.slice(1, 4));
    let schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::LargeList(item), false)]));
    let arrays = vec![Arc::clone(&sliced) as ArrayRef];
    let estimated = cudf::interop::arrow_host::estimate_list_import_device_memory_bytes(&arrays).unwrap();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    assert!(estimated >= frame.series("features").unwrap().device_allocated_size_bytes().unwrap());
    let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
        .unwrap()
        .to_host_record_batch()
        .unwrap();

    assert_eq!(materialized.schema(), schema);
    let features = materialized.column(0).as_any().downcast_ref::<LargeListArray>().unwrap();
    assert_eq!(features.value_offsets(), &[0, 2, 2, 2, 4]);
    assert_float64_large_list_value(features, 0, &[f64::INFINITY, f64::NEG_INFINITY]);
    assert_float64_large_list_value(features, 1, &[]);
    assert_float64_large_list_value(features, 2, &[]);
    assert_float64_large_list_value(features, 3, &[-0.0, f64::NAN]);
}

#[test]
fn test_integer_list_widths_round_trip_across_list_shapes() {
    let child_types = [
        ArrowDataType::Int8,
        ArrowDataType::Int16,
        ArrowDataType::Int32,
        ArrowDataType::Int64,
        ArrowDataType::UInt8,
        ArrowDataType::UInt16,
        ArrowDataType::UInt32,
        ArrowDataType::UInt64,
    ];

    for (index, child_type) in child_types.into_iter().enumerate() {
        let (offsets, values, root_validity, child_nullable, slice) = match index % 4 {
            0 => (
                vec![0, 2, 2, 2, 4],
                vec![Some(1), None, Some(3), Some(4)],
                Some(vec![true, true, false, true]),
                true,
                None,
            ),
            1 => (vec![0, 1, 1, 3], vec![Some(1), Some(2), Some(3)], None, false, None),
            2 => (vec![0, 1, 3, 3, 4], vec![Some(1), None, Some(3), Some(4)], None, true, Some((1, 2))),
            _ => (vec![0], Vec::new(), Some(Vec::new()), true, None),
        };
        let item = Arc::new(Field::new("item", child_type.clone(), child_nullable));
        let list = ListArray::new(
            Arc::clone(&item),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            integer_array(&child_type, &values),
            root_validity.map(NullBuffer::from),
        );
        let array: ArrayRef = match slice {
            Some((offset, length)) => Arc::new(list.slice(offset, length)),
            None => Arc::new(list),
        };
        let schema = Arc::new(Schema::new(vec![Field::new("items", ArrowDataType::List(item), true)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap();

        let frame = import_record_batch(&batch).unwrap();
        let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
            .unwrap()
            .to_host_record_batch()
            .unwrap();
        assert_eq!(materialized.schema(), schema);
        assert_eq!(materialized.column(0).to_data(), batch.column(0).to_data(), "{child_type:?}");
    }

    let item = Arc::new(Field::new("element", ArrowDataType::Int64, true));
    let list = LargeListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 2, 2, 3])),
        integer_array(&ArrowDataType::Int64, &[Some(10), None, Some(30)]),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("items", ArrowDataType::LargeList(item), true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(list)]).unwrap();
    let frame = import_record_batch(&batch).unwrap();
    let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
        .unwrap()
        .to_host_record_batch()
        .unwrap();
    assert_eq!(materialized.column(0).to_data(), batch.column(0).to_data());
}

#[test]
fn test_list_boolean_import_unpacks_bool8_and_reexports_exact_values() {
    let item = Arc::new(Field::new("item", ArrowDataType::Boolean, true));
    let lists = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 3, 3, 3, 4])),
        Arc::new(BooleanArray::from(vec![Some(true), Some(false), None, Some(true)])),
        Some(NullBuffer::from(vec![true, true, false, true])),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("flags", ArrowDataType::List(item), true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(lists)]).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    let series = frame.series("flags").unwrap();
    let export = series.try_prepare_device_column_export().unwrap();
    let child = match export.column().layout() {
        rapids_interop::ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected list layout, got {layout:?}"),
    };
    let child = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child.data_type().id(), rapids_interop::TypeId::Bool8);

    let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
        .unwrap()
        .to_host_record_batch()
        .unwrap();
    assert_eq!(materialized.schema(), schema);
    assert_eq!(materialized.column(0).to_data(), batch.column(0).to_data());
}

#[test]
fn test_large_list_boolean_slice_and_zero_child_shapes_round_trip() {
    let item = Arc::new(Field::new("element", ArrowDataType::Boolean, true));
    let full = LargeListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1, 3, 3, 4])),
        Arc::new(BooleanArray::from(vec![Some(false), Some(true), None, Some(false)])),
        Some(NullBuffer::from(vec![true, true, false, true])),
    );
    let sliced = full.slice(1, 2);
    let schema = Arc::new(Schema::new(vec![Field::new("flags", ArrowDataType::LargeList(Arc::clone(&item)), true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(sliced)]).unwrap();
    let frame = import_record_batch(&batch).unwrap();
    let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
        .unwrap()
        .to_host_record_batch()
        .unwrap();
    let expected = LargeListArray::new(
        item,
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 2, 2])),
        Arc::new(BooleanArray::from(vec![Some(true), None])),
        Some(NullBuffer::from(vec![true, false])),
    );
    assert_eq!(materialized.schema(), schema);
    assert_eq!(materialized.column(0).to_data(), expected.to_data());

    for rows in [3, 0] {
        let offsets = vec![0; rows + 1];
        let lists = ListArray::new(
            Arc::new(Field::new("item", ArrowDataType::Boolean, true)),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(BooleanArray::from(Vec::<Option<bool>>::new())),
            Some(NullBuffer::from(vec![false; rows])),
        );
        let schema = Arc::new(Schema::new(vec![Field::new("flags", lists.data_type().clone(), true)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(lists)]).unwrap();
        let frame = import_record_batch(&batch).unwrap();
        let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema))
            .unwrap()
            .to_host_record_batch()
            .unwrap();
        assert_eq!(materialized.schema(), schema);
        assert_eq!(materialized.column(0).to_data(), batch.column(0).to_data());
    }
}

#[test]
fn test_fixed_size_list_float32_import_synthesizes_cudf_list_offsets() {
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, true));
    let values = Arc::new(Float32Array::from(vec![Some(1.0), Some(2.0), Some(3.0), None, Some(5.0), Some(6.0)]));
    let lists = Arc::new(FixedSizeListArray::new(
        Arc::clone(&item),
        2,
        values,
        Some(NullBuffer::from(vec![true, false, true])),
    ));
    let schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 2), true)]));
    let arrays = vec![Arc::clone(&lists) as ArrayRef];
    let estimated = cudf::interop::arrow_host::estimate_list_import_device_memory_bytes(&arrays).unwrap();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    assert_eq!(frame.series("features").unwrap().dtype().unwrap(), DataType::new(TypeId::List));
    assert!(estimated >= frame.series("features").unwrap().device_allocated_size_bytes().unwrap());
    let materialized = frame.to_record_batch().unwrap();
    let features = materialized.column(0).as_any().downcast_ref::<ListArray>().unwrap();

    assert_eq!(features.value_offsets(), &[0, 2, 4, 6]);
    assert!(features.is_valid(0));
    assert!(features.is_null(1));
    assert!(features.is_valid(2));
    assert_float32_list_value(features, 0, &[Some(1.0), Some(2.0)]);
    assert_float32_list_value(features, 2, &[Some(5.0), Some(6.0)]);

    let fixed = cudf::frame::dataframe::GpuRecordBatch::new(frame, schema).unwrap().to_host_record_batch().unwrap();
    let features = fixed.column(0).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    assert_eq!(features.value_length(), 2);
    assert!(features.is_valid(0));
    assert!(features.is_null(1));
    assert!(features.is_valid(2));
    let first = features.value(0);
    assert_eq!(first.as_primitive::<arrow::datatypes::Float32Type>().values(), &[1.0, 2.0]);
    let third = features.value(2);
    assert_eq!(third.as_primitive::<arrow::datatypes::Float32Type>().values(), &[5.0, 6.0]);
}

#[test]
fn test_sliced_fixed_size_list_float32_import_uses_active_child_range() {
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let lists = FixedSizeListArray::new(
        Arc::clone(&item),
        2,
        Arc::new(Float32Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])),
        None,
    );
    let sliced = Arc::new(lists.slice(1, 2));
    let schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 2), false)]));
    let arrays = vec![Arc::clone(&sliced) as ArrayRef];
    let estimated = cudf::interop::arrow_host::estimate_list_import_device_memory_bytes(&arrays).unwrap();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();

    let frame = import_record_batch(&batch).unwrap();
    assert!(estimated >= frame.series("features").unwrap().device_allocated_size_bytes().unwrap());
    let materialized = frame.to_record_batch().unwrap();
    let features = materialized.column(0).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(features.value_offsets(), &[0, 2, 4]);
    assert_float32_list_value(features, 0, &[Some(2.0), Some(3.0)]);
    assert_float32_list_value(features, 1, &[Some(4.0), Some(5.0)]);
}

#[test]
fn test_gpu_record_batch_rejects_list_int64_claiming_float32_child_schema() {
    let source_item = Arc::new(Field::new("item", ArrowDataType::Int64, false));
    let source = ListArray::new(
        Arc::clone(&source_item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 3])),
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        None,
    );
    let source_schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(source_item), false)]));
    let source_batch = RecordBatch::try_new(source_schema, vec![Arc::new(source)]).unwrap();
    let frame = import_record_batch(&source_batch).unwrap();
    let claimed_schema = Arc::new(Schema::new(vec![Field::new(
        "features",
        ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Float32, false))),
        false,
    )]));

    let error = cudf::frame::dataframe::GpuRecordBatch::new(frame, claimed_schema)
        .expect_err("List<Int64> must not claim a List<Float32> schema contract");

    assert_eq!(error.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(error.operation(), "dataframe::GpuRecordBatch::new");
    assert_eq!(error_context_value(&error, "actual_child_cudf_type").as_deref(), Some("int64"));
    assert_eq!(error_context_value(&error, "expected_child_arrow_type").as_deref(), Some("Float32"));
    assert_eq!(error_context_value(&error, "reason").as_deref(), Some("schema_list_child_dtype_mismatch"));
}

#[test]
fn test_gpu_record_batch_rejects_list_string_claiming_float32_child_schema() {
    let source_item = Arc::new(Field::new("item", ArrowDataType::Utf8, false));
    let source = ListArray::new(
        Arc::clone(&source_item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 3])),
        Arc::new(StringArray::from(vec!["large string payload", "another payload", "tail"])),
        None,
    );
    let source_schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(source_item), false)]));
    let source_batch = RecordBatch::try_new(source_schema, vec![Arc::new(source)]).unwrap();
    let frame = import_record_batch(&source_batch).unwrap();
    let claimed_schema = Arc::new(Schema::new(vec![Field::new(
        "features",
        ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Float32, false))),
        false,
    )]));

    let error = cudf::frame::dataframe::GpuRecordBatch::new(frame, claimed_schema)
        .expect_err("List<String> must be rejected before Float32-based host memory estimation");

    assert_eq!(error.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(error.operation(), "dataframe::GpuRecordBatch::new");
    assert_eq!(error_context_value(&error, "actual_child_cudf_type").as_deref(), Some("string"));
    assert_eq!(error_context_value(&error, "expected_child_arrow_type").as_deref(), Some("Float32"));
    assert_eq!(error_context_value(&error, "reason").as_deref(), Some("schema_list_child_dtype_mismatch"));
}

#[test]
fn test_gpu_record_batch_rejects_nullable_list_child_claiming_non_nullable_schema() {
    let source_item = Arc::new(Field::new("item", ArrowDataType::Int64, true));
    let source = ListArray::new(
        Arc::clone(&source_item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 3])),
        Arc::new(Int64Array::from(vec![Some(10), None, Some(30)])),
        None,
    );
    let source_schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(source_item), false)]));
    let source_batch = RecordBatch::try_new(source_schema, vec![Arc::new(source)]).unwrap();
    let frame = import_record_batch(&source_batch).unwrap();
    let claimed_schema = Arc::new(Schema::new(vec![Field::new(
        "features",
        ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Int64, false))),
        false,
    )]));

    let error = cudf::frame::dataframe::GpuRecordBatch::new(frame, claimed_schema)
        .expect_err("a nullable list child must not claim a non-nullable child schema");

    assert_eq!(error.code(), ErrorCode::NullUnexpected);
    assert_eq!(error_context_value(&error, "child_null_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&error, "reason").as_deref(), Some("schema_list_child_nullability_mismatch"));
}

#[test]
fn test_arrow_list_constructor_rejects_child_length_mismatch_before_import() {
    let child = Arc::new(Field::new("item", ArrowDataType::Int64, true));
    let error = ListArray::try_new(
        child,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2])),
        Arc::new(Int64Array::from(vec![1])),
        None,
    )
    .expect_err("list offsets outside the child array must fail before cudf import");

    assert!(error.to_string().contains("offset"), "{error}");
}

#[test]
fn test_gpu_record_batch_rejects_fixed_size_list_dimension_mismatch_at_construction() {
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let source =
        FixedSizeListArray::new(Arc::clone(&item), 2, Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0])), None);
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(Arc::clone(&item), 2), false)]));
    let source_batch = RecordBatch::try_new(source_schema, vec![Arc::new(source)]).unwrap();
    let frame = import_record_batch(&source_batch).unwrap();
    let claimed_schema =
        Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 3), false)]));

    let error = cudf::frame::dataframe::GpuRecordBatch::new(frame, claimed_schema)
        .expect_err("a retained FixedSizeList dimension must match every cudf list row");

    assert_eq!(error.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(error.operation(), "dataframe::GpuRecordBatch::new");
    assert_eq!(error_context_value(&error, "expected_dimension").as_deref(), Some("3"));
    assert_eq!(error_context_value(&error, "minimum_actual_dimension").as_deref(), Some("2"));
    assert_eq!(error_context_value(&error, "maximum_actual_dimension").as_deref(), Some("2"));
    assert_eq!(error_context_value(&error, "evidence").as_deref(), Some("device_reduced_list_offset_width_range"));
    assert_eq!(error_context_value(&error, "reason").as_deref(), Some("schema_fixed_size_list_dimension_mismatch"));
}

#[test]
fn test_gpu_record_batch_checks_each_fixed_size_list_row_not_only_total_child_length() {
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let source = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 4])),
        Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        None,
    );
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(Arc::clone(&item)), false)]));
    let source_batch = RecordBatch::try_new(source_schema, vec![Arc::new(source)]).unwrap();
    let frame = import_record_batch(&source_batch).unwrap();
    let claimed_schema =
        Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 2), false)]));

    let error = cudf::frame::dataframe::GpuRecordBatch::new(frame, claimed_schema)
        .expect_err("equal total child length must not hide variable list row widths");

    assert_eq!(error.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(error_context_value(&error, "expected_dimension").as_deref(), Some("2"));
    assert_eq!(error_context_value(&error, "minimum_actual_dimension").as_deref(), Some("1"));
    assert_eq!(error_context_value(&error, "maximum_actual_dimension").as_deref(), Some("3"));
    assert_eq!(error_context_value(&error, "reason").as_deref(), Some("schema_fixed_size_list_dimension_mismatch"));
}

#[test]
fn test_gpu_record_batch_allows_zero_width_null_fixed_size_list_rows() {
    let item = Arc::new(Field::new("item", ArrowDataType::Int32, false));
    let source = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 4])),
        Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(Arc::clone(&item)), true)]));
    let source_batch = RecordBatch::try_new(source_schema, vec![Arc::new(source)]).unwrap();
    let frame = import_record_batch(&source_batch).unwrap();
    let fixed_schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 2), true)]));

    let materialized = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&fixed_schema))
        .unwrap()
        .to_host_record_batch()
        .unwrap();

    assert_eq!(materialized.schema(), fixed_schema);
    let features = materialized.column(0).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    assert!(features.is_valid(0));
    assert!(features.is_null(1));
    assert!(features.is_valid(2));
    assert_eq!(features.value(0).as_primitive::<arrow::datatypes::Int32Type>().values(), &[1, 2]);
    assert_eq!(features.value(2).as_primitive::<arrow::datatypes::Int32Type>().values(), &[3, 4]);
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
    let duration_s = DurationSecondArray::from(vec![Some(4), None, Some(6)]);
    let duration_ms = DurationMillisecondArray::from(vec![Some(4_000), None, Some(6_000)]);
    let duration_us = DurationMicrosecondArray::from(vec![Some(4_000), None, Some(6_000)]);
    let duration_ns = DurationNanosecondArray::from(vec![Some(4_000), None, Some(6_000)]);
    let schema = Schema::new(vec![
        Field::new("amount", ArrowDataType::Decimal128(20, 2), true),
        Field::new("date", ArrowDataType::Date32, true),
        Field::new("ts_s", ArrowDataType::Timestamp(TimeUnit::Second, None), true),
        Field::new("ts_ms", ArrowDataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("ts_us", ArrowDataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("ts_ns", ArrowDataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new("duration_s", ArrowDataType::Duration(TimeUnit::Second), true),
        Field::new("duration_ms", ArrowDataType::Duration(TimeUnit::Millisecond), true),
        Field::new("duration_us", ArrowDataType::Duration(TimeUnit::Microsecond), true),
        Field::new("duration_ns", ArrowDataType::Duration(TimeUnit::Nanosecond), true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(amount),
            Arc::new(date),
            Arc::new(ts_s),
            Arc::new(ts_ms),
            Arc::new(ts_us),
            Arc::new(ts_ns),
            Arc::new(duration_s),
            Arc::new(duration_ms),
            Arc::new(duration_us),
            Arc::new(duration_ns),
        ],
    )
    .unwrap();

    let df = import_record_batch(&batch).unwrap();

    assert_eq!(df.series("amount").unwrap().dtype().unwrap(), DataType::fixed_point(TypeId::Decimal128, -2));
    assert_eq!(df.series("date").unwrap().dtype().unwrap().id(), TypeId::TimestampDays);
    assert_eq!(df.series("ts_s").unwrap().dtype().unwrap().id(), TypeId::TimestampSeconds);
    assert_eq!(df.series("ts_ms").unwrap().dtype().unwrap().id(), TypeId::TimestampMilliseconds);
    assert_eq!(df.series("ts_us").unwrap().dtype().unwrap().id(), TypeId::TimestampMicroseconds);
    assert_eq!(df.series("ts_ns").unwrap().dtype().unwrap().id(), TypeId::TimestampNanoseconds);
    assert_eq!(df.series("duration_s").unwrap().dtype().unwrap().id(), TypeId::DurationSeconds);
    assert_eq!(df.series("duration_ms").unwrap().dtype().unwrap().id(), TypeId::DurationMilliseconds);
    assert_eq!(df.series("duration_us").unwrap().dtype().unwrap().id(), TypeId::DurationMicroseconds);
    assert_eq!(df.series("duration_ns").unwrap().dtype().unwrap().id(), TypeId::DurationNanoseconds);

    let round_trip = df.to_record_batch().unwrap();
    assert_eq!(*round_trip.schema().field(0).data_type(), ArrowDataType::Decimal128(38, 2));
    assert_eq!(*round_trip.schema().field(1).data_type(), ArrowDataType::Date32);
    assert_eq!(*round_trip.schema().field(2).data_type(), ArrowDataType::Timestamp(TimeUnit::Second, None));
    assert_eq!(*round_trip.schema().field(3).data_type(), ArrowDataType::Timestamp(TimeUnit::Millisecond, None));
    assert_eq!(*round_trip.schema().field(4).data_type(), ArrowDataType::Timestamp(TimeUnit::Microsecond, None));
    assert_eq!(*round_trip.schema().field(5).data_type(), ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));
    assert_eq!(*round_trip.schema().field(6).data_type(), ArrowDataType::Duration(TimeUnit::Second));
    assert_eq!(*round_trip.schema().field(7).data_type(), ArrowDataType::Duration(TimeUnit::Millisecond));
    assert_eq!(*round_trip.schema().field(8).data_type(), ArrowDataType::Duration(TimeUnit::Microsecond));
    assert_eq!(*round_trip.schema().field(9).data_type(), ArrowDataType::Duration(TimeUnit::Nanosecond));

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
    assert_eq!(
        round_trip.column(6).as_primitive::<arrow::datatypes::DurationSecondType>().values().as_ref(),
        &[4, 0, 6]
    );
    assert_eq!(
        round_trip.column(7).as_primitive::<arrow::datatypes::DurationMillisecondType>().values().as_ref(),
        &[4_000, 0, 6_000]
    );
    assert_eq!(
        round_trip.column(8).as_primitive::<arrow::datatypes::DurationMicrosecondType>().values().as_ref(),
        &[4_000, 0, 6_000]
    );
    assert_eq!(
        round_trip.column(9).as_primitive::<arrow::datatypes::DurationNanosecondType>().values().as_ref(),
        &[4_000, 0, 6_000]
    );
}

#[test]
fn test_gpu_record_batch_preserves_date64_and_time_schema_contract() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("date64", ArrowDataType::Date64, false),
        Field::new("time_s", ArrowDataType::Time32(TimeUnit::Second), false),
        Field::new("time_ms", ArrowDataType::Time32(TimeUnit::Millisecond), false),
        Field::new("time_us", ArrowDataType::Time64(TimeUnit::Microsecond), false),
        Field::new("time_ns", ArrowDataType::Time64(TimeUnit::Nanosecond), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Date64Array::from(vec![1_577_836_800_000, 1_580_428_800_000])),
            Arc::new(Time32SecondArray::from(vec![1, 86_399])),
            Arc::new(Time32MillisecondArray::from(vec![1_000, 86_399_000])),
            Arc::new(Time64MicrosecondArray::from(vec![1_000_000, 86_399_000_000])),
            Arc::new(Time64NanosecondArray::from(vec![1_000_000_000, 86_399_000_000_000])),
        ],
    )
    .unwrap();

    let df = import_record_batch(&batch).unwrap();

    assert_eq!(df.num_rows(), 2);
    assert_eq!(df.series("date64").unwrap().dtype().unwrap().id(), TypeId::TimestampMilliseconds);
    assert_eq!(df.series("time_s").unwrap().dtype().unwrap().id(), TypeId::DurationSeconds);
    assert_eq!(df.series("time_ms").unwrap().dtype().unwrap().id(), TypeId::DurationMilliseconds);
    assert_eq!(df.series("time_us").unwrap().dtype().unwrap().id(), TypeId::DurationMicroseconds);
    assert_eq!(df.series("time_ns").unwrap().dtype().unwrap().id(), TypeId::DurationNanoseconds);

    let materialized =
        cudf::frame::dataframe::GpuRecordBatch::new(df, Arc::clone(&schema)).unwrap().to_host_record_batch().unwrap();
    assert_eq!(materialized.schema().as_ref(), schema.as_ref());
    assert_eq!(
        materialized.column(0).as_primitive::<arrow::datatypes::Date64Type>().values().as_ref(),
        &[1_577_836_800_000, 1_580_428_800_000]
    );
    assert_eq!(
        materialized.column(1).as_primitive::<arrow::datatypes::Time32SecondType>().values().as_ref(),
        &[1, 86_399]
    );
    assert_eq!(
        materialized.column(2).as_primitive::<arrow::datatypes::Time32MillisecondType>().values().as_ref(),
        &[1_000, 86_399_000]
    );
    assert_eq!(
        materialized.column(3).as_primitive::<arrow::datatypes::Time64MicrosecondType>().values().as_ref(),
        &[1_000_000, 86_399_000_000]
    );
    assert_eq!(
        materialized.column(4).as_primitive::<arrow::datatypes::Time64NanosecondType>().values().as_ref(),
        &[1_000_000_000, 86_399_000_000_000]
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
fn test_from_record_batch_string_slice_import_uses_logical_range() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::Utf8, true)]);
    let source: Arc<dyn Array> =
        Arc::new(StringArray::from(vec![Some("skip"), Some("kept"), None, Some("also-kept"), Some("tail")]));
    let batch = RecordBatch::try_new(Arc::new(schema), vec![source.slice(1, 3)]).unwrap();

    let df = import_record_batch(&batch).unwrap();
    assert_eq!(
        df.series("s").unwrap().to_optional_strings().unwrap(),
        vec![Some("kept".to_string()), None, Some("also-kept".to_string())]
    );
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
fn test_from_record_batch_large_string_slice_import_uses_logical_range() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::LargeUtf8, true)]);
    let source: Arc<dyn Array> =
        Arc::new(LargeStringArray::from(vec![Some("skip"), Some("wide"), None, Some("also-wide"), Some("tail")]));
    let batch = RecordBatch::try_new(Arc::new(schema), vec![source.slice(1, 3)]).unwrap();

    let df = import_record_batch(&batch).unwrap();
    assert_eq!(
        df.series("s").unwrap().to_optional_strings().unwrap(),
        vec![Some("wide".to_string()), None, Some("also-wide".to_string())]
    );
}

#[test]
fn test_from_record_batch_string_view_array() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::Utf8View, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringViewArray::from_iter([Some("x"), None, Some("longer-than-inline")]))],
    )
    .unwrap();

    let df = import_record_batch(&batch).unwrap();
    let cv = df.series("s").unwrap();
    let validity = cv.testing_view().unwrap().to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);
    assert_eq!(
        cv.to_optional_strings().unwrap(),
        vec![Some("x".to_string()), None, Some("longer-than-inline".to_string())]
    );
}

#[test]
fn test_from_record_batch_string_view_slice_import_uses_logical_range() {
    let schema = Schema::new(vec![Field::new("s", ArrowDataType::Utf8View, true)]);
    let source: Arc<dyn Array> =
        Arc::new(StringViewArray::from_iter([Some("skip"), Some("x"), None, Some("longer-than-inline"), Some("tail")]));
    let batch = RecordBatch::try_new(Arc::new(schema), vec![source.slice(1, 3)]).unwrap();

    let df = import_record_batch(&batch).unwrap();
    assert_eq!(
        df.series("s").unwrap().to_optional_strings().unwrap(),
        vec![Some("x".to_string()), None, Some("longer-than-inline".to_string())]
    );
}

#[test]
fn test_gpu_record_batch_accepts_utf8_view_schema_contract() {
    let frame = crate::common::dataframe_from_columns(
        vec!["s".to_string()],
        vec![column_from_strs_nullable(&["x", "", "longer-than-inline"], &[true, false, true]).unwrap()],
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("s", ArrowDataType::Utf8View, true)]));

    let gpu_batch = cudf::frame::dataframe::GpuRecordBatch::new(frame, Arc::clone(&schema)).unwrap();
    let materialized = gpu_batch.to_host_record_batch().unwrap();

    assert_eq!(materialized.schema(), schema);
    assert_eq!(materialized.column(0).data_type(), &ArrowDataType::Utf8View);
    let strings = materialized.column(0).as_string_view();
    assert_eq!(strings.iter().collect::<Vec<_>>(), vec![Some("x"), None, Some("longer-than-inline")]);
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

// ---------------------------------------------------------------------------
// Batched pinned-staged import: DataFrame::from_record_batches
// ---------------------------------------------------------------------------

fn batches_from_ranges(ranges: &[std::ops::Range<i64>]) -> Vec<RecordBatch> {
    ranges
        .iter()
        .map(|range| {
            let ids = Int64Array::from_iter_values(range.clone());
            let scores = Float64Array::from_iter_values(range.clone().map(|value| value as f64 * 0.5));
            RecordBatch::try_from_iter(vec![("id", Arc::new(ids) as ArrayRef), ("score", Arc::new(scores) as ArrayRef)])
                .unwrap()
        })
        .collect()
}

#[test]
fn test_from_record_batches_matches_per_batch_import() {
    let runtime = crate::common::sample_runtime();
    let batches = batches_from_ranges(&[0..3, 3..8, 8..15]);

    let staged = DataFrame::from_record_batches(&runtime, &batches).unwrap();

    assert_eq!(staged.num_rows(), 15);
    assert_eq!(staged.column_names(), vec!["id", "score"]);
    let round_trip = staged.to_record_batch().unwrap();
    let ids = round_trip.column(0).as_primitive::<arrow::datatypes::Int64Type>();
    assert_eq!(ids.values(), &(0..15).collect::<Vec<i64>>()[..]);
    let scores = round_trip.column(1).as_primitive::<arrow::datatypes::Float64Type>();
    assert_eq!(scores.values(), &(0..15).map(|value| value as f64 * 0.5).collect::<Vec<f64>>()[..]);
}

#[test]
fn test_from_record_batches_supports_sliced_batches() {
    let runtime = crate::common::sample_runtime();
    let full = batches_from_ranges(std::slice::from_ref(&(0..10))).remove(0);
    let batches = vec![full.slice(1, 4), full.slice(6, 3)];

    let staged = DataFrame::from_record_batches(&runtime, &batches).unwrap();

    let round_trip = staged.to_record_batch().unwrap();
    let ids = round_trip.column(0).as_primitive::<arrow::datatypes::Int64Type>();
    assert_eq!(ids.values(), &[1, 2, 3, 4, 6, 7, 8][..]);
}

#[test]
fn test_from_record_batches_crosses_staging_slab_boundaries() {
    let runtime = crate::common::sample_runtime();
    // 3 batches x 1.5M rows x 8 bytes = 36 MB total, crossing the default
    // 16 MiB staging slab several times within and across batches.
    let per_batch: i64 = 1_500_000;
    let batches: Vec<RecordBatch> = (0..3)
        .map(|batch_index| {
            let start = batch_index * per_batch;
            let ids = Int64Array::from_iter_values(start..start + per_batch);
            RecordBatch::try_from_iter(vec![("id", Arc::new(ids) as ArrayRef)]).unwrap()
        })
        .collect();

    let staged = DataFrame::from_record_batches(&runtime, &batches).unwrap();

    assert_eq!(staged.num_rows(), 3 * per_batch as usize);
    let series = staged.series("id").unwrap();
    assert_eq!(series.min_value::<i64>().unwrap(), Some(0));
    assert_eq!(series.max_value::<i64>().unwrap(), Some(3 * per_batch - 1));
    assert_eq!(series.sum_value::<i64>().unwrap(), (0..3 * per_batch).sum::<i64>());
}

#[test]
fn test_from_record_batches_nullable_and_string_fallback() {
    let runtime = crate::common::sample_runtime();
    let batch_one = RecordBatch::try_from_iter_with_nullable(vec![
        ("n", Arc::new(Int64Array::from(vec![Some(1), None])) as ArrayRef, true),
        ("s", Arc::new(StringArray::from(vec![Some("a"), Some("b")])) as ArrayRef, true),
    ])
    .unwrap();
    let batch_two = RecordBatch::try_from_iter_with_nullable(vec![
        ("n", Arc::new(Int64Array::from(vec![Some(3)])) as ArrayRef, true),
        ("s", Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef, true),
    ])
    .unwrap();

    let staged = DataFrame::from_record_batches(&runtime, &[batch_one, batch_two]).unwrap();

    let round_trip = staged.to_record_batch().unwrap();
    let numbers = round_trip.column(0).as_primitive::<arrow::datatypes::Int64Type>();
    assert_eq!(numbers.iter().collect::<Vec<_>>(), vec![Some(1), None, Some(3)]);
    let strings = round_trip.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(strings.iter().collect::<Vec<_>>(), vec![Some("a"), Some("b"), None]);
}

#[test]
fn test_from_record_batches_rejects_empty_input() {
    let runtime = crate::common::sample_runtime();

    let error = DataFrame::from_record_batches(&runtime, &[]).unwrap_err();

    assert_eq!(error.code(), ErrorCode::EmptyInput);
}

#[test]
fn test_from_record_batches_rejects_schema_mismatch() {
    let runtime = crate::common::sample_runtime();
    let ints = RecordBatch::try_from_iter(vec![("v", Arc::new(Int64Array::from(vec![1i64])) as ArrayRef)]).unwrap();
    let floats =
        RecordBatch::try_from_iter(vec![("v", Arc::new(Float64Array::from(vec![1.0f64])) as ArrayRef)]).unwrap();

    let error = DataFrame::from_record_batches(&runtime, &[ints, floats]).unwrap_err();

    assert_eq!(error.code(), ErrorCode::TypeMismatch);
}
