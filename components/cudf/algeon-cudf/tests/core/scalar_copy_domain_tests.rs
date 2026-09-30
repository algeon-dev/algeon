//! Tests for `Scalar::copy_into_domain` via the public
//! `Runtime::series_from_scalar` API.
//!
//! `series_from_scalar` calls `Scalar::copy_into_domain` internally, so these
//! tests exercise the generic `cudf_scalar_copy_ex` C++ shim across fixed-width,
//! string, decimal, list, and struct scalar families.

use std::collections::HashMap;

use arrow::array::ArrayRef;
use cudf::{
    Runtime, Series, TimestampResolution,
    core::{DataType, TypeId},
    testing::prelude::*,
};
use rapids_interop::{
    BufferBinding, ColumnLayout, PhysicalValidity, PreparedExportProfile, SemanticColumn, TypeId as InteropTypeId,
};

use crate::common::{
    decimal32_raw_scalar, decimal128_raw_scalar, dependency_wait_call_count_for_stream, duration_raw_scalar,
    int64_array, int64_list_array, last_exec_mr_value, last_exec_stream_value, lock_event_observation,
    nested_dataframe, nullable_series, reset_dependency_observation, reset_exec_observation, run_named_cases,
    sample_runtime as common_sample_runtime, scalar, series as common_series, string_scalar, struct_array,
    timestamp_raw_scalar, utf8_array, with_test_runtime,
};

fn sample_runtime() -> Runtime {
    Runtime::builder().build().unwrap()
}

fn binding_bytes(binding: &BufferBinding<'_>) -> Vec<u8> {
    // SAFETY: `base_ptr()` is valid for `byte_len()` bytes and the binding's lifetime
    // guarantees the underlying buffer is not freed while the slice is live.
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr() as *const u8, binding.bytes().byte_len()) }.to_vec()
}

fn binding_map<'a>(bindings: &'a [BufferBinding<'a>]) -> HashMap<u32, &'a BufferBinding<'a>> {
    bindings.iter().map(|binding| (binding.id().get(), binding)).collect()
}

fn semantic_column_map(columns: &[SemanticColumn]) -> HashMap<u32, &SemanticColumn> {
    columns.iter().map(|column| (column.id().get(), column)).collect()
}

fn assert_binding_equivalent(source: &BufferBinding<'_>, copied: &BufferBinding<'_>) {
    assert_eq!(source.placement(), copied.placement());
    assert_eq!(binding_bytes(source), binding_bytes(copied));
}

fn validity_bits(column: &SemanticColumn, bindings: &HashMap<u32, &BufferBinding<'_>>) -> Option<Vec<bool>> {
    match column.validity() {
        PhysicalValidity::NoValidityBuffer => None,
        PhysicalValidity::ValidityMask { mask, .. } => {
            let bytes = binding_bytes(bindings.get(&mask.get()).expect("missing validity binding"));
            Some((0..column.len()).map(|i| (bytes[i / 8] >> (i % 8)) & 1 != 0).collect())
        },
    }
}

fn assert_fixed_width_binding_equivalent(
    source: &SemanticColumn,
    copied: &SemanticColumn,
    source_values: &BufferBinding<'_>,
    copied_values: &BufferBinding<'_>,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    copied_bindings: &HashMap<u32, &BufferBinding<'_>>,
) {
    assert_eq!(source_values.placement(), copied_values.placement());

    let source_bytes = binding_bytes(source_values);
    let copied_bytes = binding_bytes(copied_values);
    assert_eq!(source_bytes.len(), copied_bytes.len());

    let elem_size = match source.data_type().id() {
        InteropTypeId::Int8 | InteropTypeId::UInt8 | InteropTypeId::Bool8 => 1,
        InteropTypeId::Int16 | InteropTypeId::UInt16 => 2,
        InteropTypeId::Int32
        | InteropTypeId::UInt32
        | InteropTypeId::Float32
        | InteropTypeId::TimestampDays
        | InteropTypeId::DurationDays
        | InteropTypeId::Decimal32 => 4,
        InteropTypeId::Int64
        | InteropTypeId::UInt64
        | InteropTypeId::Float64
        | InteropTypeId::TimestampSeconds
        | InteropTypeId::TimestampMilliseconds
        | InteropTypeId::TimestampMicroseconds
        | InteropTypeId::TimestampNanoseconds
        | InteropTypeId::DurationSeconds
        | InteropTypeId::DurationMilliseconds
        | InteropTypeId::DurationMicroseconds
        | InteropTypeId::DurationNanoseconds
        | InteropTypeId::Decimal64 => 8,
        InteropTypeId::Decimal128 => 16,
        other => panic!("fixed-width layout requires a fixed-width type, got {other:?}"),
    };
    assert_eq!(source_bytes.len(), source.len() * elem_size);

    let validity = validity_bits(source, source_bindings)
        .or_else(|| validity_bits(copied, copied_bindings))
        .unwrap_or_else(|| vec![true; source.len()]);

    for (index, valid) in validity.into_iter().enumerate() {
        if !valid {
            continue;
        }
        let start = index * elem_size;
        let end = start + elem_size;
        assert_eq!(source_bytes[start..end], copied_bytes[start..end], "fixed-width bytes differ at valid row {index}");
    }
}

fn validity_mask_is_elidable(validity: &PhysicalValidity) -> bool {
    matches!(validity, PhysicalValidity::ValidityMask { null_count: Some(0), .. })
}

fn assert_validity_equivalent(
    source: &PhysicalValidity,
    copied: &PhysicalValidity,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    copied_bindings: &HashMap<u32, &BufferBinding<'_>>,
) {
    match (source, copied) {
        (PhysicalValidity::NoValidityBuffer, PhysicalValidity::NoValidityBuffer) => {},
        (
            PhysicalValidity::ValidityMask { mask: source_mask, null_count: source_null_count },
            PhysicalValidity::ValidityMask { mask: copied_mask, null_count: copied_null_count },
        ) => {
            assert_eq!(source_null_count, copied_null_count);
            assert_binding_equivalent(
                source_bindings.get(&source_mask.get()).expect("missing source validity binding"),
                copied_bindings.get(&copied_mask.get()).expect("missing copied validity binding"),
            );
        },
        (PhysicalValidity::ValidityMask { .. }, PhysicalValidity::NoValidityBuffer) => {
            assert!(validity_mask_is_elidable(source));
        },
        (PhysicalValidity::NoValidityBuffer, PhysicalValidity::ValidityMask { .. }) => {
            assert!(validity_mask_is_elidable(copied));
        },
    }
}

fn assert_semantic_column_equivalent(
    source: &SemanticColumn,
    copied: &SemanticColumn,
    source_columns: &HashMap<u32, &SemanticColumn>,
    copied_columns: &HashMap<u32, &SemanticColumn>,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    copied_bindings: &HashMap<u32, &BufferBinding<'_>>,
) {
    assert_eq!(source.data_type(), copied.data_type());
    assert_eq!(source.len(), copied.len());
    assert_eq!(source.offset(), copied.offset());
    assert_validity_equivalent(source.validity(), copied.validity(), source_bindings, copied_bindings);

    match (source.layout(), copied.layout()) {
        (ColumnLayout::FixedWidth { values: source_values }, ColumnLayout::FixedWidth { values: copied_values }) => {
            assert_fixed_width_binding_equivalent(
                source,
                copied,
                source_bindings.get(&source_values.get()).expect("missing source fixed-width binding"),
                copied_bindings.get(&copied_values.get()).expect("missing copied fixed-width binding"),
                source_bindings,
                copied_bindings,
            );
        },
        (
            ColumnLayout::BinaryLike { offsets: source_offsets, data: source_data },
            ColumnLayout::BinaryLike { offsets: copied_offsets, data: copied_data },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_offsets.get()).expect("missing source binary offsets binding"),
                copied_bindings.get(&copied_offsets.get()).expect("missing copied binary offsets binding"),
            );
            assert_binding_equivalent(
                source_bindings.get(&source_data.get()).expect("missing source binary data binding"),
                copied_bindings.get(&copied_data.get()).expect("missing copied binary data binding"),
            );
        },
        (
            ColumnLayout::List { offsets: source_offsets, child: source_child },
            ColumnLayout::List { offsets: copied_offsets, child: copied_child },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_offsets.get()).expect("missing source list offsets binding"),
                copied_bindings.get(&copied_offsets.get()).expect("missing copied list offsets binding"),
            );
            assert_semantic_column_equivalent(
                source_columns.get(&source_child.get()).expect("missing source list child"),
                copied_columns.get(&copied_child.get()).expect("missing copied list child"),
                source_columns,
                copied_columns,
                source_bindings,
                copied_bindings,
            );
        },
        (ColumnLayout::Struct { children: source_children }, ColumnLayout::Struct { children: copied_children }) => {
            assert_eq!(source_children.len(), copied_children.len());
            for (source_child, copied_child) in source_children.iter().zip(copied_children.iter()) {
                assert_semantic_column_equivalent(
                    source_columns.get(&source_child.get()).expect("missing source struct child"),
                    copied_columns.get(&copied_child.get()).expect("missing copied struct child"),
                    source_columns,
                    copied_columns,
                    source_bindings,
                    copied_bindings,
                );
            }
        },
        _ => panic!("layout mismatch: source={:?} copied={:?}", source.layout(), copied.layout()),
    }
}

fn assert_series_host_export_equivalent(expected: &Series, actual: &Series) {
    let expected_export = expected.try_prepare_host_column_export().unwrap();
    let actual_export = actual.try_prepare_host_column_export().unwrap();

    assert_eq!(expected_export.profile(), PreparedExportProfile::Host);
    assert_eq!(actual_export.profile(), PreparedExportProfile::Host);

    let expected_columns = semantic_column_map(expected_export.semantic_columns());
    let actual_columns = semantic_column_map(actual_export.semantic_columns());
    let expected_bindings = binding_map(expected_export.bindings());
    let actual_bindings = binding_map(actual_export.bindings());

    assert_semantic_column_equivalent(
        expected_export.column(),
        actual_export.column(),
        &expected_columns,
        &actual_columns,
        &expected_bindings,
        &actual_bindings,
    );
}

fn nested_series(rt: &Runtime, name: &str, array: ArrayRef) -> Series {
    nested_dataframe(rt, vec![(name, array)]).series(name).unwrap()
}

// ── Timestamp / Duration ────────────────────────────────────────────

fn copy_into_domain_timestamp_microseconds() {
    let rt = sample_runtime();
    let s = common_series::<i64>("t", &[1_000_000]).unwrap();
    let dur = s.astype(TypeId::DurationMicroseconds).unwrap();
    let ts = dur.astype(TypeId::TimestampMicroseconds).unwrap();
    let scalar = ts.min().unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = common_series::<i64>("out", &[1_000_000, 1_000_000])
        .unwrap()
        .astype(TypeId::DurationMicroseconds)
        .unwrap()
        .astype(TypeId::TimestampMicroseconds)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_timestamp_days() {
    let rt = sample_runtime();
    let s = common_series::<i32>("t", &[100]).unwrap();
    let dur = s.astype(TypeId::DurationDays).unwrap();
    let ts = dur.astype(TypeId::TimestampDays).unwrap();
    let scalar = ts.min().unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = common_series::<i32>("out", &[100, 100])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_duration_seconds() {
    let rt = sample_runtime();
    let s = common_series::<i64>("d", &[42]).unwrap();
    let dur = s.astype(TypeId::DurationSeconds).unwrap();
    let scalar = dur.min().unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = common_series::<i64>("out", &[42, 42]).unwrap().astype(TypeId::DurationSeconds).unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_null_timestamp() {
    let rt = sample_runtime();
    // Create a nullable timestamp column where the only element is null, so min() is null
    let s = nullable_series::<i64>("t", &[0], &[false]).unwrap();
    let ts = s.astype(TypeId::DurationMicroseconds).unwrap().astype(TypeId::TimestampMicroseconds).unwrap();
    let scalar = ts.min().unwrap();

    // series_from_scalar with a null scalar produces a column of all nulls
    let result = rt.series_from_scalar(&scalar, 3).unwrap();
    let expected = nullable_series::<i64>("out", &[0, 0, 0], &[false, false, false])
        .unwrap()
        .astype(TypeId::DurationMicroseconds)
        .unwrap()
        .astype(TypeId::TimestampMicroseconds)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_timestamp_raw_scalar_preserves_dtype_and_value() {
    let rt = sample_runtime();
    let scalar = timestamp_raw_scalar(1_700_000_000_123_456, TimestampResolution::Microseconds).unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = common_series::<i64>("out", &[1_700_000_000_123_456, 1_700_000_000_123_456])
        .unwrap()
        .astype(TypeId::DurationMicroseconds)
        .unwrap()
        .astype(TypeId::TimestampMicroseconds)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_timestamp_days_raw_scalar_preserves_dtype_and_value() {
    let rt = sample_runtime();
    let scalar = timestamp_raw_scalar(-12, TimestampResolution::Days).unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = common_series::<i32>("out", &[-12, -12])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_duration_raw_scalar_preserves_dtype_and_value() {
    let rt = sample_runtime();
    let scalar = duration_raw_scalar(-42_000, TimestampResolution::Milliseconds).unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected =
        common_series::<i64>("out", &[-42_000, -42_000]).unwrap().astype(TypeId::DurationMilliseconds).unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn series_from_scalar_cross_runtime_copy_waits_on_target_stream() {
    let _event_observation = lock_event_observation();
    let source_rt = common_sample_runtime();
    let target_rt = common_sample_runtime();
    let scalar = source_rt.scalar(42i32).unwrap();

    assert_ne!(scalar.testing_execution_domain_id(), target_rt.testing_execution_domain_id());

    let expected_exec_stream = target_rt.stream().testing_raw_handle_value();
    let expected_exec_mr = target_rt.memory_resource().testing_raw_handle_value();

    reset_dependency_observation();
    reset_exec_observation();
    let result = target_rt.series_from_scalar(&scalar, 3).unwrap();

    assert_eq!(result.as_column().testing_execution_domain_id(), target_rt.testing_execution_domain_id());
    assert_eq!(result.as_column().testing_execution_domain_stream_value(), expected_exec_stream);
    assert_eq!(result.as_column().testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert!(
        dependency_wait_call_count_for_stream(expected_exec_stream) >= 1,
        "same-device scalar relocation must order target stream after source stream"
    );
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42]);
}

// ── Decimal ─────────────────────────────────────────────────────────

fn copy_into_domain_decimal64() {
    let rt = sample_runtime();
    let s = common_series::<i64>("d", &[100]).unwrap();
    let dec = s.astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap()).unwrap();
    let scalar = dec.min().unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = common_series::<i64>("out", &[100, 100])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_decimal128() {
    let rt = sample_runtime();
    let s = common_series::<i64>("d", &[200]).unwrap();
    let dec = s.astype(DataType::fixed_point(TypeId::Decimal128, -3).unwrap()).unwrap();
    let scalar = dec.min().unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = common_series::<i64>("out", &[200, 200])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal128, -3).unwrap())
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_decimal32_raw_scalar_preserves_dtype_scale_and_value() {
    let rt = sample_runtime();
    let scalar = decimal32_raw_scalar(12345, -2).unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = with_test_runtime(|source_rt| {
        let seed = source_rt.decimal32_raw_scalar(12345, -2).unwrap();
        source_rt.series_from_scalar(&seed, 2).unwrap()
    });
    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_decimal128_raw_scalar_preserves_dtype_scale_and_value() {
    let rt = sample_runtime();
    let raw = 123456789012345678901234567890i128;
    let scalar = decimal128_raw_scalar(raw, -8).unwrap();

    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = with_test_runtime(|source_rt| {
        let seed = source_rt.decimal128_raw_scalar(raw, -8).unwrap();
        source_rt.series_from_scalar(&seed, 2).unwrap()
    });
    assert_series_host_export_equivalent(&expected, &result);
}

// ── Regression: numeric and string ──────────────────────────────────

fn copy_into_domain_numeric_i32_regression() {
    let rt = sample_runtime();
    let scalar = scalar(42i32).unwrap();

    let result = rt.series_from_scalar(&scalar, 1).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42]);
}

fn copy_into_domain_string_regression() {
    let rt = sample_runtime();
    let scalar = string_scalar("hello").unwrap();

    // count=2 so an offsets-buffer replication bug (e.g. [0,5,0,5] vs [0,5,10]) is caught.
    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    assert_eq!(result.to_strings_strict().unwrap(), vec!["hello".to_string(), "hello".to_string()]);
}

// ── End-to-end: scalar copy → DataFrame round-trip ──────────────────

fn scalar_copy_dataframe_round_trip() {
    let rt = sample_runtime();
    let s = common_series::<i64>("t", &[1_000_000]).unwrap();
    let ts = s.astype(TypeId::DurationMicroseconds).unwrap().astype(TypeId::TimestampMicroseconds).unwrap();
    let scalar = ts.min().unwrap();

    let series = rt.series_from_scalar(&scalar, 3).unwrap();
    let df = cudf::DataFrame::from_series([("ts_col", series)]).unwrap();
    let retrieved = df.series("ts_col").unwrap();

    let expected = common_series::<i64>("ts_col", &[1_000_000, 1_000_000, 1_000_000])
        .unwrap()
        .astype(TypeId::DurationMicroseconds)
        .unwrap()
        .astype(TypeId::TimestampMicroseconds)
        .unwrap();
    assert_eq!(retrieved.len(), 3);
    assert_eq!(retrieved.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
    assert_series_host_export_equivalent(&expected, &retrieved);
}

// ── List / Struct ────────────────────────────────────────────────────

fn copy_into_domain_list_scalar() {
    let rt = sample_runtime();
    let list_series = nested_series(&rt, "nested", int64_list_array(&[vec![1, 2], vec![3]]));

    // libcudf min() on list columns returns the lexicographically smallest element.
    let scalar = list_series.min().unwrap();
    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = nested_series(&rt, "nested", int64_list_array(&[vec![1, 2], vec![1, 2]]));

    assert_series_host_export_equivalent(&expected, &result);
}

fn copy_into_domain_struct_scalar() {
    let rt = sample_runtime();
    let struct_series = nested_series(
        &rt,
        "point",
        struct_array(vec![("id", int64_array(&[1, 2])), ("label", utf8_array(&["one", "two"]))]),
    );

    // libcudf min() on struct columns compares fields in order (id first, then label).
    let scalar = struct_series.min().unwrap();
    let result = rt.series_from_scalar(&scalar, 2).unwrap();
    let expected = nested_series(
        &rt,
        "point",
        struct_array(vec![("id", int64_array(&[1, 1])), ("label", utf8_array(&["one", "one"]))]),
    );

    assert_series_host_export_equivalent(&expected, &result);
}

#[test]
fn scalar_copy_domain_tests_cases() {
    run_named_cases(&[
        ("copy_into_domain_timestamp_microseconds", copy_into_domain_timestamp_microseconds),
        ("copy_into_domain_timestamp_days", copy_into_domain_timestamp_days),
        ("copy_into_domain_duration_seconds", copy_into_domain_duration_seconds),
        ("copy_into_domain_null_timestamp", copy_into_domain_null_timestamp),
        (
            "copy_into_domain_timestamp_raw_scalar_preserves_dtype_and_value",
            copy_into_domain_timestamp_raw_scalar_preserves_dtype_and_value,
        ),
        (
            "copy_into_domain_timestamp_days_raw_scalar_preserves_dtype_and_value",
            copy_into_domain_timestamp_days_raw_scalar_preserves_dtype_and_value,
        ),
        (
            "copy_into_domain_duration_raw_scalar_preserves_dtype_and_value",
            copy_into_domain_duration_raw_scalar_preserves_dtype_and_value,
        ),
        (
            "series_from_scalar_cross_runtime_copy_waits_on_target_stream",
            series_from_scalar_cross_runtime_copy_waits_on_target_stream,
        ),
        ("copy_into_domain_decimal64", copy_into_domain_decimal64),
        ("copy_into_domain_decimal128", copy_into_domain_decimal128),
        (
            "copy_into_domain_decimal32_raw_scalar_preserves_dtype_scale_and_value",
            copy_into_domain_decimal32_raw_scalar_preserves_dtype_scale_and_value,
        ),
        (
            "copy_into_domain_decimal128_raw_scalar_preserves_dtype_scale_and_value",
            copy_into_domain_decimal128_raw_scalar_preserves_dtype_scale_and_value,
        ),
        ("copy_into_domain_numeric_i32_regression", copy_into_domain_numeric_i32_regression),
        ("copy_into_domain_string_regression", copy_into_domain_string_regression),
        ("scalar_copy_dataframe_round_trip", scalar_copy_dataframe_round_trip),
        ("copy_into_domain_list_scalar", copy_into_domain_list_scalar),
        ("copy_into_domain_struct_scalar", copy_into_domain_struct_scalar),
    ]);
}
