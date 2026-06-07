//! Tests for `Scalar::copy_into_domain` via the public
//! `Runtime::series_from_scalar` API.
//!
//! `series_from_scalar` calls `Scalar::copy_into_domain` internally, so these
//! tests exercise the generic `cudf_scalar_copy_ex` C++ shim across fixed-width,
//! string, decimal, list, and struct scalar families.

use std::collections::HashMap;

use cudf::{
    JsonReadOptions, Runtime, Series,
    core::{DataType, Scalar, TypeId},
    interop::{BufferBinding, ColumnLayout, ExportProfile, PhysicalValidity, SemanticColumn},
};
use tempfile::TempDir;

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

fn semantic_column_map<'a>(columns: &'a [SemanticColumn]) -> HashMap<u32, &'a SemanticColumn> {
    columns.iter().map(|column| (column.id().get(), column)).collect()
}

fn assert_binding_equivalent(source: &BufferBinding<'_>, copied: &BufferBinding<'_>) {
    assert_eq!(source.placement(), copied.placement());
    assert_eq!(binding_bytes(source), binding_bytes(copied));
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

fn schema_nullability_is_elidable(source: &SemanticColumn, copied: &SemanticColumn) -> bool {
    match (source.schema_nullability(), copied.schema_nullability()) {
        (cudf::interop::SchemaNullability::Nullable, cudf::interop::SchemaNullability::NonNullable) => {
            validity_mask_is_elidable(source.validity())
                && matches!(copied.validity(), PhysicalValidity::NoValidityBuffer)
        },
        (cudf::interop::SchemaNullability::NonNullable, cudf::interop::SchemaNullability::Nullable) => {
            matches!(source.validity(), PhysicalValidity::NoValidityBuffer)
                && validity_mask_is_elidable(copied.validity())
        },
        _ => false,
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
    if source.schema_nullability() != copied.schema_nullability() {
        assert!(
            schema_nullability_is_elidable(source, copied),
            "schema nullability mismatch: source={source:?} copied={copied:?}"
        );
    }
    assert_validity_equivalent(source.validity(), copied.validity(), source_bindings, copied_bindings);

    match (source.layout(), copied.layout()) {
        (ColumnLayout::FixedWidth { values: source_values }, ColumnLayout::FixedWidth { values: copied_values }) => {
            assert_binding_equivalent(
                source_bindings.get(&source_values.get()).expect("missing source fixed-width binding"),
                copied_bindings.get(&copied_values.get()).expect("missing copied fixed-width binding"),
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

    assert_eq!(expected_export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(actual_export.diagnostics().profile(), ExportProfile::PreparedHostExport);

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

fn read_json_series(rt: &Runtime, filename: &str, rows: &str, column_name: &str) -> Series {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(filename);
    std::fs::write(&path, rows).unwrap();
    let frame =
        cudf::DataFrame::read_json_with_options(JsonReadOptions::builder().filepath(&path).with_runtime(rt).build())
            .unwrap();
    frame.series(column_name).unwrap()
}

// ── Timestamp / Duration ────────────────────────────────────────────

#[test]
fn copy_into_domain_timestamp_microseconds() {
    let rt = sample_runtime();
    let s = Series::new::<i64>("t", &[1_000_000]).unwrap();
    let dur = s.astype(TypeId::DurationMicroseconds).unwrap();
    let ts = dur.astype(TypeId::TimestampMicroseconds).unwrap();
    let scalar = ts.min().unwrap();

    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    let expected = Series::new::<i64>("out", &[1_000_000, 1_000_000])
        .unwrap()
        .astype(TypeId::DurationMicroseconds)
        .unwrap()
        .astype(TypeId::TimestampMicroseconds)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

#[test]
fn copy_into_domain_timestamp_days() {
    let rt = sample_runtime();
    let s = Series::new::<i32>("t", &[100]).unwrap();
    let dur = s.astype(TypeId::DurationDays).unwrap();
    let ts = dur.astype(TypeId::TimestampDays).unwrap();
    let scalar = ts.min().unwrap();

    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    let expected = Series::new::<i32>("out", &[100, 100])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

#[test]
fn copy_into_domain_duration_seconds() {
    let rt = sample_runtime();
    let s = Series::new::<i64>("d", &[42]).unwrap();
    let dur = s.astype(TypeId::DurationSeconds).unwrap();
    let scalar = dur.min().unwrap();

    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    let expected = Series::new::<i64>("out", &[42, 42]).unwrap().astype(TypeId::DurationSeconds).unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

#[test]
fn copy_into_domain_null_timestamp() {
    let rt = sample_runtime();
    // Create a nullable timestamp column where the only element is null, so min() is null
    let s = Series::new_nullable::<i64>("t", &[0], &[false]).unwrap();
    let ts = s.astype(TypeId::DurationMicroseconds).unwrap().astype(TypeId::TimestampMicroseconds).unwrap();
    let scalar = ts.min().unwrap();

    // series_from_scalar with a null scalar produces a column of all nulls
    let result = rt.series_from_scalar("out", &scalar, 3).unwrap();
    let expected = Series::new_nullable::<i64>("out", &[0, 0, 0], &[false, false, false])
        .unwrap()
        .astype(TypeId::DurationMicroseconds)
        .unwrap()
        .astype(TypeId::TimestampMicroseconds)
        .unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

// ── Decimal ─────────────────────────────────────────────────────────

#[test]
fn copy_into_domain_decimal64() {
    let rt = sample_runtime();
    let s = Series::new::<i64>("d", &[100]).unwrap();
    let dec = s.astype(DataType::fixed_point(TypeId::Decimal64, -2)).unwrap();
    let scalar = dec.min().unwrap();

    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    let expected =
        Series::new::<i64>("out", &[100, 100]).unwrap().astype(DataType::fixed_point(TypeId::Decimal64, -2)).unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

#[test]
fn copy_into_domain_decimal128() {
    let rt = sample_runtime();
    let s = Series::new::<i64>("d", &[200]).unwrap();
    let dec = s.astype(DataType::fixed_point(TypeId::Decimal128, -3)).unwrap();
    let scalar = dec.min().unwrap();

    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    let expected =
        Series::new::<i64>("out", &[200, 200]).unwrap().astype(DataType::fixed_point(TypeId::Decimal128, -3)).unwrap();
    assert_series_host_export_equivalent(&expected, &result);
}

// ── Regression: numeric and string ──────────────────────────────────

#[test]
fn copy_into_domain_numeric_i32_regression() {
    let rt = sample_runtime();
    let scalar = Scalar::from_value(42i32).unwrap();

    let result = rt.series_from_scalar("out", &scalar, 1).unwrap();
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![42]);
}

#[test]
fn copy_into_domain_string_regression() {
    let rt = sample_runtime();
    let scalar = Scalar::from_string("hello").unwrap();

    // count=2 so an offsets-buffer replication bug (e.g. [0,5,0,5] vs [0,5,10]) is caught.
    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    assert_eq!(result.to_strings().unwrap(), vec!["hello".to_string(), "hello".to_string()]);
}

// ── End-to-end: scalar copy → DataFrame round-trip ──────────────────

#[test]
fn scalar_copy_dataframe_round_trip() {
    let rt = sample_runtime();
    let s = Series::new::<i64>("t", &[1_000_000]).unwrap();
    let ts = s.astype(TypeId::DurationMicroseconds).unwrap().astype(TypeId::TimestampMicroseconds).unwrap();
    let scalar = ts.min().unwrap();

    let series = rt.series_from_scalar("ts_col", &scalar, 3).unwrap();
    let df = cudf::DataFrame::from_series([("ts_col", series)]).unwrap();
    let retrieved = df.series("ts_col").unwrap();

    let expected = Series::new::<i64>("ts_col", &[1_000_000, 1_000_000, 1_000_000])
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

#[test]
fn copy_into_domain_list_scalar() {
    let rt = sample_runtime();
    let list_series =
        read_json_series(&rt, "scalar_copy_list.jsonl", "{\"nested\":[1,2]}\n{\"nested\":[3]}\n", "nested");

    // libcudf min() on list columns returns the lexicographically smallest element.
    let scalar = list_series.min().unwrap();
    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    let expected =
        read_json_series(&rt, "scalar_copy_list_expected.jsonl", "{\"nested\":[1,2]}\n{\"nested\":[1,2]}\n", "nested")
            .rename("out");

    assert_series_host_export_equivalent(&expected, &result);
}

#[test]
fn copy_into_domain_struct_scalar() {
    let rt = sample_runtime();
    let struct_series = read_json_series(
        &rt,
        "scalar_copy_struct.jsonl",
        "{\"point\":{\"id\":1,\"label\":\"one\"}}\n{\"point\":{\"id\":2,\"label\":\"two\"}}\n",
        "point",
    );

    // libcudf min() on struct columns compares fields in order (id first, then label).
    let scalar = struct_series.min().unwrap();
    let result = rt.series_from_scalar("out", &scalar, 2).unwrap();
    let expected = read_json_series(
        &rt,
        "scalar_copy_struct_expected.jsonl",
        "{\"point\":{\"id\":1,\"label\":\"one\"}}\n{\"point\":{\"id\":1,\"label\":\"one\"}}\n",
        "point",
    )
    .rename("out");

    assert_series_host_export_equivalent(&expected, &result);
}
