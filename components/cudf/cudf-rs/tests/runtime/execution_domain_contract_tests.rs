//! Contract tests for execution-domain propagation.
//!
//! The execution-domain substrate underpins the public `Runtime v1` surface.
//! These tests use `__debug_` observation hooks to validate internal
//! domain propagation and synchronization behavior at the runtime layer.

use std::{ffi::c_void, ptr};

use cudf::{
    AggSpec, DataFrame, DataType, ErrorKind, GroupByAgg, Series, TypeId,
    core::{Column, Scalar, Table, TableView, memory::DeviceMemoryResource},
    ops::{
        binary::{self, BinaryOperator},
        concatenate, copying, dictionary,
        expr::lit,
        filter,
        groupby::{self, AggregationRequest},
        hashing, join,
        reduction::{self, ReduceAgg, ScanAgg, ScanType},
        replace, search, sorting, stream_compaction, strings,
        unary::{self, UnaryOperator},
    },
    runtime::{CudaStream, ExecutionContext},
};
use cudf_sys::{
    cudf_test_current_device_resource_raw, cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value,
    cudf_test_exec_last_sync_stream_value,
};

use crate::common::{explicit_i32_column, reset_exec_observation, reset_sync_observation, sample_runtime};

fn last_exec_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec stream observation");
    out
}

fn last_exec_mr_value() -> usize {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec mr observation");
    out as usize
}

fn current_device_resource_value() -> usize {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read current device resource");
    out as usize
}

fn last_sync_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last synchronized stream observation");
    out
}

fn explicit_i32_series(name: &str, data: &[i32], identity: u64) -> Series {
    Series::from_column(Column::from_slice(data).unwrap().__debug_rebind_execution_domain_explicit(identity).unwrap())
        .unwrap()
        .rename(name)
}

fn explicit_bool_series(name: &str, data: &[bool], identity: u64) -> Series {
    Series::from_column(Column::from_slice(data).unwrap().__debug_rebind_execution_domain_explicit(identity).unwrap())
        .unwrap()
        .rename(name)
}

fn explicit_nullable_bool_series(name: &str, data: &[bool], validity: &[bool], identity: u64) -> Series {
    Series::from_column(
        Column::from_slice_nullable(data, validity)
            .unwrap()
            .__debug_rebind_execution_domain_explicit(identity)
            .unwrap(),
    )
    .unwrap()
    .rename(name)
}

fn explicit_str_series(name: &str, data: &[&str], identity: u64) -> Series {
    Series::from_column(Column::from_strs(data).unwrap().__debug_rebind_execution_domain_explicit(identity).unwrap())
        .unwrap()
        .rename(name)
}

fn explicit_nullable_i32_series(name: &str, data: &[i32], validity: &[bool], identity: u64) -> Series {
    Series::from_column(
        Column::from_slice_nullable(data, validity)
            .unwrap()
            .__debug_rebind_execution_domain_explicit(identity)
            .unwrap(),
    )
    .unwrap()
    .rename(name)
}

fn nullable_i32_table_column(table: &Table, index: usize) -> Vec<Option<i32>> {
    let view = table.view().unwrap();
    let column = view.column(index.try_into().unwrap()).unwrap();
    let values = column.to_vec::<i32>().unwrap();
    let validity = column.validity().unwrap().unwrap_or_else(|| vec![true; values.len()]);

    values.into_iter().zip(validity).map(|(value, valid)| if valid { Some(value) } else { None }).collect()
}

#[test]
fn owned_constructors_seed_ambient_default_domain() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    assert_eq!(col.__debug_execution_domain_kind(), "ambient_default");
    assert_eq!(col.__debug_execution_domain_id(), 0);

    let scalar = Scalar::from_value(7i32).unwrap();
    assert_eq!(scalar.__debug_execution_domain_kind(), "ambient_default");
    assert_eq!(scalar.__debug_execution_domain_id(), 0);

    let table = Table::from_columns(vec![Column::from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    assert_eq!(table.__debug_execution_domain_kind(), "ambient_default");
    assert_eq!(table.__debug_execution_domain_id(), 0);

    let series = Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([series]).unwrap();
    assert_eq!(df.__debug_execution_domain_kind(), "ambient_default");
    assert_eq!(df.__debug_execution_domain_id(), 0);
    assert_eq!(df.series("id").unwrap().as_column().__debug_execution_domain_id(), 0);
}

#[test]
fn execution_domain_debug_hooks_report_consistent_metadata() {
    let runtime = sample_runtime();
    let column = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(40).unwrap();
    let scalar = Scalar::from_value(7i32).unwrap().__debug_rebind_execution_domain_explicit(41).unwrap();
    let table = Table::from_columns(vec![column.deep_copy().unwrap()]).unwrap();
    let df = DataFrame::from_named_pairs([("id", Series::from_column(column.deep_copy().unwrap()).unwrap())]).unwrap();

    assert_eq!(column.__debug_execution_domain_kind(), "explicit");
    assert_eq!(column.__debug_execution_domain_id(), 40);
    assert_ne!(column.__debug_execution_domain_stream_value(), 0);
    assert_ne!(column.__debug_execution_domain_mr_value(), 0);

    assert_eq!(scalar.__debug_execution_domain_kind(), "explicit");
    assert_eq!(scalar.__debug_execution_domain_id(), 41);
    assert_ne!(scalar.__debug_execution_domain_stream_value(), 0);

    assert_eq!(table.__debug_execution_domain_kind(), "explicit");
    assert_eq!(table.__debug_execution_domain_id(), 40);
    assert_eq!(table.__debug_execution_domain_stream_value(), column.__debug_execution_domain_stream_value());
    assert_eq!(table.__debug_execution_domain_mr_value(), column.__debug_execution_domain_mr_value());

    assert_eq!(df.__debug_execution_domain_kind(), "explicit");
    assert_eq!(df.__debug_execution_domain_id(), 40);
    assert_eq!(df.__debug_execution_domain_stream_value(), column.__debug_execution_domain_stream_value());
    assert_eq!(df.__debug_execution_domain_mr_value(), column.__debug_execution_domain_mr_value());

    assert_ne!(runtime.__debug_execution_domain_id(), 0);
}

#[test]
fn column_deep_copy_preserves_execution_domain_identity() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(41).unwrap();
    let copied = col.deep_copy().unwrap();

    assert_eq!(col.__debug_execution_domain_kind(), "explicit");
    assert_eq!(copied.__debug_execution_domain_kind(), "explicit");
    assert_eq!(copied.__debug_execution_domain_id(), col.__debug_execution_domain_id());
}

#[test]
fn column_deep_copy_uses_retained_explicit_domain_stream_and_mr() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(42).unwrap();
    let expected_stream = col.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let copied = col.deep_copy().unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(copied.__debug_execution_domain_id(), 42);
}

#[test]
fn table_deep_copy_and_release_preserve_execution_domain_identity() {
    let first = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(7).unwrap();
    let second = Column::from_slice(&[10i32, 20, 30]).unwrap().__debug_rebind_execution_domain_explicit(7).unwrap();

    let table = Table::from_columns(vec![first, second]).unwrap();
    let copied = table.deep_copy().unwrap();

    assert_eq!(table.__debug_execution_domain_kind(), "explicit");
    assert_eq!(copied.__debug_execution_domain_id(), table.__debug_execution_domain_id());

    let released = copied.into_columns().unwrap();
    assert_eq!(released.len(), 2);
    assert!(released.iter().all(|col| col.__debug_execution_domain_id() == table.__debug_execution_domain_id()));
}

#[test]
fn table_deep_copy_uses_retained_explicit_domain_stream_and_mr() {
    let first = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(43).unwrap();
    let second = Column::from_slice(&[10i32, 20, 30]).unwrap().__debug_rebind_execution_domain_explicit(43).unwrap();
    let table = Table::from_columns(vec![first, second]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let copied = table.deep_copy().unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(copied.__debug_execution_domain_id(), 43);
}

#[test]
fn dataframe_from_table_preserves_frame_and_column_domain_identity() {
    let first = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(19).unwrap();
    let second = Column::from_slice(&[10i32, 20, 30]).unwrap().__debug_rebind_execution_domain_explicit(19).unwrap();
    let table = Table::from_columns(vec![first, second]).unwrap();

    let df = DataFrame::from_table(table, vec!["a".into(), "b".into()]).unwrap();

    assert_eq!(df.__debug_execution_domain_kind(), "explicit");
    assert_eq!(df.__debug_execution_domain_id(), 19);
    assert_eq!(df.series("a").unwrap().as_column().__debug_execution_domain_id(), 19);
    assert_eq!(df.series("b").unwrap().as_column().__debug_execution_domain_id(), 19);
}

/// [INV-CT-01] Container invariant: `DataFrame::from_named_pairs` / `from_series` rejects columns
/// with different execution-domain identities (Rule 4 -- preserved in v2).
#[test]
fn dataframe_new_rejects_mixed_domain_columns() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(100).unwrap();
    let right = Column::from_slice(&[4i32, 5, 6]).unwrap().__debug_rebind_execution_domain_explicit(200).unwrap();

    let err =
        crate::common::dataframe_from_columns(vec!["left".into(), "right".into()], vec![left, right]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::new");

    let context = err.context();
    assert!(context.iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:100(")));
    assert!(context.iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:200(")));
    assert!(context.iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn dataframe_from_named_pairs_rejects_mixed_domain_series_with_relocation_guidance() {
    let left = explicit_i32_series("left", &[1, 2, 3], 101);
    let right = explicit_i32_series("right", &[4, 5, 6], 102);

    let err = DataFrame::from_named_pairs([("left", left), ("right", right)]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::from_named_pairs");

    let context = err.context();
    assert!(context.iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:101(")));
    assert!(context.iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:102(")));
    assert!(context.iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

/// [INV-CT-02] Container invariant: `Table::from_columns` rejects columns with different
/// retained provenance (Rule 4 -- preserved in v2).
#[test]
fn table_from_columns_rejects_mixed_domain_columns() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(3).unwrap();
    let right = Column::from_slice(&[4i32, 5, 6]).unwrap().__debug_rebind_execution_domain_explicit(4).unwrap();

    let err = Table::from_columns(vec![left, right]).err().expect("expected mixed-domain table construction to fail");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "table::from_columns");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:3(")));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:4(")));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn dataframe_high_level_row_ops_preserve_explicit_domain_identity() {
    let df = DataFrame::from_series([
        explicit_i32_series("id", &[3, 1, 2], 55),
        explicit_i32_series("value", &[30, 10, 20], 55),
    ])
    .unwrap();

    let sorted = df.sort_values(["id"], cudf::SortOptions::default()).unwrap();
    assert_eq!(sorted.__debug_execution_domain_kind(), "explicit");
    assert_eq!(sorted.__debug_execution_domain_id(), 55);
    assert!(
        sorted.column_names().into_iter().all(|name| sorted
            .series(name)
            .unwrap()
            .as_column()
            .__debug_execution_domain_id()
            == 55)
    );

    let gathered = df.gather(&explicit_i32_series("gather", &[2, 0], 55)).unwrap();
    assert_eq!(gathered.__debug_execution_domain_id(), 55);
    assert!(
        gathered.column_names().into_iter().all(|name| gathered
            .series(name)
            .unwrap()
            .as_column()
            .__debug_execution_domain_id()
            == 55)
    );

    let filtered = df.filter(&explicit_bool_series("mask", &[true, false, true], 55)).unwrap();
    assert_eq!(filtered.__debug_execution_domain_id(), 55);
    assert!(
        filtered.column_names().into_iter().all(|name| filtered
            .series(name)
            .unwrap()
            .as_column()
            .__debug_execution_domain_id()
            == 55)
    );
}

#[test]
fn dataframe_concat_preserves_explicit_domain_identity() {
    let left = DataFrame::from_series([explicit_i32_series("id", &[1, 2], 81)]).unwrap();
    let right = DataFrame::from_series([explicit_i32_series("id", &[3, 4], 81)]).unwrap();

    let combined = left.concat(&right).unwrap();

    assert_eq!(combined.__debug_execution_domain_kind(), "explicit");
    assert_eq!(combined.__debug_execution_domain_id(), 81);
    assert_eq!(combined.series("id").unwrap().as_column().__debug_execution_domain_id(), 81);
}

/// v2 semantics: concat with mixed explicit domains succeeds.
/// Output provenance follows the first table (left).
#[test]
fn dataframe_concat_succeeds_with_mixed_explicit_domains() {
    let left = DataFrame::from_series([explicit_i32_series("id", &[1, 2], 8)]).unwrap();
    let right = DataFrame::from_series([explicit_i32_series("id", &[3, 4], 9)]).unwrap();
    let expected_stream = left.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let combined = left.concat(&right).unwrap();

    // Output provenance inherits from the first (left) table.
    assert_eq!(combined.__debug_execution_domain_kind(), "explicit");
    assert_eq!(combined.__debug_execution_domain_id(), 8);
    assert_eq!(combined.series("id").unwrap().as_column().__debug_execution_domain_id(), 8);

    // Execution used the primary input's stream.
    assert_eq!(last_exec_stream_value(), expected_stream);

    // Verify correct values.
    reset_sync_observation();
    let host: Vec<i32> = combined.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dataframe_merge_preserves_explicit_domain_identity() {
    let left = DataFrame::from_series([
        explicit_i32_series("id", &[1, 2, 3], 64),
        explicit_i32_series("left", &[10, 20, 30], 64),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        explicit_i32_series("id", &[2, 3, 4], 64),
        explicit_i32_series("right", &[200, 300, 400], 64),
    ])
    .unwrap();

    let merged = left.merge(&right, cudf::MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(merged.__debug_execution_domain_kind(), "explicit");
    assert_eq!(merged.__debug_execution_domain_id(), 64);
    assert!(
        merged.column_names().into_iter().all(|name| merged
            .series(name)
            .unwrap()
            .as_column()
            .__debug_execution_domain_id()
            == 64)
    );
}

/// v2 semantics: merge with mixed explicit domains succeeds.
/// Output provenance follows the left table.
#[test]
fn dataframe_merge_succeeds_with_mixed_explicit_domains() {
    let left = DataFrame::from_series([
        explicit_i32_series("id", &[1, 2, 3], 11),
        explicit_i32_series("left", &[10, 20, 30], 11),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        explicit_i32_series("id", &[2, 3, 4], 12),
        explicit_i32_series("right", &[200, 300, 400], 12),
    ])
    .unwrap();
    let expected_stream = left.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let merged = left.merge(&right, cudf::MergeOptions::new().how_inner().on(["id"])).unwrap();

    // Output provenance inherits from the left table.
    assert_eq!(merged.__debug_execution_domain_kind(), "explicit");
    assert_eq!(merged.__debug_execution_domain_id(), 11);
    assert!(
        merged.column_names().into_iter().all(|name| merged
            .series(name)
            .unwrap()
            .as_column()
            .__debug_execution_domain_id()
            == 11)
    );

    // Execution used the primary (left) input's stream.
    assert_eq!(last_exec_stream_value(), expected_stream);
}

/// v2 semantics: fillna_with_series with mixed explicit domains succeeds.
/// Output provenance follows the input DataFrame (replace nulls primary = input column).
#[test]
fn dataframe_fillna_with_series_succeeds_with_mixed_explicit_domains() {
    let df = DataFrame::from_series([Series::from_column(
        Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
            .unwrap()
            .__debug_rebind_execution_domain_explicit(21)
            .unwrap(),
    )
    .unwrap()
    .rename("value")])
    .unwrap();
    let fill = explicit_i32_series("fill", &[10, 20, 30], 22);
    let expected_stream = df.__debug_execution_domain_stream_value();
    let expected_sync_stream = fill.as_column().__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let filled = df.fillna_with_series("value", &fill).unwrap();

    // Output provenance inherits from the input DataFrame.
    assert_eq!(filled.__debug_execution_domain_kind(), "explicit");
    assert_eq!(filled.__debug_execution_domain_id(), 21);

    // Execution used the primary input's stream.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    // Verify correct values.
    let host: Vec<i32> = filled.series("value").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![1, 20, 3]);
}

#[test]
fn series_to_vec_synchronizes_retained_explicit_domain_stream() {
    let series = explicit_i32_series("id", &[4, 5, 6], 71);
    let expected_stream = series.as_column().__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");

    reset_sync_observation();
    let host: Vec<i32> = series.to_vec_strict().unwrap();

    assert_eq!(host, vec![4, 5, 6]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn scalar_value_synchronizes_retained_explicit_domain_stream() {
    let scalar = Scalar::from_value(99i32).unwrap().__debug_rebind_execution_domain_explicit(72).unwrap();
    let expected_stream = scalar.__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");

    reset_sync_observation();
    let value = scalar.value::<i32>().unwrap();

    assert_eq!(value, 99);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn scalar_string_value_synchronizes_retained_explicit_domain_stream() {
    let scalar = Scalar::from_str("hello").unwrap().__debug_rebind_execution_domain_explicit(73).unwrap();
    let expected_stream = scalar.__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");

    reset_sync_observation();
    let value = scalar.string_value().unwrap();

    assert_eq!(value, "hello");
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn scalar_is_valid_synchronizes_retained_explicit_domain_stream() {
    let scalar = Scalar::from_value(99i32).unwrap().__debug_rebind_execution_domain_explicit(731).unwrap();
    let expected_stream = scalar.__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");

    reset_sync_observation();
    let is_valid = scalar.is_valid().unwrap();

    assert!(is_valid);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn series_to_strings_uses_retained_explicit_domain_stream() {
    let series = explicit_str_series("name", &["alice", "", "bob"], 732);
    let expected_stream = series.as_column().__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");

    reset_exec_observation();
    let host = series.to_strings_strict().unwrap();

    assert_eq!(host, vec!["alice".to_string(), "".to_string(), "bob".to_string()]);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dataframe_write_parquet_synchronizes_retained_explicit_domain_stream() {
    let df = DataFrame::from_series([
        explicit_i32_series("id", &[1, 2, 3], 74),
        explicit_i32_series("value", &[10, 20, 30], 74),
    ])
    .unwrap();
    let expected_stream = df.__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("explicit-domain.parquet");

    reset_sync_observation();
    df.write_parquet(&path).unwrap();

    assert!(path.exists(), "parquet write should produce an output file");
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dataframe_write_csv_synchronizes_retained_explicit_domain_stream() {
    let df = DataFrame::from_series([
        explicit_i32_series("id", &[1, 2, 3], 75),
        explicit_i32_series("value", &[10, 20, 30], 75),
    ])
    .unwrap();
    let expected_stream = df.__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("explicit-domain.csv");

    reset_sync_observation();
    df.write_csv(&path).unwrap();

    assert!(path.exists(), "csv write should produce an output file");
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn series_high_level_ops_preserve_explicit_domain_identity() {
    let base = explicit_i32_series("base", &[3, 1, 2, 1], 76);
    let sorted = base.sort_values(cudf::SortOptions::default()).unwrap();
    assert_eq!(sorted.as_column().__debug_execution_domain_kind(), "explicit");
    assert_eq!(sorted.as_column().__debug_execution_domain_id(), 76);

    let nullable = explicit_nullable_i32_series("nullable", &[1, 0, 3, 0], &[true, false, true, false], 76);
    let fill = explicit_i32_series("fill", &[10, 20, 30, 40], 76);
    let filled = nullable.fillna_with_series(&fill).unwrap();
    assert_eq!(filled.as_column().__debug_execution_domain_id(), 76);

    let values = explicit_i32_series("values", &[1, 3], 76);
    let mask = base.isin(&values).unwrap();
    assert_eq!(mask.as_column().__debug_execution_domain_id(), 76);

    let (codes, uniques) = base.factorize().unwrap();
    assert_eq!(codes.as_column().__debug_execution_domain_id(), 76);
    assert_eq!(uniques.as_column().__debug_execution_domain_id(), 76);
}

#[test]
fn series_fillna_with_series_succeeds_with_mixed_domains_and_fences_fill_stream() {
    let nullable = explicit_nullable_i32_series("nullable", &[1, 0, 3], &[true, false, true], 76);
    let fill = explicit_i32_series("fill", &[10, 20, 30], 77);
    let expected_exec_stream = nullable.as_column().__debug_execution_domain_stream_value();
    let expected_sync_stream = fill.as_column().__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let filled = nullable.fillna_with_series(&fill).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);
    assert_eq!(filled.as_column().__debug_execution_domain_id(), nullable.as_column().__debug_execution_domain_id());
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![1, 20, 3]);
}

#[test]
fn series_add_with_same_device_mixed_explicit_domains_uses_left_domain_and_fences_right_stream() {
    let left = explicit_i32_series("left", &[1, 2, 3], 77);
    let right = explicit_i32_series("right", &[10, 20, 30], 78);
    let expected_stream = left.as_column().__debug_execution_domain_stream_value();
    let fenced_stream = right.as_column().__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = left.add(&right).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn series_eq_with_same_device_mixed_explicit_domains_uses_left_domain_and_fences_right_stream() {
    let left = explicit_i32_series("left", &[1, 2, 3], 79);
    let right = explicit_i32_series("right", &[1, 0, 3], 80);
    let expected_stream = left.as_column().__debug_execution_domain_stream_value();
    let fenced_stream = right.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = left.eq(&right).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn series_isin_with_same_device_mixed_explicit_domains_uses_left_domain_and_fences_values_stream() {
    let base = explicit_i32_series("base", &[3, 1, 2, 1], 81);
    let values = explicit_i32_series("values", &[1, 3], 82);
    let expected_stream = base.as_column().__debug_execution_domain_stream_value();
    let fenced_stream = values.as_column().__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    reset_sync_observation();
    let mask = base.isin(&values).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(mask.as_column().__debug_execution_domain_id(), base.as_column().__debug_execution_domain_id());
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false, true]);
}

#[test]
fn column_view_isin_with_same_device_mixed_explicit_domains_uses_input_domain_and_fences_values_stream() {
    let input = explicit_i32_column(&[3, 1, 2, 1], 810);
    let values = explicit_i32_column(&[1, 3], 811);
    let expected_exec_stream = input.__debug_execution_domain_stream_value();
    let expected_exec_mr = input.__debug_execution_domain_mr_value();
    let fenced_stream = values.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = input.view().unwrap().isin(&values.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_exec_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, true, false, true]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn column_view_map_with_same_device_mixed_explicit_domains_uses_input_domain_and_fences_secondary_streams() {
    let input = explicit_i32_column(&[1, 2, 3, 1], 812);
    let keys = explicit_i32_column(&[1, 2, 3], 813);
    let values = explicit_i32_column(&[100, 200, 300], 814);
    let expected_exec_stream = input.__debug_execution_domain_stream_value();
    let expected_exec_mr = input.__debug_execution_domain_mr_value();
    let last_fenced_stream = values.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = input.view().unwrap().map(&keys.view().unwrap(), &values.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), last_fenced_stream);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_exec_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300, 100]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn series_fillna_uses_retained_explicit_domain_stream() {
    let nullable = explicit_nullable_i32_series("nullable", &[1, 0, 3], &[true, false, true], 80);
    let expected_stream = nullable.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    let filled = nullable.fillna(99i32).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(filled.as_column().__debug_execution_domain_id(), 80);
    let host: Vec<i32> = filled.to_vec_strict().unwrap();
    assert_eq!(host, vec![1, 99, 3]);
}

#[test]
fn dataframe_fillna_with_series_uses_retained_explicit_domain_stream_and_mr() {
    let df =
        DataFrame::from_series([explicit_nullable_i32_series("value", &[1, 0, 3], &[true, false, true], 81)]).unwrap();
    let fill = explicit_i32_series("fill", &[10, 20, 30], 81);
    let expected_stream = df.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let filled = df.fillna_with_series("value", &fill).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(filled.__debug_execution_domain_id(), 81);
    let host: Vec<i32> = filled.series("value").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![1, 20, 3]);
}

#[test]
fn dataframe_gather_uses_retained_explicit_domain_stream_and_mr() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[10, 20, 30], 92)]).unwrap();
    let indices = explicit_i32_series("gather", &[2, 0], 92);
    let expected_stream = df.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let gathered = df.gather(&indices).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(gathered.__debug_execution_domain_id(), 92);
    let host: Vec<i32> = gathered.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![30, 10]);
}

#[test]
fn dataframe_filter_uses_retained_explicit_domain_stream_and_mr() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[10, 20, 30], 93)]).unwrap();
    let mask = explicit_bool_series("mask", &[true, false, true], 93);
    let expected_stream = df.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(filtered.__debug_execution_domain_id(), 93);
    let host: Vec<i32> = filtered.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![10, 30]);
}

#[test]
fn dataframe_slice_uses_retained_explicit_domain_stream_and_mr() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[10, 20, 30, 40], 94)]).unwrap();
    let expected_stream = df.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let sliced = df.slice(1, 2).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(sliced.__debug_execution_domain_id(), 94);
    let host: Vec<i32> = sliced.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![20, 30]);
}

#[test]
fn dataframe_groupby_agg_preserves_explicit_domain_identity() {
    let df = DataFrame::from_series([
        explicit_i32_series("key", &[1, 1, 2, 2], 77),
        explicit_i32_series("value", &[10, 20, 30, 40], 77),
    ])
    .unwrap();

    let grouped = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    assert_eq!(grouped.__debug_execution_domain_kind(), "explicit");
    assert_eq!(grouped.__debug_execution_domain_id(), 77);
    assert!(
        grouped.column_names().into_iter().all(|name| grouped
            .series(name)
            .unwrap()
            .as_column()
            .__debug_execution_domain_id()
            == 77)
    );
}

#[test]
fn zero_column_dataframe_try_into_table_and_from_table_preserve_explicit_domain() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 78)]).unwrap();
    let zero = df.without_columns(["id"]).unwrap();

    assert_eq!(zero.num_columns(), 0);
    assert_eq!(zero.__debug_execution_domain_kind(), "explicit");
    assert_eq!(zero.__debug_execution_domain_id(), 78);

    let (table, names) = zero.try_into_table().unwrap();
    assert!(names.is_empty());
    assert_eq!(table.__debug_execution_domain_kind(), "explicit");
    assert_eq!(table.__debug_execution_domain_id(), 78);

    let roundtrip = DataFrame::from_table(table, names).unwrap();
    assert_eq!(roundtrip.num_columns(), 0);
    assert_eq!(roundtrip.__debug_execution_domain_kind(), "explicit");
    assert_eq!(roundtrip.__debug_execution_domain_id(), 78);
}

#[test]
fn zero_column_dataframe_to_table_copy_preserves_explicit_domain() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 79)]).unwrap();
    let zero = df.select(std::iter::empty::<&str>()).unwrap();

    assert_eq!(zero.num_columns(), 0);
    assert_eq!(zero.__debug_execution_domain_kind(), "explicit");
    assert_eq!(zero.__debug_execution_domain_id(), 79);

    let copied = zero.to_table_copy().unwrap();
    assert_eq!(copied.__debug_execution_domain_kind(), "explicit");
    assert_eq!(copied.__debug_execution_domain_id(), 79);
}

#[test]
fn dataframe_to_table_copy_uses_retained_explicit_domain_stream_and_mr() {
    let df = DataFrame::from_series([
        explicit_i32_series("id", &[1, 2, 3], 90),
        explicit_i32_series("value", &[10, 20, 30], 90),
    ])
    .unwrap();
    let expected_stream = df.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let copied = df.to_table_copy().unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(copied.__debug_execution_domain_id(), 90);
}

#[test]
fn dataframe_try_into_table_shared_copy_uses_retained_explicit_domain_stream_and_mr() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 91)]).unwrap();
    let _shared = df.series("id").unwrap();
    let expected_stream = df.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["id"]);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(table.__debug_execution_domain_id(), 91);
}

#[test]
fn zero_column_dataframe_concat_preserves_explicit_domain() {
    let left =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2], 80)]).unwrap().without_columns(["id"]).unwrap();
    let right = DataFrame::from_series([explicit_i32_series("id", &[3, 4], 80)])
        .unwrap()
        .select(std::iter::empty::<&str>())
        .unwrap();

    let combined = left.concat(&right).unwrap();

    assert_eq!(combined.num_columns(), 0);
    assert_eq!(combined.__debug_execution_domain_kind(), "explicit");
    assert_eq!(combined.__debug_execution_domain_id(), 80);
}

/// [INV-CT-01] Container construction path: inserting an ambient-domain column into an
/// explicit-domain DataFrame is rejected (Rule 4 -- preserved in v2).
#[test]
fn zero_column_explicit_dataframe_with_column_rejects_ambient_series() {
    let zero =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 81)]).unwrap().without_columns(["id"]).unwrap();
    let ambient = Series::new::<i32>("value", &[10, 20, 30]).unwrap();

    let err = zero.with_column("value", ambient).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::with_column");
    assert!(err.context().iter().any(|(key, value)| *key == "frame_domain" && value.starts_with("explicit:81(")));
    assert!(err.context().iter().any(|(key, value)| *key == "series_domain" && value.starts_with("ambient_default")));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn zero_column_explicit_dataframe_with_column_preserves_same_explicit_domain() {
    let zero =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 82)]).unwrap().without_columns(["id"]).unwrap();
    let explicit = explicit_i32_series("value", &[10, 20, 30], 82);

    let result = zero.with_column("value", explicit).unwrap();

    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 82);
    assert_eq!(result.series("value").unwrap().as_column().__debug_execution_domain_id(), 82);
}

#[test]
fn zero_column_explicit_dataframe_with_column_rejects_same_device_mixed_runtime_series() {
    let zero =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 821)]).unwrap().without_columns(["id"]).unwrap();
    let explicit = explicit_i32_series("value", &[10, 20, 30], 822);

    let err = zero.with_column("value", explicit).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::with_column");
    assert!(err.context().iter().any(|(key, value)| *key == "frame_domain" && value.starts_with("explicit:821(")));
    assert!(err.context().iter().any(|(key, value)| *key == "series_domain" && value.starts_with("explicit:822(")));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

/// Slice 5: inserting into an explicit zero-column frame must not perform an
/// implicit relocation.
#[test]
fn zero_column_explicit_dataframe_assign_rejects_mixed_domain_series() {
    let zero =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 83)]).unwrap().without_columns(["id"]).unwrap();
    let ambient = Series::new::<i32>("value", &[10, 20, 30]).unwrap();

    let err = zero.assign([("value", ambient)]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::assign");
    assert!(err.context().iter().any(|(key, value)| *key == "frame_domain" && value.starts_with("explicit:83(")));
    assert!(err.context().iter().any(|(key, value)| *key == "series_domain" && value.starts_with("ambient_default")));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn zero_column_explicit_dataframe_assign_preserves_same_explicit_domain() {
    let zero =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 88)]).unwrap().without_columns(["id"]).unwrap();
    let explicit = explicit_i32_series("value", &[10, 20, 30], 88);

    let result = zero.assign([("value", explicit)]).unwrap();

    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 88);
    assert_eq!(result.series("value").unwrap().as_column().__debug_execution_domain_id(), 88);
}

#[test]
fn zero_column_explicit_dataframe_assign_rejects_same_device_mixed_runtime_series() {
    let zero =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 89)]).unwrap().without_columns(["id"]).unwrap();
    let explicit = explicit_i32_series("value", &[10, 20, 30], 90);

    let err = zero.assign([("value", explicit)]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::assign");
    assert!(err.context().iter().any(|(key, value)| *key == "frame_domain" && value.starts_with("explicit:89(")));
    assert!(err.context().iter().any(|(key, value)| *key == "series_domain" && value.starts_with("explicit:90(")));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn zero_column_explicit_dataframe_with_columns_literal_preserves_same_explicit_domain() {
    let zero =
        DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 91)]).unwrap().without_columns(["id"]).unwrap();

    let result = zero.with_columns([lit(7i32).alias("value"), lit(1i32).alias("flag")]).unwrap();

    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 91);
    assert_eq!(result.series("value").unwrap().as_column().__debug_execution_domain_id(), 91);
    assert_eq!(result.series("flag").unwrap().as_column().__debug_execution_domain_id(), 91);
    assert_eq!(result.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 7, 7]);
    assert_eq!(result.series("flag").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 1]);
}

#[test]
fn explicit_dataframe_with_column_rejects_same_device_mixed_runtime_series() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 84)]).unwrap();
    let replacement = explicit_i32_series("value", &[10, 20, 30], 85);

    let err = df.with_column("value", replacement).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::with_column");
    assert!(err.context().iter().any(|(key, value)| *key == "frame_domain" && value.starts_with("explicit:84(")));
    assert!(err.context().iter().any(|(key, value)| *key == "series_domain" && value.starts_with("explicit:85(")));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn explicit_dataframe_assign_rejects_same_device_mixed_runtime_series() {
    let df = DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 86)]).unwrap();
    let replacement = explicit_i32_series("value", &[10, 20, 30], 87);

    let err = df.assign([("value", replacement)]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::assign");
    assert!(err.context().iter().any(|(key, value)| *key == "frame_domain" && value.starts_with("explicit:86(")));
    assert!(err.context().iter().any(|(key, value)| *key == "series_domain" && value.starts_with("explicit:87(")));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn table_view_column_retains_explicit_domain_and_sync_stream() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 120)]).unwrap();
    let view = table.view().unwrap();

    assert_eq!(view.__debug_execution_domain_kind(), "explicit");
    assert_eq!(view.__debug_execution_domain_id(), 120);

    let borrowed = view.column(0).unwrap();
    assert_eq!(borrowed.__debug_execution_domain_kind(), "explicit");
    assert_eq!(borrowed.__debug_execution_domain_id(), 120);

    reset_sync_observation();
    assert_eq!(borrowed.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(last_sync_stream_value(), table.__debug_execution_domain_stream_value());
}

#[test]
fn table_view_select_columns_retains_explicit_domain_and_sync_stream() {
    let table =
        Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 121), explicit_i32_column(&[4, 5, 6], 121)]).unwrap();
    let projected = table.view().unwrap().select_columns(&[1]).unwrap();

    assert_eq!(projected.__debug_execution_domain_kind(), "explicit");
    assert_eq!(projected.__debug_execution_domain_id(), 121);

    let borrowed = projected.column(0).unwrap();
    assert_eq!(borrowed.__debug_execution_domain_id(), 121);

    reset_sync_observation();
    assert_eq!(borrowed.to_vec::<i32>().unwrap(), vec![4, 5, 6]);
    assert_eq!(last_sync_stream_value(), table.__debug_execution_domain_stream_value());
}

#[test]
fn table_view_from_column_views_retains_explicit_domain() {
    let first = explicit_i32_column(&[7, 8, 9], 122);
    let second = explicit_i32_column(&[70, 80, 90], 122);
    let first_view = first.view().unwrap();
    let second_view = second.view().unwrap();

    let table_view = TableView::from_column_views(&[&first_view, &second_view]).unwrap();
    assert_eq!(table_view.__debug_execution_domain_kind(), "explicit");
    assert_eq!(table_view.__debug_execution_domain_id(), 122);

    let borrowed = table_view.column(1).unwrap();
    assert_eq!(borrowed.__debug_execution_domain_id(), 122);

    reset_sync_observation();
    assert_eq!(borrowed.to_vec::<i32>().unwrap(), vec![70, 80, 90]);
    assert_eq!(last_sync_stream_value(), borrowed.__debug_execution_domain_stream_value());
}

/// [INV-CT-01 extension] View-level container invariant: `TableView::from_column_views` rejects
/// column views with different retained provenance (Rule 4 -- preserved in v2).
#[test]
fn table_view_from_column_views_rejects_mixed_domains() {
    let left = explicit_i32_column(&[1, 2, 3], 123);
    let right = explicit_i32_column(&[4, 5, 6], 124);
    let left_view = left.view().unwrap();
    let right_view = right.view().unwrap();

    let err = TableView::from_column_views(&[&left_view, &right_view]).err().expect("expected mixed-domain error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "table_view::from_column_views");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:123(")));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:124(")));
    assert!(err.context().iter().any(|(key, value)| *key == "column_index" && value == "1"));
    assert!(err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
}

#[test]
fn dictionary_keys_retains_explicit_domain_and_sync_stream() {
    let input = Column::from_slice(&[429i32, 111, 213, 111]).unwrap();
    let dict = dictionary::encode(&input.view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(125)
        .unwrap();
    let dict_view = dict.view().unwrap();

    let keys = dictionary::keys(&dict_view).unwrap();
    assert_eq!(keys.__debug_execution_domain_kind(), "explicit");
    assert_eq!(keys.__debug_execution_domain_id(), 125);

    reset_sync_observation();
    let mut host = keys.to_vec::<i32>().unwrap();
    host.sort_unstable();
    assert_eq!(host, vec![111, 213, 429]);
    assert_eq!(last_sync_stream_value(), dict.__debug_execution_domain_stream_value());
}

#[test]
fn dictionary_indices_retains_explicit_domain_and_sync_stream() {
    let input = Column::from_slice(&[429i32, 111, 213, 111]).unwrap();
    let dict = dictionary::encode(&input.view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(126)
        .unwrap();
    let dict_view = dict.view().unwrap();

    let indices = dictionary::indices(&dict_view).unwrap();
    assert_eq!(indices.__debug_execution_domain_kind(), "explicit");
    assert_eq!(indices.__debug_execution_domain_id(), 126);

    reset_sync_observation();
    let host_indices = indices.to_vec::<i32>().unwrap();
    let keys = dictionary::keys(&dict_view).unwrap().to_vec::<i32>().unwrap();
    let input = [429i32, 111, 213, 111];
    for (value, index) in input.into_iter().zip(host_indices.into_iter()) {
        assert_eq!(value, keys[index as usize]);
    }
    assert_eq!(last_sync_stream_value(), dict.__debug_execution_domain_stream_value());
}

#[test]
fn search_contains_retains_explicit_domain_and_uses_exec_options() {
    let rt = sample_runtime();
    let haystack = rt.column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let needles = rt.column_from_slice(&[20i32, 50, 10]).unwrap();
    let expected_stream = haystack.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = search::contains(&haystack.view().unwrap(), &needles.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), haystack.__debug_execution_domain_id());

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn strings_contains_with_same_device_mixed_explicit_domains_fences_scalar_stream() {
    let input = Column::from_strs(&["alpha", "beta", "alphabet"])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(2001)
        .unwrap();
    let target = Scalar::from_str("alpha").unwrap().__debug_rebind_execution_domain_explicit(2002).unwrap();
    let expected_exec_stream = input.__debug_execution_domain_stream_value();
    let fenced_stream = target.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = strings::str_contains(&input.view().unwrap(), &target).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn strings_starts_with_with_same_device_mixed_explicit_domains_fences_scalar_stream() {
    let input = Column::from_strs(&["alpha", "beta", "alphabet"])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(2003)
        .unwrap();
    let target = Scalar::from_str("alp").unwrap().__debug_rebind_execution_domain_explicit(2004).unwrap();
    let expected_exec_stream = input.__debug_execution_domain_stream_value();
    let fenced_stream = target.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = strings::starts_with(&input.view().unwrap(), &target).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn strings_ends_with_with_same_device_mixed_explicit_domains_fences_scalar_stream() {
    let input = Column::from_strs(&["alpha", "beta", "alphabet"])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(2005)
        .unwrap();
    let target = Scalar::from_str("ha").unwrap().__debug_rebind_execution_domain_explicit(2006).unwrap();
    let expected_exec_stream = input.__debug_execution_domain_stream_value();
    let fenced_stream = target.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = strings::ends_with(&input.view().unwrap(), &target).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn strings_strip_with_same_device_mixed_explicit_domains_fences_scalar_stream() {
    let input = Column::from_strs(&["xxalpha", "betaxx", "xxalphabetxx"])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(2007)
        .unwrap();
    let target = Scalar::from_str("x").unwrap().__debug_rebind_execution_domain_explicit(2008).unwrap();
    let expected_exec_stream = input.__debug_execution_domain_stream_value();
    let fenced_stream = target.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = strings::strip(&input.view().unwrap(), cudf::StripSide::Both, &target).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(
        result.view().unwrap().to_strings_lossy().unwrap(),
        vec!["alpha".to_string(), "beta".to_string(), "alphabet".to_string()]
    );
}

/// v2 semantics: search::contains with mixed explicit domains succeeds.
/// Output provenance follows the haystack (the data being searched).
#[test]
fn search_contains_succeeds_with_mixed_domains() {
    let haystack = explicit_i32_column(&[10, 20, 30], 201);
    let needles = explicit_i32_column(&[20, 40], 202);
    let expected_stream = haystack.__debug_execution_domain_stream_value();
    let expected_mr = haystack.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = search::contains(&haystack.view().unwrap(), &needles.view().unwrap()).unwrap();

    // Output provenance inherits from the haystack.
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 201);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the haystack's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dictionary_encode_retains_explicit_domain_and_uses_exec_options() {
    let rt = sample_runtime();
    let input = rt.column_from_slice(&[429i32, 111, 213, 111]).unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let dict = dictionary::encode(&input.view().unwrap(), TypeId::Int32).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(dict.__debug_execution_domain_kind(), "explicit");
    assert_eq!(dict.__debug_execution_domain_id(), input.__debug_execution_domain_id());

    let dict_view = dict.view().unwrap();
    let keys = dictionary::keys(&dict_view).unwrap();
    assert_eq!(keys.__debug_execution_domain_id(), dict.__debug_execution_domain_id());

    reset_sync_observation();
    let mut host_keys = keys.to_vec::<i32>().unwrap();
    host_keys.sort_unstable();
    assert_eq!(host_keys, vec![111, 213, 429]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dictionary_decode_retains_explicit_domain_and_uses_exec_options() {
    let source = Column::from_slice(&[429i32, 111, 213, 111]).unwrap();
    let dict = dictionary::encode(&source.view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(211)
        .unwrap();
    let expected_stream = dict.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let decoded = dictionary::decode(&dict.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(decoded.__debug_execution_domain_kind(), "explicit");
    assert_eq!(decoded.__debug_execution_domain_id(), 211);

    reset_sync_observation();
    assert_eq!(decoded.view().unwrap().to_vec::<i32>().unwrap(), vec![429, 111, 213, 111]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dictionary_set_keys_retains_explicit_domain_and_uses_exec_options() {
    let dict = dictionary::encode(&Column::from_slice(&[3i32, 1, 3, 2]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(212)
        .unwrap();
    let keys = Column::from_slice(&[2i32, 3, 4]).unwrap().__debug_rebind_execution_domain_explicit(212).unwrap();
    let expected_stream = dict.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let updated = dictionary::set_keys(&dict.view().unwrap(), &keys.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(updated.__debug_execution_domain_kind(), "explicit");
    assert_eq!(updated.__debug_execution_domain_id(), 212);

    reset_sync_observation();
    let updated_view = updated.view().unwrap();
    let mut updated_keys = dictionary::keys(&updated_view).unwrap().to_vec::<i32>().unwrap();
    updated_keys.sort_unstable();
    assert_eq!(updated_keys, vec![2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: dictionary::set_keys with mixed explicit domains succeeds.
/// Output provenance follows the dictionary column (primary input).
#[test]
fn dictionary_set_keys_succeeds_with_mixed_domains() {
    let dict = dictionary::encode(&Column::from_slice(&[3i32, 1, 3, 2]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(213)
        .unwrap();
    let keys = Column::from_slice(&[2i32, 3, 4]).unwrap().__debug_rebind_execution_domain_explicit(214).unwrap();
    let expected_stream = dict.__debug_execution_domain_stream_value();
    let expected_mr = dict.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let updated = dictionary::set_keys(&dict.view().unwrap(), &keys.view().unwrap()).unwrap();

    // Output provenance inherits from the dictionary column.
    assert_eq!(updated.__debug_execution_domain_kind(), "explicit");
    assert_eq!(updated.__debug_execution_domain_id(), 213);
    assert_eq!(updated.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the dictionary column's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    let updated_view = updated.view().unwrap();
    let mut updated_keys = dictionary::keys(&updated_view).unwrap().to_vec::<i32>().unwrap();
    updated_keys.sort_unstable();
    assert_eq!(updated_keys, vec![2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dictionary_match_dictionaries_retains_explicit_domain_and_uses_exec_options() {
    let first = dictionary::encode(&Column::from_slice(&[10i32, 20, 30, 10]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(215)
        .unwrap();
    let second = dictionary::encode(&Column::from_slice(&[20i32, 30, 40, 20]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(215)
        .unwrap();
    let expected_stream = first.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let matched = dictionary::match_dictionaries(&[&first.view().unwrap(), &second.view().unwrap()]).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(matched.len(), 2);
    assert!(matched.iter().all(|column| column.__debug_execution_domain_kind() == "explicit"));
    assert!(matched.iter().all(|column| column.__debug_execution_domain_id() == 215));

    reset_sync_observation();
    let mut first_keys = dictionary::keys(&matched[0].view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    let mut second_keys = dictionary::keys(&matched[1].view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    first_keys.sort_unstable();
    second_keys.sort_unstable();
    assert_eq!(first_keys, second_keys);
    assert_eq!(first_keys, vec![10, 20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: dictionary::match_dictionaries with mixed explicit domains succeeds.
/// Output provenance follows the first dictionary (concatenate semantics).
#[test]
fn dictionary_match_dictionaries_succeeds_with_mixed_domains() {
    let first = dictionary::encode(&Column::from_slice(&[10i32, 20, 30]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(216)
        .unwrap();
    let second = dictionary::encode(&Column::from_slice(&[20i32, 30, 40]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(217)
        .unwrap();
    let expected_stream = first.__debug_execution_domain_stream_value();
    let expected_mr = first.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let matched = dictionary::match_dictionaries(&[&first.view().unwrap(), &second.view().unwrap()]).unwrap();

    // Output provenance inherits from the first dictionary.
    assert_eq!(matched.len(), 2);
    assert!(matched.iter().all(|column| column.__debug_execution_domain_kind() == "explicit"));
    assert!(matched.iter().all(|column| column.__debug_execution_domain_id() == 216));
    assert!(matched.iter().all(|column| column.__debug_execution_domain_mr_value() == expected_mr));

    // Execution used the first dictionary's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify matched dictionaries share the same key set.
    reset_sync_observation();
    let mut first_keys = dictionary::keys(&matched[0].view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    let mut second_keys = dictionary::keys(&matched[1].view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    first_keys.sort_unstable();
    second_keys.sort_unstable();
    assert_eq!(first_keys, second_keys);
    assert_eq!(first_keys, vec![10, 20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn series_isin_uses_retained_explicit_domain_stream_end_to_end() {
    let rt = sample_runtime();
    let base = rt.series("base", &[3i32, 1, 2, 1]).unwrap();
    let values = rt.series("values", &[1i32, 3]).unwrap();
    let expected_stream = base.as_column().__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let mask = base.isin(&values).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(mask.as_column().__debug_execution_domain_id(), base.as_column().__debug_execution_domain_id());

    reset_sync_observation();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false, true]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn hashing_murmurhash3_x86_32_retains_explicit_domain_and_uses_exec_options() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 218)]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let hashes = hashing::murmurhash3_x86_32(&table.view().unwrap(), 0).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(hashes.__debug_execution_domain_kind(), "explicit");
    assert_eq!(hashes.__debug_execution_domain_id(), 218);

    reset_sync_observation();
    assert_eq!(hashes.view().unwrap().to_vec::<u32>().unwrap().len(), 3);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn hashing_xxhash_64_retains_explicit_domain_and_uses_exec_options() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 219)]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let hashes = hashing::xxhash_64(&table.view().unwrap(), 123).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(hashes.__debug_execution_domain_kind(), "explicit");
    assert_eq!(hashes.__debug_execution_domain_id(), 219);

    reset_sync_observation();
    assert_eq!(hashes.view().unwrap().to_vec::<u64>().unwrap().len(), 3);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn stream_compaction_drop_nulls_retains_explicit_domain_and_uses_exec_options() {
    let first = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(220)
        .unwrap();
    let second = explicit_i32_column(&[10, 20, 30], 220);
    let table = Table::from_columns(vec![first, second]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = stream_compaction::drop_nulls(&table.view().unwrap(), &[0], 1).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 220);

    reset_sync_observation();
    let first_host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let second_host = result.view().unwrap().column(1).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(first_host, vec![1, 3]);
    assert_eq!(second_host, vec![10, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn stream_compaction_distinct_retains_explicit_domain_and_uses_exec_options() {
    let table = Table::from_columns(vec![explicit_i32_column(&[1, 2, 1, 3], 221)]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = stream_compaction::distinct(
        &table.view().unwrap(),
        &[0],
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 221);

    reset_sync_observation();
    let mut host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    host.sort_unstable();
    assert_eq!(host, vec![1, 2, 3]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: filter_isin with mixed explicit domains succeeds.
/// Output provenance follows the haystack (the DataFrame column being searched).
#[test]
fn dataframe_filter_isin_succeeds_with_mixed_domains() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let df = DataFrame::from_series([left_rt.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let values = right_rt.series("values", &[20i32, 30]).unwrap();
    let expected_stream = df.__debug_execution_domain_stream_value();
    let expected_mr = df.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let filtered = df.filter_isin("id", &values).unwrap();

    // Output provenance inherits from the DataFrame (haystack).
    assert_eq!(filtered.__debug_execution_domain_kind(), "explicit");
    assert_eq!(filtered.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(filtered.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    let host: Vec<i32> = filtered.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![20, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn column_view_factorize_uses_retained_explicit_domain_stream_end_to_end() {
    let rt = sample_runtime();
    let input = rt.column_from_slice(&[429i32, 111, 213, 111, 213, 429, 213]).unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let (codes, uniques) = input.view().unwrap().factorize().unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(codes.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(uniques.__debug_execution_domain_id(), input.__debug_execution_domain_id());

    reset_sync_observation();
    let unique_values = uniques.view().unwrap().to_vec::<i32>().unwrap();
    let code_values = codes.view().unwrap().to_vec::<i32>().unwrap();
    for code in code_values {
        assert!(unique_values.get(code as usize).is_some(), "factorize code should index into uniques");
    }
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn sorting_sort_retains_explicit_domain_and_uses_exec_options() {
    let table = Table::from_columns(vec![explicit_i32_column(&[3, 1, 2], 130)]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let sorted = sorting::sort(&table.view().unwrap(), None, None).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(sorted.__debug_execution_domain_kind(), "explicit");
    assert_eq!(sorted.__debug_execution_domain_id(), 130);

    reset_sync_observation();
    let host = sorted.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![1, 2, 3]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn sorting_sort_by_key_retains_explicit_domain_and_uses_exec_options() {
    let keys = Table::from_columns(vec![explicit_i32_column(&[3, 1, 2], 131)]).unwrap();
    let values = Table::from_columns(vec![explicit_i32_column(&[30, 10, 20], 131)]).unwrap();
    let expected_stream = values.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let sorted = sorting::sort_by_key(&values.view().unwrap(), &keys.view().unwrap(), None, None).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(sorted.__debug_execution_domain_kind(), "explicit");
    assert_eq!(sorted.__debug_execution_domain_id(), 131);

    reset_sync_observation();
    let host = sorted.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 20, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: sort_by_key with mixed explicit domains succeeds.
/// Output provenance follows the values table (primary input).
#[test]
fn sorting_sort_by_key_succeeds_with_mixed_domains() {
    let keys = Table::from_columns(vec![explicit_i32_column(&[3, 1, 2], 132)]).unwrap();
    let values = Table::from_columns(vec![explicit_i32_column(&[30, 10, 20], 133)]).unwrap();
    let expected_stream = values.__debug_execution_domain_stream_value();
    let expected_mr = values.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let sorted = sorting::sort_by_key(&values.view().unwrap(), &keys.view().unwrap(), None, None).unwrap();

    // Output provenance inherits from the values table.
    assert_eq!(sorted.__debug_execution_domain_kind(), "explicit");
    assert_eq!(sorted.__debug_execution_domain_id(), 133);
    assert_eq!(sorted.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the values table's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    let host = sorted.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 20, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn sorting_stable_sort_by_key_succeeds_with_mixed_domains() {
    let keys = Table::from_columns(vec![explicit_i32_column(&[3, 1, 2], 1330)]).unwrap();
    let values = Table::from_columns(vec![explicit_i32_column(&[30, 10, 20], 1331)]).unwrap();
    let expected_stream = values.__debug_execution_domain_stream_value();
    let expected_mr = values.__debug_execution_domain_mr_value();
    let expected_sync_stream = keys.__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let sorted = sorting::stable_sort_by_key(&values.view().unwrap(), &keys.view().unwrap(), None, None).unwrap();

    assert_eq!(sorted.__debug_execution_domain_kind(), "explicit");
    assert_eq!(sorted.__debug_execution_domain_id(), 1331);
    assert_eq!(sorted.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    let host = sorted.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 20, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn sorting_sorted_order_retains_explicit_domain_and_uses_exec_options() {
    let table = Table::from_columns(vec![explicit_i32_column(&[30, 10, 20], 134)]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let order = sorting::sorted_order(&table.view().unwrap(), None, None).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(order.__debug_execution_domain_kind(), "explicit");
    assert_eq!(order.__debug_execution_domain_id(), 134);

    reset_sync_observation();
    assert_eq!(order.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 0]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn filter_apply_boolean_mask_retains_explicit_domain_and_uses_exec_options() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 135)]).unwrap();
    let mask = Column::from_slice(&[true, false, true]).unwrap().__debug_rebind_execution_domain_explicit(135).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let filtered = filter::apply_boolean_mask(&table.view().unwrap(), &mask.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(filtered.__debug_execution_domain_kind(), "explicit");
    assert_eq!(filtered.__debug_execution_domain_id(), 135);

    reset_sync_observation();
    let host = filtered.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: filter with mixed explicit domains succeeds.
/// Output provenance follows the data being filtered (not the mask).
#[test]
fn filter_apply_boolean_mask_succeeds_with_mixed_domains() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 136)]).unwrap();
    let mask = Column::from_slice(&[true, false, true]).unwrap().__debug_rebind_execution_domain_explicit(137).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let filtered = filter::apply_boolean_mask(&table.view().unwrap(), &mask.view().unwrap()).unwrap();

    // Output provenance inherits from the data table (not the mask).
    assert_eq!(filtered.__debug_execution_domain_kind(), "explicit");
    assert_eq!(filtered.__debug_execution_domain_id(), 136);
    assert_eq!(filtered.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the data table's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    let host = filtered.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn unary_operation_retains_explicit_domain_and_uses_exec_options() {
    let input = explicit_i32_column(&[-3, 0, 5], 140);
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = unary::unary_operation(&input.view().unwrap(), UnaryOperator::Abs).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 140);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![3, 0, 5]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn unary_logical_not_retains_explicit_domain_and_uses_exec_options() {
    let input =
        Column::from_slice(&[true, false, true]).unwrap().__debug_rebind_execution_domain_explicit(141).unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = unary::logical_not(&input.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 141);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, false]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn unary_is_null_retains_explicit_domain_and_uses_exec_options() {
    let input = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(142)
        .unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = unary::is_null(&input.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 142);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, false]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn unary_is_valid_retains_explicit_domain_and_uses_exec_options() {
    let input = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(143)
        .unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = unary::is_valid(&input.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 143);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn unary_cast_retains_explicit_domain_and_uses_exec_options() {
    let input = explicit_i32_column(&[1, 2, 3], 144);
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = unary::cast(&input.view().unwrap(), &DataType::new(TypeId::Float64)).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 144);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn unary_is_nan_retains_explicit_domain_and_uses_exec_options() {
    let input =
        Column::from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap().__debug_rebind_execution_domain_explicit(145).unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = unary::is_nan(&input.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 145);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, false]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn unary_is_not_nan_retains_explicit_domain_and_uses_exec_options() {
    let input =
        Column::from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap().__debug_rebind_execution_domain_explicit(146).unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = unary::is_not_nan(&input.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 146);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn binary_operation_vv_retains_explicit_domain_and_uses_exec_options() {
    let lhs = explicit_i32_column(&[1, 2, 3], 150);
    let rhs = explicit_i32_column(&[10, 20, 30], 150);
    let expected_stream = lhs.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = binary::binary_operation_vv(
        &lhs.view().unwrap(),
        &rhs.view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 150);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn binary_operation_vs_retains_explicit_domain_and_uses_exec_options() {
    let lhs = explicit_i32_column(&[2, 3, 4], 151);
    let rhs = Scalar::from_value(10i32).unwrap().__debug_rebind_execution_domain_explicit(151).unwrap();
    let expected_stream = lhs.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result =
        binary::binary_operation_vs(&lhs.view().unwrap(), &rhs, BinaryOperator::Mul, &DataType::new(TypeId::Int32))
            .unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 151);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn binary_operation_sv_retains_explicit_domain_and_uses_exec_options() {
    let lhs = Scalar::from_value(100.0f64).unwrap().__debug_rebind_execution_domain_explicit(152).unwrap();
    let rhs = Column::from_slice(&[2.0f64, 4.0, 5.0]).unwrap().__debug_rebind_execution_domain_explicit(152).unwrap();
    let expected_stream = lhs.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result =
        binary::binary_operation_sv(&lhs, &rhs.view().unwrap(), BinaryOperator::Div, &DataType::new(TypeId::Float64))
            .unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 152);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<f64>().unwrap(), vec![50.0, 25.0, 20.0]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: binary_operation_vv with mixed explicit domains succeeds.
/// Output provenance follows the left-hand side.
#[test]
fn binary_operation_vv_succeeds_with_mixed_domains() {
    let lhs = explicit_i32_column(&[1, 2, 3], 153);
    let rhs = explicit_i32_column(&[10, 20, 30], 154);
    let expected_stream = lhs.__debug_execution_domain_stream_value();
    let expected_mr = lhs.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = binary::binary_operation_vv(
        &lhs.view().unwrap(),
        &rhs.view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();

    // Output provenance inherits from the left-hand side.
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 153);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the left-hand side's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: binary_operation_vs with mixed explicit domains succeeds.
/// Output provenance follows the left-hand side (the column).
#[test]
fn binary_operation_vs_succeeds_with_mixed_domains() {
    let lhs = explicit_i32_column(&[2, 3, 4], 155);
    let rhs = Scalar::from_value(10i32).unwrap().__debug_rebind_execution_domain_explicit(156).unwrap();
    let expected_stream = lhs.__debug_execution_domain_stream_value();
    let expected_mr = lhs.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result =
        binary::binary_operation_vs(&lhs.view().unwrap(), &rhs, BinaryOperator::Mul, &DataType::new(TypeId::Int32))
            .unwrap();

    // Output provenance inherits from the left-hand side.
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 155);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the left-hand side's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: binary_operation_sv with mixed explicit domains succeeds.
/// Output provenance follows the left-hand side (the scalar).
#[test]
fn binary_operation_sv_succeeds_with_mixed_domains() {
    let lhs = Scalar::from_value(100i32).unwrap().__debug_rebind_execution_domain_explicit(157).unwrap();
    let rhs = explicit_i32_column(&[2, 4, 5], 158);
    let expected_stream = lhs.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let result =
        binary::binary_operation_sv(&lhs, &rhs.view().unwrap(), BinaryOperator::Div, &DataType::new(TypeId::Int32))
            .unwrap();

    // Output provenance inherits from the left-hand side (scalar).
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 157);
    // Scalar does not yet expose a debug MR accessor, so at minimum assert
    // output provenance matches the actual FFI-observed MR.
    assert_eq!(result.__debug_execution_domain_mr_value(), last_exec_mr_value());

    // Execution used the left-hand side's stream.
    assert_eq!(last_exec_stream_value(), expected_stream);

    // Verify correct values.
    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![50, 25, 20]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn binary_operation_vs_with_same_device_mixed_explicit_domains_fences_scalar_stream() {
    let lhs = explicit_i32_column(&[2, 3, 4], 1580);
    let rhs = Scalar::from_value(10i32).unwrap().__debug_rebind_execution_domain_explicit(1581).unwrap();
    let expected_exec_stream = lhs.__debug_execution_domain_stream_value();
    let fenced_stream = rhs.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result =
        binary::binary_operation_vs(&lhs.view().unwrap(), &rhs, BinaryOperator::Mul, &DataType::new(TypeId::Int32))
            .unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_id(), lhs.__debug_execution_domain_id());
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40]);
}

#[test]
fn binary_operation_sv_with_same_device_mixed_explicit_domains_fences_column_stream() {
    let lhs = Scalar::from_value(100i32).unwrap().__debug_rebind_execution_domain_explicit(1582).unwrap();
    let rhs = explicit_i32_column(&[2, 4, 5], 1583);
    let expected_exec_stream = lhs.__debug_execution_domain_stream_value();
    let fenced_stream = rhs.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result =
        binary::binary_operation_sv(&lhs, &rhs.view().unwrap(), BinaryOperator::Div, &DataType::new(TypeId::Int32))
            .unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_id(), lhs.__debug_execution_domain_id());
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![50, 25, 20]);
}

#[test]
fn series_in_context_logical_and_with_same_device_mixed_explicit_domains_fences_rhs_stream() {
    let runtime = sample_runtime();
    let lhs = explicit_nullable_bool_series("lhs", &[true, false, true], &[true, false, true], 1584);
    let rhs = explicit_bool_series("rhs", &[true, true, false], 1585);
    let ctx = ExecutionContext::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let expected_exec_stream = runtime.stream().__debug_raw_handle_value();
    let fenced_stream = rhs.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = lhs.in_context(&ctx).logical_and(&rhs).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.as_column().__debug_execution_domain_id(), lhs.as_column().__debug_execution_domain_id());
    assert_eq!(result.as_column().__debug_execution_domain_stream_value(), expected_exec_stream);

    reset_sync_observation();
    assert_eq!(result.to_vec_optional::<bool>().unwrap(), vec![Some(true), None, Some(false)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn reduction_reduce_retains_explicit_domain_and_uses_exec_options() {
    let input = explicit_i32_column(&[1, 2, 3, 4], 160);
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = reduction::reduce(&input.view().unwrap(), &ReduceAgg::Sum, &DataType::new(TypeId::Int64)).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 160);

    reset_sync_observation();
    assert_eq!(result.value::<i64>().unwrap(), 10);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn reduction_scan_retains_explicit_domain_and_uses_exec_options() {
    let input = explicit_i32_column(&[1, 2, 3, 4], 161);
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result =
        reduction::scan(&input.view().unwrap(), ScanAgg::Sum, ScanType::Inclusive, cudf::NullPolicy::Exclude).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_id(), 161);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 3, 6, 10]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn reduction_minmax_retains_explicit_domain_and_uses_exec_options() {
    let input = explicit_i32_column(&[5, 1, 3, 7, 2], 162);
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let (min_scalar, max_scalar) = reduction::minmax(&input.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(min_scalar.__debug_execution_domain_kind(), "explicit");
    assert_eq!(max_scalar.__debug_execution_domain_kind(), "explicit");
    assert_eq!(min_scalar.__debug_execution_domain_id(), 162);
    assert_eq!(max_scalar.__debug_execution_domain_id(), 162);

    reset_sync_observation();
    assert_eq!(min_scalar.value::<i32>().unwrap(), 1);
    assert_eq!(max_scalar.value::<i32>().unwrap(), 7);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn concatenate_columns_retains_explicit_domain_and_uses_exec_options() {
    let left = explicit_i32_column(&[1, 2], 170);
    let right = explicit_i32_column(&[3, 4, 5], 170);
    let left_view = left.view().unwrap();
    let right_view = right.view().unwrap();
    let expected_stream = left.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = concatenate::concatenate_columns(&[&left_view, &right_view]).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 170);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: concatenate_columns with mixed explicit domains succeeds.
/// Output provenance follows the first column in the sequence.
#[test]
fn concatenate_columns_succeeds_with_mixed_domains() {
    let left = explicit_i32_column(&[1, 2], 171);
    let right = explicit_i32_column(&[3, 4], 172);
    let left_view = left.view().unwrap();
    let right_view = right.view().unwrap();
    let expected_stream = left.__debug_execution_domain_stream_value();
    let expected_mr = left.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = concatenate::concatenate_columns(&[&left_view, &right_view]).unwrap();

    // Output provenance inherits from the first column.
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 171);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the first column's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn concatenate_tables_retains_explicit_domain_and_uses_exec_options() {
    let first = Table::from_columns(vec![explicit_i32_column(&[1, 2], 173)]).unwrap();
    let second = Table::from_columns(vec![explicit_i32_column(&[3, 4], 173)]).unwrap();
    let first_view = first.view().unwrap();
    let second_view = second.view().unwrap();
    let expected_stream = first.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = concatenate::concatenate_tables(&[&first_view, &second_view]).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 173);

    reset_sync_observation();
    let host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: concatenate_tables with mixed explicit domains succeeds.
/// Output provenance follows the first table in the sequence.
#[test]
fn concatenate_tables_succeeds_with_mixed_domains() {
    let first = Table::from_columns(vec![explicit_i32_column(&[1, 2], 174)]).unwrap();
    let second = Table::from_columns(vec![explicit_i32_column(&[3, 4], 175)]).unwrap();
    let first_view = first.view().unwrap();
    let second_view = second.view().unwrap();
    let expected_stream = first.__debug_execution_domain_stream_value();
    let expected_mr = first.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = concatenate::concatenate_tables(&[&first_view, &second_view]).unwrap();

    // Output provenance inherits from the first table.
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 174);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the first table's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    let host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn copying_gather_retains_explicit_domain_and_uses_exec_options() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 176)]).unwrap();
    let gather_map = explicit_i32_column(&[2, 0], 176);
    let expected_stream = table.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = copying::gather(&table.view().unwrap(), &gather_map.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 176);

    reset_sync_observation();
    let host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![30, 10]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: gather with mixed explicit domains succeeds.
/// Output provenance follows the data table (not the gather map).
#[test]
fn copying_gather_succeeds_with_mixed_domains() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 177)]).unwrap();
    let gather_map = explicit_i32_column(&[2, 0], 178);
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = copying::gather(&table.view().unwrap(), &gather_map.view().unwrap()).unwrap();

    // Output provenance inherits from the data table (not the gather map).
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 177);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the data table's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    let host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![30, 10]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn copying_gather_unchecked_succeeds_with_mixed_domains() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 1780)]).unwrap();
    let gather_map = explicit_i32_column(&[2, 0], 1781);
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();
    let expected_sync_stream = gather_map.__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let result = unsafe { copying::gather_unchecked(&table.view().unwrap(), &gather_map.view().unwrap()) }.unwrap();

    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 1780);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    let host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![30, 10]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn replace_nulls_retains_input_explicit_domain_and_uses_exec_options() {
    let input = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(179)
        .unwrap();
    let replacement = Scalar::from_value(99i32).unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = replace::replace_nulls(&input.view().unwrap(), &replacement).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 179);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 99, 3]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn replace_nulls_with_same_device_mixed_explicit_domains_fences_scalar_stream() {
    let input = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(1790)
        .unwrap();
    let replacement = Scalar::from_value(99i32).unwrap().__debug_rebind_execution_domain_explicit(1791).unwrap();
    let expected_exec_stream = input.__debug_execution_domain_stream_value();
    let fenced_stream = replacement.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = replace::replace_nulls(&input.view().unwrap(), &replacement).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 99, 3]);
}

#[test]
fn replace_nulls_column_retains_explicit_domain_and_uses_exec_options() {
    let input = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(180)
        .unwrap();
    let replacement = explicit_i32_column(&[10, 20, 30], 180);
    let expected_stream = input.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = replace::replace_nulls_column(&input.view().unwrap(), &replacement.view().unwrap()).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 180);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: replace_nulls_column with mixed explicit domains succeeds.
/// Output provenance follows the input column (not the replacement).
#[test]
fn replace_nulls_column_succeeds_with_mixed_domains() {
    let input = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(181)
        .unwrap();
    let replacement = explicit_i32_column(&[10, 20, 30], 182);
    let expected_stream = input.__debug_execution_domain_stream_value();
    let expected_mr = input.__debug_execution_domain_mr_value();
    let expected_sync_stream = replacement.__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let result = replace::replace_nulls_column(&input.view().unwrap(), &replacement.view().unwrap()).unwrap();

    // Output provenance inherits from the input column.
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_id(), 181);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the input column's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    // Verify correct values.
    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn join_inner_join_retains_explicit_domain_and_uses_exec_options() {
    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 183)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3, 4], 183)]).unwrap();
    let left_payload = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 183)]).unwrap();
    let right_payload = Table::from_columns(vec![explicit_i32_column(&[200, 300, 400], 183)]).unwrap();
    let expected_stream = left_keys.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = join::inner_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.left.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.right.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.left.__debug_execution_domain_id(), 183);
    assert_eq!(result.right.__debug_execution_domain_id(), 183);

    reset_sync_observation();
    let left_host = result.left.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let right_host = result.right.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(left_host, vec![2, 3]);
    assert_eq!(right_host, vec![2, 3]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: inner_join with mixed explicit domains succeeds.
/// Output provenance follows the left table.
#[test]
fn join_inner_join_succeeds_with_mixed_domains() {
    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 184)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3, 4], 185)]).unwrap();
    let left_payload = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 184)]).unwrap();
    let right_payload = Table::from_columns(vec![explicit_i32_column(&[200, 300, 400], 185)]).unwrap();
    let expected_stream = left_keys.__debug_execution_domain_stream_value();
    let expected_mr = left_keys.__debug_execution_domain_mr_value();
    let expected_sync_stream = right_keys.__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let result = join::inner_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    )
    .unwrap();

    // Output provenance inherits from the left table.
    assert_eq!(result.left.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.right.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.left.__debug_execution_domain_id(), 184);
    assert_eq!(result.right.__debug_execution_domain_id(), 184);
    assert_eq!(result.left.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.right.__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the left table's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    // Verify correct values.
    reset_sync_observation();
    let left_host = result.left.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let right_host = result.right.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(left_host, vec![2, 3]);
    assert_eq!(right_host, vec![2, 3]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn join_left_join_succeeds_with_mixed_domains() {
    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 1850)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 4], 1851)]).unwrap();
    let left_payload = DataFrame::from_series([explicit_i32_series("id", &[1, 2, 3], 1850)])
        .unwrap()
        .select(std::iter::empty::<&str>())
        .unwrap()
        .to_table_copy()
        .unwrap();
    let right_payload = DataFrame::from_series([explicit_i32_series("id", &[2, 4], 1851)])
        .unwrap()
        .select(std::iter::empty::<&str>())
        .unwrap()
        .to_table_copy()
        .unwrap();
    let expected_stream = left_keys.__debug_execution_domain_stream_value();
    let expected_mr = left_keys.__debug_execution_domain_mr_value();
    let expected_sync_stream = right_keys.__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let result = join::left_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(result.left.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.right.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.left.__debug_execution_domain_id(), 1850);
    assert_eq!(result.right.__debug_execution_domain_id(), 1850);
    assert_eq!(result.left.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.right.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    let mut left_host = nullable_i32_table_column(&result.left, 0);
    let mut right_host = nullable_i32_table_column(&result.right, 0);
    left_host.sort();
    right_host.sort();
    assert_eq!(left_host, vec![Some(1), Some(2), Some(3)]);
    assert_eq!(right_host, vec![None, None, Some(2)]);
}

#[test]
fn join_full_join_succeeds_with_mixed_domains() {
    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2], 1852)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3], 1853)]).unwrap();
    let left_payload = DataFrame::from_series([explicit_i32_series("id", &[1, 2], 1852)])
        .unwrap()
        .select(std::iter::empty::<&str>())
        .unwrap()
        .to_table_copy()
        .unwrap();
    let right_payload = DataFrame::from_series([explicit_i32_series("id", &[2, 3], 1853)])
        .unwrap()
        .select(std::iter::empty::<&str>())
        .unwrap()
        .to_table_copy()
        .unwrap();
    let expected_stream = left_keys.__debug_execution_domain_stream_value();
    let expected_mr = left_keys.__debug_execution_domain_mr_value();
    let expected_sync_stream = right_keys.__debug_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let result = join::full_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(result.left.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.right.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.left.__debug_execution_domain_id(), 1852);
    assert_eq!(result.right.__debug_execution_domain_id(), 1852);
    assert_eq!(result.left.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.right.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    let mut left_host = nullable_i32_table_column(&result.left, 0);
    let mut right_host = nullable_i32_table_column(&result.right, 0);
    left_host.sort();
    right_host.sort();
    assert_eq!(left_host, vec![None, Some(1), Some(2)]);
    assert_eq!(right_host, vec![None, Some(2), Some(3)]);
}

#[test]
fn groupby_aggregate_retains_explicit_domain_and_uses_exec_options() {
    let keys = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 2], 186)]).unwrap();
    let values = explicit_i32_column(&[10, 20, 30, 40], 186);
    let values_view = values.view().unwrap();
    let aggregations = [GroupByAgg::Sum];
    let requests = [AggregationRequest { values: &values_view, aggregations: &aggregations }];
    let expected_stream = keys.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let result = groupby::groupby_aggregate(&keys.view().unwrap(), &requests, groupby::NullPolicy::Exclude).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(result.keys.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.keys.__debug_execution_domain_id(), 186);
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 1);
    assert_eq!(result.results[0][0].__debug_execution_domain_id(), 186);

    reset_sync_observation();
    let key_host = result.keys.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let sum_host = result.results[0][0].view().unwrap().to_vec::<i64>().unwrap();
    assert_eq!(key_host, vec![1, 2]);
    assert_eq!(sum_host, vec![30, 70]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

/// v2 semantics: groupby_aggregate with mixed explicit domains succeeds.
/// Output provenance follows the keys table.
#[test]
fn groupby_aggregate_succeeds_with_mixed_domains() {
    let keys = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 2], 187)]).unwrap();
    let values = explicit_i32_column(&[10, 20, 30, 40], 188);
    let values_view = values.view().unwrap();
    let aggregations = [GroupByAgg::Sum];
    let requests = [AggregationRequest { values: &values_view, aggregations: &aggregations }];
    let expected_stream = keys.__debug_execution_domain_stream_value();
    let expected_mr = keys.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = groupby::groupby_aggregate(&keys.view().unwrap(), &requests, groupby::NullPolicy::Exclude).unwrap();

    // Output provenance inherits from the keys table.
    assert_eq!(result.keys.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.keys.__debug_execution_domain_id(), 187);
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 1);
    assert_eq!(result.results[0][0].__debug_execution_domain_id(), 187);
    assert_eq!(result.keys.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.results[0][0].__debug_execution_domain_mr_value(), expected_mr);

    // Execution used the keys table's stream and MR.
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    // Verify correct values.
    reset_sync_observation();
    let key_host = result.keys.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let sum_host = result.results[0][0].view().unwrap().to_vec::<i64>().unwrap();
    assert_eq!(key_host, vec![1, 2]);
    assert_eq!(sum_host, vec![30, 70]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn groupby_aggregate_with_same_device_mixed_explicit_domains_uses_keys_domain_and_fences_values_stream() {
    let keys = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 2], 1870)]).unwrap();
    let values = explicit_i32_column(&[10, 20, 30, 40], 1871);
    let values_view = values.view().unwrap();
    let aggregations = [GroupByAgg::Sum];
    let requests = [AggregationRequest { values: &values_view, aggregations: &aggregations }];
    let expected_exec_stream = keys.__debug_execution_domain_stream_value();
    let expected_mr = keys.__debug_execution_domain_mr_value();
    let fenced_stream = values.__debug_execution_domain_stream_value();

    reset_exec_observation();
    reset_sync_observation();
    let result = groupby::groupby_aggregate(&keys.view().unwrap(), &requests, groupby::NullPolicy::Exclude).unwrap();

    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), fenced_stream);
    assert_eq!(result.keys.__debug_execution_domain_id(), keys.__debug_execution_domain_id());
    assert_eq!(result.results[0][0].__debug_execution_domain_id(), keys.__debug_execution_domain_id());

    reset_sync_observation();
    let key_host = result.keys.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let sum_host = result.results[0][0].view().unwrap().to_vec::<i64>().unwrap();
    assert_eq!(key_host, vec![1, 2]);
    assert_eq!(sum_host, vec![30, 70]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn retained_domain_carriers_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Column>();
    assert_send_sync::<Series>();
    assert_send_sync::<Table>();
    assert_send_sync::<DataFrame>();
    assert_send_sync::<Scalar>();
    assert_send_sync::<CudaStream>();
    assert_send_sync::<DeviceMemoryResource>();
}

// === v2: mixed-domain deterministic resolution ===
//
// These tests express the v2 execution semantics contract where mixed-domain
// inputs are legal at operation boundaries. The operation resolves execution
// resources from the primary input's retained provenance, and the output
// carries that primary input's domain identity.
//
// INV-ER-04 is REMOVED: mixed explicit inputs no longer cause errors.
// INV-PR-02 is CHANGED: output inherits primary input's provenance.
// INV-PR-04 is CHANGED: identity does not gate operations.

#[test]
fn mixed_domain_binary_op_succeeds_with_left_provenance() {
    // Primary input for binary ops = left-hand side (spec table)
    let lhs = explicit_i32_column(&[1, 2, 3], 300);
    let rhs = explicit_i32_column(&[10, 20, 30], 301);
    let lhs_identity = lhs.__debug_execution_domain_id();
    let lhs_stream = lhs.__debug_execution_domain_stream_value();
    let lhs_mr = lhs.__debug_execution_domain_mr_value();
    assert_ne!(lhs_identity, rhs.__debug_execution_domain_id());

    reset_exec_observation();
    let result = binary::binary_operation_vv(
        &lhs.view().unwrap(),
        &rhs.view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();

    // Output provenance = left (primary) input
    assert_eq!(result.__debug_execution_domain_id(), lhs_identity);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_mr_value(), lhs_mr);
    // FFI stream/MR = left input's retained domain
    assert_eq!(last_exec_stream_value(), lhs_stream);
    assert_eq!(last_exec_mr_value(), lhs_mr);

    // Value correctness
    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(last_sync_stream_value(), lhs_stream);
}

#[test]
fn mixed_domain_concatenate_columns_succeeds_with_first_provenance() {
    // Primary input for concatenation = first column (spec table)
    let first = explicit_i32_column(&[1, 2], 302);
    let second = explicit_i32_column(&[3, 4, 5], 303);
    let first_identity = first.__debug_execution_domain_id();
    let first_stream = first.__debug_execution_domain_stream_value();
    let first_mr = first.__debug_execution_domain_mr_value();
    assert_ne!(first_identity, second.__debug_execution_domain_id());

    let first_view = first.view().unwrap();
    let second_view = second.view().unwrap();
    reset_exec_observation();
    let result = concatenate::concatenate_columns(&[&first_view, &second_view]).unwrap();

    // Output provenance = first column's domain
    assert_eq!(result.__debug_execution_domain_id(), first_identity);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_mr_value(), first_mr);
    assert_eq!(last_exec_stream_value(), first_stream);
    assert_eq!(last_exec_mr_value(), first_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    assert_eq!(last_sync_stream_value(), first_stream);
}

#[test]
fn mixed_domain_concatenate_tables_succeeds_with_first_provenance() {
    // Primary input for concatenation = first table (spec table)
    let first = Table::from_columns(vec![explicit_i32_column(&[1, 2], 304)]).unwrap();
    let second = Table::from_columns(vec![explicit_i32_column(&[3, 4], 305)]).unwrap();
    let first_identity = first.__debug_execution_domain_id();
    let first_stream = first.__debug_execution_domain_stream_value();
    let first_mr = first.__debug_execution_domain_mr_value();
    assert_ne!(first_identity, second.__debug_execution_domain_id());

    let first_view = first.view().unwrap();
    let second_view = second.view().unwrap();
    reset_exec_observation();
    let result = concatenate::concatenate_tables(&[&first_view, &second_view]).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), first_identity);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_mr_value(), first_mr);
    assert_eq!(last_exec_stream_value(), first_stream);
    assert_eq!(last_exec_mr_value(), first_mr);

    reset_sync_observation();
    let host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), first_stream);
}

#[test]
fn mixed_domain_filter_succeeds_with_data_provenance() {
    // Primary input for filter = the data being filtered (not the mask)
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 306)]).unwrap();
    let mask = Column::from_slice(&[true, false, true]).unwrap().__debug_rebind_execution_domain_explicit(307).unwrap();
    let data_identity = table.__debug_execution_domain_id();
    let data_stream = table.__debug_execution_domain_stream_value();
    let data_mr = table.__debug_execution_domain_mr_value();
    assert_ne!(data_identity, mask.__debug_execution_domain_id());

    reset_exec_observation();
    let filtered = filter::apply_boolean_mask(&table.view().unwrap(), &mask.view().unwrap()).unwrap();

    // Output provenance = data table's domain (not the mask's)
    assert_eq!(filtered.__debug_execution_domain_id(), data_identity);
    assert_eq!(filtered.__debug_execution_domain_kind(), "explicit");
    assert_eq!(filtered.__debug_execution_domain_mr_value(), data_mr);
    assert_eq!(last_exec_stream_value(), data_stream);
    assert_eq!(last_exec_mr_value(), data_mr);

    reset_sync_observation();
    let host = filtered.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 30]);
    assert_eq!(last_sync_stream_value(), data_stream);
}

#[test]
fn mixed_domain_gather_succeeds_with_data_provenance() {
    // Primary input for gather = the data being gathered (not the gather map)
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 308)]).unwrap();
    let gather_map = explicit_i32_column(&[2, 0], 309);
    let data_identity = table.__debug_execution_domain_id();
    let data_stream = table.__debug_execution_domain_stream_value();
    let data_mr = table.__debug_execution_domain_mr_value();
    assert_ne!(data_identity, gather_map.__debug_execution_domain_id());

    reset_exec_observation();
    let result = copying::gather(&table.view().unwrap(), &gather_map.view().unwrap()).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), data_identity);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_mr_value(), data_mr);
    assert_eq!(last_exec_stream_value(), data_stream);
    assert_eq!(last_exec_mr_value(), data_mr);

    reset_sync_observation();
    let host = result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![30, 10]);
    assert_eq!(last_sync_stream_value(), data_stream);
}

#[test]
fn mixed_domain_join_succeeds_with_left_provenance() {
    // Primary input for join = left table
    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 310)]).unwrap();
    let left_payload = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 310)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3, 4], 311)]).unwrap();
    let right_payload = Table::from_columns(vec![explicit_i32_column(&[200, 300, 400], 311)]).unwrap();
    let left_identity = left_keys.__debug_execution_domain_id();
    let left_stream = left_keys.__debug_execution_domain_stream_value();
    let left_mr = left_keys.__debug_execution_domain_mr_value();
    assert_ne!(left_identity, right_keys.__debug_execution_domain_id());

    reset_exec_observation();
    let result = join::inner_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    )
    .unwrap();

    // Output provenance = left table's domain
    assert_eq!(result.left.__debug_execution_domain_id(), left_identity);
    assert_eq!(result.right.__debug_execution_domain_id(), left_identity);
    assert_eq!(result.left.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.right.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.left.__debug_execution_domain_mr_value(), left_mr);
    assert_eq!(result.right.__debug_execution_domain_mr_value(), left_mr);
    assert_eq!(last_exec_stream_value(), left_stream);
    assert_eq!(last_exec_mr_value(), left_mr);

    reset_sync_observation();
    let left_host = result.left.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let right_host = result.right.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(left_host, vec![2, 3]);
    assert_eq!(right_host, vec![2, 3]);
    assert_eq!(last_sync_stream_value(), left_stream);
}

#[test]
fn mixed_domain_sort_by_key_succeeds_with_values_provenance() {
    // Primary input for sort_by_key = values table (not the keys table)
    let keys = Table::from_columns(vec![explicit_i32_column(&[3, 1, 2], 312)]).unwrap();
    let values = Table::from_columns(vec![explicit_i32_column(&[30, 10, 20], 313)]).unwrap();
    let values_identity = values.__debug_execution_domain_id();
    let values_stream = values.__debug_execution_domain_stream_value();
    let values_mr = values.__debug_execution_domain_mr_value();
    assert_ne!(values_identity, keys.__debug_execution_domain_id());

    reset_exec_observation();
    let sorted = sorting::sort_by_key(&values.view().unwrap(), &keys.view().unwrap(), None, None).unwrap();

    assert_eq!(sorted.__debug_execution_domain_id(), values_identity);
    assert_eq!(sorted.__debug_execution_domain_kind(), "explicit");
    assert_eq!(sorted.__debug_execution_domain_mr_value(), values_mr);
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);

    reset_sync_observation();
    let host = sorted.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 20, 30]);
    assert_eq!(last_sync_stream_value(), values_stream);
}

#[test]
fn mixed_domain_replace_nulls_column_succeeds_with_input_provenance() {
    // Primary input for replace_nulls = input column (not the replacement)
    let input = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(314)
        .unwrap();
    let replacement = explicit_i32_column(&[10, 20, 30], 315);
    let input_identity = input.__debug_execution_domain_id();
    let input_stream = input.__debug_execution_domain_stream_value();
    let input_mr = input.__debug_execution_domain_mr_value();
    assert_ne!(input_identity, replacement.__debug_execution_domain_id());

    reset_exec_observation();
    let result = replace::replace_nulls_column(&input.view().unwrap(), &replacement.view().unwrap()).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), input_identity);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_mr_value(), input_mr);
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(last_sync_stream_value(), input_stream);
}

#[test]
fn mixed_domain_search_contains_succeeds_with_haystack_provenance() {
    // Primary input for search/contains = haystack (not the needles)
    let haystack = explicit_i32_column(&[10, 20, 30, 40], 316);
    let needles = explicit_i32_column(&[20, 50, 10], 317);
    let haystack_identity = haystack.__debug_execution_domain_id();
    let haystack_stream = haystack.__debug_execution_domain_stream_value();
    let haystack_mr = haystack.__debug_execution_domain_mr_value();
    assert_ne!(haystack_identity, needles.__debug_execution_domain_id());

    reset_exec_observation();
    let result = search::contains(&haystack.view().unwrap(), &needles.view().unwrap()).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), haystack_identity);
    assert_eq!(result.__debug_execution_domain_kind(), "explicit");
    assert_eq!(result.__debug_execution_domain_mr_value(), haystack_mr);
    assert_eq!(last_exec_stream_value(), haystack_stream);
    assert_eq!(last_exec_mr_value(), haystack_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), haystack_stream);
}

// ---------------------------------------------------------------------------
// Column::from_scalar — explicit-domain provenance propagation
// ---------------------------------------------------------------------------

#[test]
fn column_from_scalar_propagates_explicit_domain() {
    let scalar = Scalar::from_value(42i32).unwrap().__debug_rebind_execution_domain_explicit(900).unwrap();
    let scalar_stream = scalar.__debug_execution_domain_stream_value();
    assert_ne!(scalar_stream, 0, "explicit domain must have non-zero stream");

    reset_exec_observation();
    let col = Column::from_scalar(&scalar, 5).unwrap();

    // Output domain must inherit the scalar's explicit domain.
    assert_eq!(col.__debug_execution_domain_kind(), "explicit");
    assert_eq!(col.__debug_execution_domain_id(), 900);
    assert_eq!(col.__debug_execution_domain_stream_value(), scalar_stream);
    assert_eq!(col.__debug_execution_domain_mr_value(), last_exec_mr_value());

    // The _ex FFI path must have been called with the explicit stream.
    assert_eq!(last_exec_stream_value(), scalar_stream);

    // Round-trip the values to confirm correctness.
    reset_sync_observation();
    assert_eq!(col.view().unwrap().to_vec::<i32>().unwrap(), vec![42, 42, 42, 42, 42]);
    assert_eq!(last_sync_stream_value(), scalar_stream);
}
