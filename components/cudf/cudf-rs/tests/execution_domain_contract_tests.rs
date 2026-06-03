//! Contract tests for retained execution-domain provenance.
//!
//! Phase 1 keeps the domain substrate private, so these tests use hidden
//! debug hooks instead of a public runtime API.

use std::{ffi::c_void, ptr};

use cudf::{
    Agg, AggSpec, Column, DataFrame, ErrorKind, Scalar, Series, Table, memory::DeviceMemoryResource, stream::CudaStream,
};
use cudf_sys::{
    cudf_test_current_device_resource_raw, cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value,
    cudf_test_exec_last_sync_stream_value, cudf_test_exec_reset_observation,
};

fn reset_sync_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

fn reset_exec_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

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
    let df = DataFrame::from_columns([("id", series)]).unwrap();
    assert_eq!(df.__debug_execution_domain_kind(), "ambient_default");
    assert_eq!(df.__debug_execution_domain_id(), 0);
    assert_eq!(df.series("id").unwrap().as_column().__debug_execution_domain_id(), 0);
}

#[test]
fn column_copy_preserves_execution_domain_identity() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(41).unwrap();
    let copied = col.copy().unwrap();

    assert_eq!(col.__debug_execution_domain_kind(), "explicit");
    assert_eq!(copied.__debug_execution_domain_kind(), "explicit");
    assert_eq!(copied.__debug_execution_domain_id(), col.__debug_execution_domain_id());
}

#[test]
fn table_copy_and_release_preserve_execution_domain_identity() {
    let first = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(7).unwrap();
    let second = Column::from_slice(&[10i32, 20, 30]).unwrap().__debug_rebind_execution_domain_explicit(7).unwrap();

    let table = Table::from_columns(vec![first, second]).unwrap();
    let copied = table.copy().unwrap();

    assert_eq!(table.__debug_execution_domain_kind(), "explicit");
    assert_eq!(copied.__debug_execution_domain_id(), table.__debug_execution_domain_id());

    let released = copied.into_columns().unwrap();
    assert_eq!(released.len(), 2);
    assert!(released.iter().all(|col| col.__debug_execution_domain_id() == table.__debug_execution_domain_id()));
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

#[test]
fn dataframe_new_rejects_mixed_domain_columns() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(100).unwrap();
    let right = Column::from_slice(&[4i32, 5, 6]).unwrap().__debug_rebind_execution_domain_explicit(200).unwrap();

    let err = DataFrame::new(vec!["left".into(), "right".into()], vec![left, right]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::new");

    let context = err.context();
    assert!(context.iter().any(|(key, value)| *key == "left_domain" && value == "explicit:100"));
    assert!(context.iter().any(|(key, value)| *key == "right_domain" && value == "explicit:200"));
}

#[test]
fn table_from_columns_rejects_mixed_domain_columns() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(3).unwrap();
    let right = Column::from_slice(&[4i32, 5, 6]).unwrap().__debug_rebind_execution_domain_explicit(4).unwrap();

    let err = Table::from_columns(vec![left, right]).err().expect("expected mixed-domain table construction to fail");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "table::from_columns");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value == "explicit:3"));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value == "explicit:4"));
}

#[test]
fn dataframe_high_level_row_ops_preserve_explicit_domain_identity() {
    let df = DataFrame::from_columns([
        ("id", explicit_i32_series("id", &[3, 1, 2], 55)),
        ("value", explicit_i32_series("value", &[30, 10, 20], 55)),
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
    let left = DataFrame::from_columns([("id", explicit_i32_series("id", &[1, 2], 81))]).unwrap();
    let right = DataFrame::from_columns([("id", explicit_i32_series("id", &[3, 4], 81))]).unwrap();

    let combined = left.concat(&right).unwrap();

    assert_eq!(combined.__debug_execution_domain_kind(), "explicit");
    assert_eq!(combined.__debug_execution_domain_id(), 81);
    assert_eq!(combined.series("id").unwrap().as_column().__debug_execution_domain_id(), 81);
}

#[test]
fn dataframe_concat_rejects_mixed_explicit_domains() {
    let left = DataFrame::from_columns([("id", explicit_i32_series("id", &[1, 2], 8))]).unwrap();
    let right = DataFrame::from_columns([("id", explicit_i32_series("id", &[3, 4], 9))]).unwrap();

    let err = left.concat(&right).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::concat");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value == "explicit:8"));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value == "explicit:9"));
}

#[test]
fn dataframe_merge_preserves_explicit_domain_identity() {
    let left = DataFrame::from_columns([
        ("id", explicit_i32_series("id", &[1, 2, 3], 64)),
        ("left", explicit_i32_series("left", &[10, 20, 30], 64)),
    ])
    .unwrap();
    let right = DataFrame::from_columns([
        ("id", explicit_i32_series("id", &[2, 3, 4], 64)),
        ("right", explicit_i32_series("right", &[200, 300, 400], 64)),
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

#[test]
fn dataframe_merge_rejects_mixed_explicit_domains() {
    let left = DataFrame::from_columns([
        ("id", explicit_i32_series("id", &[1, 2, 3], 11)),
        ("left", explicit_i32_series("left", &[10, 20, 30], 11)),
    ])
    .unwrap();
    let right = DataFrame::from_columns([
        ("id", explicit_i32_series("id", &[2, 3, 4], 12)),
        ("right", explicit_i32_series("right", &[200, 300, 400], 12)),
    ])
    .unwrap();

    let err = left.merge(&right, cudf::MergeOptions::new().how_inner().on(["id"])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::merge");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value == "explicit:11"));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value == "explicit:12"));
}

#[test]
fn dataframe_fillna_with_series_rejects_mixed_explicit_domains() {
    let df = DataFrame::from_columns([(
        "value",
        Series::from_column(
            Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
                .unwrap()
                .__debug_rebind_execution_domain_explicit(21)
                .unwrap(),
        )
        .unwrap()
        .rename("value"),
    )])
    .unwrap();
    let fill = explicit_i32_series("fill", &[10, 20, 30], 22);

    let err = df.fillna_with_series("value", &fill).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "dataframe::fillna_with_series");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value == "explicit:21"));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value == "explicit:22"));
}

#[test]
fn series_view_to_vec_synchronizes_retained_explicit_domain_stream() {
    let series = explicit_i32_series("id", &[4, 5, 6], 71);
    let expected_stream = series.as_column().__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");

    reset_sync_observation();
    let host: Vec<i32> = series.view().unwrap().to_vec().unwrap();

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
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn scalar_string_value_synchronizes_retained_explicit_domain_stream() {
    let scalar = Scalar::from_string("hello").unwrap().__debug_rebind_execution_domain_explicit(73).unwrap();
    let expected_stream = scalar.__debug_execution_domain_stream_value();
    assert_ne!(expected_stream, 0, "explicit domains should expose a non-zero retained stream value");

    reset_sync_observation();
    let value = scalar.string_value().unwrap();

    assert_eq!(value, "hello");
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn dataframe_write_parquet_synchronizes_retained_explicit_domain_stream() {
    let df = DataFrame::from_columns([
        ("id", explicit_i32_series("id", &[1, 2, 3], 74)),
        ("value", explicit_i32_series("value", &[10, 20, 30], 74)),
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
    let df = DataFrame::from_columns([
        ("id", explicit_i32_series("id", &[1, 2, 3], 75)),
        ("value", explicit_i32_series("value", &[10, 20, 30], 75)),
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
fn series_fillna_uses_retained_explicit_domain_stream() {
    let nullable = explicit_nullable_i32_series("nullable", &[1, 0, 3], &[true, false, true], 80);
    let expected_stream = nullable.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    let filled = nullable.fillna(99i32).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(filled.as_column().__debug_execution_domain_id(), 80);
    let host: Vec<i32> = filled.view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![1, 99, 3]);
}

#[test]
fn dataframe_fillna_with_series_uses_retained_explicit_domain_stream_and_mr() {
    let df = DataFrame::from_columns([(
        "value",
        explicit_nullable_i32_series("value", &[1, 0, 3], &[true, false, true], 81),
    )])
    .unwrap();
    let fill = explicit_i32_series("fill", &[10, 20, 30], 81);
    let expected_stream = df.__debug_execution_domain_stream_value();
    let default_mr = current_device_resource_value();

    reset_exec_observation();
    let filled = df.fillna_with_series("value", &fill).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_ne!(last_exec_mr_value(), 0);
    assert_ne!(last_exec_mr_value(), default_mr);
    assert_eq!(filled.__debug_execution_domain_id(), 81);
    let host: Vec<i32> = filled.series("value").unwrap().column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![1, 20, 3]);
}

#[test]
fn dataframe_groupby_agg_preserves_explicit_domain_identity() {
    let df = DataFrame::from_columns([
        ("key", explicit_i32_series("key", &[1, 1, 2, 2], 77)),
        ("value", explicit_i32_series("value", &[10, 20, 30, 40], 77)),
    ])
    .unwrap();

    let grouped = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

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
