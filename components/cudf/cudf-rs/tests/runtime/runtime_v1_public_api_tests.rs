//! Public `Runtime v1` contract tests.

use std::{ffi::c_void, ptr};

use cudf::{
    Agg, AggSpec, DataFrame, DataFrameCsvReaderOptions, DataFrameCsvWriterOptions, ErrorKind, ExecutionContext,
    MergeOptions, Runtime, Series, SortOptions, core::memory::DeviceMemoryResource, io::parquet::ParquetReaderOptions,
    runtime::CudaStream,
};
use cudf_sys::{
    cudf_test_current_device_resource_raw, cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value,
    cudf_test_exec_reset_observation,
};
use tempfile::TempDir;

fn assert_send_sync<T: Send + Sync>() {}

fn obs_reset() {
    unsafe { cudf_test_exec_reset_observation() };
}

fn obs_last_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_stream_value failed");
    out
}

fn obs_last_mr() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_mr_raw failed");
    out
}

fn obs_current_device_resource() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, 0, "cudf_test_current_device_resource_raw failed");
    out
}

fn device_ptr(df: &DataFrame, name: &str) -> *const c_void {
    df.series(name).unwrap().as_column_view().unwrap().device_data_ptr().unwrap()
}

fn sample_runtime() -> Runtime {
    let stream = CudaStream::new().unwrap();
    let mr = DeviceMemoryResource::pool(16 * 1024 * 1024, 0).unwrap();
    Runtime::builder().with_stream(stream).with_memory_resource(mr).build().unwrap()
}

#[test]
fn runtime_and_execution_context_are_public_and_send_sync() {
    assert_send_sync::<Runtime>();
    assert_send_sync::<ExecutionContext<'static>>();

    let rt = Runtime::builder().build().unwrap();
    let _ctx = ExecutionContext::new().with_stream(rt.stream()).with_memory_resource(rt.memory_resource());
}

#[test]
fn runtime_seeded_series_constructor_uses_runtime_domain_and_exec_options() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let default_mr = obs_current_device_resource() as usize;

    obs_reset();
    let series = Series::new_with_runtime(&rt, "id", &[3i32, 1, 2]).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_ne!(obs_last_mr() as usize, default_mr);
    assert_eq!(series.as_column().__debug_execution_domain_kind(), "explicit");
    assert_eq!(series.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(series.as_column().__debug_execution_domain_mr_value(), rt.memory_resource().__debug_raw_handle_value());
}

#[test]
fn runtime_seeded_string_series_constructor_uses_runtime_domain_and_exec_options() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let default_mr = obs_current_device_resource() as usize;

    obs_reset();
    let series = Series::from_strings_with_runtime(&rt, "name", &["alice", "bob"]).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_ne!(obs_last_mr() as usize, default_mr);
    assert_eq!(series.as_column().__debug_execution_domain_kind(), "explicit");
    assert_eq!(series.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(series.as_column().__debug_execution_domain_mr_value(), rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(series.to_strings().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
}

#[test]
fn sort_values_with_context_on_runtime_seeded_dataframe_uses_override_stream_and_retains_identity() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([("id", Series::new_with_runtime(&rt, "id", &[3i32, 1, 2]).unwrap())]).unwrap();

    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let sorted = df.sort_values_with_context(["id"], SortOptions::default(), &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(sorted.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(sorted.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(sorted.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// [INV-AP-01] Ambient promotion rejection: borrowed context overrides cannot promote
/// ambient-default objects to explicit domains (Rule 5 — preserved in v2).
#[test]
fn sort_values_with_context_rejects_ambient_default_promotion() {
    let df = DataFrame::from_series([("id", Series::new::<i32>("id", &[3, 1, 2]).unwrap())]).unwrap();
    let override_stream = CudaStream::new().unwrap();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    let err = df.sort_values_with_context(["id"], SortOptions::default(), &ctx).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");
}

#[test]
fn merge_groupby_gather_and_filter_with_context_are_public_and_route_override_stream() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&rt, "val", &[10i64, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt, "id", &[2i32, 3, 4]).unwrap()),
        ("score", Series::new_with_runtime(&rt, "score", &[200i64, 300, 400]).unwrap()),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let joined = left.merge_with_context(&right, MergeOptions::new().how_inner().on(["id"]), &ctx).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(joined.num_rows(), 2);
    assert_eq!(joined.__debug_execution_domain_stream_value(), override_stream_value);

    let grouped = joined.groupby(["id"]).agg_with_context(&[AggSpec::new("val", Agg::Sum)], &ctx).unwrap();
    assert_eq!(grouped.__debug_execution_domain_stream_value(), override_stream_value);

    let gather_indices = Series::new_with_runtime(&rt, "idx", &[1i32, 0]).unwrap();
    obs_reset();
    let gathered = grouped.gather_with_context(&gather_indices, &ctx).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(gathered.__debug_execution_domain_stream_value(), override_stream_value);

    let mask = Series::new_with_runtime(&rt, "mask", &[true, false]).unwrap();
    obs_reset();
    let filtered = gathered.filter_with_context(&mask, &ctx).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(filtered.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(filtered.num_rows(), 1);
}

#[test]
fn parquet_read_with_runtime_seeds_explicit_domain_and_uses_runtime_stream() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_seed.parquet");
    let df = DataFrame::from_series([
        ("id", Series::new::<i32>("id", &[1, 2, 3]).unwrap()),
        ("value", Series::new::<i32>("value", &[10, 20, 30]).unwrap()),
    ])
    .unwrap();
    df.write_parquet(&path).unwrap();

    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let opts = ParquetReaderOptions::builder().filepath(&path).with_runtime(&rt).build();

    obs_reset();
    let read_df = DataFrame::read_parquet_with_options(opts).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(read_df.__debug_execution_domain_kind(), "explicit");
    assert_eq!(read_df.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(read_df.column_names(), vec!["id", "value"]);
}

#[test]
fn csv_read_with_runtime_seeds_explicit_domain_and_uses_runtime_stream() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_seed.csv");
    std::fs::write(&path, "id,value\n1,10\n2,20\n3,30\n").unwrap();

    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let opts = DataFrameCsvReaderOptions::builder().filepath(&path).with_runtime(&rt).build();

    obs_reset();
    let read_df = DataFrame::read_csv_with_options(opts).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(read_df.__debug_execution_domain_kind(), "explicit");
    assert_eq!(read_df.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(read_df.column_names(), vec!["id", "value"]);
}

#[test]
fn dataframe_write_parquet_with_options_uses_retained_runtime_stream() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write.parquet");
    let rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt, "id", &[4i32, 5, 6]).unwrap()),
        ("value", Series::new_with_runtime(&rt, "value", &[40i32, 50, 60]).unwrap()),
    ])
    .unwrap();

    obs_reset();
    df.write_parquet_with_options(cudf::io::parquet::ParquetWriterOptions::builder().filepath(&path).build()).unwrap();

    assert_eq!(obs_last_stream(), rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_parquet(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn dataframe_write_parquet_with_options_runtime_override_recovers_mixed_explicit_domains_into_target_runtime() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write_override.parquet");
    let data_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&data_rt, "id", &[4i32, 5, 6]).unwrap()),
        ("value", Series::new_with_runtime(&data_rt, "value", &[40i32, 50, 60]).unwrap()),
    ])
    .unwrap();

    obs_reset();
    df.write_parquet_with_options(
        cudf::io::parquet::ParquetWriterOptions::builder().filepath(&path).with_runtime(&target_rt).build(),
    )
    .unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().__debug_raw_handle_value());

    let mut sync_out: usize = 0;
    let status = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut sync_out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read sync stream observation");
    assert_eq!(sync_out, target_rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_parquet(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn dataframe_write_csv_with_options_uses_retained_runtime_stream() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write.csv");
    let rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt, "id", &[4i32, 5, 6]).unwrap()),
        ("value", Series::new_with_runtime(&rt, "value", &[40i32, 50, 60]).unwrap()),
    ])
    .unwrap();

    obs_reset();
    df.write_csv_with_options(DataFrameCsvWriterOptions::builder().filepath(&path).build()).unwrap();

    assert_eq!(obs_last_stream(), rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_csv(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec::<i64>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn dataframe_write_csv_with_options_runtime_override_recovers_mixed_explicit_domains_into_target_runtime() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write_override.csv");
    let data_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&data_rt, "id", &[4i32, 5, 6]).unwrap()),
        ("value", Series::new_with_runtime(&data_rt, "value", &[40i32, 50, 60]).unwrap()),
    ])
    .unwrap();

    obs_reset();
    df.write_csv_with_options(DataFrameCsvWriterOptions::builder().filepath(&path).with_runtime(&target_rt).build())
        .unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().__debug_raw_handle_value());

    let mut sync_out: usize = 0;
    let status = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut sync_out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read sync stream observation");
    assert_eq!(sync_out, target_rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_csv(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec::<i64>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn merge_with_runtime_recovers_mixed_explicit_domains_into_target_runtime() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&left_rt, "val", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[2i32, 3, 4]).unwrap()),
        ("score", Series::new_with_runtime(&right_rt, "score", &[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();

    obs_reset();
    let joined = left.merge_with_runtime(&right, MergeOptions::new().how_inner().on(["id"]), &target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(joined.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(joined.__debug_execution_domain_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(joined.column_names(), vec!["id", "val", "score"]);
    assert_eq!(joined.series("id").unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
}

#[test]
fn merge_with_context_recovers_mixed_explicit_domains_into_left_lineage() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&left_rt, "val", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[2i32, 3, 4]).unwrap()),
        ("score", Series::new_with_runtime(&right_rt, "score", &[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let joined = left.merge_with_context(&right, MergeOptions::new().how_inner().on(["id"]), &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(joined.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(joined.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(joined.column_names(), vec!["id", "val", "score"]);
    assert_eq!(joined.series("score").unwrap().to_vec::<i32>().unwrap(), vec![200, 300]);
}

#[test]
fn concat_with_runtime_recovers_mixed_explicit_domains_into_target_runtime() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2]).unwrap()),
        ("val", Series::new_with_runtime(&left_rt, "val", &[10i32, 20]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[3i32, 4]).unwrap()),
        ("val", Series::new_with_runtime(&right_rt, "val", &[30i32, 40]).unwrap()),
    ])
    .unwrap();

    obs_reset();
    let combined = left.concat_with_runtime(&right, &target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(combined.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(combined.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn merge_split_with_runtime_recovers_mixed_explicit_domains_into_target_runtime() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&left_rt, "val", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[2i32, 3, 4]).unwrap()),
        ("score", Series::new_with_runtime(&right_rt, "score", &[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();

    obs_reset();
    let split = left.merge_split_with_runtime(&right, MergeOptions::new().how_inner().on(["id"]), &target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(split.left.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(split.right.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(split.left.series("id").unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(split.right.series("score").unwrap().to_vec::<i32>().unwrap(), vec![200, 300]);
}

#[test]
fn merge_split_with_context_recovers_mixed_explicit_domains_into_left_lineage() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&left_rt, "val", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[2i32, 3, 4]).unwrap()),
        ("score", Series::new_with_runtime(&right_rt, "score", &[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let split = left.merge_split_with_context(&right, MergeOptions::new().how_inner().on(["id"]), &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(split.left.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.right.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.left.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(split.right.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(split.left.series("val").unwrap().to_vec::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(split.right.series("score").unwrap().to_vec::<i32>().unwrap(), vec![200, 300]);
}

#[test]
fn concat_with_context_recovers_mixed_explicit_domains_into_left_lineage() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2]).unwrap()),
        ("val", Series::new_with_runtime(&left_rt, "val", &[10i32, 20]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[3i32, 4]).unwrap()),
        ("val", Series::new_with_runtime(&right_rt, "val", &[30i32, 40]).unwrap()),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let combined = left.concat_with_context(&right, &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(combined.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(combined.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(combined.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn with_column_with_runtime_deep_copies_into_target_runtime() {
    let base_rt = sample_runtime();
    let extra_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&base_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&base_rt, "val", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let extra = Series::new_with_runtime(&extra_rt, "extra", &[100i32, 200, 300]).unwrap();
    let original_id_ptr = device_ptr(&df, "id");
    let original_extra_ptr = extra.as_column_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let result = df.with_column_with_runtime("extra", extra, &target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(result.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_ne!(device_ptr(&result, "id"), original_id_ptr);
    assert_ne!(device_ptr(&result, "extra"), original_extra_ptr);
    assert_eq!(result.series("extra").unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
}

#[test]
fn assign_with_runtime_deep_copies_existing_and_new_columns_into_target_runtime() {
    let base_rt = sample_runtime();
    let extra_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&base_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&base_rt, "val", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let replacement = Series::new_with_runtime(&extra_rt, "replacement", &[7i32, 8, 9]).unwrap();
    let bonus = Series::new_with_runtime(&extra_rt, "bonus", &[70i32, 80, 90]).unwrap();
    let original_id_ptr = device_ptr(&df, "id");
    let original_replacement_ptr = replacement.as_column_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let result = df.assign_with_runtime([("val", replacement), ("bonus", bonus)], &target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(result.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_ne!(device_ptr(&result, "id"), original_id_ptr);
    assert_ne!(device_ptr(&result, "val"), original_replacement_ptr);
    assert_eq!(result.series("val").unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(result.series("bonus").unwrap().to_vec::<i32>().unwrap(), vec![70, 80, 90]);
}

// === v2: *_with_context mixed-domain + context-derived provenance ===
//
// These tests express the v2 contract where *_with_context methods work with
// mixed-domain inputs and produce context-derived provenance. The explicit
// context stream is the operation authority (Rule 1), not the input provenance.
//
// INV-PR-03: output provenance reflects context overrides.
// INV-SY-03: sync targets the context's stream after mixed-domain ops.

#[test]
fn sort_values_with_context_succeeds_on_mixed_domain_inputs() {
    // Two-column DataFrame from domain A, sort with context derived from domain B.
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt_a, "id", &[3i32, 1, 2]).unwrap()),
        ("val", Series::new_with_runtime(&rt_a, "val", &[30i32, 10, 20]).unwrap()),
    ])
    .unwrap();

    // Context borrows stream from rt_b — different domain than df
    let ctx = ExecutionContext::new().with_stream(rt_b.stream()).with_memory_resource(rt_b.memory_resource());
    let ctx_stream_value = rt_b.stream().__debug_raw_handle_value();
    let ctx_mr_value = rt_b.memory_resource().__debug_raw_handle_value();
    assert_ne!(ctx_stream_value, rt_a.stream().__debug_raw_handle_value());

    obs_reset();
    let sorted = df.sort_values_with_context(["id"], SortOptions::default(), &ctx).unwrap();

    // Output FFI stream = context stream
    assert_eq!(obs_last_stream(), ctx_stream_value);
    // Output FFI MR = context MR
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    // Output provenance stream = context stream
    assert_eq!(sorted.__debug_execution_domain_stream_value(), ctx_stream_value);
    // Output provenance MR = context MR
    assert_eq!(sorted.__debug_execution_domain_mr_value(), ctx_mr_value);
    // Output identity is preserved from the primary input (df)
    assert_eq!(sorted.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    // Value correctness
    assert_eq!(sorted.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn merge_with_context_succeeds_on_mixed_domain_inputs() {
    // DataFrames from different domains, merge with explicit context.
    let rt_left = sample_runtime();
    let rt_right = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt_left, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_with_runtime(&rt_left, "val", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt_right, "id", &[2i32, 3, 4]).unwrap()),
        ("score", Series::new_with_runtime(&rt_right, "score", &[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();

    // Context uses left's stream — but the point is mixed inputs succeed
    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let left_mr_value = left.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let joined = left.merge_with_context(&right, MergeOptions::new().how_inner().on(["id"]), &ctx).unwrap();

    // Output FFI stream = context stream
    assert_eq!(obs_last_stream(), ctx_stream_value);
    // Output FFI MR = primary input MR (stream-only override)
    assert_eq!(obs_last_mr() as usize, left_mr_value);
    // Output provenance stream = context stream
    assert_eq!(joined.__debug_execution_domain_stream_value(), ctx_stream_value);
    // Output provenance MR = primary input MR (stream-only override)
    assert_eq!(joined.__debug_execution_domain_mr_value(), left_mr_value);
    // Output identity = left (primary) input's identity
    assert_eq!(joined.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    // Value correctness
    assert_eq!(joined.column_names(), vec!["id", "val", "score"]);
    assert_eq!(joined.series("id").unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
}

#[test]
fn gather_with_context_succeeds_on_mixed_domain_inputs() {
    // Data from domain A, gather map from domain B, context from domain C.
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df =
        DataFrame::from_series([("id", Series::new_with_runtime(&rt_a, "id", &[10i32, 20, 30]).unwrap())]).unwrap();
    let indices = Series::new_with_runtime(&rt_b, "idx", &[2i32, 0]).unwrap();

    // Context uses a third stream
    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let df_mr_value = df.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let gathered = df.gather_with_context(&indices, &ctx).unwrap();

    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, df_mr_value);
    assert_eq!(gathered.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(gathered.__debug_execution_domain_mr_value(), df_mr_value);
    assert_eq!(gathered.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(gathered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![30, 10]);
}

#[test]
fn filter_with_context_succeeds_on_mixed_domain_inputs() {
    // Data from domain A, mask from domain B, context from domain C.
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df =
        DataFrame::from_series([("id", Series::new_with_runtime(&rt_a, "id", &[10i32, 20, 30]).unwrap())]).unwrap();
    let mask = Series::new_with_runtime(&rt_b, "mask", &[true, false, true]).unwrap();

    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let df_mr_value = df.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let filtered = df.filter_with_context(&mask, &ctx).unwrap();

    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, df_mr_value);
    assert_eq!(filtered.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(filtered.__debug_execution_domain_mr_value(), df_mr_value);
    assert_eq!(filtered.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(filtered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 30]);
}

#[test]
fn mixed_domain_with_context_output_sync_targets_context_stream() {
    // After mixed-domain op with explicit context, synchronize_for_host_visibility
    // must target the context's stream (which is now the output's provenance stream).
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df =
        DataFrame::from_series([("id", Series::new_with_runtime(&rt_a, "id", &[10i32, 20, 30]).unwrap())]).unwrap();
    let indices = Series::new_with_runtime(&rt_b, "idx", &[2i32, 0]).unwrap();

    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let df_mr_value = df.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);
    assert_ne!(df.__debug_execution_domain_stream_value(), indices.as_column().__debug_execution_domain_stream_value());
    assert_ne!(ctx_stream_value, df.__debug_execution_domain_stream_value());
    assert_ne!(ctx_stream_value, indices.as_column().__debug_execution_domain_stream_value());

    obs_reset();
    let gathered = df.gather_with_context(&indices, &ctx).unwrap();

    // Output provenance stream = context stream
    assert_eq!(gathered.__debug_execution_domain_stream_value(), ctx_stream_value);
    // Stream-only override keeps the primary input MR.
    assert_eq!(obs_last_mr() as usize, df_mr_value);
    assert_eq!(gathered.__debug_execution_domain_mr_value(), df_mr_value);

    // Now read host data — this triggers synchronize_for_host_visibility.
    // The sync must target the context's stream, not either input provenance stream.
    let mut sync_out: usize = 0;
    let host = gathered.series("id").unwrap().to_vec::<i32>().unwrap();
    let status = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut sync_out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read sync stream observation");
    assert_eq!(sync_out, ctx_stream_value, "sync must target context stream, not either input provenance stream");
    assert_eq!(host, vec![30, 10]);
}
