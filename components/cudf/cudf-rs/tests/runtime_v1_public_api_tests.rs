//! Public `Runtime v1` contract tests.

use std::{ffi::c_void, ptr};

use cudf::{
    Agg, AggSpec, DataFrame, DataFrameCsvReaderOptions, DataFrameCsvWriterOptions, ErrorKind, ExecutionContext,
    MergeOptions, Runtime, Series, SortOptions, io::parquet::ParquetReaderOptions,
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
    df.series_view(name).unwrap().device_data_ptr().unwrap()
}

fn sample_runtime() -> Runtime {
    let stream = cudf::stream::CudaStream::new().unwrap();
    let mr = cudf::memory::DeviceMemoryResource::pool(16 * 1024 * 1024, 0).unwrap();
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
    assert_eq!(series.view().unwrap().to_strings().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
}

#[test]
fn sort_values_with_context_on_runtime_seeded_dataframe_uses_override_stream_and_retains_identity() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([("id", Series::new_with_runtime(&rt, "id", &[3i32, 1, 2]).unwrap())]).unwrap();

    let override_stream = cudf::stream::CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let sorted = df.sort_values_with_context(["id"], SortOptions::default(), &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(sorted.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(sorted.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(sorted.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn sort_values_with_context_rejects_ambient_default_promotion() {
    let df = DataFrame::from_series([("id", Series::new::<i32>("id", &[3, 1, 2]).unwrap())]).unwrap();
    let override_stream = cudf::stream::CudaStream::new().unwrap();
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
    let override_stream = cudf::stream::CudaStream::new().unwrap();
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
    let opts = ParquetReaderOptions::new(&path).with_runtime(&rt);

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
    let opts = DataFrameCsvReaderOptions::new(&path).with_runtime(&rt);

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
    df.write_parquet_with_options(cudf::io::parquet::ParquetWriterOptions::new(&path)).unwrap();

    assert_eq!(obs_last_stream(), rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_parquet(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![4, 5, 6]);
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
    df.write_csv_with_options(DataFrameCsvWriterOptions::new(&path)).unwrap();

    assert_eq!(obs_last_stream(), rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_csv(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series_view("id").unwrap().to_vec::<i64>().unwrap(), vec![4, 5, 6]);
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
    assert_eq!(joined.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
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
    let override_stream = cudf::stream::CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let joined = left.merge_with_context(&right, MergeOptions::new().how_inner().on(["id"]), &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(joined.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(joined.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(joined.column_names(), vec!["id", "val", "score"]);
    assert_eq!(joined.series_view("score").unwrap().to_vec::<i32>().unwrap(), vec![200, 300]);
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
    assert_eq!(combined.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
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
    assert_eq!(split.left.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(split.right.series_view("score").unwrap().to_vec::<i32>().unwrap(), vec![200, 300]);
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
    let override_stream = cudf::stream::CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let split = left.merge_split_with_context(&right, MergeOptions::new().how_inner().on(["id"]), &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(split.left.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.right.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.left.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(split.right.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(split.left.series_view("val").unwrap().to_vec::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(split.right.series_view("score").unwrap().to_vec::<i32>().unwrap(), vec![200, 300]);
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
    let override_stream = cudf::stream::CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let combined = left.concat_with_context(&right, &ctx).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(combined.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(combined.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(combined.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
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
    let original_extra_ptr = extra.view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let result = df.with_column_with_runtime("extra", extra, &target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(result.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_ne!(device_ptr(&result, "id"), original_id_ptr);
    assert_ne!(device_ptr(&result, "extra"), original_extra_ptr);
    assert_eq!(result.series_view("extra").unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
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
    let original_replacement_ptr = replacement.view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let result = df.assign_with_runtime([("val", replacement), ("bonus", bonus)], &target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(result.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_ne!(device_ptr(&result, "id"), original_id_ptr);
    assert_ne!(device_ptr(&result, "val"), original_replacement_ptr);
    assert_eq!(result.series_view("val").unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(result.series_view("bonus").unwrap().to_vec::<i32>().unwrap(), vec![70, 80, 90]);
}
