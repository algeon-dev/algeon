//! Public `Runtime v1` contract tests.

use std::{ffi::c_void, ptr};

use cudf::{
    Agg, AggSpec, CsvReadOptions, CsvWriteOptions, DataFrame, ErrorKind, ExecutionContext, JsonWriteOptions,
    MergeOptions, ParquetReadOptions, ParquetWriteOptions, Runtime, Series, SortOptions,
    core::{Column, Scalar, TimestampResolution, TypeId},
    runtime::CudaStream,
};
use cudf_sys::{
    cudf_test_current_device_resource_raw, cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value,
    cudf_test_exec_reset_observation,
};
use tempfile::TempDir;

use crate::common::sample_runtime;

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

fn obs_last_sync_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_sync_stream_value failed");
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

fn series_device_ptr(series: &Series) -> *const c_void {
    series.as_column_view().unwrap().device_data_ptr().unwrap()
}

#[test]
fn runtime_and_execution_context_are_public_and_send_sync() {
    assert_send_sync::<Runtime>();
    assert_send_sync::<ExecutionContext<'static>>();

    let rt = Runtime::builder().build().unwrap();
    let _ctx = ExecutionContext::new().with_stream(rt.stream()).with_memory_resource(rt.memory_resource());
}

#[test]
fn runtime_first_series_factory_uses_runtime_domain_and_exec_options() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let default_mr = obs_current_device_resource() as usize;

    obs_reset();
    let series = rt.series("id", &[3i32, 1, 2]).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_ne!(obs_last_mr() as usize, default_mr);
    assert_eq!(series.as_column().__debug_execution_domain_kind(), "explicit");
    assert_eq!(series.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(series.as_column().__debug_execution_domain_mr_value(), rt.memory_resource().__debug_raw_handle_value());
}

#[test]
fn runtime_first_string_series_factory_uses_runtime_domain_and_exec_options() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let default_mr = obs_current_device_resource() as usize;

    obs_reset();
    let series = rt.string_series("name", &["alice", "bob"]).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_ne!(obs_last_mr() as usize, default_mr);
    assert_eq!(series.as_column().__debug_execution_domain_kind(), "explicit");
    assert_eq!(series.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(series.as_column().__debug_execution_domain_mr_value(), rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(series.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
}

#[test]
fn runtime_first_series_factories_seed_runtime_domain_and_values() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let expected_mr = rt.memory_resource().__debug_raw_handle_value();

    obs_reset();
    let ints = rt.series("id", &[3i32, 1, 2]).unwrap();
    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(ints.name(), Some("id"));
    assert_eq!(ints.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(ints.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(ints.as_column().__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(ints.to_vec_strict::<i32>().unwrap(), vec![3, 1, 2]);

    obs_reset();
    let nullable = rt.nullable_series("score", &[10i32, 0, 30], &[true, false, true]).unwrap();
    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(nullable.name(), Some("score"));
    assert_eq!(nullable.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(nullable.null_count(), 1);
    assert_eq!(nullable.to_vec_optional::<i32>().unwrap(), vec![Some(10), None, Some(30)]);

    obs_reset();
    let strings = rt.string_series("name", &["alice", "bob"]).unwrap();
    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(strings.name(), Some("name"));
    assert_eq!(strings.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);

    obs_reset();
    let nullable_strings =
        rt.nullable_string_series("nickname", &["", "ignored", "zed"], &[true, false, true]).unwrap();
    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(nullable_strings.name(), Some("nickname"));
    assert_eq!(nullable_strings.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(
        nullable_strings.to_strings_optional().unwrap(),
        vec![Some("".to_string()), None, Some("zed".to_string())]
    );
}

#[test]
fn runtime_null_and_empty_series_factories_seed_runtime_domain() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let expected_mr = rt.memory_resource().__debug_raw_handle_value();

    obs_reset();
    let nulls = rt.nulls("all_null", TypeId::Int32, 4).unwrap();
    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(nulls.name(), Some("all_null"));
    assert_eq!(nulls.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(nulls.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(nulls.as_column().__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(nulls.null_count(), 4);

    obs_reset();
    let empty = rt.empty_series("empty_ids", TypeId::Int32).unwrap();
    assert_eq!(empty.name(), Some("empty_ids"));
    assert_eq!(empty.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(empty.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(empty.as_column().__debug_execution_domain_mr_value(), expected_mr);
    assert!(empty.is_empty());
}

#[test]
fn runtime_series_from_scalar_uses_runtime_exec_options_and_provenance() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let expected_mr = rt.memory_resource().__debug_raw_handle_value();
    let scalar = Scalar::from_value(7i32).unwrap();

    obs_reset();
    let series = rt.series_from_scalar("seed", &scalar, 3).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(series.name(), Some("seed"));
    assert_eq!(series.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(series.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(series.as_column().__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(series.to_vec_strict::<i32>().unwrap(), vec![7, 7, 7]);
}

#[test]
fn column_from_scalar_with_runtime_uses_runtime_exec_options_and_provenance() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let expected_mr = rt.memory_resource().__debug_raw_handle_value();
    let scalar = Scalar::from_value(11i32).unwrap();

    obs_reset();
    let column = Column::from_scalar_with_runtime(&rt, &scalar, 3).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(column.__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(column.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(column.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(column.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 11, 11]);
}

#[test]
fn in_context_sort_values_on_runtime_seeded_dataframe_uses_override_stream_and_retains_identity() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([rt.series("id", &[3i32, 1, 2]).unwrap()]).unwrap();

    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let sorted = df.in_context(&ctx).sort_values(["id"], SortOptions::default()).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(sorted.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(sorted.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn in_context_sort_values_fences_input_stream_before_override_exec() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([rt.series("id", &[3i32, 1, 2]).unwrap()]).unwrap();
    let input_stream_value = df.__debug_execution_domain_stream_value();

    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let sorted = df.in_context(&ctx).sort_values(["id"], SortOptions::default()).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(obs_last_sync_stream(), input_stream_value);
    assert_eq!(sorted.__debug_execution_domain_stream_value(), override_stream_value);

    obs_reset();
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(obs_last_sync_stream(), override_stream_value);
}

/// [INV-AP-01] Ambient promotion rejection: borrowed context overrides cannot promote
/// ambient-default objects to explicit domains (Rule 5 — preserved in v2).
#[test]
fn in_context_sort_values_rejects_ambient_default_promotion() {
    let df = DataFrame::from_series([Series::new::<i32>("id", &[3, 1, 2]).unwrap()]).unwrap();
    let override_stream = CudaStream::new().unwrap();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    let err = df.in_context(&ctx).sort_values(["id"], SortOptions::default()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");
}

#[test]
fn in_context_merge_groupby_gather_and_filter_are_public_and_route_override_stream() {
    let rt = sample_runtime();
    let left =
        DataFrame::from_series([rt.series("id", &[1i32, 2, 3]).unwrap(), rt.series("val", &[10i64, 20, 30]).unwrap()])
            .unwrap();
    let right = DataFrame::from_series([
        rt.series("id", &[2i32, 3, 4]).unwrap(),
        rt.series("score", &[200i64, 300, 400]).unwrap(),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let joined = left.in_context(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(joined.num_rows(), 2);
    assert_eq!(joined.__debug_execution_domain_stream_value(), override_stream_value);

    let grouped = joined.groupby(["id"]).in_context(&ctx).agg(&[AggSpec::new("val", Agg::Sum)]).unwrap();
    assert_eq!(grouped.__debug_execution_domain_stream_value(), override_stream_value);

    let gather_indices = rt.series("idx", &[1i32, 0]).unwrap();
    obs_reset();
    let gathered = grouped.in_context(&ctx).gather(&gather_indices).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(gathered.__debug_execution_domain_stream_value(), override_stream_value);

    let mask = rt.series("mask", &[true, false]).unwrap();
    obs_reset();
    let filtered = gathered.in_context(&ctx).filter(&mask).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(filtered.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(filtered.num_rows(), 1);
}

#[test]
fn in_context_scopes_dataframe_ops_and_groupby_shortcuts() {
    let rt = sample_runtime();
    let left =
        DataFrame::from_series([rt.series("id", &[3i32, 1, 2]).unwrap(), rt.series("val", &[30i64, 10, 20]).unwrap()])
            .unwrap();
    let right = DataFrame::from_series([
        rt.series("id", &[2i32, 3, 4]).unwrap(),
        rt.series("score", &[200i64, 300, 400]).unwrap(),
    ])
    .unwrap();
    let indices = rt.series("idx", &[1i32, 0]).unwrap();
    let mask = rt.series("mask", &[true, false]).unwrap();

    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let sorted = left.in_context(&ctx).sort_values(["id"], SortOptions::default()).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(sorted.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(sorted.series("id").unwrap().to_vec_optional::<i32>().unwrap(), vec![Some(1), Some(2), Some(3)]);

    obs_reset();
    let joined = left.in_context(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(joined.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(joined.series("id").unwrap().to_vec_optional::<i32>().unwrap(), vec![Some(3), Some(2)]);

    obs_reset();
    let gathered = joined.in_context(&ctx).gather(&indices).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(gathered.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(gathered.series("id").unwrap().to_vec_optional::<i32>().unwrap(), vec![Some(2), Some(3)]);

    obs_reset();
    let filtered = gathered.in_context(&ctx).filter(&mask).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(filtered.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(filtered.series("id").unwrap().to_vec_optional::<i32>().unwrap(), vec![Some(2)]);

    obs_reset();
    let grouped = joined.groupby(["id"]).in_context(&ctx).sum(["val"]).unwrap();
    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(grouped.__debug_execution_domain_stream_value(), override_stream_value);
    let rows: Vec<_> = grouped
        .series("id")
        .unwrap()
        .to_vec_optional::<i32>()
        .unwrap()
        .into_iter()
        .zip(grouped.series("val_sum").unwrap().to_vec_optional::<i64>().unwrap())
        .collect();
    assert_eq!(rows, vec![(Some(3), Some(30)), (Some(2), Some(20))]);
}

#[test]
fn in_context_scopes_series_sort_and_fillna_ops() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let series = data_rt.nullable_series("score", &[30i32, 0, 10, 0], &[true, false, true, false]).unwrap();
    let fill = data_rt.series("fill", &[9i32, 8, 7, 6]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let sorted = series.in_context(&ctx).sort_values(SortOptions::default()).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(sorted.as_column().__debug_execution_domain_id(), series.as_column().__debug_execution_domain_id());
    assert_eq!(sorted.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(sorted.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(sorted.to_vec_optional::<i32>().unwrap(), vec![Some(10), Some(30), None, None]);

    obs_reset();
    let filled = series.in_context(&ctx).fillna(99i32).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(filled.as_column().__debug_execution_domain_id(), series.as_column().__debug_execution_domain_id());
    assert_eq!(filled.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(filled.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![30, 99, 10, 99]);

    obs_reset();
    let filled_from_series = series.in_context(&ctx).fillna_with_series(&fill).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(
        filled_from_series.as_column().__debug_execution_domain_id(),
        series.as_column().__debug_execution_domain_id()
    );
    assert_eq!(filled_from_series.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(filled_from_series.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(filled_from_series.to_vec_strict::<i32>().unwrap(), vec![30, 8, 10, 6]);
}

#[test]
fn in_context_scopes_series_reduction_and_host_sync() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let series = data_rt.series("score", &[10i32, 20, 30]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let reduced = series.in_context(&ctx).sum().unwrap();

    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(reduced.__debug_execution_domain_id(), series.as_column().__debug_execution_domain_id());
    assert_eq!(reduced.__debug_execution_domain_stream_value(), ctx_stream_value);

    obs_reset();
    let total = series.in_context(&ctx).sum_value::<i64>().unwrap();

    assert_eq!(total, 60);
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);
}

#[test]
fn in_context_scopes_series_reduction_family() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let ints = data_rt.series("score", &[30i32, 10, 20]).unwrap();
    let floats = data_rt.series("float_score", &[2.0f64, 4.0, 4.0, 8.0]).unwrap();
    let bools = data_rt.series("flag", &[true, false, true]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let min = ints.in_context(&ctx).min().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(min.__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(min.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(min.value::<i32>().unwrap(), 10);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let max = ints.in_context(&ctx).max().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(max.__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(max.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(max.value::<i32>().unwrap(), 30);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let mean = ints.in_context(&ctx).mean().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(mean.__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(mean.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(mean.value::<f64>().unwrap(), 20.0);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let median = ints.in_context(&ctx).median().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(median.__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(median.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(median.value::<f64>().unwrap(), 20.0);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let std = floats.in_context(&ctx).std(0).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(std.__debug_execution_domain_id(), floats.as_column().__debug_execution_domain_id());
    assert_eq!(std.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert!((std.value::<f64>().unwrap() - 2.179449471770337).abs() < 1e-12);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let var = floats.in_context(&ctx).var(0).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(var.__debug_execution_domain_id(), floats.as_column().__debug_execution_domain_id());
    assert_eq!(var.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert!((var.value::<f64>().unwrap() - 4.75).abs() < 1e-12);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let any = bools.in_context(&ctx).any().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(any.__debug_execution_domain_id(), bools.as_column().__debug_execution_domain_id());
    assert_eq!(any.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert!(any.value::<bool>().unwrap());
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let all = bools.in_context(&ctx).all().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(all.__debug_execution_domain_id(), bools.as_column().__debug_execution_domain_id());
    assert_eq!(all.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert!(!all.value::<bool>().unwrap());
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let product = ints.in_context(&ctx).product().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(product.__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(product.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(product.value::<i32>().unwrap(), 6000);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    let (minmax_min, minmax_max) = ints.in_context(&ctx).minmax().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(minmax_min.__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(minmax_max.__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(minmax_min.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(minmax_max.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(minmax_min.value::<i32>().unwrap(), 10);
    assert_eq!(minmax_max.value::<i32>().unwrap(), 30);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);
}

#[test]
fn in_context_scopes_series_host_value_helpers() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let ints = data_rt.series("score", &[30i32, 10, 20]).unwrap();
    let floats = data_rt.series("float_score", &[2.0f64, 4.0, 4.0, 8.0]).unwrap();
    let bools = data_rt.series("flag", &[true, false, true]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    assert_eq!(ints.in_context(&ctx).min_value::<i32>().unwrap(), Some(10));
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert_eq!(ints.in_context(&ctx).max_value::<i32>().unwrap(), Some(30));
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert_eq!(ints.in_context(&ctx).mean_value().unwrap(), Some(20.0));
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert_eq!(ints.in_context(&ctx).median_value().unwrap(), Some(20.0));
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert!((floats.in_context(&ctx).std_value(0).unwrap().unwrap() - 2.179449471770337).abs() < 1e-12);
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert!((floats.in_context(&ctx).var_value(0).unwrap().unwrap() - 4.75).abs() < 1e-12);
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert!(bools.in_context(&ctx).any_value().unwrap());
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert!(!bools.in_context(&ctx).all_value().unwrap());
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert_eq!(ints.in_context(&ctx).product_value::<i32>().unwrap(), Some(6000));
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);

    obs_reset();
    assert_eq!(ints.in_context(&ctx).minmax_values::<i32>().unwrap(), (Some(10), Some(30)));
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);
}

#[test]
fn in_context_scopes_series_scan_and_unary_ops() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let ints = data_rt.series("score", &[3i32, 1, 2]).unwrap();
    let signed = data_rt.series("signed", &[-3i32, -1, 2]).unwrap();
    let floats = data_rt.series("float_score", &[1.0f64, f64::NAN, 3.0]).unwrap();
    let bools = data_rt.series("flag", &[true, false, true]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let cumsum = ints.in_context(&ctx).cumsum().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(cumsum.as_column().__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(cumsum.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(cumsum.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(cumsum.to_vec_strict::<i32>().unwrap(), vec![3, 4, 6]);

    obs_reset();
    let cummin = ints.in_context(&ctx).cummin().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(cummin.as_column().__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(cummin.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(cummin.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(cummin.to_vec_strict::<i32>().unwrap(), vec![3, 1, 1]);

    obs_reset();
    let cummax = ints.in_context(&ctx).cummax().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(cummax.as_column().__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(cummax.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(cummax.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(cummax.to_vec_strict::<i32>().unwrap(), vec![3, 3, 3]);

    obs_reset();
    let cumprod = ints.in_context(&ctx).cumprod().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(cumprod.as_column().__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(cumprod.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(cumprod.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(cumprod.to_vec_strict::<i32>().unwrap(), vec![3, 3, 6]);

    obs_reset();
    let is_nan = floats.in_context(&ctx).is_nan().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(is_nan.as_column().__debug_execution_domain_id(), floats.as_column().__debug_execution_domain_id());
    assert_eq!(is_nan.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(is_nan.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(is_nan.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);

    obs_reset();
    let is_not_nan = floats.in_context(&ctx).is_not_nan().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(is_not_nan.as_column().__debug_execution_domain_id(), floats.as_column().__debug_execution_domain_id());
    assert_eq!(is_not_nan.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(is_not_nan.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(is_not_nan.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);

    obs_reset();
    let abs = signed.in_context(&ctx).abs().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(abs.as_column().__debug_execution_domain_id(), signed.as_column().__debug_execution_domain_id());
    assert_eq!(abs.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(abs.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(abs.to_vec_strict::<i32>().unwrap(), vec![3, 1, 2]);

    obs_reset();
    let neg = signed.in_context(&ctx).neg().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(neg.as_column().__debug_execution_domain_id(), signed.as_column().__debug_execution_domain_id());
    assert_eq!(neg.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(neg.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(neg.to_vec_strict::<i32>().unwrap(), vec![3, 1, -2]);

    obs_reset();
    let logical_not = bools.in_context(&ctx).logical_not().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(logical_not.as_column().__debug_execution_domain_id(), bools.as_column().__debug_execution_domain_id());
    assert_eq!(logical_not.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(logical_not.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(logical_not.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn in_context_scopes_series_cast_binary_and_set_like_ops() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let ctx_rt = sample_runtime();

    let left = left_rt.nullable_series("left", &[30i32, 0, 10, 30], &[true, false, true, true]).unwrap();
    let right = right_rt.series("right", &[1i32, 2, 3, 4]).unwrap();
    let members = right_rt.series("members", &[10i32, 30]).unwrap();
    let labels = left_rt.series("labels", &[30i32, 10, 30, 20]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let null_mask = left.in_context(&ctx).is_null().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(null_mask.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(null_mask.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(null_mask.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(null_mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false, false]);

    obs_reset();
    let valid_mask = left.in_context(&ctx).is_not_null().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(valid_mask.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(valid_mask.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(valid_mask.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(valid_mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true, true]);

    obs_reset();
    let casted = left.in_context(&ctx).astype(TypeId::Float64).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(casted.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(casted.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(casted.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(casted.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(casted.to_vec_optional::<f64>().unwrap(), vec![Some(30.0), None, Some(10.0), Some(30.0)]);

    obs_reset();
    let datetimes = labels.in_context(&ctx).to_datetime(Some(TimestampResolution::Milliseconds)).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(datetimes.as_column().__debug_execution_domain_id(), labels.as_column().__debug_execution_domain_id());
    assert_eq!(datetimes.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(datetimes.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(datetimes.dtype().unwrap().id(), TypeId::TimestampMilliseconds);

    obs_reset();
    let added = left.in_context(&ctx).add(&right).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), right.as_column().__debug_execution_domain_stream_value());
    assert_eq!(added.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(added.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(added.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(added.to_vec_optional::<i32>().unwrap(), vec![Some(31), None, Some(13), Some(34)]);

    obs_reset();
    let equals = left.in_context(&ctx).eq(&right).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), right.as_column().__debug_execution_domain_stream_value());
    assert_eq!(equals.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(equals.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(equals.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(equals.to_vec_optional::<bool>().unwrap(), vec![Some(false), None, Some(false), Some(false)]);

    obs_reset();
    let isin = left.in_context(&ctx).isin(&members).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(obs_last_sync_stream(), members.as_column().__debug_execution_domain_stream_value());
    assert_eq!(isin.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(isin.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(isin.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(isin.to_vec_optional::<bool>().unwrap(), vec![Some(true), None, Some(true), Some(true)]);

    obs_reset();
    let unique = left.in_context(&ctx).unique().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(unique.as_column().__debug_execution_domain_id(), left.as_column().__debug_execution_domain_id());
    assert_eq!(unique.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(unique.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    let mut unique_values = unique.to_vec_optional::<i32>().unwrap();
    unique_values.sort_by_key(|value| value.unwrap_or(i32::MIN));
    assert_eq!(unique_values, vec![None, Some(10), Some(30)]);

    obs_reset();
    assert_eq!(left.in_context(&ctx).nunique().unwrap(), 2);
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);

    obs_reset();
    let (codes, uniques) = labels.in_context(&ctx).factorize().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(codes.name(), Some("labels_codes"));
    assert_eq!(uniques.name(), Some("labels_uniques"));
    assert_eq!(codes.as_column().__debug_execution_domain_id(), labels.as_column().__debug_execution_domain_id());
    assert_eq!(codes.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(codes.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(uniques.as_column().__debug_execution_domain_id(), labels.as_column().__debug_execution_domain_id());
    assert_eq!(uniques.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(uniques.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(codes.to_vec_strict::<i32>().unwrap(), vec![2, 0, 2, 1]);
    assert_eq!(uniques.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn in_context_scopes_series_string_accessors() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let strings =
        data_rt.nullable_string_series("name", &["  Alice", "bob  ", "", "CARA"], &[true, true, false, true]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let lower = strings.in_context(&ctx).str().lower().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(lower.as_column().__debug_execution_domain_id(), strings.as_column().__debug_execution_domain_id());
    assert_eq!(lower.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(lower.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(
        lower.to_strings_optional().unwrap(),
        vec![Some("  alice".to_string()), Some("bob  ".to_string()), None, Some("cara".to_string())]
    );

    obs_reset();
    let contains = strings.in_context(&ctx).str().contains("A").unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(contains.as_column().__debug_execution_domain_id(), strings.as_column().__debug_execution_domain_id());
    assert_eq!(contains.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(contains.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(contains.to_vec_optional::<bool>().unwrap(), vec![Some(true), Some(false), None, Some(true)]);

    obs_reset();
    let stripped = strings.in_context(&ctx).str().strip().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(stripped.as_column().__debug_execution_domain_id(), strings.as_column().__debug_execution_domain_id());
    assert_eq!(stripped.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(stripped.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(
        stripped.to_strings_optional().unwrap(),
        vec![Some("Alice".to_string()), Some("bob".to_string()), None, Some("CARA".to_string())]
    );

    obs_reset();
    let lengths = strings.in_context(&ctx).str().len().unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(lengths.as_column().__debug_execution_domain_id(), strings.as_column().__debug_execution_domain_id());
    assert_eq!(lengths.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(lengths.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(lengths.to_vec_optional::<i32>().unwrap(), vec![Some(7), Some(5), None, Some(4)]);
}

#[test]
fn in_context_scopes_series_scalar_arithmetic_and_comparison_helpers() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let ints = data_rt.nullable_series("score", &[10i32, 0, 30, 40], &[true, false, true, true]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let added = ints.in_context(&ctx).add_scalar(5i32).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(added.name(), Some("score"));
    assert_eq!(added.as_column().__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(added.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(added.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(added.to_vec_optional::<i32>().unwrap(), vec![Some(15), None, Some(35), Some(45)]);

    obs_reset();
    let matches = ints.in_context(&ctx).ge_scalar(30i32).unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(matches.name(), Some("score"));
    assert_eq!(matches.as_column().__debug_execution_domain_id(), ints.as_column().__debug_execution_domain_id());
    assert_eq!(matches.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(matches.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(matches.to_vec_optional::<bool>().unwrap(), vec![Some(false), None, Some(true), Some(true)]);
}

#[test]
fn in_context_scopes_series_string_scalar_comparison_helpers() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let strings =
        data_rt.nullable_string_series("name", &["alice", "bob", "", "carol"], &[true, true, false, true]).unwrap();

    let ctx_stream_value = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr_value = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    obs_reset();
    let equals = strings.in_context(&ctx).eq_scalar_str("bob").unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(equals.name(), Some("name"));
    assert_eq!(equals.as_column().__debug_execution_domain_id(), strings.as_column().__debug_execution_domain_id());
    assert_eq!(equals.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(equals.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(equals.to_vec_optional::<bool>().unwrap(), vec![Some(false), Some(true), None, Some(false)]);

    obs_reset();
    let greater = strings.in_context(&ctx).gt_scalar_str("avocado").unwrap();
    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, ctx_mr_value);
    assert_eq!(greater.name(), Some("name"));
    assert_eq!(greater.as_column().__debug_execution_domain_id(), strings.as_column().__debug_execution_domain_id());
    assert_eq!(greater.as_column().__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(greater.as_column().__debug_execution_domain_mr_value(), ctx_mr_value);
    assert_eq!(greater.to_vec_optional::<bool>().unwrap(), vec![Some(false), Some(true), None, Some(true)]);
}

#[test]
fn series_in_context_rejects_ambient_default_promotion() {
    let series = Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let override_stream = CudaStream::new().unwrap();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    let err = series.in_context(&ctx).sort_values(SortOptions::default()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");
}

#[test]
fn groupby_in_context_fences_input_stream_before_override_exec() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([
        rt.series("id", &[1i32, 1, 2, 2]).unwrap(),
        rt.series("val", &[10i64, 20, 30, 40]).unwrap(),
    ])
    .unwrap();
    let input_stream_value = df.__debug_execution_domain_stream_value();

    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let grouped = df.groupby(["id"]).in_context(&ctx).agg(&[AggSpec::new("val", Agg::Sum)]).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(obs_last_sync_stream(), input_stream_value);
    assert_eq!(grouped.__debug_execution_domain_stream_value(), override_stream_value);

    obs_reset();
    let rows: Vec<_> = grouped
        .series("id")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(grouped.series("val_sum").unwrap().to_vec_strict::<i64>().unwrap())
        .collect();
    assert_eq!(rows, vec![(1, 30), (2, 70)]);
    assert_eq!(obs_last_sync_stream(), override_stream_value);
}

#[test]
fn parquet_read_with_runtime_seeds_explicit_domain_and_uses_runtime_stream() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_seed.parquet");
    let df = DataFrame::from_series([
        Series::new::<i32>("id", &[1, 2, 3]).unwrap(),
        Series::new::<i32>("value", &[10, 20, 30]).unwrap(),
    ])
    .unwrap();
    df.write_parquet(&path).unwrap();

    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let opts = ParquetReadOptions::builder().filepath(&path).with_runtime(&rt).build();

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
    let opts = CsvReadOptions::builder().filepath(&path).with_runtime(&rt).build();

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
        rt.series("id", &[4i32, 5, 6]).unwrap(),
        rt.series("value", &[40i32, 50, 60]).unwrap(),
    ])
    .unwrap();

    obs_reset();
    df.write_parquet_with_options(ParquetWriteOptions::builder().filepath(&path).build()).unwrap();

    assert_eq!(obs_last_stream(), rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_parquet(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn dataframe_write_parquet_with_options_same_device_runtime_override_uses_target_runtime() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write_override.parquet");
    let data_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        data_rt.series("id", &[4i32, 5, 6]).unwrap(),
        data_rt.series("value", &[40i32, 50, 60]).unwrap(),
    ])
    .unwrap();

    obs_reset();
    df.write_parquet_with_options(ParquetWriteOptions::builder().filepath(&path).with_runtime(&target_rt).build())
        .unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().__debug_raw_handle_value());

    let mut sync_out: usize = 0;
    let status = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut sync_out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read sync stream observation");
    assert_eq!(sync_out, target_rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_parquet(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn dataframe_write_csv_with_options_uses_retained_runtime_stream() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write.csv");
    let rt = sample_runtime();
    let df = DataFrame::from_series([
        rt.series("id", &[4i32, 5, 6]).unwrap(),
        rt.series("value", &[40i32, 50, 60]).unwrap(),
    ])
    .unwrap();

    obs_reset();
    df.write_csv_with_options(CsvWriteOptions::builder().filepath(&path).build()).unwrap();

    assert_eq!(obs_last_stream(), rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_csv(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec_optional::<i64>().unwrap(), vec![Some(4), Some(5), Some(6)]);
}

#[test]
fn dataframe_write_csv_with_options_same_device_runtime_override_uses_target_runtime() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write_override.csv");
    let data_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        data_rt.series("id", &[4i32, 5, 6]).unwrap(),
        data_rt.series("value", &[40i32, 50, 60]).unwrap(),
    ])
    .unwrap();

    obs_reset();
    df.write_csv_with_options(CsvWriteOptions::builder().filepath(&path).with_runtime(&target_rt).build()).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().__debug_raw_handle_value());

    let mut sync_out: usize = 0;
    let status = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut sync_out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read sync stream observation");
    assert_eq!(sync_out, target_rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_csv(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec_optional::<i64>().unwrap(), vec![Some(4), Some(5), Some(6)]);
}

#[test]
fn dataframe_write_json_with_options_same_device_runtime_override_uses_target_runtime() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("runtime_write_override.jsonl");
    let data_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        data_rt.series("id", &[4i32, 5, 6]).unwrap(),
        data_rt.series("value", &[40i32, 50, 60]).unwrap(),
    ])
    .unwrap();

    obs_reset();
    df.write_json_with_options(JsonWriteOptions::builder().filepath(&path).with_runtime(&target_rt).build()).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(obs_last_sync_stream(), target_rt.stream().__debug_raw_handle_value());

    let loaded = DataFrame::read_json(&path).unwrap();
    assert_eq!(loaded.column_names(), vec!["id", "value"]);
    assert_eq!(loaded.series("id").unwrap().to_vec_strict::<i64>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn dataframe_copy_into_runtime_places_columns_in_target_runtime() {
    let source_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        source_rt.series("id", &[1i32, 2, 3]).unwrap(),
        source_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let original_id_ptr = device_ptr(&df, "id");

    obs_reset();
    let copied = df.copy_into_runtime(&target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(copied.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(copied.__debug_execution_domain_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(copied.__debug_execution_domain_mr_value(), target_rt.memory_resource().__debug_raw_handle_value());
    assert_ne!(device_ptr(&copied, "id"), original_id_ptr);
    assert_eq!(copied.column_names(), vec!["id", "val"]);
    assert_eq!(copied.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn series_copy_into_runtime_places_column_in_target_runtime() {
    let source_rt = sample_runtime();
    let target_rt = sample_runtime();
    let series = source_rt.series("id", &[1i32, 2, 3]).unwrap();
    let original_ptr = series_device_ptr(&series);

    obs_reset();
    let copied = series.copy_into_runtime(&target_rt).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(copied.name(), Some("id"));
    assert_eq!(copied.as_column().__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(
        copied.as_column().__debug_execution_domain_stream_value(),
        target_rt.stream().__debug_raw_handle_value()
    );
    assert_eq!(
        copied.as_column().__debug_execution_domain_mr_value(),
        target_rt.memory_resource().__debug_raw_handle_value()
    );
    assert_ne!(series_device_ptr(&copied), original_ptr);
    assert_eq!(copied.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn series_copy_into_runtime_reuses_gpu_allocation_when_already_in_target_runtime() {
    let rt = sample_runtime();
    let series = rt.series("id", &[7i32, 8, 9]).unwrap();
    let original_ptr = series_device_ptr(&series);

    obs_reset();
    let copied = series.copy_into_runtime(&rt).unwrap();

    assert_eq!(obs_last_stream(), usize::MAX);
    assert_eq!(copied.name(), Some("id"));
    assert_eq!(copied.as_column().__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(series_device_ptr(&copied), original_ptr);
    assert_eq!(copied.to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn merge_after_copy_into_runtime_recovers_mixed_explicit_domains_into_target_runtime() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2, 3]).unwrap(),
        left_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[2i32, 3, 4]).unwrap(),
        right_rt.series("score", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();

    let left = left.copy_into_runtime(&target_rt).unwrap();
    let right = right.copy_into_runtime(&target_rt).unwrap();

    obs_reset();
    let joined = left.merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(joined.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(joined.__debug_execution_domain_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(joined.column_names(), vec!["id", "val", "score"]);
    assert_eq!(joined.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![2, 3]);
}

#[test]
fn merge_in_context_uses_left_lineage_for_mixed_explicit_domains() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2, 3]).unwrap(),
        left_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[2i32, 3, 4]).unwrap(),
        right_rt.series("score", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let joined = left.in_context(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(joined.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(joined.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(joined.column_names(), vec!["id", "val", "score"]);
    assert_eq!(joined.series("score").unwrap().to_vec_strict::<i32>().unwrap(), vec![200, 300]);
}

#[test]
fn concat_after_copy_into_runtime_recovers_mixed_explicit_domains_into_target_runtime() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2]).unwrap(),
        left_rt.series("val", &[10i32, 20]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[3i32, 4]).unwrap(),
        right_rt.series("val", &[30i32, 40]).unwrap(),
    ])
    .unwrap();

    let left = left.copy_into_runtime(&target_rt).unwrap();
    let right = right.copy_into_runtime(&target_rt).unwrap();

    obs_reset();
    let combined = left.concat(&right).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(combined.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(combined.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn merge_split_after_copy_into_runtime_recovers_mixed_explicit_domains_into_target_runtime() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2, 3]).unwrap(),
        left_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[2i32, 3, 4]).unwrap(),
        right_rt.series("score", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();

    let left = left.copy_into_runtime(&target_rt).unwrap();
    let right = right.copy_into_runtime(&target_rt).unwrap();

    obs_reset();
    let split = left.merge_split(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(split.left.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(split.right.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(split.left.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(split.right.series("score").unwrap().to_vec_strict::<i32>().unwrap(), vec![200, 300]);
}

#[test]
fn merge_split_in_context_uses_left_lineage_for_mixed_explicit_domains() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2, 3]).unwrap(),
        left_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[2i32, 3, 4]).unwrap(),
        right_rt.series("score", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let split = left.in_context(&ctx).merge_split(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(split.left.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.right.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.left.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(split.right.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(split.left.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(split.right.series("score").unwrap().to_vec_strict::<i32>().unwrap(), vec![200, 300]);
}

#[test]
fn concat_in_context_uses_left_lineage_for_mixed_explicit_domains() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2]).unwrap(),
        left_rt.series("val", &[10i32, 20]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[3i32, 4]).unwrap(),
        right_rt.series("val", &[30i32, 40]).unwrap(),
    ])
    .unwrap();
    let override_stream = CudaStream::new().unwrap();
    let override_stream_value = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let combined = left.in_context(&ctx).concat(&right).unwrap();

    assert_eq!(obs_last_stream(), override_stream_value);
    assert_eq!(combined.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(combined.__debug_execution_domain_stream_value(), override_stream_value);
    assert_eq!(combined.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn with_column_after_copy_into_runtime_remains_schema_only_while_preserving_target_runtime_residency() {
    let base_rt = sample_runtime();
    let extra_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        base_rt.series("id", &[1i32, 2, 3]).unwrap(),
        base_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let extra = extra_rt.series("extra", &[100i32, 200, 300]).unwrap();
    let original_id_ptr = device_ptr(&df, "id");
    let original_extra_ptr = extra.as_column_view().unwrap().device_data_ptr().unwrap();

    let df = df.copy_into_runtime(&target_rt).unwrap();
    let extra = extra.copy_into_runtime(&target_rt).unwrap();
    let copied_id_ptr = device_ptr(&df, "id");
    let copied_extra_ptr = extra.as_column_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let result = df.with_column("extra", extra).unwrap();

    assert_eq!(obs_last_stream(), usize::MAX);
    assert_eq!(result.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(device_ptr(&result, "id"), copied_id_ptr);
    assert_ne!(device_ptr(&result, "id"), original_id_ptr);
    assert_eq!(device_ptr(&result, "extra"), copied_extra_ptr);
    assert_ne!(device_ptr(&result, "extra"), original_extra_ptr);
    assert_eq!(result.series("extra").unwrap().to_vec_strict::<i32>().unwrap(), vec![100, 200, 300]);
}

#[test]
fn assign_after_copy_into_runtime_remains_schema_only_while_preserving_target_runtime_residency() {
    let base_rt = sample_runtime();
    let extra_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        base_rt.series("id", &[1i32, 2, 3]).unwrap(),
        base_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let replacement = extra_rt.series("replacement", &[7i32, 8, 9]).unwrap();
    let bonus = extra_rt.series("bonus", &[70i32, 80, 90]).unwrap();
    let original_id_ptr = device_ptr(&df, "id");
    let original_replacement_ptr = replacement.as_column_view().unwrap().device_data_ptr().unwrap();

    let df = df.copy_into_runtime(&target_rt).unwrap();
    let replacement = replacement.copy_into_runtime(&target_rt).unwrap();
    let bonus = bonus.copy_into_runtime(&target_rt).unwrap();
    let copied_id_ptr = device_ptr(&df, "id");
    let copied_replacement_ptr = replacement.as_column_view().unwrap().device_data_ptr().unwrap();
    let copied_bonus_ptr = bonus.as_column_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let result = df.assign([("val", replacement), ("bonus", bonus)]).unwrap();

    assert_eq!(obs_last_stream(), usize::MAX);
    assert_eq!(result.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(device_ptr(&result, "id"), copied_id_ptr);
    assert_ne!(device_ptr(&result, "id"), original_id_ptr);
    assert_eq!(device_ptr(&result, "val"), copied_replacement_ptr);
    assert_ne!(device_ptr(&result, "val"), original_replacement_ptr);
    assert_eq!(device_ptr(&result, "bonus"), copied_bonus_ptr);
    assert_eq!(result.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(result.series("bonus").unwrap().to_vec_strict::<i32>().unwrap(), vec![70, 80, 90]);
}

// === v2: in_context(...) remains execution override, not explicit placement ===
//
// These tests cover mixed-domain inputs where the operation still runs under a
// borrowed execution override. Output provenance remains derived from the
// primary input identity; placement is still an explicit `copy_into_runtime`
// step, not an `in_context(...)` side effect.
//
// INV-PR-03: output provenance reflects context overrides.
// INV-SY-03: sync targets the context's stream after mixed-domain ops.

#[test]
fn in_context_sort_values_succeeds_on_mixed_domain_inputs() {
    // Two-column DataFrame from domain A, sort with context derived from domain B.
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df = DataFrame::from_series([
        rt_a.series("id", &[3i32, 1, 2]).unwrap(),
        rt_a.series("val", &[30i32, 10, 20]).unwrap(),
    ])
    .unwrap();

    // Context borrows stream from rt_b — different domain than df
    let ctx = ExecutionContext::new().with_stream(rt_b.stream()).with_memory_resource(rt_b.memory_resource());
    let ctx_stream_value = rt_b.stream().__debug_raw_handle_value();
    let ctx_mr_value = rt_b.memory_resource().__debug_raw_handle_value();
    assert_ne!(ctx_stream_value, rt_a.stream().__debug_raw_handle_value());

    obs_reset();
    let sorted = df.in_context(&ctx).sort_values(["id"], SortOptions::default()).unwrap();

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
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn in_context_merge_succeeds_on_mixed_domain_inputs() {
    // DataFrames from different domains, merge with explicit context.
    let rt_left = sample_runtime();
    let rt_right = sample_runtime();
    let left = DataFrame::from_series([
        rt_left.series("id", &[1i32, 2, 3]).unwrap(),
        rt_left.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        rt_right.series("id", &[2i32, 3, 4]).unwrap(),
        rt_right.series("score", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();

    // Context uses left's stream — but the point is mixed inputs succeed
    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let left_mr_value = left.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let joined = left.in_context(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();

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
    assert_eq!(joined.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![2, 3]);
}

#[test]
fn in_context_gather_succeeds_on_mixed_domain_inputs() {
    // Data from domain A, gather map from domain B, context from domain C.
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df = DataFrame::from_series([rt_a.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let indices = rt_b.series("idx", &[2i32, 0]).unwrap();

    // Context uses a third stream
    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let df_mr_value = df.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let gathered = df.in_context(&ctx).gather(&indices).unwrap();

    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, df_mr_value);
    assert_eq!(gathered.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(gathered.__debug_execution_domain_mr_value(), df_mr_value);
    assert_eq!(gathered.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(gathered.series("id").unwrap().to_vec_optional::<i32>().unwrap(), vec![Some(30), Some(10)]);
}

#[test]
fn in_context_filter_succeeds_on_mixed_domain_inputs() {
    // Data from domain A, mask from domain B, context from domain C.
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df = DataFrame::from_series([rt_a.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let mask = rt_b.series("mask", &[true, false, true]).unwrap();

    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let df_mr_value = df.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    obs_reset();
    let filtered = df.in_context(&ctx).filter(&mask).unwrap();

    assert_eq!(obs_last_stream(), ctx_stream_value);
    assert_eq!(obs_last_mr() as usize, df_mr_value);
    assert_eq!(filtered.__debug_execution_domain_stream_value(), ctx_stream_value);
    assert_eq!(filtered.__debug_execution_domain_mr_value(), df_mr_value);
    assert_eq!(filtered.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 30]);
}

#[test]
fn mixed_domain_in_context_output_sync_targets_context_stream() {
    // After mixed-domain op with explicit context, synchronize_for_host_visibility
    // must target the context's stream (which is now the output's provenance stream).
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let df = DataFrame::from_series([rt_a.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let indices = rt_b.series("idx", &[2i32, 0]).unwrap();

    let override_stream = CudaStream::new().unwrap();
    let ctx_stream_value = override_stream.__debug_raw_handle_value();
    let df_mr_value = df.__debug_execution_domain_mr_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);
    assert_ne!(df.__debug_execution_domain_stream_value(), indices.as_column().__debug_execution_domain_stream_value());
    assert_ne!(ctx_stream_value, df.__debug_execution_domain_stream_value());
    assert_ne!(ctx_stream_value, indices.as_column().__debug_execution_domain_stream_value());

    obs_reset();
    let gathered = df.in_context(&ctx).gather(&indices).unwrap();

    // Output provenance stream = context stream
    assert_eq!(gathered.__debug_execution_domain_stream_value(), ctx_stream_value);
    // Stream-only override keeps the primary input MR.
    assert_eq!(obs_last_mr() as usize, df_mr_value);
    assert_eq!(gathered.__debug_execution_domain_mr_value(), df_mr_value);

    // Host materialization must synchronize on the output provenance stream.
    obs_reset();
    let host = gathered.series("id").unwrap().to_vec_optional::<i32>().unwrap();
    assert_eq!(host, vec![Some(30), Some(10)]);
    assert_eq!(obs_last_sync_stream(), ctx_stream_value);
}
