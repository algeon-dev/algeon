//! Public `Runtime v1` contract tests.
//!
//! This suite keeps representative coverage for runtime seeding, explicit
//! execution overrides, copy-into-runtime placement, and mixed-domain
//! semantics. It intentionally avoids one near-identical test per public API.

use cudf::{
    AggSpec, CsvReadOptions, CsvWriteOptions, DataFrame, DeviceId, ErrorKind, ExecutionContext, GroupByAgg,
    JsonWriteOptions, MergeOptions, ParquetReadOptions, ParquetWriteOptions, Runtime, Series, SortOptions, TypeId,
    core::Scalar, runtime::CudaStream,
};
use tempfile::TempDir;

use crate::common::{
    dataframe_device_ptr as device_ptr, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    reset_exec_observation, reset_sync_observation, sample_runtime, series_device_ptr,
};

fn assert_send_sync<T: Send + Sync>() {}

fn assert_series_residency(series: &Series, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(series.as_column().__debug_execution_domain_kind(), "explicit");
    assert_eq!(series.as_column().__debug_execution_domain_id(), expected_id);
    assert_eq!(series.as_column().__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(series.as_column().__debug_execution_domain_mr_value(), expected_mr);
}

fn assert_frame_residency(df: &DataFrame, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(df.__debug_execution_domain_kind(), "explicit");
    assert_eq!(df.__debug_execution_domain_id(), expected_id);
    assert_eq!(df.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(df.__debug_execution_domain_mr_value(), expected_mr);
}

fn sorted_i32_i32_rows(df: &DataFrame, left: &str, right: &str) -> Vec<(i32, i32)> {
    let left_values: Vec<i32> = df.series(left).unwrap().to_vec_strict::<i32>().unwrap();
    let right_values: Vec<i32> = df.series(right).unwrap().to_vec_strict::<i32>().unwrap();
    let mut rows: Vec<(i32, i32)> = left_values.into_iter().zip(right_values).collect();
    rows.sort_unstable();
    rows
}

fn sorted_i32_i64_rows(df: &DataFrame, left: &str, right: &str) -> Vec<(i32, i64)> {
    let left_values: Vec<i32> = df.series(left).unwrap().to_vec_strict::<i32>().unwrap();
    let right_values: Vec<i64> = df.series(right).unwrap().to_vec_strict::<i64>().unwrap();
    let mut rows: Vec<(i32, i64)> = left_values.into_iter().zip(right_values).collect();
    rows.sort_unstable();
    rows
}

#[test]
fn runtime_and_execution_context_are_public_and_send_sync() {
    assert_send_sync::<Runtime>();
    assert_send_sync::<ExecutionContext<'static>>();

    let rt = Runtime::builder().build().unwrap();
    let _ctx = ExecutionContext::new().with_stream(rt.stream()).with_memory_resource(rt.memory_resource());
}

#[test]
fn runtime_builder_and_factory_families_seed_runtime_exec_contracts() {
    let ambient_device = DeviceId::current().unwrap();
    let via_new = Runtime::new().unwrap();
    let via_builder = Runtime::builder().build().unwrap();
    assert_eq!(via_new.device(), ambient_device);
    assert_eq!(via_builder.device(), ambient_device);
    assert_ne!(via_new.__debug_execution_domain_id(), 0);
    assert_ne!(via_builder.__debug_execution_domain_id(), 0);

    let rt = sample_runtime();
    let expected_id = rt.__debug_execution_domain_id();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let expected_mr = rt.memory_resource().__debug_raw_handle_value();

    reset_exec_observation();
    let ints = rt.series("id", &[3i32, 1, 2]).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&ints, expected_id, expected_stream, expected_mr);
    assert_eq!(ints.to_vec_strict::<i32>().unwrap(), vec![3, 1, 2]);

    reset_exec_observation();
    let strings = rt.series("name", &["alice", "bob"]).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&strings, expected_id, expected_stream, expected_mr);
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);

    reset_exec_observation();
    let nulls = rt.nulls("all_null", TypeId::Int32, 3).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&nulls, expected_id, expected_stream, expected_mr);
    assert_eq!(nulls.null_count(), 3);

    reset_exec_observation();
    let seeded = rt.series_from_scalar("seed", &Scalar::from_value(7i32).unwrap(), 2).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&seeded, expected_id, expected_stream, expected_mr);
    assert_eq!(seeded.to_vec_strict::<i32>().unwrap(), vec![7, 7]);

    reset_exec_observation();
    let column = rt.column_from_slice(&[10i32, 20, 30]).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(column.__debug_execution_domain_id(), expected_id);
    assert_eq!(column.view().unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn dataframe_in_context_routes_representative_operation_families_through_context_exec() {
    let data_rt = sample_runtime();
    let other_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let df = DataFrame::from_series([
        data_rt.series("id", &[3i32, 1, 2, 2]).unwrap(),
        data_rt.series("val", &[30i32, 10, 20, 40]).unwrap(),
    ])
    .unwrap();
    let rhs = DataFrame::from_series([
        other_rt.series("id", &[2i32, 3, 4]).unwrap(),
        other_rt.series("score", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();
    let gather_map = other_rt.series("idx", &[2i32, 0]).unwrap();

    let ctx_stream = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    reset_exec_observation();
    let sorted = df.in_context(&ctx).sort_values(["id"], SortOptions::default()).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&sorted, df.__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 2, 3]);

    reset_exec_observation();
    let gathered = df.in_context(&ctx).gather(&gather_map).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&gathered, df.__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(gathered.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(2), Some(3)]);

    reset_sync_observation();
    reset_exec_observation();
    let merged = df.in_context(&ctx).merge(&rhs, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(last_sync_stream_value(), rhs.__debug_execution_domain_stream_value());
    assert_frame_residency(&merged, df.__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted_i32_i32_rows(&merged, "id", "score"), vec![(2, 200), (2, 200), (3, 300)]);

    reset_sync_observation();
    reset_exec_observation();
    let grouped = df.groupby(["id"]).in_context(&ctx).agg(&[AggSpec::new("val", GroupByAgg::Sum)]).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(last_sync_stream_value(), df.__debug_execution_domain_stream_value());
    assert_frame_residency(&grouped, df.__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted_i32_i64_rows(&grouped, "id", "val_sum"), vec![(1, 10), (2, 60), (3, 30)]);
}

#[test]
fn series_in_context_routes_representative_operation_families_through_context_exec() {
    let data_rt = sample_runtime();
    let other_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let numeric = data_rt.nullable_series("score", &[30i32, 0, 10, 0], &[true, false, true, false]).unwrap();
    let fill = other_rt.series("fill", &[9i32, 8, 7, 6]).unwrap();
    let rhs = other_rt.series("rhs", &[1i32, 2, 3, 4]).unwrap();
    let strings = data_rt.nullable_series("name", &["Alice", "bob", "", "CARA"], &[true, true, false, true]).unwrap();

    let ctx_stream = ctx_rt.stream().__debug_raw_handle_value();
    let ctx_mr = ctx_rt.memory_resource().__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    reset_exec_observation();
    let sorted = numeric.in_context(&ctx).sort_values(SortOptions::default()).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_series_residency(&sorted, numeric.as_column().__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted.to_optional_vec::<i32>().unwrap(), vec![Some(10), Some(30), None, None]);

    reset_exec_observation();
    let filled = numeric.in_context(&ctx).fillna_with_series(&fill).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_series_residency(&filled, numeric.as_column().__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![30, 8, 10, 6]);

    reset_sync_observation();
    reset_exec_observation();
    let added = numeric.in_context(&ctx).add(&rhs).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(last_sync_stream_value(), rhs.as_column().__debug_execution_domain_stream_value());
    assert_series_residency(&added, numeric.as_column().__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(added.to_optional_vec::<i32>().unwrap(), vec![Some(31), None, Some(13), None]);

    reset_exec_observation();
    let reduced = numeric.in_context(&ctx).sum().unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(reduced.__debug_execution_domain_id(), numeric.as_column().__debug_execution_domain_id());

    reset_exec_observation();
    reset_sync_observation();
    assert_eq!(numeric.in_context(&ctx).sum_value::<i64>().unwrap(), 40);
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(last_sync_stream_value(), ctx_stream);

    reset_exec_observation();
    let contains = strings.in_context(&ctx).str().contains("A").unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_series_residency(&contains, strings.as_column().__debug_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(contains.to_optional_vec::<bool>().unwrap(), vec![Some(true), Some(false), None, Some(true)]);
}

#[test]
fn in_context_rejects_ambient_default_promotion_for_dataframe_and_series() {
    let series = Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = DataFrame::from_series([series.clone()]).unwrap();
    let override_stream = CudaStream::new().unwrap();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    let err = series.in_context(&ctx).sort_values(SortOptions::default()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");

    let err = df.in_context(&ctx).sort_values(["id"], SortOptions::default()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");
}

#[test]
fn runtime_seeded_readers_use_runtime_stream() {
    let dir = TempDir::new().unwrap();

    let parquet_path = dir.path().join("runtime_seed.parquet");
    DataFrame::from_series([
        Series::new::<i32>("id", &[1, 2, 3]).unwrap(),
        Series::new::<i32>("value", &[10, 20, 30]).unwrap(),
    ])
    .unwrap()
    .write_parquet(&parquet_path)
    .unwrap();

    let csv_path = dir.path().join("runtime_seed.csv");
    std::fs::write(&csv_path, "id,value\n1,10\n2,20\n3,30\n").unwrap();

    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();

    reset_exec_observation();
    let parquet_df = DataFrame::read_parquet_with_options(
        ParquetReadOptions::builder().filepath(&parquet_path).with_runtime(&rt).build(),
    )
    .unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(parquet_df.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(parquet_df.column_names(), vec!["id", "value"]);

    reset_exec_observation();
    let csv_df =
        DataFrame::read_csv_with_options(CsvReadOptions::builder().filepath(&csv_path).with_runtime(&rt).build())
            .unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(csv_df.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(csv_df.column_names(), vec!["id", "value"]);
}

#[test]
fn write_options_and_runtime_overrides_use_expected_runtime() {
    let dir = TempDir::new().unwrap();
    let parquet_path = dir.path().join("runtime_write.parquet");
    let csv_path = dir.path().join("runtime_write.csv");
    let json_path = dir.path().join("runtime_write.jsonl");

    let data_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        data_rt.series("id", &[4i32, 5, 6]).unwrap(),
        data_rt.series("value", &[40i32, 50, 60]).unwrap(),
    ])
    .unwrap();

    reset_exec_observation();
    df.write_parquet_with_options(ParquetWriteOptions::builder().filepath(&parquet_path).build()).unwrap();
    assert_eq!(last_exec_stream_value(), data_rt.stream().__debug_raw_handle_value());

    reset_sync_observation();
    reset_exec_observation();
    df.write_csv_with_options(CsvWriteOptions::builder().filepath(&csv_path).with_runtime(&target_rt).build()).unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(last_exec_mr_value(), target_rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(last_sync_stream_value(), target_rt.stream().__debug_raw_handle_value());

    reset_sync_observation();
    reset_exec_observation();
    df.write_json_with_options(JsonWriteOptions::builder().filepath(&json_path).with_runtime(&target_rt).build())
        .unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(last_exec_mr_value(), target_rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(last_sync_stream_value(), target_rt.stream().__debug_raw_handle_value());
}

#[test]
fn copy_into_runtime_preserves_target_residency_and_reuses_allocation_when_possible() {
    let source_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        source_rt.series("id", &[1i32, 2, 3]).unwrap(),
        source_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let series = source_rt.series("id", &[1i32, 2, 3]).unwrap();

    let original_df_ptr = device_ptr(&df, "id");
    let original_series_ptr = series_device_ptr(&series);

    reset_exec_observation();
    let copied_df = df.copy_into_runtime(&target_rt).unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_frame_residency(
        &copied_df,
        target_rt.__debug_execution_domain_id(),
        target_rt.stream().__debug_raw_handle_value(),
        target_rt.memory_resource().__debug_raw_handle_value(),
    );
    assert_ne!(device_ptr(&copied_df, "id"), original_df_ptr);

    reset_exec_observation();
    let copied_series = series.copy_into_runtime(&target_rt).unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_series_residency(
        &copied_series,
        target_rt.__debug_execution_domain_id(),
        target_rt.stream().__debug_raw_handle_value(),
        target_rt.memory_resource().__debug_raw_handle_value(),
    );
    assert_ne!(series_device_ptr(&copied_series), original_series_ptr);

    reset_exec_observation();
    let reused = copied_series.copy_into_runtime(&target_rt).unwrap();
    assert_eq!(last_exec_stream_value(), usize::MAX);
    assert_eq!(series_device_ptr(&reused), series_device_ptr(&copied_series));
}

#[test]
fn in_context_override_and_copy_into_runtime_keep_distinct_mixed_domain_semantics() {
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

    let override_stream = CudaStream::new().unwrap();
    let ctx_stream = override_stream.__debug_raw_handle_value();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    reset_exec_observation();
    let merged_in_context = left.in_context(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(merged_in_context.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(merged_in_context.__debug_execution_domain_stream_value(), ctx_stream);

    let left = left.copy_into_runtime(&target_rt).unwrap();
    let right = right.copy_into_runtime(&target_rt).unwrap();
    reset_exec_observation();
    let merged_after_copy = left.merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().__debug_raw_handle_value());
    assert_frame_residency(
        &merged_after_copy,
        target_rt.__debug_execution_domain_id(),
        target_rt.stream().__debug_raw_handle_value(),
        target_rt.memory_resource().__debug_raw_handle_value(),
    );
    assert_eq!(merged_after_copy.series("score").unwrap().to_vec_strict::<i32>().unwrap(), vec![200, 300]);
}

#[test]
fn schema_only_assign_after_copy_into_runtime_preserves_target_residency() {
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

    let df = df.copy_into_runtime(&target_rt).unwrap();
    let replacement = replacement.copy_into_runtime(&target_rt).unwrap();
    let bonus = bonus.copy_into_runtime(&target_rt).unwrap();
    let id_ptr = device_ptr(&df, "id");
    let replacement_ptr = series_device_ptr(&replacement);
    let bonus_ptr = series_device_ptr(&bonus);

    reset_exec_observation();
    let result = df.assign([("val", replacement), ("bonus", bonus)]).unwrap();
    assert_eq!(last_exec_stream_value(), usize::MAX);
    assert_frame_residency(
        &result,
        target_rt.__debug_execution_domain_id(),
        target_rt.stream().__debug_raw_handle_value(),
        target_rt.memory_resource().__debug_raw_handle_value(),
    );
    assert_eq!(device_ptr(&result, "id"), id_ptr);
    assert_eq!(device_ptr(&result, "val"), replacement_ptr);
    assert_eq!(device_ptr(&result, "bonus"), bonus_ptr);
    assert_eq!(result.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
}
