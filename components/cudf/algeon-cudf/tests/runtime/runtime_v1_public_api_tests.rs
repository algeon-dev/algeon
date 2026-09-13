//! Public `Runtime v1` contract tests.
//!
//! This suite keeps representative coverage for runtime seeding, explicit
//! execution overrides, copy-into-runtime placement, and mixed-domain
//! semantics. It intentionally avoids one near-identical test per public API.
use cudf::{
    AggSpec, DataFrame, DataType, DeviceId, ErrorCode, ExecutionBudgetEnforcement, ExecutionBudgetRequest,
    ExecutionOverrides, GroupByAgg, MergeOptions, Runtime, Series, SortOptions, TypeId,
    core::{Scalar, memory::DeviceMemoryResource},
    frame::dataframe::{JsonWriteOptions, ParquetReadOptions, ParquetWriteOptions},
    ops::expr::{col, when},
    runtime::CudaStream,
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{
    dataframe_device_ptr, dependency_wait_call_count_for_stream, error_context_value, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, lock_event_observation, observed_exec_mr_overflowed,
    observed_exec_mr_values, reset_dependency_observation, reset_exec_observation, reset_sync_observation,
    run_named_cases, sample_runtime, series, series_device_ptr,
};

fn assert_send_sync<T: Send + Sync>() {}

fn assert_series_residency(series: &Series, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(series.as_column().testing_execution_domain_kind(), "explicit");
    assert_eq!(series.as_column().testing_execution_domain_id(), expected_id);
    assert_eq!(series.as_column().testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(series.as_column().testing_execution_domain_mr_value(), expected_mr);
}

fn assert_frame_residency(df: &DataFrame, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), expected_id);
    assert_eq!(df.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(df.testing_execution_domain_mr_value(), expected_mr);
}

fn assert_scalar_residency(scalar: &Scalar, expected_id: u64, expected_stream: usize) {
    assert_eq!(scalar.testing_execution_domain_kind(), "explicit");
    assert_eq!(scalar.testing_execution_domain_id(), expected_id);
    assert_eq!(scalar.testing_execution_domain_stream_value(), expected_stream);
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

fn runtime_and_execution_context_are_public_and_send_sync() {
    assert_send_sync::<Runtime>();
    assert_send_sync::<ExecutionOverrides<'static>>();

    let rt = Runtime::builder().build().unwrap();
    let _ctx = ExecutionOverrides::new().with_stream(rt.stream()).with_memory_resource(rt.memory_resource());
}

fn runtime_builder_and_factory_families_seed_runtime_exec_contracts() {
    let current_device = DeviceId::current().unwrap();
    let via_new = Runtime::new().unwrap();
    let via_builder = Runtime::builder().build().unwrap();
    assert_eq!(via_new.device(), current_device);
    assert_eq!(via_builder.device(), current_device);
    assert_ne!(via_new.testing_execution_domain_id(), 0);
    assert_ne!(via_builder.testing_execution_domain_id(), 0);

    let rt = sample_runtime();
    let expected_id = rt.testing_execution_domain_id();
    let expected_stream = rt.stream().testing_raw_handle_value();
    let expected_mr = rt.memory_resource().testing_raw_handle_value();

    reset_exec_observation();
    let ints = rt.series(&[3i32, 1, 2]).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&ints, expected_id, expected_stream, expected_mr);
    assert_eq!(ints.to_vec_strict::<i32>().unwrap(), vec![3, 1, 2]);

    reset_exec_observation();
    let strings = rt.series(&["alice", "bob"]).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&strings, expected_id, expected_stream, expected_mr);
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);

    reset_exec_observation();
    let nulls = rt.nulls(TypeId::Int32, 3).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&nulls, expected_id, expected_stream, expected_mr);
    assert_eq!(nulls.null_count(), 3);

    reset_exec_observation();
    let scalar = rt.scalar(7i32).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_scalar_residency(&scalar, expected_id, expected_stream);
    assert_eq!(scalar.value::<i32>().unwrap(), 7);

    reset_exec_observation();
    let string_scalar = rt.string_scalar("alice").unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_scalar_residency(&string_scalar, expected_id, expected_stream);
    assert_eq!(string_scalar.string_value().unwrap(), "alice");

    reset_exec_observation();
    let null_scalar = rt.null_scalar::<i32>().unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_scalar_residency(&null_scalar, expected_id, expected_stream);
    assert!(!null_scalar.is_valid().unwrap());

    reset_exec_observation();
    let null_string_scalar = rt.null_string_scalar().unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_scalar_residency(&null_string_scalar, expected_id, expected_stream);
    assert!(!null_string_scalar.is_valid().unwrap());

    reset_exec_observation();
    let seed_scalar = rt.scalar(7i32).unwrap();
    reset_exec_observation();
    let seeded = rt.series_from_scalar(&seed_scalar, 2).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_series_residency(&seeded, expected_id, expected_stream, expected_mr);
    assert_eq!(seeded.to_vec_strict::<i32>().unwrap(), vec![7, 7]);

    reset_exec_observation();
    let column = rt.column_from_slice(&[10i32, 20, 30]).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(column.testing_execution_domain_id(), expected_id);
    assert_eq!(column.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);

    let empty_table = rt.empty_table().unwrap();
    assert_eq!(empty_table.testing_execution_domain_kind(), "explicit");
    assert_eq!(empty_table.testing_execution_domain_id(), expected_id);
    assert_eq!(empty_table.testing_view().unwrap().num_columns().unwrap(), 0);
    assert_eq!(empty_table.testing_view().unwrap().num_rows().unwrap(), 0);

    let empty_df = rt.empty_dataframe();
    assert_eq!(empty_df.testing_execution_domain_kind(), "explicit");
    assert_eq!(empty_df.testing_execution_domain_id(), expected_id);
    assert_eq!(empty_df.num_columns(), 0);
    assert_eq!(empty_df.num_rows(), 0);
}

fn runtime_builder_pool_memory_resource_and_explicit_mr_setters_use_last_call_wins() {
    let device = DeviceId::current().unwrap();

    let owned_cuda_mr = DeviceMemoryResource::cuda_on(device).unwrap();
    let runtime_from_pool = Runtime::builder()
        .with_memory_resource(owned_cuda_mr.clone())
        .with_pool_memory_resource(16 * 1024 * 1024, None)
        .build()
        .unwrap();
    assert_eq!(runtime_from_pool.device(), device);
    assert_eq!(runtime_from_pool.memory_resource().device(), device);
    assert_ne!(
        runtime_from_pool.memory_resource().testing_raw_handle_value(),
        owned_cuda_mr.testing_raw_handle_value(),
        "with_pool_memory_resource should override an earlier owned MR selection"
    );

    let owned_pool_mr = DeviceMemoryResource::pool_on(device, 16 * 1024 * 1024, None).unwrap();
    let runtime_from_owned = Runtime::builder()
        .with_pool_memory_resource(8 * 1024 * 1024, None)
        .with_memory_resource(owned_pool_mr.clone())
        .build()
        .unwrap();
    assert_eq!(runtime_from_owned.device(), device);
    assert_eq!(runtime_from_owned.memory_resource().device(), device);
    assert_eq!(
        runtime_from_owned.memory_resource().testing_raw_handle_value(),
        owned_pool_mr.testing_raw_handle_value(),
        "with_memory_resource should override an earlier deferred pool selection"
    );
}

fn device_memory_resource_explicit_device_constructors_report_requested_device() {
    let device = DeviceId::current().unwrap();

    let cuda_mr = DeviceMemoryResource::cuda_on(device).unwrap();
    assert_eq!(cuda_mr.device(), device);

    let pool_mr = DeviceMemoryResource::pool_on(device, 16 * 1024 * 1024, None).unwrap();
    assert_eq!(pool_mr.device(), device);

    let runtime = Runtime::builder().with_device(device).with_memory_resource(pool_mr.clone()).build().unwrap();
    assert_eq!(runtime.device(), device);
    assert_eq!(runtime.memory_resource().device(), device);
    assert_eq!(
        runtime.memory_resource().testing_raw_handle_value(),
        pool_mr.testing_raw_handle_value(),
        "runtime should retain the caller-supplied explicit-device MR handle"
    );
}

fn dataframe_in_context_routes_representative_operation_families_through_context_exec() {
    let _event_observation = lock_event_observation();
    let data_rt = sample_runtime();
    let other_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", data_rt.series(&[3i32, 1, 2, 2]).unwrap()),
        ("val", data_rt.series(&[30i32, 10, 20, 40]).unwrap()),
    ])
    .unwrap();
    let rhs = DataFrame::from_series([
        ("id", other_rt.series(&[2i32, 3, 4]).unwrap()),
        ("score", other_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let gather_map = other_rt.series(&[2i32, 0]).unwrap();

    let ctx_stream = ctx_rt.stream().testing_raw_handle_value();
    let ctx_mr = ctx_rt.memory_resource().testing_raw_handle_value();
    let ctx = ExecutionOverrides::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    reset_exec_observation();
    let sorted = df.with_execution(&ctx).sort_values(["id"], SortOptions::default()).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&sorted, df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 2, 3]);

    reset_exec_observation();
    let gathered = df.with_execution(&ctx).gather(&gather_map).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&gathered, df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(gathered.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(2), Some(3)]);

    reset_dependency_observation();
    reset_exec_observation();
    let merged = df.with_execution(&ctx).merge(&rhs, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(dependency_wait_call_count_for_stream(ctx_stream), 2);
    assert_frame_residency(&merged, df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted_i32_i32_rows(&merged, "id", "score"), vec![(2, 200), (2, 200), (3, 300)]);

    reset_dependency_observation();
    reset_exec_observation();
    let grouped = df.groupby(["id"]).with_execution(&ctx).agg(&[AggSpec::new("val", GroupByAgg::Sum)]).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(dependency_wait_call_count_for_stream(ctx_stream), 1);
    assert_frame_residency(&grouped, df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted_i32_i64_rows(&grouped, "id", "val_sum"), vec![(1, 10), (2, 60), (3, 30)]);

    reset_dependency_observation();
    reset_exec_observation();
    let prepared_facts = df.groupby(["id"]).with_execution(&ctx).prepare_groups().unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(dependency_wait_call_count_for_stream(ctx_stream), 1);
    assert_eq!(prepared_facts.num_groups, 3);

    reset_exec_observation();
    let filtered = df.with_execution(&ctx).filter_expr(col("id").ge(2i32) & col("val").lt(35i32)).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&filtered, df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted_i32_i32_rows(&filtered, "id", "val"), vec![(2, 20), (3, 30)]);

    reset_dependency_observation();
    reset_exec_observation();
    let keep_before = dataframe_device_ptr(&df, "val");
    let with_columns = df
        .with_execution(&ctx)
        .with_columns([
            ("score", col("id") + col("val")),
            ("id_total", col("id").agg().sum()),
            ("bucket", when(col("id").ge(2i32)).then("high").otherwise("low")),
        ])
        .unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert!(dependency_wait_call_count_for_stream(ctx_stream) >= 1);
    assert_frame_residency(&with_columns, df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(with_columns.series("val").unwrap().as_column().testing_execution_domain_stream_value(), ctx_stream);
    assert_ne!(
        dataframe_device_ptr(&with_columns, "val"),
        keep_before,
        "scoped with_columns must relocate unchanged columns when the override changes retained provenance"
    );
    assert_eq!(sorted_i32_i32_rows(&with_columns, "id", "score"), vec![(1, 11), (2, 22), (2, 42), (3, 33)]);
    assert_eq!(with_columns.series("id_total").unwrap().to_vec_strict::<i64>().unwrap(), vec![8, 8, 8, 8]);

    reset_exec_observation();
    let with_position = df.with_execution(&ctx).with_position_column("row_id").unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&with_position, df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(with_position.series("row_id").unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(with_position.series("val").unwrap().as_column().testing_execution_domain_stream_value(), ctx_stream);

    let default_with_columns = df.with_columns([("score", col("id") + col("val"))]).unwrap();
    assert_eq!(
        dataframe_device_ptr(&default_with_columns, "val"),
        keep_before,
        "ordinary with_columns should keep unchanged columns Arc-shared when no override changes provenance"
    );

    reset_exec_observation();
    let str_df = DataFrame::from_series([
        ("name", data_rt.series(&["alice", "Bob", "carol"]).unwrap()),
        ("keep", data_rt.series(&[1i32, 2, 3]).unwrap()),
    ])
    .unwrap();
    let str_with_columns =
        str_df.with_execution(&ctx).with_columns([("has_a", col("name").str().contains("a"))]).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&str_with_columns, str_df.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(str_with_columns.series("has_a").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn series_explicit_context_methods_route_representative_operation_families_through_context_exec() {
    let _event_observation = lock_event_observation();
    let data_rt = sample_runtime();
    let other_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let numeric = data_rt.nullable_series(&[30i32, 0, 10, 0], &[true, false, true, false]).unwrap();
    let fill = other_rt.series(&[9i32, 8, 7, 6]).unwrap();
    let rhs = other_rt.series(&[1i32, 2, 3, 4]).unwrap();
    let strings = data_rt.nullable_series(&["Alice", "bob", "", "CARA"], &[true, true, false, true]).unwrap();

    let ctx_stream = ctx_rt.stream().testing_raw_handle_value();
    let ctx_mr = ctx_rt.memory_resource().testing_raw_handle_value();
    let ctx = ExecutionOverrides::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    reset_exec_observation();
    let sorted = numeric.sort_values_in(SortOptions::default(), &ctx).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_series_residency(&sorted, numeric.as_column().testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(sorted.to_optional_vec::<i32>().unwrap(), vec![Some(10), Some(30), None, None]);

    reset_exec_observation();
    let filled = numeric.fillna_with_series_in(&fill, &ctx).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_series_residency(&filled, numeric.as_column().testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![30, 8, 10, 6]);

    reset_dependency_observation();
    reset_exec_observation();
    let added = numeric.add_in(&rhs, &ctx).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(dependency_wait_call_count_for_stream(ctx_stream), 2);
    assert_series_residency(&added, numeric.as_column().testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(added.to_optional_vec::<i32>().unwrap(), vec![Some(31), None, Some(13), None]);

    reset_exec_observation();
    let reduced = numeric.sum_in(&ctx).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(reduced.testing_execution_domain_id(), numeric.as_column().testing_execution_domain_id());

    reset_exec_observation();
    reset_sync_observation();
    assert_eq!(numeric.sum_value_in::<i64>(&ctx).unwrap(), 40);
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_eq!(last_sync_stream_value(), ctx_stream);

    reset_exec_observation();
    let contains = strings.str().contains_in("A", &ctx).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_series_residency(&contains, strings.as_column().testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(contains.to_optional_vec::<bool>().unwrap(), vec![Some(true), Some(false), None, Some(true)]);
}

fn explicit_context_methods_accept_runtime_seeded_dataframe_and_series() {
    let series = series::<i32>("id", &[3, 1, 2]).unwrap();
    let df = DataFrame::from_series([("id", series.clone())]).unwrap();
    let override_stream = CudaStream::new().unwrap();
    let ctx = ExecutionOverrides::new().with_stream(&override_stream);

    let sorted_series = series.sort_values_in(SortOptions::default(), &ctx).unwrap();
    assert_eq!(sorted_series.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(sorted_series.as_column().testing_execution_domain_kind(), "explicit");

    let sorted_df = df.with_execution(&ctx).sort_values(["id"], SortOptions::default()).unwrap();
    assert_eq!(sorted_df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(sorted_df.testing_execution_domain_kind(), "explicit");
}

fn explicit_runtime_operations_use_explicit_domain_only() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", rt.series(&[3i32, 1, 2]).unwrap()),
        ("val", rt.series(&[30i32, 10, 20]).unwrap()),
    ])
    .unwrap();

    let sorted = df.sort_values(["id"], SortOptions::default()).unwrap();
    assert_eq!(sorted.testing_execution_domain_kind(), "explicit");
    assert_eq!(sorted.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn runtime_seeded_readers_use_runtime_stream() {
    let dir = TempDir::new().unwrap();

    let parquet_path = dir.path().join("runtime_seed.parquet");
    let seed_rt = sample_runtime();
    DataFrame::from_series([
        ("id", seed_rt.series(&[1i32, 2, 3]).unwrap()),
        ("value", seed_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap()
    .write_parquet(&parquet_path)
    .unwrap();

    let rt = sample_runtime();
    let expected_stream = rt.stream().testing_raw_handle_value();

    reset_exec_observation();
    let parquet_df = DataFrame::read_parquet_with_options(
        ParquetReadOptions::builder().filepath(&parquet_path).with_runtime(&rt).build(),
    )
    .unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(parquet_df.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(parquet_df.column_names(), vec!["id", "value"]);
}

fn write_options_and_runtime_overrides_use_expected_runtime() {
    let dir = TempDir::new().unwrap();
    let parquet_path = dir.path().join("runtime_write.parquet");
    let json_path = dir.path().join("runtime_write.jsonl");

    let data_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", data_rt.series(&[4i32, 5, 6]).unwrap()),
        ("value", data_rt.series(&[40i32, 50, 60]).unwrap()),
    ])
    .unwrap();

    reset_exec_observation();
    df.write_parquet_with_options(ParquetWriteOptions::builder().filepath(&parquet_path).build()).unwrap();
    assert_eq!(last_exec_stream_value(), data_rt.stream().testing_raw_handle_value());

    reset_sync_observation();
    reset_exec_observation();
    df.write_json_with_options(JsonWriteOptions::builder().filepath(&json_path).with_runtime(&target_rt).build())
        .unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().testing_raw_handle_value());
    assert_eq!(last_exec_mr_value(), target_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(last_sync_stream_value(), target_rt.stream().testing_raw_handle_value());
}

fn copy_into_runtime_preserves_target_residency_and_reuses_allocation_when_possible() {
    let source_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", source_rt.series(&[1i32, 2, 3]).unwrap()),
        ("value", source_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let series = source_rt.series(&[1i32, 2, 3]).unwrap();

    let original_df_ptr = dataframe_device_ptr(&df, "id");
    let original_series_ptr = series_device_ptr(&series);

    reset_exec_observation();
    let copied_df = df.copy_into_runtime(&target_rt).unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().testing_raw_handle_value());
    assert_frame_residency(
        &copied_df,
        target_rt.testing_execution_domain_id(),
        target_rt.stream().testing_raw_handle_value(),
        target_rt.memory_resource().testing_raw_handle_value(),
    );
    assert_ne!(dataframe_device_ptr(&copied_df, "id"), original_df_ptr);

    reset_exec_observation();
    let copied_series = series.copy_into_runtime(&target_rt).unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().testing_raw_handle_value());
    assert_series_residency(
        &copied_series,
        target_rt.testing_execution_domain_id(),
        target_rt.stream().testing_raw_handle_value(),
        target_rt.memory_resource().testing_raw_handle_value(),
    );
    assert_ne!(series_device_ptr(&copied_series), original_series_ptr);

    reset_exec_observation();
    let reused = copied_series.copy_into_runtime(&target_rt).unwrap();
    assert_eq!(last_exec_stream_value(), usize::MAX);
    assert_eq!(series_device_ptr(&reused), series_device_ptr(&copied_series));
}

fn in_context_override_and_copy_into_runtime_keep_distinct_mixed_domain_semantics() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();

    let left = DataFrame::from_series([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("val", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("score", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();

    let override_stream = CudaStream::new().unwrap();
    let ctx_stream = override_stream.testing_raw_handle_value();
    let ctx = ExecutionOverrides::new().with_stream(&override_stream);

    reset_exec_observation();
    let merged_in_context =
        left.with_execution(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(merged_in_context.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(merged_in_context.testing_execution_domain_stream_value(), ctx_stream);

    let left = left.copy_into_runtime(&target_rt).unwrap();
    let right = right.copy_into_runtime(&target_rt).unwrap();
    reset_exec_observation();
    let merged_after_copy = left.merge(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();
    assert_eq!(last_exec_stream_value(), target_rt.stream().testing_raw_handle_value());
    assert_frame_residency(
        &merged_after_copy,
        target_rt.testing_execution_domain_id(),
        target_rt.stream().testing_raw_handle_value(),
        target_rt.memory_resource().testing_raw_handle_value(),
    );
    assert_eq!(merged_after_copy.series("score").unwrap().to_vec_strict::<i32>().unwrap(), vec![200, 300]);
}

fn schema_only_assign_after_copy_into_runtime_preserves_target_residency() {
    let base_rt = sample_runtime();
    let extra_rt = sample_runtime();
    let target_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", base_rt.series(&[1i32, 2, 3]).unwrap()),
        ("val", base_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let replacement = extra_rt.series(&[7i32, 8, 9]).unwrap();
    let bonus = extra_rt.series(&[70i32, 80, 90]).unwrap();

    let df = df.copy_into_runtime(&target_rt).unwrap();
    let replacement = replacement.copy_into_runtime(&target_rt).unwrap();
    let bonus = bonus.copy_into_runtime(&target_rt).unwrap();
    let id_ptr = dataframe_device_ptr(&df, "id");
    let replacement_ptr = series_device_ptr(&replacement);
    let bonus_ptr = series_device_ptr(&bonus);

    reset_exec_observation();
    let result = df.assign([("val", replacement), ("bonus", bonus)]).unwrap();
    assert_eq!(last_exec_stream_value(), usize::MAX);
    assert_frame_residency(
        &result,
        target_rt.testing_execution_domain_id(),
        target_rt.stream().testing_raw_handle_value(),
        target_rt.memory_resource().testing_raw_handle_value(),
    );
    assert_eq!(dataframe_device_ptr(&result, "id"), id_ptr);
    assert_eq!(dataframe_device_ptr(&result, "val"), replacement_ptr);
    assert_eq!(dataframe_device_ptr(&result, "bonus"), bonus_ptr);
    assert_eq!(result.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
}

fn expect_cudf_err<T>(result: cudf::Result<T>, message: &str) -> cudf::Error {
    match result {
        Ok(_) => panic!("{message}"),
        Err(err) => err,
    }
}

fn assert_all_observed_exec_mrs_are_manager_mr(token: &cudf::ExecutionBudgetToken) {
    let expected_mr = token.memory_resource().testing_raw_handle_value();
    let observed_mrs = observed_exec_mr_values();

    assert!(!observed_exec_mr_overflowed(), "exec memory-resource observation buffer overflowed");
    assert!(!observed_mrs.is_empty(), "expected at least one observed budgeted execution allocation route");
    assert!(
        observed_mrs.iter().all(|observed| *observed == expected_mr),
        "expected all observed execution memory resources to be the device manager MR {expected_mr:#x}, got {observed_mrs:?}"
    );
}

fn assert_budget_report_high_water(report: &cudf::ExecutionBudgetReport, retained_or_output_bytes: usize) {
    let high_water = report.allocator_high_water_bytes().expect("budget report should include allocator high-water");
    assert!(high_water > 0, "allocator high-water should be positive: {report:?}");
    assert!(
        high_water >= retained_or_output_bytes,
        "allocator high-water should cover retained/output bytes: high_water={high_water}, retained_or_output={retained_or_output_bytes}, report={report:?}",
    );
    assert!(
        high_water <= report.request().budget_bytes(),
        "allocator high-water should stay within budget: high_water={high_water}, report={report:?}",
    );
    assert!(
        report.allocator_current_bytes().is_some(),
        "budget report should include allocator current bytes: {report:?}",
    );
    assert!(
        report.allocator_peak_aligned_bytes().is_some_and(|bytes| bytes <= report.request().budget_bytes()),
        "aligned allocator peak should stay within budget: {report:?}",
    );
}

fn device_memory_resource_budget_tracking_usage_rejects_non_tracking_resource() {
    let mr = DeviceMemoryResource::cuda().unwrap();
    let err = mr.budget_tracking_usage().expect_err("plain CUDA MR should not expose budget tracking usage");

    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(err.operation(), "memory::device_memory_resource::budget_tracking_usage");
}

fn runtime_series_from_scalar_with_budget_reports_output_and_routes_allocator() {
    let runtime = sample_runtime();
    let scalar = runtime.scalar(42i32).unwrap();
    let token = runtime
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("scalar_series"))
        .unwrap();

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome = runtime.series_from_scalar_with_budget(&scalar, 4, &token).unwrap();
    let observed_mrs = observed_exec_mr_values();
    let series = outcome.value();
    let report = outcome.report();
    let output_bytes = series.device_allocated_size_bytes().unwrap();

    assert_eq!(
        observed_mrs.len(),
        2,
        "expected scalar staging/copy plus scalar broadcast to execute under observed memory resources"
    );
    assert_eq!(series.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42, 42]);
    assert_eq!(
        series.as_column().testing_execution_domain_mr_value(),
        token.memory_resource().testing_raw_handle_value()
    );
    assert_all_observed_exec_mrs_are_manager_mr(&token);
    assert_eq!(report.request().boundary(), "scalar_series");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::StreamBoundReservation);
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(4));
    assert_budget_report_high_water(report, output_bytes);
    let usage = token.reservation_usage().unwrap();
    assert_eq!(usage.peak_bytes as usize, report.allocator_high_water_bytes().unwrap());
    assert!(usage.current_bytes >= output_bytes as u64);
    assert!(usage.peak_aligned_bytes <= report.request().budget_bytes() as u64);
}

fn runtime_series_from_value_with_budget_reports_output_and_routes_allocator() {
    let runtime = sample_runtime();
    let token = runtime
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("value_series"))
        .unwrap();

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome = runtime.series_from_value_with_budget(42i32, 4, &token).unwrap();
    let observed_mrs = observed_exec_mr_values();
    let series = outcome.value();
    let report = outcome.report();
    let output_bytes = series.device_allocated_size_bytes().unwrap();

    assert_eq!(
        observed_mrs.len(),
        2,
        "expected value scalar construction plus scalar broadcast to execute under observed memory resources"
    );
    assert_eq!(series.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42, 42]);
    assert_eq!(
        series.as_column().testing_execution_domain_mr_value(),
        token.memory_resource().testing_raw_handle_value()
    );
    assert_all_observed_exec_mrs_are_manager_mr(&token);
    assert_eq!(report.request().boundary(), "value_series");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::StreamBoundReservation);
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(4));
}

fn runtime_nulls_with_budget_reports_output_and_routes_allocator() {
    let runtime = sample_runtime();
    let token = runtime
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("null_series"))
        .unwrap();

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome = runtime.nulls_with_budget(TypeId::Int32, 4, &token).unwrap();
    let series = outcome.value();
    let report = outcome.report();
    let output_bytes = series.device_allocated_size_bytes().unwrap();

    assert_eq!(series.to_optional_vec::<i32>().unwrap(), vec![None, None, None, None]);
    assert_eq!(series.null_count(), 4);
    assert_eq!(
        series.as_column().testing_execution_domain_mr_value(),
        token.memory_resource().testing_raw_handle_value()
    );
    assert_all_observed_exec_mrs_are_manager_mr(&token);
    assert_eq!(report.request().boundary(), "null_series");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::StreamBoundReservation);
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(4));
}

fn runtime_nulls_with_budget_decimal_scale_routes_allocator() {
    let runtime = sample_runtime();
    let token = runtime
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("decimal_null_series"))
        .unwrap();
    let dtype = DataType::fixed_point(TypeId::Decimal64, -2).unwrap();

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome = runtime.nulls_with_budget(dtype, 4, &token).unwrap();
    let series = outcome.value();
    let view = series.as_column().testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    assert_eq!(view.null_count().unwrap(), 4);
    assert_eq!(view.data_type().unwrap(), dtype);
    assert_eq!(
        series.as_column().testing_execution_domain_mr_value(),
        token.memory_resource().testing_raw_handle_value()
    );
    assert_all_observed_exec_mrs_are_manager_mr(&token);
}

fn runtime_column_from_scalar_with_budget_reports_output_and_routes_allocator() {
    let runtime = sample_runtime();
    let scalar = runtime.scalar(42i32).unwrap();
    let token = runtime
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("scalar_column"))
        .unwrap();

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome = runtime.column_from_scalar_with_budget(&scalar, 4, &token).unwrap();
    let (column, report) = outcome.into_parts();

    assert_eq!(column.to_vec::<i32>().unwrap(), vec![42, 42, 42, 42]);
    assert_eq!(column.testing_execution_domain_mr_value(), token.memory_resource().testing_raw_handle_value());
    assert_all_observed_exec_mrs_are_manager_mr(&token);

    let series = Series::try_from(column).unwrap();
    let output_bytes = series.device_allocated_size_bytes().unwrap();

    assert_eq!(report.request().boundary(), "scalar_column");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::StreamBoundReservation);
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(4));
}

fn runtime_series_from_scalar_with_budget_reports_exceeded_budget_details() {
    let runtime = sample_runtime();
    let scalar = runtime.scalar(42i32).unwrap();
    let token = runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("scalar_series")).unwrap();

    let err = expect_cudf_err(
        runtime.series_from_scalar_with_budget(&scalar, 1024, &token),
        "one-byte budget should not cover scalar-to-series allocation",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "runtime::series_from_scalar_with_budget");
    assert_eq!(
        error_context_value(&err, "budget_operation").as_deref(),
        Some("runtime::series_from_scalar_with_budget")
    );
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("scalar_series"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "budget_enforcement").as_deref(), Some("stream_bound_reservation"));
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"));
    let peak_bytes = error_context_value(&err, "budget_peak_bytes")
        .expect("budget_peak_bytes detail")
        .parse::<usize>()
        .expect("numeric budget_peak_bytes");
    assert!(
        peak_bytes <= 1,
        "failed scalar-to-series budget path may only retain peak usage within the one-byte budget before rejection"
    );
    assert_eq!(error_context_value(&err, "budget_current_bytes").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "budget_failed_allocation_count").as_deref(), Some("1"));
    assert!(error_context_value(&err, "budget_largest_failed_allocation_bytes").is_some());
}

fn runtime_series_from_value_with_budget_reports_exceeded_budget_details() {
    let runtime = sample_runtime();
    let token = runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("value_series")).unwrap();

    let err = expect_cudf_err(
        runtime.series_from_value_with_budget(42i32, 1024, &token),
        "one-byte budget should not cover scalar value-to-series allocation",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "runtime::series_from_value_with_budget");
    assert_eq!(
        error_context_value(&err, "budget_operation").as_deref(),
        Some("runtime::series_from_value_with_budget")
    );
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("value_series"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "budget_enforcement").as_deref(), Some("stream_bound_reservation"));
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"));
}

fn runtime_nulls_with_budget_reports_exceeded_budget_details() {
    let runtime = sample_runtime();
    let token = runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("null_series")).unwrap();

    let err = expect_cudf_err(
        runtime.nulls_with_budget(DataType::from(TypeId::Int32), 1024, &token),
        "one-byte budget should not cover null-series allocation",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "runtime::nulls_with_budget");
    assert_eq!(error_context_value(&err, "budget_operation").as_deref(), Some("runtime::nulls_with_budget"));
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("null_series"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "budget_enforcement").as_deref(), Some("stream_bound_reservation"));
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"));
}

fn runtime_column_from_scalar_with_budget_reports_exceeded_budget_details() {
    let runtime = sample_runtime();
    let scalar = runtime.scalar(42i32).unwrap();
    let token = runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("scalar_column")).unwrap();

    let err = expect_cudf_err(
        runtime.column_from_scalar_with_budget(&scalar, 1024, &token),
        "one-byte budget should not cover scalar-to-column allocation",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "runtime::column_from_scalar_with_budget");
    assert_eq!(
        error_context_value(&err, "budget_operation").as_deref(),
        Some("runtime::column_from_scalar_with_budget")
    );
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("scalar_column"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "budget_enforcement").as_deref(), Some("stream_bound_reservation"));
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"));
}

#[test]
fn runtime_v1_public_api_tests_cases() {
    run_named_cases(&[
        (
            "runtime_and_execution_context_are_public_and_send_sync",
            runtime_and_execution_context_are_public_and_send_sync,
        ),
        (
            "runtime_builder_and_factory_families_seed_runtime_exec_contracts",
            runtime_builder_and_factory_families_seed_runtime_exec_contracts,
        ),
        (
            "runtime_builder_pool_memory_resource_and_explicit_mr_setters_use_last_call_wins",
            runtime_builder_pool_memory_resource_and_explicit_mr_setters_use_last_call_wins,
        ),
        (
            "device_memory_resource_explicit_device_constructors_report_requested_device",
            device_memory_resource_explicit_device_constructors_report_requested_device,
        ),
        (
            "dataframe_in_context_routes_representative_operation_families_through_context_exec",
            dataframe_in_context_routes_representative_operation_families_through_context_exec,
        ),
        (
            "series_explicit_context_methods_route_representative_operation_families_through_context_exec",
            series_explicit_context_methods_route_representative_operation_families_through_context_exec,
        ),
        (
            "explicit_context_methods_accept_runtime_seeded_dataframe_and_series",
            explicit_context_methods_accept_runtime_seeded_dataframe_and_series,
        ),
        ("explicit_runtime_operations_use_explicit_domain_only", explicit_runtime_operations_use_explicit_domain_only),
        ("runtime_seeded_readers_use_runtime_stream", runtime_seeded_readers_use_runtime_stream),
        (
            "write_options_and_runtime_overrides_use_expected_runtime",
            write_options_and_runtime_overrides_use_expected_runtime,
        ),
        (
            "copy_into_runtime_preserves_target_residency_and_reuses_allocation_when_possible",
            copy_into_runtime_preserves_target_residency_and_reuses_allocation_when_possible,
        ),
        (
            "in_context_override_and_copy_into_runtime_keep_distinct_mixed_domain_semantics",
            in_context_override_and_copy_into_runtime_keep_distinct_mixed_domain_semantics,
        ),
        (
            "schema_only_assign_after_copy_into_runtime_preserves_target_residency",
            schema_only_assign_after_copy_into_runtime_preserves_target_residency,
        ),
        (
            "device_memory_resource_budget_tracking_usage_rejects_non_tracking_resource",
            device_memory_resource_budget_tracking_usage_rejects_non_tracking_resource,
        ),
        (
            "runtime_series_from_scalar_with_budget_reports_output_and_routes_allocator",
            runtime_series_from_scalar_with_budget_reports_output_and_routes_allocator,
        ),
        (
            "runtime_series_from_value_with_budget_reports_output_and_routes_allocator",
            runtime_series_from_value_with_budget_reports_output_and_routes_allocator,
        ),
        (
            "runtime_nulls_with_budget_reports_output_and_routes_allocator",
            runtime_nulls_with_budget_reports_output_and_routes_allocator,
        ),
        (
            "runtime_nulls_with_budget_decimal_scale_routes_allocator",
            runtime_nulls_with_budget_decimal_scale_routes_allocator,
        ),
        (
            "runtime_column_from_scalar_with_budget_reports_output_and_routes_allocator",
            runtime_column_from_scalar_with_budget_reports_output_and_routes_allocator,
        ),
        (
            "runtime_series_from_scalar_with_budget_reports_exceeded_budget_details",
            runtime_series_from_scalar_with_budget_reports_exceeded_budget_details,
        ),
        (
            "runtime_series_from_value_with_budget_reports_exceeded_budget_details",
            runtime_series_from_value_with_budget_reports_exceeded_budget_details,
        ),
        (
            "runtime_nulls_with_budget_reports_exceeded_budget_details",
            runtime_nulls_with_budget_reports_exceeded_budget_details,
        ),
        (
            "runtime_column_from_scalar_with_budget_reports_exceeded_budget_details",
            runtime_column_from_scalar_with_budget_reports_exceeded_budget_details,
        ),
    ]);
}
