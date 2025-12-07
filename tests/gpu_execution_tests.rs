use std::sync::{Arc, Mutex};

use arrow::{
    array::{Array, Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::{BoundaryKind, Runtime};
use datafusion::{
    common::ScalarValue,
    execution::context::TaskContext,
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        collect,
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
        test::TestMemoryExec,
    },
};
use datafusion_cudf::{CudfOptimizerConfig, GpuFallbackPolicy, optimize_plan};

static ARROW_HOST_OBSERVATION_LOCK: Mutex<()> = Mutex::new(());

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("value", DataType::Int32, false)]))
}

fn input_batch() -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap()
}

fn sort_input_batches() -> Vec<RecordBatch> {
    vec![
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from(vec![4, 2])), Arc::new(Int32Array::from(vec![40, 20]))],
        )
        .unwrap(),
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from(vec![5, 3])), Arc::new(Int32Array::from(vec![50, 30]))],
        )
        .unwrap(),
    ]
}

fn int32_values(batch: &RecordBatch, index: usize) -> Vec<i32> {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array").values().to_vec()
}

fn int64_values(batch: &RecordBatch, index: usize) -> Vec<i64> {
    batch.column(index).as_any().downcast_ref::<Int64Array>().expect("expected Int64Array").values().to_vec()
}

#[test]
fn gpu_island_executes_filter_and_projection_roundtrip() {
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    let schema = schema();
    let source: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![input_batch()]], Arc::clone(&schema), None).unwrap();
    let predicate =
        binary(col("id", &schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(2))), &schema).unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source).unwrap());
    let filter_schema = filter.schema();
    let total = binary(
        col("id", &filter_schema).unwrap(),
        Operator::Plus,
        col("value", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("id", &filter_schema).unwrap(), alias: "id".to_string() },
                ProjectionExpr { expr: total, alias: "total".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(1);
    let result = optimize_plan(projection, &config).unwrap();
    assert_eq!(result.plan.name(), "CudfGpuIslandExec");

    let batches = futures::executor::block_on(collect(result.plan, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "id");
    assert_eq!(batches[0].schema().field(1).name(), "total");
    assert_eq!(int32_values(&batches[0], 0), vec![2, 3, 4]);
    assert_eq!(int32_values(&batches[0], 1), vec![22, 33, 44]);
}

#[test]
fn gpu_island_executes_limit_after_fused_filter_and_projection() {
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    let schema = schema();
    let source: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![input_batch()]], Arc::clone(&schema), None).unwrap();
    let predicate =
        binary(col("id", &schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(2))), &schema).unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source).unwrap());
    let filter_schema = filter.schema();
    let total = binary(
        col("id", &filter_schema).unwrap(),
        Operator::Plus,
        col("value", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("id", &filter_schema).unwrap(), alias: "id".to_string() },
                ProjectionExpr { expr: total, alias: "total".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let limit: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(projection, 2));

    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let result = optimize_plan(limit, &config).unwrap();
    assert_eq!(result.plan.name(), "CudfGpuIslandExec");

    let batches = futures::executor::block_on(collect(result.plan, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "id");
    assert_eq!(batches[0].schema().field(1).name(), "total");
    assert_eq!(int32_values(&batches[0], 0), vec![2, 3]);
    assert_eq!(int32_values(&batches[0], 1), vec![22, 33]);
}

#[test]
fn gpu_island_executes_sort_and_limit_after_fused_filter_projection_across_input_batches() {
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    let schema = schema();
    let source: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[sort_input_batches()], Arc::clone(&schema), None).unwrap();
    let predicate =
        binary(col("id", &schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(3))), &schema).unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source).unwrap());
    let filter_schema = filter.schema();
    let total = binary(
        col("id", &filter_schema).unwrap(),
        Operator::Plus,
        col("value", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("id", &filter_schema).unwrap(), alias: "id".to_string() },
                ProjectionExpr { expr: total, alias: "total".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("total", &projection_schema).unwrap(),
        ArrowSortOptions::new(true, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, projection));
    let limit: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(sort, 2));

    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let result = optimize_plan(limit, &config).unwrap();
    assert_eq!(result.plan.name(), "CudfGpuIslandExec");

    let plan = Arc::clone(&result.plan);
    let batches = futures::executor::block_on(collect(plan, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "id");
    assert_eq!(batches[0].schema().field(1).name(), "total");
    assert_eq!(int32_values(&batches[0], 0), vec![5, 4]);
    assert_eq!(int32_values(&batches[0], 1), vec![55, 44]);

    let metrics = result.plan.metrics().expect("GPU island should expose execution metrics");
    assert_eq!(metrics.sum_by_name("input_batches").unwrap().as_usize(), 2);
    assert_eq!(metrics.sum_by_name("input_rows").unwrap().as_usize(), 4);
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 2);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("device_to_host_materialized_rows").unwrap().as_usize(), 2);
    assert_eq!(metrics.output_rows(), Some(2));
}

#[test]
fn gpu_island_executes_groupby_after_fused_filter_and_projection_across_input_batches() {
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let batches = vec![
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![10, 20]))],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![30, 40]))],
        )
        .unwrap(),
    ];
    let source: Arc<dyn ExecutionPlan> = TestMemoryExec::try_new_exec(&[batches], Arc::clone(&schema), None).unwrap();
    let predicate =
        binary(col("value", &schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int64(Some(20))), &schema).unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source).unwrap());
    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col("value", &filter_schema).unwrap(), alias: "amount".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let total = AggregateExprBuilder::new(sum_udaf(), vec![col("amount", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("total")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(total)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );
    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("key", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));

    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let result = optimize_plan(sort, &config).unwrap();
    assert_eq!(result.plan.name(), "CudfGpuIslandExec");

    let batches =
        futures::executor::block_on(collect(Arc::clone(&result.plan), Arc::new(TaskContext::default()))).unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "key");
    assert_eq!(batches[0].schema().field(1).name(), "total");
    assert_eq!(int32_values(&batches[0], 0), vec![1, 2]);
    assert_eq!(int64_values(&batches[0], 1), vec![30, 60]);

    let metrics = result.plan.metrics().expect("GPU island should expose execution metrics");
    assert_eq!(metrics.sum_by_name("input_batches").unwrap().as_usize(), 2);
    assert_eq!(metrics.sum_by_name("input_rows").unwrap().as_usize(), 4);
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 2);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("device_to_host_materialized_rows").unwrap().as_usize(), 2);
    assert_eq!(metrics.output_rows(), Some(2));
}

#[test]
fn gpu_island_records_one_upload_and_one_materialization_boundary() {
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();

    let schema = schema();
    let source: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![input_batch()]], Arc::clone(&schema), None).unwrap();
    let predicate =
        binary(col("id", &schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(2))), &schema).unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source).unwrap());
    let filter_schema = filter.schema();
    let total = binary(
        col("id", &filter_schema).unwrap(),
        Operator::Plus,
        col("value", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("id", &filter_schema).unwrap(), alias: "id".to_string() },
                ProjectionExpr { expr: total, alias: "total".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_min_operators_per_island(1)
        .with_runtime(runtime);
    let result = optimize_plan(projection, &config).unwrap();

    let batches = futures::executor::block_on(collect(result.plan, Arc::new(TaskContext::default()))).unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(int32_values(&batches[0], 0), vec![2, 3, 4]);
    assert_eq!(int32_values(&batches[0], 1), vec![22, 33, 44]);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 1);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let observations = observed.lock().unwrap();
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].kind, BoundaryKind::HostToDevice);
    assert_eq!(observations[0].rows, 4);
    assert_eq!(observations[0].columns, 2);
    assert!(!observations[0].synchronized_for_host);
    assert_eq!(observations[1].kind, BoundaryKind::DeviceToHost);
    assert_eq!(observations[1].rows, 3);
    assert_eq!(observations[1].columns, 2);
    assert!(observations[1].synchronized_for_host);

    let observation = cudf::testing::arrow::last_arrow_host_boundary_observation_for_test()
        .expect("GPU island should record its final host materialization boundary");
    assert_eq!(observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(observation.rows, 3);
    assert_eq!(observation.columns, 2);
    assert!(observation.synchronized_for_host);
}

#[test]
fn gpu_island_records_boundary_and_operator_metrics() {
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    let schema = schema();
    let source: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![input_batch()]], Arc::clone(&schema), None).unwrap();
    let predicate =
        binary(col("id", &schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(2))), &schema).unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source).unwrap());
    let filter_schema = filter.schema();
    let total = binary(
        col("id", &filter_schema).unwrap(),
        Operator::Plus,
        col("value", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("id", &filter_schema).unwrap(), alias: "id".to_string() },
                ProjectionExpr { expr: total, alias: "total".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let limit: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(projection, 2));

    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let result = optimize_plan(limit, &config).unwrap();

    let batches =
        futures::executor::block_on(collect(Arc::clone(&result.plan), Arc::new(TaskContext::default()))).unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(int32_values(&batches[0], 0), vec![2, 3]);
    assert_eq!(int32_values(&batches[0], 1), vec![22, 33]);

    let metrics = result.plan.metrics().expect("GPU island should expose execution metrics");
    assert_eq!(metrics.sum_by_name("input_batches").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("input_rows").unwrap().as_usize(), 4);
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("host_to_device_uploaded_rows").unwrap().as_usize(), 4);
    assert!(metrics.sum_by_name("host_to_device_uploaded_bytes").unwrap().as_usize() > 0);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("device_to_host_materialized_rows").unwrap().as_usize(), 2);
    assert!(metrics.sum_by_name("device_to_host_materialized_bytes").unwrap().as_usize() > 0);
    assert!(metrics.sum_by_name("record_batch_upload_time").unwrap().as_usize() > 0);
    assert!(metrics.sum_by_name("cudf_operator_time").unwrap().as_usize() > 0);
    assert!(metrics.sum_by_name("record_batch_materialization_time").unwrap().as_usize() > 0);
    assert_eq!(metrics.output_rows(), Some(2));
}
