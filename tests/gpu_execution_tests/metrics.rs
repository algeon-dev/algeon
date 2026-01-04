use std::sync::{Arc, Mutex};

use cudf::{BoundaryKind, Runtime};
use datafusion::{
    common::ScalarValue,
    execution::context::TaskContext,
    logical_expr::Operator,
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{
        ExecutionPlan, collect,
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        test::TestMemoryExec,
    },
};
use datafusion_cudf::{CudfOptimizerConfig, GpuFallbackPolicy, optimize_plan};

use super::support::{ARROW_HOST_OBSERVATION_LOCK, input_batch, int32_values, schema};

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
