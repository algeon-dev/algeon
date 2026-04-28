use std::sync::Arc;

use arrow::{
    array::{Array, Int32Array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    execution::context::TaskContext,
    logical_expr::Operator,
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{
        ExecutionPlan, collect,
        filter::FilterExec,
        projection::{ProjectionExec, ProjectionExpr},
        test::TestMemoryExec,
    },
};
use datafusion_cudf::{CudfOptimizerConfig, GpuFallbackPolicy, optimize_plan};

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

fn int32_values(batch: &RecordBatch, index: usize) -> Vec<i32> {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array").values().to_vec()
}

#[test]
fn gpu_island_executes_filter_and_projection_roundtrip() {
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
