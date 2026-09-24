use std::sync::Arc;

use arrow::{
    array::RecordBatch,
    datatypes::{DataType, Field, Schema},
};
use datafusion::{
    datasource::MemTable,
    execution::SessionStateBuilder,
    physical_plan::{ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions},
    prelude::SessionContext,
};
use query_engine::{
    plan::{MlAlgorithmParams, PlanNode, QueryPlanBuilder},
    source::{LocalParquetSource, SourceNode},
};

use super::CumlAlgoExec;
use crate::{
    cuml_sql::CumlOperation,
    native_exec::AlgeonNativeGpuProducerExec,
    planner::{FinalPlanDisposition, classify_final_plan},
    session::AlgeonSessionStateBuilderExt,
};

#[tokio::test]
async fn physical_planning_lowers_cuml_operations_into_typed_ml_ir() {
    let state = SessionStateBuilder::new_with_default_features().try_with_cuml_sql().unwrap().build();
    let ctx = SessionContext::new_with_state(state);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    let sql = "SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM t), n_components => 1)";
    let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
    let exec = find_cuml_exec(plan.as_ref()).expect("cuML physical node");
    assert_eq!(exec.operation(), CumlOperation::Tsvd);
    let native = exec.native_plan().unwrap();
    let ml = native
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::MlAlgorithm(ml) => Some(ml),
            _ => None,
        })
        .expect("typed ML node");
    assert_eq!(ml.inputs.len(), 1);
    assert!(matches!(&ml.spec.params, MlAlgorithmParams::Tsvd(_)));
}

#[tokio::test]
async fn sql_exec_with_native_gpu_input_has_native_final_plan_disposition() {
    let state = SessionStateBuilder::new_with_default_features().try_with_cuml_sql().unwrap().build();
    let ctx = SessionContext::new_with_state(state);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    ctx.register_table("t", Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![batch]]).unwrap())).unwrap();
    let sql = "SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM t), n_components => 1)";
    let physical = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
    let exec = find_cuml_exec(physical.as_ref()).expect("cuML physical node");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/algeon-cuml-native-classification.parquet",
        schema,
        vec!["id".into(), "d0".into(), "d1".into()],
    )));
    let child =
        Arc::new(AlgeonNativeGpuProducerExec::try_new("native_input", builder.finish_gpu_resident(source)).unwrap());
    let rebound = Arc::new(exec.clone())
        .replace_children(vec![child], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();
    assert_eq!(classify_final_plan(rebound.as_ref()), FinalPlanDisposition::Native);
}

fn find_cuml_exec(plan: &dyn ExecutionPlan) -> Option<&CumlAlgoExec> {
    plan.downcast_ref::<CumlAlgoExec>()
        .or_else(|| plan.children().into_iter().find_map(|child| find_cuml_exec(child.as_ref())))
}
