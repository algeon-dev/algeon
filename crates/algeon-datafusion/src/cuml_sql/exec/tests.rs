use std::sync::Arc;

use arrow::{
    array::{Int64Array, RecordBatch},
    compute::cast,
    datatypes::{DataType, Field, Schema},
};
use datafusion::{
    datasource::MemTable,
    execution::SessionStateBuilder,
    physical_plan::{ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions},
    prelude::SessionContext,
};
use query_engine::{
    admission::AttemptCapabilityEvidence,
    plan::{MlAlgorithmParams, PlanNode, QueryPlanBuilder},
    source::{LocalParquetSource, SourceNode},
};

use super::CumlAlgoExec;
use crate::{
    capability::CapabilityReason,
    cuml_sql::CumlOperation,
    native_exec::{AlgeonNativeGpuProducerExec, gpu_resident_producer},
    planner::{AlgeonNativeOptimizerConfig, FinalPlanDisposition, classify_final_plan, plan_with_native_diagnostics},
    report::CandidateOutcome,
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
    let native = exec.native_plan();
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
    let child_plan = builder.finish_gpu_resident(source).unwrap();
    let child_evidence = AttemptCapabilityEvidence::for_query_plan(&child_plan);
    let [child_scope] = child_evidence.execution_scopes() else {
        panic!("a query plan's evidence has one execution scope")
    };
    let child_scope_fingerprint = child_scope.execution_scope_fingerprint();
    let child = Arc::new(AlgeonNativeGpuProducerExec::try_new("native_input", child_plan).unwrap());
    let rebound = Arc::new(exec.clone())
        .replace_children(vec![child], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();
    assert_eq!(classify_final_plan(rebound.as_ref()), FinalPlanDisposition::Native);

    let producer = gpu_resident_producer(rebound.as_ref()).expect("cuML producer");
    let input_names = producer.native_relation_input_names().expect("relation input names").to_vec();
    for evidence in
        [producer.gpu_result_capability_evidence().unwrap(), producer.host_result_capability_evidence().unwrap()]
    {
        let scopes = evidence.execution_scopes();
        assert!(scopes.iter().any(|scope| scope.execution_scope_fingerprint() == child_scope_fingerprint));
        let occurrences = scopes.iter().flat_map(|scope| scope.producer_occurrences()).collect::<Vec<_>>();
        let [occurrence] = occurrences[..] else { panic!("one producer occurrence: {occurrences:?}") };
        assert_eq!(occurrence.occurrence_id(), input_names[0]);
        assert_eq!(occurrence.execution_scope_fingerprint(), child_scope_fingerprint);
    }
}

fn native_cuml_context(schema: Arc<Schema>) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .try_with_cuml_sql()
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    // One row: an empty table lowers to a zero-row literal, which DataFusion
    // keeps by cost instead of a native input producer.
    let zero = Int64Array::from(vec![0]);
    let columns = schema.fields().iter().map(|field| cast(&zero, field.data_type()).unwrap()).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    ctx
}

#[tokio::test]
async fn final_native_rule_lowers_a_relation_input_into_a_gpu_resident_producer() {
    let ctx = native_cuml_context(Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ])));
    let sql = "SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM t WHERE d0 > 0), n_components => 1)";
    let physical = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
    let exec = find_cuml_exec(physical.as_ref()).expect("cuML physical node");
    let [input] = exec.children()[..] else { panic!("one relation input") };
    assert!(gpu_resident_producer(input.as_ref()).is_some(), "input stayed host-staged: {}", input.name());
    assert_eq!(classify_final_plan(physical.as_ref()), FinalPlanDisposition::Native);
}

#[tokio::test]
async fn unlowerable_relation_input_stays_host_staged_with_a_structured_reason() {
    let ctx = native_cuml_context(Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
        Field::new("h", DataType::Float16, true),
    ])));
    let sql = "SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM t WHERE h IS NOT NULL), n_components => 1)";
    let (state, logical) = ctx.sql(sql).await.unwrap().into_parts();
    let (physical, diagnostics) = plan_with_native_diagnostics(&state, &logical).await.unwrap();
    let exec = find_cuml_exec(physical.as_ref()).expect("cuML physical node");
    let [input] = exec.children()[..] else { panic!("one relation input") };
    assert!(gpu_resident_producer(input.as_ref()).is_none(), "input must stay a DataFusion plan");
    let report = &diagnostics.report;
    let detail = report
        .outcomes()
        .iter()
        .find_map(|outcome| match outcome {
            CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedType, detail, .. } => {
                detail.column.as_ref()
            },
            _ => None,
        })
        .expect("the relation input's unsupported column is reported");
    assert_eq!(detail.name, "h");
}

fn find_cuml_exec(plan: &dyn ExecutionPlan) -> Option<&CumlAlgoExec> {
    plan.downcast_ref::<CumlAlgoExec>()
        .or_else(|| plan.children().into_iter().find_map(|child| find_cuml_exec(child.as_ref())))
}
