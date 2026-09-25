use std::sync::Arc;

use arrow::{
    array::RecordBatch,
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use datafusion::{
    datasource::MemTable, execution::SessionStateBuilder, physical_plan::ExecutionPlan, prelude::SessionContext,
};
use query_engine::plan::{PlanNode, VectorAlgorithmParams, VectorRowIdBinding};

use super::CuvsAlgoExec;
#[cfg(not(feature = "cuvs"))]
use crate::error::{DataFusionErrorExt, ErrorCode};
use crate::{
    cuvs_sql::{CUVS_BRUTE_FORCE_KNN_FUNCTION, CuvsOperation},
    session::AlgeonSessionStateBuilderExt,
};

fn vector_schema(id_name: &str, id_type: DataType) -> SchemaRef {
    let mut fields = vec![Field::new(id_name, id_type, false)];
    fields.extend((0..2).map(|index| Field::new(format!("d{index}"), DataType::Float32, false)));
    Arc::new(Schema::new(fields))
}

fn register_table(ctx: &SessionContext, name: &str, schema: SchemaRef) {
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    let provider = MemTable::try_new(schema, vec![vec![batch]]).expect("fixture MemTable");
    ctx.register_table(name, Arc::new(provider)).expect("register fixture table");
}

fn lowering_context() -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cuvs_sql()
        .expect("install cuVS SQL contracts")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_table(&ctx, "dataset_vectors", vector_schema("item_id", DataType::Int64));
    register_table(&ctx, "query_vectors", vector_schema("query_id", DataType::Utf8));
    ctx
}

fn find_cuvs_exec(plan: &dyn ExecutionPlan) -> Option<&CuvsAlgoExec> {
    if let Some(exec) = plan.downcast_ref::<CuvsAlgoExec>() {
        return Some(exec);
    }
    plan.children().into_iter().find_map(|child| find_cuvs_exec(child.as_ref()))
}

#[tokio::test]
async fn physical_planning_lowers_typed_vector_ir_and_clone_retains_planner() {
    let ctx = lowering_context();
    let sql = format!(
        "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'cosine')"
    );
    let logical = ctx.state().create_logical_plan(&sql).await.expect("cuVS logical planning");
    let physical = ctx.state().create_physical_plan(&logical).await.expect("cuVS physical planning");
    let exec = find_cuvs_exec(physical.as_ref()).expect("CuvsAlgoExec");
    assert_eq!(exec.operation(), CuvsOperation::BruteForceKnn);
    assert_eq!(exec.children().len(), 2);
    assert_eq!(exec.schema().as_ref(), logical.schema().as_arrow());
    let native = exec.native_plan();
    let vector = native
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::VectorAlgorithm(vector) => Some(vector),
            _ => None,
        })
        .expect("VectorAlgorithmNode");
    assert_eq!(vector.inputs.len(), 2);
    assert_eq!(vector.spec.params.relation_bindings().count(), 2);
    let VectorAlgorithmParams::BruteForceKnn(params) = &vector.spec.params else {
        panic!("brute-force kNN must lower to its typed parameter payload")
    };
    assert_eq!(params.dataset.row_id, VectorRowIdBinding::Column("id".to_string()));
    assert_eq!(params.queries.row_id, VectorRowIdBinding::Column("id".to_string()));
    assert_eq!(
        vector.spec.output_schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        ["query_ordinal", "query_id", "neighbor_ordinal", "neighbor_id", "rank", "distance"],
        "ordinal outputs supplement the caller's row IDs instead of replacing them"
    );

    let cloned = SessionStateBuilder::new_from_existing(ctx.state()).build();
    let cloned_physical = cloned.create_physical_plan(&logical).await.expect("cloned planner retained");
    assert!(find_cuvs_exec(cloned_physical.as_ref()).is_some());

    #[cfg(not(feature = "cuvs"))]
    {
        let error = ctx.sql(&sql).await.unwrap().collect().await.expect_err("feature-disabled execution");
        let typed = error.find_algeon_error().expect("structured cuVS feature-disabled error");
        assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
    }
}
