use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use algeon_datafusion::{
    cugraph_sql::{
        CUGRAPH_BFS_FUNCTION, CUGRAPH_OPERATION_REGISTRY, CUGRAPH_PAGERANK_FUNCTION, CugraphAlgorithm, CugraphSqlConfig,
    },
    error::{DataFusionErrorExt, ErrorCode},
    gpu_functions::{GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION},
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{BooleanArray, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::Session,
    common::{DataFusionError, Result},
    datasource::{MemTable, TableProvider, TableType},
    execution::{SessionState, SessionStateBuilder},
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
    prelude::SessionContext,
};

#[tokio::test]
async fn cugraph_sql_frontend_is_public_without_the_native_feature() {
    let _bfs_function: &str = CUGRAPH_BFS_FUNCTION;
    assert_eq!(CugraphAlgorithm::all().len(), CUGRAPH_OPERATION_REGISTRY.len());
    let ctx = feature_off_context(Arc::new(AtomicUsize::new(0)));
    let error =
        ctx.sql("SELECT * FROM cugraph_pagerank()").await.expect_err("missing edges must retain binder validation");
    assert!(error.to_string().contains("required argument 'edges' is missing"), "{error}");
}

#[tokio::test]
async fn feature_off_installation_keeps_the_registry_visible_and_unavailable() {
    let ctx = feature_off_context(Arc::new(AtomicUsize::new(0)));
    let batches = ctx
        .sql(&format!(
            "SELECT available, unavailable_reason FROM {GPU_LIST_FUNCTIONS_FUNCTION}() WHERE provider = 'cugraph'"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let available = batches[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    let reason = batches[0].column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(available.len(), CUGRAPH_OPERATION_REGISTRY.len());
    for row in 0..available.len() {
        assert!(!available.value(row));
        assert_eq!(reason.value(row), "required_feature_disabled");
    }

    let call = serde_json::json!({
        "relations": {"edges": {"table": "edges"}},
        "options": {},
    });
    let batches = ctx
        .sql(&format!(
            "SELECT valid, error_code, relations_resolved, would_execute_gpu FROM {GPU_VALIDATE_CALL_FUNCTION}('{}', '{}')",
            CUGRAPH_PAGERANK_FUNCTION,
            sql_literal(&call.to_string()),
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let valid = batches[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    let code = batches[0].column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let resolved = batches[0].column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
    let would_execute = batches[0].column(3).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(!valid.value(0));
    assert_eq!(code.value(0), "required_feature_disabled");
    assert!(!resolved.value(0));
    assert!(!would_execute.value(0));
}

#[tokio::test]
async fn feature_off_execution_rejects_before_the_edge_source_can_scan() {
    let scans = Arc::new(AtomicUsize::new(0));
    let ctx = feature_off_context(Arc::clone(&scans));
    let dataframe = ctx.sql("SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM edges))").await.unwrap();
    let plan = dataframe.create_physical_plan().await.unwrap();
    assert!(contains_zero_child_feature_disabled_exec(Arc::clone(&plan)));
    assert_eq!(scans.load(Ordering::SeqCst), 0, "physical planning must not scan the edge source");

    let error = dataframe.collect().await.expect_err("feature-off cuGraph execution must fail locally");
    let typed = error.find_algeon_error().unwrap_or_else(|| panic!("missing typed feature error: {error:?}"));
    assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(typed.operation(), "algeon_datafusion.cugraph_sql.execute");
    assert_eq!(typed.facts().get("cugraph_operation"), Some(CUGRAPH_PAGERANK_FUNCTION));
    assert_eq!(typed.facts().get("reason"), Some("required_feature_disabled"));
    assert_eq!(typed.facts().get("required_feature"), Some("cugraph"));
    assert_eq!(typed.facts().get("host_fallback"), Some("disabled"));
    assert_eq!(scans.load(Ordering::SeqCst), 0, "feature-off execution must not scan or materialize edge input");
}

#[tokio::test]
async fn feature_off_execution_remains_typed_under_the_no_cpu_requirement() {
    let scans = Arc::new(AtomicUsize::new(0));
    let ctx = feature_off_no_cpu_context(Arc::clone(&scans));
    let dataframe = ctx.sql("SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM edges))").await.unwrap();
    let plan = dataframe.create_physical_plan().await.unwrap();
    assert!(contains_zero_child_feature_disabled_exec(Arc::clone(&plan)));

    let error = dataframe.collect().await.expect_err("feature-off execution must reach its typed capability error");
    let typed =
        error.find_algeon_error().expect("feature-off error must remain structured under the no-CPU requirement");
    assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(typed.operation(), "algeon_datafusion.cugraph_sql.execute");
    assert_eq!(typed.facts().get("reason"), Some("required_feature_disabled"));
    assert_eq!(scans.load(Ordering::SeqCst), 0, "feature-off execution must not scan or materialize edge input");
}

#[tokio::test]
async fn feature_off_preserves_schema_validation_before_capability_rejection() {
    let ctx = feature_off_context(Arc::new(AtomicUsize::new(0)));
    let error = ctx
        .sql("SELECT * FROM cugraph_pagerank(edges => (SELECT * FROM bad_edges))")
        .await
        .expect_err("missing source column must remain a schema error");
    let typed = error.find_algeon_error().expect("schema validation keeps its structured error");
    assert_eq!(typed.code(), ErrorCode::FieldNotFound);
    assert_eq!(typed.operation(), "algeon_datafusion.cugraph_sql.edge_source_schema");
    assert!(error.to_string().contains("missing canonical source column 'src'"), "{error}");
}

fn feature_off_context(scans: Arc<AtomicUsize>) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .unwrap()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .unwrap()
        .build();
    feature_off_context_with_state(state, scans)
}

fn feature_off_no_cpu_context(scans: Arc<AtomicUsize>) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(
            AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
        )
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .unwrap()
        .build();
    feature_off_context_with_state(state, scans)
}

fn feature_off_context_with_state(state: SessionState, scans: Arc<AtomicUsize>) -> SessionContext {
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("edges", Arc::new(ScanSentinelProvider::new(edge_schema(), scans)))
        .expect("register edge sentinel");
    let bad_schema = Arc::new(Schema::new(vec![Field::new("source", DataType::Int64, false)]));
    let batch = RecordBatch::new_empty(Arc::clone(&bad_schema));
    ctx.register_table(
        "bad_edges",
        Arc::new(MemTable::try_new(bad_schema, vec![vec![batch]]).expect("bad edge fixture")),
    )
    .expect("register bad edge fixture");
    ctx
}

fn edge_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)]))
}

fn contains_zero_child_feature_disabled_exec(plan: Arc<dyn ExecutionPlan>) -> bool {
    (plan.name() == "CugraphFeatureDisabledExec" && plan.children().is_empty())
        || plan.children().into_iter().any(|child| contains_zero_child_feature_disabled_exec(Arc::clone(child)))
}

fn sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}

#[derive(Debug)]
struct ScanSentinelProvider {
    schema: SchemaRef,
    scans: Arc<AtomicUsize>,
}

impl ScanSentinelProvider {
    fn new(schema: SchemaRef, scans: Arc<AtomicUsize>) -> Self {
        Self { schema, scans }
    }
}

#[async_trait::async_trait]
impl TableProvider for ScanSentinelProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.scans.fetch_add(1, Ordering::SeqCst);
        Err(DataFusionError::Execution("feature-off cuGraph SQL must not scan the edge source".to_string()))
    }
}
