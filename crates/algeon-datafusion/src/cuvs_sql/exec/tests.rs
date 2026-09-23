use std::sync::Arc;
#[cfg(feature = "cuvs")]
use std::time::Duration;

use arrow::{
    array::RecordBatch,
    datatypes::{DataType, Field, Schema, SchemaRef},
};
#[cfg(feature = "cuvs")]
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr},
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, PlanProperties, ReplaceChildrenOptions,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use datafusion::{
    datasource::MemTable, execution::SessionStateBuilder, physical_plan::ExecutionPlan, prelude::SessionContext,
};
#[cfg(feature = "cuvs")]
use futures::{StreamExt, stream::once};
use query_engine::plan::{PlanNode, VectorAlgorithmParams, VectorRowIdBinding};
#[cfg(feature = "cuvs")]
use tokio::{
    sync::{mpsc, oneshot},
    time::timeout,
};

use super::CuvsAlgoExec;
#[cfg(feature = "cuvs")]
use crate::{
    attempt::attempt_ticket_for_test,
    backend::InstalledTestGpuBackend,
    native_exec::{admission::task_context_with_query_admission_ticket, below_native_admission},
};
use crate::{
    cuvs_sql::{CUVS_BRUTE_FORCE_KNN_FUNCTION, CuvsOperation},
    error::{DataFusionErrorExt, ErrorCode},
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

/// An input whose one partition reports its first poll, then waits on a
/// channel the test never sends to, as a slow upstream operator does.
#[cfg(feature = "cuvs")]
#[derive(Debug)]
struct GatedInputExec {
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    polled: std::sync::Mutex<Option<oneshot::Sender<()>>>,
    release: std::sync::Mutex<Option<mpsc::Receiver<()>>>,
}

#[cfg(feature = "cuvs")]
impl GatedInputExec {
    fn new(schema: SchemaRef) -> (Self, oneshot::Receiver<()>, mpsc::Sender<()>) {
        let (polled_sender, polled) = oneshot::channel();
        let (release, release_receiver) = mpsc::channel(1);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let exec = Self {
            schema,
            properties,
            polled: std::sync::Mutex::new(Some(polled_sender)),
            release: std::sync::Mutex::new(Some(release_receiver)),
        };
        (exec, polled, release)
    }
}

#[cfg(feature = "cuvs")]
impl DisplayAs for GatedInputExec {
    fn fmt_as(&self, _format: DisplayFormatType, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("GatedInputExec")
    }
}

#[cfg(feature = "cuvs")]
impl ExecutionPlan for GatedInputExec {
    fn name(&self) -> &str {
        "GatedInputExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _task_context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let polled = self.polled.lock().expect("gated input mutex").take();
        let mut release = self.release.lock().expect("gated input mutex").take().expect("the gated input runs once");
        let schema = Arc::clone(&self.schema);
        let stream = once(async move {
            if let Some(polled) = polled {
                let _ = polled.send(());
            }
            let _ = release.recv().await;
            Ok::<_, DataFusionError>(RecordBatch::new_empty(schema))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
    }
}

/// Cancelling the attempt stops host staging while an input stream is still
/// waiting for its first batch.
#[cfg(feature = "cuvs")]
#[tokio::test]
async fn attempt_cancellation_stops_host_staging_while_an_input_waits() {
    let backend = InstalledTestGpuBackend::new();
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .try_with_cuvs_sql()
        .expect("install cuVS SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_table(&ctx, "dataset_vectors", vector_schema("item_id", DataType::Int64));
    register_table(&ctx, "query_vectors", vector_schema("query_id", DataType::Int64));
    let sql = format!(
        "SELECT * FROM {CUVS_BRUTE_FORCE_KNN_FUNCTION}(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'l2_sqrt_expanded')"
    );
    let physical = ctx.sql(&sql).await.unwrap().create_physical_plan().await.expect("cuVS physical planning");
    let exec = find_cuvs_exec(below_native_admission(&physical).as_ref()).expect("CuvsAlgoExec").clone();
    let dataset = Arc::clone(exec.children()[0]);
    let (gated, polled, _release) = GatedInputExec::new(exec.children()[1].schema());
    let exec = (Arc::new(exec) as Arc<dyn ExecutionPlan>)
        .replace_children(
            vec![dataset, Arc::new(gated)],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .expect("gated cuVS relation");

    let source = backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority");
    let (attempt, ticket) = attempt_ticket_for_test(source, None);
    let task_context = task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), ticket);
    let mut stream = exec.execute(0, task_context).expect("the relation stream starts");
    let run = tokio::spawn(async move { stream.next().await });
    timeout(Duration::from_secs(10), polled)
        .await
        .expect("staging polls the gated input")
        .expect("the gated input reports its first poll");

    attempt.cancellation().cancel();

    let item = timeout(Duration::from_secs(10), run)
        .await
        .expect("host staging stops once the attempt is cancelled")
        .expect("the relation task joins")
        .expect("the relation stream yields its cancellation");
    let error = item.expect_err("a cancelled attempt yields no batch");
    assert_eq!(error.find_algeon_error().expect("typed cancellation").code(), ErrorCode::Cancelled);
}
