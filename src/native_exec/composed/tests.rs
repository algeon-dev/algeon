use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::PhysicalExpr,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, empty::EmptyExec,
        metrics::ExecutionPlanMetricsSet,
    },
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof,
    expr::{Expr, LimitSpec, SortExpr, SortSpec},
    graph_memory::{GraphMemoryCalibrationOverride, graph_memory_bounded_admission_evidence_for_plan},
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
        GraphEdgeColumnBindings, ProjectionOutput, QueryPlan, QueryPlanBuilder,
    },
    runtime::{ExecutionContext, QueryAdmissionRequest, resident::GpuResultHandle},
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

use super::*;
use crate::{
    backend::InstalledTestGpuBackend,
    error::{DataFusionErrorExt, Error, ErrorCode},
    native_exec::{
        NexusNativeGpuProducerExec,
        admission::{QueryAdmissionTicket, task_context_with_query_admission_ticket},
        graph::{GraphExecutionPlan, register_graph_execution_plan_discovery},
    },
};

#[derive(Debug)]
struct FakeGpuResidentProducer {
    input_name: String,
    ticket: Arc<QueryAdmissionTicket>,
    capability_proof: QueryPlanCapabilityProof,
    graph_execution: bool,
    execute_count: Arc<AtomicUsize>,
    inner: EmptyExec,
}

#[derive(Debug)]
struct FakeHostOwnedGraphExec {
    inner: EmptyExec,
}

impl DisplayAs for FakeHostOwnedGraphExec {
    fn fmt_as(&self, format: DisplayFormatType, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt_as(format, formatter)
    }
}

impl ExecutionPlan for FakeHostOwnedGraphExec {
    fn name(&self) -> &str {
        "FakeHostOwnedGraphExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.inner.properties()
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Execution("fake host-owned graph has no children".to_string()))
        }
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        self.inner.execute(partition, task_context)
    }
}

impl GraphExecutionPlan for FakeHostOwnedGraphExec {
    fn graph_execution_capability_proof(&self) -> DataFusionResult<QueryPlanCapabilityProof> {
        Err(DataFusionError::Execution("fake host-owned graph does not issue a proof".to_string()))
    }

    fn graph_result_input_name(&self) -> &str {
        "fake_input"
    }

    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        SealedAttemptMetricsTarget::new(&ExecutionPlanMetricsSet::new(), GraphClaimCalibrationCapture::default())
    }

    fn gpu_resident_output_rejection(&self) -> Option<DataFusionError> {
        Some(
            Error::new(
                ErrorCode::RequiredGpuOperationUnsupported,
                "datafusion_nexus.test.host_owned_graph",
                "fake graph output is host-owned",
            )
            .with_fact("reason", "logical_bfs_gpu_result_unsupported")
            .with_fact("output_boundary", "gpu_resident")
            .into(),
        )
    }
}

impl DisplayAs for FakeGpuResidentProducer {
    fn fmt_as(&self, format: DisplayFormatType, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt_as(format, formatter)
    }
}

impl ExecutionPlan for FakeGpuResidentProducer {
    fn name(&self) -> &str {
        "FakeGpuResidentProducer"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.inner.properties()
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Execution("fake GPU-resident producer has no children".to_string()))
        }
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        self.inner.execute(partition, task_context)
    }
}

impl GpuResidentProducer for FakeGpuResidentProducer {
    fn gpu_result_input_name(&self) -> &str {
        &self.input_name
    }

    fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
        vec![Arc::clone(&self.ticket)]
    }

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        _native_context: &'a ExecutionContext,
        _partition: usize,
        _task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>> {
        self.execute_count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(DataFusionError::Execution(
                "fake GPU-resident producer is only used to test composed admission proof completion".to_string(),
            ))
        })
    }
}

impl GraphExecutionPlan for FakeGpuResidentProducer {
    fn graph_execution_capability_proof(&self) -> DataFusionResult<QueryPlanCapabilityProof> {
        Ok(self.capability_proof.clone())
    }

    fn graph_result_input_name(&self) -> &str {
        &self.input_name
    }

    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        SealedAttemptMetricsTarget::new(&ExecutionPlanMetricsSet::new(), GraphClaimCalibrationCapture::default())
    }
}

fn fake_gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    plan.downcast_ref::<FakeGpuResidentProducer>().map(|producer| producer as &dyn GpuResidentProducer)
}

fn fake_graph_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
    plan.downcast_ref::<FakeGpuResidentProducer>()
        .filter(|producer| producer.graph_execution)
        .map(|producer| producer as &dyn GraphExecutionPlan)
}

fn fake_host_owned_graph_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
    plan.downcast_ref::<FakeHostOwnedGraphExec>().map(|graph| graph as &dyn GraphExecutionPlan)
}

fn admission_source(backend: &InstalledTestGpuBackend) -> crate::native_exec::QueryAdmissionSource {
    backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority")
}

fn typed_graph_producer_proof() -> (Arc<Schema>, QueryPlanCapabilityProof) {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/composed-admission-proof.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    )));
    let graph = builder.try_add_graph_algorithm(source, spec.clone()).expect("BFS graph plan must lower");
    let graph_plan = builder.finish_gpu_resident(graph);
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&graph_plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("BFS graph plan must have typed memory evidence");
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("BFS graph plan must issue a capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence");
    (spec.output_schema, proof)
}

fn typed_relational_producer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/composed-relational-producer.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&schema),
        schema.fields().iter().map(|field| field.name().to_owned()).collect(),
    )));
    builder.finish_gpu_resident(source)
}

fn typed_relational_producer_proof(schema: Arc<Schema>) -> QueryPlanCapabilityProof {
    let producer_plan = typed_relational_producer_plan(schema);
    QueryPlanCapabilityProof::for_query_plan(&producer_plan).expect("relational producer has an exact proof")
}

fn consumer_plan_for_inputs(input_names: &[&str], schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let inputs = input_names
        .iter()
        .map(|input_name| {
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(*input_name, Arc::clone(&schema))))
        })
        .collect::<Vec<_>>();
    let root = match inputs.as_slice() {
        [input] => *input,
        _ => builder.add_union(inputs, Arc::clone(&schema)),
    };
    builder.finish_with_host_sink(root)
}

fn consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    consumer_plan_for_inputs(&["fake_input"], schema)
}

#[test]
fn composed_constructor_preserves_host_owned_graph_gpu_output_rejection() {
    register_graph_execution_plan_discovery(fake_host_owned_graph_execution_plan);
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let child = Arc::new(FakeHostOwnedGraphExec { inner: EmptyExec::new(schema) }) as Arc<dyn ExecutionPlan>;

    let error =
        NexusNativeComposedExec::try_new(consumer_plan, vec![child], NexusNativeExecutionConfig::bounded_default())
            .expect_err("host-owned graph results must not be accepted as GPU-resident producers");

    assert_eq!(error.code(), ErrorCode::RequiredGpuOperationUnsupported);
    assert_eq!(error.fact_value("reason"), Some("logical_bfs_gpu_result_unsupported"));
    assert_eq!(error.fact_value("output_boundary"), Some("gpu_resident"));
}

fn non_attested_projection_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let projected = builder.add_projection(
        source,
        schema
            .fields()
            .iter()
            .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
            .collect(),
    );
    builder.finish_with_host_sink(projected)
}

fn filter_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let filtered =
        builder.add_filter(source, Expr::is_not_null(Expr::column(schema.field(0).name())), Arc::clone(&schema));
    builder.finish_with_host_sink(filtered)
}

fn zero_copy_projection_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let field = schema.field(0).clone();
    let projected = builder.add_projection(source, vec![ProjectionOutput::new(Expr::column(field.name()), field)]);
    builder.finish_with_host_sink(projected)
}

fn limit_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let limited = builder.add_limit(source, LimitSpec::fetch(1), schema);
    builder.finish_with_host_sink(limited)
}

fn sorted_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    consumer_plan(schema).with_appended_sort(SortSpec::new(vec![SortExpr::new(Expr::column("vertex"), false, false)]))
}

fn sorted_consumer_plan_with_shape(
    schema: Arc<Schema>,
    key_count: usize,
    row_count_upper_bound: Option<u64>,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut source = GpuResidentSource::new("fake_input", Arc::clone(&schema));
    if let Some(row_count_upper_bound) = row_count_upper_bound {
        source = source.with_row_count_upper_bound(row_count_upper_bound);
    }
    let source = builder.add_source(SourceNode::GpuResident(source));
    let sort = builder.add_sort(
        source,
        SortSpec::new((0..key_count).map(|_| SortExpr::new(Expr::column("vertex"), false, false)).collect()),
        schema,
    );
    builder.finish_with_host_sink(sort)
}

#[test]
fn complete_composed_admission_proof_rebinds_fake_producer_with_full_cpu_only_proof() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_proof) = typed_graph_producer_proof();
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let consumer_proof = QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
        .expect("consumer has a composed passthrough proof");
    let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: Arc::clone(&producer_ticket),
        capability_proof: producer_proof.clone(),
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let task_context = TaskContext::default();

    let completed = complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &task_context)
        .expect("proof completion is CPU-only and does not acquire CUDA");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("complete proof installs one ticket");

    assert!(!Arc::ptr_eq(&completed_ticket, &producer_ticket), "completion reissues the prepared ticket");
    let expected_proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        consumer_proof,
        [ComposedGraphProducerCandidate::new(
            Some("fake_input".to_string()),
            Some("fake_input".to_string()),
            Some(producer_proof.clone()),
        )],
    )
    .expect("single graph producer uses the engine composed resolver");
    assert_eq!(completed_ticket.attempt_capability_proof(), Some(&expected_proof));
}

#[test]
fn complete_composed_admission_proof_aggregates_non_attested_consumer_fallback_scope() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_proof) = typed_graph_producer_proof();
    let consumer_plan = non_attested_projection_consumer_plan(Arc::clone(&schema));
    let consumer_proof = QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer has an exact proof");
    let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_proof: producer_proof.clone(),
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let completed =
        complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &TaskContext::default())
            .expect("non-attested one-producer consumer uses ordinary bounded admission");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("fallback completion installs one admission ticket");
    let completed_proof =
        completed_ticket.attempt_capability_proof().expect("fallback completion retains the selected proof");
    let expected = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        consumer_proof,
        [ComposedGraphProducerCandidate::new(
            Some("fake_input".to_string()),
            Some("fake_input".to_string()),
            Some(producer_proof),
        )],
    )
    .expect("the general consumer proof uses the engine composed resolver");
    assert_eq!(completed_proof, &expected);
}

#[test]
fn complete_composed_admission_proof_bounds_filter_projection_and_limit_before_grant() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);

    for (label, build) in [
        ("filter", filter_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
        ("projection", zero_copy_projection_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
        ("limit", limit_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
    ] {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, producer_proof) = typed_graph_producer_proof();
        let consumer_plan = build(Arc::clone(&schema));
        let execute_count = Arc::new(AtomicUsize::new(0));
        let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: producer_ticket,
            capability_proof: producer_proof,
            graph_execution: true,
            execute_count: Arc::clone(&execute_count),
            inner: EmptyExec::new(schema),
        })];
        let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let completed =
            complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &TaskContext::default())
                .unwrap_or_else(|error| panic!("{label} consumer must complete a bounded proof: {error}"));
        assert_eq!(execute_count.load(Ordering::SeqCst), 0, "{label} admission proof does not execute producers");
        let completed_ticket = completed
            .execution_config
            .query_admission_ticket()
            .expect("bounded consumer completion installs an admission ticket");
        let proof = completed_ticket
            .attempt_capability_proof()
            .expect("bounded consumer completion retains the composed proof")
            .clone();
        let handle = backend
            .backend()
            .session_capability()
            .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(proof))
            .unwrap_or_else(|error| panic!("{label} consumer must admit on the CPU-only test backend: {error}"));
        assert_eq!(execute_count.load(Ordering::SeqCst), 0, "{label} admission does not execute producers");
        handle.finish();
        drop(handle);
        drop(completed_ticket);
        drop(prepared_inputs);
    }
}

#[test]
fn complete_composed_admission_proof_preserves_bounded_single_graph_consumer_path() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_proof) = typed_graph_producer_proof();
    let consumer_plan = sorted_consumer_plan(Arc::clone(&schema));
    let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_proof: producer_proof.clone(),
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);

    let standalone_consumer_proof = QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
        .expect("the native full sort has a composed consumer proof");
    let expected_proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        standalone_consumer_proof,
        [ComposedGraphProducerCandidate::new(
            Some("fake_input".to_string()),
            Some("fake_input".to_string()),
            Some(producer_proof.clone()),
        )],
    )
    .expect("the graph sort proof remains valid through the engine composed resolver");
    let completed =
        complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &TaskContext::default())
            .expect("the existing single graph sort must use the engine composed resolver");
    assert_eq!(
        completed
            .execution_config
            .query_admission_ticket()
            .expect("standalone path retains one admission ticket")
            .attempt_capability_proof(),
        Some(&expected_proof)
    );
}

#[test]
fn complete_composed_admission_proof_defers_string_sort_payload_until_resident_refinement() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_proof) = typed_graph_producer_proof();
    let consumer_plan = sorted_consumer_plan_with_shape(Arc::clone(&schema), 1, None);
    let execute_count = Arc::new(AtomicUsize::new(0));
    let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_proof: producer_proof,
        graph_execution: true,
        execute_count: Arc::clone(&execute_count),
        inner: EmptyExec::new(schema),
    })];
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let completed =
        complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &TaskContext::default())
            .expect("string sort retains a deferred resident-evidence phase");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("proof completion installs one admission ticket");
    let proof = completed_ticket
        .attempt_capability_proof()
        .expect("completed ticket retains the composed capability proof")
        .clone();
    let handle = backend
        .backend()
        .session_capability()
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(proof))
        .expect("missing string payload evidence is refined after the producer becomes resident");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "admission must not execute the producer");
    let active = backend.backend().admission_snapshot().expect("test backend exposes admission state");
    assert_eq!(active.active_queries, 1);
    handle.finish();
    drop(handle);
    drop(completed_ticket);
    drop(prepared_inputs);
}

#[test]
fn complete_composed_admission_proof_composes_relational_producers_with_typed_proof() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let producer_proof = typed_relational_producer_proof(Arc::clone(&schema));
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let consumer_proof =
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("relational consumer has an exact proof");
    let expected_proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        consumer_proof,
        [ComposedGraphProducerCandidate::new(Some("fake_input".to_string()), None, Some(producer_proof.clone()))],
    )
    .expect("relational composition uses the same engine composed resolver");
    let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_proof: producer_proof,
        graph_execution: false,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);

    let completed =
        complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &TaskContext::default())
            .expect("pure relational composition must use the engine composed resolver");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("relational composition retains one ticket");

    assert_eq!(completed_ticket.attempt_capability_proof(), Some(&expected_proof));
}

#[test]
fn complete_composed_admission_proof_mixed_producers_preserve_engine_selection() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, graph_proof) = typed_graph_producer_proof();
    let relational_proof = typed_relational_producer_proof(Arc::clone(&schema));
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);

    let graph_only_plan = consumer_plan_for_inputs(&["graph"], Arc::clone(&schema));
    let graph_only_ticket = QueryAdmissionTicket::new(
        config.query_admission_source().expect("test config has admission authority"),
        Some(graph_proof.clone()),
    );
    let graph_only_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "graph".to_string(),
        ticket: graph_only_ticket,
        capability_proof: graph_proof.clone(),
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(Arc::clone(&schema)),
    })];
    let graph_only = complete_composed_admission_proof(
        config.clone(),
        &graph_only_plan,
        &graph_only_inputs,
        &TaskContext::default(),
    )
    .expect("one graph producer has a complete engine-owned derivation");
    let graph_only_proof = graph_only
        .execution_config
        .query_admission_ticket()
        .expect("graph-only completion installs a ticket")
        .attempt_capability_proof()
        .expect("graph-only completion retains a proof")
        .clone();

    let mixed_plan = consumer_plan_for_inputs(&["graph", "relational"], Arc::clone(&schema));
    let mixed_graph_ticket = QueryAdmissionTicket::new(
        config.query_admission_source().expect("test config has admission authority"),
        Some(graph_proof.clone()),
    );
    let mixed_relational_ticket = QueryAdmissionTicket::new(
        config.query_admission_source().expect("test config has admission authority"),
        Some(relational_proof.clone()),
    );
    let mixed_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![
        Arc::new(FakeGpuResidentProducer {
            input_name: "graph".to_string(),
            ticket: mixed_graph_ticket,
            capability_proof: graph_proof.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
        Arc::new(FakeGpuResidentProducer {
            input_name: "relational".to_string(),
            ticket: mixed_relational_ticket,
            capability_proof: relational_proof.clone(),
            graph_execution: false,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
    ];
    let mixed = complete_composed_admission_proof(config.clone(), &mixed_plan, &mixed_inputs, &TaskContext::default())
        .expect("adding a complete relational producer must remain supported");
    let mixed_proof = mixed
        .execution_config
        .query_admission_ticket()
        .expect("mixed completion installs a ticket")
        .attempt_capability_proof()
        .expect("mixed completion retains a proof")
        .clone();
    let expected_mixed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &mixed_plan,
        QueryPlanCapabilityProof::for_query_plan(&mixed_plan).expect("mixed consumer has an exact proof"),
        [
            ComposedGraphProducerCandidate::new(
                Some("graph".to_string()),
                Some("graph".to_string()),
                Some(graph_proof.clone()),
            ),
            ComposedGraphProducerCandidate::new(Some("relational".to_string()), None, Some(relational_proof.clone())),
        ],
    )
    .expect("mixed candidates use the same engine resolver");
    assert_eq!(
        mixed_proof.required_capabilities(),
        expected_mixed.required_capabilities(),
        "mixed proof retains the resolver's capability projection"
    );
    assert_eq!(
        mixed_proof.required_floor_bytes_for(0).expect("mixed proof floor is finite"),
        expected_mixed.required_floor_bytes_for(0).expect("expected mixed proof floor is finite"),
        "mixed proof retains the resolver's finite floor"
    );
    assert!(
        mixed_proof.required_floor_bytes_for(0).is_ok(),
        "adding relational owner terms must not make a complete mixed proof unbounded"
    );
    assert!(graph_only_proof.required_floor_bytes_for(0).is_ok(), "the single graph producer remains finitely bounded");

    let three_input_plan = consumer_plan_for_inputs(&["first", "second", "relational"], Arc::clone(&schema));
    let first_ticket = QueryAdmissionTicket::new(
        config.query_admission_source().expect("test config has admission authority"),
        Some(graph_proof.clone()),
    );
    let second_ticket = QueryAdmissionTicket::new(
        config.query_admission_source().expect("test config has admission authority"),
        Some(graph_proof.clone()),
    );
    let relational_ticket = QueryAdmissionTicket::new(
        config.query_admission_source().expect("test config has admission authority"),
        Some(relational_proof.clone()),
    );
    let three_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![
        Arc::new(FakeGpuResidentProducer {
            input_name: "first".to_string(),
            ticket: first_ticket,
            capability_proof: graph_proof.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
        Arc::new(FakeGpuResidentProducer {
            input_name: "second".to_string(),
            ticket: second_ticket,
            capability_proof: graph_proof.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
        Arc::new(FakeGpuResidentProducer {
            input_name: "relational".to_string(),
            ticket: relational_ticket,
            capability_proof: relational_proof.clone(),
            graph_execution: false,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
    ];
    let three_input =
        complete_composed_admission_proof(config, &three_input_plan, &three_inputs, &TaskContext::default())
            .expect("adding a complete relational producer must not reject two graph producers");
    let three_input_ticket =
        three_input.execution_config.query_admission_ticket().expect("three-input completion installs a ticket");
    let three_input_proof =
        three_input_ticket.attempt_capability_proof().expect("three-input completion retains a proof");
    assert!(
        three_input_proof.required_floor_bytes_for(0).is_ok(),
        "two graph plus one relational producer remains finitely bounded"
    );
}

#[test]
fn complete_composed_admission_proof_preserves_the_active_stream_ticket_before_admission() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_proof) = typed_graph_producer_proof();
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let _consumer_proof =
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer has an exact proof");
    let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let active_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_proof: producer_proof,
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));

    let completed = complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, task_context.as_ref())
        .expect("proof completion must preserve the already-active stream ticket");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("completion retains the active ticket");

    assert!(Arc::ptr_eq(&completed_ticket, &active_ticket));
    assert!(active_ticket.attempt_capability_proof().is_some());
}

#[test]
fn complete_composed_admission_proof_rebinds_outer_ticket_without_producer_ticket() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let producer_plan = typed_relational_producer_plan(Arc::clone(&schema));
    let producer = NexusNativeGpuProducerExec::try_new("fake_input", producer_plan)
        .expect("ticketless relational producer should build");
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let active_ticket = QueryAdmissionTicket::new(source.clone(), None);
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);

    let completed =
        complete_composed_admission_proof(config, &consumer_plan, &[Arc::new(producer)], task_context.as_ref())
            .expect("outer ticket must absorb the complete producer and consumer proof");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("completion retains the active outer ticket");

    assert!(Arc::ptr_eq(&completed_ticket, &active_ticket));
    assert!(active_ticket.attempt_capability_proof().is_some());
}

#[test]
fn complete_composed_admission_proof_keeps_no_authority_unchanged() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let producer =
        NexusNativeGpuProducerExec::try_new("fake_input", typed_relational_producer_plan(Arc::clone(&schema)))
            .expect("ticketless relational producer should build");
    let consumer_plan = consumer_plan(schema);
    let config = NexusNativeExecutionConfig::bounded_default();

    let completed =
        complete_composed_admission_proof(config, &consumer_plan, &[Arc::new(producer)], &TaskContext::default())
            .expect("a direct caller without admission authority remains unchanged");

    assert!(completed.execution_config.query_admission_ticket().is_none());
    assert!(!completed.execution_config.has_query_admission_authority());
}

#[test]
fn complete_composed_admission_proof_rejects_backend_mismatch_before_rebind() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let producer_source = admission_source(&backend);
    let (schema, producer_proof) = typed_graph_producer_proof();
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let producer_ticket = QueryAdmissionTicket::new(producer_source, Some(producer_proof.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_proof: producer_proof,
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    drop(backend);

    let other_backend = InstalledTestGpuBackend::new();
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1)
        .with_query_admission_source(admission_source(&other_backend));
    let result = complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &TaskContext::default());
    let error = match result {
        Ok(_) => panic!("producer and root admission authorities must match"),
        Err(error) => error,
    };
    let native = error.find_nexus_error().expect("backend mismatch remains structured");
    assert_eq!(native.fact_value("reason"), Some("composed_producer_admission_backend_mismatch"));
}

#[tokio::test]
async fn prepared_composed_producers_rebind_one_ordered_scope_proof_and_grant() {
    register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
    register_graph_execution_plan_discovery(fake_graph_execution_plan);
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_proof) = typed_graph_producer_proof();
    let consumer_plan = consumer_plan_for_inputs(&["first", "second"], Arc::clone(&schema));
    let first_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let second_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
    let producers = vec![
        Arc::new(FakeGpuResidentProducer {
            input_name: "first".to_string(),
            ticket: first_ticket,
            capability_proof: producer_proof.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }) as Arc<dyn ExecutionPlan>,
        Arc::new(FakeGpuResidentProducer {
            input_name: "second".to_string(),
            ticket: second_ticket,
            capability_proof: producer_proof.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }) as Arc<dyn ExecutionPlan>,
    ];
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let composed = NexusNativeComposedExec::try_new(consumer_plan.clone(), producers, config)
        .expect("the real composed constructor must retain both prepared producer occurrences");

    let task_context = Arc::new(TaskContext::default());
    let prepared = composed
        .prepare_lakehouse_metadata(Arc::clone(&task_context))
        .await
        .expect("CPU-only composed preparation must preserve both producers");
    let prepared_occurrences = prepared
        .gpu_inputs
        .iter()
        .map(|input| gpu_resident_producer_input_name(input.as_ref()).expect("prepared producer identity"))
        .collect::<Vec<_>>();
    assert_eq!(prepared_occurrences, ["first", "second"]);

    let expected_proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer has an exact proof"),
        [
            ComposedGraphProducerCandidate::new(
                Some("first".to_string()),
                Some("first".to_string()),
                Some(producer_proof.clone()),
            ),
            ComposedGraphProducerCandidate::new(
                Some("second".to_string()),
                Some("second".to_string()),
                Some(producer_proof.clone()),
            ),
        ],
    )
    .expect("prepared producer occurrences must derive one engine-owned composed proof");
    let completed = complete_composed_admission_proof(
        prepared.execution_config.clone(),
        &prepared.plan,
        &prepared.gpu_inputs,
        task_context.as_ref(),
    )
    .expect("proof completion must remain CPU-only before context acquisition");
    let completed_ticket = completed.execution_config.query_admission_ticket().expect("composed proof owns one ticket");
    assert_eq!(completed_ticket.attempt_capability_proof(), Some(&expected_proof));

    // The scheduler receives only this opaque proof. Its selected handle
    // is the immutable grant authority; no GPU work or result handle is
    // created by this assertion.
    let required_floor = expected_proof.required_floor_bytes_for(0).expect("composed proof floor is checked");
    let grant = backend
        .backend()
        .session_capability()
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(expected_proof))
        .expect("the installed test profile must admit the composed proof");
    assert_eq!(grant.device_ordinal(), 0);
    assert!(
        grant.device_budget_bytes() >= required_floor,
        "admission grant must cover the engine-issued composed proof floor"
    );
    drop(grant);

    // Swapping the prepared occurrences changes the schedule proof. A
    // later runtime order change therefore cannot silently reuse this
    // admission evidence.
    let reversed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer has an exact proof"),
        [
            ComposedGraphProducerCandidate::new(
                Some("second".to_string()),
                Some("second".to_string()),
                Some(producer_proof.clone()),
            ),
            ComposedGraphProducerCandidate::new(
                Some("first".to_string()),
                Some("first".to_string()),
                Some(producer_proof),
            ),
        ],
    )
    .expect("reversed occurrence order is independently representable");
    assert_ne!(completed_ticket.attempt_capability_proof(), Some(&reversed));
}
