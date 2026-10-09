use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow::datatypes::{DataType, Field, Schema};
use cudf::testing;
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
use query_planning::{
    admission::AttemptCapabilityEvidence,
    expr::{Expr, LimitSpec, SortExpr, SortSpec},
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
        GraphEdgeColumnBindings, ProjectionOutput, QueryPlan, QueryPlanBuilder,
    },
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};
use query_runtime::runtime::{ExecutionContext, QueryAdmissionRequest, resident::GpuResultHandle};
use tokio::runtime::Builder as TestRuntimeBuilder;

use super::*;
use crate::{
    attempt::attempt_ticket_for_test,
    backend::InstalledTestGpuBackend,
    error::{Error, ErrorCode},
    native_exec::{
        AlgeonNativeExec, GraphClaimCalibrationCapture, QueryAdmissionSource, SealedAttemptMetricsTarget,
        SourceChunking,
        admission::{QueryAdmissionTicket, task_context_with_query_admission_ticket},
        graph::GraphExecutionPlan,
    },
};

#[derive(Debug)]
pub(crate) struct FakeGpuResidentProducer {
    input_name: String,
    ticket: Arc<QueryAdmissionTicket>,
    capability_evidence: AttemptCapabilityEvidence,
    graph_execution: bool,
    execute_count: Arc<AtomicUsize>,
    inner: EmptyExec,
}

impl FakeGpuResidentProducer {
    /// The fake runs a graph algorithm only when built with `graph_execution`.
    pub(crate) fn graph_execution_plan(&self) -> Option<&dyn GraphExecutionPlan> {
        self.graph_execution.then_some(self as &dyn GraphExecutionPlan)
    }
}

#[derive(Debug)]
pub(crate) struct FakeHostOwnedGraphExec {
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
    fn graph_execution_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.test.host_owned_graph",
            "fake host-owned graph does not issue a evidence",
        ))
    }

    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        SealedAttemptMetricsTarget::new(&ExecutionPlanMetricsSet::new(), GraphClaimCalibrationCapture::default())
    }

    fn gpu_resident_output_rejection(&self) -> Option<Error> {
        Some(
            Error::new(
                ErrorCode::RequiredGpuOperationUnsupported,
                "algeon_datafusion.test.host_owned_graph",
                "fake graph output is host-owned",
            )
            .with_fact("reason", "logical_bfs_gpu_result_unsupported")
            .with_fact("output_boundary", "gpu_resident"),
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

    fn gpu_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        Ok(self.capability_evidence.clone())
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
                "fake GPU-resident producer is only used to test composed admission evidence completion".to_string(),
            ))
        })
    }
}

impl GraphExecutionPlan for FakeGpuResidentProducer {
    fn graph_execution_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        Ok(self.capability_evidence.clone())
    }

    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        SealedAttemptMetricsTarget::new(&ExecutionPlanMetricsSet::new(), GraphClaimCalibrationCapture::default())
    }
}

/// A task context of the invocation whose one admission ticket is `ticket`.
fn invocation_task_context(ticket: &Arc<QueryAdmissionTicket>) -> Arc<TaskContext> {
    task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(ticket))
}

fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
    backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority")
}

fn typed_graph_producer_evidence() -> (Arc<Schema>, AttemptCapabilityEvidence) {
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
            LocalParquetFile::new("/tmp/composed-admission-evidence.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    )));
    let graph = builder.add_graph_algorithm(source, Vec::new(), spec.clone()).expect("BFS graph plan must lower");
    let graph_plan = builder.finish_gpu_resident(graph).unwrap();
    let mut evidence = AttemptCapabilityEvidence::for_query_plan(&graph_plan);
    evidence.attest_cugraph_graph_execution_scope(&graph_plan);
    (spec.output_schema, evidence)
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
    builder.finish_gpu_resident(source).unwrap()
}

fn typed_relational_producer_evidence(schema: Arc<Schema>) -> AttemptCapabilityEvidence {
    let producer_plan = typed_relational_producer_plan(schema);
    AttemptCapabilityEvidence::for_query_plan(&producer_plan)
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
        _ => builder.add_union(inputs, Arc::clone(&schema)).unwrap(),
    };
    builder.finish_with_host_sink(root).unwrap()
}

fn consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    consumer_plan_for_inputs(&["fake_input"], schema)
}

#[test]
fn composed_constructor_preserves_host_owned_graph_gpu_output_rejection() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let child = Arc::new(FakeHostOwnedGraphExec { inner: EmptyExec::new(schema) }) as Arc<dyn ExecutionPlan>;

    let error =
        AlgeonNativeExec::try_new_composed_for_test(consumer_plan, vec![child], AlgeonNativeExecutionConfig::default())
            .expect_err("host-owned graph results must not be accepted as GPU-resident producers");

    assert_eq!(error.code(), ErrorCode::RequiredGpuOperationUnsupported);
    assert_eq!(error.fact_value("reason"), Some("logical_bfs_gpu_result_unsupported"));
    assert_eq!(error.fact_value("output_boundary"), Some("gpu_resident"));
}

fn non_attested_projection_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let projected = builder
        .add_projection(
            source,
            schema
                .fields()
                .iter()
                .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
                .collect(),
        )
        .unwrap();
    builder.finish_with_host_sink(projected).unwrap()
}

fn filter_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let filtered = builder.add_filter(source, Expr::is_not_null(Expr::column(schema.field(0).name()))).unwrap();
    builder.finish_with_host_sink(filtered).unwrap()
}

fn zero_copy_projection_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let field = schema.field(0).clone();
    let projected =
        builder.add_projection(source, vec![ProjectionOutput::new(Expr::column(field.name()), field)]).unwrap();
    builder.finish_with_host_sink(projected).unwrap()
}

fn limit_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
    let limited = builder.add_limit(source, LimitSpec::fetch(1)).unwrap();
    builder.finish_with_host_sink(limited).unwrap()
}

fn sorted_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
    consumer_plan(schema).with_appended_sort(SortSpec::new(vec![SortExpr::new(Expr::column("vertex"), false, false)]))
}

fn complete_composed_admission_evidence_extends_the_invocation_ticket_with_full_cpu_only_evidence() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_evidence) = typed_graph_producer_evidence();
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let (_producer_ticket_attempt, producer_ticket) =
        attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: Arc::clone(&producer_ticket),
        capability_evidence: producer_evidence,
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);
    let task_context = invocation_task_context(&producer_ticket);

    let completed =
        complete_composed_admission_evidence(config, &consumer_plan, &prepared_inputs, task_context.as_ref())
            .expect("evidence completion is CPU-only and does not acquire CUDA");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("complete evidence installs one ticket");

    assert!(Arc::ptr_eq(&completed_ticket, &producer_ticket), "completion keeps the invocation's one ticket");
    let completed_evidence =
        completed_ticket.attempt_capability_evidence().expect("completed ticket retains the composed evidence");
    completed_evidence
        .validate_covers_plan(&consumer_plan)
        .expect("the completed evidence must cover the consumer plan it was derived from");
}

fn complete_composed_admission_evidence_aggregates_non_attested_consumer_fallback_scope() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_evidence) = typed_graph_producer_evidence();
    let consumer_plan = non_attested_projection_consumer_plan(Arc::clone(&schema));
    let (_producer_ticket_attempt, producer_ticket) =
        attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: Arc::clone(&producer_ticket),
        capability_evidence: producer_evidence,
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);
    let completed = complete_composed_admission_evidence(
        config,
        &consumer_plan,
        &prepared_inputs,
        invocation_task_context(&producer_ticket).as_ref(),
    )
    .expect("non-attested one-producer consumer uses ordinary bounded admission");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("fallback completion installs one admission ticket");
    let completed_evidence =
        completed_ticket.attempt_capability_evidence().expect("fallback completion retains the selected evidence");
    completed_evidence
        .validate_covers_plan(&consumer_plan)
        .expect("fallback composed evidence must cover the non-attested consumer plan");
}

fn complete_composed_admission_evidence_bounds_filter_projection_and_limit_before_grant() {
    for (label, build) in [
        ("filter", filter_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
        ("projection", zero_copy_projection_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
        ("limit", limit_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
    ] {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, producer_evidence) = typed_graph_producer_evidence();
        let consumer_plan = build(Arc::clone(&schema));
        let execute_count = Arc::new(AtomicUsize::new(0));
        let (_producer_ticket_attempt, producer_ticket) =
            attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: Arc::clone(&producer_ticket),
            capability_evidence: producer_evidence,
            graph_execution: true,
            execute_count: Arc::clone(&execute_count),
            inner: EmptyExec::new(schema),
        })];
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);
        let completed = complete_composed_admission_evidence(
            config,
            &consumer_plan,
            &prepared_inputs,
            invocation_task_context(&producer_ticket).as_ref(),
        )
        .unwrap_or_else(|error| panic!("{label} consumer must complete a bounded evidence: {error}"));
        assert_eq!(execute_count.load(Ordering::SeqCst), 0, "{label} admission evidence does not execute producers");
        let completed_ticket = completed
            .execution_config
            .query_admission_ticket()
            .expect("bounded consumer completion installs an admission ticket");
        let evidence = completed_ticket
            .attempt_capability_evidence()
            .expect("bounded consumer completion retains the composed evidence")
            .clone();
        let handle = backend
            .backend()
            .session_capability()
            .admit_query_with_request(
                QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence),
            )
            .unwrap_or_else(|error| panic!("{label} consumer must admit on the CPU-only test backend: {error}"));
        assert_eq!(execute_count.load(Ordering::SeqCst), 0, "{label} admission does not execute producers");
        handle.end_test_attempt();
        drop(handle);
        drop(completed_ticket);
        drop(prepared_inputs);
    }
}

#[test]
fn tests_tg_composed_cases_1() {
    testing::run_named_cases(&[
        (
            "complete_composed_admission_evidence_preserves_bounded_single_graph_consumer_path",
            complete_composed_admission_evidence_preserves_bounded_single_graph_consumer_path,
        ),
        (
            "complete_composed_admission_evidence_mixed_producers_preserve_engine_selection",
            complete_composed_admission_evidence_mixed_producers_preserve_engine_selection,
        ),
        (
            "complete_composed_admission_evidence_composes_relational_producers_with_typed_evidence",
            complete_composed_admission_evidence_composes_relational_producers_with_typed_evidence,
        ),
        (
            "complete_composed_admission_evidence_preserves_the_active_stream_ticket_before_admission",
            complete_composed_admission_evidence_preserves_the_active_stream_ticket_before_admission,
        ),
        (
            "complete_composed_admission_evidence_extends_the_invocation_ticket_with_full_cpu_only_evidence",
            complete_composed_admission_evidence_extends_the_invocation_ticket_with_full_cpu_only_evidence,
        ),
    ]);
}

#[test]
fn tests_tg_composed_cases_2() {
    testing::run_named_cases(&[
        (
            "complete_composed_admission_evidence_bounds_filter_projection_and_limit_before_grant",
            complete_composed_admission_evidence_bounds_filter_projection_and_limit_before_grant,
        ),
        (
            "complete_composed_admission_evidence_rebinds_outer_ticket_without_producer_ticket",
            complete_composed_admission_evidence_rebinds_outer_ticket_without_producer_ticket,
        ),
        (
            "complete_composed_admission_evidence_aggregates_non_attested_consumer_fallback_scope",
            complete_composed_admission_evidence_aggregates_non_attested_consumer_fallback_scope,
        ),
        ("prepared_composed_producers_rebind_one_ordered_scope_evidence_and_grant", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(prepared_composed_producers_rebind_one_ordered_scope_evidence_and_grant())
        }),
        (
            "complete_composed_admission_evidence_rejects_backend_mismatch_before_rebind",
            complete_composed_admission_evidence_rejects_backend_mismatch_before_rebind,
        ),
    ]);
}

fn complete_composed_admission_evidence_preserves_bounded_single_graph_consumer_path() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_evidence) = typed_graph_producer_evidence();
    let consumer_plan = sorted_consumer_plan(Arc::clone(&schema));
    let (_producer_ticket_attempt, producer_ticket) =
        attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: Arc::clone(&producer_ticket),
        capability_evidence: producer_evidence,
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);

    let completed = complete_composed_admission_evidence(
        config,
        &consumer_plan,
        &prepared_inputs,
        invocation_task_context(&producer_ticket).as_ref(),
    )
    .expect("the existing single graph sort must use the engine composed resolver");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("standalone path retains one admission ticket");
    completed_ticket
        .attempt_capability_evidence()
        .expect("standalone path retains a composed evidence")
        .validate_covers_plan(&consumer_plan)
        .expect("the sorted graph consumer's evidence must cover its own plan");
}

fn complete_composed_admission_evidence_composes_relational_producers_with_typed_evidence() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let producer_evidence = typed_relational_producer_evidence(Arc::clone(&schema));
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let (_producer_ticket_attempt, producer_ticket) =
        attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: Arc::clone(&producer_ticket),
        capability_evidence: producer_evidence,
        graph_execution: false,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);

    let completed = complete_composed_admission_evidence(
        config,
        &consumer_plan,
        &prepared_inputs,
        invocation_task_context(&producer_ticket).as_ref(),
    )
    .expect("pure relational composition must use the engine composed resolver");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("relational composition retains one ticket");

    completed_ticket
        .attempt_capability_evidence()
        .expect("relational composition retains a composed evidence")
        .validate_covers_plan(&consumer_plan)
        .expect("relational composed evidence must cover the consumer plan");
}

fn complete_composed_admission_evidence_mixed_producers_preserve_engine_selection() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, graph_evidence) = typed_graph_producer_evidence();
    let relational_evidence = typed_relational_producer_evidence(Arc::clone(&schema));
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);

    let mixed_plan = consumer_plan_for_inputs(&["graph", "relational"], Arc::clone(&schema));
    let (_mixed_attempt, mixed_ticket) =
        attempt_ticket_for_test(config.query_admission_source().expect("test config has admission authority"), None);
    let mixed_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![
        Arc::new(FakeGpuResidentProducer {
            input_name: "graph".to_string(),
            ticket: Arc::clone(&mixed_ticket),
            capability_evidence: graph_evidence.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
        Arc::new(FakeGpuResidentProducer {
            input_name: "relational".to_string(),
            ticket: Arc::clone(&mixed_ticket),
            capability_evidence: relational_evidence.clone(),
            graph_execution: false,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
    ];
    let mixed = complete_composed_admission_evidence(
        config.clone(),
        &mixed_plan,
        &mixed_inputs,
        invocation_task_context(&mixed_ticket).as_ref(),
    )
    .expect("adding a complete relational producer must remain supported");
    let mixed_completed_ticket =
        mixed.execution_config.query_admission_ticket().expect("mixed completion installs a ticket");
    let mixed_evidence =
        mixed_completed_ticket.attempt_capability_evidence().expect("mixed completion retains a evidence");
    mixed_evidence
        .validate_covers_plan(&mixed_plan)
        .expect("mixed evidence must cover the mixed-producer consumer plan");

    let three_input_plan = consumer_plan_for_inputs(&["first", "second", "relational"], Arc::clone(&schema));
    let (_three_input_attempt, three_input_ticket) =
        attempt_ticket_for_test(config.query_admission_source().expect("test config has admission authority"), None);
    let three_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![
        Arc::new(FakeGpuResidentProducer {
            input_name: "first".to_string(),
            ticket: Arc::clone(&three_input_ticket),
            capability_evidence: graph_evidence.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
        Arc::new(FakeGpuResidentProducer {
            input_name: "second".to_string(),
            ticket: Arc::clone(&three_input_ticket),
            capability_evidence: graph_evidence,
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
        Arc::new(FakeGpuResidentProducer {
            input_name: "relational".to_string(),
            ticket: Arc::clone(&three_input_ticket),
            capability_evidence: relational_evidence,
            graph_execution: false,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }),
    ];
    let three_input = complete_composed_admission_evidence(
        config,
        &three_input_plan,
        &three_inputs,
        invocation_task_context(&three_input_ticket).as_ref(),
    )
    .expect("adding a complete relational producer must not reject two graph producers");
    three_input
        .execution_config
        .query_admission_ticket()
        .expect("three-input completion installs a ticket")
        .attempt_capability_evidence()
        .expect("three-input completion retains a evidence")
        .validate_covers_plan(&three_input_plan)
        .expect("two graph plus one relational producer remains covered");
}

fn complete_composed_admission_evidence_preserves_the_active_stream_ticket_before_admission() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_evidence) = typed_graph_producer_evidence();
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let (_producer_ticket_attempt, producer_ticket) =
        attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
    let (_active_ticket_attempt, active_ticket) =
        attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_evidence: producer_evidence,
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));

    let completed =
        complete_composed_admission_evidence(config, &consumer_plan, &prepared_inputs, task_context.as_ref())
            .expect("evidence completion must preserve the already-active stream ticket");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("completion retains the active ticket");

    assert!(Arc::ptr_eq(&completed_ticket, &active_ticket));
    assert!(active_ticket.attempt_capability_evidence().is_some());
}

fn complete_composed_admission_evidence_rebinds_outer_ticket_without_producer_ticket() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let producer_plan = typed_relational_producer_plan(Arc::clone(&schema));
    let producer = AlgeonNativeExec::try_new_producer_for_test("fake_input", producer_plan)
        .expect("ticketless relational producer should build");
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let (_active_ticket_attempt, active_ticket) = attempt_ticket_for_test(source.clone(), None);
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);

    let completed =
        complete_composed_admission_evidence(config, &consumer_plan, &[Arc::new(producer)], task_context.as_ref())
            .expect("outer ticket must absorb the complete producer and consumer evidence");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("completion retains the active outer ticket");

    assert!(Arc::ptr_eq(&completed_ticket, &active_ticket));
    assert!(active_ticket.attempt_capability_evidence().is_some());
}

#[test]
fn complete_composed_admission_evidence_keeps_no_authority_unchanged() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let producer =
        AlgeonNativeExec::try_new_producer_for_test("fake_input", typed_relational_producer_plan(Arc::clone(&schema)))
            .expect("ticketless relational producer should build");
    let consumer_plan = consumer_plan(schema);
    let config = AlgeonNativeExecutionConfig::default();

    let completed =
        complete_composed_admission_evidence(config, &consumer_plan, &[Arc::new(producer)], &TaskContext::default())
            .expect("a direct caller without admission authority remains unchanged");

    assert!(completed.execution_config.query_admission_ticket().is_none());
    assert!(!completed.execution_config.has_query_admission_authority());
}

fn complete_composed_admission_evidence_rejects_backend_mismatch_before_rebind() {
    let backend = InstalledTestGpuBackend::new();
    let producer_source = admission_source(&backend);
    let (schema, producer_evidence) = typed_graph_producer_evidence();
    let consumer_plan = consumer_plan(Arc::clone(&schema));
    let (producer_attempt, producer_ticket) = attempt_ticket_for_test(producer_source, Some(producer_evidence.clone()));
    let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
        input_name: "fake_input".to_string(),
        ticket: producer_ticket,
        capability_evidence: producer_evidence,
        graph_execution: true,
        execute_count: Arc::new(AtomicUsize::new(0)),
        inner: EmptyExec::new(schema),
    })];
    drop(producer_attempt);
    drop(backend);

    let other_backend = InstalledTestGpuBackend::new();
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(admission_source(&other_backend));
    let result =
        complete_composed_admission_evidence(config, &consumer_plan, &prepared_inputs, &TaskContext::default());
    let error = match result {
        Ok(_) => panic!("producer and root admission authorities must match"),
        Err(error) => error,
    };
    assert_eq!(error.fact_value("reason"), Some("composed_producer_admission_backend_mismatch"));
}

async fn prepared_composed_producers_rebind_one_ordered_scope_evidence_and_grant() {
    let backend = InstalledTestGpuBackend::new();
    let source = admission_source(&backend);
    let (schema, producer_evidence) = typed_graph_producer_evidence();
    let consumer_plan = consumer_plan_for_inputs(&["first", "second"], Arc::clone(&schema));
    let (_attempt, invocation_ticket) = attempt_ticket_for_test(source.clone(), None);
    let producers = vec![
        Arc::new(FakeGpuResidentProducer {
            input_name: "first".to_string(),
            ticket: Arc::clone(&invocation_ticket),
            capability_evidence: producer_evidence.clone(),
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }) as Arc<dyn ExecutionPlan>,
        Arc::new(FakeGpuResidentProducer {
            input_name: "second".to_string(),
            ticket: Arc::clone(&invocation_ticket),
            capability_evidence: producer_evidence,
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(Arc::clone(&schema)),
        }) as Arc<dyn ExecutionPlan>,
    ];
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);
    let composed = AlgeonNativeExec::try_new_composed_for_test(consumer_plan.clone(), producers, config)
        .expect("the real composed constructor must retain both prepared producer occurrences");

    let task_context = invocation_task_context(&invocation_ticket);
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

    let completed = complete_composed_admission_evidence(
        prepared.execution_config.clone(),
        prepared.native_plan(),
        &prepared.gpu_inputs,
        task_context.as_ref(),
    )
    .expect("evidence completion must remain CPU-only before context acquisition");
    let completed_ticket =
        completed.execution_config.query_admission_ticket().expect("composed evidence owns one ticket");
    let completed_evidence =
        completed_ticket.attempt_capability_evidence().expect("composed evidence completion retains one");
    completed_evidence
        .validate_covers_plan(&consumer_plan)
        .expect("prepared producer occurrences must derive evidence covering the consumer plan");

    // The scheduler receives only this opaque evidence. Its selected handle
    // is the immutable grant authority; no GPU work or result handle is
    // created by this assertion.
    let grant = backend
        .backend()
        .session_capability()
        .admit_query_with_request(
            QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(completed_evidence),
        )
        .expect("the installed test profile must admit the composed evidence");
    assert_eq!(grant.device_ordinal(), 0);
    drop(grant);
}
