use std::{
    collections::BTreeSet,
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use arrow::{
    array::{ArrayRef, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, ScalarValue, Statistics, tree_node::TreeNodeRecursion},
    datasource::{MemTable, source::DataSourceExec},
    execution::{SessionStateBuilder, context::TaskContext, runtime_env::RuntimeEnvBuilder},
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{binary, col, lit},
    },
    physical_plan::{StatisticsArgs, StatisticsContext, collect, displayable, empty::EmptyExec, filter::FilterExec},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::future::BoxFuture;
use parquet::{
    arrow::ArrowWriter,
    file::{metadata::KeyValue, properties::WriterProperties},
};
use query_engine::exec::admit_query_plan;
use query_planning::{
    admission::AttemptCapabilityEvidence,
    expr::{Expr, SortExpr, SortSpec},
    mode::StatisticalAggregateExecutionMode,
    plan::{NodeId, PlanNode, ProjectionOutput, QueryPlan, WindowFunction},
    policy::PlanningPolicy,
    source::{
        GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY, GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY,
        GraphSimpleDirectedEdgeTopology, LiteralBatchSource, LocalParquetFile, LocalParquetSource,
    },
};
use query_runtime::{
    observability::{AttemptDisposition, RuntimeTerminalOutcome},
    runtime::{ExecutionCancellation, QueryAdmissionRequest, StandaloneConfig},
};
use tokio::{task::yield_now, time::timeout};

use super::*;
use crate::{
    attempt::{AlgeonAttempt, AttemptOptions, attempt_ticket_for_test},
    backend::InstalledTestGpuBackend,
    cugraph_sql::{
        CugraphAlgorithmParams, CugraphBfsOutputOptions, CugraphBfsParams, CugraphBfsSource, CugraphGraphOptions,
        CugraphGraphPredicateTable, CugraphPageRankParams, CugraphSqlConfig, CugraphTriangleCountAllParams,
        CugraphVertexIdDomain,
        exec::{
            input::{StagedCugraphSource, execute_edge_source_to_gpu, stage_cugraph_inputs},
            residency::{graph_residency_matches_edge_schema, native_graph_spec_for_residency},
        },
        result_schema_for_params, result_schema_for_vertex_domain,
    },
    error::{DataFusionErrorExt, ErrorCode, ErrorStatus},
    native::{
        ParquetReadStrategyConfig, exact_row_count_for_datafusion_local_parquet_scan,
        graph_residency_descriptor_for_datafusion_local_parquet_scan,
        graph_vertex_upper_bound_for_datafusion_local_parquet_scan, lower_physical_plan_to_native,
    },
    native_exec::{
        AlgeonNativeExec, GraphClaimCalibrationCapture, NativeOutput, QueryAdmissionSource, SealedAttemptMetricsTarget,
        SourceChunking,
        admission::{
            QueryAdmissionTicket, query_admission_ticket_from_task_context, task_context_with_query_admission_ticket,
        },
        admit_for_test, composed_admission_capability_evidence, gpu_resident_producer,
        gpu_resident_producer_unique_output_key,
        graph::{GraphExecutionPlan, graph_execution_plan},
    },
    native_report::NativeDataFusionExecutionMetrics,
    session::AlgeonSessionStateBuilderExt,
};

fn test_gpu_backend() -> InstalledTestGpuBackend {
    InstalledTestGpuBackend::new()
}

fn backend_execution_config(backend: &InstalledTestGpuBackend) -> AlgeonNativeExecutionConfig {
    backend.optimizer_config().native_execution_config().clone()
}

/// `task_context` inside a fresh attempt's invocation on `backend`; native
/// work executing in it binds that attempt's one ticket.
fn attempt_task_context(
    backend: &InstalledTestGpuBackend,
    task_context: TaskContext,
) -> (AlgeonAttempt, Arc<TaskContext>) {
    let (attempt, ticket) = attempt_ticket_for_test(admission_source(backend), None);
    (attempt, task_context_with_query_admission_ticket(Arc::new(task_context), ticket))
}

fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
    backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority")
}

fn edge_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]))
}

fn string_edge_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, true), Field::new("dst", DataType::Utf8, true)]))
}

fn memory_string_edge_source() -> Arc<dyn ExecutionPlan> {
    let schema = string_edge_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("bob"), None, Some("alice")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob"), None])) as ArrayRef,
        ],
    )
    .unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap() as Arc<dyn ExecutionPlan>
}

fn memory_string_path_target_source() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vec!["bob"])) as ArrayRef]).unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap() as Arc<dyn ExecutionPlan>
}

fn memory_edge_source(rows: usize) -> Arc<dyn ExecutionPlan> {
    let schema = edge_schema();
    let src = Arc::new(Int64Array::from_iter_values((0..rows).map(|row| row as i64))) as ArrayRef;
    let dst = Arc::new(Int64Array::from_iter_values((0..rows).map(|row| row as i64 + 1))) as ArrayRef;
    let weight = Arc::new(Float64Array::from_iter_values((0..rows).map(|_| 1.0))) as ArrayRef;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![src, dst, weight]).unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap() as Arc<dyn ExecutionPlan>
}

#[derive(Debug)]
struct CountingExec {
    input: Arc<dyn ExecutionPlan>,
    execute_count: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug)]
pub(crate) struct CountingGpuResidentProducer {
    input_name: String,
    ticket: Arc<QueryAdmissionTicket>,
    capability_evidence: AttemptCapabilityEvidence,
    execute_count: Arc<AtomicUsize>,
    inner: EmptyExec,
}

impl DisplayAs for CountingGpuResidentProducer {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt_as(t, f)
    }
}

impl ExecutionPlan for CountingGpuResidentProducer {
    fn name(&self) -> &str {
        "CountingGpuResidentProducer"
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

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        StatisticsContext::new().compute(&self.inner, &StatisticsArgs::new().with_partition(partition))
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
            Err(DataFusionError::Plan(format!(
                "CountingGpuResidentProducer expected no children, received {}",
                children.len()
            )))
        }
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        self.inner.execute(partition, task_context)
    }
}

impl GpuResidentProducer for CountingGpuResidentProducer {
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
        let execute_count = Arc::clone(&self.execute_count);
        Box::pin(async move {
            execute_count.fetch_add(1, Ordering::SeqCst);
            Err(DataFusionError::Execution(
                "counting producer must not run after composed admission rejection".to_string(),
            ))
        })
    }
}

impl GraphExecutionPlan for CountingGpuResidentProducer {
    fn graph_execution_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        Ok(self.capability_evidence.clone())
    }

    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        SealedAttemptMetricsTarget::new(&ExecutionPlanMetricsSet::new(), GraphClaimCalibrationCapture::default())
    }
}

impl CountingExec {
    fn new(input: Arc<dyn ExecutionPlan>, execute_count: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self { input, execute_count }
    }
}

impl DisplayAs for CountingExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input.fmt_as(t, f)
    }
}

impl ExecutionPlan for CountingExec {
    fn name(&self) -> &str {
        "CountingExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        StatisticsContext::new().compute(self.input.as_ref(), &StatisticsArgs::new().with_partition(partition))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Plan(format!("CountingExec expected one child, received {}", children.len())));
        }
        Ok(Arc::new(Self {
            input: children.pop().expect("child count checked"),
            execute_count: Arc::clone(&self.execute_count),
        }))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        self.execute_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.input.execute(partition, task_context)
    }
}

fn write_edge_parquet(path: &std::path::Path) {
    write_edge_parquet_with_graph_vertex_upper_bound(path, None);
}

fn write_edge_parquet_with_graph_vertex_upper_bound(path: &std::path::Path, graph_vertex_upper_bound: Option<u64>) {
    write_edge_parquet_with_footer_contract(path, graph_vertex_upper_bound, false);
}

fn write_simple_directed_edge_parquet(path: &std::path::Path) {
    write_edge_parquet_with_footer_contract(path, None, true);
}

fn write_edge_parquet_with_footer_contract(
    path: &std::path::Path,
    graph_vertex_upper_bound: Option<u64>,
    simple_directed_edge_topology: bool,
) {
    let schema = edge_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut footer_metadata = graph_vertex_upper_bound
        .map(|vertex_upper_bound| {
            vec![KeyValue::new(
                GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY.to_string(),
                vertex_upper_bound.to_string(),
            )]
        })
        .unwrap_or_default();
    if simple_directed_edge_topology {
        footer_metadata.push(KeyValue::new(
            GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY.to_string(),
            r#"{"source_column_index":0,"destination_column_index":1}"#.to_string(),
        ));
    }
    let mut writer = ArrowWriter::try_new(
        file,
        schema,
        Some(
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(2))
                .set_key_value_metadata((!footer_metadata.is_empty()).then_some(footer_metadata))
                .build(),
        ),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_string_edge_parquet(path: &std::path::Path) {
    let schema = string_edge_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob", "carol"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["bob", "carol", "alice"])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(
        file,
        schema,
        Some(
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(2))
                .set_key_value_metadata(Some(vec![KeyValue::new(
                    GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY.to_string(),
                    r#"{"source_column_index":0,"destination_column_index":1}"#.to_string(),
                )]))
                .build(),
        ),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn datafusion_parquet_edge_source(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    datafusion_parquet_edge_source_sql(path, "SELECT src, dst, weight FROM edges").await
}

async fn datafusion_parquet_edge_source_sql(path: &std::path::Path, sql: &str) -> Arc<dyn ExecutionPlan> {
    let config = SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap()
}

#[test]
fn edge_source_output_rows_rejects_inexact_datafusion_statistics() {
    let edge_source = memory_edge_source(1_000);
    let schema = edge_source.schema();
    let predicate =
        binary(col("src", &schema).unwrap(), Operator::Gt, lit(ScalarValue::Int64(Some(500))), &schema).unwrap();
    let filter = FilterExec::try_new(predicate, edge_source).unwrap();

    assert_eq!(edge_source_output_rows(&filter, &ParquetReadStrategyConfig::default()).unwrap(), None);
}

fn native_edge_source(row_group_rows: Vec<u64>, config: &AlgeonNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_edge_source_with_root(row_group_rows, config, |_, source| source)
}

fn native_edge_source_with_root(
    row_group_rows: Vec<u64>,
    config: &AlgeonNativeExecutionConfig,
    root: impl FnOnce(&mut QueryPlanBuilder, NodeId) -> NodeId,
) -> Arc<dyn ExecutionPlan> {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(row_group_rows)],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let root = root(&mut builder, source);
    native_edge_producer(builder.finish_with_host_sink(root).unwrap(), config)
}

/// A native edge relation in the form planning hands it to a graph: a
/// GPU-resident producer holding its admitted artifact.
fn native_edge_producer(plan: QueryPlan, config: &AlgeonNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
    let admitted = admit_for_test(&plan.clone_with_gpu_resident_sink(), config).unwrap();
    Arc::new(
        AlgeonNativeExec::try_new(
            admitted,
            Vec::new(),
            NativeOutput::GpuResident { input_name: "edges".to_string() },
            config.clone(),
            None,
        )
        .unwrap(),
    )
}

fn native_graph_residency_edge_source_with_stable_id(
    config: &AlgeonNativeExecutionConfig,
    stable_id: &str,
) -> Arc<dyn ExecutionPlan> {
    let file = LocalParquetFile::new("edges.parquet")
        .with_stable_id(stable_id)
        .with_file_size_bytes(1024)
        .with_row_group_rows(vec![100, 200])
        .with_graph_simple_directed_edge_topology(
            GraphSimpleDirectedEdgeTopology::new(0, 1).expect("source and destination columns must differ"),
        );
    let source = LocalParquetSource::from_files(
        vec![file],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projected = builder
        .add_projection(
            source,
            vec![
                ProjectionOutput::new(Expr::column("src"), Field::new("src", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("dst"), Field::new("dst", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("weight"), Field::new("weight", DataType::Float64, false)),
            ],
        )
        .unwrap();
    native_edge_producer(builder.finish_with_host_sink(projected).unwrap(), config)
}

fn native_graph_residency_edge_source(config: &AlgeonNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_graph_residency_edge_source_with_stable_id(config, "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z")
}

fn triangle_spec() -> CugraphAlgorithmSpec {
    let params = CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams);
    CugraphAlgorithmSpec {
        algorithm: CugraphAlgorithm::TriangleCountAll,
        has_weight: true,
        has_edge_id: false,
        graph_options: CugraphGraphOptions {
            directed: false,
            renumber: true,
            construction_policy: CugraphGraphConstructionPolicy::PythonCugraph,
        },
        output_schema: result_schema_for_params(CugraphAlgorithm::TriangleCountAll, &params),
        params,
    }
}

fn pagerank_spec() -> CugraphAlgorithmSpec {
    let params = CugraphAlgorithmParams::PageRank(CugraphPageRankParams::default());
    CugraphAlgorithmSpec {
        algorithm: CugraphAlgorithm::PageRank,
        has_weight: true,
        has_edge_id: false,
        graph_options: CugraphGraphOptions {
            directed: true,
            renumber: true,
            construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
        },
        output_schema: result_schema_for_params(CugraphAlgorithm::PageRank, &params),
        params,
    }
}

fn pagerank_utf8_spec() -> CugraphAlgorithmSpec {
    let mut spec = pagerank_spec();
    spec.has_weight = false;
    spec.output_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));
    spec
}

fn logical_string_spec(algorithm: CugraphAlgorithm) -> CugraphAlgorithmSpec {
    let params = CugraphAlgorithmParams::from_native_defaults(algorithm, None).expect("native defaults are total");
    let graph_options = algorithm.effective_default_graph_options(&CugraphGraphOptions::default());
    let output_schema =
        result_schema_for_vertex_domain(algorithm, &params, Some(&CugraphVertexIdDomain::StringLogical));
    CugraphAlgorithmSpec { algorithm, has_weight: false, has_edge_id: false, graph_options, params, output_schema }
}

#[test]
fn numeric_graph_results_preserve_the_input_vertex_domain_for_every_vertex_column() {
    for algorithm in CugraphAlgorithm::all() {
        let params = CugraphAlgorithmParams::from_native_defaults(algorithm, None).expect("native defaults are total");
        let schema = result_schema_for_vertex_domain(algorithm, &params, Some(&CugraphVertexIdDomain::Int32));
        for output in algorithm.definition().vertex_output_columns() {
            assert_eq!(
                schema.field_with_name(output.name).unwrap().data_type(),
                &DataType::Int32,
                "{} output column {}",
                algorithm.function_name(),
                output.name,
            );
        }
    }
}

fn bfs_spec() -> CugraphAlgorithmSpec {
    let params = CugraphAlgorithmParams::Bfs(Box::new(CugraphBfsParams {
        source: CugraphBfsSource::scalar_i64(0),
        depth_limit: Some(3),
        predicates: Default::default(),
        output: CugraphBfsOutputOptions { mode: CugraphBfsOutputMode::Normalized, return_target_info: false },
    }));
    CugraphAlgorithmSpec {
        algorithm: CugraphAlgorithm::Bfs,
        has_weight: false,
        has_edge_id: false,
        graph_options: CugraphGraphOptions {
            directed: true,
            renumber: true,
            construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
        },
        output_schema: result_schema_for_params(CugraphAlgorithm::Bfs, &params),
        params,
    }
}

fn bfs_utf8_spec() -> CugraphAlgorithmSpec {
    let params = CugraphAlgorithmParams::Bfs(Box::new(CugraphBfsParams {
        source: CugraphBfsSource::scalar_utf8("alice"),
        depth_limit: Some(3),
        predicates: Default::default(),
        output: CugraphBfsOutputOptions { mode: CugraphBfsOutputMode::Normalized, return_target_info: false },
    }));
    CugraphAlgorithmSpec {
        algorithm: CugraphAlgorithm::Bfs,
        has_weight: false,
        has_edge_id: false,
        graph_options: CugraphGraphOptions {
            directed: true,
            renumber: true,
            construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
        },
        output_schema: result_schema_for_params(CugraphAlgorithm::Bfs, &params),
        params,
    }
}

fn bfs_utf8_path_spec() -> CugraphAlgorithmSpec {
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.output.mode = CugraphBfsOutputMode::Path;
    params.predicates.target_vertices = Some(CugraphGraphPredicateTable::new());
    spec.output_schema = result_schema_for_params(CugraphAlgorithm::Bfs, &spec.params);
    spec
}

#[test]
fn logical_bfs_edge_id_capability_is_rejected_during_exec_construction() {
    let mut spec = bfs_utf8_spec();
    spec.has_edge_id = true;
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;

    let error = CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, AlgeonNativeExecutionConfig::default())
        .expect_err("logical BFS edge-id input must fail capability preflight");
    let algeon = error.find_algeon_error().expect("capability preflight must remain a structured adapter error");

    assert_eq!(algeon.code(), ErrorCode::RequiredGpuOperationUnsupported);
    assert_eq!(algeon.fact_value("edge_id_col"), Some("true"));
    assert_eq!(algeon.fact_value("include_edge_ids"), Some("false"));
}

#[test]
fn graph_producer_unique_key_contract_comes_from_the_parameterized_graph_output() {
    let edge_source = memory_edge_source(3);
    let pagerank = CugraphAlgoExec::try_new(
        pagerank_spec(),
        Arc::clone(&edge_source),
        Vec::new(),
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("PageRank producer should build");
    assert_eq!(
        gpu_resident_producer_unique_output_key(&pagerank),
        Some(std::collections::BTreeSet::from(["vertex".to_string()])),
    );

    let bfs = CugraphAlgoExec::try_new(
        bfs_spec(),
        Arc::clone(&edge_source),
        Vec::new(),
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("raw BFS producer should build");
    assert_eq!(
        gpu_resident_producer_unique_output_key(&bfs),
        Some(std::collections::BTreeSet::from(["vertex".to_string()])),
    );

    let mut normalized_bfs_spec = bfs_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut normalized_bfs_spec.params else { unreachable!() };
    params.output.mode = CugraphBfsOutputMode::Normalized;
    normalized_bfs_spec.output_schema = result_schema_for_params(CugraphAlgorithm::Bfs, &normalized_bfs_spec.params);
    let normalized_bfs = CugraphAlgoExec::try_new(
        normalized_bfs_spec,
        Arc::clone(&edge_source),
        Vec::new(),
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("normalized BFS producer should build");
    assert_eq!(
        gpu_resident_producer_unique_output_key(&normalized_bfs),
        Some(std::collections::BTreeSet::from(["vertex".to_string()])),
    );

    let projected_normalized_bfs = CugraphAlgoExec::try_new(
        normalized_bfs.spec,
        Arc::clone(&edge_source),
        Vec::new(),
        Some(vec![1]),
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("projected normalized BFS producer should build");
    assert_eq!(gpu_resident_producer_unique_output_key(&projected_normalized_bfs), None);

    let path_bfs = CugraphAlgoExec::try_new(
        bfs_utf8_path_spec(),
        memory_string_edge_source(),
        vec![memory_string_path_target_source()],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("path BFS producer should build");
    assert!(gpu_resident_producer(&path_bfs).is_some());
    assert_eq!(
        gpu_resident_producer_unique_output_key(&path_bfs),
        Some(std::collections::BTreeSet::from(["path_index".to_string()])),
    );

    let triangle_count = CugraphAlgoExec::try_new(
        triangle_spec(),
        edge_source,
        Vec::new(),
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("triangle-count producer should build");
    assert_eq!(
        gpu_resident_producer_unique_output_key(&triangle_count),
        Some(std::collections::BTreeSet::from(["vertex".to_string()])),
    );
}

#[test]
fn logical_bfs_include_edge_ids_capability_is_rejected_during_exec_construction() {
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.predicates.include_edge_ids = Some(CugraphGraphPredicateTable::new());
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;
    let predicate_schema = Arc::new(Schema::new(vec![Field::new("edge_id", DataType::Int64, false)]));
    let predicate_source = Arc::new(EmptyExec::new(predicate_schema)) as Arc<dyn ExecutionPlan>;

    let error = CugraphAlgoExec::try_new(
        spec,
        edge_source,
        vec![predicate_source],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect_err("logical BFS include-edge-id input must fail capability preflight");
    let algeon = error.find_algeon_error().expect("capability preflight must remain a structured adapter error");

    assert_eq!(algeon.code(), ErrorCode::RequiredGpuOperationUnsupported);
    assert_eq!(algeon.fact_value("edge_id_col"), Some("false"));
    assert_eq!(algeon.fact_value("include_edge_ids"), Some("true"));
}

#[test]
fn cugraph_algo_exec_does_not_request_input_repartitioning() {
    let exec = CugraphAlgoExec::try_new(
        triangle_spec(),
        memory_edge_source(16),
        vec![],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("cuGraph exec should build");

    assert_eq!(exec.benefits_from_input_partitioning(), vec![false]);
}

#[test]
fn logical_bfs_path_gpu_plan_decodes_the_bounded_path_schema() {
    let spec = bfs_utf8_path_spec();
    let plan = cugraph_schema_query_plan(&CugraphPlanDefinition {
        spec,
        edge_schema: string_edge_schema(),
        predicate_schemas: vec![Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]))],
        predicate_row_upper_bounds: vec![Some(1)],
        result_options: CugraphResultPlanOptions::default(),
        edge_normalize_inserted: false,
        edge_row_upper_bound: Some(3),
        edge_vertex_upper_bound: None,
    })
    .expect("logical BFS path GPU plan should build");
    let schema = plan.root_node().output_schema();

    assert_eq!(schema.fields().len(), 5);
    assert_eq!(schema.field(0), &Field::new("path_index", DataType::Int64, false));
    assert_eq!(schema.field(1), &Field::new("source", DataType::Utf8, false));
    assert_eq!(schema.field(2), &Field::new("target", DataType::Utf8, false));
    assert_eq!(schema.field(3), &Field::new("vertex", DataType::Utf8, false));
    assert_eq!(schema.field(4), &Field::new("distance", DataType::Int64, false));
    let PlanNode::Sort(sort) = &plan.root_node().kind else {
        panic!("logical BFS path decoding must restore path_index ordering");
    };
    assert_eq!(sort.spec, SortSpec::new(vec![SortExpr::new(Expr::column("path_index"), false, false)]));
}

#[test]
fn bfs_path_exec_declares_ordering_only_while_path_index_is_preserved() {
    let ordered = CugraphAlgoExec::try_new(
        bfs_utf8_path_spec(),
        memory_string_edge_source(),
        vec![memory_string_path_target_source()],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("bounded BFS path exec should build");
    assert!(ordered.properties().output_ordering().is_some());

    let projected = CugraphAlgoExec::try_new(
        bfs_utf8_path_spec(),
        memory_string_edge_source(),
        vec![memory_string_path_target_source()],
        Some(vec![1, 2, 3, 4]),
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("projected BFS path exec should build");
    assert!(projected.properties().output_ordering().is_none());
}

#[tokio::test]
async fn bfs_path_materializes_only_the_bounded_frame_under_small_host_limit() {
    let backend = test_gpu_backend();
    let mut spec = bfs_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("fixture is BFS") };
    params.depth_limit = Some(6);
    params.output.mode = CugraphBfsOutputMode::Path;
    params.predicates.target_vertices = Some(CugraphGraphPredicateTable::new());
    spec.output_schema = result_schema_for_params(spec.algorithm, &spec.params);

    let target_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let target_batch =
        RecordBatch::try_new(Arc::clone(&target_schema), vec![Arc::new(Int64Array::from(vec![6])) as ArrayRef])
            .unwrap();
    let target_source = MemorySourceConfig::try_new_exec(&[vec![target_batch]], target_schema, None).unwrap();
    let output_limit = 1024;
    let exec = CugraphAlgoExec::try_new_with_host_boundary_config(
        spec,
        memory_edge_source(1_000),
        vec![target_source],
        None,
        None,
        backend_execution_config(&backend),
        GpuIslandHostBoundaryConfig::try_new(1024 * 1024, output_limit).unwrap(),
    )
    .expect("the bounded path must plan below the deliberately small host-output limit");
    let plan = Arc::new(exec) as Arc<dyn ExecutionPlan>;
    let (_attempt, task_context) = attempt_task_context(&backend, TaskContext::default());
    let batches =
        collect(Arc::clone(&plan), task_context).await.expect("only the seven-row path should cross the host boundary");
    let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    assert_eq!(rows, 7);
    let vertices = batches
        .iter()
        .flat_map(|batch| batch.column(3).as_any().downcast_ref::<Int64Array>().unwrap().values().iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(vertices, (0_i64..=6).collect::<Vec<_>>());

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(
        &plan.metrics().expect("the graph execution must publish boundary metrics"),
    );
    assert_eq!(metrics.device_to_host_materializations, 1);
    assert_eq!(metrics.device_to_host_materialized_rows, 7);
    assert!(metrics.device_to_host_materialized_bytes <= output_limit);
}

#[test]
fn string_pagerank_plan_restores_subsumed_top_k_order_after_decode() {
    let mut result_options = CugraphResultPlanOptions::new(None, Some(3));
    result_options.subsumed_top_k_sort = Some(SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]));
    let plan = cugraph_schema_query_plan(&CugraphPlanDefinition {
        spec: pagerank_utf8_spec(),
        edge_schema: string_edge_schema(),
        predicate_schemas: Vec::new(),
        predicate_row_upper_bounds: [].to_vec(),
        result_options: result_options.clone(),
        edge_normalize_inserted: false,
        edge_row_upper_bound: None,
        edge_vertex_upper_bound: None,
    })
    .expect("string PageRank GPU plan should build");

    let codebook = plan
        .nodes()
        .iter()
        .find(|node| {
            matches!(
                &node.kind,
                PlanNode::Window(window)
                    if window.spec.functions.iter().any(|function| matches!(
                        function,
                        WindowFunction::RowNumber { output }
                            if output.name() == "__vertex_id" && output.data_type() == &DataType::Int64
                    ))
            )
        })
        .expect("logical mapping must assign dense Int64 IDs");
    let graph = plan
        .nodes()
        .iter()
        .find(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_)))
        .expect("logical mapping must retain the native graph node");
    let sort = plan
        .nodes()
        .iter()
        .find(|node| matches!(&node.kind, PlanNode::Sort(sort) if sort.input == graph.id))
        .expect("logical PageRank sort must consume the numeric graph result");
    let limit = plan
        .nodes()
        .iter()
        .find(|node| matches!(&node.kind, PlanNode::Limit(limit) if limit.input == sort.id))
        .expect("logical PageRank limit must consume the sorted numeric graph result");
    let decode = plan
        .nodes()
        .iter()
        .find(|node| matches!(&node.kind, PlanNode::Join(decode) if decode.inputs == [limit.id, codebook.id]))
        .expect("logical PageRank top-k must decode through a hash join");
    let PlanNode::Sort(post_decode_sort) = &plan.root_node().kind else {
        panic!("logical PageRank top-k root must restore ordering after decode");
    };

    assert_eq!(post_decode_sort.input, decode.id);
    assert_eq!(post_decode_sort.spec, result_options.subsumed_top_k_sort.unwrap());
    assert_eq!(plan.output_schema().field(0), &Field::new("vertex", DataType::Utf8, false));
}

#[test]
fn m1_string_algorithms_share_the_int64_codebook_and_vertex_decode_path() {
    let algorithms = [
        CugraphAlgorithm::Louvain,
        CugraphAlgorithm::EigenvectorCentrality,
        CugraphAlgorithm::KatzCentrality,
        CugraphAlgorithm::Hits,
        CugraphAlgorithm::DegreesAll,
        CugraphAlgorithm::InDegreesAll,
        CugraphAlgorithm::OutDegreesAll,
        CugraphAlgorithm::WeaklyConnectedComponents,
        CugraphAlgorithm::StronglyConnectedComponents,
        CugraphAlgorithm::TriangleCountAll,
        CugraphAlgorithm::CoreNumber,
        CugraphAlgorithm::Leiden,
        CugraphAlgorithm::Ecg,
        CugraphAlgorithm::ForceAtlas2,
    ];

    for algorithm in algorithms {
        let plan = cugraph_schema_query_plan(&CugraphPlanDefinition {
            spec: logical_string_spec(algorithm).clone(),
            edge_schema: string_edge_schema(),
            predicate_schemas: Vec::new(),
            predicate_row_upper_bounds: [].to_vec(),
            result_options: CugraphResultPlanOptions::default().clone(),
            edge_normalize_inserted: false,
            edge_row_upper_bound: Some(3),
            edge_vertex_upper_bound: None,
        })
        .unwrap_or_else(|error| panic!("{} logical plan failed: {error}", algorithm.function_name()));
        let graph = plan
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_)))
            .expect("logical plan must contain the native graph node");

        assert_eq!(graph.output_schema().field_with_name("vertex").unwrap().data_type(), &DataType::Int64);
        assert_eq!(plan.output_schema().field_with_name("vertex").unwrap().data_type(), &DataType::Utf8);
        assert!(matches!(plan.root_node().kind, PlanNode::Join(_)));
    }
}

/// MST and spectral modularity maximization run on libcugraph's legacy Int32
/// graph types. Int64 and string vertices reach them through a codebook whose
/// logical column keeps the input vertex type and whose native ids are Int32.
#[test]
fn legacy_int32_algorithms_map_int64_and_string_vertices_through_an_int32_codebook() {
    for algorithm in [CugraphAlgorithm::MinimumSpanningTree, CugraphAlgorithm::SpectralModularityMaximization] {
        let params = CugraphAlgorithmParams::from_native_defaults(algorithm, None).expect("native defaults are total");
        let graph_options = algorithm.effective_default_graph_options(&CugraphGraphOptions::default());
        let has_weight = algorithm == CugraphAlgorithm::MinimumSpanningTree;

        for (domain, vertex_type) in [
            (CugraphVertexIdDomain::Int32, DataType::Int32),
            (CugraphVertexIdDomain::Int64, DataType::Int64),
            (CugraphVertexIdDomain::StringLogical, DataType::Utf8),
        ] {
            let mut fields =
                vec![Field::new("src", vertex_type.clone(), false), Field::new("dst", vertex_type.clone(), false)];
            if has_weight {
                fields.push(Field::new("weight", DataType::Float64, false));
            }
            let definition = CugraphPlanDefinition {
                spec: CugraphAlgorithmSpec {
                    algorithm,
                    has_weight,
                    has_edge_id: false,
                    graph_options: graph_options.clone(),
                    params: params.clone(),
                    output_schema: result_schema_for_vertex_domain(algorithm, &params, Some(&domain)),
                },
                edge_schema: Arc::new(Schema::new(fields)),
                predicate_schemas: Vec::new(),
                predicate_row_upper_bounds: Vec::new(),
                result_options: CugraphResultPlanOptions::default(),
                edge_normalize_inserted: false,
                edge_row_upper_bound: Some(3),
                edge_vertex_upper_bound: None,
            };
            let case = format!("{} over {vertex_type:?} vertices", algorithm.function_name());

            preflight_cugraph_query_plan(&definition).unwrap_or_else(|error| panic!("{case} preflight: {error}"));
            let plan = cugraph_schema_query_plan(&definition).unwrap_or_else(|error| panic!("{case} plan: {error}"));
            let graph = plan
                .nodes()
                .iter()
                .find(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_)))
                .unwrap_or_else(|| panic!("{case} must contain the native graph node"));
            let codebook_row_number = plan.nodes().iter().find_map(|node| match &node.kind {
                PlanNode::Window(window) => window.spec.functions.iter().find_map(|function| match function {
                    WindowFunction::RowNumber { output } => Some(output.data_type().clone()),
                    _ => None,
                }),
                _ => None,
            });

            for column in algorithm.definition().vertex_output_columns() {
                assert_eq!(
                    graph.output_schema().field_with_name(column.name).unwrap().data_type(),
                    &DataType::Int32,
                    "{case}: native {}",
                    column.name,
                );
                assert_eq!(
                    plan.output_schema().field_with_name(column.name).unwrap().data_type(),
                    &vertex_type,
                    "{case}: decoded {}",
                    column.name,
                );
            }
            if domain == CugraphVertexIdDomain::Int32 {
                assert_eq!(codebook_row_number, None, "{case} needs no codebook");
                assert!(matches!(plan.root_node().kind, PlanNode::GraphAlgorithm(_)), "{case}");
            } else {
                assert_eq!(codebook_row_number, Some(DataType::Int32), "{case} numbers its codebook as Int32");
                assert!(matches!(plan.root_node().kind, PlanNode::Join(_)), "{case} decodes through a join");
            }
        }
    }
}

#[test]
fn m3_string_selectors_lower_to_encoded_native_side_inputs_and_decode_identity_outputs() {
    let mut cases = Vec::new();

    let mut sssp = logical_string_spec(CugraphAlgorithm::Sssp);
    let CugraphAlgorithmParams::Sssp(params) = &mut sssp.params else { unreachable!() };
    params.source_vertex = CugraphVertexSelector::Utf8("alice".to_string());
    cases.push((sssp, vec![("vertex", false), ("predecessor", true)]));

    let mut vertex_betweenness = logical_string_spec(CugraphAlgorithm::BetweennessCentrality);
    let CugraphAlgorithmParams::BetweennessCentrality(params) = &mut vertex_betweenness.params else { unreachable!() };
    params.seeds = Some(CugraphBetweennessSeeds {
        vertices: vec![
            CugraphVertexSelector::Utf8("alice".to_string()),
            CugraphVertexSelector::Utf8("carol".to_string()),
        ],
    });
    cases.push((vertex_betweenness, vec![("vertex", false)]));

    let mut edge_betweenness = logical_string_spec(CugraphAlgorithm::EdgeBetweennessCentrality);
    let CugraphAlgorithmParams::EdgeBetweennessCentrality(params) = &mut edge_betweenness.params else {
        unreachable!()
    };
    params.seeds = Some(CugraphBetweennessSeeds { vertices: vec![CugraphVertexSelector::Utf8("alice".to_string())] });
    cases.push((edge_betweenness, vec![("src", false), ("dst", false)]));

    for (spec, decoded_columns) in cases {
        let algorithm = spec.algorithm;
        let plan = cugraph_schema_query_plan(&CugraphPlanDefinition {
            spec: spec.clone(),
            edge_schema: string_edge_schema(),
            predicate_schemas: Vec::new(),
            predicate_row_upper_bounds: [].to_vec(),
            result_options: CugraphResultPlanOptions::default().clone(),
            edge_normalize_inserted: false,
            edge_row_upper_bound: Some(3),
            edge_vertex_upper_bound: None,
        })
        .unwrap_or_else(|error| panic!("{} logical selector plan failed: {error}", algorithm.function_name()));
        let graph = plan
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                PlanNode::GraphAlgorithm(graph) => Some(graph),
                _ => None,
            })
            .expect("logical selector plan must contain a native graph node");

        assert_eq!(graph.spec.params.side_input_ordinals().count(), 1, "{}", algorithm.function_name());
        for (column, nullable) in decoded_columns {
            assert_eq!(
                plan.output_schema().field_with_name(column).unwrap(),
                &Field::new(column, DataType::Utf8, nullable),
                "{}",
                algorithm.function_name()
            );
        }
    }
}

#[test]
fn string_pagerank_subsumes_only_bounded_sorts() {
    let producer = CugraphAlgoExec::try_new(
        pagerank_utf8_spec(),
        memory_string_edge_source(),
        vec![],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("string PageRank producer should build");
    let consumer_plan = |with_limit| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            &producer.gpu_result_input_name,
            producer.schema(),
        )));
        let sort =
            builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)])).unwrap();
        let root = if with_limit { builder.add_limit(sort, LimitSpec::fetch(2)).unwrap() } else { sort };
        builder.finish_gpu_resident(root).unwrap()
    };

    assert!(
        node::pushed_logical_consumer_options(&producer, &consumer_plan(false)).unwrap().is_none(),
        "an unbounded string sort must stay in the ordinary composed consumer",
    );
    assert!(
        node::pushed_logical_consumer_options(&producer, &consumer_plan(true)).unwrap().is_some(),
        "a bounded string top-k may be restored after decode by the producer",
    );
}

#[test]
fn string_pagerank_bfs_and_bfs_path_are_gpu_producers_with_unique_key_contracts() {
    let pagerank = CugraphAlgoExec::try_new(
        pagerank_utf8_spec(),
        memory_string_edge_source(),
        vec![],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("string PageRank should build as a GPU-resident graph execution");
    assert!(gpu_resident_producer(&pagerank).is_some());

    let logical = CugraphAlgoExec::try_new(
        bfs_utf8_spec(),
        memory_string_edge_source(),
        vec![],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("logical BFS should build as a GPU-resident graph execution");
    assert!(gpu_resident_producer(&logical).is_some());

    let numeric_path = CugraphAlgoExec::try_new(
        bfs_utf8_path_spec(),
        memory_string_edge_source(),
        vec![memory_string_path_target_source()],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("logical BFS path should build as a GPU-resident graph execution");
    assert!(gpu_resident_producer(&numeric_path).is_some());
    assert_eq!(
        gpu_resident_producer_unique_output_key(&numeric_path),
        Some(std::collections::BTreeSet::from(["path_index".to_string()])),
    );
}

#[test]
fn malformed_logical_bfs_mapping_fails_closed_at_gpu_boundaries() {
    let mut spec = bfs_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new());
    let source_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let source_exec = Arc::new(EmptyExec::new(Arc::clone(&source_schema))) as Arc<dyn ExecutionPlan>;
    let mut exec = CugraphAlgoExec::try_new(
        spec,
        memory_edge_source(4),
        vec![source_exec],
        None,
        None,
        AlgeonNativeExecutionConfig::default(),
    )
    .expect("numeric relation-backed BFS should build before schema corruption");
    let malformed_schema = Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false)]));
    exec.edge_source = Arc::new(EmptyExec::new(malformed_schema)) as Arc<dyn ExecutionPlan>;
    let plan = &exec as &dyn ExecutionPlan;

    assert!(
        gpu_resident_producer(plan).is_none(),
        "mapping validation errors must not classify a malformed BFS as a GPU producer"
    );
    let rejection = GraphExecutionPlan::gpu_resident_output_rejection(&exec)
        .expect("mapping validation errors must be surfaced at the GPU boundary");
    assert_eq!(rejection.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(rejection.operation(), "algeon_datafusion.cugraph_sql.plan");
    assert!(rejection.message().contains("dst"), "the missing mapping column must remain actionable");
}

#[tokio::test]
async fn cugraph_producer_preparation_stages_host_inputs_before_gpu_execution() {
    let backend = test_gpu_backend();
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let (_attempt, task_context) =
        attempt_task_context(&backend, TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let exec = Arc::new(
        CugraphAlgoExec::try_new(
            triangle_spec(),
            memory_edge_source(16),
            vec![],
            None,
            None,
            AlgeonNativeExecutionConfig::default(),
        )
        .expect("cuGraph producer should build"),
    );
    let plan = Arc::clone(&exec) as Arc<dyn ExecutionPlan>;

    let prepared = exec
        .prepare_execution_plan(plan, task_context)
        .await
        .expect("host inputs should stage during producer preparation");

    assert!(
        prepared
            .downcast_ref::<CugraphAlgoExec>()
            .expect("prepared plan should remain cuGraph")
            .prepared_staged_inputs
            .is_some()
    );
    assert!(runtime.memory_pool.reserved() > 0, "staged host batches must retain their reservation");
    drop(prepared);
    assert_eq!(runtime.memory_pool.reserved(), 0, "dropping staged inputs must release their reservation");
}

#[tokio::test]
async fn cugraph_staged_attempt_uses_exact_filtered_host_rows_once() {
    let backend = test_gpu_backend();
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(admission_source(&backend));
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(16 * 1024 * 1024, 1.0).build().unwrap());
    let (_attempt, task_context) =
        attempt_task_context(&backend, TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let edge_source = memory_edge_source(1_000);
    let schema = edge_source.schema();
    let predicate =
        binary(col("src", &schema).unwrap(), Operator::Gt, lit(ScalarValue::Int64(Some(500))), &schema).unwrap();
    let filtered = FilterExec::try_new(predicate, edge_source).unwrap();
    let execute_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let edge_source =
        Arc::new(CountingExec::new(Arc::new(filtered), Arc::clone(&execute_count))) as Arc<dyn ExecutionPlan>;
    let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
        .expect("filtered MemTable edge source should remain a valid graph plan");

    let (prepared, staged) = exec
        .prepare_staged_attempt(Arc::clone(&task_context))
        .await
        .expect("host staging should derive the exact filtered row count before admission");

    assert_eq!(execute_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(staged.row_bounds(&ParquetReadStrategyConfig::default()).unwrap().edge_row_upper_bound(), Some(499));
    assert_eq!(prepared.input_row_bounds.edge_row_upper_bound(), Some(499));
    assert!(
        prepared.native_execution_config.query_admission_ticket().is_some(),
        "the graph ticket must be minted only after staged exact rows are available"
    );
    assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 0);
    drop(staged);
    assert_eq!(runtime.memory_pool.reserved(), 0, "dropping staged host ownership must release its reservation");
}

#[tokio::test]
async fn cugraph_staged_attempt_preserves_host_edge_and_predicate_row_order() {
    let backend = test_gpu_backend();
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(admission_source(&backend));
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(16 * 1024 * 1024, 1.0).build().unwrap());
    let (_attempt, task_context) =
        attempt_task_context(&backend, TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("fixture is BFS") };
    params.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new());
    let source_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let source_batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![Arc::new(StringArray::from(vec!["alice", "bob", "carol"])) as ArrayRef],
    )
    .unwrap();
    let source_plan =
        MemorySourceConfig::try_new_exec(&[vec![source_batch.clone()]], Arc::clone(&source_schema), None).unwrap();
    let edge_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let predicate_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let edge_source =
        Arc::new(CountingExec::new(memory_string_edge_source(), Arc::clone(&edge_count))) as Arc<dyn ExecutionPlan>;
    let predicate_source =
        Arc::new(CountingExec::new(source_plan, Arc::clone(&predicate_count))) as Arc<dyn ExecutionPlan>;
    let exec = CugraphAlgoExec::try_new(spec, edge_source, vec![predicate_source], None, None, config)
        .expect("logical BFS with a host source relation should build");
    let input_limit_bytes = exec.host_boundary_config.max_input_staging_bytes();
    let output_limit_bytes = exec.host_boundary_config.max_output_staging_bytes();

    let (prepared, staged) = exec
        .prepare_staged_attempt(Arc::clone(&task_context))
        .await
        .expect("host edge and source relation should stage before the graph ticket");
    let bounds = staged.row_bounds(&ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(bounds.edge_row_upper_bound(), Some(3));
    assert_eq!(bounds.predicate_row_upper_bounds(), &[Some(3)]);
    assert_eq!(prepared.input_row_bounds.edge_row_upper_bound(), Some(3));
    assert_eq!(prepared.input_row_bounds.predicate_row_upper_bounds(), &[Some(3)]);
    assert_eq!(edge_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(predicate_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(staged.host_evidence.input_limit_bytes(), input_limit_bytes);
    assert_eq!(staged.host_evidence.input_staging_peak_bytes(), staged.reservation.as_ref().unwrap().size());
    assert_eq!(staged.host_evidence.output_limit_bytes(), output_limit_bytes);
    assert_eq!(
        staged.host_evidence.aggregate_peak_bytes(),
        staged.host_evidence.input_staging_peak_bytes() + staged.host_evidence.output_limit_bytes(),
        "host evidence must carry one checked aggregate peak for overlapping input and output reservations",
    );
    assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 0);
    drop(staged);
    assert_eq!(runtime.memory_pool.reserved(), 0, "staged edge and predicate ownership must be released together");
}

#[tokio::test]
async fn cugraph_native_staging_retains_hard_rows_and_never_executes_a_wrapped_child() {
    let config = AlgeonNativeExecutionConfig::default();
    let native = native_edge_source(vec![100, 200], &config);
    let staged = stage_cugraph_inputs(
        &triangle_spec(),
        Arc::clone(&native),
        vec![],
        Arc::new(TaskContext::default()),
        &ExecutionCancellation::new(),
        1024 * 1024,
    )
    .await
    .expect("a native edge source remains a native staging owner");
    assert!(matches!(&staged.edge, StagedCugraphSource::Native { .. }));
    assert_eq!(staged.row_bounds(&ParquetReadStrategyConfig::default()).unwrap().edge_row_upper_bound(), Some(300));
    drop(staged);

    let execute_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wrapped = Arc::new(CountingExec::new(native, Arc::clone(&execute_count))) as Arc<dyn ExecutionPlan>;
    let error = stage_cugraph_inputs(
        &triangle_spec(),
        wrapped,
        vec![],
        Arc::new(TaskContext::default()),
        &ExecutionCancellation::new(),
        1024 * 1024,
    )
    .await
    .expect_err("a CPU wrapper must not hide a native child from the host staging boundary");
    let algeon = error.find_algeon_error().expect("wrapped native rejection remains structured");
    assert_eq!(algeon.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(execute_count.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cugraph_staged_attempt_failure_cleans_host_reservation_before_admission() {
    let backend = test_gpu_backend();
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(admission_source(&backend));
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(16 * 1024 * 1024, 1.0).build().unwrap());
    let (_attempt, task_context) =
        attempt_task_context(&backend, TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let boundary = GpuIslandHostBoundaryConfig::try_new(1, 1024 * 1024).unwrap();
    let exec = CugraphAlgoExec::try_new_with_host_boundary_config(
        triangle_spec(),
        memory_edge_source(16),
        vec![],
        None,
        None,
        config,
        boundary,
    )
    .expect("the graph should build before its deliberately tiny host cap is applied");

    let error = exec
        .prepare_staged_attempt(task_context)
        .await
        .expect_err("host staging must fail before an admission ticket or query is acquired");
    let algeon = error.find_algeon_error().expect("host staging cap failure remains structured");
    assert_eq!(algeon.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 0);
    assert_eq!(runtime.memory_pool.reserved(), 0, "failed staging must release all host ownership");
}

#[tokio::test]
async fn cugraph_producer_preparation_binds_one_post_hydration_ticket_for_host_and_gpu_paths() {
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source);
    let (_attempt, task_context) = attempt_task_context(&backend, TaskContext::default());
    let exec = Arc::new(
        CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(16), vec![], None, None, config)
            .expect("cuGraph producer should build without a pre-hydration ticket"),
    );

    assert_eq!(exec.native_execution_config.query_admission_ticket_id(), None);

    let prepared = exec
        .prepare_execution_plan(Arc::clone(&exec) as Arc<dyn ExecutionPlan>, Arc::clone(&task_context))
        .await
        .expect("producer preparation should hydrate metadata and bind its invocation's ticket");
    let prepared_exec = prepared.downcast_ref::<CugraphAlgoExec>().expect("prepared plan should remain cuGraph");
    let ticket_id = prepared_exec
        .native_execution_config
        .query_admission_ticket_id()
        .expect("post-hydration cuGraph producer must carry its graph-aware ticket");
    assert!(graph_execution_plan(prepared.as_ref()).is_some());
    let prepared_ticket = prepared_exec
        .native_execution_config
        .query_admission_ticket()
        .expect("prepared cuGraph occurrence must retain its graph-aware ticket");
    assert!(
        prepared_ticket.attempt_capability_evidence().is_some(),
        "the prepared occurrence must retain one opaque graph evidence for composed admission",
    );

    let host_task_context =
        task_context_with_cugraph_execution_ticket(&prepared_exec.native_execution_config, Arc::clone(&task_context));
    assert_eq!(
        query_admission_ticket_from_task_context(&host_task_context).map(|ticket| ticket.id()),
        Some(ticket_id),
        "the host execution boundary must promote the prepared producer ticket",
    );
    assert_eq!(
        prepared_exec.native_execution_config.query_admission_ticket_id_for_task_context(host_task_context.as_ref()),
        Some(ticket_id),
        "the GPU producer context must resolve the same post-hydration ticket",
    );
}

#[tokio::test]
async fn cugraph_producer_prepared_inputs_transfer_once_across_plan_clones() {
    let backend = test_gpu_backend();
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let (_attempt, task_context) =
        attempt_task_context(&backend, TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let exec = Arc::new(
        CugraphAlgoExec::try_new(
            triangle_spec(),
            memory_edge_source(16),
            vec![],
            None,
            None,
            AlgeonNativeExecutionConfig::default(),
        )
        .expect("cuGraph producer should build"),
    );
    let plan = Arc::clone(&exec) as Arc<dyn ExecutionPlan>;
    let prepared = exec
        .prepare_execution_plan(plan, task_context)
        .await
        .expect("host inputs should stage during producer preparation");
    let prepared_exec = prepared.downcast_ref::<CugraphAlgoExec>().expect("prepared plan should remain cuGraph");
    let prepared_clone = prepared_exec.clone();

    let staged_inputs = prepared_exec.take_prepared_staged_inputs().expect("first plan clone owns the staged inputs");
    assert!(runtime.memory_pool.reserved() > 0, "transferred staged inputs retain the reservation");
    let error = prepared_clone
        .take_prepared_staged_inputs()
        .expect_err("the shared staged-input slot must be consumed exactly once");

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("staged_input_state"), Some("already_consumed"));
    drop(prepared_clone);
    drop(prepared);
    assert!(runtime.memory_pool.reserved() > 0, "only the transferred owner retains the reservation");
    drop(staged_inputs);
    assert_eq!(runtime.memory_pool.reserved(), 0, "dropping the sole staged-input owner releases the reservation");
}

#[tokio::test]
async fn cugraph_host_staging_polls_every_partition_of_a_dynamic_filter_join() {
    let dir = tempfile::tempdir().unwrap();
    for file in ["a.parquet", "b.parquet", "c.parquet"] {
        write_edge_parquet(&dir.path().join(file));
    }
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_parquet("edges", dir.path().to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let keys = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![0, 1, 2])) as ArrayRef],
    )
    .unwrap();
    ctx.register_batch("keys", keys).unwrap();
    let edge_source = ctx
        .sql("SELECT e.src, e.dst, e.weight FROM edges e JOIN keys k ON k.id = e.src")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let rendered = displayable(edge_source.as_ref()).indent(true).to_string();
    assert!(rendered.contains("mode=CollectLeft") && rendered.contains("DynamicFilter"), "{rendered}");
    assert_eq!(edge_source.properties().output_partitioning().partition_count(), 3, "{rendered}");

    // Each probe partition of the join waits until every probe partition has reported its
    // build bounds, so staging that drains partitions one at a time never completes.
    let staged = timeout(
        std::time::Duration::from_secs(30),
        stage_cugraph_inputs(
            &pagerank_spec(),
            edge_source,
            vec![],
            Arc::new(TaskContext::default()),
            &ExecutionCancellation::new(),
            1024 * 1024,
        ),
    )
    .await
    .expect("host staging must poll every probe partition of the dynamic-filter join")
    .expect("host graph inputs should stage");
    let StagedCugraphSource::Host { batches, .. } = &staged.edge else {
        panic!("a DataFusion join edge source stages on the host")
    };
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 9);
}

#[tokio::test]
async fn cugraph_input_upload_releases_the_host_reservation_before_graph_compute() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let native_context = ExecutionContext::standalone(StandaloneConfig {
        runtime: Some(Arc::new(cudf::Runtime::new().expect("test CUDA runtime"))),
        ..StandaloneConfig::default()
    })
    .unwrap();
    let staged_inputs = stage_cugraph_inputs(
        &triangle_spec(),
        memory_edge_source(16),
        vec![],
        Arc::clone(&task_context),
        &ExecutionCancellation::new(),
        1024 * 1024,
    )
    .await
    .expect("host graph inputs should stage");
    assert!(runtime.memory_pool.reserved() > 0, "staged host batches must be reserved before upload");

    let uploaded =
        staged_inputs.upload(&native_context, task_context).await.expect("staged host graph inputs should upload");

    assert_eq!(
        runtime.memory_pool.reserved(),
        0,
        "host-input reservation must be released before graph compute receives uploaded inputs",
    );
    drop(uploaded);
}

#[tokio::test]
async fn cugraph_staged_upload_rejects_missing_admission_plan_before_source_upload() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let backend = test_gpu_backend();
    let attempt = backend.admit_query().expect("test owns the backend admission slot");
    let admitted = attempt.acquire_context().expect("admitted context");
    let native_context = admitted.context();
    assert!(native_context.is_query_service_admitted());
    let staged_inputs = stage_cugraph_inputs(
        &triangle_spec(),
        memory_edge_source(16),
        vec![],
        Arc::clone(&task_context),
        &ExecutionCancellation::new(),
        1024 * 1024,
    )
    .await
    .expect("host graph inputs should stage");
    assert!(staged_inputs.admitted_graph_plan.is_none());

    let error = match staged_inputs.upload(native_context, task_context).await {
        Err(error) => error,
        Ok(_) => panic!("an admitted upload without its exact graph plan must fail closed"),
    };
    let algeon = error.find_algeon_error().expect("missing admission plan remains structured");
    assert_eq!(algeon.fact_value("reason"), Some("admission_graph_plan_missing"));
}

fn direct_composed_consumer_plan(input_names: &[&str], schema: Arc<Schema>) -> QueryPlan {
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

fn direct_composed_graph_evidence() -> (Arc<Schema>, AttemptCapabilityEvidence) {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/direct-composed-graph.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(source, Vec::new(), spec.clone()).expect("direct graph fixture must lower");
    let graph_plan = builder.finish_gpu_resident(graph).unwrap();
    let mut evidence = AttemptCapabilityEvidence::for_query_plan(&graph_plan);
    evidence.attest_cugraph_graph_execution_scope(&graph_plan);
    (spec.output_schema, evidence)
}

fn direct_composed_admission_context_config(
    backend: &InstalledTestGpuBackend,
    consumer_plan: &QueryPlan,
    children: &[Arc<dyn ExecutionPlan>],
) -> (AlgeonAttempt, AlgeonNativeExecutionConfig) {
    let evidence = composed_admission_capability_evidence(consumer_plan, children)
        .expect("direct composed fixture must issue one complete admission evidence");
    let admitted = backend
        .backend()
        .session_capability()
        .admit_query_with_request(
            QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence.clone()),
        )
        .expect("direct composed evidence must be admitted by the installed backend profile");
    drop(admitted);
    let (attempt, ticket) = attempt_ticket_for_test(admission_source(backend), Some(evidence));
    (attempt, backend_execution_config(backend).with_existing_query_admission_ticket(ticket))
}

#[tokio::test]
async fn composed_edge_source_rejects_admission_mismatch_before_running_producers() {
    let backend = test_gpu_backend();
    let admission_source = admission_source(&backend);
    let (schema, producer_evidence) = direct_composed_graph_evidence();
    let consumer_gpu_plan = direct_composed_consumer_plan(&["producer_a", "producer_b"], Arc::clone(&schema))
        .clone_with_gpu_resident_sink();
    let execute_counts = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
    let producer_names = ["producer_a", "producer_b"];
    let (_producer_attempt, producer_ticket) =
        attempt_ticket_for_test(admission_source.clone(), Some(producer_evidence.clone()));
    let children = producer_names
        .iter()
        .zip(&execute_counts)
        .map(|(input_name, execute_count)| {
            Arc::new(CountingGpuResidentProducer {
                input_name: (*input_name).to_string(),
                ticket: Arc::clone(&producer_ticket),
                capability_evidence: producer_evidence.clone(),
                execute_count: Arc::clone(execute_count),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }) as Arc<dyn ExecutionPlan>
        })
        .collect::<Vec<_>>();
    // The edge source as planning hands it to the graph: a composed producer
    // that holds its admitted GPU-resident artifact.
    let composed_edge_source = |consumer_plan: &QueryPlan, task_context: Arc<TaskContext>| {
        let admitted =
            admit_query_plan(consumer_plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default())
                .expect("composed edge-source fixture must be admissible");
        let producer = Arc::new(
            AlgeonNativeExec::try_new(
                admitted,
                children.clone(),
                NativeOutput::GpuResident { input_name: "composed_edge".to_string() },
                backend_execution_config(&backend),
                Some(EquivalenceProperties::new(Arc::clone(consumer_plan.output_schema()))),
            )
            .expect("composed edge-source fixture must build"),
        ) as Arc<dyn ExecutionPlan>;
        prepare_gpu_resident_producer_for_execution(producer, task_context)
    };

    let wrong_order_children = vec![Arc::clone(&children[1]), Arc::clone(&children[0])];
    let (_wrong_order_attempt, wrong_order_config) =
        direct_composed_admission_context_config(&backend, &consumer_gpu_plan, &wrong_order_children);
    let wrong_order_task_context = Arc::new(TaskContext::default());
    let wrong_order_context = wrong_order_config
        .execution_context_async(Arc::clone(&wrong_order_task_context))
        .await
        .expect("wrong-order fixture must acquire an admitted context");
    let edge_source = composed_edge_source(&consumer_gpu_plan, Arc::clone(&wrong_order_task_context))
        .await
        .expect("composed edge source prepares");
    let wrong_order_error =
        match execute_edge_source_to_gpu(wrong_order_context.context(), edge_source, wrong_order_task_context).await {
            Err(error) => error,
            Ok(_) => panic!("runtime producer order must not execute under a differently ordered admission evidence"),
        };
    let wrong_order_algeon = wrong_order_error.find_algeon_error().expect("order rejection must remain structured");
    assert_eq!(
        wrong_order_algeon.fact_value("schedule_reason"),
        Some("runtime_occurrence_order_mismatch"),
        "{wrong_order_algeon:#?}"
    );
    assert_eq!(execute_counts[0].load(Ordering::SeqCst), 0);
    assert_eq!(execute_counts[1].load(Ordering::SeqCst), 0);
    drop(wrong_order_context);

    let (_original_attempt, original_config) =
        direct_composed_admission_context_config(&backend, &consumer_gpu_plan, &children);
    let original_task_context = Arc::new(TaskContext::default());
    let original_context = original_config
        .execution_context_async(Arc::clone(&original_task_context))
        .await
        .expect("original fixture must acquire an admitted context");
    let changed_consumer_gpu_plan = consumer_gpu_plan
        .with_appended_sort(SortSpec::new(vec![SortExpr::new(Expr::column("vertex"), false, false)]))
        .clone_with_gpu_resident_sink();
    let edge_source = composed_edge_source(&changed_consumer_gpu_plan, Arc::clone(&original_task_context))
        .await
        .expect("changed-scope composed edge source prepares");
    let changed_scope_error =
        match execute_edge_source_to_gpu(original_context.context(), edge_source, original_task_context).await {
            Err(error) => error,
            Ok(_) => panic!("a changed consumer scope must not execute admitted producers"),
        };
    let changed_scope_algeon = changed_scope_error.find_algeon_error().expect("scope rejection must remain structured");
    assert_eq!(changed_scope_algeon.fact_value("reason"), Some("admitted_capability_scope_mismatch"));
    assert_eq!(execute_counts[0].load(Ordering::SeqCst), 0);
    assert_eq!(execute_counts[1].load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cugraph_host_staging_completes_before_backend_gpu_admission() {
    let backend = test_gpu_backend();
    // One held admission occupies the single per-device attempt slot; the
    // exec's admission queues behind it while staging runs.
    let held_queries = [backend.admit_query().expect("test owns the backend admission slot")];
    let execute_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let edge_source = Arc::new(CountingExec::new(memory_edge_source(16), Arc::clone(&execute_count)));
    let config = backend_execution_config(&backend);
    let exec = Arc::new(
        CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build"),
    );
    let (_attempt, task_context) = attempt_task_context(&backend, TaskContext::default());
    let mut stream = exec.execute(0, task_context).expect("cuGraph stream should build");
    let execution = tokio::spawn(async move { stream.try_next().await });

    timeout(std::time::Duration::from_secs(5), async {
        while execute_count.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            yield_now().await;
        }
    })
    .await
    .expect("host staging must run while the backend GPU admission slots are occupied");

    assert_eq!(execute_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(!execution.is_finished(), "execution must wait for backend admission only after staging completes");
    execution.abort();
    assert!(execution.await.expect_err("aborted execution should not complete").is_cancelled());
    drop(held_queries);
}

#[tokio::test]
async fn cugraph_producer_preparation_enforces_the_cumulative_input_cap() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let edge_source = memory_edge_source(16);
    let edge_batches = collect(Arc::clone(&edge_source), Arc::clone(&task_context)).await.unwrap();
    let edge_bytes = edge_batches.iter().map(get_record_batch_memory_size).sum();
    drop(edge_batches);
    let spec = triangle_spec();

    let error = stage_cugraph_inputs(
        &spec,
        edge_source,
        vec![memory_edge_source(1)],
        task_context,
        &ExecutionCancellation::new(),
        edge_bytes,
    )
    .await
    .expect_err("edge plus side input larger than the configured cumulative cap must be rejected");
    let algeon = error.find_algeon_error().expect("staging cap failure must remain structured");

    assert_eq!(algeon.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(algeon.status(), ErrorStatus::Permanent);
    assert_eq!(algeon.fact_value("source_kind"), Some("side_input"));
    let edge_bytes = edge_bytes.to_string();
    assert_eq!(algeon.fact_value("staged_bytes"), Some(edge_bytes.as_str()));
    assert_eq!(runtime.memory_pool.reserved(), 0, "failed cumulative staging must release its reservation");
}

#[tokio::test]
async fn cugraph_producer_preparation_counts_shared_buffers_once_across_batches_and_partitions() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let batch = collect(memory_edge_source(16), Arc::clone(&task_context)).await.unwrap().remove(0);
    let retained_bytes = get_record_batch_memory_size(&batch);
    let first = batch.slice(0, 8);
    let second = batch.slice(8, 8);
    for partitions in [vec![vec![first.clone(), second.clone()]], vec![vec![first.clone()], vec![second.clone()]]] {
        let source = MemorySourceConfig::try_new_exec(&partitions, batch.schema(), None).unwrap();
        let staged = stage_cugraph_inputs(
            &triangle_spec(),
            source,
            vec![],
            Arc::clone(&task_context),
            &ExecutionCancellation::new(),
            retained_bytes,
        )
        .await
        .expect("sliced batches must fit the limit for their one shared allocation");
        assert_eq!(staged.row_bounds(&ParquetReadStrategyConfig::default()).unwrap().edge_row_upper_bound(), Some(16));
        assert_eq!(runtime.memory_pool.reserved(), retained_bytes);
        drop(staged);
        assert_eq!(runtime.memory_pool.reserved(), 0);
    }
}

#[tokio::test]
async fn cugraph_producer_preparation_keeps_utf8_literal_sources_out_of_host_staging() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let spec = bfs_utf8_spec();
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;

    let staged = stage_cugraph_inputs(&spec, edge_source, vec![], task_context, &ExecutionCancellation::new(), 0)
        .await
        .expect("UTF-8 literal source should be constructed inside the native GPU plan");

    assert_eq!(staged.row_bounds(&ParquetReadStrategyConfig::default()).unwrap().predicate_row_upper_bounds(), &[]);
    assert_eq!(runtime.memory_pool.reserved(), 0);
}

#[tokio::test]
async fn cugraph_producer_preparation_charges_utf8_relation_sources_to_the_same_input_cap() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new());
    let source_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let source_batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef],
    )
    .unwrap();
    let source_bytes = get_record_batch_memory_size(&source_batch);
    let max_staging_bytes = source_bytes.checked_sub(1).expect("UTF-8 relation batch must retain host bytes");
    let source_exec = MemorySourceConfig::try_new_exec(&[vec![source_batch]], source_schema, None).unwrap();
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;

    let error = stage_cugraph_inputs(
        &spec,
        edge_source,
        vec![source_exec],
        task_context,
        &ExecutionCancellation::new(),
        max_staging_bytes,
    )
    .await
    .expect_err("UTF-8 relation source larger than the island-wide cap must be rejected");
    let algeon = error.find_algeon_error().expect("relation source staging cap failure must remain structured");
    let source_bytes = source_bytes.to_string();
    let max_staging_bytes = max_staging_bytes.to_string();

    assert_eq!(algeon.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(algeon.fact_value("source_kind"), Some("side_input"));
    assert_eq!(algeon.fact_value("staged_bytes"), Some("0"));
    assert_eq!(algeon.fact_value("next_batch_bytes"), Some(source_bytes.as_str()));
    assert_eq!(algeon.fact_value("max_staging_bytes"), Some(max_staging_bytes.as_str()));
    assert_eq!(runtime.memory_pool.reserved(), 0, "failed relation staging must release its reservation");
}

fn singleflight_test_exec_with_options(
    source_vertex: i64,
    depth_limit: Option<i64>,
    construction_policy: CugraphGraphConstructionPolicy,
    graph_stable_id: &str,
    output_mode: CugraphBfsOutputMode,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
) -> CugraphAlgoExec {
    let backend = test_gpu_backend();
    let config = backend_execution_config(&backend);
    let edge_source = native_graph_residency_edge_source_with_stable_id(&config, graph_stable_id);
    let mut spec = bfs_spec();
    spec.graph_options.construction_policy = construction_policy;
    {
        let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("BFS test spec") };
        params.source = CugraphBfsSource::scalar_i64(source_vertex);
        params.depth_limit = depth_limit;
        params.output.mode = output_mode;
    }
    spec.output_schema = result_schema_for_params(spec.algorithm, &spec.params);
    CugraphAlgoExec::try_new(spec, edge_source, vec![], projection, limit, config).unwrap()
}

fn singleflight_test_exec(source_vertex: i64, depth_limit: Option<i64>) -> CugraphAlgoExec {
    singleflight_test_exec_with_options(
        source_vertex,
        depth_limit,
        CugraphGraphConstructionPolicy::RawLibcugraph,
        "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
        CugraphBfsOutputMode::Normalized,
        None,
        None,
    )
}

#[test]
fn bfs_singleflight_eligibility_is_exact_and_rejects_bounded_depth() {
    let first = singleflight_test_exec(1, None).bfs_singleflight_key(0).expect("deep scalar BFS should qualify");
    let same = singleflight_test_exec(1, None).bfs_singleflight_key(0).expect("same deep BFS should qualify");
    let other_source =
        singleflight_test_exec(2, None).bfs_singleflight_key(0).expect("other scalar BFS should qualify");
    let python_noop_graph = singleflight_test_exec_with_options(
        1,
        None,
        CugraphGraphConstructionPolicy::PythonCugraph,
        "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
        CugraphBfsOutputMode::Normalized,
        None,
        None,
    )
    .bfs_singleflight_key(0)
    .expect("normalized scalar BFS should qualify");
    let other_graph = singleflight_test_exec_with_options(
        1,
        None,
        CugraphGraphConstructionPolicy::RawLibcugraph,
        "local:other-edges.parquet:1024:mtime=2026-07-13T00:00:00Z",
        CugraphBfsOutputMode::Normalized,
        None,
        None,
    )
    .bfs_singleflight_key(0)
    .expect("other resident graph should qualify");
    let raw_output = singleflight_test_exec_with_options(
        1,
        None,
        CugraphGraphConstructionPolicy::RawLibcugraph,
        "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
        CugraphBfsOutputMode::Raw,
        None,
        None,
    )
    .bfs_singleflight_key(0)
    .expect("raw-output scalar BFS should qualify");
    let projected = singleflight_test_exec_with_options(
        1,
        None,
        CugraphGraphConstructionPolicy::RawLibcugraph,
        "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
        CugraphBfsOutputMode::Normalized,
        Some(vec![0, 3]),
        None,
    )
    .bfs_singleflight_key(0)
    .expect("projected scalar BFS should qualify");
    let limited = singleflight_test_exec_with_options(
        1,
        None,
        CugraphGraphConstructionPolicy::RawLibcugraph,
        "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
        CugraphBfsOutputMode::Normalized,
        None,
        Some(17),
    )
    .bfs_singleflight_key(0)
    .expect("limited scalar BFS should qualify");

    assert_eq!(first, same);
    assert_ne!(first, other_source, "independent source semantics must never coalesce");
    assert_ne!(first, singleflight_test_exec(1, None).bfs_singleflight_key(1).unwrap());
    assert_ne!(first, other_graph, "different graph identities must never coalesce");
    assert_eq!(
        first, python_noop_graph,
        "a footer-proven simple directed edge source makes Python normalization a no-op, so it shares the raw graph key"
    );
    assert_ne!(first, raw_output, "raw and normalized BFS outputs must never coalesce");
    assert_ne!(first, projected, "different projections must never coalesce");
    assert_ne!(first, limited, "different limits must never coalesce");
    assert!(singleflight_test_exec(1, Some(3)).bfs_singleflight_key(0).is_none());
}

#[test]
fn bfs_singleflight_follower_metrics_identify_coalescing_not_a_graph_cache() {
    let exec = singleflight_test_exec(1, None);
    let key = exec.bfs_singleflight_key(0).expect("eligible BFS has a transient coalescing key");
    let batches = vec![RecordBatch::new_empty(Arc::new(Schema::empty()))];

    let context = ExecutionContext::standalone(StandaloneConfig::default()).expect("execution context builds");
    let metrics = AlgeonNativeMetrics::new(&exec.metrics, 0);
    let mut graph_metrics = GraphExecutionMetrics::new(&metrics, &context);
    exec.record_bfs_singleflight_gpu_follower_metrics(&graph_metrics, &key, &batches, Some(0), Duration::ZERO);
    exec.record_bfs_singleflight_gpu_result_follower_metrics(&graph_metrics, &key, 0, 0, Duration::ZERO);
    exec.record_bfs_singleflight_host_follower_metrics(&graph_metrics, &key, &batches, Duration::ZERO);
    graph_metrics.finish(Ok(())).expect("follower metrics publish");

    let recorded = NativeDataFusionExecutionMetrics::from_metrics(&exec.metrics.clone_inner());
    assert_eq!(recorded.graph_runtime_rows.len(), 3, "one metric row per follower boundary");
    assert!(recorded.graph_runtime_rows.iter().any(|row| {
        row.input_boundary == "singleflight_gpu_result"
            && row.output_boundary == "gpu_resident"
            && row.gpu_resident_input_handoffs == 1
    }));
    for row in recorded.graph_runtime_rows {
        assert!(row.algorithm_metadata.contains("bfs_singleflight=follower"), "{row:?}");
        assert!(!row.algorithm_metadata.contains("graph_cache="), "{row:?}");
    }
}

#[test]
fn cugraph_pre_hydration_exec_leaves_backend_ticketless() {
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let base = AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
        .with_query_admission_source(source);

    assert!(base.has_query_admission_authority());
    assert_eq!(base.query_admission_ticket_id(), None);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, base)
        .expect("cuGraph exec should build without a pre-hydration ticket");

    assert_eq!(exec.native_execution_config.query_admission_ticket_id(), None);
}

#[test]
fn cugraph_algo_exec_records_graph_residency_for_native_edge_source() {
    let config = AlgeonNativeExecutionConfig::default();
    let edge_source = native_graph_residency_edge_source(&config);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for native edge source");
    let descriptor = exec.graph_residency.as_ref().expect("native parquet edge source should be keyable");

    assert!(descriptor.source_fingerprint.as_str().starts_with("graph-source-v1-fnv64-"));
    assert_eq!(descriptor.vertex_data_type, DataType::Int64);
    assert_eq!(
        descriptor.weight_data_type,
        Some(DataType::Float64),
        "Python-compatible edge normalization must keep weight in the source fingerprint even for algorithms \
         that do not consume weights directly",
    );
}

#[test]
fn simple_directed_footer_omits_python_edge_normalization_through_native_edge_wrapper() {
    let config = AlgeonNativeExecutionConfig::default();
    let edge_source = native_graph_residency_edge_source(&config);
    let mut spec = bfs_spec();
    spec.graph_options.construction_policy = CugraphGraphConstructionPolicy::PythonCugraph;

    let exec = CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, config)
        .expect("native edge wrapper with a footer-proven topology should plan");

    assert!(
        !exec.edge_normalize_inserted(),
        "native lowering must retain the source-owned simple-directed evidence through its edge wrapper"
    );
}

#[tokio::test]
async fn cugraph_algo_exec_records_graph_residency_for_datafusion_parquet_edge_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let config = AlgeonNativeExecutionConfig::default();
    let edge_source = datafusion_parquet_edge_source(&path).await;

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for DataFusion local-Parquet edge source");
    let descriptor = exec.graph_residency.as_ref().expect("DataFusion local-Parquet edge source should be keyable");

    assert!(descriptor.source_fingerprint.as_str().starts_with("graph-source-v1-fnv64-"));
    assert_eq!(descriptor.vertex_data_type, DataType::Int64);
    assert_eq!(descriptor.weight_data_type, Some(DataType::Float64));
}

#[tokio::test]
async fn graph_residency_translates_parallel_datafusion_parquet_scan_without_order_contract() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let config = SessionConfig::new()
        .with_target_partitions(4)
        .with_repartition_file_scans(true)
        .with_repartition_file_min_size(0);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let edge_source =
        ctx.sql("SELECT src, dst, weight FROM edges").await.unwrap().create_physical_plan().await.unwrap();
    let source = edge_source
        .as_ref()
        .downcast_ref::<DataSourceExec>()
        .expect("parallel Parquet fixture should remain a direct DataFusion scan");
    let native_spec = native_graph_spec_for_residency(&pagerank_spec(), false);

    let descriptor = graph_residency_descriptor_for_datafusion_local_parquet_scan(
        source,
        &native_spec,
        &ParquetReadStrategyConfig::default(),
    )
    .expect("parallel local-Parquet scan should retain a stable graph rowset descriptor");

    assert!(descriptor.source_fingerprint.as_str().starts_with("graph-source-v1-fnv64-"));
    assert_eq!(descriptor.vertex_data_type, DataType::Int64);
    assert_eq!(descriptor.weight_data_type, Some(DataType::Float64));
    assert_eq!(
        exact_row_count_for_datafusion_local_parquet_scan(source, &ParquetReadStrategyConfig::default()),
        Some(3),
        "the cacheable local-Parquet bridge must retain the exact footer row count for graph admission"
    );
}

#[tokio::test]
async fn graph_vertex_footer_bound_is_read_from_the_complete_datafusion_parquet_scan() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet_with_graph_vertex_upper_bound(&path, Some(3));
    let edge_source = datafusion_parquet_edge_source(&path).await;
    let source = edge_source
        .as_ref()
        .downcast_ref::<DataSourceExec>()
        .expect("single-file fixture should remain a direct DataFusion scan");

    assert_eq!(
        graph_vertex_upper_bound_for_datafusion_local_parquet_scan(source, &ParquetReadStrategyConfig::default()),
        Some(3),
        "graph admission must consume the footer fact bound to the exact local-Parquet edge rowset"
    );
}

#[tokio::test]
async fn simple_directed_footer_omits_python_edge_normalization_only_for_the_proven_rowset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_simple_directed_edge_parquet(&path);
    let edge_source = datafusion_parquet_edge_source(&path).await;
    let mut spec = bfs_spec();
    spec.graph_options.construction_policy = CugraphGraphConstructionPolicy::PythonCugraph;

    let exec = CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, AlgeonNativeExecutionConfig::default())
        .expect("footer-proven simple directed edge source should plan");

    assert!(
        !exec.edge_normalize_inserted(),
        "the source certificate proves the directed Python normalization is a no-op"
    );
}

#[tokio::test]
async fn simple_directed_footer_does_not_cross_a_rowset_changing_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_simple_directed_edge_parquet(&path);
    let edge_source = datafusion_parquet_edge_source_sql(&path, "SELECT src, dst, weight FROM edges LIMIT 2").await;
    let mut spec = bfs_spec();
    spec.graph_options.construction_policy = CugraphGraphConstructionPolicy::PythonCugraph;

    let exec = CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, AlgeonNativeExecutionConfig::default())
        .expect("scan-limited source should retain normal graph planning");

    assert!(
        exec.edge_normalize_inserted(),
        "a file-level topology certificate cannot be borrowed after a rowset-changing limit"
    );
}

#[tokio::test]
async fn cugraph_algo_exec_bypasses_graph_residency_for_datafusion_parquet_scan_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let config = AlgeonNativeExecutionConfig::default();
    let edge_source = datafusion_parquet_edge_source_sql(&path, "SELECT src, dst, weight FROM edges LIMIT 2").await;

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for scan-limited DataFusion local-Parquet edge source");

    assert!(exec.graph_residency.is_none());
}

#[tokio::test]
async fn cugraph_algo_exec_bypasses_graph_residency_for_logical_bfs_datafusion_parquet_edge_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_string_edge_parquet(&path);
    let config = AlgeonNativeExecutionConfig::default();
    let edge_source = datafusion_parquet_edge_source_sql(&path, "SELECT src, dst FROM edges").await;

    let exec = CugraphAlgoExec::try_new(bfs_utf8_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph logical BFS exec should build for DataFusion local-Parquet edge source");
    assert!(exec.graph_residency.is_none());
}

#[tokio::test]
async fn cugraph_sql_bfs_table_function_records_graph_residency_for_datafusion_parquet_edge_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let config = SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let host_boundary_config = GpuIslandHostBoundaryConfig::try_new(18_000, 2_600).unwrap();
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig { host_boundary_config, ..Default::default() })
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let plan = ctx
        .sql(
            "SELECT distance, COUNT(*) AS reached \
             FROM cugraph_bfs(edges => (SELECT src, dst FROM citation_edges), \
             source_vertex => 0, depth_limit => 3, output_mode => 'normalized') \
             WHERE distance <= 3 GROUP BY distance",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let lowering = lower_physical_plan_to_native(Arc::clone(&plan), &ParquetReadStrategyConfig::default()).unwrap();
    assert_eq!(lowering.selected_count(), 1, "grouped BFS result should lower: {:#?}", lowering.report);
    let exec = cugraph_exec_in_plan(Arc::clone(&plan)).expect("planned SQL should contain CugraphAlgoExec");
    let exec = exec.downcast_ref::<CugraphAlgoExec>().unwrap();
    assert_eq!(exec.host_boundary_config, host_boundary_config);
    let descriptor =
        exec.graph_residency.as_ref().expect("stress-shaped SQL local-Parquet edge source should be keyable");
    let edge_schema = exec.edge_source.schema();
    assert!(graph_residency_matches_edge_schema(descriptor, edge_schema.as_ref(), &exec.spec));
}

#[tokio::test]
async fn cugraph_bfs_path_projection_fanout_shares_one_full_result_producer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let target_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let targets =
        RecordBatch::try_new(Arc::clone(&target_schema), vec![Arc::new(Int64Array::from(vec![2])) as ArrayRef])
            .unwrap();
    ctx.register_table("targets", Arc::new(MemTable::try_new(target_schema, vec![vec![targets]]).unwrap())).unwrap();

    let plan = ctx
        .sql(
            "WITH path AS (\
             SELECT path_index, vertex FROM cugraph_bfs(\
               edges => (SELECT src, dst FROM citation_edges), source_vertex => 0, depth_limit => 3,\
               output_mode => 'path', target_vertices => (SELECT vertex FROM targets)\
             )) \
             SELECT curr.vertex FROM path curr \
             JOIN path prev ON curr.path_index = prev.path_index + 1",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let mut producers = Vec::new();
    collect_cugraph_execs(Arc::clone(&plan), &mut producers);

    assert!(producers.len() >= 2, "the self-join should retain both projected scan occurrences");
    assert_eq!(
        producers.iter().map(|producer| producer.gpu_result_input_name.as_str()).collect::<BTreeSet<_>>().len(),
        1,
        "column pruning must not mint a second authoritative BFS occurrence"
    );
    let full_schema = producers[0].schema();
    assert!(
        full_schema.field_with_name("path_index").is_ok()
            && full_schema.field_with_name("vertex").is_ok()
            && producers.iter().all(|producer| producer.schema() == full_schema),
        "each projected scan must wrap the same full path producer schema: {:?}",
        producers
            .iter()
            .map(|producer| {
                (
                    producer.gpu_result_input_name.clone(),
                    producer.schema().fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    );
}

/// This query aggregates the ForceAtlas2 output on the native side. Keep that
/// exact terminal shape covered separately from ticket planning: a graph
/// algorithm that runs successfully but cannot feed its following COUNT/SUM
/// aggregation is not a qualifying workload.
#[tokio::test]
#[ignore = "requires a CUDA-capable cuGraph runtime; run under the repository GPU lock"]
async fn force_atlas2_count_and_coordinate_sums_execute_natively() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet_with_graph_vertex_upper_bound(&path, Some(3));

    let backend = test_gpu_backend();
    let state = backend
        .backend()
        .install_on(SessionStateBuilder::new_with_default_features())
        .unwrap()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
    let stream = attempt
        .sql(
            &ctx,
            "SELECT COUNT(*) AS vertices, SUM(x) AS sum_x, SUM(y) AS sum_y \
             FROM cugraph_force_atlas2(\
               edges => (SELECT src, dst, weight FROM citation_edges), max_iter => 20, seed => 7,\
               barnes_hut_optimize => true)",
        )
        .await;
    let execution_error = match stream {
        Ok(mut stream) => {
            let mut execution_error = None;
            while let Some(batch) = futures::StreamExt::next(&mut stream).await {
                if let Err(error) = batch {
                    execution_error = Some(error);
                    break;
                }
            }
            execution_error
        },
        Err(error) => Some(error),
    };
    let report = attempt.completion().await;
    assert!(
        execution_error.is_none(),
        "ForceAtlas2 aggregate should execute: {execution_error:?}; sealed report: {report:#?}"
    );
    assert!(
        matches!(report.disposition, AttemptDisposition::Native | AttemptDisposition::PartialNative),
        "ForceAtlas2 aggregate must retain a native final plan: {:?}",
        report.disposition
    );
    assert!(
        matches!(report.outcome, RuntimeTerminalOutcome::Succeeded),
        "ForceAtlas2 aggregate must seal a success: {:?}",
        report.outcome
    );
}

#[tokio::test]
async fn cugraph_sql_bfs_table_function_records_graph_residency_with_runner_default_session_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let plan = ctx
        .sql(
            "SELECT COUNT(*) AS reached \
             FROM cugraph_bfs(edges => (SELECT src, dst FROM citation_edges), \
             source_vertex => 0, depth_limit => 3, output_mode => 'normalized')",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let exec = cugraph_exec_in_plan(Arc::clone(&plan)).expect("planned SQL should contain CugraphAlgoExec");
    let exec = exec.downcast_ref::<CugraphAlgoExec>().unwrap();

    assert!(
        exec.graph_residency.is_some(),
        "the multi-GPU runner uses the default session configuration, so its DBLP BFS path must remain cacheable"
    );
}

fn cugraph_exec_in_plan(plan: Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.downcast_ref::<CugraphAlgoExec>().is_some() {
        return Some(plan);
    }
    plan.children().into_iter().find_map(|child| cugraph_exec_in_plan(Arc::clone(child)))
}

fn collect_cugraph_execs(plan: Arc<dyn ExecutionPlan>, output: &mut Vec<CugraphAlgoExec>) {
    if let Some(exec) = plan.downcast_ref::<CugraphAlgoExec>() {
        output.push(exec.clone());
        return;
    }
    for child in plan.children() {
        collect_cugraph_execs(Arc::clone(child), output);
    }
}

#[test]
fn graph_residency_schema_guard_rejects_descriptor_dtype_mismatch() {
    let config = AlgeonNativeExecutionConfig::default();
    let edge_source = native_graph_residency_edge_source(&config);
    let spec = triangle_spec();
    let exec = CugraphAlgoExec::try_new(spec.clone(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for native edge source");
    let descriptor = exec.graph_residency.as_ref().expect("native edge source should be keyable");
    let mismatched_schema = Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
        Field::new("weight", DataType::Float64, false),
    ]);

    assert!(graph_residency_matches_edge_schema(descriptor, edge_schema().as_ref(), &spec));
    assert!(!graph_residency_matches_edge_schema(descriptor, &mismatched_schema, &spec));
}

#[test]
fn standalone_bounded_cugraph_rejects_native_child_execution_without_admission() {
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"));
    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source).unwrap();
    let native_exec = Arc::new(AlgeonNativeExec::try_new_host_for_test(native_plan, config.clone()).unwrap())
        as Arc<dyn ExecutionPlan>;

    let error = validate_standalone_bounded_cugraph_sources(&config, native_exec.as_ref(), &[])
        .expect_err("standalone bounded cuGraph must not execute a native edge child without admission");
    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("reason"), Some("gpu_backend_required"));
    assert_eq!(error.fact_value("remediation"), Some("install_algeon_gpu_backend"));
    assert_eq!(error.fact_value("source_kind"), Some("edge"));

    let repartitioned = Arc::new(RepartitionExec::try_new(native_exec, Partitioning::UnknownPartitioning(1)).unwrap())
        as Arc<dyn ExecutionPlan>;
    let error = validate_standalone_bounded_cugraph_sources(&config, repartitioned.as_ref(), &[])
        .expect_err("fallback collect over native children should be rejected");
    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("reason"), Some("gpu_backend_required"));
    assert_eq!(error.fact_value("remediation"), Some("install_algeon_gpu_backend"));
    assert_eq!(error.fact_value("source_kind"), Some("edge"));
}

#[test]
fn admission_backed_cugraph_allows_native_child_execution() {
    let backend = test_gpu_backend();
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(admission_source(&backend));
    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source).unwrap();
    let native_exec = Arc::new(AlgeonNativeExec::try_new_host_for_test(native_plan, config.clone()).unwrap())
        as Arc<dyn ExecutionPlan>;
    let repartitioned =
        Arc::new(RepartitionExec::try_new(Arc::clone(&native_exec), Partitioning::UnknownPartitioning(1)).unwrap())
            as Arc<dyn ExecutionPlan>;

    validate_standalone_bounded_cugraph_sources(&config, native_exec.as_ref(), &[])
        .expect("query-service admission owns the native child ledger");
    validate_standalone_bounded_cugraph_sources(&config, repartitioned.as_ref(), &[])
        .expect("query-service admission owns fallback-collected native children");
}

#[tokio::test]
async fn cugraph_prepares_nested_native_source_metadata_before_graph_admission() {
    let config = AlgeonNativeExecutionConfig::default().with_lakehouse_footer_pruning(true);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![100, 200])],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let host_native = Arc::new(
        AlgeonNativeExec::try_new_host_for_test(builder.finish_with_host_sink(source).unwrap(), config.clone())
            .unwrap(),
    ) as Arc<dyn ExecutionPlan>;
    let wrapped = Arc::new(
        RepartitionExec::try_new(host_native, Partitioning::UnknownPartitioning(1))
            .expect("test wrapper should accept the native child"),
    ) as Arc<dyn ExecutionPlan>;

    let prepared = prepare_cugraph_source_metadata(wrapped, Arc::new(TaskContext::default()))
        .await
        .expect("metadata preparation should succeed");

    assert!(prepared.downcast_ref::<RepartitionExec>().is_some(), "host wrapper must be preserved");
    let children = prepared.children();
    let native = children[0]
        .downcast_ref::<AlgeonNativeExec>()
        .expect("nested native child should remain under the host wrapper");
    assert!(!native.execution_config().lakehouse_footer_pruning());
}
