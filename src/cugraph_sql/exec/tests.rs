use std::{
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::{
    array::{
        ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, ScalarValue},
    execution::{context::TaskContext, runtime_env::RuntimeEnvBuilder},
    logical_expr::Operator,
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{collect, empty::EmptyExec, filter::FilterExec},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof,
    expr::{Expr, SortExpr, SortSpec},
    graph_memory::graph_memory_bounded_admission_evidence_for_plan,
    observability::RuntimeObservationConfig,
    plan::{ProjectionOutput, QueryPlan},
    runtime::ExecutionCancellation,
    source::{
        GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY, GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY,
        LiteralBatchSource, LocalParquetFile, LocalParquetSource,
    },
};
use parquet::{
    arrow::ArrowWriter,
    file::{metadata::KeyValue, properties::WriterProperties},
};

use super::*;
use crate::{
    attempt::{AttemptOptions, NexusAttemptExtension},
    backend::InstalledTestGpuBackend,
    cugraph_sql::{
        CugraphBfsOutputOptions, CugraphBfsParams, CugraphGraphEdgePredicateTable, CugraphGraphOptions,
        CugraphGraphPredicateTable, CugraphPageRankParams, CugraphSqlConfig, CugraphTableReference,
        CugraphTriangleCountAllParams, result_schema_for_params,
    },
    error::{DataFusionErrorExt, ErrorCode},
    native_exec::{
        GraphClaimCalibrationCapture, SealedAttemptMetricsTarget, admission::QueryAdmissionTicket,
        query_admission_ticket_from_task_context,
    },
    native_report::NativeDataFusionExecutionMetrics,
    session::NexusSessionStateBuilderExt,
};

fn test_gpu_backend() -> InstalledTestGpuBackend {
    InstalledTestGpuBackend::new()
}

fn backend_execution_config(backend: &InstalledTestGpuBackend) -> NexusNativeExecutionConfig {
    backend.optimizer_config().native_execution_config().clone()
}

fn admission_source(backend: &InstalledTestGpuBackend) -> crate::native_exec::QueryAdmissionSource {
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
struct CountingGpuResidentProducer {
    input_name: String,
    ticket: Arc<crate::native_exec::admission::QueryAdmissionTicket>,
    capability_proof: QueryPlanCapabilityProof,
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

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<datafusion::common::Statistics>> {
        self.inner.partition_statistics(partition)
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

    fn prepared_admission_tickets(&self) -> Vec<Arc<crate::native_exec::admission::QueryAdmissionTicket>> {
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

fn counting_gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    plan.downcast_ref::<CountingGpuResidentProducer>().map(|producer| producer as &dyn GpuResidentProducer)
}

fn counting_graph_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
    plan.downcast_ref::<CountingGpuResidentProducer>().map(|producer| producer as &dyn GraphExecutionPlan)
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

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<datafusion::common::Statistics>> {
        self.input.partition_statistics(partition)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(self: Arc<Self>, mut children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Plan(format!("CountingExec expected one child, received {}", children.len())));
        }
        Ok(Arc::new(Self {
            input: children.pop().expect("child count checked"),
            execute_count: Arc::clone(&self.execute_count),
        }))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
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
        Some(WriterProperties::builder().set_max_row_group_row_count(Some(2)).build()),
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

    assert_eq!(edge_source_output_rows(&filter).unwrap(), None);
}

fn native_edge_source(row_group_rows: Vec<u64>, config: &NexusNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_edge_source_with_root(row_group_rows, config, |_, source| source)
}

fn graph_ticket_proof(exec: &CugraphAlgoExec) -> nexus_query_engine::admission::QueryPlanCapabilityProof {
    crate::native_exec::graph::GraphExecutionPlan::graph_execution_capability_proof(exec)
        .expect("cuGraph fixture must issue an exact internal query-plan proof")
}

fn graph_ticket_after_metadata(exec: CugraphAlgoExec) -> CugraphAlgoExec {
    let input_row_bounds =
        CugraphInputRowBounds::from_execution_plans(exec.edge_source.as_ref(), &exec.predicate_sources)
            .expect("prepared cuGraph plans should expose row-bound evidence");
    exec.issue_graph_aware_ticket_after_staging(input_row_bounds, &TaskContext::default(), None)
        .expect("prepared cuGraph staging should mint one graph-aware admission ticket")
        .0
}

fn graph_ticket_proof_for_native_source(
    exec: &CugraphAlgoExec,
) -> nexus_query_engine::admission::QueryPlanCapabilityProof {
    let native = as_nexus_native_exec(exec.edge_source.as_ref()).expect("fixture must contain a native edge source");
    let native_proof = nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_input_plan(
        &native.native_plan().clone_with_gpu_resident_sink(),
    )
    .expect("native edge fixture must have capability requirements");
    nexus_query_engine::admission::QueryPlanCapabilityProof::combine(graph_ticket_proof(exec), [native_proof])
        .expect("graph and native edge proofs must aggregate")
}

fn sorted_native_edge_source(row_group_rows: Vec<u64>, config: &NexusNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_edge_source_with_root(row_group_rows, config, |builder, source| {
        builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("src"), false, false)]), edge_schema())
    })
}

fn native_edge_source_with_root(
    row_group_rows: Vec<u64>,
    config: &NexusNativeExecutionConfig,
    root: impl FnOnce(&mut QueryPlanBuilder, nexus_query_engine::plan::NodeId) -> nexus_query_engine::plan::NodeId,
) -> Arc<dyn ExecutionPlan> {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(row_group_rows)],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let root = root(&mut builder, source);
    let plan = builder.finish_with_host_sink(root);
    Arc::new(NexusNativeExec::try_new(plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>
}

fn native_graph_residency_edge_source_with_stable_id(
    config: &NexusNativeExecutionConfig,
    stable_id: &str,
) -> Arc<dyn ExecutionPlan> {
    let file = LocalParquetFile::new("edges.parquet")
        .with_stable_id(stable_id)
        .with_file_size_bytes(1024)
        .with_row_group_rows(vec![100, 200])
        .with_graph_simple_directed_edge_topology(
            nexus_query_engine::source::GraphSimpleDirectedEdgeTopology::new(0, 1)
                .expect("source and destination columns must differ"),
        );
    let source = LocalParquetSource::from_files(
        vec![file],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projected = builder.add_projection(
        source,
        vec![
            ProjectionOutput::new(Expr::column("src"), Field::new("src", DataType::Int64, false)),
            ProjectionOutput::new(Expr::column("dst"), Field::new("dst", DataType::Int64, false)),
            ProjectionOutput::new(Expr::column("weight"), Field::new("weight", DataType::Float64, false)),
        ],
    );
    let plan = builder.finish_with_host_sink(projected);
    Arc::new(NexusNativeExec::try_new(plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>
}

fn native_graph_residency_edge_source(config: &NexusNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_graph_residency_edge_source_with_stable_id(config, "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z")
}

fn triangle_spec() -> CugraphAlgorithmSpec {
    let params = CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams);
    CugraphAlgorithmSpec {
        algorithm: CugraphAlgorithm::TriangleCountAll,
        table_name: CugraphTableReference::parse("edges").unwrap(),
        src_col: "src".to_string(),
        dst_col: "dst".to_string(),
        weight_col: Some("weight".to_string()),
        edge_id_col: None,
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
        table_name: CugraphTableReference::parse("edges").unwrap(),
        src_col: "src".to_string(),
        dst_col: "dst".to_string(),
        weight_col: Some("weight".to_string()),
        edge_id_col: None,
        graph_options: CugraphGraphOptions {
            directed: true,
            renumber: true,
            construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
        },
        output_schema: result_schema_for_params(CugraphAlgorithm::PageRank, &params),
        params,
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
        table_name: CugraphTableReference::parse("edges").unwrap(),
        src_col: "src".to_string(),
        dst_col: "dst".to_string(),
        weight_col: None,
        edge_id_col: None,
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
        table_name: CugraphTableReference::parse("edges").unwrap(),
        src_col: "src".to_string(),
        dst_col: "dst".to_string(),
        weight_col: None,
        edge_id_col: None,
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
    spec.output_schema = result_schema_for_params(CugraphAlgorithm::Bfs, &spec.params);
    spec
}

#[test]
fn logical_bfs_edge_id_capability_is_rejected_during_exec_construction() {
    let mut spec = bfs_utf8_spec();
    spec.edge_id_col = Some("edge_id".to_string());
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;

    let error =
        CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, NexusNativeExecutionConfig::bounded_default())
            .expect_err("logical BFS edge-id input must fail capability preflight");
    let nexus = error.find_nexus_error().expect("capability preflight must remain a structured adapter error");

    assert_eq!(nexus.code(), ErrorCode::RequiredGpuOperationUnsupported);
    assert_eq!(nexus.fact_value("edge_id_col"), Some("true"));
    assert_eq!(nexus.fact_value("include_edge_ids"), Some("false"));
}

#[test]
fn logical_bfs_include_edge_ids_capability_is_rejected_during_exec_construction() {
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.predicates.include_edge_ids =
        Some(CugraphGraphPredicateTable::new(CugraphTableReference::parse("edge_ids").unwrap(), "edge_id"));
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;
    let predicate_schema = Arc::new(Schema::new(vec![Field::new("edge_id", DataType::Int64, false)]));
    let predicate_source = Arc::new(EmptyExec::new(predicate_schema)) as Arc<dyn ExecutionPlan>;

    let error = CugraphAlgoExec::try_new(
        spec,
        edge_source,
        vec![predicate_source],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect_err("logical BFS include-edge-id input must fail capability preflight");
    let nexus = error.find_nexus_error().expect("capability preflight must remain a structured adapter error");

    assert_eq!(nexus.code(), ErrorCode::RequiredGpuOperationUnsupported);
    assert_eq!(nexus.fact_value("edge_id_col"), Some("false"));
    assert_eq!(nexus.fact_value("include_edge_ids"), Some("true"));
}

#[test]
fn cugraph_algo_exec_does_not_request_input_repartitioning() {
    let exec = CugraphAlgoExec::try_new(
        triangle_spec(),
        memory_edge_source(16),
        vec![],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("cuGraph exec should build");

    assert_eq!(exec.benefits_from_input_partitioning(), vec![false]);
}

#[test]
fn logical_bfs_path_decode_keeps_the_three_column_traversal_schema() {
    let spec = bfs_utf8_path_spec();

    let schema = bfs_decoded_output_schema_for_host(&spec);

    assert_eq!(schema.fields().len(), 3);
    assert_eq!(schema.field(0).name(), "vertex");
    assert_eq!(schema.field(1).name(), "distance");
    assert_eq!(schema.field(2).name(), "predecessor");
}

#[test]
fn logical_bfs_path_reconstruction_applies_the_final_schema_after_decode() {
    let spec = bfs_utf8_path_spec();
    let traversal_schema = bfs_decoded_output_schema_for_host(&spec);
    let traversal = RecordBatch::try_new(
        traversal_schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec![None, Some("alice")])) as ArrayRef,
        ],
    )
    .unwrap();

    let path = reconstruct_bfs_path_batch(&spec, &traversal, BfsPathVertex::Utf8("bob".to_string()))
        .expect("three-column traversal output should reconstruct into the five-column path schema");

    assert_eq!(path.schema(), spec.output_schema);
    assert_eq!(path.num_columns(), 5);
    assert_eq!(path.num_rows(), 2);
}

#[test]
fn logical_bfs_host_codebook_is_sorted_checked_and_null_preserving_for_arrow_strings() {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![Some("bob"), None, Some("alice")])) as ArrayRef,
        Arc::new(LargeStringArray::from(vec![Some("bob"), None, Some("carol")])) as ArrayRef,
        Arc::new(StringViewArray::from(vec![Some("carol"), None, Some("bob")])) as ArrayRef,
    ];
    for array in arrays {
        let expected_first = if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
            array.value(0).to_owned()
        } else if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
            array.value(0).to_owned()
        } else if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
            array.value(0).to_owned()
        } else {
            unreachable!("test array is one of the supported UTF8 representations")
        };
        let expected_sorted: &[&str] = match array.data_type() {
            DataType::Utf8 => &["alice", "bob"],
            DataType::LargeUtf8 | DataType::Utf8View => &["bob", "carol"],
            data_type => unreachable!("unexpected test array type: {data_type:?}"),
        };
        let schema = Arc::new(Schema::new(vec![Field::new("vertex", array.data_type().clone(), true)]));
        let source = StagedCugraphSource::Host {
            schema: Arc::clone(&schema),
            batches: vec![RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap()],
        };
        let codebook = HostBfsCodebook::from_sources_with_limit(&[(&source, vec!["vertex".to_string()])], 8)
            .expect("supported Arrow string arrays should build a host codebook");
        let batch = match &source {
            StagedCugraphSource::Host { batches, .. } => &batches[0],
            StagedCugraphSource::Native { .. } => unreachable!("test source is host-owned"),
        };
        let encoded = encode_logical_array(batch.column(0).as_ref(), &codebook, "vertex")
            .expect("host logical encoding should succeed");
        let encoded = encoded.as_any().downcast_ref::<Int32Array>().expect("encoding must use Int32");
        assert_eq!(codebook.decode_code(i64::from(encoded.value(0))).unwrap(), expected_first);
        assert!(encoded.is_null(1), "nulls must remain null in the encoded domain");
        assert_eq!(codebook.decode_code(0).unwrap(), expected_sorted[0]);
        for (index, value) in expected_sorted.iter().enumerate() {
            assert_eq!(codebook.code(value), Some(index as i32), "codebook code must follow sorted order");
        }
        assert_eq!(codebook.code("missing"), None);
        let owned_string_bytes = expected_sorted.iter().map(|value| String::from(*value).capacity()).sum::<usize>();
        let expected_memory = owned_string_bytes + expected_sorted.len() * std::mem::size_of::<String>();
        assert_eq!(codebook.memory_size().unwrap(), expected_memory, "codebook memory charges one canonical vector");
    }
}

#[test]
fn logical_bfs_host_codebook_rejects_checked_distinct_overflow() {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef])
            .unwrap();
    let source = StagedCugraphSource::Host { schema, batches: vec![batch] };
    let error = HostBfsCodebook::from_sources_with_limit(&[(&source, vec!["vertex".to_string()])], 1)
        .expect_err("the checked test limit must reject a second distinct vertex");
    let nexus = error.find_nexus_error().expect("overflow remains a structured error");
    assert_eq!(nexus.fact_value("reason"), Some("logical_bfs_codebook_overflow"));
}

#[test]
fn logical_bfs_decode_enforces_pregrant_output_overlap_bound() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let reservation = Arc::new(MemoryConsumer::new("logical-bfs-decode-test").register(&runtime.memory_pool));
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int32, false),
        Field::new("distance", DataType::Int64, false),
        Field::new("predecessor", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
            Arc::new(Int32Array::from(vec![None])) as ArrayRef,
        ],
    )
    .unwrap();
    let raw_bytes = get_record_batch_memory_size(&batch);
    reservation.try_grow(raw_bytes).unwrap();
    let source_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let source = StagedCugraphSource::Host {
        schema: Arc::clone(&source_schema),
        batches: vec![
            RecordBatch::try_new(
                Arc::clone(&source_schema),
                vec![Arc::new(StringArray::from(vec!["alice"])) as ArrayRef],
            )
            .unwrap(),
        ],
    };
    let codebook = HostBfsCodebook::from_sources_with_limit(&[(&source, vec!["vertex".to_string()])], 4).unwrap();
    let evidence = HostReservationEvidence::new(raw_bytes, raw_bytes);
    let output = EagerCugraphHostOutput::from_shared_parts_with_evidence(
        schema,
        Arc::new(vec![batch]),
        Arc::clone(&reservation),
        Some(evidence),
    );
    let error = decode_bfs_logical_host_output(
        output,
        &codebook,
        Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Utf8, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Utf8, true),
        ])),
    )
    .expect_err("decoded strings must be charged while raw output remains live");
    let nexus = error.find_nexus_error().expect("decode bound failure remains structured");
    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.fact_value("boundary"), Some("decoded_output"));
    assert_eq!(reservation.size(), raw_bytes, "failed decode must not leak a partial growth");
}

#[test]
fn logical_bfs_decode_reservation_rejects_before_materialization() {
    let raw_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int32, false),
        Field::new("distance", DataType::Int64, false),
        Field::new("predecessor", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&raw_schema),
        vec![
            Arc::new(Int32Array::from(vec![0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
            Arc::new(Int32Array::from(vec![None])) as ArrayRef,
        ],
    )
    .unwrap();
    let raw_bytes = get_record_batch_memory_size(&batch);
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(raw_bytes, 1.0).build().unwrap());
    let reservation = Arc::new(MemoryConsumer::new("logical-bfs-decode-pool-test").register(&runtime.memory_pool));
    reservation.try_grow(raw_bytes).unwrap();
    let source_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let source = StagedCugraphSource::Host {
        schema: Arc::clone(&source_schema),
        batches: vec![
            RecordBatch::try_new(
                Arc::clone(&source_schema),
                vec![Arc::new(StringArray::from(vec!["alice"])) as ArrayRef],
            )
            .unwrap(),
        ],
    };
    let codebook = HostBfsCodebook::from_sources_with_limit(&[(&source, vec!["vertex".to_string()])], 4).unwrap();
    let decoded_upper_bound = decoded_bfs_logical_host_batch_upper_bound(&batch, &codebook).unwrap();
    let decoded_actual = decode_bfs_logical_host_batch(
        &batch,
        &codebook,
        Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Utf8, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Utf8, true),
        ])),
    )
    .unwrap();
    assert!(
        decoded_upper_bound > get_record_batch_memory_size(&decoded_actual),
        "the pre-materialization bound must include Arrow builder overlap"
    );
    drop(decoded_actual);
    let output = EagerCugraphHostOutput::from_shared_parts_with_evidence(
        raw_schema,
        Arc::new(vec![batch]),
        Arc::clone(&reservation),
        Some(HostReservationEvidence::new(usize::MAX, usize::MAX)),
    );
    let error = decode_bfs_logical_host_output(
        output,
        &codebook,
        Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Utf8, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Utf8, true),
        ])),
    )
    .expect_err("decoded output must reserve before the constrained pool can materialize it");
    let nexus = error.find_nexus_error().expect("pool rejection remains structured");
    assert_eq!(nexus.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(nexus.fact_value("boundary"), Some("decoded_output"));
    let requested_bytes = decoded_upper_bound.to_string();
    assert_eq!(nexus.fact_value("requested_bytes"), Some(requested_bytes.as_str()));
    assert_eq!(reservation.size(), raw_bytes, "failed pre-materialization reservation must not leak bytes");
}

#[test]
fn logical_bfs_decode_preserves_nulls_and_releases_large_validity_reservation() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(16 * 1024 * 1024, 1.0).build().unwrap());
    let reservation = Arc::new(MemoryConsumer::new("logical-bfs-decode-roundtrip-test").register(&runtime.memory_pool));
    let row_count = 513;
    let raw_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int32, true),
        Field::new("distance", DataType::Int64, false),
        Field::new("predecessor", DataType::Int32, true),
    ]));
    let mut vertices = vec![Some(0); row_count];
    vertices[1] = None;
    let mut predecessors = vec![Some(-1); row_count];
    predecessors[2] = Some(0);
    let batch = RecordBatch::try_new(
        Arc::clone(&raw_schema),
        vec![
            Arc::new(Int32Array::from(vertices)) as ArrayRef,
            Arc::new(Int64Array::from(vec![0; row_count])) as ArrayRef,
            Arc::new(Int32Array::from(predecessors)) as ArrayRef,
        ],
    )
    .unwrap();
    let raw_bytes = get_record_batch_memory_size(&batch);
    reservation.try_grow(raw_bytes).unwrap();
    let source_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let source = StagedCugraphSource::Host {
        schema: Arc::clone(&source_schema),
        batches: vec![
            RecordBatch::try_new(
                Arc::clone(&source_schema),
                vec![Arc::new(StringArray::from(vec!["alice"])) as ArrayRef],
            )
            .unwrap(),
        ],
    };
    let codebook = HostBfsCodebook::from_sources_with_limit(&[(&source, vec!["vertex".to_string()])], 4).unwrap();
    let decoded = decode_bfs_logical_host_output(
        EagerCugraphHostOutput::from_shared_parts_with_evidence(
            raw_schema,
            Arc::new(vec![batch]),
            Arc::clone(&reservation),
            Some(HostReservationEvidence::new(usize::MAX, usize::MAX)),
        ),
        &codebook,
        Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Utf8, true),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Utf8, true),
        ])),
    )
    .expect("decoded output should fit its checked bound");
    let (batches, decoded_reservation, _) = decoded.into_parts_with_evidence();
    let vertex = batches[0].column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let predecessor = batches[0].column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(vertex.value(0), "alice");
    assert!(vertex.is_null(1), "vertex null mask must survive decode");
    assert!(predecessor.is_null(0), "negative predecessor sentinel must decode to null");
    assert_eq!(predecessor.value(2), "alice");
    assert_eq!(decoded_reservation.size(), get_record_batch_memory_size(&batches[0]));
    drop(batches);
    drop(decoded_reservation);
    drop(reservation);
    assert_eq!(runtime.memory_pool.reserved(), 0, "decoded reservation must release after output ownership ends");
}

#[test]
fn logical_bfs_projection_enforces_pregrant_output_overlap_bound() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let reservation = Arc::new(MemoryConsumer::new("logical-bfs-projection-test").register(&runtime.memory_pool));
    let spec = bfs_utf8_spec();
    let schema = bfs_decoded_output_schema_for_host(&spec);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("alice")])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
            Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
        ],
    )
    .unwrap();
    let old_bytes = get_record_batch_memory_size(&batch);
    reservation.try_grow(old_bytes).unwrap();
    let output = EagerCugraphHostOutput::from_shared_parts_with_evidence(
        schema,
        Arc::new(vec![batch]),
        Arc::clone(&reservation),
        Some(HostReservationEvidence::new(old_bytes, old_bytes)),
    );
    let projection = vec![0];
    let error = apply_host_projection_limit(output, &spec, Some(&projection), None)
        .expect_err("projection must account for old and new Arrow batches while they overlap");
    let nexus = error.find_nexus_error().expect("projection bound failure remains structured");
    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.fact_value("boundary"), Some("projected_output"));
    assert_eq!(reservation.size(), old_bytes, "failed projection must not leak a partial growth");
}

#[test]
fn logical_bfs_path_enforces_target_and_reconstruction_overlap_bound() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let output_reservation =
        Arc::new(MemoryConsumer::new("logical-bfs-path-output-test").register(&runtime.memory_pool));
    let target_reservation =
        Arc::new(MemoryConsumer::new("logical-bfs-path-target-test").register(&runtime.memory_pool));
    let spec = bfs_utf8_path_spec();
    let schema = bfs_decoded_output_schema_for_host(&spec);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob")])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec![None::<&str>, Some("alice")])) as ArrayRef,
        ],
    )
    .unwrap();
    let raw_bytes = get_record_batch_memory_size(&batch);
    output_reservation.try_grow(raw_bytes).unwrap();
    target_reservation.try_grow(1).unwrap();
    let evidence = HostReservationEvidence::new(raw_bytes + 1, raw_bytes + 1);
    let output = EagerCugraphHostOutput::from_shared_parts_with_evidence(
        schema,
        Arc::new(vec![batch]),
        Arc::clone(&output_reservation),
        Some(evidence),
    );
    let target = BoundedBfsPathTarget::for_test(BfsPathVertex::Utf8("bob".to_string()), target_reservation);
    let error = materialize_bfs_path_output_from_host(output, &spec, target, None, None, raw_bytes + 1)
        .expect_err("path reconstruction must account for traversal, target, workspace, and final output overlap");
    let nexus = error.find_nexus_error().expect("path bound failure remains structured");
    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.fact_value("boundary"), Some("reconstruction_workspace"));
    assert_eq!(output_reservation.size(), raw_bytes, "failed path reconstruction must not leak workspace growth");
}

#[tokio::test]
async fn logical_bfs_host_mapping_charges_codebook_and_encoded_peak_then_releases_on_failure() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(4 * 1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let spec = bfs_utf8_spec();
    let edge_source = memory_string_edge_source();
    let edge_batches = collect(Arc::clone(&edge_source), Arc::clone(&task_context)).await.unwrap();
    let edge_bytes = edge_batches.iter().map(get_record_batch_memory_size).sum::<usize>();
    drop(edge_batches);
    let source_batch = bfs_literal_string_source_batch(&spec).unwrap().unwrap();
    let source_bytes = get_record_batch_memory_size(&source_batch);
    let max_staging_bytes = edge_bytes.checked_add(source_bytes).expect("input bytes fit usize");
    let error = stage_cugraph_inputs(
        &spec,
        edge_source,
        vec![],
        task_context,
        &ExecutionCancellation::new(),
        max_staging_bytes,
    )
    .await
    .expect_err("the codebook owner must be charged before admission and exceed the input-only cap");
    let nexus = error.find_nexus_error().expect("host mapping cap failure remains structured");
    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.fact_value("boundary"), Some("codebook"));
    assert_eq!(runtime.memory_pool.reserved(), 0, "failed host mapping releases all staged ownership");
}

#[tokio::test]
async fn logical_bfs_codebook_reservation_rejects_before_graph_ticket() {
    let backend = test_gpu_backend();
    let config = backend_execution_config(&backend);
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(4 * 1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let spec = bfs_utf8_spec();
    let edge_source = memory_string_edge_source();
    let edge_batches = collect(Arc::clone(&edge_source), Arc::clone(&task_context)).await.unwrap();
    let edge_bytes = edge_batches.iter().map(get_record_batch_memory_size).sum::<usize>();
    let source_batch = bfs_literal_string_source_batch(&spec).unwrap().unwrap();
    let source_bytes = get_record_batch_memory_size(&source_batch);
    let edge_staged = StagedCugraphSource::Host { schema: edge_batches[0].schema(), batches: edge_batches.clone() };
    let source_staged =
        StagedCugraphSource::Host { schema: source_batch.schema(), batches: vec![source_batch.clone()] };
    let domain_sources = vec![
        (&edge_staged, vec![spec.src_col.clone(), spec.dst_col.clone()]),
        (&source_staged, vec!["__source_vertex".to_string()]),
    ];
    let codebook_bound = HostBfsCodebook::allocation_upper_bound(&domain_sources).unwrap();
    let codebook_actual =
        HostBfsCodebook::from_sources_with_limit(&domain_sources, i32::MAX as usize).unwrap().memory_size().unwrap();
    assert!(codebook_bound.bytes > codebook_actual, "pre-reservation must cover compaction overlap");
    let input_limit = edge_bytes.checked_add(source_bytes).expect("input bytes fit usize");
    let boundary = GpuIslandHostBoundaryConfig::try_new(input_limit, 1024 * 1024).unwrap();
    let exec =
        CugraphAlgoExec::try_new_with_host_boundary_config(spec, edge_source, vec![], None, None, config, boundary)
            .expect("logical BFS should build before its deliberate codebook bound");
    let error = exec
        .prepare_staged_attempt(task_context)
        .await
        .expect_err("codebook ownership must reject before ticket issuance");
    let nexus = error.find_nexus_error().expect("codebook rejection remains structured");
    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.fact_value("boundary"), Some("codebook"));
    let requested_bytes = codebook_bound.bytes.to_string();
    assert_eq!(nexus.fact_value("next_batch_bytes"), Some(requested_bytes.as_str()));
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(runtime.memory_pool.reserved(), 0);
}

#[test]
fn logical_bfs_codebook_pool_rejects_before_untracked_construction() {
    let spec = bfs_utf8_spec();
    let edge_schema = string_edge_schema();
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(StringArray::from(vec![Some("bob"), None, Some("alice")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob"), None])) as ArrayRef,
        ],
    )
    .unwrap();
    let source_batch = bfs_literal_string_source_batch(&spec).unwrap().unwrap();
    let edge = StagedCugraphSource::Host { schema: edge_schema, batches: vec![edge_batch] };
    let source = StagedCugraphSource::Host { schema: source_batch.schema(), batches: vec![source_batch] };
    let domain_sources =
        vec![(&edge, vec![spec.src_col.clone(), spec.dst_col.clone()]), (&source, vec!["__source_vertex".to_string()])];
    let codebook_bound = HostBfsCodebook::allocation_upper_bound(&domain_sources).unwrap();
    let codebook_actual =
        HostBfsCodebook::from_sources_with_limit(&domain_sources, i32::MAX as usize).unwrap().memory_size().unwrap();
    assert!(codebook_bound.bytes > codebook_actual);
    let input_bytes = [
        match &edge {
            StagedCugraphSource::Host { batches, .. } => get_record_batch_memory_size(&batches[0]),
            StagedCugraphSource::Native { .. } => unreachable!(),
        },
        match &source {
            StagedCugraphSource::Host { batches, .. } => get_record_batch_memory_size(&batches[0]),
            StagedCugraphSource::Native { .. } => unreachable!(),
        },
    ]
    .into_iter()
    .sum::<usize>();
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(input_bytes, 1.0).build().unwrap());
    let mut reservation = MemoryConsumer::new("logical-bfs-codebook-pool-test").register(&runtime.memory_pool);
    reservation.try_grow(input_bytes).unwrap();
    let max_staging_bytes = input_bytes.checked_add(codebook_bound.bytes).unwrap();
    let mut staged_bytes = input_bytes;
    let mut evidence = HostReservationEvidence::new(max_staging_bytes, usize::MAX);
    let error = prepare_bfs_logical_host_inputs(
        &spec,
        edge,
        Some(source),
        vec![],
        &mut staged_bytes,
        max_staging_bytes,
        &mut reservation,
        &mut evidence,
    )
    .expect_err("codebook ownership must fail at the reservation boundary");
    let nexus = error.find_nexus_error().expect("pool rejection remains structured");
    assert_eq!(nexus.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(nexus.fact_value("boundary"), Some("codebook"));
    let requested_bytes = codebook_bound.bytes.to_string();
    assert_eq!(nexus.fact_value("requested_bytes"), Some(requested_bytes.as_str()));
    assert_eq!(reservation.size(), input_bytes, "pool rejection must not leak the input reservation");
    assert_eq!(staged_bytes, input_bytes, "pool rejection must not advance staged accounting");
}

#[tokio::test]
async fn logical_bfs_native_source_rejects_before_ticket_or_gpu_context() {
    let backend = test_gpu_backend();
    let config = backend_execution_config(&backend);
    let schema = string_edge_schema();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_source = Arc::new(NexusNativeExec::try_new(native_plan, config).unwrap()) as Arc<dyn ExecutionPlan>;
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let exec = CugraphAlgoExec::try_new(
        bfs_utf8_spec(),
        native_source,
        vec![],
        None,
        None,
        backend_execution_config(&backend),
    )
    .expect("logical BFS should build before native-source staging rejection");
    let error = exec
        .prepare_staged_attempt(task_context)
        .await
        .expect_err("logical BFS must not stage a native logical source");
    let nexus = error.find_nexus_error().expect("native logical-source rejection remains structured");
    assert_eq!(nexus.fact_value("reason"), Some("logical_bfs_native_source_unsupported"));
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
}

#[tokio::test]
async fn cugraph_utf8_include_edges_native_predicate_rejects_owner_before_logical_staging() {
    let backend = test_gpu_backend();
    let config = backend_execution_config(&backend);
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.predicates.include_edges = Some(CugraphGraphEdgePredicateTable::new(
        CugraphTableReference::parse("allowed_edges").unwrap(),
        "src",
        "dst",
        None::<String>,
    ));

    let schema = string_edge_schema();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_predicate =
        Arc::new(NexusNativeExec::try_new(native_plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>;
    let exec = CugraphAlgoExec::try_new(
        spec.clone(),
        memory_string_edge_source(),
        vec![Arc::clone(&native_predicate)],
        None,
        None,
        config.clone(),
    )
    .expect("include_edges owner rejection occurs after metadata hydration");

    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("cuGraph stream should build");
    let error = futures::TryStreamExt::try_next(&mut stream)
        .await
        .expect_err("graph owner preflight must reject before staging the native logical predicate");
    let nexus = error.find_nexus_error().expect("owner rejection remains structured");
    assert_eq!(nexus.fact_value("reason"), Some("side_input_workspace_preflight_unavailable"));
    assert_eq!(nexus.fact_value("side_input_reason"), Some("include_edges_owner_contract_missing"));
    assert_eq!(nexus.fact_value("side_input_role"), Some("include_edges"));
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);

    let producer_exec = Arc::new(
        CugraphAlgoExec::try_new(spec, memory_string_edge_source(), vec![native_predicate], None, None, config)
            .expect("include_edges owner rejection occurs after producer metadata hydration"),
    );
    let producer_result = producer_exec
        .prepare_execution_plan(Arc::clone(&producer_exec) as Arc<dyn ExecutionPlan>, Arc::new(TaskContext::default()))
        .await;
    let producer_error = match producer_result {
        Ok(_) => panic!("graph owner preflight must reject before producer staging the native logical predicate"),
        Err(error) => error,
    };
    let producer_nexus = producer_error.find_nexus_error().expect("owner rejection remains structured");
    assert_eq!(producer_nexus.fact_value("reason"), Some("side_input_workspace_preflight_unavailable"));
    assert_eq!(producer_nexus.fact_value("side_input_reason"), Some("include_edges_owner_contract_missing"));
    assert_eq!(producer_nexus.fact_value("side_input_role"), Some("include_edges"));
}

#[test]
fn logical_and_path_bfs_decline_gpu_resident_producer_but_numeric_bfs_remains_a_producer() {
    let logical = CugraphAlgoExec::try_new(
        bfs_utf8_spec(),
        memory_string_edge_source(),
        vec![],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("logical BFS should build as a host-root graph execution");
    assert!(crate::native_exec::producer::gpu_resident_producer(&logical).is_none());

    let numeric_path = CugraphAlgoExec::try_new(
        bfs_utf8_path_spec(),
        memory_string_edge_source(),
        vec![],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("logical BFS path should build as a host-root graph execution");
    assert!(crate::native_exec::producer::gpu_resident_producer(&numeric_path).is_none());

    let numeric = CugraphAlgoExec::try_new(
        bfs_spec(),
        memory_edge_source(4),
        vec![],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("numeric non-path BFS should retain the GPU producer contract");
    assert!(crate::native_exec::producer::gpu_resident_producer(&numeric).is_some());
}

#[test]
fn malformed_logical_bfs_mapping_fails_closed_at_gpu_boundaries() {
    let mut spec = bfs_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new(
        CugraphTableReference::parse("source_vertices").unwrap(),
        "source_vertex",
    ));
    let source_schema = Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Int64, false)]));
    let source_exec = Arc::new(EmptyExec::new(Arc::clone(&source_schema))) as Arc<dyn ExecutionPlan>;
    let mut exec = CugraphAlgoExec::try_new(
        spec,
        memory_edge_source(4),
        vec![source_exec],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("numeric relation-backed BFS should build before schema corruption");
    let malformed_schema = Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false)]));
    exec.edge_source = Arc::new(EmptyExec::new(malformed_schema)) as Arc<dyn ExecutionPlan>;
    let plan = &exec as &dyn ExecutionPlan;

    assert!(
        crate::native_exec::producer::gpu_resident_producer(plan).is_none(),
        "mapping validation errors must not classify a malformed BFS as a GPU producer"
    );
    let rejection = crate::native_exec::graph::GraphExecutionPlan::gpu_resident_output_rejection(&exec)
        .expect("mapping validation errors must be surfaced at the GPU boundary");
    let nexus = rejection.find_nexus_error().expect("mapping validation error must remain structured");
    assert_eq!(nexus.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(nexus.operation(), "datafusion_nexus.cugraph_sql.plan");
    assert!(nexus.message().contains("dst"), "the missing mapping column must remain actionable");
}

#[tokio::test]
async fn cugraph_producer_preparation_stages_host_inputs_before_gpu_execution() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let exec = Arc::new(
        CugraphAlgoExec::try_new(
            triangle_spec(),
            memory_edge_source(16),
            vec![],
            None,
            None,
            NexusNativeExecutionConfig::bounded_default(),
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
    let config = NexusNativeExecutionConfig::bounded(64 * 1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(admission_source(&backend));
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(16 * 1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
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
    assert_eq!(staged.row_bounds().unwrap().edge_row_upper_bound(), Some(499));
    assert_eq!(prepared.input_row_bounds.edge_row_upper_bound(), Some(499));
    assert!(
        prepared.native_execution_config.query_admission_ticket().is_some(),
        "the graph ticket must be minted only after staged exact rows are available"
    );
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
    drop(staged);
    assert_eq!(runtime.memory_pool.reserved(), 0, "dropping staged host ownership must release its reservation");
}

#[tokio::test]
async fn cugraph_staged_attempt_preserves_host_edge_and_predicate_row_order() {
    let backend = test_gpu_backend();
    let config = NexusNativeExecutionConfig::bounded(64 * 1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(admission_source(&backend));
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(16 * 1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("fixture is BFS") };
    params.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new(
        CugraphTableReference::parse("source_vertices").unwrap(),
        "vertex_id",
    ));
    let source_schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Utf8, false)]));
    let source_batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![Arc::new(StringArray::from(vec!["alice", "bob", "carol"])) as ArrayRef],
    )
    .unwrap();
    let source_plan =
        MemorySourceConfig::try_new_exec(&[vec![source_batch.clone()]], Arc::clone(&source_schema), None).unwrap();
    let edge_schema_for_bound = string_edge_schema();
    let edge_batch_for_bound = RecordBatch::try_new(
        Arc::clone(&edge_schema_for_bound),
        vec![
            Arc::new(StringArray::from(vec![Some("bob"), None, Some("alice")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob"), None])) as ArrayRef,
        ],
    )
    .unwrap();
    let edge_for_bound =
        StagedCugraphSource::Host { schema: edge_schema_for_bound, batches: vec![edge_batch_for_bound] };
    let source_for_bound =
        StagedCugraphSource::Host { schema: Arc::clone(&source_schema), batches: vec![source_batch.clone()] };
    let bound_sources = vec![
        (&edge_for_bound, vec!["src".to_string(), "dst".to_string()]),
        (&source_for_bound, vec!["vertex_id".to_string()]),
    ];
    let codebook_bound = HostBfsCodebook::allocation_upper_bound(&bound_sources).unwrap();
    let preconstruction_input_bytes = bound_sources
        .iter()
        .flat_map(|(source, _)| match source {
            StagedCugraphSource::Host { batches, .. } => {
                batches.iter().map(get_record_batch_memory_size).collect::<Vec<_>>()
            },
            StagedCugraphSource::Native { .. } => Vec::new(),
        })
        .sum::<usize>();
    let expected_preconstruction_peak = preconstruction_input_bytes + codebook_bound.bytes;
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
    let bounds = staged.row_bounds().unwrap();

    assert_eq!(bounds.edge_row_upper_bound(), Some(3));
    assert_eq!(bounds.predicate_row_upper_bounds(), &[Some(3)]);
    assert_eq!(prepared.input_row_bounds.edge_row_upper_bound(), Some(3));
    assert_eq!(prepared.input_row_bounds.predicate_row_upper_bounds(), &[Some(3)]);
    assert_eq!(edge_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(predicate_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(staged.logical_codebook.is_some(), "logical host staging must retain its deterministic codebook");
    assert_eq!(staged.host_evidence.input_limit_bytes(), input_limit_bytes);
    assert!(staged.host_evidence.logical_codebook_bytes() > 0);
    assert!(
        staged.host_evidence.logical_encoded_input_peak_bytes() >= expected_preconstruction_peak,
        "host evidence must retain the codebook construction overlap peak",
    );
    assert!(staged.host_evidence.logical_encoded_input_peak_bytes() >= staged.reservation.as_ref().unwrap().size());
    assert_eq!(staged.host_evidence.output_limit_bytes(), output_limit_bytes);
    assert_eq!(
        staged.host_evidence.aggregate_peak_bytes(),
        staged.host_evidence.logical_encoded_input_peak_bytes() + staged.host_evidence.output_limit_bytes(),
        "host evidence must carry one checked aggregate peak for overlapping input and output reservations",
    );
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
    drop(staged);
    assert_eq!(runtime.memory_pool.reserved(), 0, "staged edge and predicate ownership must be released together");
}

#[tokio::test]
async fn cugraph_native_staging_retains_hard_rows_and_never_executes_a_wrapped_child() {
    let config = NexusNativeExecutionConfig::bounded_default();
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
    assert_eq!(staged.row_bounds().unwrap().edge_row_upper_bound(), Some(300));
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
    let nexus = error.find_nexus_error().expect("wrapped native rejection remains structured");
    assert_eq!(nexus.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(execute_count.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cugraph_staged_attempt_failure_cleans_host_reservation_before_admission() {
    let backend = test_gpu_backend();
    let config = NexusNativeExecutionConfig::bounded(64 * 1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(admission_source(&backend));
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(16 * 1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
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
    let nexus = error.find_nexus_error().expect("host staging cap failure remains structured");
    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(runtime.memory_pool.reserved(), 0, "failed staging must release all host ownership");
}

#[tokio::test]
async fn cugraph_producer_preparation_mints_one_post_hydration_ticket_for_host_and_gpu_paths() {
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let task_context = Arc::new(TaskContext::default());
    let exec = Arc::new(
        CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(16), vec![], None, None, config)
            .expect("cuGraph producer should build without a pre-hydration ticket"),
    );

    assert_eq!(exec.native_execution_config.query_admission_ticket_id(), None);

    let prepared = exec
        .prepare_execution_plan(Arc::clone(&exec) as Arc<dyn ExecutionPlan>, Arc::clone(&task_context))
        .await
        .expect("producer preparation should hydrate metadata and mint its one ticket");
    let prepared_exec = prepared.downcast_ref::<CugraphAlgoExec>().expect("prepared plan should remain cuGraph");
    let ticket_id = prepared_exec
        .native_execution_config
        .query_admission_ticket_id()
        .expect("post-hydration cuGraph producer must carry its graph-aware ticket");
    let prepared_graph = crate::native_exec::graph::graph_execution_plan(prepared.as_ref())
        .expect("prepared cuGraph execution must retain its frontend-neutral graph binding");
    assert_eq!(
        prepared_graph.graph_result_input_name(),
        prepared_exec.gpu_result_input_name,
        "the prepared graph scope and runtime result occurrence must remain bound to one input name",
    );
    let prepared_ticket = prepared_exec
        .native_execution_config
        .query_admission_ticket()
        .expect("prepared cuGraph occurrence must retain its graph-aware ticket");
    assert!(
        prepared_ticket.attempt_capability_proof().is_some(),
        "the prepared occurrence must retain one opaque graph proof for composed admission",
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
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let exec = Arc::new(
        CugraphAlgoExec::try_new(
            triangle_spec(),
            memory_edge_source(16),
            vec![],
            None,
            None,
            NexusNativeExecutionConfig::bounded_default(),
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
    let nexus = error.find_nexus_error().expect("consumed staged-input failure remains structured");

    assert_eq!(nexus.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(nexus.fact_value("staged_input_state"), Some("already_consumed"));
    drop(prepared_clone);
    drop(prepared);
    assert!(runtime.memory_pool.reserved() > 0, "only the transferred owner retains the reservation");
    drop(staged_inputs);
    assert_eq!(runtime.memory_pool.reserved(), 0, "dropping the sole staged-input owner releases the reservation");
}

#[tokio::test]
async fn cugraph_input_upload_releases_the_host_reservation_before_graph_compute() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let native_context =
        ExecutionContext::from_shared_runtime(Arc::new(cudf::Runtime::new().expect("test CUDA runtime"))).unwrap();
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut graph_metrics = GraphExecutionMetrics::new(&native_metrics, RuntimeObservationConfig::native_execution());
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

    let uploaded = staged_inputs
        .upload(&native_context, &mut graph_metrics, task_context)
        .await
        .expect("staged host graph inputs should upload");

    assert_eq!(
        runtime.memory_pool.reserved(),
        0,
        "host-input reservation must be released before graph compute receives uploaded inputs",
    );
    drop(uploaded);
}

#[tokio::test]
async fn cugraph_staged_upload_rejects_missing_admission_plan_before_source_upload() {
    let backend = test_gpu_backend();
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let config = backend_execution_config(&backend).with_query_service_handle(Arc::new(
        backend.admit_query().expect("test query must be admitted by the backend"),
    ));
    let context_lease =
        config.execution_context_async(Arc::clone(&task_context)).await.expect("backend context builds");
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut graph_metrics = GraphExecutionMetrics::new(&native_metrics, RuntimeObservationConfig::native_execution());
    let staged_inputs = stage_cugraph_inputs(
        &triangle_spec(),
        memory_edge_source(16),
        vec![],
        Arc::clone(&task_context),
        config.cancellation(),
        1024 * 1024,
    )
    .await
    .expect("host graph inputs should stage");

    let error = match staged_inputs.upload(context_lease.context(), &mut graph_metrics, task_context).await {
        Err(error) => error,
        Ok(_) => panic!("an admitted upload without its exact graph plan must fail closed"),
    };
    let nexus = error.find_nexus_error().expect("missing admission plan remains structured");
    assert_eq!(nexus.fact_value("reason"), Some("admission_graph_plan_missing"));
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
        _ => builder.add_union(inputs, Arc::clone(&schema)),
    };
    builder.finish_with_host_sink(root)
}

fn direct_composed_graph_proof() -> (Arc<Schema>, QueryPlanCapabilityProof) {
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
    let graph = builder.try_add_graph_algorithm(source, spec.clone()).expect("direct graph fixture must lower");
    let graph_plan = builder.finish_gpu_resident(graph);
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&graph_plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("direct graph fixture must have finite graph evidence");
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("direct graph fixture must issue a capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("direct graph fixture must retain finite graph evidence");
    (spec.output_schema, proof)
}

fn direct_composed_admission_context_config(
    backend: &InstalledTestGpuBackend,
    consumer_plan: &QueryPlan,
    children: &[Arc<dyn ExecutionPlan>],
) -> NexusNativeExecutionConfig {
    let proof = crate::native_exec::composed_admission_capability_proof(consumer_plan, children)
        .expect("direct composed fixture must issue one complete admission proof");
    let required_floor = proof.required_floor_bytes_for(0).expect("direct composed proof floor must be finite");
    let admitted = backend
        .backend()
        .session_capability()
        .admit_query_with_request(
            nexus_query_engine::runtime::QueryAdmissionRequest::new().with_query_plan_capability_proof(proof.clone()),
        )
        .expect("direct composed proof must be admitted by the installed backend profile");
    assert!(
        admitted.device_budget_bytes() >= required_floor,
        "direct composed admission grant must cover its proof floor"
    );
    drop(admitted);
    backend_execution_config(backend)
        .with_graph_aware_plan_query_admission_ticket(None, vec![proof])
        .expect("direct composed fixture must mint one admission ticket")
}

#[tokio::test]
async fn composed_edge_source_rejects_admission_mismatch_before_running_producers() {
    register_gpu_resident_producer_discovery(counting_gpu_resident_producer);
    register_graph_execution_plan_discovery(counting_graph_execution_plan);
    let backend = test_gpu_backend();
    let admission_source = admission_source(&backend);
    let (schema, producer_proof) = direct_composed_graph_proof();
    let consumer_host_plan = direct_composed_consumer_plan(&["producer_a", "producer_b"], Arc::clone(&schema));
    let consumer_gpu_plan = consumer_host_plan.clone_with_gpu_resident_sink();
    let execute_counts = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
    let producer_names = ["producer_a", "producer_b"];
    let children = producer_names
        .iter()
        .zip(&execute_counts)
        .map(|(input_name, execute_count)| {
            let ticket = QueryAdmissionTicket::new(admission_source.clone(), Some(producer_proof.clone()));
            Arc::new(CountingGpuResidentProducer {
                input_name: (*input_name).to_string(),
                ticket,
                capability_proof: producer_proof.clone(),
                execute_count: Arc::clone(execute_count),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }) as Arc<dyn ExecutionPlan>
        })
        .collect::<Vec<_>>();
    let composed_exec = Arc::new(
        NexusNativeComposedExec::try_new(
            consumer_host_plan.clone(),
            children.clone(),
            backend_execution_config(&backend),
        )
        .expect("direct composed fixture must build"),
    ) as Arc<dyn ExecutionPlan>;

    let wrong_order_children = vec![Arc::clone(&children[1]), Arc::clone(&children[0])];
    let wrong_order_config =
        direct_composed_admission_context_config(&backend, &consumer_gpu_plan, &wrong_order_children);
    let wrong_order_task_context = Arc::new(TaskContext::default());
    let wrong_order_context = wrong_order_config
        .execution_context_async(Arc::clone(&wrong_order_task_context))
        .await
        .expect("wrong-order fixture must acquire an admitted context");
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut graph_metrics = GraphExecutionMetrics::new(&native_metrics, RuntimeObservationConfig::native_execution());
    let wrong_order_error = match execute_edge_source_to_gpu(
        wrong_order_context.context(),
        &mut graph_metrics,
        Arc::clone(&composed_exec),
        Arc::clone(&wrong_order_task_context),
        Some(consumer_gpu_plan.clone()),
    )
    .await
    {
        Err(error) => error,
        Ok(_) => panic!("runtime producer order must not execute under a differently ordered admission proof"),
    };
    let wrong_order_nexus = wrong_order_error.find_nexus_error().expect("order rejection must remain structured");
    assert_eq!(wrong_order_nexus.fact_value("schedule_reason"), Some("runtime_occurrence_order_mismatch"));
    assert_eq!(execute_counts[0].load(Ordering::SeqCst), 0);
    assert_eq!(execute_counts[1].load(Ordering::SeqCst), 0);
    drop(wrong_order_context);

    let original_config = direct_composed_admission_context_config(&backend, &consumer_gpu_plan, &children);
    let original_task_context = Arc::new(TaskContext::default());
    let original_context = original_config
        .execution_context_async(Arc::clone(&original_task_context))
        .await
        .expect("original fixture must acquire an admitted context");
    let changed_consumer_gpu_plan = consumer_host_plan
        .with_appended_sort(SortSpec::new(vec![SortExpr::new(Expr::column("vertex"), false, false)]))
        .clone_with_gpu_resident_sink();
    let changed_scope_error = match execute_edge_source_to_gpu(
        original_context.context(),
        &mut graph_metrics,
        composed_exec,
        original_task_context,
        Some(changed_consumer_gpu_plan),
    )
    .await
    {
        Err(error) => error,
        Ok(_) => panic!("a changed consumer scope must not execute admitted producers"),
    };
    let changed_scope_nexus = changed_scope_error.find_nexus_error().expect("scope rejection must remain structured");
    assert_eq!(changed_scope_nexus.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
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
    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("cuGraph stream should build");
    let execution = tokio::spawn(async move { stream.try_next().await });

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while execute_count.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
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
    let nexus = error.find_nexus_error().expect("staging cap failure must remain structured");

    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.status(), crate::error::ErrorStatus::Permanent);
    assert_eq!(nexus.fact_value("source_kind"), Some("side_input"));
    let edge_bytes = edge_bytes.to_string();
    assert_eq!(nexus.fact_value("staged_bytes"), Some(edge_bytes.as_str()));
    assert_eq!(runtime.memory_pool.reserved(), 0, "failed cumulative staging must release its reservation");
}

#[tokio::test]
async fn cugraph_producer_preparation_charges_the_exact_utf8_literal_source_batch_to_the_input_cap() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let spec = bfs_utf8_spec();
    let source_batch = bfs_literal_string_source_batch(&spec)
        .expect("UTF-8 BFS source batch should build")
        .expect("UTF-8 BFS source should have a staged literal batch");
    let source_bytes = get_record_batch_memory_size(&source_batch);
    let max_staging_bytes = source_bytes.checked_sub(1).expect("UTF-8 source batch must retain host bytes");
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;

    let error = stage_cugraph_inputs(
        &spec,
        edge_source,
        vec![],
        task_context,
        &ExecutionCancellation::new(),
        max_staging_bytes,
    )
    .await
    .expect_err("UTF-8 literal source batch larger than the island-wide cap must be rejected");
    let nexus = error.find_nexus_error().expect("literal source staging cap failure must remain structured");
    let source_bytes = source_bytes.to_string();
    let max_staging_bytes = max_staging_bytes.to_string();

    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.fact_value("source_kind"), Some("source_vertices"));
    assert_eq!(nexus.fact_value("staged_bytes"), Some("0"));
    assert_eq!(nexus.fact_value("next_batch_bytes"), Some(source_bytes.as_str()));
    assert_eq!(nexus.fact_value("max_staging_bytes"), Some(max_staging_bytes.as_str()));
    assert_eq!(runtime.memory_pool.reserved(), 0, "failed literal staging must release its reservation");
}

#[tokio::test]
async fn cugraph_producer_preparation_charges_utf8_relation_sources_to_the_same_input_cap() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let mut spec = bfs_utf8_spec();
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("test spec is BFS") };
    params.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new(
        CugraphTableReference::parse("source_vertices").unwrap(),
        "source_vertex",
    ));
    let source_schema = Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Utf8, false)]));
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
    let nexus = error.find_nexus_error().expect("relation source staging cap failure must remain structured");
    let source_bytes = source_bytes.to_string();
    let max_staging_bytes = max_staging_bytes.to_string();

    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(nexus.fact_value("source_kind"), Some("side_input"));
    assert_eq!(nexus.fact_value("staged_bytes"), Some("0"));
    assert_eq!(nexus.fact_value("next_batch_bytes"), Some(source_bytes.as_str()));
    assert_eq!(nexus.fact_value("max_staging_bytes"), Some(max_staging_bytes.as_str()));
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

    exec.record_bfs_singleflight_gpu_follower_metrics(0, &key, &batches, Some(0), std::time::Duration::ZERO);
    exec.record_bfs_singleflight_gpu_result_follower_metrics(0, &key, 0, 0, std::time::Duration::ZERO);
    let host_metrics = NexusNativeMetrics::new(&exec.metrics, 0);
    exec.record_bfs_singleflight_host_follower_metrics(&host_metrics, &key, &batches, std::time::Duration::ZERO);

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
    let base = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);

    assert!(base.has_query_admission_authority());
    assert_eq!(base.query_admission_ticket_id(), None);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, base)
        .expect("cuGraph exec should build without a pre-hydration ticket");

    assert_eq!(exec.native_execution_config.query_admission_ticket_id(), None);
}

#[test]
fn cugraph_algo_exec_uses_cold_graph_model_as_immutable_grant_target() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024_u64;
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);

    let exec = graph_ticket_after_metadata(
        CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, config)
            .expect("cuGraph exec should build before metadata hydration"),
    );
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("graph model should derive opaque admission evidence")
        .expect("known edge rows should produce graph model evidence");
    let expected_proof = graph_ticket_proof(&exec)
        .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
        .expect("exact graph evidence must match the graph proof scope");

    assert!(
        model_ask.cold_model_requested_bytes() < want,
        "small graph estimate should not default to the controller hint"
    );
    assert!(
        model_ask.warm_model_requested_bytes() < model_ask.cold_model_requested_bytes(),
        "warm model must omit graph construction bytes"
    );
    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&expected_proof),
        "the ticket must carry the exact cuGraph execution proof"
    );
    assert_eq!(
        ticket
            .attempt_capability_proof()
            .expect("cuGraph ticket retains its engine-issued proof")
            .soft_working_set_hint_bytes(),
        Some(model_ask.cold_model_requested_bytes()),
        "the cold model may target the immutable grant but cannot alter its ledger-owned fit range"
    );
}

#[test]
fn cugraph_algo_exec_registers_frontend_neutral_execution_spis() {
    let config = NexusNativeExecutionConfig::bounded(64 * 1024 * 1024, 512, 1, 3, 1);
    let exec = CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, config)
        .expect("cuGraph exec should register its execution SPIs");
    let plan = &exec as &dyn ExecutionPlan;

    let producer = crate::native_exec::producer::gpu_resident_producer(plan)
        .expect("registered producer discovery should recognize cuGraph execution");
    assert_eq!(producer.gpu_result_input_name(), exec.gpu_result_input_name);

    let graph = crate::native_exec::graph::graph_execution_plan(plan)
        .expect("registered graph admission discovery should recognize cuGraph execution");
    assert_eq!(graph.graph_result_input_name(), exec.gpu_result_input_name);
    assert!(graph.graph_execution_capability_proof().is_ok());
}

#[test]
fn cugraph_graph_model_ask_prefers_exact_edge_rows_over_configured_raw_edges() {
    let min = 64 * 1024 * 1024;
    let config = NexusNativeExecutionConfig::bounded(min, 512, 1, 3, 1);
    let stale_config = config.clone().with_graph_memory_calibration_override(
        nexus_query_engine::graph_memory::GraphMemoryCalibrationOverride::none().with_raw_input_edges(30_000),
    );
    let edge_source = memory_edge_source(300);

    let exact_exec = CugraphAlgoExec::try_new(triangle_spec(), Arc::clone(&edge_source), vec![], None, None, config)
        .expect("triangle exec should build");
    let exact_ask =
        exact_exec.graph_model_ask_bytes_for_admission_ticket().unwrap().expect("exact rows should produce graph ask");
    let stale_exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, stale_config)
        .expect("triangle exec with stale override should build");
    let stale_override_ask = stale_exec
        .graph_model_ask_bytes_for_admission_ticket()
        .unwrap()
        .expect("exact rows should win over stale override");

    assert_eq!(stale_override_ask, exact_ask);
    assert!(exact_ask.warm_model_requested_bytes() < exact_ask.cold_model_requested_bytes());
}

#[test]
fn cugraph_raw_bfs_ticket_admits_with_the_owner_sort_workspace_preflight_bound() {
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config =
        NexusNativeExecutionConfig::bounded(64 * 1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);

    let exec = CugraphAlgoExec::try_new(bfs_spec(), memory_edge_source(300), vec![], None, None, config)
        .expect("raw BFS exec should build");
    let ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("owner preflight must let raw BFS derive bounded admission evidence")
        .expect("exact edge rows plus the owner bound must produce a raw-BFS ask");

    assert!(ask.cold_model_requested_bytes() > 0);
}

#[test]
fn cugraph_algo_exec_uses_native_edge_cold_model_as_immutable_grant_target() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024_u64;
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = native_edge_source(vec![100, 200], &config);

    let exec = graph_ticket_after_metadata(
        CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build before metadata hydration"),
    );
    let expected_proof = graph_ticket_proof_for_native_source(&exec);
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("graph model should derive opaque admission evidence")
        .expect("native edge-source rows should produce graph model evidence");
    let expected_proof = expected_proof
        .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
        .expect("exact graph evidence must match the graph proof scope");

    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&expected_proof),
        "the graph ticket must also authorize the native edge fragment it executes"
    );
    assert!(
        model_ask.cold_model_requested_bytes() < want,
        "native edge-source graph estimate should not default to the controller hint"
    );
    assert_eq!(
        ticket
            .attempt_capability_proof()
            .expect("the graph ticket retains its engine-issued proof")
            .soft_working_set_hint_bytes(),
        Some(model_ask.cold_model_requested_bytes())
    );
}

#[test]
fn cugraph_algo_exec_records_graph_residency_for_native_edge_source() {
    let config = NexusNativeExecutionConfig::bounded_default();
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
    let config = NexusNativeExecutionConfig::bounded_default();
    let edge_source = native_graph_residency_edge_source(&config);
    let mut spec = bfs_spec();
    spec.graph_options.construction_policy = CugraphGraphConstructionPolicy::PythonCugraph;

    let exec = CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, config)
        .expect("native edge wrapper with a footer-proven topology should plan");

    assert!(
        !exec.edge_normalize_inserted(),
        "native lowering must retain the source-owned simple-directed proof through its edge wrapper"
    );
}

#[tokio::test]
async fn cugraph_algo_exec_records_graph_residency_for_datafusion_parquet_edge_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let config = NexusNativeExecutionConfig::bounded_default();
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
        .downcast_ref::<datafusion::datasource::source::DataSourceExec>()
        .expect("parallel Parquet fixture should remain a direct DataFusion scan");
    let native_spec = native_graph_spec_for_residency(&pagerank_spec(), false);

    let descriptor = crate::native::graph_residency_descriptor_for_datafusion_local_parquet_scan(source, &native_spec)
        .expect("parallel local-Parquet scan should retain a stable graph rowset descriptor");

    assert!(descriptor.source_fingerprint.as_str().starts_with("graph-source-v1-fnv64-"));
    assert_eq!(descriptor.vertex_data_type, DataType::Int64);
    assert_eq!(descriptor.weight_data_type, Some(DataType::Float64));
    assert_eq!(
        crate::native::exact_row_count_for_datafusion_local_parquet_scan(source),
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
        .downcast_ref::<datafusion::datasource::source::DataSourceExec>()
        .expect("single-file fixture should remain a direct DataFusion scan");

    assert_eq!(
        crate::native::graph_vertex_upper_bound_for_datafusion_local_parquet_scan(source),
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

    let exec =
        CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, NexusNativeExecutionConfig::bounded_default())
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

    let exec =
        CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, NexusNativeExecutionConfig::bounded_default())
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
    let config = NexusNativeExecutionConfig::bounded_default();
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
    let config = NexusNativeExecutionConfig::bounded_default();
    let edge_source = datafusion_parquet_edge_source_sql(&path, "SELECT src, dst FROM edges").await;

    let exec = CugraphAlgoExec::try_new(bfs_utf8_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph logical BFS exec should build for DataFusion local-Parquet edge source");
    assert!(exec.graph_residency.is_none());
}

#[tokio::test]
async fn cugraph_algo_exec_uses_footer_rows_as_a_graph_admission_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = datafusion_parquet_edge_source(&path).await;

    let exec = graph_ticket_after_metadata(
        CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build before metadata hydration"),
    );
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("footer rows should derive graph admission evidence")
        .expect("keyable local-Parquet source has exact graph rows");
    assert_eq!(
        ticket
            .attempt_capability_proof()
            .expect("cuGraph ticket retains its engine-issued proof")
            .soft_working_set_hint_bytes(),
        Some(model_ask.cold_model_requested_bytes()),
        "the direct DataFusion scan's host staging is not a second device owner beside the graph model"
    );
}

#[tokio::test]
async fn cugraph_algo_exec_uses_footer_rows_for_cacheable_bfs_admission() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = datafusion_parquet_edge_source(&path).await;

    let exec = graph_ticket_after_metadata(
        CugraphAlgoExec::try_new(bfs_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph BFS exec should build before metadata hydration"),
    );
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("footer rows should derive graph admission evidence")
        .expect("keyable local-Parquet source has exact graph rows");
    assert_eq!(
        ticket
            .attempt_capability_proof()
            .expect("cacheable BFS ticket retains its engine-issued proof")
            .soft_working_set_hint_bytes(),
        Some(model_ask.cold_model_requested_bytes()),
        "the graph cold peak already includes its prepared GPU edge frame"
    );
}

#[tokio::test]
async fn cugraph_sql_bfs_table_function_records_graph_residency_for_datafusion_parquet_edge_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let config = SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let state = datafusion::execution::SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let plan = ctx
        .sql(
            "SELECT COUNT(*) AS reached \
             FROM cugraph_bfs('citation_edges', 0, 'src', 'dst', NULL, \
             '{\"depth_limit\": 3, \"output_mode\": \"normalized\"}')",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let exec = cugraph_exec_in_plan(Arc::clone(&plan)).expect("planned SQL should contain CugraphAlgoExec");
    let exec = exec.downcast_ref::<CugraphAlgoExec>().unwrap();
    let descriptor =
        exec.graph_residency.as_ref().expect("stress-shaped SQL local-Parquet edge source should be keyable");
    let edge_schema = exec.edge_source.schema();
    assert!(graph_residency_matches_edge_schema(descriptor, edge_schema.as_ref(), &exec.spec));
}

#[tokio::test]
async fn installed_backend_datafusion_parquet_graph_ticket_keeps_post_hydration_model_hint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet_with_graph_vertex_upper_bound(&path, Some(3));

    let backend = test_gpu_backend();
    let state = backend
        .backend()
        .install_on_with_cugraph_sql(
            datafusion::execution::SessionStateBuilder::new_with_default_features(),
            CugraphSqlConfig::default(),
        )
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
    let invocation = attempt.planning_context(&ctx).unwrap();

    let plan = invocation
        .sql(
            "SELECT COUNT(*) AS vertices, SUM(x) AS sum_x, SUM(y) AS sum_y \
             FROM cugraph_force_atlas2('citation_edges', 'src', 'dst', 'weight', \
             '{\"max_iter\": 20, \"seed\": 7, \"barnes_hut_optimize\": true}')",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let exec = cugraph_exec_in_plan(plan).expect("installed backend plan should contain CugraphAlgoExec");
    let exec = exec.downcast_ref::<CugraphAlgoExec>().unwrap();

    let planner_ticket = invocation
        .task_ctx()
        .session_config()
        .options()
        .extensions
        .get::<NexusAttemptExtension>()
        .and_then(NexusAttemptExtension::bound_ticket);
    assert!(
        planner_ticket.is_none(),
        "planning a footer-dependent graph root must not bind a stale pre-hydration ticket; hint={:?}",
        planner_ticket
            .as_ref()
            .and_then(|ticket| ticket.attempt_capability_proof())
            .and_then(nexus_query_engine::admission::QueryPlanCapabilityProof::soft_working_set_hint_bytes)
    );
    assert!(
        exec.native_execution_config.query_admission_ticket().is_none(),
        "footer-dependent graph tickets must not survive planning"
    );

    let (prepared, _staged) = exec.clone().prepare_staged_attempt(invocation.task_ctx()).await.unwrap();
    assert_eq!(
        prepared.graph_residency.as_ref().and_then(GraphResidencyDescriptor::graph_vertex_upper_bound),
        Some(3),
        "the runtime edge provenance must retain the same footer vertex bound that prices its admission ticket"
    );
    let model_ask = prepared
        .graph_model_ask_bytes_for_admission_ticket()
        .unwrap()
        .expect("hydrated DataFusion Parquet edge source must retain exact graph model evidence");
    let ticket = prepared
        .native_execution_config
        .query_admission_ticket()
        .expect("metadata hydration must mint the graph-aware ticket");
    let proof = ticket.attempt_capability_proof().expect("graph-aware ticket must retain its capability proof");

    assert!(ticket.is_attempt_bound(), "the hydrated graph ticket must replace the attempt's deferred placeholder");
    assert_eq!(edge_source_output_rows(prepared.edge_source.as_ref()).unwrap(), Some(3));
    assert_eq!(
        edge_source_graph_vertex_estimate(prepared.edge_source.as_ref()),
        Some(GraphVertexEstimate::graph_metadata(3))
    );
    let source_plan = lowered_native_edge_plan(prepared.edge_source.as_ref())
        .expect("metadata hydration must lower the DataFusion edge scan to its native source fragment");
    let source_proof = nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_input_plan(
        &source_plan.clone_with_gpu_resident_sink(),
    )
    .expect("the resolved native edge source must retain its source-read capability proof");
    let expected_proof =
        nexus_query_engine::admission::QueryPlanCapabilityProof::combine(graph_ticket_proof(&prepared), [source_proof])
            .expect("the graph root and resolved native edge source must retain one combined proof")
            .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
            .expect("exact graph evidence must match the graph proof scope");
    assert_eq!(
        proof, &expected_proof,
        "the post-hydration ticket must retain the exact graph and resolved source proofs with phase-aware model evidence"
    );
}

/// The S2 G1 query aggregates the ForceAtlas2 output on the native side. Keep
/// that exact terminal shape covered separately from ticket planning: a graph
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
        .install_on_with_cugraph_sql(
            datafusion::execution::SessionStateBuilder::new_with_default_features(),
            CugraphSqlConfig::default(),
        )
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
    let stream = attempt
        .sql(
            &ctx,
            "SELECT COUNT(*) AS vertices, SUM(x) AS sum_x, SUM(y) AS sum_y \
             FROM cugraph_force_atlas2('citation_edges', 'src', 'dst', 'weight', \
             '{\"max_iter\": 20, \"seed\": 7, \"barnes_hut_optimize\": true}')",
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
        matches!(
            report.disposition,
            nexus_query_engine::observability::AttemptDisposition::Native
                | nexus_query_engine::observability::AttemptDisposition::PartialNative
        ),
        "ForceAtlas2 aggregate must retain a native final plan: {:?}",
        report.disposition
    );
    assert!(
        matches!(report.outcome, nexus_query_engine::observability::RuntimeTerminalOutcome::Succeeded),
        "ForceAtlas2 aggregate must seal a success: {:?}",
        report.outcome
    );
}

/// Host-only diagnostic for the qualification DBLP fixture.  It is deliberately
/// ignored in normal unit runs because the full fixture is many GiB, but it
/// follows the same backend-installed table-function route as the stress
/// runner and proves that its immutable ticket retains the full graph model.
#[tokio::test]
#[ignore = "requires the canonical DBLP V12 fixture on the qualification host"]
async fn qualification_dblp_force_atlas2_ticket_uses_full_footer_metadata() {
    const DBLP_PARQUET_ROOT: &str = "/opt/dlami/nvme/fixture/data/demo-v12-footer-r1/citation_network/v1/parquet";
    let backend = test_gpu_backend();
    let state = backend
        .backend()
        .install_on_with_cugraph_sql(
            datafusion::execution::SessionStateBuilder::new_with_default_features(),
            CugraphSqlConfig::default(),
        )
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    nexus_workloads::graph::register_dblp_tables(&ctx, std::path::Path::new(DBLP_PARQUET_ROOT)).await.unwrap();
    let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
    let invocation = attempt.planning_context(&ctx).unwrap();

    let plan = invocation
        .sql(
            "SELECT COUNT(*) AS vertices, SUM(x) AS sum_x, SUM(y) AS sum_y \
             FROM cugraph_force_atlas2('edges_by_src', 'src', 'dst', 'weight', \
             '{\"max_iter\": 200, \"seed\": 20260804, \"barnes_hut_optimize\": true}')",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let exec = cugraph_exec_in_plan(plan).expect("qualification query should contain CugraphAlgoExec");
    assert!(
        invocation
            .task_ctx()
            .session_config()
            .options()
            .extensions
            .get::<NexusAttemptExtension>()
            .and_then(NexusAttemptExtension::bound_ticket)
            .is_none(),
        "the qualification graph plan must defer admission until its footer evidence is hydrated"
    );
    let (prepared, _staged) = exec
        .downcast_ref::<CugraphAlgoExec>()
        .unwrap()
        .clone()
        .prepare_staged_attempt(invocation.task_ctx())
        .await
        .unwrap();
    assert_eq!(
        prepared.graph_residency.as_ref().and_then(GraphResidencyDescriptor::graph_vertex_upper_bound),
        Some(4_894_081),
        "the qualification runtime edge provenance must retain the exact source-bound vertex model"
    );
    let model_ask = prepared
        .graph_model_ask_bytes_for_admission_ticket()
        .unwrap()
        .expect("full DBLP footer evidence must produce a graph model ask");
    let ticket = prepared.native_execution_config.query_admission_ticket().unwrap();
    let proof = ticket.attempt_capability_proof().unwrap();

    assert!(ticket.is_attempt_bound(), "the qualification graph ticket must bind directly to its attempt");
    assert_eq!(edge_source_output_rows(prepared.edge_source.as_ref()).unwrap(), Some(45_564_149));
    assert_eq!(
        edge_source_graph_vertex_estimate(prepared.edge_source.as_ref()),
        Some(GraphVertexEstimate::graph_metadata(4_894_081))
    );
    assert_eq!(
        proof.soft_working_set_hint_bytes(),
        Some(model_ask.cold_model_requested_bytes()),
        "the graph model owns the direct source upload's device lifetime"
    );
    assert!(
        model_ask.cold_model_requested_bytes() > 424_258_722,
        "full ForceAtlas2 admission model must dominate the failed source-stage-only grant"
    );
}

#[tokio::test]
async fn cugraph_sql_bfs_table_function_records_graph_residency_with_runner_default_session_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let state = datafusion::execution::SessionStateBuilder::new()
        .with_default_features()
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let plan = ctx
        .sql(
            "SELECT COUNT(*) AS reached \
             FROM cugraph_bfs('citation_edges', 0, 'src', 'dst', NULL, \
             '{\"depth_limit\": 3, \"output_mode\": \"normalized\"}')",
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
        "the P6 runner uses the default session configuration, so its DBLP BFS path must remain cacheable"
    );
}

fn cugraph_exec_in_plan(plan: Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.downcast_ref::<CugraphAlgoExec>().is_some() {
        return Some(plan);
    }
    plan.children().into_iter().find_map(|child| cugraph_exec_in_plan(Arc::clone(child)))
}

#[test]
fn graph_residency_schema_guard_rejects_descriptor_dtype_mismatch() {
    let config = NexusNativeExecutionConfig::bounded_default();
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
fn cugraph_algo_exec_keeps_keyable_native_affinity_out_of_the_admission_ticket() {
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = native_graph_residency_edge_source(&config);

    let exec = graph_ticket_after_metadata(
        CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build before metadata hydration"),
    );
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("keyable native edge source should derive graph admission evidence")
        .expect("exact native edge rows should produce graph model evidence");
    let expected_proof = graph_ticket_proof_for_native_source(&exec)
        .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
        .expect("exact graph evidence must match the graph proof scope");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&expected_proof),
        "cache affinity must not remove the native edge capability proof"
    );
}

#[test]
fn cugraph_algo_exec_uses_the_same_proof_for_unkeyable_edge_source() {
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);

    let exec = graph_ticket_after_metadata(
        CugraphAlgoExec::try_new(pagerank_spec(), memory_edge_source(100), vec![], None, None, config)
            .expect("cuGraph exec should build before metadata hydration"),
    );
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("unkeyable edge source should derive graph admission evidence")
        .expect("exact in-memory edge rows should produce graph model evidence");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");

    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(
            &graph_ticket_proof(&exec)
                .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
                .expect("exact graph evidence must match the graph proof scope"),
        )
    );
}

#[test]
fn cugraph_algo_exec_aggregates_proof_for_non_light_native_edge_source() {
    let min = 64 * 1024 * 1024;
    let backend = test_gpu_backend();
    let source = admission_source(&backend);
    let config = NexusNativeExecutionConfig::bounded(min, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_graph_memory_calibration_override(
            nexus_query_engine::graph_memory::GraphMemoryCalibrationOverride::none().with_raw_input_edges(300),
        );
    let edge_source = sorted_native_edge_source(vec![100, 200], &config);

    let exec = graph_ticket_after_metadata(
        CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build before metadata hydration"),
    );
    let expected_proof = graph_ticket_proof_for_native_source(&exec);
    let ticket =
        exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should still mint a plan ticket");

    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&expected_proof),
        "native preprocessing must be covered by the ticket that executes it"
    );
}

#[test]
fn standalone_bounded_cugraph_rejects_native_child_execution_without_admission() {
    let config = NexusNativeExecutionConfig::bounded(2048, 512, 1, 3, 1);
    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_exec =
        Arc::new(NexusNativeExec::try_new(native_plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>;

    let error = validate_standalone_bounded_cugraph_sources(&config, native_exec.as_ref(), &[])
        .expect_err("standalone bounded cuGraph must not execute a native edge child without admission");
    let nexus = error.find_nexus_error().expect("missing backend rejection remains structured");
    assert_eq!(nexus.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(nexus.fact_value("reason"), Some("gpu_backend_required"));
    assert_eq!(nexus.fact_value("remediation"), Some("install_nexus_gpu_backend"));
    assert_eq!(nexus.fact_value("source_kind"), Some("edge"));

    let repartitioned = Arc::new(RepartitionExec::try_new(native_exec, Partitioning::UnknownPartitioning(1)).unwrap())
        as Arc<dyn ExecutionPlan>;
    let error = validate_standalone_bounded_cugraph_sources(&config, repartitioned.as_ref(), &[])
        .expect_err("fallback collect over native children should be rejected");
    let nexus = error.find_nexus_error().expect("nested missing backend rejection remains structured");
    assert_eq!(nexus.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(nexus.fact_value("reason"), Some("gpu_backend_required"));
    assert_eq!(nexus.fact_value("remediation"), Some("install_nexus_gpu_backend"));
    assert_eq!(nexus.fact_value("source_kind"), Some("edge"));
}

#[test]
fn admission_backed_cugraph_allows_native_child_execution() {
    let backend = test_gpu_backend();
    let config =
        NexusNativeExecutionConfig::bounded(2048, 512, 1, 3, 1).with_query_admission_source(admission_source(&backend));
    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_exec =
        Arc::new(NexusNativeExec::try_new(native_plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>;
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
    let config = NexusNativeExecutionConfig::bounded_default().with_lakehouse_footer_pruning(true);
    let native_exec = native_edge_source(vec![100, 200], &config);
    let wrapped = Arc::new(
        RepartitionExec::try_new(native_exec, Partitioning::UnknownPartitioning(1))
            .expect("test wrapper should accept the native child"),
    ) as Arc<dyn ExecutionPlan>;

    let prepared = prepare_cugraph_source_metadata(wrapped, Arc::new(TaskContext::default()))
        .await
        .expect("metadata preparation should succeed");

    assert!(prepared.downcast_ref::<RepartitionExec>().is_some(), "host wrapper must be preserved");
    let children = prepared.children();
    let native = children[0]
        .downcast_ref::<NexusNativeExec>()
        .expect("nested native child should remain under the host wrapper");
    assert!(!native.execution_config().lakehouse_footer_pruning());
}
