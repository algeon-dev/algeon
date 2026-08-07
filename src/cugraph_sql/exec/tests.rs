use std::{fs::File, sync::Arc};

use arrow::{
    array::{ArrayRef, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    execution::{context::TaskContext, runtime_env::RuntimeEnvBuilder},
    logical_expr::Operator,
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{collect, empty::EmptyExec, filter::FilterExec},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
#[cfg(feature = "server")]
use nexus_query_engine::observability::RuntimeObservationConfig;
use nexus_query_engine::{
    expr::{Expr, SortExpr, SortSpec},
    plan::ProjectionOutput,
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
        CugraphBfsOutputOptions, CugraphBfsParams, CugraphGraphOptions, CugraphGraphPredicateTable,
        CugraphPageRankParams, CugraphSqlConfig, CugraphTableReference, CugraphTriangleCountAllParams,
        result_schema_for_params,
    },
    error::{DataFusionErrorExt, ErrorCode},
    native_exec::query_admission_ticket_from_task_context,
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
    Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]))
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
    exec.issue_graph_aware_ticket_after_metadata(&TaskContext::default())
        .expect("prepared cuGraph metadata should mint one graph-aware admission ticket")
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
    let backend = test_gpu_backend();
    let config = backend_execution_config(&backend).with_query_service_handle(Arc::new(
        backend.admit_query().expect("test query must be admitted by the backend"),
    ));
    let context_lease =
        config.execution_context_async(Arc::clone(&task_context)).await.expect("backend-admitted graph context builds");
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
    assert!(runtime.memory_pool.reserved() > 0, "staged host batches must be reserved before upload");

    let uploaded = staged_inputs
        .upload(context_lease.context(), &mut graph_metrics, task_context)
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
    let expected_proof =
        graph_ticket_proof(&exec).with_graph_memory_admission_evidence(model_ask.cold_model_evidence());

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

    let exact_ask = cugraph_graph_model_ask_bytes(&triangle_spec(), edge_source.as_ref(), true, &config)
        .unwrap()
        .expect("exact rows should produce graph ask");
    let stale_override_ask = cugraph_graph_model_ask_bytes(&triangle_spec(), edge_source.as_ref(), true, &stale_config)
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

    let ask = cugraph_graph_model_ask_bytes(&bfs_spec(), memory_edge_source(300).as_ref(), false, &config)
        .expect("owner preflight must let raw BFS derive bounded admission evidence")
        .expect("exact edge rows plus the owner bound must produce a raw-BFS ask");

    assert!(ask.cold_model_requested_bytes() > 0);
}

#[test]
fn cugraph_raw_bfs_ticket_still_refuses_when_the_owner_preflight_is_unavailable() {
    let native_spec = native_graph_spec_for_memory_estimate(&bfs_spec(), false)
        .expect("BFS without predicate inputs has a native memory model");
    let error = graph_memory_bounded_admission_evidence_for_edge_list_with_vertex_estimate(
        &native_spec,
        &edge_schema(),
        300,
        Some(300),
        None,
        None,
        GraphMemoryCalibrationOverride::none(),
        None,
    )
    .expect_err("raw BFS must not mint bounded evidence without the owner preflight bound");

    assert_eq!(error.fact_value("reason"), Some("raw_bfs_sort_workspace_preflight_unavailable"));
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
    let expected_proof = expected_proof.with_graph_memory_admission_evidence(model_ask.cold_model_evidence());

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

    let prepared = exec
        .clone()
        .prepare_lakehouse_source_metadata(invocation.task_ctx())
        .await
        .unwrap()
        .issue_graph_aware_ticket_after_metadata(invocation.task_ctx().as_ref())
        .unwrap();
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
            .with_graph_memory_admission_evidence(model_ask.cold_model_evidence());
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
    let prepared = exec
        .downcast_ref::<CugraphAlgoExec>()
        .unwrap()
        .clone()
        .prepare_lakehouse_source_metadata(invocation.task_ctx())
        .await
        .unwrap()
        .issue_graph_aware_ticket_after_metadata(invocation.task_ctx().as_ref())
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
        .with_graph_memory_admission_evidence(model_ask.cold_model_evidence());
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
        Some(&graph_ticket_proof(&exec).with_graph_memory_admission_evidence(model_ask.cold_model_evidence()))
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
