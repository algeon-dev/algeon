use std::{fs::File, sync::Arc};
#[cfg(feature = "server")]
use std::{sync::Mutex, time::Duration};

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
use nexus_query_engine::{
    expr::{Expr, SortExpr, SortSpec},
    plan::ProjectionOutput,
    runtime::ExecutionCancellation,
    source::{LiteralBatchSource, LocalParquetFile, LocalParquetSource},
};
#[cfg(feature = "server")]
use nexus_query_engine::{
    observability::{
        PlanExecutionBoundaryKind, RuntimeObservationConfig, RuntimeObservationPayload, RuntimeObservationSession,
        RuntimeTerminalOutcome, SealedRuntimeObservationSnapshot,
    },
    runtime::LakehouseCacheFillEventSnapshot,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use super::*;
#[cfg(feature = "server")]
use crate::native_exec::admission::QueryAdmissionFacts;
#[cfg(feature = "server")]
use crate::native_exec::{NativeMetricsCapture, NativeMetricsCaptureSink};
use crate::{
    backend::InstalledTestGpuBackend,
    cugraph_sql::{
        CugraphBfsOutputOptions, CugraphBfsParams, CugraphGraphOptions, CugraphGraphPredicateTable,
        CugraphPageRankParams, CugraphSqlConfig, CugraphTableReference, CugraphTriangleCountAllParams,
        result_schema_for_params,
    },
    error::{DataFusionErrorExt, ErrorCode},
    native_exec::{
        QueryAdmissionSource,
        admission::{TestAdmissionController, TestAdmissionDeviceProfile},
        task_context_with_query_admission_ticket, wrap_plan_with_query_admission_ticket,
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

fn admission_only_source() -> QueryAdmissionSource {
    admission_source_with_requirement(1024 * 1024, 1024 * 1024, 1024 * 1024)
}

fn admission_source_with_requirement(floor_bytes: u64, hint_bytes: u64, capacity_bytes: u64) -> QueryAdmissionSource {
    let controller = TestAdmissionController::new(
        TestAdmissionDeviceProfile::new(0, capacity_bytes, 1, floor_bytes),
        Some(hint_bytes),
    )
    .expect("CUDA-free test admission controller should build");
    controller.source()
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
    let mut writer = ArrowWriter::try_new(
        file,
        schema,
        Some(WriterProperties::builder().set_max_row_group_row_count(Some(2)).build()),
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
        .with_row_group_rows(vec![100, 200]);
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
    let config = backend_execution_config(&backend).with_plan_query_admission_ticket();
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

#[cfg(feature = "server")]
#[derive(Default)]
struct GraphCaptureSink {
    query_metrics: Mutex<Vec<QueryMetrics>>,
}

#[cfg(feature = "server")]
impl NativeMetricsCaptureSink for GraphCaptureSink {
    fn record_admission_facts(&self, _query_id: &str, _facts: QueryAdmissionFacts) {}

    fn record_query_metrics(&self, _query_id: &str, metrics: &QueryMetrics) {
        self.query_metrics.lock().unwrap().push(metrics.clone());
    }

    fn record_lakehouse_cache_fill_events(&self, _query_id: &str, _events: &[LakehouseCacheFillEventSnapshot]) {}
}

#[cfg(feature = "server")]
fn native_literal_edge_source(config: &NexusNativeExecutionConfig, utf8: bool) -> Arc<dyn ExecutionPlan> {
    let (schema, columns): (Arc<Schema>, Vec<ArrayRef>) = if utf8 {
        (
            string_edge_schema(),
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob", "carol"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["bob", "carol", "alice"])) as ArrayRef,
            ],
        )
    } else {
        (
            edge_schema(),
            vec![
                Arc::new(Int64Array::from(vec![0, 1, 2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2, 0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0])) as ArrayRef,
            ],
        )
    };
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    Arc::new(NexusNativeExec::try_new(builder.finish_with_host_sink(source), config.clone()).unwrap())
        as Arc<dyn ExecutionPlan>
}

#[cfg(feature = "server")]
fn observed_graph_exec(
    query_id: &'static str,
    spec: CugraphAlgorithmSpec,
    cancellation: ExecutionCancellation,
) -> (InstalledTestGpuBackend, Arc<CugraphAlgoExec>, Arc<GraphCaptureSink>) {
    let backend = test_gpu_backend();
    let sink = Arc::new(GraphCaptureSink::default());
    let erased_sink: Arc<dyn NativeMetricsCaptureSink> = sink.clone();
    let config = backend_execution_config(&backend)
        .with_cancellation(cancellation)
        .with_metrics_capture(NativeMetricsCapture::new(query_id, erased_sink))
        .with_plan_query_admission_ticket();
    let edge_source = native_literal_edge_source(&config, false);
    let exec = Arc::new(
        CugraphAlgoExec::try_new(spec, edge_source, vec![], None, None, config).expect("observed graph exec builds"),
    );
    (backend, exec, sink)
}

#[cfg(feature = "server")]
fn observed_graph_exec_with_edge_source(
    query_id: &'static str,
    spec: CugraphAlgorithmSpec,
    cancellation: ExecutionCancellation,
    edge_source: Arc<dyn ExecutionPlan>,
    host_boundary_config: GpuIslandHostBoundaryConfig,
) -> (InstalledTestGpuBackend, Arc<CugraphAlgoExec>, Arc<GraphCaptureSink>) {
    let backend = test_gpu_backend();
    let sink = Arc::new(GraphCaptureSink::default());
    let erased_sink: Arc<dyn NativeMetricsCaptureSink> = sink.clone();
    let config = backend_execution_config(&backend)
        .with_cancellation(cancellation)
        .with_metrics_capture(NativeMetricsCapture::new(query_id, erased_sink))
        .with_plan_query_admission_ticket();
    let exec = Arc::new(
        CugraphAlgoExec::try_new_with_host_boundary_config(
            spec,
            edge_source,
            vec![],
            None,
            None,
            config,
            host_boundary_config,
        )
        .expect("observed graph exec builds"),
    );
    (backend, exec, sink)
}

#[cfg(feature = "server")]
fn admitted_observed_graph_plan(exec: &Arc<CugraphAlgoExec>) -> Arc<dyn ExecutionPlan> {
    let ticket = exec
        .native_execution_config
        .query_admission_ticket()
        .expect("observed graph execution owns a plan admission ticket");
    wrap_plan_with_query_admission_ticket(Arc::clone(exec) as Arc<dyn ExecutionPlan>, ticket)
}

#[cfg(feature = "server")]
fn close_observed_graph_backend(backend: &InstalledTestGpuBackend) {
    backend.backend().close_and_wait(Duration::from_secs(10)).expect("observed graph backend closes after execution");
}

#[cfg(feature = "server")]
fn graph_plan_boundary_count(
    snapshot: &SealedRuntimeObservationSnapshot,
    expected: PlanExecutionBoundaryKind,
) -> usize {
    snapshot
        .events()
        .iter()
        .filter(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::PlanExecutionBoundary { boundary } if *boundary == expected
            )
        })
        .count()
}

#[cfg(feature = "server")]
fn sealed_graph_stage(query_id: &str, outcome: RuntimeTerminalOutcome) -> SealedRuntimeObservationSnapshot {
    RuntimeObservationSession::new(query_id, None, RuntimeObservationConfig::bounded(8, 4096))
        .seal(outcome)
        .expect("enabled graph-stage observation seals")
}

#[cfg(feature = "server")]
#[test]
fn graph_execution_accumulator_merges_stages_and_publishes_exactly_once() {
    let sink = Arc::new(GraphCaptureSink::default());
    let erased_sink: Arc<dyn NativeMetricsCaptureSink> = sink.clone();
    let capture = NativeMetricsCapture::new("graph-stage-merge", erased_sink).new_execution();
    let owner = capture.clone();
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new_with_capture(&metrics_set, 0, Some(capture));
    let mut graph_metrics = GraphExecutionMetrics::new(&native_metrics, RuntimeObservationConfig::native_execution());

    let mut edge = QueryMetrics::default().with_query_id("graph-stage-merge");
    edge.input_batches = 2;
    edge.input_rows = 7;
    edge.peak_device_memory_bytes = Some(90);
    edge.max_task_reservation_bytes = 120;
    edge.record_task_execution("shared-task", "graph_stage");
    edge.record_task_elapsed("shared-task", "graph_stage", Duration::from_nanos(5));
    edge.task_runtime_metrics.get_mut("shared-task").unwrap().max_observed_peak_bytes = Some(70);
    edge.task_operator_metrics.get_mut("graph_stage").unwrap().max_observed_peak_bytes = Some(80);

    let mut graph = QueryMetrics::default().with_query_id("graph-stage-merge");
    graph.input_batches = 3;
    graph.input_rows = 11;
    graph.peak_device_memory_bytes = Some(40);
    graph.max_task_reservation_bytes = 200;
    graph.record_task_execution("shared-task", "graph_stage");
    graph.record_task_elapsed("shared-task", "graph_stage", Duration::from_nanos(7));
    graph.task_runtime_metrics.get_mut("shared-task").unwrap().max_observed_peak_bytes = Some(130);
    graph.task_operator_metrics.get_mut("graph_stage").unwrap().max_observed_peak_bytes = Some(60);

    graph_metrics
        .merge_stage_evidence(edge, Some(sealed_graph_stage("graph-stage-merge", RuntimeTerminalOutcome::Succeeded)));
    graph_metrics
        .merge_stage_evidence(graph, Some(sealed_graph_stage("graph-stage-merge", RuntimeTerminalOutcome::Succeeded)));

    let accumulated = graph_metrics.accumulated().clone();
    assert_eq!(accumulated.input_batches, 5, "additive fields sum across stages");
    assert_eq!(accumulated.input_rows, 18, "additive fields sum across stages");
    assert_eq!(accumulated.peak_device_memory_bytes, Some(90), "query peaks take max, not sum");
    assert_eq!(accumulated.max_task_reservation_bytes, 200, "task maxima take max, not sum");
    assert_eq!(accumulated.task_runtime_metrics.len(), 1, "task-id map rows merge by key");
    assert_eq!(accumulated.task_operator_metrics.len(), 1, "operator map rows merge by key");
    let task = &accumulated.task_runtime_metrics["shared-task"];
    assert_eq!(task.elapsed_time, Duration::from_nanos(12));
    assert_eq!(task.max_observed_peak_bytes, Some(130), "task peak takes max, not sum");
    let operator = &accumulated.task_operator_metrics["graph_stage"];
    assert_eq!(operator.task_count, 2);
    assert_eq!(operator.elapsed_time, Duration::from_nanos(12));
    assert_eq!(operator.max_observed_peak_bytes, Some(80), "operator peak takes max, not sum");
    assert!(sink.query_metrics.lock().unwrap().is_empty(), "stages do not publish independently");
    assert_eq!(owner.runtime_observations().len(), 2, "capture owner retains every stage snapshot");

    graph_metrics.publish();
    graph_metrics.publish();

    let published = sink.query_metrics.lock().unwrap();
    assert_eq!(published.len(), 1, "terminal publication is idempotent");
    assert_eq!(published[0], accumulated);
    drop(published);
    let datafusion_metrics = metrics_set.clone_inner();
    assert_eq!(datafusion_metrics.sum_by_name("input_rows").unwrap().as_usize(), 18);
    assert_eq!(datafusion_metrics.sum_by_name("peak_device_memory_bytes").unwrap().as_usize(), 90);
    assert_eq!(datafusion_metrics.sum_by_name("native_task_operator_rows").unwrap().as_usize(), 1);
    assert_eq!(datafusion_metrics.sum_by_name("native_task_runtime_rows").unwrap().as_usize(), 1);
}

#[cfg(feature = "server")]
#[tokio::test]
async fn graph_success_retains_every_drainable_sealed_observation_without_duplicate_metrics() {
    let (backend, exec, sink) =
        observed_graph_exec("graph-observation-success", triangle_spec(), ExecutionCancellation::new());
    let owner = exec.metrics_capture.clone().expect("graph wrapper owns local evidence");

    let batches = collect(admitted_observed_graph_plan(&exec), Arc::new(TaskContext::default()))
        .await
        .expect("graph query succeeds");
    close_observed_graph_backend(&backend);

    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    let retained = owner.runtime_observations();
    assert_eq!(retained.len(), 1, "one admitted graph invocation retains one sealed observation");
    let snapshot = &retained[0];
    assert_eq!(snapshot.manifest().query_id.as_str(), "graph-observation-success");
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    assert!(snapshot.footer().complete);
    assert_eq!(graph_plan_boundary_count(snapshot, PlanExecutionBoundaryKind::Started), 2);
    assert_eq!(graph_plan_boundary_count(snapshot, PlanExecutionBoundaryKind::Finished), 2);
    assert_eq!(sink.query_metrics.lock().unwrap().len(), 1, "successful graph execution publishes once");
    assert_eq!(
        exec.metrics()
            .expect("graph metrics registered")
            .sum_by_name("native_graph_runtime_rows")
            .expect("graph runtime row metric registered")
            .as_usize(),
        1,
        "observed finalization must not re-record merged graph metrics",
    );
    drop(retained);

    let drained = owner.drain_runtime_observations();
    assert_eq!(drained.len(), 1);
    assert!(owner.runtime_observations().is_empty());
    assert_eq!(owner.retention_stats().drained_total, 1);
}

#[cfg(feature = "server")]
#[tokio::test]
async fn graph_native_failure_retains_failed_evidence_and_original_error() {
    let mut spec = pagerank_spec();
    let CugraphAlgorithmParams::PageRank(params) = &mut spec.params else { unreachable!("PageRank test spec") };
    params.alpha = 1.5;
    let (backend, exec, sink) = observed_graph_exec("graph-observation-failure", spec, ExecutionCancellation::new());
    let owner = exec.metrics_capture.clone().expect("graph wrapper owns local evidence");

    let error = collect(admitted_observed_graph_plan(&exec), Arc::new(TaskContext::default()))
        .await
        .expect_err("invalid native PageRank alpha remains a query failure");
    close_observed_graph_backend(&backend);

    let nexus = error.find_nexus_error().expect("DataFusion boundary preserves the typed adapter error");
    assert_eq!(nexus.code(), ErrorCode::CudfInvalidInput);
    assert_eq!(nexus.operation(), "datafusion_nexus.cugraph_sql.execute_gpu");
    assert_eq!(nexus.fact_value("native_code"), Some("cudf_invalid_input"));
    assert_eq!(nexus.fact_value("native_operation"), Some("nexus_query_engine.graph_algorithm.pagerank_params"));
    let retained = owner.runtime_observations();
    assert_eq!(retained.len(), 1, "the admitted graph invocation retains one sealed observation");
    assert_eq!(retained[0].manifest().query_id.as_str(), "graph-observation-failure");
    assert_eq!(
        retained[0].footer().terminal,
        Some(RuntimeTerminalOutcome::Failed {
            error_code: "cudf_invalid_input".to_string(),
            error_status: ErrorCode::CudfInvalidInput.status().as_str().to_string(),
        })
    );
    assert!(retained[0].footer().complete);
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Started), 2);
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Finished), 1);
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Failed), 1);
    assert_eq!(
        sink.query_metrics.lock().unwrap().len(),
        1,
        "mid-stage failure publishes prior and partial metrics once"
    );
    assert_eq!(
        exec.metrics()
            .expect("graph metrics registered")
            .sum_by_name("native_graph_runtime_rows")
            .expect("graph runtime row metric registered")
            .as_usize(),
        0,
        "invalid graph parameters are rejected before a runtime row is emitted",
    );
}

#[cfg(feature = "server")]
#[tokio::test]
async fn graph_failure_after_gpu_start_has_one_attempt_no_output_and_no_cpu_replay() {
    let execute_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let edge_source = Arc::new(CountingExec::new(memory_edge_source(16), Arc::clone(&execute_count)));
    let host_boundary_config =
        GpuIslandHostBoundaryConfig::try_new(GpuIslandHostBoundaryConfig::DEFAULT_MAX_INPUT_STAGING_BYTES, 1)
            .expect("one-byte output staging cap is valid");
    let (backend, exec, sink) = observed_graph_exec_with_edge_source(
        "graph-post-gpu-start-failure",
        triangle_spec(),
        ExecutionCancellation::new(),
        edge_source,
        host_boundary_config,
    );
    let owner = exec.metrics_capture.clone().expect("graph wrapper owns local evidence");
    let mut stream =
        admitted_observed_graph_plan(&exec).execute(0, Arc::new(TaskContext::default())).expect("graph stream builds");
    let mut output_batches = 0usize;

    let error = loop {
        match stream.try_next().await {
            Ok(Some(_batch)) => output_batches = output_batches.saturating_add(1),
            Ok(None) => panic!("forced graph failure must not end successfully"),
            Err(error) => break error,
        }
    };
    close_observed_graph_backend(&backend);

    let nexus = error.find_nexus_error().expect("post-start graph failure remains structured");
    assert_eq!(nexus.code(), ErrorCode::HostStagingLimitExceeded);
    assert_eq!(output_batches, 0, "post-start graph failure must not expose a partial output batch");
    assert_eq!(
        execute_count.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the staged CPU source must not be replayed after native graph execution starts",
    );
    let retained = owner.runtime_observations();
    assert_eq!(retained.len(), 1, "exactly one native graph attempt is observed");
    assert_eq!(retained[0].manifest().query_id.as_str(), "graph-post-gpu-start-failure");
    assert_eq!(
        retained[0].footer().terminal,
        Some(RuntimeTerminalOutcome::Failed {
            error_code: "host_staging_limit_exceeded".to_string(),
            error_status: ErrorCode::HostStagingLimitExceeded.status().as_str().to_string(),
        })
    );
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Started), 1);
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Finished), 1);
    let published = sink.query_metrics.lock().unwrap();
    assert_eq!(published.len(), 1, "the failed GPU attempt publishes exactly once");
    assert_eq!(published[0].graph_algorithm_runtime_rows.len(), 1, "exactly one graph runtime row is retained");
    assert_eq!(published[0].boundary.device_to_host_materializations, 0, "no output batch crossed to the host");
}

#[cfg(feature = "server")]
#[tokio::test]
async fn graph_pre_staging_cancellation_returns_without_a_graph_runtime_attempt() {
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();
    let (backend, exec, sink) = observed_graph_exec("graph-observation-cancelled", bfs_spec(), cancellation);
    let owner = exec.metrics_capture.clone().expect("graph wrapper owns local evidence");

    let error = collect(admitted_observed_graph_plan(&exec), Arc::new(TaskContext::default()))
        .await
        .expect_err("pre-cancelled graph query remains cancelled");
    close_observed_graph_backend(&backend);

    let nexus = error.find_nexus_error().expect("DataFusion boundary preserves the typed cancellation");
    assert_eq!(nexus.code(), ErrorCode::Cancelled);
    assert_eq!(nexus.fact_value("native_code"), Some("cancelled"));
    let retained = owner.runtime_observations();
    assert_eq!(retained.len(), 1, "the admitted invocation retains its cancelled observation");
    assert_eq!(retained[0].manifest().query_id.as_str(), "graph-observation-cancelled");
    assert_eq!(retained[0].footer().terminal, Some(RuntimeTerminalOutcome::Cancelled));
    assert!(retained[0].footer().complete);
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Started), 0);
    assert_eq!(sink.query_metrics.lock().unwrap().len(), 0, "pre-staging cancellation publishes no graph metrics");
    assert_eq!(
        exec.metrics()
            .expect("graph metrics registered")
            .sum_by_name("native_graph_runtime_rows")
            .map_or(0, |value| value.as_usize()),
        0,
        "cancellation before input staging must not start a graph runtime attempt",
    );
}

#[cfg(feature = "server")]
#[tokio::test]
async fn graph_gpu_producer_rejects_unprepared_direct_spi_execution() {
    let backend = test_gpu_backend();
    let config = backend_execution_config(&backend);
    let edge_source = native_literal_edge_source(&config, true);
    let exec = CugraphAlgoExec::try_new(bfs_utf8_spec(), edge_source, vec![], None, None, config)
        .expect("logical BFS graph exec builds");
    let task_context = Arc::new(TaskContext::default());
    let context_lease = exec
        .native_execution_config
        .execution_context_async(Arc::clone(&task_context))
        .await
        .expect("backend-admitted graph context builds");

    let result =
        GpuResidentProducer::execute_gpu_result_with_context(&exec, context_lease.context(), 0, task_context).await;
    let error = match result {
        Ok(_) => panic!("direct GPU producer execution must reject unstaged inputs"),
        Err(error) => error,
    };
    let nexus = error.find_nexus_error().expect("unprepared producer failure remains structured");

    assert_eq!(nexus.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(nexus.operation(), "datafusion_nexus.cugraph_sql.execute_gpu");
}

#[cfg(feature = "server")]
#[tokio::test]
async fn graph_logical_bfs_gpu_projection_retains_three_stages_and_publishes_once() {
    let server_observability = crate::server::ServerObservabilityConfig {
        per_query_max_records: 256,
        per_query_max_bytes: 256 * 1024,
        active_logical_capture_max_bytes: 512 * 1024,
        global_sealed_max_count: 8,
        global_sealed_max_bytes: 2 * 1024 * 1024,
        sealed_ttl: Duration::from_secs(60),
        shutdown_grace: Duration::from_secs(30),
    };
    let observation_config = server_observability.runtime_observation_config();
    let sink = Arc::new(GraphCaptureSink::default());
    let erased_sink: Arc<dyn NativeMetricsCaptureSink> = sink.clone();
    let backend = test_gpu_backend();
    let config = backend_execution_config(&backend)
        .with_metrics_capture(
            NativeMetricsCapture::new("graph-bfs-projection", erased_sink)
                .with_runtime_observation_config(observation_config),
        )
        .with_plan_query_admission_ticket();
    let edge_source = native_literal_edge_source(&config, true);
    let exec = Arc::new(
        CugraphAlgoExec::try_new(bfs_utf8_spec(), edge_source, vec![], Some(vec![0, 1]), Some(2), config)
            .expect("logical BFS projection graph exec builds"),
    );
    let owner = exec.metrics_capture.clone().expect("graph wrapper owns local evidence");
    let batches = collect(admitted_observed_graph_plan(&exec), Arc::new(TaskContext::default()))
        .await
        .expect("logical BFS GPU projection succeeds");
    close_observed_graph_backend(&backend);

    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    assert!(batches.iter().all(|batch| batch.num_columns() == 2));
    let retained = owner.runtime_observations();
    assert_eq!(retained.len(), 1, "one admitted BFS invocation retains one sealed observation");
    assert_eq!(retained[0].manifest().query_id.as_str(), "graph-bfs-projection");
    assert_eq!(retained[0].footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    assert_eq!(retained[0].manifest().limits, observation_config.limits());
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Started), 3);
    assert_eq!(graph_plan_boundary_count(&retained[0], PlanExecutionBoundaryKind::Finished), 3);
    assert_eq!(sink.query_metrics.lock().unwrap().len(), 1, "BFS projection path publishes once");
    assert_eq!(
        exec.metrics()
            .expect("graph metrics registered")
            .sum_by_name("native_graph_runtime_rows")
            .expect("graph runtime row metric registered")
            .as_usize(),
        1,
        "merged publication preserves the single graph runtime row",
    );
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
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(admission_only_source());
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
    let normalized_graph = singleflight_test_exec_with_options(
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
    assert_ne!(first, normalized_graph, "raw and Python-normalized graph preparations must never coalesce");
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
    let host_metrics = NexusNativeMetrics::new(&exec.metrics, 0);
    exec.record_bfs_singleflight_host_follower_metrics(&host_metrics, &key, &batches, std::time::Duration::ZERO);

    let recorded = NativeDataFusionExecutionMetrics::from_metrics(&exec.metrics.clone_inner());
    assert_eq!(recorded.graph_runtime_rows.len(), 2, "one metric row per follower boundary");
    for row in recorded.graph_runtime_rows {
        assert!(row.algorithm_metadata.contains("bfs_singleflight=follower"), "{row:?}");
        assert!(!row.algorithm_metadata.contains("graph_cache="), "{row:?}");
    }
}

#[test]
fn cugraph_execution_config_creates_plan_ticket_for_backend_admission_source() {
    let source = admission_only_source();
    let base = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let plan = EmptyExec::new(Arc::new(Schema::empty()));

    assert!(base.has_query_admission_authority());
    assert_eq!(base.query_admission_ticket_id(), None);

    let graph_config = cugraph_execution_config(&plan, &base);
    let ticket_id = graph_config
        .query_admission_ticket_id()
        .expect("cuGraph execution should use a plan-scoped backend admission ticket");
    let cloned = cugraph_execution_config(&plan, &graph_config);

    assert_eq!(cloned.query_admission_ticket_id(), Some(ticket_id));
}

#[test]
fn cugraph_algo_exec_keeps_graph_model_bytes_out_of_admission_grant() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024;
    let capacity = 4 * 1024 * 1024 * 1024;
    let source = admission_source_with_requirement(min, want, capacity);
    let config = NexusNativeExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, config)
        .expect("cuGraph exec should build with graph-aware admission ticket");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("graph model should derive opaque admission evidence")
        .expect("known edge rows should produce graph model evidence");

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
        Some(&nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution()),
        "the ticket must carry the canonical cuGraph execution proof"
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
    assert!(graph.graph_model_ask_for_admission().unwrap().is_some());
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
fn cugraph_algo_exec_keeps_native_edge_model_bytes_out_of_admission_grant() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024;
    let capacity = 4 * 1024 * 1024 * 1024;
    let source = admission_source_with_requirement(min, want, capacity);
    let config = NexusNativeExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = native_edge_source(vec![100, 200], &config);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should price native edge-source row evidence");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let model_ask = exec
        .graph_model_ask_bytes_for_admission_ticket()
        .expect("graph model should derive opaque admission evidence")
        .expect("native edge-source rows should produce graph model evidence");

    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution())
    );
    assert!(
        model_ask.cold_model_requested_bytes() < want,
        "native edge-source graph estimate should not default to the controller hint"
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
async fn cugraph_algo_exec_keeps_cache_affinity_out_of_the_admission_ticket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let source = admission_only_source();
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = datafusion_parquet_edge_source(&path).await;

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for keyable DataFusion edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution())
    );
}

#[tokio::test]
async fn cugraph_algo_exec_keeps_bfs_cache_affinity_out_of_the_admission_ticket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let source = admission_only_source();
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = datafusion_parquet_edge_source(&path).await;

    let exec = CugraphAlgoExec::try_new(bfs_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph BFS exec should build for keyable DataFusion edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution())
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
    let source = admission_only_source();
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = native_graph_residency_edge_source(&config);

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for keyable native edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution())
    );
}

#[test]
fn cugraph_algo_exec_uses_the_same_proof_for_unkeyable_edge_source() {
    let source = admission_only_source();
    let config = NexusNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), memory_edge_source(100), vec![], None, None, config)
        .expect("cuGraph exec should build for non-native edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");

    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution())
    );
}

#[test]
fn cugraph_algo_exec_keeps_default_ticket_for_non_light_native_edge_source() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024;
    let capacity = 4 * 1024 * 1024 * 1024;
    let source = admission_source_with_requirement(min, want, capacity);
    let config = NexusNativeExecutionConfig::bounded(min, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_graph_memory_calibration_override(
            nexus_query_engine::graph_memory::GraphMemoryCalibrationOverride::none().with_raw_input_edges(300),
        );
    let edge_source = sorted_native_edge_source(vec![100, 200], &config);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build with non-light native edge source");
    let ticket =
        exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should still mint a plan ticket");

    assert_eq!(
        ticket.attempt_capability_proof(),
        Some(&nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution()),
        "non-light native preprocessing should retain only the canonical graph proof"
    );
}

#[test]
fn cugraph_execution_config_keeps_standalone_configs_ticketless() {
    let base = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1);
    let plan = EmptyExec::new(Arc::new(Schema::empty()));

    let graph_config = cugraph_execution_config(&plan, &base);

    assert_eq!(graph_config.query_admission_ticket_id(), None);
}

#[test]
fn cugraph_execution_config_recovers_wrapped_native_edge_source_ticket() {
    let source = admission_only_source();
    let default_config =
        NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source.clone());
    let native_config = NexusNativeExecutionConfig::bounded(2048, 512, 1, 3, 1).with_query_admission_source(source);
    let native_config = native_config.with_plan_query_admission_ticket();
    let ticket = native_config.query_admission_ticket().expect("native edge source should have a plan ticket");
    let expected_ticket_id = native_config.query_admission_ticket_id();

    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_exec = Arc::new(NexusNativeExec::try_new(native_plan, native_config).unwrap()) as Arc<dyn ExecutionPlan>;
    let wrapped = wrap_plan_with_query_admission_ticket(native_exec, ticket);

    assert!(wrapped.downcast_ref::<NexusNativeExec>().is_some());
    assert_eq!(
        cugraph_execution_config(wrapped.as_ref(), &default_config).query_admission_ticket_id(),
        expected_ticket_id
    );
}

#[test]
fn cugraph_execution_config_prefers_ambient_root_ticket() {
    let source = admission_only_source();
    let base = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let ambient_config = base.clone().with_plan_query_admission_ticket();
    let local_config = base.with_plan_query_admission_ticket();
    let ambient_ticket =
        ambient_config.query_admission_ticket().expect("ambient config should have a root admission ticket");
    let task_context = task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), ambient_ticket);
    let plan = EmptyExec::new(Arc::new(Schema::empty()));

    assert_ne!(ambient_config.query_admission_ticket_id(), local_config.query_admission_ticket_id());
    assert_eq!(
        cugraph_execution_config_for_task_context(&plan, &local_config, &task_context).query_admission_ticket_id(),
        ambient_config.query_admission_ticket_id()
    );
}

#[test]
fn cugraph_execution_context_stamps_resolved_ticket_before_acquire() {
    let source = admission_only_source();
    let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
    let execution_config = execution_config.with_plan_query_admission_ticket();
    let raw_context = Arc::new(TaskContext::default());

    assert!(query_admission_ticket_from_task_context(&raw_context).is_none());

    let stamped_context = task_context_with_cugraph_execution_ticket(&execution_config, Arc::clone(&raw_context));
    let stamped_ticket = query_admission_ticket_from_task_context(&stamped_context)
        .expect("cuGraph execution should stamp its resolved ticket onto the TaskContext");

    assert_eq!(Some(stamped_ticket.id()), execution_config.query_admission_ticket_id());
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
    let config =
        NexusNativeExecutionConfig::bounded(2048, 512, 1, 3, 1).with_query_admission_source(admission_only_source());
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
