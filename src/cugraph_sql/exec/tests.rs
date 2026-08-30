use std::{
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::{
    array::{ArrayRef, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, ScalarValue, tree_node::TreeNodeRecursion},
    execution::{context::TaskContext, runtime_env::RuntimeEnvBuilder},
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{binary, col, lit},
    },
    physical_plan::{StatisticsArgs, StatisticsContext, collect, empty::EmptyExec, filter::FilterExec},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof,
    expr::{Expr, SortExpr, SortSpec},
    graph_memory::{
        GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES, graph_memory_bounded_admission_evidence_for_plan,
        graph_memory_estimates_for_plan,
    },
    observability::RuntimeObservationConfig,
    parquet_stats::{ParquetVariableWidthEncodingProof, ParquetVariableWidthPayloadEvidence},
    plan::{PlanNode, ProjectionOutput, QueryPlan, WindowFunction},
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
        CugraphAlgorithmParams, CugraphBfsOutputOptions, CugraphBfsParams, CugraphBfsSource,
        CugraphGraphEdgePredicateTable, CugraphGraphOptions, CugraphGraphPredicateTable, CugraphPageRankParams,
        CugraphSqlConfig, CugraphTableReference, CugraphTriangleCountAllParams, result_schema_for_params,
    },
    error::{DataFusionErrorExt, ErrorCode},
    native_exec::{
        GraphClaimCalibrationCapture, SealedAttemptMetricsTarget, admission::QueryAdmissionTicket,
        producer::gpu_resident_producer_unique_output_key, query_admission_ticket_from_task_context,
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<datafusion::common::Statistics>> {
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<datafusion::common::Statistics>> {
        StatisticsContext::new().compute(self.input.as_ref(), &StatisticsArgs::new().with_partition(partition))
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

fn graph_ticket_proof_with_resident_edge_plan(
    exec: &CugraphAlgoExec,
    edge_plan: &QueryPlan,
) -> nexus_query_engine::admission::QueryPlanCapabilityProof {
    graph_ticket_plan_and_proof_with_resident_edge_plan(exec, edge_plan).1
}

fn graph_ticket_plan_and_proof_with_resident_edge_plan(
    exec: &CugraphAlgoExec,
    edge_plan: &QueryPlan,
) -> (nexus_query_engine::plan::QueryPlan, nexus_query_engine::admission::QueryPlanCapabilityProof) {
    let edge_vertex_upper_bound =
        edge_source_graph_vertex_estimate(exec.edge_source.as_ref()).map(|estimate| estimate.count);
    cugraph_execution_capability_proof_with_resident_edge_plan(
        &exec.spec,
        exec.edge_source.schema(),
        exec.predicate_sources.iter().map(|source| source.schema()).collect(),
        exec.input_row_bounds.predicate_row_upper_bounds(),
        &exec.result_options,
        exec.edge_normalize_inserted(),
        exec.input_row_bounds.edge_row_upper_bound(),
        edge_vertex_upper_bound,
        Some(edge_plan),
    )
    .expect("prepared graph fixture must issue a resident-input-bound proof")
}

fn production_string_parquet_plan(
    path: &str,
    schema: Arc<Schema>,
    rows: u64,
    payload_upper_bounds: &[u64],
) -> QueryPlan {
    assert_eq!(schema.fields().len(), payload_upper_bounds.len());
    let payload_evidence = payload_upper_bounds
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            (
                index,
                ParquetVariableWidthPayloadEvidence::new(
                    index,
                    vec![0],
                    *bytes,
                    ParquetVariableWidthEncodingProof::PlainAndDictionary,
                ),
            )
        })
        .collect();
    let uncompressed_bytes = payload_upper_bounds
        .iter()
        .copied()
        .sum::<u64>()
        .saturating_add(rows.saturating_add(1).saturating_mul(8).saturating_mul(payload_upper_bounds.len() as u64));
    let file = LocalParquetFile::new(path)
        .with_row_groups(vec![0])
        .with_row_group_rows(vec![rows])
        .with_row_group_bytes(vec![uncompressed_bytes])
        .with_projected_row_group_bytes(vec![uncompressed_bytes], vec![uncompressed_bytes])
        .with_variable_width_payload_evidence(payload_evidence);
    let columns = schema.fields().iter().map(|field| field.name().clone()).collect();
    let mut builder = nexus_query_engine::plan::QueryPlanBuilder::new();
    let source = builder.add_source(nexus_query_engine::source::SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], schema, columns),
    ));
    builder.finish_gpu_resident(source)
}

fn production_string_logical_proof(
    graph_plan: &QueryPlan,
    edge_plan: &QueryPlan,
    predicate_plans: &[QueryPlan],
) -> QueryPlanCapabilityProof {
    let predicate_names = (0..predicate_plans.len()).map(cugraph_side_input_name).collect::<Vec<_>>();
    let mut resident_inputs = vec![(CUGRAPH_EDGE_INPUT_NAME, edge_plan)];
    resident_inputs.extend(predicate_names.iter().zip(predicate_plans).map(|(name, plan)| (name.as_str(), plan)));
    let graph =
        QueryPlanCapabilityProof::for_cugraph_graph_execution_plan_with_resident_inputs(graph_plan, resident_inputs)
            .expect("production-shaped string inputs must bind to the graph plan")
            .expect("logical graph plan must issue a capability proof");
    let graph_evidence =
        graph_memory_bounded_admission_evidence_for_plan(graph_plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("production-shaped logical graph plan must retain finite graph evidence");
    let graph = graph
        .with_graph_memory_admission_evidence(&graph_evidence)
        .expect("graph evidence must bind before composing logical mapping producers");

    QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt(
        graph_plan,
        graph,
        CUGRAPH_EDGE_INPUT_NAME,
        edge_plan,
        predicate_names.iter().cloned().zip(predicate_plans),
    )
    .expect("production-shaped logical mapping schedule must resolve")
    .expect("string logical mapping must materialize between its producers and graph")
}

fn production_required_floor_bytes(proof: QueryPlanCapabilityProof) -> (u64, u64, u64) {
    let backend = test_gpu_backend();
    let error = backend
        .backend()
        .session_capability()
        .admit_query_with_request(
            nexus_query_engine::runtime::QueryAdmissionRequest::new().with_query_plan_capability_proof(proof),
        )
        .expect_err("the production-shaped proof must exceed the 256 MiB test profile");
    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("impossible_requirement"));
    let parse_fact = |key| {
        error
            .fact_value(key)
            .unwrap_or_else(|| panic!("missing production admission fact '{key}'"))
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("production admission fact '{key}' must be u64"))
    };
    assert_eq!(parse_fact("max_compatible_managed_capacity_bytes"), 256 * 1024 * 1024);
    (
        parse_fact("minimum_compatible_required_floor_bytes"),
        parse_fact("minimum_compatible_dominant_owner_bytes"),
        parse_fact("minimum_compatible_owner_term_count"),
    )
}

fn sorted_native_edge_source(row_group_rows: Vec<u64>, config: &NexusNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_edge_source_with_root(row_group_rows, config, |builder, source| {
        builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("src"), false, false)]), edge_schema())
    })
}

fn filtered_native_edge_source(
    row_group_rows: Vec<u64>,
    config: &NexusNativeExecutionConfig,
) -> Arc<dyn ExecutionPlan> {
    native_edge_source_with_root(row_group_rows, config, |builder, source| {
        builder.add_filter(
            source,
            Expr::binary(Expr::column("src"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
            edge_schema(),
        )
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

fn pagerank_utf8_spec() -> CugraphAlgorithmSpec {
    let mut spec = pagerank_spec();
    spec.weight_col = None;
    spec.output_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));
    spec
}

fn logical_string_spec(algorithm: CugraphAlgorithm) -> CugraphAlgorithmSpec {
    let params = CugraphAlgorithmParams::from_native_defaults(algorithm, None);
    let graph_options = algorithm.effective_default_graph_options(&CugraphGraphOptions::default());
    let output_schema = crate::cugraph_sql::result_schema_for_vertex_domain(
        algorithm,
        &params,
        Some(&crate::cugraph_sql::CugraphVertexIdDomain::StringLogical),
    );
    CugraphAlgorithmSpec {
        algorithm,
        table_name: CugraphTableReference::parse("edges").unwrap(),
        src_col: "src".to_string(),
        dst_col: "dst".to_string(),
        weight_col: None,
        edge_id_col: None,
        graph_options,
        params,
        output_schema,
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
fn graph_producer_unique_key_contract_is_limited_to_vertex_row_results() {
    let edge_source = memory_edge_source(3);
    let pagerank = CugraphAlgoExec::try_new(
        pagerank_spec(),
        Arc::clone(&edge_source),
        Vec::new(),
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
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
        NexusNativeExecutionConfig::bounded_default(),
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
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("normalized BFS producer should build");
    assert_eq!(
        gpu_resident_producer_unique_output_key(&normalized_bfs),
        Some(std::collections::BTreeSet::from(["vertex".to_string()])),
    );

    let projected_normalized_bfs = CugraphAlgoExec::try_new(
        normalized_bfs.spec.clone(),
        Arc::clone(&edge_source),
        Vec::new(),
        Some(vec![1]),
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("projected normalized BFS producer should build");
    assert_eq!(gpu_resident_producer_unique_output_key(&projected_normalized_bfs), None);

    let path_bfs = CugraphAlgoExec::try_new(
        bfs_utf8_path_spec(),
        memory_string_edge_source(),
        Vec::new(),
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("path BFS producer should build");
    assert_eq!(gpu_resident_producer_unique_output_key(&path_bfs), None);

    let non_unique_algorithm = CugraphAlgoExec::try_new(
        triangle_spec(),
        edge_source,
        Vec::new(),
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("triangle-count producer should build");
    assert_eq!(gpu_resident_producer_unique_output_key(&non_unique_algorithm), None);
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
fn logical_bfs_path_gpu_plan_decodes_the_traversal_schema_before_host_reconstruction() {
    let spec = bfs_utf8_path_spec();
    let plan = cugraph_schema_query_plan(
        &spec,
        string_edge_schema(),
        Vec::new(),
        &[],
        &CugraphResultPlanOptions::default(),
        false,
        Some(3),
        None,
    )
    .expect("logical BFS path GPU plan should build");
    let schema = plan.root_node().output_schema();

    assert_eq!(schema.fields().len(), 3);
    assert_eq!(schema.field(0), &Field::new("vertex", DataType::Utf8, false));
    assert_eq!(schema.field(1).name(), "distance");
    assert_eq!(schema.field(2), &Field::new("predecessor", DataType::Utf8, true));
}

#[test]
fn string_pagerank_plan_restores_subsumed_top_k_order_after_decode() {
    let mut result_options = CugraphResultPlanOptions::new(None, Some(3));
    result_options.subsumed_top_k_sort = Some(SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]));
    let plan = cugraph_schema_query_plan(
        &pagerank_utf8_spec(),
        string_edge_schema(),
        Vec::new(),
        &[],
        &result_options,
        false,
        None,
        None,
    )
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
                        WindowFunction::RowNumber { alias, output_type }
                            if alias == "__vertex_id" && output_type == &DataType::Int64
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
        let plan = cugraph_schema_query_plan(
            &logical_string_spec(algorithm),
            string_edge_schema(),
            Vec::new(),
            &[],
            &CugraphResultPlanOptions::default(),
            false,
            Some(3),
            None,
        )
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
    cases.push((edge_betweenness, vec![("source", false), ("destination", false)]));

    for (spec, decoded_columns) in cases {
        let algorithm = spec.algorithm;
        let plan = cugraph_schema_query_plan(
            &spec,
            string_edge_schema(),
            Vec::new(),
            &[],
            &CugraphResultPlanOptions::default(),
            false,
            Some(3),
            None,
        )
        .unwrap_or_else(|error| panic!("{} logical selector plan failed: {error}", algorithm.function_name()));
        let graph = plan
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                PlanNode::GraphAlgorithm(graph) => Some(graph),
                _ => None,
            })
            .expect("logical selector plan must contain a native graph node");

        assert_eq!(graph.spec.params.side_input_node_ids().count(), 1, "{}", algorithm.function_name());
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
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("string PageRank producer should build");
    let consumer_plan = |with_limit| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            &producer.gpu_result_input_name,
            producer.schema(),
        )));
        let sort = builder.add_sort(
            source,
            SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
            producer.schema(),
        );
        let root = if with_limit { builder.add_limit(sort, LimitSpec::fetch(2), producer.schema()) } else { sort };
        builder.finish_gpu_resident(root)
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
fn string_pagerank_large_bound_prices_one_graph_after_both_endpoint_lookups() {
    let edge_rows = 54_957_455;
    let plan = cugraph_schema_query_plan(
        &pagerank_utf8_spec(),
        string_edge_schema(),
        Vec::new(),
        &[],
        &CugraphResultPlanOptions::default(),
        false,
        Some(edge_rows),
        None,
    )
    .expect("large string PageRank plan should build without materializing rows");

    let endpoint_joins = plan
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                &node.kind,
                PlanNode::Join(_join)
                    if node.output_schema().field_with_name("src").is_ok()
                        && node.output_schema().field_with_name("dst").is_ok()
            )
        })
        .count();
    assert_eq!(endpoint_joins, 2, "logical mapping must retain both endpoint encoding joins");

    let estimates = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect("large string PageRank plan must remain priceable");
    assert_eq!(estimates.len(), 1);
    assert_eq!(estimates[0].estimate.input_edges, edge_rows);

    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("large string PageRank plan must produce one admission token");
    assert!(evidence.cold_model_requested_bytes() > 0);

    let edge_plan = production_string_parquet_plan(
        "/tmp/production-string-pagerank.parquet",
        string_edge_schema(),
        edge_rows,
        &[2_300_000_000, 2_406_637_611],
    );
    let proof = production_string_logical_proof(&plan, &edge_plan, &[]);
    let (required_floor, dominant_owner, _) = production_required_floor_bytes(proof);
    assert!(
        required_floor > 4_706_637_611 && required_floor < 1_u64 << 40,
        "production-shaped PageRank admission must remain below one TiB instead of exploding to EiB"
    );
    assert!(dominant_owner > 0 && dominant_owner < 1_u64 << 40);
}

#[test]
fn string_jaccard_70_by_70_retires_shared_mapping_intermediates_before_the_final_sort() {
    let rows = 70;
    let spec = logical_string_spec(CugraphAlgorithm::Jaccard);
    let pair_schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Utf8, false),
        Field::new("second", DataType::Utf8, false),
    ]));
    let result_options = CugraphResultPlanOptions {
        subsumed_top_k_sort: Some(SortSpec::new(vec![SortExpr::new(Expr::column("similarity"), true, false)])),
        limit: Some(70),
        ..Default::default()
    };
    let plan = cugraph_schema_query_plan(
        &spec,
        string_edge_schema(),
        vec![Arc::clone(&pair_schema)],
        &[Some(rows)],
        &result_options,
        false,
        Some(rows),
        None,
    )
    .expect("70-edge/70-pair logical string Jaccard plan must build");
    let source_bounds = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                nexus_query_engine::source::SourceNode::GpuResident(source) => {
                    source.row_count_upper_bound.map(|bound| (source.input_name.as_str(), bound))
                },
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let side_input_name = cugraph_side_input_name(0);
    assert!(source_bounds.contains(&(CUGRAPH_EDGE_INPUT_NAME, rows)));
    assert!(source_bounds.contains(&(side_input_name.as_str(), rows)));

    let edge_plan = production_string_parquet_plan(
        "/tmp/production-string-jaccard-edges.parquet",
        string_edge_schema(),
        rows,
        &[2_300_000_000, 2_406_637_611],
    );
    let pair_plan = production_string_parquet_plan(
        "/tmp/production-string-jaccard-pairs.parquet",
        pair_schema,
        rows,
        &[2_100_000_000, 2_200_000_000],
    );
    let proof = production_string_logical_proof(&plan, &edge_plan, &[pair_plan]);
    let lifetimes = proof
        .composed_consumer_owner_lifetimes_for_test(&plan)
        .expect("the exact logical mapping scope must expose its structured schedule");
    let lifetime =
        |node_id, role| lifetimes.iter().find(|entry| entry.node_id == node_id && entry.role == role).copied().unwrap();
    let codebook = plan.nodes().iter().find(|node| matches!(node.kind, PlanNode::Window(_))).unwrap();
    let graph = plan.nodes().iter().find(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_))).unwrap();
    let PlanNode::GraphAlgorithm(graph_node) = &graph.kind else { unreachable!() };
    let final_sort = plan.node(plan.root_id()).unwrap();
    let PlanNode::Sort(sort) = &final_sort.kind else { panic!("the fixture must end after decode with a sort") };
    let pair = lifetime(graph_node.inputs[1], "consumer_output");
    let graph_output = lifetime(graph.id, "retained_output");
    let graph_execution = lifetime(graph.id, "execution_workspace");
    let codebook = lifetime(codebook.id, "consumer_output");
    let decoded = lifetime(sort.input, "consumer_output");
    let final_sort = lifetime(final_sort.id, "consumer_workspace");
    assert_eq!(pair.last_group, graph_execution.first_group, "encoded pairs live through graph execution");
    assert_eq!(pair.last_group, graph_output.first_group);
    assert_eq!(codebook.last_group, decoded.first_group, "the shared codebook lives through the final decode join");
    assert!(codebook.last_group < final_sort.first_group, "the codebook must retire before the final sort workspace");
    assert_eq!(decoded.last_group, final_sort.first_group);
    assert_eq!((pair.bytes, pair.first_group, pair.last_group), (1_760, 9, 11));
    assert_eq!((codebook.bytes, codebook.first_group, codebook.last_group), (9_006_642_739, 5, 15));
    assert_eq!((decoded.bytes, decoded.first_group, decoded.last_group), (18_013_277_878, 15, 16));
    assert_eq!((final_sort.bytes, final_sort.first_group, final_sort.last_group), (36_026_555_756, 16, 16));
    let (required_floor, dominant_owner, owner_term_count) = production_required_floor_bytes(proof);

    assert_eq!(owner_term_count, 40, "the semantic fixture includes both pair encodes and both result sorts");
    assert_eq!(dominant_owner, final_sort.bytes);
    assert!(
        required_floor < 101_492_260_864,
        "last-use retirement must keep the 70/70 floor below the production managed capacity: {required_floor}",
    );
    assert!(dominant_owner < required_floor);
}

#[test]
fn string_bfs_maps_edge_relation_predicates_and_both_result_vertices() {
    let mut spec = bfs_utf8_spec();
    {
        let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("fixture is BFS") };
        params.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new(
            CugraphTableReference::parse("sources").unwrap(),
            "source_vertex",
        ));
        params.predicates.include_vertices =
            Some(CugraphGraphPredicateTable::new(CugraphTableReference::parse("included").unwrap(), "included_vertex"));
        params.predicates.target_vertices =
            Some(CugraphGraphPredicateTable::new(CugraphTableReference::parse("targets").unwrap(), "target_vertex"));
        params.predicates.include_edges = Some(CugraphGraphEdgePredicateTable::new(
            CugraphTableReference::parse("included_edges").unwrap(),
            "edge_src",
            "edge_dst",
            None::<String>,
        ));
    }
    let predicate_schemas = vec![
        Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Utf8, false)])),
        Arc::new(Schema::new(vec![Field::new("included_vertex", DataType::Utf8, false)])),
        Arc::new(Schema::new(vec![Field::new("target_vertex", DataType::Utf8, false)])),
        Arc::new(Schema::new(vec![
            Field::new("edge_src", DataType::Utf8, false),
            Field::new("edge_dst", DataType::Utf8, false),
        ])),
    ];
    let plan = cugraph_schema_query_plan(
        &spec,
        string_edge_schema(),
        predicate_schemas.clone(),
        &[Some(54_957_455), Some(10_446_432), Some(10_446_432), Some(54_957_455)],
        &CugraphResultPlanOptions::default(),
        false,
        Some(54_957_455),
        None,
    )
    .expect("production-shaped string BFS plan should build without materializing rows");

    let domain = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Union(union) if union.inputs.len() == 7 => Some(union),
            _ => None,
        })
        .expect("BFS codebook must include two endpoints and five side-input vertex columns");
    assert_eq!(domain.inputs.len(), 7);

    let joins = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .collect::<Vec<_>>();
    let encode_joins =
        joins.iter().filter(|join| join.spec.keys.iter().any(|key| key.right.name() == "__vertex_id_str")).count();
    let decode_joins =
        joins.iter().filter(|join| join.spec.keys.iter().any(|key| key.right.name() == "__vertex_id")).count();
    assert_eq!(encode_joins, 7, "every codebook domain column must be encoded");
    assert_eq!(decode_joins, 2, "BFS vertex and predecessor must both decode through the shared codebook");

    let edge_plan = production_string_parquet_plan(
        "/tmp/production-string-bfs-edges.parquet",
        string_edge_schema(),
        54_957_455,
        &[2_300_000_000, 2_406_637_611],
    );
    let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("fixture is BFS") };
    params.predicates.include_edges = None;
    let admission_plan = cugraph_schema_query_plan(
        &spec,
        string_edge_schema(),
        predicate_schemas[..3].to_vec(),
        &[Some(10_446_432), Some(10_446_432), Some(10_446_432)],
        &CugraphResultPlanOptions::default(),
        false,
        Some(54_957_455),
        None,
    )
    .expect("production-shaped string BFS admission plan should build");
    let predicate_plans = vec![
        production_string_parquet_plan(
            "/tmp/production-string-bfs-sources.parquet",
            Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Utf8, false)])),
            10_446_432,
            &[500_000_000],
        ),
        production_string_parquet_plan(
            "/tmp/production-string-bfs-included.parquet",
            Arc::new(Schema::new(vec![Field::new("included_vertex", DataType::Utf8, false)])),
            10_446_432,
            &[500_000_000],
        ),
        production_string_parquet_plan(
            "/tmp/production-string-bfs-targets.parquet",
            Arc::new(Schema::new(vec![Field::new("target_vertex", DataType::Utf8, false)])),
            10_446_432,
            &[500_000_000],
        ),
    ];
    let proof = production_string_logical_proof(&admission_plan, &edge_plan, &predicate_plans);
    let (required_floor, dominant_owner, _) = production_required_floor_bytes(proof);
    assert!(
        required_floor > 4_706_637_611 && required_floor < 1_u64 << 40,
        "production-shaped BFS admission must remain below one TiB instead of exploding to EiB"
    );
    assert!(dominant_owner > 0 && dominant_owner < 1_u64 << 40);
}

#[test]
fn logical_bfs_path_reconstruction_applies_the_final_schema_after_gpu_decode() {
    let spec = bfs_utf8_path_spec();
    let traversal_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("distance", DataType::Int64, true),
        Field::new("predecessor", DataType::Utf8, true),
    ]));
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
        .expect("GPU-decoded traversal output should reconstruct into the five-column path schema");

    assert_eq!(path.schema(), spec.output_schema);
    assert_eq!(path.num_columns(), 5);
    assert_eq!(path.num_rows(), 2);
}

#[test]
fn string_pagerank_and_bfs_are_gpu_producers_while_bfs_path_remains_host_owned() {
    let pagerank = CugraphAlgoExec::try_new(
        pagerank_utf8_spec(),
        memory_string_edge_source(),
        vec![],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("string PageRank should build as a GPU-resident graph execution");
    assert!(crate::native_exec::producer::gpu_resident_producer(&pagerank).is_some());

    let logical = CugraphAlgoExec::try_new(
        bfs_utf8_spec(),
        memory_string_edge_source(),
        vec![],
        None,
        None,
        NexusNativeExecutionConfig::bounded_default(),
    )
    .expect("logical BFS should build as a GPU-resident graph execution");
    assert!(crate::native_exec::producer::gpu_resident_producer(&logical).is_some());

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
    assert_eq!(staged.host_evidence.input_limit_bytes(), input_limit_bytes);
    assert_eq!(staged.host_evidence.input_staging_peak_bytes(), staged.reservation.as_ref().unwrap().size());
    assert_eq!(staged.host_evidence.output_limit_bytes(), output_limit_bytes);
    assert_eq!(
        staged.host_evidence.aggregate_peak_bytes(),
        staged.host_evidence.input_staging_peak_bytes() + staged.host_evidence.output_limit_bytes(),
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
async fn cugraph_producer_preparation_keeps_utf8_literal_sources_out_of_host_staging() {
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let spec = bfs_utf8_spec();
    let edge_source = Arc::new(EmptyExec::new(string_edge_schema())) as Arc<dyn ExecutionPlan>;

    let staged = stage_cugraph_inputs(&spec, edge_source, vec![], task_context, &ExecutionCancellation::new(), 0)
        .await
        .expect("UTF-8 literal source should be constructed inside the native GPU plan");

    assert_eq!(staged.row_bounds().unwrap().predicate_row_upper_bounds(), &[]);
    assert_eq!(runtime.memory_pool.reserved(), 0);
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
        Some(model_ask.cold_model_requested_bytes().max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)),
        "the immutable grant target includes the canonical graph live-reservation minimum"
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
        Some(model_ask.cold_model_requested_bytes().max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES))
    );
}

#[test]
fn filtered_native_edge_ticket_defers_graph_pricing_to_the_retained_runtime_frame() {
    let backend = test_gpu_backend();
    let config = NexusNativeExecutionConfig::bounded(64 * 1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(admission_source(&backend));
    let edge_source = filtered_native_edge_source(vec![100_000_000], &config);
    let native = as_nexus_native_exec(edge_source.as_ref()).expect("fixture must lower to one native edge producer");
    let edge_plan = native.native_plan().clone_with_gpu_resident_sink();
    let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
        .expect("filtered native edge source must remain a valid graph producer");
    let input_row_bounds =
        CugraphInputRowBounds::from_execution_plans(exec.edge_source.as_ref(), &exec.predicate_sources)
            .expect("native source metadata supplies its static producer bound");
    let (exec, graph_plan) = exec
        .issue_graph_aware_ticket_after_staging(input_row_bounds, &TaskContext::default(), Some(&edge_plan))
        .expect("eligible filtered producer must mint one deferred graph ticket");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("one graph-aware ticket is issued");
    let graph = graph_ticket_proof_with_resident_edge_plan(&exec, &edge_plan);
    let expected = QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt(
        &graph_plan,
        graph,
        CUGRAPH_EDGE_INPUT_NAME,
        &edge_plan,
        std::iter::empty(),
    )
    .expect("deferred graph contract must derive")
    .expect("Source-Filter-GPU sink is eligible");

    assert_eq!(ticket.attempt_capability_proof(), Some(&expected));
    assert_eq!(expected.soft_working_set_hint_bytes(), None);
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
        Some(model_ask.cold_model_requested_bytes().max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)),
        "the direct DataFusion scan shares the bounded graph live-reservation owner"
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
        Some(model_ask.cold_model_requested_bytes().max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)),
        "the bounded graph live peak includes its prepared GPU edge frame"
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
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let plan = ctx
        .sql(
            "SELECT distance, COUNT(*) AS reached \
             FROM cugraph_bfs('citation_edges', 0, 'src', 'dst', NULL, \
             '{\"depth_limit\": 3, \"output_mode\": \"normalized\"}') \
             WHERE distance <= 3 GROUP BY distance",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let lowering = crate::native::lower_physical_plan_to_native(Arc::clone(&plan));
    assert_eq!(lowering.selected_count(), 1, "grouped BFS result should lower: {:#?}", lowering.report);
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
        .expect("metadata hydration must lower the DataFusion edge scan to its native source fragment")
        .clone_with_gpu_resident_sink();
    let source_proof =
        nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&source_plan)
            .expect("the resolved native edge source must retain its source-read capability proof");
    let expected_proof = nexus_query_engine::admission::QueryPlanCapabilityProof::combine(
        graph_ticket_proof_with_resident_edge_plan(&prepared, &source_plan),
        [source_proof],
    )
    .expect("the graph root and resolved native edge source must retain one combined proof")
    .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
    .expect("exact graph evidence must match the graph proof scope");
    assert_eq!(
        proof, &expected_proof,
        "the post-hydration ticket must retain the exact graph and resolved source proofs with phase-aware model evidence"
    );
}

#[tokio::test]
async fn installed_backend_string_pagerank_ticket_orders_source_mapping_and_graph_owners() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("string-edges.parquet");
    write_string_edge_parquet(&path);

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
    ctx.register_parquet("string_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
    let invocation = attempt.planning_context(&ctx).unwrap();
    let plan = invocation
        .sql(
            "SELECT vertex, value FROM cugraph_pagerank(\
             'string_edges', 'src', 'dst', NULL, '{\"alpha\": 0.85}')",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let exec = cugraph_exec_in_plan(plan).expect("installed backend plan should contain CugraphAlgoExec");
    let exec = exec.downcast_ref::<CugraphAlgoExec>().unwrap();
    let (prepared, _staged) = exec.clone().prepare_staged_attempt(invocation.task_ctx()).await.unwrap();
    let ticket = prepared
        .native_execution_config
        .query_admission_ticket()
        .expect("metadata hydration must mint the string graph ticket");
    let proof = ticket.attempt_capability_proof().expect("string graph ticket must retain its capability proof");
    let source_plan = lowered_native_edge_plan(prepared.edge_source.as_ref())
        .expect("metadata hydration must lower the string edge scan")
        .clone_with_gpu_resident_sink();
    let model_ask = prepared
        .graph_model_ask_bytes_for_admission_ticket()
        .unwrap()
        .expect("string PageRank must retain exact graph model evidence");
    let (graph_plan, graph) = graph_ticket_plan_and_proof_with_resident_edge_plan(&prepared, &source_plan);
    let graph = graph
        .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
        .expect("graph evidence must bind before the mapping schedule");
    let independently_derived =
        nexus_query_engine::admission::QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt(
            &graph_plan,
            graph,
            CUGRAPH_EDGE_INPUT_NAME,
            &source_plan,
            std::iter::empty(),
        )
        .expect("string mapping schedule must be derivable")
        .expect("string mapping must materialize between the source and graph");

    assert_eq!(
        proof.ordered_schedule_phase_group_counts_for_test(),
        independently_derived.ordered_schedule_phase_group_counts_for_test(),
        "the admitted proof must retain the phase shape derived from the exact native plan"
    );
    let (producer_phases, consumer_phase_groups) = proof
        .ordered_schedule_phase_group_counts_for_test()
        .expect("string PageRank must carry an ordered source-to-graph schedule");
    assert_eq!(producer_phases, 1);
    assert!(consumer_phase_groups > 2, "mapping operators must be priced in addition to graph phases");

    let context = ticket
        .acquire_context(invocation.task_ctx(), ExecutionCancellation::new())
        .await
        .expect("the exact string mapping schedule must resolve to an admission grant");
    assert!(context.admission_facts().query_grant_bytes > 0);
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
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
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
