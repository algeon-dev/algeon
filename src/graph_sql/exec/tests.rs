use std::{fs::File, sync::Arc};

use arrow::{
    array::{ArrayRef, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    execution::context::TaskContext,
    logical_expr::Operator,
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{empty::EmptyExec, filter::FilterExec},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use nexus_query_engine::{
    expr::{Expr, SortExpr, SortSpec},
    plan::ProjectionOutput,
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryService,
        QueryServiceConfig, QueryServiceTenantConfig, TenantId,
    },
    source::{LiteralBatchSource, LocalParquetFile, LocalParquetSource},
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use super::*;
use crate::{
    graph_sql::{
        CugraphBfsOutputOptions, CugraphBfsParams, CugraphGraphOptions, CugraphPageRankParams, CugraphSqlConfig,
        CugraphTableReference, CugraphTriangleCountAllParams, result_schema_for_params,
    },
    native_exec::{
        QueryAdmissionSource, task_context_with_query_admission_ticket, wrap_plan_with_query_admission_ticket,
    },
    session::CudfSessionStateBuilderExt,
};

fn admission_only_source() -> QueryAdmissionSource {
    admission_source_with_band(1024 * 1024, 1024 * 1024, 1024 * 1024)
}

fn admission_source_with_band(min_bytes: u64, want_bytes: u64, capacity_bytes: u64) -> QueryAdmissionSource {
    let tenant_id = TenantId::new("cugraph-test");
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let service = QueryService::new_admission_only(
        QueryServiceConfig {
            device_capacity_bytes: Some(capacity_bytes),
            max_concurrent_queries: 1,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), min_bytes, want_bytes)],
        deployment,
    )
    .expect("admission-only query service should build");
    QueryAdmissionSource::new(service, tenant_id)
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

fn native_edge_source(row_group_rows: Vec<u64>, config: &NativeCudfExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_edge_source_with_root(row_group_rows, config, |_, source| source)
}

fn sorted_native_edge_source(row_group_rows: Vec<u64>, config: &NativeCudfExecutionConfig) -> Arc<dyn ExecutionPlan> {
    native_edge_source_with_root(row_group_rows, config, |builder, source| {
        builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("src"), false, false)]), edge_schema())
    })
}

fn native_edge_source_with_root(
    row_group_rows: Vec<u64>,
    config: &NativeCudfExecutionConfig,
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
    Arc::new(NativeCudfExec::try_new(plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>
}

fn native_graph_residency_edge_source_with_stable_id(
    config: &NativeCudfExecutionConfig,
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
    Arc::new(NativeCudfExec::try_new(plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>
}

fn native_graph_residency_edge_source(config: &NativeCudfExecutionConfig) -> Arc<dyn ExecutionPlan> {
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

fn singleflight_test_exec_with_options(
    source_vertex: i64,
    depth_limit: Option<i64>,
    construction_policy: CugraphGraphConstructionPolicy,
    graph_stable_id: &str,
    output_mode: CugraphBfsOutputMode,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
) -> CugraphAlgoExec {
    let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(admission_only_source())
        .with_query_service_resident_graph_cache_max_retained_bytes(Some(4096));
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
    assert_ne!(first, other_graph, "different resident graphs must never coalesce");
    assert_ne!(first, normalized_graph, "raw and Python-normalized graph preparations must never coalesce");
    assert_ne!(first, raw_output, "raw and normalized BFS outputs must never coalesce");
    assert_ne!(first, projected, "different projections must never coalesce");
    assert_ne!(first, limited, "different limits must never coalesce");
    assert!(singleflight_test_exec(1, Some(3)).bfs_singleflight_key(0).is_none());
}

#[test]
fn cugraph_execution_config_creates_plan_ticket_for_backend_admission_source() {
    let source = admission_only_source();
    let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
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
fn cugraph_algo_exec_mints_graph_aware_ticket_below_tenant_want() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024;
    let capacity = 4 * 1024 * 1024 * 1024;
    let source = admission_source_with_band(min, want, capacity);
    let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, config)
        .expect("cuGraph exec should build with graph-aware admission ticket");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let requested = ticket.requested_grant_bytes().expect("known edge rows should produce graph-aware ask");
    let warm_requested =
        ticket.warm_graph_requested_grant_bytes().expect("known cacheable graph should carry an exact-hit warm ask");

    assert!(requested >= min);
    assert!(requested < want, "small graph ask should not default to tenant want");
    assert!(warm_requested < requested, "warm ask must omit graph construction bytes");
}

#[test]
fn cugraph_algo_exec_registers_frontend_neutral_execution_spis() {
    let config = NativeCudfExecutionConfig::bounded(64 * 1024 * 1024, 512, 1, 3, 1);
    let exec = CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, config)
        .expect("cuGraph exec should register its execution SPIs");
    let plan = &exec as &dyn ExecutionPlan;

    let producer = crate::native_exec::producer::gpu_resident_producer(plan)
        .expect("registered producer discovery should recognize cuGraph execution");
    assert_eq!(producer.gpu_result_input_name(), exec.gpu_result_input_name);

    let graph = crate::native_exec::graph::graph_execution_plan(plan)
        .expect("registered graph admission discovery should recognize cuGraph execution");
    assert!(graph.graph_model_ask_for_admission().unwrap().is_some());
    assert_eq!(graph.graph_placement_hint_for_admission(), exec.graph_placement_hint_for_admission_ticket());
}

#[test]
fn cugraph_graph_model_ask_prefers_exact_edge_rows_over_configured_raw_edges() {
    let min = 64 * 1024 * 1024;
    let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1);
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
    assert!(exact_ask.warm_bytes() < exact_ask.cold_bytes());
}

#[test]
fn cugraph_algo_exec_uses_native_edge_source_rows_for_graph_aware_ticket() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024;
    let capacity = 4 * 1024 * 1024 * 1024;
    let source = admission_source_with_band(min, want, capacity);
    let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = native_edge_source(vec![100, 200], &config);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should price native edge-source row evidence");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let requested =
        ticket.requested_grant_bytes().expect("native edge-source row evidence should produce graph-aware ask");

    assert!(requested >= min);
    assert!(requested < want, "native edge-source graph ask should not default to tenant want");
}

#[test]
fn cugraph_algo_exec_records_graph_residency_for_native_edge_source() {
    let config = NativeCudfExecutionConfig::unbounded();
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
    let config = NativeCudfExecutionConfig::unbounded();
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
    let config = NativeCudfExecutionConfig::unbounded();
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
    let config = NativeCudfExecutionConfig::unbounded();
    let edge_source = datafusion_parquet_edge_source_sql(&path, "SELECT src, dst FROM edges").await;

    let exec = CugraphAlgoExec::try_new(bfs_utf8_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph logical BFS exec should build for DataFusion local-Parquet edge source");
    let placement_hint = cugraph_graph_placement_hint(
        &exec.spec,
        exec.graph_residency.as_ref(),
        exec.edge_normalize_inserted(),
        &exec.predicate_sources,
    );

    assert!(exec.graph_residency.is_none());
    assert!(placement_hint.is_empty());
}

#[tokio::test]
async fn cugraph_algo_exec_mints_graph_cache_placement_hint_for_datafusion_parquet_edge_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let source = admission_only_source();
    let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = datafusion_parquet_edge_source(&path).await;

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for keyable DataFusion edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let resident_key = ticket
        .placement_hint()
        .resident_graph_key()
        .expect("keyable DataFusion edge source should carry a graph placement hint");

    assert_eq!(resident_key.orientation, nexus_query_engine::plan::GraphResidencyOrientation::Transposed);
    assert_eq!(resident_key.source_column, "src");
    assert_eq!(resident_key.destination_column, "dst");
    assert_eq!(resident_key.weight_column.as_deref(), Some("weight"));
}

#[tokio::test]
async fn cugraph_algo_exec_mints_bfs_graph_cache_placement_hint_for_datafusion_parquet_edge_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.parquet");
    write_edge_parquet(&path);
    let source = admission_only_source();
    let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = datafusion_parquet_edge_source(&path).await;

    let exec = CugraphAlgoExec::try_new(bfs_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph BFS exec should build for keyable DataFusion edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let resident_key = ticket
        .placement_hint()
        .resident_graph_key()
        .expect("keyable DataFusion BFS edge source should carry a graph placement hint");

    assert_eq!(resident_key.orientation, nexus_query_engine::plan::GraphResidencyOrientation::Forward);
    assert_eq!(resident_key.source_column, "src");
    assert_eq!(resident_key.destination_column, "dst");
    assert_eq!(resident_key.weight_column, None);
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
    let placement_hint = cugraph_graph_placement_hint(
        &exec.spec,
        Some(descriptor),
        exec.edge_normalize_inserted(),
        &exec.predicate_sources,
    );

    assert!(graph_residency_matches_edge_schema(descriptor, edge_schema.as_ref(), &exec.spec));
    assert_eq!(
        placement_hint.resident_graph_key().map(|key| key.orientation),
        Some(nexus_query_engine::plan::GraphResidencyOrientation::Forward)
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
    let config = NativeCudfExecutionConfig::unbounded();
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
fn cugraph_algo_exec_mints_graph_cache_placement_hint_for_keyable_native_edge_source() {
    let source = admission_only_source();
    let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
    let edge_source = native_graph_residency_edge_source(&config);

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build for keyable native edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
    let resident_key =
        ticket.placement_hint().resident_graph_key().expect("keyable cached graph query should carry a placement hint");

    assert_eq!(resident_key.orientation, nexus_query_engine::plan::GraphResidencyOrientation::Transposed);
    assert_eq!(resident_key.source_column, "src");
    assert_eq!(resident_key.destination_column, "dst");
    assert_eq!(resident_key.weight_column.as_deref(), Some("weight"));
}

#[test]
fn cugraph_algo_exec_keeps_empty_graph_cache_placement_hint_for_unkeyable_edge_source() {
    let source = admission_only_source();
    let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);

    let exec = CugraphAlgoExec::try_new(pagerank_spec(), memory_edge_source(100), vec![], None, None, config)
        .expect("cuGraph exec should build for non-native edge source");
    let ticket = exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");

    assert!(ticket.placement_hint().is_empty());
}

#[test]
fn cugraph_algo_exec_keeps_default_ticket_for_non_light_native_edge_source() {
    let min = 64 * 1024 * 1024;
    let want = 2 * 1024 * 1024 * 1024;
    let capacity = 4 * 1024 * 1024 * 1024;
    let source = admission_source_with_band(min, want, capacity);
    let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_graph_memory_calibration_override(
            nexus_query_engine::graph_memory::GraphMemoryCalibrationOverride::none().with_raw_input_edges(300),
        );
    let edge_source = sorted_native_edge_source(vec![100, 200], &config);

    let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
        .expect("cuGraph exec should build with non-light native edge source");
    let ticket =
        exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should still mint a plan ticket");

    assert_eq!(ticket.requested_grant_bytes(), None, "non-light native preprocessing should keep tenant-default ask");
}

#[test]
fn cugraph_execution_config_keeps_standalone_configs_ticketless() {
    let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1);
    let plan = EmptyExec::new(Arc::new(Schema::empty()));

    let graph_config = cugraph_execution_config(&plan, &base);

    assert_eq!(graph_config.query_admission_ticket_id(), None);
}

#[test]
fn cugraph_execution_config_recovers_wrapped_native_edge_source_ticket() {
    let source = admission_only_source();
    let default_config =
        NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source.clone());
    let native_config = NativeCudfExecutionConfig::bounded(2048, 512, 1, 3, 1).with_query_admission_source(source);
    let native_config = native_config.with_plan_query_admission_ticket();
    let ticket = native_config.query_admission_ticket().expect("native edge source should have a plan ticket");
    let expected_ticket_id = native_config.query_admission_ticket_id();

    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_exec = Arc::new(NativeCudfExec::try_new(native_plan, native_config).unwrap()) as Arc<dyn ExecutionPlan>;
    let wrapped = wrap_plan_with_query_admission_ticket(native_exec, ticket);

    assert!(wrapped.downcast_ref::<NativeCudfExec>().is_some());
    assert_eq!(
        cugraph_execution_config(wrapped.as_ref(), &default_config).query_admission_ticket_id(),
        expected_ticket_id
    );
}

#[test]
fn cugraph_execution_config_prefers_ambient_root_ticket() {
    let source = admission_only_source();
    let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
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
    let execution_config = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
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
    let config = NativeCudfExecutionConfig::bounded(2048, 512, 1, 3, 1);
    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_exec = Arc::new(NativeCudfExec::try_new(native_plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>;

    let error = validate_standalone_bounded_cugraph_sources(&config, native_exec.as_ref(), &[])
        .expect_err("standalone bounded cuGraph must not execute a native edge child without admission");
    assert!(error.to_string().contains("cannot execute native cudf child plans"), "{error}");

    let repartitioned = Arc::new(RepartitionExec::try_new(native_exec, Partitioning::UnknownPartitioning(1)).unwrap())
        as Arc<dyn ExecutionPlan>;
    let error = validate_standalone_bounded_cugraph_sources(&config, repartitioned.as_ref(), &[])
        .expect_err("fallback collect over native children should be rejected");

    assert!(error.to_string().contains("cannot execute native cudf child plans"), "{error}");
}

#[test]
fn admission_backed_cugraph_allows_native_child_execution() {
    let config =
        NativeCudfExecutionConfig::bounded(2048, 512, 1, 3, 1).with_query_admission_source(admission_only_source());
    let schema = Arc::new(Schema::empty());
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let native_plan = builder.finish_with_host_sink(source);
    let native_exec = Arc::new(NativeCudfExec::try_new(native_plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>;
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
    let config = NativeCudfExecutionConfig::unbounded().with_lakehouse_footer_pruning(true);
    let native_exec = native_edge_source(vec![100, 200], &config);
    let wrapped = Arc::new(
        RepartitionExec::try_new(native_exec, Partitioning::UnknownPartitioning(1))
            .expect("test wrapper should accept the native child"),
    ) as Arc<dyn ExecutionPlan>;

    let prepared = prepare_cugraph_source_metadata(wrapped, &TaskContext::default()).await;

    assert!(prepared.downcast_ref::<RepartitionExec>().is_some(), "host wrapper must be preserved");
    let children = prepared.children();
    let native =
        children[0].downcast_ref::<NativeCudfExec>().expect("nested native child should remain under the host wrapper");
    assert!(!native.execution_config().lakehouse_footer_pruning());
}
