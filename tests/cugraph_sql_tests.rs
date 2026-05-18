use std::{
    fs,
    path::Path,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, BooleanArray, Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    execution::SessionStateBuilder,
    logical_expr::{col, lit},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::displayable,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfOptimizerConfig, CudfSessionStateBuilderExt, CugraphAlgorithm, CugraphAlgorithmParams,
    CugraphGraphConstructionPolicy, CugraphSqlConfig, GpuFallbackPolicy, NativeCudfComposedExec,
    parse_cugraph_algorithm_spec, stable_result_schema,
};

#[tokio::test]
async fn cugraph_sql_planning_registers_udtfs_on_session_state_builder() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let functions = state.table_functions();

    assert!(functions.contains_key("cugraph_pagerank"));
    assert!(functions.contains_key("cugraph_bfs"));
    assert!(functions.contains_key("cugraph_louvain"));
    assert!(!functions.contains_key("cugraph_degrees"));
}

#[tokio::test]
async fn cugraph_sql_planning_exposes_stable_output_schemas() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    assert_sql_schema(&ctx, "SELECT * FROM cugraph_pagerank('graph_edges')", CugraphAlgorithm::PageRank).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_bfs('graph_edges', 0)", CugraphAlgorithm::Bfs).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_louvain('graph_edges')", CugraphAlgorithm::Louvain).await;
}

#[test]
fn cugraph_sql_planning_parses_defaults_explicit_columns_and_null_weight() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[lit("graph_edges")], &config).unwrap();
    assert_eq!(spec.table_name.to_string(), "graph_edges");
    assert_eq!(spec.src_col, "src");
    assert_eq!(spec.dst_col, "dst");
    assert_eq!(spec.weight_col, None);
    assert_eq!(spec.graph_options.directed, true);
    assert_eq!(spec.graph_options.renumber, true);
    assert_eq!(spec.graph_options.construction_policy, CugraphGraphConstructionPolicy::PythonCugraph);

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Louvain,
        &[
            lit("main.graph_edges"),
            lit("from_id"),
            lit("to_id"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"directed":false,"renumber":false}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.table_name.to_string(), "main.graph_edges");
    assert_eq!(spec.src_col, "from_id");
    assert_eq!(spec.dst_col, "to_id");
    assert_eq!(spec.weight_col, None);
    assert_eq!(spec.graph_options.directed, false);
    assert_eq!(spec.graph_options.renumber, false);
}

#[test]
fn cugraph_sql_planning_parses_algorithm_options() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRank,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            lit("weight"),
            lit(r#"{"alpha":0.7,"epsilon":0.001,"max_iterations":25}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.weight_col, Some("weight".to_string()));
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::PageRank(datafusion_cudf::CugraphPageRankParams {
            alpha: 0.7,
            epsilon: 0.001,
            max_iterations: 25,
        })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"depth_limit":3}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Bfs(datafusion_cudf::CugraphBfsParams { source_vertex: 42, depth_limit: Some(3) })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Louvain,
        &[lit("graph_edges"), lit("src"), lit("dst"), lit("weight"), lit(r#"{"directed":false,"resolution":2.5}"#)],
        &config,
    )
    .unwrap();
    assert_eq!(spec.params, CugraphAlgorithmParams::Louvain(datafusion_cudf::CugraphLouvainParams { resolution: 2.5 }));
}

#[test]
fn cugraph_sql_planning_rejects_invalid_arguments() {
    let config = CugraphSqlConfig::default();
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[col("graph_edges")], &config),
        "table_name must be a literal",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[lit("SELECT src FROM edges")], &config),
        "must be a simple table",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"unknown":1}"#),
            ],
            &config,
        ),
        "unknown option 'unknown'",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"alpha":1.2}"#),
            ],
            &config,
        ),
        "alpha must be in [0, 1]",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iterations":0}"#),
            ],
            &config,
        ),
        "max_iterations must be at least 1",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"depth_limit":-1}"#),
            ],
            &config,
        ),
        "depth_limit must be non-negative",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Louvain,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"resolution":0}"#),
            ],
            &config,
        ),
        "resolution must be positive",
    );
}

async fn assert_sql_schema(ctx: &SessionContext, sql: &str, algorithm: CugraphAlgorithm) {
    let df = ctx.sql(sql).await.unwrap();
    let actual = df.schema().inner();
    let expected = stable_result_schema(algorithm);
    assert_eq!(actual.fields().len(), expected.fields().len());
    for (actual, expected) in actual.fields().iter().zip(expected.fields()) {
        assert_eq!(actual.name(), expected.name());
        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(actual.is_nullable(), expected.is_nullable());
    }
}

fn assert_error_contains(result: datafusion::common::Result<datafusion_cudf::CugraphAlgorithmSpec>, expected: &str) {
    let error = result.unwrap_err().to_string();
    assert!(error.contains(expected), "expected error containing {expected:?}, got {error:?}",);
}

#[test]
fn cugraph_sql_planning_schema_contracts_are_exact() {
    assert_eq!(
        schema_fields(CugraphAlgorithm::PageRank),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Bfs),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Louvain),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false),],
    );
}

#[tokio::test]
async fn cugraph_sql_planning_direct_table_louvain_physical_plan_preserves_edge_schema_order() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-louvain-planning");
    let path = dir.join("tiny_communities.parquet");
    write_tiny_communities(&path);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_communities", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('tiny_communities', 'src', 'dst', 'weight', '{\"directed\":false}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert_eq!(
        plan.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["vertex", "partition"]
    );

    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
#[ignore = "requires CUDA plus Python cudf/cugraph for the official oracle"]
async fn cugraph_sql_gpu_pagerank_matches_python_cugraph_oracle() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-pagerank");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let Some(oracle) = python_pagerank_oracle(&path) else {
        eprintln!("skipping cuGraph PageRank e2e because the Python cuGraph oracle is unavailable");
        fs::remove_dir_all(dir).ok();
        return;
    };
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(
        "CREATE VIEW graph_edges AS \
         SELECT src, dst, weight FROM tiny_directed_edges WHERE active",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('graph_edges', 'src', 'dst', 'weight') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let actual = pagerank_rows(&batches);

    assert_eq!(actual.len(), oracle.rows.len());
    for (actual, expected) in actual.iter().zip(&oracle.rows) {
        assert_eq!(actual.vertex, expected.vertex);
        let delta = (actual.value - expected.value).abs();
        assert!(
            delta <= 1.0e-6,
            "vertex {} PageRank mismatch: actual={} expected={} delta={}",
            actual.vertex,
            actual.value,
            expected.value,
            delta
        );
    }
    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
#[ignore = "requires CUDA plus Python cudf/cugraph for the official oracle"]
async fn cugraph_sql_gpu_bfs_matches_python_cugraph_oracle() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-bfs");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let Some(oracle) = python_bfs_oracle(&path) else {
        eprintln!("skipping cuGraph BFS e2e because the Python cuGraph oracle is unavailable");
        fs::remove_dir_all(dir).ok();
        return;
    };
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(
        "CREATE VIEW graph_edges AS \
         SELECT src, dst, weight FROM tiny_directed_edges WHERE active",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('graph_edges', 0, 'src', 'dst', 'weight') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let actual = bfs_rows(&batches);

    assert_eq!(actual, oracle.rows);
    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
#[ignore = "requires CUDA plus Python cudf/cugraph for the official oracle"]
async fn cugraph_sql_gpu_louvain_matches_python_cugraph_oracle() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-louvain");
    let path = dir.join("tiny_communities.parquet");
    write_tiny_communities(&path);

    let Some(oracle) = python_louvain_oracle(&path) else {
        eprintln!("skipping cuGraph Louvain e2e because the Python cuGraph oracle is unavailable");
        fs::remove_dir_all(dir).ok();
        return;
    };
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_communities", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let batches = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('tiny_communities', 'src', 'dst', 'weight', '{\"directed\":false,\"resolution\":1.0}') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let actual = louvain_rows(&batches);

    assert_eq!(actual, oracle.rows);
    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
#[ignore = "requires CUDA plus Python cudf/cugraph for the official oracle"]
async fn cugraph_sql_gpu_pagerank_filter_sort_limit_stays_gpu_resident_until_collect() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-pagerank-compose");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let Some(oracle) = python_pagerank_oracle(&path) else {
        eprintln!("skipping cuGraph PageRank composition e2e because the Python cuGraph oracle is unavailable");
        fs::remove_dir_all(dir).ok();
        return;
    };
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(
        "CREATE VIEW graph_edges AS \
         SELECT src, dst, weight FROM tiny_directed_edges WHERE active",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('graph_edges', 'src', 'dst', 'weight') \
             WHERE value > 0 \
             ORDER BY value DESC, vertex ASC \
             LIMIT 3",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("NativeCudfComposedExec"),
        "PageRank post-algorithm filter/sort/limit should be a composed native GPU handoff:\n{plan_display}"
    );
    assert!(
        plan_display.contains("CugraphAlgoExec"),
        "composed native plan should keep cuGraph as the GPU-resident producer child:\n{plan_display}"
    );
    let composed =
        find_composed_exec(plan.as_ref()).expect("PageRank composition plan should contain NativeCudfComposedExec");
    let native_plan = composed.native_plan().to_string();
    for expected in ["GpuResidentSource", "Filter", "Sort", "Limit"] {
        assert!(
            native_plan.contains(expected),
            "composed native consumer should include {expected} over the cuGraph result:\n{native_plan}"
        );
    }

    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let actual = pagerank_rows(&batches);
    let mut expected = oracle.rows.clone();
    expected.retain(|row| row.value > 0.0);
    expected.sort_by(|left, right| {
        right.value.partial_cmp(&left.value).unwrap().then_with(|| left.vertex.cmp(&right.vertex))
    });
    expected.truncate(3);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.vertex, expected.vertex);
        let delta = (actual.value - expected.value).abs();
        assert!(
            delta <= 1.0e-6,
            "vertex {} composed PageRank mismatch: actual={} expected={} delta={}",
            actual.vertex,
            actual.value,
            expected.value,
            delta
        );
    }
    fs::remove_dir_all(dir).ok();
}

fn schema_fields(algorithm: CugraphAlgorithm) -> Vec<Field> {
    stable_result_schema(algorithm).fields().iter().map(|field| field.as_ref().clone()).collect()
}

fn find_composed_exec(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> Option<&NativeCudfComposedExec> {
    if let Some(composed) = plan.as_any().downcast_ref::<NativeCudfComposedExec>() {
        return Some(composed);
    }
    plan.children().into_iter().find_map(|child| find_composed_exec(child.as_ref()))
}

#[derive(Debug, Clone)]
struct PageRankRow {
    vertex: i64,
    value: f64,
}

#[derive(Debug)]
struct PythonPageRankOracle {
    versions: String,
    rows: Vec<PageRankRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BfsRow {
    vertex: i64,
    distance: i64,
    predecessor: i64,
}

#[derive(Debug)]
struct PythonBfsOracle {
    versions: String,
    rows: Vec<BfsRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LouvainRow {
    vertex: i64,
    partition: i64,
}

#[derive(Debug)]
struct PythonLouvainOracle {
    versions: String,
    rows: Vec<LouvainRow>,
}

fn pagerank_rows(batches: &[RecordBatch]) -> Vec<PageRankRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let values = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for index in 0..batch.num_rows() {
            rows.push(PageRankRow { vertex: vertices.value(index), value: values.value(index) });
        }
    }
    rows
}

fn bfs_rows(batches: &[RecordBatch]) -> Vec<BfsRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let predecessors = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for index in 0..batch.num_rows() {
            rows.push(BfsRow {
                vertex: vertices.value(index),
                distance: distances.value(index),
                predecessor: predecessors.value(index),
            });
        }
    }
    rows
}

fn louvain_rows(batches: &[RecordBatch]) -> Vec<LouvainRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let partitions = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for index in 0..batch.num_rows() {
            rows.push(LouvainRow { vertex: vertices.value(index), partition: partitions.value(index) });
        }
    }
    rows
}

fn python_pagerank_oracle(path: &Path) -> Option<PythonPageRankOracle> {
    let script = r#"
import json
import sys

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

df = cudf.read_parquet(sys.argv[1])
df = df[df["active"]][["src", "dst", "weight"]]
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight", renumber=True)
result = cugraph.pagerank(graph, alpha=0.85, tol=1.0e-6, max_iter=100)
value_col = "pagerank" if "pagerank" in result.columns else [name for name in result.columns if name != "vertex"][0]
result = result.rename(columns={value_col: "value"})
result["vertex"] = result["vertex"].astype("int64")
result["value"] = result["value"].astype("float64")
result = result.sort_values("vertex")
rows = [{"vertex": int(row.vertex), "value": float(row.value)} for row in result.to_pandas().itertuples()]
print(json.dumps({
    "versions": {
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    },
    "rows": rows,
}))
"#;
    let output = match Command::new("python3").arg("-c").arg(script).arg(path).output() {
        Ok(output) => output,
        Err(error) => {
            eprintln!("Python cuGraph oracle unavailable: {error}");
            return None;
        },
    };
    if output.status.code() == Some(77) {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    assert!(
        output.status.success(),
        "Python cuGraph oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let versions = value["versions"].to_string();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| PageRankRow { vertex: row["vertex"].as_i64().unwrap(), value: row["value"].as_f64().unwrap() })
        .collect();
    Some(PythonPageRankOracle { versions, rows })
}

fn python_bfs_oracle(path: &Path) -> Option<PythonBfsOracle> {
    let script = r#"
import json
import sys

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

df = cudf.read_parquet(sys.argv[1])
df = df[df["active"]][["src", "dst", "weight"]]
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight", renumber=True)
result = cugraph.bfs(graph, 0)
result["vertex"] = result["vertex"].astype("int64")
result["distance"] = result["distance"].astype("int64")
result["predecessor"] = result["predecessor"].astype("int64")
result = result.sort_values("vertex")
rows = [
    {"vertex": int(row.vertex), "distance": int(row.distance), "predecessor": int(row.predecessor)}
    for row in result.to_pandas().itertuples()
]
print(json.dumps({
    "versions": {
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    },
    "rows": rows,
}))
"#;
    let output = match Command::new("python3").arg("-c").arg(script).arg(path).output() {
        Ok(output) => output,
        Err(error) => {
            eprintln!("Python cuGraph oracle unavailable: {error}");
            return None;
        },
    };
    if output.status.code() == Some(77) {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    assert!(
        output.status.success(),
        "Python cuGraph oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let versions = value["versions"].to_string();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| BfsRow {
            vertex: row["vertex"].as_i64().unwrap(),
            distance: row["distance"].as_i64().unwrap(),
            predecessor: row["predecessor"].as_i64().unwrap(),
        })
        .collect();
    Some(PythonBfsOracle { versions, rows })
}

fn python_louvain_oracle(path: &Path) -> Option<PythonLouvainOracle> {
    let script = r#"
import json
import sys

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

df = cudf.read_parquet(sys.argv[1])
graph = cugraph.Graph(directed=False)
graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight", renumber=True)
result, modularity = cugraph.louvain(graph, max_level=100, resolution=1.0, threshold=1.0e-7)
result["vertex"] = result["vertex"].astype("int64")
result["partition"] = result["partition"].astype("int64")
result = result.sort_values("vertex")
rows = [
    {"vertex": int(row.vertex), "partition": int(row.partition)}
    for row in result.to_pandas().itertuples()
]
print(json.dumps({
    "versions": {
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    },
    "modularity": float(modularity),
    "rows": rows,
}))
"#;
    let output = match Command::new("python3").arg("-c").arg(script).arg(path).output() {
        Ok(output) => output,
        Err(error) => {
            eprintln!("Python cuGraph oracle unavailable: {error}");
            return None;
        },
    };
    if output.status.code() == Some(77) {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    assert!(
        output.status.success(),
        "Python cuGraph oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let versions = value["versions"].to_string();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| LouvainRow {
            vertex: row["vertex"].as_i64().unwrap(),
            partition: row["partition"].as_i64().unwrap(),
        })
        .collect();
    Some(PythonLouvainOracle { versions, rows })
}

fn write_tiny_directed_edges(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 0, 1, 2, 3, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3, 3, 4, 5])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true, true, true, true, true, false])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_tiny_communities(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 0, 4, 5, 3, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
