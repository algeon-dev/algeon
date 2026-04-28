//! Fixture-scale SQL e2e coverage for relation-backed cuGraph BFS predicates.
//!
//! This is an explicit tools acceptance command because it needs CUDA and
//! Python cuGraph.
//! It writes a deterministic numeric graph fixture under `fixture/graph/cugraph/`, which
//! is gitignored, then compares normalized
//! `cugraph_bfs(... include_edges_table ...)` output against a Python cuGraph
//! oracle built from the same allowed-edge relation.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    cugraph_sql::CugraphSqlConfig, planner::NexusNativeOptimizerConfig, session::NexusSessionStateBuilderExt,
};

const DEFAULT_VERTEX_COUNT: i64 = 12_000;
const DEFAULT_STRING_VERTEX_COUNT: i64 = 4_000;

type NumericBfsRawEdge = (i64, i64, i64);
type NumericBfsAllowedEdge = (i64, i64);

pub async fn run_numeric_include_edges_acceptance() {
    let fixture = ensure_numeric_bfs_predicate_fixture();
    let oracle = python_bfs_include_edges_oracle(&fixture)
        .expect("Python cuGraph oracle is required for the selected BFS predicate acceptance test");

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bfs_raw_edges", fixture.raw_edges.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet("bfs_allowed_edges", fixture.allowed_edges.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();

    let sql_start = Instant::now();
    let batches = ctx
        .sql(
            "SELECT vertex, distance, reachable \
             FROM cugraph_bfs('bfs_raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"bfs_allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\",\
                     \"output_mode\":\"normalized\"}') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let sql_ms = sql_start.elapsed().as_secs_f64() * 1000.0;
    let (actual, reachable) = normalized_bfs_rows(&batches);

    assert_eq!(
        actual, oracle.rows,
        "SQL relation-backed include_edges BFS must match Python cuGraph over the filtered allowed-edge graph",
    );
    assert_eq!(actual.len(), fixture.vertex_count as usize);
    assert!(
        reachable.iter().all(|value| *value),
        "the deterministic fixture should make every vertex reachable through allowed edges",
    );
    assert!(
        actual.last().is_some_and(|row| row.distance > 1),
        "the fixture must prove predicate filtering by avoiding raw shortcut edges",
    );

    eprintln!(
        "cugraph_bfs_include_edges_fixture rows={} raw_edges={} allowed_edges={} sql_ms={:.3} python_total_ms={:.3} python_read_ms={:.3} python_join_ms={:.3} python_graph_ms={:.3} python_bfs_ms={:.3} versions={}",
        actual.len(),
        fixture.raw_edge_count,
        fixture.allowed_edge_count,
        sql_ms,
        oracle.total_ms,
        oracle.read_ms,
        oracle.join_ms,
        oracle.graph_ms,
        oracle.bfs_ms,
        oracle.versions,
    );

    if let Some(max_slowdown) = optional_max_slowdown() {
        assert!(
            sql_ms <= oracle.total_ms * max_slowdown,
            "SQL BFS predicate path exceeded configured slowdown: sql_ms={sql_ms:.3}, python_total_ms={:.3}, max_slowdown={max_slowdown}",
            oracle.total_ms,
        );
    }
}

pub async fn run_utf8_include_edges_acceptance() {
    let fixture = ensure_string_bfs_predicate_fixture();
    let oracle = python_bfs_utf8_include_edges_oracle(&fixture)
        .expect("Python cuGraph oracle is required for the selected UTF-8 BFS predicate acceptance test");

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bfs_utf8_raw_edges", fixture.raw_edges.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet(
        "bfs_utf8_allowed_edges",
        fixture.allowed_edges.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let sql_start = Instant::now();
    let batches = ctx
        .sql(
            "SELECT vertex, distance, reachable \
             FROM cugraph_bfs('bfs_utf8_raw_edges', 'v00000000', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"bfs_utf8_allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\",\
                     \"output_mode\":\"normalized\"}') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let sql_ms = sql_start.elapsed().as_secs_f64() * 1000.0;
    let (actual, reachable) = normalized_utf8_bfs_rows(&batches);

    assert_eq!(
        actual, oracle.rows,
        "SQL Utf8 relation-backed include_edges BFS must match Python cuGraph over the filtered allowed-edge graph",
    );
    assert_eq!(actual.len(), fixture.vertex_count as usize);
    assert!(
        reachable.iter().all(|value| *value),
        "the deterministic Utf8 fixture should make every vertex reachable through allowed edges",
    );
    assert!(
        actual.last().is_some_and(|row| row.distance > 1),
        "the Utf8 fixture must prove predicate filtering by avoiding raw shortcut edges",
    );

    eprintln!(
        "cugraph_bfs_utf8_include_edges_fixture rows={} raw_edges={} allowed_edges={} sql_ms={:.3} python_total_ms={:.3} python_read_ms={:.3} python_join_ms={:.3} python_graph_ms={:.3} python_bfs_ms={:.3} versions={}",
        actual.len(),
        fixture.raw_edge_count,
        fixture.allowed_edge_count,
        sql_ms,
        oracle.total_ms,
        oracle.read_ms,
        oracle.join_ms,
        oracle.graph_ms,
        oracle.bfs_ms,
        oracle.versions,
    );

    if let Some(max_slowdown) = optional_string_max_slowdown().or_else(optional_max_slowdown) {
        assert!(
            sql_ms <= oracle.total_ms * max_slowdown,
            "SQL Utf8 BFS predicate path exceeded configured slowdown: sql_ms={sql_ms:.3}, python_total_ms={:.3}, max_slowdown={max_slowdown}",
            oracle.total_ms,
        );
    }
}

pub async fn run_utf8_path_acceptance() {
    let fixture = ensure_string_bfs_predicate_fixture();
    let oracle = python_bfs_utf8_path_oracle(&fixture)
        .expect("Python cuGraph oracle is required for the selected UTF-8 BFS path acceptance test");

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bfs_utf8_raw_edges", fixture.raw_edges.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet(
        "bfs_utf8_allowed_edges",
        fixture.allowed_edges.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();
    ctx.register_parquet(
        "bfs_utf8_target_vertices",
        fixture.target_vertices.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let sql_start = Instant::now();
    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('bfs_utf8_raw_edges', 'v00000000', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"bfs_utf8_target_vertices\",\
                     \"target_vertex_col\":\"vertex_id\",\
                     \"include_edges_table\":\"bfs_utf8_allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}') \
             ORDER BY path_index",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let sql_ms = sql_start.elapsed().as_secs_f64() * 1000.0;
    let actual = utf8_bfs_path_rows(&batches);

    assert_eq!(actual, oracle.rows, "SQL Utf8 BFS path output must match the Python cuGraph predecessor path");
    assert!(actual.len() > 1, "path fixture should include at least one intermediate edge");

    eprintln!(
        "cugraph_bfs_utf8_path_fixture rows={} raw_edges={} allowed_edges={} target_vertices={} sql_ms={:.3} python_total_ms={:.3} python_read_ms={:.3} python_join_ms={:.3} python_graph_ms={:.3} python_bfs_ms={:.3} python_post_ms={:.3} versions={}",
        actual.len(),
        fixture.raw_edge_count,
        fixture.allowed_edge_count,
        fixture.target_vertex_count,
        sql_ms,
        oracle.total_ms,
        oracle.read_ms,
        oracle.join_ms,
        oracle.graph_ms,
        oracle.bfs_ms,
        oracle.post_ms,
        oracle.versions,
    );

    if let Some(max_slowdown) = optional_string_max_slowdown().or_else(optional_max_slowdown) {
        assert!(
            sql_ms <= oracle.total_ms * max_slowdown,
            "SQL Utf8 BFS path exceeded configured slowdown: sql_ms={sql_ms:.3}, python_total_ms={:.3}, max_slowdown={max_slowdown}",
            oracle.total_ms,
        );
    }
}

pub async fn run_utf8_vertex_predicates_acceptance() {
    let fixture = ensure_string_bfs_predicate_fixture();

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bfs_utf8_raw_edges", fixture.raw_edges.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet(
        "bfs_utf8_included_vertices",
        fixture.included_vertices.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();
    ctx.register_parquet(
        "bfs_utf8_excluded_vertices",
        fixture.excluded_vertices.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();
    ctx.register_parquet(
        "bfs_utf8_target_vertices",
        fixture.target_vertices.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let cases = [
        (
            "include_vertices",
            &fixture.included_vertices,
            "{\"construction_policy\":\"raw_libcugraph\",\
              \"include_vertices_table\":\"bfs_utf8_included_vertices\",\
              \"include_vertex_col\":\"vertex_id\",\
              \"output_mode\":\"normalized\"}",
        ),
        (
            "exclude_vertices",
            &fixture.excluded_vertices,
            "{\"construction_policy\":\"raw_libcugraph\",\
              \"exclude_vertices_table\":\"bfs_utf8_excluded_vertices\",\
              \"exclude_vertex_col\":\"vertex_id\",\
              \"output_mode\":\"normalized\"}",
        ),
        (
            "target_vertices",
            &fixture.target_vertices,
            "{\"construction_policy\":\"raw_libcugraph\",\
              \"target_vertices_table\":\"bfs_utf8_target_vertices\",\
              \"target_vertex_col\":\"vertex_id\",\
              \"return_target_info\":true,\
              \"output_mode\":\"normalized\"}",
        ),
    ];

    for (mode, predicate_path, options_json) in cases {
        let oracle = python_bfs_utf8_vertex_predicate_oracle(&fixture, predicate_path, mode)
            .expect("Python cuGraph oracle is required for the selected UTF-8 BFS predicate acceptance test");
        let select_list = if mode == "target_vertices" {
            "vertex, distance, reachable, target_found, target_distance"
        } else {
            "vertex, distance, reachable"
        };
        let sql = format!(
            "SELECT {select_list} \
             FROM cugraph_bfs('bfs_utf8_raw_edges', 'v00000000', 'src', 'dst', NULL, '{options_json}') \
             ORDER BY vertex"
        );
        let sql_start = Instant::now();
        let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
        let sql_ms = sql_start.elapsed().as_secs_f64() * 1000.0;
        let actual = normalized_utf8_bfs_reachability_rows(&batches);

        assert_eq!(actual, oracle.rows, "SQL Utf8 BFS {mode} predicate output must match the Python cuGraph oracle",);
        assert_eq!(actual.len(), fixture.vertex_count as usize);
        assert!(
            actual.iter().any(|row| !row.reachable),
            "{mode} fixture case should prove predicate filtering / early-stop with unreachable rows",
        );
        if mode == "target_vertices" {
            assert!(
                actual.iter().all(|row| row.target_found == Some(true) && row.target_distance.is_some()),
                "target_vertices fixture should expose target metadata on every SQL output row",
            );
        }

        eprintln!(
            "cugraph_bfs_utf8_{mode}_fixture rows={} raw_edges={} included_vertices={} excluded_vertices={} target_vertices={} sql_ms={:.3} python_total_ms={:.3} python_read_ms={:.3} python_predicate_ms={:.3} python_graph_ms={:.3} python_bfs_ms={:.3} python_post_ms={:.3} versions={}",
            actual.len(),
            fixture.raw_edge_count,
            fixture.included_vertex_count,
            fixture.excluded_vertex_count,
            fixture.target_vertex_count,
            sql_ms,
            oracle.total_ms,
            oracle.read_ms,
            oracle.predicate_ms,
            oracle.graph_ms,
            oracle.bfs_ms,
            oracle.post_ms,
            oracle.versions,
        );

        if let Some(max_slowdown) = optional_string_max_slowdown().or_else(optional_max_slowdown) {
            assert!(
                sql_ms <= oracle.total_ms * max_slowdown,
                "SQL Utf8 BFS {mode} predicate path exceeded configured slowdown: sql_ms={sql_ms:.3}, python_total_ms={:.3}, max_slowdown={max_slowdown}",
                oracle.total_ms,
            );
        }
    }
}

#[derive(Debug)]
struct BfsPredicateFixture {
    raw_edges: PathBuf,
    allowed_edges: PathBuf,
    vertex_count: i64,
    raw_edge_count: usize,
    allowed_edge_count: usize,
}

#[derive(Debug)]
struct StringBfsPredicateFixture {
    raw_edges: PathBuf,
    allowed_edges: PathBuf,
    included_vertices: PathBuf,
    excluded_vertices: PathBuf,
    target_vertices: PathBuf,
    vertex_count: i64,
    raw_edge_count: usize,
    allowed_edge_count: usize,
    included_vertex_count: usize,
    excluded_vertex_count: usize,
    target_vertex_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BfsRow {
    vertex: i64,
    distance: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StringBfsRow {
    vertex: String,
    distance: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StringBfsReachabilityRow {
    vertex: String,
    distance: Option<i64>,
    reachable: bool,
    target_found: Option<bool>,
    target_distance: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StringBfsPathRow {
    path_index: i64,
    source: String,
    target: String,
    vertex: String,
    distance: i64,
}

#[derive(Debug)]
struct PythonBfsOracle {
    versions: String,
    rows: Vec<BfsRow>,
    total_ms: f64,
    read_ms: f64,
    join_ms: f64,
    graph_ms: f64,
    bfs_ms: f64,
}

#[derive(Debug)]
struct PythonUtf8BfsOracle {
    versions: String,
    rows: Vec<StringBfsRow>,
    total_ms: f64,
    read_ms: f64,
    join_ms: f64,
    graph_ms: f64,
    bfs_ms: f64,
}

#[derive(Debug)]
struct PythonUtf8BfsReachabilityOracle {
    versions: String,
    rows: Vec<StringBfsReachabilityRow>,
    total_ms: f64,
    read_ms: f64,
    predicate_ms: f64,
    graph_ms: f64,
    bfs_ms: f64,
    post_ms: f64,
}

#[derive(Debug)]
struct PythonUtf8BfsPathOracle {
    versions: String,
    rows: Vec<StringBfsPathRow>,
    total_ms: f64,
    read_ms: f64,
    join_ms: f64,
    graph_ms: f64,
    bfs_ms: f64,
    post_ms: f64,
}

fn ensure_numeric_bfs_predicate_fixture() -> BfsPredicateFixture {
    let vertex_count = std::env::var("NEXUS_CUGRAPH_BFS_PREDICATE_VERTEX_COUNT")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value >= 2)
        .unwrap_or(DEFAULT_VERTEX_COUNT);
    let root = std::env::var_os("NEXUS_CUGRAPH_BFS_PREDICATE_FIXTURE_DIR").map(PathBuf::from).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixture")
            .join("data")
            .join("cugraph")
            .join("bfs_predicate_numeric")
            .join("parquet")
    });
    fs::create_dir_all(&root).unwrap();
    let raw_edges = root.join("raw_edges.parquet");
    let allowed_edges = root.join("allowed_edges.parquet");
    let metadata = root.join("metadata.txt");

    let expected_metadata = format!("vertex_count={vertex_count}\nversion=1\n");
    if raw_edges.exists()
        && allowed_edges.exists()
        && fs::read_to_string(&metadata).is_ok_and(|contents| contents == expected_metadata)
    {
        let (raw_edge_count, allowed_edge_count) = expected_edge_counts(vertex_count);
        return BfsPredicateFixture { raw_edges, allowed_edges, vertex_count, raw_edge_count, allowed_edge_count };
    }

    let (raw, allowed) = generate_numeric_bfs_edges(vertex_count);
    write_raw_edges(&raw_edges, &raw);
    write_allowed_edges(&allowed_edges, &allowed);
    fs::write(metadata, expected_metadata).unwrap();
    BfsPredicateFixture {
        raw_edges,
        allowed_edges,
        vertex_count,
        raw_edge_count: raw.len(),
        allowed_edge_count: allowed.len(),
    }
}

fn ensure_string_bfs_predicate_fixture() -> StringBfsPredicateFixture {
    let vertex_count = std::env::var("NEXUS_CUGRAPH_BFS_PREDICATE_STRING_VERTEX_COUNT")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value >= 2)
        .unwrap_or(DEFAULT_STRING_VERTEX_COUNT);
    let root =
        std::env::var_os("NEXUS_CUGRAPH_BFS_PREDICATE_STRING_FIXTURE_DIR").map(PathBuf::from).unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixture")
                .join("data")
                .join("cugraph")
                .join("bfs_predicate_utf8")
                .join("parquet")
        });
    fs::create_dir_all(&root).unwrap();
    let raw_edges = root.join("raw_edges.parquet");
    let allowed_edges = root.join("allowed_edges.parquet");
    let included_vertices = root.join("included_vertices.parquet");
    let excluded_vertices = root.join("excluded_vertices.parquet");
    let target_vertices = root.join("target_vertices.parquet");
    let metadata = root.join("metadata.txt");

    let expected_metadata = format!("vertex_count={vertex_count}\nversion=2\n");
    if raw_edges.exists()
        && allowed_edges.exists()
        && included_vertices.exists()
        && excluded_vertices.exists()
        && target_vertices.exists()
        && fs::read_to_string(&metadata).is_ok_and(|contents| contents == expected_metadata)
    {
        let (raw_edge_count, allowed_edge_count) = expected_edge_counts(vertex_count);
        let (included_vertex_count, excluded_vertex_count, target_vertex_count) =
            expected_string_vertex_predicate_counts(vertex_count);
        return StringBfsPredicateFixture {
            raw_edges,
            allowed_edges,
            included_vertices,
            excluded_vertices,
            target_vertices,
            vertex_count,
            raw_edge_count,
            allowed_edge_count,
            included_vertex_count,
            excluded_vertex_count,
            target_vertex_count,
        };
    }

    let (raw, allowed) = generate_numeric_bfs_edges(vertex_count);
    let (included, excluded, targets) = generate_string_vertex_predicates(vertex_count);
    write_string_raw_edges(&raw_edges, &raw);
    write_string_allowed_edges(&allowed_edges, &allowed);
    write_string_vertices(&included_vertices, "vertex_id", &included);
    write_string_vertices(&excluded_vertices, "vertex_id", &excluded);
    write_string_vertices(&target_vertices, "vertex_id", &targets);
    fs::write(metadata, expected_metadata).unwrap();
    StringBfsPredicateFixture {
        raw_edges,
        allowed_edges,
        included_vertices,
        excluded_vertices,
        target_vertices,
        vertex_count,
        raw_edge_count: raw.len(),
        allowed_edge_count: allowed.len(),
        included_vertex_count: included.len(),
        excluded_vertex_count: excluded.len(),
        target_vertex_count: targets.len(),
    }
}

fn expected_edge_counts(vertex_count: i64) -> (usize, usize) {
    let (raw, allowed) = generate_numeric_bfs_edges(vertex_count);
    (raw.len(), allowed.len())
}

fn expected_string_vertex_predicate_counts(vertex_count: i64) -> (usize, usize, usize) {
    let (included, excluded, targets) = generate_string_vertex_predicates(vertex_count);
    (included.len(), excluded.len(), targets.len())
}

fn generate_numeric_bfs_edges(vertex_count: i64) -> (Vec<NumericBfsRawEdge>, Vec<NumericBfsAllowedEdge>) {
    let mut raw = Vec::new();
    let mut allowed = Vec::new();
    let mut edge_id = 0_i64;

    for src in 0..(vertex_count - 1) {
        let dst = src + 1;
        raw.push((edge_id, src, dst));
        allowed.push((src, dst));
        edge_id += 1;
    }
    for src in (0..vertex_count).step_by(17) {
        let dst = (src + 5).min(vertex_count - 1);
        if dst > src {
            raw.push((edge_id, src, dst));
            allowed.push((src, dst));
            edge_id += 1;
        }
    }
    for src in (0..vertex_count).step_by(53) {
        let dst = vertex_count - 1 - (src % 29);
        if dst > src + 5 {
            raw.push((edge_id, src, dst));
            edge_id += 1;
        }
    }

    (raw, allowed)
}

fn generate_string_vertex_predicates(vertex_count: i64) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let excluded =
        (0..vertex_count).filter(|vertex| *vertex != 0 && (*vertex - 13).rem_euclid(97) == 0).collect::<Vec<_>>();
    let included = (0..vertex_count).filter(|vertex| !excluded.contains(vertex)).collect::<Vec<_>>();
    let target = (vertex_count / 3).clamp(1, vertex_count - 1);
    let target = if excluded.contains(&target) { target + 1 } else { target }.min(vertex_count - 1);
    (included, excluded, vec![target])
}

fn write_raw_edges(path: &Path, rows: &[(i64, i64, i64)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Int64, false),
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(rows.iter().map(|(edge_id, _, _)| *edge_id).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Int64Array::from(rows.iter().map(|(_, src, _)| *src).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Int64Array::from(rows.iter().map(|(_, _, dst)| *dst).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64; rows.len()])) as ArrayRef,
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, schema, &batch);
}

fn write_allowed_edges(path: &Path, rows: &[(i64, i64)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, false),
        Field::new("allowed_dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(rows.iter().map(|(src, _)| *src).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Int64Array::from(rows.iter().map(|(_, dst)| *dst).collect::<Vec<_>>())) as ArrayRef,
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, schema, &batch);
}

fn write_string_raw_edges(path: &Path, rows: &[(i64, i64, i64)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Int64, false),
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(rows.iter().map(|(edge_id, _, _)| *edge_id).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(StringArray::from(rows.iter().map(|(_, src, _)| vertex_label(*src)).collect::<Vec<_>>()))
                as ArrayRef,
            Arc::new(StringArray::from(rows.iter().map(|(_, _, dst)| vertex_label(*dst)).collect::<Vec<_>>()))
                as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64; rows.len()])) as ArrayRef,
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, schema, &batch);
}

fn write_string_allowed_edges(path: &Path, rows: &[(i64, i64)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Utf8, false),
        Field::new("allowed_dst", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(rows.iter().map(|(src, _)| vertex_label(*src)).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(StringArray::from(rows.iter().map(|(_, dst)| vertex_label(*dst)).collect::<Vec<_>>())) as ArrayRef,
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, schema, &batch);
}

fn write_string_vertices(path: &Path, column_name: &str, rows: &[i64]) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(rows.iter().map(|vertex| vertex_label(*vertex)).collect::<Vec<_>>()))
            as ArrayRef],
    )
    .unwrap();
    write_single_batch_parquet(path, schema, &batch);
}

fn vertex_label(vertex: i64) -> String {
    format!("v{vertex:08}")
}

fn write_single_batch_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) {
    let temp = path
        .with_extension(format!("parquet.tmp.{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
    let file = fs::File::create(&temp).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    fs::rename(temp, path).unwrap();
}

fn python_bfs_include_edges_oracle(fixture: &BfsPredicateFixture) -> Option<PythonBfsOracle> {
    let script = r#"
import json
import sys
import time

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

t0 = time.perf_counter()
edges = cudf.read_parquet(sys.argv[1])
allowed = cudf.read_parquet(sys.argv[2])
t1 = time.perf_counter()
filtered = edges.merge(
    allowed,
    left_on=["src", "dst"],
    right_on=["allowed_src", "allowed_dst"],
    how="inner",
)[["src", "dst"]]
t2 = time.perf_counter()
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(filtered, source="src", destination="dst", renumber=True)
t3 = time.perf_counter()
result = cugraph.bfs(graph, start=0).sort_values("vertex").reset_index(drop=True)
t4 = time.perf_counter()
result["vertex"] = result["vertex"].astype("int64")
result["distance"] = result["distance"].astype("int64")
rows = [
    {"vertex": int(row.vertex), "distance": int(row.distance)}
    for row in result[["vertex", "distance"]].to_pandas().itertuples(index=False)
]
print(json.dumps({
    "versions": json.dumps({
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    }, sort_keys=True),
    "rows": rows,
    "total_ms": (t4 - t0) * 1000.0,
    "read_ms": (t1 - t0) * 1000.0,
    "join_ms": (t2 - t1) * 1000.0,
    "graph_ms": (t3 - t2) * 1000.0,
    "bfs_ms": (t4 - t3) * 1000.0,
}, sort_keys=True))
"#;
    let output = match Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&fixture.raw_edges)
        .arg(&fixture.allowed_edges)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            handle_python_oracle_unavailable(format!("Python cuGraph oracle unavailable: {error}"));
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
    }
    assert!(
        output.status.success(),
        "Python cuGraph BFS predicate oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Some(parse_python_bfs_oracle(&output.stdout))
}

fn python_bfs_utf8_include_edges_oracle(fixture: &StringBfsPredicateFixture) -> Option<PythonUtf8BfsOracle> {
    let script = r#"
import json
import sys
import time

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

t0 = time.perf_counter()
edges = cudf.read_parquet(sys.argv[1])
allowed = cudf.read_parquet(sys.argv[2])
t1 = time.perf_counter()
filtered = edges.merge(
    allowed,
    left_on=["src", "dst"],
    right_on=["allowed_src", "allowed_dst"],
    how="inner",
)[["src", "dst"]]
t2 = time.perf_counter()
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(filtered, source="src", destination="dst", renumber=True)
t3 = time.perf_counter()
result = cugraph.bfs(graph, start="v00000000").sort_values("vertex").reset_index(drop=True)
t4 = time.perf_counter()
result["distance"] = result["distance"].astype("int64")
rows = [
    {"vertex": str(row.vertex), "distance": int(row.distance)}
    for row in result[["vertex", "distance"]].to_pandas().itertuples(index=False)
]
print(json.dumps({
    "versions": json.dumps({
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    }, sort_keys=True),
    "rows": rows,
    "total_ms": (t4 - t0) * 1000.0,
    "read_ms": (t1 - t0) * 1000.0,
    "join_ms": (t2 - t1) * 1000.0,
    "graph_ms": (t3 - t2) * 1000.0,
    "bfs_ms": (t4 - t3) * 1000.0,
}, sort_keys=True))
"#;
    let output = match Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&fixture.raw_edges)
        .arg(&fixture.allowed_edges)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            handle_python_oracle_unavailable(format!("Python cuGraph oracle unavailable: {error}"));
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
    }
    assert!(
        output.status.success(),
        "Python cuGraph Utf8 BFS predicate oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Some(parse_python_utf8_bfs_oracle(&output.stdout))
}

fn python_bfs_utf8_path_oracle(fixture: &StringBfsPredicateFixture) -> Option<PythonUtf8BfsPathOracle> {
    let script = r#"
import json
import math
import sys
import time

try:
    import cudf
    import cugraph
    import pandas as pd
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

source = "v00000000"
t0 = time.perf_counter()
edges = cudf.read_parquet(sys.argv[1])
allowed = cudf.read_parquet(sys.argv[2])
targets = cudf.read_parquet(sys.argv[3])
t1 = time.perf_counter()
target_values = targets["vertex_id"].to_pandas().tolist()
if len(target_values) != 1:
    raise ValueError(f"path oracle requires exactly one target vertex, found {len(target_values)}")
target = str(target_values[0])
filtered = edges.merge(
    allowed,
    left_on=["src", "dst"],
    right_on=["allowed_src", "allowed_dst"],
    how="inner",
)[["src", "dst"]]
t2 = time.perf_counter()
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(filtered, source="src", destination="dst", renumber=True)
t3 = time.perf_counter()
result = cugraph.bfs(graph, start=source)
t4 = time.perf_counter()
pdf = result[["vertex", "distance", "predecessor"]].to_pandas()
nodes = {}
for row in pdf.itertuples(index=False):
    vertex = str(row.vertex)
    distance = int(row.distance)
    predecessor = None
    if not pd.isna(row.predecessor):
        predecessor = str(row.predecessor)
    nodes[vertex] = (distance, predecessor)

path = []
if target in nodes and nodes[target][0] < 2147483647:
    current = target
    seen = set()
    while True:
        if current in seen:
            raise ValueError("predecessor cycle while reconstructing path")
        seen.add(current)
        path.append(current)
        if current == source:
            break
        predecessor = nodes[current][1]
        if predecessor is None:
            raise ValueError(f"missing predecessor before source for {current}")
        current = predecessor
    path.reverse()
rows = [
    {
        "path_index": index,
        "source": source,
        "target": target,
        "vertex": vertex,
        "distance": int(nodes[vertex][0]),
    }
    for index, vertex in enumerate(path)
]
t5 = time.perf_counter()
print(json.dumps({
    "versions": json.dumps({
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    }, sort_keys=True),
    "rows": rows,
    "total_ms": (t5 - t0) * 1000.0,
    "read_ms": (t1 - t0) * 1000.0,
    "join_ms": (t2 - t1) * 1000.0,
    "graph_ms": (t3 - t2) * 1000.0,
    "bfs_ms": (t4 - t3) * 1000.0,
    "post_ms": (t5 - t4) * 1000.0,
}, sort_keys=True))
"#;
    let output = match Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&fixture.raw_edges)
        .arg(&fixture.allowed_edges)
        .arg(&fixture.target_vertices)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            handle_python_oracle_unavailable(format!("Python cuGraph oracle unavailable: {error}"));
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
    }
    assert!(
        output.status.success(),
        "Python cuGraph Utf8 BFS path oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Some(parse_python_utf8_bfs_path_oracle(&output.stdout))
}

fn python_bfs_utf8_vertex_predicate_oracle(
    fixture: &StringBfsPredicateFixture,
    predicate_path: &Path,
    mode: &str,
) -> Option<PythonUtf8BfsReachabilityOracle> {
    let script = r#"
import json
import sys
import time

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

mode = sys.argv[3]
t0 = time.perf_counter()
edges = cudf.read_parquet(sys.argv[1])[["src", "dst"]]
predicate = cudf.read_parquet(sys.argv[2])
t1 = time.perf_counter()
labels = cudf.concat([edges["src"], edges["dst"]]).drop_duplicates().sort_values()
filtered = edges
target_distance = None
if mode == "include_vertices":
    vertices = predicate["vertex_id"]
    filtered = edges[edges["src"].isin(vertices) & edges["dst"].isin(vertices)]
elif mode == "exclude_vertices":
    vertices = predicate["vertex_id"]
    filtered = edges[~edges["src"].isin(vertices) & ~edges["dst"].isin(vertices)]
elif mode == "target_vertices":
    pass
else:
    raise ValueError(f"unsupported mode {mode}")
t2 = time.perf_counter()
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(filtered, source="src", destination="dst", renumber=True)
t3 = time.perf_counter()
result = cugraph.bfs(graph, start="v00000000")
t4 = time.perf_counter()
result["distance"] = result["distance"].astype("int64")
if mode == "target_vertices":
    target_hits = result.merge(predicate, left_on="vertex", right_on="vertex_id", how="inner")
    reachable_targets = target_hits[target_hits["distance"] < 2147483647]
    if len(reachable_targets) > 0:
        target_distance = int(reachable_targets["distance"].min())
target_found = target_distance is not None

distance_rows = result[["vertex", "distance"]].to_pandas()
distance_by_vertex = {
    str(row.vertex): int(row.distance)
    for row in distance_rows.itertuples(index=False)
}
rows = []
for vertex in [str(value) for value in labels.to_pandas().tolist()]:
    distance = distance_by_vertex.get(vertex)
    if distance is not None and distance >= 2147483647:
        distance = None
    if target_distance is not None and distance is not None and distance > target_distance:
        distance = None
    rows.append({
        "vertex": vertex,
        "distance": distance,
        "reachable": distance is not None,
        "target_found": target_found if mode == "target_vertices" else None,
        "target_distance": target_distance if mode == "target_vertices" else None,
    })
t5 = time.perf_counter()
print(json.dumps({
    "versions": json.dumps({
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    }, sort_keys=True),
    "rows": rows,
    "total_ms": (t5 - t0) * 1000.0,
    "read_ms": (t1 - t0) * 1000.0,
    "predicate_ms": (t2 - t1) * 1000.0,
    "graph_ms": (t3 - t2) * 1000.0,
    "bfs_ms": (t4 - t3) * 1000.0,
    "post_ms": (t5 - t4) * 1000.0,
}, sort_keys=True))
"#;
    let output = match Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&fixture.raw_edges)
        .arg(predicate_path)
        .arg(mode)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            handle_python_oracle_unavailable(format!("Python cuGraph oracle unavailable: {error}"));
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
    }
    assert!(
        output.status.success(),
        "Python cuGraph Utf8 BFS {mode} predicate oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Some(parse_python_utf8_bfs_reachability_oracle(&output.stdout))
}

fn normalized_bfs_rows(batches: &[RecordBatch]) -> (Vec<BfsRow>, Vec<bool>) {
    let mut rows = Vec::new();
    let mut reachable = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let reachable_column = batch.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        for row in 0..batch.num_rows() {
            assert!(!distances.is_null(row), "reachable fixture vertex should have a normalized distance");
            rows.push(BfsRow { vertex: vertices.value(row), distance: distances.value(row) });
            reachable.push(reachable_column.value(row));
        }
    }
    (rows, reachable)
}

fn normalized_utf8_bfs_rows(batches: &[RecordBatch]) -> (Vec<StringBfsRow>, Vec<bool>) {
    let mut rows = Vec::new();
    let mut reachable = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let reachable_column = batch.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        for row in 0..batch.num_rows() {
            assert!(!distances.is_null(row), "reachable Utf8 fixture vertex should have a normalized distance");
            rows.push(StringBfsRow { vertex: vertices.value(row).to_string(), distance: distances.value(row) });
            reachable.push(reachable_column.value(row));
        }
    }
    (rows, reachable)
}

fn normalized_utf8_bfs_reachability_rows(batches: &[RecordBatch]) -> Vec<StringBfsReachabilityRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let reachable_column = batch.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        let target_found_column =
            (batch.num_columns() > 3).then(|| batch.column(3).as_any().downcast_ref::<BooleanArray>().unwrap());
        let target_distance_column =
            (batch.num_columns() > 4).then(|| batch.column(4).as_any().downcast_ref::<Int64Array>().unwrap());
        for row in 0..batch.num_rows() {
            let distance = (!distances.is_null(row)).then(|| distances.value(row));
            let reachable = reachable_column.value(row);
            assert_eq!(
                reachable,
                distance.is_some(),
                "normalized reachable must agree with nullable distance for vertex {}",
                vertices.value(row),
            );
            let target_found = target_found_column.map(|column| column.value(row));
            let target_distance =
                target_distance_column.and_then(|column| (!column.is_null(row)).then(|| column.value(row)));
            rows.push(StringBfsReachabilityRow {
                vertex: vertices.value(row).to_string(),
                distance,
                reachable,
                target_found,
                target_distance,
            });
        }
    }
    rows
}

fn utf8_bfs_path_rows(batches: &[RecordBatch]) -> Vec<StringBfsPathRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let path_indices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let sources = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let targets = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let vertices = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let distances = batch.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
        for row in 0..batch.num_rows() {
            rows.push(StringBfsPathRow {
                path_index: path_indices.value(row),
                source: sources.value(row).to_string(),
                target: targets.value(row).to_string(),
                vertex: vertices.value(row).to_string(),
                distance: distances.value(row),
            });
        }
    }
    rows
}

fn parse_python_bfs_oracle(stdout: &[u8]) -> PythonBfsOracle {
    let value: serde_json::Value = serde_json::from_slice(stdout).unwrap();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| BfsRow { vertex: row["vertex"].as_i64().unwrap(), distance: row["distance"].as_i64().unwrap() })
        .collect();
    PythonBfsOracle {
        versions: value["versions"].as_str().unwrap().to_string(),
        rows,
        total_ms: json_f64(&value, "total_ms"),
        read_ms: json_f64(&value, "read_ms"),
        join_ms: json_f64(&value, "join_ms"),
        graph_ms: json_f64(&value, "graph_ms"),
        bfs_ms: json_f64(&value, "bfs_ms"),
    }
}

fn parse_python_utf8_bfs_oracle(stdout: &[u8]) -> PythonUtf8BfsOracle {
    let value: serde_json::Value = serde_json::from_slice(stdout).unwrap();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| StringBfsRow {
            vertex: row["vertex"].as_str().unwrap().to_string(),
            distance: row["distance"].as_i64().unwrap(),
        })
        .collect();
    PythonUtf8BfsOracle {
        versions: value["versions"].as_str().unwrap().to_string(),
        rows,
        total_ms: json_f64(&value, "total_ms"),
        read_ms: json_f64(&value, "read_ms"),
        join_ms: json_f64(&value, "join_ms"),
        graph_ms: json_f64(&value, "graph_ms"),
        bfs_ms: json_f64(&value, "bfs_ms"),
    }
}

fn parse_python_utf8_bfs_path_oracle(stdout: &[u8]) -> PythonUtf8BfsPathOracle {
    let value: serde_json::Value = serde_json::from_slice(stdout).unwrap();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| StringBfsPathRow {
            path_index: row["path_index"].as_i64().unwrap(),
            source: row["source"].as_str().unwrap().to_string(),
            target: row["target"].as_str().unwrap().to_string(),
            vertex: row["vertex"].as_str().unwrap().to_string(),
            distance: row["distance"].as_i64().unwrap(),
        })
        .collect();
    PythonUtf8BfsPathOracle {
        versions: value["versions"].as_str().unwrap().to_string(),
        rows,
        total_ms: json_f64(&value, "total_ms"),
        read_ms: json_f64(&value, "read_ms"),
        join_ms: json_f64(&value, "join_ms"),
        graph_ms: json_f64(&value, "graph_ms"),
        bfs_ms: json_f64(&value, "bfs_ms"),
        post_ms: json_f64(&value, "post_ms"),
    }
}

fn parse_python_utf8_bfs_reachability_oracle(stdout: &[u8]) -> PythonUtf8BfsReachabilityOracle {
    let value: serde_json::Value = serde_json::from_slice(stdout).unwrap();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| StringBfsReachabilityRow {
            vertex: row["vertex"].as_str().unwrap().to_string(),
            distance: row["distance"].as_i64(),
            reachable: row["reachable"].as_bool().unwrap(),
            target_found: row["target_found"].as_bool(),
            target_distance: row["target_distance"].as_i64(),
        })
        .collect();
    PythonUtf8BfsReachabilityOracle {
        versions: value["versions"].as_str().unwrap().to_string(),
        rows,
        total_ms: json_f64(&value, "total_ms"),
        read_ms: json_f64(&value, "read_ms"),
        predicate_ms: json_f64(&value, "predicate_ms"),
        graph_ms: json_f64(&value, "graph_ms"),
        bfs_ms: json_f64(&value, "bfs_ms"),
        post_ms: json_f64(&value, "post_ms"),
    }
}

fn json_f64(value: &serde_json::Value, key: &str) -> f64 {
    value[key].as_f64().unwrap_or_else(|| panic!("Python oracle field {key} must be a finite JSON number"))
}

fn optional_max_slowdown() -> Option<f64> {
    std::env::var("NEXUS_CUGRAPH_BFS_PREDICATE_MAX_SLOWDOWN")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn optional_string_max_slowdown() -> Option<f64> {
    std::env::var("NEXUS_CUGRAPH_BFS_PREDICATE_STRING_MAX_SLOWDOWN")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn handle_python_oracle_unavailable(message: impl AsRef<str>) -> ! {
    crate::cugraph_optional_fixture::handle_cugraph_python_oracle_unavailable("cuGraph Python oracle", message);
}
