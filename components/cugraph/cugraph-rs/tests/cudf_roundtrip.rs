//! Integration tests for cugraph → cudf result pathways.
//!
//! These tests run actual GPU algorithms and convert results to cudf DataFrames,
//! verifying the full round-trip: GPU algorithm → result → cudf DataFrame → host Vec.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use cudf::{
    core::Table,
    interop::{InteropErrorKind, RuntimeCompletionAuthority},
    io::parquet,
};
use cugraph::{
    array::DeviceArray,
    centrality::PageRankParams,
    cudf_interop::{
        DataFrameGraphOptions, ParquetGraphOptions, TableGraphOptions, graph_from_dataframe, graph_from_parquet,
        graph_from_prepared_table_export, graph_from_table_view,
    },
    graph::{Forward, Graph, GraphBuilder, Transposed},
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};

static NEXT_TEMP_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);
const EDGE_COLUMN_NAMES: &[&str] = &["src", "dst", "weight"];

fn explicit_runtime() -> cudf::Runtime {
    cudf::Runtime::builder().build().expect("runtime")
}

fn bfs_graph_table(runtime: &cudf::Runtime) -> Table {
    let src = runtime.column_from_slice(&[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = runtime.column_from_slice(&[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");
    let wgt = runtime.column_from_slice(&[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]).expect("wgt");
    Table::from_columns(vec![src, dst, wgt]).expect("bfs graph table")
}

fn bfs_graph_dataframe(runtime: &cudf::Runtime) -> cudf::DataFrame {
    let src = runtime.series("src", &[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = runtime.series("dst", &[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");
    let wgt = runtime.series("weight", &[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]).expect("weight");
    cudf::DataFrame::from_series([src, dst, wgt]).expect("bfs graph dataframe")
}

fn pagerank_graph_dataframe(runtime: &cudf::Runtime) -> cudf::DataFrame {
    let src = runtime.series("src", &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = runtime.series("dst", &[1i32, 2, 0, 0, 1, 2]).expect("dst");
    let wgt = runtime.series("weight", &[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0]).expect("weight");
    cudf::DataFrame::from_series([src, dst, wgt]).expect("pagerank graph dataframe")
}

fn i32_series_device_ptr(series: &cudf::Series) -> usize {
    series.as_column_view().expect("column view").device_slice::<i32>().expect("i32 device slice").as_ptr() as usize
}

fn f32_series_device_ptr(series: &cudf::Series) -> usize {
    series.as_column_view().expect("column view").device_slice::<f32>().expect("f32 device slice").as_ptr() as usize
}

fn assert_named_edge_columns(df: &cudf::DataFrame) {
    assert_eq!(df.column_names().as_slice(), EDGE_COLUMN_NAMES);
}

fn build_directed_named_dataframe_graph(
    handle: &ResourceHandle,
    df: &cudf::DataFrame,
) -> cugraph::error::Result<Graph<Forward>> {
    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::directed())
        .renumber();
    graph_from_dataframe::<Forward>(handle, df, &opts)
}

fn build_undirected_named_dataframe_graph(
    handle: &ResourceHandle,
    df: &cudf::DataFrame,
) -> cugraph::error::Result<Graph<Transposed>> {
    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber();
    graph_from_dataframe::<Transposed>(handle, df, &opts)
}

fn build_directed_prepared_graph(
    handle: &ResourceHandle,
    export: &cudf::interop::PreparedTableExport<'_>,
) -> cugraph::error::Result<Graph<Forward>> {
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::directed()).renumber();
    graph_from_prepared_table_export::<Forward>(handle, export, &opts)
}

fn build_undirected_prepared_graph(
    handle: &ResourceHandle,
    export: &cudf::interop::PreparedTableExport<'_>,
) -> cugraph::error::Result<Graph<Transposed>> {
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::undirected()).renumber();
    graph_from_prepared_table_export::<Transposed>(handle, export, &opts)
}

fn bfs_distance_map(handle: &ResourceHandle, graph: &mut Graph<Forward>, source: i32) -> HashMap<i32, i32> {
    let sources = DeviceArray::from_slice(handle, &[source]).expect("sources");
    let result = graph.bfs(handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(handle).expect("distances");
    vertices.into_iter().zip(distances).collect()
}

fn pagerank_score_map(handle: &ResourceHandle, graph: &mut Graph<Transposed>) -> HashMap<i32, f32> {
    let result = graph.pagerank_simple(handle, &PageRankParams::default()).expect("pagerank");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let scores: Vec<f32> = result.scores().expect("scores").to_vec_f32(handle).expect("scores");
    vertices.into_iter().zip(scores).collect()
}

fn write_parquet_table(view: &cudf::core::TableView<'_>, path: &Path, column_names: &[&str], runtime: &cudf::Runtime) {
    let write_opts = parquet::ParquetTableWriteOptions::builder()
        .filepath(path)
        .column_names(column_names)
        .with_runtime(runtime)
        .build();
    parquet::write_parquet(view, &write_opts).expect("write parquet");
}

fn read_parquet_table(path: &Path, runtime: &cudf::Runtime) -> Table {
    let read_opts = parquet::ParquetReadOptions::builder().filepath(path).with_runtime(runtime).build();
    parquet::read_parquet(&read_opts).expect("read parquet").table
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let id = NEXT_TEMP_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("cugraph_{prefix}_{}_{}", std::process::id(), id))
}

struct TempParquetFixture {
    dir: PathBuf,
    path: PathBuf,
}

impl TempParquetFixture {
    fn bfs_graph() -> Self {
        let dir = unique_temp_dir("bfs_graph");
        std::fs::create_dir_all(&dir).expect("create fixture dir");

        let path = dir.join("bfs_graph.parquet");
        let runtime = explicit_runtime();
        let table = bfs_graph_table(&runtime);
        let view = table.view().expect("table view");
        write_parquet_table(&view, &path, &["src", "dst", "weight"], &runtime);

        Self { dir, path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn path_str(&self) -> &str {
        self.path.to_str().expect("utf8 path")
    }
}

impl Drop for TempParquetFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Run BFS and convert results to a cudf DataFrame, then read
/// back via column accessors and verify the values match the known graph topology.
#[test]
fn bfs_results_to_cudf_dataframe() {
    let handle = ResourceHandle::new().expect("handle");

    // Build graph from host data: directed, same topology as the unit test in bfs.rs.
    // 0->1, 1->3, 1->4, 2->0, 2->1, 2->3, 3->5, 4->5
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();

    let params = BfsParams::default().with_predecessors();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    // ── Convert BFS results to cudf DataFrame ───────────────────────────
    let df = result.to_dataframe().expect("to_dataframe");

    // ── Read back via DataFrame column accessors ────────────────────────
    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vert to_vec");
    let distances: Vec<i32> = df.series("distance").expect("distance column").to_vec_strict().expect("dist to_vec");

    assert_eq!(vertices.len(), 6, "should have 6 vertices");
    assert_eq!(distances.len(), 6, "should have 6 distances");

    // Build vertex → distance map and verify expected BFS distances.
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();

    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
    assert_eq!(dist_map[&1], 1, "vertex 1 should have distance 1");
    assert_eq!(dist_map[&2], i32::MAX, "vertex 2 unreachable from 0");
    assert_eq!(dist_map[&3], 2, "vertex 3 should have distance 2");
    assert_eq!(dist_map[&4], 2, "vertex 4 should have distance 2");
    assert_eq!(dist_map[&5], 3, "vertex 5 should have distance 3");
}

/// Run BFS from the Parquet-loaded graph, convert results to cudf DataFrame,
/// write to Parquet, read back, and verify.
#[test]
fn bfs_results_roundtrip_through_parquet() {
    // ── Load graph from Parquet ─────────────────────────────────────────
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let props = GraphProperties::directed();
    let fixture = TempParquetFixture::bfs_graph();

    let table = read_parquet_table(fixture.path(), &runtime);
    let view = table.view().expect("table view");

    let opts = TableGraphOptions::new(0, 1).properties(props).renumber();
    let mut graph = graph_from_table_view::<Forward>(&handle, &view, &opts).expect("graph_from_table_view");

    // ── Run BFS ─────────────────────────────────────────────────────────
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();
    let params = BfsParams::default();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    // ── Convert results to cudf DataFrame ───────────────────────────────
    let df = result.to_dataframe().expect("to_dataframe");

    // Snapshot the expected values before writing to Parquet.
    let expected_vertices: Vec<i32> = df.series("vertex").expect("v").to_vec_strict().expect("v vec");
    let expected_distances: Vec<i32> = df.series("distance").expect("d").to_vec_strict().expect("d vec");

    // ── Write results to a temporary Parquet file ───────────────────────
    let result_table = df.to_table_copy().expect("to_table");
    let result_view = result_table.view().expect("result table view");

    let tmp_dir = unique_temp_dir("bfs_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("bfs_result.parquet");

    write_parquet_table(&result_view, &out_path, &["vertex", "distance"], &runtime);

    // ── Read back and verify ────────────────────────────────────────────
    let read_table = read_parquet_table(&out_path, &runtime);
    let read_view = read_table.view().expect("read table view");

    assert_eq!(read_view.num_columns().expect("ncols"), 2);

    let read_verts: Vec<i32> = read_view.column(0).expect("col 0").to_vec().expect("col 0 vec");
    let read_dists: Vec<i32> = read_view.column(1).expect("col 1").to_vec().expect("col 1 vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch after Parquet round-trip");
    assert_eq!(read_dists, expected_distances, "distance column mismatch after Parquet round-trip");

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

/// Build a graph from Parquet using `graph_from_parquet`, then run BFS and
/// verify the results match the known graph topology.
#[test]
fn graph_from_parquet_then_bfs() {
    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties::directed();
    let fixture = TempParquetFixture::bfs_graph();

    let opts = ParquetGraphOptions::new(fixture.path_str(), 0, 1).weight_col(2).properties(props).renumber();
    let loaded = graph_from_parquet::<Forward>(&handle, &opts).expect("graph_from_parquet");
    let mut graph = loaded.graph;

    // Run BFS from vertex 0
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();

    let params = BfsParams::default();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    // Verify we get results for all vertices via typed accessors
    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&handle).expect("distances");

    assert!(!vertices.is_empty(), "graph_from_parquet should produce a non-empty graph");
    assert_eq!(vertices.len(), distances.len());

    // Source vertex should have distance 0
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
}

/// Build a graph from Parquet without weights using `graph_from_parquet`.
#[test]
fn graph_from_parquet_without_weights() {
    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties::directed();
    let fixture = TempParquetFixture::bfs_graph();

    let opts = ParquetGraphOptions::new(fixture.path_str(), 0, 1).properties(props).renumber();
    let loaded = graph_from_parquet::<Forward>(&handle, &opts).expect("graph_from_parquet without weights");
    let mut graph = loaded.graph;

    // Verify the graph is valid by running BFS
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();

    let params = BfsParams::default();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    assert!(!vertices.is_empty(), "graph should have vertices");
}

/// `graph_from_parquet` should return an error for out-of-bounds column indices.
#[test]
fn graph_from_parquet_rejects_out_of_bounds_column() {
    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties::directed();
    let fixture = TempParquetFixture::bfs_graph();

    // The test Parquet has 3 columns (src, dst, wgt), so index 10 is OOB.
    let opts = ParquetGraphOptions::new(fixture.path_str(), 0, 10).properties(props).renumber();
    let result = graph_from_parquet::<Forward>(&handle, &opts);

    let err = match result {
        Ok(_) => panic!("should reject out-of-bounds column index"),
        Err(e) => e,
    };

    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "graph_from_table_view");
}

/// `graph_from_parquet` should return an error for a non-existent file.
#[test]
fn graph_from_parquet_rejects_nonexistent_file() {
    let handle = ResourceHandle::new().expect("handle");

    let props = GraphProperties::directed();

    let opts = ParquetGraphOptions::new("/tmp/this_file_definitely_does_not_exist_12345.parquet", 0, 1)
        .properties(props)
        .renumber();
    let result = graph_from_parquet::<Forward>(&handle, &opts);

    let err = match result {
        Ok(_) => panic!("should reject non-existent file"),
        Err(e) => e,
    };

    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "graph_from_parquet");
}

/// `graph_from_parquet` should reject nullable source columns.
#[test]
fn graph_from_parquet_rejects_nulls_in_src_column() {
    let runtime = explicit_runtime();
    // Build a 3-column edge table where src contains one null.
    let src = runtime.column_from_slice_nullable(&[0i32, 1, 2], &[true, false, true]).expect("src");
    let dst = runtime.column_from_slice(&[1i32, 2, 0]).expect("dst");
    let wgt = runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).expect("wgt");

    let table = Table::from_columns(vec![src, dst, wgt]).expect("table");
    let tmp_dir = unique_temp_dir("parquet_nulls");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let path = tmp_dir.join("edges_with_null_src.parquet");

    let table_view = table.view().expect("table view");
    write_parquet_table(&table_view, &path, &["0", "1", "2"], &runtime);

    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties::directed();

    let opts =
        ParquetGraphOptions::new(path.to_str().expect("utf8 path"), 0, 1).weight_col(2).properties(props).renumber();
    let result = graph_from_parquet::<Forward>(&handle, &opts);

    let err = match result {
        Ok(_) => panic!("should reject nullable src column"),
        Err(e) => e,
    };

    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "graph_from_table_view");
    assert!(err.message().contains("null"), "unexpected error message: {}", err.message());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// ── graph_from_table_view / graph_from_dataframe tests ──────────────────────

/// Load test Parquet into a cudf Table, call graph_from_table_view(),
/// run BFS from vertex 0, verify distances match expected topology.
#[test]
fn graph_from_table_view_then_bfs() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let table = read_parquet_table(fixture.path(), &runtime);
    let view = table.view().expect("table view");

    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::directed()).renumber();
    let mut graph = graph_from_table_view::<Forward>(&handle, &view, &opts).expect("graph_from_table_view");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();
    let params = BfsParams::default();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&handle).expect("distances");

    assert!(!vertices.is_empty(), "should produce a non-empty graph");

    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
}

/// Call graph_from_table_view without weights (wgt_col: None).
#[test]
fn graph_from_table_view_without_weights() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let table = read_parquet_table(fixture.path(), &runtime);
    let view = table.view().expect("table view");

    let opts = TableGraphOptions::new(0, 1).properties(GraphProperties::directed()).renumber();
    let mut graph =
        graph_from_table_view::<Forward>(&handle, &view, &opts).expect("graph_from_table_view without weights");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();
    let params = BfsParams::default();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    assert!(!vertices.is_empty(), "graph should have vertices");
}

/// Pass dst_col: 99 — should get ErrorKind::InvalidInput.
#[test]
fn graph_from_table_view_rejects_out_of_bounds_column() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let table = read_parquet_table(fixture.path(), &runtime);
    let view = table.view().expect("table view");

    let opts = TableGraphOptions::new(0, 99).properties(GraphProperties::directed()).renumber();
    let result = graph_from_table_view::<Forward>(&handle, &view, &opts);

    let err = result.expect_err("should reject out-of-bounds column index");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

/// Build a table with a nullable src column — should reject with error
/// message containing "null".
#[test]
fn graph_from_table_view_rejects_nullable_column() {
    let runtime = explicit_runtime();
    let src = runtime.column_from_slice_nullable(&[0i32, 1, 2], &[true, false, true]).expect("src");
    let dst = runtime.column_from_slice(&[1i32, 2, 0]).expect("dst");
    let wgt = runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).expect("wgt");

    let table = Table::from_columns(vec![src, dst, wgt]).expect("table");
    let view = table.view().expect("table view");

    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties::directed();

    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(props).renumber();
    let result = graph_from_table_view::<Forward>(&handle, &view, &opts);

    let err = result.expect_err("should reject nullable column");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert!(err.message().contains("null"), "error message should mention null, got: {}", err.message());
}

/// Build a DataFrame with named columns, call graph_from_dataframe()
/// with string names, run BFS, verify results.
#[test]
fn graph_from_dataframe_by_name() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();

    let src = runtime.series("src", &[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = runtime.series("dst", &[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");
    let wgt = runtime.series("weight", &[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]).expect("wgt");

    let df = cudf::DataFrame::from_series([src, dst, wgt]).expect("dataframe");

    let props = GraphProperties::directed();

    let opts = DataFrameGraphOptions::new("src", "dst").weight_col("weight").properties(props).renumber();
    let mut graph = graph_from_dataframe::<Forward>(&handle, &df, &opts).expect("graph_from_dataframe");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();
    let params = BfsParams::default();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&handle).expect("distances");

    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
    assert_eq!(dist_map[&1], 1, "vertex 1 should have distance 1");
}

/// Pass src_col: "nonexistent" — should get ErrorKind::InvalidInput.
#[test]
fn graph_from_dataframe_rejects_unknown_column_name() {
    let runtime = explicit_runtime();
    let src = runtime.series("src", &[0i32, 1]).expect("src");
    let dst = runtime.series("dst", &[1i32, 0]).expect("dst");

    let df = cudf::DataFrame::from_series([src, dst]).expect("dataframe");

    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties::directed();

    let opts = DataFrameGraphOptions::new("nonexistent", "dst").properties(props).renumber();
    let result = graph_from_dataframe::<Forward>(&handle, &df, &opts);

    let err = result.expect_err("should reject unknown column name");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

/// Validate the primary prepared-export graph build path against the explicit
/// deep-copy-import fallback path using the same producer-rooted table export.
#[test]
fn prepared_export_vs_deep_copy_import_bfs_equivalence() {
    let handle = ResourceHandle::new().expect("handle");
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = bfs_graph_dataframe(&producer_runtime);
    let export = producer_df.try_prepare_device_table_export().expect("prepared export");
    let mut prepared_graph = build_directed_prepared_graph(&handle, &export).expect("prepared graph");

    let imported_df = cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&export, &consumer_runtime)
        .expect("deep-copy import");
    assert_named_edge_columns(&imported_df);
    let mut fallback_graph = build_directed_named_dataframe_graph(&handle, &imported_df).expect("fallback graph");

    let prepared_map = bfs_distance_map(&handle, &mut prepared_graph, 0);
    let fallback_map = bfs_distance_map(&handle, &mut fallback_graph, 0);

    assert_eq!(prepared_map, fallback_map, "deep-copy import fallback should preserve BFS semantics");
}

/// Validate that the deep-copy-import fallback produces the same PageRank
/// semantics as direct prepared-export consumption.
#[test]
fn prepared_export_vs_deep_copy_import_pagerank_equivalence() {
    let handle = ResourceHandle::new().expect("handle");
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = pagerank_graph_dataframe(&producer_runtime);
    let export = producer_df.try_prepare_device_table_export().expect("prepared export");
    let mut prepared_graph = build_undirected_prepared_graph(&handle, &export).expect("prepared graph");

    let imported_df = cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&export, &consumer_runtime)
        .expect("deep-copy import");
    assert_named_edge_columns(&imported_df);
    let mut fallback_graph = build_undirected_named_dataframe_graph(&handle, &imported_df).expect("fallback graph");

    let prepared_map = pagerank_score_map(&handle, &mut prepared_graph);
    let fallback_map = pagerank_score_map(&handle, &mut fallback_graph);

    assert_eq!(prepared_map.len(), fallback_map.len(), "vertex coverage should match");
    for (vertex, prepared_score) in prepared_map {
        let fallback_score = fallback_map.get(&vertex).expect("fallback result should contain every prepared vertex");
        assert!(
            (prepared_score - fallback_score).abs() <= 1.0e-6,
            "vertex {} score mismatch: prepared={} fallback={}",
            vertex,
            prepared_score,
            fallback_score
        );
    }
}

/// Validate that prepared-host export remains a correct single-GPU fallback:
/// the host-materialized export imports into a consumer runtime and preserves
/// the same BFS semantics as direct prepared-device consumption.
#[test]
fn prepared_host_export_import_bfs_matches_prepared_device_path() {
    let handle = ResourceHandle::new().expect("handle");
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = bfs_graph_dataframe(&producer_runtime);
    let prepared_export = producer_df.try_prepare_device_table_export().expect("prepared device export");
    let mut prepared_graph = build_directed_prepared_graph(&handle, &prepared_export).expect("prepared graph");

    let host_export = producer_df.try_prepare_host_table_export().expect("prepared host export");
    let imported_df = cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&host_export, &consumer_runtime)
        .expect("host fallback import");
    assert_named_edge_columns(&imported_df);
    let mut fallback_graph = build_directed_named_dataframe_graph(&handle, &imported_df).expect("fallback graph");

    let prepared_map = bfs_distance_map(&handle, &mut prepared_graph, 0);
    let fallback_map = bfs_distance_map(&handle, &mut fallback_graph, 0);

    assert_eq!(prepared_map, fallback_map, "prepared host export fallback should preserve BFS semantics");
}

/// Validate that prepared-host export preserves one algorithm-shaped path
/// after host-to-device fallback import into the consumer runtime.
#[test]
fn prepared_host_export_import_pagerank_matches_prepared_device_path() {
    let handle = ResourceHandle::new().expect("handle");
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = pagerank_graph_dataframe(&producer_runtime);
    let prepared_export = producer_df.try_prepare_device_table_export().expect("prepared device export");
    let mut prepared_graph = build_undirected_prepared_graph(&handle, &prepared_export).expect("prepared graph");

    let host_export = producer_df.try_prepare_host_table_export().expect("prepared host export");
    let imported_df = cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&host_export, &consumer_runtime)
        .expect("host fallback import");
    assert_named_edge_columns(&imported_df);
    let mut fallback_graph = build_undirected_named_dataframe_graph(&handle, &imported_df).expect("fallback graph");

    let prepared_map = pagerank_score_map(&handle, &mut prepared_graph);
    let fallback_map = pagerank_score_map(&handle, &mut fallback_graph);

    assert_eq!(prepared_map.len(), fallback_map.len(), "vertex coverage should match");
    for (vertex, prepared_score) in prepared_map {
        let fallback_score = fallback_map.get(&vertex).expect("fallback result should contain every prepared vertex");
        assert!(
            (prepared_score - fallback_score).abs() <= 1.0e-6,
            "vertex {} score mismatch: prepared={} fallback={}",
            vertex,
            prepared_score,
            fallback_score
        );
    }
}

/// Owned-transfer round-trips should preserve high-level schema names and, for
/// uniquely owned cudf-rs producers, zero-copy device-buffer identity.
#[test]
fn owned_transfer_zero_copy_import_preserves_device_buffers_and_bfs_semantics() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();

    let src = runtime.series("src", &[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = runtime.series("dst", &[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");
    let wgt = runtime.series("weight", &[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]).expect("weight");

    let src_ptr = i32_series_device_ptr(&src);
    let dst_ptr = i32_series_device_ptr(&dst);
    let wgt_ptr = f32_series_device_ptr(&wgt);

    let df = cudf::DataFrame::from_series([src, dst, wgt]).expect("dataframe");
    let transfer = df.try_into_owned_transfer().expect("owned transfer");
    let imported_df = cudf::DataFrame::try_from_owned_device_table_transfer(transfer).expect("owned-transfer import");

    assert_named_edge_columns(&imported_df);
    assert_eq!(i32_series_device_ptr(&imported_df.series("src").expect("src series")), src_ptr);
    assert_eq!(i32_series_device_ptr(&imported_df.series("dst").expect("dst series")), dst_ptr);
    assert_eq!(f32_series_device_ptr(&imported_df.series("weight").expect("weight series")), wgt_ptr);

    let mut graph = build_directed_named_dataframe_graph(&handle, &imported_df).expect("graph_from_dataframe");
    let dist_map = bfs_distance_map(&handle, &mut graph, 0);

    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], 1);
    assert_eq!(dist_map[&3], 2);
    assert_eq!(dist_map[&4], 2);
    assert_eq!(dist_map[&5], 3);
}

/// The owned-transfer fast path should remain semantically equivalent to
/// consuming the same graph through a direct DataFrame adapter path.
#[test]
fn owned_transfer_import_pagerank_matches_direct_dataframe_path() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();

    let direct_df = pagerank_graph_dataframe(&runtime);
    let mut direct_graph = build_undirected_named_dataframe_graph(&handle, &direct_df).expect("direct graph");

    let transferred_df = pagerank_graph_dataframe(&runtime);
    let transfer = transferred_df.try_into_owned_transfer().expect("owned transfer");
    let imported_df = cudf::DataFrame::try_from_owned_device_table_transfer(transfer).expect("owned-transfer import");
    assert_named_edge_columns(&imported_df);

    let mut imported_graph = build_undirected_named_dataframe_graph(&handle, &imported_df).expect("imported graph");

    let direct_map = pagerank_score_map(&handle, &mut direct_graph);
    let imported_map = pagerank_score_map(&handle, &mut imported_graph);

    assert_eq!(direct_map.len(), imported_map.len(), "vertex coverage should match");
    for (vertex, direct_score) in direct_map {
        let imported_score = imported_map.get(&vertex).expect("imported result should contain every direct vertex");
        assert!(
            (direct_score - imported_score).abs() <= 1.0e-6,
            "vertex {} score mismatch: direct={} imported={}",
            vertex,
            direct_score,
            imported_score
        );
    }
}

/// Runtime-backed owned-transfer release is mutually exclusive with native
/// import: once cudf claims the transfer back into native ownership, the
/// releaser must reject later release attempts.
#[test]
fn owned_transfer_with_release_import_invalidates_releaser() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();

    let producer_df = bfs_graph_dataframe(&runtime);
    let (transfer, releaser) = producer_df.try_into_owned_transfer_with_release().expect("owned transfer with release");
    let imported_df = cudf::DataFrame::try_from_owned_device_table_transfer(transfer).expect("owned-transfer import");

    let mut graph = build_directed_named_dataframe_graph(&handle, &imported_df).expect("graph_from_dataframe");
    let dist_map = bfs_distance_map(&handle, &mut graph, 0);
    assert!(!dist_map.is_empty(), "imported owned-transfer graph should produce bfs results");

    let err = releaser
        .release(RuntimeCompletionAuthority::same_device_stream(runtime.stream()))
        .expect_err("releaser should be invalid after native import has already claimed the owned-transfer root");
    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    let err_text = format!("{}", err);
    assert!(
        err_text.contains("ownership state changed") || err_text.contains("no longer has importable native ownership"),
        "unexpected releaser error: {}",
        err
    );
}

/// Runtime-backed table-transfer release should accept a same-device consumer
/// stream when the transfer is released instead of imported.
#[test]
fn owned_transfer_with_release_accepts_same_device_stream_completion() {
    let runtime = explicit_runtime();
    let producer_df = bfs_graph_dataframe(&runtime);
    let (transfer, releaser) = producer_df.try_into_owned_transfer_with_release().expect("owned transfer with release");

    assert_eq!(
        transfer.release_contract().completion(),
        cudf::interop::TransferCompletionRule::ReleaseWaitsOnConsumerSignal
    );
    assert_eq!(transfer.release_contract().synchronization(), cudf::interop::ReleaseSynchronization::Asynchronous);

    drop(transfer);
    releaser
        .release(RuntimeCompletionAuthority::same_device_stream(runtime.stream()))
        .expect("runtime-backed owned-transfer release should accept same-device stream completion");
}

// ── to_dataframe() tests ────────────────────────────────────────────────────

/// Run PageRank on a triangle graph, call .to_dataframe(), verify
/// 2 columns named "vertex" and "value", row count == 3.
#[test]
fn centrality_result_to_dataframe() {
    use cugraph::centrality::PageRankParams;

    let handle = ResourceHandle::new().expect("handle");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0, 0, 1, 2]).expect("dst");

    let src_view = src.view();
    let dst_view = dst.view();
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");
    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "value");

    let num_rows = df.num_rows();
    assert_eq!(num_rows, 3, "should have 3 vertices");

    // Verify all vertex ids are present
    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let mut sorted = vertices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);

    // Verify values are finite positive floats
    let values: Vec<f32> = df.series("value").expect("value column").to_vec_strict().expect("value to_vec");
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite() && v > 0.0, "value[{}] = {} should be finite positive", i, v);
    }
}

/// Run ForceAtlas2 on a triangle, call .to_dataframe(), verify
/// 3 columns named "vertex", "x", "y", row count == 3.
#[test]
fn layout_result_to_dataframe() {
    use cugraph::{
        layout::{ForceAtlas2Options, ForceAtlas2Params},
        params::IterationCount,
        random::RngState,
    };

    let handle = ResourceHandle::new().expect("handle");
    let mut rng = RngState::new(&handle, 42).expect("rng");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0, 0, 1, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view())
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
        params: ForceAtlas2Params { max_iter: IterationCount::new(50).unwrap(), ..Default::default() },
        ..Default::default()
    };

    let result = graph.force_atlas2(&handle, &mut rng, &options).expect("fa2");
    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 3);
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "x");
    assert_eq!(names[2], "y");

    let num_rows = df.num_rows();
    assert_eq!(num_rows, 3);

    // Verify coordinates are finite
    let x_vals: Vec<f32> = df.series("x").expect("x column").to_vec_strict().expect("x to_vec");
    let y_vals: Vec<f32> = df.series("y").expect("y column").to_vec_strict().expect("y to_vec");
    for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
        assert!(x.is_finite(), "x[{}] = {} should be finite", i, x);
        assert!(y.is_finite(), "y[{}] = {} should be finite", i, y);
    }
}

/// Run BFS with compute_predecessors: true, call .to_dataframe(), verify
/// 3 columns: "vertex", "distance", "predecessor".
#[test]
fn bfs_result_to_dataframe_with_predecessors() {
    let handle = ResourceHandle::new().expect("handle");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();
    let params = BfsParams::default().with_predecessors();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 3, "should have 3 columns with predecessors");
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "distance");
    assert_eq!(names[2], "predecessor");

    // Verify source vertex has distance 0
    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let distances: Vec<i32> = df.series("distance").expect("distance column").to_vec_strict().expect("distance to_vec");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex should have distance 0");
}

/// Run BFS with compute_predecessors: false, call .to_dataframe(), verify
/// only 2 columns: "vertex", "distance".
#[test]
fn bfs_result_to_dataframe_without_predecessors() {
    let handle = ResourceHandle::new().expect("handle");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view();
    let params = BfsParams::default();
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");
    assert!(result.predecessors().expect("predecessors").is_none(), "predecessors should be omitted");

    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 2, "should have 2 columns without predecessors");
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "distance");
}

/// Run PageRank, .to_dataframe(), write to Parquet, read back, verify
/// column names and values match.
#[test]
fn centrality_to_dataframe_write_parquet_roundtrip() {
    use cugraph::centrality::PageRankParams;

    let handle = ResourceHandle::new().expect("handle");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0, 0, 1, 2]).expect("dst");

    let src_view = src.view();
    let dst_view = dst.view();
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");
    let df = result.to_dataframe().expect("to_dataframe");

    // Snapshot expected values
    let expected_vertices: Vec<i32> = df.series("vertex").expect("v").to_vec_strict().expect("v vec");
    let expected_values: Vec<f32> = df.series("value").expect("val").to_vec_strict().expect("val vec");

    // Write to Parquet
    let tmp_dir = unique_temp_dir("centrality_df_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("centrality.parquet");

    let table = df.to_table_copy().expect("to_table");
    let table_view = table.view().expect("table view");
    write_parquet_table(&table_view, &out_path, &["vertex", "value"], &explicit_runtime());

    // Read back and verify
    let read_table = read_parquet_table(&out_path, &explicit_runtime());
    let read_view = read_table.view().expect("read table view");

    assert_eq!(read_view.num_columns().expect("ncols"), 2);

    let read_verts: Vec<i32> = read_view.column(0).expect("col 0").to_vec().expect("col 0 vec");
    let read_vals: Vec<f32> = read_view.column(1).expect("col 1").to_vec().expect("col 1 vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch");
    assert_eq!(read_vals, expected_values, "value column mismatch");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
