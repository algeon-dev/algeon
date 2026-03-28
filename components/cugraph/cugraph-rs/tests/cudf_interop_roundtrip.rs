//! Integration tests for cugraph → cudf result pathway (Task 3).
//!
//! These tests run actual GPU algorithms and convert results to cudf DataFrames,
//! verifying the full round-trip: GPU algorithm → result → cudf DataFrame → host Vec.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use cudf::{Column, io::parquet};
use cugraph::{
    array::DeviceArray,
    cudf_interop::{
        DataFrameGraphOptions, ParquetGraphOptions, TableGraphOptions, graph_from_dataframe, graph_from_parquet,
        graph_from_table_view,
    },
    graph::{Forward, GraphBuilder, Transposed},
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};

static NEXT_TEMP_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

fn bfs_graph_table() -> cudf::Table {
    let src = Column::from_slice(&[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = Column::from_slice(&[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");
    let wgt = Column::from_slice(&[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]).expect("wgt");
    cudf::Table::from_columns(vec![src, dst, wgt]).expect("bfs graph table")
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
        let table = bfs_graph_table();
        let view = table.view().expect("table view");
        let write_opts =
            parquet::ParquetWriterOptions::new(&path).column_names(&["src", "dst", "weight"]).expect("writer opts");

        parquet::write_parquet(&view, &write_opts).expect("write bfs fixture");

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
    let vertices: Vec<i32> = df.series_view("vertex").expect("vertex column").to_vec().expect("vert to_vec");
    let distances: Vec<i32> = df.series_view("distance").expect("distance column").to_vec().expect("dist to_vec");

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
    let props = GraphProperties::directed();
    let fixture = TempParquetFixture::bfs_graph();

    let reader_opts = parquet::ParquetReaderOptions::new(fixture.path());
    let read_result = parquet::read_parquet_with_metadata(reader_opts).expect("read parquet");
    let table = read_result.table;
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
    let expected_vertices: Vec<i32> = df.series_view("vertex").expect("v").to_vec().expect("v vec");
    let expected_distances: Vec<i32> = df.series_view("distance").expect("d").to_vec().expect("d vec");

    // ── Write results to a temporary Parquet file ───────────────────────
    let result_table = df.to_table_copy().expect("to_table");
    let result_view = result_table.view().expect("result table view");

    let tmp_dir = std::env::temp_dir().join("cugraph_test_bfs_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("bfs_result.parquet");

    let write_opts =
        parquet::ParquetWriterOptions::new(&out_path).column_names(&["vertex", "distance"]).expect("writer options");

    parquet::write_parquet(&result_view, &write_opts).expect("write parquet");

    // ── Read back and verify ────────────────────────────────────────────
    let read_opts = parquet::ParquetReaderOptions::new(&out_path);
    let read_result = parquet::read_parquet_with_metadata(read_opts).expect("read parquet");
    let read_view = read_result.table.view().expect("read table view");

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
    // Build a 3-column edge table where src contains one null.
    let src = Column::from_slice_nullable(&[0i32, 1, 2], &[true, false, true]).expect("src");
    let dst = Column::from_slice(&[1i32, 2, 0]).expect("dst");
    let wgt = Column::from_slice(&[1.0f32, 2.0, 3.0]).expect("wgt");

    let table = cudf::Table::from_columns(vec![src, dst, wgt]).expect("table");
    let tmp_dir = std::env::temp_dir().join("cugraph_test_graph_from_parquet_nulls");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let path = tmp_dir.join("edges_with_null_src.parquet");

    let table_view = table.view().expect("table view");
    let write_opts = parquet::ParquetWriterOptions::new(&path);
    parquet::write_parquet(&table_view, &write_opts).expect("write parquet");

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

// ── Task 6: graph_from_table_view / graph_from_dataframe tests ──────────────

/// Load test Parquet into a cudf Table, call graph_from_table_view(),
/// run BFS from vertex 0, verify distances match expected topology.
#[test]
fn graph_from_table_view_then_bfs() {
    let handle = ResourceHandle::new().expect("handle");
    let fixture = TempParquetFixture::bfs_graph();

    let reader_opts = parquet::ParquetReaderOptions::new(fixture.path());
    let read_result = parquet::read_parquet_with_metadata(reader_opts).expect("read parquet");
    let table = read_result.table;
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
    let fixture = TempParquetFixture::bfs_graph();

    let reader_opts = parquet::ParquetReaderOptions::new(fixture.path());
    let read_result = parquet::read_parquet_with_metadata(reader_opts).expect("read parquet");
    let table = read_result.table;
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
    let fixture = TempParquetFixture::bfs_graph();

    let reader_opts = parquet::ParquetReaderOptions::new(fixture.path());
    let read_result = parquet::read_parquet_with_metadata(reader_opts).expect("read parquet");
    let table = read_result.table;
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
    let src = Column::from_slice_nullable(&[0i32, 1, 2], &[true, false, true]).expect("src");
    let dst = Column::from_slice(&[1i32, 2, 0]).expect("dst");
    let wgt = Column::from_slice(&[1.0f32, 2.0, 3.0]).expect("wgt");

    let table = cudf::Table::from_columns(vec![src, dst, wgt]).expect("table");
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

    let src = Column::from_slice(&[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = Column::from_slice(&[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");
    let wgt = Column::from_slice(&[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]).expect("wgt");

    let df = cudf::DataFrame::new(vec!["src".into(), "dst".into(), "weight".into()], vec![src, dst, wgt])
        .expect("dataframe");

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
    let src = Column::from_slice(&[0i32, 1]).expect("src");
    let dst = Column::from_slice(&[1i32, 0]).expect("dst");

    let df = cudf::DataFrame::new(vec!["src".into(), "dst".into()], vec![src, dst]).expect("dataframe");

    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties::directed();

    let opts = DataFrameGraphOptions::new("nonexistent", "dst").properties(props).renumber();
    let result = graph_from_dataframe::<Forward>(&handle, &df, &opts);

    let err = result.expect_err("should reject unknown column name");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

// ── Task 7: to_dataframe() tests ────────────────────────────────────────────

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

    let result = graph.pagerank(&handle, &PageRankParams::default()).expect("pagerank");
    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "value");

    let num_rows = df.num_rows();
    assert_eq!(num_rows, 3, "should have 3 vertices");

    // Verify all vertex ids are present
    let vertices: Vec<i32> = df.series_view("vertex").expect("vertex column").to_vec().expect("vertex to_vec");
    let mut sorted = vertices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);

    // Verify values are finite positive floats
    let values: Vec<f32> = df.series_view("value").expect("value column").to_vec().expect("value to_vec");
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

    let options = ForceAtlas2Options {
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
    let x_vals: Vec<f32> = df.series_view("x").expect("x column").to_vec().expect("x to_vec");
    let y_vals: Vec<f32> = df.series_view("y").expect("y column").to_vec().expect("y to_vec");
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
    let vertices: Vec<i32> = df.series_view("vertex").expect("vertex column").to_vec().expect("vertex to_vec");
    let distances: Vec<i32> = df.series_view("distance").expect("distance column").to_vec().expect("distance to_vec");
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

    let result = graph.pagerank(&handle, &PageRankParams::default()).expect("pagerank");
    let df = result.to_dataframe().expect("to_dataframe");

    // Snapshot expected values
    let expected_vertices: Vec<i32> = df.series_view("vertex").expect("v").to_vec().expect("v vec");
    let expected_values: Vec<f32> = df.series_view("value").expect("val").to_vec().expect("val vec");

    // Write to Parquet
    let tmp_dir = std::env::temp_dir().join("cugraph_test_centrality_df_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("centrality.parquet");

    let table = df.to_table_copy().expect("to_table");
    let table_view = table.view().expect("table view");
    let write_opts =
        parquet::ParquetWriterOptions::new(&out_path).column_names(&["vertex", "value"]).expect("writer options");
    parquet::write_parquet(&table_view, &write_opts).expect("write parquet");

    // Read back and verify
    let read_opts = parquet::ParquetReaderOptions::new(&out_path);
    let read_result = parquet::read_parquet_with_metadata(read_opts).expect("read parquet");
    let read_view = read_result.table.view().expect("read table view");

    assert_eq!(read_view.num_columns().expect("ncols"), 2);

    let read_verts: Vec<i32> = read_view.column(0).expect("col 0").to_vec().expect("col 0 vec");
    let read_vals: Vec<f32> = read_view.column(1).expect("col 1").to_vec().expect("col 1 vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch");
    assert_eq!(read_vals, expected_values, "value column mismatch");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
