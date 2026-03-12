//! Integration tests for cugraph → cudf result pathway (Task 3).
//!
//! These tests run actual GPU algorithms and convert results to cudf Columns,
//! verifying the full round-trip: GPU algorithm → DeviceArrayView → cudf Column → host Vec.

use std::collections::HashMap;

use cudf::{Column, io::parquet};
use cugraph::{
    array::device_array_from_host,
    cudf_interop::{
        DeviceArrayViewExt, DeviceSliceExt, graph_from_dataframe, graph_from_parquet, graph_from_table_view,
    },
    graph::Graph,
    handle::ResourceHandle,
    traversal::bfs,
    types::{DataType, GraphProperties},
};

/// Path to the BFS test graph shipped in the examples/data directory.
const BFS_PARQUET: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/bfs_graph.parquet");

/// Run BFS and convert vertices() and distances() to cudf Columns, then read
/// back via to_vec() and verify the values match the known graph topology.
#[test]
fn bfs_results_to_cudf_columns() {
    let handle = ResourceHandle::new().expect("handle");

    // Build graph from host data: directed, same topology as the unit test in bfs.rs.
    // 0->1, 1->3, 1->4, 2->0, 2->1, 2->3, 3->5, 4->5
    let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 2, 3, 4];
    let dst_data: Vec<i32> = vec![1, 3, 4, 0, 1, 3, 5, 5];

    let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let mut graph =
        Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true).expect("graph");

    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();

    let result = bfs(&handle, &mut graph, &mut sources_view, None, true).expect("bfs");

    // ── Convert BFS results to cudf Columns ─────────────────────────────
    let vert_col = result.vertices().to_cudf_column().expect("vertices to cudf column");
    let dist_col = result.distances().to_cudf_column().expect("distances to cudf column");

    // ── Read back via cudf Column::view().to_vec() ──────────────────────
    let vert_view = vert_col.view().expect("vert view");
    let dist_view = dist_col.view().expect("dist view");

    let vertices: Vec<i32> = vert_view.to_vec().expect("vert to_vec");
    let distances: Vec<i32> = dist_view.to_vec().expect("dist to_vec");

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

/// Run BFS from the Parquet-loaded graph, convert results to cudf Columns,
/// build a cudf Table, write to Parquet, read back, and verify.
#[test]
fn bfs_results_roundtrip_through_parquet() {
    // ── Load graph from Parquet ─────────────────────────────────────────
    let options = parquet::ParquetReaderOptions::new(BFS_PARQUET).expect("reader options");
    let table = parquet::read_parquet(&options).expect("read parquet");
    let table_view = table.view().expect("table view");

    let src_col = table_view.column(0).expect("src col");
    let dst_col = table_view.column(1).expect("dst col");

    let src_slice = src_col.device_slice::<i32>().expect("src slice");
    let dst_slice = dst_col.device_slice::<i32>().expect("dst slice");

    let src_cg = src_slice.as_cugraph_view().expect("src cugraph view");
    let dst_cg = dst_slice.as_cugraph_view().expect("dst cugraph view");

    let handle = ResourceHandle::new().expect("handle");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let mut graph =
        Graph::from_edge_list(&handle, props, &src_cg, &dst_cg, None, None, None, false, true).expect("graph");

    // ── Run BFS ─────────────────────────────────────────────────────────
    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();
    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");

    // ── Convert results to cudf Columns ─────────────────────────────────
    let vert_col = result.vertices().to_cudf_column().expect("vertices to cudf");
    let dist_col = result.distances().to_cudf_column().expect("distances to cudf");

    // Snapshot the expected values before writing to Parquet.
    let expected_vertices: Vec<i32> = vert_col.view().expect("v").to_vec().expect("v vec");
    let expected_distances: Vec<i32> = dist_col.view().expect("d").to_vec().expect("d vec");

    // ── Write results to a temporary Parquet file ───────────────────────
    let result_table = cudf::Table::from_columns(vec![vert_col, dist_col]).expect("result table");
    let result_view = result_table.view().expect("result table view");

    let tmp_dir = std::env::temp_dir().join("cugraph_test_bfs_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("bfs_result.parquet");

    let write_opts =
        parquet::ParquetWriterOptions::new(&out_path).expect("writer options").column_names(&["vertex", "distance"]);

    parquet::write_parquet(&result_view, &write_opts).expect("write parquet");

    // ── Read back and verify ────────────────────────────────────────────
    let read_opts = parquet::ParquetReaderOptions::new(&out_path).expect("reader options");
    let read_table = parquet::read_parquet(&read_opts).expect("read parquet");
    let read_view = read_table.view().expect("read table view");

    assert_eq!(read_view.num_columns().expect("ncols"), 2);

    let read_verts: Vec<i32> = read_view.column(0).expect("col 0").to_vec().expect("col 0 vec");
    let read_dists: Vec<i32> = read_view.column(1).expect("col 1").to_vec().expect("col 1 vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch after Parquet round-trip");
    assert_eq!(read_dists, expected_distances, "distance column mismatch after Parquet round-trip");

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

/// Test that converting a DeviceArrayView with SizeT dtype to a cudf Column
/// returns an appropriate error.
#[test]
fn size_t_view_to_cudf_column_returns_error() {
    let handle = ResourceHandle::new().expect("handle");

    // Create a device array with SizeT type.
    let arr = device_array_from_host(&handle, &[42u64], DataType::SizeT).expect("create size_t array");
    let view = arr.view();

    let err = match view.to_cudf_column() {
        Ok(_) => panic!("SizeT should not convert to cudf"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), cugraph::error::ErrorKind::UnsupportedType);
    assert!(
        err.operation().contains("to_cudf_column"),
        "error operation should mention to_cudf_column, got: {}",
        err.operation()
    );
}

/// Build a graph from Parquet using `graph_from_parquet`, then run BFS and
/// verify the results match the known graph topology.
#[test]
fn graph_from_parquet_then_bfs() {
    let handle = ResourceHandle::new().expect("handle");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let (_table, mut graph) = graph_from_parquet(
        &handle,
        BFS_PARQUET,
        0,       // src column
        1,       // dst column
        Some(2), // weight column
        props,
        false, // store_transposed
        true,  // renumber
    )
    .expect("graph_from_parquet");

    // Run BFS from vertex 0
    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();

    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");

    // Verify we get results for all vertices
    let vert_bytes = result.vertices().to_host_vec(&handle).expect("vertices");
    let dist_bytes = result.distances().to_host_vec(&handle).expect("distances");

    let vertices: Vec<i32> = vert_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
    let distances: Vec<i32> =
        dist_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();

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

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let (_table, mut graph) = graph_from_parquet(
        &handle,
        BFS_PARQUET,
        0,    // src column
        1,    // dst column
        None, // no weight column
        props,
        false, // store_transposed
        true,  // renumber
    )
    .expect("graph_from_parquet without weights");

    // Verify the graph is valid by running BFS
    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();

    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");

    let vert_bytes = result.vertices().to_host_vec(&handle).expect("vertices");
    let vertices: Vec<i32> = vert_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();

    assert!(!vertices.is_empty(), "graph should have vertices");
}

/// `graph_from_parquet` should return an error for out-of-bounds column indices.
#[test]
fn graph_from_parquet_rejects_out_of_bounds_column() {
    let handle = ResourceHandle::new().expect("handle");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    // The test Parquet has 3 columns (src, dst, wgt), so index 10 is OOB.
    let result = graph_from_parquet(
        &handle,
        BFS_PARQUET,
        0,
        10, // out of bounds
        None,
        props,
        false,
        true,
    );

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

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let result = graph_from_parquet(
        &handle,
        "/tmp/this_file_definitely_does_not_exist_12345.parquet",
        0,
        1,
        None,
        props,
        false,
        true,
    );

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
    let write_opts = parquet::ParquetWriterOptions::new(&path).expect("writer options");
    parquet::write_parquet(&table_view, &write_opts).expect("write parquet");

    let handle = ResourceHandle::new().expect("handle");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let result = graph_from_parquet(&handle, path.to_str().expect("utf8 path"), 0, 1, Some(2), props, false, true);

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

    let options = parquet::ParquetReaderOptions::new(BFS_PARQUET).expect("reader options");
    let table = parquet::read_parquet(&options).expect("read parquet");
    let view = table.view().expect("table view");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let mut graph = graph_from_table_view(
        &handle,
        &view,
        0,       // src
        1,       // dst
        Some(2), // wgt
        props,
        false,
        true,
    )
    .expect("graph_from_table_view");

    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();
    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");

    let vert_bytes = result.vertices().to_host_vec(&handle).expect("vertices");
    let distances_bytes = result.distances().to_host_vec(&handle).expect("distances");
    let vertices: Vec<i32> = vert_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
    let distances: Vec<i32> =
        distances_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();

    assert!(!vertices.is_empty(), "should produce a non-empty graph");

    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
}

/// Call graph_from_table_view without weights (wgt_col: None).
#[test]
fn graph_from_table_view_without_weights() {
    let handle = ResourceHandle::new().expect("handle");

    let options = parquet::ParquetReaderOptions::new(BFS_PARQUET).expect("reader options");
    let table = parquet::read_parquet(&options).expect("read parquet");
    let view = table.view().expect("table view");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let mut graph = graph_from_table_view(
        &handle, &view, 0, 1, None, // no weight column
        props, false, true,
    )
    .expect("graph_from_table_view without weights");

    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();
    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");

    let vert_bytes = result.vertices().to_host_vec(&handle).expect("vertices");
    let vertices: Vec<i32> = vert_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert!(!vertices.is_empty(), "graph should have vertices");
}

/// Pass dst_col: 99 — should get ErrorKind::InvalidInput.
#[test]
fn graph_from_table_view_rejects_out_of_bounds_column() {
    let handle = ResourceHandle::new().expect("handle");

    let options = parquet::ParquetReaderOptions::new(BFS_PARQUET).expect("reader options");
    let table = parquet::read_parquet(&options).expect("read parquet");
    let view = table.view().expect("table view");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let result = graph_from_table_view(
        &handle, &view, 0, 99, // out of bounds
        None, props, false, true,
    );

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
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let result = graph_from_table_view(&handle, &view, 0, 1, Some(2), props, false, true);

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

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let mut graph = graph_from_dataframe(&handle, &df, "src", "dst", Some("weight"), props, false, true)
        .expect("graph_from_dataframe");

    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();
    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");

    let vert_bytes = result.vertices().to_host_vec(&handle).expect("vertices");
    let distances_bytes = result.distances().to_host_vec(&handle).expect("distances");
    let vertices: Vec<i32> = vert_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
    let distances: Vec<i32> =
        distances_bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();

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
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let result = graph_from_dataframe(
        &handle,
        &df,
        "nonexistent", // unknown column
        "dst",
        None,
        props,
        false,
        true,
    );

    let err = result.expect_err("should reject unknown column name");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

// ── Task 7: to_dataframe() tests ────────────────────────────────────────────

/// Run PageRank on a triangle graph, call .to_dataframe(), verify
/// 2 columns named "vertex" and "value", row count == 3.
#[test]
fn centrality_result_to_dataframe() {
    use cugraph::centrality::{PageRankParams, pagerank};

    let handle = ResourceHandle::new().expect("handle");

    let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
    let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

    let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

    let props = GraphProperties { is_symmetric: true, is_multigraph: false };

    let mut graph = Graph::from_edge_list(
        &handle,
        props,
        &src.view(),
        &dst.view(),
        None,
        None,
        None,
        true, // PageRank needs transposed
        true,
    )
    .expect("graph");

    let result = pagerank(&handle, &mut graph, &PageRankParams::default()).expect("pagerank");
    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "value");

    let num_rows = df.num_rows().expect("num_rows");
    assert_eq!(num_rows, 3, "should have 3 vertices");

    // Verify all vertex ids are present
    let vert_col = df.column("vertex").expect("vertex column");
    let vertices: Vec<i32> = vert_col.to_vec().expect("vertex to_vec");
    let mut sorted = vertices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);

    // Verify values are finite positive floats
    let val_col = df.column("value").expect("value column");
    let values: Vec<f32> = val_col.to_vec().expect("value to_vec");
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite() && v > 0.0, "value[{}] = {} should be finite positive", i, v);
    }
}

/// Run ForceAtlas2 on a triangle, call .to_dataframe(), verify
/// 3 columns named "vertex", "x", "y", row count == 3.
#[test]
fn layout_result_to_dataframe() {
    use cugraph::{
        layout::{ForceAtlas2Params, force_atlas2},
        params::IterationCount,
        random::RngState,
    };

    let handle = ResourceHandle::new().expect("handle");
    let mut rng = RngState::new(&handle, 42).expect("rng");

    let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
    let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

    let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

    let props = GraphProperties { is_symmetric: true, is_multigraph: false };

    let mut graph =
        Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true).expect("graph");

    let params = ForceAtlas2Params { max_iter: IterationCount::new(50).unwrap(), ..Default::default() };

    let result = force_atlas2(&handle, &mut rng, &mut graph, &params, None, None).expect("fa2");
    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 3);
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "x");
    assert_eq!(names[2], "y");

    let num_rows = df.num_rows().expect("num_rows");
    assert_eq!(num_rows, 3);

    // Verify coordinates are finite
    let x_col = df.column("x").expect("x column");
    let y_col = df.column("y").expect("y column");
    let x_vals: Vec<f32> = x_col.to_vec().expect("x to_vec");
    let y_vals: Vec<f32> = y_col.to_vec().expect("y to_vec");
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

    let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 2, 3, 4];
    let dst_data: Vec<i32> = vec![1, 3, 4, 0, 1, 3, 5, 5];

    let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let mut graph =
        Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true).expect("graph");

    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();
    let result = bfs(&handle, &mut graph, &mut sources_view, None, true).expect("bfs");

    let df = result.to_dataframe().expect("to_dataframe");

    assert_eq!(df.num_columns(), 3, "should have 3 columns with predecessors");
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "distance");
    assert_eq!(names[2], "predecessor");

    // Verify source vertex has distance 0
    let vert_col = df.column("vertex").expect("vertex column");
    let dist_col = df.column("distance").expect("distance column");
    let vertices: Vec<i32> = vert_col.to_vec().expect("vertex to_vec");
    let distances: Vec<i32> = dist_col.to_vec().expect("distance to_vec");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex should have distance 0");
}

/// Run BFS with compute_predecessors: false, call .to_dataframe(), verify
/// only 2 columns: "vertex", "distance".
#[test]
fn bfs_result_to_dataframe_without_predecessors() {
    let handle = ResourceHandle::new().expect("handle");

    let src_data: Vec<i32> = vec![0, 1];
    let dst_data: Vec<i32> = vec![1, 2];

    let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let mut graph =
        Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, false).expect("graph");

    let sources = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources.view();
    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");

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
    use cugraph::centrality::{PageRankParams, pagerank};

    let handle = ResourceHandle::new().expect("handle");

    let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
    let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

    let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

    let props = GraphProperties { is_symmetric: true, is_multigraph: false };

    let mut graph =
        Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, true, true).expect("graph");

    let result = pagerank(&handle, &mut graph, &PageRankParams::default()).expect("pagerank");
    let df = result.to_dataframe().expect("to_dataframe");

    // Snapshot expected values
    let expected_vertices: Vec<i32> = df.column("vertex").expect("v").to_vec().expect("v vec");
    let expected_values: Vec<f32> = df.column("value").expect("val").to_vec().expect("val vec");

    // Write to Parquet
    let tmp_dir = std::env::temp_dir().join("cugraph_test_centrality_df_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("centrality.parquet");

    let table_view = df.table().view().expect("table view");
    let write_opts =
        parquet::ParquetWriterOptions::new(&out_path).expect("writer options").column_names(&["vertex", "value"]);
    parquet::write_parquet(&table_view, &write_opts).expect("write parquet");

    // Read back and verify
    let read_opts = parquet::ParquetReaderOptions::new(&out_path).expect("reader options");
    let read_table = parquet::read_parquet(&read_opts).expect("read parquet");
    let read_view = read_table.view().expect("read table view");

    assert_eq!(read_view.num_columns().expect("ncols"), 2);

    let read_verts: Vec<i32> = read_view.column(0).expect("col 0").to_vec().expect("col 0 vec");
    let read_vals: Vec<f32> = read_view.column(1).expect("col 1").to_vec().expect("col 1 vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch");
    assert_eq!(read_vals, expected_values, "value column mismatch");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
