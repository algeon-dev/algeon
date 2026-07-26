//! Integration tests for cugraph → cudf result pathway (Task 3).
//!
//! These tests run actual GPU algorithms and convert results to cudf Columns,
//! verifying the full round-trip: GPU algorithm → DeviceArrayView → cudf Column → host Vec.

use std::collections::HashMap;

use cudf::io::parquet;

use cugraph::array::device_array_from_host;
use cugraph::cudf_interop::{DeviceArrayViewExt, DeviceSliceExt};
use cugraph::graph::Graph;
use cugraph::handle::ResourceHandle;
use cugraph::traversal::bfs;
use cugraph::types::{DataType, GraphProperties};

/// Path to the BFS test graph shipped in the examples/data directory.
const BFS_PARQUET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/data/bfs_graph.parquet"
);

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

    let props = GraphProperties {
        is_symmetric: false,
        is_multigraph: false,
    };

    let mut graph = Graph::from_edge_list(
        &handle,
        props,
        &src.view(),
        &dst.view(),
        None,
        None,
        None,
        false,
        true,
    )
    .expect("graph");

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

    let props = GraphProperties {
        is_symmetric: false,
        is_multigraph: false,
    };

    let mut graph = Graph::from_edge_list(
        &handle,
        props,
        &src_cg,
        &dst_cg,
        None,
        None,
        None,
        false,
        true,
    )
    .expect("graph");

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
    let result_table =
        cudf::Table::from_columns(vec![vert_col, dist_col]).expect("result table");
    let result_view = result_table.view().expect("result table view");

    let tmp_dir = std::env::temp_dir().join("cugraph_test_bfs_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("bfs_result.parquet");

    let write_opts = parquet::ParquetWriterOptions::new(&out_path)
        .expect("writer options")
        .column_names(&["vertex", "distance"]);

    parquet::write_parquet(&result_view, &write_opts).expect("write parquet");

    // ── Read back and verify ────────────────────────────────────────────
    let read_opts = parquet::ParquetReaderOptions::new(&out_path).expect("reader options");
    let read_table = parquet::read_parquet(&read_opts).expect("read parquet");
    let read_view = read_table.view().expect("read table view");

    assert_eq!(read_view.num_columns().expect("ncols"), 2);

    let read_verts: Vec<i32> = read_view
        .column(0)
        .expect("col 0")
        .to_vec()
        .expect("col 0 vec");
    let read_dists: Vec<i32> = read_view
        .column(1)
        .expect("col 1")
        .to_vec()
        .expect("col 1 vec");

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
    let arr =
        device_array_from_host(&handle, &[42u64], DataType::SizeT).expect("create size_t array");
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
