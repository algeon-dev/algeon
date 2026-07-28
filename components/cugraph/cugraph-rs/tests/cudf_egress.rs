//! Integration tests for graph/result exports back into cuDF.
mod common;

use std::collections::HashMap;

use common::{
    cudf::{explicit_runtime, read_parquet_table, unique_temp_dir, write_parquet_table},
    support,
    support::{
        BFS_GRAPH_DST_I32, BFS_GRAPH_SRC_I32, BFS_SOURCE_VERTEX_I32, DIRECTED_STAR_DST_I32, DIRECTED_STAR_SRC_I32,
        TEST_LAYOUT_PRIMARY_SEED, TEST_LAYOUT_SETTLE_ITERATIONS,
    },
};
use cugraph::{
    array::DeviceArray,
    centrality::{BetweennessCentralityParams, PageRankParams},
    graph::{Forward, GraphBuilder, Transposed},
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};

#[test]
fn layout_result_to_dataframe() {
    use cugraph::{layout::ForceAtlas2Params, params::IterationCount, random::RngState};

    let runtime = explicit_runtime();
    let handle = ResourceHandle::for_runtime_device(&runtime).expect("handle");
    let mut rng = RngState::new(&handle, TEST_LAYOUT_PRIMARY_SEED).expect("rng");

    let src = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SETTLE_ITERATIONS).unwrap(),
        ..Default::default()
    };

    let df = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2").to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "x");
    assert_eq!(df.column_names()[2], "y");
    assert_eq!(df.num_rows(), 3);

    let x_vals: Vec<f32> = df.series("x").expect("x column").to_vec_strict().expect("x to_vec");
    let y_vals: Vec<f32> = df.series("y").expect("y column").to_vec_strict().expect("y to_vec");
    for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
        assert!(x.is_finite(), "x[{i}] = {x} should be finite");
        assert!(y.is_finite(), "y[{i}] = {y} should be finite");
    }
}

#[test]
fn bfs_result_to_dataframe_with_predecessors() {
    let runtime = explicit_runtime();
    let handle = ResourceHandle::for_runtime_device(&runtime).expect("handle");

    let src = DeviceArray::from_slice(&handle, BFS_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, BFS_GRAPH_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let params = BfsParams::default().with_predecessors();
    let sources_view = sources.view().expect("sources_view");
    let df = graph.bfs(&handle, &sources_view, &params).expect("bfs").to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3, "should have 3 columns with predecessors");
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "distance");
    assert_eq!(df.column_names()[2], "predecessor");

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let distances: Vec<i32> = df.series("distance").expect("distance column").to_vec_strict().expect("distance to_vec");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex should have distance 0");
}

#[test]
fn centrality_to_dataframe_write_parquet_roundtrip() {
    let runtime = explicit_runtime();
    let handle = ResourceHandle::for_runtime_device(&runtime).expect("handle");

    let src = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let df = graph
        .pagerank(&handle, &PageRankParams::default())
        .expect("pagerank")
        .to_dataframe(&runtime)
        .expect("to_dataframe");
    let expected_vertices: Vec<i32> = df.series("vertex").expect("v").to_vec_strict().expect("v vec");
    let expected_values: Vec<f32> = df.series("value").expect("val").to_vec_strict().expect("val vec");

    let tmp_dir = unique_temp_dir("centrality_df_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("centrality.parquet");

    let table = df.to_table_copy().expect("to_table");
    let table_view = table.view().expect("table view");
    write_parquet_table(&table_view, &out_path, &["vertex", "value"], &explicit_runtime());

    let read_table = read_parquet_table(&out_path, &explicit_runtime());
    let read_view = read_table.view().expect("read table view");

    assert_eq!(read_view.num_columns().expect("ncols"), 2);
    let read_verts: Vec<i32> = read_view.column(0).expect("col 0").to_vec().expect("col 0 vec");
    let read_vals: Vec<f32> = read_view.column(1).expect("col 1").to_vec().expect("col 1 vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch after Parquet round-trip");
    assert_eq!(read_vals, expected_values, "value column mismatch after Parquet round-trip");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn betweenness_result_to_dataframe() {
    let runtime = explicit_runtime();
    let handle = ResourceHandle::for_runtime_device(&runtime).expect("handle");

    let src = DeviceArray::from_slice(&handle, DIRECTED_STAR_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_STAR_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let df = graph
        .betweenness_centrality_all(&handle, &BetweennessCentralityParams::default())
        .expect("betweenness")
        .to_dataframe(&runtime)
        .expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "value");
    assert_eq!(df.num_rows(), 4);

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let values: Vec<f32> = df.series("value").expect("value column").to_vec_strict().expect("value to_vec");

    assert_eq!(vertices.len(), 4);
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|value| value.is_finite() && *value >= 0.0));
}
