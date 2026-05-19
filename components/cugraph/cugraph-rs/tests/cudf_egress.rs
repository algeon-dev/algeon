//! Integration tests for graph/result exports back into cuDF.
mod common;

use std::collections::HashMap;

use common::{
    cudf::{explicit_runtime, read_parquet_dataframe, unique_temp_dir, write_parquet_dataframe},
    support,
    support::{
        BFS_GRAPH_DST_I32, BFS_GRAPH_SRC_I32, BFS_SOURCE_VERTEX_I32, DIRECTED_STAR_DST_I32, DIRECTED_STAR_SRC_I32,
        LOUVAIN_GRAPH_DST_I32, LOUVAIN_GRAPH_SRC_I32, LOUVAIN_GRAPH_WEIGHT_F32, SSSP_GRAPH_DST_I32, SSSP_GRAPH_SRC_I32,
        SSSP_GRAPH_WEIGHT_F32, TEST_LAYOUT_PRIMARY_SEED, TEST_LAYOUT_SETTLE_ITERATIONS,
    },
};
use cugraph::{
    array::DeviceArray,
    centrality::{BetweennessCentralityParams, PageRankParams},
    community::LouvainParams,
    graph::{Forward, GraphBuilder, Transposed},
    handle::ResourceHandle,
    traversal::{BfsParams, SsspParams},
    types::GraphProperties,
};

#[test]
fn layout_result_to_dataframe() {
    use cugraph::{layout::ForceAtlas2Params, params::IterationCount, random::RngState};

    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("layout_result_to_dataframe").expect("entered");
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

    let result = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2");
    let expected_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let expected_x = result.x().expect("x").to_vec_f32(&handle).expect("x copy");
    let expected_y = result.y().expect("y").to_vec_f32(&handle).expect("y copy");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "x");
    assert_eq!(df.column_names()[2], "y");
    assert_eq!(df.num_rows(), 3);

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let x_vals: Vec<f32> = df.series("x").expect("x column").to_vec_strict().expect("x to_vec");
    let y_vals: Vec<f32> = df.series("y").expect("y column").to_vec_strict().expect("y to_vec");
    assert_eq!(vertices, expected_vertices);
    assert_eq!(x_vals, expected_x);
    assert_eq!(y_vals, expected_y);
    for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
        assert!(x.is_finite(), "x[{i}] = {x} should be finite");
        assert!(y.is_finite(), "y[{i}] = {y} should be finite");
    }
}

#[test]
fn bfs_result_to_dataframe_with_predecessors() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("bfs_result_to_dataframe_with_predecessors").expect("entered");

    let src = DeviceArray::from_slice(&handle, BFS_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, BFS_GRAPH_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let params = BfsParams::default().with_predecessors();
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");
    let expected_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let expected_distances = result.distances().expect("distances").to_vec_i32(&handle).expect("distance copy");
    let expected_predecessors = result
        .predecessors()
        .expect("predecessors")
        .expect("predecessors should exist")
        .to_vec_i32(&handle)
        .expect("predecessor copy");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3, "should have 3 columns with predecessors");
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "distance");
    assert_eq!(df.column_names()[2], "predecessor");

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let distances: Vec<i32> = df.series("distance").expect("distance column").to_vec_strict().expect("distance to_vec");
    let predecessors: Vec<i32> =
        df.series("predecessor").expect("predecessor column").to_vec_strict().expect("predecessor to_vec");
    assert_eq!(vertices, expected_vertices);
    assert_eq!(distances, expected_distances);
    assert_eq!(predecessors, expected_predecessors);
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex should have distance 0");
}

#[test]
fn sssp_result_to_dataframe_with_predecessors() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("sssp_result_to_dataframe_with_predecessors").expect("entered");

    let src = DeviceArray::from_slice(&handle, SSSP_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, SSSP_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&handle, SSSP_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.sssp(&handle, 0i32, &SsspParams::default().with_predecessors()).expect("sssp");
    let expected_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let expected_distances = result.distances().expect("distances").to_vec_f32(&handle).expect("distance copy");
    let expected_predecessors = result
        .predecessors()
        .expect("predecessors")
        .expect("predecessors should exist")
        .to_vec_i32(&handle)
        .expect("predecessor copy");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3, "should have 3 columns with predecessors");
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "distance");
    assert_eq!(df.column_names()[2], "predecessor");

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let distances: Vec<f32> = df.series("distance").expect("distance column").to_vec_strict().expect("distance to_vec");
    let predecessors: Vec<i32> =
        df.series("predecessor").expect("predecessor column").to_vec_strict().expect("predecessor to_vec");
    assert_eq!(vertices, expected_vertices);
    assert_eq!(distances, expected_distances);
    assert_eq!(predecessors, expected_predecessors);
}

#[test]
fn centrality_to_dataframe_write_parquet_roundtrip() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("centrality_to_dataframe_write_parquet_roundtrip").expect("entered");

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

    write_parquet_dataframe(&df, &out_path);

    let read_df = read_parquet_dataframe(&out_path, &explicit_runtime());

    assert_eq!(read_df.num_columns(), 2);
    let read_verts: Vec<i32> = read_df.series("vertex").expect("vertex").to_vec_strict().expect("vertex vec");
    let read_vals: Vec<f32> = read_df.series("value").expect("value").to_vec_strict().expect("value vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch after Parquet round-trip");
    assert_eq!(read_vals, expected_values, "value column mismatch after Parquet round-trip");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn betweenness_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("betweenness_result_to_dataframe").expect("entered");

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

#[test]
fn louvain_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("louvain_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.louvain(&handle, &LouvainParams::default()).expect("louvain");
    let expected_rows = result.vertex_ids().expect("vertex_ids").len();
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "partition");
    assert_eq!(df.num_rows(), expected_rows);
}

#[test]
fn pagerank_result_preserves_external_vertex_ids_for_renumbered_graphs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("pagerank_result_preserves_external_vertex_ids_for_renumbered_graphs").expect("entered");

    let expected_vertices = vec![10_i32, 20, 40, 80];
    let src_data = vec![10_i32, 20, 40, 80, 20, 40, 80, 10];
    let dst_data = vec![20_i32, 40, 80, 10, 10, 20, 40, 80];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank(&handle, &PageRankParams::default()).expect("pagerank");

    let mut raw_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    raw_vertices.sort_unstable();
    assert_eq!(raw_vertices, expected_vertices, "raw PageRank result leaked internal vertex ids");

    let df = result.to_dataframe(&runtime).expect("to_dataframe");
    let mut df_vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    df_vertices.sort_unstable();
    assert_eq!(df_vertices, expected_vertices, "PageRank DataFrame vertex column leaked internal vertex ids");
}

#[test]
fn louvain_result_preserves_external_vertex_ids_for_renumbered_graphs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("louvain_result_preserves_external_vertex_ids_for_renumbered_graphs").expect("entered");

    let expected_vertices = vec![10_i32, 20, 40, 80];
    let src_data = vec![10_i32, 20, 40, 80, 20, 40, 80, 10];
    let dst_data = vec![20_i32, 40, 80, 10, 10, 20, 40, 80];
    let weights_data = vec![1.0_f32; src_data.len()];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let weights = DeviceArray::from_slice(&handle, &weights_data).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.louvain(&handle, &LouvainParams::default()).expect("louvain");

    let mut raw_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    raw_vertices.sort_unstable();
    assert_eq!(raw_vertices, expected_vertices, "raw Louvain result leaked internal vertex ids");

    let df = result.to_dataframe(&runtime).expect("to_dataframe");
    let mut df_vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    df_vertices.sort_unstable();
    assert_eq!(df_vertices, expected_vertices, "Louvain DataFrame vertex column leaked internal vertex ids");
}

#[test]
fn layout_result_preserves_external_vertex_ids_for_renumbered_graphs() {
    use cugraph::{layout::ForceAtlas2Params, params::IterationCount, random::RngState};

    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("layout_result_preserves_external_vertex_ids_for_renumbered_graphs").expect("entered");
    let mut rng = RngState::new(&handle, TEST_LAYOUT_PRIMARY_SEED).expect("rng");

    let expected_vertices = vec![10_i32, 20, 40, 80];
    let src_data = vec![10_i32, 20, 40, 80, 20, 40, 80, 10];
    let dst_data = vec![20_i32, 40, 80, 10, 10, 20, 40, 80];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
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
    let result = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2");

    let mut raw_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    raw_vertices.sort_unstable();
    assert_eq!(raw_vertices, expected_vertices, "raw layout result leaked internal vertex ids");

    let df = result.to_dataframe(&runtime).expect("to_dataframe");
    let mut df_vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    df_vertices.sort_unstable();
    assert_eq!(df_vertices, expected_vertices, "layout DataFrame vertex column leaked internal vertex ids");
}
