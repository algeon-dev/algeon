//! Integration tests for the cuDF bridge surface: typed slices, representative
//! series ingress, host-slice parity, and result-to-Series conversion.
mod common;

use common::{
    cudf::{assert_score_maps_close_f32, explicit_runtime, python_weighted_reciprocal_expected_pagerank_map},
    support::{
        BFS_SOURCE_VERTEX_I32, SIMPLE_CHAIN_DST_I32, SIMPLE_CHAIN_SRC_I32, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, UNDIRECTED_TRIANGLE_DST_I32, UNDIRECTED_TRIANGLE_SRC_I32,
        UNDIRECTED_TRIANGLE_WEIGHT_F32,
    },
};
use cugraph::{
    VertexSelection,
    array::DeviceArray,
    centrality::PageRankParams,
    cudf_interop::{IntoSeries, SeriesGraphOptions, device_array_view_from_slice, graph_from_series},
    graph::{Forward, GraphBuilder, Transposed},
    handle::ResourceHandle,
    layout::{ForceAtlas2Params, LayoutResult},
    params::IterationCount,
    random::RngState,
    traversal::BfsParams,
    types::GraphProperties,
};

#[test]
fn device_array_view_from_slice_i64() {
    let runtime = explicit_runtime();
    let series = runtime.series(&[100i64, 200, 300]).expect("series");
    let col_view = series.as_column();
    let slice = col_view.device_slice::<i64>().expect("device_slice");

    let view = device_array_view_from_slice::<i64>(&slice).expect("type should match");
    assert_eq!(view.len(), 3);
}

#[test]
fn device_array_view_from_slice_rejects_mismatch() {
    let runtime = explicit_runtime();
    let series = runtime.series(&[1i32, 2, 3]).expect("series");
    let col_view = series.as_column();
    let slice = col_view.device_slice::<i32>().expect("device_slice");

    let err = device_array_view_from_slice::<f32>(&slice).expect_err("should reject i32 -> f32 mismatch");
    assert_eq!(err.kind(), cugraph::ErrorKind::UnsupportedType);
}

#[test]
fn graph_from_series_weighted_pagerank() {
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");

    let src = runtime.series(UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = runtime.series(UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let wgt = runtime.series(UNDIRECTED_TRIANGLE_WEIGHT_F32).expect("wgt");

    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph = graph_from_series::<Transposed>(&handle, &src, &dst, Some(&wgt), &opts).expect("graph_from_series");

    let result = graph.pagerank(&handle, &PageRankParams::default()).expect("pagerank");
    let scores: Vec<f32> = result.scores().expect("scores").to_vec_f32(&handle).expect("copy");

    assert_eq!(scores.len(), 3, "should have 3 vertices");
    for (i, &score) in scores.iter().enumerate() {
        assert!(score > 0.0 && score.is_finite(), "score[{i}] = {score} should be positive finite");
    }
}

#[test]
fn graph_from_series_python_cugraph_weighted_undirected_simple_keeps_minimum_reciprocal_weight() {
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");

    let src = runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("wgt");

    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph = graph_from_series::<Transposed>(&handle, &src, &dst, Some(&wgt), &opts).expect("graph_from_series");

    let actual = common::cudf::pagerank_score_map(&handle, &mut graph);
    let expected = python_weighted_reciprocal_expected_pagerank_map(&handle);
    assert_score_maps_close_f32(&actual, &expected, 1.0e-6, "graph_from_series");
}

#[test]
fn graph_bfs_host_slice_matches_typed_view_path() {
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");

    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph = graph_from_series::<Forward>(&handle, &src, &dst, None, &opts).expect("graph_from_series");

    let direct_sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let direct_sources_view = direct_sources.view().expect("direct_sources_view");
    let direct = graph.bfs(&handle, &direct_sources_view, &BfsParams::default()).expect("bfs");
    let from_host = graph
        .bfs(&handle, VertexSelection::Host(&[BFS_SOURCE_VERTEX_I32][..]), &BfsParams::default())
        .expect("bfs via explicit host selection");

    let direct_vertices: Vec<i32> = direct.vertex_ids().expect("vids").to_vec_i32(&handle).expect("copy");
    let direct_distances: Vec<i32> = direct.distances().expect("dists").to_vec_i32(&handle).expect("copy");
    let host_vertices: Vec<i32> = from_host.vertex_ids().expect("vids").to_vec_i32(&handle).expect("copy");
    let host_distances: Vec<i32> = from_host.distances().expect("dists").to_vec_i32(&handle).expect("copy");

    assert_eq!(host_vertices, direct_vertices);
    assert_eq!(host_distances, direct_distances);
}

#[test]
fn into_series_bfs_results() {
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");

    let src = DeviceArray::from_slice(&handle, SIMPLE_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, SIMPLE_CHAIN_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let vertex_series = result.vertex_ids().expect("vids").to_series(&runtime).expect("VertexIds::to_series");
    let distance_series = result.distances().expect("dists").to_series(&runtime).expect("Distances::to_series");

    assert_eq!(vertex_series.to_vec_strict::<i32>().expect("vertex vec").len(), 3);
    assert_eq!(distance_series.to_vec_strict::<i32>().expect("distance vec"), vec![0, 1, 2]);
}

#[test]
fn into_series_layout_results() {
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let mut rng = RngState::new(&handle, 42).expect("rng");

    let src = DeviceArray::from_slice(&handle, UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ForceAtlas2Params { max_iter: IterationCount::new(50).unwrap(), ..Default::default() };
    let result: LayoutResult = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2");

    let vertex_series = result.vertex_ids().expect("vertex_ids").to_series(&runtime).expect("VertexIds::to_series");
    let x_series = result.x().expect("x").to_series(&runtime).expect("ScoreValues::to_series x");
    let y_series = result.y().expect("y").to_series(&runtime).expect("ScoreValues::to_series y");

    assert_eq!(vertex_series.to_vec_strict::<i32>().expect("vertex vec").len(), 3);

    let x_values = x_series.to_vec_strict::<f32>().expect("x vec");
    let y_values = y_series.to_vec_strict::<f32>().expect("y vec");
    assert_eq!(x_values.len(), 3);
    assert_eq!(y_values.len(), 3);
    assert!(x_values.iter().all(|value| value.is_finite()));
    assert!(y_values.iter().all(|value| value.is_finite()));
}
