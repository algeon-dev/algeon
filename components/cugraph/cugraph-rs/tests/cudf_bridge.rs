//! Integration tests for the cuDF bridge surface: typed slices, representative
//! series ingress, host-slice parity, and result-to-Series conversion.
mod common;

use common::{
    cudf::explicit_runtime,
    support::{
        BFS_SOURCE_VERTEX_I32, SIMPLE_CHAIN_DST_I32, SIMPLE_CHAIN_SRC_I32, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, UNDIRECTED_TRIANGLE_DST_I32, UNDIRECTED_TRIANGLE_SRC_I32,
        UNDIRECTED_TRIANGLE_WEIGHT_F32,
    },
};
use cugraph::{
    VertexSelection,
    array::DeviceArray,
    centrality::PageRankOptions,
    cudf_interop::{IntoSeries, SeriesGraphOptions, device_array_view_from_slice, graph_from_series},
    graph::{Forward, GraphBuilder, Transposed},
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};

#[test]
fn device_array_view_from_slice_i64() {
    let runtime = explicit_runtime();
    let series = runtime.series("v", &[100i64, 200, 300]).expect("series");
    let col_view = series.as_column_view().expect("column view");
    let slice = col_view.device_slice::<i64>().expect("device_slice");

    let view = device_array_view_from_slice::<i64>(&slice).expect("type should match");
    assert_eq!(view.len(), 3);
}

#[test]
fn device_array_view_from_slice_rejects_mismatch() {
    let runtime = explicit_runtime();
    let series = runtime.series("v", &[1i32, 2, 3]).expect("series");
    let col_view = series.as_column_view().expect("column view");
    let slice = col_view.device_slice::<i32>().expect("device_slice");

    let err = device_array_view_from_slice::<f32>(&slice).expect_err("should reject i32 -> f32 mismatch");
    assert_eq!(err.kind(), cugraph::ErrorKind::UnsupportedType);
}

#[test]
fn graph_from_series_weighted_pagerank() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();

    let src = runtime.series("src", UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = runtime.series("dst", UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let wgt = runtime.series("wgt", UNDIRECTED_TRIANGLE_WEIGHT_F32).expect("wgt");

    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph = graph_from_series::<Transposed>(&handle, &src, &dst, Some(&wgt), &opts).expect("graph_from_series");

    let result = graph.pagerank(&handle, &PageRankOptions::default()).expect("pagerank");
    let scores: Vec<f32> = result.scores().expect("scores").to_vec_f32(&handle).expect("copy");

    assert_eq!(scores.len(), 3, "should have 3 vertices");
    for (i, &score) in scores.iter().enumerate() {
        assert!(score > 0.0 && score.is_finite(), "score[{i}] = {score} should be positive finite");
    }
}

#[test]
fn graph_bfs_host_slice_matches_typed_view_path() {
    let runtime = explicit_runtime();
    let handle = ResourceHandle::for_runtime_device(&runtime).expect("handle");

    let src = runtime.series("src", TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series("dst", TRIANGLE_TAIL_DST_I32).expect("dst");

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
    let handle = ResourceHandle::new().expect("handle");

    let src = DeviceArray::from_slice(&handle, SIMPLE_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, SIMPLE_CHAIN_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let vertex_series = result.vertex_ids().expect("vids").to_series("my_vertex").expect("VertexIds::to_series");
    let distance_series = result.distances().expect("dists").to_series("my_distance").expect("Distances::to_series");

    assert_eq!(vertex_series.name(), Some("my_vertex"));
    assert_eq!(distance_series.name(), Some("my_distance"));
    assert_eq!(vertex_series.to_vec_strict::<i32>().expect("vertex vec").len(), 3);
    assert_eq!(distance_series.to_vec_strict::<i32>().expect("distance vec"), vec![0, 1, 2]);
}
