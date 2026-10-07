//! Integration tests for the cuDF bridge surface: typed slices, representative
//! series ingress, host-slice parity, and result-to-Series conversion.
#[path = "../support/common/mod.rs"]
mod common;

use common::{
    cudf::explicit_runtime,
    support::{
        BFS_SOURCE_VERTEX_I32, SIMPLE_CHAIN_DST_I32, SIMPLE_CHAIN_SRC_I32, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, UNDIRECTED_TRIANGLE_DST_I32, UNDIRECTED_TRIANGLE_SRC_I32,
    },
};
use cugraph::{
    VertexSelection,
    array::DeviceArray,
    cudf_interop::device_array_view_from_slice,
    graph::{Forward, GraphBuilder},
    handle::ResourceHandle,
    layout::{ForceAtlas2Params, LayoutResult},
    params::IterationCount,
    random::RngState,
    traversal::BfsParams,
    types::GraphProperties,
};

#[test]
fn device_array_view_from_slice_i64() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime();
    let series = runtime.series(&[100i64, 200, 300]).expect("series");
    let col_view = series.as_column();
    let slice = col_view.device_slice::<i64>().expect("device_slice");

    let view = device_array_view_from_slice::<i64>(&slice).expect("type should match");
    assert_eq!(view.len(), 3);
}

#[test]
fn device_array_view_from_slice_rejects_mismatch() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime();
    let series = runtime.series(&[1i32, 2, 3]).expect("series");
    let col_view = series.as_column();
    let slice = col_view.device_slice::<i32>().expect("device_slice");

    let err = device_array_view_from_slice::<f32>(&slice).expect_err("should reject i32 -> f32 mismatch");
    assert_eq!(err.kind(), cugraph::ErrorKind::UnsupportedType);
}

#[test]
fn graph_bfs_host_slice_matches_typed_view_path() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("graph_bfs_host_slice_matches_typed_view_path").expect("entered");

    let src = DeviceArray::from_slice(&handle, TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, TRIANGLE_TAIL_DST_I32).expect("dst");
    let src = src.view().expect("src view");
    let dst = dst.view().expect("dst view");
    let mut graph = GraphBuilder::new(&src, &dst).renumber().build::<Forward>(&handle).expect("graph");

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
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("into_series_bfs_results").expect("entered");

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
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("into_series_layout_results").expect("entered");
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
