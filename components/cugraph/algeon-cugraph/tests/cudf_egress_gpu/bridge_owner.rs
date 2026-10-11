//! Integration tests for the cuDF bridge surface: typed slices, representative
//! series ingress, host-slice parity, and result-to-Series conversion.
use common::{
    cudf::explicit_runtime,
    support::{BFS_SOURCE_VERTEX_I32, TRIANGLE_TAIL_DST_I32, TRIANGLE_TAIL_SRC_I32},
};
use cugraph::{
    VertexSelection,
    array::DeviceArray,
    cudf_interop::device_array_view_from_slice,
    graph::{Forward, GraphBuilder},
    handle::ResourceHandle,
    traversal::BfsParams,
};

use crate::common;

pub(super) fn device_array_view_from_slice_i64() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime();
    let series = runtime.series(&[100i64, 200, 300]).expect("series");
    let col_view = series.as_column();
    let slice = col_view.device_slice::<i64>().expect("device_slice");

    let view = device_array_view_from_slice::<i64>(&slice).expect("type should match");
    assert_eq!(view.len(), 3);
}

pub(super) fn device_array_view_from_slice_rejects_mismatch() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime();
    let series = runtime.series(&[1i32, 2, 3]).expect("series");
    let col_view = series.as_column();
    let slice = col_view.device_slice::<i32>().expect("device_slice");

    let err = device_array_view_from_slice::<f32>(&slice).expect_err("should reject i32 -> f32 mismatch");
    assert_eq!(err.kind(), cugraph::ErrorKind::UnsupportedType);
}

pub(super) fn graph_bfs_host_slice_matches_typed_view_path() {
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
