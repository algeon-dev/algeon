//! Integration tests for the `Distances` result contract.
//!
//! Keep this file focused on runtime result typing that is not already covered
//! by source-local unit tests. Happy-path accessor smoke tests live elsewhere.

mod common;

use common::{
    cudf::explicit_runtime,
    support::{FOUR_VERTEX_CHAIN_DST_I64, FOUR_VERTEX_CHAIN_SRC_I64, SIMPLE_CHAIN_DST_I32, SIMPLE_CHAIN_SRC_I32},
};
use cugraph::{DeviceArray, Forward, GraphBuilder, ResourceHandle, structure::Distances, traversal::BfsParams};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(explicit_runtime());
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

#[test]
fn bfs_distances_i32_supports_runtime_decoding_and_widening() {
    let handle = test_handle();
    let src = DeviceArray::from_slice(&handle, SIMPLE_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, SIMPLE_CHAIN_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let distances = result.distances().expect("distances");
    assert!(matches!(distances, Distances::I32(_)), "BFS i32 graph should produce Distances::I32");
    assert_eq!(distances.len(), 3);
    assert!(!distances.is_empty());
    let as_i32 = distances.to_vec_i32(&handle).expect("to_vec_i32");
    assert_eq!(as_i32.len(), 3);
    assert!(as_i32.contains(&0), "source distance should be 0");
    let as_i64 = distances.to_vec_i64(&handle).expect("to_vec_i64");
    assert_eq!(as_i64, vec![0, 1, 2]);
    let as_f64 = distances.to_vec_f64(&handle).expect("to_vec_f64");
    assert_eq!(as_f64, vec![0.0, 1.0, 2.0]);
    assert!(as_f64.contains(&0.0), "source distance should be 0.0");
}

#[test]
fn bfs_distances_i64_graph_returns_i64() {
    let handle = test_handle();
    let src = DeviceArray::from_slice(&handle, FOUR_VERTEX_CHAIN_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, FOUR_VERTEX_CHAIN_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let distances = result.distances().expect("distances");
    assert!(matches!(distances, Distances::I64(_)), "BFS i64 graph should produce Distances::I64");
}
