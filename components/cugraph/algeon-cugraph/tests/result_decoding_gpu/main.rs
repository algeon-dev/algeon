#[path = "../support/common/mod.rs"]
mod common;

use common::{
    cudf::explicit_runtime,
    support::{FOUR_VERTEX_CHAIN_DST_I32, FOUR_VERTEX_CHAIN_SRC_I32},
};
use cugraph::{DeviceArray, Forward, GraphBuilder, ResourceHandle, traversal::BfsParams};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(explicit_runtime());
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

#[test]
fn bfs_predecessors_are_none_when_not_requested() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("bfs_predecessors_are_none_when_not_requested").expect("entered");
    let src = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");
    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&entered, &sources_view, &BfsParams::default()).expect("bfs");

    assert!(result.predecessors().expect("predecessors").is_none());
}
