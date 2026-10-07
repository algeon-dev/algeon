#[path = "../support/common/mod.rs"]
mod common;

use common::{
    cudf::explicit_runtime,
    support::{
        DIRECTED_STAR_DST_I64, DIRECTED_STAR_SRC_I64, DIRECTED_STAR_UNIT_WEIGHT_F64, DIRECTED_TRIANGLE_CYCLE_DST_I64,
        DIRECTED_TRIANGLE_CYCLE_SRC_I64,
    },
};
use cugraph::{
    DeviceArray, Forward, GraphBuilder, ResourceHandle, Transposed, centrality::PageRankParams, error::ErrorKind,
    types::DataType,
};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(explicit_runtime());
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

#[test]
fn graph_reports_i64_vertex_and_f64_weight_dtypes() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("graph_reports_i64_vertex_and_f64_weight_dtypes").expect("entered");
    let src = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_DST_I64).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f64, 2.0, 3.0]).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let graph = GraphBuilder::new(&src_view, &dst_view).weights(&wgt_view).build::<Forward>(&handle).expect("graph");

    assert_eq!(graph.vertex_dtype(), DataType::Int64);
    assert_eq!(graph.weight_dtype(), DataType::Float64);
}

#[test]
fn typed_results_refuse_lossy_narrowing() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("typed_results_refuse_lossy_narrowing").expect("entered");
    let src = DeviceArray::from_slice(&handle, DIRECTED_STAR_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_STAR_DST_I64).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, DIRECTED_STAR_UNIT_WEIGHT_F64).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank(&handle, &PageRankParams::default()).expect("pagerank");

    let vertex_err = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect_err("must reject narrowing");
    assert_eq!(vertex_err.kind(), ErrorKind::InvalidState);

    let score_err = result.scores().expect("scores").to_vec_f32(&handle).expect_err("must reject narrowing");
    assert_eq!(score_err.kind(), ErrorKind::InvalidState);
}
