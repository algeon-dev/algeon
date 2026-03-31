use cugraph::{DataType, DeviceArray, Forward, GraphBuilder, PageRankParams, ResourceHandle, Transposed};
use cugraph_sys::cugraph_data_type_id_t;

#[test]
fn datatype_try_from_rejects_unknown_ids() {
    for raw_id in [99, 100, 255, 1000, u32::MAX] {
        let c_type = cugraph_data_type_id_t(raw_id);
        assert!(DataType::try_from(c_type).is_err(), "unexpected mapping for raw dtype id {raw_id}");
    }
}

#[test]
fn graph_reports_i64_vertex_and_f64_weight_dtypes() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i64, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 2, 0]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f64, 2.0, 3.0]).expect("wgt");

    let graph =
        GraphBuilder::new(&src.view(), &dst.view()).weights(&wgt.view()).build::<Forward>(&handle).expect("graph");

    assert_eq!(graph.vertex_dtype(), DataType::Int64);
    assert_eq!(graph.weight_dtype(), DataType::Float64);
}

#[test]
fn typed_results_refuse_lossy_narrowing() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[1i64, 2, 3, 0, 0, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[0i64, 0, 0, 1, 2, 3]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f64; 6]).expect("wgt");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view())
        .weights(&wgt.view())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");

    let vertex_err = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect_err("must reject narrowing");
    assert_eq!(vertex_err.kind(), cugraph::ErrorKind::InvalidInput);

    let score_err = result.scores().expect("scores").to_vec_f32(&handle).expect_err("must reject narrowing");
    assert_eq!(score_err.kind(), cugraph::ErrorKind::InvalidInput);
}
