use cugraph::{DeviceArray, Forward, GraphBuilder, ResourceHandle, error::ErrorKind, types::GraphProperties};

#[test]
fn builder_rejects_mismatched_src_dst_lengths() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");

    let err = GraphBuilder::new(&src.view(), &dst.view())
        .build::<Forward>(&handle)
        .expect_err("must reject mismatched lengths");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn builder_rejects_mismatched_weights_length() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("wgt");

    let err = GraphBuilder::new(&src.view(), &dst.view())
        .weights(&wgt.view())
        .build::<Forward>(&handle)
        .expect_err("must reject mismatched weights length");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("weights"));
}

#[test]
fn builder_rejects_symmetrize_with_edge_ids() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
    let edge_ids = DeviceArray::from_slice(&handle, &[10i32, 20, 30]).expect("edge_ids");

    let err = GraphBuilder::new(&src.view(), &dst.view())
        .edge_ids(&edge_ids.view())
        .symmetrize()
        .build::<Forward>(&handle)
        .expect_err("must reject symmetrize plus edge ids");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("symmetrize"));
}

#[test]
fn builder_rejects_mismatched_temporal_dtypes() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
    let start = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("start");
    let end = DeviceArray::from_slice(&handle, &[0i64, 1, 2]).expect("end");

    let err = GraphBuilder::new(&src.view(), &dst.view())
        .edge_start_time_ids(&start.view())
        .edge_end_time_ids(&end.view())
        .build::<Forward>(&handle)
        .expect_err("must reject mismatched temporal dtypes");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("same dtype"));
}

#[test]
fn builder_rejects_vertices_dtype_mismatch_with_src_dst() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
    // vertices are i64 but src/dst are i32 — upstream would silently truncate
    let verts = DeviceArray::from_slice(&handle, &[0i64, 1, 2]).expect("verts");

    let err = GraphBuilder::new(&src.view(), &dst.view())
        .vertices(&verts.view())
        .build::<Forward>(&handle)
        .expect_err("must reject vertices dtype != src/dst dtype");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("vertices"));
    assert!(err.message().contains("same dtype"));
}

#[test]
fn builder_accepts_vertices_matching_src_dst_dtype() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
    let verts = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("verts");

    // Same dtype (i32) — should succeed
    GraphBuilder::new(&src.view(), &dst.view())
        .vertices(&verts.view())
        .renumber()
        .build::<Forward>(&handle)
        .expect("vertices with matching dtype should build");
}

#[test]
fn builder_chainable_setters_preserve_build_path() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 0]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("wgt");

    let graph = GraphBuilder::new(&src.view(), &dst.view())
        .properties(GraphProperties::directed())
        .weights(&wgt.view())
        .renumber()
        .drop_self_loops()
        .drop_multi_edges()
        .do_expensive_check()
        .build::<Forward>(&handle)
        .expect("graph");

    assert_eq!(graph.vertex_dtype(), cugraph::DataType::Int32);
}
