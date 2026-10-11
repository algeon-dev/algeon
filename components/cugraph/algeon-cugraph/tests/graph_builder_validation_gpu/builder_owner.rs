use cugraph::{DeviceArray, ErrorKind, Forward, GraphBuilder};

use crate::test_handle;

pub(super) fn test_graph_creation_rejects_mismatched_edge_ids_dtype() {
    let handle = test_handle();
    let entered = handle.enter("test_graph_creation_rejects_mismatched_edge_ids_dtype").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let edge_ids = DeviceArray::from_slice(&entered, &[10i64, 20, 30]).expect("edge_ids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_ids(&edge_ids_view)
        .build::<Forward>(&entered)
        .expect_err("should reject edge_ids dtype mismatch");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_ids must have the same dtype as src/dst"));
}

pub(super) fn test_graph_creation_rejects_mismatched_edge_type_ids_length() {
    let handle = test_handle();
    let entered = handle.enter("test_graph_creation_rejects_mismatched_edge_type_ids_length").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let edge_types = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("edge_types");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_types_view = edge_types.view().expect("edge_types_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_type_ids(&edge_types_view)
        .build::<Forward>(&entered)
        .expect_err("should reject mismatched edge_type_ids length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_type_ids"));
}

pub(super) fn test_graph_creation_rejects_mismatched_edge_ids_length() {
    let handle = test_handle();
    let entered = handle.enter("test_graph_creation_rejects_mismatched_edge_ids_length").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let edge_ids = DeviceArray::from_slice(&entered, &[10i32, 20]).expect("edge_ids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_ids(&edge_ids_view)
        .build::<Forward>(&entered)
        .expect_err("should reject mismatched edge_ids length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_ids"));
}

pub(super) fn test_rejects_symmetrize_with_edge_type_ids() {
    let handle = test_handle();
    let entered = handle.enter("test_rejects_symmetrize_with_edge_type_ids").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let etids = DeviceArray::from_slice(&entered, &[0i32, 1, 0]).expect("etids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let etids_view = etids.view().expect("etids_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_type_ids(&etids_view)
        .symmetrize()
        .build::<Forward>(&entered)
        .expect_err("should reject symmetrize + edge_type_ids");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("symmetrize"));
    assert!(err.message().contains("edge_type_ids"));
}

pub(super) fn test_rejects_start_time_without_end_time() {
    let handle = test_handle();
    let entered = handle.enter("test_rejects_start_time_without_end_time").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let start = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("start");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let start_view = start.view().expect("start_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_start_time_ids(&start_view)
        .build::<Forward>(&entered)
        .expect_err("should reject start_time without end_time");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_end_time_ids"));
    assert!(err.message().contains("was not"));
}
