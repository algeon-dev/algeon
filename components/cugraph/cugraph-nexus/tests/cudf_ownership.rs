//! Integration tests for owned-transfer import and release contracts.
mod common;

use common::{
    cudf::{
        assert_named_edge_columns, build_directed_named_dataframe_graph, explicit_runtime, explicit_runtime_owner,
        f32_series_device_ptr, i32_series_device_ptr, runtime_bound_handle,
    },
    support::{
        BFS_GRAPH_DST_I32, BFS_GRAPH_SRC_I32, BFS_GRAPH_WEIGHT_F32, bfs_distance_map_i32_graph as bfs_distance_map,
        bfs_graph_dataframe,
    },
};
use cudf::interop::runtime_release::TransferCompletionAuthority;
use cugraph::handle::ResourceHandle;
use rapids_interop::{InteropErrorKind, ReleaseSynchronization, TransferCompletionRule};

fn test_handle() -> ResourceHandle {
    let runtime = explicit_runtime_owner();
    runtime_bound_handle(&runtime)
}

#[test]
fn owned_transfer_zero_copy_import_preserves_device_buffers_and_bfs_semantics() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle =
        handle.enter("owned_transfer_zero_copy_import_preserves_device_buffers_and_bfs_semantics").expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(BFS_GRAPH_SRC_I32).expect("src");
    let dst = runtime.series(BFS_GRAPH_DST_I32).expect("dst");
    let wgt = runtime.series(BFS_GRAPH_WEIGHT_F32).expect("weight");

    let src_ptr = i32_series_device_ptr(&src);
    let dst_ptr = i32_series_device_ptr(&dst);
    let wgt_ptr = f32_series_device_ptr(&wgt);

    let df = cudf::DataFrame::from_series([("src", src), ("dst", dst), ("weight", wgt)]).expect("dataframe");
    let transfer = df.try_into_owned_transfer().expect("owned transfer");
    let imported_df =
        cudf::DataFrame::try_import_from_owned_device_table_transfer_into_runtime(transfer, None, &runtime)
            .expect("owned-transfer import");

    assert_named_edge_columns(&imported_df);
    assert_eq!(i32_series_device_ptr(&imported_df.series("src").expect("src series")), src_ptr);
    assert_eq!(i32_series_device_ptr(&imported_df.series("dst").expect("dst series")), dst_ptr);
    assert_eq!(f32_series_device_ptr(&imported_df.series("weight").expect("weight series")), wgt_ptr);

    let mut graph = build_directed_named_dataframe_graph(&handle, &imported_df).expect("graph_from_dataframe");
    let dist_map = bfs_distance_map(&handle, &mut graph, 0);

    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], 1);
    assert_eq!(dist_map[&3], 2);
    assert_eq!(dist_map[&4], 2);
    assert_eq!(dist_map[&5], 3);
}

#[test]
fn owned_transfer_with_release_import_invalidates_releaser() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("owned_transfer_with_release_import_invalidates_releaser").expect("entered");
    let runtime = explicit_runtime();

    let producer_df = bfs_graph_dataframe(&runtime);
    let (transfer, releaser) = producer_df.try_into_owned_transfer_with_release().expect("owned transfer with release");
    let imported_df =
        cudf::DataFrame::try_import_from_owned_device_table_transfer_into_runtime(transfer, None, &runtime)
            .expect("owned-transfer import");

    let mut graph = build_directed_named_dataframe_graph(&handle, &imported_df).expect("graph_from_dataframe");
    let dist_map = bfs_distance_map(&handle, &mut graph, 0);
    assert!(!dist_map.is_empty(), "imported owned-transfer graph should produce bfs results");

    let err = releaser
        .release(TransferCompletionAuthority::same_device_stream(runtime.stream()))
        .expect_err("releaser should be invalid after native import has already claimed the owned-transfer root");
    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    let err_text = format!("{}", err);
    assert!(
        err_text.contains("ownership state changed") || err_text.contains("no longer has importable native ownership"),
        "unexpected releaser error: {}",
        err
    );
}

#[test]
fn owned_transfer_with_release_accepts_same_device_stream_completion() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime();
    let producer_df = bfs_graph_dataframe(&runtime);
    let (transfer, releaser) = producer_df.try_into_owned_transfer_with_release().expect("owned transfer with release");

    assert_eq!(transfer.release_contract().completion(), TransferCompletionRule::ReleaseWaitsOnConsumerSignal);
    assert_eq!(transfer.release_contract().synchronization(), ReleaseSynchronization::Asynchronous);

    drop(transfer);
    releaser
        .release(TransferCompletionAuthority::same_device_stream(runtime.stream()))
        .expect("runtime-backed owned-transfer release should accept same-device stream completion");
}
