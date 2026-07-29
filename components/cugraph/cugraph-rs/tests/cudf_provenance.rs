//! Integration tests for cuDF producer provenance and borrowed-column
//! compatibility requirements.
mod common;

use common::{
    cudf::{
        assert_score_maps_close_f32, explicit_runtime, explicit_runtime_owner, python_undirected_multigraph_dataframe,
        python_weighted_reciprocal_expected_pagerank_map, runtime_bound_handle,
    },
    support::{
        BFS_SOURCE_VERTEX_I32, SIMPLE_CHAIN_DST_I64, SIMPLE_CHAIN_SRC_I64, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, TRIANGLE_TAIL_WEIGHT_F32, bfs_distance_map_i32_result, bfs_distance_map_i64_result,
    },
};
use cudf::testing::prelude::*;
use cugraph::{
    array::DeviceArray,
    cudf_interop::{
        BorrowedInputOrderingPlan, SeriesGraphOptions, TableGraphOptions, graph_from_borrowed_columns,
        graph_from_borrowed_table_view, graph_from_series, testing::borrowed_columns_ordering_plan,
    },
    graph::{Forward, Transposed},
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};
use rapids_interop::{BorrowedColumnView, CudaStreamHandle, ExecutionBinding};

fn reset_exec_observation() {
    unsafe { cudf_sys::cudf_test_exec_reset_observation() };
}

fn last_sync_stream_value() -> usize {
    let mut out = 0usize;
    let status = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last synchronized stream");
    out
}

fn reset_event_observation() {
    unsafe { cudf_sys::cudf_test_event_reset_observation() };
}

fn observed_stream_wait_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream wait observation count");
    out
}

fn test_handle() -> ResourceHandle {
    let runtime = explicit_runtime_owner();
    runtime_bound_handle(&runtime)
}

fn borrowed_with_device<'a>(column: &BorrowedColumnView<'a>, device: i32) -> BorrowedColumnView<'a> {
    let execution = ExecutionBinding::explicit(
        device,
        CudaStreamHandle::from_raw_on_device(column.execution().cuda_stream().as_raw(), device).expect("stream handle"),
        column.execution().memory_resource(),
    );
    BorrowedColumnView::try_new(
        column.data_type(),
        column.len(),
        column.offset(),
        column.data_ptr(),
        *column.null_mask(),
        execution,
    )
    .expect("borrowed column with overridden device")
}

#[test]
fn runtime_bound_handle_same_provenance_borrowed_columns_plan_has_no_host_fence() {
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle =
        handle.enter("runtime_bound_handle_same_provenance_borrowed_columns_plan_has_no_host_fence").expect("entered");

    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");

    let plan = borrowed_columns_ordering_plan(&handle, &src_borrowed, &dst_borrowed, None).expect("ordering plan");
    assert_eq!(plan, BorrowedInputOrderingPlan::RuntimeNoHostFence);
}

#[test]
fn runtime_bound_handle_same_device_different_stream_plan_uses_device_dependencies() {
    let handle_runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&handle_runtime);
    let handle = handle
        .enter("runtime_bound_handle_same_device_different_stream_plan_uses_device_dependencies")
        .expect("entered");
    let rt_a = explicit_runtime();
    let rt_b = explicit_runtime();

    let src = rt_a.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = rt_b.series(TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");

    let plan = borrowed_columns_ordering_plan(&handle, &src_borrowed, &dst_borrowed, None).expect("ordering plan");
    assert_eq!(plan, BorrowedInputOrderingPlan::RuntimeDeviceSideDependencies { dependency_count: 2 });
}

#[test]
fn runtime_bound_handle_cross_device_borrowed_columns_are_rejected() {
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle = handle.enter("runtime_bound_handle_cross_device_borrowed_columns_are_rejected").expect("entered");

    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let foreign_device = handle.device() + 1;
    let src_foreign = borrowed_with_device(&src_borrowed, foreign_device);
    let dst_foreign = borrowed_with_device(&dst_borrowed, foreign_device);

    let err =
        borrowed_columns_ordering_plan(&handle, &src_foreign, &dst_foreign, None).expect_err("cross-device reject");
    assert_eq!(err.kind(), cugraph::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "cudf_interop::testing::borrowed_columns_ordering_plan");
}

#[test]
fn graph_from_series_allows_mixed_explicit_runtimes_on_same_device() {
    let handle = test_handle();
    let handle = handle.enter("graph_from_series_allows_mixed_explicit_runtimes_on_same_device").expect("entered");
    let rt_a = explicit_runtime();
    let rt_b = explicit_runtime();

    let src = rt_a.series(SIMPLE_CHAIN_SRC_I64).expect("src");
    let dst = rt_b.series(SIMPLE_CHAIN_DST_I64).expect("dst");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph =
        graph_from_series::<Forward>(&handle, &src, &dst, None, &opts).expect("same-device mixed runtimes should work");

    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let dist_map = bfs_distance_map_i64_result(&handle, &result);
    assert_eq!(dist_map[&2], 2);
}

#[test]
fn graph_from_borrowed_columns_allows_mixed_explicit_runtimes_on_same_device() {
    let handle = test_handle();
    let handle =
        handle.enter("graph_from_borrowed_columns_allows_mixed_explicit_runtimes_on_same_device").expect("entered");
    let rt_a = explicit_runtime();
    let rt_b = explicit_runtime();
    let rt_c = explicit_runtime();

    let src = rt_a.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = rt_b.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let wgt = rt_c.series(TRIANGLE_TAIL_WEIGHT_F32).expect("wgt");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let wgt_borrowed = wgt.try_borrow_column_view().expect("wgt borrowed");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph =
        graph_from_borrowed_columns::<Forward>(&handle, &src_borrowed, &dst_borrowed, Some(&wgt_borrowed), &opts)
            .expect("borrowed mixed runtimes should work");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let dist_map = bfs_distance_map_i32_result(&handle, &result);
    assert_eq!(dist_map[&3], 3);
}

#[test]
fn graph_from_borrowed_columns_runtime_bound_same_device_different_stream_avoids_host_fence() {
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle = handle
        .enter("graph_from_borrowed_columns_runtime_bound_same_device_different_stream_avoids_host_fence")
        .expect("entered");
    let rt_a = explicit_runtime();
    let rt_b = explicit_runtime();

    let src = rt_a.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = rt_b.series(TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let consumer_stream_raw = runtime.stream().testing_raw_handle_value();

    reset_exec_observation();
    reset_event_observation();
    let baseline_sync = last_sync_stream_value();
    let baseline_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);

    let opts = SeriesGraphOptions::default().renumber();
    let _graph =
        graph_from_borrowed_columns::<Forward>(&handle, &src_borrowed, &dst_borrowed, None, &opts).expect("graph");

    assert_eq!(
        last_sync_stream_value(),
        baseline_sync,
        "runtime-bound borrowed path should not host-synchronize producer streams"
    );
    assert!(
        observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_wait,
        "runtime stream should wait on producer barriers instead"
    );
}

#[test]
fn graph_from_borrowed_columns_python_cugraph_runtime_bound_avoids_host_fence() {
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle =
        handle.enter("graph_from_borrowed_columns_python_cugraph_runtime_bound_avoids_host_fence").expect("entered");
    let rt_a = explicit_runtime();
    let rt_b = explicit_runtime();
    let rt_c = explicit_runtime();

    let src = rt_a.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = rt_b.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = rt_c.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("wgt");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let wgt_borrowed = wgt.try_borrow_column_view().expect("wgt borrowed");
    let consumer_stream_raw = runtime.stream().testing_raw_handle_value();

    reset_exec_observation();
    reset_event_observation();
    let baseline_sync = last_sync_stream_value();
    let baseline_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);

    let opts = SeriesGraphOptions::default()
        .properties(GraphProperties::undirected())
        .renumber()
        .construction_policy(cugraph::cudf_interop::GraphConstructionPolicy::PythonCugraph);
    let _graph =
        graph_from_borrowed_columns::<Transposed>(&handle, &src_borrowed, &dst_borrowed, Some(&wgt_borrowed), &opts)
            .expect("graph");

    assert_eq!(
        last_sync_stream_value(),
        baseline_sync,
        "python-normalized runtime-bound borrowed path should not fall back to host fencing"
    );
    assert!(
        observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_wait,
        "runtime stream should inherit producer ordering through device-side waits"
    );
}

#[test]
fn graph_from_borrowed_columns_python_cugraph_weighted_undirected_simple_keeps_minimum_reciprocal_weight() {
    let handle = test_handle();
    let handle = handle
        .enter("graph_from_borrowed_columns_python_cugraph_weighted_undirected_simple_keeps_minimum_reciprocal_weight")
        .expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("wgt");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let wgt_borrowed = wgt.try_borrow_column_view().expect("wgt borrowed");

    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph =
        graph_from_borrowed_columns::<Transposed>(&handle, &src_borrowed, &dst_borrowed, Some(&wgt_borrowed), &opts)
            .expect("graph_from_borrowed_columns");

    let actual = common::cudf::pagerank_score_map(&handle, &mut graph);
    let expected = python_weighted_reciprocal_expected_pagerank_map(&handle);
    assert_score_maps_close_f32(&actual, &expected, 1.0e-6, "graph_from_borrowed_columns");
}

#[test]
fn graph_from_borrowed_table_view_python_cugraph_accepts_undirected_multigraph() {
    let handle = test_handle();
    let handle =
        handle.enter("graph_from_borrowed_table_view_python_cugraph_accepts_undirected_multigraph").expect("entered");
    let runtime = explicit_runtime();
    let df = python_undirected_multigraph_dataframe(&runtime);
    let borrowed = df.try_borrow_table_view().expect("borrowed table view");
    let props = GraphProperties { is_symmetric: true, is_multigraph: true };

    let opts = TableGraphOptions::new(0, 1).properties(props).renumber();
    let mut graph =
        graph_from_borrowed_table_view::<Forward>(&handle, &borrowed, &opts).expect("graph_from_borrowed_table_view");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), 3);
}
