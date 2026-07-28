//! Integration tests for cuDF producer provenance and borrowed-column
//! compatibility requirements.
mod common;

use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use common::{
    cudf::{
        assert_score_maps_close_f32, explicit_runtime, explicit_runtime_owner, python_undirected_multigraph_dataframe,
        python_weighted_reciprocal_expected_pagerank_map, runtime_bound_handle,
    },
    support::{
        BFS_SOURCE_VERTEX_I32, DISCONNECTED_TWO_CYCLES_DST_I32, DISCONNECTED_TWO_CYCLES_SRC_I32, SIMPLE_CHAIN_DST_I64,
        SIMPLE_CHAIN_SRC_I64, SSSP_GRAPH_DST_I32, SSSP_GRAPH_SRC_I32, SSSP_GRAPH_WEIGHT_F32, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, TRIANGLE_TAIL_WEIGHT_F32, bfs_distance_map_i32_result, bfs_distance_map_i64_result,
    },
};
use cudf::testing::prelude::*;
use cudf_sys::test_support;
use cugraph::{
    array::DeviceArray,
    centrality::PageRankParams,
    cudf_interop::{
        BorrowedInputOrderingPlan, SeriesGraphOptions, TableGraphOptions, graph_from_borrowed_columns,
        graph_from_borrowed_table_view, graph_from_series, testing::borrowed_columns_ordering_plan,
    },
    graph::{Forward, Transposed},
    handle::ResourceHandle,
    labeling::{ConnectedComponentsParams, WeaklyConnectedComponentsResult},
    params::NonNegativeF64,
    traversal::{BfsParams, SsspParams, SsspResult},
    types::GraphProperties,
};
use rapids_interop::{BorrowedColumnView, CudaStreamHandle, ExecutionBinding};

fn reset_exec_observation() {
    unsafe { test_support::cudf_test_exec_reset_observation() };
}

fn last_sync_stream_value() -> usize {
    let mut out = 0usize;
    let status = unsafe { test_support::cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last synchronized stream");
    out
}

fn reset_event_observation() {
    unsafe { test_support::cudf_test_event_reset_observation() };
}

fn observed_stream_wait_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { test_support::cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream wait observation count");
    out
}

fn test_handle() -> ResourceHandle {
    let runtime = explicit_runtime_owner();
    runtime_bound_handle(&runtime)
}

struct LocalProducerOwner {
    runtime: Arc<cudf::Runtime>,
    dropped: Arc<AtomicBool>,
}

impl LocalProducerOwner {
    fn same_binding_as(runtime: &Arc<cudf::Runtime>) -> (Self, Arc<AtomicBool>) {
        let dropped = Arc::new(AtomicBool::new(false));
        let producer_runtime =
            Arc::new(runtime.with_retained_lifetime(Arc::new(())).expect("same-binding producer runtime derivation"));
        assert!(!Arc::ptr_eq(runtime, &producer_runtime));
        assert_eq!(runtime.binding_id(), producer_runtime.binding_id());
        (Self { runtime: producer_runtime, dropped: Arc::clone(&dropped) }, dropped)
    }
}

impl Drop for LocalProducerOwner {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
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

fn sssp_distance_map_i32_f32(handle: &cugraph::EnteredContext<'_>, result: &SsspResult) -> HashMap<i32, f32> {
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let distances = result.distances().expect("distances").to_vec_f32(handle).expect("distances");
    vertices.into_iter().zip(distances).collect()
}

fn assert_f32_maps_close(actual: &HashMap<i32, f32>, expected: &HashMap<i32, f32>, tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} entry count mismatch");
    for (vertex, expected_distance) in expected {
        let actual_distance = actual.get(vertex).expect("vertex entry");
        assert!(
            (actual_distance - expected_distance).abs() <= tolerance,
            "{label} mismatch for vertex {vertex}: expected {expected_distance}, got {actual_distance}"
        );
    }
}

fn normalized_wcc_labels(
    handle: &cugraph::EnteredContext<'_>,
    result: &WeaklyConnectedComponentsResult,
) -> BTreeMap<i32, usize> {
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let labels = result.labels().expect("labels").to_vec_i32(handle).expect("labels");
    let mut label_ids = BTreeMap::<i32, usize>::new();
    let mut next_id = 0usize;
    vertices
        .into_iter()
        .zip(labels)
        .map(|(vertex, label)| {
            let id = *label_ids.entry(label).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            (vertex, id)
        })
        .collect()
}

#[test]
fn runtime_bound_handle_same_provenance_borrowed_columns_plan_has_no_host_fence() {
    let _guard = common::cudf::cuda_test_serial_guard();
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
    let _guard = common::cudf::cuda_test_serial_guard();
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
    let _guard = common::cudf::cuda_test_serial_guard();
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
fn graph_built_on_one_runtime_rejects_forward_algorithms_on_different_binding() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let build_runtime = explicit_runtime_owner();
    let build_handle = runtime_bound_handle(&build_runtime);
    let build_entered = build_handle
        .enter("graph_built_on_one_runtime_runs_forward_algorithms_on_same_device_runtime_build")
        .expect("build entered");

    let src = build_runtime.series(SSSP_GRAPH_SRC_I32).expect("src");
    let dst = build_runtime.series(SSSP_GRAPH_DST_I32).expect("dst");
    let weight = build_runtime.series(SSSP_GRAPH_WEIGHT_F32).expect("weight");
    let opts = SeriesGraphOptions::default().renumber();
    let mut graph =
        graph_from_series::<Forward>(&build_entered, &src, &dst, Some(&weight), &opts).expect("build graph");

    drop(build_entered);
    let build_device = build_handle.device();

    {
        let consumer_runtime = explicit_runtime_owner();
        let consumer_handle = runtime_bound_handle(&consumer_runtime);
        let consumer_entered = consumer_handle
            .enter("graph_built_on_one_runtime_runs_forward_algorithms_on_same_device_runtime_consume")
            .expect("consumer entered");
        assert_eq!(build_device, consumer_handle.device());

        let consumer_sources =
            DeviceArray::from_slice(&consumer_entered, &[BFS_SOURCE_VERTEX_I32]).expect("consumer sources");
        let consumer_sources_view = consumer_sources.view().expect("consumer sources view");
        let error = graph
            .bfs(&consumer_entered, &consumer_sources_view, &BfsParams::default())
            .expect_err("different execution binding must fail before BFS graph work");
        assert_eq!(error.kind(), cugraph::ErrorKind::InvalidState);
        assert_eq!(error.operation(), "bfs");
    }

    drop(graph);
}

#[test]
fn graph_built_on_one_runtime_rejects_transposed_algorithms_on_different_binding() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let build_runtime = explicit_runtime_owner();
    let build_handle = runtime_bound_handle(&build_runtime);
    let build_entered = build_handle
        .enter("graph_built_on_one_runtime_runs_transposed_algorithms_on_same_device_runtime_build")
        .expect("build entered");

    let src = build_runtime.series(DISCONNECTED_TWO_CYCLES_SRC_I32).expect("src");
    let dst = build_runtime.series(DISCONNECTED_TWO_CYCLES_DST_I32).expect("dst");
    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph = graph_from_series::<Transposed>(&build_entered, &src, &dst, None, &opts).expect("build graph");

    drop(build_entered);
    let build_device = build_handle.device();

    {
        let consumer_runtime = explicit_runtime_owner();
        let consumer_handle = runtime_bound_handle(&consumer_runtime);
        let consumer_entered = consumer_handle
            .enter("graph_built_on_one_runtime_runs_transposed_algorithms_on_same_device_runtime_consume")
            .expect("consumer entered");
        assert_eq!(build_device, consumer_handle.device());

        let error = graph
            .weakly_connected_components(&consumer_entered, &ConnectedComponentsParams::default())
            .expect_err("different execution binding must fail before WCC graph work");
        assert_eq!(error.kind(), cugraph::ErrorKind::InvalidState);
        assert_eq!(error.operation(), "weakly_connected_components");
    }

    drop(graph);
}

#[test]
fn parked_forward_graph_resumes_on_another_thread_with_same_runtime() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let entered =
        handle.enter("parked_forward_graph_resumes_on_another_thread_with_same_runtime_build").expect("build entered");

    let src = runtime.series(SSSP_GRAPH_SRC_I32).expect("src");
    let dst = runtime.series(SSSP_GRAPH_DST_I32).expect("dst");
    let weight = runtime.series(SSSP_GRAPH_WEIGHT_F32).expect("weight");
    let opts = SeriesGraphOptions::default().renumber();
    let mut graph = graph_from_series::<Forward>(&entered, &src, &dst, Some(&weight), &opts).expect("build graph");

    let sources = DeviceArray::from_slice(&entered, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources view");
    let expected_bfs = graph
        .bfs(&entered, &sources_view, &BfsParams::default())
        .map(|result| bfs_distance_map_i32_result(&entered, &result))
        .expect("baseline bfs");
    let sssp_params = SsspParams::default().with_cutoff(NonNegativeF64::new(10.0).expect("cutoff"));
    let expected_sssp = graph
        .sssp(&entered, 0i32, &sssp_params)
        .map(|result| sssp_distance_map_i32_f32(&entered, &result))
        .expect("baseline sssp");

    let parked = graph.park(&entered).expect("park graph");
    drop(entered);
    drop(handle);

    let resume_runtime = Arc::clone(&runtime);
    let join = thread::spawn(move || {
        let handle = runtime_bound_handle(&resume_runtime);
        let entered = handle
            .enter("parked_forward_graph_resumes_on_another_thread_with_same_runtime_resume")
            .expect("resume entered");
        let mut graph = parked.resume(&entered).expect("resume graph");
        let sources = DeviceArray::from_slice(&entered, &[BFS_SOURCE_VERTEX_I32]).expect("resume sources");
        let sources_view = sources.view().expect("resume sources view");
        let actual_bfs = graph
            .bfs(&entered, &sources_view, &BfsParams::default())
            .map(|result| bfs_distance_map_i32_result(&entered, &result))
            .expect("resumed bfs");
        let sssp_params = SsspParams::default().with_cutoff(NonNegativeF64::new(10.0).expect("cutoff"));
        let actual_sssp = graph
            .sssp(&entered, 0i32, &sssp_params)
            .map(|result| sssp_distance_map_i32_f32(&entered, &result))
            .expect("resumed sssp");
        (actual_bfs, actual_sssp)
    });
    let (actual_bfs, actual_sssp) = join.join().expect("resume thread");

    assert_eq!(actual_bfs, expected_bfs);
    assert_f32_maps_close(&actual_sssp, &expected_sssp, 1.0e-5, "resumed sssp");
}

#[test]
fn parked_transposed_graph_resumes_on_another_thread_with_same_runtime() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let entered = handle
        .enter("parked_transposed_graph_resumes_on_another_thread_with_same_runtime_build")
        .expect("build entered");

    let src = runtime.series(DISCONNECTED_TWO_CYCLES_SRC_I32).expect("src");
    let dst = runtime.series(DISCONNECTED_TWO_CYCLES_DST_I32).expect("dst");
    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph = graph_from_series::<Transposed>(&entered, &src, &dst, None, &opts).expect("build graph");

    let expected_pagerank = common::cudf::pagerank_score_map(&entered, &mut graph);
    let expected_wcc = graph
        .weakly_connected_components(&entered, &ConnectedComponentsParams::default())
        .map(|result| normalized_wcc_labels(&entered, &result))
        .expect("baseline wcc");

    let parked = graph.park(&entered).expect("park graph");
    drop(entered);
    drop(handle);

    let resume_runtime = Arc::clone(&runtime);
    let join = thread::spawn(move || {
        let handle = runtime_bound_handle(&resume_runtime);
        let entered = handle
            .enter("parked_transposed_graph_resumes_on_another_thread_with_same_runtime_resume")
            .expect("resume entered");
        let mut graph = parked.resume(&entered).expect("resume graph");
        let actual_pagerank = common::cudf::pagerank_score_map(&entered, &mut graph);
        let actual_wcc = graph
            .weakly_connected_components(&entered, &ConnectedComponentsParams::default())
            .map(|result| normalized_wcc_labels(&entered, &result))
            .expect("resumed wcc");
        (actual_pagerank, actual_wcc)
    });
    let (actual_pagerank, actual_wcc) = join.join().expect("resume thread");

    assert_score_maps_close_f32(&actual_pagerank, &expected_pagerank, 1.0e-6, "resumed pagerank");
    assert_eq!(actual_wcc, expected_wcc);
}

#[test]
fn parked_graph_resume_rejects_different_runtime_owner() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let entered = handle.enter("parked_graph_resume_rejects_different_runtime_owner_build").expect("entered");

    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let opts = SeriesGraphOptions::default().renumber();
    let graph = graph_from_series::<Forward>(&entered, &src, &dst, None, &opts).expect("build graph");
    let parked = graph.park(&entered).expect("park graph");

    let other_runtime = explicit_runtime_owner();
    let other_handle = runtime_bound_handle(&other_runtime);
    let other_entered =
        other_handle.enter("parked_graph_resume_rejects_different_runtime_owner_resume").expect("other entered");

    let err = parked.resume(&other_entered).expect_err("resume should reject different runtime owner");
    assert_eq!(err.kind(), cugraph::ErrorKind::InvalidState);
    assert_eq!(err.operation(), "ParkedGraph::resume");
}

#[test]
fn parked_graph_drops_on_another_thread_without_resume() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let entered = handle.enter("parked_graph_drops_on_another_thread_without_resume_build").expect("entered");

    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let opts = SeriesGraphOptions::default().renumber();
    let graph = graph_from_series::<Forward>(&entered, &src, &dst, None, &opts).expect("build graph");
    let parked = graph.park(&entered).expect("park graph");
    drop(entered);
    drop(handle);

    thread::spawn(move || drop(parked)).join().expect("drop thread");
}

#[test]
fn graph_use_rejects_different_runtime_binding_before_park() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let build_runtime = explicit_runtime_owner();
    let build_handle = runtime_bound_handle(&build_runtime);
    let build_entered =
        build_handle.enter("graph_park_rejects_graph_used_by_different_runtime_owner_build").expect("build entered");

    let src = build_runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = build_runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let opts = SeriesGraphOptions::default().renumber();
    let mut graph = graph_from_series::<Forward>(&build_entered, &src, &dst, None, &opts).expect("build graph");

    {
        let other_runtime = explicit_runtime_owner();
        let other_handle = runtime_bound_handle(&other_runtime);
        let other_entered = other_handle
            .enter("graph_park_rejects_graph_used_by_different_runtime_owner_touch")
            .expect("other entered");
        let sources = DeviceArray::from_slice(&other_entered, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
        let sources_view = sources.view().expect("sources view");
        let error = graph
            .bfs(&other_entered, &sources_view, &BfsParams::default())
            .expect_err("different execution binding must fail before BFS graph work");
        assert_eq!(error.kind(), cugraph::ErrorKind::InvalidState);
        assert_eq!(error.operation(), "bfs");
    }

    graph.park(&build_entered).expect("rejected use must not taint exact-binding park");
}

#[test]
fn result_built_on_one_runtime_decodes_after_producer_runtime_drops() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let build_runtime = explicit_runtime_owner();
    let build_handle = runtime_bound_handle(&build_runtime);
    let build_entered = build_handle
        .enter("result_built_on_one_runtime_decodes_after_producer_runtime_drops_build")
        .expect("build entered");

    let src = build_runtime.series(DISCONNECTED_TWO_CYCLES_SRC_I32).expect("src");
    let dst = build_runtime.series(DISCONNECTED_TWO_CYCLES_DST_I32).expect("dst");
    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph = graph_from_series::<Transposed>(&build_entered, &src, &dst, None, &opts).expect("build graph");

    drop(build_entered);
    let build_device = build_handle.device();

    let producer_dropped;
    let result = {
        let (producer_owner, dropped) = LocalProducerOwner::same_binding_as(&build_runtime);
        producer_dropped = dropped;
        let producer_handle = runtime_bound_handle(&producer_owner.runtime);
        let producer_entered = producer_handle
            .enter("result_built_on_one_runtime_decodes_after_producer_runtime_drops_produce")
            .expect("producer entered");
        assert_eq!(build_device, producer_handle.device());

        graph.pagerank(&producer_entered, &PageRankParams::default()).expect("pagerank")
    };
    assert!(producer_dropped.load(Ordering::SeqCst), "the local producer owner must drop after submission");

    drop(graph);

    let consumer_runtime = explicit_runtime_owner();
    let consumer_handle = runtime_bound_handle(&consumer_runtime);
    let consumer_entered = consumer_handle
        .enter("result_built_on_one_runtime_decodes_after_producer_runtime_drops_decode")
        .expect("consumer entered");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&consumer_entered).expect("vertices");
    let scores = result.scores().expect("scores").to_vec_f32(&consumer_entered).expect("scores");
    assert_eq!(vertices.len(), scores.len());
    assert!(!vertices.is_empty());

    drop(result);
}

#[test]
fn path_result_built_on_one_runtime_decodes_after_producer_runtime_drops() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let build_runtime = explicit_runtime_owner();
    let build_handle = runtime_bound_handle(&build_runtime);
    let build_entered = build_handle
        .enter("path_result_built_on_one_runtime_decodes_after_producer_runtime_drops_build")
        .expect("build entered");

    let src = build_runtime.series(SSSP_GRAPH_SRC_I32).expect("src");
    let dst = build_runtime.series(SSSP_GRAPH_DST_I32).expect("dst");
    let weight = build_runtime.series(SSSP_GRAPH_WEIGHT_F32).expect("weight");
    let opts = SeriesGraphOptions::default().renumber();
    let mut graph =
        graph_from_series::<Forward>(&build_entered, &src, &dst, Some(&weight), &opts).expect("build graph");

    drop(build_entered);
    let build_device = build_handle.device();

    let producer_dropped;
    let result = {
        let (producer_owner, dropped) = LocalProducerOwner::same_binding_as(&build_runtime);
        producer_dropped = dropped;
        let producer_handle = runtime_bound_handle(&producer_owner.runtime);
        let producer_entered = producer_handle
            .enter("path_result_built_on_one_runtime_decodes_after_producer_runtime_drops_produce")
            .expect("producer entered");
        assert_eq!(build_device, producer_handle.device());

        let sources = DeviceArray::from_slice(&producer_entered, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
        let sources_view = sources.view().expect("sources view");
        graph.bfs(&producer_entered, &sources_view, &BfsParams::default()).expect("bfs")
    };
    assert!(producer_dropped.load(Ordering::SeqCst), "the local producer owner must drop after submission");

    drop(graph);

    let consumer_runtime = explicit_runtime_owner();
    let consumer_handle = runtime_bound_handle(&consumer_runtime);
    let consumer_entered = consumer_handle
        .enter("path_result_built_on_one_runtime_decodes_after_producer_runtime_drops_decode")
        .expect("consumer entered");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&consumer_entered).expect("vertices");
    let distances = result.distances().expect("distances").to_vec_i32(&consumer_entered).expect("distances");
    assert_eq!(vertices.len(), distances.len());
    assert!(!vertices.is_empty());

    drop(result);
}

#[test]
fn graph_from_series_allows_mixed_explicit_runtimes_on_same_device() {
    let _guard = common::cudf::cuda_test_serial_guard();
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
    let _guard = common::cudf::cuda_test_serial_guard();
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
    let _guard = common::cudf::cuda_test_serial_guard();
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
    let _guard = common::cudf::cuda_test_serial_guard();
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
    let _guard = common::cudf::cuda_test_serial_guard();
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
    let _guard = common::cudf::cuda_test_serial_guard();
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
