//! Integration tests for prepared-device exports and their fallback behavior.
mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use common::{
    cudf::{
        assert_named_edge_columns, build_undirected_named_dataframe_graph, build_undirected_prepared_graph,
        explicit_runtime, pagerank_score_map,
    },
    support::{
        BFS_SOURCE_VERTEX_I32, DIRECTED_TRIANGLE_CYCLE_DST_I32, DIRECTED_TRIANGLE_CYCLE_SRC_I32, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, TRIANGLE_TAIL_WEIGHT_F32, bfs_distance_map_i32_result, triangle_graph_dataframe,
    },
};
use cugraph::{
    array::DeviceArray,
    cudf_interop::{
        SeriesGraphOptions, TableGraphOptions, graph_from_prepared_columns, graph_from_prepared_table_export,
    },
    error::ErrorKind,
    graph::Forward,
    handle::ResourceHandle,
    traversal::BfsParams,
};
use rapids_interop::{
    ActiveLeaseBackend, CompletionBinding, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation,
    LeaseBackend, LeaseCompletionResult, LeaseExportContext, ReadStartBinding,
};

fn test_handle() -> ResourceHandle {
    let runtime = explicit_runtime();
    ResourceHandle::for_runtime_device(&runtime).expect("handle")
}

#[derive(Clone)]
struct CountingLeaseBackend {
    complete_calls: Arc<AtomicUsize>,
    drop_incomplete_calls: Arc<AtomicUsize>,
    incomplete_drop_behavior: IncompleteDropBehavior,
}

impl ActiveLeaseBackend for CountingLeaseBackend {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.incomplete_drop_behavior
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseCompletionResult> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<()> {
        self.drop_incomplete_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Clone)]
struct FailingActivationBackend;

impl ActiveLeaseBackend for FailingActivationBackend {
    fn activate(
        &self,
        _export: &LeaseExportContext,
        _read_start: ReadStartBinding,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseActivation> {
        Err(InteropError::new(InteropErrorKind::InvalidView, "test backend forced activation failure"))
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        IncompleteDropBehavior::GuaranteedFallback
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseCompletionResult> {
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<()> {
        Ok(())
    }
}

#[test]
fn graph_from_prepared_columns_i32_bfs() {
    let handle = test_handle();
    let runtime = explicit_runtime();

    let src = runtime.series("src", TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series("dst", TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_export = src.try_prepare_device_column_export().expect("src export");
    let dst_export = dst.try_prepare_device_column_export().expect("dst export");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph =
        graph_from_prepared_columns::<Forward>(&handle, &src_export, &dst_export, None, &opts).expect("prepared graph");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let dist_map = bfs_distance_map_i32_result(&handle, &result);
    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], 1);
    assert_eq!(dist_map[&2], 2);
    assert_eq!(dist_map[&3], 3);
}

#[test]
fn graph_from_prepared_columns_completes_earlier_leases_when_later_activation_fails() {
    let handle = test_handle();
    let runtime = explicit_runtime();

    let src = runtime.series("src", DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = runtime.series("dst", DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");

    let complete_calls = Arc::new(AtomicUsize::new(0));
    let drop_incomplete_calls = Arc::new(AtomicUsize::new(0));
    let src_export = src.try_prepare_device_column_export().expect("src export").with_lease_backend(LeaseBackend::new(
        CountingLeaseBackend {
            complete_calls: Arc::clone(&complete_calls),
            drop_incomplete_calls: Arc::clone(&drop_incomplete_calls),
            incomplete_drop_behavior: IncompleteDropBehavior::ExplicitCompletionRequired,
        },
    ));
    let dst_export = dst
        .try_prepare_device_column_export()
        .expect("dst export")
        .with_lease_backend(LeaseBackend::new(FailingActivationBackend));

    let opts = SeriesGraphOptions::default().renumber();
    let err = graph_from_prepared_columns::<Forward>(&handle, &src_export, &dst_export, None, &opts)
        .expect_err("second activation should fail");

    assert_eq!(err.kind(), ErrorKind::InvalidState);
    assert_eq!(complete_calls.load(Ordering::SeqCst), 1, "src lease should be completed explicitly");
    assert_eq!(
        drop_incomplete_calls.load(Ordering::SeqCst),
        0,
        "explicit completion should avoid relying on incomplete-drop behavior"
    );
    assert!(err.message().contains("forced activation failure"), "error should preserve activation failure context");
}

#[test]
fn graph_from_prepared_table_export_bfs() {
    let handle = test_handle();
    let runtime = explicit_runtime();

    let src = runtime.series("src", TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series("dst", TRIANGLE_TAIL_DST_I32).expect("dst");
    let wgt = runtime.series("weight", TRIANGLE_TAIL_WEIGHT_F32).expect("weight");
    let df = cudf::DataFrame::from_series([src, dst, wgt]).expect("dataframe");

    let export = df.try_prepare_device_table_export().expect("table export");
    let opts = TableGraphOptions::new(0, 1).weight_col(2).renumber();
    let mut graph = graph_from_prepared_table_export::<Forward>(&handle, &export, &opts).expect("prepared table graph");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let dist_map = bfs_distance_map_i32_result(&handle, &result);
    assert_eq!(dist_map[&3], 3);
}

#[test]
fn graph_from_prepared_columns_rejects_host_exports() {
    let handle = test_handle();
    let runtime = explicit_runtime();

    let src = runtime.series("src", DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = runtime.series("dst", DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");

    let src_export = src.try_prepare_host_column_export().expect("src host export");
    let dst_export = dst.try_prepare_host_column_export().expect("dst host export");

    let opts = SeriesGraphOptions::default().renumber();
    let err = graph_from_prepared_columns::<Forward>(&handle, &src_export, &dst_export, None, &opts)
        .expect_err("host prepared export should be rejected");

    assert_eq!(err.kind(), ErrorKind::UnsupportedType);
    assert!(
        err.message().contains("prepared device export"),
        "expected prepared-device-only error, got: {}",
        err.message()
    );
}

#[test]
fn graph_from_prepared_table_export_rejects_host_exports() {
    let handle = test_handle();
    let runtime = explicit_runtime();

    let src = runtime.series("src", DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = runtime.series("dst", DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");
    let df = cudf::DataFrame::from_series([src, dst]).expect("dataframe");

    let export = df.try_prepare_host_table_export().expect("host table export");
    let opts = TableGraphOptions::new(0, 1).renumber();
    let err = graph_from_prepared_table_export::<Forward>(&handle, &export, &opts)
        .expect_err("host prepared table export should be rejected");

    assert_eq!(err.kind(), ErrorKind::UnsupportedType);
    assert!(
        err.message().contains("prepared device export inputs"),
        "expected prepared-device-only error, got: {}",
        err.message()
    );
}

#[test]
fn prepared_export_vs_deep_copy_import_pagerank_equivalence() {
    let handle = test_handle();
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = triangle_graph_dataframe(&producer_runtime);
    let export = producer_df.try_prepare_device_table_export().expect("prepared export");
    let mut prepared_graph = build_undirected_prepared_graph(&handle, &export).expect("prepared graph");

    let imported_df = cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&export, &consumer_runtime)
        .expect("deep-copy import");
    assert_named_edge_columns(&imported_df);
    let mut fallback_graph = build_undirected_named_dataframe_graph(&handle, &imported_df).expect("fallback graph");

    let prepared_map = pagerank_score_map(&handle, &mut prepared_graph);
    let fallback_map = pagerank_score_map(&handle, &mut fallback_graph);

    assert_eq!(prepared_map.len(), fallback_map.len(), "vertex coverage should match");
    for (vertex, prepared_score) in prepared_map {
        let fallback_score = fallback_map.get(&vertex).expect("fallback result should contain every prepared vertex");
        assert!(
            (prepared_score - fallback_score).abs() <= 1.0e-6,
            "vertex {} score mismatch: prepared={} fallback={}",
            vertex,
            prepared_score,
            fallback_score
        );
    }
}

#[test]
fn prepared_host_export_import_pagerank_matches_prepared_device_path() {
    let handle = test_handle();
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = triangle_graph_dataframe(&producer_runtime);
    let prepared_export = producer_df.try_prepare_device_table_export().expect("prepared device export");
    let mut prepared_graph = build_undirected_prepared_graph(&handle, &prepared_export).expect("prepared graph");

    let host_export = producer_df.try_prepare_host_table_export().expect("prepared host export");
    let imported_df =
        cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&host_export, &consumer_runtime)
            .expect("host fallback import");
    assert_named_edge_columns(&imported_df);
    let mut fallback_graph = build_undirected_named_dataframe_graph(&handle, &imported_df).expect("fallback graph");

    let prepared_map = pagerank_score_map(&handle, &mut prepared_graph);
    let fallback_map = pagerank_score_map(&handle, &mut fallback_graph);

    assert_eq!(prepared_map.len(), fallback_map.len(), "vertex coverage should match");
    for (vertex, prepared_score) in prepared_map {
        let fallback_score = fallback_map.get(&vertex).expect("fallback result should contain every prepared vertex");
        assert!(
            (prepared_score - fallback_score).abs() <= 1.0e-6,
            "vertex {} score mismatch: prepared={} fallback={}",
            vertex,
            prepared_score,
            fallback_score
        );
    }
}
