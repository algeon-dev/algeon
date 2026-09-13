//! Integration tests for prepared-device exports and their fallback behavior.
mod common;

use std::sync::{
    Arc, Mutex, MutexGuard, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use common::{
    cudf::{
        assert_named_edge_columns, assert_score_maps_close_f32, build_undirected_named_dataframe_graph,
        build_undirected_prepared_graph, explicit_runtime, explicit_runtime_owner, pagerank_score_map,
        python_undirected_multigraph_dataframe, python_weighted_reciprocal_expected_pagerank_map, runtime_bound_handle,
    },
    support::{
        BFS_SOURCE_VERTEX_I32, DIRECTED_TRIANGLE_CYCLE_DST_I32, DIRECTED_TRIANGLE_CYCLE_SRC_I32, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, TRIANGLE_TAIL_WEIGHT_F32, bfs_distance_map_i32_result, bfs_distance_map_i64_result,
        triangle_graph_dataframe,
    },
};
use cudf::{
    interop::contract::{
        ActiveLeaseBackend, CompletionBinding, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation,
        LeaseBackend, LeaseCompletionResult, LeaseExportContext, ReadStartBinding,
    },
    testing::prelude::*,
};
use cudf_sys::test_support;
use cugraph::{
    array::DeviceArray,
    cudf_interop::{
        GraphConstructionPolicy, PreparedGraphOptions, TableGraphOptions, TableVertexPairOptions,
        graph_from_prepared_columns, graph_from_prepared_table_export, vertex_pairs_from_prepared_columns,
        vertex_pairs_from_prepared_table_export,
    },
    error::ErrorKind,
    graph::{Forward, Graph, GraphBuilder},
    handle::ResourceHandle,
    traversal::{BfsParams, BfsPredicateParams},
    types::GraphProperties,
};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(explicit_runtime());
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

fn pair_test_graph(handle: &cugraph::EnteredContext<'_>) -> Graph<Forward> {
    let src = DeviceArray::from_slice(handle, &[0i32, 1, 2]).expect("pair graph src");
    let dst = DeviceArray::from_slice(handle, &[1i32, 2, 0]).expect("pair graph dst");
    let src = src.view().expect("pair graph src view");
    let dst = dst.view().expect("pair graph dst view");
    GraphBuilder::new(&src, &dst)
        .properties(GraphProperties::undirected())
        .build::<Forward>(handle)
        .expect("pair graph")
}

fn reset_event_observation() {
    unsafe { test_support::cudf_test_event_reset_observation() };
}

fn event_synchronize_call_count() -> i32 {
    let mut out = -1;
    let status = unsafe { test_support::cudf_test_event_synchronize_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read event synchronize observation count");
    out
}

fn lock_event_observation() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    ) -> cudf::interop::contract::Result<LeaseCompletionResult> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> cudf::interop::contract::Result<()> {
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
    ) -> cudf::interop::contract::Result<LeaseActivation> {
        Err(InteropError::new(InteropErrorKind::InvalidView, "test backend forced activation failure"))
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        IncompleteDropBehavior::GuaranteedFallback
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> cudf::interop::contract::Result<LeaseCompletionResult> {
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> cudf::interop::contract::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordedReadStartBinding {
    SynchronousByReturn,
    ConsumerStream { raw: usize, device: i32 },
}

impl From<ReadStartBinding> for RecordedReadStartBinding {
    fn from(value: ReadStartBinding) -> Self {
        match value {
            ReadStartBinding::SynchronousByReturn => Self::SynchronousByReturn,
            ReadStartBinding::ConsumerStream { raw, device } => Self::ConsumerStream { raw: raw as usize, device },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordedCompletionBinding {
    SynchronousByReturn,
    SameDeviceStream { raw: usize, device: i32 },
    CrossDeviceEvent { raw: usize, device: i32 },
}

impl From<CompletionBinding> for RecordedCompletionBinding {
    fn from(value: CompletionBinding) -> Self {
        match value {
            CompletionBinding::SynchronousByReturn => Self::SynchronousByReturn,
            CompletionBinding::SameDeviceStream { raw, device } => Self::SameDeviceStream { raw: raw as usize, device },
            CompletionBinding::CrossDeviceEvent { raw, device } => Self::CrossDeviceEvent { raw: raw as usize, device },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordedLeaseBindings {
    activate_read_start: RecordedReadStartBinding,
    activate_completion: RecordedCompletionBinding,
    complete_completion: RecordedCompletionBinding,
}

#[derive(Clone)]
struct RecordingLeaseBackend {
    recorded: Arc<Mutex<Vec<RecordedLeaseBindings>>>,
}

impl ActiveLeaseBackend for RecordingLeaseBackend {
    fn activate(
        &self,
        _export: &LeaseExportContext,
        read_start: ReadStartBinding,
        completion: CompletionBinding,
    ) -> cudf::interop::contract::Result<LeaseActivation> {
        self.recorded.lock().expect("recorded").push(RecordedLeaseBindings {
            activate_read_start: read_start.into(),
            activate_completion: completion.into(),
            complete_completion: RecordedCompletionBinding::SynchronousByReturn,
        });
        Ok(LeaseActivation::noop())
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        IncompleteDropBehavior::GuaranteedFallback
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> cudf::interop::contract::Result<LeaseCompletionResult> {
        let mut recorded = self.recorded.lock().expect("recorded");
        let last = recorded.last_mut().expect("activate should run before complete");
        last.complete_completion = completion.into();
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> cudf::interop::contract::Result<()> {
        Ok(())
    }
}

#[test]
fn vertex_pairs_from_prepared_columns_and_table_export_preserve_selected_pair_rows() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle
        .enter("vertex_pairs_from_prepared_columns_and_table_export_preserve_selected_pair_rows")
        .expect("entered");
    let runtime = explicit_runtime();
    let mut graph = pair_test_graph(&handle);

    let first = runtime.series(&[0i32, 0, 1, 2]).expect("first");
    let second = runtime.series(&[1i32, 2, 2, 0]).expect("second");
    let first_export = first.try_prepare_device_column_export().expect("first export");
    let second_export = second.try_prepare_device_column_export().expect("second export");

    let pairs = vertex_pairs_from_prepared_columns(&handle, &mut graph, &first_export, &second_export)
        .expect("prepared column pairs");
    assert_eq!(pairs.len().expect("pair length"), 4);
    assert_eq!(pairs.first().expect("first pairs").to_vec_i32(&handle).expect("first copy"), vec![0, 0, 1, 2]);
    assert_eq!(pairs.second().expect("second pairs").to_vec_i32(&handle).expect("second copy"), vec![1, 2, 2, 0]);

    let ignored = runtime.series(&[9i32, 9, 9, 9]).expect("ignored");
    let pair_table =
        cudf::DataFrame::from_series([("ignored", ignored), ("second", second), ("first", first)]).expect("pair table");
    let table_export = pair_table.try_prepare_device_table_export().expect("table export");
    let pairs =
        vertex_pairs_from_prepared_table_export(&handle, &mut graph, &table_export, &TableVertexPairOptions::new(2, 1))
            .expect("prepared table pairs");
    assert_eq!(pairs.first().expect("first pairs").to_vec_i32(&handle).expect("first copy"), vec![0, 0, 1, 2]);
    assert_eq!(pairs.second().expect("second pairs").to_vec_i32(&handle).expect("second copy"), vec![1, 2, 2, 0]);
}

#[test]
fn vertex_pairs_from_prepared_columns_rejects_mismatched_dtype_row_count_and_nulls() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle
        .enter("vertex_pairs_from_prepared_columns_rejects_mismatched_dtype_row_count_and_nulls")
        .expect("entered");
    let runtime = explicit_runtime();
    let mut graph = pair_test_graph(&handle);

    let first = runtime.series(&[0i32, 1]).expect("first");
    let second_i64 = runtime.series(&[1i64, 2]).expect("second i64");
    let first_export = first.try_prepare_device_column_export().expect("first export");
    let second_i64_export = second_i64.try_prepare_device_column_export().expect("second i64 export");
    let err = vertex_pairs_from_prepared_columns(&handle, &mut graph, &first_export, &second_i64_export)
        .expect_err("mismatched pair dtypes must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.to_string().contains("first_dtype"), "error should retain first dtype context: {err}");
    assert!(err.to_string().contains("second_dtype"), "error should retain second dtype context: {err}");

    let second_short = runtime.series(&[1i32]).expect("second short");
    let second_short_export = second_short.try_prepare_device_column_export().expect("second short export");
    let err = vertex_pairs_from_prepared_columns(&handle, &mut graph, &first_export, &second_short_export)
        .expect_err("mismatched pair row counts must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.to_string().contains("first_len"), "error should retain first length context: {err}");
    assert!(err.to_string().contains("second_len"), "error should retain second length context: {err}");

    let second_with_null = runtime.nullable_series(&[1i32, 0], &[true, false]).expect("second with null");
    let second_with_null_export = second_with_null.try_prepare_device_column_export().expect("second null export");
    let err = vertex_pairs_from_prepared_columns(&handle, &mut graph, &first_export, &second_with_null_export)
        .expect_err("null pair values must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.to_string().contains("column_name: second"), "error should identify second pair column: {err}");
    assert!(err.to_string().contains("null_count: 1"), "error should retain null count: {err}");
}

#[test]
fn vertex_pairs_from_prepared_columns_runtime_bound_handle_uses_consumer_stream_authorities() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle
        .enter("vertex_pairs_from_prepared_columns_runtime_bound_handle_uses_consumer_stream_authorities")
        .expect("entered");
    let mut graph = pair_test_graph(&handle);
    let first = runtime.series(&[0i32, 1]).expect("first");
    let second = runtime.series(&[1i32, 2]).expect("second");
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let first_export = first
        .try_prepare_device_column_export()
        .expect("first export")
        .with_lease_backend(LeaseBackend::new(RecordingLeaseBackend { recorded: Arc::clone(&recorded) }));
    let second_export = second.try_prepare_device_column_export().expect("second export");

    let _pairs =
        vertex_pairs_from_prepared_columns(&handle, &mut graph, &first_export, &second_export).expect("prepared pairs");

    let expected_stream_raw = runtime.stream().testing_raw_handle_value() as *mut std::ffi::c_void;
    let expected_device = runtime.device().ordinal() as i32;
    let recorded = recorded.lock().expect("recorded");
    assert_eq!(recorded.len(), 1, "first export should record one activation/completion pair");
    assert_eq!(
        recorded[0],
        RecordedLeaseBindings {
            activate_read_start: RecordedReadStartBinding::ConsumerStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
            activate_completion: RecordedCompletionBinding::SameDeviceStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
            complete_completion: RecordedCompletionBinding::SameDeviceStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
        }
    );
}

#[test]
fn vertex_pairs_from_prepared_columns_runtime_bound_handle_defers_completion_sync_until_pair_drop() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle = handle
        .enter("vertex_pairs_from_prepared_columns_runtime_bound_handle_defers_completion_sync_until_pair_drop")
        .expect("entered");
    let mut graph = pair_test_graph(&handle);
    let first = runtime.series(&[0i32, 1]).expect("first");
    let second = runtime.series(&[1i32, 2]).expect("second");
    let first_export = first.try_prepare_device_column_export().expect("first export");
    let second_export = second.try_prepare_device_column_export().expect("second export");
    assert!(first_export.clone_retained_owner().is_some(), "first export should retain its producer owner");
    assert!(second_export.clone_retained_owner().is_some(), "second export should retain its producer owner");

    reset_event_observation();
    let _ = event_synchronize_call_count();

    let pairs =
        vertex_pairs_from_prepared_columns(&handle, &mut graph, &first_export, &second_export).expect("prepared pairs");
    let before_drop = event_synchronize_call_count();
    drop(pairs);
    let after_drop = event_synchronize_call_count();

    assert!(
        after_drop > before_drop,
        "dropping vertex pairs should synchronize deferred prepared-column completion witnesses"
    );
}

#[test]
fn graph_from_prepared_columns_i32_bfs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle.enter("graph_from_prepared_columns_i32_bfs").expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_export = src.try_prepare_device_column_export().expect("src export");
    let dst_export = dst.try_prepare_device_column_export().expect("dst export");

    let opts = PreparedGraphOptions::default().renumber();
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
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle
        .enter("graph_from_prepared_columns_completes_earlier_leases_when_later_activation_fails")
        .expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = runtime.series(DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");

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

    let opts = PreparedGraphOptions::default().renumber();
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
fn graph_from_prepared_columns_runtime_bound_handle_uses_consumer_stream_authorities() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle
        .enter("graph_from_prepared_columns_runtime_bound_handle_uses_consumer_stream_authorities")
        .expect("entered");
    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let recorded = Arc::new(Mutex::new(Vec::new()));

    let src_export = src
        .try_prepare_device_column_export()
        .expect("src export")
        .with_lease_backend(LeaseBackend::new(RecordingLeaseBackend { recorded: Arc::clone(&recorded) }));
    let dst_export = dst.try_prepare_device_column_export().expect("dst export");
    let opts = PreparedGraphOptions::default().renumber();

    let _graph = graph_from_prepared_columns::<Forward>(&handle, &src_export, &dst_export, None, &opts).expect("graph");

    let expected_stream_raw = runtime.stream().testing_raw_handle_value() as *mut std::ffi::c_void;
    let expected_device = runtime.device().ordinal() as i32;
    let recorded = recorded.lock().expect("recorded");
    assert_eq!(recorded.len(), 1, "src export should record one activation/completion pair");
    assert_eq!(
        recorded[0],
        RecordedLeaseBindings {
            activate_read_start: RecordedReadStartBinding::ConsumerStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
            activate_completion: RecordedCompletionBinding::SameDeviceStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
            complete_completion: RecordedCompletionBinding::SameDeviceStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
        }
    );
}

#[test]
fn graph_from_prepared_columns_runtime_bound_handle_defers_completion_sync_until_graph_drop() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle = handle
        .enter("graph_from_prepared_columns_runtime_bound_handle_defers_completion_sync_until_graph_drop")
        .expect("entered");
    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_export = src.try_prepare_device_column_export().expect("src export");
    let dst_export = dst.try_prepare_device_column_export().expect("dst export");
    assert!(src_export.clone_retained_owner().is_some(), "series prepared export should retain its producer owner");
    assert!(dst_export.clone_retained_owner().is_some(), "series prepared export should retain its producer owner");
    let opts = PreparedGraphOptions::default().renumber();

    reset_event_observation();
    let _ = event_synchronize_call_count();

    let graph = graph_from_prepared_columns::<Forward>(&handle, &src_export, &dst_export, None, &opts).expect("graph");
    let before_drop = event_synchronize_call_count();

    drop(graph);
    let after_drop = event_synchronize_call_count();

    assert!(
        after_drop > before_drop,
        "dropping the graph should perform deferred prepared-column completion synchronization"
    );
}

#[test]
fn graph_from_prepared_table_export_bfs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle.enter("graph_from_prepared_table_export_bfs").expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let wgt = runtime.series(TRIANGLE_TAIL_WEIGHT_F32).expect("weight");
    let df = cudf::DataFrame::from_series([("src", src), ("dst", dst), ("weight", wgt)]).expect("dataframe");

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
fn graph_from_prepared_table_export_i32_f32_edge_ids_drive_predicate_bfs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle =
        handle.enter("graph_from_prepared_table_export_i32_f32_edge_ids_drive_predicate_bfs").expect("entered");
    let runtime = explicit_runtime();
    let df = cudf::DataFrame::from_series([
        ("src", runtime.series(&[0i32, 0, 1, 2]).expect("src")),
        ("dst", runtime.series(&[1i32, 2, 3, 3]).expect("dst")),
        ("weight", runtime.series(&[1.0f32, 2.0, 3.0, 4.0]).expect("weight")),
        ("edge_id", runtime.series(&[10i32, 11, 12, 13]).expect("edge id")),
    ])
    .expect("dataframe");
    let export = df.try_prepare_device_table_export().expect("table export");
    let options = TableGraphOptions::new(0, 1)
        .weight_col(2)
        .edge_id_col(3)
        .construction_policy(GraphConstructionPolicy::RawLibcugraph);
    let mut graph = graph_from_prepared_table_export::<Forward>(&handle, &export, &options).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let allowed = DeviceArray::from_slice(&handle, &[11i32, 13]).expect("allowed edge ids");
    let sources = sources.view().expect("sources view");
    let allowed = allowed.view().expect("allowed view");
    let predicates = BfsPredicateParams { include_edge_ids: Some(&allowed), ..BfsPredicateParams::default() };
    let result =
        graph.bfs_with_predicates(&handle, &sources, &BfsParams::default(), &predicates).expect("predicate bfs");
    let distances = bfs_distance_map_i32_result(&handle, &result);
    assert_eq!(distances[&0], 0);
    assert_eq!(distances[&1], i32::MAX);
    assert_eq!(distances[&2], 1);
    assert_eq!(distances[&3], 2);
}

#[test]
fn graph_from_prepared_table_export_i64_f64_edge_ids_drive_predicate_bfs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle =
        handle.enter("graph_from_prepared_table_export_i64_f64_edge_ids_drive_predicate_bfs").expect("entered");
    let runtime = explicit_runtime();
    let df = cudf::DataFrame::from_series([
        ("src", runtime.series(&[0i64, 0, 1, 2]).expect("src")),
        ("dst", runtime.series(&[1i64, 2, 3, 3]).expect("dst")),
        ("weight", runtime.series(&[1.0f64, 2.0, 3.0, 4.0]).expect("weight")),
        ("edge_id", runtime.series(&[10i64, 11, 12, 13]).expect("edge id")),
    ])
    .expect("dataframe");
    let export = df.try_prepare_device_table_export().expect("table export");
    let options = TableGraphOptions::new(0, 1)
        .weight_col(2)
        .edge_id_col(3)
        .construction_policy(GraphConstructionPolicy::RawLibcugraph);
    let mut graph = graph_from_prepared_table_export::<Forward>(&handle, &export, &options).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");
    let allowed = DeviceArray::from_slice(&handle, &[11i64, 13]).expect("allowed edge ids");
    let sources = sources.view().expect("sources view");
    let allowed = allowed.view().expect("allowed view");
    let predicates = BfsPredicateParams { include_edge_ids: Some(&allowed), ..BfsPredicateParams::default() };
    let result =
        graph.bfs_with_predicates(&handle, &sources, &BfsParams::default(), &predicates).expect("predicate bfs");
    let distances = bfs_distance_map_i64_result(&handle, &result);
    assert_eq!(distances[&0], 0);
    assert_eq!(distances[&1], i64::MAX);
    assert_eq!(distances[&2], 1);
    assert_eq!(distances[&3], 2);
}

#[test]
fn graph_from_prepared_table_export_rejects_invalid_edge_id_binding() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle.enter("graph_from_prepared_table_export_rejects_invalid_edge_id_binding").expect("entered");
    let runtime = explicit_runtime();
    let df = cudf::DataFrame::from_series([
        ("src", runtime.series(&[0i32, 1]).expect("src")),
        ("dst", runtime.series(&[1i32, 2]).expect("dst")),
        ("edge_id", runtime.series(&[10i64, 11]).expect("edge id")),
    ])
    .expect("dataframe");
    let export = df.try_prepare_device_table_export().expect("table export");

    let duplicate =
        TableGraphOptions::new(0, 1).edge_id_col(1).construction_policy(GraphConstructionPolicy::RawLibcugraph);
    let error = graph_from_prepared_table_export::<Forward>(&handle, &export, &duplicate)
        .expect_err("duplicate selection must fail");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.message().contains("distinct columns"));

    let out_of_range =
        TableGraphOptions::new(0, 1).edge_id_col(3).construction_policy(GraphConstructionPolicy::RawLibcugraph);
    let error = graph_from_prepared_table_export::<Forward>(&handle, &export, &out_of_range)
        .expect_err("out-of-range selection must fail");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.message().contains("out of bounds"));

    let mismatched =
        TableGraphOptions::new(0, 1).edge_id_col(2).construction_policy(GraphConstructionPolicy::RawLibcugraph);
    let error = graph_from_prepared_table_export::<Forward>(&handle, &export, &mismatched)
        .expect_err("mismatched edge-ID dtype must fail");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.message().contains("edge_ids must have the same dtype"));
}

#[test]
fn graph_from_prepared_table_export_runtime_bound_handle_uses_consumer_stream_authorities() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle
        .enter("graph_from_prepared_table_export_runtime_bound_handle_uses_consumer_stream_authorities")
        .expect("entered");
    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let df = cudf::DataFrame::from_series([("src", src), ("dst", dst)]).expect("dataframe");
    let recorded = Arc::new(Mutex::new(Vec::new()));

    let export = df
        .try_prepare_device_table_export()
        .expect("table export")
        .with_lease_backend(LeaseBackend::new(RecordingLeaseBackend { recorded: Arc::clone(&recorded) }));
    let opts = TableGraphOptions::new(0, 1).renumber();

    let _graph = graph_from_prepared_table_export::<Forward>(&handle, &export, &opts).expect("graph");

    let expected_stream_raw = runtime.stream().testing_raw_handle_value() as *mut std::ffi::c_void;
    let expected_device = runtime.device().ordinal() as i32;
    let recorded = recorded.lock().expect("recorded");
    assert_eq!(recorded.len(), 1, "table export should record one activation/completion pair");
    assert_eq!(
        recorded[0],
        RecordedLeaseBindings {
            activate_read_start: RecordedReadStartBinding::ConsumerStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
            activate_completion: RecordedCompletionBinding::SameDeviceStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
            complete_completion: RecordedCompletionBinding::SameDeviceStream {
                raw: expected_stream_raw as usize,
                device: expected_device,
            },
        }
    );
}

#[test]
fn graph_from_prepared_table_export_runtime_bound_handle_defers_completion_sync_until_graph_drop() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle = handle
        .enter("graph_from_prepared_table_export_runtime_bound_handle_defers_completion_sync_until_graph_drop")
        .expect("entered");
    let src = runtime.series(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.series(TRIANGLE_TAIL_DST_I32).expect("dst");
    let edge_ids = runtime.series(&[10i32, 11, 12, 13]).expect("edge ids");
    let df = cudf::DataFrame::from_series([("src", src), ("dst", dst), ("edge_id", edge_ids)]).expect("dataframe");

    let export = df.try_prepare_device_table_export().expect("table export");
    assert!(export.clone_retained_owner().is_some(), "dataframe prepared export should retain its producer owner");
    let opts = TableGraphOptions::new(0, 1)
        .edge_id_col(2)
        .renumber()
        .construction_policy(GraphConstructionPolicy::RawLibcugraph);

    reset_event_observation();
    let _ = event_synchronize_call_count();

    let graph = graph_from_prepared_table_export::<Forward>(&handle, &export, &opts).expect("graph");
    let before_drop = event_synchronize_call_count();

    drop(graph);
    let after_drop = event_synchronize_call_count();

    assert!(
        after_drop > before_drop,
        "dropping the graph should perform deferred prepared-table completion synchronization"
    );
}

#[test]
fn graph_from_prepared_columns_runtime_bound_owned_column_export_defers_completion_sync_until_graph_drop() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle = handle
        .enter("graph_from_prepared_columns_runtime_bound_owned_column_export_defers_completion_sync_until_graph_drop")
        .expect("entered");
    let src = runtime.column_from_slice(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.column_from_slice(TRIANGLE_TAIL_DST_I32).expect("dst");

    let src_export = src.try_into_prepared_device_column_export().expect("src export");
    let dst_export = dst.try_into_prepared_device_column_export().expect("dst export");
    assert!(
        src_export.clone_retained_owner().is_some(),
        "owned column prepared export should retain its producer owner"
    );
    assert!(
        dst_export.clone_retained_owner().is_some(),
        "owned column prepared export should retain its producer owner"
    );
    let opts = PreparedGraphOptions::default().renumber();

    reset_event_observation();
    let _ = event_synchronize_call_count();

    let graph = graph_from_prepared_columns::<Forward>(&handle, &src_export, &dst_export, None, &opts).expect("graph");
    let before_drop = event_synchronize_call_count();

    drop(graph);
    let after_drop = event_synchronize_call_count();

    assert!(
        after_drop > before_drop,
        "dropping the graph should perform deferred completion synchronization for owned low-level column exports"
    );
}

#[test]
fn graph_from_prepared_table_export_runtime_bound_owned_table_export_defers_completion_sync_until_graph_drop() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let runtime = explicit_runtime_owner();
    let handle = runtime_bound_handle(&runtime);
    let handle = handle
        .enter(
            "graph_from_prepared_table_export_runtime_bound_owned_table_export_defers_completion_sync_until_graph_drop",
        )
        .expect("entered");
    let src = runtime.column_from_slice(TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = runtime.column_from_slice(TRIANGLE_TAIL_DST_I32).expect("dst");
    let table = cudf::core::Table::from_columns(vec![src, dst]).expect("table");

    let export = table.try_into_prepared_device_table_export().expect("table export");
    assert!(export.clone_retained_owner().is_some(), "owned table prepared export should retain its producer owner");
    let opts = TableGraphOptions::new(0, 1).renumber();

    reset_event_observation();
    let _ = event_synchronize_call_count();

    let graph = graph_from_prepared_table_export::<Forward>(&handle, &export, &opts).expect("graph");
    let before_drop = event_synchronize_call_count();

    drop(graph);
    let after_drop = event_synchronize_call_count();

    assert!(
        after_drop > before_drop,
        "dropping the graph should perform deferred completion synchronization for owned low-level table exports"
    );
}

#[test]
fn graph_from_prepared_columns_python_cugraph_weighted_undirected_simple_keeps_minimum_reciprocal_weight() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle
        .enter("graph_from_prepared_columns_python_cugraph_weighted_undirected_simple_keeps_minimum_reciprocal_weight")
        .expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("weight");

    let src_export = src.try_prepare_device_column_export().expect("src export");
    let dst_export = dst.try_prepare_device_column_export().expect("dst export");
    let wgt_export = wgt.try_prepare_device_column_export().expect("weight export");

    let opts = PreparedGraphOptions::default().properties(cugraph::types::GraphProperties::undirected()).renumber();
    let mut graph = graph_from_prepared_columns::<cugraph::graph::Transposed>(
        &handle,
        &src_export,
        &dst_export,
        Some(&wgt_export),
        &opts,
    )
    .expect("graph_from_prepared_columns");

    let actual = pagerank_score_map(&handle, &mut graph);
    let expected = python_weighted_reciprocal_expected_pagerank_map(&handle);
    assert_score_maps_close_f32(&actual, &expected, 1.0e-6, "graph_from_prepared_columns");
}

#[test]
fn graph_from_prepared_table_export_python_cugraph_accepts_undirected_multigraph() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle =
        handle.enter("graph_from_prepared_table_export_python_cugraph_accepts_undirected_multigraph").expect("entered");
    let runtime = explicit_runtime();
    let df = python_undirected_multigraph_dataframe(&runtime);
    let export = df.try_prepare_device_table_export().expect("table export");
    let props = cugraph::types::GraphProperties { is_symmetric: true, is_multigraph: true };

    let opts = TableGraphOptions::new(0, 1).properties(props).renumber();
    let mut graph =
        graph_from_prepared_table_export::<Forward>(&handle, &export, &opts).expect("graph_from_prepared_table_export");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), 3);
}

#[test]
fn graph_from_prepared_columns_rejects_host_exports() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle.enter("graph_from_prepared_columns_rejects_host_exports").expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = runtime.series(DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");

    let src_export = src.try_prepare_host_column_export().expect("src host export");
    let dst_export = dst.try_prepare_host_column_export().expect("dst host export");

    let opts = PreparedGraphOptions::default().renumber();
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
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle.enter("graph_from_prepared_table_export_rejects_host_exports").expect("entered");
    let runtime = explicit_runtime();

    let src = runtime.series(DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = runtime.series(DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");
    let df = cudf::DataFrame::from_series([("src", src), ("dst", dst)]).expect("dataframe");

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
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle.enter("prepared_export_vs_deep_copy_import_pagerank_equivalence").expect("entered");
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = triangle_graph_dataframe(&producer_runtime);
    let export = producer_df.try_prepare_device_table_export().expect("prepared export");
    let mut prepared_graph = build_undirected_prepared_graph(&handle, &export).expect("prepared graph");

    let imported_df =
        cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&export, None, &consumer_runtime)
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
    let _guard = common::cudf::cuda_test_serial_guard();
    let _event_observation = lock_event_observation();
    let handle = test_handle();
    let handle = handle.enter("prepared_host_export_import_pagerank_matches_prepared_device_path").expect("entered");
    let producer_runtime = explicit_runtime();
    let consumer_runtime = explicit_runtime();

    let producer_df = triangle_graph_dataframe(&producer_runtime);
    let prepared_export = producer_df.try_prepare_device_table_export().expect("prepared device export");
    let mut prepared_graph = build_undirected_prepared_graph(&handle, &prepared_export).expect("prepared graph");

    let host_export = producer_df.try_prepare_host_table_export().expect("prepared host export");
    let imported_df =
        cudf::DataFrame::try_import_from_prepared_table_export_into_runtime(&host_export, None, &consumer_runtime)
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
