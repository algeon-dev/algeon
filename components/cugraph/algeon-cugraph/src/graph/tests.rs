use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use cudf::{
    DeviceId, DeviceMemoryResource, Runtime,
    interop::{
        contract::{
            ActiveLeaseBackend, CompletionAuthority, CompletionWitness, CompletionWitnessBackend, InteropErrorKind,
            LeaseBackend, ProducerReadiness, ReadStartAuthority, Result as InteropResult,
        },
        cuda_stream_handle,
    },
    runtime::{DeviceEntryGuard, DeviceFaultNativeStatus, DeviceHealthStatus},
    testing::{
        device_safety::{DeviceEntryTestCall as TestCudaCall, DeviceSafetyTestScope as TestCudaHooks},
        interop::column_export_with_lease_backend_for_test,
        run_named_cases,
    },
};
use cugraph_sys::{
    cugraph_edgelist_get_destinations, cugraph_edgelist_get_edge_weights, cugraph_edgelist_get_sources,
    cugraph_edgelist_t, cugraph_type_erased_device_array_view_t,
};

use super::*;
use crate::{
    array::{DeviceArray, RawDeviceArrayView},
    cuda::fail_next_fatal_device_resource_destroy,
    error::ErrorKind,
    handle::ResourceHandle,
};

#[derive(Debug)]

struct ReleaseProbe(Arc<AtomicBool>);

impl Drop for ReleaseProbe {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A runtime whose execution domains retain a probe that records when the
/// last of them is released.
fn probed_runtime() -> (Arc<cudf::Runtime>, Arc<AtomicBool>) {
    let released = Arc::new(AtomicBool::new(false));
    let runtime = cudf::Runtime::builder()
        .build()
        .expect("runtime")
        .with_retained_lifetime(Arc::new(ReleaseProbe(Arc::clone(&released))))
        .expect("probed runtime");
    (Arc::new(runtime), released)
}

struct TestCompletionWitness {
    fail: bool,
}

impl CompletionWitnessBackend for TestCompletionWitness {
    fn is_ready(&self) -> InteropResult<bool> {
        Ok(!self.fail)
    }

    fn synchronize(&self) -> InteropResult<()> {
        if self.fail {
            return Err(InteropError::new(InteropErrorKind::ReadinessUnavailable, "injected sync failure"));
        }
        Ok(())
    }
}

/// Completes every stream lease with a witness that fails its wait when
/// `fail_witness` is set.
struct WitnessLeaseBackend {
    fail_witness: bool,
}

impl ActiveLeaseBackend for WitnessLeaseBackend {
    fn complete(
        &self,
        _readiness: ProducerReadiness<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> InteropResult<Option<CompletionWitness>> {
        Ok(Some(CompletionWitness::new(Arc::new(TestCompletionWitness { fail: self.fail_witness }))))
    }

    fn drop_incomplete(
        &self,
        _readiness: ProducerReadiness<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> InteropResult<()> {
        Ok(())
    }
}

/// Completes one stream-ordered lease over a prepared export that retains
/// `owner`, so only the returned pending completion keeps `owner` alive.
fn pending_input_completion(runtime: &cudf::Runtime, owner: Arc<()>, fail_witness: bool) -> DeferredInputCompletion {
    let series = runtime.series(&[0i32, 1]).expect("series");
    let export = column_export_with_lease_backend_for_test(
        &series.try_prepare_device_column_export().expect("prepared export"),
        LeaseBackend::new(WitnessLeaseBackend { fail_witness }),
    )
    .expect("lease backend")
    .with_retained_owner(owner);
    let stream = cuda_stream_handle(runtime.stream()).expect("stream handle");
    let lease = export
        .try_activate_read_lease(
            ReadStartAuthority::ConsumerStream { stream },
            CompletionAuthority::SameDeviceStream { stream },
        )
        .expect("lease");
    DeferredInputCompletion::new(vec![lease.complete().expect("complete")]).expect("pending")
}

fn dummy_runtime_bound_graph(
    handle: &EnteredContext<'_>,
    pending_input_completion: Option<DeferredInputCompletion>,
) -> Graph<Forward> {
    Graph {
        raw: ptr::null_mut(),
        device: handle.device(),
        properties: GraphProperties::default(),
        vertex_dtype: DataType::Int32,
        weight_dtype: DataType::Float32,
        has_weights: false,
        edge_id_dtype: None,
        pending_input_completion,
        resource_handle_leases: RefCell::new(handle.runtime_bound_lease().into_iter().collect()),
        ordered_binding: Cell::new(handle.runtime_bound_lease().map(|lease| lease.runtime().binding_id())),
        untracked_resource_handle_use: Cell::new(false),
        _nosend_nosync: PhantomData,
        _transposition: PhantomData,
    }
}

#[test]
fn native_owner_cases_1() {
    run_named_cases(&[
        (
            "graph::tests::test_graph_synchronize_input_reads_releases_owner_before_graph_destruction",
            test_graph_synchronize_input_reads_releases_owner_before_graph_destruction,
        ),
        (
            "graph::tests::test_graph_synchronize_input_reads_failure_retains_unfinished_owner",
            test_graph_synchronize_input_reads_failure_retains_unfinished_owner,
        ),
        ("graph::tests::test_graph_park_releases_pending_input_owner", test_graph_park_releases_pending_input_owner),
        (
            "graph::tests::test_graph_park_surfaces_pending_input_completion_error",
            test_graph_park_surfaces_pending_input_completion_error,
        ),
        (
            "graph::tests::resume_under_executes_on_borrower_binding_and_reparks_there",
            resume_under_executes_on_borrower_binding_and_reparks_there,
        ),
        (
            "graph::tests::exact_resume_rejects_a_stale_binding_after_borrower_repark",
            exact_resume_rejects_a_stale_binding_after_borrower_repark,
        ),
        (
            "graph::tests::resume_under_rejects_non_runtime_bound_handles",
            resume_under_rejects_non_runtime_bound_handles,
        ),
        (
            "graph::tests::test_graph_park_rejects_non_runtime_bound_transfer_handle",
            test_graph_park_rejects_non_runtime_bound_transfer_handle,
        ),
        (
            "graph::tests::test_graph_retains_one_native_handle_owner_for_one_binding_after_many_hot_hits",
            test_graph_retains_one_native_handle_owner_for_one_binding_after_many_hot_hits,
        ),
        (
            "graph::tests::test_graph_last_lease_releases_runtime_owner_after_graph_drop",
            test_graph_last_lease_releases_runtime_owner_after_graph_drop,
        ),
        (
            "graph::tests::test_graph_rejects_runtime_with_different_execution_binding",
            test_graph_rejects_runtime_with_different_execution_binding,
        ),
        (
            "graph::tests::test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use",
            test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use,
        ),
    ]);
}

fn test_graph_synchronize_input_reads_releases_owner_before_graph_destruction() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_synchronize_input_reads_releases_owner").expect("entered");
    let owner = Arc::new(());
    let pending = pending_input_completion(&runtime, Arc::clone(&owner), false);
    let mut graph = dummy_runtime_bound_graph(&entered, Some(pending));

    assert_eq!(Arc::strong_count(&owner), 2);
    graph.synchronize_input_reads().expect("construction input fence");
    assert_eq!(Arc::strong_count(&owner), 1, "producer ownership must end while the graph remains live");
    assert!(graph.pending_input_completion.is_none());
    graph.park(&entered).expect("completed graph remains parkable");
}

fn test_graph_synchronize_input_reads_failure_retains_unfinished_owner() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_synchronize_input_reads_failure").expect("entered");
    let owner = Arc::new(());
    let pending = pending_input_completion(&runtime, Arc::clone(&owner), true);
    let mut graph = dummy_runtime_bound_graph(&entered, Some(pending));

    let error = graph.synchronize_input_reads().expect_err("failed construction input fence");
    assert_eq!(error.kind(), ErrorKind::InvalidState);
    assert_eq!(error.operation(), "Graph::synchronize_input_reads");
    assert!(graph.pending_input_completion.is_some());
    assert_eq!(Arc::strong_count(&owner), 2, "an unfinished read must continue retaining producer storage");
}

fn test_graph_park_releases_pending_input_owner() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_park_releases_pending_input_owner").expect("entered");

    let owner = Arc::new(());
    let pending = pending_input_completion(&runtime, Arc::clone(&owner), false);
    let graph = dummy_runtime_bound_graph(&entered, Some(pending));

    assert_eq!(Arc::strong_count(&owner), 2, "graph should retain the prepared input owner before park");
    let parked = graph.park(&entered).expect("park");
    assert_eq!(Arc::strong_count(&owner), 1, "park should release prepared input owners");
    drop(parked);
}

fn test_graph_park_surfaces_pending_input_completion_error() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_park_surfaces_pending_input_completion_error").expect("entered");

    let owner = Arc::new(());
    let pending = pending_input_completion(&runtime, Arc::clone(&owner), true);
    let graph = dummy_runtime_bound_graph(&entered, Some(pending));

    let err = graph.park(&entered).expect_err("park should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidState);
    assert_eq!(err.operation(), "Graph::park");
    assert_eq!(Arc::strong_count(&owner), 2, "a failed input fence must leak the retained input owner");
}

fn triangle_parked_graph(handle: &EnteredContext<'_>) -> ParkedGraph<Forward> {
    let src = DeviceArray::from_slice(handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(handle, &[1i32, 2, 0]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(handle).expect("graph");
    graph.park(handle).expect("park")
}

fn resume_under_executes_on_borrower_binding_and_reparks_there() {
    let storage_runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let borrower_runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    assert_ne!(
        storage_runtime.binding_id(),
        borrower_runtime.binding_id(),
        "borrower and storage runtimes must use different bindings"
    );
    let storage_handle = ResourceHandle::for_runtime(Arc::clone(&storage_runtime)).expect("handle");
    let borrower_handle = ResourceHandle::for_runtime(Arc::clone(&borrower_runtime)).expect("handle");

    let parked = {
        let entered = storage_handle.enter("resume_under_build").expect("entered");
        triangle_parked_graph(&entered)
    };
    let reparked = {
        let entered = borrower_handle.enter("resume_under_borrow").expect("entered");
        let mut graph = parked.resume_under(&entered).expect("resume_under");
        assert_eq!(graph.edge_count(&entered).expect("borrower-domain edgelist"), 3);
        graph.park_under(&entered).expect("park_under")
    };
    // Exact resume follows the park binding, which the borrower move updated.
    {
        let entered = borrower_handle.enter("resume_under_exact").expect("entered");
        let _graph = reparked.resume(&entered).expect("exact resume under the parking binding");
    }
}

fn exact_resume_rejects_a_stale_binding_after_borrower_repark() {
    let storage_runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let borrower_runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let storage_handle = ResourceHandle::for_runtime(Arc::clone(&storage_runtime)).expect("handle");
    let borrower_handle = ResourceHandle::for_runtime(Arc::clone(&borrower_runtime)).expect("handle");

    let parked = {
        let entered = storage_handle.enter("stale_resume_build").expect("entered");
        triangle_parked_graph(&entered)
    };
    let reparked = {
        let entered = borrower_handle.enter("stale_resume_borrow").expect("entered");
        let graph = parked.resume_under(&entered).expect("resume_under");
        graph.park_under(&entered).expect("park_under")
    };
    {
        let entered = storage_handle.enter("stale_resume_exact").expect("entered");
        let err = reparked.resume(&entered).expect_err("exact resume must reject the stale binding");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "ParkedGraph::resume");
    }
}

fn resume_under_rejects_non_runtime_bound_handles() {
    let storage_runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let storage_handle = ResourceHandle::for_runtime(Arc::clone(&storage_runtime)).expect("handle");
    let local_handle = ResourceHandle::new().expect("local handle");

    let parked = {
        let entered = storage_handle.enter("non_runtime_resume_build").expect("entered");
        triangle_parked_graph(&entered)
    };
    {
        let entered = local_handle.enter("non_runtime_resume_borrow").expect("entered");
        let err = parked.resume_under(&entered).expect_err("resume_under must reject non-runtime handles");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "ParkedGraph::resume_under");
    }
}

fn test_graph_park_rejects_non_runtime_bound_transfer_handle() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let build_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("build handle");
    let build_entered =
        build_handle.enter("test_graph_park_rejects_non_runtime_bound_transfer_handle_build").expect("entered");
    let local_handle = ResourceHandle::new().expect("local handle");
    let local_entered =
        local_handle.enter("test_graph_park_rejects_non_runtime_bound_transfer_handle_local").expect("entered");
    let graph = dummy_runtime_bound_graph(&build_entered, None);

    let err = graph.park(&local_entered).expect_err("park should reject non-runtime-bound handle");
    assert_eq!(err.kind(), ErrorKind::InvalidState);
    assert_eq!(err.operation(), "Graph::park");
}

fn test_graph_retains_one_native_handle_owner_for_one_binding_after_many_hot_hits() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let build_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("build handle");
    let build_entered =
        build_handle.enter("test_graph_retains_one_native_handle_owner_for_one_binding_build").expect("entered");
    let graph = dummy_runtime_bound_graph(&build_entered, None);
    assert_eq!(graph.resource_handle_leases.borrow().len(), 1);
    drop(build_entered);

    for _ in 0..100 {
        let algorithm_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("algorithm handle");
        let algorithm_entered = algorithm_handle
            .enter("test_graph_retains_one_native_handle_owner_for_one_binding_algorithm")
            .expect("entered");
        graph
            .retain_runtime_bound_handle(
                &algorithm_entered,
                "test_graph_retains_one_native_handle_owner_for_one_binding",
            )
            .expect("retain runtime handle");
    }

    assert_eq!(graph.resource_handle_leases.borrow().len(), 1);
}

fn test_graph_last_lease_releases_runtime_owner_after_graph_drop() {
    let (runtime, released) = probed_runtime();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_last_lease_releases_runtime_owner_after_graph_drop").expect("entered");
    let graph = dummy_runtime_bound_graph(&entered, None);

    drop(entered);
    drop(handle);
    drop(runtime);
    assert!(!released.load(Ordering::SeqCst), "the graph lease must retain the runtime-bound owner");
    drop(graph);
    assert!(released.load(Ordering::SeqCst), "graph teardown must release the final runtime-bound owner lease");
}

#[test]
fn test_graph_drop_quarantines_native_pointer_when_device_entry_fails() {
    let graph = Graph::<Forward> {
        raw: std::ptr::NonNull::<cugraph_graph_t>::dangling().as_ptr(),
        device: 0,
        properties: GraphProperties::default(),
        vertex_dtype: DataType::Int32,
        weight_dtype: DataType::Float32,
        has_weights: false,
        edge_id_dtype: None,
        pending_input_completion: None,
        resource_handle_leases: RefCell::new(Vec::new()),
        ordered_binding: Cell::new(None),
        untracked_resource_handle_use: Cell::new(false),
        _nosend_nosync: PhantomData,
        _transposition: PhantomData,
    };
    let reset =
        TestCudaHooks::install(vec![TestCudaCall::GetDevice { status: cudf_sys::CUDF_ERROR_FATAL_CUDA, device: 0 }]);
    let before = reset.health_snapshot(cudf::DeviceId::DEFAULT).quarantined_teardown_count();

    drop(graph);

    assert!(reset.health_snapshot(cudf::DeviceId::DEFAULT).quarantined_teardown_count() > before);
}

#[test]
fn fatal_native_graph_destroy_is_contained_and_rejects_later_owner_work() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let device = runtime.device();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("fatal_native_graph_destroy_build").expect("entered");
    let src = DeviceArray::from_slice(&entered, &[0_i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1_i32, 2, 0]).expect("dst");
    let src_view = src.view().expect("src view");
    let dst_view = dst.view().expect("dst view");
    let graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");
    drop(src_view);
    drop(dst_view);
    drop(src);
    drop(dst);
    drop(entered);
    drop(handle);
    drop(runtime);

    let scope = TestCudaHooks::install(
        (0..8).map(|_| TestCudaCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: device.ordinal() as i32 }),
    );
    fail_next_fatal_device_resource_destroy(700);

    // A native bridge regression that unwound or terminated across the C
    // boundary would abort this test process instead of returning here.
    drop(graph);

    let snapshot = scope.health_snapshot(device);
    assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
    let first = snapshot.first_fault().expect("fatal graph destroy fault");
    assert_eq!(first.operation(), "Graph::drop");
    assert_eq!(first.native_status(), Some(DeviceFaultNativeStatus::Cuda(700)));
    assert!(snapshot.quarantined_teardown_count() >= 1);
    let error = DeviceEntryGuard::enter(device, "later_cugraph_owner_work")
        .expect_err("fatal graph teardown must reject later owner-device work");
    assert_eq!(error.code(), cudf::ErrorCode::FatalCuda);
}

#[test]
fn parked_graph_event_and_runtime_chain_fail_closed_on_device_query_failure() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let parked = {
        let entered = handle.enter("parked_fail_closed_build").expect("entered");
        let src = DeviceArray::from_slice(&entered, &[0_i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1_i32, 2, 0]).expect("dst");
        let src_view = src.view().expect("src view");
        let dst_view = dst.view().expect("dst view");
        GraphBuilder::new(&src_view, &dst_view)
            .build::<Forward>(&entered)
            .expect("graph")
            .park(&entered)
            .expect("parked graph")
    };
    drop(handle);
    drop(runtime);

    // ParkedGraph teardown traverses the graph, its CudaEvent, the shared
    // ResourceHandle owner, then ExecutionDomain stream/MR owners. Every
    // injected query failure must consume/quarantine the corresponding
    // pointer without invoking a destructor on an unproven device.
    let scope = TestCudaHooks::install(
        (0..8).map(|_| TestCudaCall::GetDevice { status: cudf_sys::CUDF_ERROR_FATAL_CUDA, device: 0 }),
    );
    drop(parked);

    let snapshot = scope.health_snapshot(cudf::DeviceId::DEFAULT);
    assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
    assert!(
        snapshot.quarantined_teardown_count() >= 3,
        "graph, event, and runtime-bound native owner must all fail closed"
    );
}

#[test]
fn test_graph_drop_recovers_poisoned_use_gate_and_releases_owner() {
    let (runtime, released) = probed_runtime();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let graph = {
        let entered = handle.enter("test_graph_drop_recovers_poisoned_use_gate_build").expect("entered build context");
        let src = DeviceArray::from_slice(&entered, &[0_i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1_i32, 2, 0]).expect("dst");
        let src_view = src.view().expect("src view");
        let dst_view = dst.view().expect("dst view");
        GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph")
    };

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _entered = handle.enter("test_graph_drop_recovers_poisoned_use_gate_poison").expect("entered");
        panic!("intentional runtime-bound handle use poison")
    }));
    assert!(unwind.is_err());
    handle
        .enter("test_graph_drop_recovers_poisoned_use_gate_normal_retry")
        .expect_err("normal handle entry remains fail-closed");

    drop(handle);
    drop(runtime);
    assert!(!released.load(Ordering::SeqCst), "the graph retains the poisoned runtime-bound owner");
    drop(graph);
    assert!(released.load(Ordering::SeqCst), "Graph::drop must recover the gate and release the owner");
}

#[test]
fn test_parked_graph_drop_recovers_poisoned_use_gate_and_releases_owner() {
    let (runtime, released) = probed_runtime();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let parked = {
        let entered =
            handle.enter("test_parked_graph_drop_recovers_poisoned_use_gate_build").expect("entered build context");
        let src = DeviceArray::from_slice(&entered, &[0_i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1_i32, 2, 0]).expect("dst");
        let src_view = src.view().expect("src view");
        let dst_view = dst.view().expect("dst view");
        let graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");
        graph.park(&entered).expect("park graph")
    };

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _entered = handle.enter("test_parked_graph_drop_recovers_poisoned_use_gate_poison").expect("entered");
        panic!("intentional parked-graph owner use poison")
    }));
    assert!(unwind.is_err());
    handle
        .enter("test_parked_graph_drop_recovers_poisoned_use_gate_normal_retry")
        .expect_err("normal handle entry remains fail-closed");

    drop(handle);
    drop(runtime);
    assert!(!released.load(Ordering::SeqCst), "the parked graph retains the poisoned runtime-bound owner");
    drop(parked);
    assert!(released.load(Ordering::SeqCst), "ParkedGraph::drop must recover the gate and release the owner");
}

#[test]
fn runtime_bound_native_owner_runs_graph_work_across_threads_and_drops_on_another_thread() {
    const ITERATIONS: usize = 100;
    const STEP_TIMEOUT: Duration = Duration::from_secs(10);

    let (runtime, released) = probed_runtime();
    let (mut parked, native_owner) = {
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("build handle");
        let entered = handle.enter("cross_thread_native_owner_build").expect("entered build context");
        let native_owner = entered.handle_ptr() as usize;
        let src = DeviceArray::from_slice(&entered, &[0_i32, 1, 2, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1_i32, 2, 0, 1]).expect("dst");
        let src_view = src.view().expect("src view");
        let dst_view = dst.view().expect("dst view");
        let graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("cross-thread graph");
        (graph.park(&entered).expect("initial park"), native_owner)
    };

    for iteration in 0..ITERATIONS {
        let worker_runtime = Arc::clone(&runtime);
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let handle = ResourceHandle::for_runtime(worker_runtime).expect("worker handle");
            let entered = handle.enter("cross_thread_native_owner_step").expect("entered worker context");
            let observed_owner = entered.handle_ptr() as usize;
            let mut graph = parked.resume(&entered).expect("resume graph");
            let degrees = graph.out_degrees_all(&entered).expect("real cuGraph out-degrees operation");
            drop(degrees);
            let parked = graph.park(&entered).expect("repark graph");
            result_tx.send((parked, observed_owner)).expect("send worker result");
        });

        let (next, observed_owner) = result_rx
            .recv_timeout(STEP_TIMEOUT)
            .unwrap_or_else(|error| panic!("cross-thread iteration {iteration} timed out: {error}"));
        worker.join().unwrap_or_else(|payload| std::panic::resume_unwind(payload));
        assert_eq!(observed_owner, native_owner, "every thread must enter the same native owner generation");
        parked = next;
    }

    drop(runtime);
    assert!(!released.load(Ordering::SeqCst), "the parked graph retains the runtime-bound native owner");
    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    let drop_worker = std::thread::spawn(move || {
        drop(parked);
        dropped_tx.send(()).expect("report parked graph drop");
    });
    dropped_rx.recv_timeout(STEP_TIMEOUT).expect("parked graph drop timed out");
    drop_worker.join().unwrap_or_else(|payload| std::panic::resume_unwind(payload));
    assert!(released.load(Ordering::SeqCst), "cross-thread final drop must release the runtime owner without a cycle");
}

fn test_graph_rejects_runtime_with_different_execution_binding() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let build_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("build handle");
    let build_entered =
        build_handle.enter("test_graph_rejects_runtime_with_different_execution_binding_build").expect("entered");
    let graph = dummy_runtime_bound_graph(&build_entered, None);

    let companion = Arc::new(cudf::Runtime::builder().build().expect("independent runtime"));
    assert_ne!(runtime.binding_id(), companion.binding_id());
    let other_handle = ResourceHandle::for_runtime(companion).expect("other handle");
    let other_entered =
        other_handle.enter("test_graph_rejects_runtime_with_different_execution_binding_use").expect("entered");

    let err = graph
        .retain_runtime_bound_handle(&other_entered, "test_graph_rejects_runtime_with_different_execution_binding")
        .expect_err("different execution binding must fail before graph work");
    assert_eq!(err.kind(), ErrorKind::InvalidState);
    assert_eq!(graph.resource_handle_leases.borrow().len(), 1);
}

fn test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let runtime_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("runtime handle");
    let runtime_entered = runtime_handle
        .enter("test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use_runtime")
        .expect("entered");
    let local_handle = ResourceHandle::new().expect("local handle");
    let local_entered =
        local_handle.enter("test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use_local").expect("entered");
    let graph = dummy_runtime_bound_graph(&runtime_entered, None);

    graph
        .retain_runtime_bound_handle(&local_entered, "test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use")
        .expect("taint graph");
    let err = graph.park(&runtime_entered).expect_err("park should reject tainted graph");
    assert_eq!(err.kind(), ErrorKind::InvalidState);
    assert_eq!(err.operation(), "Graph::park");
}

#[test]
fn test_weighted_graph_frugal_quarters_preserve_edges_under_cap() {
    fn construct_edges(
        sources: &[i64],
        destinations: &[i64],
        weights: &[f64],
        cap: usize,
    ) -> (Vec<(i64, i64, u64)>, u64) {
        let device = DeviceId::current().expect("CUDA device");
        let upstream = DeviceMemoryResource::cuda_async_on(device, None, Some(0)).expect("CUDA async resource");
        let budget = upstream.budget_tracking_limit(cap).expect("construction allocation cap");
        let runtime = Arc::new(Runtime::builder().with_memory_resource(budget.clone()).build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(runtime).expect("resource handle");
        let entered = handle.enter("weighted_frugal_quarters").expect("entered context");
        let graph = {
            let source = DeviceArray::from_slice(&entered, sources).expect("source array");
            let destination = DeviceArray::from_slice(&entered, destinations).expect("destination array");
            let weight = DeviceArray::from_slice(&entered, weights).expect("weight array");
            let source_view = source.view().expect("source view");
            let destination_view = destination.view().expect("destination view");
            let weight_view = weight.view().expect("weight view");
            GraphBuilder::new(&source_view, &destination_view)
                .weights(&weight_view)
                .properties(GraphProperties { is_symmetric: false, is_multigraph: true })
                .renumber()
                .build::<Forward>(&entered)
                .expect("weighted graph must fit its construction cap")
        };
        let construction_peak = budget.budget_tracking_usage().expect("construction usage").peak_bytes;
        let mut raw_edgelist = ptr::null_mut();
        let mut error = ptr::null_mut();
        // SAFETY: the graph and entered handle remain alive until the returned edgelist is copied and freed.
        unsafe {
            let status = cugraph_decompress_to_edgelist(
                entered.handle_ptr(),
                graph.as_ptr(&entered, "weighted_frugal_quarters").expect("graph handle"),
                to_bool_t(false),
                &mut raw_edgelist,
                &mut error,
            );
            check_error(status, error, entered.device(), "weighted_frugal_quarters").expect("decompress graph");
        }
        // SAFETY: successful decompression transfers the result to this owner until all host copies complete.
        let _edgelist =
            unsafe { DecompressedEdgelist::from_raw(raw_edgelist, entered.device(), entered.runtime_bound_lease()) }
                .expect("owned edgelist");
        let read_column =
            |getter: unsafe extern "C" fn(*mut cugraph_edgelist_t) -> *mut cugraph_type_erased_device_array_view_t,
             expected_dtype| {
                // SAFETY: these views borrow the live edgelist; its owner outlives every host copy.
                let view = unsafe {
                    RawDeviceArrayView::from_borrowed(
                        getter(raw_edgelist),
                        Some(entered.device()),
                        "weighted_frugal_quarters",
                        "edgelist column",
                    )
                }
                .expect("edgelist column view");
                assert_eq!(view.dtype().expect("column dtype"), expected_dtype);
                view.to_host_vec(&entered).expect("host column")
            };
        let source = read_column(cugraph_edgelist_get_sources, DataType::Int64);
        let destination = read_column(cugraph_edgelist_get_destinations, DataType::Int64);
        let weight = read_column(cugraph_edgelist_get_edge_weights, DataType::Float64);
        let edges = source
            .as_chunks::<8>()
            .0
            .iter()
            .zip(destination.as_chunks::<8>().0.iter())
            .zip(weight.as_chunks::<8>().0.iter())
            .map(|((source, destination), weight)| {
                (i64::from_ne_bytes(*source), i64::from_ne_bytes(*destination), u64::from_ne_bytes(*weight))
            })
            .collect::<Vec<_>>();
        (edges, construction_peak)
    }

    fn source_for_row(row: usize, skewed: bool) -> i64 {
        if skewed { if row < ROWS * 7 / 8 { 0 } else { 1 + (row % 16) as i64 } } else { (row / 1024) as i64 }
    }

    fn verify_edges(edges: &[(i64, i64, u64)], skewed: bool, description: &str) {
        assert_eq!(edges.len(), ROWS, "{description} edge count, skewed={skewed}");
        let mut remaining = vec![true; ROWS];
        for &(source, destination, weight_bits) in edges {
            assert!((0..1024).contains(&destination), "{description} unexpected destination {destination}");
            let weight_index = ((f64::from_bits(weight_bits) - 1.0) * 256.0) as i64;
            assert!((0..251).contains(&weight_index), "{description} unexpected weight {weight_bits}");
            assert_eq!((1.0 + weight_index as f64 / 256.0).to_bits(), weight_bits, "{description} weight bits");
            // Destination and weight encode row modulo 1024 and 251. These moduli are coprime,
            // and 113 is the inverse of 1024 modulo 251, so this enumerates every exact input tuple.
            let first_row = destination + 1024 * (((weight_index - destination).rem_euclid(251) * 113).rem_euclid(251));
            let mut row = first_row as usize;
            let mut matched = false;
            while row < ROWS {
                if remaining[row] && source_for_row(row, skewed) == source {
                    remaining[row] = false;
                    matched = true;
                    break;
                }
                row += 1024 * 251;
            }
            assert!(
                matched,
                "{description} extra or changed edge ({source}, {destination}, {weight_bits}), skewed={skewed}"
            );
        }
        assert!(remaining.iter().all(|row| !row), "{description} missing input edges, skewed={skewed}");
    }

    const ROWS: usize = 1 << 20;
    const MIB: usize = 1 << 20;
    let destination = (0..ROWS).map(|row| (row % 1024) as i64).collect::<Vec<_>>();
    let weights = (0..ROWS).map(|row| 1.0 + (row % 251) as f64 / 256.0).collect::<Vec<_>>();
    for skewed in [false, true] {
        let source = (0..ROWS).map(|row| source_for_row(row, skewed)).collect::<Vec<_>>();
        let (nonfrugal, _) = construct_edges(&source, &destination, &weights, 512 * MIB);
        let cap = if skewed { 80 * MIB } else { 58 * MIB };
        let (frugal, peak) = construct_edges(&source, &destination, &weights, cap);
        verify_edges(&nonfrugal, skewed, "nonfrugal input");
        verify_edges(&frugal, skewed, "frugal input");
        assert!(peak <= cap as u64);
        eprintln!("weighted graph construction skewed={skewed}, peak={peak}, cap={cap}");
    }
}
