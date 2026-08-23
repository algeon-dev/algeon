use std::{
    any::Any,
    cell::{Cell, RefCell},
    marker::PhantomData,
    ptr,
    sync::{Arc, mpsc},
    time::Duration,
};

use rapids_interop::{CompletionWitnessBackend, CompletionWitnessKind};

use super::*;
use crate::{
    array::DeviceArray,
    cuda::{TestCudaCall, TestCudaHooks, fail_next_fatal_device_resource_destroy},
    error::ErrorKind,
    handle::ResourceHandle,
};

#[derive(Debug)]
struct FailingCompletionWitness;

impl CompletionWitnessBackend for FailingCompletionWitness {
    fn is_ready(&self) -> rapids_interop::Result<bool> {
        Ok(false)
    }

    fn synchronize(&self) -> rapids_interop::Result<()> {
        Err(InteropError::new(rapids_interop::InteropErrorKind::ReadinessUnavailable, "injected sync failure"))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
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
        untracked_resource_handle_use: Cell::new(false),
        _nosend_nosync: PhantomData,
        _transposition: PhantomData,
    }
}

#[test]
fn test_graph_creation_triangle() {
    let handle = ResourceHandle::new().expect("handle");
    let entered = handle.enter("test_graph_creation_triangle").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered);
    assert!(graph.is_ok(), "graph creation failed: {:?}", graph.err());
}

#[test]
fn test_graph_park_releases_pending_input_owner() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_park_releases_pending_input_owner").expect("entered");

    let owner = Arc::new(());
    let retained_owner: Arc<dyn Any + Send + Sync> = owner.clone();
    let pending = DeferredInputCompletion::new(
        vec![retained_owner],
        vec![CompletionWitness::new(CompletionWitnessKind::ConsumerStream)],
    )
    .expect("pending");
    let graph = dummy_runtime_bound_graph(&entered, Some(pending));

    assert_eq!(Arc::strong_count(&owner), 2, "graph should retain the prepared input owner before park");
    let parked = graph.park(&entered).expect("park");
    assert_eq!(Arc::strong_count(&owner), 1, "park should release prepared input owners");
    drop(parked);
}

#[test]
fn test_graph_park_surfaces_pending_input_completion_error() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_park_surfaces_pending_input_completion_error").expect("entered");

    let owner = Arc::new(());
    let retained_owner: Arc<dyn Any + Send + Sync> = owner.clone();
    let pending = DeferredInputCompletion::new(
        vec![retained_owner],
        vec![CompletionWitness::with_backend(
            CompletionWitnessKind::ConsumerStream,
            Arc::new(FailingCompletionWitness),
        )],
    )
    .expect("pending");
    let graph = dummy_runtime_bound_graph(&entered, Some(pending));

    let err = graph.park(&entered).expect_err("park should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidState);
    assert_eq!(err.operation(), "Graph::park");
    assert_eq!(Arc::strong_count(&owner), 1, "failed park should drop graph and retained input owner");
}

#[test]
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

#[test]
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

#[test]
fn test_graph_last_lease_releases_runtime_owner_after_graph_drop() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let affinity = runtime.affinity_anchor();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
    let entered = handle.enter("test_graph_last_lease_releases_runtime_owner_after_graph_drop").expect("entered");
    let graph = dummy_runtime_bound_graph(&entered, None);

    drop(entered);
    drop(handle);
    drop(runtime);
    assert!(affinity.is_alive(), "the graph lease must retain the runtime-bound owner");
    drop(graph);
    assert!(!affinity.is_alive(), "graph teardown must release the final runtime-bound owner lease");
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
    assert_eq!(snapshot.status(), cudf::runtime::DeviceHealthStatus::Unhealthy);
    let first = snapshot.first_fault().expect("fatal graph destroy fault");
    assert_eq!(first.operation(), "Graph::drop");
    assert_eq!(first.native_status(), Some(cudf::runtime::DeviceFaultNativeStatus::Cuda(700)));
    assert!(snapshot.quarantined_teardown_count() >= 1);
    let error = cudf::runtime::DeviceEntryGuard::enter(device, "later_cugraph_owner_work")
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
    assert_eq!(snapshot.status(), cudf::runtime::DeviceHealthStatus::Unhealthy);
    assert!(
        snapshot.quarantined_teardown_count() >= 3,
        "graph, event, and runtime-bound native owner must all fail closed"
    );
}

#[test]
fn test_graph_drop_recovers_poisoned_use_gate_and_releases_owner() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let affinity = runtime.affinity_anchor();
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
    assert!(affinity.is_alive(), "the graph retains the poisoned runtime-bound owner");
    drop(graph);
    assert!(!affinity.is_alive(), "Graph::drop must recover the gate and release the owner");
}

#[test]
fn test_parked_graph_drop_recovers_poisoned_use_gate_and_releases_owner() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let affinity = runtime.affinity_anchor();
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
    assert!(affinity.is_alive(), "the parked graph retains the poisoned runtime-bound owner");
    drop(parked);
    assert!(!affinity.is_alive(), "ParkedGraph::drop must recover the gate and release the owner");
}

#[test]
fn runtime_bound_native_owner_runs_graph_work_across_threads_and_drops_on_another_thread() {
    const ITERATIONS: usize = 100;
    const STEP_TIMEOUT: Duration = Duration::from_secs(10);

    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let affinity = runtime.affinity_anchor();
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
    assert!(affinity.is_alive(), "the parked graph retains the runtime-bound native owner");
    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    let drop_worker = std::thread::spawn(move || {
        drop(parked);
        dropped_tx.send(()).expect("report parked graph drop");
    });
    dropped_rx.recv_timeout(STEP_TIMEOUT).expect("parked graph drop timed out");
    drop_worker.join().unwrap_or_else(|payload| std::panic::resume_unwind(payload));
    assert!(!affinity.is_alive(), "cross-thread final drop must release the runtime owner without a cycle");
}

#[test]
fn test_graph_rejects_same_affinity_with_different_execution_binding() {
    let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
    let build_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("build handle");
    let build_entered =
        build_handle.enter("test_graph_rejects_same_affinity_with_different_execution_binding_build").expect("entered");
    let graph = dummy_runtime_bound_graph(&build_entered, None);

    let independent = cudf::Runtime::builder().build().expect("independent runtime");
    let companion = Arc::new(independent.inherit_affinity_from(&runtime).expect("companion affinity"));
    assert_eq!(runtime.affinity_id(), companion.affinity_id());
    assert_ne!(runtime.binding_id(), companion.binding_id());
    let other_handle = ResourceHandle::for_runtime(companion).expect("other handle");
    let other_entered =
        other_handle.enter("test_graph_rejects_same_affinity_with_different_execution_binding_use").expect("entered");

    let err = graph
        .retain_runtime_bound_handle(
            &other_entered,
            "test_graph_rejects_same_affinity_with_different_execution_binding",
        )
        .expect_err("different execution binding must fail before graph work");
    assert_eq!(err.kind(), ErrorKind::InvalidState);
    assert_eq!(graph.resource_handle_leases.borrow().len(), 1);
}

#[test]
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
fn test_graph_creation_rejects_mismatched_src_dst_lengths() {
    let handle = ResourceHandle::new().expect("handle");
    let entered = handle.enter("test_graph_creation_rejects_mismatched_src_dst_lengths").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let err = match GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered) {
        Ok(_) => panic!("should reject mismatched edge-list lengths"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn test_graph_creation_with_edge_ids_and_edge_type_ids() {
    let handle = ResourceHandle::new().expect("handle");
    let entered = handle.enter("test_graph_creation_with_edge_ids_and_edge_type_ids").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let edge_ids = DeviceArray::from_slice(&entered, &[100i32, 101, 102]).expect("edge_ids");
    let edge_types = DeviceArray::from_slice(&entered, &[0i32, 1, 0]).expect("edge_types");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");
    let edge_types_view = edge_types.view().expect("edge_types_view");

    let graph = GraphBuilder::new(&src_view, &dst_view)
        .edge_ids(&edge_ids_view)
        .edge_type_ids(&edge_types_view)
        .renumber()
        .build::<Forward>(&entered);
    assert!(graph.is_ok(), "graph creation with edge_ids and edge_type_ids failed: {:?}", graph.err());
}

#[test]
fn test_graph_creation_with_weights_and_edge_properties() {
    let handle = ResourceHandle::new().expect("handle");
    let entered = handle.enter("test_graph_creation_with_weights_and_edge_properties").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 3]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 3, 0]).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0, 4.0]).expect("wgt");
    let edge_ids = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 3]).expect("edge_ids");
    let edge_types = DeviceArray::from_slice(&entered, &[0i32, 0, 1, 1]).expect("edge_types");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");
    let edge_types_view = edge_types.view().expect("edge_types_view");

    let graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .edge_ids(&edge_ids_view)
        .edge_type_ids(&edge_types_view)
        .renumber()
        .build::<Forward>(&entered);
    assert!(graph.is_ok(), "graph creation with weights + edge properties failed: {:?}", graph.err());
}

#[test]
fn test_graph_creation_rejects_mismatched_edge_ids_length() {
    let handle = ResourceHandle::new().expect("handle");
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

#[test]
fn test_graph_creation_rejects_mismatched_edge_ids_dtype() {
    let handle = ResourceHandle::new().expect("handle");
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

#[test]
fn test_graph_creation_rejects_mismatched_edge_type_ids_length() {
    let handle = ResourceHandle::new().expect("handle");
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

#[test]
fn test_rejects_symmetrize_with_edge_ids() {
    let handle = ResourceHandle::new().expect("handle");
    let entered = handle.enter("test_rejects_symmetrize_with_edge_ids").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let eids = DeviceArray::from_slice(&entered, &[10i32, 20, 30]).expect("eids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let eids_view = eids.view().expect("eids_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_ids(&eids_view)
        .symmetrize()
        .build::<Forward>(&entered)
        .expect_err("should reject symmetrize + edge_ids");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("symmetrize"));
    assert!(err.message().contains("edge_ids"));
}

#[test]
fn test_rejects_symmetrize_with_edge_type_ids() {
    let handle = ResourceHandle::new().expect("handle");
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

#[test]
fn test_rejects_start_time_without_end_time() {
    let handle = ResourceHandle::new().expect("handle");
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

#[test]
fn test_builder_defaults_to_directed() {
    let handle = ResourceHandle::new().expect("handle");
    let entered = handle.enter("test_builder_defaults_to_directed").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");

    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let builder = GraphBuilder::new(&src_view, &dst_view);
    let debug = format!("{:?}", builder);

    // properties defaults to directed simple graph
    assert!(debug.contains("is_symmetric: false"), "should default to directed");
    assert!(debug.contains("is_multigraph: false"), "should default to simple graph");
}

#[test]
fn test_from_raw_views_preserves_known_dtypes_in_builder_state() {
    let handle = ResourceHandle::new().expect("handle");
    let entered = handle.enter("test_from_raw_views_preserves_known_dtypes_in_builder_state").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, &[1.0f64, 2.0, 3.0]).expect("wgt");

    let src_raw = src.raw_view().expect("src_raw");
    let dst_raw = dst.raw_view().expect("dst_raw");
    let wgt_raw = wgt.raw_view().expect("wgt_raw");

    let builder = GraphBuilder::from_raw_views(GraphProperties::default(), &src_raw, &dst_raw)
        .expect("builder from raw views")
        .weights_raw(&wgt_raw)
        .expect("weights_raw");

    assert_eq!(builder.vertex_dtype, DataType::Int32);
    assert_eq!(builder.weight_dtype, DataType::Float64);
}
