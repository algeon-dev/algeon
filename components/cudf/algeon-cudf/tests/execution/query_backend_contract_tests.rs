//! Query-backend substrate contract tests.
//!
//! These tests lock the current library-level execution and host/device
//! boundary semantics before the execution resolver refactor centralizes the
//! implementation.

use std::{
    collections::HashMap,
    sync::{
        Arc, Barrier, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use arrow::{
    array::{
        Array, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array, LargeListBuilder,
        LargeStringArray, LargeStringBuilder, ListArray, ListBuilder, StringArray, StringBuilder, StringViewArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    BoundaryKind, DataFrame, DenseMatrixLayout, DeviceId, ExecutionOverrides, QueryExecutionContext, Runtime, TypeId,
    core::{
        DeviceMemoryResource,
        error::{ErrorCode, ErrorKind, ErrorStatus, FactKeyName, FactValue},
    },
    frame::dataframe::{GpuRecordBatch, HostBatchCursorConfig},
    ops::expr::col,
    runtime::{BindingExtensionInitError, QueryStageKind},
    testing::{self, core::adopt_owned_memory_resource_for_test, prelude::*},
};
use cudf_sys::test_support;
use rapids_interop::{CompletionAuthority, ReadStartAuthority};

use crate::common::{
    dependency_wait_call_count_for_stream, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    lock_event_observation, reset_exec_observation, reset_sync_observation, run_named_cases, sample_runtime,
};

fn assert_frame_residency(df: &DataFrame, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), expected_id);
    assert_eq!(df.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(df.testing_execution_domain_mr_value(), expected_mr);
}

fn stats_memory_resource() -> DeviceMemoryResource {
    let device = DeviceId::current().expect("failed to resolve device for stats memory resource");
    let mut raw = std::ptr::null_mut();
    let status = unsafe { test_support::cudf_test_memory_resource_create_stats(&mut raw) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create stats memory resource");
    // SAFETY: the hook returned one newly owned resource for the current
    // device, and this test transfers its sole ownership to cudf.
    unsafe { adopt_owned_memory_resource_for_test(raw, device) }.expect("failed to adopt stats memory resource")
}

fn allocation_count(resource: &DeviceMemoryResource) -> i64 {
    let mut out = 0;
    let status = unsafe {
        test_support::cudf_test_memory_resource_allocation_count(resource.as_borrowed_sys().as_raw(), &mut out)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read allocation count");
    out
}

fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array")
}

fn sample_record_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int32, false),
        Field::new("value", ArrowDataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as Arc<dyn Array>,
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])) as Arc<dyn Array>,
        ],
    )
    .unwrap()
}

fn record_batch_buffer_memory_size(batch: &RecordBatch) -> usize {
    batch.columns().iter().fold(0usize, |total, column| {
        total.checked_add(column.get_buffer_memory_size()).expect("test RecordBatch buffer memory must fit usize")
    })
}

fn empty_string_array(data_type: &ArrowDataType) -> Arc<dyn Array> {
    match data_type {
        ArrowDataType::Utf8 => Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        ArrowDataType::LargeUtf8 => Arc::new(LargeStringArray::from(Vec::<Option<&str>>::new())),
        ArrowDataType::Utf8View => Arc::new(StringViewArray::from(Vec::<Option<&str>>::new())),
        data_type => panic!("expected a string data type, got {data_type:?}"),
    }
}

fn empty_child_list_gpu_batch(
    runtime: &Runtime,
    child_type: ArrowDataType,
    rows: usize,
    root_validity: Option<Vec<bool>>,
) -> GpuRecordBatch {
    let item = Arc::new(Field::new("item", child_type, true));
    let list = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0; rows + 1])),
        empty_string_array(item.data_type()),
        root_validity.map(NullBuffer::from),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::List(item), true)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(list)]).unwrap();
    let frame = DataFrame::from_record_batch(runtime, &host).unwrap();
    GpuRecordBatch::new(frame, schema).unwrap()
}

fn fixed_size_list_float32_gpu_batch(runtime: &Runtime, values: Vec<Option<f32>>) -> GpuRecordBatch {
    assert_eq!(values.len() % 2, 0);
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, true));
    let fixed = FixedSizeListArray::new(Arc::clone(&item), 2, Arc::new(Float32Array::from(values)), None);
    let schema = Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::FixedSizeList(item, 2), false)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(fixed)]).unwrap();
    let frame = DataFrame::from_record_batch(runtime, &host).unwrap();
    GpuRecordBatch::new(frame, schema).unwrap()
}

fn uniform_list_float32_gpu_batch(
    runtime: &Runtime,
    offsets: Vec<i32>,
    values: Vec<Option<f32>>,
    root_validity: Option<Vec<bool>>,
) -> GpuRecordBatch {
    assert!(!offsets.is_empty());
    let element = Arc::new(Field::new("element", ArrowDataType::Float32, true));
    let lists = ListArray::new(
        Arc::clone(&element),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Float32Array::from(values)),
        root_validity.map(NullBuffer::from),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(element), true)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(lists)]).unwrap();
    let frame = DataFrame::from_record_batch(runtime, &host).unwrap();
    GpuRecordBatch::new(frame, schema).unwrap()
}

struct RetainedLifetimeDropCounter(Arc<AtomicUsize>);

impl Drop for RetainedLifetimeDropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn retained_lifetime_sentinel() -> (Arc<dyn Send + Sync>, Arc<AtomicUsize>) {
    let drops = Arc::new(AtomicUsize::new(0));
    let sentinel: Arc<dyn Send + Sync> = Arc::new(RetainedLifetimeDropCounter(Arc::clone(&drops)));
    (sentinel, drops)
}

struct BindingExtensionDropCounter(Arc<AtomicUsize>);

impl Drop for BindingExtensionDropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct RuntimeRetainingBindingExtension {
    _runtime: Runtime,
    drops: Arc<AtomicUsize>,
}

impl Drop for RuntimeRetainingBindingExtension {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct BlockingBindingExtension {
    destructor_started: Option<mpsc::Sender<()>>,
    destructor_release: Arc<(Mutex<bool>, Condvar)>,
    drops: Arc<AtomicUsize>,
}

struct DestructorReentrantBindingExtension {
    runtime: Arc<Runtime>,
    observed: Arc<Mutex<Option<cudf::Error>>>,
}

struct FactoryCycleA;
struct FactoryCycleB;
struct CrossTypeLiveExtension;
struct CrossTypeConstructorExtension;

struct DestructorCycleA {
    runtime: Arc<Runtime>,
    rendezvous: Arc<Barrier>,
    outcome: mpsc::Sender<String>,
}

struct DestructorCycleB {
    runtime: Arc<Runtime>,
    rendezvous: Arc<Barrier>,
    outcome: mpsc::Sender<String>,
}

impl Drop for DestructorCycleA {
    fn drop(&mut self) {
        self.rendezvous.wait();
        let nested = self.runtime.get_or_try_init_binding_extension(
            "binding_extension_destructor_cycle_a_to_b",
            || -> std::result::Result<DestructorCycleB, ()> {
                panic!("cross-type destructor wait must not run the other factory")
            },
        );
        let outcome = match nested {
            Err(BindingExtensionInitError::Lifecycle(error)) => error.to_string(),
            _ => "unexpected destructor-cycle lookup outcome".to_string(),
        };
        let _ = self.outcome.send(outcome);
        self.rendezvous.wait();
    }
}

impl Drop for DestructorCycleB {
    fn drop(&mut self) {
        self.rendezvous.wait();
        let nested = self.runtime.get_or_try_init_binding_extension(
            "binding_extension_destructor_cycle_b_to_a",
            || -> std::result::Result<DestructorCycleA, ()> {
                panic!("cross-type destructor wait must not run the other factory")
            },
        );
        let outcome = match nested {
            Err(BindingExtensionInitError::Lifecycle(error)) => error.to_string(),
            _ => "unexpected destructor-cycle lookup outcome".to_string(),
        };
        let _ = self.outcome.send(outcome);
        self.rendezvous.wait();
    }
}

impl Drop for DestructorReentrantBindingExtension {
    fn drop(&mut self) {
        let nested = self.runtime.get_or_try_init_binding_extension(
            "binding_extension_nested_destruction",
            || -> std::result::Result<Self, ()> { panic!("destruction-time reinitialization must not run a factory") },
        );
        if let Err(BindingExtensionInitError::Lifecycle(error)) = nested {
            *self.observed.lock().expect("observed destruction error lock") = Some(error);
        }
    }
}

impl Drop for BlockingBindingExtension {
    fn drop(&mut self) {
        if let Some(started) = self.destructor_started.take() {
            started.send(()).expect("signal destructor start");
        }
        let (released, changed) = self.destructor_release.as_ref();
        let mut released = released.lock().expect("destructor release lock");
        while !*released {
            released = changed.wait(released).expect("wait for destructor release");
        }
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn binding_extension_is_shared_only_by_the_exact_execution_binding() {
    let base = sample_runtime();
    let same_binding = base.with_retained_lifetime(Arc::new(())).unwrap();
    let different_binding = sample_runtime().inherit_affinity_from(&base).unwrap();
    assert_eq!(base.binding_id(), same_binding.binding_id());
    assert_eq!(base.affinity_id(), different_binding.affinity_id());
    assert_ne!(base.binding_id(), different_binding.binding_id());

    let factory_calls = Arc::new(AtomicUsize::new(0));
    let first = base
        .get_or_try_init_binding_extension("binding_extension_exact_binding_first", {
            let factory_calls = Arc::clone(&factory_calls);
            move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
            }
        })
        .unwrap();
    let inherited = same_binding
        .get_or_try_init_binding_extension(
            "binding_extension_exact_binding_inherited",
            || -> std::result::Result<BindingExtensionDropCounter, ()> {
                panic!("same-binding hot hit must not execute its factory")
            },
        )
        .unwrap();
    let independent = different_binding
        .get_or_try_init_binding_extension("binding_extension_exact_binding_independent", {
            let factory_calls = Arc::clone(&factory_calls);
            move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
            }
        })
        .unwrap();

    assert!(first.same_generation(&inherited));
    assert!(!first.same_generation(&independent));
    assert_eq!(factory_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn binding_extension_cold_n_way_initialization_runs_factory_exactly_once() {
    let runtime = Arc::new(sample_runtime());
    let worker_count = 8;
    let start = Arc::new(Barrier::new(worker_count));
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::new();
    for _ in 0..worker_count {
        let runtime = Arc::clone(&runtime);
        let start = Arc::clone(&start);
        let factory_calls = Arc::clone(&factory_calls);
        workers.push(std::thread::spawn(move || {
            start.wait();
            runtime
                .get_or_try_init_binding_extension("binding_extension_cold_n_way", || {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(20));
                    Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
                })
                .unwrap()
        }));
    }

    let leases = workers.into_iter().map(|worker| worker.join().expect("extension worker")).collect::<Vec<_>>();
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    assert!(leases.iter().all(|lease| leases[0].same_generation(lease)));
}

fn runtime_retaining_binding_extension_drops_without_a_cycle() {
    let drops = Arc::new(AtomicUsize::new(0));
    let runtime = sample_runtime();
    let affinity = runtime.affinity_anchor();
    let extension = runtime
        .get_or_try_init_binding_extension("runtime_retaining_binding_extension", {
            let drops = Arc::clone(&drops);
            let retained_runtime = runtime.clone();
            move || Ok::<_, ()>(RuntimeRetainingBindingExtension { _runtime: retained_runtime, drops })
        })
        .unwrap();

    drop(runtime);
    assert!(affinity.is_alive(), "the lease and retained Runtime keep the execution domain alive");
    drop(extension);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(!affinity.is_alive(), "the extension's retained Runtime must not form a domain cycle");
}

#[test]
fn binding_extension_reinitialization_waits_for_blocked_destructor() {
    let runtime = Arc::new(sample_runtime());
    let (started_tx, started_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let drops = Arc::new(AtomicUsize::new(0));
    let first = runtime
        .get_or_try_init_binding_extension("binding_extension_blocked_destructor_first", {
            let release = Arc::clone(&release);
            let drops = Arc::clone(&drops);
            move || {
                Ok::<_, ()>(BlockingBindingExtension {
                    destructor_started: Some(started_tx),
                    destructor_release: release,
                    drops,
                })
            }
        })
        .unwrap();

    let drop_worker = std::thread::spawn(move || drop(first));
    started_rx.recv_timeout(Duration::from_secs(5)).expect("destructor started");

    let factory_calls = Arc::new(AtomicUsize::new(0));
    let (result_tx, result_rx) = mpsc::channel();
    let reinit_worker = {
        let runtime = Arc::clone(&runtime);
        let release = Arc::clone(&release);
        let drops = Arc::clone(&drops);
        let factory_calls = Arc::clone(&factory_calls);
        std::thread::spawn(move || {
            let result =
                runtime.get_or_try_init_binding_extension("binding_extension_blocked_destructor_reinit", move || {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, ()>(BlockingBindingExtension {
                        destructor_started: None,
                        destructor_release: release,
                        drops,
                    })
                });
            result_tx.send(result).expect("send reinitialization result");
        })
    };

    assert!(
        result_rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "reinitialization must wait until native destruction completes"
    );
    assert_eq!(factory_calls.load(Ordering::SeqCst), 0, "factory must not overlap destruction");

    {
        let (released, changed) = release.as_ref();
        *released.lock().expect("release lock") = true;
        changed.notify_all();
    }
    drop_worker.join().expect("drop worker");
    let second = result_rx.recv_timeout(Duration::from_secs(5)).expect("reinitialization result").unwrap();
    reinit_worker.join().expect("reinit worker");
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    drop(second);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn binding_extension_value_remains_live_while_execution_guard_is_held() {
    let runtime = sample_runtime();
    let drops = Arc::new(AtomicUsize::new(0));
    let first = runtime
        .get_or_try_init_binding_extension("binding_extension_guard_first", {
            let drops = Arc::clone(&drops);
            move || Ok::<_, ()>(BindingExtensionDropCounter(drops))
        })
        .unwrap();
    let active = first.clone();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let entered = active.enter("binding_extension_guard_enter").unwrap();
        entered_tx.send(()).expect("signal entered extension use");
        release_rx.recv().expect("release entered extension use");
        assert!(Arc::strong_count(&entered.value().0) >= 1);
        drop(entered);
        drop(active);
    });

    entered_rx.recv_timeout(Duration::from_secs(5)).expect("extension use entered");
    drop(first);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    release_tx.send(()).expect("release extension use");
    worker.join().expect("entered extension worker");
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn different_binding_extension_execution_gates_do_not_serialize() {
    let first_runtime = sample_runtime();
    let second_runtime = sample_runtime();
    let first = first_runtime
        .get_or_try_init_binding_extension("different_binding_gate_first", || {
            Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
        })
        .unwrap();
    let second = second_runtime
        .get_or_try_init_binding_extension("different_binding_gate_second", || {
            Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
        })
        .unwrap();
    let _first_entered = first.enter("different_binding_gate_first_enter").unwrap();
    let (acquired_tx, acquired_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _second_entered = second.enter("different_binding_gate_second_enter").unwrap();
        acquired_tx.send(()).expect("signal second gate");
    });

    acquired_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("a different binding gate must remain independently enterable");
    worker.join().expect("different binding worker");
}

fn binding_extension_factory_error_is_retryable() {
    let runtime = sample_runtime();
    let first = runtime.get_or_try_init_binding_extension(
        "binding_extension_factory_error",
        || -> std::result::Result<BindingExtensionDropCounter, &'static str> { Err("expected factory error") },
    );
    assert!(matches!(first, Err(BindingExtensionInitError::Factory("expected factory error"))));

    let retry = runtime
        .get_or_try_init_binding_extension("binding_extension_factory_retry", || {
            Ok::<_, &'static str>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
        })
        .expect("factory retry");
    drop(retry);
}

fn binding_extension_reentrancy_and_execution_poison_fail_closed() {
    let runtime = Arc::new(sample_runtime());
    let outer = runtime
        .get_or_try_init_binding_extension("binding_extension_outer_initialization", {
            let runtime = Arc::clone(&runtime);
            move || {
                let nested = runtime
                    .get_or_try_init_binding_extension("binding_extension_nested_initialization", || {
                        Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
                    });
                let error = match nested {
                    Err(BindingExtensionInitError::Lifecycle(error)) => error,
                    _ => panic!("nested initialization must return a lifecycle error"),
                };
                assert!(error.to_string().contains("nested_binding_extension_initialization"));
                Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
            }
        })
        .unwrap();

    let entered = outer.enter("binding_extension_first_entry").unwrap();
    let nested = outer.enter("binding_extension_nested_entry").expect_err("nested entry must fail");
    assert!(nested.to_string().contains("nested_binding_extension_execution"));
    drop(entered);

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _entered = outer.enter("binding_extension_poison_entry").unwrap();
        panic!("intentional execution-gate poison");
    }));
    assert!(unwind.is_err());
    let poisoned = outer.enter("binding_extension_after_poison").expect_err("poison must fail closed");
    assert!(poisoned.to_string().contains("binding_extension_execution_gate_poisoned"));

    drop(outer);
    let next_generation = runtime
        .get_or_try_init_binding_extension("binding_extension_generation_after_gate_poison", || {
            Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
        })
        .expect("lifecycle can create the next generation");
    let still_poisoned = next_generation
        .enter("binding_extension_next_generation_after_gate_poison")
        .expect_err("the stable execution gate remains fail-closed across generations");
    assert!(still_poisoned.to_string().contains("binding_extension_execution_gate_poisoned"));
}

fn binding_extension_poisoned_use_gate_still_admits_value_opaque_teardown() {
    let runtime = sample_runtime();
    let lease = runtime
        .get_or_try_init_binding_extension("binding_extension_teardown_setup", || {
            Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
        })
        .unwrap();
    let native_drops = Arc::new(AtomicUsize::new(0));
    let native_owner = Arc::new(Mutex::new(Some(BindingExtensionDropCounter(Arc::clone(&native_drops)))));

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _entered = lease.enter("binding_extension_poison_before_teardown").unwrap();
        panic!("intentional use-path poison before teardown")
    }));
    assert!(unwind.is_err());

    lease.with_teardown_gate({
        let native_owner = Arc::clone(&native_owner);
        move || drop(native_owner.lock().expect("native owner lock").take())
    });
    lease.with_teardown_gate({
        let native_owner = Arc::clone(&native_owner);
        move || drop(native_owner.lock().expect("native owner lock").take())
    });
    assert_eq!(native_drops.load(Ordering::SeqCst), 1, "native ownership is consumed exactly once");

    let error = lease.enter("binding_extension_normal_use_after_teardown").expect_err("use remains poisoned");
    assert!(error.to_string().contains("binding_extension_execution_gate_poisoned"));
}

#[test]
fn binding_extension_teardown_gate_serializes_after_use_poison() {
    let runtime = sample_runtime();
    let lease = runtime
        .get_or_try_init_binding_extension("binding_extension_teardown_serialization_setup", || {
            Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
        })
        .unwrap();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _entered = lease.enter("binding_extension_teardown_serialization_poison").unwrap();
        panic!("intentional poison for teardown serialization")
    }));
    assert!(unwind.is_err());

    let first = lease.clone();
    let second = lease.clone();
    let (first_entered_tx, first_entered_rx) = mpsc::channel();
    let (release_first_tx, release_first_rx) = mpsc::channel();
    let (second_started_tx, second_started_rx) = mpsc::channel();
    let (second_entered_tx, second_entered_rx) = mpsc::channel();
    let first_worker = std::thread::spawn(move || {
        first.with_teardown_gate(|| {
            first_entered_tx.send(()).expect("signal first teardown entry");
            release_first_rx.recv().expect("release first teardown entry");
        });
    });
    first_entered_rx.recv_timeout(Duration::from_secs(5)).expect("first teardown entered");
    let second_worker = std::thread::spawn(move || {
        second_started_tx.send(()).expect("signal second teardown start");
        second.with_teardown_gate(|| second_entered_tx.send(()).expect("signal second teardown entry"));
    });
    second_started_rx.recv_timeout(Duration::from_secs(5)).expect("second teardown started");
    assert!(
        second_entered_rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "second teardown must wait for the same stable gate"
    );
    release_first_tx.send(()).expect("release first teardown");
    second_entered_rx.recv_timeout(Duration::from_secs(5)).expect("second teardown entered");
    first_worker.join().expect("first teardown worker");
    second_worker.join().expect("second teardown worker");
}

#[test]
fn binding_extension_factory_cross_type_cycle_returns_instead_of_deadlocking() {
    let runtime = Arc::new(sample_runtime());
    let rendezvous = Arc::new(Barrier::new(2));
    let (outcome_tx, outcome_rx) = mpsc::channel();

    let worker_a = {
        let outer_runtime = Arc::clone(&runtime);
        let nested_runtime = Arc::clone(&runtime);
        let rendezvous = Arc::clone(&rendezvous);
        let outcome = outcome_tx.clone();
        std::thread::spawn(move || {
            outer_runtime
                .get_or_try_init_binding_extension("binding_extension_factory_cycle_a", move || {
                    rendezvous.wait();
                    let nested = nested_runtime
                        .get_or_try_init_binding_extension("binding_extension_factory_cycle_a_to_b", || {
                            Ok::<_, ()>(FactoryCycleB)
                        });
                    let message = match nested {
                        Err(BindingExtensionInitError::Lifecycle(error)) => error.to_string(),
                        _ => "unexpected factory-cycle lookup outcome".to_string(),
                    };
                    outcome.send(message).expect("send factory cycle A outcome");
                    rendezvous.wait();
                    Ok::<_, ()>(FactoryCycleA)
                })
                .expect("outer A initialization")
        })
    };
    let worker_b = {
        let outer_runtime = Arc::clone(&runtime);
        let nested_runtime = Arc::clone(&runtime);
        let rendezvous = Arc::clone(&rendezvous);
        let outcome = outcome_tx;
        std::thread::spawn(move || {
            outer_runtime
                .get_or_try_init_binding_extension("binding_extension_factory_cycle_b", move || {
                    rendezvous.wait();
                    let nested = nested_runtime
                        .get_or_try_init_binding_extension("binding_extension_factory_cycle_b_to_a", || {
                            Ok::<_, ()>(FactoryCycleA)
                        });
                    let message = match nested {
                        Err(BindingExtensionInitError::Lifecycle(error)) => error.to_string(),
                        _ => "unexpected factory-cycle lookup outcome".to_string(),
                    };
                    outcome.send(message).expect("send factory cycle B outcome");
                    rendezvous.wait();
                    Ok::<_, ()>(FactoryCycleB)
                })
                .expect("outer B initialization")
        })
    };

    let first = outcome_rx.recv_timeout(Duration::from_secs(5)).expect("first factory-cycle outcome");
    let second = outcome_rx.recv_timeout(Duration::from_secs(5)).expect("second factory-cycle outcome");
    assert!(first.contains("binding_extension_factory_cross_type_wait_forbidden"));
    assert!(second.contains("binding_extension_factory_cross_type_wait_forbidden"));
    drop(worker_a.join().expect("factory cycle A worker"));
    drop(worker_b.join().expect("factory cycle B worker"));
}

fn binding_extension_factory_can_read_live_and_initialize_vacant_different_types() {
    let runtime = Arc::new(sample_runtime());
    let live = runtime
        .get_or_try_init_binding_extension("binding_extension_cross_type_live_setup", || {
            Ok::<_, ()>(CrossTypeLiveExtension)
        })
        .unwrap();
    let read_live = Arc::new(AtomicBool::new(false));
    let initialized_vacant = Arc::new(AtomicBool::new(false));
    let constructor = runtime
        .get_or_try_init_binding_extension("binding_extension_cross_type_constructor", {
            let runtime = Arc::clone(&runtime);
            let expected_live = live;
            let read_live = Arc::clone(&read_live);
            let initialized_vacant = Arc::clone(&initialized_vacant);
            move || {
                let observed_live = runtime
                    .get_or_try_init_binding_extension(
                        "binding_extension_cross_type_read_live",
                        || -> std::result::Result<CrossTypeLiveExtension, ()> {
                            panic!("already-Live different-type factory must not run")
                        },
                    )
                    .expect("read already-Live different type");
                read_live.store(expected_live.same_generation(&observed_live), Ordering::SeqCst);
                let vacant = runtime
                    .get_or_try_init_binding_extension("binding_extension_cross_type_claim_vacant", || {
                        Ok::<_, ()>(FactoryCycleA)
                    })
                    .expect("initialize Vacant different type");
                initialized_vacant.store(true, Ordering::SeqCst);
                drop(vacant);
                Ok::<_, ()>(CrossTypeConstructorExtension)
            }
        })
        .unwrap();

    assert!(read_live.load(Ordering::SeqCst));
    assert!(initialized_vacant.load(Ordering::SeqCst));
    drop(constructor);
}

#[test]
fn binding_extension_destructor_cross_type_cycle_returns_instead_of_deadlocking() {
    let runtime = Arc::new(sample_runtime());
    let rendezvous = Arc::new(Barrier::new(2));
    let (outcome_tx, outcome_rx) = mpsc::channel();
    let lease_a = runtime
        .get_or_try_init_binding_extension("binding_extension_destructor_cycle_a_setup", {
            let runtime = Arc::clone(&runtime);
            let rendezvous = Arc::clone(&rendezvous);
            let outcome = outcome_tx.clone();
            move || Ok::<_, ()>(DestructorCycleA { runtime, rendezvous, outcome })
        })
        .unwrap();
    let lease_b = runtime
        .get_or_try_init_binding_extension("binding_extension_destructor_cycle_b_setup", {
            let runtime = Arc::clone(&runtime);
            let rendezvous = Arc::clone(&rendezvous);
            move || Ok::<_, ()>(DestructorCycleB { runtime, rendezvous, outcome: outcome_tx })
        })
        .unwrap();

    let drop_a = std::thread::spawn(move || drop(lease_a));
    let drop_b = std::thread::spawn(move || drop(lease_b));
    let first = outcome_rx.recv_timeout(Duration::from_secs(5)).expect("first destructor-cycle outcome");
    let second = outcome_rx.recv_timeout(Duration::from_secs(5)).expect("second destructor-cycle outcome");
    assert!(first.contains("binding_extension_destructor_cross_type_wait_forbidden"));
    assert!(second.contains("binding_extension_destructor_cross_type_wait_forbidden"));
    drop_a.join().expect("destructor cycle A worker");
    drop_b.join().expect("destructor cycle B worker");
}

fn binding_extension_same_thread_destructor_reinitialization_is_rejected() {
    let runtime = Arc::new(sample_runtime());
    let observed = Arc::new(Mutex::new(None));
    let lease = runtime
        .get_or_try_init_binding_extension("binding_extension_destructor_reentrancy_setup", {
            let runtime = Arc::clone(&runtime);
            let observed = Arc::clone(&observed);
            move || Ok::<_, ()>(DestructorReentrantBindingExtension { runtime, observed })
        })
        .unwrap();

    drop(lease);
    let error = observed.lock().expect("observed destruction error lock").take().expect("lifecycle error");
    assert!(error.to_string().contains("nested_binding_extension_destruction"));
}

fn binding_extension_factory_panic_poisons_lifecycle() {
    let runtime = sample_runtime();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = runtime.get_or_try_init_binding_extension(
            "binding_extension_panicking_factory",
            || -> std::result::Result<BindingExtensionDropCounter, ()> { panic!("intentional factory panic") },
        );
    }));
    assert!(unwind.is_err());

    let retry = runtime.get_or_try_init_binding_extension("binding_extension_after_factory_panic", || {
        Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
    });
    let error = match retry {
        Err(BindingExtensionInitError::Lifecycle(error)) => error,
        _ => panic!("panicking factory must poison the stable lifecycle slot"),
    };
    assert!(error.to_string().contains("binding_extension_factory_panicked"));
}

fn binding_extension_hot_churn_reuses_one_type_slot() {
    let runtime = sample_runtime();
    let factory_calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..100 {
        let lease = runtime
            .get_or_try_init_binding_extension("binding_extension_hot_churn", {
                let factory_calls = Arc::clone(&factory_calls);
                move || {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, ()>(BindingExtensionDropCounter(Arc::new(AtomicUsize::new(0))))
                }
            })
            .unwrap();
        drop(lease);
    }
    assert_eq!(factory_calls.load(Ordering::SeqCst), 100);
}

fn retained_lifetime_add_unions_distinct_tokens_without_duplicate_growth() {
    let base = sample_runtime();
    assert_eq!(base.testing_retained_lifetime_count(), 0);
    let (token_a, drops_a) = retained_lifetime_sentinel();
    let (token_b, drops_b) = retained_lifetime_sentinel();
    let runtime_a = base.with_retained_lifetime(token_a).unwrap();
    let runtime_ab = runtime_a.with_retained_lifetime(Arc::clone(&token_b)).unwrap();
    let runtime_ab_duplicate = runtime_ab.with_retained_lifetime(token_b).unwrap();

    assert_eq!(runtime_a.testing_retained_lifetime_count(), 1);
    assert_eq!(runtime_ab.testing_retained_lifetime_count(), 2);
    assert_eq!(runtime_ab_duplicate.testing_retained_lifetime_count(), 2);
    let output = runtime_ab_duplicate.series(&[1_i32, 2, 3]).unwrap();
    drop(runtime_a);
    drop(runtime_ab);
    drop(runtime_ab_duplicate);
    assert_eq!(drops_a.load(Ordering::SeqCst), 0);
    assert_eq!(drops_b.load(Ordering::SeqCst), 0);

    drop(output);
    assert_eq!(drops_a.load(Ordering::SeqCst), 1);
    assert_eq!(drops_b.load(Ordering::SeqCst), 1);
}

fn retained_lifetime_inherit_unions_input_and_source_tokens_without_duplicate_growth() {
    let (token_a, drops_a) = retained_lifetime_sentinel();
    let (token_b, drops_b) = retained_lifetime_sentinel();
    let input_a = sample_runtime().with_retained_lifetime(token_a).unwrap();
    let source_b = sample_runtime().with_retained_lifetime(token_b).unwrap();
    let merged = input_a.inherit_retained_lifetime_from(&source_b).unwrap();
    let merged_duplicate = merged.inherit_retained_lifetime_from(&source_b).unwrap();

    assert_eq!(input_a.testing_retained_lifetime_count(), 1);
    assert_eq!(source_b.testing_retained_lifetime_count(), 1);
    assert_eq!(merged.testing_retained_lifetime_count(), 2);
    assert_eq!(merged_duplicate.testing_retained_lifetime_count(), 2);
    let output = merged_duplicate.series(&[1_i32, 2, 3]).unwrap();
    drop(input_a);
    drop(source_b);
    drop(merged);
    drop(merged_duplicate);
    assert_eq!(drops_a.load(Ordering::SeqCst), 0);
    assert_eq!(drops_b.load(Ordering::SeqCst), 0);

    drop(output);
    assert_eq!(drops_a.load(Ordering::SeqCst), 1);
    assert_eq!(drops_b.load(Ordering::SeqCst), 1);
}

fn retained_lifetime_survives_runtime_clone_and_derived_runtime_until_last_owner_drops() {
    let (sentinel, drops) = retained_lifetime_sentinel();
    let source = sample_runtime().with_retained_lifetime(sentinel).unwrap();
    let source_clone = source.clone();
    let derived = sample_runtime().inherit_retained_lifetime_from(&source).unwrap();

    drop(source);
    drop(source_clone);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(derived);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn retained_lifetime_survives_runtime_output_domain_clone_until_output_drop() {
    let (sentinel, drops) = retained_lifetime_sentinel();
    let runtime = sample_runtime().with_retained_lifetime(sentinel).unwrap();
    let output = runtime.series(&[1_i32, 2, 3]).unwrap();

    drop(runtime);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(output);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn retained_lifetime_survives_borrowed_override_until_output_drop() {
    let (sentinel, drops) = retained_lifetime_sentinel();
    let runtime = sample_runtime().with_retained_lifetime(sentinel).unwrap();
    let override_runtime = sample_runtime();
    let overrides = ExecutionOverrides::new()
        .with_stream(override_runtime.stream())
        .with_memory_resource(override_runtime.memory_resource());
    let output = runtime.series_in(&[1_i32, 2, 3], &overrides).unwrap();

    assert_eq!(
        output.as_column().testing_execution_domain_stream_value(),
        override_runtime.stream().testing_raw_handle_value()
    );
    assert_eq!(
        output.as_column().testing_execution_domain_mr_value(),
        override_runtime.memory_resource().testing_raw_handle_value()
    );
    drop(runtime);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(output);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn runtime_affinity_survives_derived_overrides_and_outputs_without_counting_outputs_as_runtime_leases() {
    let base = Arc::new(sample_runtime());
    let affinity_id = base.affinity_id();
    let binding_id = base.binding_id();
    let affinity_anchor = base.affinity_anchor();
    let lease_anchor = base.lease_anchor();
    let derived = Arc::new(base.with_retained_lifetime(Arc::new(())).unwrap());
    let override_runtime = sample_runtime();
    let overrides = ExecutionOverrides::new()
        .with_stream(override_runtime.stream())
        .with_memory_resource(override_runtime.memory_resource());
    let output = derived.series_in(&[1_i32, 2, 3], &overrides).unwrap();

    assert_eq!(derived.affinity_id(), affinity_id);
    assert_eq!(derived.binding_id(), binding_id);
    assert!(Runtime::shares_affinity(&base, &derived));
    assert!(base.matches_lease_anchor(&lease_anchor));
    assert!(derived.matches_lease_anchor(&lease_anchor));
    assert!(lease_anchor.is_current());

    drop(override_runtime);
    drop(base);
    assert!(lease_anchor.is_current(), "derived Runtime keeps the counted owner lease current");
    drop(derived);

    assert!(!lease_anchor.is_current(), "GPU outputs must not count as current Runtime leases");
    assert!(affinity_anchor.is_alive(), "the output execution domain retains affinity identity");
    drop(output);
    assert!(!affinity_anchor.is_alive());
}

fn independent_runtimes_have_distinct_affinity_and_companions_adopt_affinity_not_owner_lease() {
    let base = Arc::new(sample_runtime());
    let independent = Arc::new(sample_runtime());
    assert_ne!(base.affinity_id(), independent.affinity_id());
    assert!(!Runtime::shares_affinity(&base, &independent));

    let base_lease = base.lease_anchor();
    let companion = Arc::new(independent.inherit_affinity_from(&base).unwrap());
    assert_eq!(companion.affinity_id(), base.affinity_id());
    assert_ne!(companion.binding_id(), base.binding_id());
    assert!(Runtime::shares_affinity(&base, &companion));
    assert!(!companion.matches_lease_anchor(&base_lease));

    drop(base);
    assert!(!base_lease.is_current(), "a companion has its own counted owner lease");
    assert!(companion.affinity_anchor().is_alive());
}

fn host_record_batch_import_and_export_are_observable_boundaries() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let runtime_stream = runtime.stream().testing_raw_handle_value();
    let runtime_mr = runtime.memory_resource().testing_raw_handle_value();

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let import_count_before = testing::arrow::arrow_host_import_call_count_for_test();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let import_count_after = testing::arrow::arrow_host_import_call_count_for_test();
    let import_observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();

    assert!(import_count_after > import_count_before);
    assert_eq!(import_observation.kind, BoundaryKind::HostToDevice);
    assert_eq!(import_observation.query_stage, Some(QueryStageKind::InteropImport));
    assert_eq!(import_observation.placement, runtime.placement());
    assert_eq!(import_observation.rows, 4);
    assert_eq!(import_observation.columns, 2);
    assert!(!import_observation.synchronized_for_host);
    assert!(import_observation.memory.payload_bytes.unwrap() > 0);
    assert!(
        import_observation.memory.device_allocated_bytes.unwrap() >= import_observation.memory.payload_bytes.unwrap()
    );
    assert_eq!(import_observation.memory.uploaded_bytes, import_observation.memory.payload_bytes);
    assert_eq!(import_observation.memory.downloaded_bytes, None);
    assert_frame_residency(&df, runtime.testing_execution_domain_id(), runtime_stream, runtime_mr);
    assert_eq!(df.column_names(), vec!["id", "value"]);

    reset_sync_observation();
    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let export_count_before = testing::arrow::arrow_host_export_call_count_for_test();
    let exported = df.to_record_batch().unwrap();
    let export_observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();

    assert!(testing::arrow::arrow_host_export_call_count_for_test() > export_count_before);
    assert_eq!(export_observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(export_observation.query_stage, Some(QueryStageKind::FinalMaterialize));
    assert_eq!(export_observation.placement, df.placement());
    assert_eq!(export_observation.rows, 4);
    assert_eq!(export_observation.columns, 2);
    assert!(export_observation.synchronized_for_host);
    assert_eq!(export_observation.memory.downloaded_bytes, export_observation.memory.payload_bytes);
    assert_eq!(export_observation.memory.uploaded_bytes, None);
    assert_eq!(last_sync_stream_value(), runtime_stream);
    assert_eq!(int32_array(&exported, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(int32_array(&exported, 1).values().as_ref(), &[10, 20, 30, 40]);
}

fn runtime_boundary_observer_receives_opt_in_observations() {
    let _observation_lock = lock_event_observation();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();

    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let _exported = df.to_record_batch().unwrap();

    let observations = observed.lock().unwrap();
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].kind, BoundaryKind::HostToDevice);
    assert_eq!(observations[0].query_stage, Some(QueryStageKind::InteropImport));
    assert_eq!(observations[0].placement, runtime.placement());
    assert_eq!(observations[0].rows, 4);
    assert_eq!(observations[0].columns, 2);
    assert!(!observations[0].synchronized_for_host);
    assert_eq!(observations[1].kind, BoundaryKind::DeviceToHost);
    assert_eq!(observations[1].query_stage, Some(QueryStageKind::FinalMaterialize));
    assert_eq!(observations[1].placement, df.placement());
    assert_eq!(observations[1].rows, 4);
    assert_eq!(observations[1].columns, 2);
    assert!(observations[1].synchronized_for_host);
}

fn dataframe_query_stage_memory_observation_is_explicit_and_observable() {
    let _observation_lock = lock_event_observation();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let observation = df.observe_query_stage(QueryStageKind::OperatorOutput).unwrap();
    let recorded = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(observation.kind, BoundaryKind::QueryStage);
    assert_eq!(observation.query_stage, Some(QueryStageKind::OperatorOutput));
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
    assert_eq!(observation.memory.uploaded_bytes, None);
    assert_eq!(observation.memory.downloaded_bytes, None);
    assert!(observation.memory.payload_bytes.unwrap() > 0);
    assert!(observation.memory.device_allocated_bytes.unwrap() >= observation.memory.payload_bytes.unwrap());
    assert_eq!(recorded, observation);
    assert_eq!(*observed.lock().unwrap().last().unwrap(), observation);

    let memory = df.query_memory_observation().unwrap();
    assert_eq!(memory.payload_bytes, observation.memory.payload_bytes);
    assert_eq!(memory.device_allocated_bytes, observation.memory.device_allocated_bytes);
    assert_eq!(memory.reservation_bytes, None);
}

fn query_execution_context_packages_runtime_handles_for_query_operations() {
    let _observation_lock = lock_event_observation();
    // Keep these as distinct same-device runtimes so the test proves the query
    // facade replaces stream/MR without changing the input workflow provenance.
    let data_runtime = sample_runtime();
    let query_runtime = sample_runtime();
    let query = query_runtime.query_execution();

    fn assert_query_context(_: &QueryExecutionContext<'_>) {}
    assert_query_context(&query);

    assert_eq!(query.runtime().placement(), query_runtime.placement());
    assert_eq!(query.placement(), query_runtime.placement());
    assert_eq!(query.device(), query_runtime.device());
    assert_eq!(query.stream().testing_raw_handle_value(), query_runtime.stream().testing_raw_handle_value());
    assert_eq!(
        query.memory_resource().testing_raw_handle_value(),
        query_runtime.memory_resource().testing_raw_handle_value()
    );
    assert_eq!(query.execution_binding().unwrap().device(), query_runtime.device().ordinal() as i32);

    let imported = DataFrame::from_record_batch(&data_runtime, &sample_record_batch()).unwrap();
    reset_exec_observation();
    let filtered = imported.with_execution(query.overrides()).filter_expr(col("id").ge(2i32)).unwrap();

    assert_eq!(last_exec_stream_value(), query_runtime.stream().testing_raw_handle_value());
    assert_eq!(last_exec_mr_value(), query_runtime.memory_resource().testing_raw_handle_value());
    let filtered_placement = filtered.placement();
    assert_eq!(filtered_placement.device, query_runtime.device());
    assert_eq!(filtered_placement.workflow_id, imported.placement().workflow_id);
    assert_eq!(filtered_placement.stream_debug, query_runtime.stream().testing_raw_handle_value());
    assert_eq!(filtered_placement.memory_resource_debug, query_runtime.memory_resource().testing_raw_handle_value());
}

fn gpu_record_batch_wraps_frame_schema_and_keeps_host_materialization_explicit() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let host = sample_record_batch();
    let schema = host.schema();
    let df = DataFrame::from_record_batch(&runtime, &host).unwrap();

    let gpu = GpuRecordBatch::new(df.clone(), Arc::clone(&schema)).unwrap();

    assert_eq!(gpu.schema().as_ref(), schema.as_ref());
    assert_eq!(gpu.frame().column_names(), vec!["id", "value"]);
    assert_eq!(gpu.num_rows(), 4);
    assert_eq!(gpu.num_columns(), 2);
    assert_eq!(gpu.placement(), df.placement());

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let export_count_before = testing::arrow::arrow_host_export_call_count_for_test();
    let materialized = gpu.to_host_record_batch().unwrap();
    let observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();

    assert!(testing::arrow::arrow_host_export_call_count_for_test() > export_count_before);
    assert_eq!(materialized.schema().as_ref(), schema.as_ref());
    assert_eq!(observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(observation.placement, df.placement());
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(observation.synchronized_for_host);

    let (unwrapped, unwrapped_schema, mapped_schema) = gpu.into_parts();
    assert_eq!(unwrapped.column_names(), vec!["id", "value"]);
    assert_eq!(unwrapped_schema.as_ref(), schema.as_ref());
    assert_eq!(mapped_schema.row_count(), 4);
    assert_eq!(mapped_schema.len(), 2);
}

fn host_batch_cursor_materializes_bounded_chunks_without_losing_rows_or_schema() {
    let runtime = sample_runtime();
    let host = sample_record_batch();
    let schema = host.schema();
    let df = DataFrame::from_record_batch(&runtime, &host).unwrap();
    let gpu = GpuRecordBatch::new(df, Arc::clone(&schema)).unwrap();
    let mut cursor = gpu
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: 1,
            max_rows: 2,
            ..Default::default()
        })
        .unwrap();
    let mut chunks = Vec::new();
    while let Some(batch) = cursor.next_batch().unwrap() {
        assert_eq!(batch.schema().as_ref(), schema.as_ref());
        assert_eq!(batch.num_rows(), 1, "one row is the unsplittable lower bound");
        chunks.push(batch);
    }

    assert_eq!(chunks.len(), host.num_rows());
    assert_eq!(chunks.iter().map(RecordBatch::num_rows).sum::<usize>(), host.num_rows());
    assert_eq!(cursor.remaining_rows(), 0);
}

fn host_batch_cursor_exports_one_schema_preserving_batch_for_an_empty_gpu_result() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let host = sample_record_batch();
    let schema = host.schema();
    let frame = DataFrame::from_record_batch(&runtime, &host).unwrap();
    let mask = runtime.series(&[false; 4]).unwrap();
    let frame = frame.filter(&mask).unwrap();
    assert_eq!(frame.num_rows(), 0);
    let gpu = GpuRecordBatch::new(frame, Arc::clone(&schema)).unwrap();
    let mut cursor = gpu.into_host_batch_cursor(HostBatchCursorConfig::default()).unwrap();

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    assert_eq!(cursor.remaining_rows(), 0);
    assert!(cursor.has_next_batch());
    let first = cursor.next_batch().unwrap().expect("an initially empty result still has one typed host batch");
    assert_eq!(first.schema().as_ref(), schema.as_ref());
    assert_eq!(first.num_rows(), 0);
    assert_eq!(testing::arrow::arrow_host_export_call_count_for_test(), 1);
    let observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();
    assert_eq!(observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(observation.rows, 0);
    assert_eq!(observation.columns, schema.fields().len());

    assert!(cursor.next_batch().unwrap().is_none());
    assert!(!cursor.has_next_batch());
    assert_eq!(testing::arrow::arrow_host_export_call_count_for_test(), 1);
}

#[test]
fn host_batch_cursor_rejects_four_concurrent_oversized_utf8_rows_before_d2h_and_recovers() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let value = "x".repeat(16 * 1024 * 1024);
    let schema = Arc::new(Schema::new(vec![Field::new("value", ArrowDataType::Utf8, false)]));
    let host = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![value.as_str()])) as Arc<dyn Array>],
    )
    .unwrap();
    let gpu = GpuRecordBatch::new(DataFrame::from_record_batch(&runtime, &host).unwrap(), schema).unwrap();
    let config = HostBatchCursorConfig::default();
    let barrier = Arc::new(Barrier::new(4));

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let workers = (0..4)
        .map(|_| {
            let mut cursor = gpu.clone().into_host_batch_cursor(config).unwrap();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                testing::arrow::reset_arrow_host_interop_observation_for_test();
                barrier.wait();
                let error = cursor
                    .next_batch()
                    .expect_err("a 16 MiB scalar must be rejected before device-to-host materialization");
                assert_eq!(error.code(), ErrorCode::HostMaterializationLimitExceeded);
                assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
                assert_eq!(error.status(), ErrorStatus::Permanent);
                assert_eq!(error.detail_value("reason"), Some("host_materialization_hard_limit_exceeded"));
                assert_eq!(error.detail_value("candidate_rows"), Some("1"));
                assert_eq!(
                    error.detail_value("max_materialization_bytes"),
                    Some(HostBatchCursorConfig::DEFAULT_MAX_MATERIALIZATION_BYTES.to_string().as_str())
                );
                assert!(
                    error.detail_value("estimated_materialization_bytes").unwrap().parse::<usize>().unwrap()
                        > HostBatchCursorConfig::DEFAULT_MAX_MATERIALIZATION_BYTES
                );
                assert!(
                    testing::arrow::last_arrow_host_boundary_observation_for_test().is_none(),
                    "hard-limit rejection must happen before the host export boundary"
                );
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }

    let small_host = sample_record_batch();
    let mut small_cursor =
        GpuRecordBatch::new(DataFrame::from_record_batch(&runtime, &small_host).unwrap(), small_host.schema())
            .unwrap()
            .into_host_batch_cursor(config)
            .unwrap();
    assert_eq!(small_cursor.next_batch().unwrap().unwrap(), small_host);
    assert!(small_cursor.next_batch().unwrap().is_none());
}

fn host_batch_cursor_hard_ceiling_is_inclusive_and_preparation_is_copy_free() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let schema = Arc::new(Schema::new(vec![Field::new("value", ArrowDataType::Utf8, false)]));
    let host = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["boundary-value"])) as Arc<dyn Array>],
    )
    .unwrap();
    let gpu = GpuRecordBatch::new(DataFrame::from_record_batch(&runtime, &host).unwrap(), schema).unwrap();
    let probe = gpu
        .clone()
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: usize::MAX,
            max_materialization_bytes: usize::MAX,
            max_rows: 1,
        })
        .unwrap();
    let exact_materialization_bytes = probe.prepare_next_batch().unwrap().unwrap().estimated_materialization_bytes();

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let mut equal = gpu
        .clone()
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: exact_materialization_bytes,
            max_materialization_bytes: exact_materialization_bytes,
            max_rows: 1,
        })
        .unwrap();
    let prepared = equal.prepare_next_batch().unwrap().unwrap();
    assert_eq!(prepared.estimated_materialization_bytes(), exact_materialization_bytes);
    assert!(testing::arrow::last_arrow_host_boundary_observation_for_test().is_none());
    assert_eq!(equal.materialize_prepared_batch(prepared).unwrap(), host);
    assert_eq!(
        testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap().kind,
        BoundaryKind::DeviceToHost
    );

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let below = gpu
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: exact_materialization_bytes - 1,
            max_materialization_bytes: exact_materialization_bytes - 1,
            max_rows: 1,
        })
        .unwrap();
    let error = below.prepare_next_batch().expect_err("one byte below the exact payload must reject");
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert!(testing::arrow::last_arrow_host_boundary_observation_for_test().is_none());
}

fn host_batch_cursor_estimates_final_string_layouts_before_d2h() {
    let runtime = sample_runtime();
    let values = (0..32).map(|index| (index != 7).then_some("twenty-byte-string!!")).collect::<Vec<_>>();
    let source_schema = Arc::new(Schema::new(vec![Field::new("value", ArrowDataType::Utf8, true)]));
    let source =
        RecordBatch::try_new(Arc::clone(&source_schema), vec![Arc::new(StringArray::from(values)) as Arc<dyn Array>])
            .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &source).unwrap();

    let mut estimates = Vec::new();
    for target_type in [ArrowDataType::Utf8, ArrowDataType::LargeUtf8, ArrowDataType::Utf8View] {
        let schema = Arc::new(Schema::new(vec![Field::new("value", target_type.clone(), true)]));
        let mut cursor = GpuRecordBatch::new(frame.clone(), schema)
            .unwrap()
            .into_host_batch_cursor(HostBatchCursorConfig {
                target_materialization_bytes: usize::MAX,
                max_materialization_bytes: usize::MAX,
                max_rows: usize::MAX,
            })
            .unwrap();
        let prepared = cursor.prepare_next_batch().unwrap().unwrap();
        let estimate = prepared.estimated_materialization_bytes();
        let materialized = cursor.materialize_prepared_batch(prepared).unwrap();
        let actual = record_batch_buffer_memory_size(&materialized);
        assert_eq!(materialized.column(0).data_type(), &target_type);
        assert!(actual <= estimate, "{target_type:?}: actual={actual} estimate={estimate}");
        estimates.push(estimate);
    }

    assert!(estimates[0] < estimates[1], "LargeUtf8 must account for 64-bit offsets");
    assert!(estimates[1] < estimates[2], "Utf8View must account for one 16-byte view per row");
}

fn host_batch_cursor_estimates_large_list_string_offsets_validity_and_children() {
    let runtime = sample_runtime();
    let mut builder = LargeListBuilder::new(LargeStringBuilder::with_capacity(8, 256));
    builder.values().append_value("first child value over twelve bytes");
    builder.values().append_null();
    builder.append(true);
    builder.append(false);
    builder.values().append_value("third child value over twelve bytes");
    builder.values().append_value("fourth child value over twelve bytes");
    builder.append(true);
    let values = builder.finish();
    let schema = Arc::new(Schema::new(vec![Field::new("values", values.data_type().clone(), true)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)]).unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &host).unwrap();
    let mut cursor = GpuRecordBatch::new(frame, schema)
        .unwrap()
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: usize::MAX,
            max_materialization_bytes: usize::MAX,
            max_rows: usize::MAX,
        })
        .unwrap();

    let prepared = cursor.prepare_next_batch().unwrap().unwrap();
    let estimate = prepared.estimated_materialization_bytes();
    let materialized = cursor.materialize_prepared_batch(prepared).unwrap();
    let actual = record_batch_buffer_memory_size(&materialized);

    assert_eq!(materialized.column(0).data_type(), host.column(0).data_type());
    assert!(actual <= estimate, "LargeList<LargeUtf8>: actual={actual} estimate={estimate}");
}

fn host_batch_cursor_materializes_lists_with_zero_active_string_children() {
    let runtime = sample_runtime();

    for child_type in [ArrowDataType::Utf8, ArrowDataType::LargeUtf8, ArrowDataType::Utf8View] {
        let cases = [
            (10, Some(vec![false; 10]), None),
            (10, Some(vec![true; 10]), None),
            (12, Some(vec![true; 12]), Some((1, 10))),
            (0, Some(Vec::new()), None),
        ];

        for (rows, validity, slice) in cases {
            let gpu = empty_child_list_gpu_batch(&runtime, child_type.clone(), rows, validity);
            let gpu = if let Some((offset, length)) = slice {
                GpuRecordBatch::new(gpu.frame().slice(offset, length).unwrap(), Arc::clone(gpu.schema())).unwrap()
            } else {
                gpu
            };
            let expected_rows = slice.map_or(rows, |(_, length)| length);
            let mut cursor = gpu
                .into_host_batch_cursor(HostBatchCursorConfig {
                    target_materialization_bytes: usize::MAX,
                    max_materialization_bytes: usize::MAX,
                    max_rows: usize::MAX,
                })
                .unwrap();
            let prepared = cursor.prepare_next_batch().unwrap().unwrap();
            let estimate = prepared.estimated_materialization_bytes();
            let materialized = cursor.materialize_prepared_batch(prepared).unwrap();
            let actual = record_batch_buffer_memory_size(&materialized);

            assert_eq!(materialized.num_rows(), expected_rows);
            assert_eq!(
                materialized.column(0).data_type(),
                &ArrowDataType::List(Arc::new(Field::new("item", child_type.clone(), true)))
            );
            assert!(actual <= estimate, "{child_type:?}: actual={actual} estimate={estimate}");
        }
    }
}

fn host_batch_cursor_estimates_numeric_list_and_fixed_size_list_buffers() {
    let runtime = sample_runtime();
    let list_item = Arc::new(Field::new("item", ArrowDataType::Int64, false));
    let list = ListArray::new(
        Arc::clone(&list_item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 3, 6])),
        Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60])),
        None,
    );
    let list_schema = Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::List(list_item), false)]));
    let list_batch = RecordBatch::try_new(Arc::clone(&list_schema), vec![Arc::new(list)]).unwrap();

    let fixed_item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let fixed = FixedSizeListArray::new(
        Arc::clone(&fixed_item),
        2,
        Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        None,
    );
    let fixed_schema =
        Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::FixedSizeList(fixed_item, 2), false)]));
    let fixed_batch = RecordBatch::try_new(Arc::clone(&fixed_schema), vec![Arc::new(fixed)]).unwrap();

    for host in [list_batch, fixed_batch] {
        let frame = DataFrame::from_record_batch(&runtime, &host).unwrap();
        let mut cursor = GpuRecordBatch::new(frame, host.schema())
            .unwrap()
            .into_host_batch_cursor(HostBatchCursorConfig {
                target_materialization_bytes: usize::MAX,
                max_materialization_bytes: usize::MAX,
                max_rows: usize::MAX,
            })
            .unwrap();
        let prepared = cursor.prepare_next_batch().unwrap().unwrap();
        let estimate = prepared.estimated_materialization_bytes();
        let materialized = cursor.materialize_prepared_batch(prepared).unwrap();
        let actual = record_batch_buffer_memory_size(&materialized);

        assert_eq!(materialized.schema(), host.schema());
        assert!(actual <= estimate, "{:?}: actual={actual} estimate={estimate}", host.column(0).data_type());
    }

    let nullable_item = Arc::new(Field::new("item", ArrowDataType::Int32, false));
    let nullable = ListArray::new(
        Arc::clone(&nullable_item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 4])),
        Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let physical_schema =
        Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::List(Arc::clone(&nullable_item)), true)]));
    let physical = RecordBatch::try_new(physical_schema, vec![Arc::new(nullable)]).unwrap();
    let fixed_schema =
        Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::FixedSizeList(nullable_item, 2), true)]));
    let frame = DataFrame::from_record_batch(&runtime, &physical).unwrap();
    let mut cursor = GpuRecordBatch::new(frame, fixed_schema)
        .unwrap()
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: usize::MAX,
            max_materialization_bytes: usize::MAX,
            max_rows: usize::MAX,
        })
        .unwrap();
    let prepared = cursor.prepare_next_batch().unwrap().unwrap();
    let estimate = prepared.estimated_materialization_bytes();
    let materialized = cursor.materialize_prepared_batch(prepared).unwrap();

    assert!(record_batch_buffer_memory_size(&materialized) <= estimate);
}

fn host_batch_cursor_estimates_bit_packed_boolean_list_values() {
    let runtime = sample_runtime();
    let item = Arc::new(Field::new("item", ArrowDataType::Boolean, true));
    let list = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 3, 3, 5])),
        Arc::new(BooleanArray::from(vec![Some(true), Some(false), None, Some(true), Some(false)])),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::List(item), true)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(list)]).unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &host).unwrap();
    let mut cursor = GpuRecordBatch::new(frame, schema)
        .unwrap()
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: usize::MAX,
            max_materialization_bytes: usize::MAX,
            max_rows: usize::MAX,
        })
        .unwrap();
    let prepared = cursor.prepare_next_batch().unwrap().unwrap();
    let estimate = prepared.estimated_materialization_bytes();
    let materialized = cursor.materialize_prepared_batch(prepared).unwrap();

    assert_eq!(materialized.column(0).to_data(), host.column(0).to_data());
    assert!(record_batch_buffer_memory_size(&materialized) <= estimate);
}

fn fixed_size_list_structural_proof_is_bounded_and_reused_by_cursor_slices() {
    let runtime = sample_runtime();
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let fixed = FixedSizeListArray::new(
        Arc::clone(&item),
        2,
        Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
        None,
    );
    let schema = Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::FixedSizeList(item, 2), false)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(fixed)]).unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &host).unwrap();

    testing::dataframe::reset_fixed_size_list_structural_proof_build_count_for_test();
    testing::boundary::reset_boundary_observation_for_test();
    let gpu = GpuRecordBatch::new(frame, schema).unwrap();
    assert_eq!(testing::dataframe::fixed_size_list_structural_proof_build_count_for_test(), 1);
    let observation = testing::boundary::last_boundary_observation_for_test().unwrap();
    assert_eq!(observation.kind, BoundaryKind::BoundedControlScalar);
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(observation.synchronized_for_host);
    assert_eq!(observation.memory.downloaded_bytes, Some(2 * std::mem::size_of::<i32>()));

    testing::boundary::reset_boundary_observation_for_test();
    let mut cursor = gpu
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: usize::MAX,
            max_materialization_bytes: usize::MAX,
            max_rows: 1,
        })
        .unwrap();
    let first = cursor.prepare_next_batch().unwrap().unwrap();
    assert_eq!(testing::dataframe::fixed_size_list_structural_proof_build_count_for_test(), 1);
    assert!(testing::boundary::last_boundary_observation_for_test().is_none());
    cursor.materialize_prepared_batch(first).unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let _second = cursor.prepare_next_batch().unwrap().unwrap();
    assert_eq!(
        testing::dataframe::fixed_size_list_structural_proof_build_count_for_test(),
        1,
        "cursor slices must retain the original structural proof instead of rebuilding it"
    );
    assert!(testing::boundary::last_boundary_observation_for_test().is_none());
}

fn fixed_size_list_export_retains_proof_binding_range_and_lease_root() {
    let runtime = sample_runtime();
    let gpu = fixed_size_list_float32_gpu_batch(
        &runtime,
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
    );

    let validated = gpu.try_prepare_fixed_size_list_export(0).unwrap();

    assert_eq!(validated.field_index(), 0);
    assert_eq!(validated.dimension(), 2);
    assert_eq!(validated.rows(), 3);
    assert_eq!(validated.child_len(), 6);
    assert_eq!(validated.child_byte_offset(), validated.child_offset() * std::mem::size_of::<f32>());
    assert_eq!(validated.binding_id(), runtime.binding_id());
    assert_eq!(validated.placement(), runtime.placement());
    let observation = validated.structural_observation().unwrap();
    assert_eq!(observation.kind, BoundaryKind::BoundedControlScalar);
    assert_eq!(observation.memory.downloaded_bytes, Some(2 * std::mem::size_of::<i32>()));
    let finite_observation = validated.finite_observation().unwrap();
    assert_eq!(finite_observation.kind, BoundaryKind::BoundedControlScalar);
    assert_eq!(finite_observation.rows, validated.child_len());
    assert_eq!(finite_observation.columns, 1);
    assert_eq!(finite_observation.placement, runtime.placement());
    assert_eq!(finite_observation.memory.downloaded_bytes, Some(std::mem::size_of::<i32>()));
    assert_eq!(validated.export().column().id(), validated.root_column_id());
    assert!(validated.export().semantic_columns().iter().any(|column| column.id() == validated.child_column_id()));
    let values = validated.child_values_binding();
    assert_eq!(values.id(), validated.child_values_buffer_id());
    assert!(!values.bytes().base_ptr().is_null());
    assert!(values.bytes().byte_len() >= validated.child_byte_offset() + validated.child_len() * 4);

    let lease = validated
        .export()
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    lease.complete().unwrap();
}

fn fixed_size_list_direct_pack_copies_only_active_child_once_on_target_runtime() {
    let _event_observation = lock_event_observation();
    let producer_runtime = sample_runtime();
    let full = fixed_size_list_float32_gpu_batch(
        &producer_runtime,
        vec![Some(f32::NAN), Some(f32::INFINITY), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
    );
    let sliced_frame = full.frame().slice(1, 2).unwrap();
    let sliced = GpuRecordBatch::new(sliced_frame, Arc::clone(full.schema())).unwrap();
    let validated = sliced.try_prepare_fixed_size_list_export(0).unwrap();

    let stats = stats_memory_resource();
    let target_runtime = Runtime::builder().with_memory_resource(stats.clone()).build().unwrap();
    let producer_stream = producer_runtime.stream().testing_raw_handle_value();
    let target_stream = target_runtime.stream().testing_raw_handle_value();
    assert_ne!(producer_stream, target_stream, "test requires a cross-stream pack");
    unsafe { test_support::cudf_test_event_reset_observation() };
    reset_sync_observation();
    let before = allocation_count(&stats);
    let packed = validated.pack_active_child_into_runtime(&target_runtime, DenseMatrixLayout::RowMajor).unwrap();
    let after = allocation_count(&stats);

    assert_eq!(
        after - before,
        1,
        "direct validated-child packing must allocate only the final matrix, not an imported list root, child copy, or gather map"
    );
    assert_eq!(packed.shape(), [2, 2]);
    assert_eq!(packed.layout(), DenseMatrixLayout::RowMajor);
    assert_eq!(packed.binding_id(), target_runtime.binding_id());
    assert_eq!(packed.placement(), target_runtime.placement());
    assert_eq!(
        last_sync_stream_value(),
        usize::MAX,
        "successful direct packing must leave the reset no-synchronization sentinel unchanged"
    );
    assert!(
        dependency_wait_call_count_for_stream(target_stream) >= 1,
        "target stream must wait for the prepared producer"
    );
    assert!(
        dependency_wait_call_count_for_stream(producer_stream) >= 1,
        "producer stream must wait for direct-pack completion before source deallocation"
    );
    assert_eq!(packed.into_column().to_vec::<f32>().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
}

fn fixed_size_list_export_rejects_non_finite_active_child_values() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let runtime = sample_runtime();
        let gpu = fixed_size_list_float32_gpu_batch(&runtime, vec![Some(1.0), Some(invalid), Some(3.0), Some(4.0)]);

        let error = gpu
            .try_prepare_fixed_size_list_export(0)
            .expect_err("proof-bearing FixedSizeList export must reject non-finite child values");

        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(
            error.fact_value(FactKeyName::ValidationReason),
            Some(&FactValue::Str("non_finite_vector_value".into()))
        );
        assert_eq!(
            error.fact_value(FactKeyName::ExecutionBindingId),
            Some(&FactValue::U64(runtime.binding_id().get()))
        );
        assert_eq!(error.fact_value(FactKeyName::ControlPlaneScalarBytes), Some(&FactValue::Usize(4)));
        assert_eq!(error.detail_value("non_finite_count"), Some("1"));
        assert_eq!(error.detail_value("validated_values"), Some("4"));
    }
}

fn fixed_size_list_float32_finite_validation_covers_only_the_active_child_range() {
    let runtime = sample_runtime();
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let fixed = FixedSizeListArray::new(
        Arc::clone(&item),
        2,
        Arc::new(Float32Array::from(vec![f32::NAN, f32::INFINITY, 3.0, 4.0])),
        None,
    );
    let schema = Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::FixedSizeList(item, 2), false)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(fixed)]).unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &host).unwrap().slice(1, 1).unwrap();
    let gpu = GpuRecordBatch::new(frame, schema).unwrap();

    let validated = gpu.try_prepare_fixed_size_list_export(0).unwrap();

    assert_eq!(validated.rows(), 1);
    assert_eq!(validated.child_len(), 2);
    assert_eq!(validated.finite_observation().unwrap().rows, 2);
}

fn fixed_size_list_export_rejects_detached_foreign_proof_identity() {
    let runtime = sample_runtime();
    let producer = fixed_size_list_float32_gpu_batch(&runtime, vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
    let foreign_proof_owner =
        fixed_size_list_float32_gpu_batch(&runtime, vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)]);

    let error = testing::dataframe::trigger_fixed_size_list_proof_identity_mismatch_for_test(
        &producer,
        &foreign_proof_owner,
        0,
    )
    .expect_err("a structural proof must not detach from its producer column");

    assert_eq!(error.code(), ErrorCode::InvariantViolation);
    assert_eq!(error.detail_value("reason"), Some("fixed_size_list_proof_identity_mismatch"));
}

fn fixed_size_list_export_rejects_null_child_values() {
    let runtime = sample_runtime();
    let gpu = fixed_size_list_float32_gpu_batch(&runtime, vec![Some(1.0), None, Some(3.0), Some(4.0)]);

    let error =
        gpu.try_prepare_fixed_size_list_export(0).expect_err("dense vector export must reject nullable child values");

    assert_eq!(error.code(), ErrorCode::NullUnexpected);
    assert_eq!(error.detail_value("reason"), Some("null_vector_value"));
}

fn fixed_size_list_export_rejects_null_vector_rows() {
    let runtime = sample_runtime();
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let fixed = FixedSizeListArray::new(
        Arc::clone(&item),
        2,
        Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        Some(NullBuffer::from(vec![true, false])),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("values", ArrowDataType::FixedSizeList(item, 2), true)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(fixed)]).unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &host).unwrap();
    let gpu = GpuRecordBatch::new(frame, schema).unwrap();

    let error =
        gpu.try_prepare_fixed_size_list_export(0).expect_err("dense vector export must reject null vector rows");

    assert_eq!(error.code(), ErrorCode::NullUnexpected);
    assert_eq!(error.detail_value("reason"), Some("null_vector_row"));
}

fn uniform_list_float32_arrow_ingress_exports_exact_proof_range_and_lease_root() {
    let runtime = sample_runtime();
    let gpu = uniform_list_float32_gpu_batch(
        &runtime,
        vec![0, 2, 4, 6],
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
        None,
    );

    let validated = gpu.try_prepare_uniform_list_export(0).unwrap();

    assert_eq!(validated.field_index(), 0);
    assert_eq!(validated.dimension(), 2);
    assert_eq!(validated.rows(), 3);
    assert_eq!(validated.child_len(), 6);
    assert_eq!(validated.child_byte_offset(), validated.child_offset() * std::mem::size_of::<f32>());
    assert_eq!(validated.binding_id(), runtime.binding_id());
    assert_eq!(validated.placement(), runtime.placement());
    let offsets_observation = validated.offsets_observation();
    assert_eq!(offsets_observation.kind, BoundaryKind::BoundedControlScalar);
    assert_eq!(offsets_observation.rows, 3);
    assert_eq!(offsets_observation.columns, 2);
    assert_eq!(offsets_observation.memory.downloaded_bytes, Some(2 * std::mem::size_of::<i32>()));
    let finite_observation = validated.finite_observation().unwrap();
    assert_eq!(finite_observation.kind, BoundaryKind::BoundedControlScalar);
    assert_eq!(finite_observation.rows, 6);
    assert_eq!(finite_observation.columns, 1);
    assert_eq!(finite_observation.memory.downloaded_bytes, Some(std::mem::size_of::<i32>()));
    assert_eq!(validated.export().column().id(), validated.root_column_id());
    assert!(validated.export().bindings().iter().any(|binding| binding.id() == validated.offsets_buffer_id()));
    assert!(validated.export().semantic_columns().iter().any(|column| column.id() == validated.child_column_id()));
    assert_eq!(validated.child_values_binding().id(), validated.child_values_buffer_id());

    let lease = validated
        .export()
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    lease.complete().unwrap();
}

fn uniform_list_direct_pack_reorders_active_child_with_one_target_allocation() {
    let producer_runtime = sample_runtime();
    let full = uniform_list_float32_gpu_batch(
        &producer_runtime,
        vec![0, 2, 4, 6],
        vec![Some(f32::NAN), Some(f32::INFINITY), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
        None,
    );
    let sliced_frame = full.frame().slice(1, 2).unwrap();
    let sliced = GpuRecordBatch::new(sliced_frame, Arc::clone(full.schema())).unwrap();
    let validated = sliced.try_prepare_uniform_list_export(0).unwrap();

    let stats = stats_memory_resource();
    let target_runtime = Runtime::builder().with_memory_resource(stats.clone()).build().unwrap();
    let before = allocation_count(&stats);
    let packed = validated.pack_active_child_into_runtime(&target_runtime, DenseMatrixLayout::ColumnMajor).unwrap();
    let after = allocation_count(&stats);

    assert_eq!(
        after - before,
        1,
        "column-major direct packing must allocate only its final output and no gather-index buffer"
    );
    assert_eq!(packed.shape(), [2, 2]);
    assert_eq!(packed.layout(), DenseMatrixLayout::ColumnMajor);
    assert_eq!(packed.binding_id(), target_runtime.binding_id());
    assert_eq!(packed.placement(), target_runtime.placement());
    assert_eq!(packed.into_column().to_vec::<f32>().unwrap(), vec![3.0, 5.0, 4.0, 6.0]);
}

fn uniform_list_export_fails_closed_when_empty_dimension_is_unavailable() {
    let runtime = sample_runtime();
    let gpu = uniform_list_float32_gpu_batch(&runtime, vec![0], vec![], None);

    testing::boundary::reset_boundary_observation_for_test();
    let error = gpu
        .try_prepare_uniform_list_export(0)
        .expect_err("empty variable-width List cannot prove a positive dimension");

    assert_eq!(error.code(), ErrorCode::EmptyInput);
    assert_eq!(
        error.fact_value(FactKeyName::ValidationReason),
        Some(&FactValue::Str("uniform_list_dimension_unavailable".into()))
    );
    assert!(testing::boundary::last_boundary_observation_for_test().is_none());
}

fn uniform_list_export_rejects_zero_width_and_ragged_rows() {
    let runtime = sample_runtime();
    let zero_width = uniform_list_float32_gpu_batch(&runtime, vec![0, 0, 0], vec![], None);
    let error = zero_width
        .try_prepare_uniform_list_export(0)
        .expect_err("zero-width rows cannot establish a positive vector dimension");
    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(
        error.fact_value(FactKeyName::ValidationReason),
        Some(&FactValue::Str("non_positive_vector_dimension".into()))
    );

    let ragged = uniform_list_float32_gpu_batch(
        &runtime,
        vec![0, 2, 5],
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
        None,
    );
    let error = ragged
        .try_prepare_uniform_list_export(0)
        .expect_err("ragged rows cannot establish one uniform vector dimension");
    assert_eq!(error.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(
        error.fact_value(FactKeyName::ValidationReason),
        Some(&FactValue::Str("non_uniform_vector_dimension".into()))
    );
    assert_eq!(error.detail_value("minimum_row_width"), Some("2"));
    assert_eq!(error.detail_value("maximum_row_width"), Some("3"));
}

fn uniform_list_export_rejects_root_and_child_nulls() {
    let runtime = sample_runtime();
    let root_null = uniform_list_float32_gpu_batch(
        &runtime,
        vec![0, 2, 4],
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
        Some(vec![true, false]),
    );
    let error =
        root_null.try_prepare_uniform_list_export(0).expect_err("dense vector export must reject null list rows");
    assert_eq!(error.code(), ErrorCode::NullUnexpected);
    assert_eq!(error.detail_value("reason"), Some("null_vector_row"));

    let child_null =
        uniform_list_float32_gpu_batch(&runtime, vec![0, 2, 4], vec![Some(1.0), None, Some(3.0), Some(4.0)], None);
    let error =
        child_null.try_prepare_uniform_list_export(0).expect_err("dense vector export must reject null child values");
    assert_eq!(error.code(), ErrorCode::NullUnexpected);
    assert_eq!(error.detail_value("reason"), Some("null_vector_value"));
}

fn uniform_list_export_rejects_non_finite_active_child_values() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let runtime = sample_runtime();
        let gpu = uniform_list_float32_gpu_batch(
            &runtime,
            vec![0, 2, 4],
            vec![Some(1.0), Some(invalid), Some(3.0), Some(4.0)],
            None,
        );

        let error = gpu
            .try_prepare_uniform_list_export(0)
            .expect_err("uniform-list proof must reject non-finite active child values");

        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(
            error.fact_value(FactKeyName::ValidationReason),
            Some(&FactValue::Str("non_finite_vector_value".into()))
        );
        assert_eq!(
            error.fact_value(FactKeyName::ExecutionBindingId),
            Some(&FactValue::U64(runtime.binding_id().get()))
        );
        assert_eq!(error.fact_value(FactKeyName::ControlPlaneScalarBytes), Some(&FactValue::Usize(4)));
        assert_eq!(error.detail_value("non_finite_count"), Some("1"));
        assert_eq!(error.detail_value("validated_values"), Some("4"));
    }
}

fn uniform_list_export_covers_only_the_sliced_active_child_range() {
    let runtime = sample_runtime();
    let full = uniform_list_float32_gpu_batch(
        &runtime,
        vec![0, 2, 4, 6],
        vec![Some(f32::NAN), Some(f32::INFINITY), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
        None,
    );
    let frame = full.frame().slice(1, 2).unwrap();
    let sliced = GpuRecordBatch::new(frame, Arc::clone(full.schema())).unwrap();

    let validated = sliced.try_prepare_uniform_list_export(0).unwrap();

    assert_eq!(validated.rows(), 2);
    assert_eq!(validated.dimension(), 2);
    assert_eq!(validated.child_len(), 4);
    assert_eq!(validated.offsets_observation().rows, 2);
    assert_eq!(validated.finite_observation().unwrap().rows, 4);
}

fn uniform_list_export_rejects_foreign_offset_proof_identity() {
    let runtime = sample_runtime();
    let producer =
        uniform_list_float32_gpu_batch(&runtime, vec![0, 2, 4], vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)], None);
    let proof_owner = uniform_list_float32_gpu_batch(
        &runtime,
        vec![0, 2, 4],
        vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)],
        None,
    );

    let error = testing::dataframe::trigger_uniform_list_proof_identity_mismatch_for_test(&producer, &proof_owner, 0)
        .expect_err("uniform-list offset proof must not detach from its producer");

    assert_eq!(error.code(), ErrorCode::InvariantViolation);
    assert_eq!(error.detail_value("reason"), Some("uniform_list_proof_identity_mismatch"));
}

fn host_batch_cursor_decimal32_and_decimal64_limits_use_decimal128_destination_width() {
    let runtime = sample_runtime();
    for (storage_type, source_width) in [(TypeId::Decimal32, 4usize), (TypeId::Decimal64, 8usize)] {
        let series = match storage_type {
            TypeId::Decimal32 => runtime
                .series(&[123i32])
                .unwrap()
                .astype(cudf::DataType::fixed_point(storage_type, -2).unwrap())
                .unwrap(),
            TypeId::Decimal64 => runtime
                .series(&[123i64])
                .unwrap()
                .astype(cudf::DataType::fixed_point(storage_type, -2).unwrap())
                .unwrap(),
            _ => unreachable!(),
        };
        let frame = DataFrame::from_series([("amount", series)]).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("amount", ArrowDataType::Decimal128(38, 2), false)]));
        let gpu = GpuRecordBatch::new(frame, schema).unwrap();

        testing::arrow::reset_arrow_host_interop_observation_for_test();
        let cursor = gpu
            .clone()
            .into_host_batch_cursor(HostBatchCursorConfig {
                target_materialization_bytes: 32,
                max_materialization_bytes: 32,
                max_rows: 1,
            })
            .unwrap();
        let error = cursor
            .prepare_next_batch()
            .expect_err("Decimal32/64 must be limited by the widened Decimal128 Arrow destination");
        assert_eq!(error.code(), ErrorCode::HostMaterializationLimitExceeded);
        assert_eq!(error.status(), ErrorStatus::Permanent);
        assert!(
            error.detail_value("estimated_materialization_bytes").unwrap().parse::<usize>().unwrap() > 32,
            "destination estimate must exceed the hard limit"
        );
        assert!(source_width < 32, "source payload alone would have passed this limit");
        assert!(testing::arrow::last_arrow_host_boundary_observation_for_test().is_none());

        let mut materializing = gpu
            .into_host_batch_cursor(HostBatchCursorConfig {
                target_materialization_bytes: usize::MAX,
                max_materialization_bytes: usize::MAX,
                max_rows: 1,
            })
            .unwrap();
        let prepared = materializing.prepare_next_batch().unwrap().unwrap();
        let estimate = prepared.estimated_materialization_bytes();
        let batch = materializing.materialize_prepared_batch(prepared).unwrap();
        assert_eq!(batch.column(0).data_type(), &ArrowDataType::Decimal128(38, 2));
        assert!(record_batch_buffer_memory_size(&batch) <= estimate);
    }
}

fn host_batch_cursor_rejects_oversized_nested_list_row_before_d2h() {
    let runtime = sample_runtime();
    let value = "n".repeat(16 * 1024 * 1024);
    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.values().append_value(value);
    builder.append(true);
    let array = builder.finish();
    let schema = Arc::new(Schema::new(vec![Field::new("values", array.data_type().clone(), false)]));
    let host = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();
    let gpu = GpuRecordBatch::new(DataFrame::from_record_batch(&runtime, &host).unwrap(), schema).unwrap();
    let cursor = gpu.into_host_batch_cursor(HostBatchCursorConfig::default()).unwrap();

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let error = cursor.prepare_next_batch().expect_err("an oversized nested row must reject before D2H");
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(error.detail_value("reason"), Some("host_materialization_hard_limit_exceeded"));
    assert!(testing::arrow::last_arrow_host_boundary_observation_for_test().is_none());
}

fn materialized_host_chunk_does_not_retain_gpu_or_runtime_backing() {
    let runtime = sample_runtime();
    let affinity_anchor = runtime.affinity_anchor();
    let lease_anchor = runtime.lease_anchor();
    let host = sample_record_batch();
    let schema = host.schema();
    let gpu = GpuRecordBatch::new(DataFrame::from_record_batch(&runtime, &host).unwrap(), schema).unwrap();
    let mut cursor = gpu
        .into_host_batch_cursor(HostBatchCursorConfig {
            target_materialization_bytes: usize::MAX,
            max_materialization_bytes: usize::MAX,
            max_rows: usize::MAX,
        })
        .unwrap();

    let materialized = cursor.next_batch().unwrap().unwrap();
    assert_eq!(materialized.num_rows(), host.num_rows());
    assert!(cursor.next_batch().unwrap().is_none());

    drop(cursor);
    drop(runtime);
    assert!(!lease_anchor.is_current(), "a host Arrow chunk must not retain a counted Runtime lease");
    assert!(!affinity_anchor.is_alive(), "a host Arrow chunk must not retain GPU execution-domain backing");

    assert_eq!(materialized.num_rows(), host.num_rows(), "the detached host chunk remains usable");
}

fn gpu_record_batch_host_materialization_uses_stored_arrow_schema_contract() {
    let runtime = sample_runtime();
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("label", ArrowDataType::LargeUtf8, true)
                .with_metadata(HashMap::from([("field_owner".to_string(), "planner".to_string())])),
        ])
        .with_metadata(HashMap::from([("schema_owner".to_string(), "planner".to_string())])),
    );
    let host = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(LargeStringArray::from(vec!["alpha", "beta"])) as Arc<dyn Array>],
    )
    .unwrap();
    let df = DataFrame::from_record_batch(&runtime, &host).unwrap();

    let raw_export = df.to_record_batch().unwrap();
    assert_eq!(raw_export.schema().field(0).data_type(), &ArrowDataType::Utf8);

    let gpu = GpuRecordBatch::new(df, Arc::clone(&schema)).unwrap();
    let materialized = gpu.to_host_record_batch().unwrap();
    let materialized_schema = materialized.schema();

    assert_eq!(materialized_schema.as_ref(), schema.as_ref());
    assert_eq!(materialized_schema.metadata().get("schema_owner").map(String::as_str), Some("planner"));
    assert!(materialized_schema.field(0).is_nullable());
    assert_eq!(materialized_schema.field(0).metadata().get("field_owner").map(String::as_str), Some("planner"));
    assert_eq!(materialized.column(0).data_type(), &ArrowDataType::LargeUtf8);

    let labels = materialized.column(0).as_any().downcast_ref::<LargeStringArray>().unwrap();
    assert_eq!(labels.value(0), "alpha");
    assert_eq!(labels.value(1), "beta");
}

#[test]
fn arrow_host_testing_counts_cross_threads_and_keeps_last_observation_thread_local() {
    let _observation_lock = lock_event_observation();
    let import_count_before = testing::arrow::arrow_host_import_call_count_for_test();
    let export_count_before = testing::arrow::arrow_host_export_call_count_for_test();
    assert!(testing::arrow::last_arrow_host_boundary_observation_for_test().is_none());

    std::thread::spawn(|| {
        let runtime = sample_runtime();
        let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();

        let _ = df.to_record_batch().unwrap();

        assert_eq!(
            testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap().kind,
            BoundaryKind::DeviceToHost
        );
    })
    .join()
    .unwrap();

    assert!(testing::arrow::arrow_host_import_call_count_for_test() > import_count_before);
    assert!(testing::arrow::arrow_host_export_call_count_for_test() > export_count_before);
    assert!(testing::arrow::last_arrow_host_boundary_observation_for_test().is_none());
}

fn gpu_record_batch_rejects_incompatible_external_schema_before_materialization() {
    let runtime = sample_runtime();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();

    let wrong_type_schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int64, false),
        Field::new("value", ArrowDataType::Int32, false),
    ]));
    let wrong_type =
        GpuRecordBatch::new(df.clone(), wrong_type_schema).expect_err("schema dtype mismatch should be rejected");
    assert_eq!(wrong_type.code(), cudf::ErrorCode::UnsupportedDtype);
    assert_eq!(wrong_type.operation(), "dataframe::GpuRecordBatch::new");
    assert_eq!(wrong_type.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(wrong_type.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Int64)));
    assert_eq!(wrong_type.detail_value("field_name"), Some("id"));
    assert_eq!(wrong_type.detail_value("field_index"), Some("0"));
    assert_eq!(wrong_type.detail_value("frame_dtype_scale"), Some("0"));
    assert_eq!(wrong_type.detail_value("schema_cudf_dtype_scale"), Some("0"));
    assert_eq!(wrong_type.detail_value("frame_dtype"), None);
    assert_eq!(wrong_type.detail_value("schema_cudf_dtype"), None);
    assert_eq!(wrong_type.detail_value("reason"), Some("schema_dtype_mismatch"));

    let wrong_name_schema = Arc::new(Schema::new(vec![
        Field::new("other", ArrowDataType::Int32, false),
        Field::new("value", ArrowDataType::Int32, false),
    ]));
    let wrong_name = GpuRecordBatch::new(df.clone(), wrong_name_schema)
        .expect_err("schema field names must match DataFrame columns in order");
    assert_eq!(wrong_name.code(), cudf::ErrorCode::MissingColumn);
    assert_eq!(wrong_name.operation(), "dataframe::GpuRecordBatch::new");
    assert_eq!(wrong_name.detail_value("field_name"), Some("other"));

    let wrong_count_schema = Arc::new(Schema::new(vec![Field::new("id", ArrowDataType::Int32, false)]));
    let wrong_count =
        GpuRecordBatch::new(df, wrong_count_schema).expect_err("schema field count must match DataFrame column count");
    assert_eq!(wrong_count.operation(), "dataframe::GpuRecordBatch::new");
    assert_eq!(wrong_count.detail_value("resource"), Some("column_name_count"));
}

fn query_style_chain_keeps_one_override_until_host_materialization() {
    let _observation_lock = lock_event_observation();
    let data_runtime = sample_runtime();
    let ctx_runtime = sample_runtime();
    let ctx_stream = ctx_runtime.stream().testing_raw_handle_value();
    let ctx_mr = ctx_runtime.memory_resource().testing_raw_handle_value();
    let ctx =
        ExecutionOverrides::new().with_stream(ctx_runtime.stream()).with_memory_resource(ctx_runtime.memory_resource());

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let import_count_before = testing::arrow::arrow_host_import_call_count_for_test();
    let imported = DataFrame::from_record_batch(&data_runtime, &sample_record_batch()).unwrap();
    assert!(testing::arrow::arrow_host_import_call_count_for_test() > import_count_before);
    assert_frame_residency(
        &imported,
        data_runtime.testing_execution_domain_id(),
        data_runtime.stream().testing_raw_handle_value(),
        data_runtime.memory_resource().testing_raw_handle_value(),
    );

    reset_exec_observation();
    let filtered = imported.with_execution(&ctx).filter_expr(col("id").ge(2i32)).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&filtered, imported.testing_execution_domain_id(), ctx_stream, ctx_mr);

    reset_exec_observation();
    let projected = filtered.with_execution(&ctx).with_columns([("total", col("id") + col("value"))]).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&projected, imported.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(projected.series("total").unwrap().to_vec_strict::<i32>().unwrap(), vec![22, 33, 44]);

    reset_sync_observation();
    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let export_count_before = testing::arrow::arrow_host_export_call_count_for_test();
    let exported = projected.to_record_batch().unwrap();
    let export_observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();

    assert!(testing::arrow::arrow_host_export_call_count_for_test() > export_count_before);
    assert_eq!(export_observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(export_observation.query_stage, Some(QueryStageKind::FinalMaterialize));
    assert_eq!(export_observation.placement, projected.placement());
    assert_eq!(export_observation.rows, 3);
    assert_eq!(export_observation.columns, 3);
    assert!(export_observation.synchronized_for_host);
    assert_eq!(last_sync_stream_value(), ctx_stream);
    assert_eq!(exported.column(0).len(), 3);
    assert_eq!(int32_array(&exported, 0).values().as_ref(), &[2, 3, 4]);
    assert_eq!(int32_array(&exported, 1).values().as_ref(), &[20, 30, 40]);
    assert_eq!(int32_array(&exported, 2).values().as_ref(), &[22, 33, 44]);
}

fn explicit_runtime_relocation_is_an_observable_boundary() {
    let _observation_lock = lock_event_observation();
    let source_runtime = sample_runtime();
    let target_runtime = sample_runtime();
    let df = DataFrame::from_record_batch(&source_runtime, &sample_record_batch()).unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let relocated = df.copy_into_runtime(&target_runtime).unwrap();
    let observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(observation.kind, BoundaryKind::DeviceToDeviceCopy);
    assert_eq!(observation.query_stage, Some(QueryStageKind::OperatorOutput));
    assert_eq!(observation.placement, target_runtime.placement());
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
    assert_eq!(relocated.placement(), target_runtime.placement());
}

fn arrow_c_device_export_and_import_are_observable_boundaries() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = sample_runtime();
    let consumer_runtime = sample_runtime();
    let df = DataFrame::from_record_batch(&producer_runtime, &sample_record_batch()).unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let export = df.try_prepare_device_arrow_export().unwrap();
    let export_observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(export_observation.kind, BoundaryKind::DeviceAbiHandoff);
    assert_eq!(export_observation.query_stage, Some(QueryStageKind::InteropExport));
    assert_eq!(export_observation.placement, df.placement());
    assert_eq!(export_observation.rows, 4);
    assert_eq!(export_observation.columns, 2);
    assert!(!export_observation.synchronized_for_host);

    let foreign = export.into_foreign();
    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &consumer_runtime).unwrap();
    let import_observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(import_observation.kind, BoundaryKind::DeviceAbiHandoff);
    assert_eq!(import_observation.query_stage, Some(QueryStageKind::InteropImport));
    assert_eq!(import_observation.placement, consumer_runtime.placement());
    assert_eq!(import_observation.rows, 4);
    assert_eq!(import_observation.columns, 2);
    assert!(!import_observation.synchronized_for_host);
    assert_eq!(imported.placement(), consumer_runtime.placement());
}

fn prepared_interop_lease_activation_is_an_observable_boundary() {
    let _observation_lock = lock_event_observation();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let export = df.try_prepare_device_table_export().unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    let observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(observation.kind, BoundaryKind::InteropLease);
    assert_eq!(observation.query_stage, Some(QueryStageKind::InteropExport));
    assert_eq!(observation.placement, df.placement());
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
    assert_eq!(*observed.lock().unwrap().last().unwrap(), observation);
    lease.complete().unwrap();
}

fn owned_transfer_export_is_an_observable_boundary() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let source_placement = df.placement();

    testing::boundary::reset_boundary_observation_for_test();
    let transfer = df.try_into_owned_transfer().unwrap();
    let observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(observation.kind, BoundaryKind::OwnedTransfer);
    assert_eq!(observation.query_stage, Some(QueryStageKind::InteropExport));
    assert_eq!(observation.placement, source_placement);
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
    assert_eq!(transfer.table().num_rows(), 4);
    assert_eq!(transfer.table().columns().len(), 2);
}

#[test]
fn query_backend_contract_tests_cases() {
    run_named_cases(&[
        (
            "binding_extension_is_shared_only_by_the_exact_execution_binding",
            binding_extension_is_shared_only_by_the_exact_execution_binding,
        ),
        (
            "runtime_retaining_binding_extension_drops_without_a_cycle",
            runtime_retaining_binding_extension_drops_without_a_cycle,
        ),
        ("binding_extension_factory_error_is_retryable", binding_extension_factory_error_is_retryable),
        (
            "binding_extension_reentrancy_and_execution_poison_fail_closed",
            binding_extension_reentrancy_and_execution_poison_fail_closed,
        ),
        (
            "binding_extension_poisoned_use_gate_still_admits_value_opaque_teardown",
            binding_extension_poisoned_use_gate_still_admits_value_opaque_teardown,
        ),
        (
            "binding_extension_factory_can_read_live_and_initialize_vacant_different_types",
            binding_extension_factory_can_read_live_and_initialize_vacant_different_types,
        ),
        (
            "binding_extension_same_thread_destructor_reinitialization_is_rejected",
            binding_extension_same_thread_destructor_reinitialization_is_rejected,
        ),
        ("binding_extension_factory_panic_poisons_lifecycle", binding_extension_factory_panic_poisons_lifecycle),
        ("binding_extension_hot_churn_reuses_one_type_slot", binding_extension_hot_churn_reuses_one_type_slot),
        (
            "retained_lifetime_add_unions_distinct_tokens_without_duplicate_growth",
            retained_lifetime_add_unions_distinct_tokens_without_duplicate_growth,
        ),
        (
            "retained_lifetime_inherit_unions_input_and_source_tokens_without_duplicate_growth",
            retained_lifetime_inherit_unions_input_and_source_tokens_without_duplicate_growth,
        ),
        (
            "retained_lifetime_survives_runtime_clone_and_derived_runtime_until_last_owner_drops",
            retained_lifetime_survives_runtime_clone_and_derived_runtime_until_last_owner_drops,
        ),
        (
            "retained_lifetime_survives_runtime_output_domain_clone_until_output_drop",
            retained_lifetime_survives_runtime_output_domain_clone_until_output_drop,
        ),
        (
            "retained_lifetime_survives_borrowed_override_until_output_drop",
            retained_lifetime_survives_borrowed_override_until_output_drop,
        ),
        (
            "runtime_affinity_survives_derived_overrides_and_outputs_without_counting_outputs_as_runtime_leases",
            runtime_affinity_survives_derived_overrides_and_outputs_without_counting_outputs_as_runtime_leases,
        ),
        (
            "independent_runtimes_have_distinct_affinity_and_companions_adopt_affinity_not_owner_lease",
            independent_runtimes_have_distinct_affinity_and_companions_adopt_affinity_not_owner_lease,
        ),
        (
            "host_record_batch_import_and_export_are_observable_boundaries",
            host_record_batch_import_and_export_are_observable_boundaries,
        ),
        (
            "runtime_boundary_observer_receives_opt_in_observations",
            runtime_boundary_observer_receives_opt_in_observations,
        ),
        (
            "dataframe_query_stage_memory_observation_is_explicit_and_observable",
            dataframe_query_stage_memory_observation_is_explicit_and_observable,
        ),
        (
            "query_execution_context_packages_runtime_handles_for_query_operations",
            query_execution_context_packages_runtime_handles_for_query_operations,
        ),
        (
            "gpu_record_batch_wraps_frame_schema_and_keeps_host_materialization_explicit",
            gpu_record_batch_wraps_frame_schema_and_keeps_host_materialization_explicit,
        ),
        (
            "host_batch_cursor_materializes_bounded_chunks_without_losing_rows_or_schema",
            host_batch_cursor_materializes_bounded_chunks_without_losing_rows_or_schema,
        ),
        (
            "host_batch_cursor_exports_one_schema_preserving_batch_for_an_empty_gpu_result",
            host_batch_cursor_exports_one_schema_preserving_batch_for_an_empty_gpu_result,
        ),
        (
            "host_batch_cursor_hard_ceiling_is_inclusive_and_preparation_is_copy_free",
            host_batch_cursor_hard_ceiling_is_inclusive_and_preparation_is_copy_free,
        ),
        (
            "host_batch_cursor_estimates_final_string_layouts_before_d2h",
            host_batch_cursor_estimates_final_string_layouts_before_d2h,
        ),
        (
            "host_batch_cursor_estimates_large_list_string_offsets_validity_and_children",
            host_batch_cursor_estimates_large_list_string_offsets_validity_and_children,
        ),
        (
            "host_batch_cursor_materializes_lists_with_zero_active_string_children",
            host_batch_cursor_materializes_lists_with_zero_active_string_children,
        ),
        (
            "host_batch_cursor_estimates_numeric_list_and_fixed_size_list_buffers",
            host_batch_cursor_estimates_numeric_list_and_fixed_size_list_buffers,
        ),
        (
            "host_batch_cursor_estimates_bit_packed_boolean_list_values",
            host_batch_cursor_estimates_bit_packed_boolean_list_values,
        ),
        (
            "fixed_size_list_structural_proof_is_bounded_and_reused_by_cursor_slices",
            fixed_size_list_structural_proof_is_bounded_and_reused_by_cursor_slices,
        ),
        (
            "fixed_size_list_export_retains_proof_binding_range_and_lease_root",
            fixed_size_list_export_retains_proof_binding_range_and_lease_root,
        ),
        (
            "fixed_size_list_direct_pack_copies_only_active_child_once_on_target_runtime",
            fixed_size_list_direct_pack_copies_only_active_child_once_on_target_runtime,
        ),
        (
            "fixed_size_list_export_rejects_non_finite_active_child_values",
            fixed_size_list_export_rejects_non_finite_active_child_values,
        ),
        (
            "fixed_size_list_float32_finite_validation_covers_only_the_active_child_range",
            fixed_size_list_float32_finite_validation_covers_only_the_active_child_range,
        ),
        (
            "fixed_size_list_export_rejects_detached_foreign_proof_identity",
            fixed_size_list_export_rejects_detached_foreign_proof_identity,
        ),
        ("fixed_size_list_export_rejects_null_child_values", fixed_size_list_export_rejects_null_child_values),
        ("fixed_size_list_export_rejects_null_vector_rows", fixed_size_list_export_rejects_null_vector_rows),
        (
            "uniform_list_float32_arrow_ingress_exports_exact_proof_range_and_lease_root",
            uniform_list_float32_arrow_ingress_exports_exact_proof_range_and_lease_root,
        ),
        (
            "uniform_list_direct_pack_reorders_active_child_with_one_target_allocation",
            uniform_list_direct_pack_reorders_active_child_with_one_target_allocation,
        ),
        (
            "uniform_list_export_fails_closed_when_empty_dimension_is_unavailable",
            uniform_list_export_fails_closed_when_empty_dimension_is_unavailable,
        ),
        (
            "uniform_list_export_rejects_zero_width_and_ragged_rows",
            uniform_list_export_rejects_zero_width_and_ragged_rows,
        ),
        ("uniform_list_export_rejects_root_and_child_nulls", uniform_list_export_rejects_root_and_child_nulls),
        (
            "uniform_list_export_rejects_non_finite_active_child_values",
            uniform_list_export_rejects_non_finite_active_child_values,
        ),
        (
            "uniform_list_export_covers_only_the_sliced_active_child_range",
            uniform_list_export_covers_only_the_sliced_active_child_range,
        ),
        (
            "uniform_list_export_rejects_foreign_offset_proof_identity",
            uniform_list_export_rejects_foreign_offset_proof_identity,
        ),
        (
            "host_batch_cursor_decimal32_and_decimal64_limits_use_decimal128_destination_width",
            host_batch_cursor_decimal32_and_decimal64_limits_use_decimal128_destination_width,
        ),
        (
            "host_batch_cursor_rejects_oversized_nested_list_row_before_d2h",
            host_batch_cursor_rejects_oversized_nested_list_row_before_d2h,
        ),
        (
            "materialized_host_chunk_does_not_retain_gpu_or_runtime_backing",
            materialized_host_chunk_does_not_retain_gpu_or_runtime_backing,
        ),
        (
            "gpu_record_batch_host_materialization_uses_stored_arrow_schema_contract",
            gpu_record_batch_host_materialization_uses_stored_arrow_schema_contract,
        ),
        (
            "gpu_record_batch_rejects_incompatible_external_schema_before_materialization",
            gpu_record_batch_rejects_incompatible_external_schema_before_materialization,
        ),
        (
            "query_style_chain_keeps_one_override_until_host_materialization",
            query_style_chain_keeps_one_override_until_host_materialization,
        ),
        (
            "explicit_runtime_relocation_is_an_observable_boundary",
            explicit_runtime_relocation_is_an_observable_boundary,
        ),
        (
            "arrow_c_device_export_and_import_are_observable_boundaries",
            arrow_c_device_export_and_import_are_observable_boundaries,
        ),
        (
            "prepared_interop_lease_activation_is_an_observable_boundary",
            prepared_interop_lease_activation_is_an_observable_boundary,
        ),
        ("owned_transfer_export_is_an_observable_boundary", owned_transfer_export_is_an_observable_boundary),
    ]);
}
