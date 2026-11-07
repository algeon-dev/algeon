use std::{
    ffi::c_void,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use cudf::{
    ErrorCode, Runtime,
    core::column::DeviceBuffer,
    interop::{
        contract::{CompletionAuthority, InteropErrorKind, PreparedColumnExport, ReadStartAuthority},
        cuda_stream_handle,
    },
    runtime::{DeviceEntryGuard, DeviceHealthStatus, observe_device_faults},
    testing::device_safety::{DeviceEntryTestCall, DeviceSafetyTestScope},
};
use cudf_sys::test_support;

use crate::common::lock_event_observation;

#[derive(Default)]
struct CopyGate {
    released: Mutex<bool>,
    condition: Condvar,
}

impl CopyGate {
    fn release(&self) {
        *self.released.lock().unwrap_or_else(|error| error.into_inner()) = true;
        self.condition.notify_all();
    }
}

extern "C" fn wait_for_copy_release(raw: *mut c_void) {
    // SAFETY: submission transfers exactly one Arc reference to this callback.
    let gate = unsafe { Arc::from_raw(raw.cast::<CopyGate>()) };
    let mut released = gate.released.lock().unwrap_or_else(|error| error.into_inner());
    while !*released {
        released = gate.condition.wait(released).unwrap_or_else(|error| error.into_inner());
    }
}

struct PendingCopy(Arc<CopyGate>);

impl Drop for PendingCopy {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn enqueue_read(export: &PreparedColumnExport<'_>, runtime: &Runtime, output: &mut DeviceBuffer) -> PendingCopy {
    let _entry = DeviceEntryGuard::enter(runtime.device(), "interop_cleanup_test::enqueue_read").unwrap();
    let stream = cuda_stream_handle(runtime.stream()).unwrap();
    let binding = &export.bindings()[0];
    let output_ptr = output.as_mut_ptr().unwrap();
    let gate = Arc::new(CopyGate::default());
    let raw = Arc::into_raw(Arc::clone(&gate));
    // SAFETY: the callback owns its Arc, makes no CUDA calls and releases on
    // scope exit. The activated lease orders the source before this stream;
    // both allocations cover the copy and stay held through its completion.
    let status = unsafe {
        test_support::cudf_test_cuda_launch_host_function(
            stream.as_raw(),
            Some(wait_for_copy_release),
            raw.cast_mut().cast(),
        )
    };
    if status != cudf_sys::CUDF_SUCCESS {
        // SAFETY: a failed launch did not consume the callback's Arc.
        let _gate = unsafe { Arc::from_raw(raw) };
        panic!("CUDA callback launch failed with {status}");
    }
    let pending = PendingCopy(gate);
    assert_eq!(
        unsafe {
            test_support::cudf_test_cuda_copy_device_async(
                stream.as_raw(),
                output_ptr,
                binding.bytes().base_ptr(),
                binding.bytes().byte_len(),
            )
        },
        cudf_sys::CUDF_SUCCESS
    );
    pending
}

fn assert_copied_values(runtime: &Runtime, output: &DeviceBuffer) {
    let _entry = DeviceEntryGuard::enter(runtime.device(), "interop_cleanup_test::assert_copied_values").unwrap();
    runtime.stream().synchronize().unwrap();
    let mut values = [0i32; 3];
    // SAFETY: synchronization finished the device copy. These host and device
    // allocations cover the complete fixed-width payload.
    assert_eq!(
        unsafe {
            test_support::cudf_test_cuda_copy_to_host(
                values.as_mut_ptr().cast(),
                output.as_ptr().unwrap(),
                size_of_val(&values),
            )
        },
        cudf_sys::CUDF_SUCCESS
    );
    assert_eq!(values, [1, 2, 3]);
}

struct SynchronizationFaults;

impl Drop for SynchronizationFaults {
    fn drop(&mut self) {
        unsafe {
            test_support::cudf_test_event_allow_synchronize();
            test_support::cudf_test_stream_allow_synchronize();
        }
    }
}

fn stream_sync_calls() -> i32 {
    let mut calls = 0;
    assert_eq!(unsafe { test_support::cudf_test_stream_synchronize_call_count(&mut calls) }, cudf_sys::CUDF_SUCCESS);
    calls
}

fn release_after_fallback<T>(pending: &PendingCopy, cleanup: impl FnOnce() -> T) -> T {
    let baseline = stream_sync_calls();
    thread::scope(|threads| {
        let waiter = threads.spawn(|| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while stream_sync_calls() == baseline && Instant::now() < deadline {
                thread::yield_now();
            }
            let fallback_ran = stream_sync_calls() > baseline;
            pending.0.release();
            fallback_ran
        });
        let result = cleanup();
        assert!(waiter.join().unwrap(), "cleanup must synchronize the pending consumer read");
        result
    })
}

fn borrowed_witness_failure_synchronizes_pending_cross_stream_read() {
    let _observations = lock_event_observation();
    let producer = Runtime::new().unwrap();
    let source = producer.column_from_slice(&[1i32, 2, 3]).unwrap();
    let export = source.try_prepare_device_column_export().unwrap();
    let consumer = Runtime::new().unwrap();
    let mut output = consumer.device_buffer(12).unwrap();
    let stream = cuda_stream_handle(consumer.stream()).unwrap();
    let lease = export
        .try_activate_read_lease(
            ReadStartAuthority::ConsumerStream { stream },
            CompletionAuthority::SameDeviceStream { stream },
        )
        .unwrap();
    let pending = enqueue_read(&export, &consumer, &mut output);
    let faults = SynchronizationFaults;
    unsafe {
        test_support::cudf_test_stream_synchronize_reset();
        test_support::cudf_test_event_fail_synchronize();
    }
    let scope = DeviceSafetyTestScope::install([]);
    release_after_fallback(&pending, || {
        assert_eq!(lease.complete().unwrap_err().kind(), InteropErrorKind::ReadinessUnavailable);
    });
    assert_eq!(scope.health_snapshot(consumer.device()).status(), DeviceHealthStatus::Healthy);
    drop(faults);
    assert_copied_values(&consumer, &output);
}

fn incomplete_drop_finishes_cross_stream_read_before_owner_release() {
    let _observations = lock_event_observation();
    let producer = Runtime::new().unwrap();
    let export = producer.column_from_slice(&[1i32, 2, 3]).unwrap().try_into_prepared_device_column_export().unwrap();
    let owner = Arc::downgrade(export.retained_owner().unwrap());
    let consumer = Runtime::new().unwrap();
    let mut output = consumer.device_buffer(12).unwrap();
    let stream = cuda_stream_handle(consumer.stream()).unwrap();
    let lease = export
        .try_activate_read_lease(
            ReadStartAuthority::ConsumerStream { stream },
            CompletionAuthority::SameDeviceStream { stream },
        )
        .unwrap();
    let pending = enqueue_read(&export, &consumer, &mut output);
    release_after_fallback(&pending, || drop(lease));
    drop(export);
    assert!(owner.upgrade().is_none());
    assert_copied_values(&consumer, &output);
}

fn completion_setup_failure_retains_pending_read_after_device_entry_loss() {
    let _observations = lock_event_observation();
    let producer = Runtime::new().unwrap();
    let export = producer.column_from_slice(&[1i32, 2, 3]).unwrap().try_into_prepared_device_column_export().unwrap();
    let owner = Arc::downgrade(export.retained_owner().unwrap());
    let consumer = Runtime::new().unwrap();
    let mut output = consumer.device_buffer(12).unwrap();
    let stream = cuda_stream_handle(consumer.stream()).unwrap();
    let lease = export
        .try_activate_read_lease(
            ReadStartAuthority::ConsumerStream { stream },
            CompletionAuthority::SameDeviceStream { stream },
        )
        .unwrap();
    let pending = enqueue_read(&export, &consumer, &mut output);
    let scope = DeviceSafetyTestScope::install([DeviceEntryTestCall::GetDevice {
        status: cudf_sys::CUDF_ERROR_CUDA,
        device: 0,
    }]);
    assert_eq!(lease.complete().unwrap_err().kind(), InteropErrorKind::ReadinessUnavailable);
    assert_eq!(scope.health_snapshot(consumer.device()).status(), DeviceHealthStatus::Unhealthy);
    assert_eq!(scope.remaining_call_count(), 0);
    assert_eq!(scope.health_snapshot(consumer.device()).quarantined_teardown_count(), 1);
    drop(export);
    assert!(owner.upgrade().is_some(), "invalid device authority must preserve storage for the blocked copy");
    drop(scope);
    drop(pending);
    assert_copied_values(&consumer, &output);
}

fn incomplete_cleanup_failure_retains_pending_read_and_quarantines_device() {
    let _observations = lock_event_observation();
    let producer = Runtime::new().unwrap();
    let export = producer.column_from_slice(&[1i32, 2, 3]).unwrap().try_into_prepared_device_column_export().unwrap();
    let owner = Arc::downgrade(export.retained_owner().unwrap());
    let consumer = Runtime::new().unwrap();
    let mut output = consumer.device_buffer(12).unwrap();
    let stream = cuda_stream_handle(consumer.stream()).unwrap();
    let lease = export
        .try_activate_read_lease(
            ReadStartAuthority::ConsumerStream { stream },
            CompletionAuthority::SameDeviceStream { stream },
        )
        .unwrap();
    let pending = enqueue_read(&export, &consumer, &mut output);
    let faults = SynchronizationFaults;
    let scope = DeviceSafetyTestScope::install([]);
    unsafe { test_support::cudf_test_stream_fail_synchronize() };
    drop(lease);
    drop(export);
    assert!(owner.upgrade().is_some(), "failed fallback must retain storage while the copy is blocked");
    let health = scope.health_snapshot(consumer.device());
    assert_eq!(health.status(), DeviceHealthStatus::Unhealthy);
    assert_eq!(health.quarantined_teardown_count(), 1);
    assert_eq!(health.first_fault().unwrap().operation(), "interop::ActiveReadLease::drop");
    assert_eq!(
        DeviceEntryGuard::enter(consumer.device(), "interop_cleanup_test::after_failure").unwrap_err().code(),
        ErrorCode::FatalCuda
    );
    drop(faults);
    drop(scope);
    drop(pending);
    assert_copied_values(&consumer, &output);
}

fn completion_cleanup_failure_retains_pending_read_and_quarantines_device() {
    let _observations = lock_event_observation();
    let producer = Runtime::new().unwrap();
    let export = producer.column_from_slice(&[1i32, 2, 3]).unwrap().try_into_prepared_device_column_export().unwrap();
    let owner = Arc::downgrade(export.retained_owner().unwrap());
    let consumer = Runtime::new().unwrap();
    let mut output = consumer.device_buffer(12).unwrap();
    let stream = cuda_stream_handle(consumer.stream()).unwrap();
    let lease = export
        .try_activate_read_lease(
            ReadStartAuthority::ConsumerStream { stream },
            CompletionAuthority::SameDeviceStream { stream },
        )
        .unwrap();
    let pending = enqueue_read(&export, &consumer, &mut output);
    let completion = lease.complete().unwrap();
    drop(export);
    let faults = SynchronizationFaults;
    let scope = DeviceSafetyTestScope::install([]);
    unsafe { test_support::cudf_test_event_fail_synchronize() };
    let observed = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&observed);
    let _observer = observe_device_faults(consumer.device(), move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
        panic!("test cleanup observer failure");
    });
    drop(completion);
    assert_eq!(observed.load(Ordering::SeqCst), 1);
    assert!(owner.upgrade().is_some(), "failed completion must retain storage while the copy is blocked");
    let health = scope.health_snapshot(consumer.device());
    assert_eq!(health.status(), DeviceHealthStatus::Unhealthy);
    assert_eq!(health.quarantined_teardown_count(), 1);
    assert_eq!(health.first_fault().unwrap().operation(), "interop::CompletionWitness::synchronize");
    drop(faults);
    drop(scope);
    drop(pending);
    assert_copied_values(&consumer, &output);
}

pub(crate) const CASES: &[(&str, fn())] = &[
    (
        "incomplete_drop_finishes_cross_stream_read_before_owner_release",
        incomplete_drop_finishes_cross_stream_read_before_owner_release,
    ),
    (
        "completion_setup_failure_retains_pending_read_after_device_entry_loss",
        completion_setup_failure_retains_pending_read_after_device_entry_loss,
    ),
    (
        "borrowed_witness_failure_synchronizes_pending_cross_stream_read",
        borrowed_witness_failure_synchronizes_pending_cross_stream_read,
    ),
    (
        "incomplete_cleanup_failure_retains_pending_read_and_quarantines_device",
        incomplete_cleanup_failure_retains_pending_read_and_quarantines_device,
    ),
    (
        "completion_cleanup_failure_retains_pending_read_and_quarantines_device",
        completion_cleanup_failure_retains_pending_read_and_quarantines_device,
    ),
];
