//! Multi-GPU integration tests.
//!
//! These tests run by default on hosts with 2+ visible GPUs and skip cleanly
//! on single-GPU hosts. Set `CUDF_SKIP_MULTI_GPU=1` to disable them
//! explicitly.

mod common;

use cudf::{
    DataFrame, DeviceId, Runtime, Series, SortOptions, core::memory::DeviceMemoryResource, runtime::CudaStream,
};

const SKIP_MULTI_GPU_ENV: &str = "CUDF_SKIP_MULTI_GPU";

fn should_run_multi_gpu_test() -> bool {
    if std::env::var(SKIP_MULTI_GPU_ENV).unwrap_or_default() == "1" {
        eprintln!("skipping multi-GPU test because {SKIP_MULTI_GPU_ENV}=1");
        return false;
    }

    let count = DeviceId::count().expect("failed to query visible CUDA devices");
    if count < 2 {
        eprintln!("skipping multi-GPU test because only {count} CUDA device is visible");
        return false;
    }

    true
}

fn available_test_devices() -> Vec<DeviceId> {
    let count = DeviceId::count().expect("failed to query visible CUDA devices");
    (0..count).map(|ordinal| DeviceId::new(ordinal).expect("device count returned an invalid ordinal")).collect()
}

fn first_two_test_devices() -> (DeviceId, DeviceId) {
    let mut devices = available_test_devices().into_iter();
    let first = devices.next().expect("multi-GPU tests require at least 2 devices");
    let second = devices.next().expect("multi-GPU tests require at least 2 devices");
    (first, second)
}

#[test]
fn two_runtimes_on_different_devices() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    assert_eq!(rt0.device(), device0);
    assert_eq!(rt1.device(), device1);
    assert_ne!(rt0.device(), rt1.device());
}

#[test]
fn runtimes_execute_on_every_visible_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    for device in available_test_devices() {
        let rt = Runtime::builder().with_device(device).build().unwrap();
        assert_eq!(rt.device(), device, "runtime should report its pinned device");

        common::reset_exec_observation();
        let series = Series::new_with_runtime::<i32>(&rt, "x", &[device.ordinal() as i32, 99]).unwrap();
        assert_eq!(
            common::last_exec_device(),
            device.ordinal() as i32,
            "runtime-backed column creation should execute on device {}",
            device.ordinal()
        );

        let values: Vec<i32> = series.to_vec::<i32>().unwrap();
        assert_eq!(values, vec![device.ordinal() as i32, 99]);
    }
}

#[test]
fn runtime_operations_on_different_devices_produce_correct_results() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();

    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let s0 = Series::new_with_runtime::<i32>(&rt0, "id", &[3, 1, 4]).unwrap();
    let df0 = DataFrame::from_columns([("id", s0)]).unwrap();
    common::reset_exec_observation();
    let sorted0 = df0.sort_values(["id"], SortOptions::new()).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device0.ordinal() as i32,
        "sort should execute on device {}",
        device0.ordinal()
    );
    let result0: Vec<i32> = sorted0.series("id").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result0, vec![1, 3, 4]);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let s1 = Series::new_with_runtime::<i32>(&rt1, "id", &[8, 2, 5]).unwrap();
    let df1 = DataFrame::from_columns([("id", s1)]).unwrap();
    common::reset_exec_observation();
    let sorted1 = df1.sort_values(["id"], SortOptions::new()).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "sort should execute on device {}",
        device1.ordinal()
    );
    let result1: Vec<i32> = sorted1.series("id").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result1, vec![2, 5, 8]);
}

#[test]
fn device_guard_restores_device_across_scopes() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);
    let before = DeviceId::current().unwrap();
    assert_eq!(before, device0, "test setup should begin on the primary device");

    {
        let _rt1 = Runtime::builder().with_device(device1).build().unwrap();
        // During build, DeviceGuard temporarily switches to the secondary device, then restores.
    }
    let after = DeviceId::current().unwrap();
    assert_eq!(after, before, "DeviceGuard did not restore the previous device");
}

#[test]
fn columns_from_different_devices_have_different_domains() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create a column on the primary ambient device.
    common::reset_exec_observation();
    let s0 = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device0.ordinal() as i32,
        "ambient column should be created on device {}",
        device0.ordinal()
    );

    // Create a column on the secondary device via a pinned runtime.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    common::reset_exec_observation();
    let s1 = Series::new_with_runtime::<i32>(&rt1, "y", &[4, 5, 6]).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "runtime column should be created on device {}",
        device1.ordinal()
    );

    // Mixing columns from different devices into one DataFrame must fail.
    let result = DataFrame::from_columns([("x", s0), ("y", s1)]);
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");

    common::set_current_device(device0);
}

// ---------------------------------------------------------------------------
// Device-pinned execution observation tests
//
// These tests verify that `_with_runtime` operations actually execute on the
// runtime's pinned device, not just the calling thread's ambient device.
// ---------------------------------------------------------------------------

#[test]
fn runtime_pinned_device_is_used_for_execution() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Ensure the ambient device is the primary device.
    let ambient = DeviceId::current().unwrap();
    assert_eq!(ambient, device0, "test setup should begin on the primary device");

    // Create a runtime pinned to the secondary device.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    // Reset observation, then create a column through the runtime.
    common::reset_exec_observation();
    let _series = Series::new_with_runtime::<i32>(&rt1, "x", &[10, 20, 30]).unwrap();

    // The FFI should have executed on the runtime's device, not the ambient one.
    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "expected execution on device {}, but observed device {observed}",
        device1.ordinal()
    );

    // The ambient device should still be the primary device.
    let after = DeviceId::current().unwrap();
    assert_eq!(after, device0, "runtime leaked device state to the calling thread");
}

#[test]
fn runtime_device_flows_through_sort() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_, device1) = first_two_test_devices();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    let ids = Series::new_with_runtime::<i32>(&rt1, "id", &[3, 1, 4]).unwrap();
    let df = DataFrame::from_columns([("id", ids)]).unwrap();

    common::reset_exec_observation();
    let _sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "sort_values should execute on device {}, observed device {observed}",
        device1.ordinal()
    );
}

#[test]
fn runtime_device_flows_through_host_materialization() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_, device1) = first_two_test_devices();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let series = Series::new_with_runtime::<i32>(&rt1, "x", &[10, 20]).unwrap();

    common::reset_exec_observation();
    let _vals: Vec<i32> = series.to_vec::<i32>().unwrap();

    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "to_vec should execute on device {}, observed device {observed}",
        device1.ordinal()
    );
}

#[test]
fn concurrent_runtimes_on_different_devices() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    let handle0 = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = Series::new_with_runtime::<i32>(&rt0, "a", &[1, 2, 3]).unwrap();
        common::last_exec_device()
    });

    let handle1 = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = Series::new_with_runtime::<i32>(&rt1, "b", &[4, 5, 6]).unwrap();
        common::last_exec_device()
    });

    let dev0 = handle0.join().expect("thread 0 panicked");
    let dev1 = handle1.join().expect("thread 1 panicked");

    assert_eq!(dev0, device0.ordinal() as i32, "thread 0 should execute on device {}", device0.ordinal());
    assert_eq!(dev1, device1.ordinal() as i32, "thread 1 should execute on device {}", device1.ordinal());
}

#[test]
fn shared_runtime_concurrent_access() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_, device1) = first_two_test_devices();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let rt1_clone = rt1.clone();

    let handle_a = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = Series::new_with_runtime::<i32>(&rt1, "a", &[1, 2]).unwrap();
        common::last_exec_device()
    });

    let handle_b = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = Series::new_with_runtime::<i32>(&rt1_clone, "b", &[3, 4]).unwrap();
        common::last_exec_device()
    });

    let dev_a = handle_a.join().expect("thread A panicked");
    let dev_b = handle_b.join().expect("thread B panicked");

    assert_eq!(
        dev_a,
        device1.ordinal() as i32,
        "shared runtime thread A should execute on device {}",
        device1.ordinal()
    );
    assert_eq!(
        dev_b,
        device1.ordinal() as i32,
        "shared runtime thread B should execute on device {}",
        device1.ordinal()
    );
}

// ---------------------------------------------------------------------------
// Handle device provenance tests (Phase 3)
// ---------------------------------------------------------------------------

#[test]
fn stream_records_creation_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create a stream on the ambient primary device.
    let stream0 = CudaStream::new().unwrap();
    assert_eq!(stream0.device(), device0, "stream should record the ambient device");

    // Create a stream on the secondary device via RuntimeBuilder's DeviceGuard.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    assert_eq!(rt1.stream().device(), device1, "runtime stream should record the pinned device");
}

#[test]
fn memory_resource_records_creation_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create an MR on the ambient primary device.
    let mr0 = DeviceMemoryResource::cuda().unwrap();
    assert_eq!(mr0.device(), device0, "MR should record the ambient device");

    // Create an MR on the secondary device via RuntimeBuilder's DeviceGuard.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    assert_eq!(rt1.memory_resource().device(), device1, "runtime MR should record the pinned device");
}

// ---------------------------------------------------------------------------
// RuntimeBuilder cross-device validation tests (Phase 4)
// ---------------------------------------------------------------------------

#[test]
fn builder_rejects_cross_device_stream() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create a stream on the primary device.
    let stream0 = CudaStream::new().unwrap();
    assert_eq!(stream0.device(), device0);

    // Try to build a runtime on the secondary device with the primary-device stream.
    let result = Runtime::builder().with_device(device1).with_stream(stream0).build();

    assert!(result.is_err(), "builder should reject cross-device stream");
    let err = result.unwrap_err();
    assert!(
        format!("{err:?}").contains("cross-device") || format!("{err:?}").contains("device"),
        "error should mention device mismatch: {err:?}"
    );
}

#[test]
fn builder_rejects_cross_device_mr() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create an MR on the primary device.
    let mr0 = DeviceMemoryResource::cuda().unwrap();
    assert_eq!(mr0.device(), device0);

    // Try to build a runtime on the secondary device with the primary-device MR.
    let result = Runtime::builder().with_device(device1).with_memory_resource(mr0).build();

    assert!(result.is_err(), "builder should reject cross-device MR");
}

#[test]
fn builder_rejects_mismatched_handle_devices() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create a stream on the primary device and an MR on the secondary device.
    let stream0 = CudaStream::new().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    // Clone the MR from the secondary device's runtime.
    let mr1 = rt1.memory_resource().clone();

    let result = Runtime::builder().with_stream(stream0).with_memory_resource(mr1).build();

    assert!(result.is_err(), "builder should reject mismatched handle devices");
}

#[test]
fn builder_infers_device_from_handles() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_, device1) = first_two_test_devices();

    // Create handles on the secondary device through a runtime.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let stream1 = rt1.stream().clone();
    let mr1 = rt1.memory_resource().clone();
    drop(rt1);

    // Build without explicit with_device — it should infer the device from the handles.
    let rt = Runtime::builder().with_stream(stream1).with_memory_resource(mr1).build().unwrap();
    assert_eq!(rt.device(), device1, "builder should infer the device from the supplied handles");
}

// ---------------------------------------------------------------------------
// ExecutionDomain provenance + borrowed override tests (Phase 5)
// ---------------------------------------------------------------------------

#[test]
fn same_provenance_differs_by_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    // Columns created on different devices should have different provenance.
    let s0 = Series::new_with_runtime::<i32>(&rt0, "x", &[1, 2]).unwrap();
    let s1 = Series::new_with_runtime::<i32>(&rt1, "y", &[3, 4]).unwrap();

    // Attempting to build a DataFrame from columns on different devices should fail.
    let result = DataFrame::from_columns([("x", s0), ("y", s1)]);
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");
}
