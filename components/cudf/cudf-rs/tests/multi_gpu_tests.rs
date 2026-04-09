//! Multi-GPU integration tests.
//!
//! These tests run by default on hosts with 2+ visible GPUs and skip cleanly
//! on single-GPU hosts. Set `CUDF_SKIP_MULTI_GPU=1` to disable them
//! explicitly.

mod common;

use cudf::{
    CsvWriteOptions, DataFrame, DeviceId, ErrorKind, ExecutionContext, GroupByAgg, JsonWriteOptions, MeltOptions,
    MergeOptions, ParquetWriteOptions, Runtime, Series, SortOptions, TypeId,
    core::{Column, DeviceBuffer, Scalar, Table, memory::DeviceMemoryResource},
    ops::{
        copying, dictionary,
        expr::{col, lit},
        filter,
        groupby::{self, AggregationRequest, NullPolicy},
        join, replace, search, sorting, strings,
    },
    runtime::CudaStream,
};
use tempfile::TempDir;

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

fn assert_cross_device_relocation_error(err: cudf::Error, operation: &'static str) {
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), operation);
    assert!(
        err.message().contains("cross-device") || err.message().contains("explicit relocation"),
        "expected cross-device relocation guidance, got {:?}",
        err
    );
    assert!(
        err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")),
        "expected explicit relocation remedy in error context, got {:?}",
        err.context()
    );
}

fn cross_device_write_dataframe() -> (Runtime, Runtime, DataFrame) {
    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let target_rt = Runtime::builder().with_device(device1).build().unwrap();
    let df = DataFrame::from_series([
        data_rt.series("id", &[4i32, 5, 6]).unwrap(),
        data_rt.series("value", &[40i32, 50, 60]).unwrap(),
    ])
    .unwrap();
    (data_rt, target_rt, df)
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
        let series = rt.series("x", &[device.ordinal() as i32, 99]).unwrap();
        assert_eq!(
            common::last_exec_device(),
            device.ordinal() as i32,
            "runtime-backed column creation should execute on device {}",
            device.ordinal()
        );

        let values: Vec<i32> = series.to_vec_strict::<i32>().unwrap();
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
    let s0 = rt0.series("id", &[3, 1, 4]).unwrap();
    let df0 = DataFrame::from_series([s0]).unwrap();
    common::reset_exec_observation();
    let sorted0 = df0.sort_values(["id"], SortOptions::new()).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device0.ordinal() as i32,
        "sort should execute on device {}",
        device0.ordinal()
    );
    let result0: Vec<i32> = sorted0.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(result0, vec![1, 3, 4]);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let s1 = rt1.series("id", &[8, 2, 5]).unwrap();
    let df1 = DataFrame::from_series([s1]).unwrap();
    common::reset_exec_observation();
    let sorted1 = df1.sort_values(["id"], SortOptions::new()).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "sort should execute on device {}",
        device1.ordinal()
    );
    let result1: Vec<i32> = sorted1.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(result1, vec![2, 5, 8]);
}

#[test]
fn device_guard_restores_device_across_scopes() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
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

    let _device_guard = common::DeviceRestoreGuard::new();
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
    let s1 = rt1.series("y", &[4, 5, 6]).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "runtime column should be created on device {}",
        device1.ordinal()
    );

    // Mixing columns from different devices into one DataFrame must fail.
    let result = DataFrame::from_series([s0, s1]);
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");
}

#[test]
fn adopted_device_buffer_preserves_creation_device_provenance() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);
    let buffer = DeviceBuffer::from_host_data(&[1i32, 2, 3]).unwrap();

    common::set_current_device(device1);
    let adopted = Column::from_device_buffer(buffer, 3, TypeId::Int32).unwrap();

    common::set_current_device(device0);
    let ambient = Column::from_slice(&[4i32, 5, 6]).unwrap();
    let frame = crate::common::dataframe_from_columns(vec!["adopted".into(), "ambient".into()], vec![adopted, ambient]);
    assert!(
        frame.is_ok(),
        "adopted column should retain ambient-default provenance from device {} even after switching to device {}",
        device0.ordinal(),
        device1.ordinal()
    );
}

#[test]
fn adopted_nullable_device_buffer_preserves_creation_device_provenance() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);
    let data_buffer = DeviceBuffer::from_host_data(&[10i32, 20, 30]).unwrap();
    let mut mask = [0u8; 64];
    mask[0] = 0x05;
    let mask_buffer = DeviceBuffer::from_host_data(&mask).unwrap();

    common::set_current_device(device1);
    let adopted = Column::from_device_buffer_nullable(data_buffer, mask_buffer, 1, 3, TypeId::Int32).unwrap();

    common::set_current_device(device0);
    let ambient = Column::from_slice(&[40i32, 50, 60]).unwrap();
    let frame = crate::common::dataframe_from_columns(vec!["adopted".into(), "ambient".into()], vec![adopted, ambient]);
    assert!(
        frame.is_ok(),
        "nullable adopted column should retain ambient-default provenance from device {} even after switching to device {}",
        device0.ordinal(),
        device1.ordinal()
    );
}

// ---------------------------------------------------------------------------
// Device-pinned execution observation tests
//
// These tests verify that runtime-first high-level factories actually execute
// on the runtime's pinned device, not just the calling thread's ambient device.
// ---------------------------------------------------------------------------

#[test]
fn runtime_pinned_device_is_used_for_execution() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Ensure the ambient device is the primary device.
    let ambient = DeviceId::current().unwrap();
    assert_eq!(ambient, device0, "test setup should begin on the primary device");

    // Create a runtime pinned to the secondary device.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    // Reset observation, then create a column through the runtime.
    common::reset_exec_observation();
    let _series = rt1.series("x", &[10, 20, 30]).unwrap();

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

    let ids = rt1.series("id", &[3, 1, 4]).unwrap();
    let df = DataFrame::from_series([ids]).unwrap();

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
fn runtime_device_flows_through_manual_mask_filter_paths() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let ids = rt1.series("id", &[3, 1, 4]).unwrap();
    let scores = rt1.series("score", &[30.0, 10.0, 40.0]).unwrap();
    let mask = rt1.series("mask", &[true, false, true]).unwrap();
    let df = DataFrame::from_series([ids, scores]).unwrap();

    common::reset_exec_observation();
    let filtered = df.filter(&mask).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "filter should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
    let filtered_ids: Vec<i32> = filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(filtered_ids, vec![3, 4]);

    let ctx = ExecutionContext::new().with_stream(rt1.stream()).with_memory_resource(rt1.memory_resource());
    common::reset_exec_observation();
    let filtered_with_ctx = df.in_context(&ctx).filter(&mask).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "in_context().filter should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
    let filtered_ctx_ids: Vec<i32> = filtered_with_ctx.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(filtered_ctx_ids, vec![3, 4]);

    let ambient_after = DeviceId::current().unwrap();
    assert_eq!(ambient_after, device0, "filter paths should not leak runtime device state to the calling thread");
}

#[test]
fn runtime_gt_scalar_mask_filter_regression() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let ids = rt1.series("id", &[3, 1, 4]).unwrap();
    let scores = rt1.series("score", &[30.0, 10.0, 40.0]).unwrap();
    let df = DataFrame::from_series([ids, scores]).unwrap();

    common::reset_exec_observation();
    let mask = df.series("score").unwrap().gt_scalar(20.0f64).unwrap();
    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "gt_scalar should execute on device {}, observed device {observed}",
        device1.ordinal()
    );

    let plain = df.filter(&mask);
    assert!(plain.is_ok(), "filter() with gt_scalar-derived mask should succeed: {plain:?}");

    let ctx = ExecutionContext::new().with_stream(rt1.stream()).with_memory_resource(rt1.memory_resource());
    let with_ctx = df.in_context(&ctx).filter(&mask);
    assert!(with_ctx.is_ok(), "in_context().filter() with gt_scalar-derived mask should succeed: {with_ctx:?}");
}

#[test]
fn runtime_device_flows_through_host_materialization() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_, device1) = first_two_test_devices();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let series = rt1.series("x", &[10, 20]).unwrap();

    common::reset_exec_observation();
    let _vals: Vec<i32> = series.to_vec_strict::<i32>().unwrap();

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
        let _s = rt0.series("a", &[1, 2, 3]).unwrap();
        common::last_exec_device()
    });

    let handle1 = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = rt1.series("b", &[4, 5, 6]).unwrap();
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
        let _s = rt1.series("a", &[1, 2]).unwrap();
        common::last_exec_device()
    });

    let handle_b = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = rt1_clone.series("b", &[3, 4]).unwrap();
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

    let _device_guard = common::DeviceRestoreGuard::new();
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

    let _device_guard = common::DeviceRestoreGuard::new();
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

    let _device_guard = common::DeviceRestoreGuard::new();
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

    let _device_guard = common::DeviceRestoreGuard::new();
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

    let _device_guard = common::DeviceRestoreGuard::new();
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
    let s0 = rt0.series("x", &[1, 2]).unwrap();
    let s1 = rt1.series("y", &[3, 4]).unwrap();

    // Attempting to build a DataFrame from columns on different devices should fail.
    let result = DataFrame::from_series([s0, s1]);
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");
}

#[test]
fn merge_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2, 3]).unwrap(),
        left_rt.series("val", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[2i32, 3, 4]).unwrap(),
        right_rt.series("score", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();

    let result = left.merge(&right, MergeOptions::new().how_inner().on(["id"]));
    assert!(result.is_err(), "cross-device merge should fail until inputs are explicitly relocated");

    let ctx = ExecutionContext::new().with_stream(left_rt.stream()).with_memory_resource(left_rt.memory_resource());
    let with_ctx = left.in_context(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"]));
    assert!(with_ctx.is_err(), "borrowed context override must not repair cross-device merge");
}

#[test]
fn concat_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left = DataFrame::from_series([left_rt.series("id", &[1i32, 2, 3]).unwrap()]).unwrap();
    let right = DataFrame::from_series([right_rt.series("id", &[4i32, 5, 6]).unwrap()]).unwrap();

    let err = left.concat(&right).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::concat");
}

#[test]
fn merge_split_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left = DataFrame::from_series([
        left_rt.series("id", &[1i32, 2, 3]).unwrap(),
        left_rt.series("left", &[10i32, 20, 30]).unwrap(),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        right_rt.series("id", &[2i32, 3, 4]).unwrap(),
        right_rt.series("right", &[200i32, 300, 400]).unwrap(),
    ])
    .unwrap();

    let err = left.merge_split(&right, MergeOptions::new().how_inner().on(["id"])).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::merge_split");
}

#[test]
fn gather_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let indices_rt = Runtime::builder().with_device(device1).build().unwrap();

    let df = DataFrame::from_series([data_rt.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let indices = indices_rt.series("idx", &[2i32, 0]).unwrap();

    let err = df.gather(&indices).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::gather");
}

#[test]
fn with_column_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let df = DataFrame::from_series([left_rt.series("id", &[1i32, 2, 3]).unwrap()]).unwrap();
    let extra = right_rt.series("extra", &[10i32, 20, 30]).unwrap();

    let err = df.with_column("extra", extra).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::with_column");
}

#[test]
fn assign_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let df = DataFrame::from_series([left_rt.series("id", &[1i32, 2, 3]).unwrap()]).unwrap();
    let extra = right_rt.series("extra", &[10i32, 20, 30]).unwrap();

    let err = df.assign([("extra", extra)]).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::assign");
}

#[test]
fn filter_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let mask_rt = Runtime::builder().with_device(device1).build().unwrap();

    let df = DataFrame::from_series([data_rt.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let mask = mask_rt.series("mask", &[true, false, true]).unwrap();

    let err = df.filter(&mask).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::filter");
}

#[test]
fn filter_isin_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let values_rt = Runtime::builder().with_device(device1).build().unwrap();

    let df = DataFrame::from_series([data_rt.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let values = values_rt.series("values", &[20i32, 30]).unwrap();

    let err = df.filter_isin("id", &values).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::filter_isin");
}

#[test]
fn dataframe_fillna_with_series_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let fill_rt = Runtime::builder().with_device(device1).build().unwrap();

    let df =
        DataFrame::from_series([input_rt.nullable_series("value", &[10i32, 0, 30], &[true, false, true]).unwrap()])
            .unwrap();
    let fill = fill_rt.series("fill", &[99i32, 99, 99]).unwrap();

    let err = df.fillna_with_series("value", &fill).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::fillna_with_series");
}

#[test]
fn series_fillna_with_series_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let fill_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.nullable_series("value", &[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = fill_rt.series("fill", &[99i32, 99, 99]).unwrap();

    let err = input.fillna_with_series(&fill).unwrap_err();
    assert_cross_device_relocation_error(err, "series::fillna_with_series");
}

#[test]
fn series_concat_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left = left_rt.series("id", &[1i32, 2, 3]).unwrap();
    let right = right_rt.series("id", &[4i32, 5, 6]).unwrap();

    let err = Series::concat(&[&left, &right]).unwrap_err();
    assert_cross_device_relocation_error(err, "concatenate::concatenate_columns");
}

#[test]
fn series_add_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left = left_rt.series("left", &[1i32, 2, 3]).unwrap();
    let right = right_rt.series("right", &[10i32, 20, 30]).unwrap();

    let err = left.add(&right).unwrap_err();
    assert_cross_device_relocation_error(err, "column::add");
}

#[test]
fn series_eq_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left = left_rt.series("left", &[1i32, 2, 3]).unwrap();
    let right = right_rt.series("right", &[1i32, 0, 3]).unwrap();

    let err = left.eq(&right).unwrap_err();
    assert_cross_device_relocation_error(err, "column::eq");
}

#[test]
fn series_isin_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let values_rt = Runtime::builder().with_device(device1).build().unwrap();

    let base = data_rt.series("base", &[3i32, 1, 2, 1]).unwrap();
    let values = values_rt.series("values", &[1i32, 3]).unwrap();

    let err = base.isin(&values).unwrap_err();
    assert_cross_device_relocation_error(err, "column_view::isin");
}

#[test]
fn search_contains_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let haystack_rt = Runtime::builder().with_device(device0).build().unwrap();
    let needles_rt = Runtime::builder().with_device(device1).build().unwrap();

    let haystack = haystack_rt.series("haystack", &[10i32, 20, 30]).unwrap();
    let needles = needles_rt.series("needles", &[20i32, 40]).unwrap();

    let err = match search::contains(&haystack.as_column_view().unwrap(), &needles.as_column_view().unwrap()) {
        Ok(_) => panic!("cross-device search::contains should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "search::contains");
}

#[test]
fn replace_nulls_column_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let replacement_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.nullable_series("value", &[1i32, 0, 3], &[true, false, true]).unwrap();
    let replacement = replacement_rt.series("fill", &[10i32, 20, 30]).unwrap();

    let err =
        match replace::replace_nulls_column(&input.as_column_view().unwrap(), &replacement.as_column_view().unwrap()) {
            Ok(_) => panic!("cross-device replace_nulls_column should fail until inputs are explicitly relocated"),
            Err(err) => err,
        };
    assert_cross_device_relocation_error(err, "replace::replace_nulls_column");
}

#[test]
fn replace_nulls_scalar_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let scalar_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.nullable_series("value", &[1i32, 0, 3], &[true, false, true]).unwrap();
    let scalar = scalar_rt.series("seed", &[99i32]).unwrap().sum().unwrap();

    let err = match replace::replace_nulls(&input.as_column_view().unwrap(), &scalar) {
        Ok(_) => panic!("cross-device replace_nulls should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "replace::replace_nulls");
}

#[test]
fn dataframe_fillna_with_scalar_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let df_rt = Runtime::builder().with_device(device0).build().unwrap();
    let scalar_rt = Runtime::builder().with_device(device1).build().unwrap();

    let df =
        DataFrame::from_series([df_rt.nullable_series("value", &[1i32, 0, 3], &[true, false, true]).unwrap()]).unwrap();
    let scalar = scalar_rt.series("seed", &[99i32]).unwrap().sum().unwrap();

    let err = df.fillna_with_scalar("value", &scalar).unwrap_err();
    assert_cross_device_relocation_error(err, "dataframe::fillna_with_scalar");
}

#[test]
fn low_level_join_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left_keys = cudf::core::Table::from_columns(vec![
        left_rt.series("id", &[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let left_payload = cudf::core::Table::from_columns(vec![
        left_rt.series("val", &[10i32, 20, 30]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let right_keys = cudf::core::Table::from_columns(vec![
        right_rt.series("id", &[2i32, 3, 4]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let right_payload = cudf::core::Table::from_columns(vec![
        right_rt.series("val", &[200i32, 300, 400]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();

    let err = match join::inner_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    ) {
        Ok(_) => panic!("cross-device low-level join should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::inner_join");
}

#[test]
fn low_level_left_join_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left_keys = cudf::core::Table::from_columns(vec![
        left_rt.series("id", &[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let right_keys = cudf::core::Table::from_columns(vec![
        right_rt.series("id", &[2i32, 4]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let left_payload = cudf::core::Table::from_columns(vec![]).unwrap();
    let right_payload = cudf::core::Table::from_columns(vec![]).unwrap();

    let err = match join::left_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    ) {
        Ok(_) => panic!("cross-device low-level left_join should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::left_join");
}

#[test]
fn low_level_full_join_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();

    let left_keys = cudf::core::Table::from_columns(vec![
        left_rt.series("id", &[1i32, 2]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let right_keys = cudf::core::Table::from_columns(vec![
        right_rt.series("id", &[2i32, 3]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let left_payload = cudf::core::Table::from_columns(vec![]).unwrap();
    let right_payload = cudf::core::Table::from_columns(vec![]).unwrap();

    let err = match join::full_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    ) {
        Ok(_) => panic!("cross-device low-level full_join should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::full_join");
}

#[test]
fn column_view_map_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let values_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.series("input", &[1i32, 2, 3, 1]).unwrap();
    let keys = input_rt.series("keys", &[1i32, 2, 3]).unwrap();
    let values = values_rt.series("values", &[100i32, 200, 300]).unwrap();

    let err =
        match input.as_column_view().unwrap().map(&keys.as_column_view().unwrap(), &values.as_column_view().unwrap()) {
            Ok(_) => panic!("cross-device ColumnView::map should fail until inputs are explicitly relocated"),
            Err(err) => err,
        };
    assert_cross_device_relocation_error(err, "column_view::map");
}

#[test]
fn sorting_sort_by_key_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let values_rt = Runtime::builder().with_device(device0).build().unwrap();
    let keys_rt = Runtime::builder().with_device(device1).build().unwrap();

    let values = Table::from_columns(vec![
        values_rt.series("value", &[30i32, 10, 20]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let keys =
        Table::from_columns(vec![keys_rt.series("key", &[3i32, 1, 2]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();

    let err = match sorting::sort_by_key(&values.view().unwrap(), &keys.view().unwrap(), None, None) {
        Ok(_) => panic!("cross-device sort_by_key should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "sorting::sort_by_key");
}

#[test]
fn sorting_stable_sort_by_key_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let values_rt = Runtime::builder().with_device(device0).build().unwrap();
    let keys_rt = Runtime::builder().with_device(device1).build().unwrap();

    let values = Table::from_columns(vec![
        values_rt.series("value", &[30i32, 10, 20]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let keys =
        Table::from_columns(vec![keys_rt.series("key", &[3i32, 1, 2]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();

    let err = match sorting::stable_sort_by_key(&values.view().unwrap(), &keys.view().unwrap(), None, None) {
        Ok(_) => panic!("cross-device stable_sort_by_key should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "sorting::stable_sort_by_key");
}

#[test]
fn low_level_filter_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let mask_rt = Runtime::builder().with_device(device1).build().unwrap();

    let table =
        Table::from_columns(vec![data_rt.series("id", &[10i32, 20, 30]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let mask = mask_rt.series("mask", &[true, false, true]).unwrap();

    let err = match filter::apply_boolean_mask(&table.view().unwrap(), &mask.as_column_view().unwrap()) {
        Ok(_) => panic!("cross-device low-level filter should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "filter::apply_boolean_mask");
}

#[test]
fn low_level_gather_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let indices_rt = Runtime::builder().with_device(device1).build().unwrap();

    let table =
        Table::from_columns(vec![data_rt.series("id", &[10i32, 20, 30]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let gather_map = indices_rt.series("idx", &[2i32, 0]).unwrap();

    let err = match copying::gather(&table.view().unwrap(), &gather_map.as_column_view().unwrap()) {
        Ok(_) => panic!("cross-device low-level gather should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "copying::gather");
}

#[test]
fn low_level_gather_unchecked_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let indices_rt = Runtime::builder().with_device(device1).build().unwrap();

    let table =
        Table::from_columns(vec![data_rt.series("id", &[10i32, 20, 30]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let gather_map = indices_rt.series("idx", &[2i32, 0]).unwrap();

    let err = match unsafe { copying::gather_unchecked(&table.view().unwrap(), &gather_map.as_column_view().unwrap()) }
    {
        Ok(_) => panic!("cross-device low-level gather_unchecked should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "copying::gather_unchecked");
}

#[test]
fn dictionary_set_keys_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let dict_rt = Runtime::builder().with_device(device0).build().unwrap();
    let keys_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = dict_rt.series("input", &[3i32, 1, 3, 2]).unwrap();
    let dict = dictionary::encode(&input.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let keys = keys_rt.series("keys", &[2i32, 3, 4]).unwrap();

    let err = match dictionary::set_keys(&dict.view().unwrap(), &keys.as_column_view().unwrap()) {
        Ok(_) => panic!("cross-device dictionary::set_keys should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "dictionary::set_keys");
}

#[test]
fn dictionary_match_dictionaries_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let first_rt = Runtime::builder().with_device(device0).build().unwrap();
    let second_rt = Runtime::builder().with_device(device1).build().unwrap();

    let first_input = first_rt.series("src", &[10i32, 20, 30]).unwrap();
    let second_input = second_rt.series("dst", &[20i32, 30, 40]).unwrap();
    let first = dictionary::encode(&first_input.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let second = dictionary::encode(&second_input.as_column_view().unwrap(), TypeId::Int32).unwrap();

    let err = match dictionary::match_dictionaries(&[&first.view().unwrap(), &second.view().unwrap()]) {
        Ok(_) => {
            panic!("cross-device dictionary::match_dictionaries should fail until inputs are explicitly relocated")
        },
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "dictionary::match_dictionaries");
}

#[test]
fn groupby_aggregate_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let keys_rt = Runtime::builder().with_device(device0).build().unwrap();
    let values_rt = Runtime::builder().with_device(device1).build().unwrap();

    let keys =
        Table::from_columns(vec![keys_rt.series("key", &[1i32, 1, 2, 2]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let values = values_rt.series("value", &[10i32, 20, 30, 40]).unwrap();
    let values_view = values.as_column_view().unwrap();
    let aggregations = [GroupByAgg::Sum];
    let requests = [AggregationRequest { values: &values_view, aggregations: &aggregations }];

    let err = match groupby::groupby_aggregate(&keys.view().unwrap(), &requests, NullPolicy::Exclude) {
        Ok(_) => panic!("cross-device groupby_aggregate should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "groupby::groupby_aggregate");
}

#[test]
fn binary_operation_vs_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let values_rt = Runtime::builder().with_device(device0).build().unwrap();
    let scalar_rt = Runtime::builder().with_device(device1).build().unwrap();

    let values = values_rt.series("value", &[2i32, 3, 4]).unwrap();
    let scalar = scalar_rt.series("seed", &[10i32]).unwrap().sum().unwrap();

    let err = match cudf::ops::binary::binary_operation_vs(
        &values.as_column_view().unwrap(),
        &scalar,
        cudf::ops::binary::BinaryOperator::Mul,
        &cudf::DataType::new(TypeId::Int32),
    ) {
        Ok(_) => panic!("cross-device binary_operation_vs should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "binary::binary_operation_vs");
}

#[test]
fn binary_operation_sv_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let scalar_rt = Runtime::builder().with_device(device0).build().unwrap();
    let values_rt = Runtime::builder().with_device(device1).build().unwrap();

    let scalar = scalar_rt.series("seed", &[100i32]).unwrap().sum().unwrap();
    let values = values_rt.series("value", &[2i32, 4, 5]).unwrap();

    let err = match cudf::ops::binary::binary_operation_sv(
        &scalar,
        &values.as_column_view().unwrap(),
        cudf::ops::binary::BinaryOperator::Div,
        &cudf::DataType::new(TypeId::Int32),
    ) {
        Ok(_) => panic!("cross-device binary_operation_sv should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "binary::binary_operation_sv");
}

#[test]
fn strings_contains_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let scalar_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.str_series("input", &["alpha", "beta", "alphabet"]).unwrap();
    let scalar = scalar_rt.str_series("seed", &["alpha"]).unwrap().min().unwrap();

    let err = match strings::str_contains(&input.as_column_view().unwrap(), &scalar) {
        Ok(_) => panic!("cross-device strings::contains should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "strings::contains");
}

#[test]
fn strings_starts_with_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let scalar_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.str_series("input", &["alpha", "beta", "alphabet"]).unwrap();
    let scalar = scalar_rt.str_series("seed", &["alp"]).unwrap().min().unwrap();

    let err = match strings::starts_with(&input.as_column_view().unwrap(), &scalar) {
        Ok(_) => panic!("cross-device strings::starts_with should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "strings::starts_with");
}

#[test]
fn strings_ends_with_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let scalar_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.str_series("input", &["alpha", "beta", "alphabet"]).unwrap();
    let scalar = scalar_rt.str_series("seed", &["ha"]).unwrap().min().unwrap();

    let err = match strings::ends_with(&input.as_column_view().unwrap(), &scalar) {
        Ok(_) => panic!("cross-device strings::ends_with should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "strings::ends_with");
}

#[test]
fn strings_strip_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let input_rt = Runtime::builder().with_device(device0).build().unwrap();
    let scalar_rt = Runtime::builder().with_device(device1).build().unwrap();

    let input = input_rt.str_series("input", &["xxalpha", "betaxx", "xxalphabetxx"]).unwrap();
    let scalar = scalar_rt.str_series("seed", &["x"]).unwrap().min().unwrap();

    let err = match strings::strip(&input.as_column_view().unwrap(), cudf::StripSide::Both, &scalar) {
        Ok(_) => panic!("cross-device strings::strip should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "strings::strip");
}

// ---------------------------------------------------------------------------
// Additional ambient-seeding regression probes
// ---------------------------------------------------------------------------

#[test]
fn runtime_expr_literal_with_columns_regression() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let x = rt1.series("x", &[10, 20, 30]).unwrap();
    let y = rt1.series("y", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([x, y]).unwrap();

    common::reset_exec_observation();
    let result = match df.with_columns([(col("x") + lit(1i32)).alias("x1"), (col("y") * lit(10i32)).alias("y10")]) {
        Ok(result) => result,
        Err(err) => panic!("with_columns() with Expr::lit(...) should succeed on a runtime-pinned dataframe: {err:?}"),
    };
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "expression evaluation should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
    assert_eq!(result.series("x1").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
    assert_eq!(result.series("y10").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn runtime_head_tail_slice_regression() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let ids = rt1.series("id", &[10, 20, 30, 40]).unwrap();
    let scores = rt1.series("score", &[1, 2, 3, 4]).unwrap();
    let df = DataFrame::from_series([ids, scores]).unwrap();

    common::reset_exec_observation();
    let head = match df.head(2) {
        Ok(result) => result,
        Err(err) => {
            panic!("head() should succeed on a runtime-pinned dataframe with ambient device on another GPU: {err:?}")
        },
    };
    assert_eq!(head.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20]);

    common::reset_exec_observation();
    let tail = match df.tail(2) {
        Ok(result) => result,
        Err(err) => {
            panic!("tail() should succeed on a runtime-pinned dataframe with ambient device on another GPU: {err:?}")
        },
    };
    assert_eq!(tail.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![30, 40]);

    common::reset_exec_observation();
    let slice = match df.slice(1, 2) {
        Ok(result) => result,
        Err(err) => {
            panic!("slice() should succeed on a runtime-pinned dataframe with ambient device on another GPU: {err:?}")
        },
    };
    assert_eq!(slice.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
}

#[test]
fn runtime_melt_regression() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let a = rt1.str_series("A", &["a", "b", "c"]).unwrap();
    let b = rt1.series("B", &[1, 3, 5]).unwrap();
    let c = rt1.series("C", &[2, 4, 6]).unwrap();
    let df = DataFrame::from_series(vec![a, b, c]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B", "C"]).build();
    common::reset_exec_observation();
    let result = match df.melt(opts) {
        Ok(result) => result,
        Err(err) => {
            panic!("melt() should succeed on a runtime-pinned dataframe with ambient device on another GPU: {err:?}")
        },
    };
    assert_eq!(result.series("A").unwrap().to_strings_strict().unwrap(), vec!["a", "b", "c", "a", "b", "c"]);
    assert_eq!(result.series("variable").unwrap().to_strings_strict().unwrap(), vec!["B", "B", "B", "C", "C", "C"]);
    assert_eq!(result.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3, 5, 2, 4, 6]);
}

#[test]
fn runtime_column_map_regression() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let input = rt1.series("input", &[1, 2, 3, 1]).unwrap();
    let keys = rt1.series("keys", &[1, 2, 3]).unwrap();
    let values = rt1.series("values", &[100, 200, 300]).unwrap();

    common::reset_exec_observation();
    let result =
        match input.as_column_view().unwrap().map(&keys.as_column_view().unwrap(), &values.as_column_view().unwrap()) {
            Ok(result) => result,
            Err(err) => panic!("ColumnView::map() should succeed when all logical inputs are runtime-pinned: {err:?}"),
        };
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300, 100]);
}

#[test]
fn runtime_nulls_factory_executes_on_target_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    common::reset_exec_observation();
    let series = rt1.nulls("x", TypeId::Int32, 3).unwrap();

    assert_eq!(series.null_count(), 3);
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "Runtime::nulls() should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
}

#[test]
fn dataframe_write_parquet_with_options_without_relocation_rejects_cross_device_runtime_override() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_data_rt, target_rt, df) = cross_device_write_dataframe();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("cross_device_write.parquet");

    let err = match df
        .write_parquet_with_options(ParquetWriteOptions::builder().filepath(&path).with_runtime(&target_rt).build())
    {
        Ok(()) => panic!("cross-device parquet write should fail until the dataframe is explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "dataframe::write_parquet_with_options");
}

#[test]
fn dataframe_write_csv_with_options_without_relocation_rejects_cross_device_runtime_override() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_data_rt, target_rt, df) = cross_device_write_dataframe();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("cross_device_write.csv");

    let err =
        match df.write_csv_with_options(CsvWriteOptions::builder().filepath(&path).with_runtime(&target_rt).build()) {
            Ok(()) => panic!("cross-device csv write should fail until the dataframe is explicitly relocated"),
            Err(err) => err,
        };
    assert_cross_device_relocation_error(err, "dataframe::write_csv_with_options");
}

#[test]
fn dataframe_write_json_with_options_without_relocation_rejects_cross_device_runtime_override() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (_data_rt, target_rt, df) = cross_device_write_dataframe();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("cross_device_write.jsonl");

    let err = match df
        .write_json_with_options(JsonWriteOptions::builder().filepath(&path).with_runtime(&target_rt).build())
    {
        Ok(()) => panic!("cross-device json write should fail until the dataframe is explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "dataframe::write_json_with_options");
}

#[test]
fn runtime_series_from_scalar_executes_on_target_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let scalar = Scalar::from_value(7i32).unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    common::reset_exec_observation();
    let series = rt1.series_from_scalar("x", &scalar, 4).unwrap();

    assert_eq!(series.to_vec_strict::<i32>().unwrap(), vec![7, 7, 7, 7]);
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "Runtime::series_from_scalar() should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
}

#[test]
fn runtime_column_from_scalar_executes_on_target_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let scalar = Scalar::from_value(13i32).unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    common::reset_exec_observation();
    let column = rt1.column_from_scalar(&scalar, 4).unwrap();

    assert_eq!(column.view().unwrap().to_vec::<i32>().unwrap(), vec![13, 13, 13, 13]);
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "Runtime::column_from_scalar() should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
}
