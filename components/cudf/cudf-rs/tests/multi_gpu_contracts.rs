//! Multi-GPU integration tests.
//!
//! These tests run by default on hosts with 2+ visible GPUs and skip cleanly
//! on single-GPU hosts. Set `CUDF_SKIP_MULTI_GPU=1` to disable them
//! explicitly.
use cudf::testing::prelude::*;

mod common;

use cudf::{
    DataFrame, DeviceId, ErrorKind, ExecutionOverrides, MeltOptions, MergeOptions, ParquetWriteOptions, Runtime,
    SortOptions, TypeId,
    core::memory::DeviceMemoryResource,
    ops::{
        expr::{col, lit},
        join, strings,
    },
    runtime::CudaStream,
};
use tempfile::TempDir;

use crate::common::scalar;

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

fn assert_cross_device_relocation_error_any_operation(err: cudf::Error, operations: &[&'static str]) {
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(
        operations.iter().any(|operation| err.operation() == *operation),
        "expected one of {:?}, got {:?}",
        operations,
        err.operation()
    );
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

fn assert_constructor_mixed_provenance_error(
    err: cudf::Error,
    operation: &'static str,
    expected_column_name: &'static str,
) {
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), operation);
    assert!(
        err.message().contains("share one retained provenance"),
        "expected constructor provenance guidance, got {:?}",
        err
    );
    assert!(
        err.context().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")),
        "expected explicit relocation remedy in error context, got {:?}",
        err.context()
    );
    assert!(
        err.context().iter().any(|(key, value)| *key == "column_name" && value == expected_column_name),
        "expected failing column_name context, got {:?}",
        err.context()
    );
}

fn cross_device_write_dataframe() -> (Runtime, Runtime, DataFrame) {
    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let target_rt = Runtime::builder().with_device(device1).build().unwrap();
    let df = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([
            data_rt.series("id", &[4i32, 5, 6]).unwrap(),
            data_rt.series("value", &[40i32, 50, 60]).unwrap(),
        ])
        .map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();
    (data_rt, target_rt, df)
}

fn first_two_test_runtimes() -> (Runtime, Runtime) {
    let (device0, device1) = first_two_test_devices();
    let left_rt = Runtime::builder().with_device(device0).build().unwrap();
    let right_rt = Runtime::builder().with_device(device1).build().unwrap();
    (left_rt, right_rt)
}

fn assert_cross_device_result<T>(result: Result<T, cudf::Error>, operation: &'static str, context: &'static str) {
    let err = match result {
        Ok(_) => panic!("{context}"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, operation);
}

fn assert_cross_device_result_any_operation<T>(
    result: Result<T, cudf::Error>,
    operations: &[&'static str],
    context: &'static str,
) {
    let err = match result {
        Ok(_) => panic!("{context}"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error_any_operation(err, operations);
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
    let df0 = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([s0]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();
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
    let df1 = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([s1]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();
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
fn with_column_requires_explicit_relocation_for_cross_device_series() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();
    let df = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([left_rt.series("id", &[1i32, 2, 3]).unwrap()]).map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();
    let foreign = right_rt.series("value", &[10i32, 20, 30]).unwrap();

    assert_cross_device_result(
        df.with_column("value", foreign),
        "dataframe::with_column",
        "cross-device with_column should require explicit relocation before structural insertion",
    );
}

#[test]
fn assign_requires_explicit_relocation_for_cross_device_series() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();
    let df = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([left_rt.series("id", &[1i32, 2, 3]).unwrap()]).map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();
    let foreign = right_rt.series("value", &[10i32, 20, 30]).unwrap();

    assert_cross_device_result(
        df.assign([("value", foreign)]),
        "dataframe::assign",
        "cross-device assign should require explicit relocation before structural insertion",
    );
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
fn memory_resource_drop_restores_previous_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);
    let mr = DeviceMemoryResource::pool(16 * 1024 * 1024, None).unwrap();

    common::set_current_device(device1);
    let before_drop = DeviceId::current().unwrap();
    assert_eq!(before_drop, device1);

    drop(mr);

    let after_drop = DeviceId::current().unwrap();
    assert_eq!(after_drop, before_drop, "memory-resource drop did not restore the previous device");
}

#[test]
fn columns_from_different_devices_have_different_domains() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create a column on the calling thread's current primary device.
    common::reset_exec_observation();
    let s0 = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device0.ordinal() as i32,
        "current-device column should be created on device {}",
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
    let result = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([s0, s1]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }));
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");
}

#[test]
fn adopted_device_buffer_uses_matching_runtime_provenance_after_current_device_switch() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);
    let runtime = Runtime::builder().with_device(device0).build().unwrap();
    let buffer = runtime.device_buffer_from_host_data(&[1i32, 2, 3]).unwrap();

    common::set_current_device(device1);
    let adopted = runtime.column_from_device_buffer(buffer, 3, TypeId::Int32).unwrap();

    let peer = runtime.column_from_slice(&[4i32, 5, 6]).unwrap();
    let frame = crate::common::dataframe_from_columns(vec!["adopted".into(), "peer".into()], vec![adopted, peer]);
    assert!(
        frame.is_ok(),
        "adopted column should bind to runtime device {} even after switching the calling thread's current device to {}",
        device0.ordinal(),
        device1.ordinal()
    );
}

#[test]
fn adopted_nullable_device_buffer_uses_matching_runtime_provenance_after_current_device_switch() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);
    let runtime = Runtime::builder().with_device(device0).build().unwrap();
    let data_buffer = runtime.device_buffer_from_host_data(&[10i32, 20, 30]).unwrap();
    let mut mask = [0u8; 64];
    mask[0] = 0x05;
    let mask_buffer = runtime.device_buffer_from_host_data(&mask).unwrap();

    common::set_current_device(device1);
    let adopted = runtime.column_from_device_buffer_nullable(data_buffer, mask_buffer, 1, 3, TypeId::Int32).unwrap();

    let peer = runtime.column_from_slice(&[40i32, 50, 60]).unwrap();
    let frame = crate::common::dataframe_from_columns(vec!["adopted".into(), "peer".into()], vec![adopted, peer]);
    assert!(
        frame.is_ok(),
        "nullable adopted column should bind to runtime device {} even after switching the calling thread's current device to {}",
        device0.ordinal(),
        device1.ordinal()
    );
}

#[test]
fn adopted_nullable_device_buffer_rejects_cross_device_data_and_mask_buffers() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();

    common::set_current_device(device0);
    let runtime0 = Runtime::builder().with_device(device0).build().unwrap();
    let data_buffer = runtime0.device_buffer_from_host_data(&[10i32, 20, 30]).unwrap();

    common::set_current_device(device1);
    let mut mask = [0u8; 64];
    mask[0] = 0x05;
    let runtime1 = Runtime::builder().with_device(device1).build().unwrap();
    let mask_buffer = runtime1.device_buffer_from_host_data(&mask).unwrap();

    let err = crate::common::column_from_device_buffer_nullable(data_buffer, mask_buffer, 1, 3, TypeId::Int32)
        .expect_err("cross-device nullable adoption should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "column::from_device_buffer_nullable_with_runtime");
}

#[test]
fn runtime_adopted_device_buffer_rejects_cross_device_runtime() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();

    common::set_current_device(device0);
    let runtime0 = Runtime::builder().with_device(device0).build().unwrap();
    let buffer = runtime0.device_buffer_from_host_data(&[1i32, 2, 3]).unwrap();

    let runtime = Runtime::builder().with_device(device1).build().unwrap();
    let err = runtime
        .column_from_device_buffer(buffer, 3, TypeId::Int32)
        .expect_err("cross-device runtime adoption should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "column::from_device_buffer_with_runtime");
}

#[test]
fn runtime_adopted_nullable_device_buffer_rejects_cross_device_runtime() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();

    common::set_current_device(device0);
    let runtime0 = Runtime::builder().with_device(device0).build().unwrap();
    let data_buffer = runtime0.device_buffer_from_host_data(&[10i32, 20, 30]).unwrap();
    let mut mask = [0u8; 64];
    mask[0] = 0x05;
    let mask_buffer = runtime0.device_buffer_from_host_data(&mask).unwrap();

    let runtime = Runtime::builder().with_device(device1).build().unwrap();
    let err = runtime
        .column_from_device_buffer_nullable(data_buffer, mask_buffer, 1, 3, TypeId::Int32)
        .expect_err("cross-device nullable runtime adoption should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "column::from_device_buffer_nullable_with_runtime");
}

// ---------------------------------------------------------------------------
// Device-pinned execution observation tests
//
// These tests verify that runtime-first high-level factories actually execute
// on the runtime's pinned device, not just the calling thread's current CUDA device.
// ---------------------------------------------------------------------------

#[test]
fn runtime_pinned_device_is_used_for_execution() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Ensure the calling thread starts on the primary device.
    let current_device = DeviceId::current().unwrap();
    assert_eq!(current_device, device0, "test setup should begin on the primary device");

    // Create a runtime pinned to the secondary device.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    // Reset observation, then create a column through the runtime.
    common::reset_exec_observation();
    let _series = rt1.series("x", &[10, 20, 30]).unwrap();

    // The FFI should have executed on the runtime's device, not the calling thread's current device.
    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "expected execution on device {}, but observed device {observed}",
        device1.ordinal()
    );

    // The calling thread's current device should still be the primary device.
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
    let df = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([ids]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();

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
    let df = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([ids, scores]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();

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

    let ctx = ExecutionOverrides::new().with_stream(rt1.stream()).with_memory_resource(rt1.memory_resource());
    common::reset_exec_observation();
    let filtered_with_ctx = df.with_execution(&ctx).filter(&mask).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "with_execution().filter should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
    let filtered_ctx_ids: Vec<i32> = filtered_with_ctx.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(filtered_ctx_ids, vec![3, 4]);

    let current_device_after = DeviceId::current().unwrap();
    assert_eq!(
        current_device_after, device0,
        "filter paths should not leak runtime device state to the calling thread"
    );
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
    let df = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([ids, scores]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();

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

    let ctx = ExecutionOverrides::new().with_stream(rt1.stream()).with_memory_resource(rt1.memory_resource());
    let with_ctx = df.with_execution(&ctx).filter(&mask);
    assert!(with_ctx.is_ok(), "with_execution().filter() with gt_scalar-derived mask should succeed: {with_ctx:?}");
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

    // Create a stream on the current primary device.
    let stream0 = CudaStream::new().unwrap();
    assert_eq!(stream0.device(), device0, "stream should record the current device");

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

    // Create an MR on the current primary device.
    let mr0 = DeviceMemoryResource::cuda().unwrap();
    assert_eq!(mr0.device(), device0, "MR should record the current device");

    // Create an MR on the secondary device via RuntimeBuilder's DeviceGuard.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    assert_eq!(rt1.memory_resource().device(), device1, "runtime MR should record the pinned device");
}

#[test]
fn builder_pool_memory_resource_ignores_ambient_device_and_uses_runtime_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt = Runtime::builder().with_device(device1).with_pool_memory_resource(16 * 1024 * 1024, None).build().unwrap();

    assert_eq!(rt.device(), device1);
    assert_eq!(rt.memory_resource().device(), device1);
    assert_eq!(
        DeviceId::current().unwrap(),
        device0,
        "runtime build should restore the prior ambient device after deferred pool creation"
    );

    common::reset_exec_observation();
    let series = rt.series("x", &[1i32, 2, 3]).unwrap();
    assert_eq!(common::last_exec_device(), device1.ordinal() as i32);
    assert_eq!(series.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(
        DeviceId::current().unwrap(),
        device0,
        "runtime execution should continue restoring the ambient device after operations"
    );
}

#[test]
fn explicit_device_memory_resource_constructors_ignore_ambient_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let cuda_mr = DeviceMemoryResource::cuda_on(device1).unwrap();
    assert_eq!(cuda_mr.device(), device1);
    assert_eq!(
        DeviceId::current().unwrap(),
        device0,
        "cuda_on should restore the calling thread's previous ambient device"
    );

    let pool_mr = DeviceMemoryResource::pool_on(device1, 16 * 1024 * 1024, None).unwrap();
    assert_eq!(pool_mr.device(), device1);
    assert_eq!(
        DeviceId::current().unwrap(),
        device0,
        "pool_on should restore the calling thread's previous ambient device"
    );

    let rt = Runtime::builder().with_device(device1).with_memory_resource(pool_mr.clone()).build().unwrap();
    assert_eq!(rt.device(), device1);
    assert_eq!(rt.memory_resource().device(), device1);
    assert_eq!(
        rt.memory_resource().testing_raw_handle_value(),
        pool_mr.testing_raw_handle_value(),
        "builder should accept a pool_on handle for the matching device"
    );
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
    let result = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([s0, s1]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }));
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");
}

#[test]
fn dataframe_constructors_share_mixed_provenance_rejection_contract() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    let left = rt0.series("left", &[1i32, 2]).unwrap();
    let right = rt1.series("right", &[3i32, 4]).unwrap();

    let from_columns_err =
        DataFrame::from_columns(::std::iter::IntoIterator::into_iter([left.clone(), right.clone()]).map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }))
        .unwrap_err();
    let explicit_columns_err = DataFrame::from_columns([("x", left), ("y", right)]).unwrap_err();

    assert_constructor_mixed_provenance_error(from_columns_err, "dataframe::from_columns", "right");
    assert_constructor_mixed_provenance_error(explicit_columns_err, "dataframe::from_columns", "y");
}

#[test]
fn merge_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([
            left_rt.series("id", &[1i32, 2, 3]).unwrap(),
            left_rt.series("val", &[10i32, 20, 30]).unwrap(),
        ])
        .map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();
    let right = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([
            right_rt.series("id", &[2i32, 3, 4]).unwrap(),
            right_rt.series("score", &[200i32, 300, 400]).unwrap(),
        ])
        .map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();

    assert_cross_device_result_any_operation(
        left.merge(&right, MergeOptions::new().how_inner().on(["id"])),
        &["dataframe::merge", "dataframe::with_execution::merge"],
        "cross-device merge should fail until inputs are explicitly relocated",
    );

    let ctx = ExecutionOverrides::new().with_stream(left_rt.stream()).with_memory_resource(left_rt.memory_resource());
    assert_cross_device_result_any_operation(
        left.with_execution(&ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])),
        &["dataframe::merge", "dataframe::with_execution::merge"],
        "borrowed context override must not repair cross-device merge",
    );
}

#[test]
fn gather_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (data_rt, indices_rt) = first_two_test_runtimes();

    let df = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([data_rt.series("id", &[10i32, 20, 30]).unwrap()]).map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();
    let indices = indices_rt.series("idx", &[2i32, 0]).unwrap();

    assert_cross_device_result(
        df.gather(&indices),
        "dataframe::gather",
        "cross-device gather should fail until inputs are explicitly relocated",
    );
}

#[test]
fn series_fillna_with_series_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (input_rt, fill_rt) = first_two_test_runtimes();

    let input = input_rt.nullable_series("value", &[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = fill_rt.series("fill", &[99i32, 99, 99]).unwrap();

    assert_cross_device_result(
        input.fillna_with_series(&fill),
        "series::fillna_with_series",
        "cross-device fillna_with_series should fail until inputs are explicitly relocated",
    );
}

#[test]
fn dataframe_fillna_with_scalar_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (df_rt, scalar_rt) = first_two_test_runtimes();

    let df = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([df_rt
            .nullable_series("value", &[1i32, 0, 3], &[true, false, true])
            .unwrap()])
        .map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();
    let scalar = scalar_rt.series("seed", &[99i32]).unwrap().sum().unwrap();

    assert_cross_device_result(
        df.fillna_with_scalar("value", &scalar),
        "dataframe::fillna_with_scalar",
        "cross-device fillna_with_scalar should fail until inputs are explicitly relocated",
    );
}

#[test]
fn low_level_join_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

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
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        &left_payload.testing_view().unwrap(),
        &right_payload.testing_view().unwrap(),
        join::NullEquality::Equal,
    ) {
        Ok(_) => panic!("cross-device low-level join should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::inner_join");
}

#[test]
fn strings_contains_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (input_rt, scalar_rt) = first_two_test_runtimes();

    let input = input_rt.series("input", &["alpha", "beta", "alphabet"]).unwrap();
    let scalar = scalar_rt.series("seed", &["alpha"]).unwrap().min().unwrap();

    let err = match strings::str_contains(&input.testing_view().unwrap(), &scalar) {
        Ok(_) => panic!("cross-device strings::contains should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "strings::contains");
}

// ---------------------------------------------------------------------------
// Additional runtime-seeding regression probes
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
    let df = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([x, y]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();

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
    let df = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([ids, scores]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();

    common::reset_exec_observation();
    let head = match df.head(2) {
        Ok(result) => result,
        Err(err) => {
            panic!(
                "head() should succeed on a runtime-pinned dataframe with the calling thread on another GPU: {err:?}"
            )
        },
    };
    assert_eq!(head.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20]);

    common::reset_exec_observation();
    let tail = match df.tail(2) {
        Ok(result) => result,
        Err(err) => {
            panic!(
                "tail() should succeed on a runtime-pinned dataframe with the calling thread on another GPU: {err:?}"
            )
        },
    };
    assert_eq!(tail.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![30, 40]);

    common::reset_exec_observation();
    let slice = match df.slice(1, 2) {
        Ok(result) => result,
        Err(err) => {
            panic!(
                "slice() should succeed on a runtime-pinned dataframe with the calling thread on another GPU: {err:?}"
            )
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
    let a = rt1.series("A", &["a", "b", "c"]).unwrap();
    let b = rt1.series("B", &[1, 3, 5]).unwrap();
    let c = rt1.series("C", &[2, 4, 6]).unwrap();
    let df = DataFrame::from_columns(::std::iter::IntoIterator::into_iter(vec![a, b, c]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B", "C"]).build();
    common::reset_exec_observation();
    let result = match df.melt(opts) {
        Ok(result) => result,
        Err(err) => {
            panic!(
                "melt() should succeed on a runtime-pinned dataframe with the calling thread on another GPU: {err:?}"
            )
        },
    };
    assert_eq!(result.series("A").unwrap().to_strings_strict().unwrap(), vec!["a", "b", "c", "a", "b", "c"]);
    assert_eq!(result.series("variable").unwrap().to_strings_strict().unwrap(), vec!["B", "B", "B", "C", "C", "C"]);
    assert_eq!(result.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3, 5, 2, 4, 6]);
}

#[test]
fn runtime_lookup_regression() {
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
    let result = match cudf::ops::lookup::probe(
        &input.testing_view().unwrap(),
        &keys.testing_view().unwrap(),
        &values.testing_view().unwrap(),
    ) {
        Ok(result) => result,
        Err(err) => panic!("lookup::probe() should succeed when all logical inputs are runtime-pinned: {err:?}"),
    };
    assert_eq!(result.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300, 100]);
}

#[test]
fn runtime_lookup_table_regression() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let keys = rt1.series("keys", &[1, 2, 3]).unwrap();
    let values = rt1.series("values", &[100, 200, 300]).unwrap();
    let table = cudf::LookupTable::new(keys, values).unwrap();
    let input = rt1.series("input", &[1, 4, 2, 3]).unwrap();

    common::reset_exec_observation();
    let result = match table.probe(&input) {
        Ok(result) => result,
        Err(err) => panic!("LookupTable::probe() should succeed when all logical inputs are runtime-pinned: {err:?}"),
    };

    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "LookupTable::probe() should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(100), None, Some(200), Some(300)]);
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
fn runtime_series_from_scalar_executes_on_target_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let scalar = scalar(7i32).unwrap();
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

    let scalar = scalar(13i32).unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    common::reset_exec_observation();
    let column = rt1.column_from_scalar(&scalar, 4).unwrap();

    assert_eq!(column.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![13, 13, 13, 13]);
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "Runtime::column_from_scalar() should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
}
