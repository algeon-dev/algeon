//! Multi-GPU integration tests.
//!
//! These tests run by default on hosts with 2+ visible GPUs and skip cleanly
//! on single-GPU hosts. Set `CUDF_SKIP_MULTI_GPU=1` to disable them
//! explicitly.
use cudf::testing::prelude::*;

mod common;

use cudf::{
    DataFrame, DeviceId, ErrorCode, ErrorKind, ExecutionOverrides, MergeOptions, Runtime, Series, SortOptions, TypeId,
    core::{
        error::{FactKeyName, FactValue},
        memory::DeviceMemoryResource,
    },
    frame::dataframe::{MeltOptions, ParquetWriteOptions},
    ops::{
        ast::AstExpression,
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
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), operation);
    assert!(
        err.message().contains("cross-device") || err.message().contains("explicit relocation"),
        "expected cross-device relocation guidance, got {:?}",
        err
    );
    assert!(
        err.details().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")),
        "expected explicit relocation remedy in error context, got {:?}",
        err.details()
    );
}

fn assert_cross_device_relocation_error_any_operation(err: cudf::Error, operations: &[&'static str]) {
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
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
        err.details().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")),
        "expected explicit relocation remedy in error context, got {:?}",
        err.details()
    );
}

fn assert_cross_device_contract(err: &cudf::Error, operation: &'static str) {
    let left_device = crate::common::error_context_value(err, "left_device");
    let right_device = crate::common::error_context_value(err, "right_device");

    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.operation(), operation);
    assert!(crate::common::error_context_value(err, "left_domain").is_some(), "expected left_domain in {:?}", err);
    assert!(crate::common::error_context_value(err, "right_domain").is_some(), "expected right_domain in {:?}", err);
    assert!(matches!(err.fact_value(FactKeyName::LeftDevice), Some(FactValue::DeviceId(_))));
    assert!(matches!(err.fact_value(FactKeyName::RightDevice), Some(FactValue::DeviceId(_))));
    assert!(left_device.is_some(), "expected left_device in {:?}", err);
    assert!(right_device.is_some(), "expected right_device in {:?}", err);
    assert_ne!(left_device, right_device, "cross-device contract should expose different device ids: {:?}", err);
    assert!(
        err.details().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")),
        "expected explicit relocation remedy in error context, got {:?}",
        err.details()
    );
}

fn assert_constructor_mixed_provenance_error(
    err: cudf::Error,
    operation: &'static str,
    expected_column_name: &'static str,
) {
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), operation);
    assert!(
        err.message().contains("share one retained provenance"),
        "expected constructor provenance guidance, got {:?}",
        err
    );
    assert!(
        err.details().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")),
        "expected explicit relocation remedy in error context, got {:?}",
        err.details()
    );
    assert!(
        crate::common::error_context_value(&err, "column_name").as_deref() == Some(expected_column_name),
        "expected failing column_name context, got facts={:?} details={:?}",
        err.facts(),
        err.details()
    );
    assert_eq!(err.detail_value("column_name"), None);
}

fn assert_borrowed_override_runtime_binding_mismatch(
    err: cudf::Error,
    resource: &'static str,
    left_device: DeviceId,
    right_device: DeviceId,
) {
    let left_device = left_device.ordinal().to_string();
    let right_device = right_device.ordinal().to_string();

    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.operation(), "execution_overrides::resolve_output_domain");
    assert_eq!(crate::common::error_context_value(&err, "resource").as_deref(), Some(resource));
    assert_eq!(
        err.fact_value(FactKeyName::LeftDevice),
        Some(&FactValue::DeviceId(left_device.parse::<i32>().unwrap()))
    );
    assert_eq!(
        err.fact_value(FactKeyName::RightDevice),
        Some(&FactValue::DeviceId(right_device.parse::<i32>().unwrap()))
    );
    assert_eq!(crate::common::error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert!(
        crate::common::error_context_value(&err, "remedy").is_some(),
        "expected remedy context, got {:?}",
        err.details()
    );
    assert_eq!(crate::common::error_context_value(&err, "domain_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "override_stream_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "override_mr_device").as_deref(), None);
}

fn assert_runtime_builder_binding_mismatch(
    err: cudf::Error,
    resource: &'static str,
    left_device: DeviceId,
    right_device: DeviceId,
) {
    let left_device = left_device.ordinal().to_string();
    let right_device = right_device.ordinal().to_string();

    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.operation(), "runtime::RuntimeBuilder::build");
    assert_eq!(crate::common::error_context_value(&err, "resource").as_deref(), Some(resource));
    assert_eq!(
        err.fact_value(FactKeyName::LeftDevice),
        Some(&FactValue::DeviceId(left_device.parse::<i32>().unwrap()))
    );
    assert_eq!(
        err.fact_value(FactKeyName::RightDevice),
        Some(&FactValue::DeviceId(right_device.parse::<i32>().unwrap()))
    );
    assert_eq!(crate::common::error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert!(
        crate::common::error_context_value(&err, "remedy").is_some(),
        "expected remedy context, got {:?}",
        err.details()
    );
    assert_eq!(crate::common::error_context_value(&err, "target_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "stream_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "mr_device").as_deref(), None);
}

fn dataframe1(name: &str, series: Series) -> DataFrame {
    DataFrame::from_series([(name, series)]).unwrap()
}

fn dataframe2(name1: &str, series1: Series, name2: &str, series2: Series) -> DataFrame {
    DataFrame::from_series([(name1, series1), (name2, series2)]).unwrap()
}

fn dataframe3(name1: &str, series1: Series, name2: &str, series2: Series, name3: &str, series3: Series) -> DataFrame {
    DataFrame::from_series([(name1, series1), (name2, series2), (name3, series3)]).unwrap()
}

fn cross_device_write_dataframe() -> (Runtime, Runtime, DataFrame) {
    let (device0, device1) = first_two_test_devices();
    let data_rt = Runtime::builder().with_device(device0).build().unwrap();
    let target_rt = Runtime::builder().with_device(device1).build().unwrap();
    let df =
        dataframe2("id", data_rt.series(&[4i32, 5, 6]).unwrap(), "value", data_rt.series(&[40i32, 50, 60]).unwrap());
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
        let series = rt.series(&[device.ordinal() as i32, 99]).unwrap();
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
    let s0 = rt0.series(&[3, 1, 4]).unwrap();
    let df0 = dataframe1("id", s0);
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
    let s1 = rt1.series(&[8, 2, 5]).unwrap();
    let df1 = dataframe1("id", s1);
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
    let df = dataframe1("id", left_rt.series(&[1i32, 2, 3]).unwrap());
    let foreign = right_rt.series(&[10i32, 20, 30]).unwrap();

    let err = df
        .with_column("value", foreign)
        .expect_err("cross-device with_column should require explicit relocation before structural insertion");
    assert_cross_device_contract(&err, "dataframe::with_column");
    assert_eq!(crate::common::error_context_value(&err, "column_name").as_deref(), Some("value"));
    assert_eq!(crate::common::error_context_value(&err, "frame_domain").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "series_domain").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "frame_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "series_device").as_deref(), None);
}

#[test]
fn assign_requires_explicit_relocation_for_cross_device_series() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();
    let df = dataframe1("id", left_rt.series(&[1i32, 2, 3]).unwrap());
    let foreign = right_rt.series(&[10i32, 20, 30]).unwrap();

    let err = df
        .assign([("value", foreign)])
        .expect_err("cross-device assign should require explicit relocation before structural insertion");
    assert_cross_device_contract(&err, "dataframe::assign");
    assert_eq!(crate::common::error_context_value(&err, "column_name").as_deref(), Some("value"));
    assert_eq!(crate::common::error_context_value(&err, "frame_domain").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "series_domain").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "frame_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "series_device").as_deref(), None);
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
    let s1 = rt1.series(&[4, 5, 6]).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "runtime column should be created on device {}",
        device1.ordinal()
    );

    // Mixing columns from different devices into one DataFrame must fail.
    let result = DataFrame::from_series([("x", s0), ("y", s1)]);
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
    let left_device = device0.ordinal().to_string();
    let right_device = device1.ordinal().to_string();
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.operation(), "column::from_device_buffer_nullable_with_runtime");
    assert_eq!(crate::common::error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "data_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "mask_device").as_deref(), None);
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
    let left_device = device1.ordinal().to_string();
    let right_device = device0.ordinal().to_string();
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.operation(), "column::from_device_buffer_with_runtime");
    assert_eq!(crate::common::error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "runtime_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "buffer_device").as_deref(), None);
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
    let left_device = device1.ordinal().to_string();
    let right_device = device0.ordinal().to_string();
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.operation(), "column::from_device_buffer_nullable_with_runtime");
    assert_eq!(crate::common::error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert_eq!(crate::common::error_context_value(&err, "runtime_device").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "buffer_device").as_deref(), None);
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
    let _series = rt1.series(&[10, 20, 30]).unwrap();

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

    let ids = rt1.series(&[3, 1, 4]).unwrap();
    let df = dataframe1("id", ids);

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
    let ids = rt1.series(&[3, 1, 4]).unwrap();
    let scores = rt1.series(&[30.0, 10.0, 40.0]).unwrap();
    let mask = rt1.series(&[true, false, true]).unwrap();
    let df = dataframe2("id", ids, "score", scores);

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
    let ids = rt1.series(&[3, 1, 4]).unwrap();
    let scores = rt1.series(&[30.0, 10.0, 40.0]).unwrap();
    let df = dataframe2("id", ids, "score", scores);

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
    let series = rt1.series(&[10, 20]).unwrap();

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
        let _s = rt0.series(&[1, 2, 3]).unwrap();
        common::last_exec_device()
    });

    let handle1 = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = rt1.series(&[4, 5, 6]).unwrap();
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
        let _s = rt1.series(&[1, 2]).unwrap();
        common::last_exec_device()
    });

    let handle_b = std::thread::spawn(move || {
        common::reset_exec_observation();
        let _s = rt1_clone.series(&[3, 4]).unwrap();
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
// Handle device provenance tests
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
    let series = rt.series(&[1i32, 2, 3]).unwrap();
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
// RuntimeBuilder cross-device validation tests
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
    assert_runtime_builder_binding_mismatch(result.unwrap_err(), "stream", device1, device0);
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
    assert_runtime_builder_binding_mismatch(result.unwrap_err(), "memory_resource", device1, device0);
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
    assert_runtime_builder_binding_mismatch(result.unwrap_err(), "stream_and_memory_resource", device0, device1);
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
// ExecutionDomain provenance and borrowed override tests
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
    let s0 = rt0.series(&[1, 2]).unwrap();
    let s1 = rt1.series(&[3, 4]).unwrap();

    // Attempting to build a DataFrame from columns on different devices should fail.
    let result = DataFrame::from_series([("x", s0), ("y", s1)]);
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

    let left = rt0.series(&[1i32, 2]).unwrap();
    let right = rt1.series(&[3i32, 4]).unwrap();

    let named_series_err = DataFrame::from_series([("left", left.clone()), ("right", right.clone())]).unwrap_err();
    let explicit_schema_err = DataFrame::from_series([("x", left), ("y", right)]).unwrap_err();

    assert_constructor_mixed_provenance_error(named_series_err, "dataframe::from_series", "right");
    assert_constructor_mixed_provenance_error(explicit_schema_err, "dataframe::from_series", "y");
}

#[test]
fn borrowed_stream_override_rejects_cross_device_context_with_canonical_keys() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = ExecutionOverrides::new().with_stream(right_rt.stream());

    let err = df.with_execution(&ctx).sort_values(["id"], SortOptions::new()).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "stream", left_device, right_device);
}

#[test]
fn query_execution_context_rejects_cross_device_input_runtime() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let query = right_rt.query_execution();

    let err = df.with_execution(query.overrides()).sort_values(["id"], SortOptions::new()).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "stream", left_device, right_device);
}

#[test]
fn with_columns_borrowed_stream_override_rejects_cross_device_context_with_canonical_keys() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = ExecutionOverrides::new().with_stream(right_rt.stream());

    let err = df.with_execution(&ctx).with_columns([("id_plus_one", col("id") + lit(1i32))]).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "stream", left_device, right_device);
}

#[test]
fn borrowed_memory_resource_override_rejects_cross_device_context_with_canonical_keys() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = ExecutionOverrides::new().with_memory_resource(right_rt.memory_resource());

    let err = df.with_execution(&ctx).sort_values(["id"], SortOptions::new()).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "memory_resource", left_device, right_device);
}

#[test]
fn with_columns_borrowed_memory_resource_override_rejects_cross_device_context_with_canonical_keys() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = ExecutionOverrides::new().with_memory_resource(right_rt.memory_resource());

    let err = df.with_execution(&ctx).with_columns([("id_plus_one", col("id") + lit(1i32))]).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "memory_resource", left_device, right_device);
}

#[test]
fn merge_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left =
        dataframe2("id", left_rt.series(&[1i32, 2, 3]).unwrap(), "score", left_rt.series(&[10i32, 20, 30]).unwrap());
    let right = dataframe2(
        "id",
        right_rt.series(&[2i32, 3, 4]).unwrap(),
        "value",
        right_rt.series(&[200i32, 300, 400]).unwrap(),
    );

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

    let df = dataframe1("id", data_rt.series(&[10i32, 20, 30]).unwrap());
    let indices = indices_rt.series(&[2i32, 0]).unwrap();

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

    let input = input_rt.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = fill_rt.series(&[99i32, 99, 99]).unwrap();

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

    let df = dataframe1("value", df_rt.nullable_series(&[1i32, 0, 3], &[true, false, true]).unwrap());
    let scalar = scalar_rt.series(&[99i32]).unwrap().sum().unwrap();

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

    let left_keys =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let left_payload = cudf::core::Table::from_columns(vec![
        left_rt.series(&[10i32, 20, 30]).unwrap().as_column().deep_copy().unwrap(),
    ])
    .unwrap();
    let right_keys =
        cudf::core::Table::from_columns(vec![right_rt.series(&[2i32, 3, 4]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right_payload = cudf::core::Table::from_columns(vec![
        right_rt.series(&[200i32, 300, 400]).unwrap().as_column().deep_copy().unwrap(),
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
fn join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left_keys =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right_keys =
        cudf::core::Table::from_columns(vec![right_rt.series(&[2i32, 3, 4]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();

    let err = match join::inner_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        join::NullEquality::Equal,
    ) {
        Ok(_) => panic!("cross-device join indices should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::inner_join_indices");
}

#[test]
fn conditional_join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right =
        cudf::core::Table::from_columns(vec![right_rt.series(&[2i32, 3, 4]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();

    let err = match join::conditional_inner_join_indices(
        &left.testing_view().unwrap(),
        &right.testing_view().unwrap(),
        &predicate,
    ) {
        Ok(_) => panic!("cross-device conditional join indices should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::conditional_inner_join_indices");
}

#[test]
fn conditional_join_indices_in_without_relocation_reject_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right =
        cudf::core::Table::from_columns(vec![right_rt.series(&[2i32, 3, 4]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();
    let ctx = ExecutionOverrides::new().with_stream(left_rt.stream()).with_memory_resource(left_rt.memory_resource());

    let err = match join::conditional_inner_join_indices_in(
        &left.testing_view().unwrap(),
        &right.testing_view().unwrap(),
        &predicate,
        &ctx,
    ) {
        Ok(_) => panic!("cross-device conditional join indices should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::conditional_inner_join_indices_in");
}

#[test]
fn mixed_join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left_equality =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let left_conditional =
        cudf::core::Table::from_columns(vec![left_rt.series(&[5i32, 6, 7]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right_equality =
        cudf::core::Table::from_columns(vec![right_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right_conditional =
        cudf::core::Table::from_columns(vec![right_rt.series(&[6i32, 7, 8]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();

    let err = match join::mixed_inner_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        join::NullEquality::Equal,
    ) {
        Ok(_) => panic!("cross-device mixed join indices should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::mixed_inner_join_indices");
}

#[test]
fn mixed_join_indices_in_without_relocation_reject_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left_equality =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let left_conditional =
        cudf::core::Table::from_columns(vec![left_rt.series(&[5i32, 6, 7]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right_equality =
        cudf::core::Table::from_columns(vec![right_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right_conditional =
        cudf::core::Table::from_columns(vec![right_rt.series(&[6i32, 7, 8]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();
    let ctx = ExecutionOverrides::new().with_stream(left_rt.stream()).with_memory_resource(left_rt.memory_resource());

    let err = match join::mixed_inner_join_indices_in(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        join::NullEquality::Equal,
        &ctx,
    ) {
        Ok(_) => panic!("cross-device mixed join indices should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::mixed_inner_join_indices_in");
}

#[test]
fn filter_join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 3, 5]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right =
        cudf::core::Table::from_columns(vec![right_rt.series(&[2i32, 4, 6]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let left_indices = left_rt.series(&[0i32, 1, 2]).unwrap().as_column().deep_copy().unwrap();
    let right_indices = right_rt.series(&[0i32, 1, 2]).unwrap().as_column().deep_copy().unwrap();
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();

    let err = match join::filter_join_indices(
        &left.testing_view().unwrap(),
        &right.testing_view().unwrap(),
        &left_indices.testing_view().unwrap(),
        &right_indices.testing_view().unwrap(),
        &predicate,
        join::JoinType::Inner,
    ) {
        Ok(_) => panic!("cross-device filter join indices should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::filter_join_indices");
}

#[test]
fn filter_join_indices_in_without_relocation_reject_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (left_rt, right_rt) = first_two_test_runtimes();

    let left =
        cudf::core::Table::from_columns(vec![left_rt.series(&[1i32, 3, 5]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let right =
        cudf::core::Table::from_columns(vec![right_rt.series(&[2i32, 4, 6]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let left_indices = left_rt.series(&[0i32, 1, 2]).unwrap().as_column().deep_copy().unwrap();
    let right_indices = right_rt.series(&[0i32, 1, 2]).unwrap().as_column().deep_copy().unwrap();
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();
    let ctx = ExecutionOverrides::new().with_stream(left_rt.stream()).with_memory_resource(left_rt.memory_resource());

    let err = match join::filter_join_indices_in(
        &left.testing_view().unwrap(),
        &right.testing_view().unwrap(),
        &left_indices.testing_view().unwrap(),
        &right_indices.testing_view().unwrap(),
        &predicate,
        join::JoinType::Inner,
        &ctx,
    ) {
        Ok(_) => panic!("cross-device filter join indices should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "join::filter_join_indices_in");
}

#[test]
fn hash_join_probe_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (build_rt, probe_rt) = first_two_test_runtimes();

    let build_keys =
        cudf::core::Table::from_columns(vec![build_rt.series(&[1i32, 2, 3]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let probe_keys =
        cudf::core::Table::from_columns(vec![probe_rt.series(&[2i32, 3, 4]).unwrap().as_column().deep_copy().unwrap()])
            .unwrap();
    let hash_join = join::HashJoin::new(build_keys, join::HashJoinOptions::new()).unwrap();

    let err = match hash_join.inner_join_indices(&probe_keys.testing_view().unwrap()) {
        Ok(_) => panic!("cross-device hash-join probe should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error(err, "hash_join::inner_join_indices");
}

#[test]
fn strings_contains_without_relocation_rejects_cross_device_runtime_inputs() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (input_rt, scalar_rt) = first_two_test_runtimes();

    let input = input_rt.series(&["alpha", "beta", "alphabet"]).unwrap();
    let scalar = scalar_rt.series(&["alpha"]).unwrap().min().unwrap();

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
    let x = rt1.series(&[10, 20, 30]).unwrap();
    let y = rt1.series(&[1, 2, 3]).unwrap();
    let df = dataframe2("x", x, "y", y);

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
    let ids = rt1.series(&[10, 20, 30, 40]).unwrap();
    let scores = rt1.series(&[1, 2, 3, 4]).unwrap();
    let df = dataframe2("id", ids, "score", scores);

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
    let a = rt1.series(&["a", "b", "c"]).unwrap();
    let b = rt1.series(&[1, 3, 5]).unwrap();
    let c = rt1.series(&[2, 4, 6]).unwrap();
    let df = dataframe3("A", a, "B", b, "C", c);

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
    let input = rt1.series(&[1, 2, 3, 1]).unwrap();
    let keys = rt1.series(&[1, 2, 3]).unwrap();
    let values = rt1.series(&[100, 200, 300]).unwrap();

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
    let keys = rt1.series(&[1, 2, 3]).unwrap();
    let values = rt1.series(&[100, 200, 300]).unwrap();
    let table = cudf::LookupTable::new(keys, values).unwrap();
    let input = rt1.series(&[1, 4, 2, 3]).unwrap();

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
    let series = rt1.nulls(TypeId::Int32, 3).unwrap();

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
    assert_cross_device_contract(&err, "dataframe::write_parquet_with_options");
    assert_eq!(crate::common::error_context_value(&err, "frame_domain").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "write_domain").as_deref(), None);
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
    let series = rt1.series_from_scalar(&scalar, 4).unwrap();

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

/// Device-domain concurrency (P0-6 §10.5): one pipeline per device running
/// concurrently must not interact — each device keeps its own memory domain
/// installed, work completes correctly on both, and neither pipeline blocks
/// on host-side state owned by the other.
#[test]
fn concurrent_per_device_pipelines_share_no_state() {
    if !should_run_multi_gpu_test() {
        return;
    }
    let (device0, device1) = first_two_test_devices();

    let domain0 = cudf::DeviceMemoryManager::install_for_device(device0).unwrap();
    let domain1 = cudf::DeviceMemoryManager::install_for_device(device1).unwrap();
    let mr0 = domain0.resource().testing_raw_handle_value();
    let mr1 = domain1.resource().testing_raw_handle_value();
    assert_ne!(mr0, mr1, "device memory domains must be independent");

    let worker = |device: DeviceId, pipeline_id: u64| {
        std::thread::spawn(move || {
            let rt = Runtime::builder().with_device(device).build().unwrap();
            let _lease = rt.stream_reservation(pipeline_id, 256 * 1024 * 1024).unwrap();
            for _ in 0..20 {
                let data: Vec<i32> = (0..4096).rev().collect();
                let series = rt.series(&data).unwrap();
                let sorted = series.sort_values(SortOptions::default()).unwrap();
                assert_eq!(sorted.sum_value::<i64>().unwrap(), (0..4096i64).sum::<i64>());
            }
            rt.memory_resource().testing_raw_handle_value()
        })
    };

    let pipeline0 = worker(device0, 50);
    let pipeline1 = worker(device1, 51);
    assert_eq!(pipeline0.join().unwrap(), mr0);
    assert_eq!(pipeline1.join().unwrap(), mr1);
}
