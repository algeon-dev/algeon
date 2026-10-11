//! Run these ignored tests explicitly on hosts with at least two visible GPUs.
use cudf::testing::prelude::*;

use crate::common::{
    column_from_device_buffer_nullable, dataframe_from_columns, error_context_value, query_domain_member, series,
};

#[path = "../support/common/mod.rs"]
mod common;

use cudf::{
    DataFrame, DataFrameHashJoinOptions, DataFrameJoinIndices, DataFrameJoinOptions, DataFrameMixedJoinOptions,
    DataFramePredicateJoinOptions, DenseMatrixLayout, DeviceId, ErrorCode, ErrorKind, Exec, InstallSpec, JoinType,
    MergeOptions, Runtime, Series, SortOptions, TypeId,
    core::{
        error::{FactKeyName, FactValue},
        memory::DeviceMemoryResource,
    },
    frame::dataframe::ParquetWriteOptions,
    ops::{
        ast::AstExpression,
        expr::{col, lit},
    },
    runtime::{CudaStream, DeviceEntryGuard, DeviceHealthStatus, device_health_snapshot},
};
use tempfile::TempDir;

use crate::common::scalar;

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_matrix_pack_rejects_cross_device_input() {
    let (device0, device1) = first_two_test_devices();
    let input_runtime = Runtime::builder().with_device(device0).build().unwrap();
    let output_runtime = Runtime::builder().with_device(device1).build().unwrap();
    let input = input_runtime.column_from_slice(&[1.0f32, 2.0]).unwrap();

    let error = output_runtime
        .pack_columns(&[&input], DenseMatrixLayout::RowMajor)
        .expect_err("cross-device matrix packing must be rejected");

    assert_eq!(error.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(error.operation(), "runtime::pack_columns");
    assert!(matches!(error.fact_value(FactKeyName::LeftDevice), Some(FactValue::DeviceId(_))));
    assert!(matches!(error.fact_value(FactKeyName::RightDevice), Some(FactValue::DeviceId(_))));
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn cross_device_value_drop_does_not_quarantine_teardown_device() {
    let (device_a, device_b) = first_two_test_devices();
    let runtime_b = Runtime::builder().with_device(device_b).build().unwrap();
    let series_b = runtime_b.series(&[1i32, 2, 3]).unwrap();
    {
        let _entry_a = DeviceEntryGuard::enter(device_a, "test::cross_device_drop").unwrap();
        drop(series_b);
    }
    assert_eq!(
        device_health_snapshot(device_b).status(),
        DeviceHealthStatus::Healthy,
        "dropping a B-owned value while A is entered must destroy it, not quarantine B"
    );
    let series_b_again = runtime_b.series(&[4i32]).unwrap();
    drop(series_b_again);
    assert_eq!(
        device_health_snapshot(device_b).status(),
        DeviceHealthStatus::Healthy,
        "B must still admit teardown after a cross-device drop"
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn cross_device_pool_resource_release_does_not_quarantine_its_device() {
    let (device_a, device_b) = first_two_test_devices();
    let pool_b = DeviceMemoryResource::pool_on(device_b, 16 * 1024 * 1024, None).unwrap();
    // Building a runtime on the pool registers the runtime stream, so the
    // pool's final release fences that stream before native destruction.
    let runtime_b = Runtime::builder().with_device(device_b).with_memory_resource(pool_b.clone()).build().unwrap();
    let series_b = runtime_b.series(&[1i32, 2, 3]).unwrap();
    drop(series_b);
    drop(runtime_b);
    {
        let _entry_a = DeviceEntryGuard::enter(device_a, "test::cross_device_pool_release").unwrap();
        drop(pool_b);
    }
    assert_eq!(
        device_health_snapshot(device_b).status(),
        DeviceHealthStatus::Healthy,
        "releasing B's pool while A is entered must fence and destroy it, not quarantine B"
    );
    let runtime_b = Runtime::builder().with_device(device_b).build().unwrap();
    assert_eq!(runtime_b.series(&[4i32]).unwrap().len(), 1, "B still admits work after the release");
}

fn available_test_devices() -> Vec<DeviceId> {
    let count = DeviceId::count().expect("failed to query visible CUDA devices");
    assert!(count >= 2, "requires at least 2 GPUs");
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
    let left_device = error_context_value(err, "left_device");
    let right_device = error_context_value(err, "right_device");

    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.operation(), operation);
    assert!(error_context_value(err, "left_domain").is_some(), "expected left_domain in {:?}", err);
    assert!(error_context_value(err, "right_domain").is_some(), "expected right_domain in {:?}", err);
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
        error_context_value(&err, "column_name").as_deref() == Some(expected_column_name),
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
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some(resource));
    assert_eq!(
        err.fact_value(FactKeyName::LeftDevice),
        Some(&FactValue::DeviceId(left_device.parse::<i32>().unwrap()))
    );
    assert_eq!(
        err.fact_value(FactKeyName::RightDevice),
        Some(&FactValue::DeviceId(right_device.parse::<i32>().unwrap()))
    );
    assert_eq!(error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert!(error_context_value(&err, "remedy").is_some(), "expected remedy context, got {:?}", err.details());
    assert_eq!(error_context_value(&err, "domain_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "override_stream_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "override_mr_device").as_deref(), None);
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
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some(resource));
    assert_eq!(
        err.fact_value(FactKeyName::LeftDevice),
        Some(&FactValue::DeviceId(left_device.parse::<i32>().unwrap()))
    );
    assert_eq!(
        err.fact_value(FactKeyName::RightDevice),
        Some(&FactValue::DeviceId(right_device.parse::<i32>().unwrap()))
    );
    assert_eq!(error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert!(error_context_value(&err, "remedy").is_some(), "expected remedy context, got {:?}", err.details());
    assert_eq!(error_context_value(&err, "target_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "stream_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "mr_device").as_deref(), None);
}

fn dataframe1(name: &str, series: Series) -> DataFrame {
    DataFrame::from_series([(name, series)]).unwrap()
}

fn dataframe2(name1: &str, series1: Series, name2: &str, series2: Series) -> DataFrame {
    DataFrame::from_series([(name1, series1), (name2, series2)]).unwrap()
}

fn conditional_join_options() -> DataFramePredicateJoinOptions {
    DataFramePredicateJoinOptions::new().left_columns(["left_val"]).right_columns(["right_val"])
}

fn mixed_join_options() -> DataFrameMixedJoinOptions {
    DataFrameMixedJoinOptions::new()
        .on(["id"])
        .left_predicate_columns(["left_val"])
        .right_predicate_columns(["right_val"])
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
#[ignore = "requires at least 2 GPUs"]
fn two_runtimes_on_different_devices() {
    let (device0, device1) = first_two_test_devices();
    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    assert_eq!(rt0.device(), device0);
    assert_eq!(rt1.device(), device1);
    assert_ne!(rt0.device(), rt1.device());
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtimes_execute_on_every_visible_device() {
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
#[ignore = "requires at least 2 GPUs"]
fn runtime_operations_on_different_devices_produce_correct_results() {
    let (device0, device1) = first_two_test_devices();

    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let s0 = rt0.series(&[3, 1, 4]).unwrap();
    let df0 = dataframe1("id", s0);
    common::reset_exec_observation();
    let sorted0 = df0.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()).unwrap();
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
    let sorted1 = df1.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()).unwrap();
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
#[ignore = "requires at least 2 GPUs"]
fn with_column_requires_explicit_relocation_for_cross_device_series() {
    let (left_rt, right_rt) = first_two_test_runtimes();
    let df = dataframe1("id", left_rt.series(&[1i32, 2, 3]).unwrap());
    let foreign = right_rt.series(&[10i32, 20, 30]).unwrap();

    let err = df
        .with_column("value", foreign)
        .expect_err("cross-device with_column should require explicit relocation before structural insertion");
    assert_cross_device_contract(&err, "dataframe::with_column");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("value"));
    assert_eq!(error_context_value(&err, "frame_domain").as_deref(), None);
    assert_eq!(error_context_value(&err, "series_domain").as_deref(), None);
    assert_eq!(error_context_value(&err, "frame_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "series_device").as_deref(), None);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn assign_requires_explicit_relocation_for_cross_device_series() {
    let (left_rt, right_rt) = first_two_test_runtimes();
    let df = dataframe1("id", left_rt.series(&[1i32, 2, 3]).unwrap());
    let foreign = right_rt.series(&[10i32, 20, 30]).unwrap();

    let err = df
        .assign([("value", foreign)])
        .expect_err("cross-device assign should require explicit relocation before structural insertion");
    assert_cross_device_contract(&err, "dataframe::assign");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("value"));
    assert_eq!(error_context_value(&err, "frame_domain").as_deref(), None);
    assert_eq!(error_context_value(&err, "series_domain").as_deref(), None);
    assert_eq!(error_context_value(&err, "frame_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "series_device").as_deref(), None);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn device_guard_restores_device_across_scopes() {
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
#[ignore = "requires at least 2 GPUs"]
fn memory_resource_drop_restores_previous_device() {
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
#[ignore = "requires at least 2 GPUs"]
fn columns_from_different_devices_have_different_domains() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    // Create a column on the calling thread's current primary device.
    common::reset_exec_observation();
    let s0 = series::<i32>("x", &[1, 2, 3]).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device0.ordinal() as i32,
        "current-device column should be created on device {}",
        device0.ordinal()
    );

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    common::reset_exec_observation();
    let s1 = rt1.series(&[4, 5, 6]).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "runtime column should be created on device {}",
        device1.ordinal()
    );

    let result = DataFrame::from_series([("x", s0), ("y", s1)]);
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn adopted_device_buffer_uses_matching_runtime_provenance_after_current_device_switch() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);
    let runtime = Runtime::builder().with_device(device0).build().unwrap();
    let buffer = runtime.device_buffer_from_host_data(&[1i32, 2, 3]).unwrap();

    common::set_current_device(device1);
    let adopted = runtime.column_from_device_buffer(buffer, 3, TypeId::Int32).unwrap();

    let peer = runtime.column_from_slice(&[4i32, 5, 6]).unwrap();
    let frame = dataframe_from_columns(vec!["adopted".into(), "peer".into()], vec![adopted, peer]);
    assert!(
        frame.is_ok(),
        "adopted column should bind to runtime device {} even after switching the calling thread's current device to {}",
        device0.ordinal(),
        device1.ordinal()
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn adopted_nullable_device_buffer_uses_matching_runtime_provenance_after_current_device_switch() {
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
    let frame = dataframe_from_columns(vec!["adopted".into(), "peer".into()], vec![adopted, peer]);
    assert!(
        frame.is_ok(),
        "nullable adopted column should bind to runtime device {} even after switching the calling thread's current device to {}",
        device0.ordinal(),
        device1.ordinal()
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn adopted_nullable_device_buffer_rejects_cross_device_data_and_mask_buffers() {
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

    let err = column_from_device_buffer_nullable(data_buffer, mask_buffer, 1, 3, TypeId::Int32)
        .expect_err("cross-device nullable adoption should fail");
    let left_device = device0.ordinal().to_string();
    let right_device = device1.ordinal().to_string();
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.operation(), "column::from_device_buffer_nullable_with_runtime");
    assert_eq!(error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert_eq!(error_context_value(&err, "data_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "mask_device").as_deref(), None);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_adopted_device_buffer_rejects_cross_device_runtime() {
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
    assert_eq!(error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert_eq!(error_context_value(&err, "runtime_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "buffer_device").as_deref(), None);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_adopted_nullable_device_buffer_rejects_cross_device_runtime() {
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
    assert_eq!(error_context_value(&err, "left_device").as_deref(), Some(left_device.as_str()));
    assert_eq!(error_context_value(&err, "right_device").as_deref(), Some(right_device.as_str()));
    assert_eq!(error_context_value(&err, "runtime_device").as_deref(), None);
    assert_eq!(error_context_value(&err, "buffer_device").as_deref(), None);
}

// These tests verify that runtime-first high-level factories execute on the
// runtime's pinned device, independent of the calling thread's current CUDA device.

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_pinned_device_is_used_for_execution() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let current_device = DeviceId::current().unwrap();
    assert_eq!(current_device, device0, "test setup should begin on the primary device");

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    common::reset_exec_observation();
    let _series = rt1.series(&[10, 20, 30]).unwrap();

    // The FFI runs on the runtime's device, which differs from the thread's current device.
    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "expected execution on device {}, but observed device {observed}",
        device1.ordinal()
    );

    let after = DeviceId::current().unwrap();
    assert_eq!(after, device0, "runtime leaked device state to the calling thread");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_device_flows_through_sort() {
    let (_, device1) = first_two_test_devices();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    let ids = rt1.series(&[3, 1, 4]).unwrap();
    let df = dataframe1("id", ids);

    common::reset_exec_observation();
    let _sorted = df.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()).unwrap();

    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "sort_values should execute on device {}, observed device {observed}",
        device1.ordinal()
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_device_flows_through_manual_mask_filter_paths() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let ids = rt1.series(&[3, 1, 4]).unwrap();
    let scores = rt1.series(&[30.0, 10.0, 40.0]).unwrap();
    let mask = rt1.series(&[true, false, true]).unwrap();
    let df = dataframe2("id", ids, "score", scores);

    common::reset_exec_observation();
    let filtered = df.on(Exec::RETAINED).filter(&mask).unwrap();
    assert_eq!(
        common::last_exec_device(),
        device1.ordinal() as i32,
        "filter should execute on device {}, observed device {}",
        device1.ordinal(),
        common::last_exec_device()
    );
    let filtered_ids: Vec<i32> = filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(filtered_ids, vec![3, 4]);

    let ctx = rt1.exec();
    common::reset_exec_observation();
    let filtered_with_ctx = df.on(ctx).filter(&mask).unwrap();
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
#[ignore = "requires at least 2 GPUs"]
fn runtime_gt_scalar_mask_filter_regression() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let ids = rt1.series(&[3, 1, 4]).unwrap();
    let scores = rt1.series(&[30.0, 10.0, 40.0]).unwrap();
    let df = dataframe2("id", ids, "score", scores);

    common::reset_exec_observation();
    let mask = df.series("score").unwrap().on(Exec::RETAINED).gt_scalar(20.0f64).unwrap();
    let observed = common::last_exec_device();
    assert_eq!(
        observed,
        device1.ordinal() as i32,
        "gt_scalar should execute on device {}, observed device {observed}",
        device1.ordinal()
    );

    let plain = df.on(Exec::RETAINED).filter(&mask);
    assert!(plain.is_ok(), "filter() with gt_scalar-derived mask should succeed: {plain:?}");

    let ctx = rt1.exec();
    let with_ctx = df.on(ctx).filter(&mask);
    assert!(with_ctx.is_ok(), "with_execution().filter() with gt_scalar-derived mask should succeed: {with_ctx:?}");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_device_flows_through_host_materialization() {
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
#[ignore = "requires at least 2 GPUs"]
fn concurrent_runtimes_on_different_devices() {
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
#[ignore = "requires at least 2 GPUs"]
fn shared_runtime_concurrent_access() {
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

#[test]
#[ignore = "requires at least 2 GPUs"]
fn stream_records_creation_device() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let stream0 = CudaStream::new().unwrap();
    assert_eq!(stream0.device(), device0, "stream should record the current device");

    // Create a stream on the secondary device via RuntimeBuilder's DeviceGuard.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    assert_eq!(rt1.stream().device(), device1, "runtime stream should record the pinned device");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn memory_resource_records_creation_device() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let mr0 = DeviceMemoryResource::cuda().unwrap();
    assert_eq!(mr0.device(), device0, "MR should record the current device");

    // Create an MR on the secondary device via RuntimeBuilder's DeviceGuard.
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    assert_eq!(rt1.memory_resource().device(), device1, "runtime MR should record the pinned device");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn builder_pool_memory_resource_ignores_ambient_device_and_uses_runtime_device() {
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
#[ignore = "requires at least 2 GPUs"]
fn explicit_device_memory_resource_constructors_ignore_ambient_device() {
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

#[test]
#[ignore = "requires at least 2 GPUs"]
fn builder_rejects_cross_device_stream() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let stream0 = CudaStream::new().unwrap();
    assert_eq!(stream0.device(), device0);

    let result = Runtime::builder().with_device(device1).with_stream(stream0).build();

    assert!(result.is_err(), "builder should reject cross-device stream");
    assert_runtime_builder_binding_mismatch(result.unwrap_err(), "stream", device1, device0);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn builder_rejects_cross_device_mr() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let mr0 = DeviceMemoryResource::cuda().unwrap();
    assert_eq!(mr0.device(), device0);

    let result = Runtime::builder().with_device(device1).with_memory_resource(mr0).build();

    assert!(result.is_err(), "builder should reject cross-device MR");
    assert_runtime_builder_binding_mismatch(result.unwrap_err(), "memory_resource", device1, device0);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn builder_rejects_mismatched_handle_devices() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let stream0 = CudaStream::new().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let mr1 = rt1.memory_resource().clone();

    let result = Runtime::builder().with_stream(stream0).with_memory_resource(mr1).build();

    assert!(result.is_err(), "builder should reject mismatched handle devices");
    assert_runtime_builder_binding_mismatch(result.unwrap_err(), "stream_and_memory_resource", device0, device1);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn builder_infers_device_from_handles() {
    let (_, device1) = first_two_test_devices();

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let stream1 = rt1.stream().clone();
    let mr1 = rt1.memory_resource().clone();
    drop(rt1);

    // Build without explicit with_device; it should infer the device from the handles.
    let rt = Runtime::builder().with_stream(stream1).with_memory_resource(mr1).build().unwrap();
    assert_eq!(rt.device(), device1, "builder should infer the device from the supplied handles");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn same_provenance_differs_by_device() {
    let (device0, device1) = first_two_test_devices();
    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    let s0 = rt0.series(&[1, 2]).unwrap();
    let s1 = rt1.series(&[3, 4]).unwrap();

    let result = DataFrame::from_series([("x", s0), ("y", s1)]);
    assert!(result.is_err(), "mixing columns from different devices should fail provenance check");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn dataframe_constructors_share_mixed_provenance_rejection_contract() {
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
#[ignore = "requires at least 2 GPUs"]
fn borrowed_stream_override_rejects_cross_device_context_with_canonical_keys() {
    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = Exec::RETAINED.with_stream(right_rt.stream());

    let err = df.on(ctx).sort_values(["id"], SortOptions::new()).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "stream", left_device, right_device);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn query_execution_context_rejects_cross_device_input_runtime() {
    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let err = df.on(right_rt.exec()).sort_values(["id"], SortOptions::new()).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "stream", left_device, right_device);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn with_columns_borrowed_stream_override_rejects_cross_device_context_with_canonical_keys() {
    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = Exec::RETAINED.with_stream(right_rt.stream());

    let err = df.on(ctx).with_columns([("id_plus_one", col("id") + lit(1i32))]).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "stream", left_device, right_device);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn borrowed_memory_resource_override_rejects_cross_device_context_with_canonical_keys() {
    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = Exec::RETAINED.with_memory_resource(right_rt.memory_resource());

    let err = df.on(ctx).sort_values(["id"], SortOptions::new()).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "memory_resource", left_device, right_device);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn with_columns_borrowed_memory_resource_override_rejects_cross_device_context_with_canonical_keys() {
    let (left_device, right_device) = first_two_test_devices();
    let (left_rt, right_rt) = first_two_test_runtimes();

    let df = dataframe1("id", left_rt.series(&[3i32, 1, 2]).unwrap());
    let ctx = Exec::RETAINED.with_memory_resource(right_rt.memory_resource());

    let err = df.on(ctx).with_columns([("id_plus_one", col("id") + lit(1i32))]).unwrap_err();

    assert_borrowed_override_runtime_binding_mismatch(err, "memory_resource", left_device, right_device);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn merge_without_relocation_rejects_cross_device_runtime_inputs() {
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
        left.on(Exec::RETAINED).merge(&right, MergeOptions::new().how_inner().on(["id"])),
        &["dataframe::merge"],
        "cross-device merge should fail until inputs are explicitly relocated",
    );

    let ctx = left_rt.exec();
    assert_cross_device_result_any_operation(
        left.on(ctx).merge(&right, MergeOptions::new().how_inner().on(["id"])),
        &["dataframe::merge"],
        "borrowed context override must not repair cross-device merge",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn gather_without_relocation_rejects_cross_device_runtime_inputs() {
    let (data_rt, indices_rt) = first_two_test_runtimes();

    let df = dataframe1("id", data_rt.series(&[10i32, 20, 30]).unwrap());
    let indices = indices_rt.series(&[2i32, 0]).unwrap();

    assert_cross_device_result(
        df.on(Exec::RETAINED).gather(&indices),
        "dataframe::gather",
        "cross-device gather should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn series_fillna_with_series_without_relocation_rejects_cross_device_runtime_inputs() {
    let (input_rt, fill_rt) = first_two_test_runtimes();

    let input = input_rt.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = fill_rt.series(&[99i32, 99, 99]).unwrap();

    assert_cross_device_result(
        input.on(Exec::RETAINED).fillna_with_series(&fill),
        "series::fillna_with_series",
        "cross-device fillna_with_series should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn dataframe_fillna_with_scalar_without_relocation_rejects_cross_device_runtime_inputs() {
    let (df_rt, scalar_rt) = first_two_test_runtimes();

    let df = dataframe1("value", df_rt.nullable_series(&[1i32, 0, 3], &[true, false, true]).unwrap());
    let scalar = scalar_rt.series(&[99i32]).unwrap().on(Exec::RETAINED).sum().unwrap();

    assert_cross_device_result(
        df.on(Exec::RETAINED).fillna_with_scalar("value", &scalar),
        "dataframe::fillna_with_scalar",
        "cross-device fillna_with_scalar should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    let (left_rt, right_rt) = first_two_test_runtimes();

    let left = dataframe1("id", left_rt.series(&[1i32, 2, 3]).unwrap());
    let right = dataframe1("id", right_rt.series(&[2i32, 3, 4]).unwrap());

    assert_cross_device_result(
        left.on(Exec::RETAINED).inner_join_indices(&right, DataFrameJoinOptions::new().on(["id"])),
        "join::inner_join_indices_in",
        "cross-device join indices should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn conditional_join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    let (left_rt, right_rt) = first_two_test_runtimes();

    let left = dataframe1("left_val", left_rt.series(&[1i32, 2, 3]).unwrap());
    let right = dataframe1("right_val", right_rt.series(&[2i32, 3, 4]).unwrap());
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();

    assert_cross_device_result(
        left.on(Exec::RETAINED).conditional_inner_join_indices(&right, &predicate, conditional_join_options()),
        "join::conditional_inner_join_indices_in",
        "cross-device conditional join indices should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn conditional_join_indices_in_without_relocation_reject_cross_device_runtime_inputs() {
    let (left_rt, right_rt) = first_two_test_runtimes();

    let left = dataframe1("left_val", left_rt.series(&[1i32, 2, 3]).unwrap());
    let right = dataframe1("right_val", right_rt.series(&[2i32, 3, 4]).unwrap());
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();
    let ctx = left_rt.exec();

    assert_cross_device_result(
        left.on(ctx).conditional_inner_join_indices(&right, &predicate, conditional_join_options()),
        "join::conditional_inner_join_indices_in",
        "cross-device conditional join indices should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn mixed_join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    let (left_rt, right_rt) = first_two_test_runtimes();

    let left =
        dataframe2("id", left_rt.series(&[1i32, 2, 3]).unwrap(), "left_val", left_rt.series(&[5i32, 6, 7]).unwrap());
    let right =
        dataframe2("id", right_rt.series(&[1i32, 2, 3]).unwrap(), "right_val", right_rt.series(&[6i32, 7, 8]).unwrap());
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();

    assert_cross_device_result(
        left.on(Exec::RETAINED).mixed_inner_join_indices(&right, &predicate, mixed_join_options()),
        "join::mixed_inner_join_indices_in",
        "cross-device mixed join indices should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn mixed_join_indices_in_without_relocation_reject_cross_device_runtime_inputs() {
    let (left_rt, right_rt) = first_two_test_runtimes();

    let left =
        dataframe2("id", left_rt.series(&[1i32, 2, 3]).unwrap(), "left_val", left_rt.series(&[5i32, 6, 7]).unwrap());
    let right =
        dataframe2("id", right_rt.series(&[1i32, 2, 3]).unwrap(), "right_val", right_rt.series(&[6i32, 7, 8]).unwrap());
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();
    let ctx = left_rt.exec();

    assert_cross_device_result(
        left.on(ctx).mixed_inner_join_indices(&right, &predicate, mixed_join_options()),
        "join::mixed_inner_join_indices_in",
        "cross-device mixed join indices should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn filter_join_indices_without_relocation_reject_cross_device_runtime_inputs() {
    let (left_rt, right_rt) = first_two_test_runtimes();

    let left = dataframe1("left_val", left_rt.series(&[1i32, 3, 5]).unwrap());
    let right = dataframe1("right_val", right_rt.series(&[2i32, 4, 6]).unwrap());
    let left_indices = DataFrameJoinIndices::from_series(left_rt.series(&[0i32, 1, 2]).unwrap());
    let right_indices = DataFrameJoinIndices::from_series(right_rt.series(&[0i32, 1, 2]).unwrap());
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();

    assert_cross_device_result(
        left.on(Exec::RETAINED).filter_join_indices(
            &right,
            &left_indices,
            &right_indices,
            &predicate,
            conditional_join_options(),
            JoinType::Inner,
        ),
        "join::filter_join_indices_in",
        "cross-device filter join indices should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn filter_join_indices_in_without_relocation_reject_cross_device_runtime_inputs() {
    let (left_rt, right_rt) = first_two_test_runtimes();

    let left = dataframe1("left_val", left_rt.series(&[1i32, 3, 5]).unwrap());
    let right = dataframe1("right_val", right_rt.series(&[2i32, 4, 6]).unwrap());
    let left_indices = DataFrameJoinIndices::from_series(left_rt.series(&[0i32, 1, 2]).unwrap());
    let right_indices = DataFrameJoinIndices::from_series(right_rt.series(&[0i32, 1, 2]).unwrap());
    let predicate =
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap();
    let ctx = left_rt.exec();

    assert_cross_device_result(
        left.on(ctx).filter_join_indices(
            &right,
            &left_indices,
            &right_indices,
            &predicate,
            conditional_join_options(),
            JoinType::Inner,
        ),
        "join::filter_join_indices_in",
        "cross-device filter join indices should fail until inputs are explicitly relocated",
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn hash_join_probe_without_relocation_rejects_cross_device_runtime_inputs() {
    let (build_rt, probe_rt) = first_two_test_runtimes();

    let build = DataFrame::from_series([("id", build_rt.series(&[1i32, 2, 3]).unwrap())]).unwrap();
    let probe = DataFrame::from_series([("id", probe_rt.series(&[2i32, 3, 4]).unwrap())]).unwrap();
    let hash_join = build.on(Exec::RETAINED).build_hash_join(["id"], DataFrameHashJoinOptions::new()).unwrap();

    let err = match hash_join.inner_probe(&probe, ["id"], Exec::RETAINED) {
        Ok(_) => panic!("cross-device hash-join probe should fail until inputs are explicitly relocated"),
        Err(err) => err,
    };
    assert_cross_device_relocation_error_any_operation(
        err,
        &["dataframe::hash_join::inner_probe", "hash_join::inner_join_indices_in"],
    );
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_expr_literal_with_columns_regression() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let x = rt1.series(&[10, 20, 30]).unwrap();
    let y = rt1.series(&[1, 2, 3]).unwrap();
    let df = dataframe2("x", x, "y", y);

    common::reset_exec_observation();
    let result = match df
        .on(Exec::RETAINED)
        .with_columns([(col("x") + lit(1i32)).alias("x1"), (col("y") * lit(10i32)).alias("y10")])
    {
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
#[ignore = "requires at least 2 GPUs"]
fn runtime_head_tail_slice_regression() {
    let _device_guard = common::DeviceRestoreGuard::new();
    let (device0, device1) = first_two_test_devices();
    common::set_current_device(device0);

    let rt1 = Runtime::builder().with_device(device1).build().unwrap();
    let ids = rt1.series(&[10, 20, 30, 40]).unwrap();
    let scores = rt1.series(&[1, 2, 3, 4]).unwrap();
    let df = dataframe2("id", ids, "score", scores);

    common::reset_exec_observation();
    let head = match df.on(Exec::RETAINED).head(2) {
        Ok(result) => result,
        Err(err) => {
            panic!(
                "head() should succeed on a runtime-pinned dataframe with the calling thread on another GPU: {err:?}"
            )
        },
    };
    assert_eq!(head.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20]);

    common::reset_exec_observation();
    let tail = match df.on(Exec::RETAINED).tail(2) {
        Ok(result) => result,
        Err(err) => {
            panic!(
                "tail() should succeed on a runtime-pinned dataframe with the calling thread on another GPU: {err:?}"
            )
        },
    };
    assert_eq!(tail.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![30, 40]);

    common::reset_exec_observation();
    let slice = match df.on(Exec::RETAINED).slice(1, 2) {
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
#[ignore = "requires at least 2 GPUs"]
fn runtime_nulls_factory_executes_on_target_device() {
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
#[ignore = "requires at least 2 GPUs"]
fn dataframe_write_parquet_with_options_without_relocation_rejects_cross_device_runtime_override() {
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
    assert_eq!(error_context_value(&err, "frame_domain").as_deref(), None);
    assert_eq!(error_context_value(&err, "write_domain").as_deref(), None);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn runtime_series_from_scalar_executes_on_target_device() {
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
#[ignore = "requires at least 2 GPUs"]
fn runtime_column_from_scalar_executes_on_target_device() {
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

/// Concurrent pipelines must not interact: each device keeps its own memory domain
/// installed, work completes correctly on both, and neither pipeline blocks
/// on host-side state owned by the other.
#[test]
#[ignore = "requires at least 2 GPUs"]
fn concurrent_per_device_pipelines_share_no_state() {
    let (device0, device1) = first_two_test_devices();

    let domain0 = cudf::DeviceMemoryManager::install(device0, InstallSpec::default()).unwrap().0;
    let domain1 = cudf::DeviceMemoryManager::install(device1, InstallSpec::default()).unwrap().0;
    let mr0 = domain0.resource().testing_raw_handle_value();
    let mr1 = domain1.resource().testing_raw_handle_value();
    assert_ne!(mr0, mr1, "device memory domains must be independent");

    let worker = |device: DeviceId, pipeline_id: u64| {
        std::thread::spawn(move || {
            let rt = Runtime::builder().with_device(device).build().unwrap();
            let _lease = query_domain_member(&rt, pipeline_id, 256 * 1024 * 1024).unwrap();
            for _ in 0..20 {
                let data: Vec<i32> = (0..4096).rev().collect();
                let series = rt.series(&data).unwrap();
                let sorted = series.on(Exec::RETAINED).sort_values(SortOptions::default()).unwrap();
                assert_eq!(sorted.on(Exec::RETAINED).sum_value::<i64>().unwrap(), (0..4096i64).sum::<i64>());
            }
            rt.memory_resource().testing_raw_handle_value()
        })
    };

    let pipeline0 = worker(device0, 50);
    let pipeline1 = worker(device1, 51);
    assert_eq!(pipeline0.join().unwrap(), mr0);
    assert_eq!(pipeline1.join().unwrap(), mr1);
}
