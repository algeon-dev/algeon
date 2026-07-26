use cudf::{
    DataFrame, ErrorCode, Runtime,
    core::Table,
    runtime::{DeviceEntryGuard, DeviceFaultKind, DeviceHealthStatus},
    testing::device_safety::{DeviceEntryTestCall, DeviceSafetyTestScope},
};

#[test]
fn fatal_native_owner_destroy_failures_never_abort_and_poison_is_sticky() {
    let runtime = Runtime::new().expect("runtime");
    let column = runtime.column_from_slice(&[1_i32, 2, 3]).expect("column");
    let table =
        Table::from_columns(vec![runtime.column_from_slice(&[4_i32, 5, 6]).expect("table column")]).expect("table");
    let parquet_dir = tempfile::TempDir::new().expect("Parquet temp dir");
    let parquet_path = parquet_dir.path().join("device-health.parquet");
    let parquet_options =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&parquet_path).column_names(&["value"]).build();
    cudf::io::parquet::write_parquet(&table, &parquet_options).expect("Parquet fixture");
    let scalar = runtime.scalar(7_i32).expect("scalar");
    let buffer = runtime.device_buffer(64).expect("device buffer");
    let device = runtime.device();
    let scope =
        DeviceSafetyTestScope::install((0..4).map(|_| DeviceEntryTestCall::GetDevice {
            status: cudf_sys::CUDF_SUCCESS,
            device: device.ordinal() as i32,
        }));

    unsafe { cudf_sys::cudf_test_fail_next_provenance_destroy(cudf_sys::CUDF_ERROR_FATAL_CUDA) };
    drop(column);
    unsafe { cudf_sys::cudf_test_fail_next_provenance_destroy(cudf_sys::CUDF_ERROR_FATAL_CUDA) };
    drop(table);
    unsafe { cudf_sys::cudf_test_fail_next_provenance_destroy(cudf_sys::CUDF_ERROR_FATAL_CUDA) };
    drop(scalar);
    unsafe { cudf_sys::cudf_test_fail_next_provenance_destroy(cudf_sys::CUDF_ERROR_FATAL_CUDA) };
    drop(buffer);

    let snapshot = scope.health_snapshot(device);
    assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
    assert_eq!(snapshot.fault_count(), 4);
    assert_eq!(snapshot.quarantined_teardown_count(), 4);
    let first = snapshot.first_fault().expect("fatal teardown records first fault");
    assert_eq!(first.kind(), DeviceFaultKind::FatalDeviceOperation);
    assert_eq!(first.operation(), "Column::drop");
    assert_eq!(first.native_status(), Some(cudf_sys::CUDF_ERROR_FATAL_CUDA));

    let error = DeviceEntryGuard::enter(device, "native_teardown_contract::later_work")
        .expect_err("fatal teardown must permanently reject later device work");
    assert_eq!(error.code(), ErrorCode::FatalCuda);
    let compute_error =
        runtime.column_from_slice(&[8_i32, 9]).expect_err("unhealthy device must reject later compute before FFI");
    assert_eq!(compute_error.code(), ErrorCode::FatalCuda);
    let parquet_error = DataFrame::read_parquet(&runtime, &parquet_path)
        .expect_err("unhealthy device must reject later Parquet decode before FFI");
    assert_eq!(parquet_error.code(), ErrorCode::FatalCuda);
    let memory_resource_error = runtime
        .memory_resource()
        .limiting(1024)
        .expect_err("unhealthy device must reject later memory-resource creation before FFI");
    assert_eq!(memory_resource_error.code(), ErrorCode::FatalCuda);
    assert_eq!(scope.health_snapshot(device).rejected_operation_count(), 4);
    assert_eq!(scope.remaining_call_count(), 0);
}
