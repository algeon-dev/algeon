use cudf::{
    Column, DataFrame, Runtime, Series, Table,
    interop::{
        ExportProfile, InteropErrorKind, OwnedColumnTransfer, ReleaseSynchronization, TransferCompletionRule,
        TryIntoOwnedDeviceColumnTransfer, TryIntoOwnedDeviceTableTransfer,
    },
};
use static_assertions::assert_not_impl_any;

assert_not_impl_any!(Series: TryIntoOwnedDeviceColumnTransfer);
assert_not_impl_any!(DataFrame: TryIntoOwnedDeviceTableTransfer);

fn binding_ptr_eq(transfer: &OwnedColumnTransfer, expected_ptr: *const std::ffi::c_void) -> bool {
    transfer.bindings().iter().any(|binding| binding.bytes().base_ptr().as_ptr() as *const _ == expected_ptr)
}

#[test]
fn column_try_into_owned_device_transfer_preserves_zero_copy_metadata_and_contract() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let expected_ptr = {
        let view = column.view().unwrap();
        view.device_data_ptr().unwrap()
    };
    let expected_stream = column.__debug_execution_domain_stream_value();

    let transfer = column.try_into_owned_device_column_transfer().unwrap();

    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer);
    assert_eq!(transfer.column().len(), 3);
    assert!(binding_ptr_eq(&transfer, expected_ptr));
    assert_eq!(
        transfer.release_contract().completion(),
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes
    );
    assert_eq!(transfer.release_contract().synchronization(), ReleaseSynchronization::MaySynchronizeInternally);
    assert_eq!(transfer.release_contract().release_device(), runtime.device().ordinal() as i32);

    match transfer.readiness() {
        cudf::interop::ProducerReadiness::DeviceStreamOrdered { stream } => {
            assert_eq!(stream.as_raw() as usize, expected_stream);
            assert_eq!(stream.device(), runtime.device().ordinal() as i32);
        },
        readiness => panic!("expected device-stream readiness, got {readiness:?}"),
    }
}

#[test]
fn table_try_into_owned_device_transfer_preserves_table_shape_and_contract() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = Series::new_with_runtime(&runtime, "id", &[1i32, 2, 3]).unwrap();
    let vals = Series::new_with_runtime(&runtime, "value", &[10i64, 20, 30]).unwrap();
    let (table, _names) = DataFrame::from_columns([("id", ids), ("value", vals)]).unwrap().try_into_table().unwrap();

    let transfer = table.try_into_owned_device_table_transfer().unwrap();

    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer);
    assert_eq!(transfer.table().num_rows(), 3);
    assert_eq!(transfer.columns().len(), 2);
    assert_eq!(
        transfer.release_contract().completion(),
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes
    );
    assert_eq!(transfer.release_contract().release_device(), runtime.device().ordinal() as i32);
}

#[test]
fn ambient_default_column_owned_transfer_is_rejected() {
    let column = Column::from_slice(&[1i32, 2, 3]).unwrap();

    let err = column.try_into_owned_device_column_transfer().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
}

#[test]
fn ambient_default_table_owned_transfer_is_rejected() {
    let c0 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c1 = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();

    let err = table.try_into_owned_device_table_transfer().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
}
