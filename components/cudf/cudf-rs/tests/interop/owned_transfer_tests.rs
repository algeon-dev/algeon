use cudf::{
    DataFrame, Runtime, Series,
    core::{Column, Table},
    interop::{
        BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, ExportDiagnostics, ExportProfile,
        InteropErrorKind, OwnedColumnTransfer, OwnedTableTransfer, PhysicalValidity, Placement, ProducerReadiness,
        ReleaseContract, ReleaseSynchronization, SchemaNullability, SemanticColumn, SemanticTable, TableId,
        TransferCompletionRule, TryIntoOwnedDeviceColumnTransfer, TryIntoOwnedDeviceTableTransfer,
    },
};
use rapids_interop::{DataType as InteropDataType, TypeId as InteropTypeId};
use static_assertions::assert_impl_all;

assert_impl_all!(Series: TryIntoOwnedDeviceColumnTransfer);
assert_impl_all!(DataFrame: TryIntoOwnedDeviceTableTransfer);

fn binding_ptr_eq(transfer: &OwnedColumnTransfer, expected_ptr: *const std::ffi::c_void) -> bool {
    transfer.bindings().iter().any(|binding| binding.bytes().base_ptr() == expected_ptr)
}

fn fake_ptr(addr: usize) -> *const std::ffi::c_void {
    addr as *const std::ffi::c_void
}

fn device_binding(id: u32, addr: usize, byte_len: usize, device: i32) -> BufferBinding<'static> {
    BufferBinding::new(
        BufferId::new(id),
        BoundBytes::from_raw_parts(fake_ptr(addr) as *mut std::ffi::c_void, byte_len).unwrap(),
        Placement::CudaDevice { device },
    )
}

fn fixed_width_column_semantics(id: u32, values: u32, len: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        InteropDataType::new(InteropTypeId::Int32),
        len,
        0,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::FixedWidth { values: BufferId::new(values) },
    )
}

fn manual_owned_transfer_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::OwnedDeviceTransfer)
}

fn manual_release_contract(device: i32) -> ReleaseContract {
    ReleaseContract::new(
        device,
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
        ReleaseSynchronization::MaySynchronizeInternally,
    )
}

fn manual_readiness(device: i32) -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceStreamOrdered {
        stream: cudf::interop::CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut _, device).unwrap(),
    }
}

fn wrap_owned_table_transfer(transfer: OwnedTableTransfer) -> OwnedTableTransfer {
    OwnedTableTransfer::try_new_with_semantic_columns(
        transfer.table().clone(),
        transfer.columns().to_vec(),
        transfer.semantic_columns().to_vec(),
        transfer.bindings().to_vec(),
        *transfer.readiness(),
        *transfer.diagnostics(),
        *transfer.release_contract(),
        transfer,
    )
    .unwrap()
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
fn series_try_into_owned_device_transfer_preserves_zero_copy_metadata_and_contract() {
    let runtime = Runtime::builder().build().unwrap();
    let series = Series::new_with_runtime(&runtime, "numbers", &[1i32, 2, 3]).unwrap();
    let expected_ptr = series.as_column_view().unwrap().device_data_ptr().unwrap();
    let expected_stream = series.as_column().__debug_execution_domain_stream_value();

    let transfer = series.try_into_owned_device_column_transfer().unwrap();

    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer);
    assert_eq!(transfer.column().len(), 3);
    assert!(binding_ptr_eq(&transfer, expected_ptr));
    assert_eq!(
        transfer.release_contract().completion(),
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes
    );
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
fn series_try_into_owned_device_transfer_deep_copies_when_series_data_is_shared() {
    let runtime = Runtime::builder().build().unwrap();
    let series = Series::new_with_runtime(&runtime, "numbers", &[7i32, 8, 9]).unwrap();
    let shared = series.clone();
    let original_ptr = shared.as_column_view().unwrap().device_data_ptr().unwrap();

    let transfer = series.try_into_owned_device_column_transfer().unwrap();
    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
    let imported_ptr = imported.view().unwrap().device_data_ptr().unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_ne!(imported_ptr, original_ptr);
    assert_eq!(shared.to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(shared.as_column_view().unwrap().device_data_ptr().unwrap(), original_ptr);
}

#[test]
fn dataframe_try_into_owned_device_transfer_preserves_names_shape_and_contract() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = Series::new_with_runtime(&runtime, "id", &[1i32, 2, 3]).unwrap();
    let values = Series::new_with_runtime(&runtime, "value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_columns([("id", ids), ("value", values)]).unwrap();
    let expected_names = frame.column_names().into_iter().map(str::to_string).collect::<Vec<_>>();

    let transfer = frame.try_into_owned_device_table_transfer().unwrap();

    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer);
    assert_eq!(transfer.table().num_rows(), 3);
    assert_eq!(transfer.columns().len(), 2);
    assert_eq!(transfer.table().names(), Some(expected_names.as_slice()));
    assert_eq!(
        transfer.release_contract().completion(),
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes
    );
    assert_eq!(transfer.release_contract().release_device(), runtime.device().ordinal() as i32);
}

#[test]
fn dataframe_try_into_owned_device_transfer_deep_copies_when_dataframe_columns_are_shared() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = Series::new_with_runtime(&runtime, "id", &[1i32, 2, 3]).unwrap();
    let values = Series::new_with_runtime(&runtime, "value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_columns([("id", ids), ("value", values)]).unwrap();
    let shared = frame.clone();
    let original_id_ptr = shared.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();
    let original_value_ptr = shared.series("value").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    let transfer = frame.try_into_owned_device_table_transfer().unwrap();
    let imported = DataFrame::try_from_owned_device_table_transfer(transfer).unwrap();
    let imported_id_ptr = imported.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();
    let imported_value_ptr = imported.series("value").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    assert_eq!(imported.column_names(), vec!["id", "value"]);
    assert_eq!(imported.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(imported.series("value").unwrap().to_vec::<i64>().unwrap(), vec![10, 20, 30]);
    assert_ne!(imported_id_ptr, original_id_ptr);
    assert_ne!(imported_value_ptr, original_value_ptr);
    assert_eq!(shared.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap(), original_id_ptr);
    assert_eq!(
        shared.series("value").unwrap().as_column_view().unwrap().device_data_ptr().unwrap(),
        original_value_ptr
    );
}

#[test]
fn owned_column_transfer_import_round_trips_original_column_without_copy() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[7i32, 8, 9]).unwrap();
    let expected_ptr = column.view().unwrap().device_data_ptr().unwrap();
    let expected_stream = column.__debug_execution_domain_stream_value();
    let expected_mr = column.__debug_execution_domain_mr_value();

    let transfer = column.try_into_owned_device_column_transfer().unwrap();
    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(imported.view().unwrap().device_data_ptr().unwrap(), expected_ptr);
    assert_eq!(imported.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(imported.__debug_execution_domain_mr_value(), expected_mr);
}

#[test]
fn owned_table_transfer_import_round_trips_original_table_without_copy() {
    let runtime = Runtime::builder().build().unwrap();
    let first = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let second = Column::from_slice_with_runtime(&runtime, &[10i32, 20, 30]).unwrap();
    let expected_first_ptr = first.view().unwrap().device_data_ptr().unwrap();
    let expected_second_ptr = second.view().unwrap().device_data_ptr().unwrap();
    let table = Table::from_columns(vec![first, second]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();

    let transfer = table.try_into_owned_device_table_transfer().unwrap();
    let imported = Table::try_from_owned_device_table_transfer(transfer).unwrap();
    let imported_view = imported.view().unwrap();

    assert_eq!(imported_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(imported_view.column(1).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(imported_view.column(0).unwrap().device_data_ptr().unwrap(), expected_first_ptr);
    assert_eq!(imported_view.column(1).unwrap().device_data_ptr().unwrap(), expected_second_ptr);
    assert_eq!(imported.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(imported.__debug_execution_domain_mr_value(), expected_mr);
}

#[test]
fn owned_column_transfer_import_rejects_non_cudf_owner_state() {
    let transfer = OwnedColumnTransfer::try_new(
        fixed_width_column_semantics(1, 11, 3),
        vec![device_binding(11, 0x1000, 12, 0)],
        manual_readiness(0),
        manual_owned_transfer_diagnostics(),
        manual_release_contract(0),
        String::from("not-a-cudf-column"),
    )
    .unwrap();

    let err = match Column::try_from_owned_device_column_transfer(transfer) {
        Ok(_) => panic!("expected non-cudf owner state to be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn owned_table_transfer_import_rejects_non_cudf_owner_state() {
    let transfer = OwnedTableTransfer::try_new(
        SemanticTable::new(TableId::new(7), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column_semantics(1, 11, 3)],
        vec![device_binding(11, 0x1000, 12, 0)],
        manual_readiness(0),
        manual_owned_transfer_diagnostics(),
        manual_release_contract(0),
        String::from("not-a-cudf-table"),
    )
    .unwrap();

    let err = match Table::try_from_owned_device_table_transfer(transfer) {
        Ok(_) => panic!("expected non-cudf owner state to be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn owned_table_transfer_import_rejects_excessively_nested_wrappers() {
    let runtime = Runtime::builder().build().unwrap();
    let table = Table::from_columns(vec![Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap()]).unwrap();

    let mut transfer = table.try_into_owned_device_table_transfer().unwrap();
    for _ in 0..40 {
        transfer = wrap_owned_table_transfer(transfer);
    }

    let err = match Table::try_from_owned_device_table_transfer(transfer) {
        Ok(_) => panic!("expected excessively nested owned-transfer wrappers to be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    assert!(err.to_string().contains("nesting"), "unexpected error: {err}");
}

#[test]
fn ambient_default_column_owned_transfer_is_rejected() {
    let column = Column::from_slice(&[1i32, 2, 3]).unwrap();

    let err = column.try_into_owned_device_column_transfer().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
}

#[test]
fn ambient_default_series_owned_transfer_is_rejected() {
    let series = Series::new("values", &[1i32, 2, 3]).unwrap();

    let err = series.try_into_owned_device_column_transfer().unwrap_err();

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

#[test]
fn ambient_default_dataframe_owned_transfer_is_rejected() {
    let ids = Series::new("id", &[1i32, 2, 3]).unwrap();
    let values = Series::new("value", &[10i32, 20, 30]).unwrap();
    let frame = DataFrame::from_columns([("id", ids), ("value", values)]).unwrap();

    let err = frame.try_into_owned_device_table_transfer().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
}
