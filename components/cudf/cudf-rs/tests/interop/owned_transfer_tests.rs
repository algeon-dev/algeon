use std::{
    ffi::c_void,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use cudf::{
    DataFrame, DeviceId, JsonReadOptions, Runtime, Series, TypeId,
    core::{Column, Table},
    interop::{
        borrowed::{
            BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, CudaStreamHandle, PhysicalValidity, Placement,
            SchemaNullability, SemanticColumn, SemanticTable, TableId,
        },
        prepared::{ExportDiagnostics, ExportProfile, InteropError, InteropErrorKind, ProducerReadiness},
        runtime::RuntimeCompletionAuthority,
        transfer::{
            OwnedColumnTransfer, OwnedTableTransfer, ReleaseContract, ReleaseSynchronization,
            RuntimeOwnedColumnTransferReleaser, RuntimeOwnedTableTransferReleaser, TransferCompletionRule,
            TryIntoOwnedDeviceColumnTransfer, TryIntoOwnedDeviceTableTransfer,
        },
    },
    ops::dictionary,
    runtime::{CudaEvent, CudaStream},
};
use rapids_interop::{DataType as InteropDataType, TypeId as InteropTypeId};
use static_assertions::assert_impl_all;
use tempfile::TempDir;

use crate::common::lock_event_observation;

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
        stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut _, device).unwrap(),
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

struct DropFlag {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct ForeignColumnOwner {
    _column: Column,
    _drop_flag: DropFlag,
}

struct ForeignTableOwner {
    _table: Table,
    _drop_flag: DropFlag,
}

fn fixed_width_binding_len<T>(len: usize) -> usize {
    len * std::mem::size_of::<T>()
}

fn reset_event_observation() {
    unsafe { cudf_sys::cudf_test_event_reset_observation() };
}

fn observed_event_synchronize_call_count() -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_synchronize_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read event synchronize observation count");
    out
}

fn observed_stream_wait_call_count_for_event_raw(event_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_stream_wait_call_count_for_event_raw(event_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-event wait observation count");
    out
}

fn observed_raw_record_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream raw event-record observation count");
    out
}

fn block_event_synchronize() {
    unsafe { cudf_sys::cudf_test_event_block_synchronize() };
}

fn unblock_event_synchronize() {
    unsafe { cudf_sys::cudf_test_event_unblock_synchronize() };
}

fn fail_event_synchronize() {
    unsafe { cudf_sys::cudf_test_event_fail_synchronize() };
}

fn allow_event_synchronize() {
    unsafe { cudf_sys::cudf_test_event_allow_synchronize() };
}

struct EventSynchronizeBlockGuard;

impl EventSynchronizeBlockGuard {
    fn new() -> Self {
        block_event_synchronize();
        Self
    }
}

impl Drop for EventSynchronizeBlockGuard {
    fn drop(&mut self) {
        unblock_event_synchronize();
    }
}

struct EventSynchronizeFailureGuard;

impl EventSynchronizeFailureGuard {
    fn new() -> Self {
        fail_event_synchronize();
        Self
    }
}

impl Drop for EventSynchronizeFailureGuard {
    fn drop(&mut self) {
        allow_event_synchronize();
    }
}

fn wait_for_condition(timeout: Duration, mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        thread::yield_now();
    }

    assert!(predicate(), "condition was not met within {:?}", timeout);
}

fn has_multiple_gpus() -> bool {
    DeviceId::count().expect("failed to query visible CUDA devices") >= 2
}

fn foreign_owned_column_transfer(
    column: Column,
    device: i32,
    drops: Arc<AtomicUsize>,
) -> (OwnedColumnTransfer, *const std::ffi::c_void) {
    let (len, values_ptr) = {
        let view = column.view().unwrap();
        (view.len().unwrap(), view.device_data_ptr().unwrap())
    };

    let transfer = OwnedColumnTransfer::try_new(
        fixed_width_column_semantics(1, 11, len),
        vec![device_binding(11, values_ptr as usize, fixed_width_binding_len::<i32>(len), device)],
        manual_readiness(device),
        manual_owned_transfer_diagnostics(),
        manual_release_contract(device),
        ForeignColumnOwner { _column: column, _drop_flag: DropFlag { drops } },
    )
    .unwrap();

    (transfer, values_ptr)
}

fn foreign_owned_table_transfer(
    table: Table,
    device: i32,
    drops: Arc<AtomicUsize>,
) -> (OwnedTableTransfer, [*const std::ffi::c_void; 2]) {
    let (num_rows, first_ptr, second_ptr) = {
        let view = table.view().unwrap();
        (
            view.num_rows().unwrap() as usize,
            view.column(0).unwrap().device_data_ptr().unwrap(),
            view.column(1).unwrap().device_data_ptr().unwrap(),
        )
    };

    let transfer = OwnedTableTransfer::try_new(
        SemanticTable::new(TableId::new(7), num_rows, vec![ColumnId::new(1), ColumnId::new(2)], None),
        vec![fixed_width_column_semantics(1, 11, num_rows), fixed_width_column_semantics(2, 12, num_rows)],
        vec![
            device_binding(11, first_ptr as usize, fixed_width_binding_len::<i32>(num_rows), device),
            device_binding(12, second_ptr as usize, fixed_width_binding_len::<i32>(num_rows), device),
        ],
        manual_readiness(device),
        manual_owned_transfer_diagnostics(),
        manual_release_contract(device),
        ForeignTableOwner { _table: table, _drop_flag: DropFlag { drops } },
    )
    .unwrap();

    (transfer, [first_ptr, second_ptr])
}

#[derive(Clone)]
struct ColumnSemanticSnapshot {
    root: SemanticColumn,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
}

#[derive(Clone)]
struct TableSemanticSnapshot {
    table: SemanticTable,
    columns: Vec<SemanticColumn>,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
}

fn clone_binding_as_static(binding: &BufferBinding<'_>) -> BufferBinding<'static> {
    // SAFETY: These tests keep the original export snapshot and retained owner
    // alive for the full lifetime of the cloned bindings, so widening the
    // binding borrow here does not outlive the underlying bytes.
    BufferBinding::new(
        binding.id(),
        BoundBytes::from_raw_parts(binding.bytes().base_ptr() as *mut std::ffi::c_void, binding.bytes().byte_len())
            .unwrap(),
        *binding.placement(),
    )
}

fn read_json_dataframe(filename: &str, rows: &str) -> DataFrame {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(filename);
    std::fs::write(&path, rows).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::read_json_with_options(JsonReadOptions::builder().filepath(&path).with_runtime(&runtime).build())
        .unwrap()
}

fn snapshot_foreign_owned_column_transfer(
    column: Column,
    drops: Arc<AtomicUsize>,
) -> (OwnedColumnTransfer, ColumnSemanticSnapshot) {
    let (device, snapshot) = {
        let export = column.try_prepare_device_column_export().unwrap();
        let device = match export.readiness() {
            ProducerReadiness::DeviceStreamOrdered { stream } => stream.device(),
            ProducerReadiness::DeviceEventOrdered { event } => event.device(),
            ProducerReadiness::HostVisible => panic!("owned device transfer tests require device-backed readiness"),
        };
        let snapshot = ColumnSemanticSnapshot {
            root: export.column().clone(),
            semantic_columns: export.semantic_columns().to_vec(),
            bindings: export.bindings().iter().map(clone_binding_as_static).collect(),
        };
        (device, snapshot)
    };
    let transfer = OwnedColumnTransfer::try_new_with_semantic_columns(
        snapshot.root.clone(),
        snapshot.semantic_columns.clone(),
        snapshot.bindings.clone(),
        manual_readiness(device),
        manual_owned_transfer_diagnostics(),
        manual_release_contract(device),
        ForeignColumnOwner { _column: column, _drop_flag: DropFlag { drops } },
    )
    .unwrap();
    (transfer, snapshot)
}

fn snapshot_foreign_owned_table_transfer(
    table: Table,
    drops: Arc<AtomicUsize>,
) -> (OwnedTableTransfer, TableSemanticSnapshot) {
    let (device, snapshot) = {
        let export = table.try_prepare_device_table_export().unwrap();
        let device = match export.readiness() {
            ProducerReadiness::DeviceStreamOrdered { stream } => stream.device(),
            ProducerReadiness::DeviceEventOrdered { event } => event.device(),
            ProducerReadiness::HostVisible => panic!("owned device transfer tests require device-backed readiness"),
        };
        let snapshot = TableSemanticSnapshot {
            table: export.table().clone(),
            columns: export.columns().to_vec(),
            semantic_columns: export.semantic_columns().to_vec(),
            bindings: export.bindings().iter().map(clone_binding_as_static).collect(),
        };
        (device, snapshot)
    };
    let transfer = OwnedTableTransfer::try_new_with_semantic_columns(
        snapshot.table.clone(),
        snapshot.columns.clone(),
        snapshot.semantic_columns.clone(),
        snapshot.bindings.clone(),
        manual_readiness(device),
        manual_owned_transfer_diagnostics(),
        manual_release_contract(device),
        ForeignTableOwner { _table: table, _drop_flag: DropFlag { drops } },
    )
    .unwrap();
    (transfer, snapshot)
}

fn binding_ptr_map<'a>(bindings: &'a [BufferBinding<'a>]) -> std::collections::HashMap<u32, &'a BufferBinding<'a>> {
    bindings.iter().map(|binding| (binding.id().get(), binding)).collect()
}

fn semantic_column_map<'a>(columns: &'a [SemanticColumn]) -> std::collections::HashMap<u32, &'a SemanticColumn> {
    columns.iter().map(|column| (column.id().get(), column)).collect()
}

fn assert_binding_preserves_zero_copy(source: &BufferBinding<'_>, imported: &BufferBinding<'_>) {
    assert_eq!(source.placement(), imported.placement());
    assert_eq!(source.bytes().base_ptr(), imported.bytes().base_ptr());
    assert_eq!(source.bytes().byte_len(), imported.bytes().byte_len());
}

fn assert_semantic_column_matches_snapshot(
    expected: &SemanticColumn,
    actual: &SemanticColumn,
    expected_columns: &std::collections::HashMap<u32, &SemanticColumn>,
    actual_columns: &std::collections::HashMap<u32, &SemanticColumn>,
    expected_bindings: &std::collections::HashMap<u32, &BufferBinding<'_>>,
    actual_bindings: &std::collections::HashMap<u32, &BufferBinding<'_>>,
) {
    assert_eq!(expected.data_type(), actual.data_type());
    assert_eq!(expected.len(), actual.len());
    assert_eq!(expected.offset(), actual.offset());
    assert_eq!(expected.validity(), actual.validity());
    assert_eq!(expected.schema_nullability(), actual.schema_nullability());

    match (expected.layout(), actual.layout()) {
        (ColumnLayout::FixedWidth { values: expected_values }, ColumnLayout::FixedWidth { values: actual_values }) => {
            assert_binding_preserves_zero_copy(
                expected_bindings.get(&expected_values.get()).expect("missing expected fixed-width binding"),
                actual_bindings.get(&actual_values.get()).expect("missing imported fixed-width binding"),
            );
        },
        (
            ColumnLayout::BinaryLike { offsets: expected_offsets, data: expected_data },
            ColumnLayout::BinaryLike { offsets: actual_offsets, data: actual_data },
        ) => {
            assert_binding_preserves_zero_copy(
                expected_bindings.get(&expected_offsets.get()).expect("missing expected string offsets binding"),
                actual_bindings.get(&actual_offsets.get()).expect("missing imported string offsets binding"),
            );
            assert_binding_preserves_zero_copy(
                expected_bindings.get(&expected_data.get()).expect("missing expected string chars binding"),
                actual_bindings.get(&actual_data.get()).expect("missing imported string chars binding"),
            );
        },
        (
            ColumnLayout::List { offsets: expected_offsets, child: expected_child },
            ColumnLayout::List { offsets: actual_offsets, child: actual_child },
        ) => {
            assert_binding_preserves_zero_copy(
                expected_bindings.get(&expected_offsets.get()).expect("missing expected list offsets binding"),
                actual_bindings.get(&actual_offsets.get()).expect("missing imported list offsets binding"),
            );
            assert_semantic_column_matches_snapshot(
                expected_columns.get(&expected_child.get()).expect("missing expected list child"),
                actual_columns.get(&actual_child.get()).expect("missing imported list child"),
                expected_columns,
                actual_columns,
                expected_bindings,
                actual_bindings,
            );
        },
        (ColumnLayout::Struct { children: expected_children }, ColumnLayout::Struct { children: actual_children }) => {
            assert_eq!(expected_children.len(), actual_children.len());
            for (expected_child, actual_child) in expected_children.iter().zip(actual_children.iter()) {
                assert_semantic_column_matches_snapshot(
                    expected_columns.get(&expected_child.get()).expect("missing expected struct child"),
                    actual_columns.get(&actual_child.get()).expect("missing imported struct child"),
                    expected_columns,
                    actual_columns,
                    expected_bindings,
                    actual_bindings,
                );
            }
        },
        (
            ColumnLayout::Dictionary { indices: expected_indices, keys: expected_keys },
            ColumnLayout::Dictionary { indices: actual_indices, keys: actual_keys },
        ) => {
            assert_binding_preserves_zero_copy(
                expected_bindings.get(&expected_indices.get()).expect("missing expected dictionary indices binding"),
                actual_bindings.get(&actual_indices.get()).expect("missing imported dictionary indices binding"),
            );
            assert_semantic_column_matches_snapshot(
                expected_columns.get(&expected_keys.get()).expect("missing expected dictionary keys"),
                actual_columns.get(&actual_keys.get()).expect("missing imported dictionary keys"),
                expected_columns,
                actual_columns,
                expected_bindings,
                actual_bindings,
            );
        },
        _ => panic!("layout mismatch: expected={:?} actual={:?}", expected.layout(), actual.layout()),
    }
}

fn assert_column_export_matches_snapshot(snapshot: &ColumnSemanticSnapshot, imported: &Column) {
    let export = imported.try_prepare_device_column_export().unwrap();
    let expected_columns = semantic_column_map(&snapshot.semantic_columns);
    let actual_columns = semantic_column_map(export.semantic_columns());
    let expected_bindings = binding_ptr_map(&snapshot.bindings);
    let actual_bindings = binding_ptr_map(export.bindings());

    assert_semantic_column_matches_snapshot(
        &snapshot.root,
        export.column(),
        &expected_columns,
        &actual_columns,
        &expected_bindings,
        &actual_bindings,
    );
}

fn assert_table_export_matches_snapshot(snapshot: &TableSemanticSnapshot, imported: &Table) {
    let export = imported.try_prepare_device_table_export().unwrap();
    let expected_columns = semantic_column_map(&snapshot.semantic_columns);
    let actual_columns = semantic_column_map(export.semantic_columns());
    let expected_bindings = binding_ptr_map(&snapshot.bindings);
    let actual_bindings = binding_ptr_map(export.bindings());

    assert_eq!(snapshot.table.num_rows(), export.table().num_rows());
    assert_eq!(snapshot.table.columns().len(), export.table().columns().len());

    for (expected_column, actual_column) in snapshot.columns.iter().zip(export.columns().iter()) {
        let expected_root =
            expected_columns.get(&expected_column.id().get()).expect("missing expected table root column");
        let actual_root = actual_columns.get(&actual_column.id().get()).expect("missing imported table root column");
        assert_semantic_column_matches_snapshot(
            expected_root,
            actual_root,
            &expected_columns,
            &actual_columns,
            &expected_bindings,
            &actual_bindings,
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum OwnedTransferMode {
    Plain,
    RuntimeRelease(ReleaseSynchronization),
}

impl OwnedTransferMode {
    fn label(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::RuntimeRelease(ReleaseSynchronization::Asynchronous) => "runtime_release_async",
            Self::RuntimeRelease(ReleaseSynchronization::MaySynchronizeInternally) => "runtime_release_may_synchronize",
            Self::RuntimeRelease(ReleaseSynchronization::RequiresCallerSynchronization) => {
                "runtime_release_requires_caller_synchronization"
            },
        }
    }

    fn expected_completion(self) -> TransferCompletionRule {
        match self {
            Self::Plain => TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            Self::RuntimeRelease(_) => TransferCompletionRule::ReleaseWaitsOnConsumerSignal,
        }
    }

    fn expected_synchronization(self) -> ReleaseSynchronization {
        match self {
            Self::Plain => ReleaseSynchronization::MaySynchronizeInternally,
            Self::RuntimeRelease(sync) => sync,
        }
    }
}

fn owned_transfer_modes() -> [OwnedTransferMode; 3] {
    [
        OwnedTransferMode::Plain,
        OwnedTransferMode::RuntimeRelease(ReleaseSynchronization::Asynchronous),
        OwnedTransferMode::RuntimeRelease(ReleaseSynchronization::MaySynchronizeInternally),
    ]
}

fn sample_named_dataframe(runtime: &Runtime) -> DataFrame {
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let values = runtime.series("value", &[10i64, 20, 30]).unwrap();
    DataFrame::from_series([ids, values]).unwrap()
}

fn sample_table(runtime: &Runtime) -> Table {
    let (table, _names) = sample_named_dataframe(runtime).try_into_table().unwrap();
    table
}

fn assert_release_contract_matches(contract: &ReleaseContract, expected_device: i32, mode: OwnedTransferMode) {
    assert_eq!(contract.completion(), mode.expected_completion(), "{}", mode.label());
    assert_eq!(contract.synchronization(), mode.expected_synchronization(), "{}", mode.label());
    assert_eq!(contract.release_device(), expected_device, "{}", mode.label());
}

fn assert_device_stream_readiness_matches(
    readiness: &ProducerReadiness<'_>,
    expected_stream: usize,
    expected_device: i32,
    mode: OwnedTransferMode,
) {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            assert_eq!(stream.as_raw() as usize, expected_stream, "{}", mode.label());
            assert_eq!(stream.device(), expected_device, "{}", mode.label());
        },
        readiness => panic!("{}: expected device-stream readiness, got {readiness:?}", mode.label()),
    }
}

fn assert_owned_column_transfer_contract(
    transfer: &OwnedColumnTransfer,
    expected_len: usize,
    expected_ptr: *const c_void,
    expected_stream: usize,
    expected_device: i32,
    mode: OwnedTransferMode,
) {
    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer, "{}", mode.label());
    assert_eq!(transfer.column().len(), expected_len, "{}", mode.label());
    assert!(binding_ptr_eq(transfer, expected_ptr), "{}", mode.label());
    assert_release_contract_matches(transfer.release_contract(), expected_device, mode);
    assert_device_stream_readiness_matches(transfer.readiness(), expected_stream, expected_device, mode);
}

fn assert_owned_table_transfer_contract(
    transfer: &OwnedTableTransfer,
    expected_rows: usize,
    expected_columns: usize,
    expected_names: Option<&[String]>,
    expected_device: i32,
    mode: OwnedTransferMode,
) {
    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer, "{}", mode.label());
    assert_eq!(transfer.table().num_rows(), expected_rows, "{}", mode.label());
    assert_eq!(transfer.columns().len(), expected_columns, "{}", mode.label());
    assert_eq!(transfer.table().names(), expected_names, "{}", mode.label());
    assert_release_contract_matches(transfer.release_contract(), expected_device, mode);
}

fn take_owned_column_transfer(
    column: Column,
    mode: OwnedTransferMode,
) -> (OwnedColumnTransfer, Option<RuntimeOwnedColumnTransferReleaser>) {
    match mode {
        OwnedTransferMode::Plain => (column.try_into_owned_transfer().unwrap(), None),
        OwnedTransferMode::RuntimeRelease(ReleaseSynchronization::Asynchronous) => {
            let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();
            (transfer, Some(releaser))
        },
        OwnedTransferMode::RuntimeRelease(sync) => {
            let (transfer, releaser) = column.try_into_owned_transfer_with_release_synchronization(sync).unwrap();
            (transfer, Some(releaser))
        },
    }
}

fn take_owned_series_transfer(
    series: Series,
    mode: OwnedTransferMode,
) -> (OwnedColumnTransfer, Option<RuntimeOwnedColumnTransferReleaser>) {
    match mode {
        OwnedTransferMode::Plain => (series.try_into_owned_transfer().unwrap(), None),
        OwnedTransferMode::RuntimeRelease(ReleaseSynchronization::Asynchronous) => {
            let (transfer, releaser) = series.try_into_owned_transfer_with_release().unwrap();
            (transfer, Some(releaser))
        },
        OwnedTransferMode::RuntimeRelease(sync) => {
            let (transfer, releaser) = series.try_into_owned_transfer_with_release_synchronization(sync).unwrap();
            (transfer, Some(releaser))
        },
    }
}

fn take_owned_table_transfer(
    table: Table,
    mode: OwnedTransferMode,
) -> (OwnedTableTransfer, Option<RuntimeOwnedTableTransferReleaser>) {
    match mode {
        OwnedTransferMode::Plain => (table.try_into_owned_transfer().unwrap(), None),
        OwnedTransferMode::RuntimeRelease(ReleaseSynchronization::Asynchronous) => {
            let (transfer, releaser) = table.try_into_owned_transfer_with_release().unwrap();
            (transfer, Some(releaser))
        },
        OwnedTransferMode::RuntimeRelease(sync) => {
            let (transfer, releaser) = table.try_into_owned_transfer_with_release_synchronization(sync).unwrap();
            (transfer, Some(releaser))
        },
    }
}

fn take_owned_dataframe_transfer(
    frame: DataFrame,
    mode: OwnedTransferMode,
) -> (OwnedTableTransfer, Option<RuntimeOwnedTableTransferReleaser>) {
    match mode {
        OwnedTransferMode::Plain => (frame.try_into_owned_transfer().unwrap(), None),
        OwnedTransferMode::RuntimeRelease(ReleaseSynchronization::Asynchronous) => {
            let (transfer, releaser) = frame.try_into_owned_transfer_with_release().unwrap();
            (transfer, Some(releaser))
        },
        OwnedTransferMode::RuntimeRelease(sync) => {
            let (transfer, releaser) = frame.try_into_owned_transfer_with_release_synchronization(sync).unwrap();
            (transfer, Some(releaser))
        },
    }
}

fn finish_column_runtime_release(releaser: Option<RuntimeOwnedColumnTransferReleaser>) {
    if let Some(releaser) = releaser {
        let consumer_stream = CudaStream::new().unwrap();
        releaser.release(RuntimeCompletionAuthority::same_device_stream(&consumer_stream)).unwrap();
    }
}

fn finish_table_runtime_release(releaser: Option<RuntimeOwnedTableTransferReleaser>) {
    if let Some(releaser) = releaser {
        let consumer_stream = CudaStream::new().unwrap();
        releaser.release(RuntimeCompletionAuthority::same_device_stream(&consumer_stream)).unwrap();
    }
}

fn assert_foreign_column_snapshot_round_trip(
    transfer: OwnedColumnTransfer,
    snapshot: &ColumnSemanticSnapshot,
    drops: &Arc<AtomicUsize>,
    value_assertion: impl FnOnce(&Column),
) {
    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
    value_assertion(&imported);
    assert_eq!(imported.__debug_execution_domain_kind(), "explicit");
    assert_column_export_matches_snapshot(snapshot, &imported);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

fn assert_ambient_default_owned_transfer_rejected(err: InteropError) {
    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
}

#[test]
fn column_owned_device_transfer_contracts_cover_direct_and_runtime_release_modes() {
    for mode in owned_transfer_modes() {
        let runtime = Runtime::builder().build().unwrap();
        let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
        let expected_ptr = column.view().unwrap().device_data_ptr().unwrap();
        let expected_stream = column.__debug_execution_domain_stream_value();
        let expected_device = runtime.device().ordinal() as i32;

        let (transfer, releaser) = take_owned_column_transfer(column, mode);
        assert_owned_column_transfer_contract(&transfer, 3, expected_ptr, expected_stream, expected_device, mode);

        drop(transfer);
        finish_column_runtime_release(releaser);
    }
}

#[test]
fn table_owned_device_transfer_contracts_cover_direct_and_runtime_release_modes() {
    for mode in owned_transfer_modes() {
        let runtime = Runtime::builder().build().unwrap();
        let table = sample_table(&runtime);
        let expected_device = runtime.device().ordinal() as i32;

        let (transfer, releaser) = take_owned_table_transfer(table, mode);
        assert_owned_table_transfer_contract(&transfer, 3, 2, None, expected_device, mode);

        drop(transfer);
        finish_table_runtime_release(releaser);
    }
}

#[test]
fn series_owned_device_transfer_contracts_cover_direct_and_runtime_release_modes() {
    for mode in owned_transfer_modes() {
        let runtime = Runtime::builder().build().unwrap();
        let series = runtime.series("numbers", &[1i32, 2, 3]).unwrap();
        let expected_ptr = series.as_column_view().unwrap().device_data_ptr().unwrap();
        let expected_stream = series.as_column().__debug_execution_domain_stream_value();
        let expected_device = runtime.device().ordinal() as i32;

        let (transfer, releaser) = take_owned_series_transfer(series, mode);
        assert_owned_column_transfer_contract(&transfer, 3, expected_ptr, expected_stream, expected_device, mode);

        drop(transfer);
        finish_column_runtime_release(releaser);
    }
}

#[test]
fn series_try_into_owned_device_transfer_deep_copies_when_series_data_is_shared() {
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("numbers", &[7i32, 8, 9]).unwrap();
    let shared = series.clone();
    let original_ptr = shared.as_column_view().unwrap().device_data_ptr().unwrap();

    let transfer = series.try_into_owned_transfer().unwrap();
    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
    let imported_ptr = imported.view().unwrap().device_data_ptr().unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_ne!(imported_ptr, original_ptr);
    assert_eq!(shared.to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(shared.as_column_view().unwrap().device_data_ptr().unwrap(), original_ptr);
}

#[test]
fn dataframe_owned_device_transfer_contracts_cover_direct_and_runtime_release_modes() {
    for mode in owned_transfer_modes() {
        let runtime = Runtime::builder().build().unwrap();
        let frame = sample_named_dataframe(&runtime);
        let expected_names = frame.column_names().into_iter().map(str::to_string).collect::<Vec<_>>();
        let expected_device = runtime.device().ordinal() as i32;

        let (transfer, releaser) = take_owned_dataframe_transfer(frame, mode);
        assert_owned_table_transfer_contract(&transfer, 3, 2, Some(expected_names.as_slice()), expected_device, mode);

        drop(transfer);
        finish_table_runtime_release(releaser);
    }
}

#[test]
fn runtime_release_owned_column_transfer_release_waits_on_same_device_stream() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();
    drop(transfer);

    let consumer_stream = CudaStream::new().unwrap();
    let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

    reset_event_observation();
    let _sync_block = EventSynchronizeBlockGuard::new();
    let baseline_sync = observed_event_synchronize_call_count();
    let baseline_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);

    let (tx, rx) = mpsc::channel();
    let release_thread = thread::spawn(move || {
        let result = releaser.release(RuntimeCompletionAuthority::same_device_stream(&consumer_stream));
        tx.send(result).unwrap();
    });

    rx.recv_timeout(Duration::from_secs(2))
        .expect("runtime-backed same-device release did not return promptly")
        .unwrap();
    wait_for_condition(Duration::from_secs(2), || {
        observed_raw_record_call_count_for_stream_raw(consumer_stream_raw) >= baseline_record + 1
    });
    wait_for_condition(Duration::from_secs(2), || observed_event_synchronize_call_count() >= baseline_sync + 1);
    release_thread.join().unwrap();
}

#[test]
fn runtime_release_owned_column_transfer_release_waits_on_consumer_event() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();
    drop(transfer);

    let consumer_stream = CudaStream::new().unwrap();
    let consumer_event = CudaEvent::new().unwrap();
    consumer_event.record(&consumer_stream).unwrap();
    let consumer_event_raw = consumer_event.__debug_interop_raw_handle_value();

    reset_event_observation();
    let _sync_block = EventSynchronizeBlockGuard::new();
    let baseline_sync = observed_event_synchronize_call_count();
    let baseline_wait = observed_stream_wait_call_count_for_event_raw(consumer_event_raw);

    let release_event = consumer_event.clone();
    let (tx, rx) = mpsc::channel();
    let release_thread = thread::spawn(move || {
        let result = releaser.release(RuntimeCompletionAuthority::cross_device_event(&release_event));
        tx.send(result).unwrap();
    });

    rx.recv_timeout(Duration::from_secs(2))
        .expect("runtime-backed consumer-event release did not return promptly")
        .unwrap();
    wait_for_condition(Duration::from_secs(2), || {
        observed_stream_wait_call_count_for_event_raw(consumer_event_raw) >= baseline_wait + 1
    });
    wait_for_condition(Duration::from_secs(2), || observed_event_synchronize_call_count() >= baseline_sync + 1);
    release_thread.join().unwrap();
}

#[test]
fn runtime_release_owned_column_transfer_rejects_synchronous_by_return_completion() {
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();
    drop(transfer);

    let err = releaser.release(RuntimeCompletionAuthority::SynchronousByReturn).unwrap_err();
    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn runtime_release_owned_column_transfer_with_may_synchronize_internally_blocks_caller_thread_until_consumer_signal() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let (transfer, releaser) = column
        .try_into_owned_transfer_with_release_synchronization(ReleaseSynchronization::MaySynchronizeInternally)
        .unwrap();
    drop(transfer);

    let consumer_stream = CudaStream::new().unwrap();
    let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

    reset_event_observation();
    let sync_block = EventSynchronizeBlockGuard::new();
    let baseline_sync = observed_event_synchronize_call_count();
    let baseline_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);

    let (tx, rx) = mpsc::channel();
    let release_thread = thread::spawn(move || {
        let result = releaser.release(RuntimeCompletionAuthority::same_device_stream(&consumer_stream));
        tx.send(result).unwrap();
    });

    wait_for_condition(Duration::from_secs(2), || {
        observed_raw_record_call_count_for_stream_raw(consumer_stream_raw) >= baseline_record + 1
    });
    wait_for_condition(Duration::from_secs(2), || observed_event_synchronize_call_count() >= baseline_sync + 1);
    assert!(
        rx.recv_timeout(Duration::from_millis(200)).is_err(),
        "blocking runtime-backed release unexpectedly returned before synchronize unblocked",
    );

    drop(sync_block);
    rx.recv_timeout(Duration::from_secs(2))
        .expect("blocking runtime-backed release did not finish after synchronize unblocked")
        .unwrap();
    release_thread.join().unwrap();
}

#[test]
fn runtime_release_owned_column_transfer_synchronization_selection_rejects_requires_caller_synchronization() {
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();

    let err = match column
        .try_into_owned_transfer_with_release_synchronization(ReleaseSynchronization::RequiresCallerSynchronization)
    {
        Ok(_) => panic!("expected caller-managed synchronization to be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn runtime_release_owned_column_transfer_rejects_wrong_device_stream_synchronously() {
    if !has_multiple_gpus() {
        eprintln!("skipping wrong-device runtime release test because fewer than 2 CUDA devices are visible");
        return;
    }

    let runtime = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();
    drop(transfer);

    let other_runtime = Runtime::builder().with_device(DeviceId::new(1).unwrap()).build().unwrap();
    let err = releaser.release(RuntimeCompletionAuthority::same_device_stream(other_runtime.stream())).unwrap_err();
    assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);
}

#[test]
fn runtime_release_owned_column_transfer_import_claims_native_ownership_and_invalidates_releaser() {
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[7i32, 8, 9]).unwrap();
    let expected_ptr = column.view().unwrap().device_data_ptr().unwrap();
    let expected_stream = column.__debug_execution_domain_stream_value();
    let expected_mr = column.__debug_execution_domain_mr_value();

    let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();
    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(imported.view().unwrap().device_data_ptr().unwrap(), expected_ptr);
    assert_eq!(imported.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(imported.__debug_execution_domain_mr_value(), expected_mr);

    drop(imported);

    let consumer_stream = CudaStream::new().unwrap();
    let err = releaser.release(RuntimeCompletionAuthority::same_device_stream(&consumer_stream)).unwrap_err();
    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn runtime_release_owned_column_transfer_wait_failure_keeps_transfer_importable() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[7i32, 8, 9]).unwrap();
    let expected_ptr = column.view().unwrap().device_data_ptr().unwrap();
    let (transfer, releaser) = column
        .try_into_owned_transfer_with_release_synchronization(ReleaseSynchronization::MaySynchronizeInternally)
        .unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let consumer_event = CudaEvent::new().unwrap();
    consumer_event.record(&consumer_stream).unwrap();

    reset_event_observation();
    let _sync_failure = EventSynchronizeFailureGuard::new();

    let err = releaser.release(RuntimeCompletionAuthority::cross_device_event(&consumer_event)).unwrap_err();
    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);

    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(imported.view().unwrap().device_data_ptr().unwrap(), expected_ptr);
}

#[test]
fn dataframe_try_into_owned_device_transfer_deep_copies_when_dataframe_columns_are_shared() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let values = runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, values]).unwrap();
    let shared = frame.clone();
    let original_id_ptr = shared.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();
    let original_value_ptr = shared.series("value").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    let transfer = frame.try_into_owned_transfer().unwrap();
    let imported = DataFrame::try_from_owned_device_table_transfer(transfer).unwrap();
    let imported_id_ptr = imported.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();
    let imported_value_ptr = imported.series("value").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    assert_eq!(imported.column_names(), vec!["id", "value"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(imported.series("value").unwrap().to_vec_strict::<i64>().unwrap(), vec![10, 20, 30]);
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
    let column = runtime.column_from_slice(&[7i32, 8, 9]).unwrap();
    let expected_ptr = column.view().unwrap().device_data_ptr().unwrap();
    let expected_stream = column.__debug_execution_domain_stream_value();
    let expected_mr = column.__debug_execution_domain_mr_value();

    let transfer = column.try_into_owned_transfer().unwrap();
    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(imported.view().unwrap().device_data_ptr().unwrap(), expected_ptr);
    assert_eq!(imported.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(imported.__debug_execution_domain_mr_value(), expected_mr);
}

#[test]
fn owned_table_transfer_import_round_trips_original_table_without_copy() {
    let runtime = Runtime::builder().build().unwrap();
    let first = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let second = runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let expected_first_ptr = first.view().unwrap().device_data_ptr().unwrap();
    let expected_second_ptr = second.view().unwrap().device_data_ptr().unwrap();
    let table = Table::from_columns(vec![first, second]).unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();

    let transfer = table.try_into_owned_transfer().unwrap();
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
fn owned_column_transfer_import_accepts_foreign_owner_state_zero_copy() {
    let runtime = Runtime::builder().build().unwrap();
    let device = runtime.device().ordinal() as i32;
    let source = runtime.column_from_slice(&[11i32, 22, 33]).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, expected_ptr) = foreign_owned_column_transfer(source, device, Arc::clone(&drops));

    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
    let imported_view = imported.view().unwrap();
    let reexport = imported.try_prepare_device_column_export().unwrap();

    assert_eq!(imported_view.to_vec::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(imported_view.device_data_ptr().unwrap(), expected_ptr);
    assert_eq!(imported.__debug_execution_domain_kind(), "explicit");
    assert!(
        reexport.bindings().iter().any(|binding| binding.bytes().base_ptr() == expected_ptr),
        "prepared export must preserve the foreign device address",
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn owned_column_transfer_import_retains_foreign_owner_until_drop() {
    let runtime = Runtime::builder().build().unwrap();
    let device = runtime.device().ordinal() as i32;
    let source = runtime.column_from_slice(&[5i32, 6, 7]).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, _) = foreign_owned_column_transfer(source, device, Arc::clone(&drops));

    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    let _ = imported.view().unwrap();
    let _ = imported.try_prepare_device_column_export().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(imported);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn owned_table_transfer_import_accepts_foreign_owner_state_and_round_trips_columns_zero_copy() {
    let runtime = Runtime::builder().build().unwrap();
    let device = runtime.device().ordinal() as i32;
    let first = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let second = runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![first, second]).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, expected_ptrs) = foreign_owned_table_transfer(table, device, Arc::clone(&drops));

    let imported = Table::try_from_owned_device_table_transfer(transfer).unwrap();
    let reexport = imported.try_prepare_device_table_export().unwrap();
    {
        let imported_view = imported.view().unwrap();
        assert_eq!(imported_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
        assert_eq!(imported_view.column(1).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
        assert_eq!(imported_view.column(0).unwrap().device_data_ptr().unwrap(), expected_ptrs[0]);
        assert_eq!(imported_view.column(1).unwrap().device_data_ptr().unwrap(), expected_ptrs[1]);
    }
    assert_eq!(imported.__debug_execution_domain_kind(), "explicit");
    assert!(
        reexport.bindings().iter().any(|binding| binding.bytes().base_ptr() == expected_ptrs[0]),
        "prepared export must preserve the first foreign device address",
    );
    assert!(
        reexport.bindings().iter().any(|binding| binding.bytes().base_ptr() == expected_ptrs[1]),
        "prepared export must preserve the second foreign device address",
    );

    let released = imported.into_columns().unwrap();
    assert_eq!(released.len(), 2);
    assert_eq!(released[0].view().unwrap().device_data_ptr().unwrap(), expected_ptrs[0]);
    assert_eq!(released[1].view().unwrap().device_data_ptr().unwrap(), expected_ptrs[1]);

    let rebuilt = Table::from_columns(released).unwrap();
    let rebuilt_view = rebuilt.view().unwrap();
    assert_eq!(rebuilt_view.column(0).unwrap().device_data_ptr().unwrap(), expected_ptrs[0]);
    assert_eq!(rebuilt_view.column(1).unwrap().device_data_ptr().unwrap(), expected_ptrs[1]);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn owned_table_transfer_into_columns_preserves_foreign_owner_liveness() {
    let runtime = Runtime::builder().build().unwrap();
    let device = runtime.device().ordinal() as i32;
    let first = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let second = runtime.column_from_slice(&[4i32, 5, 6]).unwrap();
    let table = Table::from_columns(vec![first, second]).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, _) = foreign_owned_table_transfer(table, device, Arc::clone(&drops));

    let imported = Table::try_from_owned_device_table_transfer(transfer).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    let mut columns = imported.into_columns().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(columns.pop().unwrap());
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(columns);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn owned_column_transfer_import_accepts_foreign_string_owner_state_zero_copy() {
    let runtime = Runtime::builder().build().unwrap();
    let source = runtime.column_from_strs(&["alpha", "beta", "gamma"]).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, snapshot) = snapshot_foreign_owned_column_transfer(source, Arc::clone(&drops));

    assert_foreign_column_snapshot_round_trip(transfer, &snapshot, &drops, |imported| {
        assert_eq!(imported.view().unwrap().to_strings_lossy().unwrap(), vec!["alpha", "beta", "gamma"]);
    });
}

#[test]
fn owned_column_transfer_import_accepts_foreign_list_owner_state_zero_copy() {
    let frame = read_json_dataframe(
        "owned_transfer_list.jsonl",
        "{\"lists\":[1,2]}\n{\"lists\":[3]}\n{\"lists\":[]}\n{\"lists\":[4,5,6]}\n",
    );
    let source = frame.series("lists").unwrap().as_column().deep_copy().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, snapshot) = snapshot_foreign_owned_column_transfer(source, Arc::clone(&drops));

    assert_foreign_column_snapshot_round_trip(transfer, &snapshot, &drops, |imported| {
        assert_eq!(imported.view().unwrap().data_type().unwrap().id(), TypeId::List);
    });
}

#[test]
fn owned_column_transfer_import_accepts_foreign_struct_owner_state_zero_copy() {
    let frame = read_json_dataframe(
        "owned_transfer_struct.jsonl",
        "{\"point\":{\"coords\":[1,2],\"label\":\"one\"}}\n\
         {\"point\":{\"coords\":[],\"label\":\"two\"}}\n\
         {\"point\":{\"coords\":[3],\"label\":\"three\"}}\n",
    );
    let source = frame.series("point").unwrap().as_column().deep_copy().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, snapshot) = snapshot_foreign_owned_column_transfer(source, Arc::clone(&drops));

    assert_foreign_column_snapshot_round_trip(transfer, &snapshot, &drops, |imported| {
        assert_eq!(imported.view().unwrap().data_type().unwrap().id(), TypeId::Struct);
    });
}

#[test]
fn owned_column_transfer_import_accepts_foreign_dictionary_owner_state_zero_copy() {
    let runtime = Runtime::builder().build().unwrap();
    let source = runtime.column_from_strs(&["beta", "alpha", "beta", "gamma"]).unwrap();
    let dictionary = dictionary::encode(&source.view().unwrap(), TypeId::Int32).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, snapshot) = snapshot_foreign_owned_column_transfer(dictionary, Arc::clone(&drops));

    assert_foreign_column_snapshot_round_trip(transfer, &snapshot, &drops, |imported| {
        let decoded = dictionary::decode(&imported.view().unwrap()).unwrap();
        assert_eq!(decoded.view().unwrap().to_strings_lossy().unwrap(), vec!["beta", "alpha", "beta", "gamma"]);
        assert_eq!(imported.view().unwrap().data_type().unwrap().id(), TypeId::Dictionary32);
    });
}

#[test]
fn owned_table_transfer_import_accepts_foreign_nested_columns_zero_copy() {
    let frame = read_json_dataframe(
        "owned_transfer_nested_table.jsonl",
        "{\"name\":\"alice\",\"lists\":[1,2],\"point\":{\"coords\":[1,2],\"label\":\"one\"},\"category\":\"beta\"}\n\
         {\"name\":\"bob\",\"lists\":[3],\"point\":{\"coords\":[],\"label\":\"two\"},\"category\":\"alpha\"}\n\
         {\"name\":\"cara\",\"lists\":[],\"point\":{\"coords\":[3],\"label\":\"three\"},\"category\":\"beta\"}\n",
    );
    let name = frame.series("name").unwrap();
    let lists = frame.series("lists").unwrap();
    let point = frame.series("point").unwrap();
    let category_source = frame.series("category").unwrap().as_column().deep_copy().unwrap();
    let category = Series::from_column(dictionary::encode(&category_source.view().unwrap(), TypeId::Int32).unwrap())
        .unwrap()
        .rename("category");
    let frame = DataFrame::from_series([name, lists, point, category]).unwrap();
    let (source, _names) = frame.try_into_table().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (transfer, snapshot) = snapshot_foreign_owned_table_transfer(source, Arc::clone(&drops));

    let imported = Table::try_from_owned_device_table_transfer(transfer).unwrap();
    let released = imported.into_columns().unwrap();
    let rebuilt = Table::from_columns(released).unwrap();

    assert_eq!(rebuilt.__debug_execution_domain_kind(), "explicit");
    assert_table_export_matches_snapshot(&snapshot, &rebuilt);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn owned_table_transfer_import_rejects_excessively_nested_wrappers() {
    let runtime = Runtime::builder().build().unwrap();
    let table = Table::from_columns(vec![runtime.column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let mut transfer = table.try_into_owned_transfer().unwrap();
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
fn ambient_default_owned_transfer_is_rejected_across_column_series_table_and_dataframe() {
    let column = Column::from_slice(&[1i32, 2, 3]).unwrap();
    assert_ambient_default_owned_transfer_rejected(column.try_into_owned_transfer().unwrap_err());

    let series = Series::new("values", &[1i32, 2, 3]).unwrap();
    assert_ambient_default_owned_transfer_rejected(series.try_into_owned_transfer().unwrap_err());

    let c0 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c1 = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    assert_ambient_default_owned_transfer_rejected(table.try_into_owned_transfer().unwrap_err());

    let ids = Series::new("id", &[1i32, 2, 3]).unwrap();
    let values = Series::new("value", &[10i32, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, values]).unwrap();
    assert_ambient_default_owned_transfer_rejected(frame.try_into_owned_transfer().unwrap_err());
}

// ---------------------------------------------------------------------------
// Abandoned transfer cleanup (Drop safety)
// ---------------------------------------------------------------------------

#[test]
fn runtime_release_owned_column_transfer_abandoned_without_import_or_release_does_not_leak() {
    // When both the transfer root and the releaser are dropped without either
    // importing or releasing, the RuntimeOwnedTransferInner Drop impl must
    // clean up the retained Column (GPU memory) without panicking.
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();

    let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();

    // Drop both without import or release — should not panic or deadlock.
    drop(transfer);
    drop(releaser);
}

#[test]
fn runtime_release_owned_table_transfer_abandoned_without_import_or_release_does_not_leak() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.series("value", &[10i64, 20, 30]).unwrap();
    let (table, _names) = DataFrame::from_series([ids, vals]).unwrap().try_into_table().unwrap();

    let (transfer, releaser) = table.try_into_owned_transfer_with_release().unwrap();

    drop(transfer);
    drop(releaser);
}

#[test]
fn runtime_release_owned_column_transfer_abandoned_releaser_only_does_not_leak() {
    // The releaser is dropped first, then the transfer is imported normally.
    // This verifies the releaser drop alone does not corrupt state.
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[7i32, 8, 9]).unwrap();

    let (transfer, releaser) = column.try_into_owned_transfer_with_release().unwrap();
    drop(releaser);

    let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
}
