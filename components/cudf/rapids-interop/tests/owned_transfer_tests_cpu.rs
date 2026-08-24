use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use rapids_interop::{
    BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, CudaStreamHandle, DataType, ExportDiagnostics,
    ExportProfile, InteropErrorKind, OwnedColumnTransfer, OwnedTableTransfer, OwnedTableTransferParts,
    PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferPartition, PhysicalValidity, Placement,
    ProducerReadiness, ReleaseContract, ReleaseSynchronization, SchemaNullability, SemanticColumn, SemanticTable,
    TableId, TransferCompletionRule, TypeId,
};

fn fake_ptr(addr: usize) -> *const core::ffi::c_void {
    addr as *const core::ffi::c_void
}

fn device_binding(id: u32, addr: usize, byte_len: usize, device: i32) -> BufferBinding<'static> {
    BufferBinding::new(
        BufferId::new(id),
        BoundBytes::from_raw_parts(fake_ptr(addr) as *mut core::ffi::c_void, byte_len).unwrap(),
        Placement::CudaDevice { device },
    )
}

fn fixed_width_column(id: u32, values: u32, len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::Int32),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::FixedWidth { values: BufferId::new(values) },
    )
}

fn owned_transfer_readiness() -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceStreamOrdered {
        stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, 0).unwrap(),
    }
}

fn owned_transfer_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::OwnedDeviceTransfer)
}

fn release_contract() -> ReleaseContract {
    release_contract_with(
        0,
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
        ReleaseSynchronization::MaySynchronizeInternally,
    )
}

fn release_contract_with(
    device: i32,
    completion: TransferCompletionRule,
    synchronization: ReleaseSynchronization,
) -> ReleaseContract {
    ReleaseContract::new(device, completion, synchronization)
}

fn owned_fixed_width_table_transfer_on_device(
    table_id: u32,
    column_id: u32,
    buffer_id: u32,
    addr: usize,
    device: i32,
    num_rows: usize,
) -> OwnedTableTransfer {
    owned_fixed_width_table_transfer_with_release_contract(
        table_id,
        column_id,
        buffer_id,
        addr,
        device,
        num_rows,
        release_contract_with(
            device,
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::MaySynchronizeInternally,
        ),
    )
}

fn owned_fixed_width_table_transfer_with_release_contract(
    table_id: u32,
    column_id: u32,
    buffer_id: u32,
    addr: usize,
    device: i32,
    num_rows: usize,
    release_contract: ReleaseContract,
) -> OwnedTableTransfer {
    OwnedTableTransfer::try_new(
        SemanticTable::new(TableId::new(table_id), num_rows, vec![ColumnId::new(column_id)], None),
        vec![fixed_width_column(column_id, buffer_id, num_rows, 0)],
        vec![device_binding(buffer_id, addr, num_rows * std::mem::size_of::<i32>(), device)],
        ProducerReadiness::DeviceStreamOrdered {
            stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, device).unwrap(),
        },
        owned_transfer_diagnostics(),
        release_contract,
        DropFlagOwner { dropped: Arc::new(AtomicBool::new(false)) },
    )
    .unwrap()
}

struct DropFlagOwner {
    dropped: Arc<AtomicBool>,
}

impl Drop for DropFlagOwner {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn owned_column_transfer_reports_owned_device_profile_and_release_contract() {
    let dropped = Arc::new(AtomicBool::new(false));
    let transfer = OwnedColumnTransfer::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        owned_transfer_readiness(),
        owned_transfer_diagnostics(),
        release_contract(),
        DropFlagOwner { dropped: Arc::clone(&dropped) },
    )
    .unwrap();

    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer);
    assert_eq!(transfer.release_contract(), &release_contract());
    assert_eq!(transfer.readiness(), &owned_transfer_readiness());

    drop(transfer);
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn owned_column_transfer_can_recover_typed_owner() {
    let transfer = OwnedColumnTransfer::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        owned_transfer_readiness(),
        owned_transfer_diagnostics(),
        release_contract(),
        String::from("column-owner"),
    )
    .unwrap();

    let owner = transfer.try_into_owner::<String>().unwrap();

    assert_eq!(owner, "column-owner");
}

#[test]
fn owned_column_transfer_owner_type_mismatch_returns_original_transfer() {
    let dropped = Arc::new(AtomicBool::new(false));
    let transfer = OwnedColumnTransfer::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        owned_transfer_readiness(),
        owned_transfer_diagnostics(),
        release_contract(),
        DropFlagOwner { dropped: Arc::clone(&dropped) },
    )
    .unwrap();

    let transfer = transfer.try_into_owner::<String>().unwrap_err().into_transfer();

    assert_eq!(transfer.release_contract(), &release_contract());
    assert!(!dropped.load(Ordering::SeqCst));

    drop(transfer);
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn owned_column_transfer_rejects_non_transfer_profile() {
    let err = OwnedColumnTransfer::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        owned_transfer_readiness(),
        ExportDiagnostics::new(ExportProfile::PreparedDeviceExport),
        release_contract(),
        DropFlagOwner { dropped: Arc::new(AtomicBool::new(false)) },
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn owned_column_transfer_rejects_host_visible_readiness() {
    let err = OwnedColumnTransfer::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        ProducerReadiness::HostVisible,
        owned_transfer_diagnostics(),
        release_contract(),
        DropFlagOwner { dropped: Arc::new(AtomicBool::new(false)) },
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn owned_column_transfer_accepts_all_supported_release_contract_combinations() {
    let cases = [
        (
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::MaySynchronizeInternally,
        ),
        (
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::RequiresCallerSynchronization,
        ),
        (TransferCompletionRule::ReleaseWaitsOnConsumerSignal, ReleaseSynchronization::Asynchronous),
        (TransferCompletionRule::ReleaseWaitsOnConsumerSignal, ReleaseSynchronization::MaySynchronizeInternally),
    ];

    for (index, (completion, synchronization)) in cases.into_iter().enumerate() {
        let release_contract = release_contract_with(0, completion, synchronization);
        let transfer = OwnedColumnTransfer::try_new(
            fixed_width_column(1 + index as u32, 11 + index as u32, 3, 0),
            vec![device_binding(11 + index as u32, 0x1000 + (index * 0x1000), 12, 0)],
            owned_transfer_readiness(),
            owned_transfer_diagnostics(),
            release_contract,
            format!("owner-{index}"),
        )
        .unwrap();

        assert_eq!(transfer.release_contract(), &release_contract);
    }
}

#[test]
fn owned_column_transfer_rejects_contradictory_release_contract_combinations() {
    let cases = [
        (TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes, ReleaseSynchronization::Asynchronous),
        (TransferCompletionRule::ReleaseWaitsOnConsumerSignal, ReleaseSynchronization::RequiresCallerSynchronization),
    ];

    for (completion, synchronization) in cases {
        let err = OwnedColumnTransfer::try_new(
            fixed_width_column(1, 11, 3, 0),
            vec![device_binding(11, 0x1000, 12, 0)],
            owned_transfer_readiness(),
            owned_transfer_diagnostics(),
            release_contract_with(0, completion, synchronization),
            String::from("owner"),
        )
        .unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    }
}

#[test]
fn owned_column_transfer_rejects_negative_release_device() {
    let err = OwnedColumnTransfer::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        owned_transfer_readiness(),
        owned_transfer_diagnostics(),
        release_contract_with(
            -1,
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::MaySynchronizeInternally,
        ),
        String::from("owner"),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
}

#[test]
fn owned_table_transfer_reports_owned_device_profile_and_release_contract() {
    let dropped = Arc::new(AtomicBool::new(false));
    let columns = vec![fixed_width_column(1, 11, 3, 0), fixed_width_column(2, 12, 3, 0)];
    let table = SemanticTable::new(TableId::new(7), 3, vec![ColumnId::new(1), ColumnId::new(2)], None);
    let transfer = OwnedTableTransfer::try_new(
        table,
        columns,
        vec![device_binding(11, 0x1000, 12, 0), device_binding(12, 0x2000, 24, 0)],
        owned_transfer_readiness(),
        owned_transfer_diagnostics(),
        release_contract(),
        DropFlagOwner { dropped: Arc::clone(&dropped) },
    )
    .unwrap();

    assert_eq!(transfer.diagnostics().profile(), ExportProfile::OwnedDeviceTransfer);
    assert_eq!(transfer.release_contract(), &release_contract());
    assert_eq!(transfer.table().num_rows(), 3);

    drop(transfer);
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn owned_table_transfer_can_recover_typed_owner() {
    let columns = vec![fixed_width_column(1, 11, 3, 0)];
    let table = SemanticTable::new(TableId::new(7), 3, vec![ColumnId::new(1)], None);
    let transfer = OwnedTableTransfer::try_new(
        table,
        columns,
        vec![device_binding(11, 0x1000, 12, 0)],
        owned_transfer_readiness(),
        owned_transfer_diagnostics(),
        release_contract(),
        String::from("table-owner"),
    )
    .unwrap();

    let owner = transfer.try_into_owner::<String>().unwrap();

    assert_eq!(owner, "table-owner");
}

#[test]
fn owned_table_transfer_accepts_named_parts_constructor() {
    let columns = vec![fixed_width_column(1, 11, 3, 0)];
    let table = SemanticTable::new(TableId::new(7), 3, vec![ColumnId::new(1)], None);
    let release_contract = release_contract();
    let transfer = OwnedTableTransfer::try_new_with_semantic_columns(OwnedTableTransferParts {
        table,
        columns: columns.clone(),
        semantic_columns: columns,
        bindings: vec![device_binding(11, 0x1000, 12, 0)],
        readiness: owned_transfer_readiness(),
        diagnostics: owned_transfer_diagnostics(),
        release_contract,
        owner: String::from("table-owner"),
    })
    .unwrap();

    assert_eq!(transfer.release_contract(), &release_contract);
    assert_eq!(transfer.try_into_owner::<String>().unwrap(), "table-owner");
}

#[test]
fn owned_table_transfer_accepts_all_supported_release_contract_combinations() {
    let cases = [
        (
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::MaySynchronizeInternally,
        ),
        (
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::RequiresCallerSynchronization,
        ),
        (TransferCompletionRule::ReleaseWaitsOnConsumerSignal, ReleaseSynchronization::Asynchronous),
        (TransferCompletionRule::ReleaseWaitsOnConsumerSignal, ReleaseSynchronization::MaySynchronizeInternally),
    ];

    for (index, (completion, synchronization)) in cases.into_iter().enumerate() {
        let release_contract = release_contract_with(0, completion, synchronization);
        let transfer = OwnedTableTransfer::try_new(
            SemanticTable::new(TableId::new(7 + index as u32), 3, vec![ColumnId::new(1)], None),
            vec![fixed_width_column(1, 11 + index as u32, 3, 0)],
            vec![device_binding(11 + index as u32, 0x1000 + (index * 0x1000), 12, 0)],
            owned_transfer_readiness(),
            owned_transfer_diagnostics(),
            release_contract,
            format!("owner-{index}"),
        )
        .unwrap();

        assert_eq!(transfer.release_contract(), &release_contract);
    }
}

#[test]
fn owned_table_transfer_rejects_contradictory_release_contract_combinations() {
    let cases = [
        (TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes, ReleaseSynchronization::Asynchronous),
        (TransferCompletionRule::ReleaseWaitsOnConsumerSignal, ReleaseSynchronization::RequiresCallerSynchronization),
    ];

    for (completion, synchronization) in cases {
        let err = OwnedTableTransfer::try_new(
            SemanticTable::new(TableId::new(7), 3, vec![ColumnId::new(1)], None),
            vec![fixed_width_column(1, 11, 3, 0)],
            vec![device_binding(11, 0x1000, 12, 0)],
            owned_transfer_readiness(),
            owned_transfer_diagnostics(),
            release_contract_with(0, completion, synchronization),
            String::from("owner"),
        )
        .unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    }
}

#[test]
fn owned_table_transfer_rejects_negative_release_device() {
    let err = OwnedTableTransfer::try_new(
        SemanticTable::new(TableId::new(7), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 11, 3, 0)],
        vec![device_binding(11, 0x1000, 12, 0)],
        owned_transfer_readiness(),
        owned_transfer_diagnostics(),
        release_contract_with(
            -1,
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::MaySynchronizeInternally,
        ),
        String::from("owner"),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
}

#[test]
fn partitioned_owned_table_transfer_accepts_two_partitions_on_different_devices() {
    let left = owned_fixed_width_table_transfer_on_device(10, 1, 11, 0x1000, 0, 3);
    let right = owned_fixed_width_table_transfer_on_device(11, 2, 12, 0x2000, 1, 3);

    let transfer = PartitionedOwnedTableTransfer::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedOwnedTableTransferPartition::new(vec![0], left),
            PartitionedOwnedTableTransferPartition::new(vec![1], right),
        ],
    )
    .unwrap();

    assert_eq!(transfer.num_rows(), 3);
    assert_eq!(transfer.schema_names().unwrap(), &["left".to_string(), "right".to_string()]);
    assert_eq!(transfer.partitions().len(), 2);
    assert_eq!(transfer.partitions()[0].top_level_column_indices(), &[0]);
    assert_eq!(transfer.partitions()[1].top_level_column_indices(), &[1]);
    assert_eq!(
        transfer.partitions()[0].transfer().release_contract().completion(),
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes
    );
    assert_eq!(transfer.partitions()[1].transfer().release_contract().release_device(), 1);
}

#[test]
fn partitioned_owned_table_transfer_preserves_per_partition_release_contracts() {
    let left = owned_fixed_width_table_transfer_with_release_contract(
        10,
        1,
        11,
        0x1000,
        0,
        3,
        release_contract_with(
            0,
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::RequiresCallerSynchronization,
        ),
    );
    let right = owned_fixed_width_table_transfer_with_release_contract(
        11,
        2,
        12,
        0x2000,
        1,
        3,
        release_contract_with(
            1,
            TransferCompletionRule::ReleaseWaitsOnConsumerSignal,
            ReleaseSynchronization::Asynchronous,
        ),
    );

    let transfer = PartitionedOwnedTableTransfer::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedOwnedTableTransferPartition::new(vec![0], left),
            PartitionedOwnedTableTransferPartition::new(vec![1], right),
        ],
    )
    .unwrap();

    assert_eq!(transfer.partitions()[0].transfer().release_contract().release_device(), 0);
    assert_eq!(transfer.partitions()[1].transfer().release_contract().release_device(), 1);
    assert_eq!(
        transfer.partitions()[0].transfer().release_contract().completion(),
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes
    );
    assert_eq!(
        transfer.partitions()[1].transfer().release_contract().completion(),
        TransferCompletionRule::ReleaseWaitsOnConsumerSignal
    );
    assert_eq!(
        transfer.partitions()[0].transfer().release_contract().synchronization(),
        ReleaseSynchronization::RequiresCallerSynchronization
    );
    assert_eq!(
        transfer.partitions()[1].transfer().release_contract().synchronization(),
        ReleaseSynchronization::Asynchronous
    );
}
