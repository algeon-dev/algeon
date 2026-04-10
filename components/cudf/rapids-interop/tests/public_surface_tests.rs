use rapids_interop::{
    ActiveColumnReadLease, ActiveLeaseActivation, ActiveLeaseBackend, ActivePartitionedTableReadLease,
    ActiveTableReadLease, BorrowedColumnView, BorrowedTableView, BoundBytes, BufferBinding, BufferId, ColumnId,
    ColumnLayout, CompletionAuthority, CompletionBinding, CompletionWitness, CompletionWitnessBackend,
    CompletionWitnessKind, CudaEventHandle, CudaStreamHandle, DataType, DeviceMemoryResourceHandle, ExecutionBinding,
    ExportDiagnostics, ExportProfile, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation,
    LeaseBackend, LeaseCompletionResult, LeaseExportContext, NullMask, NullMaskKind, OwnedColumnTransfer,
    OwnedTableTransfer, PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferPartition,
    PartitionedPreparedTableExport, PartitionedPreparedTablePartition, PhysicalValidity, Placement,
    PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority, ReadStartBinding,
    ReleaseContract, ReleaseSynchronization, Result, SchemaNullability, SemanticColumn, SemanticTable, TableId,
    TransferCompletionRule, TryBorrowColumnView, TryBorrowTableView, TryIntoOwnedDeviceColumnTransfer,
    TryIntoOwnedDevicePartitionedTableTransfer, TryIntoOwnedDeviceTableTransfer, TryPrepareDeviceColumnExport,
    TryPrepareDevicePartitionedTableExport, TryPrepareDeviceTableExport, TryPrepareHostColumnExport,
    TryPrepareHostTableExport, TypeId,
};

#[test]
fn crate_root_reexports_public_interop_surface() {
    fn assert_type<T>() {}
    fn assert_impl_try_into_owned_column<T: TryIntoOwnedDeviceColumnTransfer>() {}
    fn assert_impl_try_into_owned_table<T: TryIntoOwnedDeviceTableTransfer>() {}
    fn assert_impl_try_into_owned_partitioned_table<T: TryIntoOwnedDevicePartitionedTableTransfer>() {}

    assert_type::<Result<()>>();
    assert_type::<TypeId>();
    assert_type::<DataType>();
    assert_type::<InteropErrorKind>();
    assert_type::<InteropError>();
    assert_type::<CudaStreamHandle<'static>>();
    assert_type::<CudaEventHandle<'static>>();
    assert_type::<DeviceMemoryResourceHandle<'static>>();
    assert_type::<ExecutionBinding<'static>>();
    assert_type::<BufferId>();
    assert_type::<ColumnId>();
    assert_type::<TableId>();
    assert_type::<PhysicalValidity>();
    assert_type::<SchemaNullability>();
    assert_type::<ColumnLayout>();
    assert_type::<SemanticColumn>();
    assert_type::<SemanticTable>();
    assert_type::<Placement>();
    assert_type::<BoundBytes<'static>>();
    assert_type::<BufferBinding<'static>>();
    assert_type::<ProducerReadiness<'static>>();
    assert_type::<ReadStartAuthority<'static>>();
    assert_type::<CompletionAuthority<'static>>();
    assert_type::<ReadStartBinding>();
    assert_type::<CompletionBinding>();
    assert_type::<LeaseExportContext>();
    assert_type::<CompletionWitnessKind>();
    assert_type::<CompletionWitness>();
    assert_type::<IncompleteDropBehavior>();
    assert_type::<LeaseActivation>();
    assert_type::<LeaseBackend>();
    assert_type::<LeaseCompletionResult>();
    assert_type::<ExportProfile>();
    assert_type::<ExportDiagnostics>();
    assert_type::<TransferCompletionRule>();
    assert_type::<ReleaseSynchronization>();
    assert_type::<ReleaseContract>();
    assert_type::<NullMaskKind>();
    assert_type::<NullMask<'static>>();
    assert_type::<BorrowedColumnView<'static>>();
    assert_type::<BorrowedTableView<'static>>();
    assert_type::<ActiveColumnReadLease<'static, 'static, 'static>>();
    assert_type::<ActiveTableReadLease<'static, 'static, 'static>>();
    assert_type::<ActivePartitionedTableReadLease<'static, 'static, 'static>>();
    assert_type::<OwnedColumnTransfer>();
    assert_type::<OwnedTableTransfer>();
    assert_type::<PartitionedPreparedTablePartition<'static>>();
    assert_type::<PartitionedPreparedTableExport<'static>>();
    assert_type::<PartitionedOwnedTableTransferPartition>();
    assert_type::<PartitionedOwnedTableTransfer>();
    assert_type::<PreparedColumnExport<'static>>();
    assert_type::<PreparedTableExport<'static>>();
    assert_type::<&'static dyn ActiveLeaseActivation>();
    assert_type::<&'static dyn ActiveLeaseBackend>();
    assert_type::<&'static dyn CompletionWitnessBackend>();
    assert_type::<&'static dyn TryBorrowColumnView>();
    assert_type::<&'static dyn TryBorrowTableView>();
    assert_type::<&'static dyn TryPrepareDeviceColumnExport>();
    assert_type::<&'static dyn TryPrepareDeviceTableExport>();
    assert_type::<&'static dyn TryPrepareDevicePartitionedTableExport>();
    assert_type::<&'static dyn TryPrepareHostColumnExport>();
    assert_type::<&'static dyn TryPrepareHostTableExport>();
    assert_impl_try_into_owned_column::<OwnedColumnTransfer>();
    assert_impl_try_into_owned_table::<OwnedTableTransfer>();
    assert_impl_try_into_owned_partitioned_table::<PartitionedOwnedTableTransfer>();
}
