use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use rapids_interop::{
    ActiveLeaseBackend, BorrowedColumnView, BorrowedTableView, BoundBytes, BufferBinding, BufferId, ColumnId,
    ColumnLayout, CompletionAuthority, CompletionBinding, CudaEventHandle, CudaStreamHandle, DataType,
    DeviceMemoryResourceHandle, ExecutionBinding, ExportDiagnostics, ExportProfile, IncompleteDropBehavior,
    LeaseBackend, LeaseCompletionResult, LeaseExportContext, OwnedColumnTransfer, OwnedTableTransfer, PhysicalValidity,
    Placement, PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority, ReadStartBinding,
    ReleaseContract, ReleaseSynchronization, SchemaNullability, SemanticColumn, SemanticTable, TableId,
    TransferCompletionRule, TypeId,
};
use static_assertions::{assert_impl_all, assert_not_impl_any};

assert_impl_all!(CudaStreamHandle<'static>: Send, Sync);
assert_impl_all!(CudaEventHandle<'static>: Send, Sync);
assert_impl_all!(DeviceMemoryResourceHandle<'static>: Send, Sync);
assert_impl_all!(BoundBytes<'static>: Send, Sync);
assert_impl_all!(BufferBinding<'static>: Send, Sync);
assert_impl_all!(ExecutionBinding<'static>: Send, Sync);
assert_impl_all!(ProducerReadiness<'static>: Send, Sync);
assert_impl_all!(ReadStartAuthority<'static>: Send, Sync);
assert_impl_all!(CompletionAuthority<'static>: Send, Sync);
assert_impl_all!(OwnedColumnTransfer: Send, Sync);
assert_impl_all!(OwnedTableTransfer: Send, Sync);
assert_impl_all!(PreparedColumnExport<'static>: Send, Sync);
assert_impl_all!(PreparedTableExport<'static>: Send, Sync);

assert_not_impl_any!(BorrowedColumnView<'static>: Send, Sync);
assert_not_impl_any!(BorrowedTableView<'static>: Send, Sync);

/// Sentinel address for contract tests — never dereferenced, only used to
/// satisfy non-null validation in `BoundBytes` / `CudaStreamHandle` constructors.
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

fn prepared_readiness() -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceStreamOrdered {
        stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, 0).unwrap(),
    }
}

fn prepared_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedDeviceExport)
}

fn owned_transfer_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::OwnedDeviceTransfer)
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

fn release_contract(device: i32) -> ReleaseContract {
    ReleaseContract::new(
        device,
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
        ReleaseSynchronization::MaySynchronizeInternally,
    )
}

struct DropCounterOwner {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropCounterOwner {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct DropCountingBackend {
    drop_calls: Arc<AtomicUsize>,
    complete_calls: Arc<AtomicUsize>,
}

impl ActiveLeaseBackend for DropCountingBackend {
    fn activate(
        &self,
        _export: &LeaseExportContext,
        _read_start: ReadStartBinding,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<rapids_interop::LeaseActivation> {
        Ok(rapids_interop::LeaseActivation::noop())
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        IncompleteDropBehavior::GuaranteedFallback
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseCompletionResult> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<()> {
        self.drop_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn owned_column_transfer_owner_drops_once_after_cross_thread_handoff() {
    let drops = Arc::new(AtomicUsize::new(0));
    let transfer = OwnedColumnTransfer::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        prepared_readiness(),
        owned_transfer_diagnostics(),
        release_contract(0),
        DropCounterOwner { drops: Arc::clone(&drops) },
    )
    .unwrap();

    thread::spawn(move || drop(transfer)).join().unwrap();

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_column_export_drop_on_worker_thread_runs_fallback_once_and_releases_root() {
    let drop_calls = Arc::new(AtomicUsize::new(0));
    let complete_calls = Arc::new(AtomicUsize::new(0));
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![device_binding(11, 0x1000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap()
    .with_lease_backend(LeaseBackend::new(DropCountingBackend {
        drop_calls: Arc::clone(&drop_calls),
        complete_calls: Arc::clone(&complete_calls),
    }));

    thread::scope(|scope| {
        scope
            .spawn(|| {
                let lease = export
                    .try_activate_read_lease(
                        ReadStartAuthority::SynchronousByReturn,
                        CompletionAuthority::SynchronousByReturn,
                    )
                    .unwrap();
                drop(lease);
            })
            .join()
            .unwrap();
    });

    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(complete_calls.load(Ordering::SeqCst), 0);

    let reacquired = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(reacquired.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(complete_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_table_export_can_be_completed_from_worker_thread() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(9), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 11, 3, 0)],
        vec![device_binding(11, 0x1000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    thread::scope(|scope| {
        scope
            .spawn(|| {
                let lease = export
                    .try_activate_read_lease(
                        ReadStartAuthority::SynchronousByReturn,
                        CompletionAuthority::SynchronousByReturn,
                    )
                    .unwrap();
                assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
            })
            .join()
            .unwrap();
    });
}
