use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use rapids_interop::{
    ActiveLeaseBackend, BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority,
    CompletionWitness, CudaEventHandle, CudaStreamHandle, DataType, DeviceMemoryResourceHandle, ExecutionBinding,
    LeaseBackend, PhysicalValidity, Placement, PreparedAccess, PreparedColumnExport, PreparedTableExport,
    ProducerReadiness, ReadStartAuthority, SemanticColumn, SemanticTable, TypeId,
};
use static_assertions::assert_impl_all;

assert_impl_all!(CudaStreamHandle<'static>: Send, Sync);
assert_impl_all!(CudaEventHandle<'static>: Send, Sync);
assert_impl_all!(DeviceMemoryResourceHandle<'static>: Send, Sync);
assert_impl_all!(BoundBytes<'static>: Send, Sync);
assert_impl_all!(BufferBinding<'static>: Send, Sync);
assert_impl_all!(ExecutionBinding<'static>: Send, Sync);
assert_impl_all!(ProducerReadiness<'static>: Send, Sync);
assert_impl_all!(PreparedAccess<'static>: Send, Sync);
assert_impl_all!(ReadStartAuthority<'static>: Send, Sync);
assert_impl_all!(CompletionAuthority<'static>: Send, Sync);
assert_impl_all!(PreparedColumnExport<'static>: Send, Sync);
assert_impl_all!(PreparedTableExport<'static>: Send, Sync);

/// Sentinel address for contract tests; never dereferenced, only used to
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
        stream: unsafe { CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, 0) }.unwrap(),
    }
}

fn device_access(readiness: ProducerReadiness<'static>) -> PreparedAccess<'static> {
    PreparedAccess::Device {
        readiness,
        backend: LeaseBackend::new(DropCountingBackend { drop_calls: Arc::default(), complete_calls: Arc::default() }),
    }
}

fn fixed_width_column(id: u32, values: u32, len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::Int32),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
        ColumnLayout::FixedWidth { values: BufferId::new(values) },
    )
}

#[derive(Debug)]
struct DropCountingBackend {
    drop_calls: Arc<AtomicUsize>,
    complete_calls: Arc<AtomicUsize>,
}

unsafe impl ActiveLeaseBackend for DropCountingBackend {
    fn complete(
        &self,
        _readiness: ProducerReadiness<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<Option<CompletionWitness>> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }

    fn drop_incomplete(
        &self,
        _readiness: ProducerReadiness<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<()> {
        self.drop_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn prepared_column_export_drop_on_worker_thread_runs_fallback_once_and_releases_root() {
    let drop_calls = Arc::new(AtomicUsize::new(0));
    let complete_calls = Arc::new(AtomicUsize::new(0));
    let export = unsafe {
        PreparedColumnExport::try_new(
            ColumnId::new(1),
            vec![fixed_width_column(1, 11, 3, 0)],
            vec![device_binding(11, 0x1000, 12, 0)],
            PreparedAccess::Device {
                readiness: prepared_readiness(),
                backend: LeaseBackend::new(DropCountingBackend {
                    drop_calls: Arc::clone(&drop_calls),
                    complete_calls: Arc::clone(&complete_calls),
                }),
            },
        )
    }
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
    assert!(reacquired.complete().unwrap().is_ready().unwrap());
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(complete_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_table_export_can_be_completed_from_worker_thread() {
    let export = unsafe {
        PreparedTableExport::try_new(
            SemanticTable::new(3, vec![ColumnId::new(1)], None),
            vec![fixed_width_column(1, 11, 3, 0)],
            vec![device_binding(11, 0x1000, 12, 0)],
            device_access(prepared_readiness()),
        )
    }
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
                assert!(lease.complete().unwrap().is_ready().unwrap());
            })
            .join()
            .unwrap();
    });
}
