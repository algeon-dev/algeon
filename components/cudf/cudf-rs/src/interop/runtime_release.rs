use std::{
    fmt,
    mem::ManuallyDrop,
    sync::{Arc, Mutex, MutexGuard},
    thread,
};

use rapids_interop::ReleaseSynchronization;

use super::*;
use crate::runtime::{CudaEvent, CudaStream};

/// Consumer-supplied completion authority for owned-transfer release.
///
/// Unlike prepared-export lease completion, owned-transfer release may defer
/// the actual drop onto a background thread according to the transfer root's
/// [`ReleaseSynchronization`]. This authority therefore accepts `cudf-rs`
/// runtime handles so the releaser can capture owned CUDA wait primitives
/// before returning to the caller.
#[derive(Clone, Copy)]
pub enum TransferCompletionAuthority<'a> {
    /// Completion is guaranteed before the release call returns.
    SynchronousByReturn,
    /// Completion will be observed on the supplied CUDA stream.
    SameDeviceStream { stream: &'a CudaStream },
    /// Completion will be observed through the supplied CUDA event.
    CrossDeviceEvent { event: &'a CudaEvent },
}

impl<'a> TransferCompletionAuthority<'a> {
    /// Use a synchronous-by-return completion contract.
    pub const fn synchronous_by_return() -> Self {
        Self::SynchronousByReturn
    }

    /// Use a same-device stream completion contract.
    pub const fn same_device_stream(stream: &'a CudaStream) -> Self {
        Self::SameDeviceStream { stream }
    }

    /// Use a cross-device event completion contract.
    pub const fn cross_device_event(event: &'a CudaEvent) -> Self {
        Self::CrossDeviceEvent { event }
    }
}

impl fmt::Debug for TransferCompletionAuthority<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SynchronousByReturn => f.write_str("TransferCompletionAuthority::SynchronousByReturn"),
            Self::SameDeviceStream { stream } => f
                .debug_struct("TransferCompletionAuthority::SameDeviceStream")
                .field("device", &stream.device())
                .finish(),
            Self::CrossDeviceEvent { event } => f
                .debug_struct("TransferCompletionAuthority::CrossDeviceEvent")
                .field("device", &event.device())
                .finish(),
        }
    }
}

pub(super) struct OwnedColumnTransferState {
    // Retain the owning column so owned-transfer buffer bindings continue to
    // point at live device memory for the full transfer lifetime.
    retained_column: Column,
}

impl OwnedColumnTransferState {
    pub(super) fn new(column: Column) -> Self {
        Self { retained_column: column }
    }

    pub(super) fn into_column(self) -> Column {
        self.retained_column
    }
}

pub(super) struct OwnedTableTransferState {
    // Retain the owning table so owned-transfer buffer bindings continue to
    // point at live device memory for the full transfer lifetime.
    retained_table: Table,
}

impl OwnedTableTransferState {
    pub(super) fn new(table: Table) -> Self {
        Self { retained_table: table }
    }

    pub(super) fn into_table(self) -> Table {
        self.retained_table
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RuntimeOwnedTransferStatus {
    Pending,
    Releasing,
    Imported,
    Released,
}

impl RuntimeOwnedTransferStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Releasing => "releasing",
            Self::Imported => "imported",
            Self::Released => "released",
        }
    }
}

struct RuntimeOwnedTransferInner<T> {
    status: RuntimeOwnedTransferStatus,
    owner: Option<ManuallyDrop<T>>,
}

impl<T> Drop for RuntimeOwnedTransferInner<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            // The retained owner was never imported or released. Drop it now
            // to avoid leaking the underlying GPU resources (Column/Table).
            drop(ManuallyDrop::into_inner(owner));
        }
    }
}

struct RuntimeOwnedTransferState<T> {
    inner: Mutex<RuntimeOwnedTransferInner<T>>,
    release_device: i32,
    release_synchronization: ReleaseSynchronization,
}

enum RuntimeOwnedReleaseCompletion {
    SameDeviceStream { stream: CudaStream },
    CrossDeviceEvent { event: CudaEvent },
}

impl RuntimeOwnedReleaseCompletion {
    fn try_capture(
        completion: TransferCompletionAuthority<'_>,
        release_device: i32,
        operation: &'static str,
    ) -> std::result::Result<Self, InteropError> {
        match completion {
            TransferCompletionAuthority::SynchronousByReturn => Err(InteropError::new(
                InteropErrorKind::ProfileUnsupported,
                "runtime-backed owned-transfer release requires a consumer signal completion authority",
            )
            .with_context("operation", operation)
            .with_context("completion", "synchronous_by_return")
            .with_context("release_device", release_device.to_string())),
            TransferCompletionAuthority::SameDeviceStream { stream } => {
                let stream_device = stream.device().ordinal_raw();
                if stream_device != release_device {
                    return Err(InteropError::new(
                        InteropErrorKind::ReadinessMismatch,
                        "runtime-backed owned-transfer release stream must live on the release CUDA device",
                    )
                    .with_context("operation", operation)
                    .with_context("completion", "same_device_stream")
                    .with_context("left_device", release_device.to_string())
                    .with_context("right_device", stream_device.to_string()));
                }

                Ok(Self::SameDeviceStream { stream: stream.clone() })
            },
            TransferCompletionAuthority::CrossDeviceEvent { event } => {
                Ok(Self::CrossDeviceEvent { event: event.clone() })
            },
        }
    }

    fn wait_for_release(&self, release_device: i32, operation: &'static str) -> std::result::Result<(), InteropError> {
        match self {
            Self::SameDeviceStream { stream } => runtime_lease::synchronize_runtime_release_completion(
                TransferCompletionAuthority::same_device_stream(stream),
                release_device,
                operation,
            ),
            Self::CrossDeviceEvent { event } => runtime_lease::synchronize_runtime_release_completion(
                TransferCompletionAuthority::cross_device_event(event),
                release_device,
                operation,
            ),
        }
    }

    fn synchronize_fallback(&self, operation: &'static str) -> crate::Result<()> {
        match self {
            Self::SameDeviceStream { stream } => runtime_lease::synchronize_completion_fallback(
                TransferCompletionAuthority::same_device_stream(stream),
                operation,
            ),
            Self::CrossDeviceEvent { event } => runtime_lease::synchronize_completion_fallback(
                TransferCompletionAuthority::cross_device_event(event),
                operation,
            ),
        }
    }
}

impl<T> RuntimeOwnedTransferState<T> {
    fn new(owner: T, release_device: i32, release_synchronization: ReleaseSynchronization) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(RuntimeOwnedTransferInner {
                status: RuntimeOwnedTransferStatus::Pending,
                owner: Some(ManuallyDrop::new(owner)),
            }),
            release_device,
            release_synchronization,
        })
    }

    fn lock_inner(&self) -> MutexGuard<'_, RuntimeOwnedTransferInner<T>> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn restore_pending_owner(&self, owner: ManuallyDrop<T>) {
        let mut inner = self.lock_inner();
        inner.owner = Some(owner);
        inner.status = RuntimeOwnedTransferStatus::Pending;
    }

    fn take_for_import(
        &self,
        operation: &'static str,
        transfer_kind: &'static str,
    ) -> std::result::Result<T, InteropError> {
        let mut inner = self.lock_inner();
        match inner.status {
            RuntimeOwnedTransferStatus::Pending => {
                inner.status = RuntimeOwnedTransferStatus::Imported;
                let owner = inner.owner.take().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::InvalidView,
                        format!("runtime-backed {transfer_kind} lost its retained owner before import"),
                    )
                    .with_context("operation", operation)
                    .with_context("state", RuntimeOwnedTransferStatus::Pending.as_str())
                    .with_context("release_device", self.release_device.to_string())
                })?;
                Ok(ManuallyDrop::into_inner(owner))
            },
            status => Err(InteropError::new(
                InteropErrorKind::ProfileUnsupported,
                format!("runtime-backed {transfer_kind} no longer has importable native ownership"),
            )
            .with_context("operation", operation)
            .with_context("state", status.as_str())
            .with_context("release_device", self.release_device.to_string())),
        }
    }

    fn wait_for_release_contract(
        &self,
        completion: &RuntimeOwnedReleaseCompletion,
        operation: &'static str,
    ) -> std::result::Result<(), InteropError> {
        match completion.wait_for_release(self.release_device, operation) {
            Ok(()) => Ok(()),
            Err(wait_error) => match completion.synchronize_fallback(operation) {
                Ok(()) => Ok(()),
                Err(fallback_error) => Err(wait_error.with_context("fallback_error", fallback_error.to_string())),
            },
        }
    }

    fn finish_release_job(
        self: Arc<Self>,
        mut owner: ManuallyDrop<T>,
        completion: RuntimeOwnedReleaseCompletion,
        operation: &'static str,
    ) -> std::result::Result<(), (ManuallyDrop<T>, InteropError)> {
        if let Err(error) = self.wait_for_release_contract(&completion, operation) {
            return Err((owner, error));
        }

        // Switch to the release device before dropping GPU-owning types.
        // In multi-GPU scenarios the spawned release thread may not be on the
        // correct device, which would cause CUDA resource destruction on the
        // wrong device context.
        let mut previous_device: i32 = -1;
        let get_ok = unsafe { cudf_sys::cudf_cuda_get_device(&mut previous_device) } == cudf_sys::CUDF_SUCCESS;
        if get_ok && previous_device != self.release_device {
            let _ = unsafe { cudf_sys::cudf_cuda_set_device(self.release_device) };
        }

        // Drop the retained owner inside catch_unwind so that a panicking
        // Drop impl on Column/Table does not prevent the state machine from
        // reaching Released.
        let drop_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            ManuallyDrop::drop(&mut owner);
        }));

        // Restore previous device (best-effort).
        if get_ok && previous_device != self.release_device {
            unsafe { cudf_sys::cudf_cuda_set_device(previous_device) };
        }

        let mut inner = self.lock_inner();
        inner.status = RuntimeOwnedTransferStatus::Released;

        // If T::drop panicked, propagate after marking the terminal state so
        // later observers always see Released.
        if let Err(payload) = drop_result {
            std::panic::resume_unwind(payload);
        }

        Ok(())
    }

    fn release(
        state: &Arc<Self>,
        completion: TransferCompletionAuthority<'_>,
        operation: &'static str,
        transfer_kind: &'static str,
    ) -> std::result::Result<(), InteropError>
    where
        T: Send + 'static,
    {
        let owner = {
            let mut inner = state.lock_inner();
            match inner.status {
                RuntimeOwnedTransferStatus::Pending => {
                    inner.status = RuntimeOwnedTransferStatus::Releasing;
                    inner.owner.take().ok_or_else(|| {
                        InteropError::new(
                            InteropErrorKind::InvalidView,
                            format!("runtime-backed {transfer_kind} lost its retained owner before release"),
                        )
                        .with_context("operation", operation)
                        .with_context("state", RuntimeOwnedTransferStatus::Pending.as_str())
                        .with_context("release_device", state.release_device.to_string())
                    })?
                },
                status => {
                    return Err(InteropError::new(
                        InteropErrorKind::ProfileUnsupported,
                        format!("runtime-backed {transfer_kind} release is unavailable after ownership state changed"),
                    )
                    .with_context("operation", operation)
                    .with_context("state", status.as_str())
                    .with_context("release_device", state.release_device.to_string()));
                },
            }
        };

        let completion = match RuntimeOwnedReleaseCompletion::try_capture(completion, state.release_device, operation) {
            Ok(completion) => completion,
            Err(error) => {
                let mut inner = state.lock_inner();
                inner.owner = Some(owner);
                inner.status = RuntimeOwnedTransferStatus::Pending;
                return Err(error);
            },
        };

        match state.release_synchronization {
            ReleaseSynchronization::Asynchronous => {
                let release_state = Arc::clone(state);
                thread::spawn(move || {
                    let job_state = Arc::clone(&release_state);
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        job_state.finish_release_job(owner, completion, operation)
                    })) {
                        Ok(Ok(())) => {},
                        Ok(Err((owner, error))) => {
                            release_state.restore_pending_owner(owner);
                            eprintln!("[cudf::interop] async release failed: {error}");
                        },
                        Err(payload) => {
                            let msg = payload
                                .downcast_ref::<&str>()
                                .copied()
                                .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
                                .unwrap_or("unknown panic");
                            eprintln!("[cudf::interop] async release thread panicked: {msg}");
                        },
                    }
                });
                Ok(())
            },
            ReleaseSynchronization::MaySynchronizeInternally => {
                match Arc::clone(state).finish_release_job(owner, completion, operation) {
                    Ok(()) => Ok(()),
                    Err((owner, error)) => {
                        state.restore_pending_owner(owner);
                        Err(error)
                    },
                }
            },
            ReleaseSynchronization::RequiresCallerSynchronization => {
                let mut inner = state.lock_inner();
                inner.owner = Some(owner);
                inner.status = RuntimeOwnedTransferStatus::Pending;
                Err(InteropError::new(
                    InteropErrorKind::ProfileUnsupported,
                    format!("runtime-backed {transfer_kind} release does not support caller-managed synchronization"),
                )
                .with_context("operation", operation)
                .with_context("completion", "release_waits_on_consumer_signal")
                .with_context("synchronization", "requires_caller_synchronization")
                .with_context("release_device", state.release_device.to_string()))
            },
        }
    }
}

/// CUDA-backed releaser for a runtime-backed owned column transfer.
///
/// `release(...)` follows the matching transfer root's declared
/// [`ReleaseSynchronization`]. Async transfers return once release has been
/// accepted and scheduled, while `MaySynchronizeInternally` transfers may
/// block the caller thread while carrying out the release contract.
pub struct OwnedColumnTransferReleaser {
    state: Arc<RuntimeOwnedTransferState<Column>>,
}

impl OwnedColumnTransferReleaser {
    /// Accept release for this transfer and schedule any required
    /// consumer-signal wait according to the matching transfer root's
    /// declared release synchronization.
    ///
    /// This fails if ownership has already been claimed by native import,
    /// already accepted by a prior release call, or if the supplied
    /// completion authority does not match the release contract.
    pub fn release(self, completion: TransferCompletionAuthority<'_>) -> std::result::Result<(), InteropError> {
        RuntimeOwnedTransferState::release(
            &self.state,
            completion,
            "interop::OwnedColumnTransferReleaser::release",
            "owned column transfer",
        )
    }
}

/// CUDA-backed releaser for a runtime-backed owned table transfer.
///
/// `release(...)` follows the matching transfer root's declared
/// [`ReleaseSynchronization`]. Async transfers return once release has been
/// accepted and scheduled, while `MaySynchronizeInternally` transfers may
/// block the caller thread while carrying out the release contract.
pub struct OwnedTableTransferReleaser {
    state: Arc<RuntimeOwnedTransferState<Table>>,
}

impl OwnedTableTransferReleaser {
    /// Accept release for this transfer and schedule any required
    /// consumer-signal wait according to the matching transfer root's
    /// declared release synchronization.
    ///
    /// This fails if ownership has already been claimed by native import,
    /// already accepted by a prior release call, or if the supplied
    /// completion authority does not match the release contract.
    pub fn release(self, completion: TransferCompletionAuthority<'_>) -> std::result::Result<(), InteropError> {
        RuntimeOwnedTransferState::release(
            &self.state,
            completion,
            "interop::OwnedTableTransferReleaser::release",
            "owned table transfer",
        )
    }
}

/// One partition inside a partitioned runtime-backed owned-transfer releaser.
///
/// This preserves the original top-level column indices for the partition and
/// exposes the same one-shot runtime-backed release behavior as the underlying
/// single-table releaser.
pub struct PartitionedOwnedTableTransferReleaserPartition {
    top_level_column_indices: Vec<usize>,
    releaser: OwnedTableTransferReleaser,
}

impl PartitionedOwnedTableTransferReleaserPartition {
    /// Create one partition releaser wrapper from partition metadata and the
    /// underlying single-table runtime-backed releaser.
    pub fn new(top_level_column_indices: Vec<usize>, releaser: OwnedTableTransferReleaser) -> Self {
        Self { top_level_column_indices, releaser }
    }

    /// Return the original top-level schema indices covered by this partition.
    pub fn top_level_column_indices(&self) -> &[usize] {
        &self.top_level_column_indices
    }

    /// Accept release for this partition and carry out any required
    /// consumer-signal wait according to the matching transfer root's
    /// declared release synchronization.
    ///
    /// This is mutually exclusive with importing the matching partition's
    /// `OwnedTableTransfer` back into a native `cudf-rs` `Table`.
    pub fn release(self, completion: TransferCompletionAuthority<'_>) -> std::result::Result<(), InteropError> {
        self.releaser.release(completion)
    }

    /// Consume this wrapper and return the underlying single-table releaser.
    pub fn into_releaser(self) -> OwnedTableTransferReleaser {
        self.releaser
    }
}

/// CUDA-backed releaser set for a partitioned runtime-owned table transfer.
///
/// Each entry preserves one retained-provenance partition, one release
/// contract, and one exclusive ownership state machine.
pub struct PartitionedOwnedTableTransferReleaser {
    partitions: Vec<PartitionedOwnedTableTransferReleaserPartition>,
}

impl PartitionedOwnedTableTransferReleaser {
    /// Create a partitioned releaser wrapper from already-built partition
    /// releasers.
    pub fn new(partitions: Vec<PartitionedOwnedTableTransferReleaserPartition>) -> Self {
        Self { partitions }
    }

    /// Borrow the per-partition releasers in schema order.
    pub fn partitions(&self) -> &[PartitionedOwnedTableTransferReleaserPartition] {
        &self.partitions
    }

    /// Consume this wrapper and return the per-partition releasers.
    pub fn into_partitions(self) -> Vec<PartitionedOwnedTableTransferReleaserPartition> {
        self.partitions
    }
}

pub(super) struct RuntimeOwnedColumnTransferOwner {
    state: Arc<RuntimeOwnedTransferState<Column>>,
}

impl RuntimeOwnedColumnTransferOwner {
    pub(super) fn into_column(self, operation: &'static str) -> std::result::Result<Column, InteropError> {
        self.state.take_for_import(operation, "owned column transfer")
    }
}

pub(super) struct RuntimeOwnedTableTransferOwner {
    state: Arc<RuntimeOwnedTransferState<Table>>,
}

impl RuntimeOwnedTableTransferOwner {
    pub(super) fn into_table(self, operation: &'static str) -> std::result::Result<Table, InteropError> {
        self.state.take_for_import(operation, "owned table transfer")
    }
}

pub(super) fn runtime_owned_column_owner_and_releaser(
    column: Column,
    release_device: i32,
    release_synchronization: ReleaseSynchronization,
) -> (RuntimeOwnedColumnTransferOwner, OwnedColumnTransferReleaser) {
    let state = RuntimeOwnedTransferState::new(column, release_device, release_synchronization);
    (RuntimeOwnedColumnTransferOwner { state: Arc::clone(&state) }, OwnedColumnTransferReleaser { state })
}

pub(super) fn runtime_owned_table_owner_and_releaser(
    table: Table,
    release_device: i32,
    release_synchronization: ReleaseSynchronization,
) -> (RuntimeOwnedTableTransferOwner, OwnedTableTransferReleaser) {
    let state = RuntimeOwnedTransferState::new(table, release_device, release_synchronization);
    (RuntimeOwnedTableTransferOwner { state: Arc::clone(&state) }, OwnedTableTransferReleaser { state })
}

pub(super) fn to_static_binding(
    operation: &'static str,
    binding: BufferBinding<'_>,
) -> Result<BufferBinding<'static>, InteropError> {
    // SAFETY: This lifetime erasure is only used while constructing owned
    // transfer roots. The raw pointer still refers to producer-owned device
    // memory, and the corresponding transfer stores that owner in
    // `OwnedColumnTransferState` or `OwnedTableTransferState`. Those retained
    // states are boxed into the shared `OwnedTransferOwner`, so the source
    // `Column`/`Table` outlives every `BufferBinding<'static>` created here.
    Ok(BufferBinding::new(
        binding.id(),
        BoundBytes::from_raw_parts(binding.bytes().base_ptr() as *mut std::ffi::c_void, binding.bytes().byte_len())
            .map_err(|error| error.with_context("operation", operation))?,
        *binding.placement(),
    ))
}

pub(super) fn to_static_bindings(
    operation: &'static str,
    bindings: Vec<BufferBinding<'_>>,
) -> Result<Vec<BufferBinding<'static>>, InteropError> {
    bindings.into_iter().map(|binding| to_static_binding(operation, binding)).collect()
}

pub(super) fn to_static_readiness(
    operation: &'static str,
    readiness: ProducerReadiness<'_>,
) -> Result<ProducerReadiness<'static>, InteropError> {
    match readiness {
        ProducerReadiness::HostVisible => Ok(ProducerReadiness::HostVisible),
        ProducerReadiness::DeviceStreamOrdered { stream } => Ok(ProducerReadiness::DeviceStreamOrdered {
            stream: CudaStreamHandle::from_raw_on_device(stream.as_raw(), stream.device()).map_err(|error| {
                error.with_context("operation", operation).with_context("readiness", "device_stream_ordered")
            })?,
        }),
        ProducerReadiness::DeviceEventOrdered { event } => Ok(ProducerReadiness::DeviceEventOrdered {
            event: CudaEventHandle::from_raw_on_device(event.as_raw(), event.device()).map_err(|error| {
                error.with_context("operation", operation).with_context("readiness", "device_event_ordered")
            })?,
        }),
    }
}

pub(super) fn into_owned_column_for_transfer(series: Series, operation: &'static str) -> Result<Column, InteropError> {
    match Arc::try_unwrap(series.into_inner()) {
        Ok(column) => Ok(column),
        Err(shared) => shared
            .copy_into_domain(Arc::clone(shared.execution_domain()), operation)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error)),
    }
}

pub(super) fn into_owned_table_for_transfer(
    frame: DataFrame,
    operation: &'static str,
) -> Result<(Table, Vec<String>), InteropError> {
    frame.try_into_table().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))
}
