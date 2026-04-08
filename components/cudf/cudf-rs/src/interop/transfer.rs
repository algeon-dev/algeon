pub use rapids_interop::{
    OwnedColumnTransfer, OwnedTableTransfer, PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferPartition,
    TransferCompletionRule, TryIntoOwnedDeviceColumnTransfer, TryIntoOwnedDevicePartitionedTableTransfer,
    TryIntoOwnedDeviceTableTransfer,
};

use super::*;

pub(super) struct OwnedColumnTransferState {
    // Retain the owning column so owned-transfer buffer bindings continue to
    // point at live device memory for the full transfer lifetime.
    _column: Column,
}

impl OwnedColumnTransferState {
    pub(super) fn new(column: Column) -> Self {
        Self { _column: column }
    }

    pub(super) fn into_column(self) -> Column {
        self._column
    }
}

pub(super) struct OwnedTableTransferState {
    // Retain the owning table so owned-transfer buffer bindings continue to
    // point at live device memory for the full transfer lifetime.
    _table: Table,
}

impl OwnedTableTransferState {
    pub(super) fn new(table: Table) -> Self {
        Self { _table: table }
    }

    pub(super) fn into_table(self) -> Table {
        self._table
    }
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

pub(super) fn to_static_readiness(readiness: ProducerReadiness<'_>) -> ProducerReadiness<'static> {
    match readiness {
        ProducerReadiness::HostVisible => ProducerReadiness::HostVisible,
        ProducerReadiness::DeviceStreamOrdered { stream } => ProducerReadiness::DeviceStreamOrdered {
            stream: CudaStreamHandle::from_raw_on_device(stream.as_raw(), stream.device())
                .expect("existing readiness stream handle must remain non-null and device-qualified"),
        },
        ProducerReadiness::DeviceEventOrdered { event } => ProducerReadiness::DeviceEventOrdered {
            event: CudaEventHandle::from_raw_on_device(event.as_raw(), event.device())
                .expect("existing readiness event handle must remain non-null and device-qualified"),
        },
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
