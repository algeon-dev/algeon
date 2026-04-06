//! External interop boundaries.
//!
//! This module re-exports the shared RAPIDS Rust interop contract from
//! [`rapids_interop`] and implements `cudf-rs` producer adapters for:
//!
//! - [`crate::Column`]
//! - [`crate::ColumnView`]
//! - [`crate::Table`]
//! - [`crate::TableView`]
//! - [`crate::Series`]
//! - [`crate::DataFrame`]
//! - [`PartitionedDataFrameExport`]
//!
//! # Contract Summary
//!
//! - The shared [`rapids_interop`] names are the primary interop contract.
//! - `cudf-rs` now supports consumer-side import from the shared borrowed
//!   compatibility descriptors back into low-level owned [`crate::Column`] and
//!   [`crate::Table`] values via explicit destination [`crate::Runtime`].
//! - `cudf-rs` also supports zero-copy consumer-side import from
//!   `cudf-rs`-produced shared owned-transfer roots back into low-level owned
//!   [`crate::Column`] and [`crate::Table`] values.
//! - `cudf-rs` now also supports low-level consumer-side import from shared
//!   prepared-device export roots back into owned [`crate::Column`] and
//!   [`crate::Table`] values via explicit same-device destination
//!   [`crate::Runtime`]s.
//! - [`BorrowedColumnView`] and [`BorrowedTableView`] remain the borrowed GPU
//!   compatibility profile.
//! - [`PreparedColumnExport`] and [`PreparedTableExport`] are producer-rooted
//!   prepared export profiles for activated access.
//! - [`PartitionedPreparedTableExport`] and [`PartitionedOwnedTableTransfer`]
//!   are explicit multi-device wrappers around multiple already-valid
//!   single-device table roots.
//! - [`ActiveColumnReadLease`] and [`ActiveTableReadLease`] model activated
//!   read access with explicit read-start and completion contracts.
//! - [`CompletionWitness`] and [`LeaseCompletionResult`] make completion and
//!   drop fallback visible at the shared contract layer.
//! - [`RuntimeReadStartAuthority`], [`RuntimeCompletionAuthority`],
//!   [`RuntimeActiveColumnReadLease`], [`RuntimeActiveTableReadLease`], and
//!   their related `Runtime*` result and witness types are CUDA-specific
//!   compatibility shims layered on that same shared contract. Prefer the
//!   shared names for new interop APIs unless you specifically need `cudf-rs`
//!   runtime handles and producer-owned [`crate::runtime::CudaEvent`]
//!   witnesses.
//! - Borrowing and prepared export do not transfer ownership.
//! - Borrowed compatibility and prepared export both avoid host copies on the
//!   input path because they describe existing device buffers.
//! - Prepared host export is a separate materialized profile: it copies the
//!   active logical range to host-visible bytes and is not zero-copy.
//! - Owned device transfer is a separate zero-copy ownership move profile for
//!   true owner types such as [`crate::Column`] and [`crate::Table`].
//! - [`PartitionedDataFrameExport`] is an interop-only heterogeneous producer
//!   surface; it does not change the ordinary single-provenance semantics of
//!   [`crate::DataFrame`] or [`crate::Table`].
//! - Consumer algorithms may still allocate or copy for outputs; this module
//!   does not promise zero-copy output.
//! - Current device-export support covers fixed-width layouts, the first
//!   binary-like string layout slice, `List`, `Struct`, and `Dictionary`.
//! - Nested-layout support is still intentionally narrow in this slice:
//!   current coverage is the recursive semantic-column graph that can be
//!   expressed by the existing fixed-width, binary-like string, list, struct,
//!   and dictionary slices on one CUDA device.
//! - [`crate::Series`] and [`crate::DataFrame`] are convenience adapters, not
//!   the contract itself.
//! - Explicit execution provenance is required for producer projection.
//! - Ambient/default-only objects are rejected at the interop boundary.

mod projection;
mod runtime_lease;
mod table_export;

use std::sync::Arc;

use projection::{prepared_column_projection_from_view, prepared_host_column_projection_from_view};
pub use rapids_interop::{
    ActiveColumnReadLease, ActiveLeaseActivation, ActiveLeaseBackend, ActiveTableReadLease, BorrowedColumnView,
    BorrowedTableView, BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority,
    CompletionBinding, CompletionWitness, CompletionWitnessBackend, CompletionWitnessKind, CudaEventHandle,
    CudaStreamHandle, DataType as InteropDataType, DeviceMemoryResourceHandle, ExecutionBinding, ExportDiagnostics,
    ExportProfile, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation, LeaseBackend,
    LeaseCompletionResult, LeaseExportContext, LeaseReadiness, NullMask, NullMaskKind, OwnedColumnTransfer,
    OwnedTableTransfer, PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferPartition,
    PartitionedPreparedTableExport, PartitionedPreparedTablePartition, PhysicalValidity, Placement,
    PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority, ReadStartBinding,
    ReleaseContract, ReleaseSynchronization, SchemaNullability, SemanticColumn, SemanticTable, TableId,
    TransferCompletionRule, TryBorrowColumnView, TryBorrowTableView, TryIntoOwnedDeviceColumnTransfer,
    TryIntoOwnedDevicePartitionedTableTransfer, TryIntoOwnedDeviceTableTransfer, TryPrepareDeviceColumnExport,
    TryPrepareDevicePartitionedTableExport, TryPrepareDeviceTableExport, TryPrepareHostColumnExport,
    TryPrepareHostTableExport, TypeId as InteropTypeId,
};
pub use runtime_lease::{
    RuntimeActiveColumnReadLease, RuntimeActiveTableReadLease, RuntimeCompletionAuthority, RuntimeCompletionWitness,
    RuntimeLeaseCompletionResult, RuntimeReadStartAuthority, TryActivateRuntimeColumnReadLease,
    TryActivateRuntimeTableReadLease, cuda_explicit_completion_required_lease_backend,
};
use table_export::{
    prepare_device_table_export_from_columns, prepare_device_table_export_from_columns_with_readiness,
    prepare_host_table_export_from_columns, prepare_host_table_export_from_view,
    prepare_partitioned_device_table_export_from_columns, prepare_partitioned_host_table_export_from_columns,
    prepare_table_export_from_view, prepare_table_export_from_view_with_readiness, transfer_from_owned_table,
    transfer_partitioned_owned_table_from_columns,
};

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_to_owned},
        error,
        table::{Table, TableView},
        types::{DataType, TypeId, validity_to_bitmask},
    },
    frame::{dataframe::DataFrame, series::Series},
    internal::{
        hash::{FastIndexMap, fast_index_map, fast_index_map_with_capacity},
        resolve::validate_no_duplicate_names,
    },
    ops::dictionary,
    runtime::{CudaEvent, Runtime, execution_domain::ExecutionDomain},
};

fn map_type_id(type_id: TypeId) -> InteropTypeId {
    match type_id {
        TypeId::Empty => InteropTypeId::Empty,
        TypeId::Int8 => InteropTypeId::Int8,
        TypeId::Int16 => InteropTypeId::Int16,
        TypeId::Int32 => InteropTypeId::Int32,
        TypeId::Int64 => InteropTypeId::Int64,
        TypeId::UInt8 => InteropTypeId::UInt8,
        TypeId::UInt16 => InteropTypeId::UInt16,
        TypeId::UInt32 => InteropTypeId::UInt32,
        TypeId::UInt64 => InteropTypeId::UInt64,
        TypeId::Float32 => InteropTypeId::Float32,
        TypeId::Float64 => InteropTypeId::Float64,
        TypeId::Bool8 => InteropTypeId::Bool8,
        TypeId::TimestampDays => InteropTypeId::TimestampDays,
        TypeId::TimestampSeconds => InteropTypeId::TimestampSeconds,
        TypeId::TimestampMilliseconds => InteropTypeId::TimestampMilliseconds,
        TypeId::TimestampMicroseconds => InteropTypeId::TimestampMicroseconds,
        TypeId::TimestampNanoseconds => InteropTypeId::TimestampNanoseconds,
        TypeId::DurationDays => InteropTypeId::DurationDays,
        TypeId::DurationSeconds => InteropTypeId::DurationSeconds,
        TypeId::DurationMilliseconds => InteropTypeId::DurationMilliseconds,
        TypeId::DurationMicroseconds => InteropTypeId::DurationMicroseconds,
        TypeId::DurationNanoseconds => InteropTypeId::DurationNanoseconds,
        TypeId::Dictionary32 => InteropTypeId::Dictionary32,
        TypeId::String => InteropTypeId::String,
        TypeId::List => InteropTypeId::List,
        TypeId::Decimal32 => InteropTypeId::Decimal32,
        TypeId::Decimal64 => InteropTypeId::Decimal64,
        TypeId::Decimal128 => InteropTypeId::Decimal128,
        TypeId::Struct => InteropTypeId::Struct,
    }
}

fn map_interop_type_id(type_id: InteropTypeId) -> TypeId {
    match type_id {
        InteropTypeId::Empty => TypeId::Empty,
        InteropTypeId::Int8 => TypeId::Int8,
        InteropTypeId::Int16 => TypeId::Int16,
        InteropTypeId::Int32 => TypeId::Int32,
        InteropTypeId::Int64 => TypeId::Int64,
        InteropTypeId::UInt8 => TypeId::UInt8,
        InteropTypeId::UInt16 => TypeId::UInt16,
        InteropTypeId::UInt32 => TypeId::UInt32,
        InteropTypeId::UInt64 => TypeId::UInt64,
        InteropTypeId::Float32 => TypeId::Float32,
        InteropTypeId::Float64 => TypeId::Float64,
        InteropTypeId::Bool8 => TypeId::Bool8,
        InteropTypeId::TimestampDays => TypeId::TimestampDays,
        InteropTypeId::TimestampSeconds => TypeId::TimestampSeconds,
        InteropTypeId::TimestampMilliseconds => TypeId::TimestampMilliseconds,
        InteropTypeId::TimestampMicroseconds => TypeId::TimestampMicroseconds,
        InteropTypeId::TimestampNanoseconds => TypeId::TimestampNanoseconds,
        InteropTypeId::DurationDays => TypeId::DurationDays,
        InteropTypeId::DurationSeconds => TypeId::DurationSeconds,
        InteropTypeId::DurationMilliseconds => TypeId::DurationMilliseconds,
        InteropTypeId::DurationMicroseconds => TypeId::DurationMicroseconds,
        InteropTypeId::DurationNanoseconds => TypeId::DurationNanoseconds,
        InteropTypeId::Dictionary32 => TypeId::Dictionary32,
        InteropTypeId::String => TypeId::String,
        InteropTypeId::List => TypeId::List,
        InteropTypeId::Decimal32 => TypeId::Decimal32,
        InteropTypeId::Decimal64 => TypeId::Decimal64,
        InteropTypeId::Decimal128 => TypeId::Decimal128,
        InteropTypeId::Struct => TypeId::Struct,
    }
}

fn map_data_type(data_type: DataType) -> InteropDataType {
    if data_type.id().is_decimal() {
        InteropDataType::fixed_point(map_type_id(data_type.id()), data_type.scale())
    } else {
        InteropDataType::new(map_type_id(data_type.id()))
    }
}

fn map_cudf_error(kind: InteropErrorKind, operation: &'static str, error: crate::Error) -> InteropError {
    let mut interop = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        interop = interop.with_context(key, value.clone());
    }
    interop
}

fn map_interop_error(kind: InteropErrorKind, operation: &'static str, error: InteropError) -> InteropError {
    let mut mapped = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }
    mapped
}

fn execution_binding<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<ExecutionBinding<'a>, InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Err(InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "borrowed interop requires explicit execution provenance; ambient/default-only objects are not valid producers",
        )
        .with_context("operation", operation)
        .with_context("policy", "ambient_interop_unsupported")
        .with_context("device_snapshot", domain.device_snapshot().to_string())),
        Some(options) => {
            let stream = domain.explicit_stream().expect("explicit domain missing stream");
            let mr = domain.explicit_memory_resource().expect("explicit domain missing memory resource");
            let stream_ptr = stream.interop_raw_handle().map_err(|error| {
                map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error)
            })?;
            let mr_ptr = mr.interop_raw_handle().map_err(|error| {
                map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error)
            })?;
            Ok(ExecutionBinding::explicit(
                options.device,
                CudaStreamHandle::from_raw_on_device(stream_ptr, options.device)?,
                DeviceMemoryResourceHandle::from_raw(mr_ptr)?,
            ))
        },
    }
}

fn prepared_device_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Err(InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "prepared device export requires explicit execution provenance; ambient/default-only objects are not valid producers",
        )
        .with_context("operation", operation)
        .with_context("policy", "ambient_interop_unsupported")
        .with_context("device_snapshot", domain.device_snapshot().to_string())),
        Some(options) => {
            let stream = domain.explicit_stream().expect("explicit domain missing stream");
            let stream_ptr = stream
                .interop_raw_handle()
                .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok((
                options.device,
                ProducerReadiness::DeviceStreamOrdered {
                    stream: CudaStreamHandle::from_raw_on_device(stream_ptr, options.device)?,
                },
            ))
        },
    }
}

fn prepared_device_event_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
    event: &'a CudaEvent,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Err(InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "prepared device export requires explicit execution provenance; ambient/default-only objects are not valid producers",
        )
        .with_context("operation", operation)
        .with_context("policy", "ambient_interop_unsupported")
        .with_context("device_snapshot", domain.device_snapshot().to_string())),
        Some(options) => {
            let event_device = event.device().ordinal_raw();
            if event_device != options.device {
                return Err(InteropError::new(
                    InteropErrorKind::ReadinessMismatch,
                    "prepared export readiness event must live on the producer CUDA device",
                )
                .with_context("operation", operation)
                .with_context("producer_device", options.device.to_string())
                .with_context("event_device", event_device.to_string()));
            }
            let event_ptr = event
                .interop_raw_handle()
                .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok((
                options.device,
                ProducerReadiness::DeviceEventOrdered {
                    event: CudaEventHandle::from_raw_on_device(event_ptr, event_device)?,
                },
            ))
        },
    }
}

fn export_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedDeviceExport)
}

fn host_export_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedHostExport)
}

fn owned_transfer_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::OwnedDeviceTransfer)
}

fn owned_transfer_release_contract(device: i32) -> ReleaseContract {
    ReleaseContract::new(
        device,
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
        ReleaseSynchronization::MaySynchronizeInternally,
    )
}

fn pointer_device(
    operation: &'static str,
    ptr: *const std::ffi::c_void,
    fallback_device: i32,
) -> Result<i32, InteropError> {
    if ptr.is_null() {
        return Ok(fallback_device);
    }

    let mut device = 0i32;
    error::check_status_in(
        unsafe { cudf_sys::cudf_cuda_pointer_get_device(ptr, &mut device) },
        "interop::pointer_device",
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::PlacementUnsupported, operation, error))?;
    Ok(device)
}

fn bound_bytes<'a>(
    operation: &'static str,
    ptr: *const std::ffi::c_void,
    byte_len: usize,
) -> Result<BoundBytes<'a>, InteropError> {
    BoundBytes::from_raw_parts(ptr as *mut std::ffi::c_void, byte_len)
        .map_err(|error| error.with_context("operation", operation))
}

fn retained_host_binding(
    operation: &'static str,
    id: BufferId,
    buffer: &Arc<[u8]>,
) -> Result<BufferBinding<'static>, InteropError> {
    Ok(BufferBinding::new(
        id,
        BoundBytes::from_raw_parts(buffer.as_ptr() as *mut std::ffi::c_void, buffer.len())
            .map_err(|error| error.with_context("operation", operation))?,
        Placement::Host,
    ))
}

struct PreparedColumnProjection<'a> {
    column: SemanticColumn,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    next_buffer_id: u32,
    next_column_id: u32,
}

struct PreparedHostColumnProjection {
    column: SemanticColumn,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
    retained_host_buffers: Vec<Arc<[u8]>>,
    next_buffer_id: u32,
    next_column_id: u32,
}

struct OwnedColumnTransferState {
    // Retain the owning column so owned-transfer buffer bindings continue to
    // point at live device memory for the full transfer lifetime.
    _column: Column,
}

impl OwnedColumnTransferState {
    fn into_column(self) -> Column {
        self._column
    }
}

struct OwnedTableTransferState {
    // Retain the owning table so owned-transfer buffer bindings continue to
    // point at live device memory for the full transfer lifetime.
    _table: Table,
}

impl OwnedTableTransferState {
    fn into_table(self) -> Table {
        self._table
    }
}

struct PreparedDeviceImportColumnGraph {
    ffi_columns: Vec<cudf_sys::cudf_interop_semantic_column_t>,
    ffi_bindings: Vec<cudf_sys::cudf_interop_buffer_binding_t>,
    _child_id_buffers: Vec<Box<[u32]>>,
}

struct PreparedDeviceImportTableGraph {
    ffi_table: cudf_sys::cudf_interop_semantic_table_t,
    ffi_columns: Vec<cudf_sys::cudf_interop_semantic_column_t>,
    ffi_bindings: Vec<cudf_sys::cudf_interop_buffer_binding_t>,
    _child_id_buffers: Vec<Box<[u32]>>,
    _table_column_ids: Box<[u32]>,
}

struct BorrowedPartitionColumnGroup<'a> {
    top_level_column_indices: Vec<usize>,
    columns: Vec<&'a Column>,
    domain: Arc<ExecutionDomain>,
}

struct OwnedPartitionColumnGroup {
    top_level_column_indices: Vec<usize>,
    columns: Vec<Column>,
    domain: Arc<ExecutionDomain>,
}

/// Interop-only high-level producer for heterogeneous retained-provenance
/// column sets that still share one row-count relationship.
///
/// Provenance here means the CUDA device and execution-domain lineage on
/// which each column's buffers were produced and remain valid.
///
/// Unlike [`crate::DataFrame`], this type does not claim one shared retained
/// provenance across all columns and does not participate in general dataframe
/// algebra. Its role is to drive partitioned device export, heterogeneous
/// host-visible export, and partitioned owned transfer honestly.
pub struct PartitionedDataFrameExport {
    columns: FastIndexMap<String, Column>,
    row_count: usize,
}

impl PartitionedDataFrameExport {
    /// Create a new heterogeneous interop producer from ordered column names
    /// and owned columns.
    pub fn new(names: Vec<String>, columns: Vec<Column>) -> crate::Result<Self> {
        if names.len() != columns.len() {
            return Err(crate::Error::new(
                crate::ErrorKind::InvalidOperation,
                "number of names must equal number of columns",
            )
            .with_operation("interop::PartitionedDataFrameExport::new")
            .with_context("names_count", names.len().to_string())
            .with_context("columns_count", columns.len().to_string()));
        }

        validate_no_duplicate_names(&names, "interop::PartitionedDataFrameExport::new")?;

        if columns.is_empty() {
            return Ok(Self { columns: fast_index_map(), row_count: 0 });
        }

        let first_rows = columns[0].view()?.len()?;
        for (index, column) in columns.iter().enumerate().skip(1) {
            let rows = column.view()?.len()?;
            if rows != first_rows {
                return Err(crate::Error::new(
                    crate::ErrorKind::InvalidOperation,
                    format!("column {index} has {rows} rows but column 0 has {first_rows} rows"),
                )
                .with_operation("interop::PartitionedDataFrameExport::new")
                .with_context("column_index", index.to_string())
                .with_context("column_rows", rows.to_string())
                .with_context("expected_rows", first_rows.to_string()));
            }
        }

        let mut map: FastIndexMap<String, Column> = fast_index_map_with_capacity(names.len());
        for (name, column) in names.into_iter().zip(columns) {
            map.insert(name, column);
        }

        Ok(Self { columns: map, row_count: first_rows })
    }

    /// Convenience constructor from a fixed-size array of `(name, Column)`
    /// pairs.
    pub fn from_columns<const N: usize>(pairs: [(&str, Column); N]) -> crate::Result<Self> {
        let (names, columns): (Vec<_>, Vec<_>) =
            pairs.into_iter().map(|(name, column)| (name.to_string(), column)).unzip();
        Self::new(names, columns)
    }

    pub fn num_rows(&self) -> usize {
        self.row_count
    }

    fn column_names_owned(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    fn grouped_borrowed_columns(&self) -> Vec<BorrowedPartitionColumnGroup<'_>> {
        let mut groups: Vec<BorrowedPartitionColumnGroup<'_>> = Vec::new();

        for (index, column) in self.columns.values().enumerate() {
            if let Some(group) = groups
                .iter_mut()
                .find(|group| ExecutionDomain::same_provenance(&group.domain, column.execution_domain()))
            {
                group.top_level_column_indices.push(index);
                group.columns.push(column);
            } else {
                groups.push(BorrowedPartitionColumnGroup {
                    top_level_column_indices: vec![index],
                    columns: vec![column],
                    domain: Arc::clone(column.execution_domain()),
                });
            }
        }

        groups
    }

    fn grouped_owned_columns(self) -> Vec<OwnedPartitionColumnGroup> {
        let mut groups: Vec<OwnedPartitionColumnGroup> = Vec::new();

        for (index, (_, column)) in self.columns.into_iter().enumerate() {
            if let Some(group) = groups
                .iter_mut()
                .find(|group| ExecutionDomain::same_provenance(&group.domain, column.execution_domain()))
            {
                group.top_level_column_indices.push(index);
                group.columns.push(column);
            } else {
                groups.push(OwnedPartitionColumnGroup {
                    top_level_column_indices: vec![index],
                    domain: Arc::clone(column.execution_domain()),
                    columns: vec![column],
                });
            }
        }

        groups
    }

    /// Project this heterogeneous producer into the shared RAPIDS partitioned
    /// prepared device export profile.
    pub fn try_prepare_device_partitioned_table_export(
        &self,
    ) -> Result<PartitionedPreparedTableExport<'_>, InteropError> {
        <Self as TryPrepareDevicePartitionedTableExport>::try_prepare_device_partitioned_table_export(self)
    }

    /// Project this heterogeneous producer into one host-visible prepared
    /// table export root by materializing supported top-level columns to owned
    /// host bytes.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        <Self as TryPrepareHostTableExport>::try_prepare_host_table_export(self)
    }

    /// Move this heterogeneous producer into the shared RAPIDS partitioned
    /// owned device transfer profile.
    pub fn try_into_owned_device_partitioned_table_transfer(
        self,
    ) -> Result<PartitionedOwnedTableTransfer, InteropError> {
        <Self as TryIntoOwnedDevicePartitionedTableTransfer>::try_into_owned_device_partitioned_table_transfer(self)
    }
}

fn to_static_binding(
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
        BoundBytes::from_raw_parts(binding.bytes().base_ptr().as_ptr(), binding.bytes().byte_len())
            .map_err(|error| error.with_context("operation", operation))?,
        *binding.placement(),
    ))
}

fn to_static_bindings(
    operation: &'static str,
    bindings: Vec<BufferBinding<'_>>,
) -> Result<Vec<BufferBinding<'static>>, InteropError> {
    bindings.into_iter().map(|binding| to_static_binding(operation, binding)).collect()
}

fn to_static_readiness(readiness: ProducerReadiness<'_>) -> ProducerReadiness<'static> {
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

fn allocate_buffer_id(operation: &'static str, next_buffer_id: &mut u32) -> Result<BufferId, InteropError> {
    let id = BufferId::new(*next_buffer_id);
    *next_buffer_id = next_buffer_id.checked_add(1).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "buffer id allocation overflowed")
            .with_context("operation", operation)
            .with_context("next_buffer_id", id.get().to_string())
    })?;
    Ok(id)
}

fn allocate_column_id(operation: &'static str, next_column_id: &mut u32) -> Result<ColumnId, InteropError> {
    let id = ColumnId::new(*next_column_id);
    *next_column_id = next_column_id.checked_add(1).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "column id allocation overflowed")
            .with_context("operation", operation)
            .with_context("next_column_id", id.get().to_string())
    })?;
    Ok(id)
}

fn validity_for_prepared_projection<'a>(
    operation: &'static str,
    nullable: bool,
    null_mask_ptr: Option<*const std::ffi::c_void>,
    null_count: usize,
    active_elements: usize,
    validity_id: BufferId,
    default_device: i32,
) -> Result<(PhysicalValidity, Option<BufferBinding<'a>>), InteropError> {
    if !nullable {
        return Ok((PhysicalValidity::NoValidityBuffer, None));
    }

    let mask_byte_len = active_elements.div_ceil(8);
    let mask_ptr = match null_mask_ptr {
        Some(ptr) => ptr,
        None if mask_byte_len == 0 => std::ptr::null(),
        None => {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "nullable column did not expose a null-mask pointer",
            )
            .with_context("operation", operation));
        },
    };
    let mask_device = pointer_device(operation, mask_ptr, default_device)?;
    Ok((
        PhysicalValidity::ValidityMask { mask: validity_id, null_count: Some(null_count) },
        Some(BufferBinding::new(
            validity_id,
            bound_bytes(operation, mask_ptr, mask_byte_len)?,
            Placement::CudaDevice { device: mask_device },
        )),
    ))
}

fn host_validity_for_prepared_projection(
    operation: &'static str,
    validity: Option<Vec<bool>>,
    null_count: usize,
    validity_id: BufferId,
) -> Result<(PhysicalValidity, Option<BufferBinding<'static>>, Option<Arc<[u8]>>), InteropError> {
    let Some(validity) = validity else {
        return Ok((PhysicalValidity::NoValidityBuffer, None, None));
    };

    let mask_bytes: Arc<[u8]> = Arc::from(validity_to_bitmask(&validity));
    let binding = retained_host_binding(operation, validity_id, &mask_bytes)?;
    Ok((
        PhysicalValidity::ValidityMask { mask: validity_id, null_count: Some(null_count) },
        Some(binding),
        Some(mask_bytes),
    ))
}

fn encode_host_string_offsets(operation: &'static str, strings: &[String]) -> Result<Arc<[u8]>, InteropError> {
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    let mut total_bytes = 0usize;
    offsets.push(0i32);

    for value in strings {
        total_bytes = total_bytes.checked_add(value.len()).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "string host export byte range overflowed")
                .with_context("operation", operation)
        })?;
        let offset = i32::try_from(total_bytes).map_err(|_| {
            InteropError::new(
                InteropErrorKind::InvalidView,
                "string host export offsets exceed the current 32-bit offsets contract",
            )
            .with_context("operation", operation)
            .with_context("byte_len", total_bytes.to_string())
        })?;
        offsets.push(offset);
    }

    let bytes: Vec<u8> = offsets.iter().flat_map(|value| value.to_ne_bytes()).collect();
    Ok(Arc::from(bytes))
}

fn encode_host_i32_values(values: &[i32]) -> Arc<[u8]> {
    Arc::from(values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<u8>>())
}

fn borrow_from_materialized_view<'a>(
    operation: &'static str,
    data_type: DataType,
    num_children: i32,
    len: usize,
    offset: usize,
    data_ptr: *const std::ffi::c_void,
    null_mask: NullMask<'a>,
    execution: ExecutionBinding<'a>,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let interop_type = map_data_type(data_type);
    if interop_type.id().fixed_width_size_bytes().is_none() {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedType,
            "borrowed interop v1 supports fixed-width column types only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
    }
    if num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "borrowed interop v1 does not support child-bearing column layouts",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("num_children", num_children.to_string()));
    }

    BorrowedColumnView::try_new(interop_type, len, offset, data_ptr, null_mask, execution)
        .map_err(|error| error.with_context("operation", operation))
}

fn null_mask_for_view<'a>(
    operation: &'static str,
    nullable: bool,
    null_mask_ptr: Option<*const std::ffi::c_void>,
    offset: usize,
    len: usize,
    null_count: usize,
) -> Result<NullMask<'a>, InteropError> {
    if !nullable {
        return Ok(NullMask::absent());
    }

    let bit_len = offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "null mask bit length overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
    })?;

    let ptr = null_mask_ptr.ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "nullable column did not expose a null-mask pointer")
            .with_context("operation", operation)
    })?;

    NullMask::present(ptr, bit_len, null_count).map_err(|error| error.with_context("operation", operation))
}

/// Extract all fields from a `ColumnView` and produce a `BorrowedColumnView`.
///
/// The execution binding is provided externally so that callers can choose
/// whether to derive it from the view itself or from the owning column/table.
/// The returned view's lifetime is tied to `'a` (the execution binding and the
/// device memory it describes), not to the `ColumnView` borrow.
fn borrow_from_view_with_execution<'a>(
    view: &ColumnView<'_>,
    execution: ExecutionBinding<'a>,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offset = usize::try_from(
        view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
            .with_context("operation", operation)
    })?;
    let data_ptr =
        view.device_data_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count = usize::try_from(
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0").with_context("operation", operation)
    })?;
    let null_mask_ptr =
        view.null_mask_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_mask = null_mask_for_view(operation, nullable, null_mask_ptr, offset, len, null_count)?;

    borrow_from_materialized_view(operation, data_type, num_children, len, offset, data_ptr, null_mask, execution)
}

fn borrow_from_column_view<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let execution = execution_binding(operation, view.execution_domain())?;
    borrow_from_view_with_execution(view, execution, operation)
}

fn borrow_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let execution = execution_binding(operation, column.execution_domain())?;
    borrow_from_view_with_execution(&view, execution, operation)
}

fn checked_column_index(operation: &'static str, index: usize) -> Result<i32, InteropError> {
    let index_i32 = i32::try_from(index).map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column index exceeds i32 range")
            .with_context("operation", operation)
            .with_context("index", index.to_string())
    })?;
    Ok(index_i32)
}

fn checked_child_index(operation: &'static str, index: usize) -> Result<i32, InteropError> {
    let index_i32 = i32::try_from(index).map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "child index exceeds i32 range")
            .with_context("operation", operation)
            .with_context("index", index.to_string())
    })?;
    Ok(index_i32)
}

fn checked_i32_from_usize(field: &'static str, value: usize, operation: &'static str) -> Result<i32, InteropError> {
    i32::try_from(value).map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, format!("{field} exceeds i32::MAX"))
            .with_context("operation", operation)
            .with_context(field, value.to_string())
    })
}

fn prepared_import_profile_supported(profile: ExportProfile, operation: &'static str) -> Result<(), InteropError> {
    match profile {
        ExportProfile::PreparedDeviceExport | ExportProfile::PreparedHostExport => Ok(()),
        ExportProfile::BorrowedDeviceCompatibility => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "borrowed compatibility descriptors must be imported through the borrowed-view API",
        )
        .with_context("operation", operation)
        .with_context("profile", "borrowed_device_compatibility")
        .with_context("consumer", "cudf-rs")),
        ExportProfile::OwnedDeviceTransfer => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "owned device transfer roots must be imported through the owned-transfer API",
        )
        .with_context("operation", operation)
        .with_context("profile", "owned_device_transfer")
        .with_context("consumer", "cudf-rs")),
    }
}

fn prepared_device_export_producer_device(
    readiness: &ProducerReadiness<'_>,
    operation: &'static str,
) -> Result<i32, InteropError> {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => Ok(stream.device()),
        ProducerReadiness::DeviceEventOrdered { event } => Ok(event.device()),
        ProducerReadiness::HostVisible => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "prepared device export import requires device-backed readiness",
        )
        .with_context("operation", operation)
        .with_context("readiness", "host_visible")
        .with_context("profile", "prepared_device_export")),
    }
}

fn validate_import_runtime_device(
    operation: &'static str,
    expected_device: i32,
    runtime: &Runtime,
) -> Result<(), InteropError> {
    let actual_device = runtime.device().ordinal() as i32;
    if actual_device != expected_device {
        return Err(InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "borrowed descriptor import requires the destination runtime to use the same CUDA device",
        )
        .with_context("operation", operation)
        .with_context("descriptor_device", expected_device.to_string())
        .with_context("runtime_device", actual_device.to_string()));
    }
    Ok(())
}

fn validate_prepared_import_runtime_device(
    operation: &'static str,
    expected_device: i32,
    runtime: &Runtime,
) -> Result<(), InteropError> {
    let actual_device = runtime.device().ordinal() as i32;
    if actual_device != expected_device {
        return Err(InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "prepared device export import requires the destination runtime to use the same CUDA device",
        )
        .with_context("operation", operation)
        .with_context("producer_device", expected_device.to_string())
        .with_context("runtime_device", actual_device.to_string()));
    }
    Ok(())
}

fn interop_binding_from_prepared_binding(binding: &BufferBinding<'_>) -> cudf_sys::cudf_interop_buffer_binding_t {
    let (placement_kind, device) = match binding.placement() {
        Placement::Host => (cudf_sys::cudf_interop_placement_kind_t::CUDF_INTEROP_PLACEMENT_HOST, -1),
        Placement::CudaDevice { device } => {
            (cudf_sys::cudf_interop_placement_kind_t::CUDF_INTEROP_PLACEMENT_CUDA_DEVICE, *device)
        },
    };

    cudf_sys::cudf_interop_buffer_binding_t {
        buffer_id: binding.id().get(),
        data: binding.bytes().base_ptr().as_ptr() as *const _,
        byte_len: binding.bytes().byte_len(),
        placement_kind,
        device,
    }
}

fn interop_semantic_column_from_prepared_column(
    column: &SemanticColumn,
    operation: &'static str,
    child_id_buffers: &mut Vec<Box<[u32]>>,
) -> Result<cudf_sys::cudf_interop_semantic_column_t, InteropError> {
    let (validity_kind, validity_buffer_id, null_count) = match column.validity() {
        PhysicalValidity::NoValidityBuffer => {
            (cudf_sys::cudf_interop_validity_kind_t::CUDF_INTEROP_VALIDITY_NONE, 0, 0)
        },
        PhysicalValidity::ValidityMask { mask, null_count } => (
            cudf_sys::cudf_interop_validity_kind_t::CUDF_INTEROP_VALIDITY_MASK,
            mask.get(),
            match null_count {
                Some(count) => checked_i32_from_usize("null_count", *count, operation)?,
                None => -1,
            },
        ),
    };

    let mut ffi = cudf_sys::cudf_interop_semantic_column_t {
        column_id: column.id().get(),
        type_id: map_interop_type_id(column.data_type().id()).to_raw(),
        scale: column.data_type().scale(),
        len: checked_i32_from_usize("len", column.len(), operation)?,
        offset: checked_i32_from_usize("offset", column.offset(), operation)?,
        validity_kind,
        validity_buffer_id,
        null_count,
        layout_kind: cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_FIXED_WIDTH,
        values_buffer_id: 0,
        offsets_buffer_id: 0,
        data_buffer_id: 0,
        indices_buffer_id: 0,
        child_column_id: 0,
        child_column_ids: std::ptr::null(),
        num_child_column_ids: 0,
        keys_column_id: 0,
    };

    match column.layout() {
        ColumnLayout::FixedWidth { values } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_FIXED_WIDTH;
            ffi.values_buffer_id = values.get();
        },
        ColumnLayout::BinaryLike { offsets, data } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_BINARY_LIKE;
            ffi.offsets_buffer_id = offsets.get();
            ffi.data_buffer_id = data.get();
        },
        ColumnLayout::List { offsets, child } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_LIST;
            ffi.offsets_buffer_id = offsets.get();
            ffi.child_column_id = child.get();
        },
        ColumnLayout::Struct { children } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_STRUCT;
            let child_ids: Box<[u32]> = children.iter().map(|child| child.get()).collect::<Vec<_>>().into_boxed_slice();
            ffi.num_child_column_ids = checked_i32_from_usize("num_child_column_ids", child_ids.len(), operation)?;
            ffi.child_column_ids = child_ids.as_ptr();
            child_id_buffers.push(child_ids);
        },
        ColumnLayout::Dictionary { indices, keys } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_DICTIONARY;
            ffi.indices_buffer_id = indices.get();
            ffi.keys_column_id = keys.get();
        },
    }

    Ok(ffi)
}

fn prepared_device_import_column_graph(
    export: &PreparedColumnExport<'_>,
    operation: &'static str,
) -> Result<PreparedDeviceImportColumnGraph, InteropError> {
    let mut child_id_buffers = Vec::new();
    let ffi_columns = export
        .semantic_columns()
        .iter()
        .map(|column| interop_semantic_column_from_prepared_column(column, operation, &mut child_id_buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let ffi_bindings = export.bindings().iter().map(interop_binding_from_prepared_binding).collect();
    Ok(PreparedDeviceImportColumnGraph { ffi_columns, ffi_bindings, _child_id_buffers: child_id_buffers })
}

fn prepared_device_import_table_graph(
    export: &PreparedTableExport<'_>,
    operation: &'static str,
) -> Result<PreparedDeviceImportTableGraph, InteropError> {
    let mut child_id_buffers = Vec::new();
    let ffi_columns = export
        .semantic_columns()
        .iter()
        .map(|column| interop_semantic_column_from_prepared_column(column, operation, &mut child_id_buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let ffi_bindings = export.bindings().iter().map(interop_binding_from_prepared_binding).collect::<Vec<_>>();
    let table_column_ids: Box<[u32]> =
        export.table().columns().iter().map(|column| column.get()).collect::<Vec<_>>().into();
    let ffi_table = cudf_sys::cudf_interop_semantic_table_t {
        num_rows: checked_i32_from_usize("num_rows", export.table().num_rows(), operation)?,
        column_ids: table_column_ids.as_ptr(),
        num_columns: checked_i32_from_usize("num_columns", table_column_ids.len(), operation)?,
    };

    Ok(PreparedDeviceImportTableGraph {
        ffi_table,
        ffi_columns,
        ffi_bindings,
        _child_id_buffers: child_id_buffers,
        _table_column_ids: table_column_ids,
    })
}

fn synchronize_runtime_completion_result(
    result: RuntimeLeaseCompletionResult,
    operation: &'static str,
) -> Result<(), InteropError> {
    match result {
        RuntimeLeaseCompletionResult::FallbackSynchronized => Ok(()),
        RuntimeLeaseCompletionResult::Completed(witness) => witness
            .synchronize()
            .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)),
    }
}

fn synchronize_shared_completion_result(
    result: LeaseCompletionResult,
    operation: &'static str,
) -> Result<(), InteropError> {
    match result {
        LeaseCompletionResult::FallbackSynchronized => Ok(()),
        LeaseCompletionResult::Completed(witness) => witness
            .synchronize()
            .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error)),
    }
}

fn borrowed_null_mask_parts(
    borrowed: &BorrowedColumnView<'_>,
    operation: &'static str,
) -> Result<(*const std::ffi::c_void, i32), InteropError> {
    match borrowed.null_mask().kind() {
        NullMaskKind::Absent => Ok((std::ptr::null(), 0)),
        NullMaskKind::Present { ptr, null_count, .. } => {
            Ok((ptr.as_ptr() as *const std::ffi::c_void, checked_i32_from_usize("null_count", *null_count, operation)?))
        },
    }
}

fn import_column_from_borrowed_view(
    borrowed: &BorrowedColumnView<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Column, InteropError> {
    validate_import_runtime_device(operation, borrowed.execution().device(), runtime)?;
    let len = checked_i32_from_usize("len", borrowed.len(), operation)?;
    let offset = checked_i32_from_usize("offset", borrowed.offset(), operation)?;
    let (null_mask, null_count) = borrowed_null_mask_parts(borrowed, operation)?;
    let exec_options = runtime
        .execution_domain()
        .explicit_exec_options_raw()
        .expect("runtime always provides explicit execution options");
    let type_id = map_interop_type_id(borrowed.data_type().id());
    let scale = borrowed.data_type().scale();
    let mut handle = std::ptr::null_mut();

    error::check_status_in(
        unsafe {
            cudf_sys::cudf_column_copy_from_borrowed_view_ex(
                type_id.to_raw(),
                scale,
                len,
                offset,
                borrowed.data_ptr(),
                null_mask,
                null_count,
                &exec_options,
                &mut handle,
            )
        },
        operation,
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

    Ok(unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) })
}

fn import_table_from_borrowed_view(
    borrowed: &BorrowedTableView<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Table, InteropError> {
    validate_import_runtime_device(operation, borrowed.execution().device(), runtime)?;
    if borrowed.num_columns() == 0 {
        if borrowed.num_rows() != 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "borrowed table import cannot materialize a zero-column table with non-zero row count",
            )
            .with_context("operation", operation)
            .with_context("num_rows", borrowed.num_rows().to_string()));
        }

        return Table::from_columns(Vec::new())
            .map(|table| table.with_execution_domain(Arc::clone(runtime.execution_domain())))
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error));
    }

    let columns = borrowed
        .columns()
        .iter()
        .map(|column| import_column_from_borrowed_view(column, runtime, operation))
        .collect::<Result<Vec<_>, _>>()?;

    Table::from_columns(columns).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))
}

fn import_column_from_owned_transfer(
    transfer: OwnedColumnTransfer,
    operation: &'static str,
) -> Result<Column, InteropError> {
    transfer.try_into_owner::<OwnedColumnTransferState>().map(OwnedColumnTransferState::into_column).map_err(|_| {
        InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "cudf-rs owned column import requires a cudf-rs-produced owned device transfer root",
        )
        .with_context("operation", operation)
        .with_context("consumer", "cudf-rs")
        .with_context("profile", "owned_device_transfer")
    })
}

fn import_table_from_owned_transfer(
    transfer: OwnedTableTransfer,
    operation: &'static str,
) -> Result<Table, InteropError> {
    transfer.try_into_owner::<OwnedTableTransferState>().map(OwnedTableTransferState::into_table).map_err(|_| {
        InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "cudf-rs owned table import requires a cudf-rs-produced owned device transfer root",
        )
        .with_context("operation", operation)
        .with_context("consumer", "cudf-rs")
        .with_context("profile", "owned_device_transfer")
    })
}

fn import_column_from_prepared_export(
    export: &PreparedColumnExport<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Column, InteropError> {
    prepared_import_profile_supported(export.diagnostics().profile(), operation)?;
    let graph = prepared_device_import_column_graph(export, operation)?;
    let exec_options = runtime
        .execution_domain()
        .explicit_exec_options_raw()
        .expect("runtime always provides explicit execution options");
    let num_semantic_columns = checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)?;
    let num_bindings = checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)?;

    match export.diagnostics().profile() {
        ExportProfile::PreparedDeviceExport => {
            let producer_device = prepared_device_export_producer_device(export.readiness(), operation)?;
            validate_prepared_import_runtime_device(operation, producer_device, runtime)?;
            let lease = export.try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(runtime.stream()),
                RuntimeCompletionAuthority::same_device_stream(runtime.stream()),
            )?;
            let mut handle = std::ptr::null_mut();

            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_column_copy_from_prepared_device_export_ex(
                        export.column().id().get(),
                        graph.ffi_columns.as_ptr(),
                        num_semantic_columns,
                        graph.ffi_bindings.as_ptr(),
                        num_bindings,
                        &exec_options,
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) };
            let completion = lease
                .complete()
                .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_runtime_completion_result(completion, operation)?;
            Ok(imported)
        },
        ExportProfile::PreparedHostExport => {
            if !matches!(export.readiness(), ProducerReadiness::HostVisible) {
                return Err(InteropError::new(
                    InteropErrorKind::ProfileUnsupported,
                    "prepared host export import requires host-visible readiness",
                )
                .with_context("operation", operation)
                .with_context("profile", "prepared_host_export"));
            }

            let lease = export
                .try_activate_read_lease(
                    ReadStartAuthority::SynchronousByReturn,
                    CompletionAuthority::SynchronousByReturn,
                )
                .map_err(|error| error.with_context("operation", operation))?;
            let mut handle = std::ptr::null_mut();

            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_column_copy_from_prepared_host_export_ex(
                        export.column().id().get(),
                        graph.ffi_columns.as_ptr(),
                        num_semantic_columns,
                        graph.ffi_bindings.as_ptr(),
                        num_bindings,
                        &exec_options,
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) };
            let completion = lease
                .complete()
                .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_shared_completion_result(completion, operation)?;
            Ok(imported)
        },
        // `prepared_import_profile_supported` rejects non-prepared profiles
        // before this match executes.
        ExportProfile::BorrowedDeviceCompatibility | ExportProfile::OwnedDeviceTransfer => unreachable!(),
    }
}

fn import_table_from_prepared_export(
    export: &PreparedTableExport<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Table, InteropError> {
    prepared_import_profile_supported(export.diagnostics().profile(), operation)?;
    let graph = prepared_device_import_table_graph(export, operation)?;
    let exec_options = runtime
        .execution_domain()
        .explicit_exec_options_raw()
        .expect("runtime always provides explicit execution options");
    let num_semantic_columns = checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)?;
    let num_bindings = checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)?;

    match export.diagnostics().profile() {
        ExportProfile::PreparedDeviceExport => {
            let producer_device = prepared_device_export_producer_device(export.readiness(), operation)?;
            validate_prepared_import_runtime_device(operation, producer_device, runtime)?;
            let lease = export.try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(runtime.stream()),
                RuntimeCompletionAuthority::same_device_stream(runtime.stream()),
            )?;
            let mut handle = std::ptr::null_mut();

            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_table_copy_from_prepared_device_export_ex(
                        &graph.ffi_table,
                        graph.ffi_columns.as_ptr(),
                        num_semantic_columns,
                        graph.ffi_bindings.as_ptr(),
                        num_bindings,
                        &exec_options,
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported =
                unsafe { Table::from_raw(handle) }.with_execution_domain(Arc::clone(runtime.execution_domain()));
            let completion = lease
                .complete()
                .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_runtime_completion_result(completion, operation)?;
            Ok(imported)
        },
        ExportProfile::PreparedHostExport => {
            if !matches!(export.readiness(), ProducerReadiness::HostVisible) {
                return Err(InteropError::new(
                    InteropErrorKind::ProfileUnsupported,
                    "prepared host export import requires host-visible readiness",
                )
                .with_context("operation", operation)
                .with_context("profile", "prepared_host_export"));
            }

            let lease = export
                .try_activate_read_lease(
                    ReadStartAuthority::SynchronousByReturn,
                    CompletionAuthority::SynchronousByReturn,
                )
                .map_err(|error| error.with_context("operation", operation))?;
            let mut handle = std::ptr::null_mut();

            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_table_copy_from_prepared_host_export_ex(
                        &graph.ffi_table,
                        graph.ffi_columns.as_ptr(),
                        num_semantic_columns,
                        graph.ffi_bindings.as_ptr(),
                        num_bindings,
                        &exec_options,
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported =
                unsafe { Table::from_raw(handle) }.with_execution_domain(Arc::clone(runtime.execution_domain()));
            let completion = lease
                .complete()
                .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_shared_completion_result(completion, operation)?;
            Ok(imported)
        },
        // `prepared_import_profile_supported` rejects non-prepared profiles
        // before this match executes.
        ExportProfile::BorrowedDeviceCompatibility | ExportProfile::OwnedDeviceTransfer => unreachable!(),
    }
}

fn prepare_from_column_view<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let (default_device, readiness) = prepared_device_readiness(operation, view.execution_domain())?;
    prepare_from_column_view_with_readiness(view, operation, default_device, readiness)
}

fn prepare_from_column_view_with_readiness<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let projection = prepared_column_projection_from_view(view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_host_from_column_view(
    view: &ColumnView<'_>,
    operation: &'static str,
) -> Result<PreparedColumnExport<'static>, InteropError> {
    let projection = prepared_host_column_projection_from_view(view, operation, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(projection.retained_host_buffers))
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_host_from_owned_column(
    column: &Column,
    operation: &'static str,
) -> Result<PreparedColumnExport<'static>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let projection = prepared_host_column_projection_from_view(&view, operation, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(projection.retained_host_buffers))
    .map_err(|error| error.with_context("operation", operation))
}

fn transfer_from_owned_column(column: Column, operation: &'static str) -> Result<OwnedColumnTransfer, InteropError> {
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let projection = {
        let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?
    };
    OwnedColumnTransfer::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        // SAFETY: `to_static_bindings` depends on this retained owner to keep
        // the source column alive for the transfer lifetime.
        to_static_bindings(operation, projection.bindings)?,
        to_static_readiness(readiness),
        owned_transfer_diagnostics(),
        owned_transfer_release_contract(default_device),
        OwnedColumnTransferState { _column: column },
    )
    .map_err(|error| error.with_context("operation", operation))
}

impl<'view> ColumnView<'view> {
    /// Project this low-level borrowed column view into the shared RAPIDS
    /// borrowed interop ABI.
    ///
    /// This is a borrowed-only projection: it does not transfer ownership and
    /// avoids host copies on the input path. The current borrowed compatibility
    /// profile accepts only fixed-width column types and preserves the column's
    /// execution binding.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }

    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        <Self as TryPrepareDeviceColumnExport>::try_prepare_device_column_export(self)
    }

    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        <Self as TryPrepareHostColumnExport>::try_prepare_host_column_export(self)
    }

    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared device export profile with producer event-ordered readiness.
    ///
    /// This still returns a producer-rooted export; the supplied event only
    /// refines the read-start fact carried by the export's shared readiness
    /// contract.
    pub fn try_prepare_device_column_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedColumnExport<'a>, InteropError> {
        let operation = "interop::ColumnView::try_prepare_device_column_export_with_event";
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_from_column_view_with_readiness(self, operation, default_device, readiness)
    }
}

impl TryBorrowColumnView for ColumnView<'_> {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_column_view(self, "interop::ColumnView::try_borrow_column_view")
    }
}

impl TryPrepareDeviceColumnExport for ColumnView<'_> {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_column_view(self, "interop::ColumnView::try_prepare_device_column_export")
    }
}

impl TryPrepareHostColumnExport for ColumnView<'_> {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_column_view(self, "interop::ColumnView::try_prepare_host_column_export")
    }
}

impl Column {
    /// Deep-copy a shared borrowed interop column descriptor into an owned
    /// `cudf-rs` column inside the provided explicit runtime.
    ///
    /// This is a device-to-device copy on the descriptor's CUDA device. It is
    /// not zero-copy, and the destination runtime must be pinned to the same
    /// CUDA device as the borrowed execution binding.
    pub fn try_from_borrowed_column_view_with_runtime(
        borrowed: &BorrowedColumnView<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_column_from_borrowed_view(
            borrowed,
            runtime,
            "interop::Column::try_from_borrowed_column_view_with_runtime",
        )
    }

    /// Consume a shared owned device transfer root back into an owned
    /// `cudf-rs` column.
    ///
    /// This is a zero-copy ownership round-trip for transfer roots produced by
    /// `cudf-rs` low-level owned columns. It preserves the producer's retained
    /// execution provenance and does not deep-copy into a destination runtime.
    pub fn try_from_owned_device_column_transfer(transfer: OwnedColumnTransfer) -> Result<Self, InteropError> {
        import_column_from_owned_transfer(transfer, "interop::Column::try_from_owned_device_column_transfer")
    }

    /// Deep-copy a shared prepared export root into an owned
    /// `cudf-rs` column inside the provided explicit runtime.
    ///
    /// Prepared device exports activate on the destination runtime's CUDA
    /// stream and perform a same-device device-to-device copy. Prepared host
    /// exports activate synchronously by return and upload their host-visible
    /// bytes into new device ownership inside the destination runtime. Both
    /// paths synchronize completion before returning so producer liveness can
    /// end honestly.
    pub fn try_from_prepared_column_export_with_runtime(
        export: &PreparedColumnExport<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_column_from_prepared_export(
            export,
            runtime,
            "interop::Column::try_from_prepared_column_export_with_runtime",
        )
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    ///
    /// Borrowed interop remains explicit on [`crate::ColumnView`]. Use
    /// [`crate::Column::view`] followed by
    /// [`crate::ColumnView::try_borrow_column_view`] when you need a
    /// lifetime-bound borrowed descriptor.
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        <Self as TryPrepareDeviceColumnExport>::try_prepare_device_column_export(self)
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        <Self as TryPrepareHostColumnExport>::try_prepare_host_column_export(self)
    }

    /// Move this owned column into the shared RAPIDS owned device transfer
    /// profile.
    pub fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer, InteropError> {
        <Self as TryIntoOwnedDeviceColumnTransfer>::try_into_owned_device_column_transfer(self)
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// device export profile with producer event-ordered readiness.
    pub fn try_prepare_device_column_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedColumnExport<'a>, InteropError> {
        let operation = "interop::Column::try_prepare_device_column_export_with_event";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
        PreparedColumnExport::try_new_with_semantic_columns(
            projection.column,
            projection.semantic_columns,
            projection.bindings,
            readiness,
            export_diagnostics(),
        )
        .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
        .map_err(|error| error.with_context("operation", operation))
    }
}

impl TryPrepareDeviceColumnExport for Column {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_owned_column(self, "interop::Column::try_prepare_device_column_export")
    }
}

impl TryPrepareHostColumnExport for Column {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_owned_column(self, "interop::Column::try_prepare_host_column_export")
    }
}

impl TryIntoOwnedDeviceColumnTransfer for Column {
    fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer, InteropError> {
        transfer_from_owned_column(self, "interop::Column::try_into_owned_device_column_transfer")
    }
}

impl TableView<'_> {
    /// Project this low-level borrowed table view into the shared RAPIDS
    /// borrowed interop ABI.
    ///
    /// This is a borrowed-only projection: it does not transfer ownership and
    /// avoids host copies on the input path. The current borrowed compatibility
    /// profile accepts only fixed-width columns and preserves the table
    /// execution binding.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }

    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        <Self as TryPrepareDeviceTableExport>::try_prepare_device_table_export(self)
    }

    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared host export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        <Self as TryPrepareHostTableExport>::try_prepare_host_table_export(self)
    }

    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared device export profile with producer event-ordered readiness.
    ///
    /// This still returns a producer-rooted export; the supplied event only
    /// refines the read-start fact carried by the export's shared readiness
    /// contract.
    pub fn try_prepare_device_table_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedTableExport<'a>, InteropError> {
        let operation = "interop::TableView::try_prepare_device_table_export_with_event";
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_table_export_from_view_with_readiness(self, None, operation, default_device, readiness)
    }
}

impl TryBorrowTableView for TableView<'_> {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        let operation = "interop::TableView::try_borrow_table_view";
        let num_rows = usize::try_from(
            self.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
                .with_context("operation", operation)
        })?;
        let num_columns = usize::try_from(
            self.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
                .with_context("operation", operation)
        })?;
        let execution = execution_binding(operation, self.execution_domain())?;
        let mut columns = Vec::with_capacity(num_columns);
        for index in 0..num_columns {
            let index_i32 = checked_column_index(operation, index)?;
            let view = self
                .column(index_i32)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let borrowed = borrow_from_view_with_execution(&view, execution, operation)?;
            columns.push(borrowed);
        }
        BorrowedTableView::try_new(num_rows, columns, execution)
            .map_err(|error| error.with_context("operation", operation))
    }
}

impl TryPrepareDeviceTableExport for TableView<'_> {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        prepare_table_export_from_view(
            self,
            self.execution_domain(),
            None,
            "interop::TableView::try_prepare_device_table_export",
        )
    }
}

impl TryPrepareHostTableExport for TableView<'_> {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_host_table_export_from_view(self, None, "interop::TableView::try_prepare_host_table_export")
    }
}

impl Table {
    /// Deep-copy a shared borrowed interop table descriptor into an owned
    /// `cudf-rs` table inside the provided explicit runtime.
    ///
    /// This imports each borrowed column through a device-to-device copy on
    /// the descriptor's CUDA device, then rebuilds one owned table in the
    /// destination runtime's retained provenance. It is not zero-copy, and the
    /// destination runtime must be pinned to the same CUDA device as the
    /// borrowed execution binding.
    pub fn try_from_borrowed_table_view_with_runtime(
        borrowed: &BorrowedTableView<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_table_from_borrowed_view(borrowed, runtime, "interop::Table::try_from_borrowed_table_view_with_runtime")
    }

    /// Consume a shared owned device transfer root back into an owned
    /// `cudf-rs` table.
    ///
    /// This is a zero-copy ownership round-trip for transfer roots produced by
    /// `cudf-rs` low-level owned tables. It preserves the producer's retained
    /// execution provenance and does not deep-copy into a destination runtime.
    pub fn try_from_owned_device_table_transfer(transfer: OwnedTableTransfer) -> Result<Self, InteropError> {
        import_table_from_owned_transfer(transfer, "interop::Table::try_from_owned_device_table_transfer")
    }

    /// Deep-copy a shared prepared export root into an owned
    /// `cudf-rs` table inside the provided explicit runtime.
    ///
    /// Prepared device exports activate on the destination runtime's CUDA
    /// stream and perform same-device device-to-device copies for the current
    /// prepared-device layout set. Prepared host exports activate
    /// synchronously by return and upload their host-visible bytes into new
    /// device ownership inside the destination runtime. Both paths
    /// synchronize completion before returning so producer liveness can end
    /// honestly.
    pub fn try_from_prepared_table_export_with_runtime(
        export: &PreparedTableExport<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_table_from_prepared_export(
            export,
            runtime,
            "interop::Table::try_from_prepared_table_export_with_runtime",
        )
    }

    /// Project this owned low-level table into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    ///
    /// Borrowed interop remains explicit on [`crate::TableView`]. Use
    /// [`crate::Table::view`] followed by
    /// [`crate::TableView::try_borrow_table_view`] when you need a
    /// lifetime-bound borrowed descriptor.
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        <Self as TryPrepareDeviceTableExport>::try_prepare_device_table_export(self)
    }

    /// Project this owned low-level table into the shared RAPIDS prepared host
    /// export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        <Self as TryPrepareHostTableExport>::try_prepare_host_table_export(self)
    }

    /// Move this owned table into the shared RAPIDS owned device transfer
    /// profile.
    pub fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer, InteropError> {
        <Self as TryIntoOwnedDeviceTableTransfer>::try_into_owned_device_table_transfer(self)
    }

    /// Project this owned low-level table into the shared RAPIDS prepared
    /// device export profile with producer event-ordered readiness.
    pub fn try_prepare_device_table_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedTableExport<'a>, InteropError> {
        let operation = "interop::Table::try_prepare_device_table_export_with_event";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_table_export_from_view_with_readiness(&view, None, operation, default_device, readiness)
    }
}

impl TryPrepareDeviceTableExport for Table {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        let operation = "interop::Table::try_prepare_device_table_export";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepare_table_export_from_view(&view, self.execution_domain(), None, operation)
    }
}

impl TryPrepareHostTableExport for Table {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        let operation = "interop::Table::try_prepare_host_table_export";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepare_host_table_export_from_view(&view, None, operation)
    }
}

impl TryIntoOwnedDeviceTableTransfer for Table {
    fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer, InteropError> {
        transfer_from_owned_table(self, "interop::Table::try_into_owned_device_table_transfer")
    }
}

impl Series {
    /// Convenience producer that lends this high-level `Series` as a shared
    /// borrowed interop column view without host copies on the input path.
    ///
    /// `Series` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and is
    /// limited to the current borrowed compatibility profile's fixed-width type
    /// support.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }

    /// Convenience producer that projects this high-level `Series` into the
    /// shared prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    ///
    /// Owned device transfer remains on low-level [`crate::Column`] because
    /// `Series` is a named, potentially `Arc`-shared wrapper rather than a
    /// uniquely owned device buffer root.
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        <Self as TryPrepareDeviceColumnExport>::try_prepare_device_column_export(self)
    }

    /// Convenience producer that projects this high-level `Series` into the
    /// shared prepared host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        <Self as TryPrepareHostColumnExport>::try_prepare_host_column_export(self)
    }

    /// Convenience producer that projects this high-level `Series` into the
    /// shared prepared device export profile with producer event-ordered
    /// readiness.
    pub fn try_prepare_device_column_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedColumnExport<'a>, InteropError> {
        self.as_column().try_prepare_device_column_export_with_event(event)
    }
}

impl TryBorrowColumnView for Series {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_owned_column(self.as_column(), "interop::Series::try_borrow_column_view")
    }
}

impl TryPrepareDeviceColumnExport for Series {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_owned_column(self.as_column(), "interop::Series::try_prepare_device_column_export")
    }
}

impl TryPrepareHostColumnExport for Series {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_owned_column(self.as_column(), "interop::Series::try_prepare_host_column_export")
    }
}

impl DataFrame {
    /// Convenience producer that lends this high-level `DataFrame` as a shared
    /// borrowed interop table view without host copies on the input path.
    ///
    /// `DataFrame` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and does
    /// not promise zero-copy output for downstream consumer algorithms.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }

    /// Convenience producer that projects this high-level `DataFrame` into the
    /// shared prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    ///
    /// Owned device transfer remains on low-level [`crate::Table`] because
    /// `DataFrame` is a named, potentially column-shared wrapper around table
    /// storage rather than the table owner itself.
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        <Self as TryPrepareDeviceTableExport>::try_prepare_device_table_export(self)
    }

    /// Convenience producer that projects this high-level `DataFrame` into the
    /// shared prepared host export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        <Self as TryPrepareHostTableExport>::try_prepare_host_table_export(self)
    }

    /// Convenience producer that projects this high-level `DataFrame` into the
    /// shared prepared device export profile with producer event-ordered
    /// readiness.
    pub fn try_prepare_device_table_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedTableExport<'a>, InteropError> {
        let operation = "interop::DataFrame::try_prepare_device_table_export_with_event";
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_device_table_export_from_columns_with_readiness(
            self.columns_map().values().map(AsRef::as_ref),
            self.num_rows(),
            Some(self.column_names_owned()),
            operation,
            default_device,
            readiness,
        )
    }
}

impl TryBorrowTableView for DataFrame {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        let execution = execution_binding("interop::DataFrame::try_borrow_table_view", self.execution_domain())?;
        let columns = self
            .columns_map()
            .values()
            .map(|column| borrow_from_owned_column(column.as_ref(), "interop::DataFrame::try_borrow_table_view"))
            .collect::<Result<Vec<_>, _>>()?;
        BorrowedTableView::try_new(self.num_rows(), columns, execution)
            .map_err(|error| error.with_context("operation", "interop::DataFrame::try_borrow_table_view"))
    }
}

impl TryPrepareDeviceTableExport for DataFrame {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        prepare_device_table_export_from_columns(
            self.columns_map().values().map(AsRef::as_ref),
            self.num_rows(),
            Some(self.column_names_owned()),
            self.execution_domain(),
            "interop::DataFrame::try_prepare_device_table_export",
        )
    }
}

impl TryPrepareHostTableExport for DataFrame {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_host_table_export_from_columns(
            self.columns_map().values().map(AsRef::as_ref),
            self.num_rows(),
            Some(self.column_names_owned()),
            "interop::DataFrame::try_prepare_host_table_export",
        )
    }
}

impl TryPrepareDevicePartitionedTableExport for PartitionedDataFrameExport {
    fn try_prepare_device_partitioned_table_export(&self) -> Result<PartitionedPreparedTableExport<'_>, InteropError> {
        prepare_partitioned_device_table_export_from_columns(
            self,
            "interop::PartitionedDataFrameExport::try_prepare_device_partitioned_table_export",
        )
    }
}

impl TryPrepareHostTableExport for PartitionedDataFrameExport {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_partitioned_host_table_export_from_columns(
            self,
            "interop::PartitionedDataFrameExport::try_prepare_host_table_export",
        )
    }
}

impl TryIntoOwnedDevicePartitionedTableTransfer for PartitionedDataFrameExport {
    fn try_into_owned_device_partitioned_table_transfer(self) -> Result<PartitionedOwnedTableTransfer, InteropError> {
        transfer_partitioned_owned_table_from_columns(
            self,
            "interop::PartitionedDataFrameExport::try_into_owned_device_partitioned_table_transfer",
        )
    }
}
