//! External interop boundaries.
//!
//! This module re-exports the shared RAPIDS Rust interop contract from
//! [`rapids_interop`] and implements `cudf-rs` producer adapters for:
//!
//! - [`crate::ColumnView`]
//! - [`crate::TableView`]
//! - [`crate::Series`]
//! - [`crate::DataFrame`]
//!
//! # Contract Summary
//!
//! - The shared [`rapids_interop`] names are the primary interop contract.
//! - [`BorrowedColumnView`] and [`BorrowedTableView`] remain the borrowed GPU
//!   compatibility profile.
//! - [`PreparedColumnExport`] and [`PreparedTableExport`] are producer-rooted
//!   prepared export profiles for activated access.
//! - [`PartitionedPreparedTableExport`] and [`PartitionedOwnedTableTransfer`]
//!   are explicit multi-device wrappers around multiple already-valid
//!   single-device table roots.
//! - [`ActiveColumnReadLease`] and [`ActiveTableReadLease`] model activated
//!   read access with explicit completion.
//! - [`CompletionWitness`] and [`LeaseCompletionResult`] make completion and
//!   drop fallback visible at the shared contract layer.
//! - [`RuntimeCompletionAuthority`], [`RuntimeActiveColumnReadLease`],
//!   [`RuntimeActiveTableReadLease`], and their related `Runtime*` result and
//!   witness types are CUDA-specific compatibility shims layered on that same
//!   shared contract. Prefer the shared names for new interop APIs unless you
//!   specifically need `cudf-rs` runtime handles and producer-owned
//!   [`crate::runtime::CudaEvent`] witnesses.
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

mod runtime_lease;

use std::sync::Arc;

use indexmap::IndexMap;
pub use rapids_interop::{
    ActiveColumnReadLease, ActiveLeaseActivation, ActiveLeaseBackend, ActiveTableReadLease, BorrowedColumnView,
    BorrowedTableView, BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority,
    CompletionBinding, CompletionWitness, CompletionWitnessBackend, CompletionWitnessKind, CudaEventHandle,
    CudaStreamHandle, DataType as InteropDataType, DeviceMemoryResourceHandle, ExecutionBinding, ExportDiagnostics,
    ExportProfile, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation, LeaseBackend,
    LeaseCompletionResult, LeaseExportContext, LeaseReadiness, NullMask, NullMaskKind, OwnedColumnTransfer,
    OwnedTableTransfer, PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferPartition,
    PartitionedPreparedTableExport, PartitionedPreparedTablePartition, PhysicalValidity, Placement,
    PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReleaseContract, ReleaseSynchronization,
    SchemaNullability, SemanticColumn, SemanticTable, TableId, TransferCompletionRule, TryBorrowColumnView,
    TryBorrowTableView, TryIntoOwnedDeviceColumnTransfer, TryIntoOwnedDevicePartitionedTableTransfer,
    TryIntoOwnedDeviceTableTransfer, TryPrepareDeviceColumnExport, TryPrepareDevicePartitionedTableExport,
    TryPrepareDeviceTableExport, TryPrepareHostColumnExport, TryPrepareHostTableExport, TypeId as InteropTypeId,
};
pub use runtime_lease::{
    RuntimeActiveColumnReadLease, RuntimeActiveTableReadLease, RuntimeCompletionAuthority, RuntimeCompletionWitness,
    RuntimeLeaseCompletionResult, TryActivateRuntimeColumnReadLease, TryActivateRuntimeTableReadLease,
    cuda_explicit_completion_required_lease_backend,
};

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_to_owned},
        error,
        table::{Table, TableView},
        types::{DataType, TypeId, validity_to_bitmask},
    },
    frame::{dataframe::DataFrame, series::Series},
    internal::resolve::validate_no_duplicate_names,
    ops::dictionary,
    runtime::{CudaEvent, execution_domain::ExecutionDomain},
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
    _column: Column,
}

struct OwnedTableTransferState {
    _table: Table,
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
/// Unlike [`crate::DataFrame`], this type does not claim one shared retained
/// provenance across all columns and does not participate in general dataframe
/// algebra. Its role is to drive partitioned device export, heterogeneous
/// host-visible export, and partitioned owned transfer honestly.
pub struct PartitionedDataFrameExport {
    columns: IndexMap<String, Column>,
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
            return Ok(Self { columns: IndexMap::new(), row_count: 0 });
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

        let mut map = IndexMap::with_capacity(names.len());
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

fn prepared_host_leaf_or_string_column_projection_from_view(
    view: &ColumnView<'_>,
    operation: &'static str,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedHostColumnProjection, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let interop_type = map_data_type(data_type);
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count = usize::try_from(
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0").with_context("operation", operation)
    })?;

    if interop_type.id() == InteropTypeId::String {
        if num_children < 1 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "string column view must expose an offsets child",
            )
            .with_context("operation", operation)
            .with_context("type_id", "string")
            .with_context("num_children", num_children.to_string()));
        }

        let mut next_column_id = next_column_id;
        let mut next_buffer_id = next_buffer_id;
        let column_id = allocate_column_id(operation, &mut next_column_id)?;
        let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let data_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        let strings =
            view.to_strings().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let validity =
            view.validity().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

        let offsets_bytes = encode_host_string_offsets(operation, &strings)?;
        let data_bytes: Arc<[u8]> =
            Arc::from(strings.iter().flat_map(|value| value.as_bytes().iter().copied()).collect::<Vec<u8>>());

        let offsets_binding = retained_host_binding(operation, offsets_id, &offsets_bytes)?;
        let data_binding = retained_host_binding(operation, data_id, &data_bytes)?;
        let (validity, validity_binding, validity_bytes) =
            host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

        let mut bindings = vec![offsets_binding, data_binding];
        let mut retained_host_buffers = vec![offsets_bytes, data_bytes];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        if let Some(bytes) = validity_bytes {
            retained_host_buffers.push(bytes);
        }

        let column = SemanticColumn::new(
            column_id,
            interop_type,
            len,
            0,
            validity,
            SchemaNullability::Unknown,
            ColumnLayout::BinaryLike { offsets: offsets_id, data: data_id },
        );
        return Ok(PreparedHostColumnProjection {
            column: column.clone(),
            semantic_columns: vec![column],
            bindings,
            retained_host_buffers,
            next_buffer_id,
            next_column_id,
        });
    }

    if interop_type.id().fixed_width_size_bytes().is_none() {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedType,
            "prepared host export currently supports fixed-width, string, list, struct, and dictionary32 column types within the current recursive host layout set",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
    }
    if num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "prepared host export currently supports fixed-width leaf column layouts on the non-list fallback path only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("num_children", num_children.to_string()));
    }

    let mut next_column_id = next_column_id;
    let mut next_buffer_id = next_buffer_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;
    let values_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

    let values_bytes: Arc<[u8]> = Arc::from(
        view.to_fixed_width_host_bytes()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    );
    let validity = view.validity().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let values_binding = retained_host_binding(operation, values_id, &values_bytes)?;
    let (validity, validity_binding, validity_bytes) =
        host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

    let mut bindings = vec![values_binding];
    let mut retained_host_buffers = vec![values_bytes];
    if let Some(binding) = validity_binding {
        bindings.push(binding);
    }
    if let Some(bytes) = validity_bytes {
        retained_host_buffers.push(bytes);
    }

    let column = SemanticColumn::new(
        column_id,
        interop_type,
        len,
        0,
        validity,
        SchemaNullability::Unknown,
        ColumnLayout::FixedWidth { values: values_id },
    );
    Ok(PreparedHostColumnProjection {
        column: column.clone(),
        semantic_columns: vec![column],
        bindings,
        retained_host_buffers,
        next_buffer_id,
        next_column_id,
    })
}

fn prepared_host_column_projection_with_current_supported_child_projection_from_view(
    view: &ColumnView<'_>,
    operation: &'static str,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedHostColumnProjection, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let interop_type = map_data_type(data_type);
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count = usize::try_from(
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0").with_context("operation", operation)
    })?;

    if interop_type.id() != InteropTypeId::List {
        return prepared_host_leaf_or_string_column_projection_from_view(
            view,
            operation,
            next_column_id,
            next_buffer_id,
        );
    }

    if num_children < 2 {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            "list column view must expose offsets and child columns",
        )
        .with_context("operation", operation)
        .with_context("type_id", "list")
        .with_context("num_children", num_children.to_string()));
    }

    let root_offset = usize::try_from(
        view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
            .with_context("operation", operation)
    })?;
    let root_end = root_offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "list host export active root range overflowed")
            .with_context("operation", operation)
            .with_context("offset", root_offset.to_string())
            .with_context("len", len.to_string())
    })?;

    let mut next_column_id = next_column_id;
    let mut next_buffer_id = next_buffer_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;
    let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

    let offsets_view =
        view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offsets_type =
        offsets_view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offsets_elem_size = map_data_type(offsets_type).id().fixed_width_size_bytes().ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "list offsets child must be fixed-width")
            .with_context("operation", operation)
            .with_context("offsets_type_id", format!("{:?}", offsets_type.id()).to_lowercase())
    })?;
    if offsets_elem_size != std::mem::size_of::<i32>() {
        return Err(InteropError::new(InteropErrorKind::InvalidView, "list offsets child must use 32-bit offsets")
            .with_context("operation", operation)
            .with_context("offsets_elem_size", offsets_elem_size.to_string()));
    }
    let host_offsets = offsets_view
        .to_vec::<i32>()
        .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    if host_offsets.len() <= root_end {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            "list offsets child is too small for the active logical range",
        )
        .with_context("operation", operation)
        .with_context("offset", root_offset.to_string())
        .with_context("len", len.to_string())
        .with_context("offsets_len", host_offsets.len().to_string()));
    }

    let base_child_offset = host_offsets[root_offset];
    if base_child_offset < 0 {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            "list offsets child must not expose negative offsets",
        )
        .with_context("operation", operation)
        .with_context("offset_index", root_offset.to_string())
        .with_context("offset_value", base_child_offset.to_string()));
    }
    let mut normalized_offsets = Vec::with_capacity(len + 1);
    let mut previous = base_child_offset;
    for (position, raw_offset) in host_offsets[root_offset..=root_end].iter().copied().enumerate() {
        if raw_offset < previous {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "list offsets child must be monotonically non-decreasing",
            )
            .with_context("operation", operation)
            .with_context("offset_index", (root_offset + position).to_string())
            .with_context("previous_offset", previous.to_string())
            .with_context("offset_value", raw_offset.to_string()));
        }
        normalized_offsets.push(raw_offset - base_child_offset);
        previous = raw_offset;
    }
    let offsets_bytes = encode_host_i32_values(&normalized_offsets);
    let offsets_binding = retained_host_binding(operation, offsets_id, &offsets_bytes)?;

    let child_view =
        view.list_sliced_child().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let child_projection =
        prepared_host_column_projection_from_view(&child_view, operation, next_column_id, next_buffer_id)?;
    next_buffer_id = child_projection.next_buffer_id;
    next_column_id = child_projection.next_column_id;

    let validity = view.validity().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let (validity, validity_binding, validity_bytes) =
        host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

    let mut bindings = vec![offsets_binding];
    let mut retained_host_buffers = vec![offsets_bytes];
    if let Some(binding) = validity_binding {
        bindings.push(binding);
    }
    if let Some(bytes) = validity_bytes {
        retained_host_buffers.push(bytes);
    }
    bindings.extend(child_projection.bindings);
    retained_host_buffers.extend(child_projection.retained_host_buffers);

    let column = SemanticColumn::new(
        column_id,
        interop_type,
        len,
        0,
        validity,
        SchemaNullability::Unknown,
        ColumnLayout::List { offsets: offsets_id, child: child_projection.column.id() },
    );
    let mut semantic_columns = vec![column.clone()];
    semantic_columns.extend(child_projection.semantic_columns);
    Ok(PreparedHostColumnProjection {
        column,
        semantic_columns,
        bindings,
        retained_host_buffers,
        next_buffer_id,
        next_column_id,
    })
}

fn prepared_host_column_projection_from_view(
    view: &ColumnView<'_>,
    operation: &'static str,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedHostColumnProjection, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let interop_type = map_data_type(data_type);
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count = usize::try_from(
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0").with_context("operation", operation)
    })?;

    if interop_type.id() == InteropTypeId::Struct {
        let root_offset = usize::try_from(
            view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
                .with_context("operation", operation)
        })?;
        if root_offset != 0 {
            let normalized = copy_view_to_owned(view)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let normalized_view =
                normalized.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            return prepared_host_column_projection_from_view(
                &normalized_view,
                operation,
                next_column_id,
                next_buffer_id,
            );
        }

        let child_count = usize::try_from(num_children).map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "struct child count must be >= 0")
                .with_context("operation", operation)
                .with_context("type_id", "struct")
                .with_context("num_children", num_children.to_string())
        })?;
        if child_count == 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "struct column view must expose at least one child column",
            )
            .with_context("operation", operation)
            .with_context("type_id", "struct"));
        }

        let mut next_column_id = next_column_id;
        let mut next_buffer_id = next_buffer_id;
        let column_id = allocate_column_id(operation, &mut next_column_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity =
            view.validity().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (validity, validity_binding, validity_bytes) =
            host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

        let mut bindings = Vec::new();
        let mut retained_host_buffers = Vec::new();
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        if let Some(bytes) = validity_bytes {
            retained_host_buffers.push(bytes);
        }

        let mut child_ids = Vec::with_capacity(child_count);
        let mut semantic_columns = Vec::new();
        for index in 0..child_count {
            let child_index = checked_child_index(operation, index)?;
            let child_view = view
                .child(child_index)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let child_projection =
                prepared_host_column_projection_from_view(&child_view, operation, next_column_id, next_buffer_id)?;
            next_buffer_id = child_projection.next_buffer_id;
            next_column_id = child_projection.next_column_id;
            child_ids.push(child_projection.column.id());
            bindings.extend(child_projection.bindings);
            retained_host_buffers.extend(child_projection.retained_host_buffers);
            semantic_columns.extend(child_projection.semantic_columns);
        }

        let column = SemanticColumn::new(
            column_id,
            interop_type,
            len,
            0,
            validity,
            SchemaNullability::Unknown,
            ColumnLayout::Struct { children: child_ids },
        );
        let mut all_columns = vec![column.clone()];
        all_columns.extend(semantic_columns);
        return Ok(PreparedHostColumnProjection {
            column,
            semantic_columns: all_columns,
            bindings,
            retained_host_buffers,
            next_buffer_id,
            next_column_id,
        });
    }

    if interop_type.id() == InteropTypeId::Dictionary32 {
        if num_children < 2 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "dictionary column view must expose indices and keys columns",
            )
            .with_context("operation", operation)
            .with_context("type_id", "dictionary32")
            .with_context("num_children", num_children.to_string()));
        }

        let mut next_column_id = next_column_id;
        let mut next_buffer_id = next_buffer_id;
        let column_id = allocate_column_id(operation, &mut next_column_id)?;
        let indices_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        let indices_view = dictionary::indices(view)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_type = indices_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_elem_size = map_data_type(indices_type).id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "dictionary indices child must be fixed-width")
                .with_context("operation", operation)
                .with_context("indices_type_id", format!("{:?}", indices_type.id()).to_lowercase())
        })?;
        if indices_elem_size != std::mem::size_of::<i32>() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "dictionary indices child must use 32-bit indices",
            )
            .with_context("operation", operation)
            .with_context("indices_elem_size", indices_elem_size.to_string()));
        }

        let indices_bytes: Arc<[u8]> = Arc::from(
            indices_view
                .to_fixed_width_host_bytes()
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        );
        let indices_binding = retained_host_binding(operation, indices_id, &indices_bytes)?;

        let keys_view =
            dictionary::keys(view).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let keys_projection =
            prepared_host_column_projection_from_view(&keys_view, operation, next_column_id, next_buffer_id)?;
        next_buffer_id = keys_projection.next_buffer_id;
        next_column_id = keys_projection.next_column_id;

        let validity =
            view.validity().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (validity, validity_binding, validity_bytes) =
            host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

        let mut bindings = vec![indices_binding];
        let mut retained_host_buffers = vec![indices_bytes];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        if let Some(bytes) = validity_bytes {
            retained_host_buffers.push(bytes);
        }
        bindings.extend(keys_projection.bindings);
        retained_host_buffers.extend(keys_projection.retained_host_buffers);

        let column = SemanticColumn::new(
            column_id,
            interop_type,
            len,
            0,
            validity,
            SchemaNullability::Unknown,
            ColumnLayout::Dictionary { indices: indices_id, keys: keys_projection.column.id() },
        );
        let mut semantic_columns = vec![column.clone()];
        semantic_columns.extend(keys_projection.semantic_columns);
        return Ok(PreparedHostColumnProjection {
            column,
            semantic_columns,
            bindings,
            retained_host_buffers,
            next_buffer_id,
            next_column_id,
        });
    }

    prepared_host_column_projection_with_current_supported_child_projection_from_view(
        view,
        operation,
        next_column_id,
        next_buffer_id,
    )
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

fn prepared_column_projection_from_materialized_view<'a>(
    operation: &'static str,
    data_type: DataType,
    num_children: i32,
    len: usize,
    offset: usize,
    data_ptr: *const std::ffi::c_void,
    nullable: bool,
    null_mask_ptr: Option<*const std::ffi::c_void>,
    null_count: usize,
    default_device: i32,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedColumnProjection<'a>, InteropError> {
    let interop_type = map_data_type(data_type);
    let active_elements = offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "active element range overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
    })?;
    let mut next_column_id = next_column_id;
    let mut next_buffer_id = next_buffer_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;

    if num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "prepared device export currently supports fixed-width leaf, string, list, and struct column layouts only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("num_children", num_children.to_string()));
    }

    let values_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let elem_size = interop_type.id().fixed_width_size_bytes().ok_or_else(|| {
        InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "prepared device export currently supports fixed-width leaf column layouts on the non-nested fallback path only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
    })?;
    let values_byte_len = active_elements.checked_mul(elem_size).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "active value byte range overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
            .with_context("elem_size", elem_size.to_string())
    })?;

    let values_device = pointer_device(operation, data_ptr, default_device)?;
    let (validity, validity_binding) = validity_for_prepared_projection(
        operation,
        nullable,
        null_mask_ptr,
        null_count,
        active_elements,
        validity_id,
        default_device,
    )?;
    let mut bindings = vec![BufferBinding::new(
        values_id,
        bound_bytes(operation, data_ptr, values_byte_len)?,
        Placement::CudaDevice { device: values_device },
    )];
    if let Some(binding) = validity_binding {
        bindings.push(binding);
    }

    let column = SemanticColumn::new(
        column_id,
        interop_type,
        len,
        offset,
        validity,
        SchemaNullability::Unknown,
        ColumnLayout::FixedWidth { values: values_id },
    );
    Ok(PreparedColumnProjection {
        column: column.clone(),
        semantic_columns: vec![column],
        bindings,
        next_buffer_id,
        next_column_id,
    })
}

fn prepared_column_projection_from_view<'a>(
    view: &ColumnView<'_>,
    operation: &'static str,
    default_device: i32,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedColumnProjection<'a>, InteropError> {
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
    let active_elements = offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "active element range overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
    })?;
    let mut next_column_id = next_column_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;

    if map_data_type(data_type).id() == InteropTypeId::String {
        let mut next_buffer_id = next_buffer_id;
        let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let data_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        if num_children < 1 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "string column view must expose an offsets child",
            )
            .with_context("operation", operation)
            .with_context("type_id", "string")
            .with_context("num_children", num_children.to_string()));
        }

        let offsets_view =
            view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_type = offsets_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_elem_size = map_data_type(offsets_type).id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "string offsets child must be fixed-width")
                .with_context("operation", operation)
                .with_context("offsets_type_id", format!("{:?}", offsets_type.id()).to_lowercase())
        })?;
        if offsets_elem_size != std::mem::size_of::<i32>() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "string offsets child must use 32-bit offsets",
            )
            .with_context("operation", operation)
            .with_context("offsets_elem_size", offsets_elem_size.to_string()));
        }
        let offsets_required_elements = offset.checked_add(len).and_then(|v| v.checked_add(1)).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "string offsets range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
        })?;
        let offsets_byte_len = offsets_required_elements.checked_mul(offsets_elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "string offsets byte range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
        })?;
        let offsets_ptr = offsets_view
            .device_data_ptr()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_device = pointer_device(operation, offsets_ptr, default_device)?;

        let (chars_ptr, chars_byte_len) = view
            .string_chars_ptr_and_size()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let chars_device = pointer_device(operation, chars_ptr, default_device)?;

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = vec![
            BufferBinding::new(
                offsets_id,
                bound_bytes(operation, offsets_ptr, offsets_byte_len)?,
                Placement::CudaDevice { device: offsets_device },
            ),
            BufferBinding::new(
                data_id,
                bound_bytes(operation, chars_ptr, chars_byte_len)?,
                Placement::CudaDevice { device: chars_device },
            ),
        ];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            SchemaNullability::Unknown,
            ColumnLayout::BinaryLike { offsets: offsets_id, data: data_id },
        );
        return Ok(PreparedColumnProjection {
            column: column.clone(),
            semantic_columns: vec![column],
            bindings,
            next_buffer_id,
            next_column_id,
        });
    }

    if map_data_type(data_type).id() == InteropTypeId::Dictionary32 {
        let mut next_buffer_id = next_buffer_id;
        let indices_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        if num_children < 2 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "dictionary column view must expose indices and keys columns",
            )
            .with_context("operation", operation)
            .with_context("type_id", "dictionary32")
            .with_context("num_children", num_children.to_string()));
        }

        let indices_view = dictionary::indices(view)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_type = indices_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_elem_size = map_data_type(indices_type).id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "dictionary indices child must be fixed-width")
                .with_context("operation", operation)
                .with_context("indices_type_id", format!("{:?}", indices_type.id()).to_lowercase())
        })?;
        if indices_elem_size != std::mem::size_of::<i32>() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "dictionary indices child must use 32-bit indices",
            )
            .with_context("operation", operation)
            .with_context("indices_elem_size", indices_elem_size.to_string()));
        }
        let indices_byte_len = active_elements.checked_mul(indices_elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "dictionary indices byte range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
        })?;
        let indices_ptr = indices_view
            .device_data_ptr()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_device = pointer_device(operation, indices_ptr, default_device)?;

        let keys_view =
            dictionary::keys(view).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let keys_projection = prepared_column_projection_from_view(
            &keys_view,
            operation,
            default_device,
            next_column_id,
            next_buffer_id,
        )?;
        next_buffer_id = keys_projection.next_buffer_id;
        next_column_id = keys_projection.next_column_id;

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = vec![BufferBinding::new(
            indices_id,
            bound_bytes(operation, indices_ptr, indices_byte_len)?,
            Placement::CudaDevice { device: indices_device },
        )];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        bindings.extend(keys_projection.bindings);

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            SchemaNullability::Unknown,
            ColumnLayout::Dictionary { indices: indices_id, keys: keys_projection.column.id() },
        );
        let mut semantic_columns = vec![column.clone()];
        semantic_columns.extend(keys_projection.semantic_columns);
        return Ok(PreparedColumnProjection { column, semantic_columns, bindings, next_buffer_id, next_column_id });
    }

    if map_data_type(data_type).id() == InteropTypeId::List {
        let mut next_buffer_id = next_buffer_id;
        let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        if num_children < 2 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "list column view must expose offsets and child columns",
            )
            .with_context("operation", operation)
            .with_context("type_id", "list")
            .with_context("num_children", num_children.to_string()));
        }

        let offsets_view =
            view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_type = offsets_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_elem_size = map_data_type(offsets_type).id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "list offsets child must be fixed-width")
                .with_context("operation", operation)
                .with_context("offsets_type_id", format!("{:?}", offsets_type.id()).to_lowercase())
        })?;
        if offsets_elem_size != std::mem::size_of::<i32>() {
            return Err(InteropError::new(InteropErrorKind::InvalidView, "list offsets child must use 32-bit offsets")
                .with_context("operation", operation)
                .with_context("offsets_elem_size", offsets_elem_size.to_string()));
        }
        let offsets_required_elements = offset.checked_add(len).and_then(|v| v.checked_add(1)).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "list offsets range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
        })?;
        let offsets_byte_len = offsets_required_elements.checked_mul(offsets_elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "list offsets byte range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
        })?;
        let offsets_ptr = offsets_view
            .device_data_ptr()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_device = pointer_device(operation, offsets_ptr, default_device)?;

        let child_view = view
            .list_sliced_child()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let child_projection = prepared_column_projection_from_view(
            &child_view,
            operation,
            default_device,
            next_column_id,
            next_buffer_id,
        )?;
        next_buffer_id = child_projection.next_buffer_id;
        next_column_id = child_projection.next_column_id;

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = vec![BufferBinding::new(
            offsets_id,
            bound_bytes(operation, offsets_ptr, offsets_byte_len)?,
            Placement::CudaDevice { device: offsets_device },
        )];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        bindings.extend(child_projection.bindings);

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            SchemaNullability::Unknown,
            ColumnLayout::List { offsets: offsets_id, child: child_projection.column.id() },
        );
        let mut semantic_columns = vec![column.clone()];
        semantic_columns.extend(child_projection.semantic_columns);
        return Ok(PreparedColumnProjection { column, semantic_columns, bindings, next_buffer_id, next_column_id });
    }

    if map_data_type(data_type).id() == InteropTypeId::Struct {
        let mut next_buffer_id = next_buffer_id;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let child_count = usize::try_from(num_children).map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "struct child count must be >= 0")
                .with_context("operation", operation)
                .with_context("type_id", "struct")
                .with_context("num_children", num_children.to_string())
        })?;
        if child_count == 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "struct column view must expose at least one child column",
            )
            .with_context("operation", operation)
            .with_context("type_id", "struct"));
        }

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = Vec::new();
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        let mut child_ids = Vec::with_capacity(child_count);
        let mut semantic_columns = Vec::new();

        for index in 0..child_count {
            let child_index = checked_child_index(operation, index)?;
            let child_view = view
                .child(child_index)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let child_projection = prepared_column_projection_from_view(
                &child_view,
                operation,
                default_device,
                next_column_id,
                next_buffer_id,
            )?;
            next_buffer_id = child_projection.next_buffer_id;
            next_column_id = child_projection.next_column_id;
            child_ids.push(child_projection.column.id());
            bindings.extend(child_projection.bindings);
            semantic_columns.extend(child_projection.semantic_columns);
        }

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            SchemaNullability::Unknown,
            ColumnLayout::Struct { children: child_ids },
        );
        let mut all_columns = vec![column.clone()];
        all_columns.extend(semantic_columns);
        return Ok(PreparedColumnProjection {
            column,
            semantic_columns: all_columns,
            bindings,
            next_buffer_id,
            next_column_id,
        });
    }

    prepared_column_projection_from_materialized_view(
        operation,
        data_type,
        num_children,
        len,
        offset,
        data_ptr,
        nullable,
        null_mask_ptr,
        null_count,
        default_device,
        column_id.get(),
        next_buffer_id,
    )
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
        to_static_bindings(operation, projection.bindings)?,
        to_static_readiness(readiness),
        owned_transfer_diagnostics(),
        owned_transfer_release_contract(default_device),
        OwnedColumnTransferState { _column: column },
    )
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_table_export_from_view<'a>(
    view: &'a TableView<'_>,
    domain: &Arc<ExecutionDomain>,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'a>, InteropError> {
    let num_rows = usize::try_from(
        view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
            .with_context("operation", operation)
    })?;
    let num_columns = usize::try_from(
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
            .with_context("operation", operation)
    })?;
    let (default_device, readiness) = prepared_device_readiness(operation, domain)?;
    prepare_table_export_from_view_with_readiness(
        view,
        names,
        operation,
        num_rows,
        num_columns,
        default_device,
        readiness,
    )
}

fn prepare_host_table_export_from_view(
    view: &TableView<'_>,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let num_rows = usize::try_from(
        view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
            .with_context("operation", operation)
    })?;
    let num_columns = usize::try_from(
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
            .with_context("operation", operation)
    })?;

    let mut columns = Vec::with_capacity(num_columns);
    let mut semantic_columns = Vec::new();
    let mut bindings = Vec::new();
    let mut retained_host_buffers = Vec::new();
    let mut next_buffer_id = 0u32;
    let mut next_column_id = 0u32;

    for index in 0..num_columns {
        let index_i32 = checked_column_index(operation, index)?;
        let column_view =
            view.column(index_i32).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection =
            prepared_host_column_projection_from_view(&column_view, operation, next_column_id, next_buffer_id)?;
        next_buffer_id = projection.next_buffer_id;
        next_column_id = projection.next_column_id;
        bindings.extend(projection.bindings);
        retained_host_buffers.extend(projection.retained_host_buffers);
        columns.push(projection.column.clone());
        semantic_columns.extend(projection.semantic_columns);
    }

    let table =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        columns,
        semantic_columns,
        bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(retained_host_buffers))
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_table_export_from_view_with_readiness<'a>(
    view: &'a TableView<'_>,
    names: Option<Vec<String>>,
    operation: &'static str,
    num_rows: usize,
    num_columns: usize,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
) -> Result<PreparedTableExport<'a>, InteropError> {
    let mut columns = Vec::with_capacity(num_columns);
    let mut semantic_columns = Vec::new();
    let mut bindings = Vec::new();
    let mut next_buffer_id = 0u32;
    let mut next_column_id = 0u32;
    for index in 0..num_columns {
        let index_i32 = checked_column_index(operation, index)?;
        let column_view =
            view.column(index_i32).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection = prepared_column_projection_from_view(
            &column_view,
            operation,
            default_device,
            next_column_id,
            next_buffer_id,
        )?;
        next_buffer_id = projection.next_buffer_id;
        next_column_id = projection.next_column_id;
        bindings.extend(projection.bindings);
        columns.push(projection.column.clone());
        semantic_columns.extend(projection.semantic_columns);
    }
    let table =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        columns,
        semantic_columns,
        bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
    .map_err(|error| error.with_context("operation", operation))
}

fn transfer_from_owned_table(table: Table, operation: &'static str) -> Result<OwnedTableTransfer, InteropError> {
    let num_rows = {
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        usize::try_from(
            view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
                .with_context("operation", operation)
        })?
    };
    let num_columns = {
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        usize::try_from(
            view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
                .with_context("operation", operation)
        })?
    };
    let (default_device, readiness) = prepared_device_readiness(operation, table.execution_domain())?;
    let (columns, semantic_columns, bindings) = {
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let mut columns = Vec::with_capacity(num_columns);
        let mut semantic_columns = Vec::new();
        let mut bindings = Vec::new();
        let mut next_buffer_id = 0u32;
        let mut next_column_id = 0u32;
        for index in 0..num_columns {
            let index_i32 = checked_column_index(operation, index)?;
            let column_view = view
                .column(index_i32)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let projection = prepared_column_projection_from_view(
                &column_view,
                operation,
                default_device,
                next_column_id,
                next_buffer_id,
            )?;
            next_buffer_id = projection.next_buffer_id;
            next_column_id = projection.next_column_id;
            bindings.extend(projection.bindings);
            columns.push(projection.column.clone());
            semantic_columns.extend(projection.semantic_columns);
        }
        (columns, semantic_columns, bindings)
    };

    let table_semantic =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), None);
    OwnedTableTransfer::try_new_with_semantic_columns(
        table_semantic,
        columns,
        semantic_columns,
        to_static_bindings(operation, bindings)?,
        to_static_readiness(readiness),
        owned_transfer_diagnostics(),
        owned_transfer_release_contract(default_device),
        OwnedTableTransferState { _table: table },
    )
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_partitioned_device_table_export_from_columns<'a>(
    producer: &'a PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<PartitionedPreparedTableExport<'a>, InteropError> {
    let mut partitions = Vec::new();

    for group in producer.grouped_borrowed_columns() {
        let (default_device, readiness) = prepared_device_readiness(operation, &group.domain)?;
        let mut columns = Vec::with_capacity(group.columns.len());
        let mut semantic_columns = Vec::new();
        let mut bindings = Vec::new();
        let mut next_buffer_id = 0u32;
        let mut next_column_id = 0u32;

        for column in group.columns {
            let view =
                column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let projection =
                prepared_column_projection_from_view(&view, operation, default_device, next_column_id, next_buffer_id)?;
            next_buffer_id = projection.next_buffer_id;
            next_column_id = projection.next_column_id;
            bindings.extend(projection.bindings);
            columns.push(projection.column.clone());
            semantic_columns.extend(projection.semantic_columns);
        }

        let table = SemanticTable::new(
            TableId::new(0),
            producer.num_rows(),
            columns.iter().map(|column| column.id()).collect(),
            None,
        );
        let export = PreparedTableExport::try_new_with_semantic_columns(
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            export_diagnostics(),
        )
        .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
        .map_err(|error| error.with_context("operation", operation))?;

        partitions.push(PartitionedPreparedTablePartition::new(group.top_level_column_indices, export));
    }

    PartitionedPreparedTableExport::try_new(Some(producer.column_names_owned()), producer.num_rows(), partitions)
        .map_err(|error| error.with_context("operation", operation))
}

fn prepare_partitioned_host_table_export_from_columns(
    producer: &PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let mut columns = Vec::with_capacity(producer.columns.len());
    let mut semantic_columns = Vec::new();
    let mut bindings = Vec::new();
    let mut retained_host_buffers = Vec::new();
    let mut next_buffer_id = 0u32;
    let mut next_column_id = 0u32;

    for column in producer.columns.values() {
        let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection = prepared_host_column_projection_from_view(&view, operation, next_column_id, next_buffer_id)?;
        next_buffer_id = projection.next_buffer_id;
        next_column_id = projection.next_column_id;
        bindings.extend(projection.bindings);
        retained_host_buffers.extend(projection.retained_host_buffers);
        columns.push(projection.column.clone());
        semantic_columns.extend(projection.semantic_columns);
    }

    let table = SemanticTable::new(
        TableId::new(0),
        producer.num_rows(),
        columns.iter().map(|column| column.id()).collect(),
        Some(producer.column_names_owned()),
    );
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        columns,
        semantic_columns,
        bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(retained_host_buffers))
    .map_err(|error| error.with_context("operation", operation))
}

fn transfer_partitioned_owned_table_from_columns(
    producer: PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<PartitionedOwnedTableTransfer, InteropError> {
    let schema_names = producer.column_names_owned();
    let row_count = producer.num_rows();
    let mut partitions = Vec::new();

    for group in producer.grouped_owned_columns() {
        let table = Table::from_columns(group.columns)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let transfer = transfer_from_owned_table(table, operation)?;
        partitions.push(PartitionedOwnedTableTransferPartition::new(group.top_level_column_indices, transfer));
    }

    PartitionedOwnedTableTransfer::try_new(Some(schema_names), row_count, partitions)
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
    /// Project this owned low-level column into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
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
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_table_export_from_view_with_readiness(
            self,
            None,
            operation,
            num_rows,
            num_columns,
            default_device,
            readiness,
        )
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
    /// Project this owned low-level table into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
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
        let num_rows = usize::try_from(
            view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
                .with_context("operation", operation)
        })?;
        let num_columns = usize::try_from(
            view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
                .with_context("operation", operation)
        })?;
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        let mut columns = Vec::with_capacity(num_columns);
        let mut semantic_columns = Vec::new();
        let mut bindings = Vec::new();
        let mut next_buffer_id = 0u32;
        let mut next_column_id = 0u32;
        for index in 0..num_columns {
            let index_i32 = checked_column_index(operation, index)?;
            let column_view = view
                .column(index_i32)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let projection = prepared_column_projection_from_view(
                &column_view,
                operation,
                default_device,
                next_column_id,
                next_buffer_id,
            )?;
            next_buffer_id = projection.next_buffer_id;
            next_column_id = projection.next_column_id;
            bindings.extend(projection.bindings);
            columns.push(projection.column.clone());
            semantic_columns.extend(projection.semantic_columns);
        }
        let table =
            SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), None);
        PreparedTableExport::try_new_with_semantic_columns(
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            export_diagnostics(),
        )
        .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
        .map_err(|error| error.with_context("operation", operation))
    }
}

impl TryPrepareDeviceTableExport for Table {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        let operation = "interop::Table::try_prepare_device_table_export";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let num_rows = usize::try_from(
            view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
                .with_context("operation", operation)
        })?;
        let num_columns = usize::try_from(
            view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
                .with_context("operation", operation)
        })?;
        let (default_device, readiness) = prepared_device_readiness(operation, self.execution_domain())?;
        let mut columns = Vec::with_capacity(num_columns);
        let mut semantic_columns = Vec::new();
        let mut bindings = Vec::new();
        let mut next_buffer_id = 0u32;
        let mut next_column_id = 0u32;
        for index in 0..num_columns {
            let index_i32 = checked_column_index(operation, index)?;
            let column_view = view
                .column(index_i32)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let projection = prepared_column_projection_from_view(
                &column_view,
                operation,
                default_device,
                next_column_id,
                next_buffer_id,
            )?;
            next_buffer_id = projection.next_buffer_id;
            next_column_id = projection.next_column_id;
            bindings.extend(projection.bindings);
            columns.push(projection.column.clone());
            semantic_columns.extend(projection.semantic_columns);
        }
        let table =
            SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), None);
        PreparedTableExport::try_new_with_semantic_columns(
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            export_diagnostics(),
        )
        .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
        .map_err(|error| error.with_context("operation", operation))
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
        let mut columns = Vec::with_capacity(self.columns_map().len());
        let mut semantic_columns = Vec::new();
        let mut bindings = Vec::new();
        let mut next_buffer_id = 0u32;
        let mut next_column_id = 0u32;
        for (index, column) in self.columns_map().values().enumerate() {
            let _ = checked_column_index(operation, index)?;
            let view =
                column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let projection =
                prepared_column_projection_from_view(&view, operation, default_device, next_column_id, next_buffer_id)?;
            next_buffer_id = projection.next_buffer_id;
            next_column_id = projection.next_column_id;
            bindings.extend(projection.bindings);
            columns.push(projection.column.clone());
            semantic_columns.extend(projection.semantic_columns);
        }
        let table = SemanticTable::new(
            TableId::new(0),
            self.num_rows(),
            columns.iter().map(|column| column.id()).collect(),
            Some(self.column_names_owned()),
        );
        PreparedTableExport::try_new_with_semantic_columns(
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            export_diagnostics(),
        )
        .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
        .map_err(|error| error.with_context("operation", operation))
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
        let operation = "interop::DataFrame::try_prepare_device_table_export";
        let (default_device, readiness) = prepared_device_readiness(operation, self.execution_domain())?;
        let mut columns = Vec::with_capacity(self.columns_map().len());
        let mut semantic_columns = Vec::new();
        let mut bindings = Vec::new();
        let mut next_buffer_id = 0u32;
        let mut next_column_id = 0u32;
        for (index, column) in self.columns_map().values().enumerate() {
            let _ = checked_column_index(operation, index)?;
            let view =
                column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let projection =
                prepared_column_projection_from_view(&view, operation, default_device, next_column_id, next_buffer_id)?;
            next_buffer_id = projection.next_buffer_id;
            next_column_id = projection.next_column_id;
            bindings.extend(projection.bindings);
            columns.push(projection.column.clone());
            semantic_columns.extend(projection.semantic_columns);
        }
        let table = SemanticTable::new(
            TableId::new(0),
            self.num_rows(),
            columns.iter().map(|column| column.id()).collect(),
            Some(self.column_names_owned()),
        );
        PreparedTableExport::try_new_with_semantic_columns(
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            export_diagnostics(),
        )
        .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
        .map_err(|error| error.with_context("operation", operation))
    }
}

impl TryPrepareHostTableExport for DataFrame {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        let operation = "interop::DataFrame::try_prepare_host_table_export";
        let (table, _) = self
            .clone()
            .try_into_table()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepare_host_table_export_from_view(&view, Some(self.column_names_owned()), operation)
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
