use super::*;

pub(super) fn map_type_id(type_id: TypeId) -> InteropTypeId {
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

pub(super) fn map_interop_type_id(type_id: InteropTypeId) -> TypeId {
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

pub(super) fn map_data_type(data_type: DataType) -> InteropDataType {
    if data_type.id().is_decimal() {
        InteropDataType::fixed_point(map_type_id(data_type.id()), data_type.scale())
    } else {
        InteropDataType::new(map_type_id(data_type.id()))
    }
}

pub(super) fn map_cudf_error(kind: InteropErrorKind, operation: &'static str, error: crate::Error) -> InteropError {
    let mut interop = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        interop = interop.with_context(key, value.clone());
    }
    interop
}

pub(super) fn map_interop_error(kind: InteropErrorKind, operation: &'static str, error: InteropError) -> InteropError {
    let mut mapped = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }
    mapped
}

pub(super) fn execution_binding<'a>(
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
            let stream_ptr = stream
                .interop_raw_handle()
                .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
            let mr_ptr = mr
                .interop_raw_handle()
                .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
            Ok(ExecutionBinding::explicit(
                options.device,
                CudaStreamHandle::from_raw_on_device(stream_ptr, options.device)?,
                DeviceMemoryResourceHandle::from_raw(mr_ptr)?,
            ))
        },
    }
}

pub(super) fn prepared_device_readiness<'a>(
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

pub(super) fn prepared_device_event_readiness<'a>(
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

pub(super) fn export_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedDeviceExport)
}

pub(super) fn host_export_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedHostExport)
}

pub(super) fn owned_transfer_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::OwnedDeviceTransfer)
}

pub(super) fn owned_transfer_release_contract(device: i32) -> ReleaseContract {
    ReleaseContract::new(
        device,
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
        ReleaseSynchronization::MaySynchronizeInternally,
    )
}

pub(super) fn owned_transfer_runtime_release_contract(device: i32) -> ReleaseContract {
    ReleaseContract::new(
        device,
        TransferCompletionRule::ReleaseWaitsOnConsumerSignal,
        ReleaseSynchronization::MaySynchronizeInternally,
    )
}

pub(super) fn pointer_device(
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

pub(super) fn bound_bytes<'a>(
    operation: &'static str,
    ptr: *const std::ffi::c_void,
    byte_len: usize,
) -> Result<BoundBytes<'a>, InteropError> {
    BoundBytes::from_raw_parts(ptr as *mut std::ffi::c_void, byte_len)
        .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn retained_host_binding(
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

pub(super) struct PreparedColumnProjection<'a> {
    pub(super) column: SemanticColumn,
    pub(super) semantic_columns: Vec<SemanticColumn>,
    pub(super) bindings: Vec<BufferBinding<'a>>,
    pub(super) next_buffer_id: u32,
    pub(super) next_column_id: u32,
}

pub(super) struct PreparedHostColumnProjection {
    pub(super) column: SemanticColumn,
    pub(super) semantic_columns: Vec<SemanticColumn>,
    pub(super) bindings: Vec<BufferBinding<'static>>,
    pub(super) retained_host_buffers: Vec<Arc<[u8]>>,
    pub(super) next_buffer_id: u32,
    pub(super) next_column_id: u32,
}

pub(super) struct PreparedDeviceImportColumnGraph {
    pub(super) ffi_columns: Vec<cudf_sys::cudf_interop_semantic_column_t>,
    pub(super) ffi_bindings: Vec<cudf_sys::cudf_interop_buffer_binding_t>,
    pub(super) _child_id_buffers: Vec<Box<[u32]>>,
}

pub(super) struct PreparedDeviceImportTableGraph {
    pub(super) ffi_table: cudf_sys::cudf_interop_semantic_table_t,
    pub(super) ffi_columns: Vec<cudf_sys::cudf_interop_semantic_column_t>,
    pub(super) ffi_bindings: Vec<cudf_sys::cudf_interop_buffer_binding_t>,
    pub(super) _child_id_buffers: Vec<Box<[u32]>>,
    pub(super) _table_column_ids: Box<[u32]>,
}

pub(super) fn allocate_buffer_id(operation: &'static str, next_buffer_id: &mut u32) -> Result<BufferId, InteropError> {
    let id = BufferId::new(*next_buffer_id);
    *next_buffer_id = next_buffer_id.checked_add(1).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "buffer id allocation overflowed")
            .with_context("operation", operation)
            .with_context("next_buffer_id", id.get().to_string())
    })?;
    Ok(id)
}

pub(super) fn allocate_column_id(operation: &'static str, next_column_id: &mut u32) -> Result<ColumnId, InteropError> {
    let id = ColumnId::new(*next_column_id);
    *next_column_id = next_column_id.checked_add(1).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "column id allocation overflowed")
            .with_context("operation", operation)
            .with_context("next_column_id", id.get().to_string())
    })?;
    Ok(id)
}

pub(super) fn validity_for_prepared_projection<'a>(
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

pub(super) fn host_validity_for_prepared_projection(
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

pub(super) fn encode_host_string_offsets(
    operation: &'static str,
    strings: &[String],
) -> Result<Arc<[u8]>, InteropError> {
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

pub(super) fn encode_host_i32_values(values: &[i32]) -> Arc<[u8]> {
    Arc::from(values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<u8>>())
}

pub(super) fn checked_column_index(operation: &'static str, index: usize) -> Result<i32, InteropError> {
    let index_i32 = i32::try_from(index).map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column index exceeds i32 range")
            .with_context("operation", operation)
            .with_context("index", index.to_string())
    })?;
    Ok(index_i32)
}

pub(super) fn checked_child_index(operation: &'static str, index: usize) -> Result<i32, InteropError> {
    let index_i32 = i32::try_from(index).map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "child index exceeds i32 range")
            .with_context("operation", operation)
            .with_context("index", index.to_string())
    })?;
    Ok(index_i32)
}

pub(super) fn checked_i32_from_usize(
    field: &'static str,
    value: usize,
    operation: &'static str,
) -> Result<i32, InteropError> {
    i32::try_from(value).map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, format!("{field} exceeds i32::MAX"))
            .with_context("operation", operation)
            .with_context(field, value.to_string())
    })
}
