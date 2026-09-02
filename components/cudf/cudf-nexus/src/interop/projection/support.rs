use std::sync::Arc;

use super::*;

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

pub(super) fn map_data_type(data_type: DataType) -> InteropDataType {
    if data_type.id().is_decimal() {
        InteropDataType::fixed_point(map_type_id(data_type.id()), data_type.scale())
    } else {
        InteropDataType::new(map_type_id(data_type.id()))
    }
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

pub(super) fn projection_overflow_error(
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    reason: &'static str,
    offset: usize,
    row_count: usize,
    remedy: &'static str,
) -> InteropError {
    invalid_view_contract_error(operation, message, resource, reason, remedy)
        .with_context("offset", offset.to_string())
        .with_context("row_count", row_count.to_string())
}

pub(super) fn negative_projection_offset_error(operation: &'static str, offset: i64) -> InteropError {
    invalid_view_contract_error(
        operation,
        "column view offset must be >= 0",
        "offset",
        "negative_value",
        "normalize the view so offset >= 0 before projection",
    )
    .with_context("offset", offset.to_string())
}

pub(super) fn projection_child_count_error(
    operation: &'static str,
    message: &'static str,
    type_id: &'static str,
    child_count: usize,
    required_child_count: usize,
    reason: &'static str,
    remedy: &'static str,
) -> InteropError {
    invalid_view_contract_error(operation, message, "child_count", reason, remedy)
        .with_context("type_id", type_id)
        .with_context("child_count", child_count.to_string())
        .with_context("required_child_count", required_child_count.to_string())
}

pub(super) fn projection_child_fixed_width_error(
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    type_id: String,
    remedy: &'static str,
) -> InteropError {
    invalid_view_contract_error(operation, message, resource, "requires_fixed_width_child", remedy)
        .with_context("type_id", type_id)
}

pub(super) fn projection_child_width_error(
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    actual_width_bytes: usize,
    required_width_bytes: usize,
    remedy: &'static str,
) -> InteropError {
    invalid_view_contract_error(operation, message, resource, "requires_i32_width", remedy)
        .with_context("actual_width_bytes", actual_width_bytes.to_string())
        .with_context("required_width_bytes", required_width_bytes.to_string())
}

pub(super) fn projection_offsets_too_small_error(
    operation: &'static str,
    message: &'static str,
    offset: usize,
    row_count: usize,
    offset_count: usize,
) -> InteropError {
    invalid_view_contract_error(
        operation,
        message,
        "offsets_range",
        "too_small_for_active_range",
        "expose at least offset + row_count + 1 offsets for the active logical range",
    )
    .with_context("offset", offset.to_string())
    .with_context("row_count", row_count.to_string())
    .with_context("offset_count", offset_count.to_string())
}

pub(super) fn projection_missing_null_mask_error(operation: &'static str, row_count: usize) -> InteropError {
    invalid_view_contract_error(
        operation,
        "nullable column did not expose a null-mask pointer",
        "null_mask_buffer",
        "missing_for_nullable_column",
        "expose a null_mask_buffer for nullable projection inputs or mark the column non-nullable",
    )
    .with_context("row_count", row_count.to_string())
}

pub(super) fn projection_struct_requires_child_error(operation: &'static str) -> InteropError {
    projection_child_count_error(
        operation,
        "struct column view must expose at least one child column",
        "struct",
        0,
        1,
        "requires_child_column",
        "expose at least one child column before projecting struct data",
    )
}

pub(super) fn active_projection_range_overflow_error(
    operation: &'static str,
    offset: usize,
    row_count: usize,
) -> InteropError {
    projection_overflow_error(
        operation,
        "active element range overflowed",
        "row_range",
        "end_overflowed_usize",
        offset,
        row_count,
        "reduce offset + row_count so the active projection range fits within usize",
    )
}

pub(super) fn offsets_range_overflow_error(
    operation: &'static str,
    message: &'static str,
    offset: usize,
    row_count: usize,
) -> InteropError {
    projection_overflow_error(
        operation,
        message,
        "offsets_range",
        "end_overflowed_usize",
        offset,
        row_count,
        "reduce offset + row_count so the projected offsets range fits within usize",
    )
}

pub(super) fn projection_byte_range_overflow_error(
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    offset: usize,
    row_count: usize,
) -> InteropError {
    projection_overflow_error(
        operation,
        message,
        resource,
        "size_overflowed_usize",
        offset,
        row_count,
        "reduce row_count or the projected byte width so the active byte range fits within usize",
    )
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

pub(in crate::interop) struct PreparedColumnProjection<'a> {
    pub(in crate::interop) column: SemanticColumn,
    pub(in crate::interop) semantic_columns: Vec<SemanticColumn>,
    pub(in crate::interop) bindings: Vec<BufferBinding<'a>>,
    pub(in crate::interop) next_buffer_id: u32,
    pub(in crate::interop) next_column_id: u32,
}

pub(in crate::interop) struct PreparedHostColumnProjection {
    pub(in crate::interop) column: SemanticColumn,
    pub(in crate::interop) semantic_columns: Vec<SemanticColumn>,
    pub(in crate::interop) bindings: Vec<BufferBinding<'static>>,
    pub(in crate::interop) retained_host_buffers: Vec<Arc<[u8]>>,
    pub(in crate::interop) next_buffer_id: u32,
    pub(in crate::interop) next_column_id: u32,
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
            return Err(projection_missing_null_mask_error(operation, active_elements));
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

pub(super) struct PreparedHostValidity {
    validity: PhysicalValidity,
    retained_binding: Option<RetainedHostValidityBinding>,
}

struct RetainedHostValidityBinding {
    binding: BufferBinding<'static>,
    retained_buffer: Arc<[u8]>,
}

impl PreparedHostValidity {
    pub(super) fn retain_into(
        self,
        bindings: &mut Vec<BufferBinding<'static>>,
        retained_host_buffers: &mut Vec<Arc<[u8]>>,
    ) -> PhysicalValidity {
        let Self { validity, retained_binding } = self;
        if let Some(retained_binding) = retained_binding {
            bindings.push(retained_binding.binding);
            retained_host_buffers.push(retained_binding.retained_buffer);
        }
        validity
    }
}

pub(super) fn host_validity_for_prepared_projection(
    operation: &'static str,
    validity: Option<Arc<[u8]>>,
    null_count: usize,
    validity_id: BufferId,
) -> Result<PreparedHostValidity, InteropError> {
    let Some(mask_bytes) = validity else {
        return Ok(PreparedHostValidity { validity: PhysicalValidity::NoValidityBuffer, retained_binding: None });
    };

    let binding = retained_host_binding(operation, validity_id, &mask_bytes)?;
    Ok(PreparedHostValidity {
        validity: PhysicalValidity::ValidityMask { mask: validity_id, null_count: Some(null_count) },
        retained_binding: Some(RetainedHostValidityBinding { binding, retained_buffer: mask_bytes }),
    })
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
            invalid_view_contract_error(
                operation,
                "string host export byte range overflowed",
                "byte_len",
                "size_overflowed_usize",
                "reduce total host-export string bytes so the active byte range fits within usize",
            )
        })?;
        let offset = checked_invalid_view_i32_from_usize("byte_len", total_bytes, operation)?;
        offsets.push(offset);
    }

    let bytes: Vec<u8> = offsets.iter().flat_map(|value| value.to_ne_bytes()).collect();
    Ok(Arc::from(bytes))
}

pub(super) fn encode_host_i32_values(values: &[i32]) -> Arc<[u8]> {
    Arc::from(values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<u8>>())
}
