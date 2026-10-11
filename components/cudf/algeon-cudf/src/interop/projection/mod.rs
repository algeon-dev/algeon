use super::{
    BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, ColumnView, DataType, InteropDataType, InteropError,
    InteropErrorKind, InteropTypeId, PhysicalValidity, Placement, SemanticColumn, TypeId, checked_child_index,
    checked_invalid_view_i32_from_usize, copy_view_to_owned, dictionary, error, invalid_view_contract_error,
    map_cudf_error,
};

mod device;
mod host;
mod support;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) mod testing_hooks;

use support::{
    PreparedColumnProjection, PreparedHostColumnProjection, active_projection_range_overflow_error, allocate_buffer_id,
    allocate_column_id, bound_bytes, encode_host_i32_values, encode_host_string_offsets,
    host_validity_for_prepared_projection, map_data_type, negative_projection_offset_error,
    offsets_range_overflow_error, pointer_device, projection_byte_range_overflow_error, projection_child_count_error,
    projection_child_fixed_width_error, projection_child_width_error, projection_offsets_too_small_error,
    projection_overflow_error, projection_struct_requires_child_error, retained_host_binding,
    validity_for_prepared_projection,
};

pub(super) fn prepared_column_projection_from_view<'a>(
    view: &ColumnView<'a>,
    operation: &'static str,
    default_device: i32,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedColumnProjection<'a>, InteropError> {
    device::prepared_column_projection_from_view(view, operation, default_device, next_column_id, next_buffer_id)
}

pub(super) fn prepared_host_column_projection_from_view(
    view: &ColumnView<'_>,
    operation: &'static str,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedHostColumnProjection, InteropError> {
    host::prepared_host_column_projection_from_view(view, operation, next_column_id, next_buffer_id)
}
