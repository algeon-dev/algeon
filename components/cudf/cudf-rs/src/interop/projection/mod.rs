use super::*;

mod device;
mod host;
mod support;
pub(crate) mod testing_hooks;

use support::*;

fn schema_nullability(nullable: bool) -> SchemaNullability {
    if nullable { SchemaNullability::Nullable } else { SchemaNullability::NonNullable }
}

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
