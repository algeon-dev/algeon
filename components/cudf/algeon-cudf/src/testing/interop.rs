use std::sync::Arc;

use rapids_interop::{
    InteropError, InteropErrorKind, LeaseBackend, PreparedAccess, PreparedColumnExport, PreparedTableExport,
};

use crate::interop::{
    projection::testing_hooks::{
        trigger_active_range_overflow_for_test as projection_trigger_active_range_overflow_for_test,
        trigger_dictionary_indices_width_mismatch_for_test as projection_trigger_dictionary_indices_width_mismatch_for_test,
        trigger_host_offsets_too_small_for_test as projection_trigger_host_offsets_too_small_for_test,
        trigger_list_offsets_width_mismatch_for_test as projection_trigger_list_offsets_width_mismatch_for_test,
        trigger_missing_null_mask_for_test as projection_trigger_missing_null_mask_for_test,
        trigger_missing_string_offsets_child_for_test as projection_trigger_missing_string_offsets_child_for_test,
        trigger_negative_offset_for_test as projection_trigger_negative_offset_for_test,
        trigger_struct_missing_child_for_test as projection_trigger_struct_missing_child_for_test,
    },
    testing_hooks::trigger_column_index_limit_for_test as interop_trigger_column_index_limit_for_test,
};

/// Trigger the shared interop projection active-range overflow contract used
/// by integration tests.
pub fn trigger_interop_projection_active_range_overflow_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_active_range_overflow_for_test()
}

/// Trigger the shared interop projection negative-offset validation contract.
pub fn trigger_interop_projection_negative_offset_for_test() -> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_negative_offset_for_test()
}

/// Trigger the shared interop projection missing-string-offsets-child
/// contract.
pub fn trigger_interop_projection_missing_string_offsets_child_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_missing_string_offsets_child_for_test()
}

/// Trigger the shared interop projection list-offsets-width contract.
pub fn trigger_interop_projection_list_offsets_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_list_offsets_width_mismatch_for_test()
}

/// Trigger the shared interop projection dictionary-indices-width contract.
pub fn trigger_interop_projection_dictionary_indices_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_dictionary_indices_width_mismatch_for_test()
}

/// Trigger the shared interop projection host-offsets-range contract.
pub fn trigger_interop_projection_host_offsets_too_small_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_host_offsets_too_small_for_test()
}

/// Trigger the shared interop projection missing-null-mask contract.
pub fn trigger_interop_projection_missing_null_mask_for_test() -> std::result::Result<(), rapids_interop::InteropError>
{
    projection_trigger_missing_null_mask_for_test()
}

/// Trigger the shared interop projection struct-child-count contract.
pub fn trigger_interop_projection_struct_missing_child_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_struct_missing_child_for_test()
}

/// Trigger the shared interop invalid-view column-index i32-limit contract.
pub fn trigger_interop_column_index_limit_for_test() -> std::result::Result<(), rapids_interop::InteropError> {
    interop_trigger_column_index_limit_for_test()
}

fn device_access_with_backend<'a>(
    access: &PreparedAccess<'a>,
    backend: LeaseBackend,
) -> std::result::Result<PreparedAccess<'a>, InteropError> {
    match access {
        PreparedAccess::Device { readiness, .. } => Ok(PreparedAccess::Device { readiness: *readiness, backend }),
        PreparedAccess::Host => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "lease backends apply only to prepared device exports",
        )),
    }
}

/// Rebuild a prepared device column export with `backend`, keeping its graph,
/// readiness, and retained owner.
pub fn column_export_with_lease_backend_for_test<'a>(
    export: &PreparedColumnExport<'a>,
    backend: LeaseBackend,
) -> std::result::Result<PreparedColumnExport<'a>, InteropError> {
    let rebuilt = PreparedColumnExport::try_new(
        export.column().id(),
        export.semantic_columns().to_vec(),
        export.bindings().to_vec(),
        device_access_with_backend(export.access(), backend)?,
    )?;
    Ok(match export.retained_owner() {
        Some(owner) => rebuilt.with_retained_owner(Arc::clone(owner)),
        None => rebuilt,
    })
}

/// Rebuild a prepared device table export with `backend`, keeping its graph,
/// readiness, and retained owner.
pub fn table_export_with_lease_backend_for_test<'a>(
    export: &PreparedTableExport<'a>,
    backend: LeaseBackend,
) -> std::result::Result<PreparedTableExport<'a>, InteropError> {
    let rebuilt = PreparedTableExport::try_new(
        export.table().clone(),
        export.semantic_columns().to_vec(),
        export.bindings().to_vec(),
        device_access_with_backend(export.access(), backend)?,
    )?;
    Ok(match export.retained_owner() {
        Some(owner) => rebuilt.with_retained_owner(Arc::clone(owner)),
        None => rebuilt,
    })
}
