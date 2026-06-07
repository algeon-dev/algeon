//! Consumer-side import adapters for reconstructing owned `Column`, `Table`,
//! `Series`, and `DataFrame` values from shared interop roots.

use std::sync::Arc;

use rapids_interop::{
    BorrowedColumnView, BorrowedTableView, BufferBinding, ColumnLayout, CompletionAuthority, ExportProfile,
    InteropError, InteropErrorKind, LeaseCompletionResult, OwnedColumnTransfer, OwnedTableTransfer, PhysicalValidity,
    Placement, PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority, SemanticColumn,
};

use super::{
    OwnedColumnTransferState, OwnedTableTransferState, PreparedDeviceImportColumnGraph, PreparedDeviceImportTableGraph,
    RuntimeCompletionAuthority, RuntimeLeaseCompletionResult, RuntimeReadStartAuthority,
    TryActivateRuntimeColumnReadLease, TryActivateRuntimeTableReadLease, checked_i32_from_usize, map_cudf_error,
    map_interop_error, map_interop_type_id,
};
use crate::{
    core::{column::Column, error, table::Table},
    frame::{dataframe::DataFrame, series::Series},
    runtime::Runtime,
};

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

fn validate_runtime_device(
    operation: &'static str,
    expected_label: &'static str,
    message: &'static str,
    expected_device: i32,
    runtime: &Runtime,
) -> Result<(), InteropError> {
    let actual_device = runtime.device().ordinal() as i32;
    if actual_device != expected_device {
        return Err(InteropError::new(InteropErrorKind::ExecutionBindingMismatch, message)
            .with_context("operation", operation)
            .with_context(expected_label, expected_device.to_string())
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
        data: binding.bytes().base_ptr() as *const _,
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

fn prepared_device_import_column_graph_from_parts(
    semantic_columns: &[SemanticColumn],
    bindings: &[BufferBinding<'_>],
    operation: &'static str,
) -> Result<PreparedDeviceImportColumnGraph, InteropError> {
    let mut child_id_buffers = Vec::new();
    let ffi_columns = semantic_columns
        .iter()
        .map(|column| interop_semantic_column_from_prepared_column(column, operation, &mut child_id_buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let ffi_bindings = bindings.iter().map(interop_binding_from_prepared_binding).collect();
    Ok(PreparedDeviceImportColumnGraph { ffi_columns, ffi_bindings, _child_id_buffers: child_id_buffers })
}

fn prepared_device_import_column_graph(
    export: &PreparedColumnExport<'_>,
    operation: &'static str,
) -> Result<PreparedDeviceImportColumnGraph, InteropError> {
    prepared_device_import_column_graph_from_parts(export.semantic_columns(), export.bindings(), operation)
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
        rapids_interop::NullMaskKind::Absent => Ok((std::ptr::null(), 0)),
        rapids_interop::NullMaskKind::Present { ptr, null_count, .. } => {
            Ok((ptr.as_ptr() as *const std::ffi::c_void, checked_i32_from_usize("null_count", *null_count, operation)?))
        },
    }
}

pub(super) fn import_column_from_borrowed_view(
    borrowed: &BorrowedColumnView<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Column, InteropError> {
    validate_runtime_device(
        operation,
        "descriptor_device",
        "borrowed descriptor import requires the destination runtime to use the same CUDA device",
        borrowed.execution().device(),
        runtime,
    )?;
    let exec_options = runtime
        .execution_domain()
        .explicit_exec_options_raw()
        .expect("runtime always provides explicit execution options");
    let mut handle = std::ptr::null_mut();

    match borrowed.column() {
        Some(root_column) => {
            let graph = prepared_device_import_column_graph_from_parts(
                borrowed.semantic_columns(),
                borrowed.bindings(),
                operation,
            )?;
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_column_copy_from_prepared_device_export_ex(
                        root_column.id().get(),
                        graph.ffi_columns.as_ptr(),
                        checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)?,
                        graph.ffi_bindings.as_ptr(),
                        checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)?,
                        &exec_options,
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        },
        None => {
            let len = checked_i32_from_usize("len", borrowed.len(), operation)?;
            let offset = checked_i32_from_usize("offset", borrowed.offset(), operation)?;
            let (null_mask, null_count) = borrowed_null_mask_parts(borrowed, operation)?;
            let type_id = map_interop_type_id(borrowed.data_type().id());
            let scale = borrowed.data_type().scale();
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
        },
    }

    Ok(unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) })
}

pub(super) fn import_table_from_borrowed_view(
    borrowed: &BorrowedTableView<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Table, InteropError> {
    validate_runtime_device(
        operation,
        "descriptor_device",
        "borrowed descriptor import requires the destination runtime to use the same CUDA device",
        borrowed.execution().device(),
        runtime,
    )?;
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

pub(super) fn import_column_from_owned_transfer(
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

pub(super) fn import_table_from_owned_transfer(
    transfer: OwnedTableTransfer,
    operation: &'static str,
) -> Result<Table, InteropError> {
    const MAX_OWNED_TABLE_TRANSFER_NESTING: usize = 32;

    fn unsupported_owned_table_root(operation: &'static str) -> InteropError {
        InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "cudf-rs owned table import requires a cudf-rs-produced owned device transfer root",
        )
        .with_context("operation", operation)
        .with_context("consumer", "cudf-rs")
        .with_context("profile", "owned_device_transfer")
    }

    let mut nesting = 0usize;
    let mut current = transfer;
    loop {
        match current.try_into_owner::<OwnedTableTransferState>() {
            Ok(state) => return Ok(state.into_table()),
            Err(transfer) => match transfer.try_into_owner::<OwnedTableTransfer>() {
                Ok(inner_transfer) => {
                    nesting += 1;
                    if nesting > MAX_OWNED_TABLE_TRANSFER_NESTING {
                        return Err(InteropError::new(
                            InteropErrorKind::ProfileUnsupported,
                            "cudf-rs owned table import rejected an owned-transfer root with excessive nesting",
                        )
                        .with_context("operation", operation)
                        .with_context("consumer", "cudf-rs")
                        .with_context("profile", "owned_device_transfer")
                        .with_context("max_nesting", MAX_OWNED_TABLE_TRANSFER_NESTING.to_string())
                        .with_context("observed_nesting", nesting.to_string()));
                    }
                    current = inner_transfer;
                },
                Err(_) => return Err(unsupported_owned_table_root(operation)),
            },
        }
    }
}

pub(super) fn import_column_from_prepared_export(
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
            validate_runtime_device(
                operation,
                "producer_device",
                "prepared device export import requires the destination runtime to use the same CUDA device",
                producer_device,
                runtime,
            )?;
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

pub(super) fn import_table_from_prepared_export(
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
            validate_runtime_device(
                operation,
                "producer_device",
                "prepared device export import requires the destination runtime to use the same CUDA device",
                producer_device,
                runtime,
            )?;
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

pub(super) fn wrap_imported_series(column: Column, operation: &'static str) -> Result<Series, InteropError> {
    Series::from_column(column).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))
}

pub(super) fn wrap_imported_dataframe(
    table: Table,
    names: Vec<String>,
    operation: &'static str,
) -> Result<DataFrame, InteropError> {
    DataFrame::from_table(table, names).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))
}

pub(super) fn required_dataframe_import_names(
    names: Option<&[String]>,
    operation: &'static str,
    explicit_variant: &'static str,
) -> Result<Vec<String>, InteropError> {
    names.map(|names| names.to_vec()).ok_or_else(|| {
        InteropError::new(
            InteropErrorKind::InvalidView,
            format!(
                "high-level DataFrame import requires schema names on the shared root; use {explicit_variant} when the root does not carry schema names"
            ),
        )
        .with_context("operation", operation)
        .with_context("explicit_names_variant", explicit_variant)
    })
}
