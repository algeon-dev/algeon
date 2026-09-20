//! Consumer-side import adapters for reconstructing owned `Column`, `Table`,
//! `Series`, and `DataFrame` values from shared interop roots.

use std::sync::Arc;

use rapids_interop::{
    CompletionAuthority, InteropError, InteropErrorKind, LeaseCompletion, PreparedAccess, PreparedColumnExport,
    PreparedTableExport, ReadStartAuthority,
};

use super::{
    import_graph::{checked_i32_from_usize, prepared_device_import_column_graph, prepared_device_import_table_graph},
    map_cudf_error, map_interop_error,
};
use crate::{
    core::{column::Column, error, table::Table},
    frame::{dataframe::DataFrame, series::Series},
    runtime::{Runtime, execution_domain::EnteredExecution},
};

fn validate_runtime_device(
    operation: &'static str,
    message: &'static str,
    expected_device: i32,
    runtime: &Runtime,
) -> Result<(), InteropError> {
    let actual_device = runtime.device().ordinal() as i32;
    if actual_device != expected_device {
        return Err(InteropError::new(InteropErrorKind::ExecutionBindingMismatch, message)
            .with_context("operation", operation)
            .with_context("left_device", expected_device.to_string())
            .with_context("right_device", actual_device.to_string()));
    }
    Ok(())
}

fn missing_dataframe_import_names_error(operation: &'static str) -> InteropError {
    InteropError::new(
        InteropErrorKind::InvalidView,
        "high-level DataFrame import requires schema names on the shared root or in the names parameter",
    )
    .with_context("operation", operation)
    .with_context("resource", "schema_names")
    .with_context("reason", "missing_schema_names")
    .with_context("remedy", "pass Some(names) when the root does not carry schema names")
}

fn runtime_exec_options(runtime: &Runtime, operation: &'static str) -> Result<EnteredExecution, InteropError> {
    runtime
        .execution_domain()
        .enter_execution()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))
}

/// Imports copy producer storage on the consumer stream and return an owned
/// value, so the copy must finish before the import returns.
fn synchronize_lease_completion(
    completion: Result<LeaseCompletion, InteropError>,
    operation: &'static str,
) -> Result<(), InteropError> {
    completion
        .and_then(|mut completion| completion.synchronize())
        .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

fn runtime_shared_stream_authorities<'a>(
    runtime: &'a Runtime,
    operation: &'static str,
) -> Result<(ReadStartAuthority<'a>, CompletionAuthority<'a>), InteropError> {
    let stream =
        super::cuda_stream_handle(runtime.stream()).map_err(|error| error.with_context("operation", operation))?;
    Ok((ReadStartAuthority::ConsumerStream { stream }, CompletionAuthority::SameDeviceStream { stream }))
}

pub(super) fn import_column_from_prepared_export(
    export: &PreparedColumnExport<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Column, InteropError> {
    let graph = prepared_device_import_column_graph(export, operation)?;
    let exec_options = runtime_exec_options(runtime, operation)?;
    let num_semantic_columns = checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)?;
    let num_bindings = checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)?;

    match export.access() {
        PreparedAccess::Device { readiness, .. } => {
            validate_runtime_device(
                operation,
                "prepared device export import requires the destination runtime to use the same CUDA device",
                readiness.device(),
                runtime,
            )?;
            let (read_start, completion) = runtime_shared_stream_authorities(runtime, operation)?;
            let lease = export.try_activate_read_lease(read_start, completion)?;
            let mut handle = std::ptr::null_mut();

            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_column_copy_from_prepared_device_export_ex(
                        export.column().id().get(),
                        graph.ffi_columns.as_ptr(),
                        num_semantic_columns,
                        graph.ffi_bindings.as_ptr(),
                        num_bindings,
                        exec_options.options(),
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            synchronize_lease_completion(lease.complete(), operation)?;
            Ok(imported)
        },
        PreparedAccess::Host => {
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
                        exec_options.options(),
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            synchronize_lease_completion(lease.complete(), operation)?;
            Ok(imported)
        },
    }
}

pub(super) fn import_table_from_prepared_export(
    export: &PreparedTableExport<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Table, InteropError> {
    let graph = prepared_device_import_table_graph(export, operation)?;
    let exec_options = runtime_exec_options(runtime, operation)?;
    let num_semantic_columns = checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)?;
    let num_bindings = checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)?;

    match export.access() {
        PreparedAccess::Device { readiness, .. } => {
            validate_runtime_device(
                operation,
                "prepared device export import requires the destination runtime to use the same CUDA device",
                readiness.device(),
                runtime,
            )?;
            let (read_start, completion) = runtime_shared_stream_authorities(runtime, operation)?;
            let lease = export.try_activate_read_lease(read_start, completion)?;
            let mut handle = std::ptr::null_mut();

            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_table_copy_from_prepared_device_export_ex(
                        &graph.ffi_table,
                        graph.ffi_columns.as_ptr(),
                        num_semantic_columns,
                        graph.ffi_bindings.as_ptr(),
                        num_bindings,
                        exec_options.options(),
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Table::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            synchronize_lease_completion(lease.complete(), operation)?;
            Ok(imported)
        },
        PreparedAccess::Host => {
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
                        exec_options.options(),
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Table::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            synchronize_lease_completion(lease.complete(), operation)?;
            Ok(imported)
        },
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

pub(super) fn dataframe_import_names(
    names: Option<Vec<String>>,
    root_names: Option<&[String]>,
    operation: &'static str,
) -> Result<Vec<String>, InteropError> {
    names.or_else(|| root_names.map(ToOwned::to_owned)).ok_or_else(|| missing_dataframe_import_names_error(operation))
}
