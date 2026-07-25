//! Consumer-side import adapters for reconstructing owned `Column`, `Table`,
//! `Series`, and `DataFrame` values from shared interop roots.

use std::{any::Any, sync::Arc};

use rapids_interop::{
    BorrowedColumnView, BorrowedTableView, CompletionAuthority, InteropError, InteropErrorKind, LeaseCompletionResult,
    OwnedColumnTransfer, OwnedTableTransfer, PartitionedPreparedTableExport, PreparedColumnExport,
    PreparedExportProfile, PreparedTableExport, ProducerReadiness, ReadStartAuthority,
};

use super::{
    OwnedColumnTransferState, OwnedTableTransferState, RuntimeOwnedColumnTransferOwner, RuntimeOwnedTableTransferOwner,
    import_graph::{
        checked_i32_from_usize, map_interop_type_id, owned_device_import_table_graph,
        prepared_device_import_column_graph, prepared_device_import_column_graph_from_parts,
        prepared_device_import_table_graph,
    },
    map_cudf_error, map_interop_error,
};
use crate::{
    core::{column::Column, error, table::Table},
    frame::{dataframe::DataFrame, series::Series},
    runtime::{CudaEvent, Runtime, device_guard::DeviceGuard, execution_domain::EnteredExecution},
};

struct OwnedTransferImportRetention<T> {
    _retained_transfer: T,
    _readiness_barrier: Option<CudaEvent>,
}

fn order_runtime_stream_after_owned_transfer_readiness(
    readiness: &ProducerReadiness<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Option<CudaEvent>, InteropError> {
    let _guard = DeviceGuard::new(runtime.device())
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let consumer_stream_raw = runtime
        .stream()
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            let barrier = CudaEvent::new()
                .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            let barrier_raw = barrier
                .interop_raw_handle()
                .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, stream.as_raw()) },
                operation,
                [("readiness", "device_stream_ordered".to_string()), ("profile", "owned_device_transfer".to_string())],
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(consumer_stream_raw, barrier_raw) },
                operation,
                [("readiness", "device_stream_ordered".to_string()), ("profile", "owned_device_transfer".to_string())],
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(Some(barrier))
        },
        ProducerReadiness::DeviceEventOrdered { event } => {
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(consumer_stream_raw, event.as_raw()) },
                operation,
                [("readiness", "device_event_ordered".to_string()), ("profile", "owned_device_transfer".to_string())],
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(None)
        },
        ProducerReadiness::HostVisible => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "owned device transfer import requires device-backed readiness",
        )
        .with_context("operation", operation)
        .with_context("readiness", "host_visible")
        .with_context("profile", "owned_device_transfer")),
    }
}

fn partitioned_prepared_import_profile_supported(
    profile: PreparedExportProfile,
    operation: &'static str,
) -> Result<(), InteropError> {
    match profile {
        PreparedExportProfile::Device => Ok(()),
        PreparedExportProfile::Host => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "partitioned prepared table import requires device-backed prepared export partitions",
        )
        .with_context("operation", operation)
        .with_context("profile", "prepared_host_export")
        .with_context("consumer", "cudf")),
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

fn owned_transfer_producer_device(
    readiness: &ProducerReadiness<'_>,
    operation: &'static str,
) -> Result<i32, InteropError> {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => Ok(stream.device()),
        ProducerReadiness::DeviceEventOrdered { event } => Ok(event.device()),
        ProducerReadiness::HostVisible => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "owned device transfer import requires device-backed readiness",
        )
        .with_context("operation", operation)
        .with_context("readiness", "host_visible")
        .with_context("profile", "owned_device_transfer")),
    }
}

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

fn partition_runtime_count_mismatch_error(
    operation: &'static str,
    expected_count: usize,
    actual_count: usize,
) -> InteropError {
    InteropError::new(
        InteropErrorKind::ExecutionBindingMismatch,
        "partitioned prepared table export import requires one destination runtime per partition",
    )
    .with_context("operation", operation)
    .with_context("resource", "execution_bindings")
    .with_context("reason", "partition_runtime_count_mismatch")
    .with_context("expected_count", expected_count.to_string())
    .with_context("actual_count", actual_count.to_string())
    .with_context("remedy", "provide exactly one destination runtime per partition")
}

fn missing_dataframe_import_names_error(operation: &'static str, explicit_variant: &'static str) -> InteropError {
    InteropError::new(
        InteropErrorKind::InvalidView,
        format!(
            "high-level DataFrame import requires schema names on the shared root; use {explicit_variant} when the root does not carry schema names"
        ),
    )
    .with_context("operation", operation)
    .with_context("resource", "schema_names")
    .with_context("reason", "missing_schema_names")
    .with_context("remedy", format!("use {explicit_variant} when the root does not carry schema names"))
}

fn runtime_exec_options(runtime: &Runtime, operation: &'static str) -> Result<EnteredExecution, InteropError> {
    runtime
        .execution_domain()
        .enter_execution()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))
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

fn runtime_shared_stream_authorities<'a>(
    runtime: &'a Runtime,
    operation: &'static str,
) -> Result<(ReadStartAuthority<'a>, CompletionAuthority<'a>), InteropError> {
    let stream =
        super::cuda_stream_handle(runtime.stream()).map_err(|error| error.with_context("operation", operation))?;
    Ok((ReadStartAuthority::ConsumerStream { stream }, CompletionAuthority::SameDeviceStream { stream }))
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
        "borrowed descriptor import requires the destination runtime to use the same CUDA device",
        borrowed.execution().device(),
        runtime,
    )?;
    let exec_options = runtime_exec_options(runtime, operation)?;
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
                        exec_options.options(),
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
                        exec_options.options(),
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        },
    }

    unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
        .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))
}

pub(super) fn import_table_from_borrowed_view(
    borrowed: &BorrowedTableView<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Table, InteropError> {
    validate_runtime_device(
        operation,
        "borrowed descriptor import requires the destination runtime to use the same CUDA device",
        borrowed.execution().device(),
        runtime,
    )?;
    let columns = borrowed
        .columns()
        .iter()
        .map(|column| import_column_from_borrowed_view(column, runtime, operation))
        .collect::<Result<Vec<_>, _>>()?;

    Table::from_columns_in_domain_with_num_rows(
        columns,
        borrowed.num_rows(),
        Arc::clone(runtime.execution_domain()),
        operation,
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))
}

pub(super) fn import_column_from_owned_transfer(
    transfer: OwnedColumnTransfer,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Column, InteropError> {
    let producer_device = owned_transfer_producer_device(transfer.readiness(), operation)?;
    validate_runtime_device(
        operation,
        "owned device transfer import requires the destination runtime to use the same CUDA device",
        producer_device,
        runtime,
    )?;
    let readiness_barrier =
        order_runtime_stream_after_owned_transfer_readiness(transfer.readiness(), runtime, operation)?;
    match transfer.try_into_owner::<OwnedColumnTransferState>() {
        Ok(state) => Ok(state.into_column().with_execution_domain(Arc::clone(runtime.execution_domain()))),
        Err(mismatch) => match mismatch.into_transfer().try_into_owner::<RuntimeOwnedColumnTransferOwner>() {
            Ok(owner) => {
                Ok(owner.into_column(operation)?.with_execution_domain(Arc::clone(runtime.execution_domain())))
            },
            Err(mismatch) => {
                let transfer = mismatch.into_transfer();
                let graph = prepared_device_import_column_graph_from_parts(
                    transfer.semantic_columns(),
                    transfer.bindings(),
                    operation,
                )?;
                let num_semantic_columns =
                    checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)?;
                let num_bindings = checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)?;
                let root_column_id = transfer.column().id().get();
                // SAFETY: The C++ handle produced by
                // cudf_column_create_from_owned_device_transfer wraps a foreign
                // view-only handle state. It never frees or reallocates the
                // underlying GPU buffers; ownership lives exclusively in
                // `retained_owner`.
                let retained_owner: Arc<dyn Any + Send + Sync> = Arc::new(OwnedTransferImportRetention {
                    _retained_transfer: transfer,
                    _readiness_barrier: readiness_barrier,
                });
                let mut handle = std::ptr::null_mut();
                error::check_status_in(
                    unsafe {
                        cudf_sys::cudf_column_create_from_owned_device_transfer(
                            root_column_id,
                            graph.ffi_columns.as_ptr(),
                            num_semantic_columns,
                            graph.ffi_bindings.as_ptr(),
                            num_bindings,
                            &mut handle,
                        )
                    },
                    operation,
                )
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

                unsafe {
                    Column::from_raw_with_domain_and_owner(
                        handle,
                        Arc::clone(runtime.execution_domain()),
                        retained_owner,
                    )
                }
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))
            },
        },
    }
}

pub(super) fn import_table_from_owned_transfer(
    transfer: OwnedTableTransfer,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Table, InteropError> {
    const MAX_OWNED_TABLE_TRANSFER_NESTING: usize = 32;

    let producer_device = owned_transfer_producer_device(transfer.readiness(), operation)?;
    validate_runtime_device(
        operation,
        "owned device transfer import requires the destination runtime to use the same CUDA device",
        producer_device,
        runtime,
    )?;
    let readiness_barrier =
        order_runtime_stream_after_owned_transfer_readiness(transfer.readiness(), runtime, operation)?;
    let mut nesting = 0usize;
    let mut current = transfer;
    loop {
        match current.try_into_owner::<OwnedTableTransferState>() {
            Ok(state) => {
                return Ok(state.into_table().with_execution_domain(Arc::clone(runtime.execution_domain())));
            },
            Err(mismatch) => match mismatch.into_transfer().try_into_owner::<RuntimeOwnedTableTransferOwner>() {
                Ok(owner) => {
                    return Ok(owner
                        .into_table(operation)?
                        .with_execution_domain(Arc::clone(runtime.execution_domain())));
                },
                Err(mismatch) => match mismatch.into_transfer().try_into_owner::<OwnedTableTransfer>() {
                    Ok(inner_transfer) => {
                        nesting += 1;
                        if nesting > MAX_OWNED_TABLE_TRANSFER_NESTING {
                            return Err(InteropError::new(
                                InteropErrorKind::ProfileUnsupported,
                                "cudf owned table import rejected an owned-transfer root with excessive nesting",
                            )
                            .with_context("operation", operation)
                            .with_context("consumer", "cudf")
                            .with_context("profile", "owned_device_transfer")
                            .with_context("max_nesting", MAX_OWNED_TABLE_TRANSFER_NESTING.to_string())
                            .with_context("observed_nesting", nesting.to_string()));
                        }
                        current = inner_transfer;
                    },
                    Err(mismatch) => {
                        let transfer = mismatch.into_transfer();
                        let graph = owned_device_import_table_graph(&transfer, operation)?;
                        let num_semantic_columns =
                            checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)?;
                        let num_bindings = checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)?;
                        let owner_count = transfer.table().columns().len();
                        // SAFETY: Same invariant as the column path. The C++
                        // table handle is foreign and view-only, and ownership
                        // lives exclusively in `shared_owner`.
                        let shared_owner: Arc<dyn Any + Send + Sync> = Arc::new(OwnedTransferImportRetention {
                            _retained_transfer: transfer,
                            _readiness_barrier: readiness_barrier,
                        });
                        let retained_owners = (0..owner_count).map(|_| Some(Arc::clone(&shared_owner))).collect();
                        let mut handle = std::ptr::null_mut();
                        error::check_status_in(
                            unsafe {
                                cudf_sys::cudf_table_create_from_owned_device_transfer(
                                    &graph.ffi_table,
                                    graph.ffi_columns.as_ptr(),
                                    num_semantic_columns,
                                    graph.ffi_bindings.as_ptr(),
                                    num_bindings,
                                    &mut handle,
                                )
                            },
                            operation,
                        )
                        .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
                        return unsafe {
                            Table::from_raw_with_domain_and_owners(
                                handle,
                                Arc::clone(runtime.execution_domain()),
                                retained_owners,
                            )
                        }
                        .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error));
                    },
                },
            },
        }
    }
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

    match export.profile() {
        PreparedExportProfile::Device => {
            let producer_device = prepared_device_export_producer_device(export.readiness(), operation)?;
            validate_runtime_device(
                operation,
                "prepared device export import requires the destination runtime to use the same CUDA device",
                producer_device,
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
            let completion = lease
                .complete()
                .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_shared_completion_result(completion, operation)?;
            Ok(imported)
        },
        PreparedExportProfile::Host => {
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
                        exec_options.options(),
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let completion = lease
                .complete()
                .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_shared_completion_result(completion, operation)?;
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

    match export.profile() {
        PreparedExportProfile::Device => {
            let producer_device = prepared_device_export_producer_device(export.readiness(), operation)?;
            validate_runtime_device(
                operation,
                "prepared device export import requires the destination runtime to use the same CUDA device",
                producer_device,
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
            let completion = lease
                .complete()
                .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_shared_completion_result(completion, operation)?;
            Ok(imported)
        },
        PreparedExportProfile::Host => {
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
                        exec_options.options(),
                        &mut handle,
                    )
                },
                operation,
            )
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

            let imported = unsafe { Table::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let completion = lease
                .complete()
                .map_err(|error| map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_shared_completion_result(completion, operation)?;
            Ok(imported)
        },
    }
}

pub(super) fn import_partitioned_local_tables_from_prepared_export(
    export: &PartitionedPreparedTableExport<'_>,
    runtimes: &[&Runtime],
    operation: &'static str,
) -> Result<Vec<(Vec<usize>, Table)>, InteropError> {
    if export.partitions().len() != runtimes.len() {
        return Err(partition_runtime_count_mismatch_error(operation, export.partitions().len(), runtimes.len()));
    }

    let mut imported = Vec::with_capacity(export.partitions().len());
    for (partition_index, (partition, runtime)) in export.partitions().iter().zip(runtimes.iter().copied()).enumerate()
    {
        partitioned_prepared_import_profile_supported(partition.export().profile(), operation)
            .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;

        let producer_device = prepared_device_export_producer_device(partition.export().readiness(), operation)
            .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;
        validate_runtime_device(
            operation,
            "partitioned prepared device export import requires each destination runtime to use the matching partition CUDA device",
            producer_device,
            runtime,
        )
        .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;

        let graph = prepared_device_import_table_graph(partition.export(), operation)
            .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;
        let exec_options = runtime_exec_options(runtime, operation)?;
        let num_semantic_columns =
            checked_i32_from_usize("num_semantic_columns", graph.ffi_columns.len(), operation)
                .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;
        let num_bindings = checked_i32_from_usize("num_bindings", graph.ffi_bindings.len(), operation)
            .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;
        let (read_start, completion) = runtime_shared_stream_authorities(runtime, operation)
            .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;
        let lease = partition
            .try_activate_read_lease(read_start, completion)
            .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;
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
        .map_err(|error| {
            map_cudf_error(InteropErrorKind::InvalidView, operation, error)
                .with_context("partition_index", partition_index.to_string())
        })?;

        let table = unsafe { Table::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let completion = lease.complete().map_err(|error| {
            map_interop_error(InteropErrorKind::ReadinessUnavailable, operation, error)
                .with_context("partition_index", partition_index.to_string())
        })?;
        synchronize_shared_completion_result(completion, operation)
            .map_err(|error| error.with_context("partition_index", partition_index.to_string()))?;

        imported.push((partition.top_level_column_indices().to_vec(), table));
    }

    Ok(imported)
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
    names.map(|names| names.to_vec()).ok_or_else(|| missing_dataframe_import_names_error(operation, explicit_variant))
}
