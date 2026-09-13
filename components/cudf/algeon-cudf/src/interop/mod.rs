//! External interop boundary.
//!
//! This module moves data between `cudf` and other systems such as Arrow and
//! RAPIDS. Most callers should use the methods on [`crate::Series`] for one
//! column or [`crate::DataFrame`] for a table. Use [`crate::core::Column`] and
//! [`crate::core::Table`] only when explicit low-level ownership or borrowing
//! control is required.
//!
//! To export from `cudf`, choose one of three contracts:
//!
//! - `try_borrow_*_view` lends device buffers without transferring ownership.
//! - `try_prepare_*_export` prepares data for consumer access or copying.
//! - `try_into_owned_transfer` moves device-buffer ownership to the consumer.
//!
//! To import into `cudf`, call `try_import_from_*_into_runtime` with the
//! destination [`crate::Runtime`]. [`runtime_release`] and [`arrow_device`]
//! expose `cudf`-specific advanced controls; [`arrow_host`] converts through
//! host `RecordBatch` values. Import the shared types used by these APIs from
//! [`contract`]. Independent producers and consumers can instead depend on
//! `algeon-interop` directly and import the same types from `rapids_interop`.

pub mod arrow_device;
pub mod arrow_host;
mod borrowed;
mod column;
mod consumer_import;
/// Native-independent interop contracts used by the `cudf` adapters.
///
/// This re-exports the `rapids_interop` crate, so values imported through this
/// module have the same types as values imported directly from that crate.
pub use rapids_interop as contract;
mod contracts;
mod errors;
pub(crate) mod execution;
pub mod frame;
mod import_graph;
pub(crate) mod projection;
mod runtime_lease;
pub mod runtime_release;
mod table;
mod table_export;
pub(crate) mod testing_hooks;

use std::sync::Arc;

use borrowed::*;
use column::{prepare_from_owned_column, prepare_host_from_owned_column, transfer_from_owned_column};
use consumer_import::{
    dataframe_import_names, import_column_from_borrowed_view, import_column_from_owned_transfer,
    import_column_from_prepared_export, import_partitioned_local_tables_from_prepared_export,
    import_table_from_borrowed_view, import_table_from_owned_transfer, import_table_from_prepared_export,
    wrap_imported_dataframe, wrap_imported_series,
};
use contracts::*;
use errors::*;
use execution::*;
pub use frame::PartitionedDataFrameExport;
use prepared::*;
use projection::{prepared_column_projection_from_view, prepared_host_column_projection_from_view};
use rapids_interop::{
    OwnedColumnTransfer, OwnedTableTransfer, OwnedTableTransferParts, PartitionedOwnedTableTransfer,
    PartitionedOwnedTableTransferPartition, ReleaseContract, ReleaseSynchronization, TransferCompletionRule,
};
use runtime_release::*;
pub use table::{PartitionedLocalTableResult, PartitionedLocalTableResults};
mod prepared {
    pub use rapids_interop::{
        ExportDiagnostics, ExportProfile, InteropError, InteropErrorKind, PartitionedPreparedTableExport,
        PartitionedPreparedTablePartition, PreparedColumnExport, PreparedTableExport, ProducerReadiness,
    };
}
pub use runtime_lease::cuda_explicit_completion_required_lease_backend;
use table_export::{
    prepare_device_table_export_from_columns, prepare_device_table_export_from_columns_with_readiness,
    prepare_host_table_export_from_columns, prepare_host_table_export_from_view,
    prepare_partitioned_device_table_export_from_columns, prepare_partitioned_host_table_export_from_columns,
    prepare_table_export_from_view, prepare_table_export_from_view_with_readiness, transfer_from_owned_table,
    transfer_from_owned_table_with_names, transfer_partitioned_owned_table_from_columns,
    transfer_partitioned_owned_table_from_columns_with_runtime_release,
    transfer_partitioned_owned_table_from_columns_with_runtime_release_synchronization,
};

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_to_owned},
        error,
        table::{Table, TableView},
        types::{DataType, TypeId},
    },
    frame::{dataframe::DataFrame, series::Series},
    internal::{
        hash::{FastIndexMap, fast_index_map, fast_index_map_with_capacity},
        resolve::validate_no_duplicate_names,
    },
    ops::dictionary,
    runtime::{
        BoundaryKind, BoundaryObservation, CudaEvent, ExecutionPlacement, QueryStageKind, Runtime,
        execution_domain::ExecutionDomain, record_boundary_observation_in_domain,
    },
};

fn boundary_observation(
    kind: BoundaryKind,
    placement: ExecutionPlacement,
    rows: usize,
    columns: usize,
    synchronized_for_host: bool,
) -> BoundaryObservation {
    let observation = BoundaryObservation::new(kind, placement, rows, columns, synchronized_for_host);
    match kind {
        BoundaryKind::InteropLease | BoundaryKind::OwnedTransfer => {
            observation.with_query_stage(QueryStageKind::InteropExport)
        },
        _ => observation,
    }
}

fn record_observation(domain: &Arc<ExecutionDomain>, observation: BoundaryObservation) {
    record_boundary_observation_in_domain(domain, observation);
}

fn column_boundary_observation(
    kind: BoundaryKind,
    column: &Column,
    operation: &'static str,
) -> Result<BoundaryObservation, InteropError> {
    let rows = column.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    Ok(boundary_observation(kind, column.execution_domain().placement(), rows, 1, false))
}

fn column_view_boundary_observation(
    kind: BoundaryKind,
    view: &ColumnView<'_>,
    operation: &'static str,
) -> Result<BoundaryObservation, InteropError> {
    let rows = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    Ok(boundary_observation(kind, view.execution_domain().placement(), rows, 1, false))
}

fn table_boundary_observation(
    kind: BoundaryKind,
    table: &Table,
    operation: &'static str,
) -> Result<BoundaryObservation, InteropError> {
    let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    table_view_boundary_observation(kind, &view, operation)
}

fn table_view_boundary_observation(
    kind: BoundaryKind,
    view: &TableView<'_>,
    operation: &'static str,
) -> Result<BoundaryObservation, InteropError> {
    let rows = view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let columns =
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    Ok(boundary_observation(kind, view.execution_domain().placement(), rows, columns, false))
}

/// Borrow a `cudf` CUDA stream as a shared RAPIDS interop stream handle.
///
/// This does not transfer ownership. The returned handle is valid only for the
/// lifetime of `stream`.
pub fn cuda_stream_handle(
    stream: &crate::runtime::CudaStream,
) -> Result<rapids_interop::CudaStreamHandle<'_>, InteropError> {
    let raw = stream.interop_raw_handle().map_err(|error| {
        map_cudf_error(InteropErrorKind::ReadinessUnavailable, "interop::cuda_stream_handle", error)
    })?;
    rapids_interop::CudaStreamHandle::from_raw_on_device(raw, stream.device().ordinal_raw())
        .map_err(|error| error.with_context("operation", "interop::cuda_stream_handle"))
}

/// Borrow a `cudf` CUDA event as a shared RAPIDS interop event handle.
///
/// This does not transfer ownership. The returned handle is valid only for the
/// lifetime of `event`.
pub fn cuda_event_handle(
    event: &crate::runtime::CudaEvent,
) -> Result<rapids_interop::CudaEventHandle<'_>, InteropError> {
    let raw = event
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, "interop::cuda_event_handle", error))?;
    rapids_interop::CudaEventHandle::from_raw_on_device(raw, event.device().ordinal_raw())
        .map_err(|error| error.with_context("operation", "interop::cuda_event_handle"))
}
