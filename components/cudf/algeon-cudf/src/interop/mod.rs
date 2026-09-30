//! External interop boundary.
//!
//! This module moves data between `cudf` and other systems such as Arrow and
//! RAPIDS. Most callers should use the methods on [`crate::Series`] for one
//! column or [`crate::DataFrame`] for a table. Use [`crate::core::Column`] and
//! [`crate::core::Table`] only when explicit low-level ownership or borrowing
//! control is required.
//!
//! To export from `cudf`, call `try_prepare_*_export` to prepare data for
//! consumer access or copying.
//!
//! To import into `cudf`, call `try_import_from_*_into_runtime` with the
//! destination [`crate::Runtime`]. [`arrow_host`] converts through host
//! `RecordBatch` values. Import the shared types used by these APIs from
//! [`contract`].
//! Independent producers and consumers can instead depend on `algeon-interop`
//! directly and import the same types from `rapids_interop`.

pub mod arrow_host;
mod column;
mod consumer_import;
mod errors;
pub(crate) mod execution;
mod frame;
mod import_graph;
pub(crate) mod projection;
mod runtime_lease;
mod table;
mod table_export;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) mod testing_hooks;

use column::{prepare_from_owned_column, prepare_host_from_owned_column};
use consumer_import::{
    dataframe_import_names, import_column_from_prepared_export, import_table_from_prepared_export,
    wrap_imported_dataframe, wrap_imported_series,
};
use errors::*;
use execution::*;
use prepared::*;
use projection::{prepared_column_projection_from_view, prepared_host_column_projection_from_view};
/// Native-independent interop contracts used by the `cudf` adapters.
///
/// This re-exports the `rapids_interop` crate, so values imported through this
/// module have the same types as values imported directly from that crate.
pub use rapids_interop as contract;
use rapids_interop::{
    BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, CudaEventHandle, CudaStreamHandle,
    DataType as InteropDataType, DeviceMemoryResourceHandle, ExecutionBinding, PhysicalValidity, Placement,
    SemanticColumn, SemanticTable, TypeId as InteropTypeId,
};
mod prepared {
    pub use rapids_interop::{
        InteropError, InteropErrorKind, PreparedAccess, PreparedColumnExport, PreparedTableExport, ProducerReadiness,
    };
}
use table_export::{
    prepare_device_table_export_from_columns, prepare_device_table_export_from_columns_with_readiness,
    prepare_host_table_export_from_columns, prepare_host_table_export_from_view, prepare_table_export_from_view,
    prepare_table_export_from_view_with_readiness,
};

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_to_owned},
        error,
        table::{Table, TableView},
        types::{DataType, TypeId},
    },
    frame::{dataframe::DataFrame, series::Series},
    ops::dictionary,
    runtime::{CudaEvent, CudaStream, Runtime, execution_domain::ExecutionDomain},
};

/// Borrow a `cudf` CUDA stream as a shared RAPIDS interop stream handle.
///
/// This does not transfer ownership. The returned handle is valid only for the
/// lifetime of `stream`.
pub fn cuda_stream_handle(stream: &CudaStream) -> Result<rapids_interop::CudaStreamHandle<'_>, InteropError> {
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
pub fn cuda_event_handle(event: &CudaEvent) -> Result<rapids_interop::CudaEventHandle<'_>, InteropError> {
    let raw = event
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, "interop::cuda_event_handle", error))?;
    rapids_interop::CudaEventHandle::from_raw_on_device(raw, event.device().ordinal_raw())
        .map_err(|error| error.with_context("operation", "interop::cuda_event_handle"))
}
