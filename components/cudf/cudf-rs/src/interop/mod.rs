//! External interop boundaries.
//!
//! This module re-exports the shared RAPIDS Rust interop contract from
//! [`rapids_interop`] and implements `cudf-rs` producer adapters for:
//!
//! - [`crate::core::Column`]
//! - [`crate::core::ColumnView`]
//! - [`crate::core::Table`]
//! - [`crate::core::TableView`]
//! - [`crate::Series`]
//! - [`crate::DataFrame`]
//! - [`PartitionedDataFrameExport`]
//!
//! # Contract Summary
//!
//! - The shared [`rapids_interop`] names are the primary interop contract.
//! - Prepared export plus activated access is the primary consumer-neutral
//!   single-GPU contract.
//! - `cudf-rs` now supports consumer-side import from the shared borrowed
//!   compatibility descriptors back into low-level owned
//!   [`crate::core::Column`] and [`crate::core::Table`] values via explicit
//!   destination [`crate::Runtime`].
//! - `cudf-rs` also supports zero-copy consumer-side import from shared owned-transfer
//!   roots back into low-level owned [`crate::core::Column`] and [`crate::core::Table`]
//!   values. The current foreign-owner consumer slice covers the supported
//!   fixed-width, binary-like string, list, struct, and dictionary layouts
//!   while preserving `cudf-rs` round-trip fast paths.
//! - `cudf-rs` now also supports low-level consumer-side import from shared
//!   prepared-device export roots back into owned [`crate::core::Column`] and
//!   [`crate::core::Table`] values via explicit same-device destination
//!   [`crate::Runtime`]s.
//! - [`BorrowedColumnView`] and [`BorrowedTableView`] remain the borrowed GPU
//!   compatibility profile.
//! - [`PreparedColumnExport`] and [`PreparedTableExport`] are producer-rooted
//!   prepared export profiles for activated access.
//! - Deep-copy import into an explicit destination [`crate::Runtime`] remains
//!   the portability fallback when a consumer needs owned `cudf-rs` values
//!   instead of borrowing or moving ownership directly.
//! - [`PartitionedPreparedTableExport`] and [`PartitionedOwnedTableTransfer`]
//!   are explicit multi-device wrappers around multiple already-valid
//!   single-device table roots.
//! - [`ActiveColumnReadLease`] and [`ActiveTableReadLease`] model activated
//!   read access with explicit read-start and completion contracts.
//! - [`CompletionWitness`] and [`LeaseCompletionResult`] make completion and
//!   drop fallback visible at the shared contract layer.
//! - Use [`cuda_stream_handle`] and [`cuda_event_handle`] when a caller already
//!   has `cudf-rs` runtime handles and needs to activate a shared prepared
//!   lease with explicit CUDA stream or event facts.
//! - [`OwnedColumnTransferReleaser`] and [`OwnedTableTransferReleaser`] are
//!   CUDA-backed release helpers for the signal-driven owned-transfer producer
//!   path. Their caller-visible behavior follows the matching transfer root's
//!   declared [`ReleaseSynchronization`]: async roots return once release has
//!   been accepted and scheduled, while `MaySynchronizeInternally` roots may
//!   block while carrying out the release contract.
//! - [`PartitionedOwnedTableTransferReleaser`] mirrors that release path for
//!   partitioned owned transfer by preserving one releaser per
//!   retained-provenance partition. Import and release remain mutually
//!   exclusive independently for each partition.
//! - Borrowing and prepared export do not transfer ownership.
//! - Borrowed compatibility and prepared export both avoid host copies on the
//!   input path because they describe existing device buffers.
//! - Prepared host export is a separate materialized profile: it copies the
//!   active logical range to host-visible bytes and is not zero-copy.
//! - Owned device transfer is a separate zero-copy ownership move profile for
//!   true owner types such as [`crate::core::Column`] and
//!   [`crate::core::Table`], not the default consumer story.
//! - [`PartitionedDataFrameExport`] is an interop-only heterogeneous producer
//!   surface; it does not change the ordinary single-provenance semantics of
//!   [`crate::DataFrame`] or [`crate::core::Table`].
//! - Consumer algorithms may still allocate or copy for outputs; this module
//!   does not promise zero-copy output.
//! - Current device-export support covers fixed-width layouts, the first
//!   binary-like string layout slice, `List`, `Struct`, and `Dictionary`.
//! - Nested-layout support is still intentionally narrow in this slice:
//!   current coverage is the recursive semantic-column graph that can be
//!   expressed by the existing fixed-width, binary-like string, list, struct,
//!   and dictionary slices on one CUDA device.
//! - [`crate::Series`] and [`crate::DataFrame`] are convenience adapters, not
//!   the contract itself.
//! - Producer projection requires explicit retained provenance.
//! - Public `cudf-rs` object construction now satisfies that contract through
//!   explicit [`crate::Runtime`] ownership.
//!
//! # Module Position
//!
//! `cudf::interop` deliberately remains at the crate root rather than under
//! `cudf::core::interop`. The interop module spans both high-level types
//! ([`crate::DataFrame`], [`crate::Series`]) and low-level types
//! ([`crate::core::Column`], [`crate::core::ColumnView`],
//! [`crate::core::Table`], [`crate::core::TableView`]), so it does not
//! belong exclusively under `cudf::core`. High-level users should prefer
//! the [`crate::DataFrame`] and [`crate::Series`] interop adapters;
//! advanced users may use the low-level [`crate::core::Column`] /
//! [`crate::core::ColumnView`] adapters directly.
//!
//! # `ColumnView` in Interop Signatures
//!
//! [`crate::core::ColumnView`] appears in interop producer signatures (e.g.,
//! [`crate::core::ColumnView::try_borrow_column_view`],
//! [`crate::core::ColumnView::try_prepare_device_column_export`]). This is
//! intentional: interop operates at the `cudf::core::*` level where
//! `ColumnView` is an appropriate advanced type. The public API layering
//! refactor (Slice 3) removed `ColumnView` from `DataFrame`'s public
//! surface, but it remains correct in interop signatures.
//!
//! # Consumer Import Naming
//!
//! Consumer import methods use the `try_import_from_*_into_runtime` naming
//! convention. These are interop-boundary placement operations — they bring
//! external data into a target [`crate::Runtime`]. This is distinct from the
//! removed `DataFrame` operation-specific `_with_runtime` methods (Slice 4),
//! which hid placement as a side effect of a compute operation.

pub mod borrowed;
mod column;
mod consumer_import;
mod contracts;
mod errors;
mod execution;
pub mod frame;
mod import_graph;
mod projection;
mod runtime_lease;
mod table;
mod table_export;
mod table_support;
pub mod transfer;

use std::sync::Arc;

use borrowed::*;
use column::{prepare_from_owned_column, prepare_host_from_owned_column, transfer_from_owned_column};
use consumer_import::{
    import_column_from_borrowed_view, import_column_from_owned_transfer, import_column_from_prepared_export,
    import_partitioned_local_tables_from_prepared_export, import_table_from_borrowed_view,
    import_table_from_owned_transfer, import_table_from_prepared_export, required_dataframe_import_names,
    wrap_imported_dataframe, wrap_imported_series,
};
use contracts::*;
use errors::*;
use execution::*;
pub use frame::PartitionedDataFrameExport;
use prepared::*;
use projection::{prepared_column_projection_from_view, prepared_host_column_projection_from_view};
pub use table::{PartitionedLocalTableResult, PartitionedLocalTableResults};
pub mod prepared {
    pub use rapids_interop::{
        ActiveColumnReadLease, ActiveLeaseActivation, ActiveLeaseBackend, ActivePartitionedTableReadLease,
        ActiveTableReadLease, CompletionAuthority, CompletionBinding, CompletionWitness, CompletionWitnessBackend,
        CompletionWitnessKind, ExportDiagnostics, ExportProfile, IncompleteDropBehavior, InteropError,
        InteropErrorKind, LeaseActivation, LeaseBackend, LeaseCompletionResult, LeaseExportContext, LeaseReadiness,
        PartitionedPreparedTableExport, PartitionedPreparedTablePartition, PreparedColumnExport, PreparedTableExport,
        ProducerReadiness, ReadStartAuthority, ReadStartBinding, TryPrepareDeviceColumnExport,
        TryPrepareDevicePartitionedTableExport, TryPrepareDeviceTableExport, TryPrepareHostColumnExport,
        TryPrepareHostTableExport,
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
use transfer::*;

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
    runtime::{CudaEvent, Runtime, execution_domain::ExecutionDomain},
};

/// Borrow a `cudf-rs` CUDA stream as a shared RAPIDS interop stream handle.
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

/// Borrow a `cudf-rs` CUDA event as a shared RAPIDS interop event handle.
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
