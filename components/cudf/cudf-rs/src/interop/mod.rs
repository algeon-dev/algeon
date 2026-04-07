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
//! - `cudf-rs` now supports consumer-side import from the shared borrowed
//!   compatibility descriptors back into low-level owned
//!   [`crate::core::Column`] and [`crate::core::Table`] values via explicit
//!   destination [`crate::Runtime`].
//! - `cudf-rs` also supports zero-copy consumer-side import from
//!   `cudf-rs`-produced shared owned-transfer roots back into low-level owned
//!   [`crate::core::Column`] and [`crate::core::Table`] values. Owned-transfer
//!   roots
//!   produced by other libraries are currently rejected with
//!   [`InteropErrorKind::ProfileUnsupported`].
//! - `cudf-rs` now also supports low-level consumer-side import from shared
//!   prepared-device export roots back into owned [`crate::core::Column`] and
//!   [`crate::core::Table`] values via explicit same-device destination
//!   [`crate::Runtime`]s.
//! - [`BorrowedColumnView`] and [`BorrowedTableView`] remain the borrowed GPU
//!   compatibility profile.
//! - [`PreparedColumnExport`] and [`PreparedTableExport`] are producer-rooted
//!   prepared export profiles for activated access.
//! - [`PartitionedPreparedTableExport`] and [`PartitionedOwnedTableTransfer`]
//!   are explicit multi-device wrappers around multiple already-valid
//!   single-device table roots.
//! - [`ActiveColumnReadLease`] and [`ActiveTableReadLease`] model activated
//!   read access with explicit read-start and completion contracts.
//! - [`CompletionWitness`] and [`LeaseCompletionResult`] make completion and
//!   drop fallback visible at the shared contract layer.
//! - [`RuntimeReadStartAuthority`], [`RuntimeCompletionAuthority`],
//!   [`RuntimeActiveColumnReadLease`], [`RuntimeActiveTableReadLease`], and
//!   their related `Runtime*` result and witness types are CUDA-specific
//!   compatibility shims layered on that same shared contract. Prefer the
//!   shared names for new interop APIs unless you specifically need `cudf-rs`
//!   runtime handles and producer-owned [`crate::runtime::CudaEvent`]
//!   witnesses.
//! - Borrowing and prepared export do not transfer ownership.
//! - Borrowed compatibility and prepared export both avoid host copies on the
//!   input path because they describe existing device buffers.
//! - Prepared host export is a separate materialized profile: it copies the
//!   active logical range to host-visible bytes and is not zero-copy.
//! - Owned device transfer is a separate zero-copy ownership move profile for
//!   true owner types such as [`crate::core::Column`] and
//!   [`crate::core::Table`].
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
//! - Explicit execution provenance is required for producer projection.
//! - Ambient/default-only objects are rejected at the interop boundary.

mod borrowed;
mod column;
mod common;
mod consumer_import;
mod frame;
mod projection;
mod runtime_lease;
mod table;
mod table_export;
mod transfer;

use std::sync::Arc;

use borrowed::*;
use column::{prepare_from_owned_column, prepare_host_from_owned_column, transfer_from_owned_column};
use common::*;
use consumer_import::{
    import_column_from_borrowed_view, import_column_from_owned_transfer, import_column_from_prepared_export,
    import_table_from_borrowed_view, import_table_from_owned_transfer, import_table_from_prepared_export,
    required_dataframe_import_names, wrap_imported_dataframe, wrap_imported_series,
};
pub use frame::PartitionedDataFrameExport;
use projection::{prepared_column_projection_from_view, prepared_host_column_projection_from_view};
pub use rapids_interop::{
    ActiveColumnReadLease, ActiveLeaseActivation, ActiveLeaseBackend, ActiveTableReadLease, BorrowedColumnView,
    BorrowedTableView, BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority,
    CompletionBinding, CompletionWitness, CompletionWitnessBackend, CompletionWitnessKind, CudaEventHandle,
    CudaStreamHandle, DataType as InteropDataType, DeviceMemoryResourceHandle, ExecutionBinding, ExportDiagnostics,
    ExportProfile, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation, LeaseBackend,
    LeaseCompletionResult, LeaseExportContext, LeaseReadiness, NullMask, NullMaskKind, OwnedColumnTransfer,
    OwnedTableTransfer, PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferPartition,
    PartitionedPreparedTableExport, PartitionedPreparedTablePartition, PhysicalValidity, Placement,
    PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority, ReadStartBinding,
    ReleaseContract, ReleaseSynchronization, SchemaNullability, SemanticColumn, SemanticTable, TableId,
    TransferCompletionRule, TryBorrowColumnView, TryBorrowTableView, TryIntoOwnedDeviceColumnTransfer,
    TryIntoOwnedDevicePartitionedTableTransfer, TryIntoOwnedDeviceTableTransfer, TryPrepareDeviceColumnExport,
    TryPrepareDevicePartitionedTableExport, TryPrepareDeviceTableExport, TryPrepareHostColumnExport,
    TryPrepareHostTableExport, TypeId as InteropTypeId,
};
pub use runtime_lease::{
    RuntimeActiveColumnReadLease, RuntimeActiveTableReadLease, RuntimeCompletionAuthority, RuntimeCompletionWitness,
    RuntimeLeaseCompletionResult, RuntimeReadStartAuthority, TryActivateRuntimeColumnReadLease,
    TryActivateRuntimeTableReadLease, cuda_explicit_completion_required_lease_backend,
};
use table_export::{
    prepare_device_table_export_from_columns, prepare_device_table_export_from_columns_with_readiness,
    prepare_host_table_export_from_columns, prepare_host_table_export_from_view,
    prepare_partitioned_device_table_export_from_columns, prepare_partitioned_host_table_export_from_columns,
    prepare_table_export_from_view, prepare_table_export_from_view_with_readiness, transfer_from_owned_table,
    transfer_from_owned_table_with_names, transfer_partitioned_owned_table_from_columns,
};
use transfer::*;

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_to_owned},
        error,
        table::{Table, TableView},
        types::{DataType, TypeId, validity_to_bitmask},
    },
    frame::{dataframe::DataFrame, series::Series},
    internal::{
        hash::{FastIndexMap, fast_index_map, fast_index_map_with_capacity},
        resolve::validate_no_duplicate_names,
    },
    ops::dictionary,
    runtime::{CudaEvent, Runtime, execution_domain::ExecutionDomain},
};
