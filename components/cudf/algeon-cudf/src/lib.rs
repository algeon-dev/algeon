//! Workspace-internal safe libcudf execution component for Algeon.
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion); workspace code
//! uses this crate for its component contracts. See the
//! [API documentation boundary](https://algeon.dev/design#api-documentation).
//!
//! Algeon integrations start with [`Runtime`] execution provenance and the
//! prepared contracts in [`interop`]. Operations retain device, stream,
//! memory-resource, and owner identity; allocation-capable paths expose local
//! preflight facts; failures cross the component boundary as structured
//! [`Error`] values. [`DataFrame`] and [`Series`] are the physical GPU frame
//! owners used by the query engine.
//!
//! # Quick Start
//!
//! ```no_run
//! use cudf::{DataFrame, Runtime};
//!
//! // Build an explicit runtime for this workflow.
//! let rt = Runtime::new().unwrap();
//!
//! // Create unnamed Series from host data inside that runtime.
//! let ids = rt.series(&[1i32, 2, 3]).unwrap();
//! let scores = rt.series(&[10.0f64, 20.0, 30.0]).unwrap();
//!
//! // Build a DataFrame with explicit schema names.
//! let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();
//!
//! assert_eq!(df.num_columns(), 2);
//! assert_eq!(df.num_rows(), 3);
//!
//! // Retrieve a column back as a Series (zero-copy, Arc-shared).
//! let score = df.series("score").unwrap();
//! assert_eq!(score.len(), 3);
//!
//! // Common reductions can materialize directly to host values.
//! let total = score.on(rt.exec()).sum_value::<f64>().unwrap();
//! let mean = score.on(rt.exec()).mean_value().unwrap();
//!
//! assert_eq!(total, 60.0);
//! assert_eq!(mean, Some(20.0));
//! ```
//!
//! # Primary Entry Points
//!
//! - [`DataFrame`]: A named, ordered collection of GPU columns with
//!   immutable-value semantics.
//! - [`Series`]: A named, `Arc`-shared GPU column; the primary column type.
//! - [`Runtime`]: Owns the default CUDA stream and retained provenance for a
//!   workflow, using the device-level memory manager by default.
//!
//! # Root-Level Operations
//!
//! - [`SortOptions`] / [`FrameOps::sort_values`](crate::FrameOps::sort_values): GPU-accelerated sorting.
//! - [`MergeOptions`] / [`FrameOps::merge`](crate::FrameOps::merge): GPU-accelerated joins.
//! - [`GroupBy`] / [`DataFrame::groupby`]: GPU-accelerated groupby aggregation.
//! - [`SeriesOps::sum_value`], [`SeriesOps::mean_value`], [`SeriesOps::minmax_values`]:
//!   high-level reductions that return host values directly.
//!
//! # Specialized High-Level Types
//!
//! - [`frame::dataframe`]: DataFrame-specific builders and result snapshots
//!   such as [`frame::dataframe::DropNaOptions`] and
//!   [`frame::dataframe::InfoDetail`].
//! - [`frame::factorization`]: typed categorical encoding results and
//!   reusable mappings such as [`frame::factorization::Factorized`],
//!   [`frame::factorization::Codebook`], and
//!   [`frame::factorization::EncodedSeries`].
//!
//! # Low-Level Primitives and Interop
//!
//! The `core` module provides direct ownership of GPU buffers, device-memory
//! descriptors, and conversion boundaries. [`Runtime`] manages default
//! execution resources and retained provenance for a pipeline, while [`Exec`]
//! selects the stream and memory resource for individual operations. On
//! [`Series`], GPU operations run through [`.on(exec)`](Series::on); on
//! [`DataFrame`], operations run through [`.on(exec)`](DataFrame::on).
//!
//! - [`core::Column`]: Owned GPU column with physical buffer descriptors.
//! - [`core::Table`]: Ordered collection of equal-length owned GPU columns.
//! - [`core::Scalar`]: Single device-resident value for explicit GPU scalar workflows.
//! - [`Runtime`] / [`Exec`]: Stream and allocator lifecycle controls.
//! - [`interop::arrow_host`]: Arrow `RecordBatch` conversion via host staging.
//! - [`interop`]: Device-resident interchange adapters, runtime-backed export
//!   contracts, and handle projections. Shared types re-export through
//!   [`interop::contract`].
//! - [`runtime`]: CUDA stream, event, device inventory, and memory pool controls.
//!
//! Cross-library GPU interchange uses the prepared export descriptors in
//! [`interop::contract`] and the conversion helpers in [`interop`].
//! [`core::DeviceSlice`] exposes only a raw pointer and length without
//! null-mask, offset, readiness, completion, or execution provenance metadata.
//!
//! Interop contracts re-export from [`interop::contract`]. When holding
//! `cudf` runtime handles, [`interop::cuda_stream_handle`] and
//! [`interop::cuda_event_handle`] project them into shared descriptors.
//! External producers and consumers can depend directly on `algeon-interop`
//! without a dependency on `cudf`.
//!
//! [`interop::arrow_host`] handles host-copy `RecordBatch` interchange.
//! Device-resident data leaves `cudf` through prepared RAPIDS exports.
//!
//! # Memory Architecture and Scopes
//!
//! Multi-GPU pipelines isolate allocator state per device without taking host
//! locks across GPU execution.
//!
//! - *Device domain*: Each CUDA device configures one memory domain (upstream
//!   CUDA resource, pool, and reservation-aware adapter). The adapter installs
//!   as the per-device current RMM resource at domain initialization and remains
//!   active for the process lifetime. Intermediate allocations resolve through
//!   this adapter and account to their execution stream. Device domains share
//!   no allocator state with one another. Services configure devices during
//!   startup via [`DeviceMemoryManager::install`].
//! - *Query domain*: [`QueryMemoryDomain`] enforces an allocation ceiling across
//!   associated streams. Streams reserve capacity through
//!   [`StreamReservationLease`]. Manager allocations route to that reservation,
//!   and stage execution tokens share the reservation resource without raising
//!   the cap.
//! - *Operation scope*: Operations execute with an explicit device, stream, and
//!   memory resource. The memory resource governs output allocations;
//!   intermediate buffers allocate through the device domain and charge to the
//!   operation stream.
//!
//! ```no_run
//! use cudf::{
//!     DeviceId, DeviceMemoryManager, InstallSpec, QueryMemoryDomainOptions,
//!     Runtime,
//! };
//!
//! DeviceMemoryManager::install(
//!     DeviceId::new(0).unwrap(),
//!     InstallSpec::default(),
//! )
//! .unwrap();
//!
//! let rt = Runtime::builder().build().unwrap();
//! let domain = rt
//!     .device_memory_manager()
//!     .unwrap()
//!     .query_memory_domain(7, QueryMemoryDomainOptions::new(1 << 30))
//!     .unwrap();
//! let lease = domain.reserve_stream(rt.stream(), 1, 1 << 30).unwrap();
//! # drop(lease);
//! ```
//!
//! # Stream Ordering and Retained Provenance
//!
//! Owned GPU types ([`core::Column`], [`core::Table`], [`DataFrame`],
//! [`core::Scalar`]) carry retained provenance recording the CUDA stream and
//! allocator handle that generated them.
//!
//! - *Same-runtime operations*: Objects produced within one [`Runtime`] share
//!   execution provenance and submit work without cross-stream synchronization.
//! - *Mixed-provenance data operations*: Data operations that assemble an output
//!   domain from multiple runtimes (`merge`, `concat`, gather, filter, and
//!   column substitution) insert device-side event dependencies in both
//!   directions (`cudaEventRecord` followed by `cudaStreamWaitEvent`). The
//!   output stream waits for queued input work before execution begins;
//!   subsequently, input streams wait for operation completion to prevent
//!   premature buffer reuse on drop. Host threads do not block during this
//!   synchronization.
//! - *Mixed-provenance structural operations*: Schema-level operations that
//!   retain input column pointers as-is ([`DataFrame::with_column`],
//!   [`DataFrame::assign`]) reject cross-runtime inputs instead of inserting
//!   fences. Relocation requires explicit copies via
//!   [`DataFrame::copy_into_runtime`] or [`Series::copy_into_runtime`].
//!
//! Placing collaborating objects in the same [`Runtime`] avoids cross-stream
//! event synchronization.
//!
//!
//! # Multi-GPU
//!
//! Each [`Runtime`] is pinned to one CUDA device. When an operation is
//! invoked through a device-pinned runtime, the FFI layer re-enters the
//! runtime's device before enqueuing GPU work, and restores the calling
//! thread's previous CUDA device afterward.
//!
//! ```no_run
//! use cudf::{DeviceId, Runtime, Series};
//!
//! let rt0 = Runtime::builder()
//!     .with_device(DeviceId::new(0).unwrap())
//!     .build()
//!     .unwrap();
//! let rt1 = Runtime::builder()
//!     .with_device(DeviceId::new(1).unwrap())
//!     .build()
//!     .unwrap();
//!
//! // Each series lives on its respective GPU.
//! let s0 = rt0.series(&[1i32, 2, 3]).unwrap();
//! let s1 = rt1.series(&[4i32, 5, 6]).unwrap();
//! ```

pub mod core;
pub mod device;
pub mod diagnostics;
pub mod frame;
pub mod hash;
pub mod interop;
// Internal helpers; not part of the public API.
pub(crate) mod internal;
pub mod io;
pub mod ops;
pub mod runtime;
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub mod testing;

//; Primary entry points;
//; Types and error;
pub use core::{
    error::{
        BudgetExhaustionDetail, CudfAllocationAttribution, CudfErrorClassification, CudfErrorDetail,
        CudfErrorFailureSource, Error, ErrorCode, ErrorKind, ErrorStatus, Result, allocation_attribution,
        classify_cudf_error,
    },
    memory::{
        AllocatorEvent, AllocatorEventAction, AllocatorEventCompleteness, AllocatorEventDomainIdentity,
        AllocatorEventRead, AllocatorEventRingBounds, CurrentResourceRouteLease, DeviceMemoryFailureAllocationDomain,
        DeviceMemoryFailureInfo, DeviceMemoryFailureSource, DeviceMemoryManager, DeviceMemoryManagerBase,
        DeviceMemoryManagerFailureCallbackState, DeviceMemoryManagerInstallationCause,
        DeviceMemoryManagerInstallationError, DeviceMemoryManagerInstallationIdentity,
        DeviceMemoryManagerInstallationOutcome, DeviceMemoryManagerInstallationOutcomeKind,
        DeviceMemoryManagerInstallationRetry, DeviceMemoryManagerInstallationState,
        DeviceMemoryManagerInstallationStatus, DeviceMemoryManagerOptions, DeviceMemoryPoolUsage, DeviceMemoryResource,
        DeviceMemoryResourceUsage, InstallSpec, QueryMemoryDomain, QueryMemoryDomainAllocation,
        QueryMemoryDomainIdentity, QueryMemoryDomainOptions, QueryMemoryDomainUsage, RawCurrentResourceRouteLease,
        RawStreamReservationLease, ReservationUsage, StreamReservationLease,
    },
    scalar::ScalarValue,
    types::{CudfType, DataType, NullEquality, NullPolicy, TimestampResolution, TypeId},
};

pub use device::inventory::DeviceMemory;
pub use frame::{
    dataframe::{
        DataFrame, DataFrameConditionalFullJoinSizeData, DataFrameCrossJoinOptions, DataFrameFilteredJoin,
        DataFrameFilteredJoinOptions, DataFrameFilteredJoinProbeState, DataFrameHashJoin,
        DataFrameHashJoinMaterialized, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices, DataFrameJoinIndices,
        DataFrameJoinOptions, DataFrameJoinPairIndices, DataFrameMemoryIdentity, DataFrameMixedJoinOptions,
        DataFrameMixedJoinSizeData, DataFrameMixedSingleSideJoinSizeData, DataFramePredicateJoinOptions,
        DataFramePreparedSort, DataFrameProjectedCrossJoinPlan, FrameMemoryEffect, FrameOps, MultiParquetReadOptions,
        ParquetReadOptions, ParquetSourceChunkReadOutput, ParquetSourceReadPreflight, SourceRead, SourceReadInput,
    },
    series::{
        IntoSeries, ScalarReductionOutput, ScalarReductionOutputShape, Series, SeriesDatetimeOps, SeriesOps,
        SeriesPreparedScalarReduction, SeriesStringOps,
    },
};
//; High-level operations;
pub use ops::Expr;
pub use ops::{
    groupby::{
        AggSpec, GroupBy, GroupByAgg, GroupByOptions, PreparedDataFrameGroupBy, PreparedGroupByFacts, ShiftSpec,
    },
    join::{DataFrameMixedLeftJoinFinalized, DataFrameMixedLeftJoinState, JoinType, MergeOptions},
    reduction::ReduceAgg,
    replace::FillPolicy,
    sorting::SortOptions,
    stream_compaction::DuplicateKeepOption,
    strings::{
        REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT, ReplaceWithBackrefsOutputSizePreflight,
        ReplaceWithBackrefsPreparedOutputSizeFacts, SeriesPreparedReplaceWithBackrefs, StringSliceOptions, StripSide,
    },
};
pub use runtime::{
    AttemptStreamProviderAlias, AttemptStreamProviderRegistration, BudgetedExecutionOutcome, BudgetedOutput,
    CacheAllocationDomain, CacheAllocationDomainId, CacheAllocationLease, CacheAllocationScope, CacheAllocationScopeId,
    CacheAllocationScopeIdentity, DenseMatrixLayout, DeviceId, Exec, ExecutionBudgetEnforcement, ExecutionBudgetReport,
    ExecutionBudgetRequest, ExecutionBudgetToken, ExecutionPlacement, PackedMatrix, Runtime, RuntimeBuilder,
    StreamDependencyOutcome, ValidatedPackedMatrix,
};
