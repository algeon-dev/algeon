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
//! # Advanced Usage
//!
//! The `core` types give direct control over owned GPU objects, device-memory
//! descriptors, and import/export boundaries. Most users should not need these.
//! Use [`Runtime`] to own default execution resources plus retained provenance
//! for a pipeline, and [`Exec`] to select where one operation runs. For
//! `Series`, every GPU operation runs through [`Series::on`], such as
//! `series.on(exec).sum()` or `series.on(exec).str().contains("x")`; for
//! table-style flows, use [`DataFrame::on`].
//!
//! - [`core::Column`]: An owned GPU column (analogous to `cudf::column`).
//! - [`core::Table`]: An owned GPU table (collection of columns).
//! - [`core::Scalar`]: A single value on the GPU. Most high-level reduction
//!   workflows can stay on [`Series`] via `*_value` helpers and only drop to
//!   `Scalar` when you need explicit device-resident scalar behavior.
//! - [`Runtime`] / [`Exec`]: explicit stream and allocator control.
//! - [`interop::arrow_host`]: Arrow `RecordBatch` conversion through host memory.
//! - [`interop`]: `cudf`-specific interop adapters, runtime-backed
//!   release helpers, and CUDA handle projection. `cudf` users can import
//!   their shared contract types from [`interop::contract`].
//! - For lower-level CUDA stream and event management, see the [`runtime`] module.
//!
//! If you need to hand device-backed cudf data to another Rust GPU library,
//! prefer the shared descriptors and prepared export types in
//! [`interop::contract`] plus the adapter methods in [`interop`] over
//! [`core::DeviceSlice`]. `core::DeviceSlice` is only a raw pointer + length
//! helper and does not carry null-mask, offset-base, readiness, completion,
//! or execution-provenance semantics.
//!
//! Interop activation uses the shared names in [`interop::contract`]. If you
//! already have `cudf` runtime handles, use [`interop::cuda_stream_handle`]
//! and [`interop::cuda_event_handle`] to project them into the shared contract
//! explicitly. Independent producers and consumers can depend on
//! `algeon-interop` and import the same contracts from `rapids_interop`
//! without depending on `cudf`.
//!
//! [`interop::arrow_host`] is the host-copy `RecordBatch` path. Device-resident
//! data leaves `cudf` through the prepared RAPIDS exports above. Arrow is not
//! part of that path.
//!
//! # Concurrency Model: Device Memory Domains
//!
//! cudf is built for multi-GPU nodes running independent per-device work,
//! with several data pipelines sharing each GPU. Three layers carry the
//! model, and none of them takes a host-side lock around GPU work:
//!
//! - **Device domain**: each CUDA device gets exactly one memory domain
//!   (cuda resource → pool → reservation-aware adaptor). The adaptor is
//!   installed as RMM's per-device current resource when the domain is
//!   created and **never changes afterwards**. libcudf temporaries
//!   resolve through this long-lived adaptor and are accounted
//!   to the stream they allocate on. Device domains share no allocator or
//!   pool state with one another; the only cross-device state is a
//!   registry-lookup lock that never spans GPU work. cudf owns the
//!   memory management of the devices it uses;
//!   services should claim devices explicitly at startup via
//!   [`DeviceMemoryManager::install`].
//! - **Query**: one immutable [`QueryMemoryDomain`] caps allocations across
//!   its streams. Each stream has one shared [`StreamReservationLease`].
//!   Ordinary manager allocations route to that member, and stage tokens
//!   share its direct allocation resource without changing the cap.
//! - **Operation**: every operation executes with explicit
//!   device + stream + memory resource. The memory resource governs
//!   **outputs only**, as upstream libcudf's `mr` parameter does;
//!   temporaries belong to the device domain and are charged to the
//!   operation's stream.
//!
//! ```no_run
//! use cudf::{
//!     DeviceId, DeviceMemoryManager, InstallSpec, QueryMemoryDomainOptions,
//!     Runtime,
//! };
//!
//! // Service startup: one explicit memory-domain install per device.
//! DeviceMemoryManager::install(
//!     DeviceId::new(0).unwrap(),
//!     InstallSpec::default(),
//! )
//! .unwrap();
//!
//! // One query domain owns its cap; streams attach shared members.
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
//! # Stream Synchronization Strategy
//!
//! Every owned GPU object ([`core::Column`], [`core::Table`], [`DataFrame`],
//! [`core::Scalar`]) carries *retained provenance*; a record of the CUDA
//! stream and allocator handle that produced it. This differs from upstream
//! libcudf, where callers pass a stream to each function call and are
//! responsible for all cross-stream ordering themselves.
//!
//! **Single-runtime path:** Objects created inside the same explicit
//! [`Runtime`] share one retained-provenance bundle. No cross-runtime
//! synchronization overhead is incurred.
//!
//! **Mixed-provenance data operations:** When inputs from different
//! [`Runtime`] instances are combined by data operations that can safely
//! reassemble onto one output domain (for example `merge`, `concat`,
//! gather/filter-style row ops, and scalar/column replacement paths),
//! cudf automatically inserts device-side CUDA event ordering in both
//! directions (`cudaEventRecord` + `cudaStreamWaitEvent`): before the
//! operation, the output stream waits for work already enqueued on each
//! input's retained stream; after the operation is enqueued, each
//! cross-stream input's retained stream waits for the operation's completion
//! event, so dropping an input can never release its buffers under a
//! consumer that is still reading them. Neither direction blocks the calling
//! CPU thread.
//!
//! **Mixed-provenance structural operations:** Schema-changing operations
//! that retain incoming columns as-is, such as [`DataFrame::with_column`] and
//! [`DataFrame::assign`], do **not** auto-fence mixed provenance. They reject
//! cross-runtime inputs and require explicit relocation first via
//! [`DataFrame::copy_into_runtime`] or [`Series::copy_into_runtime`].
//!
//! This automatic ordering is a correctness fallback that does not improve
//! throughput. Keep collaborating objects in the same [`Runtime`] to avoid the
//! cross-runtime event ordering described above.
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
