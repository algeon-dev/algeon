//! Safe Rust bindings for cuDF — GPU-accelerated DataFrames.
//!
//! Start with [`DataFrame`], [`Series`], and [`Runtime`]. They provide a
//! cuDF-Python-like experience with immutable-value semantics, `Arc`-shared
//! columns, and explicit runtime ownership for GPU execution resources. Use
//! [`core::Column`]/[`core::Table`] only when you need fine-grained GPU
//! memory control.
//!
//! The crate root is intentionally narrow. DataFrame-specific option and
//! result types live under [`frame::dataframe`]; factorization-specific types
//! live under [`frame::factorization`].
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
//! let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();
//!
//! assert_eq!(df.num_columns(), 2);
//! assert_eq!(df.num_rows(), 3);
//!
//! // Retrieve a column back as a Series (zero-copy, Arc-shared).
//! let score = df.series("score").unwrap();
//! assert_eq!(score.len(), 3);
//!
//! // Common reductions can materialize directly to host values.
//! let total = score.sum_value::<f64>().unwrap();
//! let mean = score.mean_value().unwrap();
//!
//! assert_eq!(total, 60.0);
//! assert_eq!(mean, Some(20.0));
//! ```
//!
//! # Primary Entry Points
//!
//! - [`DataFrame`] — A named, ordered collection of GPU columns with
//!   immutable-value semantics.
//! - [`Series`] — A named, `Arc`-shared GPU column; the primary column type.
//! - [`Runtime`] — Owns the default CUDA stream, memory resource, and retained
//!   provenance for a workflow.
//!
//! # Root-Level Operations
//!
//! - [`SortOptions`] / [`DataFrame::sort_values`] — GPU-accelerated sorting.
//! - [`MergeOptions`] / [`DataFrame::merge`] — GPU-accelerated joins.
//! - [`GroupBy`] / [`DataFrame::groupby`] — GPU-accelerated groupby aggregation.
//! - [`LookupTable`] / [`Series::lookup`] / [`DataFrame::with_lookup`] —
//!   row-aligned lookup operators and reusable mappings.
//! - [`Series::sum_value`], [`Series::mean_value`], [`Series::minmax_values`]
//!   — high-level reductions that return host values directly.
//!
//! # Specialized High-Level Types
//!
//! - [`frame::dataframe`] — DataFrame-specific builders and result snapshots
//!   such as [`frame::dataframe::MeltOptions`],
//!   [`frame::dataframe::DropNaOptions`], and
//!   [`frame::dataframe::InfoDetail`].
//! - [`frame::factorization`] — typed categorical encoding results and
//!   reusable mappings such as [`frame::factorization::Factorized`],
//!   [`frame::factorization::Codebook`], and
//!   [`frame::factorization::EncodedSeries`].
//!
//! # Advanced Usage
//!
//! The `core` types give direct control over owned GPU objects, device-memory
//! descriptors, and import/export boundaries. Most users should not need these.
//! Use [`Runtime`] to own default execution resources plus retained provenance
//! for a pipeline, and [`ExecutionOverrides`] when you need a borrowed
//! per-operation override. For `Series`, prefer explicit methods such as
//! `series.sum_in(&ctx)` or `series.str().contains_in("x", &ctx)`; for
//! higher-level table-style flows, use scoped surfaces such as
//! [`DataFrame::with_execution`].
//!
//! - [`core::Column`] — An owned GPU column (analogous to `cudf::column`).
//! - [`core::Table`] — An owned GPU table (collection of columns).
//! - [`core::Scalar`] — A single value on the GPU. Most high-level reduction
//!   workflows can stay on [`Series`] via `*_value` helpers and only drop to
//!   `Scalar` when you need explicit device-resident scalar behavior.
//! - [`Runtime`] / [`ExecutionOverrides`] — explicit stream and allocator control.
//! - [`interop::arrow_host`] — Arrow `RecordBatch` conversion through host memory.
//! - [`interop`] — `cudf-rs`-specific interop adapters, runtime-backed
//!   release helpers, CUDA handle projection, and Arrow C Device ABI glue.
//!   Import shared contract nouns from [`rapids_interop`].
//! - For lower-level CUDA stream and event management, see the [`runtime`] module.
//!
//! If you need to hand device-backed cudf data to another Rust GPU library,
//! prefer the shared descriptors and prepared export types in
//! [`rapids_interop`] plus the adapter methods in [`interop`] over
//! [`core::DeviceSlice`]. `core::DeviceSlice` is only a raw pointer + length
//! helper and does not carry null-mask, offset-base, readiness, completion,
//! or execution-provenance semantics.
//!
//! Interop activation uses the shared names in [`rapids_interop`]. If you
//! already have `cudf-rs` runtime handles, use [`interop::cuda_stream_handle`]
//! and [`interop::cuda_event_handle`] to project them into the shared contract
//! explicitly.
//!
//! Arrow naming can be easy to misread here:
//! [`interop::arrow_host`] is the host-copy `RecordBatch` path, while
//! [`interop::arrow_device`] is the Arrow C Device ABI path for
//! device-backed interop.
//!
//! # Stream Synchronization Strategy
//!
//! Every owned GPU object ([`core::Column`], [`core::Table`], [`DataFrame`],
//! [`core::Scalar`]) carries *retained provenance* — a record of the CUDA
//! stream and memory resource that produced it. This differs from upstream
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
//! cudf-rs automatically inserts a device-side CUDA event barrier from each
//! secondary input stream into the authoritative output stream
//! (`cudaEventRecord` + `cudaStreamWaitEvent`) before the operation
//! proceeds. This preserves correctness without blocking the calling CPU
//! thread.
//!
//! **Mixed-provenance structural operations:** Schema-changing operations
//! that retain incoming columns as-is, such as [`DataFrame::with_column`] and
//! [`DataFrame::assign`], do **not** auto-fence mixed provenance. They reject
//! cross-runtime inputs and require explicit relocation first via
//! [`DataFrame::copy_into_runtime`] or [`Series::copy_into_runtime`].
//!
//! This automatic ordering remains a correctness fallback, not a throughput
//! feature. For maximum GPU concurrency and the clearest mental model, keep
//! collaborating objects in the same [`Runtime`].
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
pub mod frame;
pub mod interop;
// Internal helpers — not part of the public API.
pub(crate) mod internal;
pub mod io;
pub mod ops;
pub mod runtime;
#[doc(hidden)]
pub mod testing;

// — Primary entry points —
// — Types and error —
pub use core::{
    error::{Error, ErrorCode, ErrorKind, ErrorStatus, Result},
    scalar::IntoRuntimeScalar,
    types::{CudfType, DataType, NullEquality, NullPolicy, TimestampResolution, TypeId},
};

pub use frame::{
    dataframe::DataFrame,
    series::{HostColumnData, IntoSeries, Series},
};
// — High-level operations —
pub use ops::Expr;
pub use ops::{
    groupby::{AggSpec, GroupBy, GroupByAgg, GroupByOptions},
    join::{JoinType, MergeOptions},
    lookup::LookupTable,
    replace::FillPolicy,
    sorting::SortOptions,
    stream_compaction::DuplicateKeepOption,
    strings::{StringSliceOptions, StripSide},
};
pub use runtime::{
    BoundaryKind, BoundaryObservation, BoundaryObserver, DeviceId, ExecutionOverrides, ExecutionPlacement,
    QueryExecutionContext, Runtime, RuntimeBuilder, StreamDependencyOutcome,
};
