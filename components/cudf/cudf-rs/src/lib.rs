//! Safe Rust bindings for cuDF — GPU-accelerated DataFrames.
//!
//! Start with [`DataFrame`] and [`Series`]. They provide a cuDF-Python-like
//! experience with immutable-value semantics, `Arc`-shared columns, and
//! GPU-only data operations. Use [`Column`]/[`Table`] only when you need
//! fine-grained GPU memory control.
//!
//! # Quick Start
//!
//! ```no_run
//! use cudf::{DataFrame, Series};
//!
//! // Create named Series from host data.
//! let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
//! let scores = Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();
//!
//! // Build a DataFrame from Series.
//! let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();
//!
//! assert_eq!(df.num_columns(), 2);
//! assert_eq!(df.num_rows(), 3);
//!
//! // Retrieve a column back as a Series (zero-copy, Arc-shared).
//! let score = df.series("score").unwrap();
//! assert_eq!(score.len(), 3);
//! ```
//!
//! # Primary Types
//!
//! - [`DataFrame`] — A named, ordered collection of GPU columns with
//!   immutable-value semantics.
//! - [`Series`] — A named, `Arc`-shared GPU column; the primary column type.
//!
//! # High-Level Operations
//!
//! - [`SortOptions`] / [`DataFrame::sort_values`] — GPU-accelerated sorting.
//! - [`MergeOptions`] / [`DataFrame::merge`] — GPU-accelerated joins.
//! - [`GroupBy`] / [`DataFrame::groupby`] — GPU-accelerated groupby aggregation.
//!
//! # Advanced Usage
//!
//! The core types give direct control over GPU memory layout and lifetimes.
//! Most users should not need these.
//! Use [`Runtime`] to own default execution resources plus retained provenance
//! for a pipeline, and [`ExecutionContext`] when you need a borrowed
//! per-operation override.
//!
//! - [`Column`] — An owned GPU column (analogous to `cudf::column`).
//! - [`ColumnView`] — A non-owning borrow of a column, tied by lifetime.
//! - [`Table`] — An owned GPU table (collection of columns).
//! - [`TableView`] — A non-owning borrow of a table.
//! - [`Scalar`] — A single value on the GPU.
//! - [`Runtime`] / [`ExecutionContext`] — explicit stream and allocator control.
//! - [`interop`] — the preferred external borrowed GPU-view interop boundary.
//! - For lower-level CUDA stream management, see the [`runtime`] module.
//!
//! If you need to hand device-backed cudf data to another Rust GPU library,
//! prefer the borrowed descriptors in [`interop`] over [`core::column::DeviceSlice`].
//! `DeviceSlice` is only a raw pointer + length helper and does not carry
//! null-mask, offset-base, or execution-provenance semantics.
//!
//! # Stream Synchronization Strategy
//!
//! Every owned GPU object ([`Column`], [`Table`], [`DataFrame`], [`Scalar`])
//! carries *retained provenance* — a record of the CUDA stream and memory
//! resource that produced it.  This differs from upstream libcudf, where
//! callers pass a stream to each function call and are responsible for all
//! cross-stream ordering themselves.
//!
//! **Single-stream (default) path:** Objects created without an explicit
//! [`Runtime`] use the process-global default stream.  No synchronization
//! overhead is incurred.
//!
//! **Mixed-provenance operations:** When inputs from different [`Runtime`]
//! instances are combined (e.g. `merge`, `concat`), cudf-rs automatically
//! fences the secondary stream with `cudaStreamSynchronize` before the
//! operation proceeds on the output stream.  This is a *host fence*: it
//! blocks the calling CPU thread until the secondary stream's GPU work
//! completes.
//!
//! The host fence is intentionally conservative.  It guarantees correctness
//! without requiring users to understand CUDA event semantics, but it
//! serializes cross-stream GPU execution.  For maximum GPU throughput, keep
//! all objects in the same [`Runtime`].
//!
//! A future version may replace the host fence with lighter-weight CUDA
//! event barriers (`cudaEventRecord` + `cudaStreamWaitEvent`) that avoid
//! blocking the CPU thread.
//!
//! # Multi-GPU
//!
//! Each [`Runtime`] is pinned to one CUDA device. When an operation is
//! invoked through a device-pinned runtime, the FFI layer re-enters the
//! runtime's device before enqueuing GPU work, and restores the calling
//! thread's ambient device afterward.
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
//! let s0 = Series::new_with_runtime::<i32>(&rt0, "a", &[1, 2, 3]).unwrap();
//! let s1 = Series::new_with_runtime::<i32>(&rt1, "b", &[4, 5, 6]).unwrap();
//! ```
//!
//! Objects created without an explicit [`Runtime`] use the *ambient default*
//! path, which snapshots the calling thread's current CUDA device. The
//! ambient path is suitable for single-GPU use but does not pin device
//! selection at execution time — use [`Runtime`] for multi-GPU correctness.

pub mod core;
pub mod frame;
pub mod interop;
// Internal helpers — not part of the public API.
pub(crate) mod internal;
pub mod io;
pub mod ops;
pub mod runtime;

// — Primary entry points —
// — Core types (advanced — most users should prefer DataFrame/Series) —
// — Types and error —
pub use core::{
    column::{Column, ColumnView},
    error::{Error, ErrorKind, ErrorStatus, Result},
    scalar::Scalar,
    table::{Table, TableView},
    types::{CudfType, DataType, NullEquality, NullPolicy, TimestampResolution, TypeId},
};

pub use frame::{
    dataframe::{
        ColumnMemoryAccounting, CsvReadOptions, CsvWriteOptions, DataFrame, DataFrameColumnInfo, DataFrameInfo,
        DataFrameJoinResult, DataFrameMemoryAccounting, DataTypeCount, DistinctOptions, DropNaOptions, InfoDetail,
        JsonWriteOptions, MeltOptions, ParquetWriteOptions,
    },
    series::{IntoSeries, Series},
};
pub use io::{json::JsonReaderOptions, parquet::ParquetReaderOptions};
// — High-level operations —
pub use ops::Expr;
pub use ops::{
    groupby::{Agg, AggSpec, GroupBy, GroupByOptions},
    join::{JoinType, MergeOptions},
    replace::FillPolicy,
    sorting::SortOptions,
    stream_compaction::DuplicateKeepOption,
    strings::StripSide,
};
#[doc(hidden)]
pub use runtime::__DebugExecutionContext;
pub use runtime::{DeviceId, ExecutionContext, Runtime, RuntimeBuilder};
