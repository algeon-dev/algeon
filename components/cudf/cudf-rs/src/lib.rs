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
//! - For lower-level CUDA stream management, see the [`runtime`] module.
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
        ColumnMemoryAccounting, DataFrame, DataFrameColumnInfo, DataFrameCsvReaderOptions, DataFrameCsvWriterOptions,
        DataFrameInfo, DataFrameJoinResult, DataFrameJsonReaderOptions, DataFrameJsonWriterOptions,
        DataFrameMemoryAccounting, DataframeParquetReaderOptions, DataFrameParquetWriterOptions, DataTypeCount,
        DistinctOptions, DropNaOptions, InfoDetail, MeltOptions,
    },
    series::{IntoSeries, Series},
};
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
pub use runtime::{ExecutionContext, Runtime, RuntimeBuilder};
