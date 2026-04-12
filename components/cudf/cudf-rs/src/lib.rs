//! Safe Rust bindings for cuDF — GPU-accelerated DataFrames.
//!
//! Start with [`DataFrame`] and [`Series`]. They provide a cuDF-Python-like
//! experience with immutable-value semantics, `Arc`-shared columns, and
//! GPU-only data operations. Use [`core::Column`]/[`core::Table`] only when
//! you need fine-grained GPU memory control.
//!
//! # Quick Start
//!
//! ```no_run
//! use cudf::{DataFrame, Runtime};
//!
//! // Build an explicit runtime for this workflow.
//! let rt = Runtime::new().unwrap();
//!
//! // Create named Series from host data inside that runtime.
//! let ids = rt.series("id", &[1i32, 2, 3]).unwrap();
//! let scores = rt.series("score", &[10.0f64, 20.0, 30.0]).unwrap();
//!
//! // Build a DataFrame from Series — names come from the Series.
//! let df = DataFrame::from_series([ids, scores]).unwrap();
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
//! - [`Series::sum_value`], [`Series::mean_value`], [`Series::minmax_values`]
//!   — high-level reductions that return host values directly.
//!
//! # Advanced Usage
//!
//! The `core` types give direct control over GPU memory layout and lifetimes.
//! Most users should not need these.
//! Use [`Runtime`] to own default execution resources plus retained provenance
//! for a pipeline, and [`ExecutionContext`] when you need a borrowed
//! per-operation override.
//!
//! - [`core::Column`] — An owned GPU column (analogous to `cudf::column`).
//! - [`core::ColumnView`] — A non-owning borrow of a column, tied by lifetime.
//! - [`core::Table`] — An owned GPU table (collection of columns).
//! - [`core::TableView`] — A non-owning borrow of a table.
//! - [`core::Scalar`] — A single value on the GPU. Most high-level reduction
//!   workflows can stay on [`Series`] via `*_value` helpers and only drop to
//!   `Scalar` when you need explicit device-resident scalar behavior.
//! - [`Runtime`] / [`ExecutionContext`] — explicit stream and allocator control.
//! - [`DataFrame::table_view`] — zero-copy bridge from high-level
//!   `DataFrame` into low-level [`core::TableView`] workflows.
//! - [`interop`] — the preferred external interop boundary for the shared
//!   borrowed compatibility profile, producer-rooted prepared exports, and
//!   activated read-lease APIs.
//! - For lower-level CUDA stream and event management, see the [`runtime`] module.
//!
//! If you need to hand device-backed cudf data to another Rust GPU library,
//! prefer the shared descriptors and prepared export types in [`interop`] over
//! [`core::DeviceSlice`]. `core::DeviceSlice` is only a raw pointer + length
//! helper and does not carry null-mask, offset-base, readiness, completion,
//! or execution-provenance semantics.
//!
//! For new interop APIs, prefer the shared names in [`interop`]. The
//! `interop::Runtime*` names remain available as CUDA-specific compatibility
//! wrappers over the same shared completion and liveness contract; they are not
//! a separate interop model.
//!
//! # Stream Synchronization Strategy
//!
//! Every owned GPU object ([`core::Column`], [`core::Table`], [`DataFrame`],
//! [`core::Scalar`]) carries *retained provenance* — a record of the CUDA
//! stream and memory resource that produced it. This differs from upstream
//! libcudf, where callers pass a stream to each function call and are
//! responsible for all cross-stream ordering themselves.
//!
//! **Single-stream (default) path:** Objects created without an explicit
//! [`Runtime`] use the process-global default stream. No synchronization
//! overhead is incurred.
//!
//! **Mixed-provenance operations:** When inputs from different [`Runtime`]
//! instances are combined (e.g. `merge`, `concat`), cudf-rs automatically
//! fences the secondary stream with `cudaStreamSynchronize` before the
//! operation proceeds on the output stream. This is a *host fence*: it
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
//! let s0 = rt0.series("a", &[1i32, 2, 3]).unwrap();
//! let s1 = rt1.series("b", &[4i32, 5, 6]).unwrap();
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
// — Types and error —
pub use core::{
    error::{Error, ErrorKind, ErrorStatus, Result},
    types::{CudfType, DataType, NullEquality, NullPolicy, TimestampResolution, TypeId},
};

pub use frame::{
    dataframe::{
        ColumnMemoryAccounting, CsvReadOptions, CsvWriteOptions, DataFrame, DataFrameColumnInfo, DataFrameInfo,
        DataFrameJoinResult, DataFrameMemoryAccounting, DataTypeCount, DropDuplicatesOptions, DropNaHow, DropNaOptions,
        InfoDetail, JsonReadOptions, JsonWriteOptions, MeltOptions, ParquetReadOptions, ParquetWriteOptions,
    },
    series::{HostColumnData, IntoSeries, Series},
};
// — High-level operations —
pub use ops::Expr;
pub use ops::{
    groupby::{AggSpec, GroupBy, GroupByAgg, GroupByOptions},
    join::{JoinType, MergeOptions},
    replace::FillPolicy,
    sorting::SortOptions,
    stream_compaction::DuplicateKeepOption,
    strings::StripSide,
};
#[doc(hidden)]
pub use runtime::__DebugExecutionContext;
pub use runtime::{DeviceId, ExecutionContext, Runtime, RuntimeBuilder};

#[doc(hidden)]
pub fn __debug_validity_to_host_null_mask_bytes(validity: &[bool]) -> Vec<u8> {
    crate::core::types::validity_to_ffi_bitmask(validity)
}
