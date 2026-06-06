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
//! - [`SeriesView`] — A named, borrowed view of a column.
//! - [`DataFrameView`] — A named, borrowed view of all columns in a DataFrame.
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
//! Use [`Runtime`] to seed an explicit execution domain, and
//! [`ExecutionContext`] when you need a borrowed per-operation override.
//!
//! - [`Column`] — An owned GPU column (analogous to `cudf::column`).
//! - [`ColumnView`] — A non-owning borrow of a column, tied by lifetime.
//! - [`Table`] — An owned GPU table (collection of columns).
//! - [`TableView`] — A non-owning borrow of a table.
//! - [`Scalar`] — A single value on the GPU.
//! - [`Runtime`] / [`ExecutionContext`] — explicit stream and allocator control.
//! - For lower-level CUDA stream management, see the [`runtime`] module.

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
    types::{CudfType, DataType, NullEquality, NullPolicy, TypeId},
};

pub use frame::{
    dataframe::{
        ColumnMemoryAccounting, DataFrame, DataFrameColumnInfo, DataFrameCsvReaderOptions, DataFrameCsvWriterOptions,
        DataFrameInfo, DataFrameJoinResult, DataFrameMemoryAccounting, DataFrameView, DataTypeCount, DistinctOptions,
        DropNaOptions, InfoDetail,
    },
    series::{IntoSeries, Series, SeriesView},
};
// — High-level operations —
pub use ops::Expr;
pub use ops::{
    groupby::{Agg, AggSpec, GroupBy, GroupByOptions},
    join::{JoinType, MergeOptions},
    sorting::SortOptions,
};
#[doc(hidden)]
pub use runtime::__DebugExecutionContext;
pub use runtime::{ExecutionContext, Runtime, RuntimeBuilder};
