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
//!
//! - [`Column`] — An owned GPU column (analogous to `cudf::column`).
//! - [`ColumnView`] — A non-owning borrow of a column, tied by lifetime.
//! - [`Table`] — An owned GPU table (collection of columns).
//! - [`TableView`] — A non-owning borrow of a table.
//! - [`Scalar`] — A single value on the GPU.
//! - For CUDA stream management, see the [`stream`] module.

#[cfg(feature = "arrow")]
pub mod arrow;
pub mod binary;
pub mod column;
pub mod concatenate;
pub mod copying;
pub mod dataframe;
pub mod dataframe_view;
pub mod dictionary;
pub mod error;
pub(crate) mod execution_domain;
pub mod expr;
pub mod filter;
pub mod groupby;
pub mod hashing;
// Internal helpers — not part of the public API.
pub(crate) mod internal;
pub mod io;
pub mod join;
pub mod memory;
pub mod ops;
pub mod reduction;
pub mod replace;
pub mod scalar;
pub mod search;
pub mod series;
pub mod series_view;
pub mod sorting;
pub mod stream;
pub mod stream_compaction;
pub mod table;
pub mod types;
pub mod unary;

// — Primary entry points —
// — Core types (advanced — most users should prefer DataFrame/Series) —
pub use column::{Column, ColumnView};
pub use dataframe::{
    DataFrame, DataFrameCsvReaderOptions, DataFrameCsvWriterOptions, DataFrameJoinResult, DistinctOptions,
    DropNaOptions,
};
pub use dataframe_view::DataFrameView;
// — Types and error —
pub use error::{Error, ErrorKind, ErrorStatus, Result};
// — High-level operations —
pub use expr::Expr;
pub use groupby::{Agg, AggSpec, GroupBy, GroupByOptions};
pub use join::{JoinType, MergeOptions};
pub use scalar::Scalar;
pub use series::{IntoSeries, Series};
pub use series_view::SeriesView;
pub use sorting::SortOptions;
pub use table::{Table, TableView};
pub use types::{CudfType, DataType, NullEquality, NullPolicy, TypeId};
