//! Safe Rust bindings for cuDF — GPU-accelerated DataFrames.
//!
//! This crate provides an idiomatic Rust API over the cuDF C++ library via
//! the `cudf-sys` FFI crate. The primary user-facing types are [`DataFrame`]
//! and [`Series`], which provide a cuDF-Python-like experience with zero-copy
//! metadata operations and GPU-only data operations.
//!
//! # Key Types
//!
//! **Façade layer** (start here):
//! - [`DataFrame`] — A named, ordered collection of GPU columns with
//!   immutable-value semantics.
//! - [`Series`] — A named, `Arc`-shared GPU column; the primary column type.
//!
//! **Core layer** (for advanced / low-level use):
//! - [`Column`] — An owned GPU column (analogous to `cudf::column`).
//! - [`ColumnView`] — A non-owning borrow of a column, tied by lifetime.
//! - [`Table`] — An owned GPU table (collection of columns).
//! - [`TableView`] — A non-owning borrow of a table.
//! - [`Scalar`] — A single value on the GPU.
//! - [`CudaStream`] — A CUDA stream for async operations.
//!
//! # Example
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
//! assert_eq!(df.num_rows().unwrap(), 3);
//!
//! // Retrieve a column back as a Series (zero-copy, Arc-shared).
//! let score = df.series("score").unwrap();
//! assert_eq!(score.len().unwrap(), 3);
//! ```

#[cfg(feature = "arrow")]
pub mod arrow;
pub mod binary;
pub mod column;
pub mod concatenate;
pub mod copying;
pub mod dataframe;
pub mod dictionary;
pub mod error;
pub mod filter;
pub mod groupby;
pub mod hashing;
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
pub mod sorting;
pub mod stream;
pub mod stream_compaction;
pub mod table;
pub mod types;
pub mod unary;

// Re-export façade types at crate root.
// Future phases will add: SortOptions, MergeOptions, JoinType, Agg, AggSpec, GroupBy, etc.

// Re-export core types at crate root.
pub use column::{Column, ColumnView, DeviceSlice};
pub use dataframe::DataFrame;
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use scalar::Scalar;
pub use series::Series;
pub use stream::CudaStream;
pub use table::{Table, TableView};
pub use types::{CudfType, DataType, TypeId};
