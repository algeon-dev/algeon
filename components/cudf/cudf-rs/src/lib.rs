//! Safe Rust bindings for cuDF — GPU-accelerated DataFrames.
//!
//! This crate provides an idiomatic Rust API over the cuDF C++ library via
//! the `cudf-sys` FFI crate. It follows the same architecture as the Java
//! JNI bindings: a C shim layer wraps C++ calls with `extern "C"` linkage,
//! and Rust consumes them through FFI.
//!
//! # Key Types
//!
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
//! use cudf::column::Column;
//! use cudf::table::Table;
//!
//! // Create columns from host data.
//! let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let values = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
//!
//! // Create a table from columns.
//! let table = Table::from_columns(vec![ids, values]).unwrap();
//!
//! // Query the table through a view.
//! let view = table.view().unwrap();
//! assert_eq!(view.num_columns().unwrap(), 2);
//! assert_eq!(view.num_rows().unwrap(), 3);
//! ```

pub mod binary;
pub mod column;
pub mod concatenate;
pub mod copying;
pub mod dictionary;
pub mod error;
pub mod filter;
pub mod groupby;
pub mod hashing;
pub mod io;
pub mod join;
pub mod memory;
pub mod reduction;
pub mod replace;
pub mod scalar;
pub mod search;
pub mod sorting;
pub mod stream;
pub mod stream_compaction;
pub mod table;
pub mod types;
pub mod unary;

// Re-export key types at crate root for convenience.
pub use column::{Column, ColumnView, DeviceSlice};
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use scalar::Scalar;
pub use stream::CudaStream;
pub use table::{Table, TableView};
pub use types::{CudfType, DataType, TypeId};
