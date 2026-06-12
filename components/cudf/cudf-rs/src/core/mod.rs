//! Core ownership, memory, and type primitives.

pub mod column;
mod debug;
pub mod error;
pub mod memory;
pub mod scalar;
pub mod table;
pub mod types;

pub(crate) use column::ColumnView;
pub use column::{Column, DeviceBuffer, DeviceSlice};
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use memory::DeviceMemoryResource;
pub use scalar::Scalar;
pub use table::Table;
pub(crate) use table::TableView;
pub use types::{CudfType, DataType, NullEquality, NullPolicy, TimestampResolution, TypeId};
