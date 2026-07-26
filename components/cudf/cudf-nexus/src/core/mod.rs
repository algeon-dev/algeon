//! Core ownership, memory, and type primitives.

pub mod column;
mod debug;
pub mod error;
pub mod memory;
pub(crate) mod native_teardown;
pub(crate) mod row_range;
pub mod scalar;
pub(crate) mod schema;
pub mod table;
pub(crate) mod testing_hooks;
pub mod types;

pub(crate) use column::ColumnView;
pub use column::{Column, DeviceBuffer, DeviceSlice};
pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Result};
pub use memory::{DeviceMemoryResource, DeviceMemoryResourceUsage};
pub use scalar::Scalar;
pub use table::Table;
pub(crate) use table::TableView;
pub use types::{CudfType, DataType, NullEquality, NullPolicy, TimestampResolution, TypeId};
