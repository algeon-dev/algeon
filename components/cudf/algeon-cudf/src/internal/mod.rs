//! Internal helpers shared across high-level API modules.
//!
//! Not part of the public API. All submodules are `pub(crate)`.
//!
//! # Submodules
//!
//! - [`temp_table`]: `TempTableView`: crate-private self-borrowing temporary
//!   table view used to keep `ColumnView` inputs alive while building
//!   `TableView`s.
//! - [`resolve`]: Column-name resolution and validation helpers.
//! - [`reassemble`]: DataFrame reassembly from raw `Table` + names.

pub(crate) mod reassemble;
pub(crate) mod resolve;
pub(crate) mod temp_table;
pub(crate) mod testing_hooks;
