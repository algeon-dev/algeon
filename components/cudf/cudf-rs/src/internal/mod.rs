//! Internal helpers shared across high-level API modules.
//!
//! Not part of the public API. Most submodules are `pub(crate)`.
//! [`temp_table`] is public because
//! [`TempTableView`](temp_table::TempTableView) is returned by
//! [`DataFrame::temp_table_view`](crate::DataFrame::temp_table_view).
//!
//! # Submodules
//!
//! - [`temp_table`] — `TempTableView`: self-borrowing temporary table view.
//! - [`resolve`] — Column-name resolution and validation helpers.
//! - [`reassemble`] — DataFrame reassembly from raw `Table` + names.

pub(crate) mod reassemble;
pub(crate) mod resolve;
pub mod temp_table;
