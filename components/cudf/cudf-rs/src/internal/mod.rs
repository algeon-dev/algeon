//! Internal helpers shared across façade modules.
//!
//! This module is `pub(crate)` — nothing here is part of the public API.
//!
//! # Submodules
//!
//! - [`temp_table`] — `TempTableView`: self-borrowing temporary table view.
//! - [`resolve`] — Column-name resolution and validation helpers.
//! - [`reassemble`] — DataFrame reassembly from raw `Table` + names.

pub(crate) mod reassemble;
pub(crate) mod resolve;
pub(crate) mod temp_table;
