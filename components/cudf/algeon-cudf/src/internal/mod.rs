//! Internal helpers shared across high-level API modules.
//!
//! Not part of the public API. All submodules are `pub(crate)`.
//!
//! # Submodules
//!
//! - [`resolve`]: Column-name resolution and validation helpers.
//! - [`reassemble`]: DataFrame reassembly from raw `Table` + names.

pub(crate) mod reassemble;
pub(crate) mod resolve;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) mod testing_hooks;
