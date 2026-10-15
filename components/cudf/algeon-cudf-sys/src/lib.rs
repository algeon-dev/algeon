//! Low-level FFI bindings to the cuDF C shim library (`libcudf_rust`).
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion); workspace code
//! uses this crate for its component contracts. See the
//! [API documentation boundary](https://algeon.dev/design#api-documentation).
//!
//! Raw `extern "C"` declarations matching `libcudf_rust`, with borrowed
//! runtime-handle adapters and typed `_ex` option builders. Safe owner types
//! and query-engine contracts belong in the `cudf` crate.
//!
//! # Safety
//!
//! Raw functions in this crate are unsafe. Callers uphold handle ownership,
//! device, stream, memory-resource, completion, and callback lifetime
//! requirements documented in the C headers, [README](https://github.com/algeon-dev/algeon/blob/main/components/cudf/algeon-cudf-sys/README.md),
//! and [SAFETY.md](https://github.com/algeon-dev/algeon/blob/main/components/cudf/algeon-cudf-sys/SAFETY.md).
//!
//! Borrowed stream and memory-resource owner-handle tokens, plus an explicit
//! CUDA device ordinal, are attached to a size-tagged execution contract
//! (`cudf_exec_options_t`) for production `_ex` entry points. Execution-
//! sensitive production entries require non-null options with an explicit
//! device, stream, and memory resource; null or partial options are reserved
//! for narrow permissive diagnostics and legacy regression hooks.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::allow_attributes)] // reason = "raw C FFI bindings crate with module-level allows"

mod abi_ext;
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_safety_doc,
    clippy::unnecessary_cast
)]
mod bindings;
mod execution;

#[cfg(feature = "test-hooks")]
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_safety_doc,
    clippy::unnecessary_cast
)]
pub mod test_support;

pub use abi_ext::*;
pub use bindings::*;
pub use execution::*;
