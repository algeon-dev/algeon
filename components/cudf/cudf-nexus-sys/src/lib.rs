//! Low-level FFI bindings to the cuDF C shim library (`libcudf_rust`).
//!
//! This crate provides raw `extern "C"` function declarations matching the
//! C shim layer that wraps cuDF's C++ API, plus borrowed runtime-handle
//! adapters and typed `_ex` option builders. It is not intended to be used
//! directly as the owning API — use the `cudf` crate for safe owner types and
//! higher-level semantics.
//!
//! # Safety
//!
//! Every raw function call in this crate is unsafe. Callers must uphold the
//! handle ownership, device, stream, memory-resource, completion, and callback
//! lifetime requirements documented by the C headers. See the concise
//! [README](https://github.com/DataFusion-Nexus/datafusion-nexus/blob/main/components/cudf/cudf-nexus-sys/README.md)
//! and [SAFETY.md](https://github.com/DataFusion-Nexus/datafusion-nexus/blob/main/components/cudf/cudf-nexus-sys/SAFETY.md)
//! before adding or changing raw-handle code.
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
