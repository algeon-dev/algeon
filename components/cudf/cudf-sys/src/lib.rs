//! Low-level FFI bindings to the cuDF C shim library (`libcudf_rust`).
//!
//! This crate provides raw `extern "C"` function declarations matching the
//! C shim layer that wraps cuDF's C++ API, plus borrowed runtime-handle
//! adapters and typed `_ex` option builders. It is not intended to be used
//! directly as the owning API — use the `cudf` crate for safe owner types and
//! higher-level semantics.
//!
//! Borrowed stream and memory-resource owner-handle tokens, plus an explicit
//! CUDA device ordinal, are attached to a size-tagged execution contract
//! (`cudf_exec_options_t`) for production `_ex` entry points. Execution-
//! sensitive production entries require non-null options with an explicit
//! device, stream, and memory resource; null or partial options are reserved
//! for narrow permissive diagnostics and legacy regression hooks.

#![allow(non_camel_case_types)]

mod execution;
mod ffi;
mod types;

pub use execution::*;
pub use ffi::*;
pub use types::*;
