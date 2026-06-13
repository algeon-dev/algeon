//! Low-level FFI bindings to the cuDF C shim library (`libcudf_rust`).
//!
//! This crate provides raw `extern "C"` function declarations matching the
//! C shim layer that wraps cuDF's C++ API, plus borrowed runtime-handle
//! adapters and typed `_ex` option builders. It is not intended to be used
//! directly as the owning API — use the `cudf` crate for safe owner types and
//! higher-level semantics.
//!
//! Borrowed stream and memory-resource owner-handle tokens can be attached to a
//! size-tagged execution contract (`cudf_exec_options_t`) for `_ex` entry
//! points. Passing a null options pointer, or null handles inside the struct,
//! keeps the legacy behavior of using cuDF's default stream and current device
//! memory resource.

#![allow(non_camel_case_types)]

mod execution;
mod ffi;
mod types;

pub use execution::*;
pub use ffi::*;
pub use types::*;
