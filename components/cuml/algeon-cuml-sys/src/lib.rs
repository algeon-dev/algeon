//! Raw C bindings to the cuML native shim. Prefer the safe `cuml` crate.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, clippy::all)]

include!("bindings.rs");

#[cfg(feature = "test-hooks")]
pub mod test_support;
