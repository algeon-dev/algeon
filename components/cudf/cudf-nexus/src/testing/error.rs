use crate::{Error, ErrorCode};

/// Testing-only: construct a truthful `cudf::Error` without exposing the
/// crate-internal write surface on the primary API.
pub fn new_with_for_test(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Error {
    crate::core::error::testing_hooks::new_with_for_test(code, operation, message)
}

/// Testing-only: invoke the crate-internal `InteropError -> cudf::Error`
/// translator so integration tests can lock down boundary translation
/// contracts without exposing a public construction helper.
pub fn from_interop_in_for_test(error: rapids_interop::InteropError, operation: &'static str) -> Error {
    crate::core::error::testing_hooks::from_interop_in_for_test(error, operation)
}

/// Testing-only: re-anchor an error at a new `&'static str` operation.
#[must_use]
pub fn reanchor_for_test(err: Error, operation: &'static str) -> Error {
    crate::core::error::testing_hooks::reanchor_for_test(err, operation)
}

/// Testing-only: build the canonical foreign invariant contract error shape.
pub fn foreign_invariant_violation_for_test() -> Error {
    crate::core::error::testing_hooks::foreign_invariant_violation_for_test()
}

/// Testing-only: fill missing foreign invariant contract details without
/// overwriting existing payload.
pub fn annotate_foreign_invariant_contract_if_absent_for_test() -> Error {
    crate::core::error::testing_hooks::annotate_foreign_invariant_contract_if_absent_for_test()
}
