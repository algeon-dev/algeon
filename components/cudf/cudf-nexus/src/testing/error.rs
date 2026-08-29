use crate::{BudgetExhaustionDetail, CudfErrorDetail, Error, ErrorCode};

/// Testing-only: construct a truthful `cudf::Error` without exposing the
/// crate-internal write surface on the primary API.
pub fn new_with_for_test(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Error {
    crate::core::error::testing_hooks::new_with_for_test(code, operation, message)
}

/// Testing-only: attach a typed FFI failure payload, mirroring the payload
/// copied at the failed native shim boundary.
pub fn with_ffi_detail_for_test(error: Error, detail: CudfErrorDetail) -> Error {
    crate::core::error::testing_hooks::with_ffi_detail_for_test(error, detail)
}

/// Testing-only: build the typed budget-exhaustion counters attached by a
/// hard budget token's `annotate_error` path.
pub fn budget_exhaustion_detail_for_test(
    hard_budget_bytes: usize,
    peak_bytes: usize,
    peak_aligned_bytes: usize,
    current_bytes: usize,
    largest_failed_allocation_bytes: usize,
    failed_allocation_count: usize,
    boundary: &'static str,
    operation: &'static str,
) -> BudgetExhaustionDetail {
    crate::core::error::testing_hooks::budget_exhaustion_detail_for_test(
        hard_budget_bytes,
        peak_bytes,
        peak_aligned_bytes,
        current_bytes,
        largest_failed_allocation_bytes,
        failed_allocation_count,
        boundary,
        operation,
    )
}

/// Testing-only: attach typed budget-exhaustion counters, mirroring the
/// annotation a budget scope adds to every error that passes through it.
pub fn with_budget_exhaustion_for_test(error: Error, detail: BudgetExhaustionDetail) -> Error {
    crate::core::error::testing_hooks::with_budget_exhaustion_for_test(error, detail)
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
