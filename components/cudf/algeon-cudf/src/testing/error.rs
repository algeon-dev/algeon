use crate::{BudgetExhaustionDetail, CudfErrorDetail, Error, ErrorCode, core::error::testing_hooks};

/// Constructs a truthful `cudf::Error` without exposing the crate-internal write
/// surface on the primary API.
pub fn new_with_for_test(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Error {
    testing_hooks::new_with_for_test(code, operation, message)
}

/// Testing-only: attach a typed FFI failure payload, mirroring the payload
/// copied at the failed native shim boundary.
pub fn with_ffi_detail_for_test(error: Error, detail: CudfErrorDetail) -> Error {
    testing_hooks::with_ffi_detail_for_test(error, detail)
}

/// Testing-only: build the budget scope a token's `annotate_error` attaches,
/// with member counters that are zero unless named here.
pub fn budget_exhaustion_detail_for_test(
    member_limit_bytes: u64,
    peak_bytes: u64,
    peak_aligned_bytes: u64,
    current_bytes: u64,
    current_aligned_bytes: u64,
    largest_failed_allocation_bytes: u64,
    failed_allocation_count: u64,
    boundary: &'static str,
    operation: &'static str,
) -> BudgetExhaustionDetail {
    testing_hooks::budget_exhaustion_detail_for_test(
        member_limit_bytes,
        peak_bytes,
        peak_aligned_bytes,
        current_bytes,
        current_aligned_bytes,
        largest_failed_allocation_bytes,
        failed_allocation_count,
        boundary,
        operation,
    )
}

/// Testing-only: attach typed budget-exhaustion counters, mirroring the
/// annotation a budget scope adds to every error that passes through it.
pub fn with_budget_exhaustion_for_test(error: Error, detail: BudgetExhaustionDetail) -> Error {
    testing_hooks::with_budget_exhaustion_for_test(error, detail)
}

/// Invokes the crate-internal `InteropError -> cudf::Error` translator so
/// integration tests can lock down boundary translation contracts without
/// exposing a public construction helper.
pub fn from_interop_in_for_test(error: rapids_interop::InteropError, operation: &'static str) -> Error {
    testing_hooks::from_interop_in_for_test(error, operation)
}

/// Re-anchors an error at a new `&'static str` operation. Intended for tests.
#[must_use]
pub fn reanchor_for_test(err: Error, operation: &'static str) -> Error {
    testing_hooks::reanchor_for_test(err, operation)
}

/// Builds the canonical foreign invariant contract error shape. Intended for tests.
pub fn foreign_invariant_violation_for_test() -> Error {
    testing_hooks::foreign_invariant_violation_for_test()
}

/// Fills missing foreign invariant contract details without overwriting an
/// existing payload.
pub fn annotate_foreign_invariant_contract_if_absent_for_test() -> Error {
    testing_hooks::annotate_foreign_invariant_contract_if_absent_for_test()
}
