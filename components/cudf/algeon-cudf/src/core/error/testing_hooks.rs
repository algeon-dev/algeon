use std::{ffi::c_void, io};

use rapids_interop::InteropError;

use super::{BudgetExhaustionDetail, CudfErrorDetail, Error, ErrorCode};
use crate::core::{Result, memory::ReservationUsage, types::TypeId};

pub(crate) fn new_with_for_test(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Error {
    Error::new_with(code, operation, message)
}

/// Testing-only: attach a typed FFI failure payload, mirroring the payload
/// copied at the failed native shim boundary.
pub(crate) fn with_ffi_detail_for_test(error: Error, detail: CudfErrorDetail) -> Error {
    error.with_ffi_detail(detail)
}

/// Testing-only: build the budget scope a token's `annotate_error` attaches,
/// with member counters that are zero unless named here.
pub(crate) fn budget_exhaustion_detail_for_test(
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
    let usage = ReservationUsage {
        reservation_id: 0,
        attached: true,
        direct_total_allocations: 0,
        routed_total_allocations: 0,
        allocation_limit_bytes: member_limit_bytes,
        current_bytes,
        peak_bytes,
        total_bytes: 0,
        current_aligned_bytes,
        peak_aligned_bytes,
        total_allocations: 0,
        current_allocations: 0,
        peak_allocations: 0,
        failed_allocation_count,
        largest_failed_allocation_bytes,
    };
    BudgetExhaustionDetail::new(boundary, operation, Some(usage))
}

/// Testing-only: attach typed budget-exhaustion counters, mirroring the
/// annotation a budget scope adds to every error that passes through it.
pub(crate) fn with_budget_exhaustion_for_test(error: Error, detail: BudgetExhaustionDetail) -> Error {
    error.with_budget_exhaustion_if_absent(detail)
}

pub(crate) fn from_interop_in_for_test(error: InteropError, operation: &'static str) -> Error {
    super::from_interop_in(error, operation)
}

pub(crate) fn ffi_status_error_for_test(
    status: cudf_sys::cudf_status_t,
    message: impl Into<String>,
    operation: &'static str,
) -> Error {
    super::error_for_ffi_status(status, operation, message.into())
}

/// Translate a synthetic FFI status/message pair through the same leaf-code
/// classifier used by the production status boundary.
pub fn ffi_io_status_error_for_test(message: impl Into<String>, operation: &'static str) -> Error {
    ffi_status_error_for_test(cudf_sys::CUDF_ERROR_IO, message, operation)
}

/// Translate a synthetic CUDA FFI status/message pair through the same
/// leaf-code classifier used by the production status boundary.
pub fn ffi_cuda_status_error_for_test(message: impl Into<String>, operation: &'static str) -> Error {
    ffi_status_error_for_test(cudf_sys::CUDF_ERROR_CUDA, message, operation)
}

#[must_use]
pub(crate) fn reanchor_for_test(err: Error, operation: &'static str) -> Error {
    err.reanchor(operation)
}

pub(crate) fn foreign_invariant_violation_for_test() -> Error {
    super::foreign_invariant_violation(
        "testing::foreign_invariant_contract::build",
        "foreign metadata contract violated",
        "foreign_buffer",
        "invalid_foreign_metadata",
        "ensure the foreign buffer reports valid metadata",
    )
}

pub(crate) fn annotate_foreign_invariant_contract_if_absent_for_test() -> Error {
    super::annotate_foreign_invariant_contract_if_absent(
        Error::new_with(
            ErrorCode::InvariantViolation,
            "testing::foreign_invariant_contract::annotate",
            "annotate foreign invariant contract",
        )
        .with_detail("resource", "existing_resource")
        .with_detail("remedy", "existing_remedy"),
        "replacement_resource",
        "missing_reason",
        "replacement_remedy",
    )
}

/// Trigger the shared invariant-violation helper path used by integration
/// tests.
pub fn trigger_invariant_violation_for_test() -> Result<()> {
    Err(super::invariant_violation(
        "testing::trigger_invariant_violation_for_test",
        "injected invariant violation for tests",
    ))
}

/// Trigger the shared null-on-success foreign-handle contract path used by
/// integration tests.
pub fn trigger_null_handle_contract_error_for_test() -> Result<()> {
    super::require_non_null::<c_void>(
        std::ptr::null_mut(),
        "testing::trigger_null_handle_contract_error_for_test",
        "test_handle",
    )
    .map(|_| ())
}

/// Trigger the shared range-limit helper path used by integration tests.
pub fn trigger_range_limit_error_for_test() -> Result<()> {
    super::usize_to_i32(usize::MAX, "testing::trigger_range_limit_error_for_test", "row_count").map(|_| ())
}

/// Trigger a synthetic truthful-contract error carrying typed facts,
/// repeated details, a source chain, and multiple reanchor hops.
pub fn trigger_truthful_error_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::ColumnLengthMismatch,
        "testing::truthful_error_contract::inner",
        "column length mismatch",
    )
    .with_fact(super::FACT_EXPECTED_ROWS, 3usize)
    .with_fact(super::FACT_ACTUAL_ROWS, 2usize)
    .with_detail("detail", "alpha")
    .with_detail("detail", "beta")
    .with_source(io::Error::other("root cause"))
    .reanchor("testing::truthful_error_contract::middle")
    .reanchor("testing::truthful_error_contract::outer"))
}

/// Trigger a synthetic error that reanchors twice at the same operation so
/// integration tests can lock the no-deduplication contract.
pub fn trigger_same_operation_reanchor_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::truthful_error_contract::same",
        "same operation reanchor",
    )
    .reanchor("testing::truthful_error_contract::same")
    .reanchor("testing::truthful_error_contract::same"))
}

/// Trigger a synthetic error that applies `reanchor_if_needed` with the same
/// operation so integration tests can lock the no-op contract.
pub fn trigger_reanchor_if_needed_same_operation_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::truthful_error_contract::same_if_needed",
        "reanchor_if_needed same operation",
    )
    .reanchor_if_needed("testing::truthful_error_contract::same_if_needed"))
}

/// Trigger a synthetic error that exercises the shared `*_if_absent` helpers
/// so integration tests can lock the preserve-existing-and-fill-missing
/// contract.
pub fn trigger_if_absent_error_annotation_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::truthful_error_contract::if_absent",
        "if_absent helper contract",
    )
    .with_fact(super::FACT_KEYS_COUNT, 7usize)
    .with_detail("resource", "existing_resource")
    .with_fact_if_absent(super::FACT_KEYS_COUNT, 9usize)
    .with_fact_if_absent(super::FACT_TYPE_ID, TypeId::Int32)
    .with_detail_if_absent("resource", "replacement_resource")
    .with_detail_if_absent("reason", "missing_value"))
}

/// Trigger a synthetic error that exercises the shared `with_column` helper
/// so integration tests can lock the canonical `column_name` fact contract.
pub fn trigger_with_column_error_annotation_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::truthful_error_contract::with_column::inner",
        "with_column helper contract",
    )
    .with_column("alpha")
    .reanchor("testing::truthful_error_contract::with_column::outer"))
}

/// Attempt to write a canonical fact snake_case key through the crate-internal
/// detail writer path. The writer rejects the overlap-rule violation
/// structurally and panics; this hook lets integration tests lock that
/// behavior without exercising any public construction API.
pub fn trigger_canonical_key_in_detail_path_for_test() {
    let _ = Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::canonical_key_in_detail_path",
        "canonical key must not reach the detail path",
    )
    .with_detail("expected_rows", "3");
}
