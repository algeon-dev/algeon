use std::{ffi::c_void, io};

use rapids_interop::InteropError;

use super::{BudgetExhaustionDetail, CudfErrorDetail, Error, ErrorCode};
use crate::core::{Result, types::TypeId};

pub(crate) fn new_with_for_test(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Error {
    Error::new_with(code, operation, message)
}

/// Testing-only: attach a typed FFI failure payload, mirroring the payload
/// copied at the failed native shim boundary.
pub(crate) fn with_ffi_detail_for_test(error: Error, detail: CudfErrorDetail) -> Error {
    error.with_ffi_detail(detail)
}

/// Testing-only: build the typed budget-exhaustion counters attached by a
/// hard budget token's `annotate_error` path.
pub(crate) fn budget_exhaustion_detail_for_test(
    hard_budget_bytes: usize,
    peak_bytes: usize,
    peak_aligned_bytes: usize,
    current_bytes: usize,
    largest_failed_allocation_bytes: usize,
    failed_allocation_count: usize,
    boundary: &'static str,
    operation: &'static str,
) -> BudgetExhaustionDetail {
    BudgetExhaustionDetail::new(
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

pub(crate) fn ffi_io_status_error_for_test(message: impl Into<String>, operation: &'static str) -> Error {
    ffi_status_error_for_test(cudf_sys::CUDF_ERROR_IO, message, operation)
}

pub(crate) fn ffi_cuda_status_error_for_test(message: impl Into<String>, operation: &'static str) -> Error {
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

pub(crate) fn trigger_invariant_violation_for_test() -> Result<()> {
    Err(super::invariant_violation(
        "testing::trigger_invariant_violation_for_test",
        "injected invariant violation for tests",
    ))
}

pub(crate) fn trigger_null_handle_contract_error_for_test() -> Result<()> {
    super::require_non_null::<c_void>(
        std::ptr::null_mut(),
        "testing::trigger_null_handle_contract_error_for_test",
        "test_handle",
    )
    .map(|_| ())
}

pub(crate) fn trigger_range_limit_error_for_test() -> Result<()> {
    super::usize_to_i32(usize::MAX, "testing::trigger_range_limit_error_for_test", "row_count").map(|_| ())
}

pub(crate) fn trigger_truthful_error_contract_for_test() -> Result<()> {
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

pub(crate) fn trigger_same_operation_reanchor_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::truthful_error_contract::same",
        "same operation reanchor",
    )
    .reanchor("testing::truthful_error_contract::same")
    .reanchor("testing::truthful_error_contract::same"))
}

pub(crate) fn trigger_reanchor_if_needed_same_operation_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::truthful_error_contract::same_if_needed",
        "reanchor_if_needed same operation",
    )
    .reanchor_if_needed("testing::truthful_error_contract::same_if_needed"))
}

pub(crate) fn trigger_if_absent_error_annotation_contract_for_test() -> Result<()> {
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

pub(crate) fn trigger_with_column_error_annotation_contract_for_test() -> Result<()> {
    Err(Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::truthful_error_contract::with_column::inner",
        "with_column helper contract",
    )
    .with_column("alpha")
    .reanchor("testing::truthful_error_contract::with_column::outer"))
}

/// Attempt to write a canonical fact snake_case key through the detail path.
///
/// The crate-internal writer rejects this structurally and panics. Integration
/// tests use this hook to lock that behavior without depending on any public
/// construction API.
pub(crate) fn trigger_canonical_key_in_detail_path_for_test() {
    let _ = Error::new_with(
        ErrorCode::InvariantViolation,
        "testing::canonical_key_in_detail_path",
        "canonical key must not reach the detail path",
    )
    .with_detail("expected_rows", "3");
}
