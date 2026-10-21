//! cuDF error projection at the native boundary.
//!
//! Provides the canonical translation of `cudf::Error` into [`QueryError`]. Both
//! the runtime and query engine invoke this projection to ensure unified classification,
//! consistent fact metadata, and uniform redaction across native failures.
//!
//! Allocation attribution derives from `cudf`. The typed FFI failure source
//! establishes whether physical exhaustion or query-limit rejection occurred,
//! whereas budget-scope annotations record usage snapshots of traversed scopes
//! (see `cudf::allocation_attribution`). Attempt retry decisions follow sealed attempt
//! advice in coordination with error status.

use std::borrow::Cow;

use cudf::core::error::FactKeyName;
use query_planning::error::{MemoryFailure, MemoryUsage, QueryError, QueryErrorCode, redact_uri_tokens};

/// Wraps a `cudf::Error` at the native-engine boundary.
///
/// `operation` should be the stable dotted id for the native operation that
/// called cuDF, and `message` should describe the failure for humans. cuDF
/// classification and a URI-redacted message are preserved as `cudf_*` facts.
pub fn cudf_query_error(operation: &'static str, message: impl Into<String>, source: cudf::Error) -> QueryError {
    let error = cudf_query_error_without_source(operation, message, &source);
    error.with_source(source)
}

/// Wraps a `cudf::Error` without attaching its display text as a source.
///
/// Use this when source text may contain sensitive reader locations. It keeps
/// cuDF classification and redacted `cudf_message` facts, but omits the
/// unsanitized source chain.
pub fn cudf_query_error_without_source(
    operation: &'static str,
    message: impl Into<String>,
    source: &cudf::Error,
) -> QueryError {
    cudf_query_error_with_code(query_error_code_for_cudf_error(source), operation, message, source)
}

/// Wraps a rejected Arrow-to-cuDF dtype conversion as
/// [`QueryErrorCode::UnsupportedType`].
///
/// Native type support reports unsupported Arrow types as `UnsupportedType`.
/// [`query_error_code_for_cudf_error`] maps cuDF `UnsupportedDtype` to the
/// `CudfUnsupported` code. The `cudf_*` facts and source chain match
/// [`cudf_query_error`], plus `cudf_reason` when cuDF attached one.
pub fn cudf_unsupported_type_error(
    operation: &'static str,
    message: impl Into<String>,
    source: cudf::Error,
) -> QueryError {
    let mut error = cudf_query_error_with_code(QueryErrorCode::UnsupportedType, operation, message, &source);
    if let Some(reason) = source.detail_value("reason") {
        error = error.with_fact("cudf_reason", reason);
    }
    error.with_source(source)
}

fn cudf_query_error_with_code(
    code: QueryErrorCode,
    operation: &'static str,
    message: impl Into<String>,
    source: &cudf::Error,
) -> QueryError {
    let mut error = QueryError::new(code, operation, message)
        .with_fact("cudf_code", source.code())
        .with_fact("cudf_kind", source.kind())
        .with_fact("cudf_status", source.status())
        .with_fact("cudf_operation", source.operation());
    error = with_cudf_ffi_detail_facts(error, source);
    error = with_cudf_column_element_limit_facts(error, source);
    if let Some(detail) = source.budget_exhaustion() {
        error = error.with_memory_failure(memory_failure(detail));
    }
    match cudf_message_fact(source) {
        Some(cudf_message) => error.with_fact("cudf_message", cudf_message),
        None => error,
    }
}

/// Classifies a `cudf::Error` into the native-engine error taxonomy.
///
/// Notable permanent mappings: `NotFound` is treated as invalid input,
/// `ExecutionContext` is treated as an invariant violation, and unknown kinds
/// fall back to dependency failure.
pub fn query_error_code_for_cudf_error(error: &cudf::Error) -> QueryErrorCode {
    match cudf::allocation_attribution(error) {
        cudf::CudfAllocationAttribution::ConfirmedQueryLimit => {
            return QueryErrorCode::QueryMemoryLimitExceeded;
        },
        cudf::CudfAllocationAttribution::ConfirmedPhysical => return QueryErrorCode::DeviceOutOfMemory,
        cudf::CudfAllocationAttribution::UnattributedAllocationFailure => {
            return QueryErrorCode::DeviceAllocationFailed;
        },
        cudf::CudfAllocationAttribution::NonAllocationFailure => {},
    }
    match cudf::classify_cudf_error(error) {
        cudf::CudfErrorClassification::SizeOverflow => QueryErrorCode::SizeOverflow,
        cudf::CudfErrorClassification::HostMaterializationLimitExceeded => {
            QueryErrorCode::HostMaterializationLimitExceeded
        },
        cudf::CudfErrorClassification::ResourceExhausted => QueryErrorCode::DeviceAllocationFailed,
        cudf::CudfErrorClassification::TemporaryFailure => QueryErrorCode::CudfTemporaryFailure,
        cudf::CudfErrorClassification::PersistentFailure => QueryErrorCode::DeviceFault,
        cudf::CudfErrorClassification::InvalidInput => QueryErrorCode::CudfInvalidInput,
        cudf::CudfErrorClassification::Unsupported => QueryErrorCode::CudfUnsupported,
        cudf::CudfErrorClassification::DependencyFailure => QueryErrorCode::CudfDependencyFailure,
        cudf::CudfErrorClassification::InvariantViolation => QueryErrorCode::InternalInvariantViolation,
    }
}

/// Converts the budget scope a cuDF error passed through into planning's
/// backend-neutral memory context.
pub fn memory_failure(detail: &cudf::BudgetExhaustionDetail) -> MemoryFailure {
    MemoryFailure {
        boundary: Cow::Owned(detail.boundary().to_owned()),
        operation: Cow::Borrowed(detail.operation()),
        usage: detail.usage().map(|usage| MemoryUsage {
            limit_bytes: usage.allocation_limit_bytes,
            peak_bytes: usage.peak_bytes,
            peak_aligned_bytes: usage.peak_aligned_bytes,
            current_bytes: usage.current_bytes,
            current_aligned_bytes: usage.current_aligned_bytes,
            failed_allocation_count: usage.failed_allocation_count,
            largest_failed_allocation_bytes: usage.largest_failed_allocation_bytes,
        }),
    }
}

/// Returns the trimmed foreign cuDF message, redacting URI-like tokens.
pub fn cudf_message_fact(source: &cudf::Error) -> Option<String> {
    let message = source.message().trim();
    if message.is_empty() {
        return None;
    }
    Some(redact_uri_tokens(message))
}

/// Projects the column, element count, limit and row cap of a cuDF size
/// overflow, which the client message of `size_overflow` reads.
fn with_cudf_column_element_limit_facts(mut error: QueryError, source: &cudf::Error) -> QueryError {
    if source.code() != cudf::ErrorCode::SizeOverflow {
        return error;
    }
    if let Some(column) = source.fact_value(FactKeyName::ColumnName) {
        error = error.with_fact("column", column.to_string());
    }
    for key in ["element_count", "element_limit", "row_cap"] {
        if let Some(value) = source.detail_value(key) {
            error = error.with_fact(key, value);
        }
    }
    error
}

fn with_cudf_ffi_detail_facts(mut error: QueryError, source: &cudf::Error) -> QueryError {
    let Some(detail) = source.ffi_detail() else {
        return error;
    };
    error = error
        .with_fact("failure_source", detail.failure_source)
        .with_fact("requested_allocation_bytes", detail.requested_bytes)
        .with_fact("allocation_alignment", detail.alignment)
        .with_fact("cuda_status", detail.cuda_error);
    if detail.domain_id != 0 {
        error = error.with_fact("allocation_domain_id", detail.domain_id);
    }
    if detail.member_id != 0 {
        error = error.with_fact("allocation_member_id", detail.member_id);
    }
    if detail.generation != 0 {
        error = error.with_fact("allocation_domain_generation", detail.generation);
    }
    error
}

#[cfg(test)]
mod tests;
