//! cuDF error projection at the native boundary.
//!
//! This is the single place a `cudf::Error` becomes a [`QueryError`]. The
//! runtime and the engine both call it, so one classification, one fact set,
//! and one redaction policy cover every native failure.
//!
//! Allocation attribution is owned by `cudf`: the typed FFI failure
//! source proves a query-limit rejection or physical exhaustion, while a
//! budget-scope annotation is only a usage snapshot of the scope the error
//! passed through. See `cudf::allocation_attribution`. Sealed attempt advice,
//! not this status alone, governs retry; `Persistent` requires escalation or
//! intervention.

use query_planning::error::{QueryError, QueryErrorCode, redact_uri_tokens};

/// Wraps a `cudf::Error` at the native-engine boundary.
///
/// `operation` should be the stable dotted id for the native operation that
/// called cuDF, and `message` should describe the failure for humans. cuDF
/// classification and a URI-redacted message are preserved as `cudf_*` facts.
pub fn cudf_query_error(operation: &'static str, message: impl Into<String>, source: cudf::Error) -> QueryError {
    let error = cudf_query_error_without_source(operation, message, &source);
    error.with_source(source)
}

/// Wraps a `cudf::Error` and preserves budget-token diagnostics when the lower
/// layer attached them.
pub fn cudf_query_error_with_budget_details(
    operation: &'static str,
    message: impl Into<String>,
    source: cudf::Error,
) -> QueryError {
    let budget_details = cudf_budget_details(&source);
    let mut error = cudf_query_error(operation, message, source);
    for (key, value) in budget_details {
        error = error.with_fact(key, value);
    }
    error
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
/// Native type support reports unsupported Arrow types as `UnsupportedType`
/// rather than the `CudfUnsupported` code [`query_error_code_for_cudf_error`]
/// assigns to cuDF `UnsupportedDtype`. The `cudf_*` facts and source chain
/// match [`cudf_query_error`], plus `cudf_reason` when cuDF attached one.
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
    match cudf_message_fact(source) {
        Some(cudf_message) => error.with_fact("cudf_message", cudf_message),
        None => error,
    }
}

/// Wraps a `cudf::Error` without its source chain, preserving budget-token
/// diagnostics when the lower layer attached them.
pub fn cudf_query_error_without_source_with_budget_details(
    operation: &'static str,
    message: impl Into<String>,
    source: &cudf::Error,
) -> QueryError {
    let budget_details = cudf_budget_details(source);
    let mut error = cudf_query_error_without_source(operation, message, source);
    for (key, value) in budget_details {
        error = error.with_fact(key, value);
    }
    error
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

/// Returns the trimmed foreign cuDF message, redacting URI-like tokens.
pub fn cudf_message_fact(source: &cudf::Error) -> Option<String> {
    let message = source.message().trim();
    if message.is_empty() {
        return None;
    }
    Some(redact_uri_tokens(message))
}

fn cudf_budget_details(source: &cudf::Error) -> Vec<(&'static str, String)> {
    source
        .details()
        .iter()
        .filter(|(key, _)| {
            matches!(
                *key,
                "budget_boundary"
                    | "budget_bytes"
                    | "budget_enforcement"
                    | "budget_status"
                    | "budget_peak_bytes"
                    | "budget_current_bytes"
                    | "budget_peak_aligned_bytes"
                    | "budget_current_aligned_bytes"
                    | "budget_failed_allocation_count"
                    | "budget_largest_failed_allocation_bytes"
            )
        })
        .map(|(key, value)| (*key, value.clone()))
        .collect()
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
