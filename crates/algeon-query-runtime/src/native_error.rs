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
use query_planning::error::{
    MemoryFailure, MemoryUsage, QueryError, QueryErrorCode, RAPIDS_LIBRARY_FACT, redact_uri_tokens,
};

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
/// `RapidsUnsupported` code. The `cudf_*` facts and source chain match
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
        .with_fact(RAPIDS_LIBRARY_FACT, "cudf")
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
        cudf::CudfErrorClassification::TemporaryFailure => QueryErrorCode::RapidsTemporaryFailure,
        cudf::CudfErrorClassification::PersistentFailure => QueryErrorCode::DeviceFault,
        cudf::CudfErrorClassification::InvalidInput => QueryErrorCode::RapidsInvalidInput,
        cudf::CudfErrorClassification::Unsupported => QueryErrorCode::RapidsUnsupported,
        cudf::CudfErrorClassification::DependencyFailure => QueryErrorCode::RapidsDependencyFailure,
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
mod tests {
    use std::collections::BTreeMap;

    use cudf::{io::parquet::RemoteParquetSource, testing::error::new_with_for_test};
    use query_planning::error::{QueryErrorCode, redact_uri_tokens};

    use super::{cudf_query_error, cudf_query_error_without_source};

    fn sample_cudf_error() -> cudf::Error {
        RemoteParquetSource::s3("gs://bucket/key.parquet").expect_err("non-s3 scheme must be rejected")
    }

    fn cudf_allocation_error_for_test(
        code: cudf::ErrorCode,
        failure_source: Option<cudf::CudfErrorFailureSource>,
        with_budget_scope: bool,
    ) -> cudf::Error {
        use cudf::testing::error;
        let mut error = error::new_with_for_test(code, "cudf.test.allocation_attribution", "synthetic cudf failure");
        if let Some(failure_source) = failure_source {
            error = error::with_ffi_detail_for_test(
                error,
                cudf::CudfErrorDetail {
                    status: 0,
                    failure_source,
                    domain_id: 0,
                    member_id: 0,
                    generation: 0,
                    requested_bytes: 4096,
                    alignment: 256,
                    cuda_error: -1,
                    message: "synthetic allocation failure".to_string(),
                },
            );
        }
        if with_budget_scope {
            error = error::with_budget_exhaustion_for_test(
                error,
                error::budget_exhaustion_detail_for_test(
                    1024,
                    1024,
                    1024,
                    512,
                    512,
                    4096,
                    1,
                    "test_boundary",
                    "cudf.test.allocation_attribution",
                ),
            );
        }
        error
    }

    #[test]
    fn cudf_query_error_preserves_foreign_message_as_cudf_message_fact() {
        let source = sample_cudf_error();
        let expected_message = redact_uri_tokens(source.message().trim());

        let error = cudf_query_error("query_engine.test.from_cudf", "wrapped cudf failure", source);

        assert_eq!(error.fact_value("cudf_message"), Some(expected_message.as_str()));
        assert_eq!(error.fact_value("rapids_library"), Some("cudf"));
        assert!(error.source_error().is_some());
    }

    #[test]
    fn cudf_query_error_without_source_preserves_foreign_message_without_source_chain() {
        let source = sample_cudf_error();
        let expected_message = redact_uri_tokens(source.message().trim());

        let error = cudf_query_error_without_source(
            "query_engine.test.from_cudf_without_source",
            "wrapped cudf failure",
            &source,
        );

        assert_eq!(error.fact_value("cudf_message"), Some(expected_message.as_str()));
        assert!(error.source_error().is_none());
    }

    #[test]
    fn cudf_value_errors_from_checked_arithmetic_classify_as_invalid_input() {
        for code in [cudf::ErrorCode::DivisionByZero, cudf::ErrorCode::ValueOutOfRange] {
            let source = new_with_for_test(code, "cudf.test.checked_arithmetic", "synthetic failure");

            let error = cudf_query_error("query_engine.test.from_cudf", "wrapped cudf failure", source);

            assert_eq!(error.code(), QueryErrorCode::RapidsInvalidInput, "{code}");
            assert_eq!(error.fact_value("cudf_code"), Some(code.to_string().as_str()));
            assert_eq!(error.fact_value("cudf_status"), Some("permanent"));
        }
    }

    #[test]
    fn cudf_allocation_attribution_drives_native_classification_not_scope_usage() {
        let cases = [
            (
                cudf::ErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::ReservationLimit),
                true,
                QueryErrorCode::QueryMemoryLimitExceeded,
            ),
            (
                cudf::ErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::Physical),
                true,
                QueryErrorCode::DeviceOutOfMemory,
            ),
            (
                cudf::ErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::Unknown),
                true,
                QueryErrorCode::DeviceAllocationFailed,
            ),
            (cudf::ErrorCode::OutOfMemory, None, true, QueryErrorCode::DeviceAllocationFailed),
            (cudf::ErrorCode::IoFailure, None, true, QueryErrorCode::RapidsDependencyFailure),
        ];

        for (index, (code, failure_source, with_budget_scope, expected)) in cases.into_iter().enumerate() {
            let error = cudf_query_error(
                "query_engine.test.allocation_attribution",
                "wrapped cudf failure",
                cudf_allocation_error_for_test(code, failure_source, with_budget_scope),
            );
            assert_eq!(error.code(), expected, "native mismatch for case {index}");
        }
    }

    #[test]
    fn cudf_budget_scope_becomes_typed_memory_context_projected_under_budget_keys() {
        let error = cudf_query_error(
            "query_engine.test.budget_scope",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(
                cudf::ErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::Physical),
                true,
            ),
        );

        // A scope snapshot never turns a physical failure into a cap failure.
        assert_eq!(error.code(), QueryErrorCode::DeviceOutOfMemory);
        let memory = error.context().memory.as_ref().expect("the budget scope is typed context");
        assert_eq!(memory.boundary, "test_boundary");
        assert_eq!(memory.operation, "cudf.test.allocation_attribution");
        let usage = memory.usage.expect("the scope carried member counters");
        assert_eq!((usage.limit_bytes, usage.current_aligned_bytes, usage.failed_allocation_count), (1024, 512, 1));
        let wire = error.wire_facts().map(|(key, value)| (key, value.into_owned())).collect::<BTreeMap<_, _>>();
        for (key, value) in [
            ("budget_boundary", "test_boundary"),
            ("budget_operation", "cudf.test.allocation_attribution"),
            ("budget_enforcement", "query_domain_member"),
            ("budget_status", "exceeded_or_lower_layer_rejected"),
            ("budget_bytes", "1024"),
            ("budget_peak_bytes", "1024"),
            ("budget_peak_aligned_bytes", "1024"),
            ("budget_current_bytes", "512"),
            ("budget_current_aligned_bytes", "512"),
            ("budget_largest_failed_allocation_bytes", "4096"),
            ("budget_failed_allocation_count", "1"),
        ] {
            assert_eq!(wire.get(key).map(String::as_str), Some(value), "{key}");
        }
        assert_eq!(error.fact_value("budget_bytes"), None, "budget keys are projected, not stored");

        let unscoped = cudf_query_error(
            "query_engine.test.budget_scope",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(cudf::ErrorCode::OutOfMemory, None, false),
        );
        assert!(unscoped.context().memory.is_none());
        assert!(unscoped.wire_facts().all(|(key, _)| !key.starts_with("budget_")));
    }
}
