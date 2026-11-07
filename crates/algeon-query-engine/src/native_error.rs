//! Native error projection.
//!
//! The runtime owns the cuDF projection boundary to ensure identical classification
//! of `cudf::Error` across the runtime and query engine. Because cuVS is scoped
//! exclusively to the engine, its error projection is defined in this module.

#[cfg(feature = "cuvs")]
use std::collections::BTreeMap;
#[cfg(any(feature = "cuvs", feature = "cuml"))]
use std::error::Error as StdError;

#[cfg(feature = "cuml")]
use cudf::CudfAllocationAttribution;
#[cfg(any(feature = "cuvs", feature = "cuml"))]
use cudf::core::error::FactKeyName;
#[cfg(feature = "cuml")]
use cuml::{ErrorCode, ErrorKind, ErrorStatus};
use query_planning::error::{MemoryFailure, MemoryUsage, QueryError, QueryErrorCode};
#[cfg(any(feature = "cuvs", feature = "cuml"))]
use query_planning::error::{RAPIDS_LIBRARY_FACT, redact_uri_tokens};
pub(crate) use query_runtime::native_error::{
    cudf_query_error_without_source, cudf_unsupported_type_error, memory_failure,
};
use query_runtime::runtime::ExecutionContext;

/// Memory-scope label of an allocation failure sampled at the attempt's
/// allocation domain. It does not name an operation's budget scope.
const ATTEMPT_DOMAIN_BOUNDARY: &str = "attempt_allocation_domain";

/// Records the attempt allocation domain's counters on an allocation failure
/// that ran outside every budget scope.
///
/// Operations executing under a budget token annotate errors with allocator counters.
/// When an unbudgeted operation fails, it reaches the plan boundary containing only
/// the requested allocation size. The domain limit reflects the attempt grant.
/// Peak usage accounts for the failing operation's working set, whereas current usage
/// excludes it because the working set is released prior to sampling here.
/// Previously recorded scopes are preserved, and sampling errors leave the error unchanged.
pub(crate) fn with_attempt_domain_memory(context: &ExecutionContext, error: QueryError) -> QueryError {
    if !matches!(
        error.code(),
        QueryErrorCode::QueryMemoryLimitExceeded
            | QueryErrorCode::DeviceOutOfMemory
            | QueryErrorCode::DeviceAllocationFailed
    ) || error.context().memory.is_some()
    {
        return error;
    }
    let Ok(Some(usage)) = context.attempt_allocation_domain_usage() else {
        return error;
    };
    let operation = error.operation();
    error.with_memory_failure(MemoryFailure {
        boundary: ATTEMPT_DOMAIN_BOUNDARY.into(),
        operation: operation.into(),
        usage: Some(MemoryUsage {
            limit_bytes: usage.allocation_limit_bytes,
            peak_bytes: usage.peak_bytes,
            peak_aligned_bytes: usage.peak_aligned_bytes,
            current_bytes: usage.current_bytes,
            current_aligned_bytes: usage.current_aligned_bytes,
            failed_allocation_count: usage.failed_allocation_count,
            largest_failed_allocation_bytes: usage.largest_failed_allocation_bytes,
        }),
    })
}

/// Wraps a `cuvs::Error` at the native vector-execution boundary.
///
/// Classification is derived only from the typed cuVS kind and status; foreign
/// message text remains diagnostic and never drives retry policy.
#[cfg(feature = "cuvs")]
pub(crate) fn cuvs_query_error(operation: &'static str, message: impl Into<String>, source: cuvs::Error) -> QueryError {
    let code = query_error_code_for_cuvs_error(&source);
    let error = QueryError::new(code, operation, message)
        .with_fact(RAPIDS_LIBRARY_FACT, "cuvs")
        .with_fact("cuvs_kind", source.kind())
        .with_fact("cuvs_status", source.status())
        .with_fact("cuvs_operation", source.operation());
    let cuvs_message = source.message().trim();
    let mut error =
        if cuvs_message.is_empty() { error } else { error.with_fact("cuvs_message", redact_uri_tokens(cuvs_message)) };
    // cuVS context is an ordered container that allows duplicate keys with
    // last-write-wins reads (`context_value`). Collapse to the effective value
    // per key before projecting into the unique fact map so a legitimately
    // repeated key (for example two `native_status` entries) converts without
    // losing the primary error.
    let mut effective_context = BTreeMap::new();
    for (key, value) in source.context().filter(|(key, _)| is_safe_cuvs_context_fact(key)) {
        effective_context.insert(key, value);
    }
    for (key, value) in effective_context {
        error = error.with_fact(key, value);
    }
    with_cudf_cause_facts(error, source.source()).with_source(source)
}

/// Wraps a prepared cuDF interop failure at a native vector boundary.
///
/// The engine validates vector schemas and selects every requested column
/// before preparing this producer-rooted export. A preparation failure at this
/// point is therefore an execution-binding invariant violation, and input
/// validation has already run.
#[cfg(feature = "cuvs")]
pub(crate) fn cudf_interop_query_error<E>(
    operation: &'static str,
    message: impl Into<String>,
    source_kind: impl Into<String>,
    source: E,
) -> QueryError
where
    E: StdError + Send + Sync + 'static,
{
    let source_kind: String = source_kind.into();
    QueryError::new(QueryErrorCode::RapidsInvariantViolation, operation, message)
        .with_fact(RAPIDS_LIBRARY_FACT, "cudf")
        .with_fact("cudf_interop_kind", source_kind)
        .with_source(source)
}

/// Classifies a `cuvs::Error` into the native-engine error taxonomy.
#[cfg(feature = "cuvs")]
pub(crate) fn query_error_code_for_cuvs_error(error: &cuvs::Error) -> QueryErrorCode {
    match error.kind() {
        cuvs::ErrorKind::InvalidInput => QueryErrorCode::RapidsInvalidInput,
        cuvs::ErrorKind::Unsupported => QueryErrorCode::RapidsUnsupported,
        cuvs::ErrorKind::OutOfMemory => QueryErrorCode::DeviceAllocationFailed,
        cuvs::ErrorKind::InvalidState => QueryErrorCode::InternalInvariantViolation,
        cuvs::ErrorKind::Unexpected => match error.status() {
            cuvs::ErrorStatus::Temporary => QueryErrorCode::RapidsTemporaryFailure,
            cuvs::ErrorStatus::Persistent => QueryErrorCode::DeviceFault,
            cuvs::ErrorStatus::Permanent => QueryErrorCode::RapidsDependencyFailure,
        },
        _ => QueryErrorCode::RapidsDependencyFailure,
    }
}

#[cfg(feature = "cuvs")]
fn is_safe_cuvs_context_fact(key: &str) -> bool {
    matches!(
        key,
        "cudf_code"
            | "cudf_kind"
            | "cudf_status"
            | "cudf_operation"
            | "interop_kind"
            | "native_status"
            | "buffer_binding"
            | "resource_binding"
            | "expected_device"
            | "actual_device"
    )
}

#[cfg(feature = "cuml")]
pub(crate) fn query_error_code_for_cuml_error(error: &cuml::Error) -> QueryErrorCode {
    classify_cuml_error(error.code(), error.kind(), error.status(), cuml::allocation_attribution(error))
}

#[cfg(feature = "cuml")]
fn classify_cuml_error(
    code: cuml::ErrorCode,
    kind: cuml::ErrorKind,
    status: cuml::ErrorStatus,
    attribution: cudf::CudfAllocationAttribution,
) -> QueryErrorCode {
    if code == ErrorCode::HostOutOfMemory {
        return QueryErrorCode::RapidsTemporaryFailure;
    }
    match attribution {
        CudfAllocationAttribution::ConfirmedQueryLimit => return QueryErrorCode::QueryMemoryLimitExceeded,
        CudfAllocationAttribution::ConfirmedPhysical => return QueryErrorCode::DeviceOutOfMemory,
        CudfAllocationAttribution::UnattributedAllocationFailure => return QueryErrorCode::DeviceAllocationFailed,
        CudfAllocationAttribution::NonAllocationFailure => {},
    }
    match kind {
        ErrorKind::InvalidInput => QueryErrorCode::RapidsInvalidInput,
        ErrorKind::Unsupported => QueryErrorCode::RapidsUnsupported,
        ErrorKind::ExecutionContext => QueryErrorCode::InternalInvariantViolation,
        ErrorKind::ResourceExhausted => QueryErrorCode::SizeOverflow,
        ErrorKind::DependencyFailure => match status {
            ErrorStatus::Temporary => QueryErrorCode::RapidsTemporaryFailure,
            ErrorStatus::Persistent => QueryErrorCode::DeviceFault,
            ErrorStatus::Permanent => QueryErrorCode::RapidsDependencyFailure,
        },
        ErrorKind::InvariantViolation => QueryErrorCode::RapidsInvariantViolation,
    }
}

#[cfg(feature = "cuml")]
pub fn cuml_query_error(operation: &'static str, message: impl Into<String>, source: cuml::Error) -> QueryError {
    let mut error = QueryError::new(query_error_code_for_cuml_error(&source), operation, message)
        .with_fact(RAPIDS_LIBRARY_FACT, "cuml")
        .with_fact("cuml_code", source.code())
        .with_fact("cuml_kind", source.kind())
        .with_fact("cuml_status", source.status())
        .with_fact("cuml_phase", source.phase())
        .with_fact("cuml_operation", source.operation());
    if !source.message().trim().is_empty() {
        error = error.with_fact("cuml_message", redact_uri_tokens(source.message().trim()));
    }
    if let Some(detail) = source.ffi_detail() {
        error = error
            .with_fact("cuml_failure_source", detail.failure_source)
            .with_fact("cuml_domain_id", detail.domain_id)
            .with_fact("cuml_member_id", detail.member_id)
            .with_fact("cuml_requested_bytes", detail.requested_bytes)
            .with_fact("cuml_cuda_error", detail.cuda_error)
            .with_fact("failure_source", detail.failure_source)
            .with_fact("requested_allocation_bytes", detail.requested_bytes);
        if detail.domain_id != 0 {
            error = error.with_fact("allocation_domain_id", detail.domain_id);
        }
        if detail.member_id != 0 {
            error = error.with_fact("allocation_member_id", detail.member_id);
        }
    }
    for (key, value) in source.details().filter(|(key, _)| {
        matches!(
            *key,
            "cudf_code"
                | "cudf_operation"
                | "reason"
                | "null_count"
                | "observed_min"
                | "observed_max"
                | "distinct_count"
                | "matrix_binding_id"
                | "resource_binding_id"
                | "matrix_device"
                | "resource_device"
        )
    }) {
        error = error.with_fact(key, value);
    }
    with_cudf_cause_facts(error, source.source()).with_source(source)
}

/// Projects the budget scope and input-validation diagnostics of every
/// `cudf::Error` in a cuVS or cuML source chain.
///
/// Both libraries wrap the cuDF error that rejected an input or ran under a
/// budget, so its details are otherwise reachable only by walking the chain.
/// The innermost budget scope wins. Projected values are counts, typed
/// reasons, and static remedies; cuDF messages stay in the source chain.
#[cfg(any(feature = "cuvs", feature = "cuml"))]
fn with_cudf_cause_facts(mut error: QueryError, first_cause: Option<&(dyn StdError + 'static)>) -> QueryError {
    let mut current = first_cause;
    while let Some(cause) = current {
        if let Some(cudf_error) = cause.downcast_ref::<cudf::Error>() {
            for (key, value) in cudf_error.details() {
                if matches!(*key, "non_finite_count" | "remedy") {
                    error = error.with_fact(key, value);
                }
            }
            if let Some(value) = cudf_error.fact_value(FactKeyName::ValidationReason) {
                error = error.with_fact(FactKeyName::ValidationReason.as_snake_case(), value);
            }
            if let Some(detail) = cudf_error.budget_exhaustion() {
                error = error.with_memory_failure(memory_failure(detail));
            }
        }
        current = cause.source();
    }
    error
}

#[cfg(all(test, any(feature = "cuvs", feature = "cuml")))]
mod tests {
    #[cfg(feature = "cuml")]
    use cudf::CudfAllocationAttribution;
    #[cfg(feature = "cuml")]
    use cuml::{ErrorCode, ErrorKind, ErrorStatus};
    use query_planning::error::QueryErrorCode;

    #[cfg(feature = "cuml")]
    use super::classify_cuml_error;
    #[cfg(feature = "cuvs")]
    use super::cuvs_query_error;

    #[cfg(feature = "cuml")]
    #[test]
    fn cuml_error_projection_uses_typed_code_kind_status_and_allocation_source() {
        use CudfAllocationAttribution::{
            ConfirmedPhysical, ConfirmedQueryLimit, NonAllocationFailure, UnattributedAllocationFailure,
        };
        let cases = [
            (
                ErrorCode::HostOutOfMemory,
                ErrorKind::ResourceExhausted,
                ErrorStatus::Temporary,
                UnattributedAllocationFailure,
                QueryErrorCode::RapidsTemporaryFailure,
            ),
            (
                ErrorCode::OutOfMemory,
                ErrorKind::ResourceExhausted,
                ErrorStatus::Temporary,
                ConfirmedQueryLimit,
                QueryErrorCode::QueryMemoryLimitExceeded,
            ),
            (
                ErrorCode::OutOfMemory,
                ErrorKind::ResourceExhausted,
                ErrorStatus::Temporary,
                ConfirmedPhysical,
                QueryErrorCode::DeviceOutOfMemory,
            ),
            (
                ErrorCode::OutOfMemory,
                ErrorKind::ResourceExhausted,
                ErrorStatus::Temporary,
                UnattributedAllocationFailure,
                QueryErrorCode::DeviceAllocationFailed,
            ),
            (
                ErrorCode::RuntimeBindingMismatch,
                ErrorKind::ExecutionContext,
                ErrorStatus::Permanent,
                NonAllocationFailure,
                QueryErrorCode::InternalInvariantViolation,
            ),
            (
                ErrorCode::CudaRuntime,
                ErrorKind::DependencyFailure,
                ErrorStatus::Temporary,
                NonAllocationFailure,
                QueryErrorCode::RapidsTemporaryFailure,
            ),
            (
                ErrorCode::FatalCuda,
                ErrorKind::DependencyFailure,
                ErrorStatus::Persistent,
                NonAllocationFailure,
                QueryErrorCode::DeviceFault,
            ),
            (
                ErrorCode::ForeignUnknown,
                ErrorKind::DependencyFailure,
                ErrorStatus::Permanent,
                NonAllocationFailure,
                QueryErrorCode::RapidsDependencyFailure,
            ),
        ];
        for (code, kind, status, attribution, expected) in cases {
            assert_eq!(classify_cuml_error(code, kind, status, attribution), expected, "{code:?} {attribution:?}");
        }
    }

    #[cfg(feature = "cuvs")]
    #[test]
    fn cuvs_query_error_maps_only_typed_kind_and_status() {
        let cases = [
            (cuvs::Error::new(cuvs::ErrorKind::InvalidInput, "cuvs.invalid"), QueryErrorCode::RapidsInvalidInput),
            (cuvs::Error::new(cuvs::ErrorKind::Unsupported, "cuvs.unsupported"), QueryErrorCode::RapidsUnsupported),
            (cuvs::Error::new(cuvs::ErrorKind::OutOfMemory, "cuvs.oom"), QueryErrorCode::DeviceAllocationFailed),
            (cuvs::Error::new(cuvs::ErrorKind::InvalidState, "cuvs.state"), QueryErrorCode::InternalInvariantViolation),
            (
                cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.temporary")
                    .with_context("native_status", "CUVS_ERROR")
                    .set_temporary(),
                QueryErrorCode::RapidsTemporaryFailure,
            ),
            (
                cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.persistent").set_persistent(),
                QueryErrorCode::DeviceFault,
            ),
        ];
        for (source, expected) in cases {
            let error = cuvs_query_error("query_engine.test.from_cuvs", "wrapped cuVS failure", source);
            assert_eq!(error.code(), expected);
            assert_eq!(error.fact_value("rapids_library"), Some("cuvs"));
            assert!(error.fact_value("cuvs_kind").is_some());
            assert!(error.fact_value("cuvs_status").is_some());
            assert!(error.fact_value("cuvs_operation").is_some());
            if error.fact_value("cuvs_operation") == Some("cuvs.temporary") {
                assert_eq!(error.fact_value("native_status"), Some("CUVS_ERROR"));
            }
        }
    }

    #[cfg(feature = "cuvs")]
    #[test]
    fn cuvs_query_error_duplicate_context_projects_effective_value_without_panic() {
        let source = cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.duplicate")
            .with_context("native_status", "CUVS_ERROR")
            .with_context("native_status", "CUVS_ERROR_RETRY")
            .set_temporary();
        assert_eq!(source.context_value("native_status"), Some("CUVS_ERROR_RETRY"));

        let error = cuvs_query_error("query_engine.test.from_cuvs_duplicate", "wrapped cuVS failure", source);

        assert_eq!(error.code(), QueryErrorCode::RapidsTemporaryFailure);
        assert_eq!(error.fact_value("native_status"), Some("CUVS_ERROR_RETRY"));
        assert!(error.source_error().is_some());
    }
}
