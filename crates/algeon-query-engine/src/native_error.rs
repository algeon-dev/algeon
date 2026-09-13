//! Native error projection facade.
//!
//! The cuDF projection is owned by the runtime's native boundary so the
//! runtime and the engine classify a `cudf::Error` identically. cuVS is used
//! by the engine alone, so its projection lives here.

#[cfg(feature = "cuvs")]
use std::{collections::BTreeMap, error::Error as StdError};

#[cfg(feature = "cuvs")]
use query_planning::error::{QueryError, QueryErrorCode, redact_uri_tokens};
pub use query_runtime::native_error::{
    cudf_budget_exhaustion, cudf_message_fact, cudf_query_error, cudf_query_error_with_budget_details,
    cudf_query_error_without_source, cudf_query_error_without_source_with_budget_details, cudf_unsupported_type_error,
    query_error_code_for_cudf_error,
};

/// Wraps a `cuvs::Error` at the native vector-execution boundary.
///
/// Classification is derived only from the typed cuVS kind and status; foreign
/// message text remains diagnostic and never drives retry policy.
#[cfg(feature = "cuvs")]
pub fn cuvs_query_error(operation: &'static str, message: impl Into<String>, source: cuvs::Error) -> QueryError {
    let code = query_error_code_for_cuvs_error(&source);
    let error = QueryError::new(code, operation, message)
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
    error.with_source(source)
}

/// Wraps a prepared cuDF interop failure at a native vector boundary.
///
/// The engine validates vector schemas and selects every requested column
/// before preparing this producer-rooted export. A preparation failure at this
/// point is therefore an execution-binding invariant, not a second user-input
/// validation surface.
#[cfg(feature = "cuvs")]
pub fn cudf_interop_query_error<E>(
    operation: &'static str,
    message: impl Into<String>,
    source_kind: impl Into<String>,
    source: E,
) -> QueryError
where
    E: StdError + Send + Sync + 'static,
{
    let source_kind: String = source_kind.into();
    QueryError::new(QueryErrorCode::CudfInvariantViolation, operation, message)
        .with_fact("cudf_interop_kind", source_kind)
        .with_source(source)
}

/// Classifies a `cuvs::Error` into the native-engine error taxonomy.
#[cfg(feature = "cuvs")]
pub fn query_error_code_for_cuvs_error(error: &cuvs::Error) -> QueryErrorCode {
    match error.kind() {
        cuvs::ErrorKind::InvalidInput => QueryErrorCode::CudfInvalidInput,
        cuvs::ErrorKind::Unsupported => QueryErrorCode::CudfUnsupported,
        cuvs::ErrorKind::OutOfMemory => QueryErrorCode::DeviceAllocationFailed,
        cuvs::ErrorKind::InvalidState => QueryErrorCode::InternalInvariantViolation,
        cuvs::ErrorKind::Unexpected => match error.status() {
            cuvs::ErrorStatus::Temporary => QueryErrorCode::CudfTemporaryFailure,
            cuvs::ErrorStatus::Persistent => QueryErrorCode::DeviceFault,
            cuvs::ErrorStatus::Permanent => QueryErrorCode::CudfDependencyFailure,
        },
        _ => QueryErrorCode::CudfDependencyFailure,
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

#[cfg(all(test, feature = "cuvs"))]
mod tests;
