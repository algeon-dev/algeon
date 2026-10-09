#[cfg(not(feature = "cugraph"))]
use super::CugraphAlgorithm;
use crate::error::{Error, ErrorCode};
#[cfg(not(feature = "cugraph"))]
use crate::gpu_functions::REQUIRED_FEATURE_DISABLED;

/// Stable feature-off execution rejection for a cuGraph SQL operation.
///
/// The frontend deliberately remains available without native cuGraph support.
/// This boundary must be reached before a graph source is scanned, staged, or
/// passed to a native executor.
#[cfg(not(feature = "cugraph"))]
pub(super) fn feature_disabled(algorithm: CugraphAlgorithm) -> Error {
    Error::new(
        ErrorCode::UnsupportedOperator,
        "algeon_datafusion.cugraph_sql.execute",
        "cuGraph SQL execution requires the cugraph Cargo feature",
    )
    .with_log_safe_fact("cugraph_operation", algorithm.function_name())
    .with_log_safe_fact("reason", REQUIRED_FEATURE_DISABLED)
    .with_log_safe_fact("required_feature", "cugraph")
    .with_log_safe_fact("host_fallback", "disabled")
}

/// A plan whose shape or invariants do not hold. User-supplied option names and
/// values are rejected through [`invalid_option`] and [`unknown_option`].
pub(super) fn plan(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidExecutionPlan, "algeon_datafusion.cugraph_sql.plan", message)
}

/// Rejection of an option value of the wrong type, outside its declared range,
/// or conflicting with another option. Callers that know the offending argument
/// attach it as the `argument` fact.
///
/// The `reason` is the code the cuVS and cuML families project from
/// `GpuOptionValidationError`, so `gpu_validate_call` reports one vocabulary.
pub(super) fn invalid_option(message: impl Into<String>) -> Error {
    option_error("missing_or_invalid_option", message)
}

/// Rejection of an option name the function does not declare.
pub(super) fn unknown_option(message: impl Into<String>) -> Error {
    option_error("unknown_option", message)
}

fn option_error(reason: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "algeon_datafusion.cugraph_sql.option", message)
        .with_log_safe_fact("reason", reason)
}

/// Builds a structured rejection for a syntactically valid option that this
/// SQL surface deliberately does not implement. Metadata validation projects
/// this code directly; it must not recover the classification from prose.
pub(super) fn unsupported_option(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedOperator, "algeon_datafusion.cugraph_sql.option", message)
        .with_fact("reason", "unsupported_option")
}
