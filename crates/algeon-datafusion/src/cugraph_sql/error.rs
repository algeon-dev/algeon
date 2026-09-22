use datafusion::common::DataFusionError;

#[cfg(not(feature = "cugraph"))]
use super::CugraphAlgorithm;
use crate::error::{Error, ErrorCode};

/// Stable feature-off execution rejection for a cuGraph SQL operation.
///
/// The frontend deliberately remains available without native cuGraph support,
/// so this boundary must be reached before a graph source is scanned, staged,
/// or handed to a native executor.
#[cfg(not(feature = "cugraph"))]
pub(super) fn feature_disabled(algorithm: CugraphAlgorithm) -> DataFusionError {
    Error::new(
        ErrorCode::UnsupportedOperator,
        "algeon_datafusion.cugraph_sql.execute",
        "cuGraph SQL execution requires the cugraph Cargo feature",
    )
    .with_log_safe_fact("cugraph_operation", algorithm.function_name())
    .with_log_safe_fact("reason", "required_feature_disabled")
    .with_log_safe_fact("required_feature", "cugraph")
    .with_log_safe_fact("host_fallback", "disabled")
    .into()
}

pub(super) fn plan(message: impl Into<String>) -> DataFusionError {
    Error::new(ErrorCode::InvalidExecutionPlan, "algeon_datafusion.cugraph_sql.plan", message).into()
}

/// Builds a structured rejection for a syntactically valid option that this
/// SQL surface deliberately does not implement. Metadata validation projects
/// this code directly; it must not recover the classification from prose.
pub(super) fn unsupported_option(message: impl Into<String>) -> DataFusionError {
    Error::new(ErrorCode::UnsupportedOperator, "algeon_datafusion.cugraph_sql.option", message)
        .with_fact("reason", "unsupported_option")
        .into()
}

/// Tags a planning rejection with the named argument or relation role it
/// concerns so `gpu_validate_call` can report a stable `details_json.path`.
/// Errors that did not originate from this module's typed constructors are
/// returned unchanged.
pub(super) fn with_binding_fact(error: DataFusionError, key: &'static str, value: &str) -> DataFusionError {
    match error {
        DataFusionError::External(source) => match source.downcast::<Error>() {
            Ok(error) => (*error).with_fact(key, value).into(),
            Err(source) => DataFusionError::External(source),
        },
        other => other,
    }
}
