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
        "datafusion_nexus.cugraph_sql.execute",
        "cuGraph SQL execution requires the cugraph Cargo feature",
    )
    .with_log_safe_fact("cugraph_operation", algorithm.function_name())
    .with_log_safe_fact("reason", "required_feature_disabled")
    .with_log_safe_fact("required_feature", "cugraph")
    .with_log_safe_fact("host_fallback", "disabled")
    .into()
}

pub(super) fn plan(message: impl Into<String>) -> DataFusionError {
    Error::new(ErrorCode::InvalidExecutionPlan, "datafusion_nexus.cugraph_sql.plan", message).into()
}

/// Builds a structured rejection for a syntactically valid option that this
/// SQL surface deliberately does not implement. Metadata validation projects
/// this code directly; it must not recover the classification from prose.
pub(super) fn unsupported_option(message: impl Into<String>) -> DataFusionError {
    Error::new(ErrorCode::UnsupportedOperator, "datafusion_nexus.cugraph_sql.option", message)
        .with_fact("reason", "unsupported_option")
        .into()
}
