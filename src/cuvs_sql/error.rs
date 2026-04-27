use datafusion::common::DataFusionError;

use super::CuvsOperation;
use crate::error::{Error, ErrorCode};

pub(super) fn invalid_call(
    operation: CuvsOperation,
    reason: &'static str,
    message: impl Into<String>,
) -> DataFusionError {
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.cuvs_sql.validate", message)
        .with_log_safe_fact("cuvs_operation", operation.descriptor().sql_function_name)
        .with_log_safe_fact("reason", reason)
        .into()
}

pub(super) fn field_not_found(operation: CuvsOperation, relation_role: &'static str, field: &str) -> DataFusionError {
    Error::new(
        ErrorCode::FieldNotFound,
        "datafusion_nexus.cuvs_sql.validate_schema",
        format!("cuVS {relation_role} relation has no field named '{field}'"),
    )
    .with_log_safe_fact("cuvs_operation", operation.descriptor().sql_function_name)
    .with_log_safe_fact("relation_role", relation_role)
    .with_fact("field", field)
    .into()
}

pub(super) fn unsupported_type(
    operation: CuvsOperation,
    relation_role: &'static str,
    field: &str,
    actual: impl ToString,
    expected: &'static str,
) -> DataFusionError {
    Error::new(
        ErrorCode::UnsupportedType,
        "datafusion_nexus.cuvs_sql.validate_schema",
        format!("cuVS {relation_role} field '{field}' has an unsupported type"),
    )
    .with_log_safe_fact("cuvs_operation", operation.descriptor().sql_function_name)
    .with_log_safe_fact("relation_role", relation_role)
    .with_fact("field", field)
    .with_log_safe_fact("actual_type", actual)
    .with_log_safe_fact("expected_type", expected)
    .into()
}

pub(super) fn feature_disabled(operation: CuvsOperation) -> DataFusionError {
    Error::new(
        ErrorCode::UnsupportedOperator,
        "datafusion_nexus.cuvs_sql.execute",
        "cuVS vector algorithm execution requires the cuvs Cargo feature",
    )
    .with_log_safe_fact("cuvs_operation", operation.descriptor().sql_function_name)
    .with_log_safe_fact("reason", "cuvs_feature_disabled")
    .with_log_safe_fact("required_feature", "cuvs")
    .with_log_safe_fact("host_fallback", "disabled")
    .into()
}

pub(super) fn bounded_execution_unsupported(
    operation: CuvsOperation,
    execution_mode: impl ToString,
) -> DataFusionError {
    Error::new(
        ErrorCode::UnsupportedOperator,
        "datafusion_nexus.cuvs_sql.bounded_admission",
        "bounded cuVS execution requires a lower-level peak-memory preflight contract",
    )
    .with_log_safe_fact("cuvs_operation", operation.descriptor().sql_function_name)
    .with_log_safe_fact("operator", "vector_algorithm")
    .with_log_safe_fact("execution_mode", execution_mode.to_string())
    .with_log_safe_fact("reason", "blocked_lower_level_contract")
    .with_log_safe_fact("memory_boundary", nexus_query_engine::plan::VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY)
    .with_log_safe_fact("required_contract", nexus_query_engine::plan::VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT)
    .with_log_safe_fact("host_fallback", "disabled")
    .into()
}
