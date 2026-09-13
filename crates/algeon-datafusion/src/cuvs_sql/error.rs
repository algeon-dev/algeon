use datafusion::common::DataFusionError;

use super::CuvsOperation;
use crate::error::{Error, ErrorCode};

pub(super) fn invalid_call(
    operation: CuvsOperation,
    reason: &'static str,
    message: impl Into<String>,
) -> DataFusionError {
    invalid_call_error(operation, reason, message).into()
}

/// Option rejection attributed to one declared value argument, so metadata
/// validation can report `options.<argument>` as the failing path.
pub(super) fn invalid_option(
    operation: CuvsOperation,
    reason: &'static str,
    argument: Option<&'static str>,
    message: impl Into<String>,
) -> DataFusionError {
    let error = invalid_call_error(operation, reason, message);
    match argument {
        Some(argument) => error.with_log_safe_fact("argument", argument).into(),
        None => error.into(),
    }
}

fn invalid_call_error(operation: CuvsOperation, reason: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "algeon_datafusion.cuvs_sql.validate", message)
        .with_log_safe_fact("cuvs_operation", operation.descriptor().sql_function_name)
        .with_log_safe_fact("reason", reason)
}

pub(super) fn field_not_found(operation: CuvsOperation, relation_role: &'static str, field: &str) -> DataFusionError {
    Error::new(
        ErrorCode::FieldNotFound,
        "algeon_datafusion.cuvs_sql.validate_schema",
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
        "algeon_datafusion.cuvs_sql.validate_schema",
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
        "algeon_datafusion.cuvs_sql.execute",
        "cuVS vector algorithm execution requires the cuvs Cargo feature",
    )
    .with_log_safe_fact("cuvs_operation", operation.descriptor().sql_function_name)
    .with_log_safe_fact("reason", "required_feature_disabled")
    .with_log_safe_fact("required_feature", "cuvs")
    .with_log_safe_fact("host_fallback", "disabled")
    .into()
}
