use datafusion::common::DataFusionError;

use super::CumlOperation;
use crate::{error::Error, gpu_functions::dense_relation_errors::DenseRelationErrorContext};

const ERRORS: DenseRelationErrorContext = DenseRelationErrorContext {
    family: "cuML",
    operation_fact: "cuml_operation",
    validate_operation: "algeon_datafusion.cuml_sql.validate",
    validate_schema_operation: "algeon_datafusion.cuml_sql.validate_schema",
    execute_operation: "algeon_datafusion.cuml_sql.execute",
    required_feature: "cuml",
    feature_disabled_message: "cuML algorithm execution requires the cuml Cargo feature",
};

pub(super) fn invalid_call(
    operation: CumlOperation,
    reason: &'static str,
    message: impl Into<String>,
) -> DataFusionError {
    ERRORS.invalid_call(operation.descriptor().sql_function_name, reason, message)
}

/// Structured invalid-call error for callers that attach further facts.
pub(super) fn invalid_call_error(operation: CumlOperation, reason: &'static str, message: impl Into<String>) -> Error {
    ERRORS.invalid_call_error(operation.descriptor().sql_function_name, reason, message)
}

pub(super) fn invalid_option(
    operation: CumlOperation,
    reason: &'static str,
    argument: Option<&'static str>,
    message: impl Into<String>,
) -> DataFusionError {
    ERRORS.invalid_option(operation.descriptor().sql_function_name, reason, argument, message)
}

pub(super) fn input_ordering_key_not_projected(
    operation: CumlOperation,
    relation_role: &'static str,
    key: String,
) -> DataFusionError {
    ERRORS.input_ordering_key_not_projected(operation.descriptor().sql_function_name, relation_role, key)
}

pub(super) fn field_not_found(operation: CumlOperation, relation_role: &'static str, field: &str) -> DataFusionError {
    ERRORS.field_not_found(operation.descriptor().sql_function_name, relation_role, field)
}

pub(super) fn unsupported_type(
    operation: CumlOperation,
    relation_role: &'static str,
    field: &str,
    actual: impl ToString,
    expected: impl ToString,
) -> DataFusionError {
    ERRORS.unsupported_type(operation.descriptor().sql_function_name, relation_role, field, actual, expected)
}

pub(super) fn nullable_feature_field(
    operation: CumlOperation,
    relation_role: &'static str,
    field: &str,
) -> DataFusionError {
    ERRORS.nullable_feature_field(operation.descriptor().sql_function_name, relation_role, field)
}

pub(super) fn feature_disabled(operation: CumlOperation) -> DataFusionError {
    ERRORS.feature_disabled(operation.descriptor().sql_function_name)
}
