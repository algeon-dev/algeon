use super::CuvsOperation;
use crate::{error::Error, gpu_functions::dense_relation_errors::DenseRelationErrorContext};

const ERRORS: DenseRelationErrorContext = DenseRelationErrorContext {
    family: "cuVS",
    operation_fact: "cuvs_operation",
    validate_operation: "algeon_datafusion.cuvs_sql.validate",
    validate_schema_operation: "algeon_datafusion.cuvs_sql.validate_schema",
    execute_operation: "algeon_datafusion.cuvs_sql.execute",
    required_feature: "cuvs",
    feature_disabled_message: "cuVS vector algorithm execution requires the cuvs Cargo feature",
};

pub(super) fn invalid_call(operation: CuvsOperation, reason: &'static str, message: impl Into<String>) -> Error {
    ERRORS.invalid_call(operation.descriptor().sql_function_name, reason, message)
}

pub(super) fn invalid_option(
    operation: CuvsOperation,
    reason: &'static str,
    argument: Option<&'static str>,
    message: impl Into<String>,
) -> Error {
    ERRORS.invalid_option(operation.descriptor().sql_function_name, reason, argument, message)
}

pub(super) fn input_ordering_key_not_projected(
    operation: CuvsOperation,
    relation_role: &'static str,
    key: String,
) -> Error {
    ERRORS.input_ordering_key_not_projected(operation.descriptor().sql_function_name, relation_role, key)
}

pub(super) fn field_not_found(operation: CuvsOperation, relation_role: &'static str, field: &str) -> Error {
    ERRORS.field_not_found(operation.descriptor().sql_function_name, relation_role, field)
}

pub(super) fn unsupported_type(
    operation: CuvsOperation,
    relation_role: &'static str,
    field: &str,
    actual: impl ToString,
    expected: impl ToString,
) -> Error {
    ERRORS.unsupported_type(operation.descriptor().sql_function_name, relation_role, field, actual, expected)
}

pub(super) fn nullable_feature_field(operation: CuvsOperation, relation_role: &'static str, field: &str) -> Error {
    ERRORS.nullable_feature_field(operation.descriptor().sql_function_name, relation_role, field)
}

pub(super) fn feature_disabled(operation: CuvsOperation) -> Error {
    ERRORS.feature_disabled(operation.descriptor().sql_function_name)
}
