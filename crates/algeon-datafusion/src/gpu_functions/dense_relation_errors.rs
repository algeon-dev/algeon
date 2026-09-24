//! Structured validation errors shared by dense-relation SQL families.

use datafusion::common::DataFusionError;

use crate::error::{Error, ErrorCode};

pub(crate) struct DenseRelationErrorContext {
    pub(crate) family: &'static str,
    pub(crate) operation_fact: &'static str,
    pub(crate) validate_operation: &'static str,
    pub(crate) validate_schema_operation: &'static str,
    pub(crate) execute_operation: &'static str,
    pub(crate) required_feature: &'static str,
    pub(crate) feature_disabled_message: &'static str,
}

impl DenseRelationErrorContext {
    pub(crate) fn invalid_call(
        &self,
        operation: &'static str,
        reason: &'static str,
        message: impl Into<String>,
    ) -> DataFusionError {
        self.invalid_call_error(operation, reason, message).into()
    }

    pub(crate) fn invalid_option(
        &self,
        operation: &'static str,
        reason: &'static str,
        argument: Option<&'static str>,
        message: impl Into<String>,
    ) -> DataFusionError {
        let error = self.invalid_call_error(operation, reason, message);
        match argument {
            Some(argument) => error.with_log_safe_fact("argument", argument).into(),
            None => error.into(),
        }
    }

    fn invalid_call_error(&self, operation: &'static str, reason: &'static str, message: impl Into<String>) -> Error {
        Error::new(ErrorCode::InvalidRequest, self.validate_operation, message)
            .with_log_safe_fact(self.operation_fact, operation)
            .with_log_safe_fact("reason", reason)
    }

    pub(crate) fn field_not_found(
        &self,
        operation: &'static str,
        relation_role: &'static str,
        field: &str,
    ) -> DataFusionError {
        Error::new(
            ErrorCode::FieldNotFound,
            self.validate_schema_operation,
            format!("{} {relation_role} relation has no field named '{field}'", self.family),
        )
        .with_log_safe_fact(self.operation_fact, operation)
        .with_log_safe_fact("relation_role", relation_role)
        .with_fact("field", field)
        .into()
    }

    pub(crate) fn unsupported_type(
        &self,
        operation: &'static str,
        relation_role: &'static str,
        field: &str,
        actual: impl ToString,
        expected: impl ToString,
    ) -> DataFusionError {
        let expected = expected.to_string();
        Error::new(
            ErrorCode::UnsupportedType,
            self.validate_schema_operation,
            format!(
                "{} {relation_role} field '{field}' has an unsupported type; supported types: {expected}",
                self.family
            ),
        )
        .with_log_safe_fact(self.operation_fact, operation)
        .with_log_safe_fact("relation_role", relation_role)
        .with_fact("field", field)
        .with_log_safe_fact("actual_type", actual)
        .with_log_safe_fact("expected_type", expected)
        .into()
    }

    pub(crate) fn feature_disabled(&self, operation: &'static str) -> DataFusionError {
        Error::new(ErrorCode::UnsupportedOperator, self.execute_operation, self.feature_disabled_message)
            .with_log_safe_fact(self.operation_fact, operation)
            .with_log_safe_fact("reason", "required_feature_disabled")
            .with_log_safe_fact("required_feature", self.required_feature)
            .with_log_safe_fact("host_fallback", "disabled")
            .into()
    }
}
