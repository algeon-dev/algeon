//! Structured validation errors shared by dense-relation SQL families.

use super::REQUIRED_FEATURE_DISABLED;
use crate::error::{ClientDetail, Error, ErrorCode};

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
    pub(crate) fn invalid_option(
        &self,
        operation: &'static str,
        reason: &'static str,
        argument: Option<&'static str>,
        message: impl Into<String>,
    ) -> Error {
        let error = self.invalid_call(operation, reason, message);
        match argument {
            Some(argument) => error.with_log_safe_fact("argument", argument),
            None => error,
        }
    }

    pub(crate) fn invalid_call(
        &self,
        operation: &'static str,
        reason: &'static str,
        message: impl Into<String>,
    ) -> Error {
        Error::new(ErrorCode::InvalidRequest, self.validate_operation, message)
            .with_log_safe_fact(self.operation_fact, operation)
            .with_log_safe_fact("reason", reason)
    }

    pub(crate) fn input_ordering_key_not_projected(
        &self,
        operation: &'static str,
        relation_role: &'static str,
        key: String,
    ) -> Error {
        self.invalid_call(
            operation,
            "input_ordering_key_not_projected",
            format!(
                "{} {relation_role} ORDER BY key is not an output column of the input; project the key in the input or drop the ORDER BY",
                self.family
            ),
        )
        .with_log_safe_fact("relation_role", relation_role)
        .with_client_detail(ClientDetail::OrderingKey { name: key })
    }

    pub(crate) fn field_not_found(&self, operation: &'static str, relation_role: &'static str, field: &str) -> Error {
        Error::new(
            ErrorCode::FieldNotFound,
            self.validate_schema_operation,
            format!("{} {relation_role} relation has no field with the requested name", self.family),
        )
        .with_log_safe_fact(self.operation_fact, operation)
        .with_log_safe_fact("relation_role", relation_role)
        .with_client_detail(ClientDetail::Field { name: field.to_string() })
    }

    pub(crate) fn unsupported_type(
        &self,
        operation: &'static str,
        relation_role: &'static str,
        field: &str,
        actual: impl ToString,
        expected: impl ToString,
    ) -> Error {
        let expected = expected.to_string();
        // The field and its type name the caller's own schema (nested Arrow
        // types print field names), so they travel as client detail and stay
        // out of the logged message and facts.
        Error::new(
            ErrorCode::UnsupportedType,
            self.validate_schema_operation,
            format!("{} {relation_role} field has an unsupported type; supported types: {expected}", self.family),
        )
        .with_log_safe_fact(self.operation_fact, operation)
        .with_log_safe_fact("relation_role", relation_role)
        .with_client_detail(ClientDetail::RelationField { field: field.to_string(), data_type: actual.to_string() })
        .with_log_safe_fact("expected_type", expected)
    }

    pub(crate) fn feature_disabled(&self, operation: &'static str) -> Error {
        Error::new(ErrorCode::UnsupportedOperator, self.execute_operation, self.feature_disabled_message)
            .with_log_safe_fact(self.operation_fact, operation)
            .with_log_safe_fact("reason", REQUIRED_FEATURE_DISABLED)
            .with_log_safe_fact("required_feature", self.required_feature)
            .with_log_safe_fact("host_fallback", "disabled")
    }
}
