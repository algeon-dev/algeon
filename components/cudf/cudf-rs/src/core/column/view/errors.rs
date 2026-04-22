use crate::core::error::{Error, ErrorCode, ErrorKind};

fn with_context_if_absent(mut err: Error, key: &'static str, value: &'static str) -> Error {
    if err.context_value(key).is_none() {
        err = err.with_context(key, value);
    }
    err
}

pub(super) fn foreign_column_view_metadata_contract_error(
    operation: &'static str,
    message: impl Into<String>,
) -> Error {
    Error::new(ErrorKind::InvariantViolation, message)
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
        .with_context("resource", "column_view_metadata")
        .with_context("reason", "invalid_foreign_column_view_metadata")
        .with_context(
            "remedy",
            "ensure the foreign column view exposes valid dtype, size, null-count, child-count, and offset metadata",
        )
}

pub(super) fn classify_column_view_metadata_error(mut err: Error) -> Error {
    if err.code() == ErrorCode::InvariantViolation || err.kind() == ErrorKind::InvariantViolation {
        err = err.with_code(ErrorCode::InvariantViolation);
        err = with_context_if_absent(err, "resource", "column_view_metadata");
        err = with_context_if_absent(err, "reason", "invalid_foreign_column_view_metadata");
        err = with_context_if_absent(
            err,
            "remedy",
            "ensure the foreign column view exposes valid dtype, size, null-count, child-count, and offset metadata",
        );
    }
    err
}

pub(super) fn foreign_string_view_contract_error(operation: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvariantViolation, message)
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
        .with_context("resource", "string_view_layout")
        .with_context("reason", "invalid_string_view_metadata")
        .with_context("remedy", "ensure the STRING column view exposes valid offsets and chars metadata")
}

pub(super) fn classify_string_view_contract_error(mut err: Error) -> Error {
    if err.code() == ErrorCode::InvariantViolation || err.kind() == ErrorKind::InvariantViolation {
        err = err.with_code(ErrorCode::InvariantViolation);
        err = with_context_if_absent(err, "resource", "string_view_layout");
        err = with_context_if_absent(err, "reason", "invalid_string_view_metadata");
        err = with_context_if_absent(
            err,
            "remedy",
            "ensure the STRING column view exposes valid offsets and chars metadata",
        );
    }
    err
}
