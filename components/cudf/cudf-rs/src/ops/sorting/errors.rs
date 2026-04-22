use crate::core::error::{self, Error, ErrorCode, ErrorKind};

pub(super) fn unsupported_sort_validation_error(
    operation: &'static str,
    message: impl Into<String>,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("reason", reason)
        .with_context("remedy", remedy)
}

pub(super) fn sort_option_count_mismatch_error(
    operation: &'static str,
    message: impl Into<String>,
    reason: &'static str,
    remedy: &'static str,
    keys_count: usize,
    actual_count: usize,
) -> Error {
    unsupported_sort_validation_error(operation, message, reason, remedy)
        .with_context("resource", "sort_options")
        .with_context("keys_count", keys_count.to_string())
        .with_context("actual_count", actual_count.to_string())
}

pub(super) fn sort_key_count_limit_error(operation: &'static str, keys_count: usize) -> Error {
    error::limit_validation_error(
        ErrorKind::InvalidInput,
        operation,
        "number of sort keys exceeds i32::MAX",
        "sort_keys",
        "keys_count",
        "i32::MAX",
        keys_count,
        "reduce the number of sort keys to fit within i32::MAX",
    )
}
