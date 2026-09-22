use crate::core::error::{Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_KEYS_COUNT, limit_validation_error};

pub(super) fn unsupported_sort_validation_error(
    operation: &'static str,
    message: impl Into<String>,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, operation, message)
        .with_detail("reason", reason)
        .with_detail("remedy", remedy)
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
        .with_detail("resource", "sort_options")
        .with_fact(FACT_KEYS_COUNT, keys_count)
        .with_fact(FACT_ACTUAL_COUNT, actual_count)
}

pub(super) fn sort_key_count_limit_error(operation: &'static str, keys_count: usize) -> Error {
    limit_validation_error(
        operation,
        "number of sort keys exceeds i32::MAX",
        "sort_keys",
        "keys_count",
        "i32::MAX",
        keys_count,
        "reduce the number of sort keys to fit within i32::MAX",
    )
}
