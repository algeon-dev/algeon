use super::error::{self, Error, ErrorCode, ErrorKind};

pub(crate) fn column_name_count_limit_error(
    operation: &'static str,
    message: &'static str,
    column_name_count: usize,
) -> Error {
    error::limit_validation_error(
        ErrorKind::InvalidInput,
        operation,
        message,
        "column_name_count",
        "column_name_count",
        "i32::MAX",
        column_name_count,
        "reduce column_name_count to fit within i32::MAX",
    )
}

pub(crate) fn column_name_count_mismatch_error(
    kind: ErrorKind,
    operation: &'static str,
    message: impl Into<String>,
    column_name_count: usize,
    table_column_count: usize,
) -> Error {
    Error::new(kind, message)
        .with_code(ErrorCode::ColumnLengthMismatch)
        .with_operation(operation)
        .with_context("column_name_count", column_name_count.to_string())
        .with_context("table_column_count", table_column_count.to_string())
        .with_context("resource", "column_name_count")
        .with_context("reason", "mismatch_with_table_column_count")
        .with_context("remedy", "ensure column_name_count matches table_column_count")
}

pub(crate) fn foreign_column_name_count_contract_error(
    operation: &'static str,
    message: impl Into<String>,
    reason: &'static str,
    column_name_count: usize,
    table_column_count: usize,
    remedy: &'static str,
) -> Error {
    Error::new(ErrorKind::InvariantViolation, message)
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
        .with_context("column_name_count", column_name_count.to_string())
        .with_context("table_column_count", table_column_count.to_string())
        .with_context("resource", "column_name_count")
        .with_context("reason", reason)
        .with_context("remedy", remedy)
}
