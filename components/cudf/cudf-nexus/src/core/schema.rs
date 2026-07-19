use super::error::{self, Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_TABLE_COLUMN_COUNT};

pub(crate) fn column_name_count_limit_error(
    operation: &'static str,
    message: &'static str,
    column_name_count: usize,
) -> Error {
    error::limit_validation_error(
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
    operation: &'static str,
    message: impl Into<String>,
    column_name_count: usize,
    table_column_count: usize,
) -> Error {
    Error::new_with(ErrorCode::ColumnLengthMismatch, operation, message)
        .with_fact(FACT_ACTUAL_COUNT, column_name_count)
        .with_fact(FACT_TABLE_COLUMN_COUNT, table_column_count)
        .with_detail("column_name_count", column_name_count.to_string())
        .with_detail("resource", "column_name_count")
        .with_detail("reason", "mismatch_with_table_column_count")
        .with_detail("remedy", "ensure column_name_count matches table_column_count")
}

pub(crate) fn foreign_column_name_count_contract_error(
    operation: &'static str,
    message: impl Into<String>,
    reason: &'static str,
    column_name_count: usize,
    table_column_count: usize,
    remedy: &'static str,
) -> Error {
    error::foreign_invariant_violation(operation, message, "column_name_count", reason, remedy)
        .with_fact(FACT_ACTUAL_COUNT, column_name_count)
        .with_fact(FACT_TABLE_COLUMN_COUNT, table_column_count)
        .with_detail("column_name_count", column_name_count.to_string())
}
