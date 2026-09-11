use crate::core::error::{self, Error};

pub(crate) fn table_column_count_limit_error(
    operation: &'static str,
    message: &'static str,
    table_column_count: usize,
) -> Error {
    error::limit_validation_error(
        operation,
        message,
        "table_column_count",
        "table_column_count",
        "i32::MAX",
        table_column_count,
        "reduce table_column_count to fit within i32::MAX",
    )
}
