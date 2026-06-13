use crate::core::error::{Error, ErrorCode, ErrorKind};

pub(super) fn unsupported_stream_compaction_validation_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("reason", reason)
        .with_context("remedy", remedy)
}

pub(super) fn key_index_out_of_bounds_error(operation: &'static str, key_index: usize, num_columns: usize) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        format!("key column index {} out of bounds for {} columns", key_index, num_columns),
    )
    .with_code(ErrorCode::IndexOutOfBounds)
    .with_operation(operation)
    .with_context("column_index", key_index.to_string())
    .with_context("table_column_count", num_columns.to_string())
}
