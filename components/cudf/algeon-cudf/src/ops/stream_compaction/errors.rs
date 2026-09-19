use crate::core::error::{Error, ErrorCode, FACT_COLUMN_INDEX, FACT_TABLE_COLUMN_COUNT};

pub(super) fn unsupported_stream_compaction_validation_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, operation, message)
        .with_detail("reason", reason)
        .with_detail("remedy", remedy)
}

pub(super) fn key_index_out_of_bounds_error(operation: &'static str, key_index: usize, num_columns: usize) -> Error {
    Error::new_with(
        ErrorCode::IndexOutOfBounds,
        operation,
        format!("key column index {} out of bounds for {} columns", key_index, num_columns),
    )
    .with_fact(FACT_COLUMN_INDEX, key_index)
    .with_fact(FACT_TABLE_COLUMN_COUNT, num_columns)
}
