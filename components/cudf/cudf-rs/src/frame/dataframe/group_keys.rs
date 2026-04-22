//! Shared high-level validation for DataFrame APIs that operate on group keys.

use crate::core::error::{Error, ErrorCode, ErrorKind, Result};

fn empty_group_keys_validation_error(operation: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, "at least one group key column is required")
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("resource", "group_keys")
        .with_context("reason", "empty_group_keys")
        .with_context("keys_count", "0")
        .with_context("remedy", "provide at least one group key column")
}

pub(crate) fn require_non_empty_group_keys(keys_len: usize, operation: &'static str) -> Result<()> {
    if keys_len == 0 {
        return Err(empty_group_keys_validation_error(operation));
    }

    Ok(())
}
