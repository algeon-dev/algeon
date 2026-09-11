//! Shared high-level validation for DataFrame APIs that operate on group keys.

use crate::core::error::{Error, ErrorCode, FACT_KEYS_COUNT, Result};

pub(crate) fn require_non_empty_group_keys(keys_len: usize, operation: &'static str) -> Result<()> {
    if keys_len == 0 {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "at least one group key column is required",
        )
        .with_fact(FACT_KEYS_COUNT, 0usize)
        .with_detail("resource", "group_keys")
        .with_detail("reason", "empty_group_keys")
        .with_detail("remedy", "provide at least one group key column"));
    }

    Ok(())
}
