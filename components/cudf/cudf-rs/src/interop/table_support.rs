use super::*;

pub(super) fn checked_column_index(operation: &'static str, index: usize) -> Result<i32, InteropError> {
    let index_i32 = i32::try_from(index).map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column index exceeds i32 range")
            .with_context("operation", operation)
            .with_context("index", index.to_string())
    })?;
    Ok(index_i32)
}
