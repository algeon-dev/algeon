use crate::core::error::{self, Result};

pub(crate) fn checked_bounds_i32(start: usize, len: usize, operation: &'static str) -> Result<(i32, i32)> {
    let start_i32: i32 = start.try_into().map_err(|_| error::row_range_limit_error(operation, "start", start))?;
    let end = start.checked_add(len).ok_or_else(|| error::row_range_overflow_error(operation, start, len))?;
    let end_i32: i32 = end
        .try_into()
        .map_err(|_| error::row_range_limit_error(operation, "end", end).with_detail("start_row", start.to_string()))?;
    Ok((start_i32, end_i32))
}
