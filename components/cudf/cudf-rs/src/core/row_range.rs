use std::sync::Arc;

use crate::{
    core::{
        column::Column,
        error::{self, Result},
    },
    runtime::execution_domain::ExecutionDomain,
};

pub(crate) fn checked_bounds_i32(start: usize, len: usize, operation: &'static str) -> Result<(i32, i32)> {
    let start_i32: i32 = start.try_into().map_err(|_| error::row_range_limit_error(operation, "start", start))?;
    let end = start.checked_add(len).ok_or_else(|| error::row_range_overflow_error(operation, start, len))?;
    let end_i32: i32 = end
        .try_into()
        .map_err(|_| error::row_range_limit_error(operation, "end", end).with_detail("start_row", start.to_string()))?;
    Ok((start_i32, end_i32))
}

pub(crate) fn gather_map_in_domain(
    start: usize,
    len: usize,
    output_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Column> {
    let (start_i32, end_i32) = checked_bounds_i32(start, len, operation)?;
    let indices: Vec<i32> = (start_i32..end_i32).collect();
    Column::from_slice_in_domain(&indices, output_domain)
}
