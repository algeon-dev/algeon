use std::sync::Arc;

use super::Column;
use crate::{core::error::Result, runtime::execution_domain::ExecutionDomain};

/// Build a zero-based `Int32` position column `0..len` inside one execution
/// domain.
pub(crate) fn position_sequence_in_domain(
    len: usize,
    domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Column> {
    Column::from_sequence_in_domain(0i32, 1i32, len, Arc::clone(domain), operation)
}
