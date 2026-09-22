use std::sync::Arc;

use crate::{
    core::{column::Column, error::Result},
    runtime::Runtime,
};

/// Trigger the factorization empty-row-code dtype invariant path used by
/// integration tests.
pub fn trigger_factorize_empty_row_code_dtype_error_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let output_domain = Arc::clone(runtime.execution_domain());
    let row_codes = Column::from_slice_in_domain::<f64>(&[], &output_domain)?;
    let sorted_keys = Column::from_slice_in_domain::<i32>(&[], &output_domain)?;
    let row_codes_view = row_codes.view()?;
    let sorted_keys_view = sorted_keys.view()?;

    let exec_options = output_domain.enter_execution()?;
    super::ordered_first_seen_keys(
        &row_codes_view,
        &sorted_keys_view,
        &output_domain,
        exec_options.options(),
        "series::factorize",
    )
    .map(|_| ())
}
