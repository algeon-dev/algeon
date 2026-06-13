use std::sync::Arc;

use crate::{
    core::{column::Column, error::Result},
    runtime::Runtime,
};

pub(crate) fn trigger_factorize_empty_row_code_dtype_error_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let output_domain = Arc::clone(runtime.execution_domain());
    let row_codes = Column::from_slice_in_domain::<f64>(&[], &output_domain)?;
    let sorted_keys = Column::from_slice_in_domain::<i32>(&[], &output_domain)?;
    let row_codes_view = row_codes.view()?;
    let sorted_keys_view = sorted_keys.view()?;

    super::ordered_first_seen_keys(
        &row_codes_view,
        &sorted_keys_view,
        &output_domain,
        output_domain.explicit_exec_options_raw().as_ref(),
        "series::factorize",
    )
    .map(|_| ())
}
