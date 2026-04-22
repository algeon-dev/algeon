use super::dataframe::trigger_merge_reassembly_key_count_error_for_test_impl;
use crate::{DataFrame, core::error::Result};

pub(crate) fn trigger_merge_reassembly_key_count_error_for_test() -> Result<DataFrame> {
    trigger_merge_reassembly_key_count_error_for_test_impl()
}
