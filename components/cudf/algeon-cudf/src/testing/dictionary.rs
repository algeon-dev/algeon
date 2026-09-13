use crate::{core::Result, ops::dictionary::testing_hooks};

/// Trigger the dictionary match result-count mismatch foreign-contract path
/// used by integration tests.
pub fn trigger_match_dictionaries_result_count_mismatch_for_test() -> Result<()> {
    testing_hooks::trigger_match_dictionaries_result_count_mismatch_for_test()
}

/// Trigger the dictionary match null-result-array foreign-contract path used
/// by integration tests.
pub fn trigger_match_dictionaries_null_result_array_for_test() -> Result<()> {
    testing_hooks::trigger_match_dictionaries_null_result_array_for_test()
}

/// Trigger the shared-dictionary input-count limit validation contract.
pub fn trigger_match_dictionaries_column_count_limit_error_for_test() -> Result<()> {
    testing_hooks::trigger_match_dictionaries_column_count_limit_error_for_test()
}

/// Trigger fatal teardown while cleaning a malformed dictionary result.
pub fn trigger_match_dictionaries_fatal_malformed_cleanup_for_test() -> Result<()> {
    testing_hooks::trigger_match_dictionaries_fatal_malformed_cleanup_for_test()
}
