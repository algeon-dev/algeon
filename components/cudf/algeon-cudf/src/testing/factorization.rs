use crate::core::Result;

/// Trigger the factorization empty-row-code dtype invariant path used by
/// integration tests.
pub fn trigger_factorize_empty_row_code_dtype_error_for_test() -> Result<()> {
    crate::ops::factorization::testing_hooks::trigger_factorize_empty_row_code_dtype_error_for_test()
}

/// Trigger the Codebook key-count limit validation path used by integration
/// tests.
pub fn trigger_codebook_key_count_limit_error_for_test() -> Result<()> {
    crate::frame::factorization::trigger_codebook_key_count_limit_error_for_test()
}
