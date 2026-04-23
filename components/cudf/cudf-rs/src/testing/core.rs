use crate::core::{DeviceMemoryResource, Result};

/// Build a stats memory resource for allocation-accounting tests.
pub fn stats_memory_resource() -> Result<DeviceMemoryResource> {
    crate::core::testing_hooks::stats_memory_resource()
}

/// Pack host validity bits into cuDF's allocation-sized null-mask layout.
pub fn validity_to_host_null_mask_bytes(validity: &[bool]) -> Vec<u8> {
    crate::core::testing_hooks::validity_to_host_null_mask_bytes(validity)
}

/// Trigger the shared invariant-violation helper path used by integration
/// tests.
pub fn trigger_invariant_violation_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_invariant_violation_for_test()
}

/// Trigger the shared null-on-success foreign-handle contract path used by
/// integration tests.
pub fn trigger_null_handle_contract_error_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_null_handle_contract_error_for_test()
}

/// Trigger the shared range-limit helper path used by integration tests.
pub fn trigger_range_limit_error_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_range_limit_error_for_test()
}

/// Trigger a synthetic truthful-contract error carrying typed facts,
/// repeated details, a source chain, and multiple reanchor hops.
pub fn trigger_truthful_error_contract_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_truthful_error_contract_for_test()
}

/// Trigger a synthetic error that reanchors twice at the same operation so
/// integration tests can lock the no-deduplication contract.
pub fn trigger_same_operation_reanchor_contract_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_same_operation_reanchor_contract_for_test()
}

/// Trigger a synthetic error that applies `reanchor_if_needed` with the same
/// operation so integration tests can lock the no-op contract.
pub fn trigger_reanchor_if_needed_same_operation_contract_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_reanchor_if_needed_same_operation_contract_for_test()
}

/// Trigger a synthetic error that exercises the shared `*_if_absent` helpers
/// so integration tests can lock the preserve-existing-and-fill-missing
/// contract.
pub fn trigger_if_absent_error_annotation_contract_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_if_absent_error_annotation_contract_for_test()
}

/// Trigger a synthetic error that exercises the shared `with_column` helper
/// so integration tests can lock the canonical `column_name` fact contract.
pub fn trigger_with_column_error_annotation_contract_for_test() -> Result<()> {
    crate::core::error::testing_hooks::trigger_with_column_error_annotation_contract_for_test()
}

/// Attempt to write a canonical fact snake_case key through the crate-internal
/// detail writer path. The writer rejects the overlap-rule violation
/// structurally and panics; this hook lets integration tests lock that
/// behavior without exercising any public construction API.
pub fn trigger_canonical_key_in_detail_path_for_test() {
    crate::core::error::testing_hooks::trigger_canonical_key_in_detail_path_for_test()
}

/// Trigger the string-scalar negative-length foreign metadata path used by
/// integration tests.
pub fn trigger_negative_string_scalar_length_for_test() -> Result<()> {
    crate::core::scalar::trigger_negative_string_length_error_for_test()
}

/// Trigger the stable owned-table column-count limit validation contract.
pub fn trigger_table_column_count_limit_error_for_test() -> Result<()> {
    crate::core::table::testing_hooks::trigger_table_column_count_limit_error_for_test()
}

/// Trigger the stable borrowed-table-view column-count limit validation
/// contract.
pub fn trigger_table_view_column_count_limit_error_for_test() -> Result<()> {
    crate::core::table::testing_hooks::trigger_table_view_column_count_limit_error_for_test()
}
