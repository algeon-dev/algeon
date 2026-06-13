use crate::{core::Result, runtime::BoundaryObservation};

/// Trigger the numeric Arrow error mapping path used by integration tests.
pub fn trigger_numeric_to_arrow_error_for_test() -> Result<()> {
    crate::interop::arrow_host::testing_hooks::trigger_numeric_to_arrow_error_for_test()
}

/// Trigger the record-batch Arrow error mapping path used by integration
/// tests.
pub fn trigger_record_batch_error_for_test() -> Result<()> {
    crate::interop::arrow_host::testing_hooks::trigger_record_batch_error_for_test()
}

/// Trigger the Utf8 Arrow downcast-mismatch path used by integration tests.
pub fn trigger_arrow_string_array_downcast_mismatch_for_test() -> Result<()> {
    crate::interop::arrow_host::testing_hooks::trigger_string_array_downcast_mismatch_for_test()
}

/// Trigger the LargeUtf8 Arrow downcast-mismatch path used by integration
/// tests.
pub fn trigger_arrow_large_string_array_downcast_mismatch_for_test() -> Result<()> {
    crate::interop::arrow_host::testing_hooks::trigger_large_string_array_downcast_mismatch_for_test()
}

/// Reset host-copy Arrow interop observation counters used by integration
/// tests.
pub fn reset_arrow_host_interop_observation_for_test() {
    crate::interop::arrow_host::testing_hooks::reset_host_interop_observation_for_test();
}

/// Read how many times the host-copy Arrow export entrypoint ran.
pub fn arrow_host_export_call_count_for_test() -> usize {
    crate::interop::arrow_host::testing_hooks::host_export_call_count_for_test()
}

/// Read how many times the host-copy Arrow import entrypoint ran.
pub fn arrow_host_import_call_count_for_test() -> usize {
    crate::interop::arrow_host::testing_hooks::host_import_call_count_for_test()
}

/// Read the last host-copy Arrow boundary observation, if one has been
/// recorded since the last reset.
pub fn last_arrow_host_boundary_observation_for_test() -> Option<BoundaryObservation> {
    crate::interop::arrow_host::testing_hooks::last_host_boundary_observation_for_test()
}
