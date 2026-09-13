use crate::{core::Result, interop::arrow_host::testing_hooks, runtime::BoundaryObservation};

/// Trigger the numeric Arrow error mapping path used by integration tests.
pub fn trigger_numeric_to_arrow_error_for_test() -> Result<()> {
    testing_hooks::trigger_numeric_to_arrow_error_for_test()
}

/// Trigger the record-batch Arrow error mapping path used by integration
/// tests.
pub fn trigger_record_batch_error_for_test() -> Result<()> {
    testing_hooks::trigger_record_batch_error_for_test()
}

/// Trigger the Utf8 Arrow downcast-mismatch path used by integration tests.
pub fn trigger_arrow_string_array_downcast_mismatch_for_test() -> Result<()> {
    testing_hooks::trigger_string_array_downcast_mismatch_for_test()
}

/// Trigger the LargeUtf8 Arrow downcast-mismatch path used by integration
/// tests.
pub fn trigger_arrow_large_string_array_downcast_mismatch_for_test() -> Result<()> {
    testing_hooks::trigger_large_string_array_downcast_mismatch_for_test()
}

/// Start a fresh host-copy Arrow interop observation window for the calling
/// thread.
///
/// The underlying call totals remain process-wide and monotonic so resets from
/// parallel tests cannot rewind another thread's observation window. This
/// snapshots the calling thread's count baselines and clears last-boundary
/// detail only for that thread. Calls dispatched to other threads after this
/// reset remain visible through the calling thread's counters.
pub fn reset_arrow_host_interop_observation_for_test() {
    testing_hooks::reset_host_interop_observation_for_test();
}

/// Read how many times the host-copy Arrow export entrypoint ran process-wide
/// since the calling thread last reset its observation window.
///
/// The counter follows exports dispatched to dedicated GPU worker threads, so
/// a caller may reset it before asynchronous execution and inspect it after
/// that execution joins.
pub fn arrow_host_export_call_count_for_test() -> usize {
    testing_hooks::host_export_call_count_for_test()
}

/// Read how many times the host-copy Arrow import entrypoint ran process-wide
/// since the calling thread last reset its observation window.
pub fn arrow_host_import_call_count_for_test() -> usize {
    testing_hooks::host_import_call_count_for_test()
}

/// Read the current thread's last host-copy Arrow boundary observation, if one
/// has been recorded since the last reset on this thread.
pub fn last_arrow_host_boundary_observation_for_test() -> Option<BoundaryObservation> {
    testing_hooks::last_host_boundary_observation_for_test()
}
