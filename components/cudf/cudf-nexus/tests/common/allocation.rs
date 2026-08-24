use std::sync::Mutex;

use cudf_sys::test_support;

/// Serializes test-only reset/read windows for the process-local allocation counter.
static ALLOCATION_ATTEMPT_COUNTER_LOCK: Mutex<()> = Mutex::new(());

/// Measure nonzero allocation attempts observed at the reservation-aware
/// current-resource adaptor boundary.
///
/// The measured closure must perform any required host-visible synchronization
/// before returning. This helper does not synchronize CUDA work on behalf of
/// the caller.
pub fn allocation_count_delta<T>(op: impl FnOnce() -> T) -> i64 {
    let _guard = ALLOCATION_ATTEMPT_COUNTER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    unsafe { test_support::cudf_test_memory_resource_reset_allocation_attempts() };
    let _result = op();
    let mut count = 0;
    let status = unsafe { test_support::cudf_test_memory_resource_allocation_attempt_count(&mut count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read allocation-attempt count");
    i64::try_from(count).expect("allocation-attempt count exceeds i64")
}
