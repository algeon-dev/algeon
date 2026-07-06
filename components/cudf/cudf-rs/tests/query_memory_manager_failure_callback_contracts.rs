mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use cudf::{DeviceId, DeviceMemoryManager, ErrorKind, ExecutionBudgetRequest, Runtime, testing::prelude::*};

#[test]
fn manager_installed_failure_callback_observes_reservation_failures_and_preserves_stream_contract() {
    let device = DeviceId::current().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |bytes: usize| {
        assert!(bytes > 0);
        callback_calls.fetch_add(1, Ordering::SeqCst);
        false
    });
    let manager = DeviceMemoryManager::install_for_device_with_failure_callback(device, callback).unwrap();
    let state = DeviceMemoryManager::failure_callback_state_for_device(device).unwrap();
    assert!(state.exists);
    assert!(state.failure_callback_installed);
    let runtime = Runtime::builder().with_device(device).with_device_memory_manager().build().unwrap();

    assert_eq!(
        runtime.device_memory_manager().unwrap().resource().testing_raw_handle_value(),
        manager.resource().testing_raw_handle_value()
    );

    let token = runtime
        .execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("manager_failure_callback"))
        .unwrap();
    let err = runtime.series_from_value_with_budget(1i32, 1024, &token).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(token.reservation_usage().unwrap().failed_allocation_count > 0);
    assert!(
        calls.load(Ordering::SeqCst) > 0,
        "manager-level failure callback must observe reservation-limit allocation failures"
    );
    drop(token);

    let first = runtime.stream_reservation(101, 64 * 1024 * 1024).unwrap();

    let err = runtime.stream_reservation(102, 64 * 1024 * 1024).unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert!(first.usage().unwrap().attached);
}
