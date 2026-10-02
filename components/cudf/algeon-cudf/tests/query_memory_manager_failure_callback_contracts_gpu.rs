mod common;

use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use cudf::{
    DeviceId, DeviceMemoryFailureAllocationDomain, DeviceMemoryManager, ErrorKind, ExecutionBudgetRequest, InstallSpec,
    QueryMemoryDomainOptions, Runtime, testing::prelude::*,
};

#[test]
fn manager_installed_failure_callback_observes_reservation_failures_and_preserves_stream_contract() {
    let device = DeviceId::current().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let query_domain_id = Arc::new(AtomicU64::new(0));
    let query_member_id = Arc::new(AtomicU64::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback_domain_id = Arc::clone(&query_domain_id);
    let callback_member_id = Arc::clone(&query_member_id);
    let callback = Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
        assert!(info.requested_bytes > 0);
        callback_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id, member_id }) =
            info.allocation_domain
        {
            callback_domain_id.store(domain_id, Ordering::SeqCst);
            callback_member_id.store(member_id, Ordering::SeqCst);
        }
    });
    let (manager, _) =
        DeviceMemoryManager::install(device, InstallSpec { on_failure: Some(callback), ..InstallSpec::default() })
            .unwrap();
    let state = DeviceMemoryManager::failure_callback_state_for_device(device).unwrap();
    assert!(state.exists);
    assert!(state.failure_callback_installed);
    assert!(!state.failure_callback_panicked);
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

    drop(first);
    let panicking_resource =
        manager.resource().with_failure_callback_details(Arc::new(|_| panic!("observation delivery failed"))).unwrap();
    let panicking_runtime =
        Runtime::builder().with_device(device).with_memory_resource(panicking_resource).build().unwrap();
    let _panicking_lease = manager.reserve_stream(panicking_runtime.stream(), 103, 1).unwrap();
    let err = panicking_runtime.series(&[1i32, 2, 3]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    let detail = err.ffi_detail().expect("canonical allocation error survives callback delivery failure");
    assert_eq!(detail.failure_source, cudf::CudfErrorFailureSource::ReservationLimit);
    assert!(!DeviceMemoryManager::failure_callback_state_for_device(device).unwrap().failure_callback_panicked);

    let domain = manager
        .query_memory_domain(90_001, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let direct = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(1).with_boundary("direct_domain_failure_callback"),
        &domain,
    )
    .unwrap();
    let err = runtime.series_from_value_with_budget(3i32, 1024, &direct).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(query_domain_id.load(Ordering::SeqCst), 90_001);
    assert_ne!(query_member_id.load(Ordering::SeqCst), 0);
}
