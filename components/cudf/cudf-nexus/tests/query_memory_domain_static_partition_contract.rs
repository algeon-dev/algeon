use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
};

use cudf::{
    DeviceId, DeviceMemoryFailureAction, DeviceMemoryFailureAllocationDomain, DeviceMemoryFailureSource,
    DeviceMemoryManager, DeviceMemoryManagerOptions, ErrorKind, QueryMemoryDomainOptions, Runtime,
};

const STATIC_PARTITION_BYTES: usize = 64 * 1024 * 1024;

fn static_partition_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn failure_source_code(source: DeviceMemoryFailureSource) -> u32 {
    match source {
        DeviceMemoryFailureSource::Physical => 0,
        DeviceMemoryFailureSource::ReservationLimit => 1,
        DeviceMemoryFailureSource::AllocationLimit => 2,
        DeviceMemoryFailureSource::Unknown(value) => value,
    }
}

#[test]
fn static_partition_allows_private_query_memory_domain() {
    let _guard = static_partition_test_lock().lock().unwrap();
    let device = DeviceId::current().unwrap();
    let options = DeviceMemoryManagerOptions {
        allocation_limit_bytes: Some(STATIC_PARTITION_BYTES),
        ..DeviceMemoryManagerOptions::default()
    };
    let manager = DeviceMemoryManager::install_for_device_with_options(device, options).unwrap();

    let domain = manager
        .query_memory_domain(
            92_001,
            QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(16 * 1024 * 1024),
        )
        .unwrap();

    assert_eq!(domain.identity().unwrap().device, device);
    assert_eq!(domain.usage().unwrap().allocation_limit_bytes, 8 * 1024 * 1024);
}

#[test]
fn static_partition_domain_oom_reports_query_identity_without_trim_or_retry() {
    let _guard = static_partition_test_lock().lock().unwrap();
    let device = DeviceId::current().unwrap();
    let manager = DeviceMemoryManager::install_for_device_with_options(
        device,
        DeviceMemoryManagerOptions {
            allocation_limit_bytes: Some(STATIC_PARTITION_BYTES),
            ..DeviceMemoryManagerOptions::default()
        },
    )
    .unwrap();

    let callback_calls = Arc::new(AtomicUsize::new(0));
    let seen_source = Arc::new(AtomicU32::new(u32::MAX));
    let seen_domain_id = Arc::new(AtomicU64::new(0));
    let seen_member_id = Arc::new(AtomicU64::new(0));
    let saw_trim_or_retry = Arc::new(AtomicBool::new(false));
    let callback = {
        let callback_calls = Arc::clone(&callback_calls);
        let seen_source = Arc::clone(&seen_source);
        let seen_domain_id = Arc::clone(&seen_domain_id);
        let seen_member_id = Arc::clone(&seen_member_id);
        let saw_trim_or_retry = Arc::clone(&saw_trim_or_retry);
        Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            seen_source.store(failure_source_code(info.source), Ordering::SeqCst);
            if let Some(DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id, member_id }) =
                info.allocation_domain
            {
                seen_domain_id.store(domain_id, Ordering::SeqCst);
                seen_member_id.store(member_id, Ordering::SeqCst);
            }
            saw_trim_or_retry.store(info.should_trim_and_retry() || info.trim_retry_count != 0, Ordering::SeqCst);
            // This must be ignored for a reservation/static-partition failure;
            // retrying or trimming a physical pool cannot make the cap grow.
            DeviceMemoryFailureAction::TrimAndRetry
        })
    };
    let callback_resource = manager.resource().with_failure_callback_details(callback).unwrap();
    let filling_runtime =
        Runtime::builder().with_device(device).with_memory_resource(manager.resource().clone()).build().unwrap();
    let held_manager_allocation = filling_runtime.device_buffer(STATIC_PARTITION_BYTES).unwrap();

    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_resource).build().unwrap();
    let domain = manager
        .query_memory_domain(92_002, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _lease = domain.reserve_stream(runtime.stream(), 92_102, 8 * 1024 * 1024).unwrap();

    let error = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
    assert_eq!(seen_source.load(Ordering::SeqCst), failure_source_code(DeviceMemoryFailureSource::ReservationLimit));
    assert_eq!(seen_domain_id.load(Ordering::SeqCst), 92_002);
    assert_eq!(seen_member_id.load(Ordering::SeqCst), 92_102);
    assert!(
        !saw_trim_or_retry.load(Ordering::SeqCst),
        "a static-partition failure must not request a physical-pool trim or retry"
    );

    drop(held_manager_allocation);
}
