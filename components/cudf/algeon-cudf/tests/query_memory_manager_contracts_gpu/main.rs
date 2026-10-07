#[path = "../support/common/mod.rs"]
mod common;

use std::{
    sync::{
        Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use cudf::{
    CudfErrorFailureSource, DataFrame, DataType, DeviceId, DeviceMemoryFailureAllocationDomain,
    DeviceMemoryFailureSource, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions,
    DeviceMemoryResource, ErrorCode, ErrorKind, ErrorStatus, Exec, ExecutionBudgetEnforcement, ExecutionBudgetRequest,
    InstallSpec, QueryMemoryDomainOptions, Result, Runtime, SortOptions, StreamReservationLease, TypeId,
    core::DeviceBuffer,
    ops::expr::col,
    runtime::CudaStream,
    testing::{core::adopt_owned_memory_resource_for_test, prelude::*},
};
use cudf_sys::test_support;
use static_assertions::assert_not_impl_any;

use crate::common::{execution_budget_token, execution_budget_token_in_domain, query_domain_member};

assert_not_impl_any!(cudf::ExecutionBudgetToken: Clone);

static NEXT_BUDGET_LIFETIME_TEST_ID: AtomicU64 = AtomicU64::new(700_000);

struct BudgetRuntimeDropProbe(Arc<AtomicUsize>);

impl Drop for BudgetRuntimeDropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct BudgetReservationDropOrderProbe {
    manager: DeviceMemoryManager,
    stream: CudaStream,
    reservation_detached: Arc<AtomicBool>,
}

impl Drop for BudgetReservationDropOrderProbe {
    fn drop(&mut self) {
        let reservation_id = NEXT_BUDGET_LIFETIME_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let detached = self
            .manager
            .query_memory_domain(reservation_id, QueryMemoryDomainOptions::new(1))
            .and_then(|domain| domain.reserve_stream(&self.stream, reservation_id, 1))
            .is_ok();
        self.reservation_detached.store(detached, Ordering::SeqCst);
    }
}

fn runtime_with_budget_drop_probe() -> (Runtime, Arc<AtomicUsize>) {
    let base = Runtime::builder().with_device_memory_manager().build().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let retained = base.with_retained_lifetime(Arc::new(BudgetRuntimeDropProbe(Arc::clone(&drops)))).unwrap();
    drop(base);
    (retained, drops)
}

fn next_budget_lifetime_test_id() -> u64 {
    NEXT_BUDGET_LIFETIME_TEST_ID.fetch_add(1, Ordering::Relaxed)
}

fn assert_send_sync<T: Send + Sync>() {}

/// The per-device memory manager and its CUDA async pool are process-global,
/// so pool usage observed through [`DeviceMemoryManager::pool_usage`] would be
/// shared mutable state if tests ever shared a process (e.g. under
/// `cargo test`); nextest runs each test in its own process, so this guard is
/// a no-op there and only matters for that shared-process case.
///
/// Tests that allocate, free, or trim through the shared device pool hold the
/// shared guard (they may run concurrently with each other). Tests that
/// assert absolute pool-usage values (`used_current_bytes == 0`,
/// before/after equality) hold the exclusive guard so no sibling can disturb
/// the pool inside their critical section.
///
/// Take the guard as the FIRST statement of the test body so every device
/// allocation drops; releasing its pool accounting; before the guard is
/// released.
static DEVICE_POOL_STATE: RwLock<()> = RwLock::new(());

fn shared_device_pool_guard() -> RwLockReadGuard<'static, ()> {
    DEVICE_POOL_STATE.read().unwrap_or_else(PoisonError::into_inner)
}

fn exclusive_device_pool_guard() -> RwLockWriteGuard<'static, ()> {
    DEVICE_POOL_STATE.write().unwrap_or_else(PoisonError::into_inner)
}

fn set_current_device_for_runtime(rt: &Runtime) {
    let status = unsafe { cudf_sys::cudf_cuda_set_device(rt.device().ordinal() as i32) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}

fn fail_first_allocation_resource(upstream: &DeviceMemoryResource) -> DeviceMemoryResource {
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        test_support::cudf_test_memory_resource_create_fail_first_allocation(
            upstream.as_borrowed_sys().as_raw(),
            &mut raw,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create fail-first memory resource");
    // SAFETY: the hook returned one newly owned resource tied to the same
    // device as `upstream`, and this test transfers its sole ownership to cudf.
    unsafe { adopt_owned_memory_resource_for_test(raw, upstream.device()) }
        .expect("failed to adopt fail-first memory resource")
}

fn current_device_resource_raw_value() -> usize {
    let mut out = std::ptr::null_mut();
    let status = unsafe { test_support::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out as usize
}

fn expect_device_buffer_error(result: Result<DeviceBuffer>) -> cudf::Error {
    match result {
        Ok(_) => panic!("expected device buffer allocation to fail"),
        Err(error) => error,
    }
}

fn failure_source_code(source: DeviceMemoryFailureSource) -> u32 {
    match source {
        DeviceMemoryFailureSource::Physical => 0,
        DeviceMemoryFailureSource::ReservationLimit => 1,
        DeviceMemoryFailureSource::AllocationLimit => 2,
        DeviceMemoryFailureSource::Unknown(value) => value,
    }
}

fn query_domain_count_for_id(manager: &DeviceMemoryManager, domain_id: u64) -> usize {
    let mut raw_manager = std::ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_memory_manager_for_device(manager.device().ordinal() as i32, &mut raw_manager) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!raw_manager.is_null());
    let mut count = 0;
    let status =
        unsafe { test_support::cudf_test_memory_manager_query_domain_count_for_id(raw_manager, domain_id, &mut count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    count
}

struct QueryDomainRoutePauseRelease;

impl Drop for QueryDomainRoutePauseRelease {
    fn drop(&mut self) {
        unsafe { test_support::cudf_test_query_domain_route_pause_release() };
    }
}

#[test]
fn default_runtime_uses_device_level_memory_manager_resource() {
    let rt = Runtime::builder().build().unwrap();
    let manager = rt.device_memory_manager().unwrap();

    assert_eq!(
        rt.memory_resource().testing_raw_handle_value(),
        manager.resource().testing_raw_handle_value(),
        "plain runtimes must use the per-device manager so normal execution does not create per-runtime current-MR switching pressure",
    );
}

#[test]
fn cuda_async_resource_exposes_pool_usage_and_explicit_trim() {
    let device = DeviceId::current().unwrap();
    let allocation_size = 32 * 1024 * 1024;
    let mr = DeviceMemoryResource::cuda_async_on(device, None, None).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(mr.clone()).build().unwrap();

    let buffer = runtime.device_buffer(allocation_size).unwrap();
    runtime.synchronize().unwrap();
    drop(buffer);
    runtime.synchronize().unwrap();

    let before = mr.pool_usage().unwrap();
    assert!(
        before.reserved_current_bytes >= allocation_size as u64,
        "async pool should retain idle bytes before explicit trim: {before:?}"
    );
    assert_eq!(before.used_current_bytes, 0);

    mr.trim_to(0).unwrap();

    let after = mr.pool_usage().unwrap();
    assert_eq!(after.used_current_bytes, 0);
    assert!(
        after.reserved_current_bytes < before.reserved_current_bytes,
        "explicit trim should release idle async-pool bytes: before={before:?} after={after:?}"
    );
}

#[test]
fn device_memory_manager_default_async_pool_exposes_usage_and_trim() {
    // The reserved/used relation is read as two independent native counters;
    // only the exclusive guard keeps sibling pool traffic from tearing the pair.
    let _pool_state = exclusive_device_pool_guard();
    let device = DeviceId::current().unwrap();
    let (manager, _) = DeviceMemoryManager::install(
        device,
        InstallSpec {
            options: DeviceMemoryManagerOptions {
                base: DeviceMemoryManagerBase::CudaAsync { initial_pool_size: None, release_threshold: None },
                allocation_limit_bytes: None,
            },
            ..InstallSpec::default()
        },
    )
    .unwrap();

    manager.trim_to(0).unwrap();
    let usage = manager.pool_usage().unwrap();
    assert!(usage.reserved_current_bytes >= usage.used_current_bytes);
    assert!(usage.reserved_high_bytes >= usage.reserved_current_bytes);
    assert!(usage.used_high_bytes >= usage.used_current_bytes);
}

#[test]
fn failure_callback_observes_physical_failure_without_trim_or_retry() {
    let device = DeviceId::current().unwrap();
    let idle_bytes = 128 * 1024 * 1024;
    let request_bytes = 1024 * 1024;
    let async_mr = DeviceMemoryResource::cuda_async_on(device, None, None).unwrap();
    let priming_runtime =
        Runtime::builder().with_device(device).with_memory_resource(async_mr.clone()).build().unwrap();

    let buffer = priming_runtime.device_buffer(idle_bytes).unwrap();
    priming_runtime.synchronize().unwrap();
    drop(buffer);
    priming_runtime.synchronize().unwrap();

    let before = async_mr.pool_usage().unwrap();
    assert_eq!(before.used_current_bytes, 0);
    assert!(
        before.reserved_current_bytes >= (10 * request_bytes) as u64,
        "test setup must leave idle async-pool bytes before the observed failure: {before:?}"
    );

    let fail_first = fail_first_allocation_resource(&async_mr);
    let rust_callback_calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&rust_callback_calls);
    let callback = Arc::new(move |_bytes: usize| {
        callback_calls.fetch_add(1, Ordering::SeqCst);
    });
    let observing_mr = fail_first.with_failure_callback(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(observing_mr).build().unwrap();

    let error = match runtime.device_buffer(request_bytes) {
        Ok(_) => panic!("observation callback must not retry the failed allocation"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::OutOfMemory);
    assert_eq!(rust_callback_calls.load(Ordering::SeqCst), 1);
    let after = async_mr.pool_usage().unwrap();
    assert_eq!(after.reserved_current_bytes, before.reserved_current_bytes);
    assert_eq!(after.used_current_bytes, before.used_current_bytes);
}

#[test]
fn detailed_failure_callback_reports_pool_usage_without_trim_or_retry() {
    let device = DeviceId::current().unwrap();
    let idle_bytes = 128 * 1024 * 1024;
    let request_bytes = 1024 * 1024;
    let async_mr = DeviceMemoryResource::cuda_async_on(device, None, None).unwrap();
    let priming_runtime =
        Runtime::builder().with_device(device).with_memory_resource(async_mr.clone()).build().unwrap();

    let buffer = priming_runtime.device_buffer(idle_bytes).unwrap();
    priming_runtime.synchronize().unwrap();
    drop(buffer);
    priming_runtime.synchronize().unwrap();

    let before = async_mr.pool_usage().unwrap();
    assert_eq!(before.used_current_bytes, 0);
    assert!(before.reserved_current_bytes >= (10 * request_bytes) as u64);

    let fail_first = fail_first_allocation_resource(&async_mr);
    let callback_calls = Arc::new(AtomicUsize::new(0));
    let seen_device = Arc::new(AtomicUsize::new(usize::MAX));
    let seen_source = Arc::new(AtomicU32::new(u32::MAX));
    let seen_attempt = Arc::new(AtomicU32::new(u32::MAX));
    let seen_trim_count = Arc::new(AtomicU32::new(u32::MAX));
    let seen_requested = Arc::new(AtomicU64::new(0));
    let seen_reserved = Arc::new(AtomicU64::new(0));
    let seen_used = Arc::new(AtomicU64::new(u64::MAX));
    let callback = {
        let callback_calls = Arc::clone(&callback_calls);
        let seen_device = Arc::clone(&seen_device);
        let seen_source = Arc::clone(&seen_source);
        let seen_attempt = Arc::clone(&seen_attempt);
        let seen_trim_count = Arc::clone(&seen_trim_count);
        let seen_requested = Arc::clone(&seen_requested);
        let seen_reserved = Arc::clone(&seen_reserved);
        let seen_used = Arc::clone(&seen_used);
        Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            seen_device.store(info.device.ordinal(), Ordering::SeqCst);
            seen_source.store(failure_source_code(info.source), Ordering::SeqCst);
            seen_attempt.store(info.callback_attempt, Ordering::SeqCst);
            seen_trim_count.store(info.trim_retry_count, Ordering::SeqCst);
            seen_requested.store(info.requested_bytes, Ordering::SeqCst);
            let pool_usage = info.pool_usage.expect("detailed callback should expose async pool usage");
            seen_reserved.store(pool_usage.reserved_current_bytes, Ordering::SeqCst);
            seen_used.store(pool_usage.used_current_bytes, Ordering::SeqCst);
        })
    };
    let observing_mr = fail_first.with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(observing_mr).build().unwrap();

    let error = match runtime.device_buffer(request_bytes) {
        Ok(_) => panic!("observation callback must not retry the failed allocation"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::OutOfMemory);
    assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
    assert_eq!(seen_device.load(Ordering::SeqCst), device.ordinal());
    assert_eq!(seen_source.load(Ordering::SeqCst), 0);
    assert_eq!(seen_attempt.load(Ordering::SeqCst), 0);
    assert_eq!(seen_trim_count.load(Ordering::SeqCst), 0);
    assert_eq!(seen_requested.load(Ordering::SeqCst), request_bytes as u64);
    assert_eq!(seen_used.load(Ordering::SeqCst), 0);
    assert!(seen_reserved.load(Ordering::SeqCst) >= (10 * request_bytes) as u64);
    let after = async_mr.pool_usage().unwrap();
    assert_eq!(after.reserved_current_bytes, before.reserved_current_bytes);
    assert_eq!(after.used_current_bytes, before.used_current_bytes);
}

#[test]
fn detailed_failure_callback_reports_reservation_limit_source_and_does_not_trim() {
    // This test asserts absolute shared-pool state (`used_current_bytes == 0`
    // and before/after usage equality), so it must exclude every sibling that
    // allocates from or trims the process-global device pool.
    let _pool_state = exclusive_device_pool_guard();
    let manager_runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device = manager_runtime.device();
    let manager = manager_runtime.device_memory_manager().unwrap();
    let idle_bytes = 128 * 1024 * 1024;

    manager.trim_to(0).unwrap();
    let buffer = manager_runtime.device_buffer(idle_bytes).unwrap();
    manager_runtime.synchronize().unwrap();
    drop(buffer);
    manager_runtime.synchronize().unwrap();

    let before = manager.pool_usage().unwrap();
    assert_eq!(before.used_current_bytes, 0);
    assert!(
        before.reserved_current_bytes >= idle_bytes as u64,
        "test setup must leave idle async-pool bytes before reservation rejection: {before:?}"
    );

    let callback_calls = Arc::new(AtomicUsize::new(0));
    let seen_source = Arc::new(AtomicU32::new(u32::MAX));
    let seen_trim_count = Arc::new(AtomicU32::new(u32::MAX));
    let seen_reservation_id = Arc::new(AtomicU64::new(0));
    let callback = {
        let callback_calls = Arc::clone(&callback_calls);
        let seen_source = Arc::clone(&seen_source);
        let seen_trim_count = Arc::clone(&seen_trim_count);
        let seen_reservation_id = Arc::clone(&seen_reservation_id);
        Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            seen_source.store(failure_source_code(info.source), Ordering::SeqCst);
            seen_trim_count.store(info.trim_retry_count, Ordering::SeqCst);
            if let Some(DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { member_id: reservation_id, .. }) =
                info.allocation_domain
            {
                seen_reservation_id.store(reservation_id, Ordering::SeqCst);
            }
        })
    };
    let callback_mr = manager.resource().with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let _lease = query_domain_member(&runtime, 9002, 1).unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
    assert_eq!(seen_source.load(Ordering::SeqCst), 1);
    assert_eq!(seen_trim_count.load(Ordering::SeqCst), 0);
    assert_eq!(seen_reservation_id.load(Ordering::SeqCst), 9002);

    let after = manager.pool_usage().unwrap();
    assert_eq!(
        after.reserved_current_bytes, before.reserved_current_bytes,
        "reservation-limit observation must not trim the shared pool: before={before:?} after={after:?}"
    );
    assert_eq!(after.used_current_bytes, before.used_current_bytes);
}

#[test]
fn query_domain_failure_callback_reports_arena_and_member_identity() {
    let manager_runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device = manager_runtime.device();
    let manager = manager_runtime.device_memory_manager().unwrap();
    let seen_source = Arc::new(AtomicU32::new(u32::MAX));
    let seen_arena_id = Arc::new(AtomicU64::new(0));
    let seen_reservation_id = Arc::new(AtomicU64::new(0));
    let callback_source = Arc::clone(&seen_source);
    let callback_arena_id = Arc::clone(&seen_arena_id);
    let callback_reservation_id = Arc::clone(&seen_reservation_id);
    let callback = Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
        callback_source.store(failure_source_code(info.source), Ordering::SeqCst);
        if let Some(DeviceMemoryFailureAllocationDomain::QueryMemoryDomain {
            domain_id: arena_id,
            member_id: reservation_id,
        }) = info.allocation_domain
        {
            callback_arena_id.store(arena_id, Ordering::SeqCst);
            callback_reservation_id.store(reservation_id, Ordering::SeqCst);
        }
    });
    let callback_mr = manager.resource().with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let arena = manager.query_memory_domain(30_005, QueryMemoryDomainOptions::new(1)).unwrap();
    let _lease = arena.reserve_stream(runtime.stream(), 30_505, 64 * 1024 * 1024).unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(seen_source.load(Ordering::SeqCst), failure_source_code(DeviceMemoryFailureSource::ReservationLimit));
    assert_eq!(seen_arena_id.load(Ordering::SeqCst), 30_005);
    assert_eq!(seen_reservation_id.load(Ordering::SeqCst), 30_505);
}

#[test]
fn device_memory_manager_runtime_uses_one_device_level_resource_with_stream_reservations() {
    assert_send_sync::<DeviceMemoryManager>();
    assert_send_sync::<cudf::QueryMemoryDomain>();
    assert_send_sync::<StreamReservationLease>();

    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = rt.device_memory_manager().unwrap();
    let manager_mr = manager.resource().testing_raw_handle_value();

    assert_eq!(rt.memory_resource().testing_raw_handle_value(), manager_mr);

    let lease = query_domain_member(&rt, 42, 64 * 1024 * 1024).unwrap();
    assert_eq!(lease.stream().testing_raw_handle_value(), rt.stream().testing_raw_handle_value());
    assert_ne!(lease.memory_resource().testing_raw_handle_value(), manager_mr);

    let usage = lease.usage().unwrap();
    assert_eq!(usage.reservation_id, 42);
    assert!(usage.attached);
    assert_eq!(usage.allocation_limit_bytes, 64 * 1024 * 1024);
}

#[test]
fn raw_stream_sibling_reservation_accounts_and_detaches_before_stream_destruction() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let arena = manager.query_memory_domain(30_901, QueryMemoryDomainOptions::new(4 * 1024 * 1024)).unwrap();
    let sibling = arena.reserve_stream(runtime.stream(), 30_902, 4 * 1024 * 1024).unwrap();
    let mut stream_handle = std::ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_create(&mut stream_handle) }, cudf_sys::CUDF_SUCCESS);
    let mut raw_stream = std::ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_raw(stream_handle, &mut raw_stream) }, cudf_sys::CUDF_SUCCESS);

    // SAFETY: the borrowed stream remains live until both attachments are dropped.
    let lease = unsafe { sibling.attach_sibling_raw_stream(raw_stream, 30_903, 4 * 1024 * 1024) }.unwrap();
    assert!(lease.usage().unwrap().attached);
    assert_eq!(lease.usage().unwrap().reservation_id, 30_903);
    drop(lease);
    // SAFETY: the borrowed stream is still live and is dropped after this attachment.
    let reattached = unsafe { sibling.attach_sibling_raw_stream(raw_stream, 30_904, 4 * 1024 * 1024) }.unwrap();
    assert_eq!(reattached.usage().unwrap().reservation_id, 30_904);
    drop(reattached);
    assert_eq!(unsafe { cudf_sys::cudf_stream_destroy(stream_handle) }, cudf_sys::CUDF_SUCCESS);
}

#[test]
fn raw_current_resource_route_accounts_and_detaches_before_stream_reuse() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let target = manager.resource().as_borrowed_sys();
    let mut stream_handle = std::ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_create(&mut stream_handle) }, cudf_sys::CUDF_SUCCESS);
    let mut raw_stream = std::ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_raw(stream_handle, &mut raw_stream) }, cudf_sys::CUDF_SUCCESS);

    // SAFETY: raw_stream remains live until every route using it is dropped.
    let route = unsafe { manager.route_current_resource_raw(raw_stream, manager.resource()) }.unwrap();
    let other_resource = DeviceMemoryResource::cuda_async_on(runtime.device(), None, None).unwrap();
    // SAFETY: raw_stream remains live, while the first route remains attached.
    assert!(unsafe { manager.route_current_resource_raw(raw_stream, &other_resource) }.is_err());
    let mut ptr = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            test_support::cudf_test_memory_resource_allocate(target.as_raw(), stream_handle, 4096, 256, &mut ptr)
        },
        cudf_sys::CUDF_SUCCESS
    );
    assert!(!ptr.is_null());
    drop(route);

    let mut rejected_ptr = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            test_support::cudf_test_memory_resource_allocate(
                target.as_raw(),
                stream_handle,
                4096,
                256,
                &mut rejected_ptr,
            )
        },
        cudf_sys::CUDF_ERROR_INVARIANT_VIOLATION
    );
    assert!(rejected_ptr.is_null());
    assert_eq!(
        unsafe { test_support::cudf_test_memory_resource_deallocate(target.as_raw(), stream_handle, ptr, 4096, 256) },
        cudf_sys::CUDF_SUCCESS
    );

    // SAFETY: the stream remains live and the preceding route has detached.
    let reused = unsafe { manager.route_current_resource_raw(raw_stream, manager.resource()) }.unwrap();
    drop(reused);
    assert_eq!(unsafe { cudf_sys::cudf_stream_destroy(stream_handle) }, cudf_sys::CUDF_SUCCESS);
}

#[test]
fn query_domain_enforces_query_cap_across_streams() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let arena = manager.query_memory_domain(30_001, QueryMemoryDomainOptions::new(12 * 1024 * 1024)).unwrap();
    let left_lease = arena.reserve_stream(left.stream(), 30_101, 16 * 1024 * 1024).unwrap();
    let right_lease = arena.reserve_stream(right.stream(), 30_202, 16 * 1024 * 1024).unwrap();

    let left_buffer = left.device_buffer(8 * 1024 * 1024).unwrap();
    let err = expect_device_buffer_error(right.device_buffer(8 * 1024 * 1024));

    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(left_lease.usage().unwrap().failed_allocation_count, 0);
    assert!(
        right_lease.usage().unwrap().failed_allocation_count > 0,
        "member usage should record failures caused by the parent domain cap"
    );
    let arena_usage = arena.usage().unwrap();
    assert_eq!(arena_usage.domain_id, 30_001);
    assert_eq!(arena_usage.active_members, 2);
    assert!(arena_usage.failed_allocation_count > 0);
    assert!(arena_usage.current_bytes >= 8 * 1024 * 1024);

    drop(left_buffer);
}

#[test]
fn query_domain_counts_retained_allocations_after_member_detach() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let arena = manager.query_memory_domain(30_003, QueryMemoryDomainOptions::new(16 * 1024 * 1024)).unwrap();
    let lease = arena.reserve_stream(runtime.stream(), 30_303, 16 * 1024 * 1024).unwrap();

    let buffer = runtime.device_buffer(4 * 1024 * 1024).unwrap();
    let while_attached = arena.usage().unwrap();
    assert_eq!(while_attached.active_members, 1);
    assert!(while_attached.current_bytes >= 4 * 1024 * 1024);

    drop(lease);

    let after_detach = arena.usage().unwrap();
    assert_eq!(after_detach.active_members, 0);
    assert!(
        after_detach.current_bytes >= 4 * 1024 * 1024,
        "retained allocation should keep charging the domain after its member detaches"
    );

    drop(buffer);

    let after_drop = arena.usage().unwrap();
    assert!(
        after_drop.current_bytes < after_detach.current_bytes,
        "dropping retained allocation must release domain accounting"
    );
}

#[test]
fn execution_budget_token_can_attach_as_query_domain_member() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            40_012,
            QueryMemoryDomainOptions::new(64 * 1024 * 1024).with_ceiling_bytes(64 * 1024 * 1024),
        )
        .unwrap();
    let left_token = execution_budget_token_in_domain(
        &left,
        ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_domain_member"),
        &domain,
    )
    .unwrap();
    let right_token = execution_budget_token_in_domain(
        &right,
        ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_domain_member"),
        &domain,
    )
    .unwrap();

    let left_outcome = left_token
        .run("runtime::series_from_value", |exec| left.on(exec)?.series_from_value(13i32, 1_000_000))
        .unwrap();
    let right_outcome = right_token
        .run("runtime::series_from_value", |exec| right.on(exec)?.series_from_value(29i32, 4_000_000))
        .unwrap();

    assert_eq!(left_token.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(right_token.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(left_outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(right_outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);

    let left_output_bytes = left_outcome.value().device_allocated_size_bytes().unwrap() as u64;
    let right_output_bytes = right_outcome.value().device_allocated_size_bytes().unwrap() as u64;
    let left_usage = left_token.reservation_usage().unwrap();
    let right_usage = right_token.reservation_usage().unwrap();
    assert!(left_usage.current_bytes >= left_output_bytes);
    assert!(right_usage.current_bytes >= right_output_bytes);
    assert!(
        left_usage.current_bytes < left_output_bytes + right_output_bytes,
        "left member usage must not include the right member allocation: left_usage={left_usage:?} left_output_bytes={left_output_bytes} right_output_bytes={right_output_bytes}",
    );

    let domain_usage = domain.usage().unwrap();
    assert_eq!(domain_usage.domain_id, 40_012);
    assert_eq!(domain_usage.active_members, 2);
    assert!(domain_usage.current_bytes >= left_output_bytes + right_output_bytes);
}

#[test]
fn frozen_query_domain_retains_live_output_and_rejects_new_allocations() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            next_budget_lifetime_test_id(),
            QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024),
        )
        .unwrap();
    let producer = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("frozen_domain_producer"),
        &domain,
    )
    .unwrap();
    let output =
        producer.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(17i32, 4096)).unwrap();
    drop(producer);

    let frozen = domain.freeze().unwrap();
    assert!(frozen.current_bytes > 0);
    assert!(frozen.current_aligned_bytes > 0);
    assert_eq!(domain.freeze().unwrap().current_aligned_bytes, frozen.current_aligned_bytes);

    let consumer = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("frozen_domain_rejected_allocation"),
        &domain,
    )
    .unwrap();
    let error =
        consumer.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(29i32, 1)).unwrap_err();
    // Allocating into a frozen domain is a routing/lifetime invariant
    // violation, not caller input: the producer was supposed to stop
    // allocating before its output was published.
    assert_eq!(error.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(error.kind(), ErrorKind::InvariantViolation);
    assert!(error.message().contains("frozen"), "unexpected error: {error}");

    drop(consumer);
    drop(output);
    let released = domain.usage().unwrap();
    assert_eq!(released.current_allocations, 0);
    assert_eq!(released.current_bytes, 0);
    assert_eq!(released.current_aligned_bytes, 0);
}

#[test]
fn execution_budget_token_retains_source_runtime_until_token_drop() {
    let _pool_state = shared_device_pool_guard();
    let (runtime, drops) = runtime_with_budget_drop_probe();
    let token =
        execution_budget_token(&runtime, ExecutionBudgetRequest::new(1024 * 1024).with_boundary("runtime_lifetime"))
            .unwrap();

    drop(runtime);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(token);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn query_domain_budget_token_retains_source_runtime_after_external_handles_drop() {
    let _pool_state = shared_device_pool_guard();
    let (runtime, drops) = runtime_with_budget_drop_probe();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            next_budget_lifetime_test_id(),
            QueryMemoryDomainOptions::new(1024 * 1024).with_ceiling_bytes(1024 * 1024),
        )
        .unwrap();
    let token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("query_domain_runtime_lifetime"),
        &domain,
    )
    .unwrap();

    drop(domain);
    drop(runtime);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(token);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn execution_budget_token_constructed_from_runtime_clone_retains_exact_source() {
    let _pool_state = shared_device_pool_guard();
    let (runtime, drops) = runtime_with_budget_drop_probe();
    let supplied_clone = runtime.clone();
    let token = execution_budget_token(
        &supplied_clone,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("runtime_clone_lifetime"),
    )
    .unwrap();

    drop(supplied_clone);
    drop(runtime);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(token);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn budgeted_output_inherits_retained_runtime_lifetime_after_token_drop() {
    let _pool_state = shared_device_pool_guard();
    let (runtime, drops) = runtime_with_budget_drop_probe();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            next_budget_lifetime_test_id(),
            QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024),
        )
        .unwrap();
    let token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("budgeted_output_runtime_lifetime"),
        &domain,
    )
    .unwrap();
    let outcome =
        token.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(17i32, 4096)).unwrap();

    drop(domain);
    drop(token);
    drop(runtime);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(outcome);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn execution_budget_token_retains_only_its_source_runtime() {
    let _pool_state = shared_device_pool_guard();
    let (runtime_a, drops_a) = runtime_with_budget_drop_probe();
    let (runtime_b, drops_b) = runtime_with_budget_drop_probe();
    let token = execution_budget_token(
        &runtime_a,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("runtime_a_lifetime"),
    )
    .unwrap();

    drop(runtime_a);
    drop(runtime_b);
    assert_eq!(drops_a.load(Ordering::SeqCst), 0);
    assert_eq!(drops_b.load(Ordering::SeqCst), 1);

    drop(token);
    assert_eq!(drops_a.load(Ordering::SeqCst), 1);
    assert_eq!(drops_b.load(Ordering::SeqCst), 1);
}

#[test]
fn execution_budget_token_retention_has_no_runtime_cycle() {
    let _pool_state = shared_device_pool_guard();
    let released = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_device_memory_manager()
        .build()
        .unwrap()
        .with_retained_lifetime(Arc::new(BudgetRuntimeDropProbe(Arc::clone(&released))))
        .unwrap();
    let token =
        execution_budget_token(&runtime, ExecutionBudgetRequest::new(1024 * 1024).with_boundary("runtime_cycle_check"))
            .unwrap();

    drop(runtime);
    assert_eq!(released.load(Ordering::SeqCst), 0, "the token retains the runtime's execution domain");

    drop(token);
    assert_eq!(released.load(Ordering::SeqCst), 1, "dropping the token must release the execution domain");
}

#[test]
fn execution_budget_token_detaches_reservation_before_retained_runtime_drop() {
    let _pool_state = shared_device_pool_guard();
    let base = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = base.device_memory_manager().unwrap();
    let reservation_detached = Arc::new(AtomicBool::new(false));
    let retained_runtime = base
        .with_retained_lifetime(Arc::new(BudgetReservationDropOrderProbe {
            manager: manager.clone(),
            stream: base.stream().clone(),
            reservation_detached: Arc::clone(&reservation_detached),
        }))
        .unwrap();
    drop(base);
    let token = execution_budget_token(
        &retained_runtime,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("reservation_drop_order"),
    )
    .unwrap();
    drop(retained_runtime);

    assert!(
        manager
            .query_memory_domain(next_budget_lifetime_test_id(), QueryMemoryDomainOptions::new(1))
            .unwrap()
            .reserve_stream(token.stream(), next_budget_lifetime_test_id(), 1)
            .is_err(),
        "the token reservation must still be attached before token drop"
    );
    assert!(!reservation_detached.load(Ordering::SeqCst));

    drop(token);
    assert!(
        reservation_detached.load(Ordering::SeqCst),
        "retained Runtime teardown must observe the token reservation already detached"
    );
}

#[test]
fn query_domain_member_usage_closes_when_budget_allocation_is_freed() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            40_016,
            QueryMemoryDomainOptions::new(16 * 1024 * 1024).with_ceiling_bytes(16 * 1024 * 1024),
        )
        .unwrap();
    let token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("domain_member_deallocation"),
        &domain,
    )
    .unwrap();

    let outcome =
        token.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(13i32, 1_000_000)).unwrap();
    let output_bytes = outcome.value().device_allocated_size_bytes().unwrap() as u64;
    assert!(token.reservation_usage().unwrap().current_bytes >= output_bytes);
    assert!(domain.usage().unwrap().current_bytes >= output_bytes);

    drop(outcome);
    runtime.synchronize().unwrap();

    let member_usage = token.reservation_usage().unwrap();
    assert_eq!(member_usage.current_bytes, 0);
    assert_eq!(member_usage.current_aligned_bytes, 0);
    assert_eq!(member_usage.current_allocations, 0);

    let domain_usage = domain.usage().unwrap();
    assert_eq!(domain_usage.current_bytes, 0);
    assert_eq!(domain_usage.current_aligned_bytes, 0);
    assert_eq!(domain_usage.current_allocations, 0);
    assert_eq!(domain_usage.active_members, 1);
}

#[test]
fn sequential_query_domain_budget_tokens_reattach_same_stream_after_drop() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_014, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();

    {
        let token = execution_budget_token_in_domain(
            &runtime,
            ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("first_domain_member"),
            &domain,
        )
        .unwrap();
        let outcome =
            token.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(7i32, 1024)).unwrap();
        assert_eq!(outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    }

    let token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("second_domain_member"),
        &domain,
    )
    .unwrap();
    let outcome =
        token.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(9i32, 1024)).unwrap();

    assert_eq!(outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
}

#[test]
fn second_live_query_domain_budget_token_on_same_stream_is_structured_rejection() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_015, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _first = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("first_live_domain_member"),
        &domain,
    )
    .unwrap();

    let err = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("second_live_domain_member"),
        &domain,
    )
    .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn query_memory_domain_preserves_device_resource_identity() {
    assert_send_sync::<cudf::QueryMemoryDomain>();

    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = rt.device_memory_manager().unwrap();
    let manager_mr = manager.resource().testing_raw_handle_value();
    let domain = manager
        .query_memory_domain(
            40_001,
            QueryMemoryDomainOptions::new(32 * 1024 * 1024).with_ceiling_bytes(64 * 1024 * 1024),
        )
        .unwrap();

    let lease = domain.reserve_stream(rt.stream(), 40_101, 32 * 1024 * 1024).unwrap();
    let buffer = rt.device_buffer(1024 * 1024).unwrap();
    rt.synchronize().unwrap();

    assert_eq!(rt.memory_resource().testing_raw_handle_value(), manager_mr);
    assert_ne!(lease.memory_resource().testing_raw_handle_value(), manager_mr);
    let domain_usage = domain.usage().unwrap();
    assert_eq!(domain_usage.domain_id, 40_001);
    assert!(domain_usage.current_bytes >= 1024 * 1024);
    let pool_usage = domain.pool_usage().unwrap();
    assert!(pool_usage.reserved_current_bytes >= pool_usage.used_current_bytes);
    assert_eq!(pool_usage, manager.pool_usage().unwrap());
    drop(buffer);
}

#[test]
fn query_memory_domain_ceiling_is_logical_accounting_only() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let requested_ceiling = 8 * 1024 * 1024;
    let domain = manager
        .query_memory_domain(
            40_017,
            QueryMemoryDomainOptions::new(requested_ceiling).with_ceiling_bytes(requested_ceiling),
        )
        .unwrap();

    let usage = domain.usage().unwrap();
    assert_eq!(usage.ceiling_bytes, requested_ceiling as u64);
    assert_eq!(usage.allocation_limit_bytes, requested_ceiling as u64);
}

#[test]
fn shared_pool_keeps_query_domain_limits_and_outputs_independent() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let domain_limit = 8 * 1024 * 1024;
    let left_domain = manager
        .query_memory_domain(40_117, QueryMemoryDomainOptions::new(domain_limit).with_ceiling_bytes(domain_limit))
        .unwrap();
    let right_domain = manager
        .query_memory_domain(40_118, QueryMemoryDomainOptions::new(domain_limit).with_ceiling_bytes(domain_limit))
        .unwrap();
    let _left_lease = left_domain.reserve_stream(left.stream(), 41_117, domain_limit).unwrap();
    let _right_lease = right_domain.reserve_stream(right.stream(), 41_118, domain_limit).unwrap();

    let right_output = right.series(&[7i32, 8, 9]).unwrap();
    let identity = left_domain.identity().unwrap();
    unsafe { test_support::cudf_test_query_domain_fail_allocation_after(identity.domain_id, 0) };
    let physical_started = Instant::now();
    let physical_error = left.series(&[1i32, 2, 3]).unwrap_err();
    let physical_failure_latency = physical_started.elapsed();
    unsafe { test_support::cudf_test_query_domain_reset_allocation_failure() };

    assert_eq!(physical_error.kind(), ErrorKind::ResourceExhausted);
    let detail = physical_error.ffi_detail().expect("physical failure keeps its canonical payload");
    assert_eq!(detail.failure_source, CudfErrorFailureSource::Physical);
    assert_eq!(detail.domain_id, 40_117);
    assert_eq!(detail.member_id, 41_117);
    assert_eq!(detail.generation, identity.generation);
    assert_eq!(left_domain.usage().unwrap().current_allocations, 0);

    let left_error = expect_device_buffer_error(left.device_buffer(domain_limit + 256));
    assert_eq!(left_error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(left_error.ffi_detail().unwrap().failure_source, CudfErrorFailureSource::ReservationLimit);
    assert_eq!(left_domain.usage().unwrap().current_allocations, 0);
    assert_eq!(right_output.to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);

    let alternating_started = Instant::now();
    let left_sorted =
        left.series(&[5i32, 1, 4, 2, 3]).unwrap().on(Exec::RETAINED).sort_values(SortOptions::default()).unwrap();
    let right_sorted = right_output.on(Exec::RETAINED).sort_values(SortOptions::default()).unwrap();
    left.synchronize().unwrap();
    right.synchronize().unwrap();
    let alternating_temp_latency = alternating_started.elapsed();
    assert_eq!(left_sorted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    assert_eq!(right_sorted.to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);

    let pool_usage = manager.pool_usage().unwrap();
    assert!(pool_usage.reserved_current_bytes >= pool_usage.used_current_bytes);
    assert!(pool_usage.reserved_current_bytes > 0);
    assert_eq!(left_domain.pool_usage().unwrap(), right_domain.pool_usage().unwrap());
    assert_eq!(right_domain.pool_usage().unwrap(), pool_usage);
    eprintln!(
        "E6 shared_pool physical_failure_ns={} alternating_temp_ns={} reserved_current_bytes={} reserved_high_bytes={} used_current_bytes={} used_high_bytes={}",
        physical_failure_latency.as_nanos(),
        alternating_temp_latency.as_nanos(),
        pool_usage.reserved_current_bytes,
        pool_usage.reserved_high_bytes,
        pool_usage.used_current_bytes,
        pool_usage.used_high_bytes,
    );
}

#[test]
fn query_domain_member_limit_is_independent_from_domain_grant() {
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            40_018,
            QueryMemoryDomainOptions::new(16 * 1024 * 1024).with_ceiling_bytes(16 * 1024 * 1024),
        )
        .unwrap();
    let small = execution_budget_token_in_domain(
        &left,
        ExecutionBudgetRequest::new(1024).with_boundary("small_member"),
        &domain,
    )
    .unwrap();
    let large = execution_budget_token_in_domain(
        &right,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("large_member"),
        &domain,
    )
    .unwrap();

    let err = small.run("runtime::series_from_value", |exec| left.on(exec)?.series_from_value(1i32, 1024)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(small.reservation_usage().unwrap().failed_allocation_count > 0);

    let outcome =
        large.run("runtime::series_from_value", |exec| right.on(exec)?.series_from_value(2i32, 1024)).unwrap();
    assert_eq!(outcome.value().len(), 1024);
    assert_eq!(large.reservation_usage().unwrap().failed_allocation_count, 0);
}

#[test]
fn query_domain_budget_output_uses_direct_member_resource_without_stream_route() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            40_019,
            QueryMemoryDomainOptions::new(16 * 1024 * 1024).with_ceiling_bytes(16 * 1024 * 1024),
        )
        .unwrap();
    let token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("direct_member"),
        &domain,
    )
    .unwrap();

    let outcome =
        token.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(7i32, 4096)).unwrap();
    let usage = token.reservation_usage().unwrap();
    assert!(usage.direct_total_allocations > 0);
    assert_eq!(usage.routed_total_allocations, 0);
    drop(outcome);
    runtime.synchronize().unwrap();
    assert_eq!(token.reservation_usage().unwrap().current_allocations, 0);
}

#[test]
fn query_memory_domain_usage_closes_when_allocation_is_freed() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_007, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _lease = domain.reserve_stream(runtime.stream(), 40_701, 8 * 1024 * 1024).unwrap();

    let buffer = runtime.device_buffer(1024).unwrap();
    let live = domain.usage().unwrap();
    assert!(live.current_bytes >= 1024);
    assert!(live.current_allocations > 0);

    drop(buffer);

    let released = domain.usage().unwrap();
    assert_eq!(released.current_bytes, 0);
    assert_eq!(released.current_aligned_bytes, 0);
    assert_eq!(released.current_allocations, 0);
}

#[test]
fn query_memory_domain_churn_reuses_grant_after_frees() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_008, QueryMemoryDomainOptions::new(4 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _lease = domain.reserve_stream(runtime.stream(), 40_801, 4 * 1024).unwrap();

    for _ in 0..16 {
        let buffer = runtime.device_buffer(1024).unwrap();
        drop(buffer);
    }

    let usage = domain.usage().unwrap();
    assert_eq!(usage.current_bytes, 0);
    assert_eq!(usage.current_allocations, 0);
    assert_eq!(usage.failed_allocation_count, 0);
    assert!(usage.total_bytes > usage.allocation_limit_bytes);
}

#[test]
fn query_memory_domain_frees_are_accounted_per_domain() {
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let left_domain = manager
        .query_memory_domain(40_009, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let right_domain = manager
        .query_memory_domain(40_010, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _left_lease = left_domain.reserve_stream(left.stream(), 40_901, 8 * 1024 * 1024).unwrap();
    let _right_lease = right_domain.reserve_stream(right.stream(), 41_001, 8 * 1024 * 1024).unwrap();

    let left_buffer = left.device_buffer(1024).unwrap();
    let right_buffer = right.device_buffer(2048).unwrap();
    assert!(left_domain.usage().unwrap().current_bytes >= 1024);
    assert!(right_domain.usage().unwrap().current_bytes >= 2048);

    drop(left_buffer);

    assert_eq!(left_domain.usage().unwrap().current_bytes, 0);
    assert!(right_domain.usage().unwrap().current_bytes >= 2048);

    drop(right_buffer);

    assert_eq!(right_domain.usage().unwrap().current_bytes, 0);
}

#[test]
fn query_memory_domain_failure_is_query_local() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let left_domain = manager
        .query_memory_domain(40_003, QueryMemoryDomainOptions::new(1).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let right_domain = manager
        .query_memory_domain(40_004, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _left_lease = left_domain.reserve_stream(left.stream(), 40_301, 8 * 1024 * 1024).unwrap();
    let _right_lease = right_domain.reserve_stream(right.stream(), 40_401, 8 * 1024 * 1024).unwrap();

    let err = expect_device_buffer_error(left.device_buffer(1024));
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(left_domain.usage().unwrap().failed_allocation_count > 0);

    let right_buffer = right.device_buffer(1024).unwrap();
    assert_eq!(right_buffer.size_bytes().unwrap(), 1024);
    assert_eq!(right_domain.usage().unwrap().failed_allocation_count, 0);
}

#[test]
fn query_memory_domain_failure_callback_reports_reservation_limit_source() {
    let manager_runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device = manager_runtime.device();
    let manager = manager_runtime.device_memory_manager().unwrap();
    let seen_source = Arc::new(AtomicU32::new(u32::MAX));
    let seen_domain_id = Arc::new(AtomicU64::new(0));
    let seen_member_id = Arc::new(AtomicU64::new(0));
    let callback_source = Arc::clone(&seen_source);
    let callback_domain_id = Arc::clone(&seen_domain_id);
    let callback_member_id = Arc::clone(&seen_member_id);
    let callback = Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
        callback_source.store(failure_source_code(info.source), Ordering::SeqCst);
        if let Some(DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id, member_id }) =
            info.allocation_domain
        {
            callback_domain_id.store(domain_id, Ordering::SeqCst);
            callback_member_id.store(member_id, Ordering::SeqCst);
        }
    });
    let callback_mr = manager.resource().with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let domain = manager
        .query_memory_domain(40_005, QueryMemoryDomainOptions::new(1).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _lease = domain.reserve_stream(runtime.stream(), 40_501, 8 * 1024 * 1024).unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(seen_source.load(Ordering::SeqCst), failure_source_code(DeviceMemoryFailureSource::ReservationLimit));
    assert_eq!(seen_domain_id.load(Ordering::SeqCst), 40_005);
    assert_eq!(seen_member_id.load(Ordering::SeqCst), 40_501);
}

#[test]
fn query_memory_domain_failure_identity_survives_fresh_thread_without_ambient_context() {
    let manager_runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device = manager_runtime.device();
    let manager = manager_runtime.device_memory_manager().unwrap();
    let seen_domain_id = Arc::new(AtomicU64::new(0));
    let seen_member_id = Arc::new(AtomicU64::new(0));
    let seen_source = Arc::new(AtomicU32::new(u32::MAX));
    let callback_domain_id = Arc::clone(&seen_domain_id);
    let callback_member_id = Arc::clone(&seen_member_id);
    let callback_source = Arc::clone(&seen_source);
    let callback = Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
        callback_source.store(failure_source_code(info.source), Ordering::SeqCst);
        if let Some(DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id, member_id }) =
            info.allocation_domain
        {
            callback_domain_id.store(domain_id, Ordering::SeqCst);
            callback_member_id.store(member_id, Ordering::SeqCst);
        }
    });
    let callback_mr = manager.resource().with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let domain = manager
        .query_memory_domain(40_012, QueryMemoryDomainOptions::new(1).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let lease = domain.reserve_stream(runtime.stream(), 41_201, 8 * 1024 * 1024).unwrap();

    let error = std::thread::spawn(move || {
        let error = runtime.series(&[1i32, 2, 3]).unwrap_err();
        drop(lease);
        drop(domain);
        error
    })
    .join()
    .unwrap();

    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(seen_source.load(Ordering::SeqCst), failure_source_code(DeviceMemoryFailureSource::ReservationLimit));
    assert_eq!(seen_domain_id.load(Ordering::SeqCst), 40_012);
    assert_eq!(seen_member_id.load(Ordering::SeqCst), 41_201);
}

#[test]
fn query_memory_domain_teardown_racing_allocation_blocks_stream_reattach_until_commit() {
    let _pool_state = shared_device_pool_guard();
    const DOMAIN_ID: u64 = 60_001;
    let runtime = Arc::new(Runtime::builder().with_device_memory_manager().build().unwrap());
    let manager = runtime.device_memory_manager().unwrap();
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 0);
    let domain = manager
        .query_memory_domain(
            DOMAIN_ID,
            QueryMemoryDomainOptions::new(64 * 1024 * 1024).with_ceiling_bytes(64 * 1024 * 1024),
        )
        .unwrap();
    let lease = domain.reserve_stream(runtime.stream(), 61_001, 64 * 1024 * 1024).unwrap();
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 1);

    if unsafe { test_support::cudf_test_query_domain_route_pause_arm(DOMAIN_ID) } == 0 {
        eprintln!("skipping deterministic query-domain teardown race; debug test hooks are disabled");
        return;
    }
    let pause_release = QueryDomainRoutePauseRelease;
    let worker_runtime = Arc::clone(&runtime);
    let allocation = std::thread::spawn(move || worker_runtime.device_buffer(16 * 1024 * 1024));
    let deadline = Instant::now() + Duration::from_secs(5);
    while unsafe { test_support::cudf_test_query_domain_route_pause_reached() } == 0 {
        assert!(Instant::now() < deadline, "query-domain allocation did not reach the armed route pause");
        std::thread::yield_now();
    }

    drop(lease);
    drop(domain);
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 1);

    let replacement = manager
        .query_memory_domain(
            60_101,
            QueryMemoryDomainOptions::new(64 * 1024 * 1024).with_ceiling_bytes(64 * 1024 * 1024),
        )
        .unwrap();
    let error = replacement.reserve_stream(runtime.stream(), 61_101, 64 * 1024 * 1024).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    drop(pause_release);
    let buffer = allocation.join().unwrap().unwrap();
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 1);

    let replacement_lease = replacement.reserve_stream(runtime.stream(), 61_102, 64 * 1024 * 1024).unwrap();
    assert_eq!(replacement_lease.usage().unwrap().current_allocations, 0);
    assert_eq!(replacement_lease.usage().unwrap().routed_total_allocations, 0);
    drop(replacement_lease);

    let replacement_lease = replacement.reserve_stream(runtime.stream(), 61_104, 64 * 1024 * 1024).unwrap();

    std::thread::spawn(move || drop(buffer)).join().unwrap();
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 0);
    let replacement_usage = replacement_lease.usage().unwrap();
    assert_eq!(replacement_usage.current_allocations, 0);
    assert_eq!(replacement_usage.routed_total_allocations, 0);

    drop(replacement_lease);
}

#[test]
fn query_memory_domain_retained_output_survives_domain_handle_drop() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();

    let retained = {
        let domain = manager
            .query_memory_domain(
                40_006,
                QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024),
            )
            .unwrap();
        let _lease = domain.reserve_stream(runtime.stream(), 40_601, 8 * 1024 * 1024).unwrap();
        runtime.device_buffer(1024).unwrap()
    };

    drop(retained);
}

#[test]
fn query_domain_pointer_registry_uses_recorded_owner_and_rejects_double_free() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_106, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let lease = domain.reserve_stream(runtime.stream(), 40_706, 8 * 1024 * 1024).unwrap();
    let member_resource = lease.memory_resource().as_borrowed_sys();
    let manager_resource = manager.resource().as_borrowed_sys();
    let stream = runtime.stream().as_borrowed_sys();
    let mut ptr = std::ptr::null_mut();

    let allocate_status = unsafe {
        test_support::cudf_test_memory_resource_allocate(member_resource.as_raw(), stream.as_raw(), 4096, 256, &mut ptr)
    };
    assert_eq!(allocate_status, cudf_sys::CUDF_SUCCESS);
    assert!(!ptr.is_null());
    assert_eq!(lease.usage().unwrap().current_allocations, 1);

    let deallocate_status = unsafe {
        test_support::cudf_test_memory_resource_deallocate(manager_resource.as_raw(), stream.as_raw(), ptr, 1, 1)
    };
    assert_eq!(deallocate_status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(lease.usage().unwrap().current_allocations, 0);
    assert_eq!(domain.usage().unwrap().current_allocations, 0);

    let second_deallocate_status = unsafe {
        test_support::cudf_test_memory_resource_deallocate(manager_resource.as_raw(), stream.as_raw(), ptr, 4096, 256)
    };
    assert_eq!(second_deallocate_status, cudf_sys::CUDF_ERROR_INVARIANT_VIOLATION);
    let message = unsafe { cudf_sys::cudf_last_error_message() };
    assert!(!message.is_null());
    let message = unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy();
    assert!(message.contains("unknown or already-freed device allocation pointer"));

    let invariant_error = match runtime.device_buffer(256) {
        Ok(_) => panic!("allocator accepted a new allocation after an invariant violation"),
        Err(error) => error,
    };
    assert_eq!(invariant_error.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(invariant_error.kind(), ErrorKind::InvariantViolation);
    let detail = invariant_error.ffi_detail().expect("sticky allocator invariant keeps its typed native payload");
    assert_eq!(detail.status, cudf_sys::CUDF_ERROR_INVARIANT_VIOLATION);
    assert!(detail.message.contains("allocator invariant violation"));
    assert!(detail.message.contains("unknown or already-freed device allocation pointer"));
}

#[test]
fn zero_byte_query_domain_allocation_does_not_change_counters_and_cleans_up_when_idle() {
    const DOMAIN_ID: u64 = 40_107;
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            DOMAIN_ID,
            QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024),
        )
        .unwrap();
    let lease = domain.reserve_stream(runtime.stream(), 40_707, 8 * 1024 * 1024).unwrap();
    let member_resource = lease.memory_resource().as_borrowed_sys();
    let stream = runtime.stream().as_borrowed_sys();
    let mut ptr = std::ptr::null_mut();

    let allocate_status = unsafe {
        test_support::cudf_test_memory_resource_allocate(member_resource.as_raw(), stream.as_raw(), 0, 256, &mut ptr)
    };
    assert_eq!(allocate_status, cudf_sys::CUDF_SUCCESS);
    let deallocate_status = unsafe {
        test_support::cudf_test_memory_resource_deallocate(member_resource.as_raw(), stream.as_raw(), ptr, 0, 256)
    };
    assert_eq!(deallocate_status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(lease.usage().unwrap().current_allocations, 0);
    assert_eq!(domain.usage().unwrap().current_allocations, 0);

    // The zero-byte release must not have latched the sticky unknown-pointer
    // invariant: a later ordinary allocation on the same member still succeeds
    // and accounts normally.
    let mut sized_ptr = std::ptr::null_mut();
    let sized_allocate_status = unsafe {
        test_support::cudf_test_memory_resource_allocate(
            member_resource.as_raw(),
            stream.as_raw(),
            4096,
            256,
            &mut sized_ptr,
        )
    };
    assert_eq!(sized_allocate_status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(lease.usage().unwrap().current_allocations, 1);
    let sized_deallocate_status = unsafe {
        test_support::cudf_test_memory_resource_deallocate(
            member_resource.as_raw(),
            stream.as_raw(),
            sized_ptr,
            4096,
            256,
        )
    };
    assert_eq!(sized_deallocate_status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(lease.usage().unwrap().current_allocations, 0);

    drop(lease);
    drop(domain);
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 0);
}

#[test]
fn query_domain_direct_output_keeps_domain_registry_until_output_drop() {
    const DOMAIN_ID: u64 = 60_002;
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            DOMAIN_ID,
            QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024),
        )
        .unwrap();
    let token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("retained_direct_member"),
        &domain,
    )
    .unwrap();
    let outcome =
        token.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(11i32, 4096)).unwrap();
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 1);

    drop(token);
    drop(domain);
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 1);

    drop(outcome);
    runtime.synchronize().unwrap();
    assert_eq!(query_domain_count_for_id(&manager, DOMAIN_ID), 0);
}

#[test]
fn fused_filter_literals_use_active_query_domain_member_after_input_member_detaches() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            60_003,
            QueryMemoryDomainOptions::new(32 * 1024 * 1024).with_ceiling_bytes(32 * 1024 * 1024),
        )
        .unwrap();
    let input_token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("filter_input"),
        &domain,
    )
    .unwrap();
    let input = input_token
        .run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(7i32, 4096))
        .unwrap()
        .into_value();
    let frame = DataFrame::from_series([("value", input)]).unwrap();
    drop(input_token);

    let filter_token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("filter_output"),
        &domain,
    )
    .unwrap();
    let filtered = frame.on(filter_token.exec()).filter_expr(col("value").ge(7i32)).unwrap();

    assert_eq!(filtered.num_rows(), 4096);
    assert!(filter_token.reservation_usage().unwrap().direct_total_allocations > 0);
}

#[test]
fn detached_query_domain_member_allocation_is_permanent_invalid_input() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            60_004,
            QueryMemoryDomainOptions::new(16 * 1024 * 1024).with_ceiling_bytes(16 * 1024 * 1024),
        )
        .unwrap();
    let token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("detached_member_input"),
        &domain,
    )
    .unwrap();
    let input = token
        .run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(7i32, 4096))
        .unwrap()
        .into_value();
    let frame = DataFrame::from_series([("value", input)]).unwrap();
    drop(token);

    let error = frame.on(Exec::RETAINED).filter_expr(col("value").ge(7i32)).unwrap_err();

    assert_eq!(error.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.status(), ErrorStatus::Permanent);
    assert!(error.to_string().contains("query memory domain member is detached"));
}

#[test]
fn fused_filter_decimal_rescaling_uses_replacement_query_domain_member() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            60_005,
            QueryMemoryDomainOptions::new(32 * 1024 * 1024).with_ceiling_bytes(32 * 1024 * 1024),
        )
        .unwrap();
    let input_token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("decimal_filter_input"),
        &domain,
    )
    .unwrap();
    let input = input_token
        .run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(7i32, 4096))
        .unwrap()
        .into_value();
    let integer_frame = DataFrame::from_series([("value", input)]).unwrap();
    let decimal_frame = input_token
        .run("dataframe::cast_column", |exec| {
            integer_frame.on(exec).cast_column("value", DataType::fixed_point(TypeId::Decimal128, -2).unwrap())
        })
        .unwrap()
        .into_value();
    drop(integer_frame);
    drop(input_token);

    let filter_token = execution_budget_token_in_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("decimal_filter_output"),
        &domain,
    )
    .unwrap();
    let filtered = decimal_frame.on(filter_token.exec()).filter_expr(col("value").ge(7i32)).unwrap();

    assert_eq!(filtered.num_rows(), 4096);
    assert!(filter_token.reservation_usage().unwrap().direct_total_allocations > 0);
}

#[test]
fn concurrent_stream_reservations_share_device_resource_without_counter_crosstalk() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = left.memory_resource().testing_raw_handle_value();

    assert_eq!(right.memory_resource().testing_raw_handle_value(), device_mr);
    assert_ne!(left.stream().testing_raw_handle_value(), right.stream().testing_raw_handle_value());

    let left_handle = std::thread::spawn(move || {
        let token =
            execution_budget_token(&left, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_query"))
                .unwrap();
        let outcome =
            token.run("runtime::series_from_value", |exec| left.on(exec)?.series_from_value(11i32, 1024)).unwrap();
        let usage = token.reservation_usage().unwrap();
        assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 11);
        (
            outcome.value().device_allocated_size_bytes().unwrap(),
            usage.peak_bytes as usize,
            usage.current_bytes as usize,
            outcome.value().as_column().testing_execution_domain_mr_value(),
        )
    });

    let right_handle = std::thread::spawn(move || {
        let token =
            execution_budget_token(&right, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_query"))
                .unwrap();
        let outcome =
            token.run("runtime::series_from_value", |exec| right.on(exec)?.series_from_value(29i32, 2048)).unwrap();
        let usage = token.reservation_usage().unwrap();
        assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 29);
        (
            outcome.value().device_allocated_size_bytes().unwrap(),
            usage.peak_bytes as usize,
            usage.current_bytes as usize,
            outcome.value().as_column().testing_execution_domain_mr_value(),
        )
    });

    let left_usage = left_handle.join().unwrap();
    let right_usage = right_handle.join().unwrap();

    assert_ne!(left_usage.3, device_mr);
    assert_ne!(right_usage.3, device_mr);
    assert!(left_usage.1 >= left_usage.0);
    assert!(right_usage.1 >= right_usage.0);
    assert!(left_usage.2 >= left_usage.0);
    assert!(right_usage.2 >= right_usage.0);
}

#[test]
fn concurrent_reserved_query_work_leaves_current_resource_on_device_adaptor() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = left.device_memory_manager().unwrap().resource().testing_raw_handle_value();

    assert_eq!(left.memory_resource().testing_raw_handle_value(), device_mr);
    assert_eq!(right.memory_resource().testing_raw_handle_value(), device_mr);
    set_current_device_for_runtime(&left);
    assert_eq!(current_device_resource_raw_value(), device_mr);

    let left_handle = std::thread::spawn(move || {
        set_current_device_for_runtime(&left);
        assert_eq!(current_device_resource_raw_value(), device_mr);
        let token = execution_budget_token(
            &left,
            ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_current_mr"),
        )
        .unwrap();
        let outcome =
            token.run("runtime::series_from_value", |exec| left.on(exec)?.series_from_value(3i32, 1024)).unwrap();
        assert_eq!(
            outcome.value().as_column().testing_execution_domain_mr_value(),
            token.memory_resource().testing_raw_handle_value()
        );
        assert_ne!(token.memory_resource().testing_raw_handle_value(), device_mr);
        assert_eq!(current_device_resource_raw_value(), device_mr);
    });

    let right_handle = std::thread::spawn(move || {
        set_current_device_for_runtime(&right);
        assert_eq!(current_device_resource_raw_value(), device_mr);
        let token = execution_budget_token(
            &right,
            ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_current_mr"),
        )
        .unwrap();
        let outcome =
            token.run("runtime::series_from_value", |exec| right.on(exec)?.series_from_value(5i32, 2048)).unwrap();
        assert_eq!(
            outcome.value().as_column().testing_execution_domain_mr_value(),
            token.memory_resource().testing_raw_handle_value()
        );
        assert_ne!(token.memory_resource().testing_raw_handle_value(), device_mr);
        assert_eq!(current_device_resource_raw_value(), device_mr);
    });

    left_handle.join().unwrap();
    right_handle.join().unwrap();
    set_current_device_for_runtime(&Runtime::builder().with_device_memory_manager().build().unwrap());
    assert_eq!(current_device_resource_raw_value(), device_mr);
}

#[test]
fn reservation_usage_releases_retained_output_by_allocation_identity() {
    let _pool_state = shared_device_pool_guard();
    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let token =
        execution_budget_token(&rt, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("retained_output"))
            .unwrap();

    let outcome = token.run("runtime::series_from_value", |exec| rt.on(exec)?.series_from_value(7i32, 4096)).unwrap();
    let output_bytes = outcome.value().device_allocated_size_bytes().unwrap();
    let while_retained = token.reservation_usage().unwrap();
    assert!(while_retained.current_bytes as usize >= output_bytes);

    drop(outcome);

    let after_drop = token.reservation_usage().unwrap();
    assert!(
        after_drop.current_bytes < while_retained.current_bytes,
        "dropping retained output must release bytes from the reservation that created it"
    );
}

#[test]
fn reservation_limit_failure_is_stream_local() {
    let _pool_state = shared_device_pool_guard();
    let small_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let large_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let small = execution_budget_token(&small_rt, ExecutionBudgetRequest::new(1).with_boundary("small_query")).unwrap();
    let large =
        execution_budget_token(&large_rt, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("large_query"))
            .unwrap();

    let err =
        small.run("runtime::series_from_value", |exec| small_rt.on(exec)?.series_from_value(1i32, 1024)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(small.reservation_usage().unwrap().failed_allocation_count > 0);

    let outcome =
        large.run("runtime::series_from_value", |exec| large_rt.on(exec)?.series_from_value(2i32, 1024)).unwrap();
    assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 2);
    assert_eq!(large.reservation_usage().unwrap().failed_allocation_count, 0);
}

#[test]
fn failure_callback_memory_resource_observes_once_and_preserves_failure() {
    let device = DeviceId::current().unwrap();
    let base = Runtime::builder().with_device(device).build().unwrap();
    let lease = query_domain_member(&base, 9003, 1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |bytes: usize| {
        assert!(bytes > 0);
        callback_calls.fetch_add(1, Ordering::SeqCst);
    });
    let callback_mr = lease.memory_resource().with_failure_callback(callback).unwrap();
    let runtime = base.with_memory_resource_override(&callback_mr, "test.callback_member").unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn failure_callback_memory_resource_catches_rust_panics() {
    let device = DeviceId::current().unwrap();
    let base = Runtime::builder().with_device(device).build().unwrap();
    let lease = query_domain_member(&base, 9003, 1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |_bytes: usize| {
        callback_calls.fetch_add(1, Ordering::SeqCst);
        panic!("failure callback panic must not cross FFI");
    });
    let callback_mr = lease.memory_resource().with_failure_callback(callback).unwrap();
    let runtime = base.with_memory_resource_override(&callback_mr, "test.callback_member").unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let err = runtime.series(&[4i32, 5, 6]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn failure_callback_memory_resource_never_retries() {
    let device = DeviceId::current().unwrap();
    let base = Runtime::builder().with_device(device).build().unwrap();
    let lease = query_domain_member(&base, 9003, 1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |_bytes: usize| {
        callback_calls.fetch_add(1, Ordering::SeqCst);
    });
    let callback_mr = lease.memory_resource().with_failure_callback(callback).unwrap();
    let runtime = base.with_memory_resource_override(&callback_mr, "test.callback_member").unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn failure_callback_memory_resource_observes_stream_reservation_limit_once() {
    let manager_runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device = manager_runtime.device();
    let manager = manager_runtime.device_memory_manager().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |bytes: usize| {
        assert!(bytes > 0);
        callback_calls.fetch_add(1, Ordering::SeqCst);
    });
    let callback_mr = manager.resource().with_failure_callback(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let _lease = query_domain_member(&runtime, 9001, 1).unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// With the device memory domain installed, concurrent execution mixing default
/// (manager-backed) runtimes and owned-pool runtimes must never disturb the
/// device's current resource. In the device-domain model nothing swaps it, so
/// any observed change means a per-op swap was reintroduced.
#[test]
fn concurrent_owned_pool_and_manager_runtimes_preserve_device_current_resource() {
    let _pool_state = shared_device_pool_guard();
    let manager_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = manager_rt.device_memory_manager().unwrap().resource().testing_raw_handle_value();
    set_current_device_for_runtime(&manager_rt);
    assert_eq!(current_device_resource_raw_value(), device_mr);

    let stop = Arc::new(AtomicBool::new(false));
    let poller = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut observations = 0u64;
            while !stop.load(Ordering::Relaxed) {
                assert_eq!(
                    current_device_resource_raw_value(),
                    device_mr,
                    "device current resource changed during concurrent mixed-runtime execution"
                );
                observations += 1;
            }
            observations
        })
    };

    let manager_worker = std::thread::spawn(move || {
        for _ in 0..40 {
            let series = manager_rt.series(&[3i32, 1, 2, 5, 4]).unwrap();
            let sorted = series.on(Exec::RETAINED).sort_values(SortOptions::default()).unwrap();
            assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
        }
    });
    let owned_pool_worker = std::thread::spawn(move || {
        let rt = Runtime::builder().with_pool_memory_resource(64 * 1024 * 1024, None).build().unwrap();
        for _ in 0..40 {
            let series = rt.series(&[30i32, 10, 20]).unwrap();
            let sorted = series.on(Exec::RETAINED).sort_values(SortOptions::default()).unwrap();
            assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
        }
    });

    manager_worker.join().unwrap();
    owned_pool_worker.join().unwrap();
    stop.store(true, Ordering::Relaxed);
    let observations = poller.join().unwrap();
    assert!(observations > 0, "invariant poller never observed the current resource");
    assert_eq!(current_device_resource_raw_value(), device_mr);
}

/// Two pipelines, each with its own runtime and stream reservation, run
/// temporary-heavy work concurrently. Temporaries
/// resolve through the device adaptor and are charged by stream, so each
/// pipeline's reservation must see its own usage (including temporaries) and
/// an over-limit failure must stay local to its pipeline.
#[test]
fn concurrent_temp_heavy_pipelines_keep_reservation_accounting_isolated() {
    let _pool_state = shared_device_pool_guard();
    const ROWS: usize = 1 << 20;
    const ROW_BYTES: usize = ROWS * std::mem::size_of::<i32>();

    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let left_lease = query_domain_member(&left, 9101, 64 * 1024 * 1024).unwrap();
    let right_lease = query_domain_member(&right, 9202, 1).unwrap();

    let left_worker = std::thread::spawn(move || {
        let data: Vec<i32> = (0..ROWS as i32).rev().collect();
        let series = left.series(&data).unwrap();
        let sorted = series.on(Exec::RETAINED).sort_values(SortOptions::default()).unwrap();
        assert_eq!(sorted.len(), ROWS);
        let usage = left_lease.usage().unwrap();
        assert_eq!(usage.reservation_id, 9101);
        assert_eq!(usage.failed_allocation_count, 0, "well-budgeted pipeline must not observe failures");
        // Input column + sort temporaries (order map, materialized output) all
        // charge this reservation: the peak must clearly exceed one column.
        assert!(
            usage.peak_bytes as usize >= 2 * ROW_BYTES,
            "sort temporaries were not charged to the owning pipeline's reservation: peak {} < {}",
            usage.peak_bytes,
            2 * ROW_BYTES
        );
        usage.peak_bytes
    });

    let right_worker = std::thread::spawn(move || {
        // A 1-byte budget cannot even hold the input column: the failure must
        // surface in this pipeline only, as a resource-exhausted error.
        let err = right.series(&[1i32, 2, 3]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
        let usage = right_lease.usage().unwrap();
        assert_eq!(usage.reservation_id, 9202);
        assert!(usage.failed_allocation_count > 0);
        assert_eq!(usage.current_bytes, 0, "failed allocations must not leak accounted bytes");
    });

    let left_peak = left_worker.join().unwrap();
    right_worker.join().unwrap();
    assert!(left_peak > 0);
}

/// Explicit memory resources flow to output allocations: an owned-pool runtime's
/// outputs carry its pool, while the device current resource (temporaries'
/// route) stays on the device adaptor.
#[test]
fn explicit_output_memory_resource_routing_survives_lock_removal() {
    let manager_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = manager_rt.device_memory_manager().unwrap().resource().testing_raw_handle_value();

    let owned = Runtime::builder().with_pool_memory_resource(64 * 1024 * 1024, None).build().unwrap();
    let owned_mr = owned.memory_resource().testing_raw_handle_value();
    assert_ne!(owned_mr, device_mr);

    let series = owned.series(&[5i32, 1, 4]).unwrap();
    let sorted = series.on(Exec::RETAINED).sort_values(SortOptions::default()).unwrap();
    assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 4, 5]);
    assert_eq!(
        sorted.as_column().testing_execution_domain_mr_value(),
        owned_mr,
        "outputs must keep routing through the explicitly provided memory resource"
    );

    set_current_device_for_runtime(&owned);
    assert_eq!(
        current_device_resource_raw_value(),
        device_mr,
        "running on an owned-pool runtime must not install its pool as the device current resource"
    );
}

#[test]
fn shared_member_lease_keeps_member_attached_until_last_clone_drops() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_020, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();

    let lease = domain.reserve_stream(runtime.stream(), 9_001, 8 * 1024 * 1024).unwrap();
    let clone = lease.clone();
    drop(lease);
    let still_attached = domain.reserve_stream(runtime.stream(), 9_002, 8 * 1024 * 1024).unwrap_err();
    assert_eq!(
        still_attached.kind(),
        ErrorKind::InvalidInput,
        "a live clone must keep the member attached: {still_attached}"
    );
    drop(clone);
    domain
        .reserve_stream(runtime.stream(), 9_003, 8 * 1024 * 1024)
        .expect("member detaches once the final shared lease drops");
}

#[test]
fn budget_token_from_shared_lease_allocates_without_a_second_attach() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            40_021,
            QueryMemoryDomainOptions::new(64 * 1024 * 1024).with_ceiling_bytes(64 * 1024 * 1024),
        )
        .unwrap();
    let lease = domain.reserve_stream(runtime.stream(), 9_101, 64 * 1024 * 1024).unwrap();

    let first = cudf::ExecutionBudgetToken::from_shared_lease(
        &runtime,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("first_shared_stage"),
        lease.clone(),
    );
    let second = cudf::ExecutionBudgetToken::from_shared_lease(
        &runtime,
        ExecutionBudgetRequest::new(128 * 1024 * 1024).with_boundary("second_shared_stage"),
        lease.clone(),
    );
    let first_outcome =
        first.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(3i32, 1_000_000)).unwrap();
    let second_outcome =
        second.run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(5i32, 1_000_000)).unwrap();
    assert_eq!(first.member_limit_bytes().unwrap(), 64 * 1024 * 1024);
    assert_eq!(second.member_limit_bytes().unwrap(), 64 * 1024 * 1024);
    assert_eq!(first_outcome.report().request().budget_bytes(), 1024 * 1024);
    assert_eq!(second_outcome.report().request().budget_bytes(), 128 * 1024 * 1024);
    assert_eq!(first_outcome.report().member_limit_bytes(), Some(64 * 1024 * 1024));
    assert_eq!(second_outcome.report().member_limit_bytes(), Some(64 * 1024 * 1024));

    let error = first
        .run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(7i32, 20_000_000))
        .unwrap_err();
    assert_eq!(error.budget_exhaustion().unwrap().usage().unwrap().allocation_limit_bytes, 64 * 1024 * 1024);

    drop(first);
    drop(second);
    let usage = lease.usage().unwrap();
    assert!(usage.current_bytes > 0, "shared member tracks both stage allocations");
    let domain_usage = domain.usage().unwrap();
    assert_eq!(domain_usage.domain_id, 40_021);
    assert!(domain_usage.active_members >= 1);
    drop(first_outcome);
    drop(second_outcome);
}
