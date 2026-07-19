mod common;

use std::{
    sync::{
        Arc, Barrier, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use cudf::{
    DataFrame, DataType, DeviceId, DeviceMemoryFailureAction, DeviceMemoryFailureAllocationDomain,
    DeviceMemoryFailureSource, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions,
    DeviceMemoryResource, ErrorCode, ErrorKind, ErrorStatus, ExecutionBudgetEnforcement, ExecutionBudgetRequest,
    ExecutionBudgetStatus, QueryMemoryDomainOptions, Runtime, SortOptions, StreamReservationLease, TypeId,
    ops::expr::col, runtime::CudaStream, testing::prelude::*,
};
use cudf_sys::test_support;
use static_assertions::assert_not_impl_any;

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
        let detached = self.manager.reserve_stream(&self.stream, reservation_id, 1).is_ok();
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
/// so pool usage observed through [`DeviceMemoryManager::pool_usage`] is
/// shared mutable state across this binary's parallel test threads.
///
/// Tests that allocate, free, or trim through the shared device pool hold the
/// shared guard (they may run concurrently with each other). Tests that
/// assert absolute pool-usage values (`used_current_bytes == 0`,
/// before/after equality) hold the exclusive guard so no sibling can disturb
/// the pool inside their critical section.
///
/// Take the guard as the FIRST statement of the test body so every device
/// allocation drops — releasing its pool accounting — before the guard is
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
    unsafe { cudf::testing::core::adopt_owned_memory_resource_for_test(raw, upstream.device()) }
        .expect("failed to adopt fail-first memory resource")
}

fn current_device_resource_raw_value() -> usize {
    let mut out = std::ptr::null_mut();
    let status = unsafe { cudf_sys::test_support::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out as usize
}

fn expect_device_buffer_error(result: cudf::Result<cudf::core::DeviceBuffer>) -> cudf::Error {
    match result {
        Ok(_) => panic!("expected device buffer allocation to fail"),
        Err(error) => error,
    }
}

fn failure_source_code(source: DeviceMemoryFailureSource) -> u32 {
    match source {
        DeviceMemoryFailureSource::Physical => 0,
        DeviceMemoryFailureSource::ReservationLimit => 1,
        DeviceMemoryFailureSource::Unknown(value) => value,
    }
}

fn query_domain_pool_count_for_id(manager: &DeviceMemoryManager, domain_id: u64) -> usize {
    let mut raw_manager = std::ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_memory_manager_for_device(manager.device().ordinal() as i32, &mut raw_manager) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!raw_manager.is_null());
    let mut count = 0;
    let status = unsafe {
        cudf_sys::test_support::cudf_test_memory_manager_query_domain_pool_count_for_id(
            raw_manager,
            domain_id,
            &mut count,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    count
}

struct QueryDomainRoutePauseRelease;

impl Drop for QueryDomainRoutePauseRelease {
    fn drop(&mut self) {
        unsafe { cudf_sys::test_support::cudf_test_query_domain_route_pause_release() };
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
    let _pool_state = shared_device_pool_guard();
    let device = DeviceId::current().unwrap();
    let manager = DeviceMemoryManager::install_for_device_with_options(
        device,
        DeviceMemoryManagerOptions {
            base: DeviceMemoryManagerBase::CudaAsync { initial_pool_size: None, release_threshold: None },
            allocation_limit_bytes: None,
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
fn failure_callback_trims_idle_async_pool_and_retries_before_rust_callback() {
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
        "test setup must leave enough idle async-pool bytes to trigger trim retry: {before:?}"
    );

    let fail_first = fail_first_allocation_resource(&async_mr);
    let rust_callback_calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&rust_callback_calls);
    let callback = Arc::new(move |_bytes: usize| {
        callback_calls.fetch_add(1, Ordering::SeqCst);
        false
    });
    let retrying_mr = fail_first.with_failure_callback(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(retrying_mr).build().unwrap();

    let buffer = runtime.device_buffer(request_bytes).unwrap();
    runtime.synchronize().unwrap();

    assert_eq!(
        rust_callback_calls.load(Ordering::SeqCst),
        0,
        "native trim-retry should recover before invoking the Rust failure callback"
    );
    let after = async_mr.pool_usage().unwrap();
    assert!(
        after.reserved_current_bytes < before.reserved_current_bytes,
        "trim-retry should release idle async-pool pages before retrying: before={before:?} after={after:?}"
    );
    assert!(after.used_current_bytes >= request_bytes as u64);
    drop(buffer);
}

#[test]
fn detailed_failure_callback_reports_pool_usage_and_can_trim_retry() {
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
            if info.should_trim_and_retry() {
                DeviceMemoryFailureAction::TrimAndRetry
            } else {
                DeviceMemoryFailureAction::Decline
            }
        })
    };
    let retrying_mr = fail_first.with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(retrying_mr).build().unwrap();

    let buffer = runtime.device_buffer(request_bytes).unwrap();
    runtime.synchronize().unwrap();

    assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
    assert_eq!(seen_device.load(Ordering::SeqCst), device.ordinal());
    assert_eq!(seen_source.load(Ordering::SeqCst), 0);
    assert_eq!(seen_attempt.load(Ordering::SeqCst), 0);
    assert_eq!(seen_trim_count.load(Ordering::SeqCst), 0);
    assert_eq!(seen_requested.load(Ordering::SeqCst), request_bytes as u64);
    assert_eq!(seen_used.load(Ordering::SeqCst), 0);
    assert!(seen_reserved.load(Ordering::SeqCst) >= (10 * request_bytes) as u64);
    let after = async_mr.pool_usage().unwrap();
    assert!(
        after.reserved_current_bytes < before.reserved_current_bytes,
        "callback-requested trim should release idle async-pool pages: before={before:?} after={after:?}"
    );
    drop(buffer);
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
    let seen_should_trim = Arc::new(AtomicBool::new(true));
    let seen_trim_count = Arc::new(AtomicU32::new(u32::MAX));
    let seen_reservation_id = Arc::new(AtomicU64::new(0));
    let callback = {
        let callback_calls = Arc::clone(&callback_calls);
        let seen_source = Arc::clone(&seen_source);
        let seen_should_trim = Arc::clone(&seen_should_trim);
        let seen_trim_count = Arc::clone(&seen_trim_count);
        let seen_reservation_id = Arc::clone(&seen_reservation_id);
        Arc::new(move |info: cudf::DeviceMemoryFailureInfo| {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            seen_source.store(failure_source_code(info.source), Ordering::SeqCst);
            seen_should_trim.store(info.should_trim_and_retry(), Ordering::SeqCst);
            seen_trim_count.store(info.trim_retry_count, Ordering::SeqCst);
            if let Some(DeviceMemoryFailureAllocationDomain::StreamReservation { reservation_id }) =
                info.allocation_domain
            {
                seen_reservation_id.store(reservation_id, Ordering::SeqCst);
            }
            DeviceMemoryFailureAction::TrimAndRetry
        })
    };
    let callback_mr = manager.resource().with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let _lease = manager.reserve_stream(runtime.stream(), 9002, 1).unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
    assert_eq!(seen_source.load(Ordering::SeqCst), 1);
    assert!(!seen_should_trim.load(Ordering::SeqCst));
    assert_eq!(seen_trim_count.load(Ordering::SeqCst), 0);
    assert_eq!(seen_reservation_id.load(Ordering::SeqCst), 9002);

    let after = manager.pool_usage().unwrap();
    assert_eq!(
        after.reserved_current_bytes, before.reserved_current_bytes,
        "reservation-limit trim-and-retry requests must not trim the shared pool: before={before:?} after={after:?}"
    );
    assert_eq!(after.used_current_bytes, before.used_current_bytes);
}

#[test]
fn reservation_arena_failure_callback_reports_arena_and_member_identity() {
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
        if let Some(DeviceMemoryFailureAllocationDomain::ReservationArena { arena_id, reservation_id }) =
            info.allocation_domain
        {
            callback_arena_id.store(arena_id, Ordering::SeqCst);
            callback_reservation_id.store(reservation_id, Ordering::SeqCst);
        }
        DeviceMemoryFailureAction::Decline
    });
    let callback_mr = manager.resource().with_failure_callback_details(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let arena = manager.reservation_arena(30_005, 1).unwrap();
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
    assert_send_sync::<cudf::ReservationArena>();
    assert_send_sync::<StreamReservationLease>();

    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = rt.device_memory_manager().unwrap();
    let manager_mr = manager.resource().testing_raw_handle_value();

    assert_eq!(rt.memory_resource().testing_raw_handle_value(), manager_mr);

    let lease = rt.stream_reservation(42, 64 * 1024 * 1024).unwrap();
    assert_eq!(lease.stream().testing_raw_handle_value(), rt.stream().testing_raw_handle_value());
    assert_eq!(lease.memory_resource().testing_raw_handle_value(), manager_mr);

    let usage = lease.usage().unwrap();
    assert_eq!(usage.reservation_id, 42);
    assert!(usage.attached);
    assert_eq!(usage.allocation_limit_bytes, 64 * 1024 * 1024);
}

#[test]
fn reservation_arena_enforces_query_cap_across_streams() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let arena = manager.reservation_arena(30_001, 12 * 1024 * 1024).unwrap();
    let left_lease = arena.reserve_stream(left.stream(), 30_101, 16 * 1024 * 1024).unwrap();
    let right_lease = arena.reserve_stream(right.stream(), 30_202, 16 * 1024 * 1024).unwrap();

    let left_buffer = left.device_buffer(8 * 1024 * 1024).unwrap();
    let err = expect_device_buffer_error(right.device_buffer(8 * 1024 * 1024));

    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(left_lease.usage().unwrap().failed_allocation_count, 0);
    assert!(
        right_lease.usage().unwrap().failed_allocation_count > 0,
        "member usage should record failures caused by the parent arena cap"
    );
    let arena_usage = arena.usage().unwrap();
    assert_eq!(arena_usage.arena_id, 30_001);
    assert_eq!(arena_usage.active_members, 2);
    assert!(arena_usage.failed_allocation_count > 0);
    assert!(arena_usage.current_bytes >= 8 * 1024 * 1024);

    drop(left_buffer);
}

#[test]
fn reservation_arena_resize_grows_and_shrinks_parent_limit() {
    let _pool_state = shared_device_pool_guard();
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = left.device_memory_manager().unwrap();
    let arena = manager.reservation_arena(30_002, 12 * 1024 * 1024).unwrap();
    let _left_lease = arena.reserve_stream(left.stream(), 30_111, 32 * 1024 * 1024).unwrap();
    let _right_lease = arena.reserve_stream(right.stream(), 30_222, 32 * 1024 * 1024).unwrap();

    let left_buffer = left.device_buffer(8 * 1024 * 1024).unwrap();
    let err = expect_device_buffer_error(right.device_buffer(8 * 1024 * 1024));
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);

    arena.resize(20 * 1024 * 1024).unwrap();
    let right_buffer = right.device_buffer(8 * 1024 * 1024).unwrap();
    let grown = arena.usage().unwrap();
    assert_eq!(grown.allocation_limit_bytes, 20 * 1024 * 1024);
    assert!(grown.current_bytes >= 16 * 1024 * 1024);

    arena.resize(1).unwrap();
    let shrunk = arena.usage().unwrap();
    assert_eq!(shrunk.allocation_limit_bytes, 1);
    assert!(shrunk.current_aligned_bytes > shrunk.allocation_limit_bytes);
    let err = expect_device_buffer_error(right.device_buffer(1));
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);

    drop(left_buffer);
    drop(right_buffer);
    arena.resize(1024 * 1024).unwrap();
    let buffer = right.device_buffer(1024).unwrap();
    assert_eq!(buffer.size_bytes().unwrap(), 1024);
}

#[test]
fn reservation_arena_counts_retained_allocations_after_member_detach() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let arena = manager.reservation_arena(30_003, 16 * 1024 * 1024).unwrap();
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
        "retained allocation should keep charging the arena after its member detaches"
    );

    drop(buffer);

    let after_drop = arena.usage().unwrap();
    assert!(
        after_drop.current_bytes < after_detach.current_bytes,
        "dropping retained allocation must release arena accounting"
    );
}

#[test]
fn execution_budget_token_can_attach_as_arena_member() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let arena = manager.reservation_arena(30_004, 64 * 1024 * 1024).unwrap();
    let token = cudf::ExecutionBudgetToken::from_runtime_in_arena(
        &runtime,
        ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("arena_member"),
        &arena,
    )
    .unwrap();

    let outcome = runtime.series_from_value_with_budget(13i32, 4096, &token).unwrap();

    assert_eq!(token.enforcement(), ExecutionBudgetEnforcement::ReservationArenaMember);
    assert_eq!(outcome.report().enforcement(), ExecutionBudgetEnforcement::ReservationArenaMember);
    assert_eq!(outcome.report().status(), ExecutionBudgetStatus::ReservationEnforced);
    assert!(arena.usage().unwrap().current_bytes >= outcome.value().device_allocated_size_bytes().unwrap() as u64);
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
    let left_token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &left,
        ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_domain_member"),
        &domain,
    )
    .unwrap();
    let right_token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &right,
        ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_domain_member"),
        &domain,
    )
    .unwrap();

    let left_outcome = left.series_from_value_with_budget(13i32, 1_000_000, &left_token).unwrap();
    let right_outcome = right.series_from_value_with_budget(29i32, 4_000_000, &right_token).unwrap();

    assert_eq!(left_token.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(right_token.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(left_outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(right_outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(left_outcome.report().status(), ExecutionBudgetStatus::ReservationEnforced);
    assert_eq!(right_outcome.report().status(), ExecutionBudgetStatus::ReservationEnforced);

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
fn execution_budget_token_retains_source_runtime_until_token_drop() {
    let _pool_state = shared_device_pool_guard();
    let (runtime, drops) = runtime_with_budget_drop_probe();
    let token = cudf::ExecutionBudgetToken::from_runtime(
        &runtime,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("runtime_lifetime"),
    )
    .unwrap();

    drop(runtime);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(token);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn arena_budget_token_retains_source_runtime_after_external_handles_drop() {
    let _pool_state = shared_device_pool_guard();
    let (runtime, drops) = runtime_with_budget_drop_probe();
    let manager = runtime.device_memory_manager().unwrap();
    let arena = manager.reservation_arena(next_budget_lifetime_test_id(), 1024 * 1024).unwrap();
    let token = cudf::ExecutionBudgetToken::from_runtime_in_arena(
        &runtime,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("arena_runtime_lifetime"),
        &arena,
    )
    .unwrap();

    drop(arena);
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
    let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
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
    let token = cudf::ExecutionBudgetToken::from_runtime(
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
    let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("budgeted_output_runtime_lifetime"),
        &domain,
    )
    .unwrap();
    let outcome = runtime.series_from_value_with_budget(17i32, 4096, &token).unwrap();

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
    let token = cudf::ExecutionBudgetToken::from_runtime(
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
fn execution_budget_token_retention_has_no_runtime_affinity_cycle() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let affinity_anchor = runtime.affinity_anchor();
    let lease_anchor = runtime.lease_anchor();
    let token = cudf::ExecutionBudgetToken::from_runtime(
        &runtime,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("runtime_cycle_check"),
    )
    .unwrap();

    drop(runtime);
    assert!(affinity_anchor.is_alive());
    assert!(lease_anchor.is_current());

    drop(token);
    assert!(!lease_anchor.is_current());
    assert!(!affinity_anchor.is_alive());
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
    let token = cudf::ExecutionBudgetToken::from_runtime(
        &retained_runtime,
        ExecutionBudgetRequest::new(1024 * 1024).with_boundary("reservation_drop_order"),
    )
    .unwrap();
    drop(retained_runtime);

    assert!(
        manager.reserve_stream(token.stream(), next_budget_lifetime_test_id(), 1).is_err(),
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
    let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("domain_member_deallocation"),
        &domain,
    )
    .unwrap();

    let outcome = runtime.series_from_value_with_budget(13i32, 1_000_000, &token).unwrap();
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
fn query_domain_member_budget_token_resize_updates_hard_member_limit() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_013, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let mut token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("domain_member_resize"),
        &domain,
    )
    .unwrap();

    token.resize(1024).unwrap();

    assert_eq!(token.request().budget_bytes(), 1024);
    assert_eq!(token.reservation_usage().unwrap().allocation_limit_bytes, 1024);
    let err = runtime.series_from_value_with_budget(1i32, 1024, &token).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
}

#[test]
fn sequential_query_domain_budget_tokens_reattach_same_stream_after_drop() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_014, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();

    {
        let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
            &runtime,
            ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("first_domain_member"),
            &domain,
        )
        .unwrap();
        let outcome = runtime.series_from_value_with_budget(7i32, 1024, &token).unwrap();
        assert_eq!(outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    }

    let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("second_domain_member"),
        &domain,
    )
    .unwrap();
    let outcome = runtime.series_from_value_with_budget(9i32, 1024, &token).unwrap();

    assert_eq!(outcome.report().enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
}

#[test]
fn second_live_query_domain_budget_token_on_same_stream_is_structured_rejection() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_015, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _first = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("first_live_domain_member"),
        &domain,
    )
    .unwrap();

    let err = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
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

    assert_eq!(rt.memory_resource().testing_raw_handle_value(), manager_mr);
    assert_ne!(lease.memory_resource().testing_raw_handle_value(), manager_mr);
    assert_eq!(domain.usage().unwrap().domain_id, 40_001);
    let pool_usage = domain.pool_usage().unwrap();
    assert!(pool_usage.reserved_current_bytes >= pool_usage.used_current_bytes);
}

#[test]
fn query_memory_domain_reports_verified_driver_ceiling() {
    assert!(cudf::QueryMemoryDomain::driver_ceiling_supported());
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
    assert!(usage.driver_ceiling_verified);
    assert!(usage.effective_driver_ceiling_bytes >= usage.ceiling_bytes);
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
    let small = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &left,
        ExecutionBudgetRequest::new(1024).with_boundary("small_member"),
        &domain,
    )
    .unwrap();
    let large = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &right,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("large_member"),
        &domain,
    )
    .unwrap();

    let err = left.series_from_value_with_budget(1i32, 1024, &small).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(small.reservation_usage().unwrap().failed_allocation_count > 0);

    let outcome = right.series_from_value_with_budget(2i32, 1024, &large).unwrap();
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
    let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("direct_member"),
        &domain,
    )
    .unwrap();

    let outcome = runtime.series_from_value_with_budget(7i32, 4096, &token).unwrap();
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
fn query_memory_domain_resize_updates_gate_but_rejects_above_ceiling() {
    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = rt.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_002, QueryMemoryDomainOptions::new(1).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let _lease = domain.reserve_stream(rt.stream(), 40_201, 8 * 1024 * 1024).unwrap();

    let err = expect_device_buffer_error(rt.device_buffer(1024));
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(domain.usage().unwrap().failed_allocation_count > 0);

    domain.resize_grant(8 * 1024 * 1024).unwrap();
    let buffer = rt.device_buffer(1024).unwrap();
    assert_eq!(buffer.size_bytes().unwrap(), 1024);

    let err = domain.resize_grant(16 * 1024 * 1024).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    _lease.resize(1024).unwrap();
    assert_eq!(_lease.usage().unwrap().allocation_limit_bytes, 1024);
}

#[test]
fn query_memory_domain_and_legacy_reservation_are_mutually_exclusive_per_stream() {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(40_011, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();

    let _domain_lease = domain.reserve_stream(runtime.stream(), 41_101, 8 * 1024 * 1024).unwrap();
    let err = runtime.stream_reservation(41_102, 8 * 1024 * 1024).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn concurrent_query_domain_and_legacy_reservation_on_same_stream_never_both_attach() {
    let _pool_state = shared_device_pool_guard();

    for iteration in 0..64 {
        let runtime = Arc::new(Runtime::builder().with_device_memory_manager().build().unwrap());
        let manager = runtime.device_memory_manager().unwrap();
        let domain = Arc::new(
            manager
                .query_memory_domain(
                    50_000 + iteration,
                    QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024),
                )
                .unwrap(),
        );
        let barrier = Arc::new(Barrier::new(3));

        let legacy_runtime = Arc::clone(&runtime);
        let legacy_barrier = Arc::clone(&barrier);
        let legacy = std::thread::spawn(move || {
            legacy_barrier.wait();
            legacy_runtime.execution_budget_token(
                ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("concurrent_legacy_reservation"),
            )
        });

        let domain_runtime = Arc::clone(&runtime);
        let domain_barrier = Arc::clone(&barrier);
        let domain_handle = Arc::clone(&domain);
        let query_domain = std::thread::spawn(move || {
            domain_barrier.wait();
            cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
                &domain_runtime,
                ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("concurrent_query_domain"),
                domain_handle.as_ref(),
            )
        });

        barrier.wait();
        let legacy_attached = legacy.join().unwrap().is_ok();
        let domain_attached = query_domain.join().unwrap().is_ok();
        assert_ne!(
            legacy_attached, domain_attached,
            "iteration {iteration}: exactly one reservation family must attach to one stream"
        );
    }
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
        DeviceMemoryFailureAction::Decline
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
        DeviceMemoryFailureAction::Decline
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
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 0);
    let domain = manager
        .query_memory_domain(
            DOMAIN_ID,
            QueryMemoryDomainOptions::new(64 * 1024 * 1024).with_ceiling_bytes(64 * 1024 * 1024),
        )
        .unwrap();
    let lease = domain.reserve_stream(runtime.stream(), 61_001, 64 * 1024 * 1024).unwrap();
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 1);

    if unsafe { cudf_sys::test_support::cudf_test_query_domain_route_pause_arm(DOMAIN_ID) } == 0 {
        eprintln!("skipping deterministic query-domain teardown race; debug test hooks are disabled");
        return;
    }
    let pause_release = QueryDomainRoutePauseRelease;
    let worker_runtime = Arc::clone(&runtime);
    let allocation = std::thread::spawn(move || worker_runtime.device_buffer(16 * 1024 * 1024));
    let deadline = Instant::now() + Duration::from_secs(5);
    while unsafe { cudf_sys::test_support::cudf_test_query_domain_route_pause_reached() } == 0 {
        assert!(Instant::now() < deadline, "query-domain allocation did not reach the armed route pause");
        std::thread::yield_now();
    }

    drop(lease);
    drop(domain);
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 1);

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
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 1);

    let replacement_lease = replacement.reserve_stream(runtime.stream(), 61_102, 64 * 1024 * 1024).unwrap();
    assert_eq!(replacement_lease.usage().unwrap().current_allocations, 0);
    assert_eq!(replacement_lease.usage().unwrap().routed_total_allocations, 0);
    drop(replacement_lease);

    let legacy = runtime.stream_reservation(61_103, 64 * 1024 * 1024).unwrap();
    drop(legacy);

    let replacement_lease = replacement.reserve_stream(runtime.stream(), 61_104, 64 * 1024 * 1024).unwrap();

    std::thread::spawn(move || drop(buffer)).join().unwrap();
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 0);
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
fn query_domain_direct_output_keeps_private_pool_until_output_drop() {
    const DOMAIN_ID: u64 = 60_002;
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(
            DOMAIN_ID,
            QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024),
        )
        .unwrap();
    let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(8 * 1024 * 1024).with_boundary("retained_direct_member"),
        &domain,
    )
    .unwrap();
    let outcome = runtime.series_from_value_with_budget(11i32, 4096, &token).unwrap();
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 1);

    drop(token);
    drop(domain);
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 1);

    drop(outcome);
    runtime.synchronize().unwrap();
    assert_eq!(query_domain_pool_count_for_id(&manager, DOMAIN_ID), 0);
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
    let input_token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("filter_input"),
        &domain,
    )
    .unwrap();
    let input = runtime.series_from_value_with_budget(7i32, 4096, &input_token).unwrap().into_value();
    let frame = DataFrame::from_series([("value", input)]).unwrap();
    drop(input_token);

    let filter_token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("filter_output"),
        &domain,
    )
    .unwrap();
    let filtered =
        frame.with_execution(&filter_token.execution_overrides()).filter_expr(col("value").ge(7i32)).unwrap();

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
    let token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("detached_member_input"),
        &domain,
    )
    .unwrap();
    let input = runtime.series_from_value_with_budget(7i32, 4096, &token).unwrap().into_value();
    let frame = DataFrame::from_series([("value", input)]).unwrap();
    drop(token);

    let error = frame.filter_expr(col("value").ge(7i32)).unwrap_err();

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
    let input_token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("decimal_filter_input"),
        &domain,
    )
    .unwrap();
    let input = runtime.series_from_value_with_budget(7i32, 4096, &input_token).unwrap().into_value();
    let integer_frame = DataFrame::from_series([("value", input)]).unwrap();
    let decimal_frame = integer_frame
        .cast_column_with_budget("value", DataType::fixed_point(TypeId::Decimal128, -2), &input_token)
        .unwrap()
        .into_value();
    drop(integer_frame);
    drop(input_token);

    let filter_token = cudf::ExecutionBudgetToken::from_runtime_in_query_domain(
        &runtime,
        ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("decimal_filter_output"),
        &domain,
    )
    .unwrap();
    let filtered =
        decimal_frame.with_execution(&filter_token.execution_overrides()).filter_expr(col("value").ge(7i32)).unwrap();

    assert_eq!(filtered.num_rows(), 4096);
    assert!(filter_token.reservation_usage().unwrap().direct_total_allocations > 0);
}

#[test]
fn query_execution_context_reserves_stream_against_device_manager_resource() {
    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let query = rt.query_execution();
    let manager_mr = query.device_memory_manager().unwrap().resource().testing_raw_handle_value();

    let lease = query.reserve_stream(77, 32 * 1024 * 1024).unwrap();

    assert_eq!(lease.stream().testing_raw_handle_value(), query.stream().testing_raw_handle_value());
    assert_eq!(lease.memory_resource().testing_raw_handle_value(), manager_mr);

    let usage = lease.usage().unwrap();
    assert_eq!(usage.reservation_id, 77);
    assert!(usage.attached);
    assert_eq!(usage.allocation_limit_bytes, 32 * 1024 * 1024);
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
        let token = left
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_query"))
            .unwrap();
        let outcome = left.series_from_value_with_budget(11i32, 1024, &token).unwrap();
        let usage = token.reservation_usage().unwrap();
        assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 11);
        assert_eq!(outcome.report().status(), ExecutionBudgetStatus::ReservationEnforced);
        (
            outcome.value().device_allocated_size_bytes().unwrap(),
            usage.peak_bytes as usize,
            usage.current_bytes as usize,
            outcome.value().as_column().testing_execution_domain_mr_value(),
        )
    });

    let right_handle = std::thread::spawn(move || {
        let token = right
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_query"))
            .unwrap();
        let outcome = right.series_from_value_with_budget(29i32, 2048, &token).unwrap();
        let usage = token.reservation_usage().unwrap();
        assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 29);
        assert_eq!(outcome.report().status(), ExecutionBudgetStatus::ReservationEnforced);
        (
            outcome.value().device_allocated_size_bytes().unwrap(),
            usage.peak_bytes as usize,
            usage.current_bytes as usize,
            outcome.value().as_column().testing_execution_domain_mr_value(),
        )
    });

    let left_usage = left_handle.join().unwrap();
    let right_usage = right_handle.join().unwrap();

    assert_eq!(left_usage.3, device_mr);
    assert_eq!(right_usage.3, device_mr);
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
        let token = left
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_current_mr"))
            .unwrap();
        let outcome = left.series_from_value_with_budget(3i32, 1024, &token).unwrap();
        assert_eq!(outcome.value().as_column().testing_execution_domain_mr_value(), device_mr);
        assert_eq!(token.memory_resource().testing_raw_handle_value(), device_mr);
        assert_eq!(current_device_resource_raw_value(), device_mr);
    });

    let right_handle = std::thread::spawn(move || {
        set_current_device_for_runtime(&right);
        assert_eq!(current_device_resource_raw_value(), device_mr);
        let token = right
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_current_mr"))
            .unwrap();
        let outcome = right.series_from_value_with_budget(5i32, 2048, &token).unwrap();
        assert_eq!(outcome.value().as_column().testing_execution_domain_mr_value(), device_mr);
        assert_eq!(token.memory_resource().testing_raw_handle_value(), device_mr);
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
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("retained_output"))
        .unwrap();

    let outcome = rt.series_from_value_with_budget(7i32, 4096, &token).unwrap();
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
    let small = small_rt.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("small_query")).unwrap();
    let large = large_rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("large_query"))
        .unwrap();

    let err = small_rt.series_from_value_with_budget(1i32, 1024, &small).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(small.reservation_usage().unwrap().failed_allocation_count > 0);

    let outcome = large_rt.series_from_value_with_budget(2i32, 1024, &large).unwrap();
    assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 2);
    assert_eq!(large.reservation_usage().unwrap().failed_allocation_count, 0);
}

#[test]
fn failure_callback_memory_resource_retries_until_callback_declines() {
    let device = DeviceId::current().unwrap();
    let upstream = DeviceMemoryResource::cuda_on(device).unwrap();
    let limiting = upstream.limiting(1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |bytes: usize| {
        assert!(bytes > 0);
        callback_calls.fetch_add(1, Ordering::SeqCst) == 0
    });
    let callback_mr = limiting.with_failure_callback(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn failure_callback_memory_resource_catches_rust_panics() {
    let device = DeviceId::current().unwrap();
    let upstream = DeviceMemoryResource::cuda_on(device).unwrap();
    let limiting = upstream.limiting(1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |_bytes: usize| -> bool {
        callback_calls.fetch_add(1, Ordering::SeqCst);
        panic!("failure callback panic must not cross FFI");
    });
    let callback_mr = limiting.with_failure_callback(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();

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
fn failure_callback_memory_resource_caps_retries_when_callback_keeps_retrying() {
    let device = DeviceId::current().unwrap();
    let upstream = DeviceMemoryResource::cuda_on(device).unwrap();
    let limiting = upstream.limiting(1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |_bytes: usize| {
        callback_calls.fetch_add(1, Ordering::SeqCst);
        true
    });
    let callback_mr = limiting.with_failure_callback(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 8);
}

#[test]
fn failure_callback_memory_resource_retries_stream_reservation_limit_failures() {
    let manager_runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device = manager_runtime.device();
    let manager = manager_runtime.device_memory_manager().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let callback = Arc::new(move |bytes: usize| {
        assert!(bytes > 0);
        callback_calls.fetch_add(1, Ordering::SeqCst) == 0
    });
    let callback_mr = manager.resource().with_failure_callback(callback).unwrap();
    let runtime = Runtime::builder().with_device(device).with_memory_resource(callback_mr).build().unwrap();
    let _lease = manager.reserve_stream(runtime.stream(), 9001, 1).unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn stream_reservation_resize_grows_limit_for_later_allocations() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let lease = runtime.stream_reservation(9101, 1).unwrap();

    let err = runtime.series(&[1i32, 2, 3]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(lease.usage().unwrap().allocation_limit_bytes, 1);

    lease.resize(1 << 20).unwrap();

    let series = runtime.series(&[1i32, 2, 3]).unwrap();
    assert_eq!(series.len(), 3);
    let usage = lease.usage().unwrap();
    assert_eq!(usage.allocation_limit_bytes, 1 << 20);
    assert!(usage.peak_aligned_bytes > 0);
}

#[test]
fn stream_reservation_resize_allows_shrink_below_live_usage_and_blocks_future_allocations() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let lease = runtime.stream_reservation(9102, 1 << 20).unwrap();
    let retained = runtime.series(&[1i32, 2, 3]).unwrap();
    let live_usage = lease.usage().unwrap();
    assert!(live_usage.current_aligned_bytes > 1);

    lease.resize(1).unwrap();

    let shrunk_usage = lease.usage().unwrap();
    assert_eq!(shrunk_usage.allocation_limit_bytes, 1);
    assert!(shrunk_usage.current_aligned_bytes > shrunk_usage.allocation_limit_bytes);

    let err = runtime.series(&[4i32, 5, 6]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);

    drop(retained);
    lease.resize(1 << 20).unwrap();
    let series = runtime.series(&[4i32, 5, 6]).unwrap();
    assert_eq!(series.len(), 3);
}

#[test]
fn execution_budget_token_resize_grows_limit_and_updates_reports() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let mut token =
        runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("resize_grow")).unwrap();

    let err = runtime.series_from_value_with_budget(1i32, 3, &token).unwrap_err();
    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(token.request().budget_bytes(), 1);
    assert_eq!(token.reservation_usage().unwrap().allocation_limit_bytes, 1);

    token.resize(1 << 20).unwrap();

    let outcome = runtime.series_from_value_with_budget(2i32, 3, &token).unwrap();
    assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap(), vec![2, 2, 2]);
    assert_eq!(token.request().budget_bytes(), 1 << 20);
    assert_eq!(outcome.report().request().budget_bytes(), 1 << 20);
    let usage = token.reservation_usage().unwrap();
    assert_eq!(usage.allocation_limit_bytes, 1 << 20);
    assert!(usage.peak_aligned_bytes > 0);
}

#[test]
fn execution_budget_token_resize_shrinks_below_live_usage_and_blocks_future_allocations() {
    let _pool_state = shared_device_pool_guard();
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let mut token =
        runtime.execution_budget_token(ExecutionBudgetRequest::new(1 << 20).with_boundary("resize_shrink")).unwrap();
    let retained = runtime.series_from_value_with_budget(3i32, 3, &token).unwrap();
    let live_usage = token.reservation_usage().unwrap();
    assert!(live_usage.current_aligned_bytes > 1);

    token.resize(1).unwrap();

    assert_eq!(token.request().budget_bytes(), 1);
    let shrunk_usage = token.reservation_usage().unwrap();
    assert_eq!(shrunk_usage.allocation_limit_bytes, 1);
    assert!(shrunk_usage.current_aligned_bytes > shrunk_usage.allocation_limit_bytes);

    let err = runtime.series_from_value_with_budget(4i32, 3, &token).unwrap_err();
    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);

    drop(retained);
    token.resize(1 << 20).unwrap();
    let outcome = runtime.series_from_value_with_budget(5i32, 3, &token).unwrap();
    assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap(), vec![5, 5, 5]);
    assert_eq!(outcome.report().request().budget_bytes(), 1 << 20);
}

/// §10.1 invariant contract (replaces the per-op guard contract): with the
/// device memory domain installed, concurrent execution mixing default
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
            let sorted = series.sort_values(SortOptions::default()).unwrap();
            assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
        }
    });
    let owned_pool_worker = std::thread::spawn(move || {
        let rt = Runtime::builder().with_pool_memory_resource(64 * 1024 * 1024, None).build().unwrap();
        for _ in 0..40 {
            let series = rt.series(&[30i32, 10, 20]).unwrap();
            let sorted = series.sort_values(SortOptions::default()).unwrap();
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

/// §10.3 accounting correctness: two pipelines (two runtimes, each with its
/// own stream reservation) run temporary-heavy work concurrently. Temporaries
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
    let left_lease = left.stream_reservation(9101, 64 * 1024 * 1024).unwrap();
    let right_lease = right.stream_reservation(9202, 1).unwrap();

    let left_worker = std::thread::spawn(move || {
        let data: Vec<i32> = (0..ROWS as i32).rev().collect();
        let series = left.series(&data).unwrap();
        let sorted = series.sort_values(SortOptions::default()).unwrap();
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

/// §10.4 outputs routing: explicit memory resources still flow to output
/// allocations after the global lock removal — an owned-pool runtime's
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
    let sorted = series.sort_values(SortOptions::default()).unwrap();
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
