use std::{
    sync::{Mutex, MutexGuard, OnceLock},
    time::Duration,
};

use super::support::{AdmissionTestController, test_ledger_with_explicit_profile};
use crate::runtime::{
    GpuBackendDeviceConfig, GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuDeviceMemoryLimit,
    GpuMemoryOwnership, QueryAdmissionRequest, QueryExecutionConfig, QueryService, QueueBounds,
    gpu_execution_pool::RESERVED_LIFECYCLE_WORKERS,
};

const TEST_DEVICE_CAPACITY_BYTES: u64 = 64 * 1024 * 1024;

fn service_claim_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().expect("multi-device registry test lock poisoned")
}

#[track_caller]
fn require_device_inventory(required: usize) {
    let available = cudf::DeviceId::count().expect("CUDA device enumeration");
    assert!(available >= required, "requires {required} GPU devices, found {available}");
}

fn test_device_profiles(
    device_ordinals: &[usize],
    max_active_attempts: usize,
    stream_pool_size: usize,
    default_query_limit_bytes: u64,
) -> Vec<GpuBackendDeviceConfig> {
    device_ordinals
        .iter()
        .copied()
        .map(|device_ordinal| {
            GpuBackendDeviceConfig::new(
                GpuBackendDeviceProfile::new(device_ordinal)
                    .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_CAPACITY_BYTES })
                    .with_default_query_limit_bytes(default_query_limit_bytes)
                    .with_source_work_per_attempt(1)
                    .with_max_active_attempts(max_active_attempts)
                    .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(
                        stream_pool_size,
                        max_active_attempts + RESERVED_LIFECYCLE_WORKERS,
                        max_active_attempts,
                    )),
                GpuMemoryOwnership::WholeDeviceExclusive,
            )
        })
        .collect()
}

fn runtime_test_service(
    config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    device_profiles: Vec<GpuBackendDeviceConfig>,
) -> QueryService {
    QueryService::new_with_device_profiles_for_test(config, queue_bounds, device_profiles)
        .expect("explicit multi-device runtime test profiles must install")
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn query_service_places_concurrent_runtime_handles_on_distinct_devices_when_available() {
    require_device_inventory(2);
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0, 1], 1, 6, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    let first_context = first.acquire_context().unwrap();
    let second_context = second.acquire_context().unwrap();
    assert_eq!(first_context.context().runtime().unwrap().device().ordinal(), first.device_ordinal());
    assert_eq!(second_context.context().runtime().unwrap().device().ordinal(), second.device_ordinal());

    let reports = service.device_ledger_reports().unwrap();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0].device_ordinal, 0);
    assert_eq!(reports[1].device_ordinal, 1);
    assert_eq!(reports[0].managed_capacity_bytes, TEST_DEVICE_CAPACITY_BYTES);
    assert_eq!(reports[1].managed_capacity_bytes, TEST_DEVICE_CAPACITY_BYTES);
    assert_eq!(reports[0].live_grants_bytes, first.device_budget_bytes());
    assert_eq!(reports[1].live_grants_bytes, second.device_budget_bytes());
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn alternating_multi_device_stream_cleanup_returns_each_pool_before_reuse() {
    require_device_inventory(2);
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0, 1], 1, 6, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.admit_query().unwrap();
    let first_context = first.acquire_context().unwrap();
    let second = service.admit_query().unwrap();
    let second_context = second.acquire_context().unwrap();
    assert_eq!((first.device_ordinal(), second.device_ordinal()), (0, 1));
    let mut active = [Some((first, first_context)), Some((second, second_context))];

    for released_ordinal in [0usize, 1, 0, 1, 0, 1] {
        let retained_ordinal = 1 - released_ordinal;
        let (released, released_context) = active[released_ordinal].take().expect("selected device has one query");
        drop(released_context);
        released.end_test_attempt();
        drop(released);
        service
            .wait_for_stream_pools_quiescent(&[(retained_ordinal, 1)], Duration::from_secs(5))
            .expect("released device cleanup completes while the other root lease remains checked out");

        let replacement = service.admit_query().expect("released device admits replacement");
        assert_eq!(replacement.device_ordinal(), released_ordinal);
        let replacement_context = replacement.acquire_context().unwrap();
        assert_eq!(replacement_context.context().runtime().unwrap().device().ordinal(), released_ordinal);
        active[released_ordinal] = Some((replacement, replacement_context));
    }

    for entry in &mut active {
        let (handle, context) = entry.take().unwrap();
        drop(context);
        handle.end_test_attempt();
        drop(handle);
    }
    service.wait_for_stream_pools_idle(Duration::from_secs(5)).unwrap();
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
}

#[test]
fn per_device_active_limit_forces_spread_and_then_blocks() {
    let service = AdmissionTestController::new_with_ledgers(
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        [0, 1].map(|ordinal| {
            test_ledger_with_explicit_profile(
                ordinal,
                TEST_DEVICE_CAPACITY_BYTES,
                TEST_DEVICE_CAPACITY_BYTES,
                1,
                Default::default(),
            )
        }),
    )
    .expect("explicit multi-device admission profiles");

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    assert!(service.try_admit_query().unwrap().is_none());
}

#[test]
fn least_loaded_placement_spreads_across_identical_devices() {
    let service = AdmissionTestController::new_with_ledgers(
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        [0, 1].map(|ordinal| {
            test_ledger_with_explicit_profile(
                ordinal,
                TEST_DEVICE_CAPACITY_BYTES,
                TEST_DEVICE_CAPACITY_BYTES / 2,
                2,
                Default::default(),
            )
        }),
    )
    .expect("explicit multi-device admission profiles");

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");

    // The normative spare-capacity ranking spreads across identical devices.
    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
}

#[test]
fn query_service_memory_failure_callback_ignores_budget_failures() {
    require_device_inventory(1);
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0], 1, 3, 64 * 1024),
    );

    let initial_report = service.device_memory_failure_reports().into_iter().next().unwrap();
    assert!(!initial_report.callback_install_unavailable, "CUDA allocation failure callback is required");
    assert!(initial_report.callback_installed);

    const GRANT_BYTES: u64 = 64 * 1024;

    // An explicit 64 KiB query cap makes the immutable attempt grant the
    // domain ceiling and the root member's hard limit. The 4 MiB allocation
    // must fail locally and must not count as a device-health failure.
    let handle = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(GRANT_BYTES))
        .unwrap()
        .expect("query should admit");
    assert_eq!(handle.device_budget_bytes(), GRANT_BYTES, "the ledger publishes the forced immutable grant");
    let context = handle.acquire_context().unwrap();
    let execution = context.context();
    let grant = execution.memory_grant().unwrap().expect("admitted context carries the immutable grant");
    assert_eq!(grant.budget_bytes(), GRANT_BYTES);
    let grant_id = grant.grant_id();
    assert_eq!(execution.attempt_allocation_domain_id(), Some(grant_id));

    let domain_before = execution
        .attempt_allocation_domain_usage()
        .unwrap()
        .expect("admitted context carries the query allocation domain");
    assert_eq!(domain_before.domain_id, grant_id);
    assert_eq!(domain_before.ceiling_bytes, GRANT_BYTES);
    assert_eq!(domain_before.allocation_limit_bytes, GRANT_BYTES);
    assert_eq!(
        domain_before.active_members, 3,
        "the root runtime, its one auxiliary provider fork, and the reserved root fork are the domain members"
    );

    let token = context
        .context()
        .execution_budget_token(
            cudf::ExecutionBudgetRequest::new(GRANT_BYTES as usize).with_boundary("algeon_failure_callback"),
        )
        .unwrap();
    let member_before = token.reservation_usage().unwrap();
    assert_eq!(member_before.allocation_limit_bytes, GRANT_BYTES);
    let error = token
        .run("runtime::series_from_value", |exec| {
            execution.runtime().unwrap().on(exec)?.series_from_value(1i32, 1024 * 1024)
        })
        .unwrap_err();
    assert_eq!(error.kind(), cudf::ErrorKind::ResourceExhausted);
    let budget_usage = *error.budget_exhaustion().and_then(|budget| budget.usage()).expect("member usage");
    assert_eq!(budget_usage.allocation_limit_bytes, GRANT_BYTES);

    let member_after = token.reservation_usage().unwrap();
    assert_eq!(member_after.allocation_limit_bytes, GRANT_BYTES, "the member limit is immutable");
    assert!(member_after.failed_allocation_count > member_before.failed_allocation_count);
    assert!(member_after.largest_failed_allocation_bytes > 0);
    assert_eq!(
        budget_usage.failed_allocation_count, member_after.failed_allocation_count,
        "the structured failure reports the same member counter"
    );

    let domain_after = execution.attempt_allocation_domain_usage().unwrap().unwrap();
    assert_eq!(domain_after.domain_id, grant_id);
    assert_eq!(domain_after.ceiling_bytes, GRANT_BYTES);
    assert_eq!(domain_after.allocation_limit_bytes, GRANT_BYTES);
    assert_eq!(domain_after.active_members, 3);
    assert_eq!(domain_after.current_bytes, member_after.current_bytes);
    assert_eq!(domain_after.peak_bytes, member_after.peak_bytes);
    assert_eq!(domain_after.current_allocations, member_after.current_allocations);
    assert_eq!(domain_after.peak_allocations, member_after.peak_allocations);
    // The root member rejects before an over-limit allocation reaches the
    // parent domain. The idle provider members contribute no live use, so the
    // domain counters still mirror the root member.
    assert_eq!(domain_after.failed_allocation_count, 0);
    assert_eq!(domain_after.largest_failed_allocation_bytes, 0);

    let report = service.device_memory_failure_reports().into_iter().next().unwrap();
    assert!(report.callback_installed);
    assert_eq!(report.allocation_failure_count, initial_report.allocation_failure_count);
    assert_eq!(report.total_failed_allocation_bytes, initial_report.total_failed_allocation_bytes);
    assert_eq!(report.largest_failed_allocation_bytes, initial_report.largest_failed_allocation_bytes);
    assert_eq!(report.last_failed_allocation_bytes, initial_report.last_failed_allocation_bytes);
    assert!(report.pool_usage_observation_count >= initial_report.pool_usage_observation_count);
}

#[test]
fn direct_test_services_do_not_retain_backend_ownership() {
    require_device_inventory(1);
    let _guard = service_claim_test_guard();
    let config = QueryExecutionConfig::default();
    let queue_bounds = QueueBounds { max_waiters: 2, max_wait: Duration::from_millis(500) };
    let profiles = test_device_profiles(&[0], 1, 3, TEST_DEVICE_CAPACITY_BYTES);
    let first_service = runtime_test_service(config.clone(), queue_bounds.clone(), profiles.clone());
    let second_service = QueryService::new_with_device_profiles_for_test(config, queue_bounds, profiles)
        .expect("test-only services borrow construction authority without becoming backend owners");

    drop(first_service);
    drop(second_service);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn escaped_query_anchor_blocks_only_its_selected_device() {
    require_device_inventory(2);
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 3, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0, 1], 1, 9, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    assert_eq!(first.device_ordinal(), 0);
    let admitted_context = first.acquire_context().unwrap();
    let escaped_context = admitted_context.context().clone();
    drop(admitted_context);
    first.end_test_attempt();

    let second = service.try_admit_query().unwrap().expect("other device remains available");
    assert_eq!(second.device_ordinal(), 1);
    assert!(service.try_admit_query().unwrap().is_none());

    drop(escaped_context);
    let third = service.admit_query().expect("the escaped device-zero anchor eventually drains");
    assert_eq!(third.device_ordinal(), 0);
}

#[test]
fn multi_device_admissions_preserve_each_profiles_fixed_default_query_limit() {
    let want_budget = TEST_DEVICE_CAPACITY_BYTES;
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let service = AdmissionTestController::new_with_ledgers(
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        [0, 1].map(|ordinal| {
            test_ledger_with_explicit_profile(
                ordinal,
                TEST_DEVICE_CAPACITY_BYTES,
                TEST_DEVICE_CAPACITY_BYTES / 2,
                2,
                Default::default(),
            )
        }),
    )
    .expect("explicit multi-device admission profiles");

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    assert_eq!(first.device_budget_bytes(), min_budget);
    assert_eq!(second.device_budget_bytes(), min_budget);
}

#[test]
fn multi_device_admissions_do_not_resize_the_fixed_default_query_limit_for_open_slots() {
    let want_budget = TEST_DEVICE_CAPACITY_BYTES;
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let service = AdmissionTestController::new_with_ledgers(
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        [0, 1].map(|ordinal| {
            test_ledger_with_explicit_profile(
                ordinal,
                TEST_DEVICE_CAPACITY_BYTES,
                TEST_DEVICE_CAPACITY_BYTES / 2,
                2,
                Default::default(),
            )
        }),
    )
    .expect("explicit multi-device admission profiles");

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");
    let third = service.try_admit_query().unwrap().expect("third query should admit");

    assert_eq!(first.device_budget_bytes(), min_budget);
    assert_eq!(second.device_budget_bytes(), min_budget);
    assert_eq!(third.device_budget_bytes(), min_budget);
}
