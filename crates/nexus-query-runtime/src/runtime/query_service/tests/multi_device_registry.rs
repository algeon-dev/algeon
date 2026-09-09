use std::{
    sync::{Mutex, MutexGuard, OnceLock, mpsc},
    time::Duration,
};

use nexus_query_planning::admission::{AttemptRequirement, EstimateQuality};

use crate::runtime::{
    DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendDeviceConfig, GpuBackendDeviceInstallationBounds,
    GpuBackendDeviceProfile, GpuDeviceMemoryLimit, GpuMemoryOwnership, KvikioCompatModeStatus, NvidiaFsProbeStatus,
    QueryExecutionConfig, QueryService, QueueBounds,
};

const TEST_DEVICE_CAPACITY_BYTES: u64 = 64 * 1024 * 1024;

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

fn service_claim_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().expect("multi-device registry test lock poisoned")
}

/// Tests that construct real CUDA backends are unsupported on hosts without
/// the required inventory. Rust's test harness has no runtime `skip` result,
/// so make that outcome explicit rather than silently returning.
#[track_caller]
fn has_required_device_inventory(required: usize) -> bool {
    let location = std::panic::Location::caller();
    match cudf::DeviceId::count() {
        Ok(available) if available >= required => true,
        Ok(available) => {
            eprintln!(
                "SKIP unsupported_gpu_inventory at {location}: requires {required} GPU device(s), found {available}"
            );
            false
        },
        Err(error) => {
            eprintln!("SKIP unsupported_gpu_inventory at {location}: could not enumerate GPUs: {error}");
            false
        },
    }
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
                        max_active_attempts,
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
    QueryService::new_with_device_profiles_and_deployment_for_test(
        config,
        queue_bounds,
        device_profiles,
        non_gds_deployment(),
    )
    .expect("explicit multi-device runtime test profiles must install")
}

#[test]
fn query_service_places_concurrent_runtime_handles_on_distinct_devices_when_available() {
    if !has_required_device_inventory(2) {
        return;
    }
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
fn alternating_multi_device_stream_cleanup_returns_each_pool_before_reuse() {
    if !has_required_device_inventory(2) {
        return;
    }
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
        released.finish();
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
        handle.finish();
        drop(handle);
    }
    service.wait_for_stream_pools_idle(Duration::from_secs(5)).unwrap();
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
}

#[test]
fn per_device_active_limit_forces_spread_and_then_blocks() {
    if !has_required_device_inventory(2) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0, 1], 1, 9, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    assert!(service.try_admit_query().unwrap().is_none());
}

#[test]
fn least_loaded_placement_spreads_across_identical_devices() {
    if !has_required_device_inventory(2) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0, 1], 2, 6, TEST_DEVICE_CAPACITY_BYTES / 2),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");

    // The normative spare-capacity ranking spreads across identical devices.
    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
}

#[test]
fn query_service_memory_failure_callback_ignores_budget_failures() {
    if !has_required_device_inventory(1) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0], 1, 3, 64 * 1024),
    );

    let initial_report = service.device_memory_failure_reports().into_iter().next().unwrap();
    if initial_report.callback_install_unavailable {
        eprintln!("SKIP unsupported_memory_failure_callback: CUDA allocation failure callback is unavailable");
        return;
    }
    assert!(initial_report.callback_installed);

    const GRANT_BYTES: u64 = 64 * 1024;

    // An explicit 64 KiB working set makes the immutable attempt grant the
    // domain ceiling and the root member's hard limit; the 4 MiB allocation
    // must fail locally rather than becoming a device-health failure.
    let handle = service
        .try_admit_query_with_request(crate::runtime::QueryAdmissionRequest::new().with_attempt_requirement(
            AttemptRequirement {
                configured_floor_bytes: GRANT_BYTES,
                capability_floor_raise_bytes: 0,
                required_phase_floor_bytes: 0,
                source_stage_bytes: None,
                soft_working_set_hint_bytes: Some(GRANT_BYTES),
                estimate_quality: EstimateQuality::Modeled,
                required_capabilities: Vec::new(),
                owner_terms: Vec::new(),
                bound_derivation: None,
                requires_composed_producer_binding: false,
                attested_cugraph_graph_execution_scopes: Vec::new(),
                reservation_shapes: Vec::new(),
                deferred_resident_phases: Vec::new(),
            },
        ))
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
        domain_before.active_members, 2,
        "the root runtime and its one admitted provider stream are the domain members"
    );

    let token = context
        .context()
        .execution_budget_token(
            cudf::ExecutionBudgetRequest::new(GRANT_BYTES as usize).with_boundary("nexus_failure_callback"),
        )
        .unwrap();
    let member_before = token.reservation_usage().unwrap();
    assert_eq!(member_before.allocation_limit_bytes, GRANT_BYTES);
    let error = execution.runtime().unwrap().series_from_value_with_budget(1i32, 1024 * 1024, &token).unwrap_err();
    assert_eq!(error.kind(), cudf::ErrorKind::ResourceExhausted);
    assert_eq!(error.detail_value("budget_enforcement"), Some("query_domain_member"));
    let grant_bytes_text = GRANT_BYTES.to_string();
    assert_eq!(error.detail_value("budget_bytes"), Some(grant_bytes_text.as_str()));

    let member_after = token.reservation_usage().unwrap();
    assert_eq!(member_after.allocation_limit_bytes, GRANT_BYTES, "the member limit is immutable");
    assert!(member_after.failed_allocation_count > member_before.failed_allocation_count);
    assert!(member_after.largest_failed_allocation_bytes > 0);
    let failed_count_text = member_after.failed_allocation_count.to_string();
    assert_eq!(
        error.detail_value("budget_failed_allocation_count"),
        Some(failed_count_text.as_str()),
        "the structured failure reports the same member counter"
    );

    let domain_after = execution.attempt_allocation_domain_usage().unwrap().unwrap();
    assert_eq!(domain_after.domain_id, grant_id);
    assert_eq!(domain_after.ceiling_bytes, GRANT_BYTES);
    assert_eq!(domain_after.allocation_limit_bytes, GRANT_BYTES);
    assert_eq!(domain_after.active_members, 2);
    assert_eq!(domain_after.current_bytes, member_after.current_bytes);
    assert_eq!(domain_after.peak_bytes, member_after.peak_bytes);
    assert_eq!(domain_after.current_allocations, member_after.current_allocations);
    assert_eq!(domain_after.peak_allocations, member_after.peak_allocations);
    // The root member rejects before an over-limit allocation reaches the
    // parent domain. The idle provider member contributes no live use, so the
    // domain counters still exactly mirror the root member.
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
    if !has_required_device_inventory(1) {
        return;
    }
    let _guard = service_claim_test_guard();
    let config = QueryExecutionConfig::default();
    let queue_bounds = QueueBounds { max_waiters: 2, max_wait: Duration::from_millis(500) };
    let profiles = test_device_profiles(&[0], 1, 3, TEST_DEVICE_CAPACITY_BYTES);
    let first_service = runtime_test_service(config.clone(), queue_bounds.clone(), profiles.clone());
    let second_service = QueryService::new_with_device_profiles_and_deployment_for_test(
        config,
        queue_bounds,
        profiles,
        non_gds_deployment(),
    )
    .expect("test-only services borrow construction authority without becoming backend owners");

    drop(first_service);
    drop(second_service);
}

#[test]
fn per_device_active_limit_is_enforced_within_one_backend() {
    if !has_required_device_inventory(1) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0], 1, 3, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");

    assert!(service.try_admit_query().unwrap().is_none());

    drop(first);
    let _second = service.admit_query().expect("ordered cleanup should wake per-device admission");
}

#[test]
fn per_device_active_limit_release_wakes_waiter_in_same_backend() {
    if !has_required_device_inventory(1) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_secs(2) },
        test_device_profiles(&[0], 1, 3, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let waiting_service = service.clone();
    let (sender, receiver) = mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let result =
            waiting_service.admit_query().map(|handle| handle.device_ordinal()).map_err(|error| error.to_string());
        sender.send(result).expect("main thread should receive waiter result");
    });

    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while service.admission_snapshot().unwrap().queued_queries == 0 {
        assert!(std::time::Instant::now() < deadline, "second query did not queue before release");
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(first);

    let admitted_device = receiver.recv_timeout(Duration::from_secs(3)).expect("waiter should wake before timeout");
    waiter.join().expect("waiter thread should finish");
    assert_eq!(admitted_device.unwrap(), 0);
}

#[test]
fn per_device_active_limit_waits_for_escaped_context_after_handle_finish() {
    if !has_required_device_inventory(1) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0], 1, 6, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let admitted_context = first.acquire_context().unwrap();
    let retained_context = admitted_context.context().clone();
    drop(admitted_context);
    first.finish();

    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    assert!(
        service.try_admit_query().unwrap().is_none(),
        "the escaped context must retain the first query's device slot and grant"
    );
    drop(retained_context);
    let second = service.admit_query().expect("the final context anchor should wake admission");
    assert_eq!(second.device_ordinal(), 0);
}

#[test]
fn escaped_query_anchor_blocks_only_its_selected_device() {
    if !has_required_device_inventory(2) {
        return;
    }
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
    first.finish();

    let second = service.try_admit_query().unwrap().expect("other device remains available");
    assert_eq!(second.device_ordinal(), 1);
    assert!(service.try_admit_query().unwrap().is_none());

    drop(escaped_context);
    let third = service.admit_query().expect("the escaped device-zero anchor eventually drains");
    assert_eq!(third.device_ordinal(), 0);
}

#[test]
fn multi_device_admissions_preserve_each_profiles_fixed_default_query_limit() {
    if !has_required_device_inventory(2) {
        return;
    }
    let _guard = service_claim_test_guard();
    let want_budget = TEST_DEVICE_CAPACITY_BYTES;
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0, 1], 2, 6, TEST_DEVICE_CAPACITY_BYTES / 2),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    assert_eq!(first.device_budget_bytes(), min_budget);
    assert_eq!(second.device_budget_bytes(), min_budget);
}

#[test]
fn multi_device_admissions_do_not_resize_the_fixed_default_query_limit_for_open_slots() {
    if !has_required_device_inventory(2) {
        return;
    }
    let _guard = service_claim_test_guard();
    let want_budget = TEST_DEVICE_CAPACITY_BYTES;
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0, 1], 2, 9, TEST_DEVICE_CAPACITY_BYTES / 2),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should admit");
    let third = service.try_admit_query().unwrap().expect("third query should admit");

    assert_eq!(first.device_budget_bytes(), min_budget);
    assert_eq!(second.device_budget_bytes(), min_budget);
    assert_eq!(third.device_budget_bytes(), min_budget);
}

#[test]
fn query_service_treats_transient_device_capacity_exhaustion_as_busy_not_rejected() {
    if !has_required_device_inventory(1) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        test_device_profiles(&[0], 1, 6, TEST_DEVICE_CAPACITY_BYTES),
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");

    assert!(service.try_admit_query().unwrap().is_none());
    drop(first);
    let _second = service.admit_query().expect("ordered cleanup should wake capacity admission");
}

#[test]
fn queued_query_service_admission_times_out_while_device_capacity_is_busy() {
    if !has_required_device_inventory(1) {
        return;
    }
    let _guard = service_claim_test_guard();
    let service = runtime_test_service(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(1) },
        test_device_profiles(&[0], 1, 6, TEST_DEVICE_CAPACITY_BYTES),
    );
    let _active = service.try_admit_query().unwrap().expect("first query should admit");
    let mut waiter = service.enqueue_admission().unwrap();

    std::thread::sleep(Duration::from_millis(5));
    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), nexus_query_planning::error::QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
}
