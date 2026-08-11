// Report-before-reuse commit: deterministic barriers for the normative
// order — cleanup complete → disposition/health commit → seal/report_ready →
// capacity release — plus health disposition coverage.

use super::service::disposition::{
    CapacityCommitDisposition, CleanupEvidence, decide_capacity_commit, decide_capacity_disposition,
};
use crate::{
    attempt::{AttemptBackendCounters, AttemptOwner, BeginAttemptRequest},
    observability::{AttemptAdmissionFact, AttemptDisposition},
    runtime::BackendLifecycle,
};

fn wait_until(deadline: Instant, mut condition: impl FnMut() -> bool, what: &str) {
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Cleanup-complete without report-ready cannot admit a replacement, and the
/// published report precedes any visible capacity reuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleanup_complete_without_report_ready_cannot_admit_replacement() {
    let service = QueryService::new_for_admission_tests(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(10), max_overtakes_per_waiter: 4 },
        test_ledger(64 * 1024 * 1024, 1, Default::default()),
    )
    .expect("admission-only service for release commit ordering");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("first admission");
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: first.admission_sequence(),
            admission_wait_ns: first.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: first.device_ordinal(),
            grant_bytes: first.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminalizes admitted attempt");
    // Hold normal observer delivery after the terminal commit. A replacement
    // must still see capacity only after the attempt completion is published.
    let delivery_gate = attempt_owner.observation_owner_for_test().arm_deferred_delivery_gate();

    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        assert!(Instant::now() < deadline, "second admission queues behind the first");
        tokio::task::yield_now().await;
    }

    let gate = first.lifetime.arm_release_commit_gate();
    let release = std::thread::spawn(move || {
        first.finish();
        drop(first);
    });
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || gate.entered(),
        "release commit reaches the pre-seal barrier",
    );

    // Cleanup completed and the commit is armed, but the report is not
    // published and no capacity is reusable yet.
    assert!(!completion.is_ready(), "attempt completion must remain unpublished before the commit gate opens");
    assert_eq!(counters.snapshot().reports_ready, 0, "report-ready counter must remain zero before the seal");
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1, "capacity is still held at the commit barrier");
    assert_eq!(snapshot.queued_queries, 1, "replacement cannot observe reusable capacity");

    gate.open();
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || delivery_gate.entered(),
        "ordinary observer delivery begins after capacity release",
    );
    let admitted = tokio::time::timeout(Duration::from_secs(10), queued)
        .await
        .expect("replacement admits after the commit")
        .expect("queued admission task")
        .expect("queued admission succeeds");
    // Capture state while deferred observer delivery is blocked. This is the
    // actual reuse boundary: the replacement has admitted, but tail delivery
    // has not run. The old generic observer path leaves completion unready
    // here; the synchronous terminal commit cannot.
    let completion_ready_before_delivery = completion.is_ready();
    let report_ready_before_delivery =
        completion.try_report().is_some_and(|report| report.timestamps.report_ready_ns.is_some());
    let reports_ready_before_delivery = counters.snapshot().reports_ready;
    delivery_gate.open();
    release.join().expect("release thread completes");
    assert!(completion_ready_before_delivery, "attempt completion is ready before capacity reuse");
    assert!(report_ready_before_delivery, "published report records its report-ready timestamp before capacity reuse");
    assert_eq!(reports_ready_before_delivery, 1, "exactly one attempt report is ready before capacity reuse");
    assert!(
        completion.try_report().expect("published attempt report").timestamps.report_ready_ns.is_some(),
        "published report records its report-ready timestamp"
    );
    let report = completion.try_report().expect("published attempt report");
    assert_eq!(report.cleanup_disposition, crate::observability::AttemptCleanupDisposition::Completed);
    assert_eq!(report.capacity_disposition, crate::observability::AttemptCapacityDisposition::Released);
    assert_eq!(counters.snapshot().reports_ready, 1, "exactly one attempt report is ready");
    admitted.finish();
    drop(admitted);
}

/// An admitted attempt can reach query-handle release without an EOF or an
/// outer wrapper terminalizing it first. The release capability must establish
/// the fallback outcome before its cleanup/capacity commit, otherwise the
/// report would lack finalization facts.
#[test]
fn unterminalized_admitted_release_commits_fallback_before_report() {
    let service = QueryService::new_for_admission_tests(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(10), max_overtakes_per_waiter: 4 },
        test_ledger(64 * 1024 * 1024, 1, Default::default()),
    )
    .expect("admission-only service for fallback release ordering");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("admission succeeds");
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("attempt admission fact");

    // Deliberately omit `AttemptOwner::commit_terminal`: this is the
    // no-EOF/drop fallback that previously tried to commit cleanup first.
    handle.finish();
    drop(handle);

    let report = completion.wait_timeout(Duration::from_secs(1)).expect("fallback report is published");
    assert_eq!(report.outcome, RuntimeTerminalOutcome::Dropped);
    assert_eq!(report.cleanup_disposition, crate::observability::AttemptCleanupDisposition::Completed);
    assert_eq!(report.capacity_disposition, crate::observability::AttemptCapacityDisposition::Released);
    let cleanup_complete = report.timestamps.cleanup_complete_ns.expect("cleanup complete timestamp");
    let report_ready = report.timestamps.report_ready_ns.expect("report-ready timestamp");
    assert!(report.timestamps.outcome_terminal_ns <= cleanup_complete, "terminal outcome precedes cleanup");
    assert!(cleanup_complete <= report_ready, "cleanup precedes report-ready");
}

/// The release fallback becomes terminal before its physical cleanup job can
/// run. A GPU-worker barrier makes the pre-cleanup window observable: report
/// publication remains deferred, but a later execution terminalization cannot
/// replace the already-resolved fallback outcome.
#[tokio::test(flavor = "current_thread")]
async fn unterminalized_release_resolves_fallback_before_gpu_cleanup_starts() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for fallback ordering");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("admission succeeds");
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("attempt admission fact");

    let pool = service
        .inner
        .resources
        .as_ref()
        .expect("runtime resources")
        .gpu_execution_pool_for_device(handle.device_ordinal())
        .expect("device GPU execution pool");
    let release_blocker = Arc::new(tokio::sync::Semaphore::new(0));
    let blocker_release = Arc::clone(&release_blocker);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::Cleanup, async move {
        let _ = started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("cleanup blocker remains open");
        let _ = finished_tx.send(());
    })
    .expect("cleanup blocker queues on the service pool");
    started_rx.await.expect("cleanup blocker starts");

    handle.finish();
    drop(handle);

    assert!(attempt_handle.is_terminal(), "release establishes fallback before its GPU cleanup task runs");
    assert!(
        attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).is_err(),
        "a later execution outcome cannot race the established release fallback"
    );
    assert!(!completion.is_ready(), "report publication remains behind the physical cleanup fence");

    release_blocker.add_permits(1);
    finished_rx.await.expect("cleanup blocker finishes");
    let report = completion.wait_timeout(Duration::from_secs(10)).expect("cleanup publishes the fallback report");
    assert_eq!(report.outcome, RuntimeTerminalOutcome::Dropped);
    assert_eq!(report.cleanup_disposition, crate::observability::AttemptCleanupDisposition::Completed);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

/// Abandoning the handle (no explicit finish) drives the same commit order:
/// seal before capacity reuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_handle_keeps_report_before_reuse_order() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for abandoned release ordering");
    let owner = RuntimeObservationOwner::new("abandoned-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()))
        .expect("first admission");
    drop(first.acquire_context().expect("first context"));

    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        assert!(Instant::now() < deadline, "second admission queues");
        tokio::task::yield_now().await;
    }

    let gate = first.lifetime.arm_release_commit_gate();
    let release = std::thread::spawn(move || drop(first));
    wait_until(Instant::now() + Duration::from_secs(10), || gate.entered(), "abandoned release reaches the barrier");
    assert!(owner.sealed_snapshot().is_none());
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    gate.open();
    release.join().expect("release thread completes");
    let admitted = tokio::time::timeout(Duration::from_secs(10), queued)
        .await
        .expect("replacement admits")
        .expect("task")
        .expect("admission succeeds");
    assert!(owner.sealed_snapshot().is_some(), "receiver abandonment does not change the commit order");
    admitted.finish();
    drop(admitted);
}

/// Release-commit barriers belong to their admitted handle, so concurrently
/// armed tests or releases cannot overwrite or consume one another's hook.
#[test]
fn release_commit_gates_are_scoped_to_their_handles() {
    let service = try_guaranteed_runtime_test_service_with_concurrency(2)
        .expect("runtime-backed service for handle-scoped release commit gates");
    let first = service.admit_query().expect("first admission");
    let second = service.admit_query().expect("second admission");
    let first_gate = first.lifetime.arm_release_commit_gate();
    let second_gate = second.lifetime.arm_release_commit_gate();

    let first_release = std::thread::spawn(move || drop(first));
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || first_gate.entered(),
        "first handle reaches its release commit gate",
    );
    assert!(!second_gate.entered(), "releasing the first handle must not consume the second handle's gate");
    first_gate.open();
    first_release.join().expect("first release thread completes");

    let second_release = std::thread::spawn(move || drop(second));
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || second_gate.entered(),
        "second handle reaches its release commit gate",
    );
    second_gate.open();
    second_release.join().expect("second release thread completes");
}

/// Residual attempt-domain state closes device admission before report
/// publication, and the leaking attempt retains its grant until the report is
/// allowed to seal.
#[test]
fn residual_terminal_domain_usage_quarantines_before_report_and_capacity_release() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for terminal usage quarantine");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let owner = attempt_owner.observation_owner_for_test();
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("admission");
    drop(first.acquire_context().expect("context"));
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: first.admission_sequence(),
            admission_wait_ns: first.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: first.device_ordinal(),
            grant_bytes: first.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("primary outcome commits first");

    let domain = first.lifetime.envelope_domain_for_test().expect("device-backed domain");
    let mut terminal_usage = domain.usage().expect("domain usage sample for injected terminal state");
    terminal_usage.active_members = 1;
    terminal_usage.current_allocations = 2;
    terminal_usage.current_bytes = 3;
    terminal_usage.current_aligned_bytes = 4;
    first.lifetime.inject_terminal_domain_usage_for_test(terminal_usage);
    let gate = first.lifetime.arm_release_commit_gate();
    first.finish();
    let release = std::thread::spawn(move || drop(first));
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || gate.entered(),
        "residual-usage release reaches the pre-seal barrier",
    );

    let report_before_gate_opens = owner.sealed_snapshot();
    let ledger_before_gate_opens = device.ledger().snapshot();
    let service_before_gate_opens = service.admission_snapshot();
    let replacement_before_gate_opens = service.try_admit_query();
    gate.open();
    release.join().expect("release thread completes");

    assert!(report_before_gate_opens.is_none(), "report must remain unsealed at the release commit gate");
    let ledger_before_gate_opens = ledger_before_gate_opens.expect("device ledger snapshot at release commit gate");
    let reason = match ledger_before_gate_opens.health {
        crate::runtime::gpu::DeviceAdmissionHealth::Closed { reason, .. } => reason,
        health => panic!("residual terminal usage must close device admission before the gate: {health:?}"),
    };
    assert!(reason.contains("active_members=1"), "quarantine reason preserves active-member evidence: {reason}");
    assert!(reason.contains("current_allocations=2"), "quarantine reason preserves allocation evidence: {reason}");
    assert!(reason.contains("current_bytes=3"), "quarantine reason preserves byte evidence: {reason}");
    assert!(reason.contains("current_aligned_bytes=4"), "quarantine reason preserves aligned-byte evidence: {reason}");
    assert_eq!(ledger_before_gate_opens.active_attempts, 1, "the leaking attempt's grant is still active");
    assert!(ledger_before_gate_opens.live_grants_bytes > 0, "the leaking attempt's capacity remains held");
    assert_eq!(
        service_before_gate_opens.expect("service snapshot at release commit gate").active_queries,
        1,
        "the leaking handle remains active until report publication"
    );
    assert!(
        !matches!(replacement_before_gate_opens, Ok(Some(_))),
        "closed admission cannot place a replacement: {replacement_before_gate_opens:?}"
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    wait_until(deadline, || completion.is_ready(), "residual-usage attempt report publishes after the gate opens");
    let report = completion.try_report().expect("published attempt report");
    assert_eq!(
        report.cleanup_disposition,
        crate::observability::AttemptCleanupDisposition::Completed,
        "the cleanup callback completed; residual usage, not cleanup failure, chose quarantine"
    );
    assert_eq!(
        report.capacity_disposition,
        crate::observability::AttemptCapacityDisposition::RetainedQuarantine,
        "the report projects the same final usage-derived health decision that closed the ledger"
    );
    wait_until(
        deadline,
        || device.ledger().snapshot().map(|snapshot| snapshot.active_attempts == 0).unwrap_or(false),
        "quarantined attempt releases its held capacity after report publication",
    );
}

/// The finalizer is a one-time transfer from its owner into the admitted
/// envelope. A second admission binding can carry observation identity but
/// cannot manufacture competing cleanup/capacity facts.
#[test]
fn admitted_finalizer_transfers_once_and_publishes_no_pending_capacity() {
    let service = QueryService::new_for_admission_tests(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(10), max_overtakes_per_waiter: 4 },
        test_ledger(64 * 1024 * 1024, 1, Default::default()),
    )
    .expect("admission-only service");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        counters,
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("admission");
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    assert!(matches!(
        attempt_owner.admission_binding().into_parts().1.take_for_admission(),
        Err(crate::attempt::AttemptLifecycleError::FinalizationUnavailable)
    ));

    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("primary outcome commits first");
    handle.finish();
    drop(handle);
    let report = completion.wait_timeout(Duration::from_secs(1)).expect("finalized report publishes");
    assert_eq!(report.capacity_disposition, crate::observability::AttemptCapacityDisposition::Released);
    assert!(report.timestamps.report_ready_ns.is_some());
}

/// A stream synchronization failure during cleanup is device-local fault
/// evidence: the device closes before report publication and the attempt's
/// capacity is never reusable, while the report still publishes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_cleanup_quarantines_device_and_never_returns_capacity() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for cleanup quarantine");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let owner =
        RuntimeObservationOwner::new("failed-cleanup-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()))
        .expect("admission");
    drop(first.acquire_context().expect("context"));

    device.stream_pool().fail_next_synchronization_for_test();
    first.finish();
    drop(first);

    let deadline = Instant::now() + Duration::from_secs(10);
    wait_until(
        deadline,
        || {
            matches!(
                device.ledger().snapshot().map(|snapshot| snapshot.health),
                Ok(crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. })
            )
        },
        "device ledger closes under quarantine",
    );
    assert!(owner.sealed_snapshot().is_some(), "report still publishes after failed cleanup");

    // The attempt's capacity is never reusable: a new admission cannot place
    // on the closed device.
    let error = service.admit_query().expect_err("closed device rejects new admission");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(
        error.fact_value("reason"),
        Some("no_healthy_compatible_device"),
        "closed device rejects immediately with its health disposition: {error}"
    );
    assert_eq!(error.fact_value("device_admission_health"), Some("closed"));
}

/// A failed attempt with device-local cleanup evidence can recommend another
/// device only after the failed ledger is quarantined and the immutable
/// compatible set proves an alternate remains healthy and can fit its floor.
#[test]
fn device_local_cleanup_fault_seals_retry_on_a_frozen_eligible_alternate() {
    let visible_devices = cudf::DeviceId::count().expect("enumerate CUDA devices for alternate retry coverage");
    assert!(
        visible_devices >= 2,
        "alternate retry coverage requires at least two visible CUDA devices; found {visible_devices}"
    );

    const CAPACITY_BYTES: u64 = 64 * 1024 * 1024;
    const ATTEMPT_FLOOR_BYTES: u64 = CAPACITY_BYTES / 8;
    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 4 },
        vec![
            runtime_test_device_profile_with_limits(0, CAPACITY_BYTES, ATTEMPT_FLOOR_BYTES, 1, 3, 1, 1),
            runtime_test_device_profile_with_limits(1, CAPACITY_BYTES, ATTEMPT_FLOOR_BYTES, 1, 3, 1, 1),
        ],
    )
    .expect("two-device runtime-backed service for alternate retry advice");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device_zero = resources.devices.context_by_ordinal(0).expect("device zero context");
    let device_one = resources.devices.context_by_ordinal(1).expect("device one context");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        counters,
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("first admission");
    assert_eq!(handle.device_ordinal(), 0, "equal-profile placement uses ordinal zero deterministically");
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    drop(handle.acquire_context().expect("root context"));
    device_zero.stream_pool().fail_next_synchronization_for_test();
    let execution_error = QueryError::new(
        QueryErrorCode::CudfTemporaryFailure,
        "nexus_query_engine.query_service.alternate_retry_test",
        "injected execution failure before device-local cleanup fault",
    );
    attempt_owner.commit_query_error(&execution_error).expect("attempt failure records before cleanup");
    handle.finish();
    drop(handle);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("faulted attempt report seals");
    assert_eq!(
        report.retry_advice,
        crate::observability::AttemptRetryAdvice {
            recovery: crate::observability::AttemptRetryRecovery::RetryAnotherEligibleDevice,
            prerequisite: crate::observability::AttemptRetryPrerequisite::QuarantinedDeviceAndEligibleAlternate,
        },
        "retry advice requires the post-quarantine alternate proof"
    );
    let fault = report.native_fault_evidence.expect("device-local cleanup failure retains native evidence");
    assert_eq!(fault.isolation, crate::observability::AttemptFaultIsolation::DeviceLocal);
    assert_eq!(fault.teardown_health, crate::observability::AttemptTeardownHealth::StreamSynchronizationFailed);
    assert!(
        matches!(device_zero.ledger().health(), Ok(crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. })),
        "the failed device is quarantined before retry advice is published"
    );
    assert!(
        matches!(device_one.ledger().health(), Ok(crate::runtime::gpu::DeviceAdmissionHealth::Open)),
        "the alternate proof retains only a currently healthy ledger"
    );

    let replacement = service.admit_query().expect("healthy alternate admits a new attempt");
    assert_eq!(replacement.device_ordinal(), 1, "the quarantined device cannot be selected again");
    replacement.finish();
    drop(replacement);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("two-device service drains");
}

/// A duplicate stream-cleanup handoff is an internal ownership violation. It
/// must still run the release report path as `CallbackAbandoned` before
/// dropping the immutable grant, so the service becomes fail-closed instead
/// of silently making capacity reusable without a terminal observation.
#[test]
fn duplicate_root_cleanup_handoff_marks_backend_unhealthy_before_capacity_release() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for cleanup collision");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("admission");
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    let context = handle.acquire_context().expect("root context");
    handle.lifetime.install_foreign_root_stream_cleanup_completion_for_test();
    drop(context);

    let primary_error = QueryError::new(
        QueryErrorCode::CudfTemporaryFailure,
        "nexus_query_engine.query_service.unknown_isolation_primary",
        "injected primary error before abandoned cleanup",
    );
    attempt_owner.commit_query_error(&primary_error).expect("primary error records before cleanup");
    handle.finish();
    drop(handle);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("abandoned-handoff report seals");
    assert!(matches!(
        report.outcome,
        RuntimeTerminalOutcome::Failed {
            error_code: crate::observability::AttemptFailureCode::CudfTemporaryFailure,
            error_status: crate::observability::AttemptFailureStatus::Temporary,
        }
    ));
    let primary_failure = report.primary_failure.as_ref().expect("primary failure remains authoritative");
    assert_eq!(primary_failure.code(), crate::observability::AttemptFailureCode::CudfTemporaryFailure);
    assert_eq!(report.cleanup_disposition, crate::observability::AttemptCleanupDisposition::CompletedWithFailures);
    assert_eq!(report.cleanup_failure_count, 1, "the abandoned callback is one cleanup failure");
    assert!(report.cleanup_failure_count_complete);
    let cleanup_failure = report.first_cleanup_failure.as_ref().expect("abandoned cleanup is retained");
    assert_eq!(cleanup_failure.phase(), crate::observability::CleanupFailurePhase::QueryServiceCallbackAbandoned);
    assert_eq!(
        cleanup_failure.failure().code(),
        crate::observability::AttemptFailureCode::CleanupCallbackAbandoned
    );
    assert!(cleanup_failure.complete(), "the sole cleanup failure remains a complete representative");
    assert_eq!(report.capacity_disposition, crate::observability::AttemptCapacityDisposition::RetainedQuarantine);
    assert_eq!(report.health_transition, Some(crate::observability::AttemptHealthTransition::BackendUnhealthy));
    let fault = report.native_fault_evidence.expect("unknown isolation remains explicit");
    assert_eq!(fault.isolation, crate::observability::AttemptFaultIsolation::Unknown);
    assert_eq!(fault.teardown_health, crate::observability::AttemptTeardownHealth::CallbackAbandoned);
    assert!(fault.device_health.is_none(), "unknown isolation cannot fabricate a device-local health sample");
    assert!(fault.first_fault.is_none(), "unknown isolation cannot fabricate a device-local native fault");
    assert_eq!(
        report.retry_advice,
        crate::observability::AttemptRetryAdvice {
            recovery: crate::observability::AttemptRetryRecovery::RecoverBackend,
            prerequisite: crate::observability::AttemptRetryPrerequisite::BackendRecovered,
        },
        "unknown isolation is sealed as backend-recovery advice rather than a device retry"
    );
    let resource_evidence = report.resource_evidence.expect("admitted attempt retains its immutable resource evidence");
    assert_eq!(
        resource_evidence.grant_cap_bytes.provenance,
        crate::observability::AttemptEvidenceProvenance::AttemptExact
    );
    assert!(resource_evidence.grant_cap_bytes.value.is_some());
    assert_eq!(
        resource_evidence.domain_cap_bytes.provenance,
        crate::observability::AttemptEvidenceProvenance::AttemptExact
    );
    assert!(resource_evidence.domain_cap_bytes.value.is_some());
    for (name, provenance) in [
        ("ledger sequence", resource_evidence.ledger_sequence.provenance),
        ("ledger sample time", resource_evidence.ledger_sample_ns.provenance),
        ("device profile", resource_evidence.configured_limit_bytes.provenance),
        ("allocator event", resource_evidence.allocator_event.provenance),
        ("unattributed allocator event", resource_evidence.unattributed_device_allocator_event.provenance),
    ] {
        assert_eq!(
            provenance,
            crate::observability::AttemptEvidenceProvenance::Unavailable,
            "unknown isolation must not project a {name}"
        );
    }
    assert!(
        matches!(
            service
                .inner
                .resources
                .as_ref()
                .expect("runtime-backed service")
                .devices
                .context_by_ordinal(0)
                .expect("device zero")
                .ledger()
                .health(),
            Ok(crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. })
        ),
        "duplicate cleanup handoff closes the scheduler ledger"
    );
    let error =
        service.admit_query().expect_err("backend rejects new admission after unknown-isolation cleanup collision");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("backend_unhealthy"));
}

/// A finite-overtake protection is queue state only. If its sole candidate
/// closes, the waiter must preserve the persistent device-health disposition
/// rather than degrading it into a retryable queue-overload error.
#[test]
fn protected_waiter_reports_device_quarantine_not_transient_overload() {
    const MIB: u64 = 1024 * 1024;

    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 1 },
        vec![runtime_test_device_profile_with_limits(0, 100 * MIB, 10 * MIB, 2, 6, 2, 2)],
    )
    .expect("runtime-backed service for protected-waiter quarantine");

    let active = service.admit_query().expect("initial admission");
    let mut protected = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(60 * MIB, Some(60 * MIB))),
        )
        .expect("large waiter queues behind the initial grant");
    assert!(protected.try_admit().expect("large waiter remains queued").is_none());

    let mut bypass = service.enqueue_admission().expect("small bypass queues");
    let bypass = bypass.try_admit().expect("small bypass placement").expect("small bypass admits");
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, vec![0]);

    service
        .close_device_admission_for_test(0, "test-protected-waiter-quarantine")
        .expect("close the only compatible device");
    let error = protected.try_admit().expect_err("protected waiter terminates after its device closes");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("no_healthy_compatible_device"));
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);

    drop(bypass);
    drop(active);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).expect("closed service drains cleanup");
}

/// An OOM-class outcome alone never quarantines: the device stays open and
/// healthy capacity is reusable after report-ready.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oom_outcome_keeps_device_open_and_capacity_reusable_after_report() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for OOM disposition");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let owner = RuntimeObservationOwner::new("oom-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()))
        .expect("admission");
    drop(first.acquire_context().expect("context"));
    owner.record_terminal_outcome(RuntimeTerminalOutcome::Failed {
        error_code: AttemptFailureCode::CudfResourceExhausted,
        error_status: AttemptFailureStatus::Temporary,
    });
    first.finish();
    drop(first);

    let deadline = Instant::now() + Duration::from_secs(10);
    wait_until(deadline, || owner.sealed_snapshot().is_some(), "OOM report seals");
    wait_until(
        deadline,
        || device.ledger().snapshot().map(|snapshot| snapshot.active_attempts == 0).unwrap_or(false),
        "capacity returns after report-ready",
    );
    assert!(
        matches!(
            device.ledger().snapshot().map(|snapshot| snapshot.health),
            Ok(crate::runtime::gpu::DeviceAdmissionHealth::Open)
        ),
        "OOM alone never quarantines the device"
    );

    let second = service.admit_query().expect("capacity is reusable after the OOM report");
    second.finish();
    drop(second);
}

/// A reservation-limit callback produces an exactly joined allocator record in
/// the terminal resource block. The release commit keeps that record distinct
/// from immutable attempt facts, the atomic ledger snapshot, and separately
/// timed usage samples.
#[test]
fn oom_release_projects_truthful_resource_evidence() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for OOM resource evidence");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("admission");
    let grant_bytes = handle.device_budget_bytes();
    attempt_handle
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes,
        })
        .expect("attempt admission fact");
    let requested_bytes = grant_bytes.checked_add(1).expect("grant leaves one byte above its immutable cap");
    let requested_bytes = usize::try_from(requested_bytes).expect("test device allocation size fits usize");
    let context = handle.acquire_context().expect("context");
    let allocation_error = match context.context().runtime().expect("query runtime").device_buffer(requested_bytes) {
        Ok(_) => panic!("allocation above the immutable grant limit is rejected"),
        Err(error) => error,
    };
    assert_eq!(allocation_error.kind(), cudf::ErrorKind::ResourceExhausted);
    drop(context);
    let terminal_error = crate::error::QueryError::from_cudf(
        "nexus_query_engine.query_service.release_commit_test",
        "attempt allocation exceeded its immutable grant limit",
        allocation_error,
    );
    assert_eq!(terminal_error.code(), QueryErrorCode::CudfResourceExhausted);
    attempt_owner.commit_query_error(&terminal_error).expect("terminal OOM outcome");
    handle.finish();
    drop(handle);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("OOM report publishes");
    assert_eq!(
        report.retry_advice,
        crate::observability::AttemptRetryAdvice {
            recovery: crate::observability::AttemptRetryRecovery::ChangeConfiguration,
            prerequisite: crate::observability::AttemptRetryPrerequisite::ConfigurationChanged,
        },
        "an exactly joined reservation-limit callback requires a configuration change"
    );
    let evidence = report.resource_evidence.expect("device-backed OOM has resource evidence");
    assert_eq!(evidence.ledger_sequence.provenance, crate::observability::AttemptEvidenceProvenance::DeviceSnapshot);
    assert!(evidence.ledger_sequence.value.is_some(), "one atomic ledger sequence is retained");
    assert_eq!(
        evidence.configured_limit_bytes.provenance,
        crate::observability::AttemptEvidenceProvenance::AttemptExact
    );
    assert_eq!(evidence.grant_cap_bytes.value, Some(grant_bytes));
    assert_eq!(evidence.grant_cap_bytes.provenance, crate::observability::AttemptEvidenceProvenance::AttemptExact);
    assert_eq!(evidence.domain_cap_bytes.provenance, crate::observability::AttemptEvidenceProvenance::AttemptExact);
    let terminal_usage = evidence.outcome_terminal_usage.value.expect("usage sampled before cleanup");
    let post_cleanup_usage = evidence.post_cleanup_usage.value.expect("usage sampled after cleanup");
    assert!(
        terminal_usage.sampled_ns <= post_cleanup_usage.sampled_ns,
        "usage samples retain their independent release ordering"
    );
    for pool in [terminal_usage.pool_usage, post_cleanup_usage.pool_usage] {
        let pool = pool.expect("query-domain pool usage is projected at both release samples");
        assert!(pool.used_current_bytes <= pool.reserved_current_bytes);
        assert!(pool.used_high_bytes <= pool.reserved_high_bytes);
    }
    assert_eq!(
        evidence.outcome_terminal_usage.provenance,
        crate::observability::AttemptEvidenceProvenance::SampledAfterFailure
    );
    assert_eq!(
        evidence.post_cleanup_usage.provenance,
        crate::observability::AttemptEvidenceProvenance::SampledAfterFailure
    );
    assert_eq!(
        evidence.allocator_event.provenance,
        crate::observability::AttemptEvidenceProvenance::AttemptExact,
        "the reservation-limit callback is joined through the completed domain-generation interval"
    );
    let allocator_event = evidence.allocator_event.value.expect("exact callback record is retained");
    assert_eq!(allocator_event.requested_bytes, u64::try_from(requested_bytes).expect("usize fits u64"));
    assert_eq!(allocator_event.source, crate::observability::AttemptAllocatorEventSource::ReservationLimit);
    assert_eq!(
        allocator_event.action,
        crate::observability::AttemptEvidenceField::attempt_exact(
            crate::observability::AttemptAllocatorEventAction::Decline
        )
    );
    assert_eq!(
        evidence.unattributed_device_allocator_event.provenance,
        crate::observability::AttemptEvidenceProvenance::Unavailable,
        "the exact attempt record takes precedence over unrelated device samples"
    );
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

/// The backend-unhealthy latch rejects new submissions with a structured
/// error while the in-flight attempt finishes untouched.
#[test]
fn backend_unhealthy_rejects_new_submissions_while_inflight_finishes() {
    let service = test_service(2, 8);
    let active = service.admit_query().expect("in-flight admission");
    service.service.force_backend_unhealthy_for_test("test-injected unknown-isolation fault");
    assert!(matches!(
        service.service.inner.fallback_device_ledger.as_ref().expect("admission-only ledger").health(),
        Ok(crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. })
    ));

    let error = service.admit_query().expect_err("new submission rejects while unhealthy");
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("backend_unhealthy"));
    assert_eq!(
        error.fact_value("detail"),
        Some("test-injected unknown-isolation fault".to_string())
            .as_deref()
            .or(Some("test-injected unknown-isolation fault"))
    );

    let queued_error = service.try_admit_query().expect_err("try admission also rejects while unhealthy");
    assert_eq!(queued_error.fact_value("reason"), Some("backend_unhealthy"));

    // In-flight work is untouched and its release commits normally.
    active.finish();
    drop(active);
    let deadline = Instant::now() + Duration::from_secs(10);
    wait_until(
        deadline,
        || service.admission_snapshot().map(|snapshot| snapshot.active_queries == 0).unwrap_or(false),
        "in-flight attempt releases",
    );
}

/// Disposition mapping: abandoned cleanup callbacks mark the backend
/// unhealthy (unknown isolation); completed cleanup with a healthy device is
/// healthy reuse.
#[test]
fn disposition_mapping_matches_isolation_rules() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for disposition mapping");
    let handle = service.admit_query().expect("admission");
    let domain = handle.lifetime.envelope_domain_for_test().expect("device-backed domain");

    let mut cleared_usage = domain.usage().expect("domain usage sample");
    cleared_usage.active_members = 0;
    cleared_usage.current_allocations = 0;
    cleared_usage.current_bytes = 0;
    cleared_usage.current_aligned_bytes = 0;
    let (disposition, usage) = decide_capacity_disposition(
        0,
        Some(&domain),
        CleanupEvidence::Completed,
        Some(cleared_usage),
    );
    assert_eq!(disposition, CapacityCommitDisposition::HealthyReuse);
    let usage = usage.expect("healthy device samples domain usage");
    assert_eq!(usage.domain_id, domain.domain_id());

    type ResidualInjector = (&'static str, fn(&mut cudf::QueryMemoryDomainUsage));
    let residuals: [ResidualInjector; 4] = [
        ("active_members", |usage| usage.active_members = 1),
        ("current_allocations", |usage| usage.current_allocations = 1),
        ("current_bytes", |usage| usage.current_bytes = 1),
        ("current_aligned_bytes", |usage| usage.current_aligned_bytes = 1),
    ];
    for (field, inject_residual) in residuals {
        let mut residual_usage = cleared_usage;
        inject_residual(&mut residual_usage);
        let (disposition, observed_usage) = decide_capacity_disposition(
            0,
            Some(&domain),
            CleanupEvidence::Completed,
            Some(residual_usage),
        );
        let reason = match disposition {
            CapacityCommitDisposition::DeviceQuarantine { reason } => reason,
            disposition => panic!("nonzero {field} must quarantine the device, got {disposition:?}"),
        };
        assert!(reason.contains(&format!("{field}=1")), "quarantine reason preserves {field}: {reason}");
        assert_eq!(observed_usage, Some(residual_usage), "quarantine returns the sampled terminal usage");
    }

    let abandonment = CleanupEvidence::from(crate::runtime::service_stream::StreamCleanupOutcome::CallbackAbandoned);
    assert_eq!(abandonment, CleanupEvidence::CallbackAbandoned);
    let abandonment_decision = decide_capacity_commit(0, Some(&domain), abandonment, None);
    let disposition = abandonment_decision.disposition;
    assert!(
        matches!(disposition, CapacityCommitDisposition::BackendUnhealthy { .. }),
        "abandoned cleanup has unknown isolation: {disposition:?}"
    );
    let abandonment_evidence =
        abandonment_decision.native_fault_evidence.expect("unknown isolation has terminal evidence");
    assert_eq!(abandonment_evidence.isolation, crate::observability::AttemptFaultIsolation::Unknown);
    assert_eq!(abandonment_evidence.teardown_health, crate::observability::AttemptTeardownHealth::CallbackAbandoned);
    assert!(abandonment_evidence.device_health.is_none(), "unknown isolation must not fabricate a device-local sample");

    let synchronization_decision =
        decide_capacity_commit(0, Some(&domain), CleanupEvidence::StreamSynchronizationFailed, None);
    let disposition = synchronization_decision.disposition;
    assert!(
        matches!(disposition, CapacityCommitDisposition::DeviceQuarantine { .. }),
        "sync failure is device-local: {disposition:?}"
    );
    let synchronization_evidence = synchronization_decision
        .native_fault_evidence
        .expect("device-local synchronization failure has terminal evidence");
    assert_eq!(synchronization_evidence.isolation, crate::observability::AttemptFaultIsolation::DeviceLocal);
    assert_eq!(
        synchronization_evidence.teardown_health,
        crate::observability::AttemptTeardownHealth::StreamSynchronizationFailed
    );

    // cuDF's same-thread test registry makes the sibling-owned health
    // snapshot deterministic without poisoning the process-wide registry
    // used by the asynchronous GPU cleanup worker. The decision consumes the
    // snapshot directly, preserving each native status ABI family.
    let device = cudf::DeviceId::new(0).expect("runtime service established device zero");
    for (native_status, expected_status) in [
        (
            cudf::runtime::DeviceFaultNativeStatus::Cudf(cudf_sys::CUDF_ERROR_FATAL_CUDA),
            crate::observability::AttemptNativeStatus::Cudf(cudf_sys::CUDF_ERROR_FATAL_CUDA),
        ),
        (cudf::runtime::DeviceFaultNativeStatus::Cuda(700), crate::observability::AttemptNativeStatus::Cuda(700)),
    ] {
        let _device_safety = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
        cudf::runtime::report_device_fault(
            device,
            cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
            "test_native_fault",
            Some(native_status),
        );
        let native_fault_decision = decide_capacity_commit(
            0,
            Some(&domain),
            CleanupEvidence::Completed,
            Some(cleared_usage),
        );
        assert!(
            matches!(native_fault_decision.disposition, CapacityCommitDisposition::DeviceQuarantine { .. }),
            "a native device fault quarantines capacity before report publication"
        );
        let evidence =
            native_fault_decision.native_fault_evidence.expect("device-local native fault retains terminal evidence");
        assert_eq!(evidence.isolation, crate::observability::AttemptFaultIsolation::DeviceLocal);
        assert_eq!(evidence.teardown_health, crate::observability::AttemptTeardownHealth::Completed);
        assert_eq!(
            evidence.device_health.expect("device-local evidence retains health snapshot").status,
            crate::observability::AttemptDeviceHealthStatus::Unhealthy
        );
        let fault = evidence.first_fault.expect("fault snapshot retains the first structured native fault");
        assert_eq!(fault.kind, crate::observability::AttemptNativeFaultKind::FatalDeviceOperation);
        assert_eq!(fault.status, Some(expected_status));
    }

    // Ledger-only releases carry no domain and always healthy-reuse without
    // touching CUDA FFI.
    let (disposition, usage) = decide_capacity_disposition(0, None, CleanupEvidence::CallbackAbandoned, None);
    assert_eq!(disposition, CapacityCommitDisposition::HealthyReuse);
    assert!(usage.is_none());

    handle.finish();
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}
