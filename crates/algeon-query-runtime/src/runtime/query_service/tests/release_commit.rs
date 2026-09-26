// Report-before-reuse commit: deterministic barriers for the normative
// order; cleanup complete → disposition/health commit → seal/report_ready →
// capacity release; plus health disposition coverage.

use super::service::disposition::{
    CapacityCommitDisposition, CleanupEvidence, decide_capacity_commit, decide_capacity_disposition,
};
use crate::native_error::cudf_query_error;

fn wait_until(deadline: Instant, mut condition: impl FnMut() -> bool, what: &str) {
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn assert_legal_resident_output(
    case: &str,
    build: impl FnOnce(
        &Arc<CudfRuntime>,
    ) -> (GpuDataFrame, SchemaRef),
) {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for resident output");
    let device = service
        .inner
        .resources
        .as_ref()
        .expect("runtime-backed service")
        .devices
        .context_by_ordinal(0)
        .expect("device context");
    let backend_lifecycle = BackendLifecycle::new();
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::new(AttemptBackendCounters::default()),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let admitted = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission");
    let context = admitted.acquire_context().expect("attempt context");
    let runtime = context.context().shared_runtime().expect("attempt runtime");
    let (frame, schema) = build(&runtime);
    let bytes = frame.device_allocated_size_bytes().expect("resident output bytes");
    let owner = context.context().retain_resident_output_owner(bytes).expect("resident owner");
    let result =
        GpuResultHandle::new_with_resident_owner(context.context(), frame, schema, owner)
            .expect("resident result");
    drop(runtime);
    drop(context);
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminal outcome");

    drop(admitted);
    wait_until(Instant::now() + Duration::from_secs(10), || completion.is_ready(), case);
    let report = completion.try_report().expect("published attempt report");
    assert_eq!(report.cleanup_failure_count, 0, "{case}");
    assert!(!matches!(
        report.capacity_disposition,
        AttemptCapacityDisposition::RetainedQuarantine
    ));
    assert!(matches!(
        device.ledger().snapshot().expect("ledger snapshot").health,
        DeviceAdmissionHealth::Open
    ));
    drop(result);
}

#[test]
fn shared_sliced_nested_and_empty_resident_outputs_pass_owner_attestation() {
    assert_legal_resident_output("shared resident output", |runtime| {
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("kept", DataType::Int32, false),
            Field::new("discarded", DataType::Int32, false),
        ]));
        let source_batch = RecordBatch::try_new(
            source_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![4, 5, 6])),
            ],
        )
        .unwrap();
        let source = DataFrame::from_record_batch(runtime.as_ref(), &source_batch).unwrap();
        let selected = source.select(["kept"]).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "kept",
            DataType::Int32,
            false,
        )]));
        (GpuDataFrame::new(selected), schema)
    });

    assert_legal_resident_output("sliced resident output", |runtime| {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
        )
        .unwrap();
        let source = DataFrame::from_record_batch(runtime.as_ref(), &batch).unwrap();
        let sliced = source.slice(1, 2).unwrap();
        (GpuDataFrame::new(sliced), schema)
    });

    assert_legal_resident_output("nested resident output", |runtime| {
        use arrow::{
            array::{ArrayRef, Int32Array, ListArray},
            buffer::{OffsetBuffer, ScalarBuffer},
        };
        let item = Arc::new(Field::new("item", DataType::Int32, false));
        let list = ListArray::new(
            Arc::clone(&item),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 3])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "items",
            DataType::List(item),
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(list) as ArrayRef],
        )
        .unwrap();
        let frame = DataFrame::from_record_batch(runtime.as_ref(), &batch).unwrap();
        (GpuDataFrame::new(frame), schema)
    });

    assert_legal_resident_output("empty resident output", |runtime| {
        (
            GpuDataFrame::new(runtime.empty_dataframe()),
            Arc::new(Schema::empty()),
        )
    });
}

/// A retained GPU result keeps the backend generation alive: close must
/// surface pending cleanup instead of false quiescence, and completes only
/// after the final owner releases.
#[test]
fn close_with_retained_gpu_result_waits_for_release_instead_of_claiming_quiescence() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for resident output");
    let backend_lifecycle = BackendLifecycle::new();
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::new(AttemptBackendCounters::default()),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let admitted = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission");
    let context = admitted.acquire_context().expect("attempt context");
    let runtime = context.context().shared_runtime().expect("attempt runtime");
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(runtime.as_ref(), &batch).unwrap();
    let frame = GpuDataFrame::new(frame);
    let bytes = frame.device_allocated_size_bytes().expect("resident output bytes");
    let owner = context.context().retain_resident_output_owner(bytes).expect("resident owner");
    let result =
        GpuResultHandle::new_with_resident_owner(context.context(), frame, schema, owner)
            .expect("resident result");
    drop(runtime);
    drop(context);
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminal outcome");

    drop(admitted);
    wait_until(Instant::now() + Duration::from_secs(10), || completion.is_ready(), "retained resident report");
    let report = completion.try_report().expect("published attempt report");
    assert_eq!(report.cleanup_failure_count, 0, "retained resident output");

    let timeout = service.close_and_wait_for_quiescence(Duration::from_secs(1)).unwrap_err();
    assert_eq!(timeout.fact_value("reason"), Some("query_service_resource_epoch_timeout"));

    drop(result);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("close completes after release");
}

/// Cleanup-complete without report-ready cannot admit a replacement, and the
/// published report precedes any visible capacity reuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleanup_complete_without_report_ready_cannot_admit_replacement() {
    let service = QueryService::new_for_admission_tests(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(10) },
        test_ledger(64 * 1024 * 1024, 1, Default::default()),
    )
    .expect("admission-only service for release commit ordering");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("first admission");
    // Hold normal observer delivery after the terminal commit. A replacement
    // must still see capacity only after the attempt completion is published.
    let delivery_gate = attempt_owner.arm_delivery_gate_for_test();

    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        assert!(Instant::now() < deadline, "second admission queues behind the first");
        yield_now().await;
    }

    // The terminal commit releases the admitted grant.
    let gate = first.lifetime.arm_release_commit_gate();
    let release = std::thread::spawn(move || {
        attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminalizes admitted attempt");
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
    let admitted = timeout(Duration::from_secs(10), queued)
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
    assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed);
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
    assert_eq!(counters.snapshot().reports_ready, 1, "exactly one attempt report is ready");
    drop(admitted);
}

/// The attempt owns its grant. Dropping the query handle neither releases
/// the grant nor ends the attempt; an attempt abandoned without a terminal
/// commit ends as `Dropped`, and that commit releases the grant before the
/// report publishes.
#[test]
fn dropping_the_handle_keeps_the_grant_until_the_attempt_ends() {
    let service = QueryService::new_for_admission_tests(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(10) },
        test_ledger(64 * 1024 * 1024, 1, Default::default()),
    )
    .expect("admission-only service for fallback release ordering");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission succeeds");

    drop(handle);
    assert!(!attempt_handle.is_terminal(), "dropping the handle does not end the attempt");
    assert!(!completion.is_ready(), "dropping the handle publishes no report");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1, "the attempt still holds its grant");

    // Deliberately omit `AttemptOwner::commit_terminal`: the abandoned
    // attempt's drop commits `Dropped`, which releases the grant.
    drop(attempt_owner);
    let report = completion.wait_timeout(Duration::from_secs(1)).expect("the dropped attempt's report is published");
    assert_eq!(report.outcome, RuntimeTerminalOutcome::Dropped);
    assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed);
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
    let cleanup_complete = report.timestamps.cleanup_complete_ns.expect("cleanup complete timestamp");
    let report_ready = report.timestamps.report_ready_ns.expect("report-ready timestamp");
    assert!(report.timestamps.outcome_terminal_ns <= cleanup_complete, "terminal outcome precedes cleanup");
    assert!(cleanup_complete <= report_ready, "cleanup precedes report-ready");
}

/// The attempt owns its admission: committing its terminal outcome releases
/// the grant and publishes the report while the handle is still held.
#[test]
fn terminal_commit_releases_the_admitted_grant_while_the_handle_is_held() {
    let service = QueryService::new_for_admission_tests(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(10) },
        test_ledger(64 * 1024 * 1024, 1, Default::default()),
    )
    .expect("admission-only service for terminal release");
    let backend_lifecycle = BackendLifecycle::new();
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::new(AttemptBackendCounters::default()),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission succeeds");

    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminal outcome commits");

    let report = completion.wait_timeout(Duration::from_secs(1)).expect("the terminal commit publishes the report");
    assert_eq!(report.outcome, RuntimeTerminalOutcome::Succeeded);
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0, "the grant is released before the handle drops");
    drop(handle);
}

/// Only the attempt's terminal commit starts the release, and the report
/// publishes after the release's physical cleanup. A GPU-worker barrier makes
/// the pre-cleanup window observable.
#[tokio::test(flavor = "current_thread")]
async fn terminal_commit_releases_and_publishes_after_gpu_cleanup() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for fallback ordering");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission succeeds");

    let pool = service
        .inner
        .resources
        .as_ref()
        .expect("runtime resources")
        .gpu_execution_pool_for_device(handle.device_ordinal())
        .expect("device GPU execution pool");
    let release_blocker = Arc::new(Semaphore::new(0));
    let blocker_release = Arc::clone(&release_blocker);
    let (started_tx, started_rx) = oneshot::channel();
    let (finished_tx, finished_rx) = oneshot::channel();
    pool.spawn(GpuExecutionJobKind::Cleanup, async move {
        let _ = started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("cleanup blocker remains open");
        let _ = finished_tx.send(());
    })
    .expect("cleanup blocker queues on the service pool");
    started_rx.await.expect("cleanup blocker starts");

    drop(handle);

    assert!(!attempt_handle.is_terminal(), "dropping the handle does not end the attempt");
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("the attempt commits its own outcome");
    assert!(!completion.is_ready(), "report publication remains behind the physical cleanup fence");

    release_blocker.add_permits(1);
    finished_rx.await.expect("cleanup blocker finishes");
    let report = completion.wait_timeout(Duration::from_secs(10)).expect("cleanup publishes the attempt report");
    assert_eq!(report.outcome, RuntimeTerminalOutcome::Succeeded);
    assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

/// An attempt abandoned without a terminal commit drives the same commit
/// order: seal before capacity reuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_attempt_keeps_report_before_reuse_order() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for abandoned release ordering");
    let attempt_owner = observed_attempt("abandoned-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let binding = attempt_owner.admission_binding();
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new(binding.clone()))
        .expect("first admission");
    drop(first.acquire_context().expect("first context"));

    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        assert!(Instant::now() < deadline, "second admission queues");
        yield_now().await;
    }

    let gate = first.lifetime.arm_release_commit_gate();
    drop(first);
    let release = std::thread::spawn(move || drop(attempt_owner));
    wait_until(Instant::now() + Duration::from_secs(10), || gate.entered(), "abandoned release reaches the barrier");
    assert!(binding.observation_handle().sealed_snapshot().is_none());
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    gate.open();
    release.join().expect("release thread completes");
    let admitted = timeout(Duration::from_secs(10), queued)
        .await
        .expect("replacement admits")
        .expect("task")
        .expect("admission succeeds");
    assert!(
        binding.observation_handle().sealed_snapshot().is_some(),
        "receiver abandonment does not change the commit order"
    );
    admitted.end_test_attempt();
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

/// Different explicit caps remain independently committed while their GPU
/// jobs overlap. A failed first attempt may enter cleaning without releasing
/// the second attempt's grant, and the first grant remains fenced until its
/// envelope cleanup completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlapping_different_caps_isolate_failure_and_cleaning_release() {
    const FIRST_CAP_BYTES: u64 = 8 * 1024 * 1024;
    const SECOND_CAP_BYTES: u64 = 12 * 1024 * 1024;

    let service = try_guaranteed_runtime_test_service_with_concurrency(2)
        .expect("runtime-backed service for overlapping cap isolation");
    let first_owner = observed_attempt("overlap-first", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let first_binding = first_owner.admission_binding();
    let second_owner = observed_attempt("overlap-second", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let first = service
        .admit_query_with_request(
            QueryAdmissionRequest::new(first_binding.clone()).with_query_cap_bytes(FIRST_CAP_BYTES),
        )
        .expect("first explicit cap admits");
    let second = service
        .admit_query_with_request(
            QueryAdmissionRequest::new(second_owner.admission_binding()).with_query_cap_bytes(SECOND_CAP_BYTES),
        )
        .expect("second explicit cap admits");
    assert_eq!(first.device_budget_bytes(), FIRST_CAP_BYTES);
    assert_eq!(second.device_budget_bytes(), SECOND_CAP_BYTES);

    let first_context = first.acquire_context().expect("first context");
    let second_context = second.acquire_context().expect("second context");
    let first_executor = first.gpu_executor().expect("first GPU executor");
    let second_executor = second.gpu_executor().expect("second GPU executor");
    let entered = Arc::new(TokioBarrier::new(3));
    let (first_release_tx, first_release_rx) = oneshot::channel();
    let (second_release_tx, second_release_rx) = oneshot::channel();
    let first_job = tokio::spawn(first_executor.run_observed_scoped(
        GpuExecutionJobKind::QueryExecution,
        first_context.context().runtime_observation().clone(),
        {
            let entered = Arc::clone(&entered);
            move |_scope| async move {
                entered.wait().await;
                first_release_rx.await.expect("first job release signal");
            }
        },
    ));
    let second_job = tokio::spawn(second_executor.run_observed_scoped(
        GpuExecutionJobKind::QueryExecution,
        second_context.context().runtime_observation().clone(),
        {
            let entered = Arc::clone(&entered);
            move |_scope| async move {
                entered.wait().await;
                second_release_rx.await.expect("second job release signal");
            }
        },
    ));
    timeout(Duration::from_secs(10), entered.wait())
        .await
        .expect("both different-cap GPU jobs overlap at the barrier");

    let overlapping = service.device_ledger_reports().expect("overlap ledger report");
    let [overlapping] = overlapping.as_slice() else {
        panic!("expected one device ledger during overlap");
    };
    assert_eq!(overlapping.active_attempts, 2);
    assert_eq!(overlapping.live_grants_bytes, FIRST_CAP_BYTES + SECOND_CAP_BYTES);
    assert!(overlapping.grants.iter().any(|grant| grant.grant_bytes == FIRST_CAP_BYTES));
    assert!(overlapping.grants.iter().any(|grant| grant.grant_bytes == SECOND_CAP_BYTES));

    first_release_tx.send(()).expect("first job remains controllable after overlap");
    first_job.await.expect("first job joins").expect("first GPU job completes");
    drop(first_executor);
    drop(second_executor);
    drop(first_context);

    // The release gate is after report sealing and before the envelope drops
    // its lane and grant. This is the cleaning window in which capacity must
    // remain committed to the failed attempt. The terminal commit releases
    // the admitted grant.
    let first_gate = first.lifetime.arm_envelope_release_gate();
    let first_release = std::thread::spawn(move || {
        first_owner
            .commit_terminal(RuntimeTerminalOutcome::Failed {
                error_code: AttemptFailureCode::CudfTemporaryFailure,
                error_status: AttemptFailureStatus::Temporary,
            })
            .expect("failed first attempt commits its terminal outcome");
        drop(first);
    });
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || first_gate.entered(),
        "failed first attempt reaches envelope cleaning gate",
    );
    let cleaning = service.device_ledger_reports().expect("cleaning ledger report");
    let [cleaning] = cleaning.as_slice() else {
        panic!("expected one device ledger while first attempt cleans");
    };
    assert_eq!(cleaning.active_attempts, 2, "cleaning first attempt still owns its active lane");
    assert_eq!(cleaning.live_grants_bytes, FIRST_CAP_BYTES + SECOND_CAP_BYTES);
    assert!(
        first_binding.observation_handle().sealed_snapshot().is_some(),
        "failed attempt report seals before envelope release"
    );
    assert!(
        service
            .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(FIRST_CAP_BYTES))
            .expect("third admission probe")
            .is_none(),
        "cleaning attempt still consumes its lane and cap",
    );

    first_gate.open();
    first_release.join().expect("first release completes");
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || {
            service.device_ledger_reports().ok().is_some_and(|reports| {
                reports.first().is_some_and(|report| {
                    report.active_attempts == 1
                        && report.live_grants_bytes == SECOND_CAP_BYTES
                        && report.grants.len() == 1
                        && report.grants[0].grant_bytes == SECOND_CAP_BYTES
                })
            })
        },
        "first failed grant releases without touching second grant",
    );

    second_release_tx.send(()).expect("second job release signal");
    second_job.await.expect("second job joins").expect("second GPU job completes");
    drop(second_context);
    drop(second);
    drop(second_owner);
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || {
            service.device_ledger_reports().ok().is_some_and(|reports| {
                reports.first().is_some_and(|report| report.active_attempts == 0 && report.grants.is_empty())
            })
        },
        "second grant releases after its job and cleanup finish",
    );
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("overlapping cap isolation service drains");
}

/// Cancelling one of two overlapping attempts releases only the cancelled
/// attempt's grant. Its capacity stays committed for the whole cleaning
/// window, and the concurrently active attempt's differently sized grant is
/// never consumed or released on its behalf.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_one_active_attempt_releases_only_its_own_differently_capped_grant() {
    const CANCELLED_CAP_BYTES: u64 = 8 * 1024 * 1024;
    const SURVIVING_CAP_BYTES: u64 = 12 * 1024 * 1024;

    let service = try_guaranteed_runtime_test_service_with_concurrency(2)
        .expect("runtime-backed service for active cancellation isolation");
    let backend_lifecycle = BackendLifecycle::new();
    let begin_attempt = || {
        AttemptOwner::begin(
            BeginAttemptRequest {
                observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
                ..BeginAttemptRequest::default()
            },
            backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
            Arc::new(AttemptBackendCounters::default()),
            None,
        )
    };
    let cancelled_attempt = begin_attempt();
    let surviving_attempt = begin_attempt();
    let cancelled_owner = cancelled_attempt.owner;
    let surviving_owner = surviving_attempt.owner;
    let cancelled_completion = cancelled_attempt.completion.completion();
    for owner in [&cancelled_owner, &surviving_owner] {
        let handle = owner.handle();
        handle.record_planning_started().expect("planning starts");
        handle.record_planning_finished().expect("planning finishes");
        handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    }

    let cancelled = service
        .admit_query_with_request(
            QueryAdmissionRequest::new(cancelled_owner.admission_binding())
                .with_query_cap_bytes(CANCELLED_CAP_BYTES)
                .with_execution_cancellation(cancelled_owner.cancellation().execution_cancellation()),
        )
        .expect("cancelled attempt admits");
    let surviving = service
        .admit_query_with_request(
            QueryAdmissionRequest::new(surviving_owner.admission_binding())
                .with_query_cap_bytes(SURVIVING_CAP_BYTES)
                .with_execution_cancellation(surviving_owner.cancellation().execution_cancellation()),
        )
        .expect("surviving attempt admits");
    assert_eq!(cancelled.device_budget_bytes(), CANCELLED_CAP_BYTES);
    assert_eq!(surviving.device_budget_bytes(), SURVIVING_CAP_BYTES);
    for owner in [&cancelled_owner, &surviving_owner] {
        let handle = owner.handle();
        handle.record_physical_execution_started().expect("physical execution starts");
        handle.record_gpu_execution_started().expect("GPU execution starts");
    }

    let cancelled_context = cancelled
        .acquire_context_with_cancellation(cancelled_owner.cancellation().execution_cancellation())
        .expect("cancelled attempt context");
    let surviving_context = surviving
        .acquire_context_with_cancellation(surviving_owner.cancellation().execution_cancellation())
        .expect("surviving attempt context");
    let cancelled_executor = cancelled.gpu_executor().expect("cancelled GPU executor");
    let surviving_executor = surviving.gpu_executor().expect("surviving GPU executor");
    let entered = Arc::new(TokioBarrier::new(3));
    let (cancelled_release_tx, cancelled_release_rx) = oneshot::channel();
    let (surviving_release_tx, surviving_release_rx) = oneshot::channel();
    let cancelled_job = tokio::spawn(cancelled_executor.run_observed_scoped(
        GpuExecutionJobKind::QueryExecution,
        cancelled_context.context().runtime_observation().clone(),
        {
            let entered = Arc::clone(&entered);
            move |_scope| async move {
                entered.wait().await;
                cancelled_release_rx.await.expect("cancelled job release signal");
            }
        },
    ));
    let surviving_job = tokio::spawn(surviving_executor.run_observed_scoped(
        GpuExecutionJobKind::QueryExecution,
        surviving_context.context().runtime_observation().clone(),
        {
            let entered = Arc::clone(&entered);
            move |_scope| async move {
                entered.wait().await;
                surviving_release_rx.await.expect("surviving job release signal");
            }
        },
    ));
    timeout(Duration::from_secs(10), entered.wait())
        .await
        .expect("both different-cap GPU jobs overlap at the barrier");

    let overlapping = service.device_ledger_reports().expect("overlap ledger report");
    let [overlapping] = overlapping.as_slice() else {
        panic!("expected one device ledger during overlap");
    };
    assert_eq!(overlapping.active_attempts, 2);
    assert_eq!(overlapping.live_grants_bytes, CANCELLED_CAP_BYTES + SURVIVING_CAP_BYTES);

    let cancellation = cancelled_owner.cancellation();
    cancellation.cancel();
    let cancellation_error = cancellation
        .check("query_runtime.tests.active_cancellation")
        .expect_err("an effective cancellation must produce a terminal error");
    assert!(
        cancelled_context.context().check_cancellation("query_runtime.tests.active_cancellation").is_err(),
        "the cancelled attempt's own execution context observes the signal",
    );
    assert!(
        surviving_context.context().check_cancellation("query_runtime.tests.active_cancellation").is_ok(),
        "cancelling one attempt must not cancel the other",
    );
    cancelled_release_tx.send(()).expect("cancelled job remains controllable");
    cancelled_job.await.expect("cancelled job joins").expect("cancelled GPU job completes");
    drop(cancelled_executor);
    drop(cancelled_context);

    // The release gate is after report sealing and before the envelope drops
    // its lane and grant: the cancelled attempt must still own its capacity.
    // The terminal commit releases the admitted grant.
    let cancelled_gate = cancelled.lifetime.arm_envelope_release_gate();
    let cancelled_release = std::thread::spawn(move || {
        cancelled_owner.commit_query_error(&cancellation_error).expect("cancelled terminal commit");
        drop(cancelled);
    });
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || cancelled_gate.entered(),
        "cancelled attempt reaches envelope cleaning gate",
    );
    let cleaning = service.device_ledger_reports().expect("cleaning ledger report");
    let [cleaning] = cleaning.as_slice() else {
        panic!("expected one device ledger while the cancelled attempt cleans");
    };
    assert_eq!(cleaning.active_attempts, 2, "a cleaning cancellation still owns its lane");
    assert_eq!(cleaning.live_grants_bytes, CANCELLED_CAP_BYTES + SURVIVING_CAP_BYTES);
    assert!(
        service
            .try_admit_query_with_request(
                QueryAdmissionRequest::test_attempt().with_query_cap_bytes(CANCELLED_CAP_BYTES),
            )
            .expect("replacement admission probe")
            .is_none(),
        "a cleaning cancelled grant is not returned early",
    );

    cancelled_gate.open();
    cancelled_release.join().expect("cancelled release completes");
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || {
            service.device_ledger_reports().ok().is_some_and(|reports| {
                reports.first().is_some_and(|report| {
                    report.active_attempts == 1
                        && report.live_grants_bytes == SURVIVING_CAP_BYTES
                        && report.grants.len() == 1
                        && report.grants[0].grant_bytes == SURVIVING_CAP_BYTES
                })
            })
        },
        "cancelled grant releases without touching the surviving grant",
    );
    let cancelled_report = cancelled_completion.try_report().expect("cancelled attempt report is published");
    assert!(matches!(cancelled_report.outcome, RuntimeTerminalOutcome::Cancelled));
    assert_eq!(cancelled_report.cleanup_failure_count, 0);

    surviving_release_tx.send(()).expect("surviving job release signal");
    surviving_job.await.expect("surviving job joins").expect("surviving GPU job completes");
    drop(surviving_executor);
    drop(surviving_context);
    surviving_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("surviving terminal commit");
    drop(surviving);
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || {
            service.device_ledger_reports().ok().is_some_and(|reports| {
                reports.first().is_some_and(|report| report.active_attempts == 0 && report.grants.is_empty())
            })
        },
        "surviving grant releases after its own job and cleanup finish",
    );
    service
        .close_and_wait_for_quiescence(Duration::from_secs(10))
        .expect("active cancellation isolation service drains");
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
    let counters = Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let observation = attempt_owner.observation_for_test();
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission");
    drop(first.acquire_context().expect("context"));
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");

    let domain = first.lifetime.envelope_domain_for_test().expect("device-backed domain");
    let mut terminal_usage = domain.usage().expect("domain usage sample for injected terminal state");
    terminal_usage.active_members = 1;
    terminal_usage.current_allocations = 2;
    terminal_usage.current_bytes = 3;
    terminal_usage.current_aligned_bytes = 4;
    first.lifetime.inject_terminal_domain_usage_for_test(terminal_usage);
    // The terminal commit releases the admitted grant.
    let gate = first.lifetime.arm_release_commit_gate();
    let release = std::thread::spawn(move || {
        attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("primary outcome commits first");
        drop(first);
    });
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || gate.entered(),
        "residual-usage release reaches the pre-seal barrier",
    );

    let report_before_gate_opens = observation.sealed_snapshot();
    let ledger_before_gate_opens = device.ledger().snapshot();
    let service_before_gate_opens = service.admission_snapshot();
    let replacement_before_gate_opens = service.try_admit_query();
    gate.open();
    release.join().expect("release thread completes");

    assert!(report_before_gate_opens.is_none(), "report must remain unsealed at the release commit gate");
    let ledger_before_gate_opens = ledger_before_gate_opens.expect("device ledger snapshot at release commit gate");
    let reason = match ledger_before_gate_opens.health {
        DeviceAdmissionHealth::Closed { reason, .. } => reason,
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
        AttemptCleanupDisposition::Completed,
        "the cleanup callback completed; residual usage, not cleanup failure, chose quarantine"
    );
    assert_eq!(
        report.capacity_disposition,
        AttemptCapacityDisposition::RetainedQuarantine,
        "the report projects the same final usage-derived health decision that closed the ledger"
    );
    wait_until(
        deadline,
        || device.ledger().snapshot().map(|snapshot| snapshot.active_attempts == 0).unwrap_or(false),
        "quarantined attempt releases its held capacity after report publication",
    );
}

/// Owner attestation must reject a residual the allocation registry still
/// owns but no published output owner holds. The residual here is a real
/// device allocation leaked into the attempt domain, not a counter mismatch.
#[test]
fn resident_owner_attestation_mismatch_is_a_cleanup_failure() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for resident attestation");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = Arc::new(AttemptBackendCounters::default());
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
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let admitted = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission");
    let grant = admitted.lifetime.memory_grant().expect("memory grant");
    // Leak a real allocation into the attempt domain and forget it. The
    // registry keeps owning it, so freeze must see a live allocation that no
    // published output owner accounts for.
    let leaked_context = admitted.acquire_context().expect("attempt context for leaked allocation");
    let leaked_runtime = leaked_context.context().shared_runtime().expect("attempt runtime");
    let leaked_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "leaked",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from(vec![7; 4096]))],
    )
    .expect("leaked host batch");
    let leaked_frame = GpuDataFrame::new(
        DataFrame::from_record_batch(leaked_runtime.as_ref(), &leaked_batch).expect("leaked device upload"),
    );
    let leaked_bytes = leaked_frame.device_allocated_size_bytes().expect("leaked allocation bytes");
    let published_owner = grant
        .retain_resident_owner(leaked_bytes)
        .expect("equal-byte owner charge succeeds")
        .expect("positive owner bytes");
    std::mem::forget(leaked_frame);
    drop(leaked_runtime);
    drop(leaked_context);
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminal outcome");

    drop(admitted);
    wait_until(Instant::now() + Duration::from_secs(10), || completion.is_ready(), "resident mismatch report");
    let report = completion.try_report().expect("published attempt report");
    assert_eq!(report.cleanup_failure_count, 1);
    assert!(report.cleanup_failure_count_complete);
    assert_eq!(
        report.first_cleanup_failure.as_ref().map(CleanupFailureSummary::phase),
        Some(CleanupFailurePhase::FinalValidation)
    );
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::RetainedQuarantine);
    assert!(matches!(
        device.ledger().snapshot().expect("ledger snapshot").health,
        DeviceAdmissionHealth::Closed { .. }
    ));
    drop(published_owner);
    drop(grant);
}

/// The admission claim is a one-time transfer from its owner into the admitted
/// envelope. A second admission binding can carry observation identity but
/// cannot claim again or manufacture competing cleanup/capacity facts.
#[test]
fn admitted_finalizer_transfers_once_and_publishes_no_pending_capacity() {
    let service = QueryService::new_for_admission_tests(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(10) },
        test_ledger(64 * 1024 * 1024, 1, Default::default()),
    )
    .expect("admission-only service");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = Arc::new(AttemptBackendCounters::default());
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
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission");
    assert!(matches!(
        attempt_owner.admission_binding().claim(),
        Err(AttemptLifecycleError::AdmissionAlreadyClaimed)
    ));

    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("primary outcome commits first");
    drop(handle);
    let report = completion.wait_timeout(Duration::from_secs(1)).expect("finalized report publishes");
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
    assert!(report.timestamps.report_ready_ns.is_some());
}

fn begin_native_attempt() -> (AttemptOwner, AttemptCompletion) {
    let backend_lifecycle = BackendLifecycle::new();
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::new(AttemptBackendCounters::default()),
        None,
    );
    let handle = attempt.owner.handle();
    handle.record_planning_started().expect("planning starts");
    handle.record_planning_finished().expect("planning finishes");
    handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let completion = attempt.completion.completion();
    (attempt.owner, completion)
}

/// The second admission claim of an attempt is rejected with a typed error
/// before any queue or ledger change. The first admission keeps its grant and
/// its lifecycle, no ledger closes, and an independent attempt still admits.
#[test]
fn second_admission_claim_is_rejected_before_queue_or_ledger_changes() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let device = service
        .inner
        .resources
        .as_ref()
        .expect("runtime-backed service")
        .devices
        .context_by_ordinal(0)
        .expect("device context");
    let (attempt_owner, completion) = begin_native_attempt();
    let first = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("first admission");

    let error = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect_err("an attempt has one admission claim");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("attempt_admission_already_claimed"));
    assert!(matches!(
        error.source_error().and_then(|source| source.downcast_ref::<AttemptLifecycleError>()),
        Some(AttemptLifecycleError::AdmissionAlreadyClaimed)
    ));
    let ledger = device.ledger().snapshot().expect("ledger snapshot");
    assert_eq!(ledger.active_attempts, 1, "the rejected claim never reached the ledger");
    assert!(matches!(device.ledger().health(), Ok(DeviceAdmissionHealth::Open)));
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);
    assert!(!completion.is_ready(), "the rejected claim leaves the first admission's attempt running");
    // The adapter reports every failed admission through the rejection view;
    // for a claimed attempt it must change nothing.
    attempt_owner.admission_rejection().reject(&error).expect("a claimed attempt ignores the rejection");
    assert!(!completion.is_ready());

    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminal outcome");
    drop(first);
    let report = completion.wait_timeout(Duration::from_secs(10)).expect("the first admission publishes the report");
    assert_eq!(report.outcome, RuntimeTerminalOutcome::Succeeded);
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
    assert!(report.primary_failure.is_none(), "the rejected claim records no primary failure");

    let independent = service.admit_query().expect("the fleet still admits independent attempts");
    drop(independent);
}

/// The attempt owner commits a terminal outcome while its admission is
/// assembling the envelope. The admission then cannot publish a grant that
/// would never run: it rolls back, and the attempt publishes exactly one
/// not-admitted report with the owner's outcome. No ledger closes.
#[test]
fn owner_terminal_during_admission_publishes_one_report_and_keeps_ledgers_open() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let device = service
        .inner
        .resources
        .as_ref()
        .expect("runtime-backed service")
        .devices
        .context_by_ordinal(0)
        .expect("device context");
    let (probe, pause) = AdmissionTransactionProbe::pausing_at(AdmissionStep::AfterWorkerScope);
    service.set_admission_transaction_probe(probe);
    let (attempt_owner, completion) = begin_native_attempt();
    let binding = attempt_owner.admission_binding();
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service.admit_query_with_request(QueryAdmissionRequest::new(binding))
    });

    pause.wait_until_entered(Duration::from_secs(10));
    attempt_owner
        .commit_failure_summary(AttemptFailureSummary::backend_closing())
        .expect("the owner commits its terminal outcome");
    assert!(!completion.is_ready(), "the claimed admission publishes the report, not the owner");
    pause.release();

    let error = admission.join().expect("admission thread joins").expect_err("a terminal attempt is not admitted");
    assert_eq!(error.fact_value("reason"), Some("attempt_terminal_before_admission"));
    let report = completion.wait_timeout(Duration::from_secs(10)).expect("the attempt publishes one report");
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::NotAdmitted);
    assert_eq!(report.primary_failure, Some(AttemptFailureSummary::backend_closing()));
    assert!(report.admission.is_none());
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || device.ledger().snapshot().map(|snapshot| snapshot.active_attempts == 0).unwrap_or(false),
        "the rolled-back envelope releases its provisional grant",
    );
    assert!(matches!(device.ledger().health(), Ok(DeviceAdmissionHealth::Open)));
    service.set_admission_transaction_probe(AdmissionTransactionProbe::disabled());
    let independent = service.admit_query().expect("the fleet still admits independent attempts");
    drop(independent);
}

/// A stream synchronization failure during cleanup is device-local fault
/// evidence: the device closes before report publication and the attempt's
/// capacity is never reusable, while the report still publishes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_cleanup_quarantines_device_and_never_returns_capacity() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for cleanup quarantine");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let attempt_owner =
        observed_attempt("failed-cleanup-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let binding = attempt_owner.admission_binding();
    let first = service.admit_query_with_request(QueryAdmissionRequest::new(binding.clone())).expect("admission");
    drop(first.acquire_context().expect("context"));

    device.stream_pool().fail_next_synchronization_for_test();
    drop(first);
    // The abandoned attempt ends as `Dropped`, which releases its grant.
    drop(attempt_owner);

    let deadline = Instant::now() + Duration::from_secs(10);
    wait_until(
        deadline,
        || {
            matches!(
                device.ledger().snapshot().map(|snapshot| snapshot.health),
                Ok(DeviceAdmissionHealth::Closed { .. })
            )
        },
        "device ledger closes under quarantine",
    );
    assert!(binding.observation_handle().sealed_snapshot().is_some(), "report still publishes after failed cleanup");

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
#[ignore = "requires at least 2 GPUs"]
fn device_local_cleanup_fault_seals_retry_on_a_frozen_eligible_alternate() {
    let visible_devices = DeviceId::count().expect("enumerate CUDA devices for alternate retry coverage");
    assert!(
        visible_devices >= 2,
        "alternate retry coverage requires at least two visible CUDA devices; found {visible_devices}"
    );

    const CAPACITY_BYTES: u64 = 64 * 1024 * 1024;
    const ATTEMPT_FLOOR_BYTES: u64 = CAPACITY_BYTES / 8;
    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5) },
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
    let counters = Arc::new(AttemptBackendCounters::default());
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
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("first admission");
    assert_eq!(handle.device_ordinal(), 0, "equal-profile placement uses ordinal zero deterministically");
    drop(handle.acquire_context().expect("root context"));
    device_zero.stream_pool().fail_next_synchronization_for_test();
    let execution_error = QueryError::new(
        QueryErrorCode::CudfTemporaryFailure,
        "query_engine.query_service.alternate_retry_test",
        "injected execution failure before device-local cleanup fault",
    );
    attempt_owner.commit_query_error(&execution_error).expect("attempt failure records before cleanup");
    drop(handle);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("faulted attempt report seals");
    assert_eq!(
        report.retry_advice,
        AttemptRetryAdvice {
            recovery: AttemptRetryRecovery::RetryAnotherEligibleDevice,
            prerequisite: AttemptRetryPrerequisite::QuarantinedDeviceAndEligibleAlternate,
        },
        "retry advice requires the post-quarantine alternate proof"
    );
    let fault = report.native_fault_evidence.expect("device-local cleanup failure retains native evidence");
    assert_eq!(fault.isolation, AttemptFaultIsolation::DeviceLocal);
    assert_eq!(fault.teardown_health, AttemptTeardownHealth::StreamSynchronizationFailed);
    assert!(
        matches!(device_zero.ledger().health(), Ok(DeviceAdmissionHealth::Closed { .. })),
        "the failed device is quarantined before retry advice is published"
    );
    assert!(
        matches!(device_one.ledger().health(), Ok(DeviceAdmissionHealth::Open)),
        "the alternate proof retains only a currently healthy ledger"
    );

    let replacement = service.admit_query().expect("healthy alternate admits a new attempt");
    assert_eq!(replacement.device_ordinal(), 1, "the quarantined device cannot be selected again");
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
    let counters = Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission");
    let context = handle.acquire_context().expect("root context");
    handle.lifetime.install_foreign_root_stream_cleanup_completion_for_test();
    drop(context);

    let primary_error = QueryError::new(
        QueryErrorCode::CudfTemporaryFailure,
        "query_engine.query_service.unknown_isolation_primary",
        "injected primary error before abandoned cleanup",
    );
    attempt_owner.commit_query_error(&primary_error).expect("primary error records before cleanup");
    drop(handle);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("abandoned-handoff report seals");
    assert!(matches!(
        report.outcome,
        RuntimeTerminalOutcome::Failed {
            error_code: AttemptFailureCode::CudfTemporaryFailure,
            error_status: AttemptFailureStatus::Temporary,
        }
    ));
    let primary_failure = report.primary_failure.as_ref().expect("primary failure remains authoritative");
    assert_eq!(primary_failure.code(), AttemptFailureCode::CudfTemporaryFailure);
    assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::CompletedWithFailures);
    assert_eq!(report.cleanup_failure_count, 1, "the abandoned callback is one cleanup failure");
    assert!(report.cleanup_failure_count_complete);
    let cleanup_failure = report.first_cleanup_failure.as_ref().expect("abandoned cleanup is retained");
    assert_eq!(cleanup_failure.phase(), CleanupFailurePhase::QueryServiceCallbackAbandoned);
    assert_eq!(cleanup_failure.failure().code(), AttemptFailureCode::CleanupCallbackAbandoned);
    assert!(cleanup_failure.complete(), "the sole cleanup failure remains a complete representative");
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::RetainedQuarantine);
    assert_eq!(report.health_transition, Some(AttemptHealthTransition::BackendUnhealthy));
    let fault = report.native_fault_evidence.expect("unknown isolation remains explicit");
    assert_eq!(fault.isolation, AttemptFaultIsolation::Unknown);
    assert_eq!(fault.teardown_health, AttemptTeardownHealth::CallbackAbandoned);
    assert!(fault.device_health.is_none(), "unknown isolation cannot fabricate a device-local health sample");
    assert!(fault.first_fault.is_none(), "unknown isolation cannot fabricate a device-local native fault");
    assert_eq!(
        report.retry_advice,
        AttemptRetryAdvice {
            recovery: AttemptRetryRecovery::RecoverBackend,
            prerequisite: AttemptRetryPrerequisite::BackendRecovered,
        },
        "unknown isolation is sealed as backend-recovery advice rather than a device retry"
    );
    let resource_evidence = report.resource_evidence.expect("admitted attempt retains its immutable resource evidence");
    assert_eq!(
        resource_evidence.grant_cap_bytes.provenance,
        AttemptEvidenceProvenance::AttemptExact
    );
    assert!(resource_evidence.grant_cap_bytes.value.is_some());
    assert_eq!(
        resource_evidence.domain_cap_bytes.provenance,
        AttemptEvidenceProvenance::AttemptExact
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
            AttemptEvidenceProvenance::Unavailable,
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
            Ok(DeviceAdmissionHealth::Closed { .. })
        ),
        "duplicate cleanup handoff closes the scheduler ledger"
    );
    let error =
        service.admit_query().expect_err("backend rejects new admission after unknown-isolation cleanup collision");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("backend_unhealthy"));
}

/// An OOM-class outcome alone never quarantines: the device stays open and
/// healthy capacity is reusable after report-ready.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oom_outcome_keeps_device_open_and_capacity_reusable_after_report() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for OOM disposition");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let attempt_owner = observed_attempt("oom-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let binding = attempt_owner.admission_binding();
    let first = service.admit_query_with_request(QueryAdmissionRequest::new(binding.clone())).expect("admission");
    drop(first.acquire_context().expect("context"));
    attempt_owner
        .commit_terminal(RuntimeTerminalOutcome::Failed {
            error_code: AttemptFailureCode::CudfResourceExhausted,
            error_status: AttemptFailureStatus::Temporary,
        })
        .expect("OOM outcome commits");
    drop(first);

    let deadline = Instant::now() + Duration::from_secs(10);
    wait_until(deadline, || binding.observation_handle().sealed_snapshot().is_some(), "OOM report seals");
    wait_until(
        deadline,
        || device.ledger().snapshot().map(|snapshot| snapshot.active_attempts == 0).unwrap_or(false),
        "capacity returns after report-ready",
    );
    assert!(
        matches!(
            device.ledger().snapshot().map(|snapshot| snapshot.health),
            Ok(DeviceAdmissionHealth::Open)
        ),
        "OOM alone never quarantines the device"
    );

    let second = service.admit_query().expect("capacity is reusable after the OOM report");
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
    let counters = Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("admission");
    let grant_bytes = handle.device_budget_bytes();
    let requested_bytes = grant_bytes.checked_add(1).expect("grant leaves one byte above its immutable cap");
    let requested_bytes = usize::try_from(requested_bytes).expect("test device allocation size fits usize");
    let context = handle.acquire_context().expect("context");
    let allocation_error = match context.context().runtime().expect("query runtime").device_buffer(requested_bytes) {
        Ok(_) => panic!("allocation above the immutable grant limit is rejected"),
        Err(error) => error,
    };
    assert_eq!(allocation_error.kind(), CudfErrorKind::ResourceExhausted);
    drop(context);
    let terminal_error = cudf_query_error(
        "query_engine.query_service.release_commit_test",
        "attempt allocation exceeded its immutable grant limit",
        allocation_error,
    );
    assert_eq!(terminal_error.code(), QueryErrorCode::QueryMemoryLimitExceeded);
    attempt_owner.commit_query_error(&terminal_error).expect("terminal OOM outcome");
    drop(handle);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("OOM report publishes");
    assert_eq!(
        report.retry_advice,
        AttemptRetryAdvice {
            recovery: AttemptRetryRecovery::ChangeConfiguration,
            prerequisite: AttemptRetryPrerequisite::ConfigurationChanged,
        },
        "an exactly joined reservation-limit callback requires a configuration change"
    );
    let evidence = report.resource_evidence.expect("device-backed OOM has resource evidence");
    assert_eq!(evidence.ledger_sequence.provenance, AttemptEvidenceProvenance::DeviceSnapshot);
    assert!(evidence.ledger_sequence.value.is_some(), "one atomic ledger sequence is retained");
    assert_eq!(
        evidence.configured_limit_bytes.provenance,
        AttemptEvidenceProvenance::AttemptExact
    );
    assert_eq!(evidence.grant_cap_bytes.value, Some(grant_bytes));
    assert_eq!(evidence.grant_cap_bytes.provenance, AttemptEvidenceProvenance::AttemptExact);
    assert_eq!(evidence.domain_cap_bytes.provenance, AttemptEvidenceProvenance::AttemptExact);
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
        AttemptEvidenceProvenance::SampledAtTerminalBoundary
    );
    assert_eq!(
        evidence.post_cleanup_usage.provenance,
        AttemptEvidenceProvenance::SampledAtTerminalBoundary
    );
    assert_eq!(
        evidence.allocator_event.provenance,
        AttemptEvidenceProvenance::AttemptExact,
        "the reservation-limit callback is joined through the completed domain-generation interval"
    );
    let allocator_event = evidence.allocator_event.value.expect("exact callback record is retained");
    assert_eq!(allocator_event.requested_bytes, u64::try_from(requested_bytes).expect("usize fits u64"));
    assert_eq!(allocator_event.source, AttemptAllocatorEventSource::ReservationLimit);
    assert_eq!(
        allocator_event.action,
        AttemptEvidenceField::attempt_exact(
            AttemptAllocatorEventAction::Decline
        )
    );
    assert_eq!(
        evidence.unattributed_device_allocator_event.provenance,
        AttemptEvidenceProvenance::Unavailable,
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
        Ok(DeviceAdmissionHealth::Closed { .. })
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
    let (disposition, usage) =
        decide_capacity_disposition(0, Some(&domain), CleanupEvidence::Completed, Some(cleared_usage));
    assert_eq!(disposition, CapacityCommitDisposition::HealthyReuse);
    let usage = usage.expect("healthy device samples domain usage");
    assert_eq!(usage.domain_id, domain.domain_id());

    type ResidualInjector = (&'static str, fn(&mut QueryMemoryDomainUsage));
    let residuals: [ResidualInjector; 4] = [
        ("active_members", |usage| usage.active_members = 1),
        ("current_allocations", |usage| usage.current_allocations = 1),
        ("current_bytes", |usage| usage.current_bytes = 1),
        ("current_aligned_bytes", |usage| usage.current_aligned_bytes = 1),
    ];
    for (field, inject_residual) in residuals {
        let mut residual_usage = cleared_usage;
        inject_residual(&mut residual_usage);
        let (disposition, observed_usage) =
            decide_capacity_disposition(0, Some(&domain), CleanupEvidence::Completed, Some(residual_usage));
        let reason = match disposition {
            CapacityCommitDisposition::DeviceQuarantine { reason } => reason,
            disposition => panic!("nonzero {field} must quarantine the device, got {disposition:?}"),
        };
        assert!(reason.contains(&format!("{field}=1")), "quarantine reason preserves {field}: {reason}");
        assert_eq!(observed_usage, Some(residual_usage), "quarantine returns the sampled terminal usage");
    }

    let abandonment = CleanupEvidence::from(StreamCleanupOutcome::CallbackAbandoned);
    assert_eq!(abandonment, CleanupEvidence::CallbackAbandoned);
    let abandonment_decision = decide_capacity_commit(0, Some(&domain), abandonment, None);
    let disposition = abandonment_decision.disposition;
    assert!(
        matches!(disposition, CapacityCommitDisposition::BackendUnhealthy { .. }),
        "abandoned cleanup has unknown isolation: {disposition:?}"
    );
    let abandonment_evidence =
        abandonment_decision.native_fault_evidence.expect("unknown isolation has terminal evidence");
    assert_eq!(abandonment_evidence.isolation, AttemptFaultIsolation::Unknown);
    assert_eq!(abandonment_evidence.teardown_health, AttemptTeardownHealth::CallbackAbandoned);
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
    assert_eq!(synchronization_evidence.isolation, AttemptFaultIsolation::DeviceLocal);
    assert_eq!(
        synchronization_evidence.teardown_health,
        AttemptTeardownHealth::StreamSynchronizationFailed
    );

    // cuDF's same-thread test registry makes the sibling-owned health
    // snapshot deterministic without poisoning the process-wide registry
    // used by the asynchronous GPU cleanup worker. The decision consumes the
    // snapshot directly, preserving each native status ABI family.
    let device = DeviceId::new(0).expect("runtime service established device zero");
    for (native_status, expected_status) in [
        (
            DeviceFaultNativeStatus::CUDF_FATAL_CUDA,
            match DeviceFaultNativeStatus::CUDF_FATAL_CUDA {
                DeviceFaultNativeStatus::Cudf(status) => {
                    AttemptNativeStatus::Cudf(status)
                },
                _ => unreachable!(),
            },
        ),
        (DeviceFaultNativeStatus::Cuda(700), AttemptNativeStatus::Cuda(700)),
    ] {
        let _device_safety = DeviceSafetyTestScope::install([]);
        report_device_fault(
            device,
            DeviceFaultKind::FatalDeviceOperation,
            "test_native_fault",
            Some(native_status),
        );
        let native_fault_decision =
            decide_capacity_commit(0, Some(&domain), CleanupEvidence::Completed, Some(cleared_usage));
        assert!(
            matches!(native_fault_decision.disposition, CapacityCommitDisposition::DeviceQuarantine { .. }),
            "a native device fault quarantines capacity before report publication"
        );
        let evidence =
            native_fault_decision.native_fault_evidence.expect("device-local native fault retains terminal evidence");
        assert_eq!(evidence.isolation, AttemptFaultIsolation::DeviceLocal);
        assert_eq!(evidence.teardown_health, AttemptTeardownHealth::Completed);
        assert_eq!(
            evidence.device_health.expect("device-local evidence retains health snapshot").status,
            AttemptDeviceHealthStatus::Unhealthy
        );
        let fault = evidence.first_fault.expect("fault snapshot retains the first structured native fault");
        assert_eq!(fault.kind, AttemptNativeFaultKind::FatalDeviceOperation);
        assert_eq!(fault.status, Some(expected_status));
    }

    // Ledger-only releases carry no domain and always healthy-reuse without
    // touching CUDA FFI.
    let (disposition, usage) = decide_capacity_disposition(0, None, CleanupEvidence::CallbackAbandoned, None);
    assert_eq!(disposition, CapacityCommitDisposition::HealthyReuse);
    assert!(usage.is_none());

    handle.end_test_attempt();
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

/// A frozen resident producer must never publish reusable ledger capacity
/// while it still holds its attempt lane. With `max_active = 1`, a consumer
/// that passes the ledger has to find a free lane; publishing `B - R` before
/// the lane is released let the consumer through the ledger and then fail its
/// non-blocking lane acquisition with an internal `lane_exhausted` error.
#[test]
fn frozen_resident_publish_never_precedes_the_producer_lane_release() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for frozen publish ordering");
    let backend_lifecycle = BackendLifecycle::new();
    let counters = Arc::new(AttemptBackendCounters::default());
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
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    let producer = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("producer admission");
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");

    // Retain a real resident owner so this attempt's terminal commit takes
    // the frozen-resident path rather than an ordinary capacity release. The
    // allocation must exist in the attempt domain: owner attestation reads
    // the allocation registry, not a synthetic charge.
    let grant = producer.lifetime.memory_grant().expect("producer memory grant");
    let producer_context = producer.acquire_context().expect("producer context");
    let producer_runtime = producer_context.context().shared_runtime().expect("producer runtime");
    let resident_schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        false,
    )]));
    let resident_batch = RecordBatch::try_new(
        Arc::clone(&resident_schema),
        vec![Arc::new(Int32Array::from(vec![11; 1024]))],
    )
    .expect("resident host batch");
    let resident_frame = GpuDataFrame::new(
        DataFrame::from_record_batch(producer_runtime.as_ref(), &resident_batch).expect("resident upload"),
    );
    let resident_bytes = resident_frame.device_allocated_size_bytes().expect("resident output bytes");
    let resident_owner =
        grant.retain_resident_owner(resident_bytes).expect("resident owner is recorded").expect("positive owner bytes");
    let resident_result = GpuResultHandle::new_with_resident_owner(
        producer_context.context(),
        resident_frame,
        resident_schema,
        Some(resident_owner),
    )
    .expect("resident result binds its allocation owner");
    drop(producer_runtime);
    drop(producer_context);

    // Stop the producer at the exact point where the report is sealed but the
    // attempt lane, grant, and permit are still held. Any ledger capacity a
    // consumer can observe here is capacity without a lane. The terminal
    // commit releases the admitted grant.
    let gate = producer.lifetime.arm_envelope_release_gate();
    let release = std::thread::spawn(move || {
        attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("terminal outcome");
        drop(producer);
    });
    wait_until(
        Instant::now() + Duration::from_secs(10),
        || gate.entered(),
        "frozen release reaches the barrier while the attempt still owns its lane",
    );

    let sealed_admission = service.try_admit_query();
    gate.open();
    release.join().expect("producer release thread completes");

    match sealed_admission {
        Ok(None) => {},
        Ok(Some(handle)) => {
            panic!(
                "a consumer admitted while the producer still held lane {:?}",
                handle.lifetime.admitted_device_requirement_for_test(0).is_some()
            )
        },
        Err(error) => panic!(
            "the frozen publish exposed reusable capacity before the lane was released: {:?}: {error}",
            error.code()
        ),
    }

    // Once the lane is released and B to R is published, the same consumer
    // request succeeds on the capacity the frozen output left behind.
    let deadline = Instant::now() + Duration::from_secs(10);
    let admitted = loop {
        if let Some(handle) = service.try_admit_query().expect("post-release admission is not an internal error") {
            break handle;
        }
        assert!(Instant::now() < deadline, "the released frozen capacity admits a consumer");
        std::thread::yield_now();
    };
    drop(admitted);

    let report = completion.try_report().expect("published producer report");
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::FrozenResident);
    drop(resident_result);
    drop(grant);
}
