//! Unit tests for the query coordinator.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::physical_plan::{empty::EmptyExec, union::UnionExec};
use futures::future::BoxFuture;
use query_engine::{
    error::{QueryError, QueryErrorCode},
    observability::{
        AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptFailureSummary, AttemptId,
        AttemptQueryId, AttemptRetryAdvice, AttemptRetryPrerequisite, AttemptRetryRecovery, AttemptTimestamps,
        QUERY_ATTEMPT_REPORT_SCHEMA_V1, QueryAttemptReport, RuntimeObservationConfig, RuntimeObservationOwner,
        RuntimeQueryId, RuntimeTerminalOutcome, SealedRuntimeObservationSnapshot,
    },
    runtime::ExecutionCancellation,
};
use tokio::{task::yield_now, time::timeout};

use super::{
    super::observability::{CapturedError, QueryStatsPayload},
    artifacts::{
        ArtifactEgressLimiter, FinalizationTestGate, LiveStatsCache, LiveStatsRead, LiveStatsWaitResult,
        LiveStatsWaiter, QUERY_STATS_HEADER_LINE, RUNTIME_ARTIFACT_CONTROL_SCHEMA, RUNTIME_ARTIFACT_CONTROL_SCHEMA_V2,
        compose_terminal_artifact, incomplete_runtime_control, install_coordinator_render_probe,
    },
    maintenance::{CoordinatorMaintenanceClock, QueryCoordinatorMaintenance},
    retention::checked_deadline,
    types::{QueryHotCounterSnapshot, QueryPhase, QueryTerminal},
    *,
};
use crate::server::{
    adapter_error::{Error as AdapterError, ErrorCode as AdapterErrorCode},
    error::adapter_error_to_status,
    planner::NativePlanningDiagnostics,
    report::{GpuCandidate, PlanningReport},
};

#[derive(Debug)]
struct TestMaintenanceClock {
    now: Mutex<Instant>,
    advanced: Notify,
    sleepers: AtomicUsize,
}

impl TestMaintenanceClock {
    fn new(now: Instant) -> Arc<Self> {
        Arc::new(Self { now: Mutex::new(now), advanced: Notify::new(), sleepers: AtomicUsize::new(0) })
    }

    fn advance(&self, duration: Duration) {
        let mut now = self.now.lock().unwrap();
        *now += duration;
        drop(now);
        self.advanced.notify_waiters();
    }

    fn advance_without_notifying(&self, duration: Duration) {
        *self.now.lock().unwrap() += duration;
    }

    async fn wait_for_sleepers(&self, expected: usize) {
        while self.sleepers.load(Ordering::SeqCst) < expected {
            yield_now().await;
        }
    }
}

impl CoordinatorMaintenanceClock for TestMaintenanceClock {
    fn now(&self) -> Instant {
        *self.now.lock().unwrap()
    }

    fn sleep_until(&self, deadline: Instant) -> BoxFuture<'_, ()> {
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            loop {
                let mut advanced = Box::pin(self.advanced.notified());
                advanced.as_mut().enable();
                if self.now() >= deadline {
                    return;
                }
                advanced.await;
            }
        })
    }
}

fn config() -> QueryCoordinatorConfig {
    QueryCoordinatorConfig {
        max_live_queries: 2,
        max_ticketed_queries: 1,
        max_active_logical_capture_bytes: 1024 * 1024,
        per_query_capture_max_bytes: 64 * 1024,
        per_query_artifact_max_bytes: 64 * 1024,
        retained_max_count: 1,
        retained_max_bytes: 64 * 1024,
        retained_ttl: Duration::from_secs(10),
        tombstone_max_count: 2,
        tombstone_ttl: Duration::from_secs(20),
        shutdown_grace: Duration::from_secs(5),
    }
}

fn register(coordinator: &Arc<QueryCoordinator>, query_id: &str) -> QueryPlanningLease {
    coordinator.register_with_id(query_id, "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096)).unwrap()
}

fn cpu_attempt_report(query_id: &str) -> QueryAttemptReport {
    QueryAttemptReport {
        schema: QUERY_ATTEMPT_REPORT_SCHEMA_V1.to_string(),
        attempt_id: AttemptId::mint(),
        query_id: Some(AttemptQueryId::new(query_id).expect("test query id is a valid attempt correlation")),
        trace_context: None,
        disposition: AttemptDisposition::CpuFallback,
        outcome: RuntimeTerminalOutcome::Succeeded,
        primary_failure: None,
        retry_advice: AttemptRetryAdvice::no_retry_needed(),
        device_ordinal: None,
        admission: None,
        admission_wait: None,
        composed_schedule: None,
        timestamps: AttemptTimestamps {
            submitted_ns: 1,
            planning_started_ns: Some(2),
            planning_finished_ns: Some(3),
            admission_requested_ns: None,
            admitted_ns: None,
            physical_execution_started_ns: Some(4),
            gpu_execution_started_ns: None,
            outcome_terminal_ns: 5,
            cleanup_complete_ns: None,
            report_ready_ns: Some(6),
        },
        cleanup_disposition: AttemptCleanupDisposition::NotRequired,
        cleanup_failure_count: 0,
        cleanup_failure_count_complete: true,
        first_cleanup_failure: None,
        capacity_disposition: AttemptCapacityDisposition::NotAdmitted,
        final_domain_usage: None,
        health_transition: None,
        native_fault_evidence: None,
        resource_evidence: None,
        observation_lost_records: 0,
        observation_retained_records: 0,
        observation_retained_bytes: 0,
        observation_truncated: false,
        observation_degraded: false,
        unix_epoch_ns: None,
    }
}

fn queue_backoff_attempt_report(query_id: &str) -> QueryAttemptReport {
    let mut report = cpu_attempt_report(query_id);
    let failure = AttemptFailureSummary::from_native_error(
        &QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.test.admission",
            "bounded admission queue is full",
        )
        .with_fact("reason", "admission_queue_full"),
    );
    report.outcome = RuntimeTerminalOutcome::failed(&failure);
    report.primary_failure = Some(failure);
    report.retry_advice = AttemptRetryAdvice {
        recovery: AttemptRetryRecovery::Backoff,
        prerequisite: AttemptRetryPrerequisite::CapacityAvailable,
    };
    report
}

fn cancelled_attempt_report_snapshot(query_id: &str) -> (QueryAttemptReport, SealedRuntimeObservationSnapshot) {
    let mut report = cpu_attempt_report(query_id);
    report.outcome = RuntimeTerminalOutcome::Cancelled;
    report.retry_advice = AttemptRetryAdvice::do_not_retry();
    sealed_attempt_report_snapshot(report)
}

fn sealed_attempt_report_snapshot(
    report: QueryAttemptReport,
) -> (QueryAttemptReport, SealedRuntimeObservationSnapshot) {
    let query_id = report.query_id.as_ref().map_or("attempt-query", |query_id| query_id.as_str());
    let terminal = report.outcome.clone();
    let snapshot_report = report.clone();
    let owner =
        RuntimeObservationOwner::new(RuntimeQueryId::new(query_id), None, RuntimeObservationConfig::bounded(16, 4096))
            .with_attempt_report_provider(Arc::new(move || snapshot_report.clone()));
    let snapshot = owner.finish_query(Some(terminal)).expect("attempt observation seals before coordinator handoff");
    // The sealed snapshot owns the immutable report projection accepted by the
    // coordinator. Return that exact projection rather than a pre-seal test
    // template, whose completeness footer may have been filled during sealing.
    let report = snapshot.attempt_report().expect("sealed snapshot carries its attempt report").clone();
    (report, snapshot)
}

fn attach_cancelled_attempt_then_fail_delivery(
    coordinator: &Arc<QueryCoordinator>,
    query_id: &str,
    config: RuntimeObservationConfig,
) {
    let planning = coordinator
        .register_with_id_with_attempt_observation(query_id, "select 1", None, None, config)
        .expect("attempt-owned query registers");
    let (report, snapshot) = cancelled_attempt_report_snapshot(query_id);
    coordinator
        .attach_attempt_report(planning.key(), Arc::new(report), Some(snapshot), None)
        .expect("sealed attempt report attaches to its exact coordinator generation");
    let delivery = tonic::Status::internal("Flight delivery failed after the attempt sealed");
    planning
        .into_execution()
        .expect("planning promotes to execution")
        .fail("delivery", CapturedError::from_status(&delivery));
}

#[test]
fn attempt_owned_snapshot_is_retained_without_a_second_observation_owner() {
    let coordinator = QueryCoordinator::new(config());
    let (query_id, planning) = coordinator
        .register_new_with_attempt_observation("select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .expect("attempt-owned registration succeeds");
    let sealed_report = cpu_attempt_report(&query_id);
    let owner = RuntimeObservationOwner::new(
        RuntimeQueryId::new(query_id.clone()),
        None,
        RuntimeObservationConfig::bounded(16, 4096),
    )
    .with_attempt_report_provider(Arc::new(move || sealed_report.clone()));
    let snapshot = owner
        .finish_query(Some(RuntimeTerminalOutcome::Succeeded))
        .expect("attempt observation seals before its report is handed off");
    let report = snapshot.attempt_report().expect("snapshot carries its sealed attempt report").clone();
    assert_eq!(snapshot.attempt_report(), Some(&report));
    assert_eq!(snapshot.footer().terminal.as_ref(), Some(&RuntimeTerminalOutcome::Succeeded));

    coordinator
        .attach_attempt_report(planning.key(), Arc::new(report.clone()), Some(snapshot), None)
        .expect("the matching sealed attempt snapshot is accepted");
    planning.into_execution().expect("planning promotes to execution").succeed();

    let (control, jsonl) = coordinator.artifact_frames(&query_id).expect("retained attempt artifact");
    let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
    assert_eq!(control["schema"], RUNTIME_ARTIFACT_CONTROL_SCHEMA_V2);
    assert_eq!(control["version"], 2);
    assert_eq!(control["state"], "sealed");
    assert_eq!(control["retryable"], false, "delivery retryability remains distinct from attempt advice");
    assert_eq!(control["attempt"]["attempt_id"], report.attempt_id.to_string());
    assert_eq!(control["attempt"]["retry_advice"]["recovery"], "no_retry_needed");
    assert_eq!(control["attempt"]["retry_advice"]["prerequisite"], "none");
    let jsonl = jsonl.expect("attempt snapshot remains available as JSONL");
    let records = std::str::from_utf8(&jsonl)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("retained JSONL record is valid"))
        .collect::<Vec<_>>();
    assert!(records.iter().any(|record| record["record_type"] == "manifest"), "{records:#?}");
    let retained_report = records
        .iter()
        .find(|record| record["record_type"] == "footer")
        .and_then(|record| record["data"].get("attempt_report"))
        .expect("sealed JSONL footer carries the canonical attempt report");
    assert_eq!(retained_report["attempt_id"], report.attempt_id.to_string());
    assert_eq!(retained_report["retry_advice"], control["attempt"]["retry_advice"]);
    let stats = coordinator.query_stats(&query_id).expect("attempt stats retained");
    let stats = std::str::from_utf8(&stats).unwrap();
    assert!(stats.contains(&format!("# attempt_id={}", report.attempt_id)), "{stats}");
    assert!(stats.contains("# attempt_retry_recovery=no_retry_needed"), "{stats}");
    assert!(stats.contains("# attempt_retry_prerequisite=none"), "{stats}");
}

#[test]
fn execution_tsv_projects_the_sealed_attempt_fact_matrix() {
    #[derive(Clone, Copy)]
    enum AttemptCase {
        NativeSucceeded,
        NativeFailed,
        NativeCancelled,
        NativeDropped,
        NativePanicked,
        CpuFallback,
        Missing,
    }

    #[derive(Clone, Copy)]
    enum DeliveryCase {
        Succeeded,
        Failed,
    }

    struct Case {
        name: &'static str,
        attempt: AttemptCase,
        delivery: DeliveryCase,
        expected_started: &'static str,
        expected_runtime_outcome: &'static str,
        expected_outcome: &'static str,
        expected_delivery_outcome: Option<&'static str>,
        expected_elapsed_ms: &'static str,
        expects_failure_summary: bool,
    }

    let cases = [
        Case {
            name: "native-succeeded",
            attempt: AttemptCase::NativeSucceeded,
            delivery: DeliveryCase::Succeeded,
            expected_started: "true",
            expected_runtime_outcome: "succeeded",
            expected_outcome: "succeeded",
            expected_delivery_outcome: None,
            expected_elapsed_ms: "0",
            expects_failure_summary: false,
        },
        Case {
            name: "native-failed",
            attempt: AttemptCase::NativeFailed,
            delivery: DeliveryCase::Failed,
            expected_started: "true",
            expected_runtime_outcome: "failed",
            expected_outcome: "failed",
            expected_delivery_outcome: None,
            expected_elapsed_ms: "0",
            expects_failure_summary: true,
        },
        Case {
            name: "native-cancelled",
            attempt: AttemptCase::NativeCancelled,
            delivery: DeliveryCase::Failed,
            expected_started: "true",
            expected_runtime_outcome: "cancelled",
            expected_outcome: "cancelled",
            expected_delivery_outcome: Some("failed"),
            expected_elapsed_ms: "0",
            expects_failure_summary: false,
        },
        Case {
            name: "native-dropped",
            attempt: AttemptCase::NativeDropped,
            delivery: DeliveryCase::Failed,
            expected_started: "true",
            expected_runtime_outcome: "dropped",
            expected_outcome: "dropped",
            expected_delivery_outcome: Some("failed"),
            expected_elapsed_ms: "0",
            expects_failure_summary: false,
        },
        Case {
            name: "native-panicked",
            attempt: AttemptCase::NativePanicked,
            delivery: DeliveryCase::Failed,
            expected_started: "true",
            expected_runtime_outcome: "panicked",
            expected_outcome: "panicked",
            expected_delivery_outcome: Some("failed"),
            expected_elapsed_ms: "0",
            expects_failure_summary: false,
        },
        Case {
            name: "cpu-fallback",
            attempt: AttemptCase::CpuFallback,
            delivery: DeliveryCase::Succeeded,
            expected_started: "false",
            expected_runtime_outcome: "not_started",
            expected_outcome: "succeeded",
            expected_delivery_outcome: None,
            expected_elapsed_ms: "0",
            expects_failure_summary: false,
        },
        Case {
            name: "delivery-failure",
            attempt: AttemptCase::NativeSucceeded,
            delivery: DeliveryCase::Failed,
            expected_started: "true",
            expected_runtime_outcome: "succeeded",
            expected_outcome: "succeeded",
            expected_delivery_outcome: Some("failed"),
            expected_elapsed_ms: "0",
            expects_failure_summary: false,
        },
        Case {
            name: "missing-attempt-report",
            attempt: AttemptCase::Missing,
            delivery: DeliveryCase::Succeeded,
            expected_started: "false",
            expected_runtime_outcome: "not_started",
            expected_outcome: "succeeded",
            expected_delivery_outcome: None,
            expected_elapsed_ms: "none",
            expects_failure_summary: false,
        },
    ];

    for case in cases {
        let coordinator = QueryCoordinator::new(config());
        let query_id = format!("attempt-facts-{}", case.name);
        let planning = match case.attempt {
            AttemptCase::Missing => register(&coordinator, &query_id),
            _ => coordinator
                .register_with_id_with_attempt_observation(
                    &query_id,
                    "select 1",
                    None,
                    None,
                    RuntimeObservationConfig::bounded(16, 4096),
                )
                .expect("attempt-owned query registers"),
        };
        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
        coordinator.capture_planning(&query_id, Some(accepted_native_diagnostics()), Arc::new(EmptyExec::new(schema)));

        let mut expected_failure_summary = None;
        if !matches!(case.attempt, AttemptCase::Missing) {
            let mut report = match case.attempt {
                AttemptCase::NativeFailed => queue_backoff_attempt_report(&query_id),
                _ => cpu_attempt_report(&query_id),
            };
            report.disposition = match case.attempt {
                AttemptCase::CpuFallback => AttemptDisposition::CpuFallback,
                _ => AttemptDisposition::Native,
            };
            report.outcome = match case.attempt {
                AttemptCase::NativeSucceeded | AttemptCase::CpuFallback => RuntimeTerminalOutcome::Succeeded,
                AttemptCase::NativeFailed => report.outcome,
                AttemptCase::NativeCancelled => RuntimeTerminalOutcome::Cancelled,
                AttemptCase::NativeDropped => RuntimeTerminalOutcome::Dropped,
                AttemptCase::NativePanicked => RuntimeTerminalOutcome::Panicked,
                AttemptCase::Missing => unreachable!("missing report does not build an attempt"),
            };
            let (report, snapshot) = sealed_attempt_report_snapshot(report);
            expected_failure_summary =
                report.primary_failure.as_ref().map(|summary| serde_json::to_string(summary).unwrap());
            coordinator
                .attach_attempt_report(planning.key(), Arc::new(report), Some(snapshot), None)
                .expect("sealed attempt report attaches to its exact coordinator generation");
        }

        let execution = planning.into_execution().expect("planning promotes to execution");
        match case.delivery {
            DeliveryCase::Succeeded => execution.succeed(),
            DeliveryCase::Failed => execution.fail("delivery", captured_error_with_stage("runtime_execution")),
        }

        let payload = coordinator.query_stats(&query_id).expect("stats retained");
        let payload = std::str::from_utf8(&payload).unwrap();
        assert_eq!(
            tsv_value(payload, "execution", "native_runtime_started"),
            case.expected_started,
            "{}: {payload}",
            case.name
        );
        assert_eq!(
            tsv_value(payload, "execution", "runtime_outcome"),
            case.expected_runtime_outcome,
            "{}: {payload}",
            case.name
        );
        assert_eq!(
            tsv_value(payload, "execution", "runtime_failure"),
            expected_failure_summary.as_deref().unwrap_or("none"),
            "{}: {payload}",
            case.name
        );
        assert_eq!(case.expects_failure_summary, expected_failure_summary.is_some(), "{}", case.name);
        assert!(payload.contains(&format!("# outcome={}\n", case.expected_outcome)), "{}: {payload}", case.name);
        assert!(payload.contains(&format!("# elapsed_ms={}\n", case.expected_elapsed_ms)), "{}: {payload}", case.name);
        match case.expected_delivery_outcome {
            Some(delivery_outcome) => assert!(
                payload.contains(&format!("# delivery_outcome={delivery_outcome}\n")),
                "{}: {payload}",
                case.name
            ),
            None => assert!(!payload.contains("# delivery_outcome="), "{}: {payload}", case.name),
        }
    }
}

#[test]
fn coordinator_owned_observation_footer_has_no_attempt_terminal() {
    let coordinator = QueryCoordinator::new(config());
    register(&coordinator, "coordinator-owned").into_execution().expect("planning promotes to execution").succeed();

    let (_, jsonl) = coordinator.artifact_frames("coordinator-owned").expect("runtime artifact retained");
    let footer = jsonl
        .expect("runtime observation JSONL is retained")
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).expect("valid runtime JSONL"))
        .find(|record| record["record_type"] == "footer")
        .expect("runtime observation includes a footer");
    assert!(footer["data"]["terminal"].is_null(), "{footer}");
}

#[test]
fn sealed_queue_backoff_advice_agrees_across_flight_jsonl_and_stats() {
    let coordinator = QueryCoordinator::new(config());
    let (query_id, planning) = coordinator
        .register_new_with_attempt_observation("select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .expect("attempt-owned registration succeeds");
    let sealed_report = queue_backoff_attempt_report(&query_id);
    let terminal = sealed_report.outcome.clone();
    let owner = RuntimeObservationOwner::new(
        RuntimeQueryId::new(query_id.clone()),
        None,
        RuntimeObservationConfig::bounded(16, 4096),
    )
    .with_attempt_report_provider(Arc::new(move || sealed_report.clone()));
    let snapshot = owner.finish_query(Some(terminal)).expect("failed attempt observation seals before handoff");
    let report = snapshot.attempt_report().expect("snapshot carries its sealed attempt report").clone();

    coordinator
        .attach_attempt_report(planning.key(), Arc::new(report), Some(snapshot), None)
        .expect("the matching sealed attempt snapshot is accepted");
    let delivery = tonic::Status::resource_exhausted("attempt failed after sealed report publication");
    planning
        .into_execution()
        .expect("planning promotes to execution")
        .fail("execution", CapturedError::from_status(&delivery));

    let (control, jsonl) = coordinator.artifact_frames(&query_id).expect("retained attempt artifact");
    let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
    let jsonl = jsonl.expect("sealed attempt JSONL is retained");
    let retained_report = std::str::from_utf8(&jsonl)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("retained JSONL record is valid"))
        .find(|record| record["record_type"] == "footer")
        .and_then(|record| record["data"].get("attempt_report").cloned())
        .expect("sealed JSONL footer carries the canonical attempt report");
    let stats_payload = coordinator.query_stats(&query_id).expect("attempt stats retained");
    let stats = std::str::from_utf8(&stats_payload).unwrap();

    let expected_advice = serde_json::json!({
        "recovery": "backoff",
        "prerequisite": "capacity_available",
    });
    assert_eq!(control["attempt"]["retry_advice"], expected_advice);
    assert_eq!(retained_report["retry_advice"], control["attempt"]["retry_advice"]);
    assert!(stats.contains("# attempt_retry_recovery=backoff\n"), "{stats}");
    assert!(stats.contains("# attempt_retry_prerequisite=capacity_available\n"), "{stats}");
}

#[test]
fn attempt_owned_snapshot_rejects_a_nonidentical_completion_report() {
    let coordinator = QueryCoordinator::new(config());
    let (query_id, _planning) = coordinator
        .register_new_with_attempt_observation("select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .expect("attempt-owned registration succeeds");
    let sealed_report = cpu_attempt_report(&query_id);
    let owner =
        RuntimeObservationOwner::new(RuntimeQueryId::new(query_id), None, RuntimeObservationConfig::bounded(16, 4096))
            .with_attempt_report_provider(Arc::new(move || sealed_report.clone()));
    let snapshot = owner
        .finish_query(Some(RuntimeTerminalOutcome::Succeeded))
        .expect("attempt observation seals before its report is handed off");

    let mut mismatched = snapshot.attempt_report().expect("snapshot carries its sealed attempt report").clone();
    mismatched.timestamps.report_ready_ns = None;
    let error = coordinator
        .attach_attempt_report(_planning.key(), Arc::new(mismatched), Some(snapshot), None)
        .expect_err("coordinator rejects a pre-ready report that only shares the attempt identity");
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::InvalidTransition);
}

#[test]
fn unavailable_projection_keeps_canonical_attempt_outcome_separate_from_delivery_failure() {
    let mut bounded = config();
    // The sealed observation JSONL cannot fit, but the bounded unavailable
    // stats control payload still can. This exercises the post-handoff
    // unavailable projection rather than the normal captured-query formatter.
    bounded.per_query_artifact_max_bytes = 512;
    let coordinator = QueryCoordinator::new(bounded);
    attach_cancelled_attempt_then_fail_delivery(
        &coordinator,
        "cancelled-attempt-unavailable",
        RuntimeObservationConfig::bounded(16, 4096),
    );

    let stats = String::from_utf8(coordinator.query_stats("cancelled-attempt-unavailable").unwrap().to_vec()).unwrap();
    assert!(stats.contains("# outcome=cancelled\n"), "{stats}");
    assert!(stats.contains("# delivery_outcome=failed\n"), "{stats}");
    assert!(stats.contains("# incomplete=true\n"), "{stats}");
}

#[test]
fn tombstone_projection_keeps_canonical_attempt_outcome_separate_from_delivery_failure() {
    let mut bounded = config();
    bounded.retained_max_count = 0;
    let coordinator = QueryCoordinator::new(bounded);
    attach_cancelled_attempt_then_fail_delivery(
        &coordinator,
        "cancelled-attempt-tombstone",
        RuntimeObservationConfig::bounded(16, 4096),
    );

    assert!(matches!(
        coordinator.artifact("cancelled-attempt-tombstone"),
        Some(CoordinatorArtifact::Unavailable(reason)) if reason.as_str() == "evicted"
    ));
    let stats = String::from_utf8(coordinator.query_stats("cancelled-attempt-tombstone").unwrap().to_vec()).unwrap();
    assert!(stats.contains("# outcome=cancelled\n"), "{stats}");
    assert!(stats.contains("# delivery_outcome=failed\n"), "{stats}");
    assert!(stats.contains("# unavailable_reason=evicted\n"), "{stats}");
}

fn accepted_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new("test_native_plan", 1));
    NativePlanningDiagnostics {
        input_plan_root: "EmptyExec".to_string(),
        input_plan_shape: "EmptyExec".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        capability_contract: Default::default(),
    }
}

fn captured_error_with_stage(stage: &str) -> CapturedError {
    CapturedError {
        code: "native_execution".to_string(),
        kind: "execution".to_string(),
        status: "transient".to_string(),
        operation: "algeon_datafusion.test".to_string(),
        message: "native execution failed".to_string(),
        facts: vec![("native_datafusion_stage".to_string(), stage.to_string())],
        diagnostic: "native execution failed".to_string(),
        source_chain: None,
        root_cause: None,
    }
}

fn ticket_payload(started_at: Instant, deadline: Instant) -> CoordinatorTicketPayload {
    CoordinatorTicketPayload {
        sql: "select 1".to_string(),
        correlation_id: None,
        prepared_handle: None,
        kind: 0,
        execution_domain: StatementExecutionDomain::Data,
        native_cancellation: ExecutionCancellation::new(),
        attempt: None,
        started_at,
        deadline,
        forest_model_digests: Vec::new(),
    }
}

fn phase(coordinator: &QueryCoordinator, query_id: &str) -> Option<QueryPhase> {
    let state = coordinator.state.lock().unwrap();
    let key = state.current.get(query_id)?;
    state.records.get(key).map(|record| record.phase)
}

fn wait_for_phase_sync(coordinator: &QueryCoordinator, query_id: &str, expected: QueryPhase) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while phase(coordinator, query_id) != Some(expected) {
        assert!(Instant::now() < deadline, "query {query_id} did not reach phase {expected:?}");
        std::thread::yield_now();
    }
}

async fn wait_for_phase(coordinator: &QueryCoordinator, query_id: &str, expected: Option<QueryPhase>) {
    timeout(Duration::from_secs(1), async {
        loop {
            if phase(coordinator, query_id) == expected {
                return;
            }
            yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("query {query_id} did not reach phase {expected:?}"));
}

fn tsv_value<'a>(payload: &'a str, section: &str, column: &str) -> &'a str {
    let marker = format!("[{section}]");
    let mut lines = payload
        .lines()
        .skip_while(|line| *line != marker)
        .skip(1)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let header = lines.next().unwrap().split('\t').collect::<Vec<_>>();
    let row = lines.next().unwrap().split('\t').collect::<Vec<_>>();
    row[header.iter().position(|name| *name == column).unwrap()]
}

#[test]
fn terminal_artifact_composer_reserves_fixed_frames_and_keeps_complete_utf8_lines() {
    let control = Arc::<[u8]>::from(&b"control"[..]);
    let jsonl = Arc::<[u8]>::from(&b"jsonl\n"[..]);
    let unicode_line = "# detail=資料\n";
    let payload = Arc::<[u8]>::from(format!("{QUERY_STATS_HEADER_LINE}{unicode_line}# later=ok\n").into_bytes());
    let cap =
        u64::try_from(control.len() + jsonl.len() + QUERY_STATS_HEADER_LINE.len() + unicode_line.len() - 1).unwrap();

    let composed = compose_terminal_artifact(Arc::clone(&control), Some(Arc::clone(&jsonl)), cap, |_| {
        QueryStatsPayload { bytes: Arc::clone(&payload), incomplete: false }
    })
    .expect("the header fits after fixed frames");

    assert_eq!(composed.bundle.stats.as_ref(), QUERY_STATS_HEADER_LINE.as_bytes());
    assert!(std::str::from_utf8(&composed.bundle.stats).is_ok());
    assert!(composed.stats_incomplete);
    assert_eq!(
        composed.bundle.charge_bytes,
        u64::try_from(
            composed.bundle.stats.len()
                + composed.bundle.runtime_control.len()
                + composed.bundle.runtime_jsonl.as_ref().map_or(0, |value| value.len()),
        )
        .unwrap()
    );
    assert!(composed.bundle.charge_bytes <= cap);

    let exact = u64::try_from(control.len() + QUERY_STATS_HEADER_LINE.len()).unwrap();
    assert!(
        compose_terminal_artifact(Arc::clone(&control), None, exact - 1, |_| QueryStatsPayload {
            bytes: Arc::from(QUERY_STATS_HEADER_LINE.as_bytes()),
            incomplete: false,
        })
        .is_none(),
        "a terminal bundle cannot retain a partial stats header"
    );
    let exact_bundle = compose_terminal_artifact(control, None, exact, |_| QueryStatsPayload {
        bytes: Arc::from(QUERY_STATS_HEADER_LINE.as_bytes()),
        incomplete: false,
    })
    .expect("the exact control plus header boundary fits");
    assert_eq!(exact_bundle.bundle.charge_bytes, exact);
}

#[test]
fn undersized_terminal_cap_installs_stats_free_tombstone_without_leaking_egress() {
    let query_id = "tiny-資料";
    let minimum =
        incomplete_runtime_control(query_id, RetainedUnavailableReason::TooLarge).len() + QUERY_STATS_HEADER_LINE.len();
    let mut tiny = config();
    tiny.per_query_artifact_max_bytes = u64::try_from(minimum - 1).unwrap();
    let coordinator = QueryCoordinator::new(tiny);

    register(&coordinator, query_id).into_execution().unwrap().succeed();

    assert_eq!(phase(&coordinator, query_id), Some(QueryPhase::Tombstone));
    assert!(matches!(
        coordinator.artifact(query_id),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::TooLarge))
    ));
    assert!(coordinator.query_stats(query_id).is_none());
    assert!(coordinator.lease_query_stats(query_id).unwrap().is_none());
    assert_eq!(coordinator.artifact_egress_usage(), (0, 0));
    let state = coordinator.lock_state();
    let key = state.current.get(query_id).unwrap();
    let record = state.records.get(key).unwrap();
    assert!(record.retained.is_none());
    assert!(record.tombstone_stats.is_none());
    assert_eq!(state.tombstone_bytes, 0);
}

#[test]
fn forced_shutdown_with_undersized_cap_installs_stats_free_tombstone() {
    let query_id = "forced-tiny-資料";
    let minimum = incomplete_runtime_control(query_id, RetainedUnavailableReason::ShutdownTimeout).len()
        + QUERY_STATS_HEADER_LINE.len();
    let mut tiny = config();
    tiny.per_query_artifact_max_bytes = u64::try_from(minimum - 1).unwrap();
    let coordinator = QueryCoordinator::new(tiny);
    let execution = register(&coordinator, query_id).into_execution().unwrap();
    let activity = execution.activity_factory()().expect("activity lease");
    let now = Instant::now();
    coordinator.begin_shutdown(now);

    assert_eq!(coordinator.force_shutdown_timeout(now + Duration::from_secs(5)), 1);
    assert_eq!(phase(&coordinator, query_id), Some(QueryPhase::Tombstone));
    assert!(matches!(
        coordinator.artifact(query_id),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::ShutdownTimeout))
    ));
    assert!(coordinator.query_stats(query_id).is_none());
    let state = coordinator.lock_state();
    assert_eq!(state.active_logical_capture_bytes, 0);
    assert_eq!(state.retained_count, 0);
    assert_eq!(state.tombstone_count, 1);
    assert_eq!(state.tombstone_bytes, 0);
    drop(state);
    drop(activity);
    drop(execution);
}

#[test]
fn terminal_waits_for_last_activity_then_finalizes_without_access() {
    let coordinator = QueryCoordinator::new(config());
    let planning = register(&coordinator, "delayed");
    let execution = planning.into_execution().unwrap();
    let activity = execution.activity_factory()().expect("activity factory acquires an opaque native token");

    execution.succeed();
    assert!(matches!(coordinator.artifact("delayed"), Some(CoordinatorArtifact::Active)));

    drop(activity);
    assert_eq!(
        coordinator
            .state
            .lock()
            .unwrap()
            .records
            .values()
            .find(|record| record.key.query_id() == "delayed")
            .map(|record| record.phase),
        Some(QueryPhase::Retained),
        "dropping the last native activity must finalize without retrieval traffic"
    );
    let Some(CoordinatorArtifact::Retained(bundle)) = coordinator.artifact("delayed") else {
        panic!("last activity must trigger finalization");
    };
    assert!(bundle.runtime_jsonl.is_some());
}

#[test]
fn overlapping_finalizers_share_one_claim_and_install_one_complete_artifact() {
    let mut finalization_config = config();
    finalization_config.retained_max_count = 2;
    let coordinator = QueryCoordinator::new(finalization_config);
    let execution = register(&coordinator, "overlapping-finalizers").into_execution().unwrap();
    let gate = FinalizationTestGate::new();
    *coordinator.finalization_gate.lock().unwrap() = Some(Arc::clone(&gate));

    let finalizer = std::thread::spawn(move || execution.succeed());
    gate.entered.wait();

    let concurrent = Arc::clone(&coordinator);
    std::thread::spawn(move || concurrent.finalize_ready()).join().unwrap();
    assert_eq!(phase(&coordinator, "overlapping-finalizers"), Some(QueryPhase::Finalizing));
    assert_eq!(coordinator.finalization_installs.load(Ordering::SeqCst), 0);

    gate.release.wait();
    finalizer.join().unwrap();

    assert_eq!(coordinator.finalization_installs.load(Ordering::SeqCst), 1);
    let Some(CoordinatorArtifact::Retained(bundle)) = coordinator.artifact("overlapping-finalizers") else {
        panic!("the claimed finalizer must retain the complete artifact");
    };
    assert!(bundle.runtime_jsonl.is_some(), "a competing finalizer must not install capture_disabled");
    for line in std::str::from_utf8(bundle.runtime_jsonl.as_ref().unwrap()).unwrap().lines() {
        serde_json::from_str::<serde_json::Value>(line).expect("every retained JSONL record is valid JSON");
    }
    let state = coordinator.state.lock().unwrap();
    let key = state.current.get("overlapping-finalizers").unwrap();
    let record = state.records.get(key).unwrap();
    assert!(record.finalization_claim.is_none());
    assert!(record.unavailable.is_none());
}

#[test]
fn stats_reader_after_cache_close_waits_for_terminal_publication() {
    let coordinator = QueryCoordinator::new(config());
    let execution = register(&coordinator, "post-close-reader").into_execution().unwrap();
    let cache = {
        let state = coordinator.lock_state();
        let key = state.current.get("post-close-reader").unwrap();
        Arc::clone(&state.records.get(key).unwrap().live_stats_cache)
    };
    let (wait_tx, wait_rx) = mpsc::channel();
    *cache.wait_observer.lock().unwrap() = Some(wait_tx);
    let gate = FinalizationTestGate::new();
    *coordinator.finalization_gate.lock().unwrap() = Some(Arc::clone(&gate));

    let finalizer = std::thread::spawn(move || execution.succeed());
    gate.entered.wait();

    let reader_coordinator = Arc::clone(&coordinator);
    let reader = std::thread::spawn(move || {
        reader_coordinator
            .query_stats("post-close-reader")
            .expect("a finalizing query remains addressable until terminal stats are published")
    });
    wait_rx.recv_timeout(Duration::from_secs(2)).expect("the post-close reader must wait for terminal publication");

    gate.release.wait();
    finalizer.join().unwrap();
    let stats = reader.join().unwrap();
    assert!(std::str::from_utf8(&stats).unwrap().contains("# outcome=succeeded"));
    assert_eq!(cache.state.lock().unwrap().waiting_readers, 0);
}

#[test]
fn stale_generation_and_observation_identity_cannot_mutate_replacement() {
    let coordinator = QueryCoordinator::new(config());
    let first = coordinator
        .register_with_id_with_attempt_observation(
            "same-id",
            "select 1",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .expect("first attempt-owned generation registers");
    let first_key = first.key().clone();
    let first_observation = first.observation_id().clone();
    drop(first);
    let second = coordinator
        .register_with_id_with_attempt_observation(
            "same-id",
            "select 2",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .expect("replacement attempt-owned generation registers");

    let error = coordinator.acquire_activity(&first_key, &first_observation).unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::StaleGeneration);
    let error = coordinator
        .attach_attempt_report(&first_key, Arc::new(cpu_attempt_report("same-id")), None, None)
        .expect_err("a delayed report cannot attach to a reused query id");
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::StaleGeneration);
    assert!(
        coordinator
            .state
            .lock()
            .unwrap()
            .records
            .get(second.key())
            .expect("replacement record remains present")
            .attempt_report
            .is_none(),
        "stale report must not become the replacement generation's canonical outcome"
    );
    assert_ne!(first_observation, *second.observation_id());
}

#[test]
fn planning_report_handoff_keeps_attempt_owned_record_live_until_the_sealed_report_arrives() {
    let coordinator = QueryCoordinator::new(config());
    let planning = coordinator
        .register_with_id_with_attempt_observation(
            "planning-handoff-gate",
            "select 1",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .expect("attempt-owned generation registers");
    let key = planning.key().clone();
    let _handoff = planning.attempt_report_delivery_terminal();
    drop(planning);

    let error = coordinator
        .register_with_id_with_attempt_observation(
            "planning-handoff-gate",
            "select 2",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .expect_err("a registered handoff must keep its generation live until the sealed report arrives");
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::DuplicateLiveQueryId);

    let (report, snapshot) = cancelled_attempt_report_snapshot("planning-handoff-gate");
    coordinator
        .attach_attempt_report(&key, Arc::new(report), Some(snapshot), None)
        .expect("the registered handoff accepts its exact sealed report");
    coordinator
        .register_with_id_with_attempt_observation(
            "planning-handoff-gate",
            "select 2",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .expect("the terminal planning generation releases its query id after report finalization");
}

#[test]
fn duplicate_live_query_id_is_rejected_before_registry_or_budget_mutation() {
    let coordinator = QueryCoordinator::new(config());
    let first = coordinator
        .register_with_id("duplicate-live", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap();
    let before = {
        let state = coordinator.state.lock().unwrap();
        (
            state.next_generation,
            state.active_logical_capture_bytes,
            state.ticketed_count,
            state.records.len(),
            state.current.len(),
        )
    };

    let error = coordinator
        .register_with_id(
            "duplicate-live",
            "select 2",
            Some("replacement-correlation".to_string()),
            Some("replacement-trace".to_string()),
            RuntimeObservationConfig::bounded(1, 1),
        )
        .unwrap_err();

    assert_eq!(error.kind(), QueryCoordinatorErrorKind::DuplicateLiveQueryId);
    assert_eq!(error.query_id(), Some("duplicate-live"));
    let state = coordinator.state.lock().unwrap();
    assert_eq!(
        (
            state.next_generation,
            state.active_logical_capture_bytes,
            state.ticketed_count,
            state.records.len(),
            state.current.len(),
        ),
        before
    );
    drop(state);
    drop(first);
}

#[test]
fn planning_render_runs_unlocked_and_stale_generation_install_is_rejected() {
    let mut planning_config = config();
    planning_config.retained_max_count = 2;
    let coordinator = QueryCoordinator::new(planning_config);
    let planning = register(&coordinator, "planning-stale");
    let stale_key = planning.key().clone();
    let replacement = Arc::new(Mutex::new(None));
    let replacement_for_probe = Arc::clone(&replacement);
    let weak = Arc::downgrade(&coordinator);
    let stale_key_for_probe = stale_key.clone();
    let probe = install_coordinator_render_probe(move |operation| {
        assert_eq!(operation, "planning render");
        let coordinator = weak.upgrade().unwrap();
        assert!(coordinator.state.try_lock().is_ok(), "planning rendered under the registry lock");
        coordinator.set_terminal(&stale_key_for_probe, QueryTerminal::Dropped);
        let fresh = register(&coordinator, "planning-stale");
        replacement_for_probe.lock().unwrap().replace(fresh);
    });
    let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));

    coordinator.capture_planning(
        "planning-stale",
        Some(accepted_native_diagnostics()),
        Arc::new(EmptyExec::new(schema)),
    );
    drop(probe);
    drop(planning);

    let replacement = replacement.lock().unwrap().take().expect("probe registered a replacement generation");
    assert_ne!(replacement.key(), &stale_key);
    let payload = String::from_utf8(coordinator.query_stats("planning-stale").unwrap().to_vec()).unwrap();
    assert!(!payload.contains("[physical_plan]"), "stale planning render mutated the replacement: {payload}");
    drop(replacement);
}

#[test]
fn high_churn_reuse_keeps_registry_metadata_bounded_and_rejects_stale_key() {
    let mut churn_config = config();
    churn_config.retained_ttl = Duration::ZERO;
    churn_config.tombstone_ttl = Duration::ZERO;
    let coordinator = QueryCoordinator::new(churn_config);
    let mut stale = None;

    for generation in 1..=256 {
        let planning = register(&coordinator, "00000000-0000-0000-0000-000000000001");
        if generation == 1 {
            stale = Some((planning.key().clone(), planning.observation_id().clone()));
        }
        planning.into_execution().unwrap().succeed();
        coordinator.maintenance_at(Instant::now());
        let state = coordinator.state.lock().unwrap();
        assert!(state.records.is_empty(), "expired churn generation remained resident");
        assert!(state.current.is_empty(), "expired churn generation remained current");
    }

    let replacement = register(&coordinator, "00000000-0000-0000-0000-000000000001");
    let (stale_key, stale_observation) = stale.unwrap();
    let error = coordinator.acquire_activity(&stale_key, &stale_observation).unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::StaleGeneration);
    let state = coordinator.state.lock().unwrap();
    assert_eq!(state.next_generation, 257);
    assert_eq!(state.records.len(), 1);
    assert_eq!(state.current.len(), 1);
    drop(state);
    drop(replacement);
}

#[test]
fn generation_overflow_is_explicit_and_does_not_register() {
    let coordinator = QueryCoordinator::new(config());
    coordinator.state.lock().unwrap().next_generation = u64::MAX;

    let error = coordinator
        .register_with_id("overflow", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap_err();

    assert_eq!(error.kind(), QueryCoordinatorErrorKind::GenerationExhausted);
    let state = coordinator.state.lock().unwrap();
    assert!(state.records.is_empty());
    assert!(state.current.is_empty());
}

#[test]
fn query_wide_observation_identity_is_stable_across_three_native_epochs() {
    let coordinator = QueryCoordinator::new(config());
    let planning = register(&coordinator, "stable-observation");
    let observation_id = planning.observation_id().clone();
    let root = planning.observation_handle().unwrap();
    let execution = planning.into_execution().unwrap();

    assert_eq!(execution.observation_id(), &observation_id);
    for _ in 0..3 {
        let epoch = execution.observation_handle().unwrap();
        assert!(root.is_same_session(&epoch));
    }
    execution.succeed();
}

#[test]
fn dropped_planning_and_execution_leases_terminalize_without_explicit_finish() {
    let mut lease_config = config();
    lease_config.retained_max_count = 2;
    let coordinator = QueryCoordinator::new(lease_config);

    drop(register(&coordinator, "planning-drop"));
    drop(register(&coordinator, "execution-drop").into_execution().unwrap());

    for query_id in ["planning-drop", "execution-drop"] {
        let stats = String::from_utf8(coordinator.query_stats(query_id).unwrap().to_vec()).unwrap();
        assert!(stats.contains("# outcome=dropped"), "{stats}");
        assert!(stats.contains("# sealed=true"), "{stats}");
        assert!(matches!(coordinator.artifact(query_id), Some(CoordinatorArtifact::Retained(_))));
    }
}

#[test]
fn query_stats_projection_and_runtime_control_encoding_run_without_registry_lock() {
    let coordinator = QueryCoordinator::new(config());
    let weak = Arc::downgrade(&coordinator);
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_for_probe = Arc::clone(&observed);
    let _probe = install_coordinator_render_probe(move |operation| {
        let coordinator = weak.upgrade().unwrap();
        assert!(coordinator.state.try_lock().is_ok(), "{operation} ran under the registry lock");
        observed_for_probe.lock().unwrap().push(operation);
    });
    let planning = register(&coordinator, "lock-probe");

    assert!(coordinator.query_stats("lock-probe").is_some());
    assert!(coordinator.artifact_frames("lock-probe").is_some());

    let execution = planning.into_execution().unwrap();
    assert!(coordinator.query_stats("lock-probe").is_some());
    assert!(coordinator.artifact_frames("lock-probe").is_some());
    execution.succeed();

    assert_eq!(
        *observed.lock().unwrap(),
        ["query-stats projection", "runtime control encoding", "runtime control encoding", "finalization projection",]
    );
}

#[test]
fn finalization_projection_panic_installs_bounded_unavailable_without_poisoning_registry() {
    let coordinator = QueryCoordinator::new(config());
    let weak = Arc::downgrade(&coordinator);
    let _probe = install_coordinator_render_probe(move |operation| {
        let coordinator = weak.upgrade().unwrap();
        assert!(coordinator.state.try_lock().is_ok(), "{operation} ran under the registry lock");
        if operation == "finalization projection" {
            panic!("deterministic finalization projection panic");
        }
    });
    let execution = register(&coordinator, "finalization-panic").into_execution().unwrap();

    execution.succeed();

    assert!(matches!(
        coordinator.artifact("finalization-panic"),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::FinalizationPanicked))
    ));
    assert_eq!(coordinator.terminal("finalization-panic"), Some(QueryTerminalOutcome::Succeeded));
    let state = coordinator.lock_state();
    let key = state.current.get("finalization-panic").unwrap();
    let record = state.records.get(key).unwrap();
    assert_eq!(record.phase, QueryPhase::Retained);
    assert_eq!(record.logical_capture_bytes, 0);
    assert_eq!(record.finalization_claim, None);
    let bundle = record.retained.as_ref().unwrap();
    assert_eq!(
        bundle.charge_bytes,
        u64::try_from(
            bundle.stats.len()
                + bundle.runtime_control.len()
                + bundle.runtime_jsonl.as_ref().map_or(0, |value| value.len()),
        )
        .unwrap()
    );
    assert!(bundle.charge_bytes <= config().per_query_artifact_max_bytes);
    assert_eq!(state.active_logical_capture_bytes, 0);
    drop(state);
    let stats = String::from_utf8(coordinator.query_stats("finalization-panic").unwrap().to_vec()).unwrap();
    assert!(stats.contains("# query_id=finalization-panic"), "{stats}");
    assert!(stats.contains("# outcome=succeeded"), "{stats}");
    assert!(stats.contains("# incomplete=true"), "{stats}");
    assert!(stats.contains("# unavailable_reason=finalization_panicked"), "{stats}");
    let (control, runtime) = coordinator.artifact_frames("finalization-panic").unwrap();
    let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
    assert_eq!(control["reason"], "finalization_panicked");
    assert_eq!(control["incomplete"], true);
    assert!(runtime.is_none());

    drop(register(&coordinator, "after-finalization-panic"));
    assert!(coordinator.is_drained());
}

#[test]
fn checked_deadline_overflow_is_immediate_and_never_panics() {
    let now = Instant::now();
    assert_eq!(checked_deadline(now, Duration::MAX), now);
}

#[test]
fn planning_lease_unwind_records_panicked_once_and_does_not_poison_registry() {
    let coordinator = QueryCoordinator::new(config());
    let planning = register(&coordinator, "planning-panic");
    let key = planning.key().clone();

    assert!(
        catch_unwind(AssertUnwindSafe(move || {
            let _planning = planning;
            panic!("planning lease unwind probe");
        }))
        .is_err()
    );
    coordinator.set_terminal(&key, QueryTerminal::Succeeded);

    let state = coordinator.state.lock().expect("lease unwind must not poison registry mutex");
    let record = state.records.get(&key).unwrap();
    assert_eq!(record.terminal_outcome, Some(QueryTerminalOutcome::Panicked));
    assert!(record.terminal.is_none(), "retained records keep only the compact terminal outcome");
    assert_eq!(record.phase, QueryPhase::Retained);
    assert_eq!(record.active_leases, 0);
}

#[test]
fn execution_lease_unwind_records_panicked_once_and_does_not_poison_registry() {
    let coordinator = QueryCoordinator::new(config());
    let execution = register(&coordinator, "execution-panic").into_execution().unwrap();
    let key = execution.key.clone();

    assert!(
        catch_unwind(AssertUnwindSafe(move || {
            let _execution = execution;
            panic!("execution lease unwind probe");
        }))
        .is_err()
    );
    coordinator.set_terminal(&key, QueryTerminal::Succeeded);

    let state = coordinator.state.lock().expect("lease unwind must not poison registry mutex");
    let record = state.records.get(&key).unwrap();
    assert_eq!(record.terminal_outcome, Some(QueryTerminalOutcome::Panicked));
    assert!(record.terminal.is_none(), "retained records keep only the compact terminal outcome");
    assert_eq!(record.phase, QueryPhase::Retained);
    assert_eq!(record.active_leases, 0);
}

#[test]
fn activity_lease_unwind_cannot_overwrite_first_terminal_or_poison_registry() {
    let coordinator = QueryCoordinator::new(config());
    let execution = register(&coordinator, "activity-panic").into_execution().unwrap();
    let key = execution.key.clone();
    let activity = execution.activity_factory()().unwrap();
    execution.succeed();

    assert!(
        catch_unwind(AssertUnwindSafe(move || {
            let _activity = activity;
            panic!("activity lease unwind probe");
        }))
        .is_err()
    );

    let state = coordinator.state.lock().expect("activity unwind must not poison registry mutex");
    let record = state.records.get(&key).unwrap();
    assert_eq!(record.terminal_outcome, Some(QueryTerminalOutcome::Succeeded));
    assert!(record.terminal.is_none(), "retained records keep only the compact terminal outcome");
    assert_eq!(record.phase, QueryPhase::Retained);
    assert_eq!(record.active_activities, 0);
}

#[test]
fn shutdown_race_into_execution_is_structured_cancelled_and_preserves_terminal() {
    let coordinator = QueryCoordinator::new(config());
    let planning = register(&coordinator, "do-put-shutdown-race");

    assert_eq!(coordinator.begin_shutdown(Instant::now()).live_count(), 1);
    let error = planning.into_execution().unwrap_err();

    assert_eq!(error.kind(), QueryCoordinatorErrorKind::Shutdown);
    let adapter = error.adapter_error("algeon_datafusion.server.flight.do_put_statement_update.into_execution");
    assert_eq!(adapter.code(), AdapterErrorCode::Cancelled);
    assert_eq!(adapter.fact_value("query_coordinator_error"), Some("shutdown"));
    assert!(matches!(coordinator.terminal("do-put-shutdown-race"), Some(QueryTerminalOutcome::Cancelled)));
    assert!(matches!(
        coordinator.artifact("do-put-shutdown-race"),
        Some(CoordinatorArtifact::Retained(_) | CoordinatorArtifact::Unavailable(_))
    ));
}

#[tokio::test]
async fn manual_clock_expires_exact_ticket_retention_and_tombstone_boundaries_without_traffic() {
    let coordinator = QueryCoordinator::new(config());
    let clock = TestMaintenanceClock::new(Instant::now());
    let maintenance = QueryCoordinatorMaintenance::start_with_clock(
        Arc::clone(&coordinator),
        Arc::<TestMaintenanceClock>::clone(&clock),
    );
    let base = clock.now();
    let planning = register(&coordinator, "ticket");
    planning.issue_ticket(ticket_payload(base, base + Duration::from_secs(1))).unwrap();

    clock.wait_for_sleepers(1).await;
    clock.advance(Duration::from_secs(1) - Duration::from_nanos(1));
    yield_now().await;
    assert_eq!(phase(&coordinator, "ticket"), Some(QueryPhase::Ticketed));
    clock.advance(Duration::from_nanos(1));
    wait_for_phase(&coordinator, "ticket", Some(QueryPhase::Retained)).await;

    clock.wait_for_sleepers(2).await;
    let retention_deadline = {
        let state = coordinator.state.lock().unwrap();
        let key = state.current.get("ticket").unwrap();
        state.records.get(key).unwrap().retention_deadline.unwrap()
    };
    let until_retention = retention_deadline.duration_since(clock.now());
    clock.advance(until_retention - Duration::from_nanos(1));
    yield_now().await;
    assert_eq!(phase(&coordinator, "ticket"), Some(QueryPhase::Retained));
    clock.advance(Duration::from_nanos(1));
    wait_for_phase(&coordinator, "ticket", Some(QueryPhase::Tombstone)).await;

    clock.wait_for_sleepers(3).await;
    let tombstone_deadline = {
        let state = coordinator.state.lock().unwrap();
        let key = state.current.get("ticket").unwrap();
        state.records.get(key).unwrap().tombstone_deadline.unwrap()
    };
    let until_tombstone = tombstone_deadline.duration_since(clock.now());
    clock.advance(until_tombstone - Duration::from_nanos(1));
    yield_now().await;
    assert_eq!(phase(&coordinator, "ticket"), Some(QueryPhase::Tombstone));
    clock.advance(Duration::from_nanos(1));
    wait_for_phase(&coordinator, "ticket", None).await;

    assert!(clock.sleepers.load(Ordering::SeqCst) >= 3, "maintenance must sleep on every exact deadline");
    let plan = coordinator.begin_shutdown(clock.now());
    assert!(!maintenance.finish(plan).await.unwrap());
}

#[test]
fn maintenance_retained_expiry_batch_enforces_tombstone_cap_immediately() {
    let mut expiry_config = config();
    expiry_config.retained_max_count = 8;
    expiry_config.retained_max_bytes = 1024 * 1024;
    expiry_config.retained_ttl = Duration::from_secs(1);
    expiry_config.tombstone_max_count = 2;
    let coordinator = QueryCoordinator::new(expiry_config);

    for query_id in ["expire-0", "expire-1", "expire-2", "expire-3", "expire-4"] {
        register(&coordinator, query_id).into_execution().unwrap().succeed();
    }
    coordinator.maintenance_at(Instant::now() + Duration::from_secs(2));

    let state = coordinator.state.lock().unwrap();
    assert_eq!(state.retained_count, 0);
    assert_eq!(state.tombstone_count, 2);
    assert_eq!(state.records.len(), 2, "the same maintenance transaction must enforce the tombstone cap");
    let remaining = state.records.keys().map(|key| key.query_id()).collect::<Vec<_>>();
    assert_eq!(remaining, ["expire-3", "expire-4"], "deadline ties use QueryKey ordering deterministically");
}

#[tokio::test]
async fn maintenance_shutdown_joins_after_last_execution_and_activity_release() {
    let coordinator = QueryCoordinator::new(config());
    let clock = TestMaintenanceClock::new(Instant::now());
    let maintenance = QueryCoordinatorMaintenance::start_with_clock(
        Arc::clone(&coordinator),
        Arc::<TestMaintenanceClock>::clone(&clock),
    );
    let planning = register(&coordinator, "graceful-shutdown");
    let execution = planning.into_execution().unwrap();
    let activity = execution.activity_factory()().expect("activity lease");

    let plan = coordinator.begin_shutdown(clock.now());
    assert_eq!(plan.live_count(), 1);
    let error = coordinator
        .register_with_id("closed", "select 2", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::RegistrationCapacity);
    let finish = tokio::spawn(maintenance.finish(plan));

    drop(execution);
    assert!(!coordinator.is_drained(), "native activity keeps finalization live during grace");
    drop(activity);

    assert!(!finish.await.unwrap().unwrap());
    assert!(matches!(coordinator.artifact("graceful-shutdown"), Some(CoordinatorArtifact::Retained(_))));
}

#[tokio::test]
async fn maintenance_rejects_quiescence_first_observed_at_absolute_deadline() {
    let coordinator = QueryCoordinator::new(config());
    let clock = TestMaintenanceClock::new(Instant::now());
    let maintenance = QueryCoordinatorMaintenance::start_with_clock(
        Arc::clone(&coordinator),
        Arc::<TestMaintenanceClock>::clone(&clock),
    );
    let execution = register(&coordinator, "late-graceful-shutdown").into_execution().unwrap();
    let activity = execution.activity_factory()().expect("activity lease");

    let plan = coordinator.begin_shutdown(clock.now());
    let finish = tokio::spawn(maintenance.finish(plan));
    clock.wait_for_sleepers(1).await;
    clock.advance_without_notifying(config().shutdown_grace);
    drop(execution);
    drop(activity);

    assert!(finish.await.unwrap().unwrap(), "quiescence at the deadline cannot report graceful success");
    assert!(matches!(coordinator.artifact("late-graceful-shutdown"), Some(CoordinatorArtifact::Retained(_))));
}

#[tokio::test]
async fn maintenance_shutdown_timeout_retains_bounded_incomplete_diagnostic_and_late_drops_are_noops() {
    let coordinator = QueryCoordinator::new(config());
    let clock = TestMaintenanceClock::new(Instant::now());
    let maintenance = QueryCoordinatorMaintenance::start_with_clock(
        Arc::clone(&coordinator),
        Arc::<TestMaintenanceClock>::clone(&clock),
    );
    let planning = coordinator
        .register_with_id_with_attempt_observation(
            "forced-shutdown",
            "select secret_literal",
            Some("correlation".to_string()),
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .unwrap();
    let (report, snapshot) = cancelled_attempt_report_snapshot("forced-shutdown");
    coordinator
        .attach_attempt_report(planning.key(), Arc::new(report), Some(snapshot), None)
        .expect("sealed cancelled attempt report attaches before shutdown delivery times out");
    let execution = planning.into_execution().unwrap();
    execution.hot_counters().record_output(3, 64);
    let activity = execution.activity_factory()().expect("activity lease");

    let plan = coordinator.begin_shutdown(clock.now());
    assert_eq!(plan.live_count(), 1);
    let finish = tokio::spawn(maintenance.finish(plan));
    clock.wait_for_sleepers(1).await;
    clock.advance(Duration::from_secs(6));

    assert!(finish.await.unwrap().unwrap());
    assert!(matches!(
        coordinator.artifact("forced-shutdown"),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::ShutdownTimeout))
    ));
    let stats = String::from_utf8(coordinator.query_stats("forced-shutdown").unwrap().to_vec()).unwrap();
    assert!(stats.contains("# outcome=cancelled\n"), "{stats}");
    assert!(stats.contains("# delivery_outcome=dropped\n"), "{stats}");
    assert!(stats.contains("# incomplete=true"), "{stats}");
    assert!(stats.contains("# unavailable_reason=shutdown_timeout"), "{stats}");
    assert!(stats.contains("# delivery_output_batches=1"), "{stats}");
    assert!(!stats.contains("secret_literal"), "{stats}");
    let (control, runtime_jsonl) = coordinator.artifact_frames("forced-shutdown").unwrap();
    let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
    assert_eq!(control["reason"], "shutdown_timeout");
    assert_eq!(control["incomplete"], true);
    assert!(runtime_jsonl.is_none());
    let before = coordinator.query_stats("forced-shutdown").unwrap();
    {
        let state = coordinator.state.lock().unwrap();
        let record = state.records.values().find(|record| record.key.query_id() == "forced-shutdown").unwrap();
        assert!(record.captured.is_none());
        assert!(record.observation_owner.is_none());
        assert_eq!(record.active_leases, 0);
        assert_eq!(record.active_activities, 0);
        let bundle = record.retained.as_ref().expect("forced shutdown retains its bounded fallback bundle");
        assert_eq!(
            bundle.charge_bytes,
            u64::try_from(
                bundle.stats.len()
                    + bundle.runtime_control.len()
                    + bundle.runtime_jsonl.as_ref().map_or(0, |value| value.len()),
            )
            .unwrap()
        );
        assert!(bundle.charge_bytes <= config().per_query_artifact_max_bytes);
    }

    drop(activity);
    drop(execution);
    assert_eq!(coordinator.query_stats("forced-shutdown").unwrap().as_ref(), before.as_ref());
}

#[test]
fn registration_cap_rejects_before_work() {
    let mut limited = config();
    limited.max_live_queries = 1;
    let coordinator = QueryCoordinator::new(limited);
    let _first = register(&coordinator, "first");
    let error = coordinator
        .register_with_id("second", "select 2", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::RegistrationCapacity);
}

#[test]
fn ticket_and_logical_capture_byte_caps_reject_before_expensive_work() {
    let now = Instant::now();
    let mut ticket_limited = config();
    ticket_limited.max_live_queries = 3;
    ticket_limited.max_ticketed_queries = 1;
    let coordinator = QueryCoordinator::new(ticket_limited);
    register(&coordinator, "first-ticket").issue_ticket(ticket_payload(now, now + Duration::from_secs(30))).unwrap();
    let error = register(&coordinator, "second-ticket")
        .issue_ticket(ticket_payload(now, now + Duration::from_secs(30)))
        .unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::TicketCapacity);
    let stats = String::from_utf8(coordinator.query_stats("second-ticket").unwrap().to_vec()).unwrap();
    assert!(stats.contains("code=service_overloaded"), "{stats}");
    assert!(stats.contains("kind=resource_exhausted"), "{stats}");
    assert!(stats.contains("status=temporary"), "{stats}");
    assert!(stats.contains("operation=algeon_datafusion.server.query_coordinator.issue_ticket"), "{stats}");

    let mut capture_limited = config();
    capture_limited.max_live_queries = 3;
    capture_limited.per_query_capture_max_bytes = 1;
    capture_limited.per_query_artifact_max_bytes = 1;
    capture_limited.max_active_logical_capture_bytes = 8194;
    let coordinator = QueryCoordinator::new(capture_limited);
    let _first = coordinator
        .register_with_id("first-capture", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap();
    let error = coordinator
        .register_with_id("second-capture", "select 2", None, None, RuntimeObservationConfig::bounded(16, 1))
        .unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::LogicalCaptureBytes);
}

#[test]
fn hot_counter_updates_do_not_require_registry_lock() {
    let coordinator = QueryCoordinator::new(config());
    let planning = register(&coordinator, "hot");
    let hot = planning.hot_counters();
    let state = coordinator.state.lock().unwrap();
    hot.record_output(3, 64);
    assert_eq!(hot.snapshot().output_batches, 1);
    drop(state);
}

#[test]
fn artifact_frames_preserve_control_schema_escape_ids_and_never_expose_sql() {
    let coordinator = QueryCoordinator::new(config());
    let query_id = "quoted-\"query\nline";
    let planning = coordinator
        .register_with_id(
            query_id,
            "select 'private-sql-literal'",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .unwrap();
    let (pending, jsonl) = coordinator.artifact_frames(query_id).unwrap();
    assert!(jsonl.is_none());
    let pending: serde_json::Value = serde_json::from_slice(&pending).unwrap();
    assert_eq!(pending["schema"], RUNTIME_ARTIFACT_CONTROL_SCHEMA);
    assert_eq!(pending["version"], 1);
    assert_eq!(pending["query_id"], query_id);
    assert_eq!(pending["state"], "pending");
    assert_eq!(pending["retryable"], true);

    let execution = planning.into_execution().unwrap();
    let (active, jsonl) = coordinator.artifact_frames(query_id).unwrap();
    assert!(jsonl.is_none());
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&active).unwrap()["state"], "active");
    execution.succeed();

    let (sealed, jsonl) = coordinator.artifact_frames(query_id).unwrap();
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&sealed).unwrap()["state"], "sealed");
    let jsonl = jsonl.expect("terminal artifact has one complete JSONL frame");
    let jsonl = std::str::from_utf8(&jsonl).unwrap();
    assert!(jsonl.contains("\"record_type\":\"manifest\""));
    assert!(jsonl.contains("\"record_type\":\"footer\""));
    assert!(!jsonl.contains("private-sql-literal"));
}

#[test]
fn retention_charges_exact_serialized_bytes_and_evicts_oldest_terminal_only() {
    let mut limited = config();
    limited.retained_max_count = 1;
    let coordinator = QueryCoordinator::new(limited);
    let active = register(&coordinator, "active").into_execution().unwrap();

    for query_id in ["older", "newer"] {
        register(&coordinator, query_id).into_execution().unwrap().succeed();
    }

    assert!(matches!(coordinator.artifact("active"), Some(CoordinatorArtifact::Active)));
    assert!(matches!(
        coordinator.artifact("older"),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::Evicted))
    ));
    let Some(CoordinatorArtifact::Retained(bundle)) = coordinator.artifact("newer") else {
        panic!("newest terminal query must be retained");
    };
    let expected = u64::try_from(bundle.stats.len() + bundle.runtime_control.len()).unwrap()
        + u64::try_from(bundle.runtime_jsonl.as_ref().unwrap().len()).unwrap();
    assert_eq!(bundle.charge_bytes, expected);
    let state = coordinator.state.lock().unwrap();
    assert_eq!(state.retained_count, 1);
    assert_eq!(state.retained_bytes, expected);
    drop(state);
    drop(active);
}

#[test]
fn structured_failure_stats_keep_log_safe_facts_and_first_terminal() {
    let coordinator = QueryCoordinator::new(config());
    let mut execution = register(&coordinator, "failed-stats").into_execution().unwrap();
    execution.hot_counters().record_output(7, 128);
    execution.finish_failed(
        "collect",
        CapturedError {
            code: "native_execution".to_string(),
            kind: "execution".to_string(),
            status: "transient".to_string(),
            operation: "test.operation".to_string(),
            message: "safe failure".to_string(),
            facts: vec![("reason".to_string(), "reservation_pressure".to_string())],
            diagnostic: "safe diagnostic".to_string(),
            source_chain: None,
            root_cause: None,
        },
    );
    execution.finish_success();

    let payload = coordinator.query_stats("failed-stats").unwrap();
    let payload = std::str::from_utf8(&payload).unwrap();
    assert!(payload.contains("# outcome=failed\n"), "{payload}");
    assert!(payload.contains("# sealed=true\n"), "{payload}");
    assert!(payload.contains("fact.reason=reservation_pressure\n"), "{payload}");
    assert!(payload.contains("operation=test.operation\n"), "{payload}");
}

#[test]
fn terminal_runtime_footer_preserves_structured_error_identity_after_capture_truncation() {
    let mut limited = config();
    limited.per_query_capture_max_bytes = 1;
    let coordinator = QueryCoordinator::new(limited);
    let planning = coordinator
        .register_with_id_with_attempt_observation(
            "terminal-error-identity",
            "select 1",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .expect("attempt-owned query registers");
    let mut report = cpu_attempt_report("terminal-error-identity");
    let failure = AttemptFailureSummary::from_native_error(&QueryError::new(
        QueryErrorCode::CudfResourceExhausted,
        "algeon_datafusion.test.terminal_error_identity",
        "native execution failed",
    ));
    report.outcome = RuntimeTerminalOutcome::failed(&failure);
    report.primary_failure = Some(failure);
    let (report, snapshot) = sealed_attempt_report_snapshot(report);
    coordinator
        .attach_attempt_report(planning.key(), Arc::new(report), Some(snapshot), None)
        .expect("sealed failed attempt report attaches before delivery terminalization");
    planning
        .into_execution()
        .expect("planning promotes to execution")
        .fail("delivery", CapturedError::from_status(&tonic::Status::internal("Flight delivery failed")));

    let jsonl = {
        let state = coordinator.lock_state();
        let key = state.current.get("terminal-error-identity").expect("query remains addressable");
        let record = state.records.get(key).expect("terminal record remains retained");
        Arc::clone(
            record
                .retained
                .as_ref()
                .and_then(|bundle| bundle.runtime_jsonl.as_ref())
                .expect("capture-truncated records retain their runtime JSONL internally"),
        )
    };
    let footer = std::str::from_utf8(&jsonl)
        .expect("runtime JSONL is UTF-8")
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("runtime JSONL is valid"))
        .find(|record| record["record_type"] == "footer")
        .expect("sealed runtime JSONL has one footer");

    assert_eq!(footer["data"]["terminal"]["kind"], "failed");
    assert_eq!(footer["data"]["terminal"]["error_code"], "cudf_resource_exhausted");
    assert_eq!(footer["data"]["terminal"]["error_status"], "temporary");
}

#[test]
fn over_budget_terminal_becomes_explicit_unavailable_tombstone() {
    let mut limited = config();
    limited.retained_max_bytes = 1;
    let coordinator = QueryCoordinator::new(limited);
    register(&coordinator, "too-large").into_execution().unwrap().succeed();

    assert!(matches!(
        coordinator.artifact("too-large"),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::TooLarge))
    ));
    let (control, jsonl) = coordinator.artifact_frames("too-large").unwrap();
    assert!(jsonl.is_none());
    let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
    assert_eq!(control["state"], "unavailable");
    assert_eq!(control["reason"], "global_budget");
    assert_eq!(control["incomplete"], true);
    let stats = String::from_utf8(coordinator.query_stats("too-large").unwrap().to_vec()).unwrap();
    assert!(stats.contains("# incomplete=true"), "{stats}");
    assert!(stats.contains("# unavailable_reason=global_budget"), "{stats}");
    let state = coordinator.state.lock().unwrap();
    assert_eq!(state.tombstone_count, 1);
    assert_eq!(state.tombstone_bytes, stats.len() as u64);
}

#[test]
fn adversarial_capture_inputs_and_transient_artifacts_stay_bounded_with_structured_loss() {
    let mut bounded = config();
    bounded.max_active_logical_capture_bytes = 64 * 1024;
    bounded.per_query_capture_max_bytes = 1024;
    bounded.per_query_artifact_max_bytes = 8 * 1024;
    bounded.retained_max_bytes = 8 * 1024;
    let coordinator = QueryCoordinator::new(bounded.clone());
    let planning = coordinator
        .register_with_id(
            "adversarial-bounded",
            "s".repeat(2 * 1024 * 1024),
            Some("c".repeat(2 * 1024 * 1024)),
            None,
            RuntimeObservationConfig::bounded(8, 1024),
        )
        .unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
    let plan = UnionExec::try_new(
        (0..4096).map(|_| Arc::new(EmptyExec::new(Arc::clone(&schema))) as Arc<dyn ExecutionPlan>).collect(),
    )
    .unwrap();
    let mut diagnostics = accepted_native_diagnostics();
    diagnostics.input_plan_shape = "diagnostic".repeat(256 * 1024);
    coordinator.capture_planning("adversarial-bounded", Some(diagnostics), plan);

    let huge = "error".repeat(256 * 1024);
    planning.into_execution().unwrap().fail(
        "runtime_execution",
        CapturedError {
            code: huge.clone(),
            kind: huge.clone(),
            status: huge.clone(),
            operation: huge.clone(),
            message: huge.clone(),
            facts: (0..4096).map(|index| (format!("key-{index}"), "v".repeat(1024))).collect(),
            diagnostic: huge.clone(),
            source_chain: Some(huge.clone()),
            root_cause: Some(huge),
        },
    );

    let stats = coordinator.query_stats("adversarial-bounded").expect("bounded terminal stats");
    assert!(stats.len() <= bounded.per_query_artifact_max_bytes as usize);
    let stats = String::from_utf8(stats.to_vec()).unwrap();
    assert!(stats.contains("# incomplete=true"), "{stats}");
    assert!(
        stats.contains("capture_loss.retained_byte_limit") || stats.contains("# unavailable_reason=global_budget"),
        "{stats}"
    );
    let state = coordinator.state.lock().unwrap();
    assert_eq!(state.active_logical_capture_bytes, 0, "terminal install releases the reservation once");
    assert!(state.retained_bytes <= bounded.retained_max_bytes);
}

#[test]
fn different_key_waiter_sleeps_until_current_render_finishes_then_retries_once() {
    let cache = Arc::new(LiveStatsCache::new());
    let first_key =
        (1, QueryHotCounterSnapshot { output_batches: 1, output_rows: 1, output_bytes: 1, overflow: false });
    let second_key =
        (2, QueryHotCounterSnapshot { output_batches: 2, output_rows: 2, output_bytes: 2, overflow: false });
    let LiveStatsRead::Render(first_renderer) = cache.begin_read(first_key) else {
        panic!("first key must elect the renderer");
    };
    let (wait_tx, wait_rx) = mpsc::channel();
    *cache.wait_observer.lock().unwrap() = Some(wait_tx);
    let LiveStatsRead::Wait(second_waiter) = cache.begin_read(second_key) else {
        panic!("different key must wait behind the active renderer");
    };
    let waiter = std::thread::spawn(move || second_waiter.wait());
    wait_rx.recv_timeout(Duration::from_secs(2)).expect("different-key waiter must enter the condvar wait");
    {
        let state = cache.state.lock().unwrap();
        assert_eq!(state.active_renderers, 1);
        assert_eq!(state.waiting_readers, 1);
    }

    first_renderer.complete(Arc::from(&b"first"[..]));
    assert!(matches!(waiter.join().unwrap(), LiveStatsWaitResult::Retry));
    let LiveStatsRead::Render(second_renderer) = cache.begin_read(second_key) else {
        panic!("second key must render once after the first flight completes");
    };
    let second = second_renderer.complete(Arc::from(&b"second"[..])).unwrap();
    assert_eq!(second.as_ref(), b"second");
    let state = cache.state.lock().unwrap();
    assert_eq!(state.active_renderers, 0);
    assert_eq!(state.waiting_readers, 0);
    assert_eq!(state.completed_key, Some(second_key));
}

#[test]
fn renderer_drop_wakes_all_waiters_with_one_shared_error_and_exact_accounting() {
    let cache = Arc::new(LiveStatsCache::new());
    let key = (1, QueryHotCounterSnapshot { output_batches: 1, output_rows: 1, output_bytes: 1, overflow: false });
    let LiveStatsRead::Render(renderer) = cache.begin_read(key) else {
        panic!("first reader must render");
    };
    let (wait_tx, wait_rx) = mpsc::channel();
    *cache.wait_observer.lock().unwrap() = Some(wait_tx);
    let waiters = (0..8)
        .map(|_| {
            let LiveStatsRead::Wait(waiter) = cache.begin_read(key) else {
                panic!("same-key reader must join the flight");
            };
            std::thread::spawn(move || waiter.wait())
        })
        .collect::<Vec<_>>();
    for _ in 0..waiters.len() {
        wait_rx.recv_timeout(Duration::from_secs(2)).expect("every waiter must block behind the renderer");
    }

    drop(renderer);
    assert!(waiters.into_iter().all(|waiter| matches!(waiter.join().unwrap(), LiveStatsWaitResult::Failed)));
    let state = cache.state.lock().unwrap();
    assert_eq!(state.active_renderers, 0);
    assert_eq!(state.waiting_readers, 0);
    assert_eq!(state.rendering_key, None);
    assert_eq!(state.completed_key, Some(key));
    assert!(state.render_failed);
}

#[test]
fn live_stats_waiter_drop_underflow_during_unwind_closes_the_cache_without_aborting() {
    let cache = Arc::new(LiveStatsCache::new());
    let waiter = LiveStatsWaiter {
        cache: Arc::clone(&cache),
        key: (1, QueryHotCounterSnapshot { output_batches: 1, output_rows: 1, output_bytes: 1, overflow: false }),
        completed: false,
    };

    let unwind = catch_unwind(AssertUnwindSafe(|| {
        let _waiter = waiter;
        panic!("live stats waiter unwind probe");
    }));

    assert!(unwind.is_err(), "the original unwind must survive the waiter drop");
    let state = cache.state.lock().unwrap();
    assert!(!state.open);
    assert!(state.abandoned);
    assert_eq!(state.waiting_readers, 0);
    assert!(state.stats.is_none());
    assert!(state.rendering_key.is_none());
}

#[test]
fn live_stats_renderer_drop_underflow_during_unwind_closes_the_cache_without_aborting() {
    let cache = Arc::new(LiveStatsCache::new());
    let key = (1, QueryHotCounterSnapshot { output_batches: 1, output_rows: 1, output_bytes: 1, overflow: false });
    let LiveStatsRead::Render(renderer) = cache.begin_read(key) else {
        panic!("first reader must elect the renderer");
    };
    cache.state.lock().unwrap().active_renderers = 0;

    let unwind = catch_unwind(AssertUnwindSafe(|| {
        let _renderer = renderer;
        panic!("live stats renderer unwind probe");
    }));

    assert!(unwind.is_err(), "the original unwind must survive the renderer drop");
    let state = cache.state.lock().unwrap();
    assert!(!state.open);
    assert!(state.abandoned);
    assert_eq!(state.active_renderers, 0);
    assert!(state.stats.is_none());
    assert!(state.rendering_key.is_none());
}

#[test]
fn artifact_egress_drop_underflow_during_unwind_quarantines_the_limiter_without_aborting() {
    let limiter = ArtifactEgressLimiter::new(1, 1024);
    let lease = limiter.try_acquire(512).unwrap();
    limiter.state.lock().unwrap().streams = 0;

    let unwind = catch_unwind(AssertUnwindSafe(|| {
        let _lease = lease;
        panic!("artifact egress unwind probe");
    }));

    assert!(unwind.is_err(), "the original unwind must survive the egress lease drop");
    let state = limiter.state.lock().unwrap();
    assert!(state.closed);
    assert_eq!((state.streams, state.bytes), (1, 1024));
    drop(state);
    let error = limiter.try_acquire(1).unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::ArtifactEgressCapacity);
}

#[test]
fn concurrent_repeated_live_stats_reads_share_one_bounded_snapshot() {
    let coordinator = QueryCoordinator::new(config());
    let execution = register(&coordinator, "live-stats-cache").into_execution().unwrap();
    let cache = {
        let state = coordinator.lock_state();
        let key = state.current.get("live-stats-cache").unwrap();
        Arc::clone(&state.records.get(key).unwrap().live_stats_cache)
    };
    let start = Arc::new(Barrier::new(33));
    let render_entered = Arc::new(Barrier::new(2));
    let render_release = Arc::new(Barrier::new(2));
    let render_count = Arc::new(AtomicUsize::new(0));
    let readers = (0..32)
        .map(|_| {
            let coordinator = Arc::clone(&coordinator);
            let start = Arc::clone(&start);
            let render_entered = Arc::clone(&render_entered);
            let render_release = Arc::clone(&render_release);
            let render_count = Arc::clone(&render_count);
            std::thread::spawn(move || {
                let _probe = install_coordinator_render_probe(move |operation| {
                    if operation == "query-stats projection" && render_count.fetch_add(1, Ordering::SeqCst) == 0 {
                        render_entered.wait();
                        render_release.wait();
                    }
                });
                start.wait();
                coordinator.query_stats("live-stats-cache").unwrap()
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    render_entered.wait();
    let wait_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if cache.state.lock().unwrap().waiting_readers == 31 {
            break;
        }
        assert!(Instant::now() < wait_deadline, "all concurrent cache misses must join the single flight");
        std::thread::yield_now();
    }
    render_release.wait();
    let snapshots = readers.into_iter().map(|reader| reader.join().unwrap()).collect::<Vec<_>>();
    assert_eq!(render_count.load(Ordering::SeqCst), 1, "only the elected builder may render the artifact");
    assert!(snapshots.iter().all(|snapshot| Arc::ptr_eq(snapshot, &snapshots[0])));
    assert!(snapshots[0].len() <= config().per_query_artifact_max_bytes as usize);
    assert_eq!(cache.state.lock().unwrap().waiting_readers, 0);
    execution.succeed();
}

#[test]
fn live_stats_egress_reserves_bound_before_render_and_reconciles_to_actual_bytes() {
    let bounded = config();
    let coordinator = QueryCoordinator::new(bounded.clone());
    let execution = register(&coordinator, "live-egress-preflight").into_execution().unwrap();
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let request_coordinator = Arc::clone(&coordinator);
    let request_entered = Arc::clone(&entered);
    let request_release = Arc::clone(&release);
    let request = std::thread::spawn(move || {
        let _probe = install_coordinator_render_probe(move |operation| {
            if operation == "query-stats projection" {
                request_entered.wait();
                request_release.wait();
            }
        });
        request_coordinator.lease_query_stats("live-egress-preflight").unwrap().unwrap()
    });
    entered.wait();
    {
        let state = coordinator.artifact_egress.state.lock().unwrap();
        assert_eq!(state.streams, 1);
        assert_eq!(state.bytes, bounded.per_query_artifact_max_bytes);
        assert_eq!(state.peak_streams, 1);
        assert_eq!(state.peak_bytes, bounded.per_query_artifact_max_bytes);
    }
    release.wait();
    let mut response = request.join().unwrap();
    let actual_bytes = u64::try_from(response.next_frame().unwrap().len()).unwrap();
    {
        let state = coordinator.artifact_egress.state.lock().unwrap();
        assert_eq!(state.streams, 1);
        assert_eq!(state.bytes, actual_bytes);
        assert_eq!(state.peak_bytes, bounded.per_query_artifact_max_bytes);
    }
    drop(response);
    let state = coordinator.artifact_egress.state.lock().unwrap();
    assert_eq!((state.streams, state.bytes), (0, 0));
    drop(state);
    execution.succeed();
}

#[test]
fn live_stats_render_panic_releases_provisional_egress_and_publishes_shared_error() {
    let coordinator = QueryCoordinator::new(config());
    let execution = register(&coordinator, "live-render-panic").into_execution().unwrap();
    let request_coordinator = Arc::clone(&coordinator);
    let panicked = std::thread::spawn(move || {
        catch_unwind(AssertUnwindSafe(|| {
            let _probe = install_coordinator_render_probe(|operation| {
                if operation == "query-stats projection" {
                    panic!("injected stats renderer panic");
                }
            });
            request_coordinator.lease_query_stats("live-render-panic")
        }))
    })
    .join()
    .unwrap();
    assert!(panicked.is_err());
    {
        let state = coordinator.artifact_egress.state.lock().unwrap();
        assert_eq!((state.streams, state.bytes), (0, 0));
    }
    let error = coordinator.lease_query_stats("live-render-panic").unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::ArtifactRenderFailed);
    let state = coordinator.artifact_egress.state.lock().unwrap();
    assert_eq!((state.streams, state.bytes), (0, 0));
    drop(state);
    execution.succeed();
}

#[test]
fn finalization_closes_and_drains_in_flight_live_stats_without_late_repopulation() {
    let coordinator = QueryCoordinator::new(config());
    let execution = register(&coordinator, "live-finalize-race").into_execution().unwrap();
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let reader_coordinator = Arc::clone(&coordinator);
    let reader_entered = Arc::clone(&entered);
    let reader_release = Arc::clone(&release);
    let reader = std::thread::spawn(move || {
        let _probe = install_coordinator_render_probe(move |operation| {
            if operation == "query-stats projection" {
                reader_entered.wait();
                reader_release.wait();
            }
        });
        reader_coordinator.query_stats("live-finalize-race").expect("reader receives terminal stats")
    });
    entered.wait();

    let finalizer = std::thread::spawn(move || execution.succeed());
    wait_for_phase_sync(&coordinator, "live-finalize-race", QueryPhase::Finalizing);
    release.wait();
    let raced_stats = reader.join().unwrap();
    finalizer.join().unwrap();

    let retained = coordinator.query_stats("live-finalize-race").unwrap();
    assert!(Arc::ptr_eq(&raced_stats, &retained));
    let state = coordinator.lock_state();
    assert_eq!(state.active_logical_capture_bytes, 0);
    let key = state.current.get("live-finalize-race").unwrap();
    let cache = &state.records.get(key).unwrap().live_stats_cache;
    let cache = cache.state.lock().unwrap();
    assert!(!cache.open);
    assert_eq!(cache.active_renderers, 0);
    assert_eq!(cache.waiting_readers, 0);
    assert!(cache.stats.is_none(), "live projection cache must be drained before active charge release");
    assert!(cache.terminal_stats.as_ref().is_some_and(|stats| Arc::ptr_eq(stats, &retained)));
}

#[test]
fn artifact_egress_lease_bounds_concurrency_and_releases_on_response_drop() {
    let mut limited = config();
    limited.max_live_queries = 1;
    let coordinator = QueryCoordinator::new(limited);
    register(&coordinator, "egress").into_execution().unwrap().succeed();

    let mut first = coordinator.lease_runtime_artifact("egress").unwrap().unwrap();
    let first_frame = first.next_frame().expect("control frame");
    assert!(!first_frame.is_empty());
    let error = coordinator.lease_runtime_artifact("egress").unwrap_err();
    assert_eq!(error.kind(), QueryCoordinatorErrorKind::ArtifactEgressCapacity);
    drop(first);
    {
        let state = coordinator.artifact_egress.state.lock().unwrap();
        assert_eq!(state.streams, 0);
        assert_eq!(state.bytes, 0);
    }
    let replacement = coordinator.lease_runtime_artifact("egress").unwrap().unwrap();
    drop(replacement);
    let state = coordinator.artifact_egress.state.lock().unwrap();
    assert_eq!(state.streams, 0);
    assert_eq!(state.bytes, 0);
}

#[test]
fn retained_unavailable_runtime_artifact_v2_keeps_attempt_projection_when_capture_truncates() {
    let coordinator = QueryCoordinator::new(config());
    let planning = coordinator
        .register_with_id_with_attempt_observation(
            "bounded-capture",
            "x".repeat(128 * 1024),
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .unwrap();
    let (report, snapshot) = cancelled_attempt_report_snapshot("bounded-capture");
    coordinator
        .attach_attempt_report(planning.key(), Arc::new(report.clone()), Some(snapshot), None)
        .expect("capture-truncated query accepts its canonical sealed report");
    planning.into_execution().unwrap().succeed();

    {
        let state = coordinator.lock_state();
        let key = state.current.get("bounded-capture").unwrap();
        let record = state.records.get(key).unwrap();
        assert_eq!(record.phase, QueryPhase::Retained);
        assert_eq!(record.unavailable, Some(RetainedUnavailableReason::CaptureTruncated));
        assert!(
            record.retained.as_ref().unwrap().runtime_jsonl.is_some(),
            "the regression requires retained JSONL behind an unavailable control"
        );
    }

    let (control, jsonl) = coordinator.artifact_frames("bounded-capture").unwrap();
    let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
    assert_eq!(control["schema"], RUNTIME_ARTIFACT_CONTROL_SCHEMA_V2);
    assert_eq!(control["version"], 2);
    assert_eq!(control["state"], "unavailable");
    assert_eq!(control["reason"], "capture_budget");
    assert_eq!(control["attempt"]["attempt_id"], report.attempt_id.to_string());
    assert_eq!(control["attempt"]["retry_advice"]["recovery"], "do_not_retry");
    assert!(jsonl.is_none(), "unavailable artifacts must not expose a JSONL frame");

    let mut response = coordinator.lease_runtime_artifact("bounded-capture").unwrap().unwrap();
    assert!(response.next_frame().is_some(), "unavailable artifact has one control frame");
    assert!(response.next_frame().is_none(), "unavailable artifact must have exactly one frame");
    drop(response);
    assert_eq!(coordinator.artifact_egress_usage(), (0, 0));
}

#[test]
fn planning_capture_projects_and_drops_the_full_execution_plan() {
    let coordinator = QueryCoordinator::new(config());
    let _planning = register(&coordinator, "plan-release");
    let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
    let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(schema));
    coordinator.capture_planning("plan-release", Some(accepted_native_diagnostics()), Arc::clone(&plan));
    assert_eq!(Arc::strong_count(&plan), 1, "capture must retain bounded projections, not the full plan DAG");
}

#[test]
fn hot_counter_saturation_is_truthfully_exported_as_incomplete() {
    let coordinator = QueryCoordinator::new(config());
    let execution = register(&coordinator, "hot-overflow").into_execution().unwrap();
    let hot = execution.hot_counters();
    hot.output_rows.store(usize::MAX, Ordering::Relaxed);
    hot.record_output(1, 1);
    execution.succeed();

    let stats = String::from_utf8(coordinator.query_stats("hot-overflow").unwrap().to_vec()).unwrap();
    assert!(stats.contains("capture_loss.counter_overflow"), "{stats}");
    assert!(stats.contains("# incomplete=true"), "{stats}");
    let (control, _) = coordinator.artifact_frames("hot-overflow").unwrap();
    let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
    assert_eq!(control["reason"], "counter_overflow");
    assert_eq!(control["incomplete"], true);
}

#[test]
fn full_stats_projection_keeps_cpu_baseline_failures_not_started() {
    for (query_id, stage) in [
        ("context-failure", "runtime_context"),
        // Error staging alone cannot turn a DataFusion-only final plan into native runtime.
        ("execution-failure", "runtime_execution"),
    ] {
        let coordinator = QueryCoordinator::new(config());
        let planning = register(&coordinator, query_id);
        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
        coordinator.capture_planning(
            query_id,
            Some(accepted_native_diagnostics()),
            Arc::new(EmptyExec::new(schema)) as Arc<dyn ExecutionPlan>,
        );
        planning.into_execution().unwrap().fail("stream", captured_error_with_stage(stage));

        let payload = coordinator.query_stats(query_id).unwrap();
        let payload = std::str::from_utf8(&payload).unwrap();
        assert_eq!(tsv_value(payload, "execution", "native_runtime_started"), "false", "{payload}");
        assert_eq!(tsv_value(payload, "execution", "runtime_outcome"), "not_started", "{payload}");
    }
}

#[test]
fn status_terminal_preserves_adapter_source_chain() {
    let coordinator = QueryCoordinator::new(config());
    let mut execution = register(&coordinator, "status-source").into_execution().unwrap();
    let error = AdapterError::new(
        AdapterErrorCode::Internal,
        "algeon_datafusion.server.flight.test.encode",
        "failed to encode response",
    )
    .with_source(std::io::Error::other("socket closed"));
    let status = adapter_error_to_status(&error);
    execution.finish_status("flight_encode", &status);

    let payload = coordinator.query_stats("status-source").unwrap();
    let payload = std::str::from_utf8(&payload).unwrap();
    assert!(payload.contains("operation=algeon_datafusion.server.flight.test.encode"), "{payload}");
    assert!(payload.contains("source_chain=socket closed"), "{payload}");
    assert!(payload.contains("diagnostic=socket closed"), "{payload}");
}
