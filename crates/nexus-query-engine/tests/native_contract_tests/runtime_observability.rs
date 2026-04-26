use std::{
    error::Error as StdError,
    fs,
    hint::black_box,
    io::{self, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

use arrow::{
    array::Int32Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use nexus_query_engine::{
    error::{QueryError, QueryErrorCode},
    exec::execute_query_plan_with_runtime_observation,
    metrics::{QueryId, TraceId},
    observability::{
        DeterministicRuntimeObservationClock, ObservationSequence, PlanEdgeRow, PlanExecutionBoundaryKind,
        PlanExecutionId, PlanNodeRow, QueryBoundaryKind, RUNTIME_OBSERVATION_JSONL_SCHEMA, RuntimeEventContext,
        RuntimeObservationConfig, RuntimeObservationExportErrorKind, RuntimeObservationExportErrorStatus,
        RuntimeObservationHandle, RuntimeObservationLossReason, RuntimeObservationPayload, RuntimeObservationRecord,
        RuntimeObservationRecordClass, RuntimeObservationRecordDisposition, RuntimeObservationSession,
        RuntimePlanNodeId, RuntimeQueryId, RuntimeQueryLifecycle, RuntimeResourceDomain, RuntimeResourceId,
        RuntimeResourceRow, RuntimeSpanId, RuntimeTaskId, RuntimeTerminalOutcome, RuntimeTopologyRecord,
        RuntimeTraceId, TaskDefinitionRow, write_runtime_observation_jsonl,
    },
    plan::QueryPlanBuilder,
    runtime::{
        ExecutionCancellation, ExecutionContext, MemoryResourcePolicy, RuntimeObservationBuildError,
        SourceSchedulerPolicy,
    },
    source::{LiteralBatchSource, SourceNode},
};

const LARGE_BYTE_LIMIT: u64 = 16 * 1024 * 1024;

fn test_session(
    config: RuntimeObservationConfig,
) -> (RuntimeObservationSession, Arc<DeterministicRuntimeObservationClock>) {
    let clock = Arc::new(DeterministicRuntimeObservationClock::new(Some(1_234_567)));
    let session = RuntimeObservationSession::with_clock(
        "query-observation",
        Some(RuntimeTraceId::new("trace-parent")),
        config,
        clock.clone(),
    );
    (session, clock)
}

fn query_started() -> RuntimeObservationPayload {
    RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started }
}

fn is_query_terminal(payload: &RuntimeObservationPayload) -> bool {
    matches!(
        payload,
        RuntimeObservationPayload::QueryBoundary {
            boundary: QueryBoundaryKind::Succeeded
                | QueryBoundaryKind::Failed
                | QueryBoundaryKind::Cancelled
                | QueryBoundaryKind::Dropped
                | QueryBoundaryKind::Panicked
        }
    )
}

#[test]
fn query_lifecycle_records_cleanup_before_exactly_one_success_terminal() {
    let (session, _) = test_session(RuntimeObservationConfig::bounded(16, LARGE_BYTE_LIMIT));
    let lifecycle = RuntimeQueryLifecycle::new(session, PlanExecutionId::new("plan-success"));
    let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("enabled lifecycle seals");

    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    assert_eq!(snapshot.events().iter().filter(|event| is_query_terminal(&event.payload)).count(), 1);
    let payloads = snapshot.events().iter().map(|event| &event.payload).collect::<Vec<_>>();
    let cleanup_finished = payloads
        .iter()
        .position(|payload| {
            matches!(payload, RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished })
        })
        .expect("cleanup finished is recorded");
    let terminal = payloads.iter().position(|payload| is_query_terminal(payload)).expect("terminal is recorded");
    assert!(cleanup_finished < terminal);
}

#[test]
fn query_lifecycle_drop_and_panic_each_record_one_terminal() {
    let (dropped_session, _) = test_session(RuntimeObservationConfig::bounded(16, LARGE_BYTE_LIMIT));
    let dropped_handle = dropped_session.handle();
    drop(RuntimeQueryLifecycle::new(dropped_session, PlanExecutionId::new("plan-drop")));
    let dropped = dropped_handle.sealed_snapshot().expect("drop seals");
    assert_eq!(dropped.footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
    assert_eq!(dropped.events().iter().filter(|event| is_query_terminal(&event.payload)).count(), 1);

    let (panic_session, _) = test_session(RuntimeObservationConfig::bounded(16, LARGE_BYTE_LIMIT));
    let panic_handle = panic_session.handle();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _lifecycle = RuntimeQueryLifecycle::new(panic_session, PlanExecutionId::new("plan-panic"));
        panic!("lifecycle unwind probe");
    }));
    assert!(result.is_err());
    let panicked = panic_handle.sealed_snapshot().expect("unwind seals");
    assert_eq!(panicked.footer().terminal, Some(RuntimeTerminalOutcome::Panicked));
    assert_eq!(panicked.events().iter().filter(|event| is_query_terminal(&event.payload)).count(), 1);
}

#[test]
fn observed_execution_cancellation_preserves_error_metrics_and_partial_events() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut plan = QueryPlanBuilder::new();
    let source = plan.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let plan = plan.finish_with_host_sink(source);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();
    let context = ExecutionContext::builder()
        .with_query_id("observed-cancel")
        .with_cancellation(cancellation)
        .build()
        .expect("context builds before execution");
    let (session, _) = test_session(RuntimeObservationConfig::bounded(16, LARGE_BYTE_LIMIT));
    let session = RuntimeObservationSession::new("observed-cancel", None, session.config());

    let outcome = execute_query_plan_with_runtime_observation(&context, &plan, session);
    assert!(matches!(outcome.result(), Err(error) if error.code() == QueryErrorCode::Cancelled));
    assert_eq!(outcome.metrics().query_id().as_str(), "observed-cancel");
    let snapshot = outcome.observation().expect("enabled execution returns sealed evidence");
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Cancelled));
    assert_eq!(snapshot.events().iter().filter(|event| is_query_terminal(&event.payload)).count(), 1);
    assert!(snapshot.events().iter().any(|event| {
        matches!(event.payload, RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started })
    }));
}

#[test]
fn explicit_observed_execution_uses_manifest_identity_on_query_mismatch() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut plan = QueryPlanBuilder::new();
    let source = plan.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let plan = plan.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_query_id("context-query")
        .with_trace_id("shared-trace")
        .build()
        .expect("context builds");
    let session = RuntimeObservationSession::new(
        "manifest-query",
        Some(RuntimeTraceId::new("shared-trace")),
        RuntimeObservationConfig::bounded(32, LARGE_BYTE_LIMIT),
    );

    let outcome = execute_query_plan_with_runtime_observation(&context, &plan, session);
    let error = match outcome.result() {
        Err(error) => error,
        Ok(_) => panic!("query identity mismatch must fail before execution"),
    };
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_query_id_mismatch"));
    assert_eq!(error.query_id().map(QueryId::as_str), Some("manifest-query"));
    assert_eq!(error.trace_id().map(TraceId::as_str), Some("shared-trace"));
    assert_eq!(outcome.metrics().query_id().as_str(), "manifest-query");
    assert_eq!(outcome.metrics().trace_id.as_ref().map(TraceId::as_str), Some("shared-trace"));
    let snapshot = outcome.observation().expect("mismatch still seals observation");
    assert_eq!(snapshot.manifest().query_id.as_str(), "manifest-query");
    assert_eq!(snapshot.manifest().trace_id.as_ref().map(RuntimeTraceId::as_str), Some("shared-trace"));
    assert_eq!(
        snapshot.footer().terminal,
        Some(RuntimeTerminalOutcome::Failed {
            error_code: QueryErrorCode::InvalidPlan.to_string(),
            error_status: QueryErrorCode::InvalidPlan.status().as_str().to_string(),
        })
    );
}

#[test]
fn explicit_observed_execution_uses_manifest_identity_on_trace_mismatch() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut plan = QueryPlanBuilder::new();
    let source = plan.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let plan = plan.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_query_id("shared-query")
        .with_trace_id("context-trace")
        .build()
        .expect("context builds");
    let session = RuntimeObservationSession::new(
        "shared-query",
        Some(RuntimeTraceId::new("manifest-trace")),
        RuntimeObservationConfig::bounded(32, LARGE_BYTE_LIMIT),
    );

    let outcome = execute_query_plan_with_runtime_observation(&context, &plan, session);
    let error = match outcome.result() {
        Err(error) => error,
        Ok(_) => panic!("trace identity mismatch must fail before execution"),
    };
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_trace_id_mismatch"));
    assert_eq!(error.query_id().map(QueryId::as_str), Some("shared-query"));
    assert_eq!(error.trace_id().map(TraceId::as_str), Some("manifest-trace"));
    assert_eq!(outcome.metrics().query_id().as_str(), "shared-query");
    assert_eq!(outcome.metrics().trace_id.as_ref().map(TraceId::as_str), Some("manifest-trace"));
    let snapshot = outcome.observation().expect("mismatch still seals observation");
    assert_eq!(snapshot.manifest().query_id.as_str(), "shared-query");
    assert_eq!(snapshot.manifest().trace_id.as_ref().map(RuntimeTraceId::as_str), Some("manifest-trace"));
}

#[test]
fn fake_clock_orders_and_clamps_events_under_recorder_sequence() {
    let (session, clock) = test_session(RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT));
    clock.set_monotonic_ns(20);
    session.record(query_started);
    clock.set_monotonic_ns(5);
    session.record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupStarted });

    let snapshot = session.live_snapshot().expect("enabled session has live snapshot");
    assert_eq!(snapshot.manifest().unix_epoch_ns, Some(1_234_567));
    assert_eq!(snapshot.events().iter().map(|event| event.sequence.get()).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(snapshot.events().iter().map(|event| event.monotonic_ns).collect::<Vec<_>>(), [20, 20]);
}

#[test]
fn concurrent_recorders_assign_unique_total_order() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 200;
    let (session, clock) =
        test_session(RuntimeObservationConfig::bounded((THREADS * PER_THREAD) as u64, LARGE_BYTE_LIMIT));
    let barrier = Arc::new(Barrier::new(THREADS));
    let recording = session.handle();
    let handles = (0..THREADS)
        .map(|_| {
            let recording = recording.clone();
            let clock = Arc::clone(&clock);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..PER_THREAD {
                    clock.advance(1);
                    assert!(matches!(
                        recording.record(query_started),
                        RuntimeObservationRecordDisposition::Retained { .. }
                    ));
                }
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().expect("recording worker completed");
    }

    let snapshot = session.live_snapshot().expect("live snapshot");
    assert_eq!(snapshot.events().len(), THREADS * PER_THREAD);
    for (index, event) in snapshot.events().iter().enumerate() {
        assert_eq!(event.sequence.get(), index as u64 + 1);
    }
    assert!(snapshot.events().windows(2).all(|events| events[0].monotonic_ns <= events[1].monotonic_ns));
}

#[test]
fn record_cap_drops_whole_records_and_reports_loss() {
    let (session, _) = test_session(RuntimeObservationConfig::bounded(2, LARGE_BYTE_LIMIT));
    assert!(matches!(session.record(query_started), RuntimeObservationRecordDisposition::Retained { .. }));
    assert!(matches!(session.record(query_started), RuntimeObservationRecordDisposition::Retained { .. }));
    match session.record(query_started) {
        RuntimeObservationRecordDisposition::Dropped { sequence, reason } => {
            assert_eq!(sequence.get(), 3);
            assert_eq!(reason, RuntimeObservationLossReason::RecordLimit);
        },
        other => panic!("third record must be dropped: {other:?}"),
    }

    let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).expect("sealed snapshot");
    assert_eq!(snapshot.events().len(), 2);
    assert!(snapshot.footer().truncated);
    assert_eq!(snapshot.footer().first_dropped_sequence.map(ObservationSequence::get), Some(3));
    assert_eq!(snapshot.footer().last_attempted_sequence.map(ObservationSequence::get), Some(3));
    assert_eq!(snapshot.footer().dropped_by_reason[0].reason, RuntimeObservationLossReason::RecordLimit);
    assert_eq!(snapshot.footer().dropped_by_reason[0].count, 1);
}

#[test]
fn byte_cap_distinguishes_oversize_from_accumulated_limit() {
    let (probe, _) = test_session(RuntimeObservationConfig::bounded(2, LARGE_BYTE_LIMIT));
    let logical_bytes = match probe.record(query_started) {
        RuntimeObservationRecordDisposition::Retained { logical_bytes, .. } => logical_bytes,
        other => panic!("probe event should be retained: {other:?}"),
    };

    let (oversize, _) = test_session(RuntimeObservationConfig::bounded(2, logical_bytes - 1));
    assert!(matches!(
        oversize.record(query_started),
        RuntimeObservationRecordDisposition::Dropped { reason: RuntimeObservationLossReason::Oversized, .. }
    ));

    let (accumulated, _) = test_session(RuntimeObservationConfig::bounded(2, logical_bytes * 2 - 1));
    assert!(matches!(accumulated.record(query_started), RuntimeObservationRecordDisposition::Retained { .. }));
    assert!(matches!(
        accumulated.record(query_started),
        RuntimeObservationRecordDisposition::Dropped { reason: RuntimeObservationLossReason::ByteLimit, .. }
    ));
    let snapshot = accumulated.seal(RuntimeTerminalOutcome::Succeeded).expect("sealed snapshot");
    assert_eq!(snapshot.footer().retained_bytes, logical_bytes);
    assert_eq!(snapshot.footer().dropped_by_reason[0].reason, RuntimeObservationLossReason::ByteLimit);
}

#[test]
fn first_terminal_wins_and_live_snapshots_remain_explicitly_incomplete() {
    let (session, _) = test_session(RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT));
    session.record(query_started);
    let live = session.live_snapshot().expect("session is live");
    assert!(!live.footer().complete);
    assert_eq!(live.footer().terminal, None);

    let first = session.seal(RuntimeTerminalOutcome::Succeeded).expect("first seal succeeds");
    let second = session
        .seal(RuntimeTerminalOutcome::Failed {
            error_code: "internal".to_string(),
            error_status: QueryErrorCode::Internal.status().as_str().to_string(),
        })
        .expect("subsequent seal returns winner");
    assert!(first.footer().complete);
    assert_eq!(first.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    assert_eq!(second.footer().terminal, first.footer().terminal);
    assert!(session.live_snapshot().is_none());

    let payload_built = AtomicBool::new(false);
    assert_eq!(
        session.record(|| {
            payload_built.store(true, Ordering::Release);
            query_started()
        }),
        RuntimeObservationRecordDisposition::Sealed
    );
    assert!(!payload_built.load(Ordering::Acquire));
}

#[test]
fn concurrent_authorized_terminal_race_returns_one_winner() {
    let (session, _) = test_session(RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT));
    let owner = Arc::new(session);
    let barrier = Arc::new(Barrier::new(2));
    let handles = [RuntimeTerminalOutcome::Succeeded, RuntimeTerminalOutcome::Cancelled]
        .into_iter()
        .map(|outcome| {
            let owner = Arc::clone(&owner);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                owner.seal(outcome).expect("enabled owner seals")
            })
        })
        .collect::<Vec<_>>();
    let snapshots =
        handles.into_iter().map(|handle| handle.join().expect("terminal racer completed")).collect::<Vec<_>>();
    assert_eq!(snapshots[0].footer().terminal, snapshots[1].footer().terminal);
    assert!(snapshots[0].footer().complete);
}

#[test]
fn jsonl_is_schema_versioned_ordered_and_round_trips() {
    let (session, clock) = test_session(RuntimeObservationConfig::bounded(16, LARGE_BYTE_LIMIT));
    let execution = PlanExecutionId::new("execution-1");
    session.record_topology(RuntimeObservationRecordClass::PlanNode, || {
        RuntimeTopologyRecord::PlanNode(PlanNodeRow {
            plan_execution_id: execution.clone(),
            node_id: RuntimePlanNodeId::new("node-b"),
            operator_kind: "filter".to_string(),
        })
    });
    session.record_topology(RuntimeObservationRecordClass::PlanNode, || {
        RuntimeTopologyRecord::PlanNode(PlanNodeRow {
            plan_execution_id: execution.clone(),
            node_id: RuntimePlanNodeId::new("node-a"),
            operator_kind: "source".to_string(),
        })
    });
    session.record_topology(RuntimeObservationRecordClass::PlanEdge, || {
        RuntimeTopologyRecord::PlanEdge(PlanEdgeRow {
            plan_execution_id: execution.clone(),
            from_node_id: RuntimePlanNodeId::new("node-a"),
            to_node_id: RuntimePlanNodeId::new("node-b"),
            input_ordinal: 0,
        })
    });
    session.record_topology(RuntimeObservationRecordClass::TaskDefinition, || {
        RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
            plan_execution_id: execution.clone(),
            task_id: RuntimeTaskId::new("task-1"),
            plan_node_id: Some(RuntimePlanNodeId::new("node-b")),
            task_kind: "unary".to_string(),
        })
    });
    session.record_topology(RuntimeObservationRecordClass::Resource, || {
        RuntimeTopologyRecord::Resource(RuntimeResourceRow {
            plan_execution_id: Some(execution.clone()),
            resource_id: RuntimeResourceId::new("resource-1"),
            resource_kind: "frame".to_string(),
            domain: RuntimeResourceDomain::Device { device_ordinal: 0 },
            producer_task_id: None,
            consumer_task_ids: Vec::new(),
            lifetime: "query".to_string(),
        })
    });
    clock.set_monotonic_ns(42);
    session.record_for_plan(execution, || RuntimeObservationPayload::PlanExecutionBoundary {
        boundary: PlanExecutionBoundaryKind::Started,
    });
    let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).expect("sealed snapshot");

    let mut jsonl = Vec::new();
    write_runtime_observation_jsonl(&snapshot, &mut jsonl).expect("JSONL export succeeds");
    let records = String::from_utf8(jsonl)
        .expect("JSONL is UTF-8")
        .lines()
        .map(|line| serde_json::from_str::<RuntimeObservationRecord>(line).expect("line parses"))
        .collect::<Vec<_>>();
    assert!(
        matches!(&records[0], RuntimeObservationRecord::Manifest(manifest) if manifest.schema == RUNTIME_OBSERVATION_JSONL_SCHEMA)
    );
    assert!(matches!(&records[1], RuntimeObservationRecord::PlanNode { row, .. } if row.node_id.as_str() == "node-a"));
    assert!(matches!(&records[2], RuntimeObservationRecord::PlanNode { row, .. } if row.node_id.as_str() == "node-b"));
    assert!(matches!(&records[3], RuntimeObservationRecord::PlanEdge { .. }));
    assert!(matches!(&records[4], RuntimeObservationRecord::TaskDefinition { .. }));
    assert!(matches!(&records[5], RuntimeObservationRecord::Resource { .. }));
    assert!(matches!(&records[6], RuntimeObservationRecord::TimelineEvent(event) if event.sequence.get() == 6));
    assert!(matches!(records.last(), Some(RuntimeObservationRecord::Footer(footer)) if footer.complete));
    for record in records {
        let encoded = serde_json::to_vec(&record).expect("record re-encodes");
        let decoded: RuntimeObservationRecord = serde_json::from_slice(&encoded).expect("record round trips");
        assert_eq!(decoded, record);
    }
}

#[test]
fn jsonl_v2_matches_fixed_golden_bytes() {
    let clock = Arc::new(DeterministicRuntimeObservationClock::new(Some(1_234)));
    let session = RuntimeObservationSession::with_clock(
        "golden-query",
        Some(RuntimeTraceId::new("golden-trace")),
        RuntimeObservationConfig::bounded(6, 4_096),
        clock.clone(),
    );
    let execution = PlanExecutionId::new("golden-execution");
    session.record_topology(RuntimeObservationRecordClass::PlanNode, || {
        RuntimeTopologyRecord::PlanNode(PlanNodeRow {
            plan_execution_id: execution.clone(),
            node_id: RuntimePlanNodeId::new("golden-node"),
            operator_kind: "hash_join".to_string(),
        })
    });
    session.record_topology(RuntimeObservationRecordClass::PlanEdge, || {
        RuntimeTopologyRecord::PlanEdge(PlanEdgeRow {
            plan_execution_id: execution.clone(),
            from_node_id: RuntimePlanNodeId::new("golden-node"),
            to_node_id: RuntimePlanNodeId::new("golden-output"),
            input_ordinal: 2,
        })
    });
    session.record_topology(RuntimeObservationRecordClass::TaskDefinition, || {
        RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
            plan_execution_id: execution.clone(),
            task_id: RuntimeTaskId::new("golden-task"),
            plan_node_id: Some(RuntimePlanNodeId::new("golden-node")),
            task_kind: "partitioned_probe".to_string(),
        })
    });
    session.record_topology(RuntimeObservationRecordClass::Resource, || {
        RuntimeTopologyRecord::Resource(RuntimeResourceRow {
            plan_execution_id: Some(execution.clone()),
            resource_id: RuntimeResourceId::new("golden-resource"),
            resource_kind: "device_frame".to_string(),
            domain: RuntimeResourceDomain::Device { device_ordinal: 3 },
            producer_task_id: None,
            consumer_task_ids: Vec::new(),
            lifetime: "query".to_string(),
        })
    });
    clock.set_monotonic_ns(7);
    session.record_with_context(
        RuntimeEventContext::for_plan(execution).with_span(RuntimeSpanId::new(17), Some(RuntimeSpanId::new(9))),
        || RuntimeObservationPayload::PlanExecutionBoundary { boundary: PlanExecutionBoundaryKind::Failed },
    );
    clock.set_monotonic_ns(11);
    session.record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished });
    clock.set_monotonic_ns(13);
    session.record(query_started);
    let snapshot = session
        .seal(RuntimeTerminalOutcome::Failed {
            error_code: "golden_failure".to_string(),
            error_status: "permanent".to_string(),
        })
        .expect("golden session seals");
    assert!(matches!(
        snapshot.events().last().map(|event| &event.payload),
        Some(RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished })
    ));

    let mut actual = Vec::new();
    write_runtime_observation_jsonl(&snapshot, &mut actual).expect("golden JSONL exports");
    let expected = include_bytes!("../fixtures/runtime_observation_v2.jsonl");
    assert_eq!(actual.as_slice(), expected, "JSONL v2 bytes changed:\n{}", String::from_utf8_lossy(&actual));
    assert!(actual.ends_with(b"\n"));
}

#[test]
fn writer_failure_does_not_mutate_or_consume_snapshot() {
    let (session, _) = test_session(RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT));
    session.record(query_started);
    let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).expect("sealed snapshot");
    let footer_before = snapshot.footer().clone();

    let mut failing = FailingWriter { remaining: 8 };
    let error = write_runtime_observation_jsonl(&snapshot, &mut failing).expect_err("writer should fail");
    assert_eq!(error.kind(), RuntimeObservationExportErrorKind::DestinationUnavailable);
    assert_eq!(error.status(), RuntimeObservationExportErrorStatus::Temporary);
    assert!(error.is_retryable());
    assert_eq!(error.record_index(), 0);
    assert_eq!(snapshot.footer(), &footer_before);

    let mut retry = Vec::new();
    write_runtime_observation_jsonl(&snapshot, &mut retry).expect("same snapshot exports to a healthy writer");
    assert!(!retry.is_empty());
}

#[test]
fn disabled_session_does_not_construct_lazy_payload() {
    let session = RuntimeObservationSession::disabled();
    let payload_built = AtomicBool::new(false);
    assert_eq!(
        session.record(|| {
            payload_built.store(true, Ordering::Release);
            query_started()
        }),
        RuntimeObservationRecordDisposition::Disabled
    );
    assert!(!payload_built.load(Ordering::Acquire));
    assert!(session.live_snapshot().is_none());
}

#[test]
fn execution_context_workers_receive_recording_handle_without_owner_authority() {
    let observed = ExecutionContext::builder()
        .with_query_id(QueryId::new("context-query"))
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT))
        .build_with_runtime_observation()
        .expect("observed context builds");
    assert_recording_handle(observed.context().runtime_observation());
    let cloned = observed.context().clone();
    cloned.runtime_observation().record(query_started);

    let snapshot = observed.context().runtime_observation().live_snapshot().expect("original sees clone record");
    assert_eq!(snapshot.manifest().query_id.as_str(), "context-query");
    assert_eq!(snapshot.events().len(), 1);
    let sealed = observed.observation().seal(RuntimeTerminalOutcome::Succeeded).expect("owner seals after worker");
    assert!(sealed.footer().complete);
    assert_eq!(cloned.runtime_observation().record(query_started), RuntimeObservationRecordDisposition::Sealed);
}

fn assert_recording_handle(_handle: &RuntimeObservationHandle) {}

fn recorded_session(query_id: &str, trace_id: Option<&str>) -> RuntimeObservationSession {
    recorded_session_with_handle(query_id, trace_id).0
}

fn recorded_session_with_handle(
    query_id: &str,
    trace_id: Option<&str>,
) -> (RuntimeObservationSession, RuntimeObservationHandle) {
    let session = RuntimeObservationSession::new(
        query_id,
        trace_id.map(RuntimeTraceId::new),
        RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT),
    );
    session.record(query_started);
    let handle = session.handle();
    (session, handle)
}

fn recover_recorded_owner(failure: RuntimeObservationBuildError) -> nexus_query_engine::error::QueryError {
    let (error, observation) = failure.into_parts();
    let live = observation.live_snapshot().expect("failed build returns a live owner");
    assert_eq!(live.events().len(), 1, "records captured before the build failure remain available");
    let sealed = observation
        .seal(RuntimeTerminalOutcome::Failed {
            error_code: error.code().to_string(),
            error_status: error.status().as_str().to_string(),
        })
        .expect("recovered owner remains sealable");
    assert!(sealed.footer().complete);
    assert_eq!(sealed.events().len(), 1);
    error
}

fn assert_erased_build_error_drop_seals_dropped(
    failure: RuntimeObservationBuildError,
    handle: RuntimeObservationHandle,
    expected_operation: &str,
) {
    assert_eq!(failure.query_error().operation(), expected_operation);
    let expected_display = failure.query_error().to_string();
    let erased: Box<dyn StdError> = Box::new(failure);
    assert_eq!(erased.to_string(), expected_display);
    let source = erased
        .source()
        .and_then(|source| source.downcast_ref::<QueryError>())
        .expect("type erasure preserves the native QueryError source");
    assert_eq!(source.operation(), expected_operation);
    assert_eq!(source.to_string(), expected_display);
    assert!(catch_unwind(AssertUnwindSafe(|| drop(erased))).is_ok(), "build-error drop must not panic");

    let snapshot = handle.sealed_snapshot().expect("dropping an unrecovered build error seals its owner");
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
    assert_eq!(snapshot.events().len(), 1, "records retained before the build failure survive terminalization");
    assert_eq!(handle.record(query_started), RuntimeObservationRecordDisposition::Sealed);
}

#[test]
fn plain_build_rejects_enabled_observation_without_owner_lifetime() {
    let error = ExecutionContext::builder()
        .with_query_id("owner-required")
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT))
        .build()
        .expect_err("plain context cannot discard seal authority");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_owner_required"));
}

#[test]
fn supplied_owner_is_recovered_from_wrong_build_method() {
    let failure = ExecutionContext::builder()
        .with_runtime_observation_session(recorded_session("owner-required", None))
        .build()
        .expect_err("owner-discarding build method is rejected");
    assert_eq!(failure.observation().live_snapshot().expect("owner is present").events().len(), 1);
    let error = recover_recorded_owner(failure);
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_owner_required"));
}

#[test]
fn supplied_session_manifest_adopts_or_validates_context_identity() {
    let supplied = RuntimeObservationSession::new(
        "manifest-query",
        Some(RuntimeTraceId::new("manifest-trace")),
        RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT),
    );
    let observed = ExecutionContext::builder()
        .with_runtime_observation_session(supplied)
        .build_with_runtime_observation()
        .expect("manifest identity is adopted");
    assert_eq!(observed.context().query_id().map(QueryId::as_str), Some("manifest-query"));
    assert_eq!(observed.context().trace_id().map(TraceId::as_str), Some("manifest-trace"));
    assert_eq!(observed.context().runtime_observation().query_id().map(|id| id.as_str()), Some("manifest-query"));

    let matching = RuntimeObservationSession::new(
        "matching-query",
        Some(RuntimeTraceId::new("matching-trace")),
        RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT),
    );
    let matching = ExecutionContext::builder()
        .with_query_id("matching-query")
        .with_trace_id("matching-trace")
        .with_runtime_observation_session(matching)
        .build_with_runtime_observation()
        .expect("matching explicit identity is accepted");
    let clone = matching.context().clone();
    assert_eq!(clone.query_id().map(QueryId::as_str), Some("matching-query"));
    assert_eq!(clone.runtime_observation().query_id().map(|id| id.as_str()), Some("matching-query"));
}

#[test]
fn runtime_query_identity_uses_one_canonical_representation() {
    for raw in ["", " ", "\t\n"] {
        assert_eq!(RuntimeQueryId::new(raw).as_str(), QueryId::UNASSIGNED);
        let decoded: RuntimeQueryId =
            serde_json::from_str(&serde_json::to_string(raw).expect("identity serializes")).expect("identity parses");
        assert_eq!(decoded.as_str(), QueryId::UNASSIGNED);
    }
    assert_eq!(RuntimeQueryId::new("normal-query").as_str(), "normal-query");

    let observed = ExecutionContext::builder()
        .with_runtime_observation_session(recorded_session(" \t", None))
        .build_with_runtime_observation()
        .expect("canonical manifest identity is adopted");
    assert_eq!(observed.context().query_id().map(QueryId::as_str), Some(QueryId::UNASSIGNED));
    assert_eq!(
        observed.context().runtime_observation().query_id().map(RuntimeQueryId::as_str),
        Some(QueryId::UNASSIGNED)
    );
}

#[test]
fn supplied_session_identity_mismatches_are_structured_errors() {
    let failure = ExecutionContext::builder()
        .with_query_id("different-query")
        .with_runtime_observation_session(recorded_session("manifest-query", None))
        .build_with_runtime_observation()
        .expect_err("query mismatch is rejected before runtime construction");
    let error = recover_recorded_owner(failure);
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_query_id_mismatch"));

    let failure = ExecutionContext::builder()
        .with_query_id("manifest-query")
        .with_trace_id("different-trace")
        .with_runtime_observation_session(recorded_session("manifest-query", Some("manifest-trace")))
        .build_with_runtime_observation()
        .expect_err("trace mismatch is rejected before runtime construction");
    let error = recover_recorded_owner(failure);
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_trace_id_mismatch"));
}

#[test]
fn supplied_owner_is_recovered_from_policy_config_and_runtime_failures() {
    let policy_failure = ExecutionContext::builder()
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(0))
        .with_runtime_observation_session(recorded_session("policy-failure", None))
        .build_with_runtime_observation()
        .expect_err("invalid policy is rejected");
    let error = recover_recorded_owner(policy_failure);
    assert_eq!(error.operation(), "nexus_query_engine.source_scheduler_policy.validate");

    let config_failure = ExecutionContext::builder()
        .with_service_stream_pool_size(0)
        .with_runtime_observation_session(recorded_session("config-failure", None))
        .build_with_runtime_observation()
        .expect_err("invalid configuration is rejected");
    let error = recover_recorded_owner(config_failure);
    assert_eq!(error.operation(), "nexus_query_engine.service_stream_pool.validate");

    let runtime_failure = ExecutionContext::builder()
        .with_memory_policy(MemoryResourcePolicy::Pool { initial_pool_size: 4_096, maximum_pool_size: Some(1_024) })
        .with_runtime_observation_session(recorded_session("runtime-failure", None))
        .build_with_runtime_observation()
        .expect_err("invalid pool sizing fails runtime construction");
    let error = recover_recorded_owner(runtime_failure);
    assert_eq!(error.operation(), "nexus_query_engine.runtime.build");
}

#[test]
fn erased_representative_build_errors_seal_dropped_without_losing_records() {
    let (session, handle) = recorded_session_with_handle("drop-wrong-method", None);
    let failure = ExecutionContext::builder()
        .with_runtime_observation_session(session)
        .build()
        .expect_err("wrong build method fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "nexus_query_engine.runtime_observation.owner");

    let (session, handle) = recorded_session_with_handle("drop-identity", None);
    let failure = ExecutionContext::builder()
        .with_query_id("different-query")
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("identity mismatch fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "nexus_query_engine.runtime_observation.identity");

    let (session, handle) = recorded_session_with_handle("drop-policy", None);
    let failure = ExecutionContext::builder()
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(0))
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("invalid policy fails");
    assert_erased_build_error_drop_seals_dropped(
        failure,
        handle,
        "nexus_query_engine.source_scheduler_policy.validate",
    );

    let (session, handle) = recorded_session_with_handle("drop-config", None);
    let failure = ExecutionContext::builder()
        .with_service_stream_pool_size(0)
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("invalid configuration fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "nexus_query_engine.service_stream_pool.validate");

    let (session, handle) = recorded_session_with_handle("drop-runtime", None);
    let failure = ExecutionContext::builder()
        .with_memory_policy(MemoryResourcePolicy::Pool { initial_pool_size: 4_096, maximum_pool_size: Some(1_024) })
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("invalid runtime construction fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "nexus_query_engine.runtime.build");
}

#[test]
fn build_error_into_parts_disarms_drop_and_returns_live_owner() {
    let (session, handle) = recorded_session_with_handle("recover-live-owner", None);
    let failure = ExecutionContext::builder()
        .with_runtime_observation_session(session)
        .build()
        .expect_err("wrong build method fails");
    let (error, observation) = failure.into_parts();
    assert_eq!(error.operation(), "nexus_query_engine.runtime_observation.owner");
    assert!(handle.sealed_snapshot().is_none(), "into_parts must not terminalize the recovered owner");
    assert_eq!(observation.live_snapshot().expect("owner remains live").events().len(), 1);
    drop(error);
    assert!(handle.sealed_snapshot().is_none(), "dropping the separated QueryError must not seal the owner");
    let sealed = observation.seal(RuntimeTerminalOutcome::Succeeded).expect("caller seals recovered owner");
    assert_eq!(sealed.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
}

#[test]
fn observed_context_identity_changes_require_manifest_equality() {
    let observed = ExecutionContext::builder()
        .with_query_id("fixed-query")
        .with_trace_id("fixed-trace")
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(8, LARGE_BYTE_LIMIT))
        .build_with_runtime_observation()
        .expect("observed context builds");
    let same = observed.context().with_query_id("fixed-query").expect("same query id is allowed");
    let same = same.with_trace_id("fixed-trace").expect("same trace id is allowed");
    same.runtime_observation().record(query_started);
    assert_eq!(observed.observation().live_snapshot().expect("owner sees clone record").events().len(), 1);

    let query_error = observed.context().with_query_id("other-query").expect_err("query id cannot diverge");
    assert_eq!(query_error.fact_value("reason"), Some("runtime_observation_query_id_mismatch"));
    let trace_error = observed.context().with_trace_id("other-trace").expect_err("trace id cannot diverge");
    assert_eq!(trace_error.fact_value("reason"), Some("runtime_observation_trace_id_mismatch"));
}

#[test]
fn native_observability_has_no_frontend_dependency_edges() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/observability");
    for entry in fs::read_dir(root).expect("observability directory is readable") {
        let entry = entry.expect("directory entry is readable");
        if entry.path().extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let source = fs::read_to_string(entry.path()).expect("Rust source is readable");
        assert!(!source.contains("datafusion::"), "native observability must not import frontend crates");
        assert!(!source.contains("use datafusion"), "native observability must not import frontend crates");
    }
}

#[test]
#[ignore = "manual release-mode runtime observation micro-measurement"]
fn runtime_observation_micro_measurement() {
    const REPETITIONS: usize = 7;
    const SINGLE_ITERATIONS: u64 = 100_000;
    const THREADS: usize = 4;
    const PER_THREAD: u64 = 25_000;

    let samples = (0..REPETITIONS)
        .map(|_| runtime_observation_measurement_sample(SINGLE_ITERATIONS, THREADS, PER_THREAD))
        .collect::<Vec<_>>();
    let baseline = samples.iter().map(|sample| sample.baseline_ns).collect::<Vec<_>>();
    let disabled = samples.iter().map(|sample| sample.disabled_ns).collect::<Vec<_>>();
    let enabled = samples.iter().map(|sample| sample.enabled_ns).collect::<Vec<_>>();
    let contended = samples.iter().map(|sample| sample.contended_ns).collect::<Vec<_>>();
    let retained_bytes = samples.iter().map(|sample| sample.retained_bytes_per_event).collect::<Vec<_>>();

    let disabled_p95 = percentile(&disabled, 0.95);
    let enabled_p95 = percentile(&enabled, 0.95);
    let contended_p95 = percentile(&contended, 0.95);
    let retained_bytes_p95 = percentile(&retained_bytes, 0.95);

    eprintln!(
        "runtime_observation_measurement repetitions={REPETITIONS} baseline_median_ns_per_event={:.2} baseline_p95_ns_per_event={:.2} disabled_median_ns_per_event={:.2} disabled_p95_ns_per_event={:.2} enabled_median_ns_per_event={:.2} enabled_p95_ns_per_event={:.2} contended_median_ns_per_event={:.2} contended_p95_ns_per_event={:.2} retained_bytes_per_event={:.2} payload_size={} event_size={} single_iterations={SINGLE_ITERATIONS} threads={THREADS} per_thread={PER_THREAD}",
        percentile(&baseline, 0.5),
        percentile(&baseline, 0.95),
        percentile(&disabled, 0.5),
        disabled_p95,
        percentile(&enabled, 0.5),
        enabled_p95,
        percentile(&contended, 0.5),
        contended_p95,
        retained_bytes_p95,
        std::mem::size_of::<RuntimeObservationPayload>(),
        std::mem::size_of::<nexus_query_engine::observability::RuntimeTimelineEvent>(),
    );

    assert!(disabled_p95 <= 2.0, "disabled p95 {disabled_p95:.2} ns/event exceeds 2 ns/event budget");
    assert!(enabled_p95 <= 125.0, "enabled p95 {enabled_p95:.2} ns/event exceeds 125 ns/event budget");
    assert!(contended_p95 <= 400.0, "contended p95 {contended_p95:.2} ns/event exceeds 400 ns/event budget");
    assert!(retained_bytes_p95 <= 48.0, "retained logical bytes {retained_bytes_p95:.2}/event exceeds 48/event budget");
}

struct RuntimeObservationMeasurementSample {
    baseline_ns: f64,
    disabled_ns: f64,
    enabled_ns: f64,
    contended_ns: f64,
    retained_bytes_per_event: f64,
}

fn runtime_observation_measurement_sample(
    single_iterations: u64,
    threads: usize,
    per_thread: u64,
) -> RuntimeObservationMeasurementSample {
    let baseline_started = Instant::now();
    for value in 0..single_iterations {
        black_box(value);
    }
    let baseline_ns = baseline_started.elapsed().as_nanos() as f64 / single_iterations as f64;

    let disabled = RuntimeObservationSession::disabled();
    let disabled_started = Instant::now();
    for _ in 0..single_iterations {
        black_box(disabled.record(query_started));
    }
    let disabled_ns = disabled_started.elapsed().as_nanos() as f64 / single_iterations as f64;

    let enabled = RuntimeObservationSession::new(
        "measurement-single",
        None,
        RuntimeObservationConfig::bounded(single_iterations, 64 * 1024 * 1024),
    );
    let enabled_started = Instant::now();
    for _ in 0..single_iterations {
        black_box(enabled.record(query_started));
    }
    let enabled_ns = enabled_started.elapsed().as_nanos() as f64 / single_iterations as f64;
    let enabled_snapshot = enabled.live_snapshot().expect("enabled measurement snapshot");
    let retained_bytes_per_event =
        enabled_snapshot.footer().retained_bytes as f64 / enabled_snapshot.footer().retained_records as f64;

    let contended_events = (threads as u64) * per_thread;
    let contended = RuntimeObservationSession::new(
        "measurement-contended",
        None,
        RuntimeObservationConfig::bounded(contended_events, 64 * 1024 * 1024),
    );
    let recording = contended.handle();
    let barrier = Arc::new(Barrier::new(threads));
    let contended_started = Instant::now();
    let handles = (0..threads)
        .map(|_| {
            let recording = recording.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..per_thread {
                    black_box(recording.record(query_started));
                }
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().expect("measurement worker completed");
    }
    let contended_ns = contended_started.elapsed().as_nanos() as f64 / contended_events as f64;

    RuntimeObservationMeasurementSample { baseline_ns, disabled_ns, enabled_ns, contended_ns, retained_bytes_per_event }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * percentile).ceil() as usize).saturating_sub(1).min(values.len() - 1);
    values[index]
}

struct FailingWriter {
    remaining: usize,
}

impl Write for FailingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "injected writer timeout"));
        }
        let written = self.remaining.min(buffer.len());
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
