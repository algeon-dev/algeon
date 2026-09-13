use std::{
    error::Error as StdError,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use arrow::{
    array::Int32Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use query_engine::{
    error::{QueryError, QueryErrorCode},
    exec::{QueryExecutionRequest, execute_query_plan},
    metrics::{QueryId, TraceId},
    observability::{
        QueryBoundaryKind, RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationPayload,
        RuntimeObservationRecordDisposition, RuntimeObservationSession, RuntimeQueryId, RuntimeTerminalOutcome,
        RuntimeTraceId,
    },
    plan::QueryPlanBuilder,
    runtime::{
        ExecutionCancellation, ExecutionContext, MemoryResourcePolicy, RuntimeObservationBuildError,
        SourceSchedulerPolicy,
    },
    source::{LiteralBatchSource, SourceNode},
};

const LARGE_BYTE_LIMIT: u64 = 16 * 1024 * 1024;

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
fn observed_execution_cancellation_preserves_error_metrics_and_partial_events() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut plan = QueryPlanBuilder::new();
    let source = plan.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let plan = plan.finish_with_host_sink(source);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();
    let observed = ExecutionContext::builder()
        .with_query_id("observed-cancel")
        .with_cancellation(cancellation)
        .with_runtime_observation_session(RuntimeObservationSession::new(
            "observed-cancel",
            None,
            RuntimeObservationConfig::bounded(16, LARGE_BYTE_LIMIT),
        ))
        .build_with_runtime_observation()
        .expect("context builds before execution");

    let outcome = execute_query_plan(observed.context(), QueryExecutionRequest::new(&plan));
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
    let _plan = plan.finish_with_host_sink(source);
    let failure = ExecutionContext::builder()
        .with_query_id("context-query")
        .with_trace_id("shared-trace")
        .with_runtime_observation_session(RuntimeObservationSession::new(
            "manifest-query",
            Some(RuntimeTraceId::new("shared-trace")),
            RuntimeObservationConfig::bounded(32, LARGE_BYTE_LIMIT),
        ))
        .build_with_runtime_observation()
        .expect_err("query identity mismatch must fail during context construction");
    let (error, observation) = failure.into_parts();
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_query_id_mismatch"));
    let snapshot =
        observation.seal(RuntimeTerminalOutcome::from_query_error(&error)).expect("mismatch returns seal authority");
    assert_eq!(snapshot.manifest().query_id.as_str(), "manifest-query");
    assert_eq!(snapshot.manifest().trace_id.as_ref().map(RuntimeTraceId::as_str), Some("shared-trace"));
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::from_query_error(&error)));
}

#[test]
fn explicit_observed_execution_uses_manifest_identity_on_trace_mismatch() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut plan = QueryPlanBuilder::new();
    let source = plan.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let _plan = plan.finish_with_host_sink(source);
    let failure = ExecutionContext::builder()
        .with_query_id("shared-query")
        .with_trace_id("context-trace")
        .with_runtime_observation_session(RuntimeObservationSession::new(
            "shared-query",
            Some(RuntimeTraceId::new("manifest-trace")),
            RuntimeObservationConfig::bounded(32, LARGE_BYTE_LIMIT),
        ))
        .build_with_runtime_observation()
        .expect_err("trace identity mismatch must fail during context construction");
    let (error, observation) = failure.into_parts();
    assert_eq!(error.fact_value("reason"), Some("runtime_observation_trace_id_mismatch"));
    let snapshot =
        observation.seal(RuntimeTerminalOutcome::from_query_error(&error)).expect("mismatch returns seal authority");
    assert_eq!(snapshot.manifest().query_id.as_str(), "shared-query");
    assert_eq!(snapshot.manifest().trace_id.as_ref().map(RuntimeTraceId::as_str), Some("manifest-trace"));
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
    assert_eq!(snapshot.events().iter().filter(|event| event.payload == query_started()).count(), 2);
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

fn recover_recorded_owner(failure: RuntimeObservationBuildError) -> QueryError {
    let (error, observation) = failure.into_parts();
    let live = observation.live_snapshot().expect("failed build returns a live owner");
    assert_eq!(live.events().len(), 1, "records captured before the build failure remain available");
    let sealed =
        observation.seal(RuntimeTerminalOutcome::from_query_error(&error)).expect("recovered owner remains sealable");
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
fn observed_context_adopts_canonical_manifest_identity() {
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
    assert_eq!(error.operation(), "query_engine.source_scheduler_policy.validate");

    let config_failure = ExecutionContext::builder()
        .with_service_stream_pool_size(0)
        .with_runtime_observation_session(recorded_session("config-failure", None))
        .build_with_runtime_observation()
        .expect_err("invalid configuration is rejected");
    let error = recover_recorded_owner(config_failure);
    assert_eq!(error.operation(), "query_engine.service_stream_pool.validate");

    let runtime_failure = ExecutionContext::builder()
        .with_memory_policy(MemoryResourcePolicy::Pool { initial_pool_size: 4_096, maximum_pool_size: Some(1_024) })
        .with_runtime_observation_session(recorded_session("runtime-failure", None))
        .build_with_runtime_observation()
        .expect_err("invalid pool sizing fails runtime construction");
    let error = recover_recorded_owner(runtime_failure);
    assert_eq!(error.operation(), "query_engine.runtime.build");
}

#[test]
fn erased_representative_build_errors_seal_dropped_without_losing_records() {
    let (session, handle) = recorded_session_with_handle("drop-wrong-method", None);
    let failure = ExecutionContext::builder()
        .with_runtime_observation_session(session)
        .build()
        .expect_err("wrong build method fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "query_engine.runtime_observation.owner");

    let (session, handle) = recorded_session_with_handle("drop-identity", None);
    let failure = ExecutionContext::builder()
        .with_query_id("different-query")
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("identity mismatch fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "query_engine.runtime_observation.identity");

    let (session, handle) = recorded_session_with_handle("drop-policy", None);
    let failure = ExecutionContext::builder()
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(0))
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("invalid policy fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "query_engine.source_scheduler_policy.validate");

    let (session, handle) = recorded_session_with_handle("drop-config", None);
    let failure = ExecutionContext::builder()
        .with_service_stream_pool_size(0)
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("invalid configuration fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "query_engine.service_stream_pool.validate");

    let (session, handle) = recorded_session_with_handle("drop-runtime", None);
    let failure = ExecutionContext::builder()
        .with_memory_policy(MemoryResourcePolicy::Pool { initial_pool_size: 4_096, maximum_pool_size: Some(1_024) })
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .expect_err("invalid runtime construction fails");
    assert_erased_build_error_drop_seals_dropped(failure, handle, "query_engine.runtime.build");
}

#[test]
fn build_error_into_parts_disarms_drop_and_returns_live_owner() {
    let (session, handle) = recorded_session_with_handle("recover-live-owner", None);
    let failure = ExecutionContext::builder()
        .with_runtime_observation_session(session)
        .build()
        .expect_err("wrong build method fails");
    let (error, observation) = failure.into_parts();
    assert_eq!(error.operation(), "query_engine.runtime_observation.owner");
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
    assert_eq!(
        observed
            .observation()
            .live_snapshot()
            .expect("owner sees clone record")
            .events()
            .iter()
            .filter(|event| event.payload == query_started())
            .count(),
        2
    );

    let query_error = observed.context().with_query_id("other-query").expect_err("query id cannot diverge");
    assert_eq!(query_error.fact_value("reason"), Some("runtime_observation_query_id_mismatch"));
    let trace_error = observed.context().with_trace_id("other-trace").expect_err("trace id cannot diverge");
    assert_eq!(trace_error.fact_value("reason"), Some("runtime_observation_trace_id_mismatch"));
}
