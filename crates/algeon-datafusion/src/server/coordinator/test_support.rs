//! Attempt reports and sealed snapshots for coordinator handoff tests.

use std::sync::Arc;

use query_planning::correlation::QueryId;
use query_runtime::observability::{
    AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptId, AttemptQueryId,
    AttemptRetryAdvice, AttemptTimestamps, QUERY_ATTEMPT_REPORT_SCHEMA, QueryAttemptReport, RuntimeObservationConfig,
    RuntimeObservationOwner, RuntimeTerminalOutcome, SealedRuntimeObservationSnapshot,
};

/// A succeeded, not-admitted attempt report for `query_id`.
pub(crate) fn cpu_attempt_report(query_id: &str) -> QueryAttemptReport {
    QueryAttemptReport {
        schema: QUERY_ATTEMPT_REPORT_SCHEMA.to_string(),
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

/// Seals a fresh attempt observation with `report`, the way an attempt hands
/// its report and snapshot to the coordinator.
pub(crate) fn sealed_attempt_report_snapshot(
    report: QueryAttemptReport,
) -> (QueryAttemptReport, SealedRuntimeObservationSnapshot) {
    let query_id = report.query_id.as_ref().map_or("attempt-query", |query_id| query_id.as_str());
    let owner = RuntimeObservationOwner::new(QueryId::new(query_id), None, RuntimeObservationConfig::bounded(16, 4096));
    seal_attempt_observation(owner, report)
}

/// Seals `observation`, with whatever a test recorded into it, under
/// `report`.
pub(crate) fn seal_attempt_observation(
    observation: RuntimeObservationOwner,
    report: QueryAttemptReport,
) -> (QueryAttemptReport, SealedRuntimeObservationSnapshot) {
    let terminal = report.outcome.clone();
    let snapshot =
        observation.with_attempt_report_provider(Arc::new(move || report.clone())).finish_query(Some(terminal));
    let snapshot = snapshot.expect("attempt observation seals before coordinator handoff");
    // The sealed snapshot owns the immutable report projection accepted by the
    // coordinator. Return that exact projection rather than a pre-seal test
    // template, whose completeness footer may have been filled during sealing.
    let report = snapshot.attempt_report().expect("sealed snapshot carries its attempt report").clone();
    (report, snapshot)
}
