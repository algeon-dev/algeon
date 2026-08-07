//! Public execution-attempt lifecycle contracts that need no CUDA device.
//!
//! These tests pin the DataFusion-independent attempt identity, correlation
//! validation, report envelope, and cancellation propagation model through the
//! crate's public API only.

use std::time::{Duration, Instant};

use nexus_query_engine::{
    attempt::{AttemptCancellationBoundary, BoundaryCancellationState, cancellation_state_at, effective_queue_expiry},
    error::QueryErrorCode,
    observability::{
        AttemptDisposition, AttemptId, AttemptQueryId, AttemptTraceContext, MAX_ATTEMPT_QUERY_ID_BYTES,
        MAX_ATTEMPT_TRACE_CONTEXT_BYTES, QUERY_ATTEMPT_REPORT_SCHEMA_V1,
    },
};

#[test]
fn attempt_query_id_validation_is_bounded_and_structural() {
    let accepted = AttemptQueryId::new("q-123").expect("in-limit id");
    assert_eq!(accepted.as_str(), "q-123");

    let at_limit = AttemptQueryId::new("x".repeat(MAX_ATTEMPT_QUERY_ID_BYTES)).expect("exact limit is accepted");
    assert_eq!(at_limit.as_str().len(), MAX_ATTEMPT_QUERY_ID_BYTES);

    let oversized = AttemptQueryId::new("x".repeat(MAX_ATTEMPT_QUERY_ID_BYTES + 1)).unwrap_err();
    assert_eq!(oversized.code(), QueryErrorCode::InvalidAttemptCorrelation);
    assert_eq!(oversized.fact_value("field"), Some("query_id"));
    assert_eq!(oversized.fact_value("reason"), Some("too_long"));

    let blank = AttemptQueryId::new("   ").unwrap_err();
    assert_eq!(blank.code(), QueryErrorCode::InvalidAttemptCorrelation);
    assert_eq!(blank.fact_value("reason"), Some("blank"));
}

#[test]
fn attempt_trace_context_validation_is_bounded_and_carried_unchanged() {
    let trace = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let accepted = AttemptTraceContext::new(trace).expect("trace context");
    assert_eq!(accepted.as_str(), trace, "trace context is carried unchanged");

    let oversized = AttemptTraceContext::new("t".repeat(MAX_ATTEMPT_TRACE_CONTEXT_BYTES + 1)).unwrap_err();
    assert_eq!(oversized.code(), QueryErrorCode::InvalidAttemptCorrelation);
    assert_eq!(oversized.fact_value("field"), Some("trace_context"));
}

#[test]
fn attempt_ids_are_unique_fixed_size_and_formatted() {
    let mut seen = std::collections::HashSet::new();
    for _ in 0..1_000 {
        let id = AttemptId::mint();
        assert_eq!(id.as_bytes().len(), 16);
        assert!(seen.insert(id), "attempt ids are never reused");
        let display = id.to_string();
        assert!(display.starts_with("attempt-"));
        assert_eq!(display.len(), "attempt-".len() + 32);
    }
}

#[test]
fn attempt_report_schema_and_disposition_labels_are_stable() {
    assert_eq!(QUERY_ATTEMPT_REPORT_SCHEMA_V1, "nexus_query_engine.query_attempt_report.v1");
    let dispositions = [
        (AttemptDisposition::Unplanned, "\"unplanned\""),
        (AttemptDisposition::CpuFallback, "\"cpu_fallback\""),
        (AttemptDisposition::PartialNative, "\"partial_native\""),
        (AttemptDisposition::Native, "\"native\""),
    ];
    for (disposition, expected) in dispositions {
        assert_eq!(serde_json::to_string(&disposition).expect("serialize"), expected);
    }
}

#[test]
fn correlation_limits_are_pinned_public_constants() {
    assert_eq!(MAX_ATTEMPT_QUERY_ID_BYTES, 256);
    assert_eq!(MAX_ATTEMPT_TRACE_CONTEXT_BYTES, 1024);
}

#[test]
fn query_attempt_report_serde_round_trips_with_stable_field_names() {
    use nexus_query_engine::observability::{
        AttemptAdmissionFact, AttemptAdmissionWaitFact, AttemptCapacityDisposition, AttemptCleanupDisposition,
        AttemptRetryAdvice, AttemptTimestamps, QueryAttemptReport, RuntimeTerminalOutcome,
    };
    let report = QueryAttemptReport {
        schema: QUERY_ATTEMPT_REPORT_SCHEMA_V1.to_string(),
        attempt_id: AttemptId::mint(),
        query_id: Some(AttemptQueryId::new("q-7").expect("query id")),
        trace_context: None,
        disposition: AttemptDisposition::Native,
        outcome: RuntimeTerminalOutcome::Succeeded,
        primary_failure: None,
        retry_advice: AttemptRetryAdvice::no_retry_needed(),
        device_ordinal: Some(1),
        admission: Some(AttemptAdmissionFact {
            admission_sequence: 9,
            admission_wait_ns: 12,
            device_ordinal: 1,
            grant_bytes: 2048,
        }),
        admission_wait: Some(AttemptAdmissionWaitFact {
            submission_sequence: 7,
            overtake_limit: 2,
            overtakes_consumed: 2,
            overtakes_remaining: 0,
            protection_device_ordinal: Some(1),
        }),
        composed_schedule: None,
        timestamps: AttemptTimestamps {
            submitted_ns: 0,
            planning_started_ns: Some(1),
            planning_finished_ns: Some(2),
            admission_requested_ns: Some(3),
            admitted_ns: Some(4),
            physical_execution_started_ns: Some(4),
            gpu_execution_started_ns: Some(5),
            outcome_terminal_ns: 6,
            cleanup_complete_ns: Some(7),
            report_ready_ns: Some(8),
        },
        cleanup_disposition: AttemptCleanupDisposition::Completed,
        cleanup_failure_count: 0,
        cleanup_failure_count_complete: true,
        first_cleanup_failure: None,
        capacity_disposition: AttemptCapacityDisposition::Released,
        final_domain_usage: None,
        health_transition: None,
        native_fault_evidence: None,
        resource_evidence: None,
        observation_lost_records: 0,
        observation_retained_records: 3,
        observation_retained_bytes: 128,
        observation_truncated: false,
        observation_degraded: false,
        unix_epoch_ns: Some(99),
    };
    let json = serde_json::to_value(&report).expect("serialize report");
    for field in [
        "schema",
        "attempt_id",
        "query_id",
        "trace_context",
        "disposition",
        "outcome",
        "primary_failure",
        "retry_advice",
        "device_ordinal",
        "admission",
        "admission_wait",
        "composed_schedule",
        "timestamps",
        "cleanup_disposition",
        "cleanup_failure_count",
        "cleanup_failure_count_complete",
        "first_cleanup_failure",
        "capacity_disposition",
        "final_domain_usage",
        "health_transition",
        "native_fault_evidence",
        "resource_evidence",
        "observation_lost_records",
        "observation_retained_records",
        "observation_retained_bytes",
        "observation_truncated",
        "observation_degraded",
        "unix_epoch_ns",
    ] {
        assert!(json.get(field).is_some(), "report field {field} is part of the v1 envelope");
    }
    assert_eq!(json["schema"], QUERY_ATTEMPT_REPORT_SCHEMA_V1);
    assert_eq!(json["disposition"], "native");
    assert_eq!(json["outcome"]["kind"], "succeeded");
    let decoded: QueryAttemptReport = serde_json::from_value(json).expect("deserialize report");
    assert_eq!(decoded, report);
    assert_eq!(decoded.total_ns(), Some(8));
    assert_eq!(decoded.planning_ns(), Some(1));
    assert_eq!(decoded.queue_wait_ns(), Some(1));
    assert_eq!(decoded.physical_execution_ns(), Some(2));
    assert_eq!(decoded.cleanup_ns(), Some(1));
}

#[test]
fn effective_queue_expiry_is_the_earlier_deadline() {
    let submitted = Instant::now();
    let max_wait = Duration::from_millis(500);
    assert_eq!(effective_queue_expiry(submitted, None, max_wait), submitted + max_wait);
    let earlier = submitted + Duration::from_millis(10);
    assert_eq!(effective_queue_expiry(submitted, Some(earlier), max_wait), earlier);
    let later = submitted + Duration::from_secs(60);
    assert_eq!(effective_queue_expiry(submitted, Some(later), max_wait), submitted + max_wait);
}

#[test]
fn cancellation_propagation_state_covers_every_later_boundary() {
    let submitted = Instant::now();
    let max_wait = Duration::from_millis(500);
    let boundaries = [
        AttemptCancellationBoundary::Planning,
        AttemptCancellationBoundary::AdmissionQueue,
        AttemptCancellationBoundary::AdmissionTransaction,
        AttemptCancellationBoundary::StreamAcquisition,
        AttemptCancellationBoundary::WorkerJob,
        AttemptCancellationBoundary::GpuExecutionSafeBoundary,
        AttemptCancellationBoundary::ResultDrain,
        AttemptCancellationBoundary::Cleanup,
    ];
    for boundary in boundaries {
        let cancelled = cancellation_state_at(boundary, true, None, submitted, max_wait, submitted);
        if matches!(boundary, AttemptCancellationBoundary::Cleanup) {
            assert_eq!(cancelled, BoundaryCancellationState::NotRequested, "cleanup runs to completion");
        } else {
            assert_eq!(cancelled, BoundaryCancellationState::CancellationRequested);
        }
    }
    // Queue expiry uses the configured maximum wait; other boundaries only
    // observe the caller deadline.
    let after_max_wait = submitted + Duration::from_secs(1);
    assert!(matches!(
        cancellation_state_at(
            AttemptCancellationBoundary::AdmissionQueue,
            false,
            None,
            submitted,
            max_wait,
            after_max_wait
        ),
        BoundaryCancellationState::DeadlineExpired { .. }
    ));
    assert_eq!(
        cancellation_state_at(
            AttemptCancellationBoundary::GpuExecutionSafeBoundary,
            false,
            None,
            submitted,
            max_wait,
            after_max_wait
        ),
        BoundaryCancellationState::NotRequested
    );
}
