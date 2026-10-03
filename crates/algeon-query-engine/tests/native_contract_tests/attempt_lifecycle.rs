//! Public execution-attempt lifecycle contracts that need no CUDA device.
//!
//! These tests pin the attempt identity, correlation
//! validation, and report envelope through the crate's public API only.

use query_planning::error::QueryErrorCode;
use query_runtime::observability::{
    AttemptDisposition, AttemptId, AttemptQueryId, AttemptTraceContext, MAX_ATTEMPT_QUERY_ID_BYTES,
    MAX_ATTEMPT_TRACE_CONTEXT_BYTES, QUERY_ATTEMPT_REPORT_SCHEMA,
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
    assert_eq!(QUERY_ATTEMPT_REPORT_SCHEMA, "query_engine.query_attempt_report");
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
    use query_runtime::observability::{
        AttemptAdmissionFact, AttemptAdmissionWaitFact, AttemptCapacityDisposition, AttemptCleanupDisposition,
        AttemptRetryAdvice, AttemptTimestamps, QueryAttemptReport, RuntimeTerminalOutcome,
    };
    let report = QueryAttemptReport {
        schema: QUERY_ATTEMPT_REPORT_SCHEMA.to_string(),
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
        admission_wait: Some(AttemptAdmissionWaitFact { submission_sequence: 7 }),
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
        assert!(json.get(field).is_some(), "report field {field} is part of the envelope");
    }
    assert_eq!(json["schema"], QUERY_ATTEMPT_REPORT_SCHEMA);
    assert_eq!(json["disposition"], "native");
    assert_eq!(json["outcome"]["kind"], "succeeded");
    assert_eq!(json["admission_wait"], serde_json::json!({ "submission_sequence": 7 }));
    let decoded: QueryAttemptReport = serde_json::from_value(json).expect("deserialize report");
    assert_eq!(decoded, report);
    assert_eq!(decoded.total_ns(), Some(8));
    assert_eq!(decoded.planning_ns(), Some(1));
    assert_eq!(decoded.post_planning_pre_admission_ns(), Some(1));
    assert_eq!(decoded.queue_wait_ns(), Some(1));
    assert_eq!(decoded.admitted_to_physical_execution_ns(), Some(0));
    assert_eq!(decoded.physical_to_gpu_dispatch_ns(), Some(1));
    assert_eq!(decoded.admitted_to_gpu_dispatch_ns(), Some(1));
    assert_eq!(decoded.submitted_to_gpu_dispatch_ns(), Some(5));
    assert_eq!(decoded.gpu_dispatch_to_terminal_ns(), Some(1));
    assert_eq!(decoded.physical_execution_ns(), Some(2));
    assert_eq!(decoded.cleanup_ns(), Some(1));
}
