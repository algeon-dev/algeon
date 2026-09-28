use query_planning::error::{AllocationSource, QueryError, QueryErrorCode};

use super::*;

fn failure(code: AttemptFailureCode, facts: Vec<AttemptFailureFact>) -> AttemptFailureSummary {
    AttemptFailureSummary {
        origin: AttemptFailureOrigin::Native,
        code,
        status: AttemptFailureStatus::Temporary,
        operation: AttemptFailureOperation::Execution,
        text: None,
        facts,
    }
}

fn reason(value: AttemptFailureFactValue) -> AttemptFailureFact {
    AttemptFailureFact { key: AttemptFailureFactKey::Reason, value }
}

#[test]
fn native_failure_summary_projects_only_known_cugraph_allocation_sources_in_key_order() {
    let error = QueryError::new(
        QueryErrorCode::CudfResourceExhausted,
        "query_engine.test.allocation_source",
        "cuGraph allocation failed",
    )
    .with_fact("reason", "backend_closing")
    .with_allocation_source(AllocationSource::RmmBadAlloc);

    let summary = AttemptFailureSummary::from_native_error(&error, AttemptFailureOperation::Execution);
    assert_eq!(
        summary.facts(),
        &[
            reason(AttemptFailureFactValue::BackendClosing),
            AttemptFailureFact {
                key: AttemptFailureFactKey::AllocationSource,
                value: AttemptFailureFactValue::CugraphRmmBadAlloc,
            },
        ]
    );

    for (source, expected) in [
        (AllocationSource::RmmOutOfMemory, AttemptFailureFactValue::CugraphRmmOutOfMemory),
        (AllocationSource::StdBadAlloc, AttemptFailureFactValue::CugraphStdBadAlloc),
    ] {
        let error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "query_engine.test.allocation_source",
            "cuGraph allocation failed",
        )
        .with_allocation_source(source);
        assert_eq!(
            AttemptFailureSummary::from_native_error(&error, AttemptFailureOperation::Execution).facts(),
            &[AttemptFailureFact { key: AttemptFailureFactKey::AllocationSource, value: expected }]
        );
    }

    let unknown = QueryError::new(
        QueryErrorCode::CudfResourceExhausted,
        "query_engine.test.allocation_source",
        "cuGraph allocation failed",
    )
    .with_allocation_source(AllocationSource::Unknown);
    assert!(AttemptFailureSummary::from_native_error(&unknown, AttemptFailureOperation::Execution).facts().is_empty());

    let untyped_fact = QueryError::new(
        QueryErrorCode::CudfResourceExhausted,
        "query_engine.test.allocation_source",
        "cuGraph allocation failed",
    )
    .with_fact("cugraph_allocation_source", "rmm_bad_alloc");
    assert!(
        AttemptFailureSummary::from_native_error(&untyped_fact, AttemptFailureOperation::Execution).facts().is_empty(),
        "free-form diagnostic facts must not mint sealed allocation provenance"
    );
}

fn exact_allocator_evidence(source: AttemptAllocatorEventSource) -> AttemptResourceEvidence {
    let mut evidence = AttemptResourceEvidence::unavailable();
    evidence.allocator_event = AttemptEvidenceField::attempt_exact(AttemptAllocatorEvent {
        sequence: 1,
        monotonic_time_ns: AttemptEvidenceField::unavailable(),
        requested_bytes: 1024,
        source,
        action: AttemptEvidenceField::unavailable(),
        callback_attempt: 0,
        trim_retry_count: 0,
        pool_usage: AttemptEvidenceField::unavailable(),
    });
    evidence
}

fn device_local_fault_evidence() -> AttemptNativeFaultEvidence {
    AttemptNativeFaultEvidence {
        isolation: AttemptFaultIsolation::DeviceLocal,
        teardown_health: AttemptTeardownHealth::Completed,
        device_health: Some(AttemptDeviceHealth {
            status: AttemptDeviceHealthStatus::Unhealthy,
            fault_count: 1,
            rejected_operation_count: 1,
            quarantined_teardown_count: 0,
        }),
        first_fault: Some(AttemptNativeFault {
            kind: AttemptNativeFaultKind::FatalDeviceOperation,
            status: Some(AttemptNativeStatus::Cuda(700)),
        }),
    }
}

fn advice_for(
    outcome: &RuntimeTerminalOutcome,
    primary_failure: Option<&AttemptFailureSummary>,
    capacity_disposition: AttemptCapacityDisposition,
    health_transition: Option<AttemptHealthTransition>,
    native_fault_evidence: Option<AttemptNativeFaultEvidence>,
    resource_evidence: Option<AttemptResourceEvidence>,
    alternate_device_eligibility: AttemptAlternateDeviceEligibility,
) -> AttemptRetryAdvice {
    AttemptRetryAdvice::from_sealed_attempt(
        outcome,
        primary_failure,
        capacity_disposition,
        health_transition,
        native_fault_evidence,
        resource_evidence,
        alternate_device_eligibility,
    )
}

#[test]
fn attempt_failure_text_is_utf8_safe_and_bounded() {
    let multibyte = "é".repeat(MAX_ATTEMPT_FAILURE_TEXT_BYTES);
    let text = AttemptFailureText::from_bytes(multibyte.as_bytes());
    assert!(text.value().len() <= MAX_ATTEMPT_FAILURE_TEXT_BYTES);
    assert!(text.truncated());
    assert!(text.value().is_char_boundary(text.value().len()));

    let invalid = AttemptFailureText::from_bytes(&[b'o', b'k', 0xff, b'!']);
    assert!(invalid.invalid_utf8());
    assert!(invalid.value().contains('\u{fffd}'));
    assert!(invalid.value().len() <= MAX_ATTEMPT_FAILURE_TEXT_BYTES);
}

#[test]
fn attempt_failure_text_redacts_uris_credentials_and_sql_shapes() {
    for sensitive in [
        "reader failed at s3://user:password@example.invalid/bucket/object",
        "authorization: bearer secret-value",
        "SELECT* FROM private_table",
    ] {
        let text = AttemptFailureText::from_bytes(sensitive.as_bytes());
        assert_eq!(text.value(), "<redacted>");
        assert!(text.redacted());
    }
}

#[test]
fn failure_summary_deserialization_enforces_all_bounds_before_publication() {
    let oversized_text = serde_json::json!({
        "value": "x".repeat(MAX_ATTEMPT_FAILURE_TEXT_BYTES + 1),
        "truncated": false,
        "redacted": false,
        "invalid_utf8": false,
    });
    assert!(serde_json::from_value::<AttemptFailureText>(oversized_text).is_err());

    let fact = serde_json::json!({ "key": "reason", "value": "backend_closing" });
    let too_many_facts = serde_json::json!({
        "origin": "native",
        "code": "internal",
        "status": "permanent",
        "operation": "execution",
        "facts": vec![fact; MAX_ATTEMPT_FAILURE_FACTS + 1],
    });
    assert!(serde_json::from_value::<AttemptFailureSummary>(too_many_facts).is_err());

    let raw_uri = serde_json::json!({
        "value": "failed at s3://private-bucket/object",
        "truncated": false,
        "redacted": false,
        "invalid_utf8": false,
    });
    assert!(serde_json::from_value::<AttemptFailureText>(raw_uri).is_err());

    let falsely_redacted = serde_json::json!({
        "value": "authorization: bearer secret-value",
        "truncated": false,
        "redacted": true,
        "invalid_utf8": false,
    });
    assert!(serde_json::from_value::<AttemptFailureText>(falsely_redacted).is_err());
}

#[test]
fn resource_evidence_preserves_explicit_unavailable_provenance_through_serde() {
    let mut evidence = AttemptResourceEvidence::unavailable();
    evidence.grant_cap_bytes = AttemptEvidenceField::attempt_exact(4096);
    evidence.ledger_sequence = AttemptEvidenceField::device_snapshot(17);
    evidence.outcome_terminal_usage = AttemptEvidenceField::sampled_at_terminal_boundary(AttemptTimedDomainUsage {
        sampled_ns: 23,
        usage: AttemptFinalDomainUsage {
            domain_id: 7,
            active: true,
            active_members: 1,
            ceiling_bytes: 4096,
            allocation_limit_bytes: 4096,
            current_bytes: 32,
            peak_bytes: 32,
            total_bytes: 32,
            current_aligned_bytes: 32,
            peak_aligned_bytes: 32,
            total_allocations: 1,
            current_allocations: 1,
            peak_allocations: 1,
            failed_allocation_count: 0,
            largest_failed_allocation_bytes: 0,
        },
        pool_usage: Some(AttemptAllocatorPoolUsage {
            reserved_current_bytes: 64,
            reserved_high_bytes: 96,
            used_current_bytes: 32,
            used_high_bytes: 48,
        }),
    });

    let encoded = serde_json::to_value(evidence).expect("resource evidence serializes");
    assert_eq!(encoded["grant_cap_bytes"]["provenance"], "attempt_exact");
    assert_eq!(encoded["ledger_sequence"]["provenance"], "device_snapshot");
    assert_eq!(encoded["outcome_terminal_usage"]["provenance"], "sampled_at_terminal_boundary");
    assert_eq!(encoded["outcome_terminal_usage"]["value"]["pool_usage"]["reserved_high_bytes"], 96);
    assert_eq!(encoded["allocator_event"]["provenance"], "unavailable");
    assert!(encoded["allocator_event"]["value"].is_null());
    assert_eq!(
        serde_json::from_value::<AttemptResourceEvidence>(encoded).expect("resource evidence deserializes"),
        evidence
    );
}

#[test]
fn sealed_retry_advice_is_exhaustive_and_evidence_gated() {
    let no_commit =
        (AttemptCapacityDisposition::NotAdmitted, None, None, None, AttemptAlternateDeviceEligibility::Unavailable);
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::Succeeded,
            None,
            no_commit.0,
            no_commit.1,
            no_commit.2,
            no_commit.3,
            no_commit.4,
        ),
        AttemptRetryAdvice::no_retry_needed()
    );
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::Cancelled,
            None,
            no_commit.0,
            no_commit.1,
            no_commit.2,
            no_commit.3,
            no_commit.4,
        ),
        AttemptRetryAdvice::do_not_retry()
    );

    let unsupported = failure(AttemptFailureCode::UnsupportedOperator, Vec::new());
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&unsupported),
            Some(&unsupported),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            no_commit.3,
            no_commit.4,
        ),
        AttemptRetryAdvice::change_configuration()
    );

    let query_cap_exceeded = failure(AttemptFailureCode::QueryMemoryLimitExceeded, Vec::new());
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&query_cap_exceeded),
            Some(&query_cap_exceeded),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            no_commit.3,
            no_commit.4,
        ),
        AttemptRetryAdvice::change_configuration()
    );

    let queue =
        failure(AttemptFailureCode::ServiceOverloaded, vec![reason(AttemptFailureFactValue::AdmissionQueueFull)]);
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&queue),
            Some(&queue),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            no_commit.3,
            no_commit.4,
        ),
        AttemptRetryAdvice::backoff()
    );
    let impossible_floor = failure(
        AttemptFailureCode::ServiceOverloaded,
        vec![reason(AttemptFailureFactValue::RequiredFloorExceedsDeviceCapacity)],
    );
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&impossible_floor),
            Some(&impossible_floor),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            no_commit.3,
            no_commit.4,
        ),
        AttemptRetryAdvice::change_configuration()
    );

    let resource_exhausted = failure(AttemptFailureCode::CudfResourceExhausted, Vec::new());
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&resource_exhausted),
            Some(&resource_exhausted),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            Some(exact_allocator_evidence(AttemptAllocatorEventSource::ReservationLimit)),
            no_commit.4,
        ),
        AttemptRetryAdvice::change_configuration()
    );
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&resource_exhausted),
            Some(&resource_exhausted),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            Some(exact_allocator_evidence(AttemptAllocatorEventSource::AllocationLimit)),
            no_commit.4,
        ),
        AttemptRetryAdvice::change_configuration(),
        "an exact cache allocation limit cannot resolve without configuration change"
    );
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&resource_exhausted),
            Some(&resource_exhausted),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            Some(exact_allocator_evidence(AttemptAllocatorEventSource::Physical)),
            no_commit.4,
        ),
        AttemptRetryAdvice::indeterminate(),
        "a physical allocator failure alone does not prove capacity will become available"
    );
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&resource_exhausted),
            Some(&resource_exhausted),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            None,
            no_commit.4,
        ),
        AttemptRetryAdvice::indeterminate(),
        "OOM status without exact allocator evidence cannot recommend a retry"
    );

    let temporary = failure(AttemptFailureCode::CudfTemporaryFailure, Vec::new());
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&temporary),
            Some(&temporary),
            no_commit.0,
            no_commit.1,
            no_commit.2,
            no_commit.3,
            no_commit.4,
        ),
        AttemptRetryAdvice::indeterminate(),
        "a temporary code alone is not retry evidence"
    );

    let unknown_isolation = AttemptNativeFaultEvidence {
        isolation: AttemptFaultIsolation::Unknown,
        teardown_health: AttemptTeardownHealth::CallbackAbandoned,
        device_health: None,
        first_fault: None,
    };
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::Succeeded,
            None,
            AttemptCapacityDisposition::RetainedQuarantine,
            Some(AttemptHealthTransition::BackendUnhealthy),
            Some(unknown_isolation),
            None,
            AttemptAlternateDeviceEligibility::NotApplicable,
        ),
        AttemptRetryAdvice::recover_backend(),
        "unknown isolation is a backend recovery requirement even when cleanup did not replace the outcome"
    );

    let local_fault = device_local_fault_evidence();
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&temporary),
            Some(&temporary),
            AttemptCapacityDisposition::RetainedQuarantine,
            Some(AttemptHealthTransition::DeviceQuarantined),
            Some(local_fault),
            None,
            AttemptAlternateDeviceEligibility::Eligible,
        ),
        AttemptRetryAdvice::retry_another_eligible_device()
    );
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&temporary),
            Some(&temporary),
            AttemptCapacityDisposition::RetainedQuarantine,
            Some(AttemptHealthTransition::DeviceQuarantined),
            Some(local_fault),
            None,
            AttemptAlternateDeviceEligibility::NoEligibleAlternate,
        ),
        AttemptRetryAdvice::do_not_retry()
    );
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&temporary),
            Some(&temporary),
            AttemptCapacityDisposition::RetainedQuarantine,
            Some(AttemptHealthTransition::DeviceQuarantined),
            Some(local_fault),
            None,
            AttemptAlternateDeviceEligibility::Unavailable,
        ),
        AttemptRetryAdvice::indeterminate()
    );

    let stream_synchronization_fault = AttemptNativeFaultEvidence {
        isolation: AttemptFaultIsolation::DeviceLocal,
        teardown_health: AttemptTeardownHealth::StreamSynchronizationFailed,
        device_health: Some(AttemptDeviceHealth {
            status: AttemptDeviceHealthStatus::Healthy,
            fault_count: 0,
            rejected_operation_count: 0,
            quarantined_teardown_count: 1,
        }),
        first_fault: None,
    };
    assert_eq!(
        advice_for(
            &RuntimeTerminalOutcome::failed(&temporary),
            Some(&temporary),
            AttemptCapacityDisposition::RetainedQuarantine,
            Some(AttemptHealthTransition::DeviceQuarantined),
            Some(stream_synchronization_fault),
            None,
            AttemptAlternateDeviceEligibility::Eligible,
        ),
        AttemptRetryAdvice::retry_another_eligible_device(),
        "a stream-synchronization failure is itself proven device-local cleanup evidence"
    );
}

#[test]
fn retry_advice_deserialization_rejects_inconsistent_pairs() {
    let error = serde_json::from_str::<AttemptRetryAdvice>(
        r#"{"recovery":"retry_another_eligible_device","prerequisite":"none"}"#,
    )
    .expect_err("retry advice must reject an impossible recovery/prerequisite pair");

    assert!(error.to_string().contains("retry recovery and prerequisite are inconsistent"));
}
