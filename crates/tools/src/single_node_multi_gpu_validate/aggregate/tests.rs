use std::collections::BTreeMap;

use super::*;

fn device(ordinal: usize, capacity_bytes: u64, max_active_attempts: usize) -> DeviceConfiguration {
    DeviceConfiguration {
        ordinal,
        model_name: Some("NVIDIA L40S".to_string()),
        uuid: Some(format!("GPU-{ordinal}")),
        physical_total_bytes: Some(46_068 * 1024 * 1024),
        capacity_bytes,
        cache_cap_bytes: 0,
        install_telemetry: true,
        max_active_attempts,
        stream_pool_size: 8,
        worker_count: 2,
        lane_count: 2,
        ownership: "whole_device_exclusive".to_string(),
        device_health_generation: 0,
        starting_health_open: true,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "fixed measurement-matrix expansion helper; each parameter is one axis of the validation matrix"
)]
fn attempt(
    sequence: usize,
    ordinal: usize,
    grant_id: Option<u64>,
    domain_id: Option<u64>,
    submitted_ns: u64,
    outcome_terminal_ns: u64,
    total_ns: u64,
    phase: SchedulePhase,
) -> AttemptObservation {
    AttemptObservation {
        sequence,
        phase,
        step_index: 0,
        query_kind: super::super::profile::FixtureKind::Tpch,
        query_label: "q01".to_string(),
        graph_workload: false,
        placement_required: true,
        attempt_id: format!("attempt-{sequence:032x}"),
        disposition: "Native".to_string(),
        outcome: "succeeded".to_string(),
        failure: None,
        terminal_evidence: terminal_evidence(),
        succeeded: true,
        reached_gpu: true,
        graph_claim_calibration: None,
        composed_schedule: None,
        device_ordinal: Some(ordinal),
        grant_bytes: Some(1024),
        admission_sequence: Some(sequence as u64),
        admission_wait_ns: Some(10),
        admission_requested_ns: Some(submitted_ns),
        admitted_ns: Some(submitted_ns.saturating_add(1)),
        admission_wait: None,
        grant_id,
        domain_id,
        domain_pool_usage: None,
        total_ns: Some(total_ns),
        queue_wait_ns: Some(5),
        physical_execution_ns: Some(total_ns / 2),
        submitted_ns,
        outcome_terminal_ns,
    }
}

fn terminal_evidence() -> AttemptTerminalEvidence {
    AttemptTerminalEvidence {
        primary_failure: None,
        cleanup_disposition: serde_json::json!("completed"),
        cleanup_failure_count: 0,
        cleanup_failure_count_complete: true,
        first_cleanup_failure: None,
        capacity_disposition: serde_json::json!("released"),
        health_transition: Some(serde_json::json!("healthy_reuse")),
        native_fault_evidence: None,
        resource_evidence: None,
        retry_advice: serde_json::json!({ "recovery": "do_not_retry", "prerequisite": "none" }),
    }
}

fn domain_exhaustion_failure() -> AttemptObservation {
    let mut failure = attempt(0, 0, Some(1), Some(10), 0, 1_000, 1_000, SchedulePhase::Measurement);
    failure.outcome = "failed".to_string();
    failure.succeeded = false;
    failure.failure = Some(AttemptFailureClass {
        origin: "Native".to_string(),
        code: "CudfResourceExhausted".to_string(),
        status: "Temporary".to_string(),
        operation: "Execution".to_string(),
    });
    failure.terminal_evidence.native_fault_evidence = Some(serde_json::json!({
        "device_health": {
            "status": "healthy",
            "quarantined_teardown_count": 0,
        }
    }));
    failure.terminal_evidence.resource_evidence = Some(serde_json::json!({
        "allocator_event": {
            "provenance": "attempt_exact",
            "value": {"source": "reservation_limit"},
        }
    }));
    failure
}

fn physical_allocator_pressure_failure() -> AttemptObservation {
    let mut failure = domain_exhaustion_failure();
    failure.terminal_evidence.resource_evidence = Some(serde_json::json!({
        "allocator_event": {
            "provenance": "attempt_exact",
            "value": {
                "source": "physical",
                "requested_bytes": 1024,
            },
        },
        "grant_cap_bytes": {"provenance": "attempt_exact", "value": 2048},
        "domain_cap_bytes": {"provenance": "attempt_exact", "value": 2048},
        "managed_capacity_bytes": {"provenance": "attempt_exact", "value": 4096},
    }));
    failure
}

fn physical_pressure_facts(released: bool) -> PhysicalPressureLeaseFacts {
    PhysicalPressureLeaseFacts {
        device_ordinal: 0,
        method: "cuda_on_sidecar_fixed_chunks",
        free_before_bytes: 4096,
        free_under_pressure_bytes: 512,
        bytes_held: 3584,
        reserve_target_bytes: 512,
        free_after_release_bytes: released.then_some(4096),
        recovered_bytes: released.then_some(3584),
    }
}

fn quarantine_evidence(target_ordinal: usize, observations: &[AttemptObservation]) -> QuarantineFallbackEvidence {
    QuarantineFallbackEvidence::from_barrier(
        target_ordinal,
        10,
        7,
        1,
        0,
        20,
        1,
        true,
        true,
        &observations.iter().map(|observation| observation.sequence).collect(),
        true,
        None,
        observations,
    )
}

fn occupancy(peak_jobs: usize) -> BTreeMap<usize, DeviceOccupancyPeaks> {
    BTreeMap::from([(
        0,
        DeviceOccupancyPeaks {
            peak_active_attempts: 2,
            peak_active_query_jobs: peak_jobs,
            peak_live_grants_bytes: 2048,
            peak_resident_owner_bytes: 4096,
            allocation_failure_count: 0,
            min_root_streams_available: 6,
            min_root_worker_slots_available: 1,
        },
    )])
}

#[test]
fn same_device_evidence_requires_distinct_grants_domains_overlap_and_witnessed_occupancy() {
    let observations = vec![
        attempt(0, 0, Some(10), Some(100), 0, 2_000, 2_000, SchedulePhase::Measurement),
        attempt(1, 0, Some(11), Some(101), 1_000, 3_000, 2_000, SchedulePhase::Measurement),
    ];
    let aggregate = MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
    let status = |name: &str| {
        aggregate.scenarios.same_device.iter().find(|clause| clause.clause == name).map(|clause| clause.status).unwrap()
    };
    assert_eq!(status("multiple_admitted_attempts_on_one_device"), ClauseStatus::Evidenced);
    assert_eq!(status("distinct_grants"), ClauseStatus::Evidenced);
    assert_eq!(status("distinct_domains"), ClauseStatus::Evidenced);
    assert_eq!(status("concurrent_attempt_intervals_on_one_device"), ClauseStatus::Evidenced);
    assert_eq!(status("witnessed_concurrent_device_occupancy"), ClauseStatus::Evidenced);
    // Nsight overlap is never claimed by this runner.
    assert_eq!(status("nsight_confirmed_overlapping_cuda_ranges"), ClauseStatus::NotEvidenced);
    assert!(scenario_clauses_pass(&aggregate.scenarios));
}

#[test]
fn a_shared_or_unavailable_identity_fails_rather_than_shrinking_the_sample() {
    let shared = vec![
        attempt(0, 0, Some(10), Some(100), 0, 2_000, 2_000, SchedulePhase::Measurement),
        attempt(1, 0, Some(10), Some(101), 1_000, 3_000, 2_000, SchedulePhase::Measurement),
    ];
    let aggregate = MeasuredAggregate::reduce(&shared, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
    let grants = aggregate.scenarios.same_device.iter().find(|c| c.clause == "distinct_grants").unwrap();
    assert_eq!(grants.status, ClauseStatus::Failed);
    assert!(!scenario_clauses_pass(&aggregate.scenarios));

    let unavailable = vec![
        attempt(0, 0, None, Some(100), 0, 2_000, 2_000, SchedulePhase::Measurement),
        attempt(1, 0, Some(11), Some(101), 1_000, 3_000, 2_000, SchedulePhase::Measurement),
    ];
    let aggregate = MeasuredAggregate::reduce(&unavailable, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
    let grants = aggregate.scenarios.same_device.iter().find(|c| c.clause == "distinct_grants").unwrap();
    assert_eq!(grants.status, ClauseStatus::Failed);
    assert!(grants.detail.contains("carried no identity"), "{}", grants.detail);
}

#[test]
fn serialized_occupancy_is_labelled_profile_derived_and_never_claims_gpu_overlap() {
    let observations = vec![attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
    let aggregate = MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
    let value = serde_json::to_value(&aggregate.scenarios.logical_slot_occupancy).unwrap();
    assert_eq!(value["basis"], "profile_derived_logical_slots");
    assert_eq!(value["proves_gpu_overlap"], false);
    assert_eq!(value["stream_identity_source"], "nsight_systems_trace");
    // The word "stream" only appears on the min-availability counter, never
    // as an identity claim.
    assert!(value["per_device"]["0"]["min_root_streams_available"].is_number());
}

#[test]
fn four_device_progress_and_heterogeneity_are_asserted_from_resolved_envelopes() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2), device(2, 2 << 30, 2), device(3, 1 << 30, 4)];
    let observations = (0..4)
        .map(|ordinal| {
            attempt(
                ordinal,
                ordinal,
                Some(ordinal as u64 + 1),
                Some(ordinal as u64 + 100),
                0,
                1_000,
                1_000,
                SchedulePhase::Measurement,
            )
        })
        .collect::<Vec<_>>();
    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
    let status = |name: &str| {
        aggregate
            .scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == name)
            .map(|clause| clause.status)
            .unwrap()
    };
    assert_eq!(status("one_ordinal_per_attempt"), ClauseStatus::Evidenced);
    assert_eq!(status("progress_on_all_eligible_devices"), ClauseStatus::Evidenced);
    assert_eq!(status("heterogeneous_ordinal_profiles"), ClauseStatus::Evidenced);
    // Quarantine fallback is a declared gap, not a pass and not a failure.
    assert_eq!(status("device_block_or_quarantine_fallback"), ClauseStatus::NotEvidenced);
    assert!(aggregate.placement.iter().all(|device| device.made_progress));
    assert!(scenario_clauses_pass(&aggregate.scenarios));
}

#[test]
fn explicit_pre_admission_cancellation_is_not_a_missing_placement() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
    let mut cancelled = attempt(0, 0, Some(1), Some(10), 0, 10, 10, SchedulePhase::Measurement);
    cancelled.outcome = "cancelled".to_string();
    cancelled.succeeded = false;
    cancelled.reached_gpu = false;
    cancelled.device_ordinal = None;
    cancelled.grant_bytes = None;
    cancelled.grant_id = None;
    cancelled.domain_id = None;
    let observations = vec![
        cancelled,
        attempt(1, 0, Some(2), Some(11), 0, 10, 10, SchedulePhase::Measurement),
        attempt(2, 1, Some(3), Some(12), 0, 10, 10, SchedulePhase::Measurement),
    ];

    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
    let clause =
        aggregate.scenarios.multiple_devices.iter().find(|clause| clause.clause == "one_ordinal_per_attempt").unwrap();
    assert_eq!(clause.status, ClauseStatus::Evidenced);
    assert!(clause.detail.contains("1 explicit pre-admission cancellations"));
}

#[test]
fn not_admitted_attempt_is_not_a_missing_placement() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
    let mut rejected = attempt(0, 0, None, None, 0, 10, 10, SchedulePhase::Measurement);
    rejected.outcome = "failed".to_string();
    rejected.terminal_evidence.capacity_disposition = serde_json::json!("not_admitted");
    let observations = vec![
        rejected,
        attempt(1, 0, Some(2), Some(11), 0, 10, 10, SchedulePhase::Measurement),
        attempt(2, 1, Some(3), Some(12), 0, 10, 10, SchedulePhase::Measurement),
    ];

    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
    let clause =
        aggregate.scenarios.multiple_devices.iter().find(|clause| clause.clause == "one_ordinal_per_attempt").unwrap();
    assert_eq!(clause.status, ClauseStatus::Evidenced);
    assert!(clause.detail.contains("1 not-admitted attempts"));
}

#[test]
fn explicit_nonexecuting_capability_probe_is_not_a_missing_placement() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
    let mut capability_probe = attempt(0, 0, None, None, 0, 10, 10, SchedulePhase::Measurement);
    capability_probe.graph_workload = true;
    capability_probe.placement_required = false;
    capability_probe.outcome = "failed".to_string();
    capability_probe.succeeded = false;
    capability_probe.reached_gpu = false;
    capability_probe.device_ordinal = None;
    let observations = vec![
        capability_probe,
        attempt(1, 0, Some(2), Some(11), 0, 10, 10, SchedulePhase::Measurement),
        attempt(2, 1, Some(3), Some(12), 0, 10, 10, SchedulePhase::Measurement),
    ];

    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
    let clause =
        aggregate.scenarios.multiple_devices.iter().find(|clause| clause.clause == "one_ordinal_per_attempt").unwrap();
    assert_eq!(clause.status, ClauseStatus::Evidenced);
    assert!(clause.detail.contains("1 non-executing capability probes"));
}

#[test]
fn controlled_quarantine_evidence_requires_a_closed_target_and_healthy_fallback() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
    let observations = vec![attempt(0, 1, Some(10), Some(100), 0, 1_000, 1_000, SchedulePhase::Measurement)];
    let evidence = quarantine_evidence(0, &observations);
    let aggregate = MeasuredAggregate::reduce_with_quarantine(
        &observations,
        &devices,
        &BTreeMap::new(),
        0,
        1_000_000_000,
        Some(evidence),
    );
    let clause = aggregate
        .scenarios
        .multiple_devices
        .iter()
        .find(|clause| clause.clause == "device_block_or_quarantine_fallback")
        .expect("fallback clause is present");
    assert_eq!(clause.status, ClauseStatus::Evidenced);
    assert_eq!(aggregate.quarantine_fallback.as_ref().map(|evidence| evidence.target_ordinal), Some(0));
}

#[test]
fn controlled_quarantine_evidence_rejects_a_post_injection_target_placement() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
    let observations = vec![attempt(0, 0, Some(10), Some(100), 0, 1_000, 1_000, SchedulePhase::Measurement)];
    let evidence = quarantine_evidence(0, &observations);
    let aggregate = MeasuredAggregate::reduce_with_quarantine(
        &observations,
        &devices,
        &BTreeMap::new(),
        0,
        1_000_000_000,
        Some(evidence),
    );
    let clause = aggregate
        .scenarios
        .multiple_devices
        .iter()
        .find(|clause| clause.clause == "device_block_or_quarantine_fallback")
        .expect("fallback clause is present");
    assert_eq!(clause.status, ClauseStatus::Failed);
    assert!(!scenario_clauses_pass(&aggregate.scenarios));
}

#[test]
fn domain_exhaustion_requires_exact_reservation_limit_cleanup_and_healthy_reuse() {
    let failure = domain_exhaustion_failure();
    let recovery = attempt(1, 0, Some(2), Some(11), 1_001, 2_000, 999, SchedulePhase::Measurement);
    let evidence = DomainExhaustionEvidence::from_observations(&[failure.clone(), recovery]);

    assert!(evidence.is_evidenced());
    assert_eq!(evidence.matched_native_resource_exhaustion_reports, 1);
    assert_eq!(evidence.failure_sequence, Some(0));
    assert_eq!(evidence.failure_device_ordinal, Some(0));
    assert_eq!(evidence.recovery_sequence, Some(1));

    let mut physical = failure;
    physical.terminal_evidence.resource_evidence = Some(serde_json::json!({
        "allocator_event": {
            "provenance": "attempt_exact",
            "value": {"source": "physical"},
        }
    }));
    let evidence = DomainExhaustionEvidence::from_observations(&[physical]);
    assert!(!evidence.is_evidenced());
    assert!(!evidence.allocator_event_attempt_exact_reservation_limit);
    assert!(!evidence.same_device_recovery_completed);
}

#[test]
fn physical_allocator_pressure_requires_sidecar_pressure_exact_physical_event_caps_release_recovery_and_close() {
    let failure = physical_allocator_pressure_failure();
    let mut recovery = attempt(1, 0, Some(2), Some(11), 1_001, 2_000, 999, SchedulePhase::Measurement);
    recovery.query_label = "edge_scan".to_string();
    let evidence = PhysicalAllocatorPressureEvidence::from_observations(
        physical_pressure_facts(false),
        physical_pressure_facts(true),
        true,
        true,
        &[failure.clone(), recovery.clone()],
    )
    .with_final_close_quiescent(true);

    assert!(evidence.is_evidenced());
    assert_eq!(evidence.status, ClauseStatus::Evidenced);
    assert_eq!(evidence.observed_physical_request_bytes, Some(1024));
    assert_eq!(evidence.failure_sequence, Some(0));
    assert_eq!(evidence.recovery_sequence, Some(1));
    let projection = serde_json::to_value(&evidence).unwrap();
    assert_eq!(projection["pressure"]["method"], "cuda_on_sidecar_fixed_chunks");
    assert_eq!(projection["observed_physical_request_bytes"], 1024);
    assert_eq!(projection["status"], "evidenced");

    let mut non_contiguous_pressure = physical_pressure_facts(false);
    let mut non_contiguous_release = physical_pressure_facts(true);
    non_contiguous_pressure.free_under_pressure_bytes = 2_048;
    non_contiguous_release.free_under_pressure_bytes = 2_048;
    let non_contiguous = PhysicalAllocatorPressureEvidence::from_observations(
        non_contiguous_pressure,
        non_contiguous_release,
        true,
        true,
        &[failure.clone(), recovery.clone()],
    )
    .with_final_close_quiescent(true);
    assert!(!non_contiguous.physical_request_exceeds_free_under_pressure);
    assert!(non_contiguous.is_evidenced());

    let close_required = PhysicalAllocatorPressureEvidence::from_observations(
        physical_pressure_facts(false),
        physical_pressure_facts(true),
        true,
        true,
        &[failure.clone(), recovery.clone()],
    );
    assert!(!close_required.is_evidenced());
    assert_eq!(close_required.status, ClauseStatus::Failed);
    assert!(!close_required.final_close_quiescent);

    let mut too_small_grant = failure;
    too_small_grant.terminal_evidence.resource_evidence.as_mut().unwrap()["grant_cap_bytes"]["value"] =
        serde_json::json!(1024);
    let failed = PhysicalAllocatorPressureEvidence::from_observations(
        physical_pressure_facts(false),
        physical_pressure_facts(true),
        true,
        true,
        &[too_small_grant, recovery],
    )
    .with_final_close_quiescent(true);
    assert!(!failed.is_evidenced());
    assert!(!failed.grant_cap_exceeds_physical_request);

    let missing_close = PhysicalAllocatorPressureEvidence::from_observations(
        physical_pressure_facts(false),
        physical_pressure_facts(true),
        true,
        true,
        &[physical_allocator_pressure_failure()],
    );
    assert!(!missing_close.is_evidenced());
    assert!(!missing_close.final_close_quiescent);
    assert!(!missing_close.same_device_recovery_completed);
}

#[test]
fn overload_evidence_requires_a_bounded_full_queue_and_sealed_public_rows() {
    let devices = vec![device(0, 1 << 30, 1), device(1, 1 << 30, 1)];
    let mut protected = attempt(0, 0, Some(1), Some(10), 0, 2_000, 2_000, SchedulePhase::Measurement);
    protected.admission_sequence = Some(4);
    protected.admission_wait_ns = Some(900);
    protected.admission_requested_ns = Some(100);
    protected.admitted_ns = Some(1_000);
    protected.admission_wait = Some(AdmissionWaitEvidence {
        submission_sequence: 21,
        overtake_limit: 0,
        overtakes_consumed: 0,
        overtakes_remaining: 0,
        protection_device_ordinal: None,
    });

    let mut unrelated = attempt(1, 1, Some(2), Some(11), 200, 500, 300, SchedulePhase::Measurement);
    unrelated.admission_sequence = Some(3);
    unrelated.admission_requested_ns = Some(200);
    unrelated.admitted_ns = Some(300);

    let mut observations = vec![
        protected,
        unrelated,
        attempt(2, 0, Some(3), Some(12), 0, 300, 300, SchedulePhase::Measurement),
        attempt(3, 1, Some(4), Some(13), 0, 400, 400, SchedulePhase::Measurement),
    ];
    observations[2].admission_sequence = Some(1);
    observations[3].admission_sequence = Some(2);

    let evidence = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 2, 0, 0, 0);
    assert!(evidence.is_evidenced());
    assert_eq!(evidence.status, ClauseStatus::Evidenced);
    assert_eq!(evidence.configured_overtake_limit, 0);
    assert_eq!(evidence.protection_device_ordinal, None);
    assert_eq!(evidence.unrelated_device_progress_count, 0);
    assert_eq!(evidence.unrelated_device_completion_count, 0);

    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000)
        .with_overload_fairness(evidence.clone());
    let artifact_projection = serde_json::to_value(&aggregate).unwrap();
    assert_eq!(artifact_projection["scenarios"]["overload_fairness"]["status"], "evidenced");
    assert!(artifact_projection["attempts"][0]["admission_wait"]["protection_device_ordinal"].is_null());
    assert_eq!(artifact_projection["attempts"][0]["admission_requested_ns"], 100);
    assert_eq!(artifact_projection["attempts"][0]["admitted_ns"], 1_000);

    let queue_not_full = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 1, 0, 0, 0);
    assert!(!queue_not_full.is_evidenced());
    assert_eq!(queue_not_full.status, ClauseStatus::Failed);

    let queue_exceeded = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 3, 0, 0, 0);
    assert!(!queue_exceeded.is_evidenced());
    assert_eq!(queue_exceeded.status, ClauseStatus::Failed);

    observations[0].admission_wait.as_mut().unwrap().overtakes_remaining = 1;
    let malformed = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 2, 0, 0, 0);
    assert!(!malformed.is_evidenced());
    assert_eq!(malformed.status, ClauseStatus::Failed);
    assert!(!malformed.wait_facts_exact);
}

#[test]
fn an_idle_eligible_device_fails_the_progress_clause() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 2 << 30, 2)];
    let observations = vec![attempt(0, 0, Some(1), Some(2), 0, 1_000, 1_000, SchedulePhase::Measurement)];
    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
    let progress = aggregate
        .scenarios
        .multiple_devices
        .iter()
        .find(|clause| clause.clause == "progress_on_all_eligible_devices")
        .unwrap();
    assert_eq!(progress.status, ClauseStatus::Failed);
    assert!(!scenario_clauses_pass(&aggregate.scenarios));
}

#[test]
fn identical_envelopes_leave_heterogeneity_unevidenced_rather_than_passing() {
    let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
    let observations = (0..2)
        .map(|ordinal| {
            attempt(ordinal, ordinal, Some(ordinal as u64), Some(ordinal as u64), 0, 10, 10, SchedulePhase::Measurement)
        })
        .collect::<Vec<_>>();
    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
    let clause = aggregate
        .scenarios
        .multiple_devices
        .iter()
        .find(|clause| clause.clause == "heterogeneous_ordinal_profiles")
        .unwrap();
    assert_eq!(clause.status, ClauseStatus::NotEvidenced);
    assert!(scenario_clauses_pass(&aggregate.scenarios));
}

#[test]
fn warmup_attempts_are_excluded_from_throughput_and_latency() {
    let observations = vec![
        attempt(0, 0, Some(1), Some(10), 0, 100, 5_000_000_000, SchedulePhase::Warmup),
        attempt(1, 0, Some(2), Some(11), 100, 200, 1_000_000_000, SchedulePhase::Measurement),
        attempt(2, 0, Some(3), Some(12), 200, 300, 3_000_000_000, SchedulePhase::Measurement),
    ];
    let aggregate =
        MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(2), 1_000_000_000, 2_000_000_000);
    assert_eq!(aggregate.window.warmup_attempts, 1);
    assert_eq!(aggregate.window.measurement_attempts, 2);
    assert_eq!(aggregate.throughput.completions, 2);
    assert_eq!(aggregate.latency.samples, 2);
    // The 5s warmup sample must not enter the distribution.
    assert_eq!(aggregate.latency.max_ns, 3_000_000_000);
    assert!((aggregate.throughput.application_qps - 1.0).abs() < f64::EPSILON);
    // Every attempt, warmup included, still has a report row.
    assert_eq!(aggregate.attempts.len(), 3);
}

#[test]
fn gpu_reaching_qps_excludes_cpu_fallback_while_application_qps_counts_it() {
    let mut fallback = attempt(1, 0, Some(2), Some(11), 0, 10, 1_000_000_000, SchedulePhase::Measurement);
    fallback.reached_gpu = false;
    fallback.disposition = "CpuFallback".to_string();
    fallback.device_ordinal = None;
    fallback.grant_id = None;
    let observations =
        vec![attempt(0, 0, Some(1), Some(10), 0, 10, 1_000_000_000, SchedulePhase::Measurement), fallback];
    let aggregate = MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 2_000_000_000);
    assert_eq!(aggregate.throughput.successful_completions, 2);
    assert_eq!(aggregate.throughput.gpu_reaching_completions, 1);
    assert!(aggregate.throughput.gpu_reaching_qps < aggregate.throughput.application_qps);
}

#[test]
fn report_and_aggregate_totals_are_equal_by_construction() {
    let observations = (0..7)
        .map(|index| {
            attempt(
                index,
                index % 2,
                Some(index as u64),
                Some(index as u64 + 50),
                index as u64 * 10,
                index as u64 * 10 + 100,
                1_000,
                if index == 0 { SchedulePhase::Warmup } else { SchedulePhase::Measurement },
            )
        })
        .collect::<Vec<_>>();
    let aggregate = MeasuredAggregate::reduce(
        &observations,
        &[device(0, 1 << 30, 2), device(1, 2 << 30, 2)],
        &occupancy(2),
        0,
        1_000_000_000,
    );
    let equality = aggregate.report_aggregate_equality;
    assert!(equality.equal);
    assert_eq!(equality.rows_total, 7);
    assert_eq!(equality.aggregate_total, 7);
    assert_eq!(equality.rows_measurement_attempts, equality.aggregate_measurement_attempts);
    assert_eq!(equality.rows_succeeded, equality.aggregate_succeeded);
    assert_eq!(equality.rows_placed, equality.aggregate_placed);
}

#[test]
fn percentiles_use_nearest_rank_over_retained_samples() {
    let sorted = (1..=100_u64).collect::<Vec<_>>();
    assert_eq!(percentile(&sorted, 50.0), 50);
    assert_eq!(percentile(&sorted, 95.0), 95);
    assert_eq!(percentile(&sorted, 99.0), 99);
    assert_eq!(percentile(&[], 95.0), 0);
    assert_eq!(percentile(&[7], 95.0), 7);
}

/// The witness must be falsifiable: two attempts that ran back to back on
/// one ordinal did not overlap, even though both were placed there.
#[test]
fn sequential_attempts_on_one_device_fail_the_overlap_clause() {
    let sequential = vec![
        attempt(0, 0, Some(10), Some(100), 1_000, 2_000, 1_000, SchedulePhase::Measurement),
        attempt(1, 0, Some(11), Some(101), 2_000, 3_000, 1_000, SchedulePhase::Measurement),
    ];
    let aggregate = MeasuredAggregate::reduce(&sequential, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
    let overlap = aggregate
        .scenarios
        .same_device
        .iter()
        .find(|clause| clause.clause == "concurrent_attempt_intervals_on_one_device")
        .unwrap();
    assert_eq!(overlap.status, ClauseStatus::Failed);
    assert!(!scenario_clauses_pass(&aggregate.scenarios));

    // A shared submit instant alone is not overlap either: zero-width
    // intersection must not count.
    let touching = vec![
        attempt(0, 0, Some(10), Some(100), 1_000, 1_000, 0, SchedulePhase::Measurement),
        attempt(1, 0, Some(11), Some(101), 1_000, 2_000, 1_000, SchedulePhase::Measurement),
    ];
    let aggregate = MeasuredAggregate::reduce(&touching, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
    assert_eq!(
        aggregate
            .scenarios
            .same_device
            .iter()
            .find(|clause| clause.clause == "concurrent_attempt_intervals_on_one_device")
            .unwrap()
            .status,
        ClauseStatus::Failed
    );
}

/// Equality must be falsifiable too: an attempt placed on an ordinal the
/// backend never reported breaks placement conservation.
#[test]
fn an_attempt_placed_on_an_unconfigured_ordinal_breaks_the_equality_check() {
    let stray = vec![
        attempt(0, 0, Some(10), Some(100), 0, 2_000, 1_000, SchedulePhase::Measurement),
        // Ordinal 7 has no device configuration, so `placement` cannot
        // account for it.
        attempt(1, 7, Some(11), Some(101), 0, 2_000, 1_000, SchedulePhase::Measurement),
    ];
    let aggregate = MeasuredAggregate::reduce(&stray, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
    let equality = aggregate.report_aggregate_equality;
    assert!(!equality.equal);
    assert_eq!(equality.rows_placed, 2);
    assert_eq!(equality.aggregate_placed, 1);
}

#[test]
fn a_single_placed_attempt_leaves_same_device_concurrency_unevidenced() {
    let observations = vec![attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
    let aggregate = MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
    assert_eq!(aggregate.scenarios.same_device.len(), 1);
    assert_eq!(aggregate.scenarios.same_device[0].status, ClauseStatus::NotEvidenced);
    assert!(scenario_clauses_pass(&aggregate.scenarios));
}

#[test]
fn a_serial_device_profile_leaves_same_device_concurrency_unevidenced() {
    let observations = vec![
        attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement),
        attempt(1, 0, Some(3), Some(4), 10, 20, 10, SchedulePhase::Measurement),
    ];
    let aggregate = MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 1)], &occupancy(1), 0, 1_000_000_000);
    assert_eq!(aggregate.scenarios.same_device.len(), 1);
    assert_eq!(aggregate.scenarios.same_device[0].status, ClauseStatus::NotEvidenced);
    assert!(scenario_clauses_pass(&aggregate.scenarios));
}

#[test]
fn per_step_aggregates_attribute_throughput_and_placement_to_their_step() {
    let observations = vec![
        AttemptObservation {
            step_index: 0,
            total_ns: Some(1_000_000),
            ..attempt(0, 0, Some(1), Some(2), 0, 10, 1_000_000, SchedulePhase::Measurement)
        },
        AttemptObservation {
            step_index: 1,
            total_ns: Some(3_000_000),
            ..attempt(1, 1, Some(3), Some(4), 0, 20, 3_000_000, SchedulePhase::Measurement)
        },
        AttemptObservation {
            step_index: 1,
            total_ns: Some(5_000_000),
            ..attempt(2, 1, Some(5), Some(6), 0, 30, 5_000_000, SchedulePhase::Measurement)
        },
    ];
    let steps = [
        StepWindow {
            step_index: 0,
            concurrency: 1,
            arrival_rate_per_second: None,
            rate_overrun: "delay",
            measurement_ns: 1_000_000_000,
        },
        StepWindow {
            step_index: 1,
            concurrency: 4,
            arrival_rate_per_second: None,
            rate_overrun: "delay",
            measurement_ns: 2_000_000_000,
        },
    ];
    let devices = [device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
    let aggregate =
        MeasuredAggregate::reduce_full(&observations, &devices, &occupancy(2), 0, 3_000_000_000, None, &steps);
    assert_eq!(aggregate.steps.len(), 2);
    let step0 = &aggregate.steps[0];
    assert_eq!(step0.throughput.completions, 1);
    assert_eq!(step0.latency.p50_ns, 1_000_000);
    assert_eq!(step0.placements.iter().map(|placement| placement.device_ordinal).collect::<Vec<_>>(), [0]);
    let step1 = &aggregate.steps[1];
    assert_eq!(step1.throughput.completions, 2);
    assert_eq!(step1.throughput.application_qps, 1.0);
    assert_eq!(step1.latency.p50_ns, 3_000_000);
    assert_eq!(step1.latency.max_ns, 5_000_000);
    assert_eq!(step1.placements.len(), 1);
    assert_eq!(step1.placements[0].attempts_placed, 2);
    // The run-level aggregate still spans every step's measurement set.
    assert_eq!(aggregate.throughput.completions, 3);
}

#[test]
fn latency_keeps_sealed_admission_waits_separate_from_plan_queue_waits() {
    let mut first = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
    first.admission_wait_ns = Some(10);
    first.queue_wait_ns = Some(1_000);
    let mut second = attempt(1, 0, Some(3), Some(4), 0, 20, 20, SchedulePhase::Measurement);
    second.admission_wait_ns = Some(20);
    second.queue_wait_ns = Some(2_000);
    let mut third = attempt(2, 0, Some(5), Some(6), 0, 30, 30, SchedulePhase::Measurement);
    third.admission_wait_ns = Some(90);
    third.queue_wait_ns = Some(3_000);

    let reduced = latency(&[&first, &second, &third]);
    assert_eq!(reduced.admission_wait_samples, 3);
    assert_eq!(reduced.admission_wait_p50_ns, 20);
    assert_eq!(reduced.admission_wait_p95_ns, 90);
    assert_eq!(reduced.admission_wait_p99_ns, 90);
    assert_eq!(reduced.queue_wait_p95_ns, 3_000);
}

#[test]
fn quarantine_cycles_evidence_requires_every_configured_cycle_to_evidence() {
    let observations = vec![
        attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement),
        attempt(1, 2, Some(3), Some(4), 0, 20, 20, SchedulePhase::Measurement),
    ];
    let first = quarantine_evidence(0, &observations);
    let second = quarantine_evidence(1, &observations[1..]);
    let evidenced = QuarantineCyclesEvidence::from_cycle_evidence(2, vec![first.clone(), second]);
    assert!(evidenced.is_evidenced());
    assert_eq!(evidenced.completed_cycles, 2);
    assert_eq!(evidenced.cycles[0].target_ordinal, 0);
    assert_eq!(evidenced.cycles[1].device_health_generation, 1);

    // Fewer cycles than configured is a failure, not a partial pass.
    let incomplete = QuarantineCyclesEvidence::from_cycle_evidence(3, vec![first.clone()]);
    assert!(!incomplete.is_evidenced());
    assert_eq!(incomplete.completed_cycles, 1);

    // A cycle with post-ack placements on its target fails the section.
    let contaminated = vec![attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
    let failed = QuarantineCyclesEvidence::from_cycle_evidence(1, vec![quarantine_evidence(0, &contaminated)]);
    assert!(!failed.is_evidenced());
    assert_eq!(failed.cycles[0].status, ClauseStatus::Failed);
}

fn cache_snapshot(physical: u64, ledger: u64, cap: u64) -> serde_json::Value {
    serde_json::json!({"cache_cap_bytes": cap, "ledger_evictable_resident_bytes": ledger, "ledger_pinned_in_use_bytes": 0, "physical_current_bytes": physical})
}

#[test]
fn opted_in_cache_quarantine_requires_complete_consistent_snapshots() {
    let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
    let mut evidence = quarantine_evidence(0, &observations);
    evidence.cache_residency = CacheQuarantineEvidence::from_snapshots(
        Some(cache_snapshot(90, 100, 100)),
        Some(cache_snapshot(0, 0, 100)),
        Some(cache_snapshot(0, 0, 100)),
    );
    evidence.require_cache_residency();
    assert_eq!(evidence.clause().status, ClauseStatus::Evidenced);
}

#[test]
fn opted_in_cache_quarantine_rejects_missing_or_zero_pre_ack() {
    let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
    let mut evidence = quarantine_evidence(0, &observations);
    evidence.require_cache_residency();
    assert_eq!(evidence.clause().status, ClauseStatus::Failed);
    evidence.cache_residency = CacheQuarantineEvidence::from_snapshots(
        Some(cache_snapshot(0, 0, 100)),
        Some(cache_snapshot(0, 100, 100)),
        Some(cache_snapshot(0, 100, 100)),
    );
    assert_eq!(evidence.clause().status, ClauseStatus::Failed);
}

#[test]
fn opted_in_cache_quarantine_rejects_physical_over_ledger_or_cap() {
    let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
    for snapshots in [
        (cache_snapshot(101, 100, 100), cache_snapshot(80, 100, 100), cache_snapshot(80, 100, 100)),
        (cache_snapshot(101, 200, 100), cache_snapshot(80, 100, 100), cache_snapshot(80, 100, 100)),
        (cache_snapshot(80, 100, 100), cache_snapshot(101, 100, 100), cache_snapshot(80, 100, 100)),
        (cache_snapshot(80, 100, 100), cache_snapshot(1, 0, 100), cache_snapshot(80, 100, 100)),
    ] {
        let mut evidence = quarantine_evidence(0, &observations);
        evidence.cache_residency =
            CacheQuarantineEvidence::from_snapshots(Some(snapshots.0), Some(snapshots.1), Some(snapshots.2));
        evidence.require_cache_residency();
        assert_eq!(evidence.clause().status, ClauseStatus::Failed);
    }
}

#[test]
fn legacy_quarantine_remains_evidenced_without_cache_observations() {
    let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
    let evidence = quarantine_evidence(0, &observations);
    assert_eq!(evidence.clause().status, ClauseStatus::Evidenced);
}

fn complete_graph_claim_row() -> GraphClaimCalibrationRow {
    GraphClaimCalibrationRow {
        node_id: "n7".to_string(),
        algorithm: "pagerank".to_string(),
        execution_state: "algorithm_started".to_string(),
        measured_peak_bytes: Some(4_000),
        measured_peak_provenance: "cache_scope_high_water".to_string(),
    }
}

fn physical_graph_claim_snapshot(rows: Vec<GraphClaimCalibrationRow>) -> GraphClaimCalibrationSnapshot {
    GraphClaimCalibrationSnapshot {
        truncated: false,
        rows,
        execution: Some(GraphClaimCalibrationExecution::Physical),
        execution_complete: true,
    }
}

fn bfs_singleflight_follower_snapshot(leader_attempt_id: Option<String>) -> GraphClaimCalibrationSnapshot {
    GraphClaimCalibrationSnapshot {
        truncated: false,
        rows: Vec::new(),
        execution: Some(GraphClaimCalibrationExecution::NonPhysical(
            GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower { leader_attempt_id },
        )),
        execution_complete: true,
    }
}

#[test]
fn graph_claim_calibration_requires_one_complete_typed_row_per_graph_attempt() {
    let mut graph = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
    graph.graph_workload = true;
    graph.graph_claim_calibration = Some(physical_graph_claim_snapshot(vec![complete_graph_claim_row()]));

    let evidence = graph_claim_calibration_evidence(&[graph.clone()]);
    assert!(evidence.passes());
    assert_eq!(evidence.graph_attempts, 1);
    assert_eq!(evidence.attempts[0].reason, "complete");

    graph.graph_claim_calibration.as_mut().expect("snapshot exists").rows.push(complete_graph_claim_row());
    let duplicate = graph_claim_calibration_evidence(&[graph]);
    assert!(!duplicate.passes());
    assert_eq!(duplicate.attempts[0].reason, "expected_exactly_one_graph_claim_row");
}

#[test]
fn graph_claim_calibration_rejects_missing_or_unmeasured_rows() {
    let mut missing = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
    missing.graph_workload = true;
    let missing_evidence = graph_claim_calibration_evidence(&[missing]);
    assert!(!missing_evidence.passes());
    assert_eq!(missing_evidence.attempts[0].reason, "missing_sealed_graph_claim_snapshot");

    let mut unavailable = attempt(1, 0, Some(3), Some(4), 0, 20, 20, SchedulePhase::Measurement);
    unavailable.graph_workload = true;
    unavailable.graph_claim_calibration = Some(physical_graph_claim_snapshot(vec![GraphClaimCalibrationRow {
        measured_peak_bytes: None,
        measured_peak_provenance: "unavailable".to_string(),
        ..complete_graph_claim_row()
    }]));
    let unavailable_evidence = graph_claim_calibration_evidence(&[unavailable]);
    assert!(!unavailable_evidence.passes());
    assert_eq!(unavailable_evidence.attempts[0].reason, "graph_peak_measurement_unavailable");
}

#[test]
fn graph_claim_calibration_accepts_a_fully_linked_singleflight_follower_across_warmup() {
    let mut leader = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Warmup);
    leader.graph_workload = true;
    leader.graph_claim_calibration = Some(physical_graph_claim_snapshot(vec![complete_graph_claim_row()]));

    let mut follower = attempt(1, 0, Some(3), Some(4), 0, 10, 10, SchedulePhase::Measurement);
    follower.graph_workload = true;
    follower.graph_claim_calibration = Some(bfs_singleflight_follower_snapshot(Some(leader.attempt_id.clone())));

    let evidence = graph_claim_calibration_evidence(&[leader, follower]);
    assert!(evidence.passes());
    assert_eq!(evidence.graph_attempts, 1);
    assert_eq!(evidence.attempts[0].reason, "complete");
}

#[test]
fn graph_claim_calibration_rejects_unlinked_or_row_carrying_singleflight_followers() {
    let mut absent_leader_id = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
    absent_leader_id.graph_workload = true;
    absent_leader_id.graph_claim_calibration = Some(bfs_singleflight_follower_snapshot(None));
    let absent_leader_id_evidence = graph_claim_calibration_evidence(&[absent_leader_id]);
    assert!(!absent_leader_id_evidence.passes());
    assert_eq!(absent_leader_id_evidence.attempts[0].reason, "singleflight_follower_missing_leader_attempt_id");

    let mut missing_leader = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
    missing_leader.graph_workload = true;
    missing_leader.graph_claim_calibration =
        Some(bfs_singleflight_follower_snapshot(Some("attempt-missing".to_string())));
    let missing_leader_evidence = graph_claim_calibration_evidence(&[missing_leader]);
    assert!(!missing_leader_evidence.passes());
    assert_eq!(missing_leader_evidence.attempts[0].reason, "singleflight_follower_leader_not_evidenced");

    let mut carrying_row = attempt(1, 0, Some(3), Some(4), 0, 10, 10, SchedulePhase::Measurement);
    carrying_row.graph_workload = true;
    let mut snapshot = bfs_singleflight_follower_snapshot(Some("attempt-missing".to_string()));
    snapshot.rows.push(complete_graph_claim_row());
    carrying_row.graph_claim_calibration = Some(snapshot);
    let carrying_row_evidence = graph_claim_calibration_evidence(&[carrying_row]);
    assert!(!carrying_row_evidence.passes());
    assert_eq!(carrying_row_evidence.attempts[0].reason, "singleflight_follower_has_graph_claim_rows");
}

#[test]
fn graph_claim_calibration_rejects_missing_or_conflicting_execution_classification() {
    let mut missing = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
    missing.graph_workload = true;
    missing.graph_claim_calibration = Some(GraphClaimCalibrationSnapshot {
        truncated: false,
        rows: vec![complete_graph_claim_row()],
        execution: None,
        execution_complete: false,
    });
    let missing_evidence = graph_claim_calibration_evidence(&[missing]);
    assert!(!missing_evidence.passes());
    assert_eq!(missing_evidence.attempts[0].reason, "graph_execution_classification_missing");

    let mut conflicting = attempt(1, 0, Some(3), Some(4), 0, 10, 10, SchedulePhase::Measurement);
    conflicting.graph_workload = true;
    conflicting.graph_claim_calibration = Some(GraphClaimCalibrationSnapshot {
        execution_complete: false,
        ..physical_graph_claim_snapshot(vec![complete_graph_claim_row()])
    });
    let conflicting_evidence = graph_claim_calibration_evidence(&[conflicting]);
    assert!(!conflicting_evidence.passes());
    assert_eq!(conflicting_evidence.attempts[0].reason, "graph_execution_classification_conflicted");
}
