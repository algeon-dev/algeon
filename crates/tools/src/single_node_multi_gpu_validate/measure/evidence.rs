use super::*;

pub(super) fn run_deadline_evidence(
    measurement_window_ns: u64,
    completion_drain_grace_ns: u64,
    elapsed_until_reports_ns: u64,
) -> RunDeadlineEvidence {
    let configured_window_ns = measurement_window_ns.saturating_add(completion_drain_grace_ns);
    RunDeadlineEvidence {
        measurement_window_ns,
        completion_drain_grace_ns,
        configured_window_ns,
        elapsed_until_reports_ns,
        overrun_ns: elapsed_until_reports_ns.saturating_sub(configured_window_ns),
        reports_sealed_within_deadline: elapsed_until_reports_ns <= configured_window_ns,
    }
}

pub(super) fn observe(
    sequence: usize,
    phase: SchedulePhase,
    step_index: usize,
    query: &SelectedQuery,
    report: &QueryAttemptReport,
    capture: OutputCapture,
) -> AttemptObservation {
    let resource_evidence = report.resource_evidence.as_ref();
    AttemptObservation {
        sequence,
        phase,
        step_index,
        query_kind: query.kind,
        query_label: query.label.clone(),
        graph_workload: query.graph,
        placement_required: query.placement_required,
        attempt_id: report.attempt_id.to_string(),
        disposition: format!("{:?}", report.disposition),
        outcome: report.outcome.stable_label().to_string(),
        failure: report.primary_failure.as_ref().map(|failure| AttemptFailureClass {
            origin: format!("{:?}", failure.origin()),
            code: format!("{:?}", failure.code()),
            status: format!("{:?}", failure.status()),
            operation: format!("{:?}", failure.operation()),
        }),
        terminal_evidence: terminal_evidence(report),
        succeeded: report.outcome.stable_label() == super::super::aggregate::SUCCEEDED_OUTCOME,
        reached_gpu: matches!(report.disposition, AttemptDisposition::Native | AttemptDisposition::PartialNative),
        graph_claim_calibration: capture.graph_claim_calibration,
        composed_schedule: report
            .composed_schedule
            .as_ref()
            .filter(|projection| projection.schema == COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1)
            .map(project_composed_schedule_projection),
        device_ordinal: report.admission.as_ref().map(|admission| admission.device_ordinal).or(report.device_ordinal),
        grant_bytes: report.admission.as_ref().map(|admission| admission.grant_bytes),
        admission_sequence: report.admission.as_ref().map(|admission| admission.admission_sequence),
        admission_wait_ns: report.admission.as_ref().map(|admission| admission.admission_wait_ns),
        admission_requested_ns: report.timestamps.admission_requested_ns,
        admitted_ns: report.timestamps.admitted_ns,
        admission_wait: report.admission_wait.map(|fact| super::super::aggregate::AdmissionWaitEvidence {
            submission_sequence: fact.submission_sequence,
            // Preserve the historical artifact shape as explicit FIFO zero
            // facts; overtake policy is no longer configurable.
            overtake_limit: 0,
            overtakes_consumed: 0,
            overtakes_remaining: 0,
            protection_device_ordinal: None,
        }),
        grant_id: resource_evidence.and_then(|evidence| evidence.grant_id.value),
        domain_id: resource_evidence.and_then(|evidence| evidence.domain_id.value),
        domain_pool_usage: resource_evidence
            .and_then(|evidence| evidence.outcome_terminal_usage.value)
            .and_then(|sample| sample.pool_usage)
            .map(|usage| super::super::aggregate::QueryDomainPoolUsageObservation {
                reserved_high_bytes: usage.reserved_high_bytes,
                used_high_bytes: usage.used_high_bytes,
            }),
        total_ns: report.total_ns(),
        queue_wait_ns: report.queue_wait_ns(),
        physical_execution_ns: report.physical_execution_ns(),
        // The engine stamps both from one observation clock shared by every
        // attempt, so these intervals are directly comparable. A host-side
        // stamp taken in the collector loop would only record collector order.
        submitted_ns: report.timestamps.submitted_ns,
        outcome_terminal_ns: report.timestamps.outcome_terminal_ns,
    }
}

/// Copies a schedule only from the sealed engine projection. This conversion
/// changes artifact field names but never derives a byte, scope, order, or
/// fingerprint in the runner.
pub(super) fn project_composed_schedule_projection(
    projection: &AttemptComposedScheduleProjection,
) -> ComposedScheduleArtifact {
    ComposedScheduleArtifact {
        schema: super::super::COMPOSED_SCHEDULE_SCHEMA,
        version: super::super::COMPOSED_SCHEDULE_VERSION,
        attempt_id: projection.attempt_id.to_string(),
        selected_ordinal: projection.device_ordinal,
        grant_id: projection.grant_id,
        grant_bytes: projection.grant_bytes,
        schedule_fingerprint: projection.fingerprint.clone(),
        engine_fingerprint_matches_contents: projection.fingerprint_matches_contents(),
        attempt_wide_fixed_bytes: projection.attempt_wide_fixed_bytes,
        occurrences: projection
            .producer_occurrences
            .iter()
            .enumerate()
            .map(|(order, occurrence)| ComposedScheduleOccurrence {
                occurrence_id: occurrence.occurrence_id.clone(),
                order,
                structural_scope: occurrence.execution_scope_fingerprint.clone(),
                isolated_peak_bytes: occurrence.isolated_peak_bytes,
                stage_peak_bytes: occurrence.stage_peak_bytes,
                retained_output_bytes: occurrence.retained_output_bytes,
            })
            .collect(),
        consumer_structural_scope: projection.consumer_scope_fingerprint.clone(),
        consumer_additional_peak_bytes: projection.consumer_additional_peak_bytes,
        consumer_stage_peak_bytes: projection.consumer_stage_peak_bytes,
        complete_phase_peak_bytes: projection.complete_peak_bytes,
    }
}

/// Projects the DataFusion adapter's immutable typed snapshot into the versioned
/// stress artifact. This intentionally reads engine fields directly rather
/// than reconstructing graph rows from DataFusion metric labels or error text.
pub(super) fn project_graph_claim_calibration_snapshot(
    snapshot: algeon_datafusion::attempt::SealedAttemptGraphClaimCalibrationSnapshot,
) -> GraphClaimCalibrationSnapshot {
    GraphClaimCalibrationSnapshot {
        truncated: snapshot.truncated(),
        execution: snapshot.execution().cloned().map(Into::into),
        execution_complete: snapshot.execution_complete(),
        rows: snapshot
            .rows()
            .iter()
            .map(|row| GraphClaimCalibrationRow {
                node_id: row.node_id.clone(),
                algorithm: row.algorithm.as_str().to_string(),
                execution_state: row.execution_state.as_str().to_string(),
                measured_peak_bytes: row.measured_peak_bytes,
                measured_peak_provenance: row.measured_peak_provenance.as_str().to_string(),
            })
            .collect(),
    }
}

/// Projects dedicated-probe report facts without persisting caller correlations or
/// arbitrary failure text. The sealed report remains the sole event record.
pub(super) fn terminal_evidence(report: &QueryAttemptReport) -> AttemptTerminalEvidence {
    AttemptTerminalEvidence {
        primary_failure: report.primary_failure.as_ref().map(redacted_failure_summary),
        cleanup_disposition: json_value(report.cleanup_disposition),
        cleanup_failure_count: report.cleanup_failure_count,
        cleanup_failure_count_complete: report.cleanup_failure_count_complete,
        first_cleanup_failure: report.first_cleanup_failure.as_ref().map(redacted_cleanup_failure_summary),
        capacity_disposition: json_value(report.capacity_disposition),
        health_transition: report.health_transition.map(json_value),
        native_fault_evidence: report.native_fault_evidence.map(json_value),
        resource_evidence: report.resource_evidence.map(json_value),
        retry_advice: json_value(report.retry_advice),
    }
}

pub(super) fn redacted_failure_summary(summary: &query_engine::observability::AttemptFailureSummary) -> Value {
    let mut value = json_value(summary);
    if let Value::Object(fields) = &mut value {
        fields.remove("text");
    }
    value
}

pub(super) fn redacted_cleanup_failure_summary(summary: &query_engine::observability::CleanupFailureSummary) -> Value {
    let mut value = json_value(summary);
    if let Value::Object(fields) = &mut value
        && let Some(Value::Object(failure)) = fields.get_mut("failure")
    {
        failure.remove("text");
    }
    value
}

pub(super) fn json_value<T: serde::Serialize>(value: T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// Closes the public backend once and retains an auditable bounded projection
/// of either outcome. Do not convert an error to a `null` artifact field: a
/// failed close is a real result that must remain visible.
pub(super) fn close_backend(backend: &AlgeonGpuBackend) -> BackendCloseEvidence {
    let before = project_close_snapshot(backend.close_snapshot());
    let outcome = match backend.close_and_wait(CLOSE_TIMEOUT) {
        Ok(snapshot) => BackendCloseOutcome::Quiescent { snapshot: project_close_snapshot(snapshot) },
        Err(error) => BackendCloseOutcome::Failed {
            failure: BackendCloseFailure {
                code: error.code().as_str(),
                kind: error.kind().as_str(),
                status: error.status().as_str(),
                operation: error.operation(),
                // These are the only close facts that carry a closed,
                // non-caller-controlled vocabulary. `cleanup_error` and any
                // source text intentionally never leave the engine.
                reason: error.fact_value("reason").map(str::to_owned),
            },
        },
    };
    let after = project_close_snapshot(backend.close_snapshot());
    BackendCloseEvidence { before, outcome, after }
}

pub(super) fn project_close_snapshot(
    snapshot: query_engine::runtime::GpuBackendCloseSnapshot,
) -> BackendCloseSnapshotEvidence {
    BackendCloseSnapshotEvidence {
        closing: snapshot.closing,
        gpu_resource_quiescent: snapshot.gpu_resource_quiescent,
        observation_quiescent: snapshot.observation_quiescent,
        active_gpu_permits: snapshot.active_gpu_permits,
        active_observation_permits: snapshot.active_observation_permits,
        cleanup_failed: snapshot.cleanup_failed,
        cleanup_deadline_elapsed_count: snapshot.cleanup_deadline_elapsed_count,
        cleanup_late_success_count: snapshot.cleanup_late_success_count,
    }
}

pub(super) fn device_configurations(backend: &AlgeonGpuBackend) -> Vec<DeviceConfiguration> {
    backend
        .device_startup_records()
        .iter()
        .map(|record| DeviceConfiguration {
            ordinal: record.device_ordinal,
            model_name: record.model_name.clone(),
            uuid: record.uuid.clone(),
            physical_total_bytes: record.physical_total_bytes,
            capacity_bytes: record.capacity_bytes,
            cache_cap_bytes: record.resource_profile.cache_cap_bytes,
            install_telemetry: record.install_telemetry,
            max_active_attempts: record.max_active_attempts,
            stream_pool_size: record.stream_pool_size,
            worker_count: record.worker_count,
            lane_count: record.lane_count,
            ownership: format!("{:?}", record.memory_ownership),
            device_health_generation: record.device_health_generation,
            // A device that enumerated a startup record is admissible at t0;
            // The quarantine scenario owns making one become ineligible mid-run.
            starting_health_open: true,
        })
        .collect()
}
