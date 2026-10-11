//! Pure operability reductions over sealed public attempt reports.
//!
//! The stress program must say whether an embedding application can diagnose
//! a non-success outcome from the report it already owns.  This module does
//! not inspect engine state or error text: it reduces the bounded projection
//! retained in [`AttemptObservation`] into per-attempt and run-wide evidence.
#![cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use super::{
    aggregate::{AttemptObservation, AttemptTerminalEvidence, ClauseStatus},
    profile::FeatureOffCugraphProbeProfile,
    schedule::SchedulePhase,
};

/// The runner-owned, schema-only graph UDTF appended to a feature-off
/// profile. Keeping this label outside user-selected catalog labels prevents
/// another graph workload from silently becoming part of the cell.
pub(super) const FEATURE_OFF_CUGRAPH_QUERY_LABEL: &str = "feature_off_cugraph_pagerank";
pub(super) const FEATURE_OFF_CUGRAPH_OPERATION: &str = "algeon_datafusion.cugraph_sql.execute";
const FEATURE_OFF_CUGRAPH_ERROR_CODE: &str = "unsupported_operator";
const FEATURE_OFF_CUGRAPH_REASON: &str = "required_feature_disabled";
const FEATURE_OFF_CUGRAPH_REQUIRED_FEATURE: &str = "cugraph";
const FEATURE_OFF_CUGRAPH_HOST_FALLBACK: &str = "disabled";

/// The bounded, typed projection captured from the actual `DataFusionError`
/// returned by the UDTF stream. It deliberately carries only equality facts
/// for the stable public contract, never error display/source text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct FeatureOffCugraphSourceError {
    pub(super) source_error_observed: bool,
    pub(super) unsupported_operator: bool,
    pub(super) operation_matches: bool,
    pub(super) reason_matches: bool,
    pub(super) required_feature_matches: bool,
    pub(super) host_fallback_matches: bool,
}

/// One selected feature-off UDTF attempt's structured result. The direct
/// source-error projection is retained beside the sealed report because the
/// report intentionally reduces adapter-only failures to its text-free
/// fallback vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct FeatureOffCugraphAttemptEvidence {
    pub(super) sequence: usize,
    pub(super) sealed_report_rejected: bool,
    pub(super) source_error_observed: bool,
    pub(super) unsupported_operator: bool,
    pub(super) operation_matches: bool,
    pub(super) reason_matches: bool,
    pub(super) required_feature_matches: bool,
    pub(super) host_fallback_matches: bool,
    pub(super) native_fault_evidence_absent: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) total_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) within_measured_latency_bound: Option<bool>,
}

/// Feature-off cell evidence. This is a load-harness projection, not a
/// unit-test result: it proves that every emitted UDTF request sealed an exact
/// structured rejection while relational attempts continued to reach GPU work
/// in the same measured window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct FeatureOffCugraphEvidence {
    pub(super) status: ClauseStatus,
    pub(super) query_label: &'static str,
    pub(super) error_code: &'static str,
    pub(super) operation: &'static str,
    pub(super) reason: &'static str,
    pub(super) required_feature: &'static str,
    pub(super) host_fallback: &'static str,
    pub(super) max_rejection_latency_ns: u64,
    pub(super) required_warmup_probe_attempts: u64,
    pub(super) required_measurement_probe_attempts: u64,
    pub(super) probe_attempts: u64,
    pub(super) warmup_probe_attempts: u64,
    pub(super) measurement_probe_attempts: u64,
    pub(super) exact_structured_rejections: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_observed_measurement_rejection_latency_ns: Option<u64>,
    pub(super) native_fault_evidence_count: u64,
    /// A failure before an attempt task starts leaves the selected query
    /// unknown, so it could otherwise hide an unissued runner-owned UDTF.
    pub(super) begin_failures: u64,
    /// A task join failure loses both its sealed report and its typed source
    /// projection, so it cannot be compatible with an every-request claim.
    pub(super) join_failures: u64,
    pub(super) relational_measurement_attempts: u64,
    pub(super) relational_gpu_reaching_successes: u64,
    pub(super) min_relational_gpu_reaching_attempts: u64,
    pub(super) attempts: Vec<FeatureOffCugraphAttemptEvidence>,
}

impl FeatureOffCugraphEvidence {
    pub(super) fn from_observations(
        profile: FeatureOffCugraphProbeProfile,
        observations: &[AttemptObservation],
        source_errors: &[(usize, FeatureOffCugraphSourceError)],
        begin_failures: u64,
        join_failures: u64,
    ) -> Self {
        let source_errors = source_errors.iter().copied().collect::<BTreeMap<_, _>>();
        let latency_cap_ns = profile.max_rejection_latency_ms.saturating_mul(1_000_000);
        let mut exact_structured_rejections = 0_u64;
        let mut warmup_probe_attempts = 0_u64;
        let mut measurement_probe_attempts = 0_u64;
        let mut max_observed_measurement_rejection_latency_ns: Option<u64> = None;
        let attempts = observations
            .iter()
            .filter(|observation| observation.query_label == FEATURE_OFF_CUGRAPH_QUERY_LABEL)
            .map(|observation| {
                let source = source_errors.get(&observation.sequence).copied().unwrap_or_default();
                let measured = observation.phase == SchedulePhase::Measurement;
                if measured {
                    measurement_probe_attempts += 1;
                    if let Some(total_ns) = observation.total_ns {
                        max_observed_measurement_rejection_latency_ns = Some(
                            max_observed_measurement_rejection_latency_ns
                                .map_or(total_ns, |current| current.max(total_ns)),
                        );
                    }
                } else {
                    warmup_probe_attempts += 1;
                }
                let within_measured_latency_bound =
                    measured.then(|| observation.total_ns.is_some_and(|total_ns| total_ns <= latency_cap_ns));
                let native_fault_evidence_absent = observation.terminal_evidence.native_fault_evidence.is_none();
                let exact = !observation.succeeded
                    && source.source_error_observed
                    && source.unsupported_operator
                    && source.operation_matches
                    && source.reason_matches
                    && source.required_feature_matches
                    && source.host_fallback_matches
                    && native_fault_evidence_absent
                    && within_measured_latency_bound.unwrap_or(true);
                exact_structured_rejections += u64::from(exact);
                FeatureOffCugraphAttemptEvidence {
                    sequence: observation.sequence,
                    sealed_report_rejected: !observation.succeeded,
                    source_error_observed: source.source_error_observed,
                    unsupported_operator: source.unsupported_operator,
                    operation_matches: source.operation_matches,
                    reason_matches: source.reason_matches,
                    required_feature_matches: source.required_feature_matches,
                    host_fallback_matches: source.host_fallback_matches,
                    native_fault_evidence_absent,
                    total_ns: observation.total_ns,
                    within_measured_latency_bound,
                }
            })
            .collect::<Vec<_>>();
        let probe_attempts = attempts.len() as u64;
        let native_fault_evidence_count = observations
            .iter()
            .filter(|observation| {
                observation.query_label == FEATURE_OFF_CUGRAPH_QUERY_LABEL
                    && observation.terminal_evidence.native_fault_evidence.is_some()
            })
            .count() as u64;
        let relational_measurement = observations
            .iter()
            .filter(|observation| observation.phase == SchedulePhase::Measurement && !observation.graph_workload);
        let relational_measurement_attempts = relational_measurement.clone().count() as u64;
        let relational_gpu_reaching_successes = relational_measurement
            .filter(|observation| observation.succeeded && observation.reached_gpu)
            .count() as u64;
        let required_warmup_probe_attempts = profile.warmup_probe_attempts as u64;
        let required_measurement_probe_attempts = profile.measurement_probe_attempts as u64;
        let required_probe_attempts =
            required_warmup_probe_attempts.saturating_add(required_measurement_probe_attempts);
        let status = if probe_attempts == required_probe_attempts
            && warmup_probe_attempts == required_warmup_probe_attempts
            && measurement_probe_attempts == required_measurement_probe_attempts
            && exact_structured_rejections == probe_attempts
            && native_fault_evidence_count == 0
            && begin_failures == 0
            && join_failures == 0
            && relational_gpu_reaching_successes >= profile.min_relational_gpu_reaching_attempts
        {
            ClauseStatus::Evidenced
        } else {
            ClauseStatus::Failed
        };
        Self {
            status,
            query_label: FEATURE_OFF_CUGRAPH_QUERY_LABEL,
            error_code: FEATURE_OFF_CUGRAPH_ERROR_CODE,
            operation: FEATURE_OFF_CUGRAPH_OPERATION,
            reason: FEATURE_OFF_CUGRAPH_REASON,
            required_feature: FEATURE_OFF_CUGRAPH_REQUIRED_FEATURE,
            host_fallback: FEATURE_OFF_CUGRAPH_HOST_FALLBACK,
            max_rejection_latency_ns: latency_cap_ns,
            required_warmup_probe_attempts,
            required_measurement_probe_attempts,
            probe_attempts,
            warmup_probe_attempts,
            measurement_probe_attempts,
            exact_structured_rejections,
            max_observed_measurement_rejection_latency_ns,
            native_fault_evidence_count,
            begin_failures,
            join_failures,
            relational_measurement_attempts,
            relational_gpu_reaching_successes,
            min_relational_gpu_reaching_attempts: profile.min_relational_gpu_reaching_attempts,
            attempts,
        }
    }

    pub(super) fn passes(&self) -> bool {
        self.status == ClauseStatus::Evidenced
    }
}

/// Whether one sealed attempt report names every fact an operator needs to
/// attribute a non-success without reading engine internals or error prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct AttemptAttribution {
    /// Successful attempts are outside the completeness denominator.  Their
    /// empty `missing_fields` is still emitted so every attempt row has the
    /// same auditable shape.
    pub(super) applicable: bool,
    pub(super) complete: bool,
    pub(super) device_ordinal_required: bool,
    pub(super) allocator_event_required: bool,
    /// Closed field names, never caller-controlled report text.
    pub(super) missing_fields: Vec<&'static str>,
}

/// One outcome class's contribution to attribution completeness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct OutcomeAttributionSummary {
    pub(super) outcome: String,
    pub(super) non_success_attempts: u64,
    pub(super) attributable_attempts: u64,
    pub(super) unattributable_attempts: u64,
}

/// Count of one specifically absent report field.  A missing field is a
/// product finding, not an inferred cause.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct MissingAttributionField {
    pub(super) field: &'static str,
    pub(super) attempts: u64,
}

/// Artifact-level attribution-completeness section.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct AttributionCompleteness {
    pub(super) status: ClauseStatus,
    pub(super) non_success_attempts: u64,
    pub(super) attributable_attempts: u64,
    pub(super) unattributable_attempts: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) attributable_fraction: Option<f64>,
    pub(super) by_outcome: Vec<OutcomeAttributionSummary>,
    pub(super) missing_fields: Vec<MissingAttributionField>,
}

/// The explicit reference-exporter cell. The endpoint stored in an artifact
/// is always reduced to host:port; credentials and paths never leave the
/// invocation boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct OtlpExportEvidence {
    pub(super) status: ClauseStatus,
    pub(super) mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) endpoint_host_port: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) interval_ms: Option<u64>,
    pub(super) export_attempts: u64,
    pub(super) successful_exports: u64,
    pub(super) failed_exports: u64,
}

impl OtlpExportEvidence {
    pub(super) fn not_requested() -> Self {
        Self {
            status: ClauseStatus::NotEvidenced,
            mode: "not_requested",
            endpoint_host_port: None,
            interval_ms: None,
            export_attempts: 0,
            successful_exports: 0,
            failed_exports: 0,
        }
    }

    pub(super) fn requested(
        endpoint_host_port: String,
        interval_ms: u64,
        export_attempts: u64,
        successful_exports: u64,
        failed_exports: u64,
    ) -> Self {
        let status = if failed_exports != 0 {
            ClauseStatus::Failed
        } else if successful_exports != 0 {
            ClauseStatus::Evidenced
        } else {
            ClauseStatus::NotEvidenced
        };
        Self {
            status,
            mode: "examples_backend_metrics_otlp_exporter",
            endpoint_host_port: Some(endpoint_host_port),
            interval_ms: Some(interval_ms),
            export_attempts,
            successful_exports,
            failed_exports,
        }
    }

    pub(super) fn passes(&self) -> bool {
        match self.mode {
            // Ordinary suite cells do not exercise the embedder's exporter.
            // They retain the explicit not-requested row without making an
            // export claim a global prerequisite.
            "not_requested" => {
                self.status == ClauseStatus::NotEvidenced
                    && self.endpoint_host_port.is_none()
                    && self.interval_ms.is_none()
                    && self.export_attempts == 0
                    && self.successful_exports == 0
                    && self.failed_exports == 0
            },
            "examples_backend_metrics_otlp_exporter" => {
                self.status == ClauseStatus::Evidenced
                    && self.endpoint_host_port.as_deref().is_some_and(|endpoint| !endpoint.is_empty())
                    && self.interval_ms.is_some_and(|interval| interval != 0)
                    && self.export_attempts != 0
                    && self.successful_exports != 0
                    && self.failed_exports == 0
            },
            _ => false,
        }
    }
}

impl AttributionCompleteness {
    /// Reduces only canonical sealed report projections.  A run with no
    /// non-success outcomes cannot prove attribution completeness, but it is
    /// not misreported as a failed observability result.
    pub(super) fn from_observations(observations: &[AttemptObservation]) -> Self {
        let mut by_outcome: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        let mut missing_fields = BTreeMap::<&'static str, u64>::new();
        let mut non_success_attempts = 0_u64;
        let mut attributable_attempts = 0_u64;

        for observation in observations.iter().filter(|observation| !observation.succeeded) {
            non_success_attempts += 1;
            let attribution = attribution_for(observation);
            let counts = by_outcome.entry(observation.outcome.clone()).or_default();
            counts.0 += 1;
            if attribution.complete {
                attributable_attempts += 1;
                counts.1 += 1;
            }
            for field in attribution.missing_fields {
                *missing_fields.entry(field).or_default() += 1;
            }
        }

        let unattributable_attempts = non_success_attempts.saturating_sub(attributable_attempts);
        let status = if non_success_attempts == 0 {
            ClauseStatus::NotEvidenced
        } else if unattributable_attempts == 0 {
            ClauseStatus::Evidenced
        } else {
            ClauseStatus::Failed
        };
        Self {
            status,
            non_success_attempts,
            attributable_attempts,
            unattributable_attempts,
            attributable_fraction: (non_success_attempts != 0)
                .then(|| attributable_attempts as f64 / non_success_attempts as f64),
            by_outcome: by_outcome
                .into_iter()
                .map(|(outcome, (non_success_attempts, attributable_attempts))| OutcomeAttributionSummary {
                    outcome,
                    non_success_attempts,
                    attributable_attempts,
                    unattributable_attempts: non_success_attempts.saturating_sub(attributable_attempts),
                })
                .collect(),
            missing_fields: missing_fields
                .into_iter()
                .map(|(field, attempts)| MissingAttributionField { field, attempts })
                .collect(),
        }
    }

    /// An observed missing field is a qualification failure; no observations
    /// remain explicitly `not_evidenced` until a fault-bearing cell is run.
    pub(super) fn passes(&self) -> bool {
        self.status != ClauseStatus::Failed
    }
}

/// Projects reference-exporter completeness for an individual attempt row.
pub(super) fn attribution_for(observation: &AttemptObservation) -> AttemptAttribution {
    if observation.succeeded {
        return AttemptAttribution {
            applicable: false,
            complete: true,
            device_ordinal_required: false,
            allocator_event_required: false,
            missing_fields: Vec::new(),
        };
    }

    let device_ordinal_required = report_owns_device(observation);
    let allocator_event_required = memory_failure_requires_allocator_event(observation);
    let mut missing_fields = Vec::new();
    if observation.outcome.trim().is_empty() {
        missing_fields.push("outcome");
    }
    // `primary_failure` is the sealed structured cause.  A cancellation
    // without one remains visible as an observability gap rather than being
    // reverse-engineered from runner control flow.
    if observation.failure.is_none() {
        missing_fields.push("cause");
    }
    if device_ordinal_required && observation.device_ordinal.is_none() {
        missing_fields.push("device_ordinal");
    }
    if allocator_event_required && !has_attempt_exact_allocator_event(&observation.terminal_evidence) {
        missing_fields.push("allocator_event");
    }
    AttemptAttribution {
        applicable: true,
        complete: missing_fields.is_empty(),
        device_ordinal_required,
        allocator_event_required,
        missing_fields,
    }
}

/// A device ordinal applies once a device-backed attempt has actually been
/// admitted, including an attempt that later failed after an evidence block
/// was emitted. A native plan is not itself a device placement: pre-admission
/// rejections and explicitly unplanned cancellations are intentionally not
/// forced to invent an ordinal.
fn report_owns_device(observation: &AttemptObservation) -> bool {
    (matches!(observation.disposition.as_str(), "Native" | "PartialNative") && observation.admission_sequence.is_some())
        || observation.terminal_evidence.native_fault_evidence.is_some()
        || observation.terminal_evidence.resource_evidence.is_some()
}

/// Determines whether a memory failure needs an exact cuDF allocator-ring
/// event. `MemoryReservationFailed` is already a sealed native-controller
/// cause; it is not evidence that the cuDF callback ran. A typed cuGraph
/// allocation label names the throwing API boundary but cannot replace a
/// cache-scope allocator record: a resident scope now publishes every
/// allocation failure class with its immutable scope identity.
fn memory_failure_requires_allocator_event(observation: &AttemptObservation) -> bool {
    matches!(
        observation.failure.as_ref().map(|failure| failure.code.as_str()),
        Some("QueryMemoryLimitExceeded" | "DeviceOutOfMemory" | "DeviceAllocationFailed")
    )
}

/// An allocator event is attributable only when its own public provenance is
/// `attempt_exact`.  Device-level or post-failure samples are useful context
/// but cannot establish that this attempt caused the event.
fn has_attempt_exact_allocator_event(evidence: &AttemptTerminalEvidence) -> bool {
    let Some(Value::Object(resource)) = evidence.resource_evidence.as_ref() else {
        return false;
    };
    let Some(Value::Object(allocator_event)) = resource.get("allocator_event") else {
        return false;
    };
    allocator_event.get("provenance").and_then(Value::as_str) == Some("attempt_exact")
        && allocator_event.get("value").is_some_and(|value| !value.is_null())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::single_node_multi_gpu_validate::{
        aggregate::{AttemptFailureClass, AttemptTerminalEvidence, SUCCEEDED_OUTCOME},
        profile::FixtureKind,
        schedule::SchedulePhase,
    };

    fn observation() -> AttemptObservation {
        AttemptObservation {
            sequence: 0,
            phase: SchedulePhase::Measurement,
            step_index: 0,
            query_kind: FixtureKind::Tpch,
            query_label: "q01".to_string(),
            graph_workload: false,
            placement_required: true,
            attempt_id: "attempt-1".to_string(),
            disposition: "Native".to_string(),
            outcome: "failed".to_string(),
            failure: Some(AttemptFailureClass {
                origin: "Native".to_string(),
                code: "MemoryReservationFailed".to_string(),
                status: "ResourceExhausted".to_string(),
                operation: "Execution".to_string(),
            }),
            terminal_evidence: AttemptTerminalEvidence {
                primary_failure: Some(json!({"code": "memory_reservation_failed"})),
                cleanup_disposition: json!("completed"),
                cleanup_failure_count: 0,
                cleanup_failure_count_complete: true,
                first_cleanup_failure: None,
                capacity_disposition: json!("released"),
                health_transition: None,
                native_fault_evidence: None,
                resource_evidence: Some(json!({
                    "allocator_event": {
                        "provenance": "attempt_exact",
                        "value": {"source": "reservation_limit"}
                    }
                })),
                retry_advice: json!({"recovery": "change_configuration"}),
            },
            succeeded: false,
            reached_gpu: true,
            graph_claim_calibration: None,
            composed_schedule: None,
            device_ordinal: Some(0),
            grant_bytes: Some(1),
            admission_sequence: Some(1),
            admission_wait_ns: Some(0),
            admission_requested_ns: Some(1),
            admitted_ns: Some(2),
            admission_wait: None,
            grant_id: Some(1),
            domain_id: Some(1),
            domain_pool_usage: None,
            total_ns: Some(10),
            queue_wait_ns: Some(1),
            physical_execution_ns: Some(5),
            submitted_ns: 0,
            outcome_terminal_ns: 10,
        }
    }

    #[test]
    fn exact_memory_evidence_is_attributable() {
        let summary = AttributionCompleteness::from_observations(&[observation()]);
        assert_eq!(summary.status, ClauseStatus::Evidenced);
        assert_eq!(summary.attributable_attempts, 1);
        assert!(summary.missing_fields.is_empty());
    }

    #[test]
    fn native_reservation_failure_is_complete_without_a_callback_event() {
        let mut reservation_failure = observation();
        reservation_failure.terminal_evidence.resource_evidence = None;

        let attribution = attribution_for(&reservation_failure);
        assert!(!attribution.allocator_event_required);
        assert!(attribution.complete);
        assert_eq!(AttributionCompleteness::from_observations(&[reservation_failure]).status, ClauseStatus::Evidenced);
    }

    fn cugraph_oom_observation(allocation_source: &str, exact_allocator_event: bool) -> AttemptObservation {
        let mut observation = observation();
        observation.failure = Some(AttemptFailureClass {
            origin: "Native".to_string(),
            code: "QueryMemoryLimitExceeded".to_string(),
            status: "Temporary".to_string(),
            operation: "Execution".to_string(),
        });
        observation.terminal_evidence.primary_failure = Some(json!({
            "code": "query_memory_limit_exceeded",
            "facts": [{
                "key": "allocation_source",
                "value": allocation_source,
            }],
        }));
        if !exact_allocator_event {
            observation.terminal_evidence.resource_evidence = None;
        }
        observation
    }

    #[test]
    fn direct_typed_cugraph_oom_without_allocator_event_is_incomplete() {
        for source in ["cugraph_rmm_bad_alloc", "cugraph_std_bad_alloc", "cugraph_rmm_out_of_memory"] {
            for exact_allocator_event in [false, true] {
                let observation = cugraph_oom_observation(source, exact_allocator_event);
                let attribution = attribution_for(&observation);
                assert!(attribution.allocator_event_required, "{source}, event={exact_allocator_event}");
                assert_eq!(attribution.complete, exact_allocator_event, "{source}, event={exact_allocator_event}");
                if !exact_allocator_event {
                    assert_eq!(attribution.missing_fields, vec!["allocator_event"], "{source}");
                }
                assert_eq!(
                    AttributionCompleteness::from_observations(&[observation]).status,
                    if exact_allocator_event { ClauseStatus::Evidenced } else { ClauseStatus::Failed },
                    "{source}, event={exact_allocator_event}"
                );
            }
        }
    }

    #[test]
    fn caller_cancellation_without_a_sealed_cause_is_a_named_gap() {
        let mut cancelled = observation();
        cancelled.outcome = "cancelled".to_string();
        cancelled.failure = None;
        cancelled.device_ordinal = None;
        cancelled.disposition = "Unplanned".to_string();
        cancelled.terminal_evidence.resource_evidence = None;
        let summary = AttributionCompleteness::from_observations(&[cancelled]);
        assert_eq!(summary.status, ClauseStatus::Failed);
        assert_eq!(summary.missing_fields, vec![MissingAttributionField { field: "cause", attempts: 1 }]);
    }

    #[test]
    fn pre_admission_backend_closing_does_not_invent_a_device_ordinal() {
        let mut closing = observation();
        closing.failure = Some(AttemptFailureClass {
            origin: "AttemptControl".to_string(),
            code: "BackendClosing".to_string(),
            status: "Permanent".to_string(),
            operation: "Admission".to_string(),
        });
        closing.device_ordinal = None;
        closing.admission_sequence = None;
        closing.admission_wait_ns = None;
        closing.admission_requested_ns = None;
        closing.admitted_ns = None;
        closing.grant_id = None;
        closing.domain_id = None;
        closing.terminal_evidence.resource_evidence = None;

        let attribution = attribution_for(&closing);
        assert!(!attribution.device_ordinal_required);
        assert!(attribution.complete);
    }

    #[test]
    fn successful_attempts_do_not_make_a_vacuous_success_claim() {
        let mut succeeded = observation();
        succeeded.outcome = SUCCEEDED_OUTCOME.to_string();
        succeeded.succeeded = true;
        succeeded.failure = None;
        let summary = AttributionCompleteness::from_observations(&[succeeded]);
        assert_eq!(summary.status, ClauseStatus::NotEvidenced);
        assert!(summary.passes());
    }

    #[test]
    fn otlp_not_requested_passes_an_ordinary_cell() {
        let cases = [
            ("not requested", OtlpExportEvidence::not_requested(), ClauseStatus::NotEvidenced, true),
            (
                "no success",
                OtlpExportEvidence::requested("127.0.0.1:4317".to_string(), 1_000, 0, 0, 0),
                ClauseStatus::NotEvidenced,
                false,
            ),
            (
                "success",
                OtlpExportEvidence::requested("127.0.0.1:4317".to_string(), 1_000, 3, 1, 0),
                ClauseStatus::Evidenced,
                true,
            ),
            (
                "failure",
                OtlpExportEvidence::requested("127.0.0.1:4317".to_string(), 1_000, 3, 2, 1),
                ClauseStatus::Failed,
                false,
            ),
        ];
        for (name, evidence, status, passes) in cases {
            assert_eq!(evidence.status, status, "{name}");
            assert_eq!(evidence.passes(), passes, "{name}");
        }
    }

    #[test]
    fn feature_off_cell_requires_typed_rejection_and_relational_gpu_progress() {
        let mut relational = observation();
        relational.outcome = SUCCEEDED_OUTCOME.to_string();
        relational.succeeded = true;
        relational.failure = None;

        let mut warmup_graph = observation();
        warmup_graph.sequence = 1;
        warmup_graph.phase = SchedulePhase::Warmup;
        warmup_graph.query_label = FEATURE_OFF_CUGRAPH_QUERY_LABEL.to_string();
        warmup_graph.graph_workload = true;
        warmup_graph.disposition = "Unplanned".to_string();
        warmup_graph.device_ordinal = None;
        warmup_graph.admission_sequence = None;
        warmup_graph.terminal_evidence.resource_evidence = None;

        let mut graph = warmup_graph.clone();
        graph.sequence = 2;
        graph.phase = SchedulePhase::Measurement;
        graph.total_ns = Some(10_000_000);

        let profile = FeatureOffCugraphProbeProfile {
            max_rejection_latency_ms: 25,
            warmup_probe_attempts: 1,
            measurement_probe_attempts: 1,
            min_relational_gpu_reaching_attempts: 1,
        };
        let exact = FeatureOffCugraphSourceError {
            source_error_observed: true,
            unsupported_operator: true,
            operation_matches: true,
            reason_matches: true,
            required_feature_matches: true,
            host_fallback_matches: true,
        };
        let evidence = FeatureOffCugraphEvidence::from_observations(
            profile,
            &[relational.clone(), warmup_graph.clone(), graph.clone()],
            &[(1, exact), (2, exact)],
            0,
            0,
        );
        assert_eq!(evidence.status, ClauseStatus::Evidenced);
        assert_eq!(evidence.exact_structured_rejections, 2);
        assert_eq!(evidence.warmup_probe_attempts, 1);
        assert_eq!(evidence.measurement_probe_attempts, 1);
        assert_eq!(evidence.relational_gpu_reaching_successes, 1);
        assert_eq!(evidence.max_observed_measurement_rejection_latency_ns, Some(10_000_000));

        let missing_declared_measurement_probe = FeatureOffCugraphEvidence::from_observations(
            FeatureOffCugraphProbeProfile { measurement_probe_attempts: 2, ..profile },
            &[relational.clone(), warmup_graph.clone(), graph.clone()],
            &[(1, exact), (2, exact)],
            0,
            0,
        );
        assert_eq!(missing_declared_measurement_probe.status, ClauseStatus::Failed);

        let task_loss = FeatureOffCugraphEvidence::from_observations(
            profile,
            &[relational.clone(), warmup_graph.clone(), graph.clone()],
            &[(1, exact), (2, exact)],
            0,
            1,
        );
        assert_eq!(task_loss.status, ClauseStatus::Failed);
        assert_eq!(task_loss.join_failures, 1);

        let begin_failure = FeatureOffCugraphEvidence::from_observations(
            profile,
            &[relational.clone(), warmup_graph.clone(), graph.clone()],
            &[(1, exact), (2, exact)],
            1,
            0,
        );
        assert_eq!(begin_failure.status, ClauseStatus::Failed);
        assert_eq!(begin_failure.begin_failures, 1);

        graph.terminal_evidence.native_fault_evidence = Some(json!({"kind": "unexpected"}));
        let rejected = FeatureOffCugraphEvidence::from_observations(
            profile,
            &[relational, warmup_graph, graph],
            &[(1, exact), (2, exact)],
            0,
            0,
        );
        assert_eq!(rejected.status, ClauseStatus::Failed);
        assert_eq!(rejected.native_fault_evidence_count, 1);
    }

    #[test]
    fn feature_off_cell_ignores_relational_native_fault_evidence() {
        let mut relational = observation();
        relational.outcome = SUCCEEDED_OUTCOME.to_string();
        relational.succeeded = true;
        relational.failure = None;
        relational.terminal_evidence.native_fault_evidence = Some(json!({"kind": "relational"}));

        let mut warmup_graph = observation();
        warmup_graph.sequence = 1;
        warmup_graph.phase = SchedulePhase::Warmup;
        warmup_graph.query_label = FEATURE_OFF_CUGRAPH_QUERY_LABEL.to_string();
        warmup_graph.graph_workload = true;
        warmup_graph.disposition = "Unplanned".to_string();
        warmup_graph.device_ordinal = None;
        warmup_graph.admission_sequence = None;
        warmup_graph.terminal_evidence.resource_evidence = None;

        let mut graph = warmup_graph.clone();
        graph.sequence = 2;
        graph.phase = SchedulePhase::Measurement;
        graph.total_ns = Some(10_000_000);

        let profile = FeatureOffCugraphProbeProfile {
            max_rejection_latency_ms: 25,
            warmup_probe_attempts: 1,
            measurement_probe_attempts: 1,
            min_relational_gpu_reaching_attempts: 1,
        };
        let exact = FeatureOffCugraphSourceError {
            source_error_observed: true,
            unsupported_operator: true,
            operation_matches: true,
            reason_matches: true,
            required_feature_matches: true,
            host_fallback_matches: true,
        };
        let evidence = FeatureOffCugraphEvidence::from_observations(
            profile,
            &[relational, warmup_graph, graph],
            &[(1, exact), (2, exact)],
            0,
            0,
        );

        assert_eq!(evidence.status, ClauseStatus::Evidenced);
        assert_eq!(evidence.native_fault_evidence_count, 0);
    }
}
