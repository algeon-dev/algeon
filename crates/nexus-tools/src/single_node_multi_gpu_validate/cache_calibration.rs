//! Paired P6.8 cache calibration evidence.
//!
//! A cache-disabled companion is derived from the same validated opt-in
//! profile by zeroing the one P5 control, `cache_cap_bytes`.  This prevents a
//! calibration from silently changing its workload mix, queue contract, or
//! device envelope while comparing the two states.

use serde::Serialize;

use super::{aggregate::ClauseStatus, artifact::MeasuredRunArtifact};

#[derive(Debug, Clone, Serialize)]
pub(super) struct CacheCalibrationClause {
    pub(super) clause: &'static str,
    pub(super) status: ClauseStatus,
    pub(super) detail: String,
}

/// Two sequential public-backend measurements and their cache-accounting
/// comparison. The full measurements remain embedded so QPS and latency are
/// evaluated from their own report rows rather than from a third cache store.
#[derive(Debug, Serialize)]
pub(super) struct CacheCalibrationEvidence {
    pub(super) cache_disabled: MeasuredRunArtifact,
    pub(super) cache_opt_in: MeasuredRunArtifact,
    pub(super) clauses: Vec<CacheCalibrationClause>,
}

impl CacheCalibrationEvidence {
    pub(super) fn new(cache_disabled: MeasuredRunArtifact, cache_opt_in: MeasuredRunArtifact) -> Self {
        let disabled_peak = peak_cache_residency(&cache_disabled);
        let opt_in_peak = peak_cache_residency(&cache_opt_in);
        let disabled_caps = cache_disabled
            .aggregate
            .placement
            .iter()
            .map(|placement| placement.configuration.cache_cap_bytes)
            .collect::<Vec<_>>();
        let opt_in_caps = cache_opt_in
            .aggregate
            .placement
            .iter()
            .map(|placement| placement.configuration.cache_cap_bytes)
            .collect::<Vec<_>>();
        let mut clauses = vec![state_clause(
            "cache_disabled_baseline",
            cache_disabled.cache_state == "disabled" && disabled_caps.iter().all(|cap| *cap == 0),
            format!("state={} with {} ordinal-local cache caps", cache_disabled.cache_state, disabled_caps.len()),
        )];
        clauses.push(state_clause(
            "cache_opt_in_profile",
            cache_opt_in.cache_state == "enabled" && opt_in_caps.iter().any(|cap| *cap != 0),
            format!("state={} with {} ordinal-local cache caps", cache_opt_in.cache_state, opt_in_caps.len()),
        ));
        clauses.push(match disabled_peak {
            Some(0) => CacheCalibrationClause {
                clause: "disabled_ledger_residency",
                status: ClauseStatus::Evidenced,
                detail: "all retained disabled-run ledger samples reported zero cache residency".to_string(),
            },
            Some(bytes) => CacheCalibrationClause {
                clause: "disabled_ledger_residency",
                status: ClauseStatus::Failed,
                detail: format!("disabled run retained {bytes} bytes of cache residency"),
            },
            None => CacheCalibrationClause {
                clause: "disabled_ledger_residency",
                status: ClauseStatus::NotEvidenced,
                detail: "disabled run retained no ledger samples for cache accounting".to_string(),
            },
        });
        clauses.push(match opt_in_peak {
            Some(bytes) if bytes > 0 => CacheCalibrationClause {
                clause: "opt_in_ledger_residency",
                status: ClauseStatus::Evidenced,
                detail: format!("opt-in run reached {bytes} bytes of ledger-accounted cache residency"),
            },
            Some(_) => CacheCalibrationClause {
                clause: "opt_in_ledger_residency",
                status: ClauseStatus::NotEvidenced,
                detail: "opt-in run completed without a retained cache-residency sample".to_string(),
            },
            None => CacheCalibrationClause {
                clause: "opt_in_ledger_residency",
                status: ClauseStatus::NotEvidenced,
                detail: "opt-in run retained no ledger samples for cache accounting".to_string(),
            },
        });
        Self { cache_disabled, cache_opt_in, clauses }
    }

    /// P6.8 requires direct residency evidence in both halves of the paired
    /// run, so unlike ordinary runner gaps a `not_evidenced` cache clause keeps
    /// the calibration non-qualifying.
    pub(super) fn qualifies(&self) -> bool {
        self.clauses.iter().all(|clause| clause.status == ClauseStatus::Evidenced)
            && self.cache_disabled.qualifying
            && self.cache_opt_in.qualifying
    }
}

fn state_clause(clause: &'static str, valid: bool, detail: String) -> CacheCalibrationClause {
    CacheCalibrationClause {
        clause,
        status: if valid { ClauseStatus::Evidenced } else { ClauseStatus::Failed },
        detail,
    }
}

/// Finds the largest total resident cache charge observed in the canonical
/// ledger reports. They are public, versioned snapshot DTOs retained verbatim
/// in the artifact, so this does not manufacture a parallel cache counter.
fn peak_cache_residency(run: &MeasuredRunArtifact) -> Option<u64> {
    let mut peak = None;
    for sample in &run.snapshots {
        let Some(reports) = sample.device_ledger_reports.as_array() else {
            continue;
        };
        for report in reports {
            let Some(evictable) = report.get("cache_evictable_resident_bytes").and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            let Some(pinned) = report.get("cache_pinned_in_use_bytes").and_then(serde_json::Value::as_u64) else {
                continue;
            };
            peak = Some(peak.unwrap_or(0_u64).max(evictable.saturating_add(pinned)));
        }
    }
    peak
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_measurements_do_not_claim_cache_residency() {
        // This test pins the conservative rule at the owning artifact layer:
        // a run with no ledger samples is never interpreted as zero cache use.
        assert_eq!(peak_cache_residency(&empty_measurement()), None);
    }

    fn empty_measurement() -> MeasuredRunArtifact {
        use crate::single_node_multi_gpu_validate::{
            aggregate::{MeasuredAggregate, MeasurementWindow, ReportAggregateEquality},
            artifact::{ArrivalPlanEvidence, BackendEvidence, TraceEvidence},
            metrics_audit::MetricsAudit,
        };

        let aggregate = MeasuredAggregate::reduce(&[], &[], &Default::default(), 0, 1);
        MeasuredRunArtifact {
            qualifying: false,
            backend: BackendEvidence {
                resource_core: 0,
                observation_core: 0,
                resource_generation: 0,
                shared_session_count: 0,
                queue_max_queued_attempts: 0,
                queue_max_queue_wait_ms: 0,
                queue_max_overtakes_per_waiter: 0,
                device_startup_records: serde_json::Value::Null,
                close: test_close_evidence(),
            },
            selected_queries: Vec::new(),
            arrival_plan: ArrivalPlanEvidence {
                arrival_mode: "concurrency",
                seed: 0,
                concurrency: 0,
                arrival_rate_per_second: None,
                max_rounds: None,
                rate_overrun: "delay",
                planned_arrivals: 0,
                planned_warmup_arrivals: None,
                planned_measurement_arrivals: None,
                truncated_by_attempt_cap: false,
                dropped_overrun_arrivals: 0,
                begin_failures: 0,
                join_failures: 0,
            },
            cache_state: "disabled",
            aggregate: MeasuredAggregate {
                window: MeasurementWindow {
                    warmup_ns: 0,
                    measurement_ns: 1,
                    warmup_attempts: 0,
                    measurement_attempts: 0,
                },
                throughput: aggregate.throughput,
                latency: aggregate.latency,
                placement: aggregate.placement,
                attempts: aggregate.attempts,
                scenarios: aggregate.scenarios,
                quarantine_fallback: None,
                report_aggregate_equality: ReportAggregateEquality {
                    rows_measurement_attempts: 0,
                    aggregate_measurement_attempts: 0,
                    rows_succeeded: 0,
                    aggregate_succeeded: 0,
                    rows_placed: 0,
                    aggregate_placed: 0,
                    rows_total: 0,
                    aggregate_total: 0,
                    equal: true,
                },
            },
            deadline: crate::single_node_multi_gpu_validate::artifact::RunDeadlineEvidence {
                configured_window_ns: 1,
                elapsed_until_reports_ns: 1,
                overrun_ns: 0,
                reports_sealed_within_deadline: true,
            },
            cancellation_barrier: None,
            deadline_barrier: None,
            domain_exhaustion: None,
            physical_allocator_pressure: None,
            metrics_audit: MetricsAudit::from_snapshots(&[], &[], 0, 250, false),
            snapshots: Vec::new(),
            trace: TraceEvidence { captured_by_runner: false, uri: None, sha256: None },
        }
    }

    fn test_close_evidence() -> crate::single_node_multi_gpu_validate::artifact::BackendCloseEvidence {
        use crate::single_node_multi_gpu_validate::artifact::{
            BackendCloseEvidence, BackendCloseOutcome, BackendCloseSnapshotEvidence,
        };

        let snapshot = BackendCloseSnapshotEvidence {
            closing: true,
            gpu_resource_quiescent: true,
            observation_quiescent: true,
            active_gpu_permits: 0,
            active_observation_permits: 0,
            cleanup_failed: false,
            cleanup_deadline_elapsed_count: 0,
            cleanup_late_success_count: 0,
        };
        BackendCloseEvidence { before: snapshot, outcome: BackendCloseOutcome::Quiescent { snapshot }, after: snapshot }
    }
}
