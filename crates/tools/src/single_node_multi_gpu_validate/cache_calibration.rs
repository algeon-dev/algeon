//! Paired cache calibration evidence.
//!
//! A cache-disabled companion is derived from the same validated opt-in
//! profile by zeroing the one cache control, `cache_cap_bytes`. This prevents a
//! calibration from silently changing its workload mix, queue contract, or
//! device envelope while comparing the two states.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

use super::{
    aggregate::{AttemptRow, ClauseStatus, SUCCEEDED_OUTCOME},
    artifact::MeasuredRunArtifact,
    profile::{CacheStressProfile, FixtureKind},
};

#[derive(Debug, Clone, Serialize)]
pub(super) struct CacheCalibrationClause {
    pub(super) clause: &'static str,
    pub(super) status: ClauseStatus,
    pub(super) detail: String,
}

/// Direct samples of the one cuDF cache allocation domain paired with the
/// ledger observation read by that same domain owner.  This deliberately does
/// not try to derive cache residency from aggregate CUDA/NVML free bytes:
/// those include unrelated attempt allocations and allocator-pool state.
#[derive(Debug, Clone, Default, Serialize)]
pub(super) struct CacheSamplingEvidence {
    pub(super) samples_with_cache_reports: u64,
    pub(super) expected_enabled_ordinals: Vec<usize>,
    pub(super) observed_enabled_ordinals: Vec<usize>,
    pub(super) missing_report_samples: u64,
    pub(super) cap_mismatch_samples: u64,
    pub(super) logical_over_cap_samples: u64,
    pub(super) physical_over_ledger_samples: u64,
    pub(super) peak_ledger_resident_bytes: u64,
    pub(super) peak_physical_current_bytes: u64,
    pub(super) source_lookup_hits: u64,
    pub(super) source_lookup_misses: u64,
    pub(super) source_promotions_inserted: u64,
    pub(super) source_promotions_coalesced: u64,
    pub(super) source_promoted_device_bytes: u64,
    pub(super) source_eviction_batches: u64,
    pub(super) source_evicted_device_bytes: u64,
    pub(super) graph_lookup_hits: u64,
    pub(super) graph_lookup_misses: u64,
    pub(super) graph_insertions: u64,
    pub(super) graph_duplicate_inserts: u64,
    pub(super) bfs_singleflight_eligible_requests: u64,
    pub(super) bfs_singleflight_physical_traversals: u64,
    pub(super) bfs_singleflight_follower_copies: u64,
    pub(super) bfs_singleflight_groups_formed: u64,
    pub(super) bfs_singleflight_leader_failures: u64,
    pub(super) bfs_singleflight_peak_width: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct CacheActivityCounters {
    source_lookup_hits: u64,
    source_lookup_misses: u64,
    source_promotions_inserted: u64,
    source_promotions_coalesced: u64,
    source_promoted_device_bytes: u64,
    source_eviction_batches: u64,
    source_evicted_device_bytes: u64,
    graph_lookup_hits: u64,
    graph_lookup_misses: u64,
    graph_insertions: u64,
    graph_duplicate_inserts: u64,
}

/// Sealed evidence from the dedicated, cold-cache same-key probe. The
/// probe runs source and BFS cohorts separately on a fresh single-ordinal
/// backend; its snapshots remain public backend DTOs, not a runner-owned
/// cache event store.
#[derive(Debug, Clone, Serialize)]
pub(super) struct CacheSameKeyProbeEvidence {
    pub(super) target_device_ordinal: usize,
    pub(super) cohort_width: usize,
    pub(super) source: CacheSameKeyCohortEvidence,
    pub(super) bfs: CacheSameKeyCohortEvidence,
    pub(super) close_quiescent: bool,
}

/// One source or BFS cohort's exact reports plus the public before/after
/// domain snapshots used by the pure cache-calibration reducer.
#[derive(Debug, Clone, Serialize)]
pub(super) struct CacheSameKeyCohortEvidence {
    /// Runner-visible cohort label. Attempt query ids are intentionally not
    /// retained; only their one-way cache/coordinator correlations are.
    pub(super) cohort_id: String,
    pub(super) expected_correlation_fingerprints: Vec<String>,
    pub(super) fixture_kind: FixtureKind,
    pub(super) query_label: String,
    pub(super) submitted_attempts: u64,
    pub(super) begin_failures: u64,
    pub(super) join_failures: u64,
    pub(super) measurement_attempts: u64,
    pub(super) measurement_successes: u64,
    pub(super) successful_attempts: u64,
    pub(super) target_ordinal_attempts: u64,
    pub(super) other_ordinal_attempts: u64,
    pub(super) cache_reports_before: Value,
    pub(super) cache_reports_after: Value,
    pub(super) source_cache_events_before: Value,
    pub(super) source_cache_events_after: Value,
    pub(super) bfs_singleflight_before: Value,
    pub(super) bfs_singleflight_after: Value,
}

/// Two sequential public-backend measurements and their cache-accounting
/// comparison. The full measurements remain embedded so QPS and latency are
/// evaluated from their own report rows rather than from a third cache store.
#[derive(Debug, Serialize)]
pub(super) struct CacheCalibrationEvidence {
    pub(super) cache_disabled: MeasuredRunArtifact,
    pub(super) cache_opt_in: MeasuredRunArtifact,
    pub(super) sampling: CacheSamplingEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) same_key_probe: Option<CacheSameKeyProbeEvidence>,
    pub(super) clauses: Vec<CacheCalibrationClause>,
}

impl CacheCalibrationEvidence {
    pub(super) fn new(
        cache_disabled: MeasuredRunArtifact,
        cache_opt_in: MeasuredRunArtifact,
        stress: Option<CacheStressProfile>,
    ) -> Self {
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
        let sampling = cache_sampling_evidence(&cache_opt_in);
        let same_key_probe = cache_opt_in.cache_same_key_probe.clone();
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
        clauses.push(cache_sampling_clause(
            "per_ordinal_six_gib_cache_cap",
            !opt_in_caps.is_empty() && opt_in_caps.iter().all(|cap| *cap == six_gib()),
            sampling.samples_with_cache_reports != 0,
            format!("configured ordinal-local caps={opt_in_caps:?}; required exactly {} bytes (6 GiB)", six_gib()),
        ));
        clauses.push(cache_sampling_clause(
            "cache_domain_matches_ledger_profile",
            sampling.cap_mismatch_samples == 0 && sampling.missing_report_samples == 0,
            sampling.samples_with_cache_reports != 0,
            format!(
                "{} cache-report samples; {} missing ordinal reports; {} cap mismatches",
                sampling.samples_with_cache_reports, sampling.missing_report_samples, sampling.cap_mismatch_samples
            ),
        ));
        clauses.push(cache_sampling_clause(
            "per_sample_cache_cap_and_physical_ownership",
            sampling.logical_over_cap_samples == 0 && sampling.physical_over_ledger_samples == 0,
            sampling.samples_with_cache_reports != 0,
            format!(
                "peak ledger={} bytes; peak physical={} bytes; logical-over-cap samples={}; physical-over-ledger samples={}",
                sampling.peak_ledger_resident_bytes,
                sampling.peak_physical_current_bytes,
                sampling.logical_over_cap_samples,
                sampling.physical_over_ledger_samples,
            ),
        ));
        if let Some(stress) = stress.as_ref() {
            clauses.extend(stress_clauses(&cache_disabled, &cache_opt_in, &sampling, stress, same_key_probe.as_ref()));
        }
        Self { cache_disabled, cache_opt_in, sampling, same_key_probe, clauses }
    }

    /// Paired calibration requires direct residency evidence in both halves of the paired
    /// run, so unlike ordinary runner gaps a `not_evidenced` cache clause keeps
    /// the calibration non-qualifying.
    pub(super) fn qualifies(&self) -> bool {
        self.clauses.iter().all(|clause| clause.status == ClauseStatus::Evidenced)
            && self.cache_disabled.qualifying
            && self.cache_opt_in.qualifying
    }
}

const fn six_gib() -> u64 {
    6 * 1024 * 1024 * 1024
}

fn state_clause(clause: &'static str, valid: bool, detail: String) -> CacheCalibrationClause {
    CacheCalibrationClause {
        clause,
        status: if valid { ClauseStatus::Evidenced } else { ClauseStatus::Failed },
        detail,
    }
}

fn cache_sampling_clause(clause: &'static str, valid: bool, sampled: bool, detail: String) -> CacheCalibrationClause {
    CacheCalibrationClause {
        clause,
        status: if !sampled {
            ClauseStatus::NotEvidenced
        } else if valid {
            ClauseStatus::Evidenced
        } else {
            ClauseStatus::Failed
        },
        detail,
    }
}

fn stress_clauses(
    cache_disabled: &MeasuredRunArtifact,
    cache_opt_in: &MeasuredRunArtifact,
    sampling: &CacheSamplingEvidence,
    stress: &CacheStressProfile,
    same_key_probe: Option<&CacheSameKeyProbeEvidence>,
) -> Vec<CacheCalibrationClause> {
    let mut clauses = vec![cache_sampling_clause(
        "s7_required_per_device_cap",
        !sampling.expected_enabled_ordinals.is_empty()
            && sampling.cap_mismatch_samples == 0
            && cache_opt_in
                .aggregate
                .placement
                .iter()
                .all(|placement| placement.configuration.cache_cap_bytes == stress.required_per_device_cap_bytes),
        sampling.samples_with_cache_reports != 0,
        format!(
            "required cap={} bytes; observed enabled ordinals={:?}",
            stress.required_per_device_cap_bytes, sampling.observed_enabled_ordinals
        ),
    )];
    if stress.minimum_source_promoted_bytes != 0 || stress.require_source_eviction {
        clauses.push(cache_sampling_clause(
            "s7_thrash_working_set_and_eviction",
            sampling.source_promoted_device_bytes >= stress.minimum_source_promoted_bytes
                && (!stress.require_source_eviction || sampling.source_eviction_batches != 0),
            sampling.samples_with_cache_reports != 0,
            format!(
                "cumulative source promotion={} bytes (minimum={}); eviction batches={} (required={})",
                sampling.source_promoted_device_bytes,
                stress.minimum_source_promoted_bytes,
                sampling.source_eviction_batches,
                stress.require_source_eviction,
            ),
        ));
    }
    if stress.require_source_coalescing {
        if let Some(probe) = same_key_probe {
            clauses.push(same_key_source_clause(probe));
        } else {
            clauses.push(cache_sampling_clause(
                "s7_source_same_key_coalescing",
                sampling.source_promotions_coalesced != 0,
                sampling.samples_with_cache_reports != 0,
                format!(
                    "source cache coalesced {} promotions across {} lookups",
                    sampling.source_promotions_coalesced,
                    sampling.source_lookup_hits.saturating_add(sampling.source_lookup_misses),
                ),
            ));
        }
    }
    if stress.require_bfs_singleflight {
        if let Some(probe) = same_key_probe {
            clauses.push(same_key_bfs_clause(probe));
        } else {
            let sampled = sampling.bfs_singleflight_eligible_requests != 0;
            clauses.push(cache_sampling_clause(
                "s7_bfs_same_key_singleflight",
                sampling.bfs_singleflight_follower_copies != 0
                    && sampling.bfs_singleflight_physical_traversals < sampling.bfs_singleflight_eligible_requests
                    && sampling.bfs_singleflight_leader_failures == 0,
                sampled,
                format!(
                    "eligible={} physical_traversals={} follower_copies={} groups={} leader_failures={} peak_width={}",
                    sampling.bfs_singleflight_eligible_requests,
                    sampling.bfs_singleflight_physical_traversals,
                    sampling.bfs_singleflight_follower_copies,
                    sampling.bfs_singleflight_groups_formed,
                    sampling.bfs_singleflight_leader_failures,
                    sampling.bfs_singleflight_peak_width,
                ),
            ));
        }
    }
    if stress.require_explained_admission_delta {
        let delta = admission_delta(cache_disabled, cache_opt_in);
        clauses.push(CacheCalibrationClause {
            clause: "s7_cache_on_admission_delta_accounted",
            status: if delta.additional_non_successes == 0 || delta.unexplained_non_successes == 0 {
                ClauseStatus::Evidenced
            } else {
                ClauseStatus::Failed
            },
            detail: format!(
                "cache-on-only non-successes={} unexplained={} (an additional outcome needs sealed nonzero cache residency)",
                delta.additional_non_successes, delta.unexplained_non_successes
            ),
        });
    }
    clauses
}

fn same_key_source_clause(probe: &CacheSameKeyProbeEvidence) -> CacheCalibrationClause {
    let before = target_cache_report(&probe.source.cache_reports_before, probe.target_device_ordinal);
    let after = target_cache_report(&probe.source.cache_reports_after, probe.target_device_ordinal);
    let required_followers = probe.cohort_width.saturating_sub(1) as u64;
    let source_before = before.and_then(|report| cache_activity_counter(report, "source", "promotions_inserted"));
    let source_after = after.and_then(|report| cache_activity_counter(report, "source", "promotions_inserted"));
    let source_entries_before = before.and_then(|report| cache_activity_counter(report, "source", "entry_count"));
    let source_entries_after = after.and_then(|report| cache_activity_counter(report, "source", "entry_count"));
    let coalesced_before = before.and_then(|report| cache_activity_counter(report, "source", "promotions_coalesced"));
    let coalesced_after = after.and_then(|report| cache_activity_counter(report, "source", "promotions_coalesced"));
    let source_loads_before =
        before.and_then(|report| cache_activity_counter(report, "source", "promotions_in_progress"));
    let source_loads_after =
        after.and_then(|report| cache_activity_counter(report, "source", "promotions_in_progress"));
    let source_events =
        source_event_delta(&probe.source.source_cache_events_before, &probe.source.source_cache_events_after);
    let source_event_count = source_events.as_ref().map(Vec::len);
    let source_events_match = source_events
        .as_ref()
        .is_some_and(|events| source_events_match_cohort(events, &probe.source, probe.target_device_ordinal, before));
    let source_ledger_delta = before
        .zip(after)
        .and_then(|(before, after)| cache_ledger_bytes(after)?.checked_sub(cache_ledger_bytes(before)?));
    let inserted_event_bytes =
        source_events.as_ref().and_then(|events| action_retained_bytes(events, "promotion_inserted"));
    let valid = before.is_some()
        && after.is_some()
        && probe.close_quiescent
        && cohort_on_target(&probe.source, probe.cohort_width, probe.target_device_ordinal)
        && source_events_match
        && source_before == Some(0)
        && source_entries_before == Some(0)
        && source_loads_before == Some(0)
        && source_loads_after == Some(0)
        && source_entries_after == Some(1)
        && source_after.zip(source_before).and_then(|(after, before)| after.checked_sub(before)) == Some(1)
        && coalesced_after
            .zip(coalesced_before)
            .and_then(|(after, before)| after.checked_sub(before))
            .is_some_and(|delta| delta >= required_followers)
        && source_ledger_delta.is_some_and(|bytes| bytes != 0 && Some(bytes) == inserted_event_bytes)
        && domain_snapshot_pair_is_accounted(
            before.expect("source cache reports were checked above"),
            after.expect("source cache reports were checked above"),
        );
    CacheCalibrationClause {
        clause: "s7_source_same_key_coalescing",
        status: if valid { ClauseStatus::Evidenced } else { ClauseStatus::Failed },
        detail: format!(
            "ordinal={} cohort={} cohort_width={} source_events={source_event_count:?}; source_event_contract={source_events_match}; source_ledger_delta={source_ledger_delta:?}; inserted_event_bytes={inserted_event_bytes:?}; source entries {:?}->{:?}; inserted {:?}->{:?}; coalesced {:?}->{:?}; measurement={}/{} successful={} target={} other={} close_quiescent={}",
            probe.target_device_ordinal,
            probe.source.cohort_id,
            probe.cohort_width,
            source_entries_before,
            source_entries_after,
            source_before,
            source_after,
            coalesced_before,
            coalesced_after,
            probe.source.measurement_successes,
            probe.source.measurement_attempts,
            probe.source.successful_attempts,
            probe.source.target_ordinal_attempts,
            probe.source.other_ordinal_attempts,
            probe.close_quiescent,
        ),
    }
}

fn same_key_bfs_clause(probe: &CacheSameKeyProbeEvidence) -> CacheCalibrationClause {
    let before = target_cache_report(&probe.bfs.cache_reports_before, probe.target_device_ordinal);
    let after = target_cache_report(&probe.bfs.cache_reports_after, probe.target_device_ordinal);
    let required_followers = probe.cohort_width.saturating_sub(1) as u64;
    let eligible =
        counter_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after, "eligible_requests");
    let physical =
        counter_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after, "physical_traversals");
    let followers =
        counter_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after, "follower_copies");
    let groups = counter_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after, "groups_formed");
    let overflow_groups =
        counter_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after, "overflow_groups");
    let leader_failures =
        counter_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after, "leader_failures");
    let cancelled_waiters =
        counter_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after, "cancelled_waiters");
    let graph_insertions =
        before.zip(after).and_then(|(before, after)| activity_delta(before, after, "graph", "insertions"));
    let graph_duplicates =
        before.zip(after).and_then(|(before, after)| activity_delta(before, after, "graph", "duplicate_inserts"));
    let bfs_events = bfs_event_delta(&probe.bfs.bfs_singleflight_before, &probe.bfs.bfs_singleflight_after);
    let bfs_event_count = bfs_events.as_ref().map(Vec::len);
    let bfs_events_match = bfs_events
        .as_ref()
        .is_some_and(|events| bfs_events_match_cohort(events, &probe.bfs, probe.target_device_ordinal));
    let valid = before.is_some()
        && after.is_some()
        && bfs_snapshot_complete(&probe.bfs.bfs_singleflight_before)
        && bfs_snapshot_complete(&probe.bfs.bfs_singleflight_after)
        && probe.close_quiescent
        && cohort_on_target(&probe.bfs, probe.cohort_width, probe.target_device_ordinal)
        && bfs_events_match
        && graph_counter(before, "available_entries") == Some(0)
        && graph_counter(before, "insertions") == Some(0)
        && graph_counter(before, "duplicate_inserts") == Some(0)
        && graph_counter(after, "available_entries") == Some(1)
        && bfs_snapshot_is_fresh(&probe.bfs.bfs_singleflight_before)
        && eligible == Some(probe.cohort_width as u64)
        && physical == Some(1)
        && followers == Some(required_followers)
        && groups == Some(1)
        && overflow_groups == Some(0)
        && leader_failures == Some(0)
        && cancelled_waiters == Some(0)
        && graph_insertions == Some(1)
        && graph_duplicates == Some(0)
        && domain_snapshot_pair_is_accounted(
            before.expect("BFS cache reports were checked above"),
            after.expect("BFS cache reports were checked above"),
        );
    CacheCalibrationClause {
        clause: "s7_bfs_same_key_singleflight",
        status: if valid { ClauseStatus::Evidenced } else { ClauseStatus::Failed },
        detail: format!(
            "ordinal={} cohort={} cohort_width={} bfs_events={bfs_event_count:?}; bfs_event_contract={bfs_events_match}; eligible={eligible:?} physical={physical:?} followers={followers:?} groups={groups:?} overflow_groups={overflow_groups:?} leader_failures={leader_failures:?} cancelled_waiters={cancelled_waiters:?} graph_insertions={graph_insertions:?} graph_duplicates={graph_duplicates:?} measurement={}/{} successful={} target={} other={} close_quiescent={}",
            probe.target_device_ordinal,
            probe.bfs.cohort_id,
            probe.cohort_width,
            probe.bfs.measurement_successes,
            probe.bfs.measurement_attempts,
            probe.bfs.successful_attempts,
            probe.bfs.target_ordinal_attempts,
            probe.bfs.other_ordinal_attempts,
            probe.close_quiescent,
        ),
    }
}

fn cohort_on_target(cohort: &CacheSameKeyCohortEvidence, width: usize, target_ordinal: usize) -> bool {
    cohort.submitted_attempts == width as u64
        && cohort.begin_failures == 0
        && cohort.join_failures == 0
        && cohort.measurement_attempts == width as u64
        && cohort.measurement_successes == width as u64
        && cohort.successful_attempts == width as u64
        && cohort.target_ordinal_attempts == width as u64
        && cohort.other_ordinal_attempts == 0
        && target_cache_report(&cohort.cache_reports_before, target_ordinal).is_some()
        && target_cache_report(&cohort.cache_reports_after, target_ordinal).is_some()
}

fn target_cache_report(reports: &Value, target_ordinal: usize) -> Option<&Value> {
    let reports = reports.as_array()?;
    if reports.len() != 1 {
        return None;
    }
    let report = reports.first()?;
    (report.get("device_ordinal").and_then(Value::as_u64) == Some(target_ordinal as u64)).then_some(report)
}

fn cache_activity_counter(report: &Value, activity: &str, field: &str) -> Option<u64> {
    report.get(activity)?.get(field)?.as_u64()
}

fn graph_counter(report: Option<&Value>, field: &str) -> Option<u64> {
    report.and_then(|report| cache_activity_counter(report, "graph", field))
}

fn activity_delta(before: &Value, after: &Value, activity: &str, field: &str) -> Option<u64> {
    cache_activity_counter(after, activity, field)?.checked_sub(cache_activity_counter(before, activity, field)?)
}

fn counter_delta(before: &Value, after: &Value, field: &str) -> Option<u64> {
    after.get(field)?.as_u64()?.checked_sub(before.get(field)?.as_u64()?)
}

fn bfs_snapshot_complete(snapshot: &Value) -> bool {
    [
        "eligible_requests",
        "physical_traversals",
        "follower_copies",
        "groups_formed",
        "overflow_groups",
        "leader_failures",
        "cancelled_waiters",
    ]
    .iter()
    .all(|field| snapshot.get(*field).and_then(Value::as_u64).is_some())
        && snapshot.get("events").and_then(Value::as_array).is_some()
}

fn bfs_snapshot_is_fresh(snapshot: &Value) -> bool {
    [
        "eligible_requests",
        "physical_traversals",
        "follower_copies",
        "groups_formed",
        "overflow_groups",
        "leader_failures",
        "cancelled_waiters",
    ]
    .iter()
    .all(|field| snapshot.get(*field).and_then(Value::as_u64) == Some(0))
        && snapshot.get("events").and_then(Value::as_array).is_some_and(Vec::is_empty)
}

fn source_event_delta<'a>(before: &'a Value, after: &'a Value) -> Option<Vec<&'a Value>> {
    event_delta(before, after)
}

fn bfs_event_delta<'a>(before: &'a Value, after: &'a Value) -> Option<Vec<&'a Value>> {
    event_delta(before.get("events")?, after.get("events")?)
}

fn event_delta<'a>(before: &'a Value, after: &'a Value) -> Option<Vec<&'a Value>> {
    let before = before.as_array()?;
    let after = after.as_array()?;
    let mut before_sequences = BTreeSet::new();
    for event in before {
        if !before_sequences.insert(event.get("sequence")?.as_u64()?) {
            return None;
        }
    }
    let mut after_sequences = BTreeSet::new();
    for event in after {
        if !after_sequences.insert(event.get("sequence")?.as_u64()?) {
            return None;
        }
    }
    let delta = after
        .iter()
        .filter(|event| !before_sequences.contains(&event.get("sequence").and_then(Value::as_u64).unwrap_or_default()))
        .collect::<Vec<_>>();
    if let Some(last_before) = before_sequences.last()
        && delta
            .iter()
            .any(|event| event.get("sequence").and_then(Value::as_u64).is_none_or(|sequence| sequence <= *last_before))
    {
        return None;
    }
    Some(delta)
}

fn source_events_match_cohort(
    events: &[&Value],
    cohort: &CacheSameKeyCohortEvidence,
    target_ordinal: usize,
    cache_report: Option<&Value>,
) -> bool {
    let Some(domain) = cache_report.and_then(|report| report.get("allocation_domain_id")).and_then(Value::as_u64)
    else {
        return false;
    };
    let Some(expected) = expected_correlations(cohort) else {
        return false;
    };
    let Some(correlations) = event_string_set(events, "correlation_fingerprint") else {
        return false;
    };
    let Some(keys) = event_string_set(events, "cache_key_fingerprint") else {
        return false;
    };
    let Some(leader_correlations) = event_correlations_for_action(events, "load_leader") else {
        return false;
    };
    let Some(follower_correlations) = event_correlations_for_action(events, "load_follower") else {
        return false;
    };
    let Some(hit_correlations) = event_correlations_for_action(events, "load_hit") else {
        return false;
    };
    let Some(promotion_correlations) = event_correlations_for_action(events, "promotion_inserted") else {
        return false;
    };
    let expected_followers = expected.difference(&leader_correlations).cloned().collect::<BTreeSet<_>>();
    events.len() == expected.len().saturating_mul(2)
        && events.iter().all(|event| {
            event.get("device_ordinal").and_then(Value::as_u64) == Some(target_ordinal as u64)
                && event.get("allocation_domain_id").and_then(Value::as_u64) == Some(domain)
                && matches!(
                    event.get("action").and_then(Value::as_str),
                    Some("load_leader" | "load_follower" | "load_hit" | "promotion_inserted")
                )
        })
        && correlations == expected
        && keys.len() == 1
        && action_count(events, "load_leader") == 1
        && action_count(events, "load_follower") == expected.len().saturating_sub(1)
        && action_count(events, "load_hit") == expected.len().saturating_sub(1)
        && action_count(events, "promotion_inserted") == 1
        && leader_correlations.len() == 1
        && leader_correlations.is_subset(&expected)
        && follower_correlations == expected_followers
        && hit_correlations == expected_followers
        && promotion_correlations == leader_correlations
}

fn bfs_events_match_cohort(events: &[&Value], cohort: &CacheSameKeyCohortEvidence, target_ordinal: usize) -> bool {
    let Some(expected) = expected_correlations(cohort) else {
        return false;
    };
    let Some(correlations) = event_string_set(events, "correlation_fingerprint") else {
        return false;
    };
    let Some(keys) = event_string_set(events, "singleflight_key_fingerprint") else {
        return false;
    };
    let Some(generations) = event_u64_set(events, "generation") else {
        return false;
    };
    let Some(leader_correlations) = event_correlations_for_action(events, "leader_join") else {
        return false;
    };
    let Some(follower_correlations) = event_correlations_for_action(events, "follower_join") else {
        return false;
    };
    let Some(traversal_correlations) = event_correlations_for_action(events, "leader_traversal") else {
        return false;
    };
    let Some(published_correlations) = event_correlations_for_action(events, "leader_published") else {
        return false;
    };
    let Some(copy_correlations) = event_correlations_for_action(events, "follower_copy") else {
        return false;
    };
    let expected_followers = expected.difference(&leader_correlations).cloned().collect::<BTreeSet<_>>();
    events.len() == expected.len().saturating_mul(2).saturating_add(1)
        && events.iter().all(|event| {
            event.get("device_ordinal").and_then(Value::as_u64) == Some(target_ordinal as u64)
                && matches!(
                    event.get("action").and_then(Value::as_str),
                    Some("leader_join" | "follower_join" | "leader_traversal" | "leader_published" | "follower_copy")
                )
        })
        && correlations == expected
        && keys.len() == 1
        && generations.len() == 1
        && action_count(events, "leader_join") == 1
        && action_count(events, "follower_join") == expected.len().saturating_sub(1)
        && action_count(events, "leader_traversal") == 1
        && action_count(events, "leader_published") == 1
        && action_count(events, "follower_copy") == expected.len().saturating_sub(1)
        && leader_correlations.len() == 1
        && leader_correlations.is_subset(&expected)
        && follower_correlations == expected_followers
        && traversal_correlations == leader_correlations
        && published_correlations == leader_correlations
        && copy_correlations == expected_followers
}

fn expected_correlations(cohort: &CacheSameKeyCohortEvidence) -> Option<BTreeSet<String>> {
    let expected = cohort.expected_correlation_fingerprints.iter().cloned().collect::<BTreeSet<_>>();
    (usize::try_from(cohort.submitted_attempts).ok() == Some(expected.len()) && !expected.is_empty())
        .then_some(expected)
}

fn event_string_set(events: &[&Value], field: &str) -> Option<BTreeSet<String>> {
    events.iter().map(|event| event.get(field)?.as_str().map(ToOwned::to_owned)).collect()
}

fn event_u64_set(events: &[&Value], field: &str) -> Option<BTreeSet<u64>> {
    events.iter().map(|event| event.get(field)?.as_u64()).collect()
}

fn event_correlations_for_action(events: &[&Value], action: &str) -> Option<BTreeSet<String>> {
    events
        .iter()
        .filter(|event| event.get("action").and_then(Value::as_str) == Some(action))
        .map(|event| event.get("correlation_fingerprint")?.as_str().map(ToOwned::to_owned))
        .collect()
}

fn action_count(events: &[&Value], action: &str) -> usize {
    events.iter().filter(|event| event.get("action").and_then(Value::as_str) == Some(action)).count()
}

fn action_retained_bytes(events: &[&Value], action: &str) -> Option<u64> {
    events
        .iter()
        .filter(|event| event.get("action").and_then(Value::as_str) == Some(action))
        .try_fold(0_u64, |sum, event| sum.checked_add(event.get("retained_device_bytes")?.as_u64()?))
}

fn cache_ledger_bytes(report: &Value) -> Option<u64> {
    report
        .get("ledger_evictable_resident_bytes")?
        .as_u64()?
        .checked_add(report.get("ledger_pinned_in_use_bytes")?.as_u64()?)
}

fn domain_snapshot_pair_is_accounted(before: &Value, after: &Value) -> bool {
    let before_domain = before.get("allocation_domain_id").and_then(Value::as_u64);
    let after_domain = after.get("allocation_domain_id").and_then(Value::as_u64);
    before_domain.is_some()
        && before_domain == after_domain
        && [before, after].into_iter().all(|report| {
            let cap = report.get("cache_cap_bytes").and_then(Value::as_u64);
            let limit = report.get("allocation_limit_bytes").and_then(Value::as_u64);
            let evictable = report.get("ledger_evictable_resident_bytes").and_then(Value::as_u64);
            let pinned = report.get("ledger_pinned_in_use_bytes").and_then(Value::as_u64);
            let physical = report.get("physical_current_bytes").and_then(Value::as_u64);
            matches!((cap, limit, evictable, pinned, physical), (Some(cap), Some(limit), Some(evictable), Some(pinned), Some(physical)) if cap != 0 && cap == limit && evictable.saturating_add(pinned) <= cap && physical <= evictable.saturating_add(pinned))
        })
}

#[derive(Debug, Clone, Copy, Default)]
struct AdmissionDelta {
    additional_non_successes: u64,
    unexplained_non_successes: u64,
}

fn admission_delta(cache_disabled: &MeasuredRunArtifact, cache_opt_in: &MeasuredRunArtifact) -> AdmissionDelta {
    let mut disabled_counts = BTreeMap::<(String, String, String), u64>::new();
    for attempt in cache_disabled.aggregate.attempts.iter().filter(|attempt| attempt.outcome != SUCCEEDED_OUTCOME) {
        *disabled_counts.entry(outcome_key(attempt)).or_default() += 1;
    }
    let mut delta = AdmissionDelta::default();
    for attempt in cache_opt_in.aggregate.attempts.iter().filter(|attempt| attempt.outcome != SUCCEEDED_OUTCOME) {
        let count = disabled_counts.entry(outcome_key(attempt)).or_default();
        if *count != 0 {
            *count -= 1;
            continue;
        }
        delta.additional_non_successes = delta.additional_non_successes.saturating_add(1);
        if !attempt_has_accounted_cache_residency(attempt) {
            delta.unexplained_non_successes = delta.unexplained_non_successes.saturating_add(1);
        }
    }
    delta
}

fn outcome_key(attempt: &AttemptRow) -> (String, String, String) {
    (
        attempt.disposition.clone(),
        attempt.outcome.clone(),
        attempt.failure.as_ref().map_or_else(String::new, |failure| failure.code.clone()),
    )
}

fn attempt_has_accounted_cache_residency(attempt: &AttemptRow) -> bool {
    let Some(evidence) = attempt.terminal_evidence.resource_evidence.as_ref() else {
        return false;
    };
    ["cache_evictable_resident_bytes", "cache_pinned_in_use_bytes"]
        .iter()
        .filter_map(|key| evidence.get(*key))
        .filter_map(|field| field.get("value").and_then(Value::as_u64))
        .any(|bytes| bytes != 0)
}

/// Finds the largest total resident cache charge observed in the canonical
/// ledger reports. They are public snapshot DTOs retained verbatim
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

fn cache_sampling_evidence(run: &MeasuredRunArtifact) -> CacheSamplingEvidence {
    let expected_caps = run
        .aggregate
        .placement
        .iter()
        .filter_map(|placement| {
            (placement.configuration.cache_cap_bytes != 0)
                .then_some((placement.configuration.ordinal, placement.configuration.cache_cap_bytes))
        })
        .collect::<BTreeMap<_, _>>();
    let mut observed_ordinals = BTreeSet::new();
    let mut activity_by_ordinal = BTreeMap::new();
    let mut samples_with_cache_reports = 0_u64;
    let mut missing_report_samples = 0_u64;
    let mut cap_mismatch_samples = 0_u64;
    let mut logical_over_cap_samples = 0_u64;
    let mut physical_over_ledger_samples = 0_u64;
    let mut peak_ledger_resident_bytes = 0_u64;
    let mut peak_physical_current_bytes = 0_u64;

    for sample in &run.snapshots {
        let Some(reports) = sample.device_cache_reports.as_array() else {
            continue;
        };
        if reports.is_empty() {
            continue;
        }
        samples_with_cache_reports = samples_with_cache_reports.saturating_add(1);
        let mut sample_ordinals = BTreeSet::new();
        for report in reports {
            let Some(ordinal) =
                report.get("device_ordinal").and_then(Value::as_u64).and_then(|value| usize::try_from(value).ok())
            else {
                cap_mismatch_samples = cap_mismatch_samples.saturating_add(1);
                continue;
            };
            let Some(expected_cap) = expected_caps.get(&ordinal).copied() else {
                cap_mismatch_samples = cap_mismatch_samples.saturating_add(1);
                continue;
            };
            observed_ordinals.insert(ordinal);
            sample_ordinals.insert(ordinal);
            let Some(cap) = report.get("cache_cap_bytes").and_then(Value::as_u64) else {
                cap_mismatch_samples = cap_mismatch_samples.saturating_add(1);
                continue;
            };
            let allocation_limit = report.get("allocation_limit_bytes").and_then(Value::as_u64);
            if cap != expected_cap || allocation_limit != Some(expected_cap) {
                cap_mismatch_samples = cap_mismatch_samples.saturating_add(1);
            }
            let evictable = report.get("ledger_evictable_resident_bytes").and_then(Value::as_u64);
            let pinned = report.get("ledger_pinned_in_use_bytes").and_then(Value::as_u64);
            let physical = report.get("physical_current_bytes").and_then(Value::as_u64);
            let (Some(evictable), Some(pinned), Some(physical)) = (evictable, pinned, physical) else {
                cap_mismatch_samples = cap_mismatch_samples.saturating_add(1);
                continue;
            };
            let logical = evictable.saturating_add(pinned);
            peak_ledger_resident_bytes = peak_ledger_resident_bytes.max(logical);
            peak_physical_current_bytes = peak_physical_current_bytes.max(physical);
            if logical > expected_cap {
                logical_over_cap_samples = logical_over_cap_samples.saturating_add(1);
            }
            // A cache fill may reserve before its device copy, so logical
            // bytes can temporarily exceed physical bytes.  The inverse would
            // be an uncharged physical cache allocation and is never legal.
            if physical > logical {
                physical_over_ledger_samples = physical_over_ledger_samples.saturating_add(1);
            }
            activity_by_ordinal.insert(ordinal, activity_from_report(report));
        }
        missing_report_samples = missing_report_samples
            .saturating_add(expected_caps.keys().filter(|ordinal| !sample_ordinals.contains(ordinal)).count() as u64);
    }

    let activity = activity_by_ordinal.values().copied().fold(CacheActivityCounters::default(), |mut total, value| {
        total.source_lookup_hits = total.source_lookup_hits.saturating_add(value.source_lookup_hits);
        total.source_lookup_misses = total.source_lookup_misses.saturating_add(value.source_lookup_misses);
        total.source_promotions_inserted =
            total.source_promotions_inserted.saturating_add(value.source_promotions_inserted);
        total.source_promotions_coalesced =
            total.source_promotions_coalesced.saturating_add(value.source_promotions_coalesced);
        total.source_promoted_device_bytes =
            total.source_promoted_device_bytes.saturating_add(value.source_promoted_device_bytes);
        total.source_eviction_batches = total.source_eviction_batches.saturating_add(value.source_eviction_batches);
        total.source_evicted_device_bytes =
            total.source_evicted_device_bytes.saturating_add(value.source_evicted_device_bytes);
        total.graph_lookup_hits = total.graph_lookup_hits.saturating_add(value.graph_lookup_hits);
        total.graph_lookup_misses = total.graph_lookup_misses.saturating_add(value.graph_lookup_misses);
        total.graph_insertions = total.graph_insertions.saturating_add(value.graph_insertions);
        total.graph_duplicate_inserts = total.graph_duplicate_inserts.saturating_add(value.graph_duplicate_inserts);
        total
    });
    CacheSamplingEvidence {
        samples_with_cache_reports,
        expected_enabled_ordinals: expected_caps.keys().copied().collect(),
        observed_enabled_ordinals: observed_ordinals.into_iter().collect(),
        missing_report_samples,
        cap_mismatch_samples,
        logical_over_cap_samples,
        physical_over_ledger_samples,
        peak_ledger_resident_bytes,
        peak_physical_current_bytes,
        source_lookup_hits: activity.source_lookup_hits,
        source_lookup_misses: activity.source_lookup_misses,
        source_promotions_inserted: activity.source_promotions_inserted,
        source_promotions_coalesced: activity.source_promotions_coalesced,
        source_promoted_device_bytes: activity.source_promoted_device_bytes,
        source_eviction_batches: activity.source_eviction_batches,
        source_evicted_device_bytes: activity.source_evicted_device_bytes,
        graph_lookup_hits: activity.graph_lookup_hits,
        graph_lookup_misses: activity.graph_lookup_misses,
        graph_insertions: activity.graph_insertions,
        graph_duplicate_inserts: activity.graph_duplicate_inserts,
        bfs_singleflight_eligible_requests: singleflight_counter(run, "eligible_requests"),
        bfs_singleflight_physical_traversals: singleflight_counter(run, "physical_traversals"),
        bfs_singleflight_follower_copies: singleflight_counter(run, "follower_copies"),
        bfs_singleflight_groups_formed: singleflight_counter(run, "groups_formed"),
        bfs_singleflight_leader_failures: singleflight_counter(run, "leader_failures"),
        bfs_singleflight_peak_width: singleflight_counter(run, "peak_width") as usize,
    }
}

fn activity_from_report(report: &Value) -> CacheActivityCounters {
    let source = report.get("source").unwrap_or(&Value::Null);
    let graph = report.get("graph").unwrap_or(&Value::Null);
    CacheActivityCounters {
        source_lookup_hits: value_u64(source, "lookup_hits"),
        source_lookup_misses: value_u64(source, "lookup_misses"),
        source_promotions_inserted: value_u64(source, "promotions_inserted"),
        source_promotions_coalesced: value_u64(source, "promotions_coalesced"),
        source_promoted_device_bytes: value_u64(source, "promoted_device_bytes"),
        source_eviction_batches: value_u64(source, "eviction_batches"),
        source_evicted_device_bytes: value_u64(source, "evicted_device_bytes"),
        graph_lookup_hits: value_u64(graph, "lookup_hits"),
        graph_lookup_misses: value_u64(graph, "lookup_misses"),
        graph_insertions: value_u64(graph, "insertions"),
        graph_duplicate_inserts: value_u64(graph, "duplicate_inserts"),
    }
}

fn singleflight_counter(run: &MeasuredRunArtifact, key: &str) -> u64 {
    run.snapshots
        .iter()
        .rev()
        .find_map(|sample| sample.cugraph_bfs_singleflight_snapshot.get(key).and_then(Value::as_u64))
        .unwrap_or_default()
}

fn value_u64(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_node_multi_gpu_validate::artifact::{
        BackendCloseEvidence, RunDeadlineEvidence, TimeSeriesEvidence,
    };

    #[test]
    fn empty_measurements_do_not_claim_cache_residency() {
        // This test pins the conservative rule at the owning artifact layer:
        // a run with no ledger samples is never interpreted as zero cache use.
        assert_eq!(peak_cache_residency(&empty_measurement()), None);
    }

    fn empty_measurement() -> MeasuredRunArtifact {
        use crate::single_node_multi_gpu_validate::{
            aggregate::{MeasuredAggregate, MeasurementWindow, ReportAggregateEquality},
            artifact::{ArrivalPlanEvidence, BackendEvidence, OperabilityEvidence, TraceEvidence},
            metrics_audit::MetricsAudit,
            operability::{AttributionCompleteness, OtlpExportEvidence},
        };

        let aggregate = MeasuredAggregate::reduce(&[], &[], &Default::default(), 0, 1);
        MeasuredRunArtifact {
            qualifying: false,
            backend: BackendEvidence {
                backend_generation: 0,
                shared_session_count: 0,
                queue_max_queued_attempts: 0,
                queue_max_queue_wait_ms: 0,
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
                duration_paced_concurrency: false,
                rate_overrun: "delay",
                planned_arrivals: 0,
                max_planned_attempts: 0,
                planned_warmup_arrivals: None,
                planned_measurement_arrivals: None,
                truncated_by_attempt_cap: false,
                steps: Vec::new(),
                storm: None,
                dropped_overrun_arrivals: 0,
                begin_failures: 0,
                join_failures: 0,
            },
            cache_state: "disabled",
            composed_schedules: Vec::new(),
            cache_same_key_probe: None,
            qps_envelope: None,
            graph_claim_calibration: None,
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
                steps: Vec::new(),
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
            deadline: RunDeadlineEvidence {
                measurement_window_ns: 1,
                completion_drain_grace_ns: 0,
                configured_window_ns: 1,
                elapsed_until_reports_ns: 1,
                overrun_ns: 0,
                reports_sealed_within_deadline: true,
            },
            cancellation_barrier: None,
            deadline_barrier: None,
            domain_exhaustion: None,
            physical_allocator_pressure: None,
            quarantine_cycles: None,
            close_under_load: None,
            completion_sink: None,
            feature_off_cugraph: None,
            metrics_audit: MetricsAudit::from_snapshots(&[], &[], 0, 250, false),
            operability: OperabilityEvidence {
                attribution: AttributionCompleteness::from_observations(&[]),
                otlp_export: OtlpExportEvidence::not_requested(),
            },
            time_series: TimeSeriesEvidence::empty(),
            snapshots: Vec::new(),
            trace: TraceEvidence { captured_by_runner: false, uri: None, sha256: None },
        }
    }

    const SIX_GIB: u64 = 6 * 1024 * 1024 * 1024;

    fn cache_measurement(enabled: bool, cache_report: Value, ledger_evictable_bytes: u64) -> MeasuredRunArtifact {
        use crate::single_node_multi_gpu_validate::{
            aggregate::{DeviceConfiguration, DeviceOccupancyPeaks, DevicePlacement},
            artifact::BackendSnapshotSample,
        };

        let mut run = empty_measurement();
        run.qualifying = true;
        run.cache_state = if enabled { "enabled" } else { "disabled" };
        run.aggregate.placement = vec![DevicePlacement {
            configuration: DeviceConfiguration {
                ordinal: 0,
                model_name: None,
                uuid: None,
                physical_total_bytes: None,
                capacity_bytes: SIX_GIB.saturating_mul(2),
                cache_cap_bytes: if enabled { SIX_GIB } else { Default::default() },
                install_telemetry: true,
                max_active_attempts: 1,
                stream_pool_size: 1,
                worker_count: 1,
                lane_count: 1,
                ownership: "whole_device_exclusive".to_string(),
                device_health_generation: 0,
                starting_health_open: true,
            },
            attempts_placed: 0,
            attempts_succeeded: 0,
            distinct_grants: 0,
            grant_bytes_min: 0,
            grant_bytes_max: 0,
            occupancy: DeviceOccupancyPeaks::default(),
            made_progress: false,
        }];
        run.snapshots = vec![BackendSnapshotSample {
            offset_ns: 0,
            backend_metrics_snapshot: Value::Null,
            admission: Value::Null,
            device_ledger_reports: serde_json::json!([{
                "cache_evictable_resident_bytes": ledger_evictable_bytes,
                "cache_pinned_in_use_bytes": 0,
            }]),
            device_cache_reports: cache_report,
            device_memory_failure_reports: Value::Null,
            gpu_execution_pool_reports: Value::Null,
            device_telemetry_snapshots: Value::Null,
            cugraph_bfs_singleflight_snapshot: Value::Null,
        }];
        run
    }

    fn cache_report(cache_cap_bytes: u64, allocation_limit_bytes: u64, physical_current_bytes: u64) -> Value {
        serde_json::json!([{
            "device_ordinal": 0,
            "cache_cap_bytes": cache_cap_bytes,
            "allocation_limit_bytes": allocation_limit_bytes,
            "ledger_evictable_resident_bytes": 1024,
            "ledger_pinned_in_use_bytes": 0,
            "physical_current_bytes": physical_current_bytes,
            "source": {},
            "graph": {},
        }])
    }

    fn clause_status(evidence: &CacheCalibrationEvidence, name: &str) -> ClauseStatus {
        evidence
            .clauses
            .iter()
            .find(|clause| clause.clause == name)
            .unwrap_or_else(|| panic!("missing cache calibration clause {name}"))
            .status
    }

    #[test]
    fn uncharged_physical_cache_bytes_fail_calibration() {
        let disabled = cache_measurement(false, Value::Null, 0);
        let enabled = cache_measurement(true, cache_report(SIX_GIB, SIX_GIB, 1025), 1024);

        let evidence = CacheCalibrationEvidence::new(disabled, enabled, None);
        assert_eq!(clause_status(&evidence, "per_sample_cache_cap_and_physical_ownership"), ClauseStatus::Failed);
        assert_eq!(evidence.sampling.physical_over_ledger_samples, 1);
    }

    #[test]
    fn cache_domain_cap_mismatch_fails_calibration() {
        let disabled = cache_measurement(false, Value::Null, 0);
        let enabled = cache_measurement(true, cache_report(SIX_GIB - 1, SIX_GIB, 1024), 1024);

        let evidence = CacheCalibrationEvidence::new(disabled, enabled, None);
        assert_eq!(clause_status(&evidence, "cache_domain_matches_ledger_profile"), ClauseStatus::Failed);
        assert_eq!(evidence.sampling.cap_mismatch_samples, 1);
    }

    #[test]
    fn missing_cache_reports_are_not_evidence() {
        let disabled = cache_measurement(false, Value::Null, 0);
        let enabled = cache_measurement(true, Value::Null, 1024);

        let evidence = CacheCalibrationEvidence::new(disabled, enabled, None);
        assert_eq!(clause_status(&evidence, "cache_domain_matches_ledger_profile"), ClauseStatus::NotEvidenced);
        assert_eq!(clause_status(&evidence, "per_sample_cache_cap_and_physical_ownership"), ClauseStatus::NotEvidenced);
    }

    #[test]
    fn cache_only_admission_failure_requires_sealed_residency() {
        use crate::single_node_multi_gpu_validate::{
            aggregate::{AttemptFailureClass, AttemptRow, AttemptTerminalEvidence},
            operability::AttemptAttribution,
            schedule::SchedulePhase,
        };

        let disabled = empty_measurement();
        let mut enabled = empty_measurement();
        enabled.aggregate.attempts.push(AttemptRow {
            sequence: 0,
            phase: SchedulePhase::Measurement,
            step_index: 0,
            query_label: "q01".to_string(),
            attempt_id: "attempt-cache-only".to_string(),
            disposition: "Native".to_string(),
            outcome: "failed".to_string(),
            failure: Some(AttemptFailureClass {
                origin: "Native".to_string(),
                code: "QueryMemoryLimitExceeded".to_string(),
                status: "Temporary".to_string(),
                operation: "Execution".to_string(),
            }),
            terminal_evidence: AttemptTerminalEvidence {
                primary_failure: None,
                cleanup_disposition: serde_json::json!("completed"),
                cleanup_failure_count: 0,
                cleanup_failure_count_complete: true,
                first_cleanup_failure: None,
                capacity_disposition: serde_json::json!("released"),
                health_transition: None,
                native_fault_evidence: None,
                resource_evidence: None,
                retry_advice: Value::Null,
            },
            attribution: AttemptAttribution {
                applicable: true,
                complete: false,
                device_ordinal_required: true,
                allocator_event_required: true,
                missing_fields: vec!["allocator_event"],
            },
            graph_claim_calibration: None,
            device_ordinal: Some(0),
            grant_bytes: None,
            admission_sequence: Some(1),
            admission_wait_ns: None,
            admission_requested_ns: None,
            admitted_ns: None,
            admission_wait: None,
            grant_id: None,
            domain_id: None,
            total_ns: None,
            queue_wait_ns: None,
            physical_execution_ns: None,
        });
        let stress = CacheStressProfile {
            require_resident_quarantine: false,
            required_per_device_cap_bytes: SIX_GIB,
            minimum_source_promoted_bytes: 0,
            require_source_eviction: false,
            require_source_coalescing: false,
            require_bfs_singleflight: false,
            require_explained_admission_delta: true,
            same_key_probe: None,
        };

        let clauses = stress_clauses(&disabled, &enabled, &CacheSamplingEvidence::default(), &stress, None);
        let admission = clauses
            .iter()
            .find(|clause| clause.clause == "s7_cache_on_admission_delta_accounted")
            .expect("admission-delta clause");
        assert_eq!(admission.status, ClauseStatus::Failed);
        assert!(admission.detail.contains("cache-on-only non-successes=1 unexplained=1"));
    }

    #[test]
    fn bfs_follower_evidence_requires_a_real_coalesced_traversal() {
        let stress = CacheStressProfile {
            require_resident_quarantine: false,
            required_per_device_cap_bytes: SIX_GIB,
            minimum_source_promoted_bytes: 0,
            require_source_eviction: false,
            require_source_coalescing: false,
            require_bfs_singleflight: true,
            require_explained_admission_delta: false,
            same_key_probe: None,
        };
        let sampling = CacheSamplingEvidence {
            bfs_singleflight_eligible_requests: 4,
            bfs_singleflight_physical_traversals: 1,
            bfs_singleflight_follower_copies: 3,
            bfs_singleflight_groups_formed: 1,
            bfs_singleflight_leader_failures: 0,
            bfs_singleflight_peak_width: 4,
            ..CacheSamplingEvidence::default()
        };

        let clauses = stress_clauses(&empty_measurement(), &empty_measurement(), &sampling, &stress, None);
        let singleflight = clauses
            .iter()
            .find(|clause| clause.clause == "s7_bfs_same_key_singleflight")
            .expect("BFS singleflight clause");
        assert_eq!(singleflight.status, ClauseStatus::Evidenced);
    }

    #[test]
    fn dedicated_same_key_probe_requires_one_cold_ordinal_local_source_and_bfs_generation() {
        let probe = same_key_probe_evidence();

        assert_eq!(same_key_source_clause(&probe).status, ClauseStatus::Evidenced);
        assert_eq!(same_key_bfs_clause(&probe).status, ClauseStatus::Evidenced);
    }

    #[test]
    fn dedicated_same_key_probe_rejects_cross_device_cache_reports() {
        type Mutate = fn(&mut CacheSameKeyProbeEvidence);
        type Clause = fn(&CacheSameKeyProbeEvidence) -> CacheCalibrationClause;
        let cases: &[(&str, Mutate, &[Clause])] = &[
            (
                "dedicated_same_key_probe_rejects_cross_device_cache_reports",
                |probe| {
                    let foreign = probe.source.cache_reports_after.as_array().unwrap()[0].clone();
                    probe.source.cache_reports_after.as_array_mut().unwrap().push(foreign);
                },
                &[same_key_source_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_stale_singleflight_counters",
                |probe| {
                    probe.bfs.bfs_singleflight_before["eligible_requests"] = serde_json::json!(1);
                    probe.bfs.bfs_singleflight_after["eligible_requests"] = serde_json::json!(3);
                },
                &[same_key_bfs_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_failed_bfs_leader",
                |probe| {
                    probe.bfs.bfs_singleflight_after["leader_failures"] = serde_json::json!(1);
                },
                &[same_key_bfs_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_duplicate_graph_charge",
                |probe| {
                    probe.bfs.cache_reports_after[0]["graph"]["duplicate_inserts"] = serde_json::json!(1);
                },
                &[same_key_bfs_clause],
            ),
            (
                "dedicated_same_key_probe_fails_when_required_events_are_missing",
                |probe| {
                    probe.source.source_cache_events_after = Value::Null;
                    probe.bfs.bfs_singleflight_after["events"] = Value::Null;
                },
                &[same_key_source_clause, same_key_bfs_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_mismatched_event_correlation",
                |probe| {
                    probe.source.source_cache_events_after[1]["correlation_fingerprint"] =
                        serde_json::json!("foreign-correlation");
                },
                &[same_key_source_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_multiple_source_keys",
                |probe| {
                    probe.source.source_cache_events_after[2]["cache_key_fingerprint"] =
                        serde_json::json!("source-key-2");
                },
                &[same_key_source_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_cross_device_bfs_event",
                |probe| {
                    probe.bfs.bfs_singleflight_after["events"][1]["device_ordinal"] = serde_json::json!(1);
                },
                &[same_key_bfs_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_duplicate_source_charge",
                |probe| {
                    probe.source.cache_reports_after[0]["ledger_evictable_resident_bytes"] = serde_json::json!(2048);
                },
                &[same_key_source_clause],
            ),
            (
                "dedicated_same_key_probe_rejects_non_measurement_attempts",
                |probe| {
                    probe.bfs.measurement_attempts = 0;
                    probe.bfs.measurement_successes = 0;
                },
                &[same_key_bfs_clause],
            ),
        ];
        for (name, mutate, clauses) in cases {
            let mut probe = same_key_probe_evidence();
            mutate(&mut probe);
            for clause in *clauses {
                assert_eq!(clause(&probe).status, ClauseStatus::Failed, "{name}");
            }
        }
    }

    fn same_key_probe_evidence() -> CacheSameKeyProbeEvidence {
        CacheSameKeyProbeEvidence {
            target_device_ordinal: 0,
            cohort_width: 2,
            source: CacheSameKeyCohortEvidence {
                cohort_id: "source-cohort".to_string(),
                expected_correlation_fingerprints: vec!["source-0".to_string(), "source-1".to_string()],
                fixture_kind: FixtureKind::Tpch,
                query_label: "q06".to_string(),
                submitted_attempts: 2,
                begin_failures: 0,
                join_failures: 0,
                measurement_attempts: 2,
                measurement_successes: 2,
                successful_attempts: 2,
                target_ordinal_attempts: 2,
                other_ordinal_attempts: 0,
                cache_reports_before: same_key_cache_report(0, 0, 0, 0, 0, 0, 0),
                cache_reports_after: same_key_cache_report(1024, 1, 0, 1, 1, 0, 0),
                source_cache_events_before: serde_json::json!([]),
                source_cache_events_after: same_key_source_events(),
                bfs_singleflight_before: same_key_bfs_snapshot(0, 0, 0, 0, 0, 0, serde_json::json!([])),
                bfs_singleflight_after: same_key_bfs_snapshot(0, 0, 0, 0, 0, 0, serde_json::json!([])),
            },
            bfs: CacheSameKeyCohortEvidence {
                cohort_id: "bfs-cohort".to_string(),
                expected_correlation_fingerprints: vec!["bfs-0".to_string(), "bfs-1".to_string()],
                fixture_kind: FixtureKind::Dblp,
                query_label: "bfs".to_string(),
                submitted_attempts: 2,
                begin_failures: 0,
                join_failures: 0,
                measurement_attempts: 2,
                measurement_successes: 2,
                successful_attempts: 2,
                target_ordinal_attempts: 2,
                other_ordinal_attempts: 0,
                // The source cohort's one retained entry remains part of the
                // same ledger-accounted allocation domain before BFS starts.
                cache_reports_before: same_key_cache_report(1024, 1, 0, 1, 1, 0, 0),
                cache_reports_after: same_key_cache_report(2048, 1, 1, 1, 1, 1, 0),
                source_cache_events_before: serde_json::json!([]),
                source_cache_events_after: serde_json::json!([]),
                bfs_singleflight_before: same_key_bfs_snapshot(0, 0, 0, 0, 0, 0, serde_json::json!([])),
                bfs_singleflight_after: same_key_bfs_snapshot(2, 1, 1, 1, 0, 0, same_key_bfs_events()),
            },
            close_quiescent: true,
        }
    }

    fn same_key_cache_report(
        ledger_evictable_resident_bytes: u64,
        source_entries: u64,
        graph_entries: u64,
        source_insertions: u64,
        source_coalesced: u64,
        graph_insertions: u64,
        graph_duplicates: u64,
    ) -> Value {
        serde_json::json!([{
            "device_ordinal": 0,
            "allocation_domain_id": 7,
            "cache_cap_bytes": SIX_GIB,
            "allocation_limit_bytes": SIX_GIB,
            "ledger_evictable_resident_bytes": ledger_evictable_resident_bytes,
            "ledger_pinned_in_use_bytes": 0,
            "physical_current_bytes": ledger_evictable_resident_bytes,
            "source": {
                "entry_count": source_entries,
                "promotions_in_progress": 0,
                "promotions_inserted": source_insertions,
                "promotions_coalesced": source_coalesced
            },
            "graph": {
                "available_entries": graph_entries,
                "insertions": graph_insertions,
                "duplicate_inserts": graph_duplicates
            }
        }])
    }

    fn same_key_bfs_snapshot(
        eligible_requests: u64,
        physical_traversals: u64,
        follower_copies: u64,
        groups_formed: u64,
        overflow_groups: u64,
        leader_failures: u64,
        events: Value,
    ) -> Value {
        serde_json::json!({
            "eligible_requests": eligible_requests,
            "physical_traversals": physical_traversals,
            "follower_copies": follower_copies,
            "groups_formed": groups_formed,
            "overflow_groups": overflow_groups,
            "leader_failures": leader_failures,
            "cancelled_waiters": 0,
            "events": events,
        })
    }

    fn same_key_source_events() -> Value {
        serde_json::json!([
            {
                "sequence": 0,
                "device_ordinal": 0,
                "allocation_domain_id": 7,
                "correlation_fingerprint": "source-0",
                "cache_key_fingerprint": "source-key-1",
                "action": "load_leader",
                "retained_device_bytes": 0,
            },
            {
                "sequence": 1,
                "device_ordinal": 0,
                "allocation_domain_id": 7,
                "correlation_fingerprint": "source-1",
                "cache_key_fingerprint": "source-key-1",
                "action": "load_follower",
                "retained_device_bytes": 0,
            },
            {
                "sequence": 2,
                "device_ordinal": 0,
                "allocation_domain_id": 7,
                "correlation_fingerprint": "source-0",
                "cache_key_fingerprint": "source-key-1",
                "action": "promotion_inserted",
                "retained_device_bytes": 1024,
            },
            {
                "sequence": 3,
                "device_ordinal": 0,
                "allocation_domain_id": 7,
                "correlation_fingerprint": "source-1",
                "cache_key_fingerprint": "source-key-1",
                "action": "load_hit",
                "retained_device_bytes": 0,
            }
        ])
    }

    fn same_key_bfs_events() -> Value {
        serde_json::json!([
            {
                "sequence": 0,
                "device_ordinal": 0,
                "singleflight_key_fingerprint": "bfs-key-1",
                "correlation_fingerprint": "bfs-0",
                "generation": 1,
                "action": "leader_join",
            },
            {
                "sequence": 1,
                "device_ordinal": 0,
                "singleflight_key_fingerprint": "bfs-key-1",
                "correlation_fingerprint": "bfs-1",
                "generation": 1,
                "action": "follower_join",
            },
            {
                "sequence": 2,
                "device_ordinal": 0,
                "singleflight_key_fingerprint": "bfs-key-1",
                "correlation_fingerprint": "bfs-0",
                "generation": 1,
                "action": "leader_traversal",
            },
            {
                "sequence": 3,
                "device_ordinal": 0,
                "singleflight_key_fingerprint": "bfs-key-1",
                "correlation_fingerprint": "bfs-0",
                "generation": 1,
                "action": "leader_published",
            },
            {
                "sequence": 4,
                "device_ordinal": 0,
                "singleflight_key_fingerprint": "bfs-key-1",
                "correlation_fingerprint": "bfs-1",
                "generation": 1,
                "action": "follower_copy",
            }
        ])
    }

    fn test_close_evidence() -> BackendCloseEvidence {
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
