//! Compact utilization/drift time series.
//!
//! The GPU-side sampler (`measure.rs`) projects each public snapshot pull
//! into the compact rows defined here; everything below the row structs is a
//! pure reduction over them, so utilization balance and drift evidence are
//! unit-testable without CUDA. The rows deliberately mirror fields the
//! artifact already retains in the raw snapshot samples — this is a compact
//! queryable projection of the same observation, never a second event store.
//!
//! Only the measured run samples, so the CPU-only feature reaches these
//! values from tests alone.
#![cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::profile::VramCalibrationProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct QueryDomainPoolClaim {
    pub(super) device_ordinal: usize,
    pub(super) grant_id: u64,
    pub(super) grant_bytes: u64,
    pub(super) reserved_high_bytes: u64,
    pub(super) used_high_bytes: u64,
}

impl QueryDomainPoolClaim {
    fn overhead_bytes(self) -> Option<u64> {
        self.reserved_high_bytes.checked_sub(self.used_high_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MatchedQueryDomainPoolUsage {
    overhead_bytes: u64,
    reserved_high_bytes: u64,
}

/// One sampler tick's compact per-device projection.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(super) struct DeviceSeriesRow {
    pub(super) offset_ns: u64,
    pub(super) device_ordinal: usize,
    /// The ledger's admission health at this tick. A closed ordinal must
    /// receive no further placements; device-health and close checks read this series.
    pub(super) health_open: bool,
    pub(super) gpu_utilization_pct: Option<u32>,
    pub(super) memory_utilization_pct: Option<u32>,
    pub(super) cuda_free_bytes: Option<u64>,
    /// NVML's physical framebuffer usage. CUDA free bytes remain the primary
    /// allocation-capacity signal; this independent device-wide measurement
    /// is retained to expose scope differences instead of pretending the two
    /// APIs measure the same allocator state.
    pub(super) nvml_used_bytes: Option<u64>,
    pub(super) active_attempts: usize,
    pub(super) managed_capacity_bytes: u64,
    pub(super) live_grants_bytes: u64,
    pub(super) live_grant_ids: Vec<u64>,
    pub(super) live_reserved_bytes: u64,
    pub(super) cache_evictable_resident_bytes: u64,
    pub(super) cache_pinned_in_use_bytes: u64,
    /// Direct cuDF allocation-domain observation. It is `None` only when the
    /// per-device cache is disabled; it is never inferred from NVML free bytes.
    pub(super) cache_physical_current_bytes: Option<u64>,
    pub(super) cache_physical_peak_bytes: Option<u64>,
    /// Cumulative allocator-failure counter value at this tick.
    pub(super) allocation_failure_count: u64,
    /// Ledger observation sequence, for cadence/staleness auditing.
    pub(super) ledger_sequence: u64,
    /// Sequence returned by a second ledger read taken immediately after the
    /// CUDA/NVML telemetry pull.  A calibration reducer may use this row's
    /// claim only when it matches `ledger_sequence` and the claim fields did
    /// not change across that bracket.  The compact row retains both sides so
    /// a consumer need not trust a runner-side assertion of simultaneity.
    pub(super) claim_snapshot_end_sequence: Option<u64>,
    /// Whether the admission-plus-cache claim was stable across the telemetry
    /// bracket.  Unstable rows remain useful for utilization/drift reporting,
    /// but are deliberately excluded from graph-model calibration.
    pub(super) claim_snapshot_stable: bool,
    pub(super) active_query_jobs: usize,
    pub(super) queued_jobs: usize,
}

/// One sampler tick's process-wide projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub(super) struct ProcessSeriesRow {
    pub(super) offset_ns: u64,
    pub(super) active_queries: usize,
    pub(super) queued_queries: usize,
    /// Backend observation generations, recorded so the audit can prove metric
    /// cardinality/identity stayed fixed across the window.
    pub(super) backend_generation: u64,
    pub(super) metric_generation: u64,
    /// Wall-clock duration of the complete public observation pull that
    /// produced this row. It includes metrics, ledger, admission, pool,
    /// failure, and telemetry snapshots; feature-off and telemetry checks report monitoring cost
    /// without timing unrelated runner work.
    pub(super) observation_pull_ns: u64,
    pub(super) completed_attempts: u64,
    pub(super) successful_native_attempts: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct SampleGap {
    pub(super) nanoseconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FleetUtilizationPolicy {
    pub(super) device_ordinals: Vec<usize>,
    pub(super) minimum_utilization_percent: u32,
    pub(super) minimum_duration_ns: u64,
    pub(super) maximum_sample_gap_ns: u64,
    pub(super) minimum_active_attempts: usize,
    pub(super) minimum_completed_attempts: u64,
    pub(super) minimum_successful_native_attempts: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct FleetUtilizationInterval {
    pub(super) start_offset_ns: u64,
    pub(super) end_offset_ns: u64,
    pub(super) duration_ns: u64,
    pub(super) minimum_utilization_milli: u64,
    pub(super) minimum_active_attempts: usize,
    pub(super) completed_attempts: u64,
    pub(super) successful_native_attempts: u64,
    pub(super) maximum_sample_gap: SampleGap,
}

pub(super) fn maximum_physical_sample_gap(rows: &[DeviceSeriesRow]) -> Option<SampleGap> {
    per_ordinal(rows)
        .filter_map(|(_, rows)| {
            let start = rows.iter().map(|row| row.offset_ns).min()?;
            let end = rows.iter().map(|row| row.offset_ns).max()?;
            let mut physical = rows
                .iter()
                .filter(|row| row.cuda_free_bytes.is_some() && row.nvml_used_bytes.is_some())
                .map(|row| row.offset_ns)
                .collect::<Vec<_>>();
            physical.sort_unstable();
            let first = *physical.first()?;
            let last = *physical.last()?;
            Some(
                physical
                    .windows(2)
                    .map(|pair| pair[1].saturating_sub(pair[0]))
                    .max()
                    .unwrap_or_default()
                    .max(first.saturating_sub(start))
                    .max(end.saturating_sub(last)),
            )
        })
        .max()
        .map(|nanoseconds| SampleGap { nanoseconds })
}

pub(super) fn fleet_utilization_interval(
    device_rows: &[DeviceSeriesRow],
    process_rows: &[ProcessSeriesRow],
    policy: &FleetUtilizationPolicy,
) -> Result<Option<FleetUtilizationInterval>, &'static str> {
    if policy.device_ordinals.is_empty()
        || policy.minimum_utilization_percent > 100
        || policy.device_ordinals.iter().copied().collect::<std::collections::BTreeSet<_>>().len()
            != policy.device_ordinals.len()
    {
        return Err("invalid fleet utilization policy");
    }
    let mut by_offset = std::collections::BTreeMap::<u64, Vec<&DeviceSeriesRow>>::new();
    for row in device_rows {
        by_offset.entry(row.offset_ns).or_default().push(row);
    }
    let mut current: Option<(u64, u64, u64, usize, u64, u64, u64)> = None;
    let mut best = None;
    for process in process_rows {
        let Some(rows) = by_offset.get(&process.offset_ns) else {
            current = None;
            continue;
        };
        let selected = policy.device_ordinals.iter().map(|ordinal| {
            let mut matches = rows.iter().filter(|row| row.device_ordinal == *ordinal);
            let row = matches.next().copied()?;
            (matches.next().is_none() && row.cuda_free_bytes.is_some() && row.nvml_used_bytes.is_some()).then_some(row)
        });
        let selected = selected.collect::<Option<Vec<_>>>();
        let Some(selected) = selected else {
            current = None;
            continue;
        };
        let grants = selected.iter().try_fold(0_u128, |sum, row| sum.checked_add(u128::from(row.live_grants_bytes)));
        let capacity =
            selected.iter().try_fold(0_u128, |sum, row| sum.checked_add(u128::from(row.managed_capacity_bytes)));
        let (Some(grants), Some(capacity)) = (grants, capacity) else {
            return Err("fleet utilization sum overflow");
        };
        let active = selected.iter().try_fold(0_usize, |sum, row| sum.checked_add(row.active_attempts));
        let Some(active) = active else {
            return Err("fleet active-attempt sum overflow");
        };
        let weighted_grants = grants.checked_mul(100).ok_or("fleet utilization product overflow")?;
        let weighted_capacity = capacity
            .checked_mul(u128::from(policy.minimum_utilization_percent))
            .ok_or("fleet utilization product overflow")?;
        let utilization_ok = capacity > 0 && weighted_grants >= weighted_capacity;
        if !utilization_ok || active < policy.minimum_active_attempts {
            current = None;
            continue;
        }
        let utilization_milli =
            u64::try_from(grants.checked_mul(1000).ok_or("fleet utilization product overflow")? / capacity)
                .map_err(|_| "fleet utilization ratio overflow")?;
        let next = match current {
            Some((start, previous, minimum, minimum_active, completed, successful, maximum_gap))
                if process.offset_ns.saturating_sub(previous) <= policy.maximum_sample_gap_ns =>
            {
                (
                    start,
                    process.offset_ns,
                    minimum.min(utilization_milli),
                    minimum_active.min(active),
                    completed,
                    successful,
                    maximum_gap.max(process.offset_ns.saturating_sub(previous)),
                )
            },
            _ => (
                process.offset_ns,
                process.offset_ns,
                utilization_milli,
                active,
                process.completed_attempts,
                process.successful_native_attempts,
                0,
            ),
        };
        current = Some(next);
        let (start, end, minimum, minimum_active, completed_start, successful_start, maximum_gap) = next;
        let duration_ns = end.saturating_sub(start);
        let completed_attempts =
            process.completed_attempts.checked_sub(completed_start).ok_or("completed-attempt counter regressed")?;
        let successful_native_attempts = process
            .successful_native_attempts
            .checked_sub(successful_start)
            .ok_or("successful-native-attempt counter regressed")?;
        if duration_ns >= policy.minimum_duration_ns
            && completed_attempts >= policy.minimum_completed_attempts
            && successful_native_attempts >= policy.minimum_successful_native_attempts
        {
            let interval = FleetUtilizationInterval {
                start_offset_ns: start,
                end_offset_ns: end,
                duration_ns,
                minimum_utilization_milli: minimum,
                minimum_active_attempts: minimum_active,
                completed_attempts,
                successful_native_attempts,
                maximum_sample_gap: SampleGap { nanoseconds: maximum_gap },
            };
            if best.as_ref().is_none_or(|best: &FleetUtilizationInterval| interval.duration_ns > best.duration_ns) {
                best = Some(interval);
            }
        }
    }
    Ok(best)
}

/// Bounded summary of the public monitoring pull cost.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct ObservationPullSummary {
    pub(super) samples: u64,
    pub(super) min_ns: u64,
    pub(super) p50_ns: u64,
    pub(super) p95_ns: u64,
    pub(super) max_ns: u64,
    pub(super) mean_ns: u64,
}

/// Per-ordinal utilization statistics over the sampled window. Means are
/// computed only over ticks that carried telemetry; a `None` utilization
/// field means no tick did.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct OrdinalUtilizationSummary {
    pub(super) device_ordinal: usize,
    pub(super) samples: u64,
    pub(super) samples_with_telemetry: u64,
    pub(super) gpu_utilization_min_pct: Option<u32>,
    pub(super) gpu_utilization_max_pct: Option<u32>,
    pub(super) gpu_utilization_mean_pct: Option<f64>,
    pub(super) memory_utilization_max_pct: Option<u32>,
    pub(super) cuda_free_bytes_min: Option<u64>,
    pub(super) peak_active_attempts: usize,
    pub(super) peak_live_grants_bytes: u64,
}

/// Baseline-vs-end comparison per ordinal. The sampler starts before load
/// and finishes after the drain, so first/last rows approximate quiescence;
/// the gap between the final submission and terminal sample makes that
/// approximation a contract.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct OrdinalDriftSummary {
    pub(super) device_ordinal: usize,
    pub(super) first_offset_ns: u64,
    pub(super) last_offset_ns: u64,
    pub(super) cuda_free_bytes_first: Option<u64>,
    pub(super) cuda_free_bytes_last: Option<u64>,
    /// Signed delta (last - first); negative means free bytes were lost
    /// across the run.
    pub(super) cuda_free_bytes_delta: Option<i64>,
    pub(super) live_grants_bytes_first: u64,
    pub(super) live_grants_bytes_last: u64,
    /// Evictable + pinned cache bytes at the first/last tick and the peak.
    pub(super) cache_resident_bytes_first: u64,
    pub(super) cache_resident_bytes_last: u64,
    pub(super) cache_resident_bytes_peak: u64,
    pub(super) allocation_failure_count_first: u64,
    pub(super) allocation_failure_count_last: u64,
    /// Fragmentation canary: counter increments in the first versus the
    /// second half of the run. An upward trend absent external pressure is a
    /// drift finding, measured rather than judged here.
    pub(super) allocation_failures_first_half: u64,
    pub(super) allocation_failures_second_half: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct DriftSummary {
    pub(super) ordinals: Vec<OrdinalDriftSummary>,
    /// Backend observation generations never regressed across the window.
    pub(super) generations_monotonic: bool,
    /// Per-ordinal ledger sequences never regressed across the window.
    pub(super) ledger_sequences_monotonic: bool,
}

/// The per-profile verdict for a load-induced claim-versus-VRAM calibration.
/// It is intentionally independent of an Nsight trace: CUDA/NVML snapshots
/// make every qualifying run auditable, while a targeted trace cross-checks
/// allocations, lifetimes, and hidden copies at a smaller scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum VramCalibrationStatus {
    Evidenced,
    Failed,
}

/// Per-ordinal comparison at the same sample where CUDA free memory was at
/// its minimum. The cache claim uses the ledger-resident delta; the direct
/// cache-domain physical delta is deliberately reported beside it rather
/// than being folded into a second claim.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct OrdinalVramCalibration {
    pub(super) device_ordinal: usize,
    pub(super) baseline_offset_ns: Option<u64>,
    /// Whether the pre-load baseline's claim stayed stable across its own
    /// telemetry bracket. An unstable baseline cannot anchor a VRAM delta.
    pub(super) baseline_claim_snapshot_stable: Option<bool>,
    pub(super) cuda_peak_offset_ns: Option<u64>,
    /// Whether the ledger claim remained unchanged across the telemetry pull
    /// that observed the physical CUDA peak. `Some(false)` is a fail-closed
    /// sample, not permission to substitute a lower stable peak.
    pub(super) cuda_peak_claim_snapshot_stable: Option<bool>,
    /// Count of post-baseline rows whose ledger claim was stable across the
    /// CUDA/NVML telemetry bracket.
    pub(super) post_baseline_stable_claim_samples: u64,
    /// The calibration must see a nonzero load and nonzero contemporaneous
    /// claim after its baseline.  Without this witness a quiescent baseline
    /// could otherwise qualify as a zero-versus-zero memory peak.
    pub(super) post_baseline_load_claim_witness: bool,
    pub(super) cuda_free_bytes_baseline: Option<u64>,
    pub(super) cuda_free_bytes_min: Option<u64>,
    pub(super) cuda_used_delta_bytes: Option<u64>,
    pub(super) nvml_used_bytes_baseline: Option<u64>,
    pub(super) nvml_used_bytes_max: Option<u64>,
    pub(super) nvml_used_delta_bytes: Option<u64>,
    /// Absolute difference between the CUDA-capacity and NVML-framebuffer
    /// deltas. `None` means NVML was unavailable, never a fabricated zero.
    pub(super) cuda_nvml_delta_abs_bytes: Option<u64>,
    pub(super) admission_claim_bytes_at_cuda_peak: Option<u64>,
    pub(super) cache_ledger_resident_delta_bytes_at_cuda_peak: Option<u64>,
    pub(super) cache_physical_delta_bytes_at_cuda_peak: Option<u64>,
    /// Exact private-pool page overhead matched to the live grant set at the
    /// physical peak. Logical grant bytes remain unchanged.
    pub(super) query_domain_pool_overhead_bytes_at_cuda_peak: Option<u64>,
    pub(super) query_domain_pool_reserved_high_bytes_at_cuda_peak: Option<u64>,
    pub(super) physical_observed_bytes_at_cuda_peak: Option<u64>,
    pub(super) accounted_claim_bytes_at_cuda_peak: Option<u64>,
    pub(super) underclaim_bytes: Option<u64>,
    pub(super) overclaim_bytes: Option<u64>,
    pub(super) observed_to_claim_ratio_milli: Option<u64>,
    pub(super) claim_to_observed_ratio_milli: Option<u64>,
    /// `None` identifies an unavailable CUDA baseline/peak, which is a
    /// fail-closed gap for an explicitly calibrated profile.
    pub(super) within_profile_band: Option<bool>,
}

/// Compact, deterministic reduction of the public sampler rows. The runner
/// records the full rows as primary evidence; this section makes its
/// claim/actual conclusion queryable without a second event store.
#[derive(Debug, Clone, Serialize)]
pub(super) struct VramCalibrationEvidence {
    pub(super) status: VramCalibrationStatus,
    pub(super) policy: VramCalibrationProfile,
    pub(super) configured_ordinals: Vec<usize>,
    pub(super) sampled_all_configured_ordinals: bool,
    pub(super) ordinals: Vec<OrdinalVramCalibration>,
}

impl VramCalibrationEvidence {
    pub(super) fn passes(&self) -> bool {
        self.status == VramCalibrationStatus::Evidenced
    }
}

/// Folds the per-device series into per-ordinal utilization statistics.
pub(super) fn utilization_summary(rows: &[DeviceSeriesRow]) -> Vec<OrdinalUtilizationSummary> {
    per_ordinal(rows)
        .map(|(ordinal, rows)| {
            let utils: Vec<u32> = rows.iter().filter_map(|row| row.gpu_utilization_pct).collect();
            OrdinalUtilizationSummary {
                device_ordinal: ordinal,
                samples: rows.len() as u64,
                samples_with_telemetry: rows.iter().filter(|row| row.gpu_utilization_pct.is_some()).count() as u64,
                gpu_utilization_min_pct: utils.iter().copied().min(),
                gpu_utilization_max_pct: utils.iter().copied().max(),
                gpu_utilization_mean_pct: (!utils.is_empty())
                    .then(|| utils.iter().map(|value| f64::from(*value)).sum::<f64>() / utils.len() as f64),
                memory_utilization_max_pct: rows.iter().filter_map(|row| row.memory_utilization_pct).max(),
                cuda_free_bytes_min: rows.iter().filter_map(|row| row.cuda_free_bytes).min(),
                peak_active_attempts: rows.iter().map(|row| row.active_attempts).max().unwrap_or_default(),
                peak_live_grants_bytes: rows.iter().map(|row| row.live_grants_bytes).max().unwrap_or_default(),
            }
        })
        .collect()
}

/// Reduces individual full-pull timings with nearest-rank percentiles, the
/// same rule the runner uses for attempt latency. `samples == 0` distinguishes
/// an empty series from an observed zero-cost pull.
pub(super) fn observation_pull_summary(rows: &[ProcessSeriesRow]) -> ObservationPullSummary {
    let mut values = rows.iter().map(|row| row.observation_pull_ns).collect::<Vec<_>>();
    values.sort_unstable();
    let sum = values.iter().map(|value| u128::from(*value)).sum::<u128>();
    ObservationPullSummary {
        samples: values.len() as u64,
        min_ns: values.first().copied().unwrap_or_default(),
        p50_ns: percentile(&values, 50),
        p95_ns: percentile(&values, 95),
        max_ns: values.last().copied().unwrap_or_default(),
        mean_ns: if values.is_empty() { 0 } else { (sum / values.len() as u128) as u64 },
    }
}

/// Folds the series into baseline-vs-end drift evidence. Halves are split at
/// the median offset of the ordinal's own samples.
pub(super) fn drift_summary(rows: &[DeviceSeriesRow], process_rows: &[ProcessSeriesRow]) -> DriftSummary {
    let ordinals = per_ordinal(rows)
        .map(|(ordinal, rows)| {
            let first = rows.first().expect("per_ordinal yields non-empty groups");
            let last = rows.last().expect("per_ordinal yields non-empty groups");
            let mid_offset = first.offset_ns + (last.offset_ns - first.offset_ns) / 2;
            let at_or_before_mid = rows.iter().rfind(|row| row.offset_ns <= mid_offset).unwrap_or(first);
            let cache = |row: &&DeviceSeriesRow| row.cache_evictable_resident_bytes + row.cache_pinned_in_use_bytes;
            OrdinalDriftSummary {
                device_ordinal: ordinal,
                first_offset_ns: first.offset_ns,
                last_offset_ns: last.offset_ns,
                cuda_free_bytes_first: first.cuda_free_bytes,
                cuda_free_bytes_last: last.cuda_free_bytes,
                cuda_free_bytes_delta: match (first.cuda_free_bytes, last.cuda_free_bytes) {
                    (Some(first_bytes), Some(last_bytes)) => Some(last_bytes as i64 - first_bytes as i64),
                    _ => None,
                },
                live_grants_bytes_first: first.live_grants_bytes,
                live_grants_bytes_last: last.live_grants_bytes,
                cache_resident_bytes_first: cache(first),
                cache_resident_bytes_last: cache(last),
                cache_resident_bytes_peak: rows.iter().map(cache).max().unwrap_or_default(),
                allocation_failure_count_first: first.allocation_failure_count,
                allocation_failure_count_last: last.allocation_failure_count,
                allocation_failures_first_half: at_or_before_mid
                    .allocation_failure_count
                    .saturating_sub(first.allocation_failure_count),
                allocation_failures_second_half: last
                    .allocation_failure_count
                    .saturating_sub(at_or_before_mid.allocation_failure_count),
            }
        })
        .collect();
    let generations_monotonic = process_rows.windows(2).all(|pair| {
        pair[0].backend_generation <= pair[1].backend_generation
            && pair[0].metric_generation <= pair[1].metric_generation
    });
    let ledger_sequences_monotonic = per_ordinal(rows)
        .all(|(_, rows)| rows.windows(2).all(|pair| pair[0].ledger_sequence <= pair[1].ledger_sequence));
    DriftSummary { ordinals, generations_monotonic, ledger_sequences_monotonic }
}

/// Compares admission/cache accounting against the device-level CUDA delta.
///
/// A baseline row is synchronously sampled before work submission, so each
/// ordinal's first row is a real before-load observation rather than a race
/// against the drive loop. Each later candidate brackets its CUDA/NVML pull
/// with two ledger reads and is eligible only when the claim stayed unchanged;
/// this is not a global atomic snapshot, so NVML remains an independent
/// cross-check and targeted Nsight runs remain the allocation-lifetime
/// authority.
pub(super) fn vram_calibration_evidence(
    rows: &[DeviceSeriesRow],
    query_domain_pool_claims: &[QueryDomainPoolClaim],
    configured_ordinals: &[usize],
    policy: VramCalibrationProfile,
) -> VramCalibrationEvidence {
    let configured_ordinals = configured_ordinals.to_vec();
    let mut ordinals = Vec::with_capacity(configured_ordinals.len());
    let mut all_sampled = true;
    for ordinal in &configured_ordinals {
        let ordinal_rows = rows.iter().filter(|row| row.device_ordinal == *ordinal).collect::<Vec<_>>();
        let Some(baseline) = ordinal_rows.first().copied() else {
            all_sampled = false;
            ordinals.push(OrdinalVramCalibration::missing(*ordinal));
            continue;
        };
        let post_baseline_rows =
            ordinal_rows.iter().skip(1).copied().filter(|row| row.cuda_free_bytes.is_some()).collect::<Vec<_>>();
        let post_baseline_stable_claim_samples =
            post_baseline_rows.iter().filter(|row| stable_claim_snapshot(row)).count() as u64;
        // The baseline is never a CUDA peak candidate. Select the physical
        // peak before considering claim stability: otherwise an unstable
        // lower-free row could be hidden by a less-demanding stable row. If
        // multiple rows observed the same physical peak, prefer a stable row
        // and then the larger contemporaneous claim.
        let cuda_free_bytes_min = post_baseline_rows.iter().filter_map(|row| row.cuda_free_bytes).min();
        let cuda_peak = cuda_free_bytes_min.and_then(|minimum| {
            post_baseline_rows.iter().copied().filter(|row| row.cuda_free_bytes == Some(minimum)).min_by_key(|row| {
                (
                    !stable_claim_snapshot(row),
                    std::cmp::Reverse(row.live_grants_bytes.saturating_add(cache_ledger_bytes(row))),
                )
            })
        });
        let baseline_claim_snapshot_stable = stable_claim_snapshot(baseline);
        let cuda_peak_claim_snapshot_stable = cuda_peak.map(stable_claim_snapshot);
        let trusted_cuda_peak =
            cuda_peak.filter(|cuda_peak| baseline_claim_snapshot_stable && stable_claim_snapshot(cuda_peak));
        let cuda_used_delta_bytes =
            cuda_peak.and_then(|cuda_peak| match (baseline.cuda_free_bytes, cuda_peak.cuda_free_bytes) {
                (Some(baseline), Some(peak)) => Some(baseline.saturating_sub(peak)),
                _ => None,
            });
        let nvml_used_bytes_max = ordinal_rows.iter().filter_map(|row| row.nvml_used_bytes).max();
        let nvml_used_delta_bytes = match (baseline.nvml_used_bytes, nvml_used_bytes_max) {
            (Some(baseline), Some(peak)) => Some(peak.saturating_sub(baseline)),
            _ => None,
        };
        let cache_ledger_baseline = cache_ledger_bytes(baseline);
        let cache_ledger_delta_bytes =
            trusted_cuda_peak.map(|cuda_peak| cache_ledger_bytes(cuda_peak).saturating_sub(cache_ledger_baseline));
        let cache_physical_delta_bytes = trusted_cuda_peak.and_then(|cuda_peak| {
            match (baseline.cache_physical_current_bytes, cuda_peak.cache_physical_current_bytes) {
                (Some(baseline), Some(peak)) => Some(peak.saturating_sub(baseline)),
                _ => None,
            }
        });
        let query_domain_pool_usage = trusted_cuda_peak.and_then(|cuda_peak| {
            matched_query_domain_pool_usage(
                query_domain_pool_claims,
                *ordinal,
                &cuda_peak.live_grant_ids,
                cuda_peak.active_attempts,
                cuda_peak.live_grants_bytes,
            )
        });
        let query_domain_pool_overhead_bytes = query_domain_pool_usage.map(|usage| usage.overhead_bytes);
        let query_domain_pool_reserved_high_bytes = query_domain_pool_usage.map(|usage| usage.reserved_high_bytes);
        let physical_observed_bytes = cuda_used_delta_bytes
            .zip(query_domain_pool_reserved_high_bytes)
            .map(|(sampled_bytes, pool_bytes)| sampled_bytes.max(pool_bytes));
        let accounted_claim_bytes = trusted_cuda_peak
            .zip(cache_ledger_delta_bytes)
            .zip(query_domain_pool_overhead_bytes)
            .and_then(|((cuda_peak, cache_delta), pool_overhead)| {
                cuda_peak.live_grants_bytes.checked_add(cache_delta)?.checked_add(pool_overhead)
            });
        let post_baseline_load_claim_witness = cuda_peak
            .zip(accounted_claim_bytes)
            .is_some_and(|(_, claim)| physical_observed_bytes.is_some_and(|used| used > 0) && claim > 0);
        let (underclaim_bytes, overclaim_bytes, observed_to_claim_ratio_milli, claim_to_observed_ratio_milli, within) =
            match (physical_observed_bytes, accounted_claim_bytes) {
                (Some(observed), Some(claim)) => {
                    let observed_limit =
                        scaled_limit(claim, policy.max_observed_to_claim_ratio_milli, policy.max_underclaim_bytes);
                    let claim_limit =
                        scaled_limit(observed, policy.max_claim_to_observed_ratio_milli, policy.max_overclaim_bytes);
                    (
                        Some(observed.saturating_sub(claim)),
                        Some(claim.saturating_sub(observed)),
                        ratio_milli(observed, claim),
                        ratio_milli(claim, observed),
                        Some(observed <= observed_limit && claim <= claim_limit),
                    )
                },
                _ => (None, None, None, None, None),
            };
        ordinals.push(OrdinalVramCalibration {
            device_ordinal: *ordinal,
            baseline_offset_ns: Some(baseline.offset_ns),
            baseline_claim_snapshot_stable: Some(baseline_claim_snapshot_stable),
            cuda_peak_offset_ns: cuda_peak.map(|row| row.offset_ns),
            cuda_peak_claim_snapshot_stable,
            post_baseline_stable_claim_samples,
            post_baseline_load_claim_witness,
            cuda_free_bytes_baseline: baseline.cuda_free_bytes,
            cuda_free_bytes_min,
            cuda_used_delta_bytes,
            nvml_used_bytes_baseline: baseline.nvml_used_bytes,
            nvml_used_bytes_max,
            nvml_used_delta_bytes,
            cuda_nvml_delta_abs_bytes: match (cuda_used_delta_bytes, nvml_used_delta_bytes) {
                (Some(cuda), Some(nvml)) => Some(cuda.abs_diff(nvml)),
                _ => None,
            },
            admission_claim_bytes_at_cuda_peak: trusted_cuda_peak.map(|row| row.live_grants_bytes),
            cache_ledger_resident_delta_bytes_at_cuda_peak: cache_ledger_delta_bytes,
            cache_physical_delta_bytes_at_cuda_peak: cache_physical_delta_bytes,
            query_domain_pool_overhead_bytes_at_cuda_peak: query_domain_pool_overhead_bytes,
            query_domain_pool_reserved_high_bytes_at_cuda_peak: query_domain_pool_reserved_high_bytes,
            physical_observed_bytes_at_cuda_peak: physical_observed_bytes,
            accounted_claim_bytes_at_cuda_peak: accounted_claim_bytes,
            underclaim_bytes,
            overclaim_bytes,
            observed_to_claim_ratio_milli,
            claim_to_observed_ratio_milli,
            within_profile_band: within,
        });
    }
    let status = if all_sampled
        && ordinals.iter().all(|ordinal| {
            ordinal.baseline_claim_snapshot_stable == Some(true)
                && ordinal.cuda_peak_claim_snapshot_stable == Some(true)
                && ordinal.within_profile_band == Some(true)
                && ordinal.post_baseline_load_claim_witness
        }) {
        VramCalibrationStatus::Evidenced
    } else {
        VramCalibrationStatus::Failed
    };
    VramCalibrationEvidence {
        status,
        policy,
        configured_ordinals,
        sampled_all_configured_ordinals: all_sampled,
        ordinals,
    }
}

fn matched_query_domain_pool_usage(
    claims: &[QueryDomainPoolClaim],
    device_ordinal: usize,
    live_grant_ids: &[u64],
    active_attempts: usize,
    live_grants_bytes: u64,
) -> Option<MatchedQueryDomainPoolUsage> {
    if live_grant_ids.len() != active_attempts {
        return None;
    }
    if live_grant_ids.is_empty() {
        return (live_grants_bytes == 0)
            .then_some(MatchedQueryDomainPoolUsage { overhead_bytes: 0, reserved_high_bytes: 0 });
    }
    let mut claims_by_grant = BTreeMap::new();
    for claim in claims.iter().filter(|claim| claim.device_ordinal == device_ordinal) {
        if claims_by_grant.insert(claim.grant_id, claim).is_some() {
            return None;
        }
    }
    let mut seen = BTreeSet::new();
    let mut matched_grants = 0_u64;
    let mut usage = MatchedQueryDomainPoolUsage { overhead_bytes: 0, reserved_high_bytes: 0 };
    for grant_id in live_grant_ids {
        if !seen.insert(*grant_id) {
            return None;
        }
        let claim = claims_by_grant.get(grant_id)?;
        matched_grants = matched_grants.checked_add(claim.grant_bytes)?;
        usage.overhead_bytes = usage.overhead_bytes.checked_add(claim.overhead_bytes()?)?;
        usage.reserved_high_bytes = usage.reserved_high_bytes.checked_add(claim.reserved_high_bytes)?;
    }
    (matched_grants == live_grants_bytes).then_some(usage)
}

impl OrdinalVramCalibration {
    fn missing(device_ordinal: usize) -> Self {
        Self {
            device_ordinal,
            baseline_offset_ns: None,
            baseline_claim_snapshot_stable: None,
            cuda_peak_offset_ns: None,
            cuda_peak_claim_snapshot_stable: None,
            post_baseline_stable_claim_samples: 0,
            post_baseline_load_claim_witness: false,
            cuda_free_bytes_baseline: None,
            cuda_free_bytes_min: None,
            cuda_used_delta_bytes: None,
            nvml_used_bytes_baseline: None,
            nvml_used_bytes_max: None,
            nvml_used_delta_bytes: None,
            cuda_nvml_delta_abs_bytes: None,
            admission_claim_bytes_at_cuda_peak: None,
            cache_ledger_resident_delta_bytes_at_cuda_peak: None,
            cache_physical_delta_bytes_at_cuda_peak: None,
            query_domain_pool_overhead_bytes_at_cuda_peak: None,
            query_domain_pool_reserved_high_bytes_at_cuda_peak: None,
            physical_observed_bytes_at_cuda_peak: None,
            accounted_claim_bytes_at_cuda_peak: None,
            underclaim_bytes: None,
            overclaim_bytes: None,
            observed_to_claim_ratio_milli: None,
            claim_to_observed_ratio_milli: None,
            within_profile_band: None,
        }
    }
}

fn cache_ledger_bytes(row: &DeviceSeriesRow) -> u64 {
    row.cache_evictable_resident_bytes.saturating_add(row.cache_pinned_in_use_bytes)
}

/// Returns whether the sampler proved a stable logical claim across its
/// telemetry bracket. The boolean is retained for readers, but the reducer
/// re-checks both recorded sequences so a malformed row cannot quietly regain
/// calibration eligibility.
fn stable_claim_snapshot(row: &DeviceSeriesRow) -> bool {
    row.claim_snapshot_stable && row.claim_snapshot_end_sequence == Some(row.ledger_sequence)
}

fn scaled_limit(bytes: u64, ratio_milli: u64, fixed_bytes: u64) -> u64 {
    let scaled =
        u128::from(bytes).saturating_mul(u128::from(ratio_milli)).saturating_div(1_000).min(u128::from(u64::MAX));
    (scaled as u64).saturating_add(fixed_bytes)
}

fn ratio_milli(numerator: u64, denominator: u64) -> Option<u64> {
    (denominator != 0).then(|| {
        (u128::from(numerator).saturating_mul(1_000).saturating_div(u128::from(denominator)).min(u128::from(u64::MAX)))
            as u64
    })
}

/// Groups rows by ordinal, preserving first-seen ordinal order and the
/// input's per-ordinal time order.
fn per_ordinal(rows: &[DeviceSeriesRow]) -> impl Iterator<Item = (usize, Vec<&DeviceSeriesRow>)> {
    let mut order = Vec::new();
    let mut groups: std::collections::BTreeMap<usize, Vec<&DeviceSeriesRow>> = std::collections::BTreeMap::new();
    for row in rows {
        if !groups.contains_key(&row.device_ordinal) {
            order.push(row.device_ordinal);
        }
        groups.entry(row.device_ordinal).or_default().push(row);
    }
    order.into_iter().map(move |ordinal| {
        let rows = groups.remove(&ordinal).expect("ordinal recorded when first seen");
        (ordinal, rows)
    })
}

fn percentile(sorted: &[u64], percent: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (percent.saturating_mul(sorted.len())).div_ceil(100);
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(ordinal: usize, offset_ns: u64) -> DeviceSeriesRow {
        DeviceSeriesRow {
            offset_ns,
            device_ordinal: ordinal,
            health_open: true,
            gpu_utilization_pct: None,
            memory_utilization_pct: None,
            cuda_free_bytes: None,
            nvml_used_bytes: None,
            active_attempts: 0,
            managed_capacity_bytes: 1,
            live_grants_bytes: 0,
            live_grant_ids: Vec::new(),
            live_reserved_bytes: 0,
            cache_evictable_resident_bytes: 0,
            cache_pinned_in_use_bytes: 0,
            cache_physical_current_bytes: None,
            cache_physical_peak_bytes: None,
            allocation_failure_count: 0,
            ledger_sequence: 0,
            claim_snapshot_end_sequence: Some(0),
            claim_snapshot_stable: true,
            active_query_jobs: 0,
            queued_jobs: 0,
        }
    }

    #[test]
    fn utilization_summary_uses_only_telemetry_ticks_for_means() {
        let mut rows = vec![row(1, 0), row(1, 1), row(1, 2)];
        rows[0].gpu_utilization_pct = Some(30);
        rows[2].gpu_utilization_pct = Some(90);
        rows[1].active_attempts = 4;
        rows[2].live_grants_bytes = 4096;
        let summary = utilization_summary(&rows);
        assert_eq!(summary.len(), 1);
        let ordinal = summary[0];
        assert_eq!(ordinal.samples, 3);
        assert_eq!(ordinal.samples_with_telemetry, 2);
        assert_eq!(ordinal.gpu_utilization_min_pct, Some(30));
        assert_eq!(ordinal.gpu_utilization_max_pct, Some(90));
        assert_eq!(ordinal.gpu_utilization_mean_pct, Some(60.0));
        assert_eq!(ordinal.peak_active_attempts, 4);
        assert_eq!(ordinal.peak_live_grants_bytes, 4096);
    }

    #[test]
    fn utilization_summary_keeps_ordinals_separate() {
        let mut rows = vec![row(0, 0), row(2, 0), row(0, 1), row(2, 1)];
        rows[0].gpu_utilization_pct = Some(10);
        rows[1].gpu_utilization_pct = Some(95);
        let summary = utilization_summary(&rows);
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].device_ordinal, 0);
        assert_eq!(summary[0].gpu_utilization_max_pct, Some(10));
        assert_eq!(summary[1].device_ordinal, 2);
        assert_eq!(summary[1].gpu_utilization_max_pct, Some(95));
    }

    #[test]
    fn drift_summary_compares_baseline_to_end_and_splits_halves() {
        let mut rows = Vec::new();
        for index in 0..4 {
            let mut row = row(0, index * 100);
            row.cuda_free_bytes = Some(1_000 - index * 10);
            row.live_grants_bytes = if index == 3 { 0 } else { 512 };
            row.allocation_failure_count = [0, 2, 2, 9][index as usize];
            row.cache_evictable_resident_bytes = index * 64;
            rows.push(row);
        }
        let summary = drift_summary(&rows, &[]);
        assert_eq!(summary.ordinals.len(), 1);
        let drift = summary.ordinals[0];
        assert_eq!(drift.cuda_free_bytes_first, Some(1_000));
        assert_eq!(drift.cuda_free_bytes_last, Some(970));
        assert_eq!(drift.cuda_free_bytes_delta, Some(-30));
        assert_eq!(drift.live_grants_bytes_first, 512);
        assert_eq!(drift.live_grants_bytes_last, 0);
        assert_eq!(drift.cache_resident_bytes_first, 0);
        assert_eq!(drift.cache_resident_bytes_last, 192);
        assert_eq!(drift.cache_resident_bytes_peak, 192);
        // Mid offset is 150: the row at 100 closes the first half at 2.
        assert_eq!(drift.allocation_failures_first_half, 2);
        assert_eq!(drift.allocation_failures_second_half, 7);
    }

    #[test]
    fn drift_summary_tracks_generation_and_sequence_monotonicity() {
        let mut rows = vec![row(0, 0), row(0, 1)];
        rows[1].ledger_sequence = 1;
        let process = [
            ProcessSeriesRow {
                offset_ns: 0,
                active_queries: 0,
                queued_queries: 0,
                backend_generation: 1,
                metric_generation: 3,
                observation_pull_ns: 10,
                completed_attempts: 0,
                successful_native_attempts: 0,
            },
            ProcessSeriesRow {
                offset_ns: 1,
                active_queries: 0,
                queued_queries: 0,
                backend_generation: 1,
                metric_generation: 4,
                observation_pull_ns: 20,
                completed_attempts: 0,
                successful_native_attempts: 0,
            },
        ];
        let summary = drift_summary(&rows, &process);
        assert!(summary.generations_monotonic);
        assert!(summary.ledger_sequences_monotonic);

        let regressed = [
            process[1],
            ProcessSeriesRow {
                offset_ns: 2,
                active_queries: 0,
                queued_queries: 0,
                backend_generation: 0,
                metric_generation: 4,
                observation_pull_ns: 30,
                completed_attempts: 0,
                successful_native_attempts: 0,
            },
        ];
        let summary = drift_summary(&rows, &regressed);
        assert!(!summary.generations_monotonic);

        let mut sequence_regressed = vec![row(0, 0), row(0, 1)];
        sequence_regressed[0].ledger_sequence = 7;
        sequence_regressed[1].ledger_sequence = 3;
        assert!(!drift_summary(&sequence_regressed, &[]).ledger_sequences_monotonic);
    }

    #[test]
    fn empty_series_produce_empty_summaries() {
        assert!(utilization_summary(&[]).is_empty());
        let drift = drift_summary(&[], &[]);
        assert!(drift.ordinals.is_empty());
        assert!(drift.generations_monotonic);
        assert!(drift.ledger_sequences_monotonic);
        assert_eq!(observation_pull_summary(&[]).samples, 0);
    }

    fn exact_calibration_policy() -> VramCalibrationProfile {
        VramCalibrationProfile {
            max_underclaim_bytes: 64,
            max_observed_to_claim_ratio_milli: 1_000,
            max_overclaim_bytes: 64,
            max_claim_to_observed_ratio_milli: 1_000,
        }
    }

    fn pool_claim(grant_id: u64, grant_bytes: u64, overhead_bytes: u64) -> QueryDomainPoolClaim {
        QueryDomainPoolClaim {
            device_ordinal: 0,
            grant_id,
            grant_bytes,
            reserved_high_bytes: grant_bytes.saturating_add(overhead_bytes),
            used_high_bytes: grant_bytes,
        }
    }

    #[test]
    fn vram_calibration_compares_claims_at_the_cuda_peak_with_nvml_as_a_cross_check() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        baseline.nvml_used_bytes = Some(5_000);
        baseline.cache_physical_current_bytes = Some(100);
        let mut peak = row(0, 20);
        peak.cuda_free_bytes = Some(8_000);
        peak.nvml_used_bytes = Some(7_900);
        peak.active_attempts = 1;
        peak.live_grants_bytes = 1_900;
        peak.live_grant_ids = vec![1];
        peak.cache_evictable_resident_bytes = 50;
        peak.cache_physical_current_bytes = Some(150);

        let evidence =
            vram_calibration_evidence(&[baseline, peak], &[pool_claim(1, 1_900, 50)], &[0], exact_calibration_policy());
        assert_eq!(evidence.status, VramCalibrationStatus::Evidenced);
        assert!(evidence.passes());
        let ordinal = evidence.ordinals[0];
        assert_eq!(ordinal.cuda_used_delta_bytes, Some(2_000));
        assert_eq!(ordinal.nvml_used_delta_bytes, Some(2_900));
        assert_eq!(ordinal.cuda_nvml_delta_abs_bytes, Some(900));
        assert_eq!(ordinal.admission_claim_bytes_at_cuda_peak, Some(1_900));
        assert_eq!(ordinal.cache_ledger_resident_delta_bytes_at_cuda_peak, Some(50));
        assert_eq!(ordinal.cache_physical_delta_bytes_at_cuda_peak, Some(50));
        assert_eq!(ordinal.query_domain_pool_overhead_bytes_at_cuda_peak, Some(50));
        assert_eq!(ordinal.query_domain_pool_reserved_high_bytes_at_cuda_peak, Some(1_950));
        assert_eq!(ordinal.physical_observed_bytes_at_cuda_peak, Some(2_000));
        assert_eq!(ordinal.accounted_claim_bytes_at_cuda_peak, Some(2_000));
        assert_eq!(ordinal.cuda_peak_claim_snapshot_stable, Some(true));
        assert_eq!(ordinal.underclaim_bytes, Some(0));
        assert_eq!(ordinal.within_profile_band, Some(true));
    }

    #[test]
    fn vram_calibration_matches_pool_overhead_by_exact_live_grant_ids() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        let mut peak = row(0, 20);
        peak.cuda_free_bytes = Some(7_700);
        peak.active_attempts = 2;
        peak.live_grants_bytes = 2_000;
        peak.live_grant_ids = vec![1, 2];
        let claims = [pool_claim(1, 1_000, 100), pool_claim(2, 1_000, 200), pool_claim(3, 1_000, 999)];

        let evidence = vram_calibration_evidence(&[baseline, peak], &claims, &[0], exact_calibration_policy());

        let ordinal = evidence.ordinals[0];
        assert_eq!(ordinal.query_domain_pool_overhead_bytes_at_cuda_peak, Some(300));
        assert_eq!(ordinal.query_domain_pool_reserved_high_bytes_at_cuda_peak, Some(2_300));
        assert_eq!(ordinal.physical_observed_bytes_at_cuda_peak, Some(2_300));
        assert_eq!(ordinal.accounted_claim_bytes_at_cuda_peak, Some(2_300));
        assert_eq!(ordinal.within_profile_band, Some(true));
    }

    #[test]
    fn vram_calibration_rejects_malformed_live_grant_sets() {
        let valid = pool_claim(1, 1_000, 100);
        let cases = [
            ("count mismatch", vec![], 1, 1_000, vec![valid]),
            ("duplicate live id", vec![1, 1], 2, 2_000, vec![valid]),
            ("missing live id", vec![2], 1, 1_000, vec![valid]),
            ("duplicate claim id", vec![1], 1, 1_000, vec![valid, valid]),
            ("grant byte mismatch", vec![1], 1, 999, vec![valid]),
            (
                "pool usage underflow",
                vec![1],
                1,
                1_000,
                vec![QueryDomainPoolClaim { reserved_high_bytes: 999, used_high_bytes: 1_000, ..valid }],
            ),
            ("grant sum overflow", vec![1, 2], 2, u64::MAX, vec![pool_claim(1, u64::MAX, 0), pool_claim(2, 1, 0)]),
        ];
        for (name, live_grant_ids, active_attempts, live_grants_bytes, claims) in cases {
            let mut baseline = row(0, 10);
            baseline.cuda_free_bytes = Some(10_000);
            let mut peak = row(0, 20);
            peak.cuda_free_bytes = Some(9_000);
            peak.active_attempts = active_attempts;
            peak.live_grants_bytes = live_grants_bytes;
            peak.live_grant_ids = live_grant_ids;

            let evidence = vram_calibration_evidence(&[baseline, peak], &claims, &[0], exact_calibration_policy());

            assert_eq!(evidence.status, VramCalibrationStatus::Failed, "{name}");
            assert_eq!(evidence.ordinals[0].query_domain_pool_overhead_bytes_at_cuda_peak, None, "{name}");
        }
    }

    #[test]
    fn vram_calibration_uses_exact_pool_reservation_when_the_sampler_misses_its_peak() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        let mut peak = row(0, 20);
        peak.cuda_free_bytes = Some(5_300);
        peak.active_attempts = 1;
        peak.live_grants_bytes = 4_800;
        peak.live_grant_ids = vec![1];
        let claims = [QueryDomainPoolClaim {
            device_ordinal: 0,
            grant_id: 1,
            grant_bytes: 4_800,
            reserved_high_bytes: 4_860,
            used_high_bytes: 4_770,
        }];

        let evidence = vram_calibration_evidence(&[baseline, peak], &claims, &[0], exact_calibration_policy());

        let ordinal = evidence.ordinals[0];
        assert_eq!(ordinal.cuda_used_delta_bytes, Some(4_700));
        assert_eq!(ordinal.query_domain_pool_reserved_high_bytes_at_cuda_peak, Some(4_860));
        assert_eq!(ordinal.physical_observed_bytes_at_cuda_peak, Some(4_860));
        assert_eq!(ordinal.accounted_claim_bytes_at_cuda_peak, Some(4_890));
        assert_eq!(ordinal.overclaim_bytes, Some(30));
        assert_eq!(ordinal.within_profile_band, Some(true));
    }

    #[test]
    fn vram_calibration_fails_for_material_underclaim_overclaim_or_missing_ordinal() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        let mut underclaimed = row(0, 20);
        underclaimed.cuda_free_bytes = Some(8_000);
        underclaimed.active_attempts = 1;
        underclaimed.live_grants_bytes = 1_000;
        underclaimed.live_grant_ids = vec![1];
        let evidence = vram_calibration_evidence(
            &[baseline.clone(), underclaimed],
            &[pool_claim(1, 1_000, 0)],
            &[0],
            exact_calibration_policy(),
        );
        assert_eq!(evidence.status, VramCalibrationStatus::Failed);
        assert_eq!(evidence.ordinals[0].underclaim_bytes, Some(1_000));
        assert_eq!(evidence.ordinals[0].within_profile_band, Some(false));

        let mut overclaimed = row(0, 20);
        overclaimed.cuda_free_bytes = Some(9_900);
        overclaimed.active_attempts = 1;
        overclaimed.live_grants_bytes = 1_000;
        overclaimed.live_grant_ids = vec![1];
        let evidence = vram_calibration_evidence(
            &[baseline.clone(), overclaimed],
            &[QueryDomainPoolClaim {
                device_ordinal: 0,
                grant_id: 1,
                grant_bytes: 1_000,
                reserved_high_bytes: 100,
                used_high_bytes: 100,
            }],
            &[0],
            exact_calibration_policy(),
        );
        assert_eq!(evidence.status, VramCalibrationStatus::Failed);
        assert_eq!(evidence.ordinals[0].overclaim_bytes, Some(900));

        let evidence = vram_calibration_evidence(&[baseline], &[], &[0, 1], exact_calibration_policy());
        assert_eq!(evidence.status, VramCalibrationStatus::Failed);
        assert!(!evidence.sampled_all_configured_ordinals);
        assert_eq!(evidence.ordinals[1].within_profile_band, None);
    }

    #[test]
    fn vram_calibration_requires_a_post_baseline_nonzero_load_and_claim_witness() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        let mut quiescent_post_baseline = row(0, 20);
        quiescent_post_baseline.cuda_free_bytes = Some(10_000);

        let evidence =
            vram_calibration_evidence(&[baseline, quiescent_post_baseline], &[], &[0], exact_calibration_policy());

        assert_eq!(evidence.status, VramCalibrationStatus::Failed);
        let ordinal = evidence.ordinals[0];
        assert_eq!(ordinal.post_baseline_stable_claim_samples, 1);
        assert!(!ordinal.post_baseline_load_claim_witness);
        assert_eq!(ordinal.within_profile_band, Some(true), "a zero-versus-zero band alone is not evidence");
    }

    #[test]
    fn vram_calibration_rejects_rows_whose_claim_changed_during_telemetry() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        let mut otherwise_matching_peak = row(0, 20);
        otherwise_matching_peak.cuda_free_bytes = Some(8_000);
        otherwise_matching_peak.live_grants_bytes = 2_000;
        // Exercise the reducer's sequence check as well as the sampler's
        // boolean: an artifact row cannot regain eligibility merely because
        // a producer asserted stable while its recorded bracket disagrees.
        otherwise_matching_peak.claim_snapshot_stable = true;
        otherwise_matching_peak.claim_snapshot_end_sequence = Some(1);

        let evidence =
            vram_calibration_evidence(&[baseline, otherwise_matching_peak], &[], &[0], exact_calibration_policy());

        assert_eq!(evidence.status, VramCalibrationStatus::Failed);
        let ordinal = evidence.ordinals[0];
        assert_eq!(ordinal.post_baseline_stable_claim_samples, 0);
        assert!(!ordinal.post_baseline_load_claim_witness);
        assert_eq!(ordinal.cuda_peak_offset_ns, Some(20));
        assert_eq!(ordinal.cuda_peak_claim_snapshot_stable, Some(false));
        assert_eq!(ordinal.within_profile_band, None);
    }

    #[test]
    fn vram_calibration_does_not_hide_an_unstable_physical_peak_with_a_lower_stable_sample() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        let mut stable_lower_use = row(0, 20);
        stable_lower_use.cuda_free_bytes = Some(8_500);
        stable_lower_use.live_grants_bytes = 1_500;
        let mut unstable_physical_peak = row(0, 30);
        unstable_physical_peak.cuda_free_bytes = Some(8_000);
        unstable_physical_peak.live_grants_bytes = 2_000;
        unstable_physical_peak.claim_snapshot_end_sequence = Some(1);

        let evidence = vram_calibration_evidence(
            &[baseline, stable_lower_use, unstable_physical_peak],
            &[],
            &[0],
            exact_calibration_policy(),
        );

        assert_eq!(evidence.status, VramCalibrationStatus::Failed);
        let ordinal = evidence.ordinals[0];
        assert_eq!(ordinal.post_baseline_stable_claim_samples, 1);
        assert_eq!(ordinal.cuda_peak_offset_ns, Some(30));
        assert_eq!(ordinal.cuda_free_bytes_min, Some(8_000));
        assert_eq!(ordinal.cuda_peak_claim_snapshot_stable, Some(false));
        assert_eq!(ordinal.accounted_claim_bytes_at_cuda_peak, None);
        assert_eq!(ordinal.within_profile_band, None);
    }

    #[test]
    fn vram_calibration_rejects_an_unstable_preload_baseline() {
        let mut baseline = row(0, 10);
        baseline.cuda_free_bytes = Some(10_000);
        baseline.claim_snapshot_stable = false;
        baseline.claim_snapshot_end_sequence = Some(1);
        let mut peak = row(0, 20);
        peak.cuda_free_bytes = Some(8_000);
        peak.live_grants_bytes = 2_000;

        let evidence = vram_calibration_evidence(&[baseline, peak], &[], &[0], exact_calibration_policy());

        assert_eq!(evidence.status, VramCalibrationStatus::Failed);
        assert_eq!(evidence.ordinals[0].baseline_claim_snapshot_stable, Some(false));
        assert_eq!(evidence.ordinals[0].within_profile_band, None);
    }

    #[test]
    fn observation_pull_summary_uses_nearest_rank_percentiles() {
        let rows = [
            ProcessSeriesRow {
                offset_ns: 0,
                active_queries: 0,
                queued_queries: 0,
                backend_generation: 1,
                metric_generation: 1,
                observation_pull_ns: 10,
                completed_attempts: 0,
                successful_native_attempts: 0,
            },
            ProcessSeriesRow {
                offset_ns: 1,
                active_queries: 0,
                queued_queries: 0,
                backend_generation: 1,
                metric_generation: 2,
                observation_pull_ns: 30,
                completed_attempts: 0,
                successful_native_attempts: 0,
            },
            ProcessSeriesRow {
                offset_ns: 2,
                active_queries: 0,
                queued_queries: 0,
                backend_generation: 1,
                metric_generation: 3,
                observation_pull_ns: 20,
                completed_attempts: 0,
                successful_native_attempts: 0,
            },
        ];
        let summary = observation_pull_summary(&rows);
        assert_eq!(summary.samples, 3);
        assert_eq!(summary.min_ns, 10);
        assert_eq!(summary.p50_ns, 20);
        assert_eq!(summary.p95_ns, 30);
        assert_eq!(summary.max_ns, 30);
        assert_eq!(summary.mean_ns, 20);
    }

    #[test]
    fn fleet_interval_requires_continuous_utilization_cadence_and_useful_work() {
        let mut devices = Vec::new();
        let mut process = Vec::new();
        for second in 0..=60_u64 {
            let offset_ns = second * 1_000_000_000;
            for ordinal in 0..4 {
                let mut sample = row(ordinal, offset_ns);
                sample.active_attempts = 1;
                sample.managed_capacity_bytes = 100;
                sample.live_grants_bytes = 80;
                sample.cuda_free_bytes = Some(1);
                sample.nvml_used_bytes = Some(1);
                devices.push(sample);
            }
            process.push(ProcessSeriesRow {
                offset_ns,
                active_queries: 4,
                queued_queries: 0,
                backend_generation: 1,
                metric_generation: second,
                observation_pull_ns: 1,
                completed_attempts: second,
                successful_native_attempts: second,
            });
        }
        let policy = FleetUtilizationPolicy {
            device_ordinals: vec![0, 1, 2, 3],
            minimum_utilization_percent: 80,
            minimum_duration_ns: 60_000_000_000,
            maximum_sample_gap_ns: 1_000_000_000,
            minimum_active_attempts: 4,
            minimum_completed_attempts: 4,
            minimum_successful_native_attempts: 4,
        };
        let interval = fleet_utilization_interval(&devices, &process, &policy).unwrap().unwrap();
        assert_eq!(interval.duration_ns, 60_000_000_000);
        assert_eq!(interval.minimum_utilization_milli, 800);
        assert_eq!(interval.maximum_sample_gap.nanoseconds, 1_000_000_000);

        devices.iter_mut().filter(|row| row.offset_ns == 30_000_000_000).for_each(|row| {
            row.live_grants_bytes = 79;
        });
        assert!(fleet_utilization_interval(&devices, &process, &policy).unwrap().is_none());

        devices.iter_mut().filter(|row| row.offset_ns == 30_000_000_000).for_each(|row| {
            row.live_grants_bytes = 80;
            row.nvml_used_bytes = None;
        });
        assert!(fleet_utilization_interval(&devices, &process, &policy).unwrap().is_none());

        let mut duplicate_policy = policy.clone();
        duplicate_policy.device_ordinals = vec![0, 0, 1, 2];
        assert_eq!(
            fleet_utilization_interval(&devices, &process, &duplicate_policy),
            Err("invalid fleet utilization policy")
        );
    }

    #[test]
    fn maximum_physical_gap_includes_leading_and_trailing_outages() {
        let mut rows = (0..=4).map(|offset| row(0, offset)).collect::<Vec<_>>();
        for sample in &mut rows[1..=2] {
            sample.cuda_free_bytes = Some(1);
            sample.nvml_used_bytes = Some(1);
        }
        assert_eq!(maximum_physical_sample_gap(&rows), Some(SampleGap { nanoseconds: 2 }));
    }
}
