//! Pure audit of public backend-metric snapshots.
//!
//! The measured runner turns the public backend snapshot into the small DTO
//! below.  Keeping the audit independent from CUDA and the backend types makes
//! the validation rules testable in the CPU-only runner feature.

use std::collections::BTreeSet;

use serde::Serialize;

use super::aggregate::ClauseStatus;

/// One audited public backend-metrics snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MetricSnapshotObservation {
    pub(super) offset_ns: u64,
    pub(super) backend_generation: u64,
    pub(super) generation: u64,
    pub(super) configured_device_ordinals: Vec<usize>,
    /// Number of descriptor-owned metric samples returned by the public
    /// projection for this snapshot, if every projection succeeded.
    pub(super) projected_series_count: Option<usize>,
    /// Independently calculated expected cardinality for the frozen descriptor
    /// registry and this backend's configured ordinal set.
    pub(super) expected_series_count: usize,
    /// Sum of the terminal `attempts.completed` series across the closed
    /// disposition/outcome vocabulary.
    pub(super) terminal_completed: u64,
    /// Aggregation of every `total` duration histogram.  Counts remain
    /// cumulative by bound and the last entry is the `+Inf` bucket.
    pub(super) total_duration_bucket_counts: Vec<u64>,
    pub(super) total_duration_count: u64,
}

/// One auditable backend-metric clause.
#[derive(Debug, Clone, Serialize)]
pub(super) struct MetricAuditClause {
    pub(super) clause: &'static str,
    pub(super) status: ClauseStatus,
    pub(super) detail: String,
}

/// Explicit delivery accounting for the pull-only backend-metric store.
///
/// The measured runner reads a public snapshot directly and does not install
/// an exporter or delivery queue. The record therefore names the pull-only mode
/// and carries no loss count.
#[derive(Debug, Clone, Serialize)]
pub(super) struct DeliveryLossEvidence {
    pub(super) mode: &'static str,
    pub(super) detail: &'static str,
}

/// Bounded metrics audit derived from timestamped public snapshots.
#[derive(Debug, Clone, Serialize)]
pub(super) struct MetricsAudit {
    pub(super) requested_cadence_ms: u64,
    pub(super) retained_snapshots: u64,
    pub(super) raw_snapshot_retention_truncated: bool,
    pub(super) snapshot_cadence: MetricAuditClause,
    pub(super) fixed_cardinality: MetricAuditClause,
    pub(super) report_to_metric_equality: MetricAuditClause,
    pub(super) duration_bucket_suitability: MetricAuditClause,
    pub(super) delivery_loss: DeliveryLossEvidence,
}

impl MetricsAudit {
    /// Audits a bounded sequence of timestamped snapshots.
    ///
    /// `reported_attempts` comes from the artifact's retained report rows,
    /// while `terminal_completed` comes from the backend-metric store. They intentionally
    /// take different paths so a dropped report or metric projection cannot be
    /// masked by one shared accumulator.
    pub(super) fn from_snapshots(
        snapshots: &[MetricSnapshotObservation],
        expected_device_ordinals: &[usize],
        reported_attempts: usize,
        requested_cadence_ms: u64,
        raw_snapshot_retention_truncated: bool,
    ) -> Self {
        let requested_cadence_ns = requested_cadence_ms.saturating_mul(1_000_000);
        let snapshot_cadence = cadence_clause(snapshots, requested_cadence_ns, raw_snapshot_retention_truncated);
        let fixed_cardinality = cardinality_clause(snapshots, expected_device_ordinals);
        let report_to_metric_equality = report_equality_clause(snapshots, reported_attempts);
        let duration_bucket_suitability = bucket_clause(snapshots);
        Self {
            requested_cadence_ms,
            retained_snapshots: snapshots.len() as u64,
            raw_snapshot_retention_truncated,
            snapshot_cadence,
            fixed_cardinality,
            report_to_metric_equality,
            duration_bucket_suitability,
            delivery_loss: DeliveryLossEvidence {
                mode: "not_applicable_pull_only",
                detail: "the runner reads pull snapshots directly and installs no exporter or delivery queue",
            },
        }
    }

    /// A metric audit is blocking only when a projection-integrity clause
    /// fails. Histogram saturation is an audit usability finding. It must stay
    /// explicit in the artifact, but it cannot turn a bounded, otherwise
    /// coherent workload into a false negative because the metric cannot
    /// estimate that workload's p99.
    /// The delivery field is intentionally not a synthetic success claim.
    pub(super) fn passes(&self) -> bool {
        [&self.snapshot_cadence, &self.fixed_cardinality, &self.report_to_metric_equality]
            .into_iter()
            .all(|clause| clause.status != ClauseStatus::Failed)
    }
}

fn clause(clause: &'static str, status: ClauseStatus, detail: impl Into<String>) -> MetricAuditClause {
    MetricAuditClause { clause, status, detail: detail.into() }
}

fn cadence_clause(
    snapshots: &[MetricSnapshotObservation],
    requested_cadence_ns: u64,
    raw_snapshot_retention_truncated: bool,
) -> MetricAuditClause {
    if raw_snapshot_retention_truncated {
        return clause(
            "snapshot_cadence",
            ClauseStatus::NotEvidenced,
            "the bounded raw snapshot retention cap was reached before the run ended",
        );
    }
    if snapshots.len() < 2 {
        return clause(
            "snapshot_cadence",
            ClauseStatus::NotEvidenced,
            format!("{} retained snapshot(s); at least two are required to measure cadence", snapshots.len()),
        );
    }
    let max_gap_ns =
        snapshots.windows(2).map(|pair| pair[1].offset_ns.saturating_sub(pair[0].offset_ns)).max().unwrap_or_default();
    // Three periods leaves scheduler and final-sample slack while still
    // detecting a stopped or materially stalled sampler.
    let maximum_allowed_ns = requested_cadence_ns.saturating_mul(3);
    if max_gap_ns <= maximum_allowed_ns {
        clause(
            "snapshot_cadence",
            ClauseStatus::Evidenced,
            format!(
                "{} snapshots retained; largest interval {} ms within the {} ms audit bound",
                snapshots.len(),
                max_gap_ns / 1_000_000,
                maximum_allowed_ns / 1_000_000
            ),
        )
    } else {
        clause(
            "snapshot_cadence",
            ClauseStatus::Failed,
            format!(
                "largest snapshot interval {} ms exceeds the {} ms audit bound",
                max_gap_ns / 1_000_000,
                maximum_allowed_ns / 1_000_000
            ),
        )
    }
}

fn cardinality_clause(
    snapshots: &[MetricSnapshotObservation],
    expected_device_ordinals: &[usize],
) -> MetricAuditClause {
    if snapshots.is_empty() {
        return clause("fixed_cardinality", ClauseStatus::NotEvidenced, "no backend metric snapshots were retained");
    }
    let expected = normalized_ordinals(expected_device_ordinals);
    if expected.len() != expected_device_ordinals.len() {
        return clause(
            "fixed_cardinality",
            ClauseStatus::Failed,
            "the immutable backend device configuration contains duplicate ordinals",
        );
    }
    let backend_generation = snapshots[0].backend_generation;
    for (index, snapshot) in snapshots.iter().enumerate() {
        if snapshot.backend_generation != backend_generation {
            return clause(
                "fixed_cardinality",
                ClauseStatus::Failed,
                format!(
                    "snapshot {index} belongs to backend generation {}, expected {backend_generation}",
                    snapshot.backend_generation
                ),
            );
        }
        let actual = normalized_ordinals(&snapshot.configured_device_ordinals);
        if actual.len() != snapshot.configured_device_ordinals.len() || actual != expected {
            return clause(
                "fixed_cardinality",
                ClauseStatus::Failed,
                format!("snapshot {index} configured ordinals do not match the immutable backend configuration"),
            );
        }
        let Some(projected_series) = snapshot.projected_series_count else {
            return clause(
                "fixed_cardinality",
                ClauseStatus::Failed,
                format!("snapshot {index} could not project every canonical descriptor"),
            );
        };
        if projected_series != snapshot.expected_series_count {
            return clause(
                "fixed_cardinality",
                ClauseStatus::Failed,
                format!(
                    "snapshot {index} projected {projected_series} series, expected {} from the closed descriptor registry",
                    snapshot.expected_series_count
                ),
            );
        }
    }
    clause(
        "fixed_cardinality",
        ClauseStatus::Evidenced,
        format!(
            "{} snapshots from backend generation {backend_generation} each retained {} configured device labels and the descriptor-owned series cardinality",
            snapshots.len(),
            expected.len()
        ),
    )
}

fn normalized_ordinals(ordinals: &[usize]) -> Vec<usize> {
    ordinals.iter().copied().collect::<BTreeSet<_>>().into_iter().collect()
}

fn report_equality_clause(snapshots: &[MetricSnapshotObservation], reported_attempts: usize) -> MetricAuditClause {
    let Some(last) = snapshots.last() else {
        return clause(
            "report_to_metric_equality",
            ClauseStatus::NotEvidenced,
            "no final snapshot was retained after attempt drain",
        );
    };
    let reported_attempts = reported_attempts as u64;
    if last.terminal_completed == reported_attempts {
        clause(
            "report_to_metric_equality",
            ClauseStatus::Evidenced,
            format!(
                "final terminal counter {} equals {} retained report rows at metrics generation {}",
                last.terminal_completed, reported_attempts, last.generation
            ),
        )
    } else {
        clause(
            "report_to_metric_equality",
            ClauseStatus::Failed,
            format!(
                "final terminal counter {} differs from {} retained report rows",
                last.terminal_completed, reported_attempts
            ),
        )
    }
}

fn bucket_clause(snapshots: &[MetricSnapshotObservation]) -> MetricAuditClause {
    let Some(last) = snapshots.last() else {
        return clause(
            "duration_bucket_suitability",
            ClauseStatus::NotEvidenced,
            "no final snapshot was retained for duration-histogram audit",
        );
    };
    if last.total_duration_bucket_counts.len() < 2 {
        return clause(
            "duration_bucket_suitability",
            ClauseStatus::Failed,
            "the total-duration histogram omitted its finite or +Inf buckets",
        );
    }
    let mut previous = 0_u64;
    for (index, count) in last.total_duration_bucket_counts.iter().copied().enumerate() {
        if count < previous {
            return clause(
                "duration_bucket_suitability",
                ClauseStatus::Failed,
                format!("total-duration cumulative bucket {index} decreased from {previous} to {count}"),
            );
        }
        previous = count;
    }
    if previous != last.total_duration_count {
        return clause(
            "duration_bucket_suitability",
            ClauseStatus::Failed,
            format!("total-duration +Inf bucket {previous} differs from histogram count {}", last.total_duration_count),
        );
    }
    if last.total_duration_count == 0 {
        return clause(
            "duration_bucket_suitability",
            ClauseStatus::NotEvidenced,
            "no terminal total-duration samples reached the duration histogram",
        );
    }
    let finite = last.total_duration_bucket_counts[last.total_duration_bucket_counts.len() - 2];
    let overflow = last.total_duration_count.saturating_sub(finite);
    if finite == 0 {
        return clause(
            "duration_bucket_suitability",
            ClauseStatus::NotEvidenced,
            format!("all {} total-duration samples overflowed the largest finite bucket", last.total_duration_count),
        );
    }
    clause(
        "duration_bucket_suitability",
        ClauseStatus::Evidenced,
        format!(
            "{} total-duration samples; {finite} fit a finite bucket and {overflow} used +Inf",
            last.total_duration_count
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(offset_ns: u64, terminal_completed: u64) -> MetricSnapshotObservation {
        MetricSnapshotObservation {
            offset_ns,
            backend_generation: 7,
            generation: offset_ns,
            configured_device_ordinals: vec![0, 1],
            projected_series_count: Some(116),
            expected_series_count: 116,
            terminal_completed,
            total_duration_bucket_counts: vec![0, 0, 0, terminal_completed, terminal_completed],
            total_duration_count: terminal_completed,
        }
    }

    #[test]
    fn audit_requires_cadence_closed_cardinality_reports_and_finite_buckets() {
        let snapshots = vec![snapshot(0, 0), snapshot(250_000_000, 2)];
        let audit = MetricsAudit::from_snapshots(&snapshots, &[0, 1], 2, 250, false);
        assert_eq!(audit.snapshot_cadence.status, ClauseStatus::Evidenced);
        assert_eq!(audit.fixed_cardinality.status, ClauseStatus::Evidenced);
        assert_eq!(audit.report_to_metric_equality.status, ClauseStatus::Evidenced);
        assert_eq!(audit.duration_bucket_suitability.status, ClauseStatus::Evidenced);
        assert!(audit.passes());
    }

    #[test]
    fn audit_fails_on_a_cardinality_or_report_count_mismatch() {
        let mut snapshots = vec![snapshot(0, 0), snapshot(250_000_000, 2)];
        snapshots[1].projected_series_count = Some(115);
        let audit = MetricsAudit::from_snapshots(&snapshots, &[0, 1], 1, 250, false);
        assert_eq!(audit.fixed_cardinality.status, ClauseStatus::Failed);
        assert_eq!(audit.report_to_metric_equality.status, ClauseStatus::Failed);
        assert!(!audit.passes());
    }

    #[test]
    fn audit_fails_a_stalled_sampler_and_reports_all_overflow_histogram() {
        let mut snapshots = vec![snapshot(0, 0), snapshot(1_000_000_000, 2)];
        snapshots[1].total_duration_bucket_counts = vec![0, 0, 0, 0, 2];
        let audit = MetricsAudit::from_snapshots(&snapshots, &[0, 1], 2, 250, false);
        assert_eq!(audit.snapshot_cadence.status, ClauseStatus::Failed);
        assert_eq!(audit.duration_bucket_suitability.status, ClauseStatus::NotEvidenced);
        assert!(!audit.passes());
    }

    #[test]
    fn all_overflow_histograms_are_reported_without_failing_projection_integrity() {
        let mut snapshots = vec![snapshot(0, 0), snapshot(250_000_000, 2)];
        snapshots[1].total_duration_bucket_counts = vec![0, 0, 0, 0, 2];
        let audit = MetricsAudit::from_snapshots(&snapshots, &[0, 1], 2, 250, false);
        assert_eq!(audit.duration_bucket_suitability.status, ClauseStatus::NotEvidenced);
        assert!(audit.passes());
    }

    #[test]
    fn retention_cap_leaves_cadence_explicitly_unevidenced() {
        let audit = MetricsAudit::from_snapshots(&[snapshot(0, 1)], &[0, 1], 1, 250, true);
        assert_eq!(audit.snapshot_cadence.status, ClauseStatus::NotEvidenced);
        assert!(audit.passes());
    }
}
