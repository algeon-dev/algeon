//! Recorder retention state, metrics reduction, and budget windows.

use std::collections::BTreeMap;

use super::{
    CleanupFailureSummary, ObservationSequence, PlanExecutionId, QueryMetrics, QueryMetricsCheckpoint,
    QueryMetricsSummary, RetainedTopologyRecord, RuntimeObservationLimits, RuntimeObservationLossReason,
    RuntimeObservationRecordClass, RuntimeObservationSealObserver, RuntimeTimelineEvent,
    SealedRuntimeObservationSnapshot, SourceReadRuntimeMetrics,
};
use crate::runtime::LakehouseCacheFillEventSnapshot;

/// Exact cleanup-failure accounting for an attempt report. It sits outside the
/// record limits, so bounded timeline loss never changes the count or the
/// first summary.
#[derive(Debug, Clone)]
pub(crate) struct CleanupFailureTally {
    count: u64,
    complete: bool,
    first: Option<CleanupFailureSummary>,
}

impl Default for CleanupFailureTally {
    fn default() -> Self {
        Self { count: 0, complete: true, first: None }
    }
}

impl CleanupFailureTally {
    /// Retains only the first cleanup failure while accounting every observed
    /// one. Overflow is marked incomplete rather than published as a
    /// saturated exact value.
    pub(super) fn record(&mut self, summary: CleanupFailureSummary) {
        if let Some(first) = &mut self.first {
            // The report retains one representative summary. Once another
            // cleanup failure is observed that representative is necessarily
            // incomplete, even though the count remains exact below.
            first.mark_incomplete();
        } else {
            self.first = Some(summary);
        }
        match self.count.checked_add(1) {
            Some(next) => self.count = next,
            None => self.complete = false,
        }
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }

    pub(crate) fn complete(&self) -> bool {
        self.complete
    }

    pub(crate) fn first(&self) -> Option<&CleanupFailureSummary> {
        self.first.as_ref()
    }

    #[cfg(test)]
    pub(super) fn with_count_for_test(count: u64) -> Self {
        Self { count, ..Self::default() }
    }
}

#[derive(Debug, Default)]
pub(super) struct RuntimeObservationRecorderState {
    pub(super) next_sequence: u64,
    pub(super) last_monotonic_ns: u64,
    pub(super) retained_records: u64,
    pub(super) retained_bytes: u64,
    pub(super) topology: Vec<RetainedTopologyRecord>,
    pub(super) events: Vec<RuntimeTimelineEvent>,
    pub(super) first_dropped_sequence: Option<ObservationSequence>,
    pub(super) last_attempted_sequence: Option<ObservationSequence>,
    pub(super) dropped_by_class: BTreeMap<RuntimeObservationRecordClass, u64>,
    pub(super) dropped_by_reason: BTreeMap<RuntimeObservationLossReason, u64>,
    pub(super) degraded: bool,
    pub(super) poison_recoveries: u64,
    pub(super) counter_overflow: bool,
    pub(super) sealed_snapshot: Option<SealedRuntimeObservationSnapshot>,
    pub(super) seal_observers: Vec<RuntimeObservationSealObserver>,
    pub(super) cleanup_failures: CleanupFailureTally,
    // Metrics reduction shares this lock with retention and loss accounting,
    // so one revision always describes totals, retained rows, and completeness
    // together. It is partitioned by plan execution because one query can run
    // several, and each execution's projection must carry only its own facts;
    // a consumer that sums projections must not count a fact twice.
    pub(super) metrics: BTreeMap<PlanExecutionId, PlanExecutionMetrics>,
    pub(super) metrics_revision: u64,
    // Query-wide share of the limits already taken by budget detail rows.
    pub(super) budget_detail_records: u64,
    pub(super) budget_detail_bytes: u64,
    // Background cache-fill completions no plan execution carries yet. A fill
    // finishes on its own schedule, so the next plan execution to freeze its
    // metrics takes them and offers them for retention. At most the record
    // limit wait; later ones only count as overflow, since no more than the
    // limit can be retained.
    pub(super) cache_fill_events: Vec<LakehouseCacheFillEventSnapshot>,
    pub(super) cache_fill_events_overflow: u64,
}

// Bounded by the number of plan executions in the query, not by data volume;
// detail rows inside are bounded by the recorder's shared record/byte limits.
#[derive(Debug, Default)]
pub(super) struct PlanExecutionMetrics {
    pub(super) summary: QueryMetricsSummary,
    pub(super) totals_saturated: bool,
    pub(super) source_read_rows: Vec<SourceReadRuntimeMetrics>,
    pub(super) source_read_rows_dropped: u64,
    // Reduction state for every family other than source reads. Only this
    // module mutates it, through the typed `MetricsRecorder` methods.
    pub(super) reduction: QueryMetrics,
    pub(super) detail_rows_dropped: u64,
    // Open budget windows and the peaks each has seen. A window is a nesting
    // scope, so there are at most a handful open at once.
    pub(super) budget_windows: BTreeMap<u64, BudgetWindowPeaks>,
    pub(super) next_budget_window: u64,
    // Set once the plan's owner projects its metrics; later facts for the
    // plan are ignored, so every projection agrees with this value.
    pub(super) frozen: Option<QueryMetricsCheckpoint>,
}

impl PlanExecutionMetrics {
    pub(super) const fn is_frozen(&self) -> bool {
        self.frozen.is_some()
    }
}

/// The largest allocator high-water and retained bytes the budget reports in
/// one window carried.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BudgetWindowPeaks {
    /// Largest allocator high-water bytes, if any report carried one.
    pub observed_peak_bytes: Option<u64>,
    /// Largest retained bytes, if any report carried one.
    pub retained_bytes: Option<u64>,
}

/// An open span over which budget reports are folded into one task's peaks.
///
/// Replaces a cursor into the retained budget events: a window sees every
/// report recorded while it is open, including ones bounded retention later
/// declines, so a task's peak never depends on what was kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "an open budget window must be closed onto its task"]
pub struct BudgetWindow(pub(super) u64);

impl RuntimeObservationRecorderState {
    pub(super) fn take_sequence(&mut self) -> ObservationSequence {
        let value = self.next_sequence.max(1);
        self.next_sequence = value.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        let sequence = ObservationSequence::new(value);
        self.last_attempted_sequence = Some(sequence);
        sequence
    }

    pub(super) fn rejection_reason(
        &self,
        limits: RuntimeObservationLimits,
        logical_bytes: u64,
    ) -> Option<RuntimeObservationLossReason> {
        if self.retained_records >= limits.max_retained_records {
            Some(RuntimeObservationLossReason::RecordLimit)
        } else if logical_bytes > limits.max_retained_bytes {
            Some(RuntimeObservationLossReason::Oversized)
        } else if self.retained_bytes.checked_add(logical_bytes).is_none_or(|sum| sum > limits.max_retained_bytes) {
            Some(RuntimeObservationLossReason::ByteLimit)
        } else {
            None
        }
    }

    pub(super) fn record_drop(
        &mut self,
        class: RuntimeObservationRecordClass,
        sequence: ObservationSequence,
        reason: RuntimeObservationLossReason,
    ) {
        self.first_dropped_sequence.get_or_insert(sequence);
        let class_count = self.dropped_by_class.get(&class).copied().unwrap_or(0);
        let next_class_count = class_count.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        self.dropped_by_class.insert(class, next_class_count);
        let reason_count = self.dropped_by_reason.get(&reason).copied().unwrap_or(0);
        let next_reason_count = reason_count.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        self.dropped_by_reason.insert(reason, next_reason_count);
    }

    /// Admits one metrics detail row.
    ///
    /// Budget reports arrive once per budgeted allocation call, so they grow
    /// with data far faster than any other family. They may take at most half
    /// of the shared limits; otherwise a large query would spend the whole
    /// capacity on them and lose every later strategy decision or operator row.
    pub(super) fn admit_detail(
        &mut self,
        limits: RuntimeObservationLimits,
        class: RuntimeObservationRecordClass,
        sequence: ObservationSequence,
        logical_bytes: u64,
    ) -> Result<(), RuntimeObservationLossReason> {
        if class != RuntimeObservationRecordClass::BudgetDetail {
            return self.admit(limits, class, sequence, logical_bytes);
        }
        let share = RuntimeObservationLimits {
            max_retained_records: limits.max_retained_records / 2,
            max_retained_bytes: limits.max_retained_bytes / 2,
        };
        let share_rejection = if self.budget_detail_records >= share.max_retained_records {
            Some(RuntimeObservationLossReason::RecordLimit)
        } else if self.budget_detail_bytes.saturating_add(logical_bytes) > share.max_retained_bytes {
            Some(RuntimeObservationLossReason::ByteLimit)
        } else {
            None
        };
        if let Some(reason) = share_rejection {
            self.record_drop(class, sequence, reason);
            return Err(reason);
        }
        self.admit(limits, class, sequence, logical_bytes)?;
        self.budget_detail_records = self.budget_detail_records.saturating_add(1);
        self.budget_detail_bytes = self.budget_detail_bytes.saturating_add(logical_bytes);
        Ok(())
    }

    pub(super) fn admit(
        &mut self,
        limits: RuntimeObservationLimits,
        class: RuntimeObservationRecordClass,
        sequence: ObservationSequence,
        logical_bytes: u64,
    ) -> Result<(), RuntimeObservationLossReason> {
        if let Some(reason) = self.rejection_reason(limits, logical_bytes) {
            self.record_drop(class, sequence, reason);
            return Err(reason);
        }
        self.retained_records = self.retained_records.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        self.retained_bytes = self.retained_bytes.checked_add(logical_bytes).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        Ok(())
    }
}
