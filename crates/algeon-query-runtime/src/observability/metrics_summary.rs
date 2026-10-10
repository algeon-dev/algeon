//! Exact query metrics reduction kept by the observation owner.
//!
//! The owner updates these totals on every typed fact *before* deciding whether
//! the fact's detail row is retained, so a total never depends on retention.
//! Trace capture can be disabled or exhausted; the totals stay exact either way.
//!
//! Only the source-read family is reduced here so far. Families that have not
//! moved yet are still written into `QueryMetrics` directly, and are listed in
//! `docs/plan/2026-09-16-query-layer-boundaries/d4-observation-authorization.md`.

use std::time::Duration;

use crate::metrics::{QueryMetrics, SourceReadRuntimeMetrics};

/// One source-read fact, recorded once at the boundary that owns it.
///
/// Each variant has exactly one recording site. That is what keeps retries,
/// abandoned prefetches, and parallel workers from counting the same work
/// twice: a worker records its physical read, the scheduler records the
/// logical commit, and neither re-reports the other's facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceReadFact {
    /// A source chunk was planned.
    ChunkPlanned { row_groups: u64 },
    /// A source chunk read was attempted.
    ChunkAttempted,
    /// Source chunks were committed as logical output.
    ChunkCommitted { chunk_count: u64, row_groups: u64 },
    /// A source chunk read will be retried.
    ChunkRetried,
    /// A source-read prefetch was started.
    PrefetchStarted,
    /// A source-read prefetch completed.
    PrefetchCompleted,
    /// A source-read prefetch was skipped.
    PrefetchSkipped,
    /// A source-read prefetch was abandoned before its output was consumed.
    PrefetchAbandoned,
    /// Time one read spent between enqueue and a worker starting it.
    QueueWait(Duration),
    /// One input batch reached execution.
    InputBatch { rows: u64 },
    /// Parquet source work, as explicit increments. Recording sites add
    /// different subsets of these; a re-read that only learns its row-group
    /// count adds no file; so each carries exactly what it adds.
    ParquetSource { files: u64, known_bytes: u64, selected_row_groups: u64, read_columns: u64 },
    /// Time spent loading source metadata.
    MetadataLoadTime(Duration),
    /// Time spent inside the Parquet read call.
    ParquetReadTime(Duration),
    /// Time spent in source-read orchestration around the Parquet call.
    SourceReadTime(Duration),
}

/// Exact source-read totals for one query.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceReadTotals {
    pub(crate) input_batches: u64,
    pub(crate) input_rows: u64,
    pub parquet_files: u64,
    pub parquet_bytes_known: u64,
    pub row_groups_selected: u64,
    pub(crate) read_columns: u64,
    pub(crate) source_chunks_planned: u64,
    pub(crate) source_chunks_attempted: u64,
    pub(crate) source_chunks_committed: u64,
    pub(crate) source_chunks_retried: u64,
    pub(crate) row_groups_planned: u64,
    pub(crate) row_groups_committed: u64,
    pub(crate) source_read_prefetches_started: u64,
    pub(crate) source_read_prefetches_completed: u64,
    pub(crate) source_read_prefetches_skipped: u64,
    pub(crate) source_read_prefetches_abandoned: u64,
    pub(crate) source_metadata_load_time: Duration,
    pub source_read_queue_wait_time: Duration,
    pub(crate) parquet_read_time: Duration,
    pub(crate) source_read_time: Duration,
}

impl SourceReadTotals {
    /// Folds one fact into the totals. Returns `false` when a counter
    /// saturated, so the caller can mark the totals as no longer exact.
    pub(crate) fn apply(&mut self, fact: &SourceReadFact) -> bool {
        match fact {
            SourceReadFact::ChunkPlanned { row_groups } => {
                add_u64(&mut self.source_chunks_planned, 1) & add_u64(&mut self.row_groups_planned, *row_groups)
            },
            SourceReadFact::ChunkAttempted => add_u64(&mut self.source_chunks_attempted, 1),
            SourceReadFact::ChunkCommitted { chunk_count, row_groups } => {
                add_u64(&mut self.source_chunks_committed, *chunk_count)
                    & add_u64(&mut self.row_groups_committed, *row_groups)
            },
            SourceReadFact::ChunkRetried => add_u64(&mut self.source_chunks_retried, 1),
            SourceReadFact::PrefetchStarted => add_u64(&mut self.source_read_prefetches_started, 1),
            SourceReadFact::PrefetchCompleted => add_u64(&mut self.source_read_prefetches_completed, 1),
            SourceReadFact::PrefetchSkipped => add_u64(&mut self.source_read_prefetches_skipped, 1),
            SourceReadFact::PrefetchAbandoned => add_u64(&mut self.source_read_prefetches_abandoned, 1),
            SourceReadFact::InputBatch { rows } => {
                add_u64(&mut self.input_batches, 1) & add_u64(&mut self.input_rows, *rows)
            },
            SourceReadFact::ParquetSource { files, known_bytes, selected_row_groups, read_columns } => {
                add_u64(&mut self.parquet_files, *files)
                    & add_u64(&mut self.parquet_bytes_known, *known_bytes)
                    & add_u64(&mut self.row_groups_selected, *selected_row_groups)
                    & add_u64(&mut self.read_columns, *read_columns)
            },
            SourceReadFact::QueueWait(elapsed) => add_duration(&mut self.source_read_queue_wait_time, *elapsed),
            SourceReadFact::MetadataLoadTime(elapsed) => add_duration(&mut self.source_metadata_load_time, *elapsed),
            SourceReadFact::ParquetReadTime(elapsed) => add_duration(&mut self.parquet_read_time, *elapsed),
            SourceReadFact::SourceReadTime(elapsed) => add_duration(&mut self.source_read_time, *elapsed),
        }
    }
}

// `&` rather than `&&` throughout `apply`: every counter in a fact must be
// updated even after an earlier one saturates.
fn add_u64(counter: &mut u64, value: u64) -> bool {
    match counter.checked_add(value) {
        Some(next) => {
            *counter = next;
            true
        },
        None => {
            *counter = u64::MAX;
            false
        },
    }
}

fn add_duration(total: &mut Duration, value: Duration) -> bool {
    match total.checked_add(value) {
        Some(next) => {
            *total = next;
            true
        },
        None => {
            *total = Duration::MAX;
            false
        },
    }
}

/// Exact query metrics reduction kept by the observation owner.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryMetricsSummary {
    pub source_read: SourceReadTotals,
}

/// Whether detail rows were captured at all.
///
/// A disabled capture retains nothing on purpose, which is not the same as a
/// bounded capture that happened to retain everything. Zero retained rows and
/// zero losses under [`Self::Disabled`] say nothing about completeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetricsCaptureMode {
    /// Trace capture was off; only totals exist.
    Disabled,
    /// Detail rows were retained up to the recorder's record and byte limits.
    Bounded,
}

impl MetricsCaptureMode {
    /// Stable report label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Bounded => "bounded",
        }
    }
}

/// What a metrics checkpoint can and cannot claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricsCompleteness {
    pub(crate) capture_mode: MetricsCaptureMode,
    /// Detail rows the recorder declined to retain.
    pub(crate) detail_rows_dropped: u64,
    /// Whether any detail row was dropped.
    pub truncated: bool,
    /// Whether a total saturated, so totals are no longer exact.
    pub(crate) totals_saturated: bool,
}

impl MetricsCompleteness {
    /// Whether the checkpoint's detail rows are every row that was recorded.
    #[cfg(test)]
    pub(crate) const fn detail_is_complete(&self) -> bool {
        matches!(self.capture_mode, MetricsCaptureMode::Bounded) && !self.truncated
    }

    /// Stable report label for the retained detail: `not_captured` when trace
    /// capture was off, `incomplete` when rows were declined, else `complete`.
    pub(crate) const fn detail_status(&self) -> &'static str {
        match (self.capture_mode, self.truncated) {
            (MetricsCaptureMode::Disabled, _) => "not_captured",
            (MetricsCaptureMode::Bounded, true) => "incomplete",
            (MetricsCaptureMode::Bounded, false) => "complete",
        }
    }
}

/// One immutable, internally consistent view of the owner's metrics.
///
/// Totals, retained detail, and completeness come from a single lock
/// acquisition and carry one `revision`, so a projection never mixes two
/// moments. Revisions only increase; a projection that already applied a
/// revision can skip it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryMetricsCheckpoint {
    pub(crate) revision: u64,
    /// Whether the owner has sealed, so no later revision will follow.
    pub(crate) sealed: bool,
    pub summary: QueryMetricsSummary,
    /// Retained source-read detail rows, in recording order.
    pub source_read_rows: Vec<SourceReadRuntimeMetrics>,
    /// Reduction of every other metrics family, with its growing detail rows
    /// already bounded. Its source-read fields are unused; see `summary`.
    pub(crate) metrics: QueryMetrics,
    pub(crate) completeness: MetricsCompleteness,
}

#[cfg(test)]
mod tests;
