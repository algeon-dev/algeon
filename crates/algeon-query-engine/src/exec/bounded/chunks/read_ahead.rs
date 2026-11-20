//! One source read kept ahead of a consumer that runs its chunks one at a time.
//!
//! The consumer reads chunk N (or takes the read started for it), starts the read of chunk N+1
//! on a free attempt auxiliary permit, then runs its stages and operator on chunk N. The read
//! of N+1 decodes on the leased stream while N's kernels run on the consumer's stream. At most
//! one read is ahead, chunks are taken in planned order, and the consumer commits them in that
//! order, so commit order and merge order do not depend on the read ahead.
//!
//! The read ahead is optional work. Chunk planning reserves one chunk of the grant for it.
//! Admission can refuse it (`source_read_ahead_admitted`), a missing permit skips it, and a
//! failed read is dropped and the consumer reads the chunk itself. The attempt holds every read
//! ahead through `ExecutionContext::register_read_ahead`, so the retry entry
//! `backoff_before_retry` releases it before any query-cap retry; a released read ahead starts
//! no further read, and its consumer continues serially. A driver whose own work fails with a
//! memory failure while a read is pending releases it and runs that work again without it
//! (`ReadAheadStep::release_after_failure`).
//!
//! Two memory failures are terminal while a read can be pending, and there the pending read's
//! bytes count against the same query cap: the aggregate's commit of a chunk's partials into
//! its merge state, which runs after the chunk's work and cannot be replayed, and work of a
//! pull-stream consumer that is not retried, which the stream does not see. Dropping the owner
//! joins and discards a pending read, which returns its permit, stream and memory on every exit
//! path. Design owner: `website/docs/design/data-path.mdx`.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use query_planning::error::QueryErrorCode;
use query_runtime::runtime::ReadAheadRelease;

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, ExecutionContext, ExecutionMetrics,
    LocalParquetSource, QueryError, Result, SourceChunk, SourceChunkBatch, source_budget::source_read_ahead_admitted,
};
use crate::exec::{
    retry::{ResourceFailureRecovery, resource_failure_recovery},
    source::{ScheduledLocalParquetRead, ScheduledLocalParquetReadOutput, ScheduledSourceRead},
};

#[cfg(test)]
mod gpu;

/// Budget boundary of a read started ahead of an aggregate, whose own reads use
/// `source_chunk_read`.
pub(in crate::exec::bounded) const AGGREGATE_READ_AHEAD_BOUNDARY: &str = "source_chunk_prefetch";
/// Budget boundary of a read started ahead of a join leaf consumer, whose own reads use
/// `join_source_read`.
pub(in crate::exec::bounded) const JOIN_LEAF_READ_AHEAD_BOUNDARY: &str = "join_source_read_prefetch";

/// Whether a source consumer of this attempt keeps one read ahead of itself.
///
/// Chunk planning adds the read-ahead slot of `SourceTaskMemoryModel` when this holds and every
/// consumer of the stream has a driver that reads ahead, and records that on the stream; a
/// driver starts reads ahead only for a stream planned that way (`SourceReadAhead::for_stream`). A compiled run that holds two or more execution
/// lanes keeps none: its lanes overlap reads and hold the auxiliary permits a read ahead would
/// take. A context that is itself an attempt work lease already holds an auxiliary permit and
/// reads inline, so a read ahead there would only keep two decoded chunks live.
/// `max_source_reads_in_flight = 1` turns it off.
pub(in crate::exec::bounded) fn source_read_ahead_applies(execution: &BoundedExecutionAttempt<'_>) -> bool {
    let context = execution.context();
    execution.run_lane_width() <= 1
        && context.source_scheduler_policy().enables_prefetch()
        && !context.is_attempt_work_lease_context()
}

/// The single read a consumer keeps ahead of itself.
///
/// `scratch_limit_bytes` is the consumer's planned chunk target. Every read ahead takes it as
/// its pass read limit, which keeps its read scratch within the slot the chunk planner reserved
/// for it.
pub(in crate::exec::bounded) struct SourceReadAhead {
    slot: Arc<ReadAheadSlot>,
    scratch_limit_bytes: Option<u64>,
    budget_boundary: &'static str,
}

/// The pending read, shared with the attempt's registry so a retry can release it.
struct ReadAheadSlot {
    context: ExecutionContext,
    metrics: ExecutionMetrics,
    state: Mutex<ReadAheadState>,
}

#[derive(Default)]
struct ReadAheadState {
    pending: Option<(Vec<ReadAheadChunkKey>, ScheduledLocalParquetRead)>,
    released: bool,
}

/// Identity of one chunk a pending read covers. A retry split keeps the chunk index and
/// changes the row groups, so both take part.
#[derive(PartialEq, Eq)]
struct ReadAheadChunkKey {
    chunk_index: usize,
    file_index: usize,
    row_group_indexes: Vec<usize>,
}

fn read_ahead_keys(chunks: &[SourceChunk]) -> Vec<ReadAheadChunkKey> {
    chunks
        .iter()
        .map(|chunk| ReadAheadChunkKey {
            chunk_index: chunk.chunk_index,
            file_index: chunk.file_index,
            row_group_indexes: chunk.row_group_indexes.clone(),
        })
        .collect()
}

impl ReadAheadSlot {
    fn lock(&self) -> MutexGuard<'_, ReadAheadState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Marks the slot released and abandons its pending read. Returns whether one was pending.
    fn release_pending(&self) -> bool {
        let pending = {
            let mut state = self.lock();
            state.released = true;
            state.pending.take()
        };
        let Some((_, read)) = pending else {
            return false;
        };
        read.abandon_in(&self.context, &self.metrics);
        true
    }
}

impl ReadAheadRelease for ReadAheadSlot {
    fn release(&self) {
        self.release_pending();
    }
}

impl SourceReadAhead {
    /// A read ahead of `stream`'s chunks registered with the attempt, or `None` unless the
    /// chunks were planned with the read-ahead slot and `source_read_ahead_applies` still holds.
    /// Its reads run under `budget_boundary`, with the planned chunk target as scratch limit.
    pub(in crate::exec::bounded) fn for_stream(
        execution: &BoundedExecutionAttempt<'_>,
        stream: &BoundedJoinLeafLocalParquetSourceStream,
        budget_boundary: &'static str,
    ) -> Option<Self> {
        if !stream.planned_read_ahead() || !source_read_ahead_applies(execution) {
            return None;
        }
        let scratch_limit_bytes = stream.planned_chunk_target_bytes();
        let slot = Arc::new(ReadAheadSlot {
            context: execution.context().clone(),
            metrics: execution.metrics().clone(),
            state: Mutex::new(ReadAheadState::default()),
        });
        let registered: Weak<dyn ReadAheadRelease> = Arc::downgrade(&slot) as Weak<dyn ReadAheadRelease>;
        execution.context().register_read_ahead(registered);
        Some(Self { slot, scratch_limit_bytes, budget_boundary })
    }

    /// Starts the read of `batch` ahead of its consumer.
    ///
    /// Starts nothing while a read is pending or after a release. A read that admission
    /// refuses, that finds no auxiliary permit free, or whose lease or submission fails is
    /// counted as a skipped prefetch and leaves the batch to its consumer. A started read
    /// counts one chunk attempt per chunk and one prefetch start. Only cancellation is
    /// returned as an error.
    pub(in crate::exec::bounded) fn start(
        &self,
        execution: &BoundedExecutionAttempt<'_>,
        source: &LocalParquetSource,
        batch: SourceChunkBatch,
        reader_dynamic_filter_count: usize,
    ) -> Result<()> {
        let scratch_limit_bytes = self.scratch_limit_bytes;
        {
            let state = self.slot.lock();
            if state.released || state.pending.is_some() {
                return Ok(());
            }
        }
        let metrics = execution.metrics();
        if !source_read_ahead_admitted(execution.context(), &batch, scratch_limit_bytes) {
            metrics.record_source_read_prefetch_skipped();
            return Ok(());
        }
        let keys = read_ahead_keys(&batch.chunks);
        let chunk_count = batch.chunk_count();
        let read = ScheduledSourceRead {
            source: source.clone(),
            batch,
            budget_bytes: execution.device_budget_bytes(),
            budget_boundary: self.budget_boundary,
            reader_dynamic_filter_count,
            scratch_limit_bytes,
        };
        let read =
            match ScheduledLocalParquetRead::spawn_prefetch(execution.context().clone(), metrics.clone(), read, 1) {
                Ok(Some(read)) => read,
                Err(error) if error.code() == QueryErrorCode::Cancelled => return Err(error),
                Ok(None) | Err(_) => {
                    metrics.record_source_read_prefetch_skipped();
                    return Ok(());
                },
            };
        for _ in 0..chunk_count {
            metrics.record_source_chunk_attempt();
        }
        metrics.record_source_read_prefetch_started();
        let released_while_starting = {
            let mut state = self.slot.lock();
            if state.released {
                Some(read)
            } else {
                state.pending = Some((keys, read));
                None
            }
        };
        // A retry on another thread released the slot while this read was starting.
        if let Some(read) = released_while_starting {
            read.abandon_in(execution.context(), metrics);
        }
        Ok(())
    }

    /// Joins the pending read when it covers exactly `chunks` and returns its output.
    ///
    /// `Ok(None)` when no read is pending, when the pending read covers other chunks (it is
    /// abandoned), or when the read failed: a failure is counted as an abandoned prefetch and
    /// the consumer reads the chunks itself, so a failed read ahead is not reported.
    /// Cancellation is returned. The output's `observed_bytes` is 0, as from
    /// `ScheduledLocalParquetRead::finish`.
    pub(in crate::exec::bounded) fn finish(
        &self,
        chunks: &[SourceChunk],
    ) -> Result<Option<ScheduledLocalParquetReadOutput>> {
        let pending = self.slot.lock().pending.take();
        let Some((keys, read)) = pending else {
            return Ok(None);
        };
        if keys != read_ahead_keys(chunks) {
            read.abandon_in(&self.slot.context, &self.slot.metrics);
            return Ok(None);
        }
        match read.finish() {
            Ok(output) => {
                self.slot.metrics.record_source_read_prefetch_completed();
                Ok(Some(output))
            },
            Err(error) if error.code() == QueryErrorCode::Cancelled => Err(error),
            Err(_) => {
                self.slot.metrics.record_source_read_prefetch_abandoned();
                Ok(None)
            },
        }
    }

    /// Abandons the pending read, if any, and starts no further one. Returns whether a read
    /// was pending.
    pub(in crate::exec::bounded) fn release(&self) -> bool {
        self.slot.release_pending()
    }
}

impl Drop for SourceReadAhead {
    fn drop(&mut self) {
        self.slot.release_pending();
    }
}

/// `chunk` as a one-chunk batch, the unit a read ahead of a single chunk covers.
pub(in crate::exec::bounded) fn read_ahead_chunk_batch(chunk: &SourceChunk) -> SourceChunkBatch {
    SourceChunkBatch {
        batch_index: chunk.chunk_index,
        estimated_source_read_bytes: chunk.estimated_source_read_bytes,
        source_batching_reason: "read_ahead".to_string(),
        chunks: vec![chunk.clone()],
    }
}

/// The read-ahead step of one chunk's work: the consumer's read ahead and the batch it reads
/// next, if any.
pub(in crate::exec::bounded) struct ReadAheadStep<'a> {
    pub(in crate::exec::bounded) read_ahead: &'a SourceReadAhead,
    pub(in crate::exec::bounded) next: Option<SourceChunkBatch>,
}

impl<'a> ReadAheadStep<'a> {
    /// The step of a consumer that reads single chunks; `next` is the chunk after this one.
    pub(in crate::exec::bounded) fn chunk(read_ahead: &'a SourceReadAhead, next: Option<&SourceChunk>) -> Self {
        Self { read_ahead, next: next.map(read_ahead_chunk_batch) }
    }

    /// Releases the read ahead after the consumer's work failed with `error`, when the failure
    /// is a memory failure (a query-cap rejection or device exhaustion) and a read was pending
    /// next to that work. Returns whether it released one; the caller then runs the work again
    /// with no read ahead, so that work's memory failure is not caused by the read ahead. Work
    /// outside such a caller is covered only by the planned slot (see the module docs).
    pub(in crate::exec::bounded) fn release_after_failure(&self, error: &QueryError) -> bool {
        resource_failure_recovery(error) != ResourceFailureRecovery::NotResource && self.read_ahead.release()
    }

    /// Starts the read of `next` ahead of the consumer. See `SourceReadAhead::start`.
    pub(in crate::exec::bounded) fn start_next(
        &self,
        execution: &BoundedExecutionAttempt<'_>,
        source: &LocalParquetSource,
        reader_dynamic_filter_count: usize,
    ) -> Result<()> {
        let Some(next) = &self.next else {
            return Ok(());
        };
        self.read_ahead.start(execution, source, next.clone(), reader_dynamic_filter_count)
    }
}
