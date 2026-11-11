//! Bounded incremental GPU-to-host Arrow materialization cursor.
use std::sync::Arc;

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use super::GpuRecordBatch;
use crate::core::error::{Error, ErrorCode, Result};

/// Bounds one GPU-to-host materialization step by estimated final Arrow buffer
/// capacity and row count.
///
/// `target_materialization_bytes` is the soft chunking target. A row that
/// cannot be split may exceed that target, but it must not exceed
/// `max_materialization_bytes`, the hard pre-copy ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostBatchCursorConfig {
    pub target_materialization_bytes: usize,
    pub max_materialization_bytes: usize,
    pub max_rows: usize,
}

impl HostBatchCursorConfig {
    pub const DEFAULT_TARGET_MATERIALIZATION_BYTES: usize = 2 * 1024 * 1024;
    pub const DEFAULT_MAX_MATERIALIZATION_BYTES: usize = 4 * 1024 * 1024;
    /// The default soft target divided by an 8-byte row, so for rows of 8
    /// bytes or wider the byte target, not this cap, ends a chunk.
    pub const DEFAULT_MAX_ROWS: usize = 256 * 1024;
}

impl Default for HostBatchCursorConfig {
    fn default() -> Self {
        Self {
            target_materialization_bytes: Self::DEFAULT_TARGET_MATERIALIZATION_BYTES,
            max_materialization_bytes: Self::DEFAULT_MAX_MATERIALIZATION_BYTES,
            max_rows: Self::DEFAULT_MAX_ROWS,
        }
    }
}

/// A row range of the cursor's GPU batch, sized for one bounded host copy.
///
/// Preparing a step allocates no device memory and copies no values to host;
/// sizing reads only the string and list offsets of the candidate range. The
/// token names a row range of the originating cursor's batch, so a caller can
/// reserve host capacity before asking that cursor to copy it.
#[derive(Debug)]
pub struct PreparedHostBatch {
    cursor_identity: Arc<()>,
    offset: usize,
    rows: usize,
    batch_index: usize,
    estimated_materialization_bytes: usize,
}

impl PreparedHostBatch {
    /// Conservative final Arrow buffer capacity that must be reserved before
    /// the host copy starts.
    pub fn estimated_materialization_bytes(&self) -> usize {
        self.estimated_materialization_bytes
    }

    /// Number of rows in this prepared chunk.
    pub fn num_rows(&self) -> usize {
        self.rows
    }
}

/// Incremental GPU-to-host Arrow materializer.
///
/// The cursor retains the original GPU batch and copies each chunk to host
/// through zero-copy views of its row range. An initially empty GPU batch
/// still yields exactly one schema-preserving host batch before reaching EOF,
/// matching a full result materialization boundary. A schema-aware upper bound
/// for final Arrow buffer capacities picks the largest row count within the
/// soft target. A single row may exceed that target, but an unsplittable row
/// that exceeds the hard materialization ceiling is rejected with a permanent
/// structured resource error before any host value copy.
pub struct HostBatchCursor {
    batch: GpuRecordBatch,
    config: HostBatchCursorConfig,
    offset: usize,
    emitted_batches: usize,
    identity: Arc<()>,
}

impl HostBatchCursor {
    pub fn new(batch: GpuRecordBatch, config: HostBatchCursorConfig) -> Result<Self> {
        if config.target_materialization_bytes == 0 || config.max_materialization_bytes == 0 || config.max_rows == 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                "dataframe::HostBatchCursor::new",
                "host batch cursor bounds must be non-zero",
            )
            .with_detail("target_materialization_bytes", config.target_materialization_bytes.to_string())
            .with_detail("max_materialization_bytes", config.max_materialization_bytes.to_string())
            .with_detail("max_rows", config.max_rows.to_string()));
        }
        if config.target_materialization_bytes > config.max_materialization_bytes {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                "dataframe::HostBatchCursor::new",
                "host batch cursor soft materialization target must not exceed its hard materialization ceiling",
            )
            .with_detail("target_materialization_bytes", config.target_materialization_bytes.to_string())
            .with_detail("max_materialization_bytes", config.max_materialization_bytes.to_string()));
        }
        Ok(Self { batch, config, offset: 0, emitted_batches: 0, identity: Arc::new(()) })
    }

    pub fn schema(&self) -> &SchemaRef {
        self.batch.schema()
    }

    pub fn remaining_rows(&self) -> usize {
        self.batch.num_rows() - self.offset
    }

    /// Whether the cursor can still yield a host batch.
    ///
    /// This differs from `remaining_rows() > 0` for an initially empty result,
    /// which has no rows but still owns one typed materialization boundary.
    pub fn has_next_batch(&self) -> bool {
        self.remaining_rows() > 0 || (self.batch.num_rows() == 0 && self.emitted_batches == 0)
    }

    /// Select and size the next GPU chunk without copying values to host.
    pub fn prepare_next_batch(&self) -> Result<Option<PreparedHostBatch>> {
        if !self.has_next_batch() {
            return Ok(None);
        }
        let (rows, estimated_bytes) = if self.batch.num_rows() == 0 {
            self.batch.estimated_empty_materialization_bytes().map(|bytes| (0, bytes))
        } else {
            self.size_next_range()
        }
        .map_err(|error| error.reanchor("dataframe::HostBatchCursor::prepare_next_batch"))?;

        if estimated_bytes > self.config.max_materialization_bytes {
            return Err(Error::new_with(
                ErrorCode::HostMaterializationLimitExceeded,
                "dataframe::HostBatchCursor::prepare_next_batch",
                "unsplittable host output row exceeds the hard materialization ceiling",
            )
            .with_detail("reason", "host_materialization_hard_limit_exceeded")
            .with_detail("row_offset", self.offset.to_string())
            .with_detail("candidate_rows", rows.to_string())
            .with_detail("estimated_materialization_bytes", estimated_bytes.to_string())
            .with_detail("max_materialization_bytes", self.config.max_materialization_bytes.to_string()));
        }

        Ok(Some(PreparedHostBatch {
            cursor_identity: Arc::clone(&self.identity),
            offset: self.offset,
            rows,
            batch_index: self.emitted_batches,
            estimated_materialization_bytes: estimated_bytes,
        }))
    }

    /// Largest row count from the cursor offset whose estimate fits the soft
    /// target, at least one row, with that count's estimate.
    fn size_next_range(&self) -> Result<(usize, usize)> {
        let candidate_rows = self.remaining_rows().min(self.config.max_rows);
        let target = self.config.target_materialization_bytes;
        let sizer = self.batch.host_row_range_sizer(self.offset)?;
        // Rows past the lower-bound fit exceed the target whatever their
        // strings and lists hold, so their offsets are never copied to host.
        let bound = largest_rows_within(candidate_rows, target, |rows| sizer.lower_bound_bytes(rows))?.max(1);
        let range = sizer.load(bound)?;
        let rows = largest_rows_within(bound, target, |rows| range.bytes(rows))?.max(1);
        Ok((rows, range.bytes(rows)?))
    }

    /// Copy one previously prepared chunk to host and advance the cursor.
    pub fn materialize_prepared_batch(&mut self, prepared: PreparedHostBatch) -> Result<RecordBatch> {
        if !Arc::ptr_eq(&self.identity, &prepared.cursor_identity)
            || self.offset != prepared.offset
            || self.emitted_batches != prepared.batch_index
        {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                "dataframe::HostBatchCursor::materialize_prepared_batch",
                "prepared host batch does not belong to the cursor's current position",
            )
            .with_detail("reason", "stale_or_foreign_prepared_host_batch")
            .with_detail("cursor_offset", self.offset.to_string())
            .with_detail("prepared_offset", prepared.offset.to_string())
            .with_detail("cursor_batch_index", self.emitted_batches.to_string())
            .with_detail("prepared_batch_index", prepared.batch_index.to_string()));
        }

        let rows = prepared.rows;
        let next_offset = self.offset.checked_add(rows).ok_or_else(|| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "dataframe::HostBatchCursor::materialize_prepared_batch",
                "host batch cursor row offset overflowed usize",
            )
            .with_detail("cursor_offset", self.offset.to_string())
            .with_detail("materialized_rows", rows.to_string())
        })?;
        let next_batch_index = self.emitted_batches.checked_add(1).ok_or_else(|| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "dataframe::HostBatchCursor::materialize_prepared_batch",
                "host batch cursor batch index overflowed usize",
            )
            .with_detail("batch_index", self.emitted_batches.to_string())
        })?;
        let batch = self.batch.to_host_record_batch_rows(self.offset, rows)?;
        self.offset = next_offset;
        self.emitted_batches = next_batch_index;
        Ok(batch)
    }

    pub fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(prepared) = self.prepare_next_batch()? else {
            return Ok(None);
        };
        self.materialize_prepared_batch(prepared).map(Some)
    }
}

/// Largest row count in `1..=max_rows` whose estimate fits `target`, or 0
/// when one row does not. The estimate must not decrease as rows grow.
fn largest_rows_within(
    max_rows: usize,
    target: usize,
    mut estimate: impl FnMut(usize) -> Result<usize>,
) -> Result<usize> {
    if estimate(max_rows)? <= target {
        return Ok(max_rows);
    }
    let mut low = 1usize;
    let mut high = max_rows - 1;
    let mut best = 0usize;
    while low <= high {
        let middle = low + (high - low) / 2;
        if estimate(middle)? <= target {
            best = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    Ok(best)
}
