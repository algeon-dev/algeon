//! Bounded incremental GPU-to-host Arrow materialization cursor.
use std::sync::Arc;

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use super::GpuRecordBatch;
use crate::{
    Exec,
    core::error::{Error, ErrorCode, Result},
};

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
    pub const DEFAULT_MAX_ROWS: usize = 16 * 1024;
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

/// A metadata-sized GPU slice ready for one bounded host copy.
///
/// Preparing a step performs no device-to-host value materialization. The
/// token keeps the exact sliced GPU candidate alive so a caller can reserve
/// host capacity before asking the originating cursor to copy it.
#[derive(Debug)]
pub struct PreparedHostBatch {
    batch: GpuRecordBatch,
    cursor_identity: Arc<()>,
    offset: usize,
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
        self.batch.num_rows()
    }
}

/// Incremental GPU-to-host Arrow materializer.
///
/// The cursor retains the original GPU batch and at most one sliced GPU chunk.
/// An initially empty GPU batch still yields exactly one schema-preserving host
/// batch before reaching EOF, matching a full result materialization boundary.
/// It uses a schema-aware upper bound for final Arrow buffer capacities to
/// binary-reduce a candidate before copying it to host. A single row may exceed
/// the soft target, but an unsplittable row that exceeds the hard
/// materialization ceiling is rejected with a permanent structured resource
/// error before any host value copy.
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
        let remaining = self.remaining_rows();
        let initial_empty_batch = self.batch.num_rows() == 0 && self.emitted_batches == 0;
        if !self.has_next_batch() {
            return Ok(None);
        }
        let candidate_rows = if initial_empty_batch { 0 } else { remaining.min(self.config.max_rows) };
        let mut candidate = self.candidate(candidate_rows)?;
        let mut estimated_bytes = candidate.estimated_materialization_bytes()?;
        if candidate_rows > 1 && estimated_bytes > self.config.target_materialization_bytes {
            // Release the oversized GPU slice before allocating binary-search probes.
            drop(candidate);
            let mut low = 1usize;
            let mut high = candidate_rows - 1;
            let mut best_rows = 1usize;
            while low <= high {
                let middle = low + (high - low) / 2;
                let probe = self.candidate(middle)?;
                let estimated_bytes = probe.estimated_materialization_bytes()?;
                // Release each GPU slice before allocating the next probe.
                drop(probe);
                if estimated_bytes <= self.config.target_materialization_bytes {
                    best_rows = middle;
                    low = middle + 1;
                } else {
                    high = middle - 1;
                }
            }
            candidate = self.candidate(best_rows)?;
            estimated_bytes = candidate.estimated_materialization_bytes()?;
        }

        if estimated_bytes > self.config.max_materialization_bytes {
            return Err(Error::new_with(
                ErrorCode::HostMaterializationLimitExceeded,
                "dataframe::HostBatchCursor::prepare_next_batch",
                "unsplittable host output row exceeds the hard materialization ceiling",
            )
            .with_detail("reason", "host_materialization_hard_limit_exceeded")
            .with_detail("row_offset", self.offset.to_string())
            .with_detail("candidate_rows", candidate.num_rows().to_string())
            .with_detail("estimated_materialization_bytes", estimated_bytes.to_string())
            .with_detail("max_materialization_bytes", self.config.max_materialization_bytes.to_string()));
        }

        Ok(Some(PreparedHostBatch {
            batch: candidate,
            cursor_identity: Arc::clone(&self.identity),
            offset: self.offset,
            batch_index: self.emitted_batches,
            estimated_materialization_bytes: estimated_bytes,
        }))
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

        let rows = prepared.batch.num_rows();
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
        let batch = prepared.batch.to_host_record_batch()?;
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

    fn candidate(&self, rows: usize) -> Result<GpuRecordBatch> {
        if rows == 0 {
            debug_assert_eq!(self.batch.num_rows(), 0);
            return Ok(self.batch.clone());
        }
        let frame = self
            .batch
            .frame()
            .on(Exec::RETAINED)
            .slice(self.offset, rows)
            .map_err(|error| error.reanchor("dataframe::HostBatchCursor::next_batch"))?;
        self.batch.with_validated_slice(frame)
    }
}
