//! Source chunk planning for local Parquet scans.
//!
//! Chunk plans split one native source node into file or row-group ranges that
//! can be scheduled under a bounded memory policy. Byte fields are device-side
//! planning estimates unless explicitly named as compressed Parquet bytes or
//! exact source-read preflight bytes.

use arrow::datatypes::DataType as ArrowDataType;

use crate::{
    backend::BackendCapabilities,
    error::{QueryError, QueryErrorCode, Result},
    source::{
        LocalParquetFile, LocalParquetSource, parquet_file_path_fact, parquet_file_stable_id_fact,
        parquet_source_path_fact, parquet_source_stable_id_fact,
    },
};

#[doc(hidden)]
pub mod assembly;
#[doc(hidden)]
pub mod decimal;
mod memory;
mod model;
mod planner;
mod policy;
#[doc(hidden)]
pub mod row_group;
#[doc(hidden)]
pub mod scan_limit;

pub use memory::SourceTaskMemoryModel;
pub use model::{SourceChunk, SourceChunkBatch, SourceChunkPlan};
pub use planner::{
    annotate_source_chunk_batching_reasons, plan_source_chunk_batches, plan_source_chunk_batches_with_row_cap,
};
pub use policy::{
    LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK, LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES,
    SourceChunkPlanningOptions, SourceChunkPlanningPolicy,
};

/// What the execution backend contributes to exact source-read sizing.
///
/// Chunk planning consumes these; it never derives a backend memory model of
/// its own. The minimum read budget is the backend reader's floor, and type
/// support decides whether a column has an exact fixed-width bound at all.
#[derive(Debug, Clone, Copy)]
pub struct SourceReadSizing<'a> {
    backend: &'a dyn BackendCapabilities,
    minimum_read_budget_bytes: u64,
}

impl<'a> SourceReadSizing<'a> {
    /// Binds chunk sizing to one backend's reader floor and type support.
    pub const fn new(backend: &'a dyn BackendCapabilities, minimum_read_budget_bytes: u64) -> Self {
        Self { backend, minimum_read_budget_bytes }
    }

    /// The backend reader's minimum source-read budget in bytes.
    pub(crate) const fn minimum_read_budget_bytes(&self) -> u64 {
        self.minimum_read_budget_bytes
    }

    /// Whether the backend can represent this Arrow type.
    pub(crate) fn supports_type(&self, data_type: &ArrowDataType) -> bool {
        self.backend.supports_type(data_type)
    }
}

/// Returns the source file referenced by `chunk` after validating that the
/// chunk's executable identity still matches the file at `file_index`.
#[doc(hidden)]
pub fn source_file_for_chunk<'a>(
    source: &'a LocalParquetSource,
    chunk: &SourceChunk,
    operation: &'static str,
) -> Result<&'a LocalParquetFile> {
    let file = source.files.get(chunk.file_index).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "local parquet source chunk references a missing source file",
        )
        .with_fact("reason", "source_chunk_file_index_missing")
        .with_fact("path", parquet_source_path_fact(&chunk.location, &chunk.path))
        .with_fact("stable_id", parquet_source_stable_id_fact(&chunk.location, chunk.stable_id.as_str()))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("source_files", source.files.len())
    })?;
    validate_source_chunk_identity(source, file, chunk, operation)?;
    Ok(file)
}

pub(super) fn validate_source_chunk_identity(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    chunk: &SourceChunk,
    operation: &'static str,
) -> Result<()> {
    if chunk.location != file.location {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "location"));
    }
    if chunk.path != file.path {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "reader_path"));
    }
    if chunk.resolved_source != file.resolved_source {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "resolved_source"));
    }
    if chunk.stable_id != file.stable_id {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "stable_id"));
    }
    if chunk.partition_values != file.partition_values {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "partition_values"));
    }
    if chunk.read_columns != source.read_columns {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "read_columns"));
    }
    Ok(())
}

pub(super) fn source_chunk_identity_mismatch_error(
    file: &LocalParquetFile,
    chunk: &SourceChunk,
    operation: &'static str,
    field: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        operation,
        "local parquet source chunk identity does not match its referenced source file",
    )
    .with_fact("reason", "source_chunk_identity_mismatch")
    .with_fact("field", field)
    .with_fact("path", parquet_source_path_fact(&chunk.location, &chunk.path))
    .with_fact("source_file_path", parquet_file_path_fact(file))
    .with_fact("stable_id", parquet_source_stable_id_fact(&chunk.location, chunk.stable_id.as_str()))
    .with_fact("source_file_stable_id", parquet_file_stable_id_fact(file))
    .with_fact("chunk_index", chunk.chunk_index)
    .with_fact("file_index", chunk.file_index)
}
