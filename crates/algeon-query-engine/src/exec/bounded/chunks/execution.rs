use parquet::file::reader::{FileReader, SerializedFileReader};
use query_runtime::observability::SourceReadFact;

use super::{
    BoundedExecutionAttempt, BoundedPlan, CommittedPartial, LocalParquetSource, Result,
    ScheduledLocalParquetReadOutput, SourceChunk, SourceChunkBatch, attach_chunk_resource_facts,
    attach_source_batch_resource_facts, observe_device_memory_bytes, read_local_parquet_chunk,
    read_local_parquet_chunk_batch_required_only_no_commit, source_adjacent::execute_source_adjacent_chunk_stages,
};

pub(in crate::exec::bounded) struct RawSourceInput {
    pub(in crate::exec::bounded) partial: CommittedPartial,
    pub(in crate::exec::bounded) source_row_group_count: u64,
    /// Rows decoded by the source read before source-adjacent stages ran.
    pub(in crate::exec::bounded) source_rows: u64,
    /// Device bytes of the decoded source frame before source-adjacent stages ran.
    pub(in crate::exec::bounded) source_bytes: u64,
}

/// Reads a source chunk with its pre-aggregate filter and projection stages
/// applied, without running a partial aggregate. Returns the raw filtered and
/// projected data with the represented source row-group count.
///
/// Used by the distinct aggregate path where all raw source data must be
/// concatenated before a single groupby can produce a correct COUNT(DISTINCT).
pub(in crate::exec::bounded) fn execute_bounded_raw_source_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
) -> Result<RawSourceInput> {
    execution.context().check_cancellation("query_engine.exec.bounded.raw_source_chunk")?;
    let source = read_bounded_chunk_source(execution, bounded.source, chunk)
        .map_err(|error| attach_chunk_resource_facts(error, chunk, "source_chunk_read", 1, 1, execution.metrics()))?;
    let source_row_group_count = reported_source_chunk_read_row_groups(chunk, source.observed_row_group_count);
    let source_rows = source.frame.num_rows() as u64;
    let source_bytes = source.observed_bytes;
    let input = CommittedPartial::with_device_bytes(source.frame, source.observed_bytes);
    let partial = execute_source_adjacent_chunk_stages(execution, bounded, chunk, input, 1, 1)?;
    Ok(RawSourceInput { partial, source_row_group_count, source_rows, source_bytes })
}

pub(in crate::exec::bounded) fn execute_bounded_raw_source_chunk_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: &SourceChunkBatch,
) -> Result<RawSourceInput> {
    execution.context().check_cancellation("query_engine.exec.bounded.raw_source_chunk_batch")?;
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let source =
        read_local_parquet_chunk_batch_required_only_no_commit(execution, bounded.source, batch, "source_chunk_read")
            .map_err(|error| {
            attach_source_batch_resource_facts(error, batch, "source_chunk_read", 1, 1, execution.metrics())
        })?;
    let source_row_group_count = reported_source_chunk_batch_read_row_groups(batch, source.observed_row_group_count);
    let source_rows = source.frame.num_rows() as u64;
    let source_bytes = source.observed_bytes;
    let input = CommittedPartial::with_device_bytes(source.frame, source.observed_bytes);
    let partial = execute_source_adjacent_chunk_stages(execution, bounded, first_chunk, input, 1, 1)?;
    Ok(RawSourceInput { partial, source_row_group_count, source_rows, source_bytes })
}

pub(in crate::exec::bounded::chunks) fn read_bounded_chunk_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<ScheduledLocalParquetReadOutput> {
    execution.context().check_cancellation("query_engine.exec.bounded.source_chunk_read")?;
    execution.record_source_read(SourceReadFact::ChunkAttempted);
    let source_token_budget = execution.device_budget_bytes();
    let frame = read_local_parquet_chunk(
        execution.context(),
        source,
        chunk,
        source_token_budget,
        "source_chunk_read",
        execution.metrics(),
    )?;
    let mut output = ScheduledLocalParquetReadOutput {
        chunk: chunk.clone(),
        frame,
        observed_bytes: 0,
        observed_row_group_count: source_chunk_static_read_row_groups(chunk),
    };
    output.observed_bytes = observe_device_memory_bytes(&output.frame, execution.metrics())?;
    Ok(output)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_static_read_row_groups(chunk: &SourceChunk) -> Option<u64> {
    if let Some(row_groups) = chunk.known_read_row_group_count() {
        return Some(row_groups as u64);
    }
    if chunk.reads_all_row_groups && chunk.path.is_file() {
        return SerializedFileReader::try_from(chunk.path.as_path())
            .ok()
            .map(|reader| reader.metadata().num_row_groups() as u64);
    }
    None
}

pub(in crate::exec::bounded::chunks) fn reported_source_chunk_read_row_groups(
    chunk: &SourceChunk,
    observed_row_group_count: Option<u64>,
) -> u64 {
    observed_row_group_count
        .or_else(|| source_chunk_static_read_row_groups(chunk))
        .unwrap_or_else(|| chunk.row_group_count() as u64)
}

pub(in crate::exec::bounded::chunks) fn reported_source_chunk_batch_read_row_groups(
    batch: &SourceChunkBatch,
    observed_row_group_count: Option<u64>,
) -> u64 {
    observed_row_group_count
        .or_else(|| source_chunk_batch_static_read_row_groups(batch))
        .unwrap_or_else(|| batch.row_group_count() as u64)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_batch_static_read_row_groups(
    batch: &SourceChunkBatch,
) -> Option<u64> {
    batch.chunks.iter().try_fold(0u64, |total, chunk| {
        source_chunk_static_read_row_groups(chunk).map(|count| total.saturating_add(count))
    })
}
