use parquet::file::reader::{FileReader, SerializedFileReader};
use query_runtime::observability::SourceReadFact;

use super::{
    BoundedExecutionAttempt, BoundedPlan, CommittedPartial, LocalParquetSource, Result,
    ScheduledLocalParquetReadOutput, SourceChunk, SourceChunkBatch, attach_chunk_resource_facts,
    attach_source_batch_resource_facts, observe_device_memory_bytes, read_local_parquet_chunk,
    read_local_parquet_chunk_batch_required_only_no_commit, source_adjacent::execute_source_adjacent_chunk_stages,
};

/// One chunk's frame after the source-adjacent stages, with the figures of
/// the decoded source frame that stage metrics compare against.
pub(in crate::exec::bounded) struct RawSourceInput {
    pub(in crate::exec::bounded) partial: CommittedPartial,
    /// Row groups the read represented. Prefers what the reader observed over
    /// the planned count, which can differ for a chunk reading a whole file.
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
/// `attempt` and `max_attempts` are the caller's retry position for this
/// chunk; they only label the facts of a failed read or stage.
pub(in crate::exec::bounded) fn execute_bounded_raw_source_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> Result<RawSourceInput> {
    execution.context().check_cancellation("query_engine.exec.bounded.raw_source_chunk")?;
    let source = read_bounded_chunk_source(execution, bounded.source, chunk).map_err(|error| {
        attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, execution.metrics())
    })?;
    let source_row_group_count = reported_source_chunk_read_row_groups(chunk, source.observed_row_group_count);
    let source_rows = source.frame.num_rows() as u64;
    let source_bytes = source.observed_bytes;
    let input = CommittedPartial::with_device_bytes(source.frame, source.observed_bytes);
    let partial = execute_source_adjacent_chunk_stages(execution, bounded, chunk, input, attempt, max_attempts)?;
    Ok(RawSourceInput { partial, source_row_group_count, source_rows, source_bytes })
}

/// Multi-chunk form of `execute_bounded_raw_source_chunk`. A batch is not
/// split by row group, so its facts always report the first attempt.
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

/// Decodes one chunk and observes its device bytes. Records the attempt fact
/// only; the caller records the commit once the output has survived its
/// stages and retries, because a chunk that fails later is read again.
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

/// Row groups a chunk will read, known without decoding. A chunk that reads a
/// whole local file has no explicit index list, so the count comes from the
/// file footer; that is a host file open on the executing thread, and a
/// footer error yields `None` while the read itself proceeds.
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
