use std::time::Duration;

use super::{object_store_fetch::HostParquetBuffer, *};
use crate::parquet_metadata::parse_complete_parquet_metadata;

#[derive(Debug, Clone, Default)]
pub(in crate::exec::source) struct HostBufferFetchRuntimeMetrics {
    pub(in crate::exec::source) fetch_mode: String,
    pub(in crate::exec::source) payload_bytes: u64,
    pub(in crate::exec::source) requested_range_count: u64,
    pub(in crate::exec::source) requested_range_bytes: u64,
    pub(in crate::exec::source) original_file_bytes: u64,
    pub(in crate::exec::source) mini_parquet_footer_bytes: u64,
}

pub(super) fn parquet_file_path_fact(file: &LocalParquetFile) -> String {
    file.location.diagnostic_location()
}

pub(super) fn parquet_chunk_path_fact(chunk: &SourceChunk) -> String {
    chunk.location.diagnostic_location()
}

pub(super) fn parquet_file_stable_id_fact(file: &LocalParquetFile) -> String {
    file.location.diagnostic_stable_id(file.stable_id.as_str())
}

pub(super) fn parquet_chunk_stable_id_fact(chunk: &SourceChunk) -> String {
    chunk.location.diagnostic_stable_id(chunk.stable_id.as_str())
}

pub(super) fn record_host_buffer_file_read_metrics(
    source: &LocalParquetSource,
    source_node_id: NodeId,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    fetch: &HostParquetBuffer,
    output_rows: u64,
    decode_elapsed: Duration,
    read_evidence: Option<&cudf::io::parquet::ParquetReadEvidence>,
    metrics: &mut QueryMetrics,
) {
    let observed_row_group_count = fetched_file_row_group_count(file, fetch);
    metrics.record_source_decode_time(decode_elapsed);
    metrics.parquet_read_time += decode_elapsed;
    metrics.source_read_time += fetch.fetch_elapsed;
    metrics.source_read_time += decode_elapsed;
    metrics.parquet_files += 1;
    metrics.parquet_bytes_known += fetch.actual_bytes;
    metrics.row_groups_selected += observed_row_group_count;
    metrics.read_columns += source.read_columns.len() as u64;
    metrics.record_input_batch(output_rows);
    let host_buffer_fetch = fetch.runtime_metrics();
    record_source_file_read_runtime(
        source,
        source_node_id,
        file,
        file_index,
        chunk_index,
        observed_row_group_count,
        output_rows,
        decode_elapsed,
        &host_buffer_fetch,
        read_evidence,
        metrics,
    );
}

pub(super) fn record_host_buffer_chunk_read_metrics(
    chunk: &SourceChunk,
    fetch: &HostParquetBuffer,
    decode_elapsed: Duration,
    metrics: &mut QueryMetrics,
) -> Option<u64> {
    metrics.record_source_decode_time(decode_elapsed);
    let observed_row_group_count = fetched_chunk_row_group_count(chunk, fetch);
    if chunk.reads_all_row_groups
        && chunk.row_group_indexes.is_empty()
        && let Some(row_group_count) = observed_row_group_count
    {
        metrics.row_groups_selected += row_group_count;
    }
    observed_row_group_count
}

fn fetched_file_row_group_count(file: &LocalParquetFile, fetch: &HostParquetBuffer) -> u64 {
    if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
        return ranged_fetch.selected_original_row_groups.len() as u64;
    }
    if let Some(row_groups) = &file.row_groups {
        return row_groups.len() as u64;
    }
    super::represented_file_row_group_count(file)
        .or_else(|| {
            parse_complete_parquet_metadata(
                fetch.bytes(),
                "nexus_query_engine.source_decode.host_buffer_file_metadata",
                parquet_file_path_fact(file),
            )
            .ok()
            .map(|metadata| metadata.num_row_groups())
        })
        .unwrap_or(0) as u64
}

fn fetched_chunk_row_group_count(chunk: &SourceChunk, fetch: &HostParquetBuffer) -> Option<u64> {
    if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
        return Some(ranged_fetch.selected_original_row_groups.len() as u64);
    }
    chunk.known_read_row_group_count().map(|count| count as u64).or_else(|| {
        parse_complete_parquet_metadata(
            fetch.bytes(),
            "nexus_query_engine.source_decode.host_buffer_chunk_metadata",
            parquet_chunk_path_fact(chunk),
        )
        .ok()
        .map(|metadata| metadata.num_row_groups() as u64)
    })
}
