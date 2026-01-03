use super::*;

pub(in crate::exec) fn record_bounded_parquet_source_metrics(
    source: &LocalParquetSource,
    chunk_plan: &SourceChunkPlan,
    metrics: &mut QueryMetrics,
) {
    let known_file_bytes = source.files.iter().filter_map(|file| file.file_size_bytes).sum::<u64>();
    let planned_chunk_bytes = chunk_plan
        .chunks
        .iter()
        .map(|chunk| {
            chunk.known_row_group_bytes.or(chunk.estimated_chunk_bytes).or(chunk.known_file_bytes).unwrap_or(0)
        })
        .sum::<u64>();

    metrics.parquet_files = metrics.parquet_files.saturating_add(source.file_count() as u64);
    metrics.parquet_bytes_known = metrics.parquet_bytes_known.saturating_add(known_file_bytes.max(planned_chunk_bytes));
    metrics.row_groups_selected = metrics.row_groups_selected.saturating_add(chunk_plan.row_group_count() as u64);
    metrics.read_columns =
        metrics.read_columns.saturating_add((source.file_count() * source.read_column_count()) as u64);
}

pub(in crate::exec) fn format_row_groups(chunk: &SourceChunk) -> String {
    if chunk.reads_all_row_groups {
        return "all".to_string();
    }

    comma_separated_usize(&chunk.row_group_indexes)
}
