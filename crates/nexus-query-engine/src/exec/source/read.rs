use parquet::file::reader::FileReader;

use super::*;

pub(super) fn read_local_parquet_file(
    context: &ExecutionContext,
    file: &LocalParquetFile,
    columns: &[&str],
    predicate: Option<Arc<cudf::ops::ast::AstExpression>>,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    context.check_cancellation("nexus_query_engine.local_parquet_source.read")?;
    let row_groups = cudf_row_groups(file)?;
    let read_timer = Instant::now();
    let options = match (&row_groups, predicate) {
        (Some(row_groups), Some(predicate)) => ParquetReadOptions::builder()
            .filepath(&file.path)
            .columns(columns)
            .row_groups(row_groups)
            .predicate(predicate)
            .with_runtime(context.runtime())
            .build(),
        (Some(row_groups), None) => ParquetReadOptions::builder()
            .filepath(&file.path)
            .columns(columns)
            .row_groups(row_groups)
            .with_runtime(context.runtime())
            .build(),
        (None, Some(predicate)) => ParquetReadOptions::builder()
            .filepath(&file.path)
            .columns(columns)
            .predicate(predicate)
            .with_runtime(context.runtime())
            .build(),
        (None, None) => {
            ParquetReadOptions::builder().filepath(&file.path).columns(columns).with_runtime(context.runtime()).build()
        },
    };

    let frame = DataFrame::read_parquet_with_options(options).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.read",
            "failed to read native local parquet source with cudf",
            error,
        )
        .with_fact("path", file.path.display())
        .with_fact("read_columns", columns.len())
        .with_fact("row_groups_selected", file.selected_row_group_count())
    })?;
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    metrics.parquet_files += 1;
    metrics.parquet_bytes_known += file.file_size_bytes.unwrap_or(0);
    metrics.row_groups_selected += local_file_selected_row_group_count(file);
    metrics.read_columns += columns.len() as u64;
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(frame)
}

fn local_file_selected_row_group_count(file: &LocalParquetFile) -> u64 {
    if let Some(row_groups) = &file.row_groups {
        return row_groups.len() as u64;
    }
    if let Some(count) = represented_file_row_group_count(file) {
        return count as u64;
    }
    parquet::file::reader::SerializedFileReader::try_from(file.path.as_path())
        .ok()
        .map(|reader| reader.metadata().num_row_groups() as u64)
        .unwrap_or(0)
}

fn represented_file_row_group_count(file: &LocalParquetFile) -> Option<usize> {
    file.row_group_bytes
        .as_ref()
        .or(file.projected_row_group_uncompressed_bytes.as_ref())
        .or(file.projected_row_group_compressed_bytes.as_ref())
        .or(file.pure_filter_row_group_uncompressed_bytes.as_ref())
        .or(file.pure_filter_row_group_compressed_bytes.as_ref())
        .or(file.row_group_rows.as_ref())
        .map(Vec::len)
}

pub(super) fn cudf_row_groups(file: &LocalParquetFile) -> Result<Option<Vec<i32>>> {
    let Some(row_groups) = &file.row_groups else {
        return Ok(None);
    };

    if row_groups.is_empty() {
        return Ok(None);
    }

    row_groups
        .iter()
        .map(|row_group| {
            i32::try_from(*row_group).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.row_groups",
                    "local parquet row group index exceeds the cudf reader range",
                )
                .with_fact("path", file.path.display())
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", i32::MAX)
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}
