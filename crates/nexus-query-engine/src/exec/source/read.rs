use std::time::Instant;

use parquet::file::reader::FileReader;

use super::{
    DataFrame, ExecutionContext, LocalParquetFile, LocalParquetSource, NodeId, QueryError, QueryMetrics, Result,
    SourceReadMode, local_parquet_file_row_group_metadata_count, parquet_file_path_fact,
    record_source_file_read_runtime,
};

#[expect(
    clippy::too_many_arguments,
    reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
)]
pub(super) fn read_local_parquet_file(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    source_node_id: NodeId,
    file: &LocalParquetFile,
    file_index: usize,
    cudf_source: &cudf::io::source::LocalParquetSource,
    cudf_chunk: &cudf::io::source::SourceChunk,
    columns: &[&str],
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    context.check_cancellation("nexus_query_engine.local_parquet_source.read")?;
    let _io_lease = context.acquire_io_lease("nexus_query_engine.local_parquet_source.read")?;
    let read_timer = Instant::now();
    let output = DataFrame::read_parquet_source_chunk_with_evidence(
        &context.runtime()?,
        cudf_source,
        cudf_chunk,
        SourceReadMode::ExecutionFrame,
    )
    .map_err(|error| {
        read_cudf_error_for_file(
            "nexus_query_engine.local_parquet_source.read",
            "failed to read native local parquet source with cudf",
            error,
            file,
        )
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("read_columns", columns.len())
        .with_fact("row_groups_selected", file.selected_row_group_count())
    })?;
    let read_evidence = output.read_evidence().cloned();
    let read_elapsed = read_timer.elapsed();
    let frame = output.into_frame();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    metrics.parquet_files += 1;
    metrics.parquet_bytes_known += file.file_size_bytes.unwrap_or(0);
    metrics.row_groups_selected += local_file_selected_row_group_count(file);
    metrics.read_columns += columns.len() as u64;
    metrics.record_input_batch(frame.num_rows() as u64);
    record_source_file_read_runtime(
        source,
        source_node_id,
        file,
        file_index,
        file_index,
        local_file_selected_row_group_count(file),
        frame.num_rows() as u64,
        read_elapsed,
        read_evidence.as_ref(),
        metrics,
    );
    Ok(frame)
}

fn read_cudf_error_for_file(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    file: &LocalParquetFile,
) -> QueryError {
    if file.location.is_local_path() {
        QueryError::from_cudf_with_budget_details(operation, message, error)
    } else {
        QueryError::from_cudf_without_source_with_budget_details(operation, message, &error)
    }
}

fn local_file_selected_row_group_count(file: &LocalParquetFile) -> u64 {
    if let Some(row_groups) = &file.row_groups {
        return row_groups.len() as u64;
    }
    if let Some(count) = local_parquet_file_row_group_metadata_count(file) {
        return count as u64;
    }
    parquet::file::reader::SerializedFileReader::try_from(file.path.as_path())
        .ok()
        .map(|reader| reader.metadata().num_row_groups() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_direct_read_errors_redact_object_store_locations() {
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let file = LocalParquetFile::from_resolved_parquet_source(crate::source::ResolvedParquetSource::remote_kvikio(
            sensitive_uri,
            crate::source::CredentialMode::SharedProcessEnv,
        ))
        .expect("valid remote direct source");
        let cudf_error = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::IoFailure,
            "io::parquet::read_parquet_table",
            format!("failed to open {sensitive_uri}"),
        );

        let error = read_cudf_error_for_file(
            "nexus_query_engine.local_parquet_source.read",
            "failed to read native local parquet source with cudf",
            cudf_error,
            &file,
        )
        .with_fact("path", parquet_file_path_fact(&file));
        let rendered = format!("{error}");
        let debug = format!("{error:?}");

        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(error.source_error().is_none());
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(!rendered.contains(sensitive), "{sensitive} leaked in display error: {rendered}");
            assert!(!debug.contains(sensitive), "{sensitive} leaked in debug error: {debug}");
        }
    }
}
