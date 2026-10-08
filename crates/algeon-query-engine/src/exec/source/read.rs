use std::{slice, time::Instant};

use cudf::{
    Exec,
    io::source::{LocalParquetSource as CudfLocalParquetSource, SourceChunk as CudfSourceChunk},
};
use parquet::file::reader::{FileReader, SerializedFileReader};

use super::{
    DataFrame, ExecutionContext, LocalParquetFile, LocalParquetSource, NodeId, QueryError, Result, SourceRead,
    SourceReadFact, SourceReadMode, cudf_query_error, cudf_query_error_without_source,
    local_parquet_file_row_group_metadata_count, parquet_file_path_fact, record_source_file_read_runtime,
    record_source_read,
};

/// Reads one whole file through the cuDF source contract and returns the execution frame
/// (`SourceReadMode::ExecutionFrame`, which includes pure-filter columns).
///
/// This is the unbudgeted whole-file path: the read runs with `Exec::RETAINED`, so its output is
/// not charged to an execution budget token, unlike the chunk read. The call holds an I/O lease
/// for the duration of the read; a server context rejects the lease once graceful shutdown has
/// begun. Cancellation is checked before the lease. A cuDF failure carries the file path and
/// selected row-group count, and the error chain is dropped for object-store files. Records the
/// read time, `ParquetSource` and input-batch facts, and one `file` metrics row.
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
    cudf_source: &CudfLocalParquetSource,
    cudf_chunk: &CudfSourceChunk,
    columns: &[&str],
) -> Result<DataFrame> {
    context.check_cancellation("query_engine.local_parquet_source.read")?;
    let _io_lease = context.acquire_io_lease("query_engine.local_parquet_source.read")?;
    let read_timer = Instant::now();
    let output = DataFrame::read_parquet_source(
        SourceRead::new(cudf_source, slice::from_ref(cudf_chunk), SourceReadMode::ExecutionFrame),
        &context.runtime()?,
        Exec::RETAINED,
    )
    .map_err(|error| {
        read_cudf_error_for_file(
            "query_engine.local_parquet_source.read",
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
    record_source_read(context, SourceReadFact::ParquetReadTime(read_elapsed));
    record_source_read(context, SourceReadFact::SourceReadTime(read_elapsed));
    record_source_read(
        context,
        SourceReadFact::ParquetSource {
            files: 1,
            known_bytes: file.file_size_bytes.unwrap_or(0),
            selected_row_groups: local_file_selected_row_group_count(file),
            read_columns: columns.len() as u64,
        },
    );
    record_source_read(context, SourceReadFact::InputBatch { rows: frame.num_rows() as u64 });
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
        context,
    );
    Ok(frame)
}

/// Same redaction rule as `cudf_contract::error`: the cuDF error chain is kept only for local
/// paths, because for object-store files it can contain the object URI.
fn read_cudf_error_for_file(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    file: &LocalParquetFile,
) -> QueryError {
    if file.location.is_local_path() {
        cudf_query_error(operation, message, error)
    } else {
        cudf_query_error_without_source(operation, message, &error)
    }
}

/// Row groups a whole-file read covers: the explicit selection, else the count in the plan's
/// metadata, else the count in the local file footer. Reports 0 when none of these is available,
/// for example an object-store file without recorded metadata.
fn local_file_selected_row_group_count(file: &LocalParquetFile) -> u64 {
    if let Some(row_groups) = &file.row_groups {
        return row_groups.len() as u64;
    }
    if let Some(count) = local_parquet_file_row_group_metadata_count(file) {
        return count as u64;
    }
    SerializedFileReader::try_from(file.path.as_path())
        .ok()
        .map(|reader| reader.metadata().num_row_groups() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use cudf::{ErrorCode, testing::error::new_with_for_test};
    use query_planning::source::{CredentialMode, ResolvedParquetSource};

    use super::*;

    #[test]
    fn remote_direct_read_errors_redact_object_store_locations() {
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let file = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
            sensitive_uri,
            CredentialMode::SharedProcessEnv,
        ))
        .expect("valid remote direct source");
        let cudf_error = new_with_for_test(
            ErrorCode::IoFailure,
            "io::parquet::read_parquet_table",
            format!("failed to open {sensitive_uri}"),
        );

        let error = read_cudf_error_for_file(
            "query_engine.local_parquet_source.read",
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
