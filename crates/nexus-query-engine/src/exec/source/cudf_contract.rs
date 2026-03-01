use std::path::{Path, PathBuf};

use super::{
    predicate::{lower_source_reader_batch_predicate, lower_source_reader_predicate},
    *,
};

/// Explicit datasource-level probe facts for a resolved remote Parquet source.
///
/// These facts come from opening libcudf's path datasource. They are not
/// per-read Parquet evidence and must not be used to satisfy
/// [`crate::source::RemoteParquetReadMode::RequireDirect`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRemoteParquetDatasourceProbe {
    /// Whether the opened datasource reports device-read support.
    pub supports_device_read: bool,
    /// Whether the opened datasource prefers device reads for the requested size.
    pub device_read_preferred: bool,
    /// Datasource size reported by libcudf, when available.
    pub size_bytes: Option<u64>,
    /// cuDF/KvikIO remote backend build status.
    pub backend_build_status: String,
    /// Conservative typed-source capability disposition at the cudf-rs boundary.
    pub source_capability_disposition: String,
    /// URI scheme only, never the full URI.
    pub uri_scheme: String,
    /// Selected Nexus read strategy.
    pub read_strategy: String,
    /// Requested remote read mode.
    pub remote_read_mode: String,
    /// Credential mode selected by the resolved source.
    pub credential_mode: String,
    /// Whether this probe supplies per-read backend/byte evidence.
    pub per_read_evidence_available: bool,
}

/// Probe libcudf datasource capability for an explicitly resolved remote
/// path-based Parquet source.
///
/// This API may contact object-store metadata services while opening the
/// datasource. It is intentionally not used by default source planning, and it
/// rejects `RequireDirect` because datasource capability is weaker than the
/// per-read backend/range evidence required by that mode.
pub fn probe_resolved_remote_parquet_datasource(
    context: &ExecutionContext,
    resolved_source: &crate::source::ResolvedParquetSource,
    preferred_read_size_bytes: Option<u64>,
) -> Result<ResolvedRemoteParquetDatasourceProbe> {
    const OPERATION: &str = "nexus_query_engine.remote_parquet_datasource.probe";
    validate_remote_parquet_datasource_probe_request(resolved_source, OPERATION)?;
    let remote_source = cudf_remote_parquet_source_for_resolved_source(context, resolved_source, OPERATION)?;
    let source_capability = cudf::io::parquet::remote_parquet_capability(&remote_source);
    let probe = cudf::io::parquet::probe_remote_parquet_datasource(&remote_source, preferred_read_size_bytes).map_err(
        |error| {
            QueryError::from_cudf_without_source(
                OPERATION,
                "failed to probe resolved remote Parquet datasource capability",
                &error,
            )
            .with_fact("resource", "remote_parquet_datasource")
            .with_fact("read_strategy", resolved_source.read_strategy.as_str())
            .with_fact("remote_read_mode", resolved_source.remote_read_mode.as_str())
            .with_fact("credential_mode", resolved_source.credential_mode.as_str())
            .with_fact("uri_scheme", source_capability.uri_scheme.as_str())
        },
    )?;
    Ok(ResolvedRemoteParquetDatasourceProbe {
        supports_device_read: probe.supports_device_read,
        device_read_preferred: probe.device_read_preferred,
        size_bytes: probe.size_bytes,
        backend_build_status: probe.backend_build_status.as_str().to_string(),
        source_capability_disposition: remote_parquet_capability_disposition_label(source_capability.disposition)
            .to_string(),
        uri_scheme: source_capability.uri_scheme,
        read_strategy: resolved_source.read_strategy.as_str().to_string(),
        remote_read_mode: resolved_source.remote_read_mode.as_str().to_string(),
        credential_mode: resolved_source.credential_mode.as_str().to_string(),
        per_read_evidence_available: false,
    })
}

pub(super) fn cudf_local_parquet_source_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<(cudf::io::source::LocalParquetSource, cudf::io::source::SourceChunk)> {
    let verified_chunk;
    let chunk =
        if let Some(reader_decimal_types) = footer_verified_reader_decimal_types_for_chunk(context, source, chunk)? {
            verified_chunk = chunk_with_reader_decimal_types(chunk, reader_decimal_types);
            &verified_chunk
        } else {
            chunk
        };
    let cudf_source = cudf_local_parquet_source_for_chunk(context, source, chunk)?;
    let row_groups = if chunk.reads_all_row_groups { None } else { Some(chunk.row_group_indexes.clone()) };
    let cudf_chunk =
        cudf::io::source::SourceChunk::from_source_file_selection(&cudf_source, 0, chunk.chunk_index, row_groups)
            .map_err(|error| {
                cudf_error_for_chunk(
                    "nexus_query_engine.local_parquet_chunk.descriptor_cudf",
                    "failed to convert native local parquet source chunk descriptor into cudf source chunk",
                    error,
                    chunk,
                )
                .with_fact("path", chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("read_columns", chunk.read_columns.len())
                .with_fact("row_groups", format_row_groups(chunk))
            })?;
    let cudf_source = attach_cudf_source_reader_predicate(
        context,
        cudf_source,
        &cudf_chunk,
        source,
        "nexus_query_engine.local_parquet_chunk.source_cudf",
        chunk.path.as_path(),
        chunk.chunk_index,
        chunk.file_index,
        chunk.read_columns.len(),
        Some(format_row_groups(chunk)),
    )?;
    Ok((cudf_source, cudf_chunk))
}

pub(super) fn cudf_local_parquet_source_chunk_batch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
) -> Result<(cudf::io::source::LocalParquetSource, Vec<cudf::io::source::SourceChunk>)> {
    if chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_chunk_batch.source_cudf",
            "local parquet source chunk batch must not be empty",
        ));
    }
    let mut metadata_cache = crate::chunk::SourceMetadataCache::default();
    let chunks = chunks
        .iter()
        .map(|chunk| {
            footer_verified_reader_decimal_types_for_chunk_with_cache(context, source, chunk, &mut metadata_cache)?
                .map(|reader_decimal_types| chunk_with_reader_decimal_types(chunk, reader_decimal_types))
                .map_or_else(|| Ok(chunk.clone()), Ok)
        })
        .collect::<Result<Vec<_>>>()?;
    let cudf_files =
        chunks.iter().map(|chunk| cudf_local_parquet_file_for_chunk(context, chunk)).collect::<Result<Vec<_>>>()?;
    let mut cudf_source = cudf_local_parquet_source_from_files(
        source,
        cudf_files,
        chunks[0].read_columns.iter().map(String::as_str),
        "nexus_query_engine.local_parquet_chunk_batch.source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("chunk_count", chunks.len())
            .with_fact("first_chunk_index", chunks[0].chunk_index)
            .with_fact("first_file_index", chunks[0].file_index)
    })?;
    let cudf_chunks = chunks
        .iter()
        .enumerate()
        .map(|(batch_file_index, chunk)| {
            let row_groups = if chunk.reads_all_row_groups { None } else { Some(chunk.row_group_indexes.clone()) };
            cudf::io::source::SourceChunk::from_source_file_selection(
                &cudf_source,
                batch_file_index,
                chunk.chunk_index,
                row_groups,
            )
            .map_err(|error| {
                cudf_error_for_chunk(
                    "nexus_query_engine.local_parquet_chunk_batch.descriptor_cudf",
                    "failed to convert native local parquet source chunk batch descriptor into cudf source chunk",
                    error,
                    chunk,
                )
                .with_fact("path", chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("batch_file_index", batch_file_index)
                .with_fact("read_columns", chunk.read_columns.len())
                .with_fact("row_groups", format_row_groups(chunk))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    cudf_source = attach_cudf_source_reader_batch_predicate(
        context,
        cudf_source,
        &cudf_chunks,
        source,
        "nexus_query_engine.local_parquet_chunk_batch.source_cudf",
        chunks[0].path.as_path(),
        chunks[0].chunk_index,
        chunks[0].file_index,
        chunks[0].read_columns.len(),
        Some(format!("batch:{}", chunks.len())),
    )?;
    Ok((cudf_source, cudf_chunks))
}

pub(super) fn cudf_local_parquet_source_file_selection(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
) -> Result<(cudf::io::source::LocalParquetSource, cudf::io::source::SourceChunk)> {
    let verified_reader_decimal_types =
        crate::chunk::footer_verified_object_store_reader_decimal_types(context, source, file)?;
    let verified_file;
    let file = if let Some(reader_decimal_types) = verified_reader_decimal_types {
        verified_file = file.clone().with_reader_decimal_types(reader_decimal_types);
        &verified_file
    } else {
        file
    };
    let cudf_source = cudf_local_parquet_source_for_file(context, source, file, file_index, chunk_index)?;
    let row_groups = file.row_groups.clone();
    let cudf_chunk =
        cudf::io::source::SourceChunk::from_source_file_selection(&cudf_source, 0, chunk_index, row_groups).map_err(
            |error| {
                cudf_error_for_file(
                    "nexus_query_engine.local_parquet_source.descriptor_cudf",
                    "failed to convert native local parquet source file descriptor into cudf source chunk",
                    error,
                    file,
                )
                .with_fact("path", file_path_fact(file))
                .with_fact("chunk_index", chunk_index)
                .with_fact("file_index", file_index)
                .with_fact("read_columns", source.read_columns.len())
            },
        )?;
    let cudf_source = attach_cudf_source_reader_predicate(
        context,
        cudf_source,
        &cudf_chunk,
        source,
        "nexus_query_engine.local_parquet_source.source_cudf",
        file.path.as_path(),
        chunk_index,
        file_index,
        source.read_columns.len(),
        None,
    )?;
    Ok((cudf_source, cudf_chunk))
}

fn footer_verified_reader_decimal_types_for_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<Option<Vec<crate::source::ParquetReaderDecimalType>>> {
    let mut metadata_cache = crate::chunk::SourceMetadataCache::default();
    footer_verified_reader_decimal_types_for_chunk_with_cache(context, source, chunk, &mut metadata_cache)
}

fn footer_verified_reader_decimal_types_for_chunk_with_cache(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    metadata_cache: &mut crate::chunk::SourceMetadataCache,
) -> Result<Option<Vec<crate::source::ParquetReaderDecimalType>>> {
    let file = crate::chunk::source_file_for_chunk(
        source,
        chunk,
        "nexus_query_engine.local_parquet_chunk.reader_decimal_type_metadata",
    )?;
    if chunk.reader_decimal_types_footer_verified
        && !chunk.reader_decimal_types.is_empty()
        && crate::chunk::object_store_decimal_reader_predicate_requires_footer_contract(source, file)
    {
        return Ok(Some(chunk.reader_decimal_types.clone()));
    }
    crate::chunk::footer_verified_object_store_reader_decimal_types_with_supplied_cache(
        context,
        source,
        file,
        &chunk.reader_decimal_types,
        metadata_cache,
    )
}

fn chunk_with_reader_decimal_types(
    chunk: &SourceChunk,
    reader_decimal_types: Vec<crate::source::ParquetReaderDecimalType>,
) -> SourceChunk {
    let mut chunk = chunk.clone();
    chunk.reader_decimal_types = reader_decimal_types;
    chunk.reader_decimal_types_footer_verified = true;
    chunk
}

pub(super) struct MiniParquetHostBufferDescriptor<'a> {
    path: &'a Path,
    location: &'a crate::source::ParquetSourceLocation,
    stable_id: &'a str,
    partition_values: &'a [LiteralValue],
    read_columns: &'a [String],
    reader_decimal_types: MiniParquetHostBufferReaderDecimalTypes,
    file_index: usize,
    chunk_index: usize,
    pub(super) mini_file_size_bytes: u64,
    pub(super) mini_row_group_bytes: &'a [u64],
    pub(super) mini_row_group_rows: &'a [u64],
}

pub(super) struct MiniParquetHostBufferReaderDecimalTypes(Vec<crate::source::ParquetReaderDecimalType>);

impl MiniParquetHostBufferReaderDecimalTypes {
    fn as_slice(&self) -> &[crate::source::ParquetReaderDecimalType] {
        &self.0
    }

    #[cfg(test)]
    fn from_footer_verified_for_test(reader_decimal_types: Vec<crate::source::ParquetReaderDecimalType>) -> Self {
        Self(reader_decimal_types)
    }
}

pub(super) fn mini_host_buffer_reader_decimal_types_for_file(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> Result<MiniParquetHostBufferReaderDecimalTypes> {
    let reader_decimal_types = crate::chunk::footer_verified_object_store_reader_decimal_types(context, source, file)?
        .unwrap_or_else(|| file.reader_decimal_types.clone());
    Ok(MiniParquetHostBufferReaderDecimalTypes(reader_decimal_types))
}

pub(super) fn mini_host_buffer_reader_decimal_types_for_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<MiniParquetHostBufferReaderDecimalTypes> {
    let reader_decimal_types = footer_verified_reader_decimal_types_for_chunk(context, source, chunk)?
        .unwrap_or_else(|| chunk.reader_decimal_types.clone());
    Ok(MiniParquetHostBufferReaderDecimalTypes(reader_decimal_types))
}

impl<'a> MiniParquetHostBufferDescriptor<'a> {
    pub(super) fn from_chunk(
        chunk: &'a SourceChunk,
        mini_file_size_bytes: u64,
        mini_row_group_bytes: &'a [u64],
        mini_row_group_rows: &'a [u64],
        reader_decimal_types: MiniParquetHostBufferReaderDecimalTypes,
    ) -> Self {
        Self {
            path: chunk.path.as_path(),
            location: &chunk.location,
            stable_id: chunk.stable_id.as_str(),
            partition_values: &chunk.partition_values,
            read_columns: &chunk.read_columns,
            reader_decimal_types,
            file_index: chunk.file_index,
            chunk_index: chunk.chunk_index,
            mini_file_size_bytes,
            mini_row_group_bytes,
            mini_row_group_rows,
        }
    }

    pub(super) fn from_file(
        file: &'a LocalParquetFile,
        source: &'a LocalParquetSource,
        file_index: usize,
        chunk_index: usize,
        mini_file_size_bytes: u64,
        mini_row_group_bytes: &'a [u64],
        mini_row_group_rows: &'a [u64],
        reader_decimal_types: MiniParquetHostBufferReaderDecimalTypes,
    ) -> Self {
        Self {
            path: file.path.as_path(),
            location: &file.location,
            stable_id: file.stable_id.as_str(),
            partition_values: &file.partition_values,
            read_columns: &source.read_columns,
            reader_decimal_types,
            file_index,
            chunk_index,
            mini_file_size_bytes,
            mini_row_group_bytes,
            mini_row_group_rows,
        }
    }
}

pub(super) fn cudf_local_parquet_source_mini_host_buffer_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    descriptor: MiniParquetHostBufferDescriptor<'_>,
) -> Result<(cudf::io::source::LocalParquetSource, cudf::io::source::SourceChunk)> {
    let partition_values = descriptor.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mini_path = mini_host_buffer_path(descriptor.file_index, descriptor.chunk_index);
    let mini_stable_id =
        format!("{}#mini:{}", descriptor.location.diagnostic_stable_id(descriptor.stable_id), descriptor.chunk_index);
    let mut cudf_file = cudf::io::source::LocalParquetFile::new(&mini_path, mini_stable_id.as_str(), partition_values)
        .map_err(|error| {
            cudf_error_for_location(
                descriptor.location,
                "nexus_query_engine.local_parquet_chunk.mini_host_buffer_file_cudf",
                "failed to build mini-Parquet host-buffer file descriptor",
                error,
            )
            .with_fact("path", path_fact(descriptor.path))
            .with_fact("chunk_index", descriptor.chunk_index)
            .with_fact("file_index", descriptor.file_index)
        })?
        .with_file_size_bytes(descriptor.mini_file_size_bytes)
        .with_row_group_metadata(descriptor.mini_row_group_bytes.to_vec(), descriptor.mini_row_group_rows.to_vec())
        .map_err(|error| {
            cudf_error_for_location(
                descriptor.location,
                "nexus_query_engine.local_parquet_chunk.mini_host_buffer_file_cudf",
                "failed to attach mini-Parquet row-group metadata to cudf source contract",
                error,
            )
            .with_fact("path", path_fact(descriptor.path))
            .with_fact("chunk_index", descriptor.chunk_index)
            .with_fact("file_index", descriptor.file_index)
        })?;
    cudf_file = attach_cudf_reader_decimal_types(
        cudf_file,
        descriptor.reader_decimal_types.as_slice(),
        "nexus_query_engine.local_parquet_chunk.mini_host_buffer_file_cudf",
        descriptor.location,
    )?;
    let cudf_source = cudf_local_parquet_source_from_file(
        source,
        cudf_file,
        descriptor.read_columns.iter().map(String::as_str),
        "nexus_query_engine.local_parquet_chunk.mini_host_buffer_source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("path", path_fact(descriptor.path))
            .with_fact("chunk_index", descriptor.chunk_index)
            .with_fact("file_index", descriptor.file_index)
            .with_fact("read_columns", descriptor.read_columns.len())
    })?;
    let cudf_chunk =
        cudf::io::source::SourceChunk::from_source_file_selection(&cudf_source, 0, descriptor.chunk_index, None)
            .map_err(|error| {
                cudf_error_for_location(
                    descriptor.location,
                    "nexus_query_engine.local_parquet_chunk.mini_host_buffer_chunk_cudf",
                    "failed to convert mini-Parquet host-buffer chunk descriptor into cudf source chunk",
                    error,
                )
                .with_fact("path", path_fact(descriptor.path))
                .with_fact("chunk_index", descriptor.chunk_index)
                .with_fact("file_index", descriptor.file_index)
            })?;
    let cudf_source = attach_cudf_source_reader_predicate(
        context,
        cudf_source,
        &cudf_chunk,
        source,
        "nexus_query_engine.local_parquet_chunk.mini_host_buffer_source_cudf",
        descriptor.path,
        descriptor.chunk_index,
        descriptor.file_index,
        descriptor.read_columns.len(),
        Some("mini:all".to_string()),
    )?;
    Ok((cudf_source, cudf_chunk))
}

fn mini_host_buffer_path(file_index: usize, chunk_index: usize) -> PathBuf {
    PathBuf::from(format!("/__cudf_host_buffer__/mini-{file_index}-{chunk_index}.parquet"))
}

fn path_fact(path: &Path) -> String {
    crate::source::redacted_path_diagnostic(path)
}

fn chunk_path_fact(chunk: &SourceChunk) -> String {
    if chunk.location.is_local_path() { path_fact(&chunk.path) } else { chunk.location.diagnostic_location() }
}

fn file_path_fact(file: &LocalParquetFile) -> String {
    if file.location.is_local_path() { path_fact(&file.path) } else { file.location.diagnostic_location() }
}

fn file_stable_id_fact(file: &LocalParquetFile) -> String {
    file.location.diagnostic_stable_id(file.stable_id.as_str())
}

fn source_has_object_store_location(source: &LocalParquetSource) -> bool {
    source.files.iter().any(|file| !file.location.is_local_path())
}

fn cudf_error_for_source(
    source: &LocalParquetSource,
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
) -> QueryError {
    if source_has_object_store_location(source) {
        QueryError::from_cudf_without_source(operation, message, &error)
    } else {
        QueryError::from_cudf(operation, message, error)
    }
}

fn cudf_error_for_location(
    location: &crate::source::ParquetSourceLocation,
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
) -> QueryError {
    if location.is_local_path() {
        QueryError::from_cudf(operation, message, error)
    } else {
        QueryError::from_cudf_without_source(operation, message, &error)
    }
}

fn cudf_error_for_chunk(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    chunk: &SourceChunk,
) -> QueryError {
    cudf_error_for_location(&chunk.location, operation, message, error)
}

fn cudf_error_for_file(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    file: &LocalParquetFile,
) -> QueryError {
    cudf_error_for_location(&file.location, operation, message, error)
}

fn cudf_local_parquet_source_for_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<cudf::io::source::LocalParquetSource> {
    crate::chunk::source_file_for_chunk(source, chunk, "nexus_query_engine.local_parquet_chunk.source_cudf")?;
    let cudf_file = cudf_local_parquet_file_for_chunk(context, chunk)?;
    cudf_local_parquet_source_from_file(
        source,
        cudf_file,
        chunk.read_columns.iter().map(String::as_str),
        "nexus_query_engine.local_parquet_chunk.source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("path", chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_fact("row_groups", format_row_groups(chunk))
    })
}

fn cudf_local_parquet_source_for_file(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
) -> Result<cudf::io::source::LocalParquetSource> {
    let cudf_file = cudf_local_parquet_file(context, file)?;
    cudf_local_parquet_source_from_file(
        source,
        cudf_file,
        source.read_columns.iter().map(String::as_str),
        "nexus_query_engine.local_parquet_source.source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("path", file_path_fact(file))
            .with_fact("chunk_index", chunk_index)
            .with_fact("file_index", file_index)
            .with_fact("read_columns", source.read_columns.len())
    })
}

fn cudf_local_parquet_source_from_file<'a>(
    source: &LocalParquetSource,
    cudf_file: cudf::io::source::LocalParquetFile,
    read_columns: impl IntoIterator<Item = &'a str>,
    operation: &'static str,
) -> Result<cudf::io::source::LocalParquetSource> {
    cudf_local_parquet_source_from_files(source, [cudf_file], read_columns, operation)
}

fn cudf_local_parquet_source_from_files<'a>(
    source: &LocalParquetSource,
    cudf_files: impl IntoIterator<Item = cudf::io::source::LocalParquetFile>,
    read_columns: impl IntoIterator<Item = &'a str>,
    operation: &'static str,
) -> Result<cudf::io::source::LocalParquetSource> {
    let cudf_schema = cudf_source_schema(source)?;
    let output_columns = cudf_source_output_columns(source)?;
    let required_columns = cudf_source_required_columns(source)?;
    let partition_columns = cudf_partition_columns(source)?;
    cudf::io::source::LocalParquetSource::data(
        cudf_files,
        cudf_schema,
        read_columns,
        output_columns,
        required_columns,
        partition_columns,
    )
    .map_err(|error| {
        cudf_error_for_source(
            source,
            operation,
            "failed to convert native local parquet source into cudf source contract",
            error,
        )
    })
}

fn attach_cudf_source_reader_predicate(
    context: &ExecutionContext,
    mut cudf_source: cudf::io::source::LocalParquetSource,
    cudf_chunk: &cudf::io::source::SourceChunk,
    source: &LocalParquetSource,
    operation: &'static str,
    path: &Path,
    chunk_index: usize,
    file_index: usize,
    read_columns: usize,
    row_groups: Option<String>,
) -> Result<cudf::io::source::LocalParquetSource> {
    if let Some(predicate) = lower_source_reader_predicate(context, source, &cudf_source, cudf_chunk)? {
        cudf_source = cudf_source.with_reader_predicate(predicate).map_err(|error| {
            let error = cudf_error_for_source(
                source,
                operation,
                "failed to attach native reader predicate to cudf source contract",
                error,
            )
            .with_fact("path", path_fact(path))
            .with_fact("chunk_index", chunk_index)
            .with_fact("file_index", file_index)
            .with_fact("read_columns", read_columns);
            if let Some(row_groups) = row_groups { error.with_fact("row_groups", row_groups) } else { error }
        })?;
    }
    Ok(cudf_source)
}

fn attach_cudf_source_reader_batch_predicate(
    context: &ExecutionContext,
    mut cudf_source: cudf::io::source::LocalParquetSource,
    cudf_chunks: &[cudf::io::source::SourceChunk],
    source: &LocalParquetSource,
    operation: &'static str,
    path: &Path,
    chunk_index: usize,
    file_index: usize,
    read_columns: usize,
    row_groups: Option<String>,
) -> Result<cudf::io::source::LocalParquetSource> {
    if let Some(predicate) = lower_source_reader_batch_predicate(context, source, &cudf_source, cudf_chunks)? {
        cudf_source = cudf_source.with_reader_predicate(predicate).map_err(|error| {
            let error = cudf_error_for_source(
                source,
                operation,
                "failed to attach native reader predicate to cudf source batch contract",
                error,
            )
            .with_fact("path", path_fact(path))
            .with_fact("chunk_index", chunk_index)
            .with_fact("file_index", file_index)
            .with_fact("read_columns", read_columns);
            if let Some(row_groups) = row_groups { error.with_fact("row_groups", row_groups) } else { error }
        })?;
    }
    Ok(cudf_source)
}

fn attach_cudf_remote_parquet_contract(
    context: &ExecutionContext,
    cudf_file: cudf::io::source::LocalParquetFile,
    resolved_source: &crate::source::ResolvedParquetSource,
    operation: &'static str,
) -> Result<cudf::io::source::LocalParquetFile> {
    if cudf_file.remote_source().is_none() {
        return Ok(cudf_file);
    }
    let profile = cudf_remote_parquet_profile(context, resolved_source)?;
    let read_mode = cudf_remote_read_mode(resolved_source.remote_read_mode);
    cudf_file.with_remote_parquet_contract(profile, read_mode).map_err(|error| {
        QueryError::from_cudf_without_source(
            operation,
            "failed to attach typed remote Parquet contract to cudf source file",
            &error,
        )
        .with_fact("resource", "remote_parquet_source")
        .with_fact("uri_scheme", resolved_remote_uri_scheme(resolved_source))
        .with_fact("read_strategy", resolved_source.read_strategy.as_str())
        .with_fact("remote_read_mode", resolved_source.remote_read_mode.as_str())
        .with_fact("credential_mode", resolved_source.credential_mode.as_str())
    })
}

fn validate_remote_parquet_datasource_probe_request(
    resolved_source: &crate::source::ResolvedParquetSource,
    operation: &'static str,
) -> Result<()> {
    if !matches!(
        resolved_source.read_strategy,
        crate::source::ParquetReadStrategy::RemoteDirect | crate::source::ParquetReadStrategy::RemoteKvikioExperimental
    ) || resolved_source.decode_input != crate::source::SourceDecodeInput::Filepath
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "remote Parquet datasource probes require a resolved remote path-based source",
        )
        .with_fact("resource", "remote_parquet_datasource")
        .with_fact("reason", "remote_datasource_probe_requires_path_remote_strategy")
        .with_fact("read_strategy", resolved_source.read_strategy.as_str())
        .with_fact("source_decode_input", resolved_source.decode_input.as_str())
        .with_fact("remote_read_mode", resolved_source.remote_read_mode.as_str())
        .with_fact("credential_mode", resolved_source.credential_mode.as_str()));
    }
    if resolved_source.remote_read_mode == crate::source::RemoteParquetReadMode::RequireDirect {
        return Err(QueryError::new(
            QueryErrorCode::CudfUnsupported,
            operation,
            "remote direct read mode requires per-read evidence, not datasource capability",
        )
        .with_fact("resource", "remote_parquet_datasource")
        .with_fact("reason", "remote_direct_evidence_unavailable")
        .with_fact("read_strategy", resolved_source.read_strategy.as_str())
        .with_fact("remote_read_mode", resolved_source.remote_read_mode.as_str())
        .with_fact("credential_mode", resolved_source.credential_mode.as_str())
        .with_fact("remote_backend_build_status", cudf::io::parquet::remote_parquet_backend_build_status().as_str()));
    }
    Ok(())
}

fn cudf_remote_parquet_source_for_resolved_source(
    context: &ExecutionContext,
    resolved_source: &crate::source::ResolvedParquetSource,
    operation: &'static str,
) -> Result<cudf::io::parquet::RemoteParquetSource> {
    let profile = cudf_remote_parquet_profile(context, resolved_source)?;
    cudf::io::parquet::RemoteParquetSource::s3(resolved_source.reader_location.as_str())
        .map(|source| {
            source.with_profile(profile).with_read_mode(cudf_remote_read_mode(resolved_source.remote_read_mode))
        })
        .map_err(|error| {
            QueryError::from_cudf_without_source(
                operation,
                "failed to build typed remote Parquet datasource probe source",
                &error,
            )
            .with_fact("resource", "remote_parquet_datasource")
            .with_fact("uri_scheme", resolved_remote_uri_scheme(resolved_source))
            .with_fact("read_strategy", resolved_source.read_strategy.as_str())
            .with_fact("remote_read_mode", resolved_source.remote_read_mode.as_str())
            .with_fact("credential_mode", resolved_source.credential_mode.as_str())
        })
}

fn resolved_remote_uri_scheme(resolved_source: &crate::source::ResolvedParquetSource) -> &'static str {
    if resolved_source.reader_location.starts_with("s3://") || resolved_source.original_location.starts_with("s3://") {
        "s3"
    } else {
        "unknown"
    }
}

fn cudf_remote_read_mode(read_mode: crate::source::RemoteParquetReadMode) -> cudf::io::parquet::RemoteReadMode {
    match read_mode {
        crate::source::RemoteParquetReadMode::RequireDirect => cudf::io::parquet::RemoteReadMode::RequireDirect,
        crate::source::RemoteParquetReadMode::PreferDirect => cudf::io::parquet::RemoteReadMode::PreferDirect,
        crate::source::RemoteParquetReadMode::AllowHostFallback => cudf::io::parquet::RemoteReadMode::AllowHostFallback,
    }
}

fn remote_parquet_capability_disposition_label(
    disposition: cudf::io::parquet::RemoteParquetCapabilityDisposition,
) -> &'static str {
    match disposition {
        cudf::io::parquet::RemoteParquetCapabilityDisposition::PathReaderUnverified => "path_reader_unverified",
        cudf::io::parquet::RemoteParquetCapabilityDisposition::KvikioRemoteDatasource => "kvikio_remote_datasource",
        cudf::io::parquet::RemoteParquetCapabilityDisposition::BackendBuildDisabled => "backend_build_disabled",
        cudf::io::parquet::RemoteParquetCapabilityDisposition::CredentialModeUnsupported => {
            "credential_mode_unsupported"
        },
        cudf::io::parquet::RemoteParquetCapabilityDisposition::DirectEvidenceUnavailable => {
            "direct_evidence_unavailable"
        },
    }
}

fn cudf_remote_parquet_profile(
    context: &ExecutionContext,
    resolved_source: &crate::source::ResolvedParquetSource,
) -> Result<cudf::io::parquet::RemoteParquetProfile> {
    let Some(profile_id) = resolved_source.storage_profile_id.as_deref() else {
        return Ok(cudf_remote_parquet_profile_from_storage(None, None));
    };
    let storage_profile = context.source_storage_profile(profile_id)?;
    Ok(cudf_remote_parquet_profile_from_storage(Some(profile_id), Some(storage_profile)))
}

fn cudf_remote_parquet_profile_from_storage(
    profile_id: Option<&str>,
    storage_profile: Option<&crate::runtime::SourceStorageProfile>,
) -> cudf::io::parquet::RemoteParquetProfile {
    let mut profile = cudf::io::parquet::RemoteParquetProfile::shared_process_env();
    if let Some(profile_id) = profile_id {
        profile = profile.with_profile_id(profile_id);
    }
    let Some(storage_profile) = storage_profile else {
        return profile;
    };
    if storage_profile.requires_scoped_remote_direct_profile() {
        profile = profile.with_credential_mode(cudf::io::parquet::RemoteCredentialMode::ScopedProfile);
    }
    let s3_profile = storage_profile.s3_profile();
    if let Some(region) = s3_profile.region.as_deref().filter(|region| !region.is_empty()) {
        profile = profile.with_region(region);
    }
    if let Some(endpoint) = s3_profile.endpoint.as_deref().filter(|endpoint| !endpoint.is_empty()) {
        profile = profile.with_endpoint(endpoint);
    }
    if let Some(virtual_hosted_style_request) = s3_profile.virtual_hosted_style_request {
        profile = profile.with_path_style_access(!virtual_hosted_style_request);
    }
    match &s3_profile.credential {
        crate::runtime::S3SourceStorageCredential::DefaultChain => {},
        crate::runtime::S3SourceStorageCredential::Unsigned => {
            profile = profile.with_unsigned_requests();
        },
        crate::runtime::S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token } => {
            profile = profile.with_static_credentials(
                access_key_id.clone(),
                secret_access_key.clone(),
                session_token.clone(),
            );
        },
    }
    profile
}

fn cudf_local_parquet_file_for_chunk(
    context: &ExecutionContext,
    chunk: &SourceChunk,
) -> Result<cudf::io::source::LocalParquetFile> {
    let partition_values = chunk.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mut cudf_file = match chunk.resolved_source.read_strategy {
        crate::source::ParquetReadStrategy::RemoteDirect
        | crate::source::ParquetReadStrategy::RemoteKvikioExperimental => match &chunk.location {
            crate::source::ParquetSourceLocation::ObjectStoreUri { uri, .. } => {
                cudf::io::source::LocalParquetFile::object_store_uri(
                    uri.clone(),
                    chunk.stable_id.as_str(),
                    partition_values,
                )
            },
            crate::source::ParquetSourceLocation::LocalPath(_) => {
                cudf::io::source::LocalParquetFile::new(&chunk.path, chunk.stable_id.as_str(), partition_values)
            },
        },
        crate::source::ParquetReadStrategy::RemoteHostBuffer
            if matches!(chunk.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) =>
        {
            let crate::source::ParquetSourceLocation::ObjectStoreUri { uri, .. } = &chunk.location else {
                unreachable!("object-store location checked above")
            };
            cudf::io::source::LocalParquetFile::object_store_uri(
                uri.clone(),
                chunk.stable_id.as_str(),
                partition_values,
            )
        },
        crate::source::ParquetReadStrategy::LocalKvikio
        | crate::source::ParquetReadStrategy::RemoteHostBuffer
        | crate::source::ParquetReadStrategy::LocalCacheKvikio => {
            cudf::io::source::LocalParquetFile::new(&chunk.path, chunk.stable_id.as_str(), partition_values)
        },
    }
    .map_err(|error| {
        cudf_error_for_chunk(
            "nexus_query_engine.local_parquet_chunk.file_cudf",
            "failed to convert native local parquet file into cudf source contract",
            error,
            chunk,
        )
        .with_fact("path", chunk_path_fact(chunk))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
    })?;
    cudf_file = attach_cudf_remote_parquet_contract(
        context,
        cudf_file,
        &chunk.resolved_source,
        "nexus_query_engine.local_parquet_chunk.file_cudf",
    )?;
    if let Some(file_size_bytes) = chunk.known_file_bytes {
        cudf_file = cudf_file.with_file_size_bytes(file_size_bytes);
    }
    cudf_file = cudf_file.with_row_groups(chunk.source_file_row_groups.clone());
    if let (Some(row_group_bytes), Some(row_group_rows)) =
        (chunk.source_file_row_group_bytes.as_ref(), chunk.source_file_row_group_rows.as_ref())
    {
        cudf_file =
            cudf_file.with_row_group_metadata(row_group_bytes.clone(), row_group_rows.clone()).map_err(|error| {
                cudf_error_for_chunk(
                    "nexus_query_engine.local_parquet_chunk.file_cudf",
                    "failed to attach native local parquet row-group metadata to cudf source contract",
                    error,
                    chunk,
                )
                .with_fact("path", chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
            })?;
    }
    if let (Some(compressed_bytes), Some(uncompressed_bytes)) =
        (&chunk.source_file_projected_compressed_bytes, &chunk.source_file_projected_uncompressed_bytes)
    {
        cudf_file = cudf_file
            .with_projected_row_group_bytes(compressed_bytes.clone(), uncompressed_bytes.clone())
            .map_err(|error| {
                cudf_error_for_chunk(
                    "nexus_query_engine.local_parquet_chunk.file_cudf",
                    "failed to attach projected row-group byte metadata to cudf source contract",
                    error,
                    chunk,
                )
                .with_fact("path", chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
            })?;
    }
    if let (Some(compressed_bytes), Some(uncompressed_bytes)) =
        (&chunk.source_file_pure_filter_compressed_bytes, &chunk.source_file_pure_filter_uncompressed_bytes)
    {
        cudf_file = cudf_file
            .with_pure_filter_row_group_bytes(compressed_bytes.clone(), uncompressed_bytes.clone())
            .map_err(|error| {
                cudf_error_for_chunk(
                    "nexus_query_engine.local_parquet_chunk.file_cudf",
                    "failed to attach pure-filter row-group byte metadata to cudf source contract",
                    error,
                    chunk,
                )
                .with_fact("path", chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
            })?;
    }
    cudf_file = attach_cudf_reader_decimal_types(
        cudf_file,
        &chunk.reader_decimal_types,
        "nexus_query_engine.local_parquet_chunk.file_cudf",
        &chunk.location,
    )?;
    Ok(cudf_file)
}

fn cudf_local_parquet_file(
    context: &ExecutionContext,
    file: &LocalParquetFile,
) -> Result<cudf::io::source::LocalParquetFile> {
    let partition_values = file.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mut cudf_file = match file.resolved_source.read_strategy {
        crate::source::ParquetReadStrategy::RemoteDirect
        | crate::source::ParquetReadStrategy::RemoteKvikioExperimental => match &file.location {
            crate::source::ParquetSourceLocation::ObjectStoreUri { uri, .. } => {
                cudf::io::source::LocalParquetFile::object_store_uri(
                    uri.clone(),
                    file.stable_id.as_str(),
                    partition_values,
                )
            },
            crate::source::ParquetSourceLocation::LocalPath(_) => {
                cudf::io::source::LocalParquetFile::new(&file.path, file.stable_id.as_str(), partition_values)
            },
        },
        crate::source::ParquetReadStrategy::RemoteHostBuffer
            if matches!(file.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) =>
        {
            let crate::source::ParquetSourceLocation::ObjectStoreUri { uri, .. } = &file.location else {
                unreachable!("object-store location checked above")
            };
            cudf::io::source::LocalParquetFile::object_store_uri(uri.clone(), file.stable_id.as_str(), partition_values)
        },
        crate::source::ParquetReadStrategy::LocalKvikio
        | crate::source::ParquetReadStrategy::RemoteHostBuffer
        | crate::source::ParquetReadStrategy::LocalCacheKvikio => {
            cudf::io::source::LocalParquetFile::new(&file.path, file.stable_id.as_str(), partition_values)
        },
    }
    .map_err(|error| {
        cudf_error_for_file(
            "nexus_query_engine.local_parquet_source.file_cudf",
            "failed to convert native local parquet file into cudf source contract",
            error,
            file,
        )
        .with_fact("path", file_path_fact(file))
        .with_fact("stable_id", file_stable_id_fact(file))
    })?;
    cudf_file = attach_cudf_remote_parquet_contract(
        context,
        cudf_file,
        &file.resolved_source,
        "nexus_query_engine.local_parquet_source.file_cudf",
    )?;
    if let Some(file_size_bytes) = file.file_size_bytes {
        cudf_file = cudf_file.with_file_size_bytes(file_size_bytes);
    }
    cudf_file = cudf_file.with_row_groups(file.row_groups.clone());
    if let (Some(row_group_bytes), Some(row_group_rows)) = (&file.row_group_bytes, &file.row_group_rows) {
        cudf_file =
            cudf_file.with_row_group_metadata(row_group_bytes.clone(), row_group_rows.clone()).map_err(|error| {
                cudf_error_for_file(
                    "nexus_query_engine.local_parquet_source.file_cudf",
                    "failed to attach native local parquet row-group metadata to cudf source contract",
                    error,
                    file,
                )
                .with_fact("path", file_path_fact(file))
                .with_fact("stable_id", file_stable_id_fact(file))
            })?;
    }
    if let (Some(compressed_bytes), Some(uncompressed_bytes)) =
        (&file.projected_row_group_compressed_bytes, &file.projected_row_group_uncompressed_bytes)
    {
        cudf_file = cudf_file
            .with_projected_row_group_bytes(compressed_bytes.clone(), uncompressed_bytes.clone())
            .map_err(|error| {
                cudf_error_for_file(
                    "nexus_query_engine.local_parquet_source.file_cudf",
                    "failed to attach projected row-group byte metadata to cudf source contract",
                    error,
                    file,
                )
                .with_fact("path", file_path_fact(file))
                .with_fact("stable_id", file_stable_id_fact(file))
            })?;
    }
    if let (Some(compressed_bytes), Some(uncompressed_bytes)) =
        (&file.pure_filter_row_group_compressed_bytes, &file.pure_filter_row_group_uncompressed_bytes)
    {
        cudf_file = cudf_file
            .with_pure_filter_row_group_bytes(compressed_bytes.clone(), uncompressed_bytes.clone())
            .map_err(|error| {
                cudf_error_for_file(
                    "nexus_query_engine.local_parquet_source.file_cudf",
                    "failed to attach pure-filter row-group byte metadata to cudf source contract",
                    error,
                    file,
                )
                .with_fact("path", file_path_fact(file))
                .with_fact("stable_id", file_stable_id_fact(file))
            })?;
    }
    cudf_file = attach_cudf_reader_decimal_types(
        cudf_file,
        &file.reader_decimal_types,
        "nexus_query_engine.local_parquet_source.file_cudf",
        &file.location,
    )?;
    Ok(cudf_file)
}

fn attach_cudf_reader_decimal_types(
    cudf_file: cudf::io::source::LocalParquetFile,
    reader_decimal_types: &[crate::source::ParquetReaderDecimalType],
    operation: &'static str,
    location: &crate::source::ParquetSourceLocation,
) -> Result<cudf::io::source::LocalParquetFile> {
    if reader_decimal_types.is_empty() {
        return Ok(cudf_file);
    }
    let types = reader_decimal_types.iter().map(cudf_reader_decimal_type).collect::<Result<Vec<_>>>()?;
    cudf_file.with_reader_decimal_types(types).map_err(|error| {
        cudf_error_for_location(
            location,
            operation,
            "failed to attach reader decimal type metadata to cudf source contract",
            error,
        )
    })
}

fn cudf_reader_decimal_type(
    reader_type: &crate::source::ParquetReaderDecimalType,
) -> Result<cudf::io::source::ParquetReaderDecimalType> {
    let type_id = match reader_type.type_id {
        crate::source::ParquetReaderDecimalTypeId::Decimal32 => cudf::TypeId::Decimal32,
        crate::source::ParquetReaderDecimalTypeId::Decimal64 => cudf::TypeId::Decimal64,
        crate::source::ParquetReaderDecimalTypeId::Decimal128 => cudf::TypeId::Decimal128,
    };
    cudf::io::source::ParquetReaderDecimalType::new(
        reader_type.column_name.as_str(),
        cudf::DataType::fixed_point(type_id, reader_type.cudf_scale),
    )
    .map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.reader_decimal_type_cudf",
            "failed to convert native reader decimal type into cudf source contract",
            error,
        )
        .with_fact("column", reader_type.column_name.as_str())
        .with_fact("reader_type", reader_type.type_id.as_str())
    })
}

fn cudf_source_schema(source: &LocalParquetSource) -> Result<cudf::io::source::SourceSchema> {
    let columns = source
        .schema
        .fields()
        .iter()
        .map(|field| {
            let dtype = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.local_parquet_chunk.schema_cudf",
                    "local parquet source schema contains a type unsupported by cudf source reads",
                )
                .with_fact("column", field.name().as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?;
            cudf::io::source::SourceColumn::new(field.name().as_str(), dtype, field.is_nullable()).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.local_parquet_chunk.schema_cudf",
                    "failed to convert native local parquet source schema into cudf source schema",
                    error,
                )
                .with_fact("column", field.name().as_str())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    cudf::io::source::SourceSchema::new(columns).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_chunk.schema_cudf",
            "failed to build cudf local parquet source schema",
            error,
        )
    })
}

fn cudf_source_output_columns(source: &LocalParquetSource) -> Result<Vec<cudf::io::source::SourceOutputColumn>> {
    source
        .projected_output_columns
        .iter()
        .zip(&source.projected_output_column_source_refs)
        .map(|(column, source_ref)| {
            let source_ref = cudf_source_column_ref(source, column, *source_ref)?;
            cudf::io::source::SourceOutputColumn::new(column.as_str(), source_ref).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.local_parquet_chunk.output_cudf",
                    "failed to convert native local parquet output column into cudf source output column",
                    error,
                )
                .with_fact("column", column)
            })
        })
        .collect()
}

fn cudf_source_required_columns(source: &LocalParquetSource) -> Result<Vec<cudf::io::source::SourceRequiredColumn>> {
    source
        .pure_filter_columns
        .iter()
        .zip(&source.pure_filter_column_source_refs)
        .map(|(column, source_ref)| {
            let source_ref = cudf_source_column_ref(source, column, *source_ref)?;
            cudf::io::source::SourceRequiredColumn::new(
                column.as_str(),
                source_ref,
                cudf::io::source::SourceColumnRole::PureFilter,
            )
            .map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.local_parquet_chunk.required_cudf",
                    "failed to convert native local parquet pure-filter column into cudf source required column",
                    error,
                )
                .with_fact("column", column)
            })
        })
        .collect()
}

fn cudf_source_column_ref(
    source: &LocalParquetSource,
    column: &str,
    source_ref: crate::source::SourceColumnRef,
) -> Result<cudf::io::source::SourceColumnRef> {
    match source_ref {
        crate::source::SourceColumnRef::File(file_index) => source
            .read_column_file_indices
            .iter()
            .position(|index| *index == file_index)
            .map(cudf::io::source::SourceColumnRef::ReadColumn)
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_chunk.output_cudf",
                    "local parquet source file column cannot be resolved to a cudf read projection index",
                )
                .with_fact("column", column)
                .with_fact("file_column_index", file_index)
            }),
        crate::source::SourceColumnRef::Partition(index) => Ok(cudf::io::source::SourceColumnRef::Partition(index)),
        crate::source::SourceColumnRef::MissingNull(_index) => Ok(cudf::io::source::SourceColumnRef::MissingNull),
    }
}

fn cudf_partition_columns(source: &LocalParquetSource) -> Result<Vec<cudf::io::source::PartitionColumn>> {
    source
        .partition_columns
        .iter()
        .map(|column| {
            let dtype = cudf_dtype_from_arrow_type(&column.data_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.local_parquet_chunk.partition_cudf",
                    "local parquet partition column type is unsupported by cudf source reads",
                )
                .with_fact("partition_column", column.name.as_str())
                .with_fact("data_type", format!("{:?}", column.data_type))
            })?;
            cudf::io::source::PartitionColumn::new(column.name.as_str(), dtype, column.nullable).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.local_parquet_chunk.partition_cudf",
                    "failed to convert native local parquet partition column into cudf source partition column",
                    error,
                )
                .with_fact("partition_column", column.name.as_str())
            })
        })
        .collect()
}

fn cudf_partition_value(value: &LiteralValue) -> Result<cudf::io::source::PartitionValue> {
    match value {
        LiteralValue::Null(data_type) => {
            let dtype = cudf_dtype_from_arrow_type(data_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.local_parquet_chunk.partition_cudf",
                    "local parquet partition null type is unsupported by cudf source reads",
                )
                .with_fact("data_type", format!("{data_type:?}"))
            })?;
            Ok(cudf::io::source::PartitionValue::Null(dtype))
        },
        LiteralValue::Boolean(value) => Ok(cudf::io::source::PartitionValue::Boolean(*value)),
        LiteralValue::Int8(value) => Ok(cudf::io::source::PartitionValue::Int8(*value)),
        LiteralValue::Int16(value) => Ok(cudf::io::source::PartitionValue::Int16(*value)),
        LiteralValue::Int32(value) => Ok(cudf::io::source::PartitionValue::Int32(*value)),
        LiteralValue::Int64(value) => Ok(cudf::io::source::PartitionValue::Int64(*value)),
        LiteralValue::UInt8(value) => Ok(cudf::io::source::PartitionValue::UInt8(*value)),
        LiteralValue::UInt16(value) => Ok(cudf::io::source::PartitionValue::UInt16(*value)),
        LiteralValue::UInt32(value) => Ok(cudf::io::source::PartitionValue::UInt32(*value)),
        LiteralValue::UInt64(value) => Ok(cudf::io::source::PartitionValue::UInt64(*value)),
        LiteralValue::Float32(value) => Ok(cudf::io::source::PartitionValue::Float32(*value)),
        LiteralValue::Float64(value) => Ok(cudf::io::source::PartitionValue::Float64(*value)),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => {
            Ok(cudf::io::source::PartitionValue::String(value.clone()))
        },
        LiteralValue::Decimal128 { raw, scale, .. } => Ok(cudf::io::source::PartitionValue::Decimal128 {
            raw: *raw,
            scale: crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale),
        }),
        LiteralValue::Date32(value) => Ok(cudf::io::source::PartitionValue::Date32(*value)),
        LiteralValue::Timestamp { raw, unit } => Ok(cudf::io::source::PartitionValue::Timestamp {
            raw: *raw,
            resolution: timestamp_unit_to_cudf_resolution(unit),
        }),
        LiteralValue::Duration { raw, unit } => Ok(cudf::io::source::PartitionValue::Duration {
            raw: *raw,
            resolution: timestamp_unit_to_cudf_resolution(unit),
        }),
        LiteralValue::IntervalMonthDayNano { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.local_parquet_chunk.partition_cudf",
            "local parquet partition value is unsupported by cudf source reads",
        )
        .with_fact("partition_value", value)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::expr::{BinaryOp, Expr, LiteralValue};

    fn remote_direct_test_source() -> LocalParquetSource {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, false),
            arrow::datatypes::Field::new("payload", arrow::datatypes::DataType::Utf8, true),
            arrow::datatypes::Field::new("filter_only", arrow::datatypes::DataType::Boolean, true),
        ]));
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/part.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("part")
            .with_file_size_bytes(4096)
            .with_row_groups(vec![1, 3])
            .with_row_group_bytes(vec![512, 768, 1024, 1280])
            .with_projected_row_group_bytes(vec![128, 192, 256, 320], vec![256, 384, 512, 640])
            .with_pure_filter_row_group_bytes(vec![32, 48, 64, 80], vec![64, 96, 128, 160])
            .with_row_group_rows(vec![10, 10, 10, 10]);

        LocalParquetSource::from_files(vec![file], schema, vec!["payload".to_string(), "filter_only".to_string()])
            .with_column_roles(vec!["payload".to_string()], vec!["filter_only".to_string()])
            .with_column_file_indices(vec![1, 2], vec![1], vec![2])
    }

    #[test]
    fn remote_parquet_profile_carries_only_non_secret_storage_hints() {
        let storage = crate::runtime::SourceStorageProfile::s3(
            "rest-s3",
            crate::runtime::S3SourceStorageProfile::default_chain()
                .with_region("us-west-2")
                .with_endpoint("http://localhost:9000")
                .with_path_style_access(true)
                .with_static_credentials("access", "secret", Some("token".to_string())),
        );

        let profile = cudf_remote_parquet_profile_from_storage(Some("rest-s3"), Some(&storage));

        assert_eq!(profile.profile_id(), Some("rest-s3"));
        assert_eq!(profile.region(), Some("us-west-2"));
        assert_eq!(profile.endpoint(), Some("http://localhost:9000"));
        assert_eq!(profile.path_style_access(), Some(true));
        assert_eq!(profile.credential_mode(), cudf::io::parquet::RemoteCredentialMode::ScopedProfile);
    }

    #[test]
    fn remote_parquet_profile_treats_region_override_as_scoped_hint() {
        let storage = crate::runtime::SourceStorageProfile::s3(
            "glue-region",
            crate::runtime::S3SourceStorageProfile::default_chain().with_region("us-west-2"),
        );

        let profile = cudf_remote_parquet_profile_from_storage(Some("glue-region"), Some(&storage));

        assert_eq!(profile.profile_id(), Some("glue-region"));
        assert_eq!(profile.region(), Some("us-west-2"));
        assert_eq!(profile.credential_mode(), cudf::io::parquet::RemoteCredentialMode::ScopedProfile);
    }

    #[test]
    fn typed_remote_parquet_contract_uses_prefer_direct_for_remote_direct() {
        let cudf_file = cudf::io::source::LocalParquetFile::object_store_uri("s3://bucket/part.parquet", "part", [])
            .expect("valid object store file");
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/part.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        );
        let file = cudf_file
            .with_remote_parquet_contract(
                cudf_remote_parquet_profile_from_storage(None, None),
                cudf::io::parquet::RemoteReadMode::PreferDirect,
            )
            .expect("remote contract attaches");
        let remote_source = file.remote_source().expect("remote source is preserved");
        let capability = cudf::io::parquet::remote_parquet_capability(remote_source);

        assert_eq!(remote_source.uri(), "s3://bucket/part.parquet");
        assert_eq!(remote_source.read_mode(), cudf::io::parquet::RemoteReadMode::PreferDirect);
        assert_ne!(
            capability.disposition,
            cudf::io::parquet::RemoteParquetCapabilityDisposition::DirectEvidenceUnavailable
        );
        assert_eq!(resolved.read_strategy, crate::source::ParquetReadStrategy::RemoteDirect);
    }

    #[test]
    fn typed_remote_parquet_contract_preserves_require_direct_policy() {
        let cudf_file = cudf::io::source::LocalParquetFile::object_store_uri("s3://bucket/part.parquet", "part", [])
            .expect("valid object store file");
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/part.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);
        let file = cudf_file
            .with_remote_parquet_contract(
                cudf_remote_parquet_profile_from_storage(None, None),
                cudf_remote_read_mode(resolved.remote_read_mode),
            )
            .expect("remote contract attaches");
        let remote_source = file.remote_source().expect("remote source is preserved");
        let capability = cudf::io::parquet::remote_parquet_capability(remote_source);

        assert_eq!(remote_source.read_mode(), cudf::io::parquet::RemoteReadMode::RequireDirect);
        if capability.backend_build_status == cudf::io::parquet::RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled
        {
            assert_eq!(
                capability.disposition,
                cudf::io::parquet::RemoteParquetCapabilityDisposition::BackendBuildDisabled
            );
        } else {
            assert_ne!(
                capability.disposition,
                cudf::io::parquet::RemoteParquetCapabilityDisposition::DirectEvidenceUnavailable
            );
        }
        assert_eq!(capability.read_mode, cudf::io::parquet::RemoteReadMode::RequireDirect);
    }

    #[test]
    fn resolved_datasource_probe_rejects_require_direct_as_capability_only() {
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/part.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);

        let error =
            validate_remote_parquet_datasource_probe_request(&resolved, "nexus_query_engine.test.probe").unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::CudfUnsupported);
        assert_eq!(error.fact_value("resource"), Some("remote_parquet_datasource"));
        assert_eq!(error.fact_value("reason"), Some("remote_direct_evidence_unavailable"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
        assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
        assert_eq!(error.fact_value("credential_mode"), Some("shared_process_env"));
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn resolved_datasource_probe_rejects_host_buffer_strategy() {
        let resolved = crate::source::ResolvedParquetSource::remote_host_buffer(
            "s3://bucket/part.parquet",
            "s3://bucket/part.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        );

        let error =
            validate_remote_parquet_datasource_probe_request(&resolved, "nexus_query_engine.test.probe").unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("resource"), Some("remote_parquet_datasource"));
        assert_eq!(error.fact_value("reason"), Some("remote_datasource_probe_requires_path_remote_strategy"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_host_buffer"));
        assert_eq!(error.fact_value("source_decode_input"), Some("host_buffer"));
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn datasource_probe_cudf_failures_preserve_facts_without_source_text() {
        let source = cudf::io::parquet::RemoteParquetSource::s3("s3://bucket/part.parquet")
            .expect("valid remote source")
            .with_profile(
                cudf::io::parquet::RemoteParquetProfile::shared_process_env()
                    .with_credential_mode(cudf::io::parquet::RemoteCredentialMode::ScopedProfile),
            );
        let cudf_error = cudf::io::parquet::probe_remote_parquet_datasource(&source, Some(4096))
            .expect_err("scoped profiles are rejected before datasource open");

        let error = QueryError::from_cudf_without_source(
            "nexus_query_engine.test.probe",
            "failed to probe resolved remote Parquet datasource capability",
            &cudf_error,
        )
        .with_fact("resource", "remote_parquet_datasource")
        .with_fact("uri_scheme", "s3");

        assert_eq!(error.fact_value("cudf_operation"), Some("io::parquet::probe_remote_parquet_datasource"));
        assert_eq!(error.fact_value("resource"), Some("remote_parquet_datasource"));
        assert_eq!(error.fact_value("uri_scheme"), Some("s3"));
        assert!(error.source_error().is_none());
        assert!(!error.to_string().contains(" source="));
        assert!(!format!("{error:?}").contains("source: Some"));
    }

    #[test]
    fn remote_direct_cudf_contract_preserves_projection_row_groups_and_reader_mode() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let source = remote_direct_test_source();
        let (cudf_source, cudf_chunk) =
            cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
                .expect("source contract lowers through nexus conversion path");

        assert_eq!(cudf_source.read_columns(), ["payload", "filter_only"]);
        assert_eq!(cudf_source.output_columns().len(), 1);
        assert_eq!(cudf_source.required_columns().len(), 1);
        assert_eq!(cudf_source.files()[0].row_groups(), Some([1, 3].as_slice()));
        assert_eq!(cudf_source.files()[0].remote_source().unwrap().uri(), "s3://bucket/part.parquet");
        assert_eq!(
            cudf_source.files()[0].remote_source().unwrap().read_mode(),
            cudf::io::parquet::RemoteReadMode::RequireDirect
        );
        assert_eq!(cudf_chunk.remote_source().unwrap().uri(), "s3://bucket/part.parquet");
        assert!(!cudf_chunk.reads_all_row_groups());
        assert_eq!(cudf_chunk.row_group_indexes(), [1, 3]);
        assert_eq!(cudf_chunk.read_columns(), ["payload", "filter_only"]);
        assert_eq!(cudf_chunk.known_row_group_bytes(), Some(768 + 1280));
        assert_eq!(cudf_chunk.known_projected_compressed_bytes(), Some(192 + 320));
        assert_eq!(cudf_chunk.known_pure_filter_compressed_bytes(), Some(48 + 80));
    }

    #[test]
    fn remote_direct_contract_keeps_filter_only_predicate_attachable() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let source = remote_direct_test_source().with_reader_predicate(crate::expr::Expr::column("filter_only"));
        let predicate = source.reader_predicate.as_ref().expect("reader predicate set");
        assert!(source.reader_predicate_is_supported(predicate));

        let (cudf_source, _cudf_chunk) =
            cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
                .expect("source contract lowers through nexus conversion path");

        assert_eq!(cudf_source.read_columns(), ["payload", "filter_only"]);
        assert!(cudf_source.reader_predicate().is_some());
        assert_eq!(cudf_source.files()[0].remote_source().unwrap().uri(), "s3://bucket/part.parquet");
        assert_eq!(
            cudf_source.files()[0].remote_source().unwrap().read_mode(),
            cudf::io::parquet::RemoteReadMode::RequireDirect
        );
    }

    #[test]
    fn remote_direct_chunk_batch_rejects_preattached_decimal_reader_type_without_footer_verification() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "discount",
            arrow::datatypes::DataType::Decimal128(15, 2),
            true,
        )]));
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/lineitem.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("lineitem")
            .with_file_size_bytes(4096)
            .with_row_groups(vec![0])
            .with_row_group_bytes(vec![4096])
            .with_row_group_rows(vec![10])
            .with_reader_decimal_types(vec![crate::source::ParquetReaderDecimalType::new(
                "discount",
                crate::source::ParquetReaderDecimalTypeId::Decimal64,
                -2,
            )]);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);
        let predicate = source.reader_predicate.as_ref().expect("reader predicate set");
        assert!(source.reader_predicate_is_supported(predicate));
        assert!(source.reader_predicate_requires_source_contract(predicate));
        let file = &source.files[0];
        let chunk = SourceChunk {
            source_node_id: crate::plan::NodeId::from_index_for_internal_use(1),
            chunk_index: 7,
            file_index: 0,
            location: file.location.clone(),
            path: file.path.clone(),
            resolved_source: file.resolved_source.clone(),
            stable_id: file.stable_id.clone(),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: file.file_size_bytes,
            known_row_group_bytes: Some(4096),
            known_projected_compressed_bytes: Some(4096),
            known_projected_uncompressed_bytes: Some(4096),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(4096),
            estimated_source_read_bytes: Some(4096),
            exact_source_read_preflight_bytes: Some(4096),
            source_file_row_groups: file.row_groups.clone(),
            source_file_row_group_bytes: file.row_group_bytes.clone(),
            source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
            source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
            source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
            source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
            source_file_row_group_rows: file.row_group_rows.clone(),
            reader_decimal_types: file.reader_decimal_types.clone(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["discount".to_string()],
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(10),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: Some("batched".to_string()),
            source_chunk_planning_mode: crate::runtime::SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test".to_string(),
        };

        let error = cudf_local_parquet_source_chunk_batch(&context, &source, &[chunk])
            .expect_err("chunk batches must verify preattached remote decimal reader metadata against the footer");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_unavailable"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
        assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
        assert_eq!(error.fact_value("missing_reader_decimal_columns"), Some("discount"));
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn remote_direct_chunk_batch_reuses_planner_verified_decimal_reader_type() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "discount",
            arrow::datatypes::DataType::Decimal128(15, 2),
            true,
        )]));
        let reader_decimal_types = vec![crate::source::ParquetReaderDecimalType::new(
            "discount",
            crate::source::ParquetReaderDecimalTypeId::Decimal64,
            -2,
        )];
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/lineitem.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("lineitem")
            .with_file_size_bytes(4096)
            .with_row_groups(vec![0])
            .with_row_group_bytes(vec![4096])
            .with_row_group_rows(vec![10])
            .with_reader_decimal_types(reader_decimal_types.clone());
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);
        let file = &source.files[0];
        let chunk = SourceChunk {
            source_node_id: crate::plan::NodeId::from_index_for_internal_use(1),
            chunk_index: 7,
            file_index: 0,
            location: file.location.clone(),
            path: file.path.clone(),
            resolved_source: file.resolved_source.clone(),
            stable_id: file.stable_id.clone(),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: file.file_size_bytes,
            known_row_group_bytes: Some(4096),
            known_projected_compressed_bytes: Some(4096),
            known_projected_uncompressed_bytes: Some(4096),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(4096),
            estimated_source_read_bytes: Some(4096),
            exact_source_read_preflight_bytes: Some(4096),
            source_file_row_groups: file.row_groups.clone(),
            source_file_row_group_bytes: file.row_group_bytes.clone(),
            source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
            source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
            source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
            source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
            source_file_row_group_rows: file.row_group_rows.clone(),
            reader_decimal_types,
            reader_decimal_types_footer_verified: true,
            read_columns: vec!["discount".to_string()],
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(10),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: Some("batched".to_string()),
            source_chunk_planning_mode: crate::runtime::SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test".to_string(),
        };

        let (cudf_source, _chunks) = cudf_local_parquet_source_chunk_batch(&context, &source, &[chunk])
            .expect("planner-verified chunk reader decimal metadata should not refetch object-store footers");

        assert_eq!(cudf_source.files()[0].reader_decimal_types()[0].data_type().id(), cudf::TypeId::Decimal64);
        assert!(cudf_source.reader_predicate().is_some());
    }

    #[test]
    fn chunk_batch_rejects_chunk_identity_mismatch_before_footer_verification() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let source = remote_direct_test_source();
        let file = &source.files[0];
        let mut chunk = SourceChunk {
            source_node_id: crate::plan::NodeId::from_index_for_internal_use(1),
            chunk_index: 7,
            file_index: 0,
            location: file.location.clone(),
            path: file.path.clone(),
            resolved_source: file.resolved_source.clone(),
            stable_id: file.stable_id.clone(),
            row_group_indexes: vec![1],
            reads_all_row_groups: false,
            known_file_bytes: file.file_size_bytes,
            known_row_group_bytes: Some(768),
            known_projected_compressed_bytes: Some(192),
            known_projected_uncompressed_bytes: Some(384),
            known_pure_filter_compressed_bytes: Some(48),
            known_pure_filter_uncompressed_bytes: Some(96),
            estimated_chunk_bytes: Some(384),
            estimated_source_read_bytes: Some(816),
            exact_source_read_preflight_bytes: Some(384),
            source_file_row_groups: file.row_groups.clone(),
            source_file_row_group_bytes: file.row_group_bytes.clone(),
            source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
            source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
            source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
            source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
            source_file_row_group_rows: file.row_group_rows.clone(),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: source.read_columns.clone(),
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(10),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: Some("batched".to_string()),
            source_chunk_planning_mode: crate::runtime::SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test".to_string(),
        };
        chunk.location = crate::source::ParquetSourceLocation::object_store_uri("s3://bucket/other.parquet").unwrap();

        let error = cudf_local_parquet_source_chunk_batch(&context, &source, &[chunk])
            .expect_err("chunk must not verify one file while decoding another object");
        let rendered = format!("{error}");

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reason"), Some("source_chunk_identity_mismatch"));
        assert_eq!(error.fact_value("field"), Some("location"));
        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(error.fact_value("source_file_path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(!rendered.contains("s3://bucket"));
    }

    #[test]
    fn mini_host_buffer_keeps_decimal_reader_predicate_attachable() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "discount",
            arrow::datatypes::DataType::Decimal128(15, 2),
            true,
        )]));
        let resolved = crate::source::ResolvedParquetSource::remote_host_buffer(
            "s3://bucket/lineitem.parquet",
            "/host/buffer/lineitem.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_host_buffer_ranged_enabled(true);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote host-buffer source")
            .with_stable_id("lineitem")
            .with_file_size_bytes(4096)
            .with_row_groups(vec![0])
            .with_row_group_bytes(vec![4096])
            .with_row_group_rows(vec![10])
            .with_reader_decimal_types(vec![crate::source::ParquetReaderDecimalType::new(
                "discount",
                crate::source::ParquetReaderDecimalTypeId::Decimal64,
                -2,
            )]);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);
        let predicate = source.reader_predicate.as_ref().expect("reader predicate set");
        assert!(source.reader_predicate_is_supported(predicate));
        assert!(source.reader_predicate_requires_source_contract(predicate));

        let mini_row_group_bytes = vec![2048];
        let mini_row_group_rows = vec![10];
        let reader_decimal_types = MiniParquetHostBufferReaderDecimalTypes::from_footer_verified_for_test(
            source.files[0].reader_decimal_types.clone(),
        );
        let descriptor = MiniParquetHostBufferDescriptor::from_file(
            &source.files[0],
            &source,
            0,
            3,
            2048,
            &mini_row_group_bytes,
            &mini_row_group_rows,
            reader_decimal_types,
        );

        let (cudf_source, _cudf_chunk) =
            cudf_local_parquet_source_mini_host_buffer_chunk(&context, &source, descriptor)
                .expect("mini-Parquet decimal reader predicate should carry verified type metadata");

        assert_eq!(cudf_source.files()[0].reader_decimal_types()[0].data_type().id(), cudf::TypeId::Decimal64);
        assert!(cudf_source.reader_predicate().is_some());
    }

    #[test]
    fn remote_direct_contract_rejects_decimal_reader_predicate_without_footer_type_metadata() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "discount",
            arrow::datatypes::DataType::Decimal128(15, 2),
            true,
        )]));
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/lineitem.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("lineitem")
            .with_file_size_bytes(4096)
            .with_row_groups(vec![0])
            .with_row_group_bytes(vec![4096])
            .with_row_group_rows(vec![10]);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);

        let error = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
            .expect_err("remote decimal reader predicates need proven footer dtype metadata");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_unavailable"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
        assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
        assert_eq!(error.fact_value("missing_reader_decimal_columns"), Some("discount"));
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn remote_direct_contract_rejects_preattached_decimal_reader_type_without_footer_verification() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "discount",
            arrow::datatypes::DataType::Decimal128(15, 2),
            true,
        )]));
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            "s3://bucket/lineitem.parquet",
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("lineitem")
            .with_file_size_bytes(4096)
            .with_row_groups(vec![0])
            .with_row_group_bytes(vec![4096])
            .with_row_group_rows(vec![10])
            .with_reader_decimal_types(vec![crate::source::ParquetReaderDecimalType::new(
                "discount",
                crate::source::ParquetReaderDecimalTypeId::Decimal32,
                -2,
            )]);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);

        let error = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
            .expect_err("direct remote decimal reader predicates must verify preattached types against the footer");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_unavailable"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn remote_direct_cudf_contract_errors_redact_object_store_locations() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "payload",
            arrow::datatypes::DataType::Utf8,
            true,
        )]));
        let resolved = crate::source::ResolvedParquetSource::remote_direct(
            sensitive_uri,
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_read_mode(crate::source::RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("secret-bucket/private/key.parquet?X-Amz-Signature=abc")
            .with_row_group_bytes(vec![10, 20])
            .with_row_group_rows(vec![1]);
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["payload".to_string()]);

        let error = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 0)
            .expect_err("mismatched row-group metadata should fail before read");
        let rendered = format!("{error}");
        let debug = format!("{error:?}");

        assert_eq!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")), true);
        assert_eq!(
            error.fact_value("stable_id").is_some_and(|value| value.starts_with("object_store:s3:stable_id_fnv64:")),
            true
        );
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(!rendered.contains(sensitive), "{sensitive} leaked in display error: {rendered}");
            assert!(!debug.contains(sensitive), "{sensitive} leaked in debug error: {debug}");
        }
        assert!(error.source_error().is_none());
    }

    #[test]
    fn mini_host_buffer_descriptor_errors_redact_object_store_identity() {
        let context = ExecutionContext::builder().build().expect("execution context builds");
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "payload",
            arrow::datatypes::DataType::Utf8,
            true,
        )]));
        let resolved = crate::source::ResolvedParquetSource::remote_host_buffer(
            sensitive_uri,
            sensitive_uri,
            crate::source::CredentialMode::SharedProcessEnv,
        )
        .with_remote_host_buffer_ranged_enabled(true);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote host-buffer source")
            .with_stable_id("secret-bucket/private/key.parquet?X-Amz-Signature=abc")
            .with_file_size_bytes(4096);
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["payload".to_string()]);
        let mini_row_group_bytes = vec![2048];
        let mini_row_group_rows = Vec::new();
        let reader_decimal_types = MiniParquetHostBufferReaderDecimalTypes::from_footer_verified_for_test(Vec::new());
        let descriptor = MiniParquetHostBufferDescriptor::from_file(
            &source.files[0],
            &source,
            0,
            3,
            2048,
            &mini_row_group_bytes,
            &mini_row_group_rows,
            reader_decimal_types,
        );

        let error = cudf_local_parquet_source_mini_host_buffer_chunk(&context, &source, descriptor)
            .expect_err("mismatched mini row-group metadata should fail before read");
        let rendered = format!("{error}");
        let debug = format!("{error:?}");

        assert_eq!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")), true);
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(!rendered.contains(sensitive), "{sensitive} leaked in display error: {rendered}");
            assert!(!debug.contains(sensitive), "{sensitive} leaked in debug error: {debug}");
        }
        assert!(error.source_error().is_none());
    }
}
