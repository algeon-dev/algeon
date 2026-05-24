use std::path::Path;

use super::{
    super::{
        predicate::{lower_source_reader_batch_predicate, lower_source_reader_predicate},
        *,
    },
    error::{cudf_error_for_chunk, cudf_error_for_file, cudf_error_for_source, path_fact},
    remote::attach_cudf_remote_parquet_contract,
    schema::{
        attach_cudf_reader_decimal_types, cudf_partition_columns, cudf_partition_value, cudf_source_output_columns,
        cudf_source_required_columns, cudf_source_schema,
    },
};

pub(in crate::exec::source) fn cudf_local_parquet_source_chunk(
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
                .with_fact("path", parquet_chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("read_columns", chunk.read_columns.len())
                .with_row_group_facts(chunk)
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

pub(in crate::exec::source) fn cudf_local_parquet_source_chunk_batch(
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
    let cudf_files = chunks
        .iter()
        .map(|chunk| {
            let file = crate::chunk::source_file_for_chunk(
                source,
                chunk,
                "nexus_query_engine.local_parquet_chunk_batch.file_cudf",
            )?;
            let cudf_file = cudf_local_parquet_file_for_chunk(context, chunk)?;
            Ok(attach_cudf_variable_width_payload_evidence(cudf_file, file))
        })
        .collect::<Result<Vec<_>>>()?;
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
                .with_fact("path", parquet_chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("batch_file_index", batch_file_index)
                .with_fact("read_columns", chunk.read_columns.len())
                .with_row_group_facts(chunk)
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

pub(in crate::exec::source) fn cudf_local_parquet_source_file_selection(
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
                .with_fact("path", parquet_file_path_fact(file))
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

pub(in crate::exec::source) fn footer_verified_reader_decimal_types_for_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<Option<Vec<crate::source::ParquetReaderDecimalType>>> {
    let mut metadata_cache = crate::chunk::SourceMetadataCache::default();
    footer_verified_reader_decimal_types_for_chunk_with_cache(context, source, chunk, &mut metadata_cache)
}

pub(in crate::exec::source) fn footer_verified_reader_decimal_types_for_chunk_with_cache(
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

pub(in crate::exec::source) fn chunk_with_reader_decimal_types(
    chunk: &SourceChunk,
    reader_decimal_types: Vec<crate::source::ParquetReaderDecimalType>,
) -> SourceChunk {
    let mut chunk = chunk.clone();
    chunk.reader_decimal_types = reader_decimal_types;
    chunk.reader_decimal_types_footer_verified = true;
    chunk
}

pub(in crate::exec::source) fn cudf_local_parquet_source_for_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<cudf::io::source::LocalParquetSource> {
    let file =
        crate::chunk::source_file_for_chunk(source, chunk, "nexus_query_engine.local_parquet_chunk.source_cudf")?;
    let cudf_file = cudf_local_parquet_file_for_chunk(context, chunk)?;
    let cudf_file = attach_cudf_variable_width_payload_evidence(cudf_file, file);
    cudf_local_parquet_source_from_file(
        source,
        cudf_file,
        chunk.read_columns.iter().map(String::as_str),
        "nexus_query_engine.local_parquet_chunk.source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("path", parquet_chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_row_group_facts(chunk)
    })
}

pub(in crate::exec::source) fn cudf_local_parquet_source_for_file(
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
            .with_fact("path", parquet_file_path_fact(file))
            .with_fact("chunk_index", chunk_index)
            .with_fact("file_index", file_index)
            .with_fact("read_columns", source.read_columns.len())
    })
}

pub(in crate::exec::source) fn cudf_local_parquet_source_from_file<'a>(
    source: &LocalParquetSource,
    cudf_file: cudf::io::source::LocalParquetFile,
    read_columns: impl IntoIterator<Item = &'a str>,
    operation: &'static str,
) -> Result<cudf::io::source::LocalParquetSource> {
    cudf_local_parquet_source_from_files(source, [cudf_file], read_columns, operation)
}

pub(in crate::exec::source) fn cudf_local_parquet_source_from_files<'a>(
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

pub(in crate::exec::source) fn attach_cudf_source_reader_predicate(
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
            if let Some(row_groups) = row_groups { error.with_fact("row_group_indexes", row_groups) } else { error }
        })?;
    }
    Ok(cudf_source)
}

pub(in crate::exec::source) fn attach_cudf_source_reader_batch_predicate(
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
            if let Some(row_groups) = row_groups { error.with_fact("row_group_indexes", row_groups) } else { error }
        })?;
    }
    Ok(cudf_source)
}

pub(in crate::exec::source) fn cudf_local_parquet_file_for_chunk(
    context: &ExecutionContext,
    chunk: &SourceChunk,
) -> Result<cudf::io::source::LocalParquetFile> {
    let partition_values = chunk.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mut cudf_file = match chunk.resolved_source.read_strategy() {
        crate::source::ParquetReadStrategy::RemoteDirect => match &chunk.location {
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
        crate::source::ParquetReadStrategy::LocalKvikio | crate::source::ParquetReadStrategy::LocalCacheKvikio => {
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
        .with_fact("path", parquet_chunk_path_fact(chunk))
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
                .with_fact("path", parquet_chunk_path_fact(chunk))
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
                .with_fact("path", parquet_chunk_path_fact(chunk))
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
                .with_fact("path", parquet_chunk_path_fact(chunk))
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

pub(in crate::exec::source) fn cudf_local_parquet_file(
    context: &ExecutionContext,
    file: &LocalParquetFile,
) -> Result<cudf::io::source::LocalParquetFile> {
    let partition_values = file.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mut cudf_file = match file.resolved_source.read_strategy() {
        crate::source::ParquetReadStrategy::RemoteDirect => match &file.location {
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
        crate::source::ParquetReadStrategy::LocalKvikio | crate::source::ParquetReadStrategy::LocalCacheKvikio => {
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
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
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
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("stable_id", parquet_file_stable_id_fact(file))
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
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("stable_id", parquet_file_stable_id_fact(file))
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
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("stable_id", parquet_file_stable_id_fact(file))
            })?;
    }
    cudf_file = attach_cudf_reader_decimal_types(
        cudf_file,
        &file.reader_decimal_types,
        "nexus_query_engine.local_parquet_source.file_cudf",
        &file.location,
    )?;
    Ok(attach_cudf_variable_width_payload_evidence(cudf_file, file))
}

fn attach_cudf_variable_width_payload_evidence(
    cudf_file: cudf::io::source::LocalParquetFile,
    file: &LocalParquetFile,
) -> cudf::io::source::LocalParquetFile {
    let evidence = file
        .variable_width_payload_evidence
        .iter()
        .map(|(column_index, evidence)| {
            let encoding_proof = match evidence.encoding_proof {
                crate::parquet_stats::ParquetVariableWidthEncodingProof::Empty => {
                    cudf::io::source::ParquetVariableWidthEncodingProof::Empty
                },
                crate::parquet_stats::ParquetVariableWidthEncodingProof::Plain => {
                    cudf::io::source::ParquetVariableWidthEncodingProof::Plain
                },
                crate::parquet_stats::ParquetVariableWidthEncodingProof::Dictionary => {
                    cudf::io::source::ParquetVariableWidthEncodingProof::Dictionary
                },
                crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary => {
                    cudf::io::source::ParquetVariableWidthEncodingProof::PlainAndDictionary
                },
            };
            (
                *column_index,
                cudf::io::source::ParquetVariableWidthPayloadEvidence::new(
                    evidence.physical_column_index,
                    evidence.selected_row_groups.clone(),
                    evidence.payload_upper_bound_bytes,
                    encoding_proof,
                ),
            )
        })
        .collect();
    cudf_file.with_variable_width_payload_evidence(evidence)
}
