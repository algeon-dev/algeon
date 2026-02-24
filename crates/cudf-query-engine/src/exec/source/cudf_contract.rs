use std::path::{Path, PathBuf};

use super::{predicate::lower_source_reader_predicate, *};

pub(super) fn cudf_local_parquet_source_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<(cudf::io::source::LocalParquetSource, cudf::io::source::SourceChunk)> {
    let cudf_source = cudf_local_parquet_source_for_chunk(source, chunk)?;
    let row_groups = if chunk.reads_all_row_groups { None } else { Some(chunk.row_group_indexes.clone()) };
    let cudf_chunk =
        cudf::io::source::SourceChunk::from_source_file_selection(&cudf_source, 0, chunk.chunk_index, row_groups)
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_chunk.descriptor_cudf",
                    "failed to convert native local parquet source chunk descriptor into cudf source chunk",
                    error,
                )
                .with_fact("path", chunk.path.display())
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
        "cudf_query_engine.local_parquet_chunk.source_cudf",
        chunk.path.as_path(),
        chunk.chunk_index,
        chunk.file_index,
        chunk.read_columns.len(),
        Some(format_row_groups(chunk)),
    )?;
    Ok((cudf_source, cudf_chunk))
}

pub(super) fn cudf_local_parquet_source_file_selection(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
) -> Result<(cudf::io::source::LocalParquetSource, cudf::io::source::SourceChunk)> {
    let cudf_source = cudf_local_parquet_source_for_file(source, file, file_index, chunk_index)?;
    let row_groups = file.row_groups.clone();
    let cudf_chunk =
        cudf::io::source::SourceChunk::from_source_file_selection(&cudf_source, 0, chunk_index, row_groups).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.descriptor_cudf",
                    "failed to convert native local parquet source file descriptor into cudf source chunk",
                    error,
                )
                .with_fact("path", file.path.display())
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
        "cudf_query_engine.local_parquet_source.source_cudf",
        file.path.as_path(),
        chunk_index,
        file_index,
        source.read_columns.len(),
        None,
    )?;
    Ok((cudf_source, cudf_chunk))
}

pub(super) struct MiniParquetHostBufferDescriptor<'a> {
    path: &'a Path,
    stable_id: &'a str,
    partition_values: &'a [LiteralValue],
    read_columns: &'a [String],
    file_index: usize,
    chunk_index: usize,
    pub(super) mini_file_size_bytes: u64,
    pub(super) mini_row_group_bytes: &'a [u64],
    pub(super) mini_row_group_rows: &'a [u64],
}

impl<'a> MiniParquetHostBufferDescriptor<'a> {
    pub(super) fn from_chunk(
        chunk: &'a SourceChunk,
        mini_file_size_bytes: u64,
        mini_row_group_bytes: &'a [u64],
        mini_row_group_rows: &'a [u64],
    ) -> Self {
        Self {
            path: chunk.path.as_path(),
            stable_id: chunk.stable_id.as_str(),
            partition_values: &chunk.partition_values,
            read_columns: &chunk.read_columns,
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
    ) -> Self {
        Self {
            path: file.path.as_path(),
            stable_id: file.stable_id.as_str(),
            partition_values: &file.partition_values,
            read_columns: &source.read_columns,
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
    let mini_stable_id = format!("{}#mini:{}", descriptor.stable_id, descriptor.chunk_index);
    let cudf_file = cudf::io::source::LocalParquetFile::new(&mini_path, mini_stable_id.as_str(), partition_values)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.local_parquet_chunk.mini_host_buffer_file_cudf",
                "failed to build mini-Parquet host-buffer file descriptor",
                error,
            )
            .with_fact("path", descriptor.path.display())
            .with_fact("chunk_index", descriptor.chunk_index)
            .with_fact("file_index", descriptor.file_index)
        })?
        .with_file_size_bytes(descriptor.mini_file_size_bytes)
        .with_row_group_metadata(descriptor.mini_row_group_bytes.to_vec(), descriptor.mini_row_group_rows.to_vec())
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.local_parquet_chunk.mini_host_buffer_file_cudf",
                "failed to attach mini-Parquet row-group metadata to cudf source contract",
                error,
            )
            .with_fact("path", descriptor.path.display())
            .with_fact("chunk_index", descriptor.chunk_index)
            .with_fact("file_index", descriptor.file_index)
        })?;
    let cudf_source = cudf_local_parquet_source_from_file(
        source,
        cudf_file,
        descriptor.read_columns.iter().map(String::as_str),
        "cudf_query_engine.local_parquet_chunk.mini_host_buffer_source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("path", descriptor.path.display())
            .with_fact("chunk_index", descriptor.chunk_index)
            .with_fact("file_index", descriptor.file_index)
            .with_fact("read_columns", descriptor.read_columns.len())
    })?;
    let cudf_chunk =
        cudf::io::source::SourceChunk::from_source_file_selection(&cudf_source, 0, descriptor.chunk_index, None)
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_chunk.mini_host_buffer_chunk_cudf",
                    "failed to convert mini-Parquet host-buffer chunk descriptor into cudf source chunk",
                    error,
                )
                .with_fact("path", descriptor.path.display())
                .with_fact("chunk_index", descriptor.chunk_index)
                .with_fact("file_index", descriptor.file_index)
            })?;
    let cudf_source = attach_cudf_source_reader_predicate(
        context,
        cudf_source,
        &cudf_chunk,
        source,
        "cudf_query_engine.local_parquet_chunk.mini_host_buffer_source_cudf",
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

fn cudf_local_parquet_source_for_chunk(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<cudf::io::source::LocalParquetSource> {
    source.files.get(chunk.file_index).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_chunk.source_cudf",
            "local parquet source chunk references a missing source file",
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("source_files", source.files.len())
    })?;
    let cudf_file = cudf_local_parquet_file_for_chunk(chunk)?;
    cudf_local_parquet_source_from_file(
        source,
        cudf_file,
        chunk.read_columns.iter().map(String::as_str),
        "cudf_query_engine.local_parquet_chunk.source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_fact("row_groups", format_row_groups(chunk))
    })
}

fn cudf_local_parquet_source_for_file(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
) -> Result<cudf::io::source::LocalParquetSource> {
    let cudf_file = cudf_local_parquet_file(file)?;
    cudf_local_parquet_source_from_file(
        source,
        cudf_file,
        source.read_columns.iter().map(String::as_str),
        "cudf_query_engine.local_parquet_source.source_cudf",
    )
    .map_err(|error| {
        error
            .with_fact("path", file.path.display())
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
    let cudf_schema = cudf_source_schema(source)?;
    let output_columns = cudf_source_output_columns(source)?;
    let required_columns = cudf_source_required_columns(source)?;
    let partition_columns = cudf_partition_columns(source)?;
    cudf::io::source::LocalParquetSource::data(
        [cudf_file],
        cudf_schema,
        read_columns,
        output_columns,
        required_columns,
        partition_columns,
    )
    .map_err(|error| {
        QueryError::from_cudf(
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
            let error = QueryError::from_cudf(
                operation,
                "failed to attach native reader predicate to cudf source contract",
                error,
            )
            .with_fact("path", path.display())
            .with_fact("chunk_index", chunk_index)
            .with_fact("file_index", file_index)
            .with_fact("read_columns", read_columns);
            if let Some(row_groups) = row_groups { error.with_fact("row_groups", row_groups) } else { error }
        })?;
    }
    Ok(cudf_source)
}

fn cudf_local_parquet_file_for_chunk(chunk: &SourceChunk) -> Result<cudf::io::source::LocalParquetFile> {
    let partition_values = chunk.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mut cudf_file = match &chunk.location {
        crate::source::ParquetSourceLocation::LocalPath(_) => {
            cudf::io::source::LocalParquetFile::new(&chunk.path, chunk.stable_id.as_str(), partition_values)
        },
        crate::source::ParquetSourceLocation::ObjectStoreUri { uri, .. } => {
            cudf::io::source::LocalParquetFile::object_store_uri(
                uri.clone(),
                chunk.stable_id.as_str(),
                partition_values,
            )
        },
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_chunk.file_cudf",
            "failed to convert native local parquet file into cudf source contract",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
    })?;
    if let Some(file_size_bytes) = chunk.known_file_bytes {
        cudf_file = cudf_file.with_file_size_bytes(file_size_bytes);
    }
    cudf_file = cudf_file.with_row_groups(chunk.source_file_row_groups.clone());
    if let (Some(row_group_bytes), Some(row_group_rows)) =
        (chunk.source_file_row_group_bytes.as_ref(), chunk.source_file_row_group_rows.as_ref())
    {
        cudf_file =
            cudf_file.with_row_group_metadata(row_group_bytes.clone(), row_group_rows.clone()).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_chunk.file_cudf",
                    "failed to attach native local parquet row-group metadata to cudf source contract",
                    error,
                )
                .with_fact("path", chunk.path.display())
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
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_chunk.file_cudf",
                    "failed to attach projected row-group byte metadata to cudf source contract",
                    error,
                )
                .with_fact("path", chunk.path.display())
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
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_chunk.file_cudf",
                    "failed to attach pure-filter row-group byte metadata to cudf source contract",
                    error,
                )
                .with_fact("path", chunk.path.display())
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
            })?;
    }
    Ok(cudf_file)
}

fn cudf_local_parquet_file(file: &LocalParquetFile) -> Result<cudf::io::source::LocalParquetFile> {
    let partition_values = file.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mut cudf_file = match &file.location {
        crate::source::ParquetSourceLocation::LocalPath(_) => {
            cudf::io::source::LocalParquetFile::new(&file.path, file.stable_id.as_str(), partition_values)
        },
        crate::source::ParquetSourceLocation::ObjectStoreUri { uri, .. } => {
            cudf::io::source::LocalParquetFile::object_store_uri(uri.clone(), file.stable_id.as_str(), partition_values)
        },
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.file_cudf",
            "failed to convert native local parquet file into cudf source contract",
            error,
        )
        .with_fact("path", file.path.display())
        .with_fact("stable_id", file.stable_id.as_str())
    })?;
    if let Some(file_size_bytes) = file.file_size_bytes {
        cudf_file = cudf_file.with_file_size_bytes(file_size_bytes);
    }
    cudf_file = cudf_file.with_row_groups(file.row_groups.clone());
    if let (Some(row_group_bytes), Some(row_group_rows)) = (&file.row_group_bytes, &file.row_group_rows) {
        cudf_file =
            cudf_file.with_row_group_metadata(row_group_bytes.clone(), row_group_rows.clone()).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.file_cudf",
                    "failed to attach native local parquet row-group metadata to cudf source contract",
                    error,
                )
                .with_fact("path", file.path.display())
                .with_fact("stable_id", file.stable_id.as_str())
            })?;
    }
    if let (Some(compressed_bytes), Some(uncompressed_bytes)) =
        (&file.projected_row_group_compressed_bytes, &file.projected_row_group_uncompressed_bytes)
    {
        cudf_file = cudf_file
            .with_projected_row_group_bytes(compressed_bytes.clone(), uncompressed_bytes.clone())
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.file_cudf",
                    "failed to attach projected row-group byte metadata to cudf source contract",
                    error,
                )
                .with_fact("path", file.path.display())
                .with_fact("stable_id", file.stable_id.as_str())
            })?;
    }
    if let (Some(compressed_bytes), Some(uncompressed_bytes)) =
        (&file.pure_filter_row_group_compressed_bytes, &file.pure_filter_row_group_uncompressed_bytes)
    {
        cudf_file = cudf_file
            .with_pure_filter_row_group_bytes(compressed_bytes.clone(), uncompressed_bytes.clone())
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.file_cudf",
                    "failed to attach pure-filter row-group byte metadata to cudf source contract",
                    error,
                )
                .with_fact("path", file.path.display())
                .with_fact("stable_id", file.stable_id.as_str())
            })?;
    }
    Ok(cudf_file)
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
                    "cudf_query_engine.local_parquet_chunk.schema_cudf",
                    "local parquet source schema contains a type unsupported by cudf source reads",
                )
                .with_fact("column", field.name().as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?;
            cudf::io::source::SourceColumn::new(field.name().as_str(), dtype, field.is_nullable()).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_chunk.schema_cudf",
                    "failed to convert native local parquet source schema into cudf source schema",
                    error,
                )
                .with_fact("column", field.name().as_str())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    cudf::io::source::SourceSchema::new(columns).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_chunk.schema_cudf",
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
                    "cudf_query_engine.local_parquet_chunk.output_cudf",
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
                    "cudf_query_engine.local_parquet_chunk.required_cudf",
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
                    "cudf_query_engine.local_parquet_chunk.output_cudf",
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
                    "cudf_query_engine.local_parquet_chunk.partition_cudf",
                    "local parquet partition column type is unsupported by cudf source reads",
                )
                .with_fact("partition_column", column.name.as_str())
                .with_fact("data_type", format!("{:?}", column.data_type))
            })?;
            cudf::io::source::PartitionColumn::new(column.name.as_str(), dtype, column.nullable).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_chunk.partition_cudf",
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
                    "cudf_query_engine.local_parquet_chunk.partition_cudf",
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
            "cudf_query_engine.local_parquet_chunk.partition_cudf",
            "local parquet partition value is unsupported by cudf source reads",
        )
        .with_fact("partition_value", value)),
    }
}
