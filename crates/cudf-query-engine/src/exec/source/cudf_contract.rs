use super::*;

pub(super) fn cudf_local_parquet_source_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<(cudf::io::source::LocalParquetSource, cudf::io::source::SourceChunk)> {
    let source = cudf_local_parquet_source_for_chunk(context, source, chunk)?;
    let row_groups = if chunk.reads_all_row_groups { None } else { Some(chunk.row_group_indexes.clone()) };
    let cudf_chunk =
        cudf::io::source::SourceChunk::from_source_file_selection(&source, 0, chunk.chunk_index, row_groups).map_err(
            |error| {
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
            },
        )?;
    Ok((source, cudf_chunk))
}

fn cudf_local_parquet_source_for_chunk(
    context: &ExecutionContext,
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
    let cudf_schema = cudf_source_schema(source)?;
    let output_columns = cudf_source_output_columns(source)?;
    let required_columns = cudf_source_required_columns(source)?;
    let partition_columns = cudf_partition_columns(source)?;
    let mut cudf_source = cudf::io::source::LocalParquetSource::data(
        [cudf_file],
        cudf_schema,
        chunk.read_columns.iter().map(String::as_str),
        output_columns,
        required_columns,
        partition_columns,
    )
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_chunk.source_cudf",
            "failed to convert native local parquet source into cudf source contract",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", chunk.read_columns.len())
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    if let Some(predicate) = lower_source_reader_predicate(context, source, chunk.path.as_path())? {
        cudf_source = cudf_source.with_reader_predicate(predicate).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.local_parquet_chunk.source_cudf",
                "failed to attach native reader predicate to cudf source contract",
                error,
            )
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_fact("row_groups", format_row_groups(chunk))
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
