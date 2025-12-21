use super::*;

pub fn validate_local_parquet_source(source: &LocalParquetSource) -> Result<()> {
    if source.files.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source must contain at least one file",
        )
        .with_fact("reason", "empty_files"));
    }

    for (file_index, file) in source.files.iter().enumerate() {
        if file.stable_id.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet source file must have a stable identity",
            )
            .with_fact("reason", "empty_stable_id")
            .with_fact("file_index", file_index));
        }
        if file.path.as_os_str().is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet source path must not be empty",
            )
            .with_fact("file_index", file_index));
        }
    }

    if source.read_columns.is_empty() && !source.is_zero_column_count_source() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source must name at least one required read column",
        )
        .with_fact("reason", "empty_read_columns"));
    }

    validate_local_parquet_source_contract(source)?;

    for projected_column in &source.projected_output_columns {
        if !source.read_columns.contains(projected_column) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet projected output column must be included in read columns",
            )
            .with_fact("reason", "projected_column_not_read")
            .with_fact("column", projected_column));
        }
    }

    for filter_column in &source.pure_filter_columns {
        if !source.read_columns.contains(filter_column) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet pure-filter column must be included in read columns",
            )
            .with_fact("reason", "filter_column_not_read")
            .with_fact("column", filter_column));
        }
    }

    for (file_index, file) in source.files.iter().enumerate() {
        for (index, row_group) in file.row_groups.iter().flatten().enumerate() {
            if i32::try_from(*row_group).is_err() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.local_parquet_source.validate",
                    "local parquet row group index exceeds the cudf reader range",
                )
                .with_fact("path", file.path.display())
                .with_fact("file_index", file_index)
                .with_fact("row_group_position", index)
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", i32::MAX));
            }
            if let Some(row_group_bytes) = &file.row_group_bytes
                && *row_group >= row_group_bytes.len()
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.local_parquet_source.validate",
                    "selected parquet row group is outside source contract row-group bytes",
                )
                .with_fact("path", file.path.display())
                .with_fact("file_index", file_index)
                .with_fact("row_group_position", index)
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", row_group_bytes.len().saturating_sub(1)));
            }
        }
    }

    Ok(())
}

pub(super) fn validate_local_parquet_source_contract(source: &LocalParquetSource) -> Result<()> {
    let diagnostics = &source.diagnostics;
    if !diagnostics.rejected_features.is_empty() {
        let features =
            diagnostics.rejected_features.iter().map(|feature| feature.as_str()).collect::<Vec<_>>().join(",");
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source contains rejected source features",
        )
        .with_fact("reason", "rejected_source_features")
        .with_fact("rejected_features", features));
    }
    if !diagnostics.direct_device_read {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source is not eligible for direct device read",
        )
        .with_fact("reason", "direct_device_read_disabled"));
    }
    if diagnostics.byte_range_disposition == ByteRangeDisposition::Rejected {
        return Err(unsupported_source_contract_fact("byte_range", diagnostics.byte_range_disposition.as_str()));
    }
    if diagnostics.partition_column_disposition != PartitionColumnDisposition::Absent {
        return Err(unsupported_source_contract_fact(
            "partition_columns",
            diagnostics.partition_column_disposition.as_str(),
        ));
    }
    match diagnostics.zero_column_count_disposition {
        ZeroColumnCountDisposition::NotRequired => {},
        ZeroColumnCountDisposition::Represented => validate_represented_zero_column_count_source(source)?,
        ZeroColumnCountDisposition::Rejected => {
            return Err(unsupported_source_contract_fact(
                "zero_column_count",
                diagnostics.zero_column_count_disposition.as_str(),
            ));
        },
    }
    if diagnostics.schema_mapping_disposition != SchemaMappingDisposition::DirectByName {
        return Err(unsupported_source_contract_fact(
            "schema_mapping",
            diagnostics.schema_mapping_disposition.as_str(),
        ));
    }
    if diagnostics.read_column_count != source.read_columns.len()
        || source.read_column_file_indices.len() != source.read_columns.len()
        || diagnostics.projected_output_column_count != source.projected_output_columns.len()
        || source.projected_output_column_file_indices.len() != source.projected_output_columns.len()
        || diagnostics.pure_filter_column_count != source.pure_filter_columns.len()
        || source.pure_filter_column_file_indices.len() != source.pure_filter_columns.len()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source diagnostics and schema mapping must match source column contract",
        )
        .with_fact("reason", "diagnostic_column_count_mismatch")
        .with_fact("diagnostic_read_columns", diagnostics.read_column_count)
        .with_fact("read_columns", source.read_columns.len())
        .with_fact("read_column_file_indices", source.read_column_file_indices.len())
        .with_fact("diagnostic_projected_output_columns", diagnostics.projected_output_column_count)
        .with_fact("projected_output_columns", source.projected_output_columns.len())
        .with_fact("projected_output_column_file_indices", source.projected_output_column_file_indices.len())
        .with_fact("diagnostic_pure_filter_columns", diagnostics.pure_filter_column_count)
        .with_fact("pure_filter_columns", source.pure_filter_columns.len())
        .with_fact("pure_filter_column_file_indices", source.pure_filter_column_file_indices.len()));
    }

    Ok(())
}

pub(super) fn validate_represented_zero_column_count_source(source: &LocalParquetSource) -> Result<()> {
    if !source.read_columns.is_empty()
        || !source.read_column_file_indices.is_empty()
        || !source.projected_output_columns.is_empty()
        || !source.projected_output_column_file_indices.is_empty()
        || !source.pure_filter_columns.is_empty()
        || !source.pure_filter_column_file_indices.is_empty()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "represented zero-column count source must not read or project parquet data columns",
        )
        .with_fact("reason", "invalid_zero_column_count_columns")
        .with_fact("read_columns", source.read_columns.len())
        .with_fact("projected_output_columns", source.projected_output_columns.len())
        .with_fact("pure_filter_columns", source.pure_filter_columns.len()));
    }

    if source.schema.field_with_name(ZERO_COLUMN_COUNT_COLUMN).is_err() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "represented zero-column count source must expose its synthetic row-count column",
        )
        .with_fact("reason", "missing_zero_column_count_column")
        .with_fact("column", ZERO_COLUMN_COUNT_COLUMN));
    }

    Ok(())
}

pub(super) fn unsupported_source_contract_fact(field: &'static str, disposition: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "cudf_query_engine.local_parquet_source.validate",
        "local parquet source contract contains a source fact this executor cannot honor",
    )
    .with_fact("reason", "unsupported_source_contract_fact")
    .with_fact("field", field)
    .with_fact("disposition", disposition)
}

pub fn read_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_local_parquet_source(source)?;

    if source.is_zero_column_count_source() {
        return read_zero_column_count_source(context, source, metrics);
    }

    let columns = source.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut frames = Vec::with_capacity(source.files.len());
    for file in &source.files {
        frames.push(read_local_parquet_file(context, file, &columns, metrics)?);
    }

    let mut frames = frames.into_iter();
    let frame = frames.next().expect("validated local parquet source contains at least one file");
    let other_frames = frames.collect::<Vec<_>>();
    if other_frames.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }

    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.concat",
            "failed to concatenate native local parquet files",
            error,
        )
        .with_fact("file_count", source.file_count())
        .with_fact("read_columns", source.read_column_count())
    })?;
    metrics.operator_time += timer.elapsed();

    Ok(GpuDataFrame::new(output))
}

pub(super) fn read_zero_column_count_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let chunk_plan = plan_local_parquet_source_chunks(
        NodeId::from_index_for_internal_use(0),
        source,
        SourceChunkPlanningPolicy::default(),
    )?;
    let counts = chunk_plan
        .chunks
        .iter()
        .map(|chunk| {
            chunk.known_row_count.ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.zero_column_count.read",
                    "zero-column count source requires parquet row-count metadata",
                )
                .with_fact("path", chunk.path.display())
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("row_groups", format_row_groups(chunk))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);
    zero_column_count_frame(context, &counts, metrics)
}

pub(super) fn read_local_parquet_file(
    context: &ExecutionContext,
    file: &LocalParquetFile,
    columns: &[&str],
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let row_groups = cudf_row_groups(file)?;
    let read_timer = Instant::now();
    let options = if let Some(row_groups) = &row_groups {
        ParquetReadOptions::builder()
            .filepath(&file.path)
            .columns(columns)
            .row_groups(row_groups)
            .with_runtime(context.runtime())
            .build()
    } else {
        ParquetReadOptions::builder().filepath(&file.path).columns(columns).with_runtime(context.runtime()).build()
    };

    let frame = DataFrame::read_parquet_with_options(options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.read",
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
    metrics.row_groups_selected += file.selected_row_group_count() as u64;
    metrics.read_columns += columns.len() as u64;
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(frame)
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
                    "cudf_query_engine.local_parquet_source.row_groups",
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

pub(super) fn read_local_parquet_chunk(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if chunk.zero_column_count {
        let row_count = chunk.known_row_count.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_chunk.zero_column_count",
                "zero-column count chunk requires parquet row-count metadata",
            )
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("row_groups", format_row_groups(chunk))
        })?;
        return zero_column_count_frame(context, &[row_count], metrics);
    }

    let row_groups = if chunk.reads_all_row_groups {
        None
    } else {
        Some(
            chunk
                .row_group_indexes
                .iter()
                .map(|row_group| {
                    i32::try_from(*row_group).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "cudf_query_engine.local_parquet_chunk.row_groups",
                            "local parquet row group index exceeds the cudf reader range",
                        )
                        .with_fact("path", chunk.path.display())
                        .with_fact("row_group", row_group)
                        .with_fact("max_row_group", i32::MAX)
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )
    };
    let columns = chunk.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let read_timer = Instant::now();
    let options = if let Some(row_groups) = &row_groups {
        ParquetReadOptions::builder()
            .filepath(&chunk.path)
            .columns(&columns)
            .row_groups(row_groups)
            .with_runtime(context.runtime())
            .build()
    } else {
        ParquetReadOptions::builder().filepath(&chunk.path).columns(&columns).with_runtime(context.runtime()).build()
    };

    let frame = DataFrame::read_parquet_with_options(options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_chunk.read",
            "failed to read bounded native local parquet source chunk with cudf",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", columns.len())
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(GpuDataFrame::new(frame))
}

pub(super) fn zero_column_count_frame(
    context: &ExecutionContext,
    row_counts: &[u64],
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let counts = row_counts
        .iter()
        .map(|row_count| {
            i64::try_from(*row_count).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.zero_column_count.frame",
                    "parquet row count exceeds native Int64 count representation",
                )
                .with_fact("row_count", row_count)
                .with_fact("max_row_count", i64::MAX)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let timer = Instant::now();
    let series = context.runtime().series(counts.as_slice()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.zero_column_count.frame",
            "failed to construct native zero-column count input",
            error,
        )
    })?;
    let frame = DataFrame::from_columns([(ZERO_COLUMN_COUNT_COLUMN, series)]).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.zero_column_count.frame",
            "failed to construct native zero-column count frame",
            error,
        )
    })?;
    let read_elapsed = timer.elapsed();
    metrics.source_read_time += read_elapsed;
    metrics.record_input_batch(row_counts.iter().copied().sum());
    Ok(GpuDataFrame::new(frame))
}

pub(super) fn record_bounded_parquet_source_metrics(
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

pub(super) fn format_row_groups(chunk: &SourceChunk) -> String {
    if chunk.reads_all_row_groups {
        return "all".to_string();
    }

    comma_separated_usize(&chunk.row_group_indexes)
}
