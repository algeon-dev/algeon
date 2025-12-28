use super::*;

pub fn read_gpu_resident_source(
    context: &ExecutionContext,
    source: &crate::source::GpuResidentSource,
    inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let handle = inputs.get(&source.input_name).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.gpu_resident_source.read",
            "GPU-resident source input was not provided to native execution",
        )
        .with_fact("input_name", source.input_name.as_str())
    })?;

    if handle.schema().as_ref() != source.schema.as_ref() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.gpu_resident_source.read",
            "GPU-resident source schema does not match the declared source schema",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("declared_fields", source.schema.fields().len())
        .with_fact("actual_fields", handle.schema().fields().len()));
    }

    let context_runtime = context.shared_runtime();
    if !std::sync::Arc::ptr_eq(handle.runtime(), &context_runtime) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.gpu_resident_source.read",
            "GPU-resident source runtime ownership does not match the execution context",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_device", handle.runtime().device().ordinal())
        .with_fact("context_device", context.runtime().device().ordinal()));
    }

    if handle.device_placement() != context.device_placement()
        || handle.placement().device != context.runtime().device()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.gpu_resident_source.read",
            "GPU-resident source device placement does not match the execution context",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_device", handle.placement().device.ordinal())
        .with_fact("context_device", context.runtime().device().ordinal()));
    }

    metrics.boundary.record_gpu_resident_input_handoff();
    Ok(handle.clone().into_gpu_frame())
}

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
        if file.partition_values.len() != source.partition_columns.len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet source file partition values must match source partition columns",
            )
            .with_fact("reason", "partition_value_count_mismatch")
            .with_fact("file_index", file_index)
            .with_fact("partition_columns", source.partition_columns.len())
            .with_fact("partition_values", file.partition_values.len()));
        }
        for (partition_index, (partition_column, partition_value)) in
            source.partition_columns.iter().zip(&file.partition_values).enumerate()
        {
            validate_partition_value_type(partition_column, partition_value).map_err(|error| {
                error
                    .with_fact("file_index", file_index)
                    .with_fact("partition_index", partition_index)
                    .with_fact("partition_column", partition_column.name.as_str())
            })?;
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
        if source.source_column_ref(projected_column).is_none() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet projected output column must be available from file or partition columns",
            )
            .with_fact("reason", "projected_column_not_available")
            .with_fact("column", projected_column));
        }
    }

    for filter_column in &source.pure_filter_columns {
        if source.source_column_ref(filter_column).is_none() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet pure-filter column must be available from file or partition columns",
            )
            .with_fact("reason", "filter_column_not_available")
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
    match diagnostics.partition_column_disposition {
        PartitionColumnDisposition::Absent => {
            if !source.partition_columns.is_empty() {
                return Err(unsupported_source_contract_fact("partition_columns", "diagnostic_absent_but_present"));
            }
        },
        PartitionColumnDisposition::Represented => validate_represented_partition_columns(source)?,
        PartitionColumnDisposition::Rejected => {
            return Err(unsupported_source_contract_fact(
                "partition_columns",
                diagnostics.partition_column_disposition.as_str(),
            ));
        },
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
        || source.projected_output_column_source_refs.len() != source.projected_output_columns.len()
        || diagnostics.pure_filter_column_count != source.pure_filter_columns.len()
        || source.pure_filter_column_source_refs.len() != source.pure_filter_columns.len()
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
        .with_fact("projected_output_column_source_refs", source.projected_output_column_source_refs.len())
        .with_fact("diagnostic_pure_filter_columns", diagnostics.pure_filter_column_count)
        .with_fact("pure_filter_columns", source.pure_filter_columns.len())
        .with_fact("pure_filter_column_source_refs", source.pure_filter_column_source_refs.len()));
    }

    Ok(())
}

pub(super) fn validate_represented_partition_columns(source: &LocalParquetSource) -> Result<()> {
    if source.partition_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "represented partition source must declare at least one partition column",
        )
        .with_fact("reason", "empty_partition_columns"));
    }

    for (partition_index, partition_column) in source.partition_columns.iter().enumerate() {
        if source.read_columns.iter().any(|column| column == &partition_column.name) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "partition column must not also be read from the parquet file",
            )
            .with_fact("reason", "partition_column_also_read")
            .with_fact("partition_index", partition_index)
            .with_fact("partition_column", partition_column.name.as_str()));
        }
        let field = source.schema.field_with_name(partition_column.name.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "represented partition column must exist in source output schema",
            )
            .with_fact("reason", "missing_partition_schema_field")
            .with_fact("partition_index", partition_index)
            .with_fact("partition_column", partition_column.name.as_str())
        })?;
        if field.data_type() != &partition_column.data_type || field.is_nullable() != partition_column.nullable {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "represented partition column schema must match source output schema",
            )
            .with_fact("reason", "partition_schema_mismatch")
            .with_fact("partition_index", partition_index)
            .with_fact("partition_column", partition_column.name.as_str())
            .with_fact("declared_type", format!("{:?}", partition_column.data_type))
            .with_fact("schema_type", format!("{:?}", field.data_type()))
            .with_fact("declared_nullable", partition_column.nullable)
            .with_fact("schema_nullable", field.is_nullable()));
        }
    }

    Ok(())
}

pub(super) fn validate_partition_value_type(
    partition_column: &PartitionColumn,
    partition_value: &LiteralValue,
) -> Result<()> {
    if matches!(partition_value, LiteralValue::Null(_)) && !partition_column.nullable {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "null partition value cannot be assigned to a non-nullable partition column",
        )
        .with_fact("reason", "non_nullable_partition_value_null")
        .with_fact("partition_type", format!("{:?}", partition_column.data_type))
        .with_fact("partition_nullable", partition_column.nullable)
        .with_fact("partition_value", partition_value));
    }

    if partition_column.data_type == partition_value.data_type()
        || (is_string_arrow_type(&partition_column.data_type) && is_string_arrow_type(&partition_value.data_type()))
    {
        return Ok(());
    }

    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "cudf_query_engine.local_parquet_source.validate",
        "partition value literal must match its partition column type",
    )
    .with_fact("reason", "partition_value_type_mismatch")
    .with_fact("partition_type", format!("{:?}", partition_column.data_type))
    .with_fact("partition_value", partition_value))
}

fn is_string_arrow_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

pub(super) fn validate_represented_zero_column_count_source(source: &LocalParquetSource) -> Result<()> {
    if !source.read_columns.is_empty()
        || !source.read_column_file_indices.is_empty()
        || !source.projected_output_columns.is_empty()
        || !source.projected_output_column_source_refs.is_empty()
        || !source.pure_filter_columns.is_empty()
        || !source.pure_filter_column_source_refs.is_empty()
        || !source.partition_columns.is_empty()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "represented zero-column count source must not read or project parquet data columns",
        )
        .with_fact("reason", "invalid_zero_column_count_columns")
        .with_fact("read_columns", source.read_columns.len())
        .with_fact("projected_output_columns", source.projected_output_columns.len())
        .with_fact("pure_filter_columns", source.pure_filter_columns.len())
        .with_fact("partition_columns", source.partition_columns.len()));
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
        let frame = read_local_parquet_file(context, file, &columns, metrics)?;
        frames.push(inject_partition_columns(context, source, &file.partition_values, frame, metrics)?.into_frame());
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

pub(super) fn inject_partition_columns(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    partition_values: &[LiteralValue],
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if source.partition_columns.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }
    if source.partition_columns.len() != partition_values.len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.inject_partitions",
            "partition value count must match source partition columns",
        )
        .with_fact("partition_columns", source.partition_columns.len())
        .with_fact("partition_values", partition_values.len()));
    }

    let rows = frame.num_rows();
    let timer = Instant::now();
    let mut output = frame;
    for (partition_column, partition_value) in source.partition_columns.iter().zip(partition_values) {
        let series = partition_value_series(context, partition_value, rows).map_err(|error| {
            error
                .with_fact("partition_column", partition_column.name.as_str())
                .with_fact("partition_value", partition_value)
        })?;
        output = output.with_column(partition_column.name.as_str(), series).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.local_parquet_source.inject_partitions",
                "failed to inject native parquet partition column",
                error,
            )
            .with_fact("partition_column", partition_column.name.as_str())
            .with_fact("partition_value", partition_value)
        })?;
    }
    metrics.source_read_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn partition_value_series(
    context: &ExecutionContext,
    value: &LiteralValue,
    rows: usize,
) -> Result<cudf::Series> {
    if let LiteralValue::Null(data_type) = value {
        let scalar = null_partition_value_scalar(context, data_type)?;
        return context.runtime().series_from_scalar(&scalar, rows).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.local_parquet_source.partition_value",
                "failed to construct native parquet partition column",
                error,
            )
        });
    }

    let scalar = match value {
        LiteralValue::Boolean(value) => context.runtime().scalar(*value),
        LiteralValue::Int8(value) => context.runtime().scalar(*value),
        LiteralValue::Int16(value) => context.runtime().scalar(*value),
        LiteralValue::Int32(value) => context.runtime().scalar(*value),
        LiteralValue::Int64(value) => context.runtime().scalar(*value),
        LiteralValue::UInt8(value) => context.runtime().scalar(*value),
        LiteralValue::UInt16(value) => context.runtime().scalar(*value),
        LiteralValue::UInt32(value) => context.runtime().scalar(*value),
        LiteralValue::UInt64(value) => context.runtime().scalar(*value),
        LiteralValue::Float32(value) => context.runtime().scalar(*value),
        LiteralValue::Float64(value) => context.runtime().scalar(*value),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => context.runtime().string_scalar(value),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            context.runtime().decimal128_raw_scalar(*raw, crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        LiteralValue::Date32(value) => {
            context.runtime().timestamp_raw_scalar(i64::from(*value), cudf::TimestampResolution::Days)
        },
        LiteralValue::Timestamp { raw, unit } => {
            context.runtime().timestamp_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Duration { raw, unit } => {
            context.runtime().duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::IntervalMonthDayNano { .. } => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.local_parquet_source.partition_value",
                "interval partition values are not supported by native parquet sources",
            )
            .with_fact("partition_value", value));
        },
        LiteralValue::Null(_) => unreachable!("typed null partition values are handled before scalar construction"),
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.partition_value",
            "failed to construct native parquet partition scalar",
            error,
        )
    })?;

    context.runtime().series_from_scalar(&scalar, rows).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.partition_value",
            "failed to construct native parquet partition column",
            error,
        )
    })
}

fn null_partition_value_scalar(context: &ExecutionContext, data_type: &ArrowDataType) -> Result<cudf::core::Scalar> {
    let scalar = match data_type {
        ArrowDataType::Boolean => context.runtime().null_scalar::<bool>(),
        ArrowDataType::Int8 => context.runtime().null_scalar::<i8>(),
        ArrowDataType::Int16 => context.runtime().null_scalar::<i16>(),
        ArrowDataType::Int32 => context.runtime().null_scalar::<i32>(),
        ArrowDataType::Int64 => context.runtime().null_scalar::<i64>(),
        ArrowDataType::UInt8 => context.runtime().null_scalar::<u8>(),
        ArrowDataType::UInt16 => context.runtime().null_scalar::<u16>(),
        ArrowDataType::UInt32 => context.runtime().null_scalar::<u32>(),
        ArrowDataType::UInt64 => context.runtime().null_scalar::<u64>(),
        ArrowDataType::Float32 => context.runtime().null_scalar::<f32>(),
        ArrowDataType::Float64 => context.runtime().null_scalar::<f64>(),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            context.runtime().null_string_scalar()
        },
        ArrowDataType::Decimal128(_precision, scale) => {
            context.runtime().null_decimal128_raw_scalar(crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        ArrowDataType::Date32 => context.runtime().null_timestamp_raw_scalar(cudf::TimestampResolution::Days),
        ArrowDataType::Timestamp(unit, None) => {
            context.runtime().null_timestamp_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        ArrowDataType::Duration(unit) => {
            context.runtime().null_duration_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.local_parquet_source.partition_value",
                "native parquet partition null literal type is not supported",
            )
            .with_fact("data_type", format!("{data_type:?}")));
        },
    };

    scalar.map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.partition_value",
            "failed to construct native parquet partition null scalar",
            error,
        )
    })
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
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    budget_bytes: u64,
    budget_boundary: &'static str,
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

    let (cudf_source, cudf_chunk) = cudf_local_parquet_source_chunk(source, chunk)?;
    let budget_bytes = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_chunk.budget",
            "bounded local parquet source chunk budget exceeds platform usize range",
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("budget_bytes", budget_bytes)
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    let token = context
        .runtime()
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "cudf_query_engine.local_parquet_chunk.budget",
                "failed to create bounded native local parquet source chunk budget token",
                error,
            )
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_fact("row_groups", format_row_groups(chunk))
        })?;
    let read_timer = Instant::now();
    let frame = DataFrame::read_parquet_source_chunk_with_budget(
        context.runtime(),
        &cudf_source,
        &cudf_chunk,
        SourceReadMode::ExecutionFrame,
        &token,
    )
    .map_err(|error| {
        QueryError::from_cudf_with_budget_details(
            "cudf_query_engine.local_parquet_chunk.read",
            "failed to read bounded native local parquet source chunk with cudf",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", chunk.read_columns.len())
        .with_fact("row_groups", format_row_groups(chunk))
    })?
    .into_value();
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(GpuDataFrame::new(frame))
}

fn cudf_local_parquet_source_chunk(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<(cudf::io::source::LocalParquetSource, cudf::io::source::SourceChunk)> {
    let source = cudf_local_parquet_source_for_chunk(source, chunk)?;
    let chunk_plan = cudf::io::source::plan_local_parquet_source_chunks(
        &source,
        cudf::io::source::SourceChunkPlanningPolicy::default(),
    )
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_chunk.plan_cudf",
            "failed to plan cudf local parquet source chunk",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", chunk.read_columns.len())
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    if chunk_plan.chunks().len() != 1 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_chunk.plan_cudf",
            "cudf local parquet source chunk conversion must produce exactly one chunk",
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("planned_chunks", chunk_plan.chunks().len())
        .with_fact("row_groups", format_row_groups(chunk)));
    }

    Ok((source, chunk_plan.chunks()[0].clone()))
}

fn cudf_local_parquet_source_for_chunk(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Result<cudf::io::source::LocalParquetSource> {
    let file = source.files.get(chunk.file_index).ok_or_else(|| {
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
    let cudf_file = cudf_local_parquet_file_for_chunk(file, chunk)?;
    let cudf_schema = cudf_source_schema(source)?;
    let output_columns = cudf_source_output_columns(source)?;
    let required_columns = cudf_source_required_columns(source)?;
    let partition_columns = cudf_partition_columns(source)?;
    cudf::io::source::LocalParquetSource::data(
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
    })
}

fn cudf_local_parquet_file_for_chunk(
    file: &LocalParquetFile,
    chunk: &SourceChunk,
) -> Result<cudf::io::source::LocalParquetFile> {
    let partition_values = chunk.partition_values.iter().map(cudf_partition_value).collect::<Result<Vec<_>>>()?;
    let mut cudf_file = cudf::io::source::LocalParquetFile::new(&chunk.path, file.stable_id.as_str(), partition_values)
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
    cudf_file = cudf_file.with_row_groups(if chunk.reads_all_row_groups {
        None
    } else {
        Some(chunk.row_group_indexes.clone())
    });
    if let (Some(row_group_bytes), Some(row_group_rows)) = (&file.row_group_bytes, &file.row_group_rows) {
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
    let frame = DataFrame::from_series([(ZERO_COLUMN_COUNT_COLUMN, series)]).map_err(|error| {
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
