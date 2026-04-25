use super::*;

pub(in crate::exec) fn local_parquet_execution_frame_schema(source: &LocalParquetSource) -> Result<Arc<Schema>> {
    if source.is_zero_column_count_source() {
        return Ok(Arc::clone(&source.schema));
    }
    if source.is_count_star_probe_source() {
        return Ok(Arc::clone(source.output_schema()));
    }

    let mut fields = Vec::with_capacity(source.projected_output_columns.len() + source.pure_filter_columns.len());
    let mut seen = std::collections::HashSet::with_capacity(fields.capacity());
    for column in source.projected_output_columns.iter().chain(&source.pure_filter_columns) {
        if !seen.insert(column.as_str()) {
            continue;
        }
        let field = source.schema.field_with_name(column).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.execution_frame_schema",
                "local parquet execution frame column is missing from the source schema",
            )
            .with_fact("column", column.as_str())
            .with_fact("source_fields", source.schema.fields().len())
        })?;
        fields.push(field.clone());
    }

    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn materialize_missing_columns(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    mut frame: DataFrame,
    row_count_override: Option<usize>,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    if source.missing_columns.is_empty() {
        return Ok(frame);
    }

    let rows = row_count_override.unwrap_or_else(|| frame.num_rows());
    let timer = Instant::now();
    for (missing_index, missing_column) in source.missing_columns.iter().enumerate() {
        let series = missing_column_series(context, missing_column, rows).map_err(|error| {
            error.with_fact("missing_index", missing_index).with_fact("missing_column", missing_column.name.as_str())
        })?;
        frame = frame.with_column(missing_column.name.as_str(), series).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.local_parquet_source.missing_column",
                "failed to materialize native parquet missing nullable column",
                error,
            )
            .with_fact("missing_index", missing_index)
            .with_fact("missing_column", missing_column.name.as_str())
            .with_fact("missing_type", format!("{:?}", missing_column.data_type))
        })?;
    }
    metrics.source_read_time += timer.elapsed();
    Ok(frame)
}

fn missing_column_series(
    context: &ExecutionContext,
    missing_column: &crate::source::MissingColumn,
    rows: usize,
) -> Result<cudf::Series> {
    if !missing_column.nullable {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.missing_column",
            "missing source columns can only represent nullable logical columns",
        )
        .with_fact("reason", "non_nullable_missing_column")
        .with_fact("missing_type", format!("{:?}", missing_column.data_type)));
    }

    let scalar = match &missing_column.data_type {
        ArrowDataType::Boolean => context.runtime()?.null_scalar::<bool>(),
        ArrowDataType::Int8 => context.runtime()?.null_scalar::<i8>(),
        ArrowDataType::Int16 => context.runtime()?.null_scalar::<i16>(),
        ArrowDataType::Int32 => context.runtime()?.null_scalar::<i32>(),
        ArrowDataType::Int64 => context.runtime()?.null_scalar::<i64>(),
        ArrowDataType::UInt8 => context.runtime()?.null_scalar::<u8>(),
        ArrowDataType::UInt16 => context.runtime()?.null_scalar::<u16>(),
        ArrowDataType::UInt32 => context.runtime()?.null_scalar::<u32>(),
        ArrowDataType::UInt64 => context.runtime()?.null_scalar::<u64>(),
        ArrowDataType::Float32 => context.runtime()?.null_scalar::<f32>(),
        ArrowDataType::Float64 => context.runtime()?.null_scalar::<f64>(),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            context.runtime()?.null_string_scalar()
        },
        ArrowDataType::Decimal128(_precision, scale) => {
            context.runtime()?.null_decimal128_raw_scalar(crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        ArrowDataType::Date32 => context.runtime()?.null_timestamp_raw_scalar(cudf::TimestampResolution::Days),
        ArrowDataType::Date64 => context.runtime()?.null_timestamp_raw_scalar(cudf::TimestampResolution::Milliseconds),
        ArrowDataType::Time32(unit) | ArrowDataType::Time64(unit) => {
            context.runtime()?.null_duration_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        ArrowDataType::Timestamp(unit, _) => {
            context.runtime()?.null_timestamp_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        ArrowDataType::Duration(unit) => {
            context.runtime()?.null_duration_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        data_type => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.local_parquet_source.missing_column",
                "missing source column type is not supported by native execution",
            )
            .with_fact("reason", "unsupported_missing_column_type")
            .with_fact("missing_type", format!("{data_type:?}")));
        },
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.missing_column",
            "failed to construct native parquet missing nullable scalar",
            error,
        )
    })?;

    context.runtime()?.series_from_scalar(&scalar, rows).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.missing_column",
            "failed to construct native parquet missing nullable column",
            error,
        )
    })
}

pub(super) fn coerce_local_parquet_source_schema(
    mut frame: DataFrame,
    source: &LocalParquetSource,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let timer = Instant::now();
    for (field_index, field) in source.schema.fields().iter().enumerate() {
        let field_name = field.name();
        if !frame.column_names().contains(&field_name.as_str()) {
            continue;
        }
        let expected_dtype = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.local_parquet_source.schema_coerce",
                "local parquet source schema type is not supported by native execution",
            )
            .with_fact("field_name", field_name.as_str())
            .with_fact("field_index", field_index)
            .with_fact("expected_type", format!("{:?}", field.data_type()))
        })?;
        let actual_dtype = frame.series(field_name.as_str()).and_then(|series| series.dtype()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.local_parquet_source.schema_coerce",
                "failed to inspect native local parquet source column type",
                error,
            )
            .with_fact("field_name", field_name.as_str())
            .with_fact("field_index", field_index)
        })?;
        if actual_dtype == expected_dtype {
            continue;
        }
        frame = match token {
            Some(token) => budgeted_value(
                frame.cast_column_with_budget(field_name.as_str(), expected_dtype, token).map_err(|error| {
                    QueryError::from_cudf_with_budget_details(
                        "nexus_query_engine.local_parquet_source.schema_coerce",
                        "failed to cast bounded native local parquet source column to declared schema type",
                        error,
                    )
                    .with_fact("field_name", field_name.as_str())
                    .with_fact("field_index", field_index)
                    .with_fact("actual_dtype", format!("{actual_dtype:?}"))
                    .with_fact("expected_type", format!("{:?}", field.data_type()))
                    .with_fact_if_absent("budget_operation", "dataframe::cast_column_with_budget")
                })?,
                metrics,
                "nexus_query_engine.local_parquet_source.schema_coerce",
            ),
            None => frame.cast_column(field_name.as_str(), expected_dtype).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.local_parquet_source.schema_coerce",
                    "failed to cast native local parquet source column to declared schema type",
                    error,
                )
                .with_fact("field_name", field_name.as_str())
                .with_fact("field_index", field_index)
                .with_fact("actual_dtype", format!("{actual_dtype:?}"))
                .with_fact("expected_type", format!("{:?}", field.data_type()))
            })?,
        };
    }
    metrics.source_read_time += timer.elapsed();
    Ok(frame)
}
