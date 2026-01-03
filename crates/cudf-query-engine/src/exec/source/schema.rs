use super::*;

pub(super) fn coerce_local_parquet_source_schema(
    mut frame: DataFrame,
    source: &LocalParquetSource,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let timer = Instant::now();
    for (field_index, field) in source.schema.fields().iter().enumerate() {
        let field_name = field.name();
        if !frame.column_names().iter().any(|name| *name == field_name.as_str()) {
            continue;
        }
        let expected_dtype = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.local_parquet_source.schema_coerce",
                "local parquet source schema type is not supported by native execution",
            )
            .with_fact("field_name", field_name.as_str())
            .with_fact("field_index", field_index)
            .with_fact("expected_type", format!("{:?}", field.data_type()))
        })?;
        let actual_dtype = frame.series(field_name.as_str()).and_then(|series| series.dtype()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.local_parquet_source.schema_coerce",
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
                        "cudf_query_engine.local_parquet_source.schema_coerce",
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
                "cudf_query_engine.local_parquet_source.schema_coerce",
            ),
            None => frame.cast_column(field_name.as_str(), expected_dtype).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.schema_coerce",
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
