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

fn validate_local_parquet_source_contract(source: &LocalParquetSource) -> Result<()> {
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

fn validate_represented_partition_columns(source: &LocalParquetSource) -> Result<()> {
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

fn validate_partition_value_type(partition_column: &PartitionColumn, partition_value: &LiteralValue) -> Result<()> {
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

fn validate_represented_zero_column_count_source(source: &LocalParquetSource) -> Result<()> {
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

fn unsupported_source_contract_fact(field: &'static str, disposition: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "cudf_query_engine.local_parquet_source.validate",
        "local parquet source contract contains a source fact this executor cannot honor",
    )
    .with_fact("reason", "unsupported_source_contract_fact")
    .with_fact("field", field)
    .with_fact("disposition", disposition)
}
