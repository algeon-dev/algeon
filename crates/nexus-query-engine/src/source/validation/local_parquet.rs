use arrow::datatypes::{DataType as ArrowDataType, Field};

use crate::{
    datatypes::{arrow_data_types_match_except_list_child_names, cudf_dtype_from_arrow_type},
    error::{QueryError, QueryErrorCode, Result},
    expr::LiteralValue,
    source::{
        ByteRangeDisposition, LocalParquetFile, LocalParquetSource, PartitionColumn, PartitionColumnDisposition,
        PredicateDisposition, ScanLimitDisposition, SchemaMappingDisposition, SourceColumnRef,
        ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition, local_parquet_file_row_group_metadata_count,
        parquet_file_path_fact, parquet_file_stable_id_fact,
    },
};

/// Validate local Parquet source invariants before native reads.
///
/// Checks file identity/path presence, partition value types, read-column
/// requirements, row-count metadata for missing-column-only reads, and reader
/// predicate support.
pub fn validate_local_parquet_source(source: &LocalParquetSource) -> Result<()> {
    if source.files.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "local parquet source must contain at least one file",
        )
        .with_fact("reason", "empty_files"));
    }

    for (file_index, file) in source.files.iter().enumerate() {
        if file.stable_id.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "local parquet source file must have a stable identity",
            )
            .with_fact("reason", "empty_stable_id")
            .with_fact("file_index", file_index));
        }
        if file.path.as_os_str().is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "local parquet source path must not be empty",
            )
            .with_fact("file_index", file_index));
        }
        if file.has_position_delete_files() {
            validate_position_delete_row_group_selection(file, file_index)?;
        }
        for (delete_index, delete_file) in file.position_delete_files.iter().enumerate() {
            if delete_file.stable_id.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.validate",
                    "position delete file must have a stable identity",
                )
                .with_fact("reason", "empty_position_delete_stable_id")
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index));
            }
            if delete_file.path.as_os_str().is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.validate",
                    "position delete source path must not be empty",
                )
                .with_fact("reason", "empty_position_delete_path")
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index));
            }
        }
        for (delete_index, delete_file) in file.equality_delete_files.iter().enumerate() {
            if delete_file.stable_id.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.validate",
                    "equality delete file must have a stable identity",
                )
                .with_fact("reason", "empty_equality_delete_stable_id")
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index));
            }
            if delete_file.path.as_os_str().is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.validate",
                    "equality delete source path must not be empty",
                )
                .with_fact("reason", "empty_equality_delete_path")
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index));
            }
            if delete_file.equality_columns.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.validate",
                    "equality delete file must name at least one equality column",
                )
                .with_fact("reason", "empty_equality_delete_columns")
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index));
            }
            for column in &delete_file.equality_columns {
                if !source.read_columns.iter().any(|read_column| read_column == column) {
                    return Err(QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.local_parquet_source.validate",
                        "equality delete key columns must be present in the source read projection",
                    )
                    .with_fact("reason", "missing_equality_delete_read_column")
                    .with_fact("file_index", file_index)
                    .with_fact("delete_index", delete_index)
                    .with_fact("column", column.as_str()));
                }
            }
        }
        if file.partition_values.len() != source.partition_columns.len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
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

    if source.read_columns.is_empty()
        && !source.is_zero_column_count_source()
        && !source.is_count_star_probe_source()
        && source.missing_columns.is_empty()
        && source.partition_columns.is_empty()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "local parquet source must name at least one required read column",
        )
        .with_fact("reason", "empty_read_columns"));
    }
    if source.read_columns.is_empty() && (!source.missing_columns.is_empty() || !source.partition_columns.is_empty()) {
        for (file_index, file) in source.files.iter().enumerate() {
            if file.row_group_rows.is_none() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.validate",
                    "local parquet source with only synthetic columns requires row-count metadata",
                )
                .with_fact("reason", "missing_row_count_metadata")
                .with_fact("file_index", file_index)
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("stable_id", parquet_file_stable_id_fact(file)));
            }
        }
    }
    if source.files.iter().any(LocalParquetFile::has_row_delete_files) && source.read_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "Iceberg row deletes require at least one physical data column",
        )
        .with_fact("reason", "row_deletes_require_data_column"));
    }
    if source.files.iter().any(LocalParquetFile::has_row_delete_files) && source.reader_predicate.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "Iceberg row deletes must be applied before source filtering",
        )
        .with_fact("reason", "row_deletes_reject_reader_predicate"));
    }
    if let Some(predicate) = &source.reader_predicate
        && !source.reader_predicate_is_supported(predicate)
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "local parquet reader predicate cannot be represented by the cudf reader contract",
        )
        .with_fact("reason", "unsupported_reader_predicate")
        .with_fact("predicate", predicate));
    }

    validate_local_parquet_source_contract(source)?;
    validate_count_star_probe_source(source)?;
    validate_output_schema_override_contract(source)?;

    for projected_column in &source.projected_output_columns {
        if source
            .source_column_ref(projected_column)
            .is_none_or(|source_ref| !source_column_ref_is_available(source, source_ref))
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "local parquet projected output column must be available from file or partition columns",
            )
            .with_fact("reason", "projected_column_not_available")
            .with_fact("column", projected_column));
        }
    }

    for filter_column in &source.pure_filter_columns {
        if source
            .source_column_ref(filter_column)
            .is_none_or(|source_ref| !source_column_ref_is_available(source, source_ref))
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
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
                    "nexus_query_engine.local_parquet_source.validate",
                    "local parquet row group index exceeds the cudf reader range",
                )
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("file_index", file_index)
                .with_fact("row_group_position", index)
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", i32::MAX));
            }
            if let Some(row_group_count) = local_parquet_file_row_group_metadata_count(file)
                && *row_group >= row_group_count
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.validate",
                    "selected parquet row group is outside source contract row-group bytes",
                )
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("file_index", file_index)
                .with_fact("row_group_position", index)
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", row_group_count.saturating_sub(1)));
            }
        }
    }

    Ok(())
}

fn validate_position_delete_row_group_selection(file: &LocalParquetFile, file_index: usize) -> Result<()> {
    let Some(row_groups) = file.row_groups.as_ref().filter(|row_groups| !row_groups.is_empty()) else {
        return Ok(());
    };
    let mut previous = row_groups[0];
    validate_position_delete_row_group_index(file, file_index, 0, previous)?;
    for (row_group_position, row_group) in row_groups.iter().copied().enumerate().skip(1) {
        validate_position_delete_row_group_index(file, file_index, row_group_position, row_group)?;
        if previous.checked_add(1) != Some(row_group) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "local parquet source with Iceberg position deletes requires contiguous row-group selections",
            )
            .with_fact("reason", "position_deletes_require_contiguous_row_groups")
            .with_fact("file_index", file_index)
            .with_fact("path", parquet_file_path_fact(file))
            .with_fact("stable_id", parquet_file_stable_id_fact(file))
            .with_fact("previous_row_group", previous)
            .with_fact("row_group", row_group));
        }
        previous = row_group;
    }
    Ok(())
}

fn validate_position_delete_row_group_index(
    file: &LocalParquetFile,
    file_index: usize,
    row_group_position: usize,
    row_group: usize,
) -> Result<()> {
    if let Some(row_group_rows) = file.row_group_rows.as_ref()
        && row_group >= row_group_rows.len()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "local parquet source with Iceberg position deletes selected a row group outside row-count metadata",
        )
        .with_fact("reason", "position_deletes_row_group_outside_row_count_metadata")
        .with_fact("file_index", file_index)
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("row_group_position", row_group_position)
        .with_fact("row_group", row_group)
        .with_fact("row_group_count", row_group_rows.len()));
    }
    Ok(())
}

fn validate_output_schema_override_contract(source: &LocalParquetSource) -> Result<()> {
    if source.output_schema_override.is_none() || source.is_count_star_probe_source() {
        return Ok(());
    }

    for (field_index, output_field) in source.output_schema().fields().iter().enumerate() {
        let source_field = source.schema.field_with_name(output_field.name()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "local parquet output schema override field must also exist in execution source schema",
            )
            .with_fact("reason", "output_schema_override_field_absent")
            .with_fact("field_index", field_index)
            .with_fact("field_name", output_field.name().as_str())
        })?;
        if !output_schema_override_field_matches_source(source_field, output_field) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "local parquet output schema override field must match execution source schema",
            )
            .with_fact("reason", "output_schema_override_field_mismatch")
            .with_fact("field_index", field_index)
            .with_fact("field_name", output_field.name().as_str())
            .with_fact("source_type", format!("{:?}", source_field.data_type()))
            .with_fact("output_type", format!("{:?}", output_field.data_type()))
            .with_fact("source_nullable", source_field.is_nullable())
            .with_fact("output_nullable", output_field.is_nullable()));
        }
    }

    Ok(())
}

fn output_schema_override_field_matches_source(source_field: &Field, output_field: &Field) -> bool {
    source_field.metadata() == output_field.metadata()
        && output_schema_override_data_type_matches_source(source_field.data_type(), output_field.data_type())
        && (!source_field.is_nullable() || output_field.is_nullable())
}

fn output_schema_override_data_type_matches_source(source_type: &ArrowDataType, output_type: &ArrowDataType) -> bool {
    match (source_type, output_type) {
        (ArrowDataType::Timestamp(source_unit, _), ArrowDataType::Timestamp(output_unit, Some(_)))
            if source_unit == output_unit =>
        {
            true
        },
        _ => arrow_data_types_match_except_list_child_names(source_type, output_type),
    }
}

fn source_column_ref_is_available(source: &LocalParquetSource, source_ref: SourceColumnRef) -> bool {
    match source_ref {
        SourceColumnRef::File(index) => source.read_column_file_indices.contains(&index),
        SourceColumnRef::Partition(index) => index < source.partition_columns.len(),
        SourceColumnRef::MissingNull(index) => index < source.missing_columns.len(),
    }
}

fn validate_count_star_probe_source(source: &LocalParquetSource) -> Result<()> {
    if !source.is_count_star_probe_source() {
        return Ok(());
    }
    if source.output_schema_override.is_none()
        || source.output_schema().field_with_name(ZERO_COLUMN_COUNT_COLUMN).is_err()
        || source.read_columns.is_empty()
        || !source.projected_output_columns.is_empty()
        || source.pure_filter_columns.len() != source.read_columns.len()
        || source.reader_predicate.is_some()
        || source.scan_limit.is_some()
        || !source.partition_columns.is_empty()
        || !source.missing_columns.is_empty()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "count-star probe source must read carrier columns and expose only the row-count weight column",
        )
        .with_fact("reason", "invalid_count_star_probe_contract")
        .with_fact("read_columns", source.read_columns.len())
        .with_fact("projected_output_columns", source.projected_output_columns.len())
        .with_fact("pure_filter_columns", source.pure_filter_columns.len())
        .with_fact("reader_predicate", source.reader_predicate.is_some())
        .with_fact("scan_limit", source.scan_limit.is_some())
        .with_fact("partition_columns", source.partition_columns.len())
        .with_fact("missing_columns", source.missing_columns.len()));
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
            "nexus_query_engine.local_parquet_source.validate",
            "local parquet source contains rejected source features",
        )
        .with_fact("reason", "rejected_source_features")
        .with_fact("rejected_features", features));
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
    match diagnostics.scan_limit_disposition {
        ScanLimitDisposition::Absent => {
            if source.scan_limit.is_some() {
                return Err(unsupported_source_contract_fact("scan_limit", "diagnostic_absent_but_present"));
            }
        },
        ScanLimitDisposition::Represented => {
            if source.scan_limit.is_none() {
                return Err(unsupported_source_contract_fact("scan_limit", "diagnostic_represented_but_absent"));
            }
        },
        ScanLimitDisposition::Rejected => {
            return Err(unsupported_source_contract_fact("scan_limit", diagnostics.scan_limit_disposition.as_str()));
        },
    }
    match diagnostics.schema_mapping_disposition {
        SchemaMappingDisposition::DirectByName => {
            if !source.missing_columns.is_empty() {
                return Err(unsupported_source_contract_fact("schema_mapping", "diagnostic_direct_but_missing"));
            }
        },
        SchemaMappingDisposition::FieldIdBindings => {
            if source.files.iter().all(|file| file.field_id_read_bindings.is_empty()) {
                return Err(unsupported_source_contract_fact(
                    "schema_mapping",
                    "diagnostic_field_id_but_bindings_empty",
                ));
            }
        },
        SchemaMappingDisposition::MissingNullableColumns => validate_represented_missing_columns(source)?,
        SchemaMappingDisposition::Rejected => {
            return Err(unsupported_source_contract_fact(
                "schema_mapping",
                diagnostics.schema_mapping_disposition.as_str(),
            ));
        },
    }
    match diagnostics.predicate_disposition {
        PredicateDisposition::ReaderPushdown => {
            if source.reader_predicate.is_none() {
                return Err(unsupported_source_contract_fact("predicate", "reader_pushdown_but_absent"));
            }
        },
        PredicateDisposition::Absent
        | PredicateDisposition::PruningOnly
        | PredicateDisposition::NativePostScanOnly
        | PredicateDisposition::Rejected => {
            if source.reader_predicate.is_some() {
                return Err(unsupported_source_contract_fact("predicate", "reader_predicate_without_diagnostic"));
            }
        },
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
            "nexus_query_engine.local_parquet_source.validate",
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

fn validate_represented_missing_columns(source: &LocalParquetSource) -> Result<()> {
    if source.missing_columns.is_empty() {
        return Err(unsupported_source_contract_fact("schema_mapping", "diagnostic_missing_but_empty"));
    }

    for (missing_index, missing_column) in source.missing_columns.iter().enumerate() {
        if source.read_columns.iter().any(|column| column == &missing_column.name)
            || source.partition_columns.iter().any(|column| column.name == missing_column.name)
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "missing source column must not also be read from parquet or partition metadata",
            )
            .with_fact("reason", "missing_column_name_collision")
            .with_fact("missing_index", missing_index)
            .with_fact("missing_column", missing_column.name.as_str()));
        }
        let field = source.schema.field_with_name(missing_column.name.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "represented missing source column must exist in source output schema",
            )
            .with_fact("reason", "missing_column_schema_field_absent")
            .with_fact("missing_index", missing_index)
            .with_fact("missing_column", missing_column.name.as_str())
        })?;
        if !arrow_data_types_match_except_list_child_names(field.data_type(), &missing_column.data_type)
            || field.is_nullable() != missing_column.nullable
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "missing source column descriptor must match source output schema",
            )
            .with_fact("reason", "missing_column_schema_mismatch")
            .with_fact("missing_index", missing_index)
            .with_fact("missing_column", missing_column.name.as_str())
            .with_fact("field_type", format!("{:?}", field.data_type()))
            .with_fact("missing_type", format!("{:?}", missing_column.data_type))
            .with_fact("field_nullable", field.is_nullable())
            .with_fact("missing_nullable", missing_column.nullable));
        }
        if !missing_column.nullable {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "missing source columns can only represent nullable logical columns",
            )
            .with_fact("reason", "non_nullable_missing_column")
            .with_fact("missing_index", missing_index)
            .with_fact("missing_column", missing_column.name.as_str())
            .with_fact("missing_type", format!("{:?}", missing_column.data_type)));
        }
        if cudf_dtype_from_arrow_type(&missing_column.data_type).is_none() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.local_parquet_source.validate",
                "missing source column type is not supported by native execution",
            )
            .with_fact("reason", "unsupported_missing_column_type")
            .with_fact("missing_index", missing_index)
            .with_fact("missing_column", missing_column.name.as_str())
            .with_fact("missing_type", format!("{:?}", missing_column.data_type)));
        }
    }

    Ok(())
}

fn validate_represented_partition_columns(source: &LocalParquetSource) -> Result<()> {
    if source.partition_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.validate",
            "represented partition source must declare at least one partition column",
        )
        .with_fact("reason", "empty_partition_columns"));
    }

    for (partition_index, partition_column) in source.partition_columns.iter().enumerate() {
        if source.read_columns.iter().any(|column| column == &partition_column.name) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "partition column must not also be read from the parquet file",
            )
            .with_fact("reason", "partition_column_also_read")
            .with_fact("partition_index", partition_index)
            .with_fact("partition_column", partition_column.name.as_str()));
        }
        let field = source.schema.field_with_name(partition_column.name.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
                "represented partition column must exist in source output schema",
            )
            .with_fact("reason", "missing_partition_schema_field")
            .with_fact("partition_index", partition_index)
            .with_fact("partition_column", partition_column.name.as_str())
        })?;
        if !arrow_data_types_match_except_list_child_names(field.data_type(), &partition_column.data_type)
            || field.is_nullable() != partition_column.nullable
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.validate",
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
            "nexus_query_engine.local_parquet_source.validate",
            "null partition value cannot be assigned to a non-nullable partition column",
        )
        .with_fact("reason", "non_nullable_partition_value_null")
        .with_fact("partition_type", format!("{:?}", partition_column.data_type))
        .with_fact("partition_nullable", partition_column.nullable)
        .with_fact("partition_value", partition_value));
    }

    if partition_column.data_type == partition_value.data_type()
        || (is_string_arrow_type(&partition_column.data_type) && is_string_arrow_type(&partition_value.data_type()))
        || timestamp_partition_value_matches(&partition_column.data_type, &partition_value.data_type())
    {
        return Ok(());
    }

    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.local_parquet_source.validate",
        "partition value literal must match its partition column type",
    )
    .with_fact("reason", "partition_value_type_mismatch")
    .with_fact("partition_type", format!("{:?}", partition_column.data_type))
    .with_fact("partition_value", partition_value))
}

fn is_string_arrow_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn timestamp_partition_value_matches(partition_type: &ArrowDataType, value_type: &ArrowDataType) -> bool {
    matches!(
        (partition_type, value_type),
        (ArrowDataType::Timestamp(partition_unit, Some(_)), ArrowDataType::Timestamp(value_unit, None))
            if partition_unit == value_unit
    )
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
            "nexus_query_engine.local_parquet_source.validate",
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
            "nexus_query_engine.local_parquet_source.validate",
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
        "nexus_query_engine.local_parquet_source.validate",
        "local parquet source contract contains a source fact this executor cannot honor",
    )
    .with_fact("reason", "unsupported_source_contract_fact")
    .with_fact("field", field)
    .with_fact("disposition", disposition)
}
