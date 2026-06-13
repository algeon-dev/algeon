use super::*;

pub(super) fn table_format_local_parquet_files(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
) -> Result<Vec<NativeLocalParquetFile>, FallbackReason> {
    if scan_plan.data_files.is_empty() {
        return Err(FallbackReason::NativeSourceEmpty);
    }
    scan_plan.data_files.iter().map(|file| table_format_local_parquet_file(scan_plan, file, requirements)).collect()
}

pub(super) fn table_format_local_parquet_files_for_group(
    scan_plan: &TableFormatScanPlan,
    group: &TableFormatSourceReadGroup,
) -> Result<Vec<NativeLocalParquetFile>, FallbackReason> {
    if group.file_indices.is_empty() {
        return Err(FallbackReason::NativeSourceEmpty);
    }
    group
        .file_indices
        .iter()
        .zip(group.field_id_bindings.iter())
        .map(|(file_index, bindings)| {
            let file = scan_plan.data_files.get(*file_index).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
            table_format_local_parquet_file(scan_plan, file, &group.requirements).map(|native_file| {
                native_file.with_field_id_read_bindings(
                    bindings.iter().map(TableFormatFieldIdReadBinding::native_binding).collect(),
                )
            })
        })
        .collect()
}

pub(super) fn table_format_local_parquet_file(
    scan_plan: &TableFormatScanPlan,
    file: &TableFormatDataFile,
    requirements: &NativeParquetColumnRequirements,
) -> Result<NativeLocalParquetFile, FallbackReason> {
    if file.format != TableFormatDataFileFormat::Parquet {
        return Err(FallbackReason::IcebergNonParquetDataFile);
    }
    if !file.read_capability.is_native_parquet_capable() {
        return Err(FallbackReason::IcebergObjectStoreReadUnsupported);
    }
    let stable_id = format!("table-format:{}:{}", scan_plan.format.code(), file.id);
    let mut native_file = table_format_native_parquet_file(file)?.with_stable_id(stable_id);
    if let Some(file_size_bytes) = file.size_bytes {
        native_file = native_file.with_file_size_bytes(file_size_bytes);
    }
    if let Some(row_groups) = file.row_groups.clone() {
        native_file = native_file.with_row_groups(row_groups);
    }
    if !file.position_delete_files.is_empty() {
        native_file = native_file.with_position_delete_files(table_format_position_delete_files(file)?);
    }
    if requirements.read_columns.is_empty()
        && file.row_groups.is_none()
        && !file.has_position_delete_files()
        && let Some(row_count) = file.row_count
    {
        native_file = native_file.with_row_group_rows(vec![row_count]);
    }
    if !requirements.partition_columns.is_empty() {
        native_file = native_file.with_partition_values(table_format_partition_values(file, requirements)?);
    }
    for file_column_index in &requirements.read_column_file_indices {
        if let Some(null_counts) = file.row_group_null_counts.get(file_column_index) {
            native_file = native_file.with_row_group_null_counts(*file_column_index, null_counts.clone());
        }
    }
    Ok(attach_row_group_byte_metadata(native_file, requirements))
}

pub(super) fn table_format_position_delete_files(
    file: &TableFormatDataFile,
) -> Result<Vec<NativeParquetPositionDeleteFile>, FallbackReason> {
    file.position_delete_files
        .iter()
        .map(|delete_file| {
            if !delete_file.is_supported_position_delete() {
                return Err(FallbackReason::IcebergDeleteFilesUnsupported);
            }
            let resolved_source =
                delete_file.resolved_parquet_source.clone().ok_or(FallbackReason::IcebergDeleteFilesUnsupported)?;
            let mut native_delete =
                NativeParquetPositionDeleteFile::from_resolved_parquet_source(delete_file.id.clone(), resolved_source)
                    .map_err(|_error| FallbackReason::IcebergObjectStoreReadUnsupported)?;
            if let Some(size_bytes) = delete_file.size_bytes {
                native_delete = native_delete.with_file_size_bytes(size_bytes);
            }
            if let Some(row_count) = delete_file.row_count {
                native_delete = native_delete.with_row_count(row_count);
            }
            if let Some(referenced_data_file) = delete_file.referenced_data_file.as_ref() {
                native_delete = native_delete.with_referenced_data_file(referenced_data_file.clone());
            }
            native_delete =
                native_delete.with_position_bounds(delete_file.position_lower_bound, delete_file.position_upper_bound);
            Ok(native_delete)
        })
        .collect()
}

pub(super) fn table_format_partition_values(
    file: &TableFormatDataFile,
    requirements: &NativeParquetColumnRequirements,
) -> Result<Vec<NativeLiteralValue>, FallbackReason> {
    requirements
        .partition_columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            if let Some(value) = requirements.partition_column_values.get(index).and_then(|value| value.clone()) {
                return Ok(value);
            }
            let value = file
                .partition_values
                .iter()
                .find(|value| value.native_materialized_column_name() == column.name)
                .ok_or(FallbackReason::NativeSourcePartitionValues)?;
            if value.transform != TableFormatPartitionTransform::Identity {
                return Err(FallbackReason::IcebergPartitionTransformUnsupported);
            }
            table_format_scalar_value_to_native_literal(&value.value, &column.data_type)
        })
        .collect()
}

pub(super) fn table_format_scalar_value_to_native_literal(
    value: &TableFormatScalarValue,
    data_type: &DataType,
) -> Result<NativeLiteralValue, FallbackReason> {
    Ok(match value {
        TableFormatScalarValue::Null => NativeLiteralValue::Null(data_type.clone()),
        TableFormatScalarValue::Boolean(value) => NativeLiteralValue::Boolean(*value),
        TableFormatScalarValue::Int32(value) => NativeLiteralValue::Int32(*value),
        TableFormatScalarValue::Int64(value) if matches!(data_type, DataType::Timestamp(_, _)) => {
            let DataType::Timestamp(unit, _) = data_type else { unreachable!("timestamp type checked in guard") };
            NativeLiteralValue::Timestamp { raw: *value, unit: *unit }
        },
        TableFormatScalarValue::Int64(value) if matches!(data_type, DataType::Decimal128(_, _)) => {
            let DataType::Decimal128(precision, scale) = data_type else {
                unreachable!("decimal type checked in guard")
            };
            NativeLiteralValue::Decimal128 { raw: i128::from(*value), precision: *precision, scale: *scale }
        },
        TableFormatScalarValue::Int64(value) => NativeLiteralValue::Int64(*value),
        TableFormatScalarValue::Int128(value) if matches!(data_type, DataType::Decimal128(_, _)) => {
            let DataType::Decimal128(precision, scale) = data_type else {
                unreachable!("decimal type checked in guard")
            };
            NativeLiteralValue::Decimal128 { raw: *value, precision: *precision, scale: *scale }
        },
        TableFormatScalarValue::Int128(_) => return Err(FallbackReason::UnsupportedType),
        TableFormatScalarValue::UInt32(value) => NativeLiteralValue::UInt32(*value),
        TableFormatScalarValue::UInt64(value) => NativeLiteralValue::UInt64(*value),
        TableFormatScalarValue::Float64(value) => NativeLiteralValue::Float64(*value),
        TableFormatScalarValue::Utf8(value) if matches!(data_type, DataType::LargeUtf8) => {
            NativeLiteralValue::LargeUtf8(value.clone())
        },
        TableFormatScalarValue::Utf8(value) => NativeLiteralValue::Utf8(value.clone()),
        TableFormatScalarValue::Date32(value) => NativeLiteralValue::Date32(*value),
    })
}

pub(super) fn table_format_local_path(location: &str) -> PathBuf {
    if let Some(path) = location.strip_prefix("file://localhost/") {
        return PathBuf::from(format!("/{path}"));
    }
    if let Some(path) = location.strip_prefix("file://") {
        return PathBuf::from(path);
    }
    if let Some(path) = location.strip_prefix("file:") {
        return PathBuf::from(path);
    }
    PathBuf::from(location)
}

pub(super) fn table_format_native_parquet_file(
    file: &TableFormatDataFile,
) -> Result<NativeLocalParquetFile, FallbackReason> {
    if !file.read_capability.is_native_parquet_capable() {
        return Err(FallbackReason::IcebergObjectStoreReadUnsupported);
    }
    let resolved_source = match file.resolved_parquet_source.clone() {
        Some(resolved_source) => resolved_source,
        None => resolve_table_format_parquet_source_with_metadata(
            &file.location,
            &file.read_capability,
            &TableFormatParquetReadStrategyConfig::from_env(),
            &TableFormatParquetSourceMetadata::from_size_and_row_count(file.size_bytes, file.row_count),
        )
        .map_err(|_error| FallbackReason::IcebergObjectStoreReadUnsupported)?,
    };
    NativeLocalParquetFile::from_resolved_parquet_source(resolved_source)
        .map_err(|_error| FallbackReason::IcebergObjectStoreReadUnsupported)
}
