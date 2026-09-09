use std::path::PathBuf;

use arrow::datatypes::TimeUnit;

use super::{
    CapabilityReason, DataType, NativeLiteralValue, NativeLocalParquetFile, NativeParquetColumnRequirements,
    NativeParquetEqualityDeleteFile, NativeParquetPositionDeleteFile, TableFormatColumnSource, TableFormatDataFile,
    TableFormatDataFileFormat, TableFormatFieldIdReadBinding, TableFormatParquetReadStrategyConfig,
    TableFormatParquetSourceMetadata, TableFormatPartitionTransform, TableFormatScalarValue, TableFormatScanPlan,
    TableFormatSourceReadGroup, attach_row_group_byte_metadata, resolve_table_format_parquet_source_with_metadata,
};

pub(super) fn table_format_local_parquet_files(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
) -> Result<Vec<NativeLocalParquetFile>, CapabilityReason> {
    if scan_plan.data_files.is_empty() {
        return Err(CapabilityReason::NativeSourceEmpty);
    }
    scan_plan.data_files.iter().map(|file| table_format_local_parquet_file(scan_plan, file, requirements)).collect()
}

pub(super) fn table_format_local_parquet_files_for_group(
    scan_plan: &TableFormatScanPlan,
    group: &TableFormatSourceReadGroup,
) -> Result<Vec<NativeLocalParquetFile>, CapabilityReason> {
    if group.file_indices.is_empty() {
        return Err(CapabilityReason::NativeSourceEmpty);
    }
    group
        .file_indices
        .iter()
        .zip(group.field_id_bindings.iter())
        .map(|(file_index, bindings)| {
            let file = scan_plan.data_files.get(*file_index).ok_or(CapabilityReason::NativeSourceSchemaMapping)?;
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
) -> Result<NativeLocalParquetFile, CapabilityReason> {
    if file.format != TableFormatDataFileFormat::Parquet {
        return Err(CapabilityReason::IcebergNonParquetDataFile);
    }
    if !file.read_capability.is_native_parquet_capable() {
        return Err(CapabilityReason::IcebergObjectStoreReadUnsupported);
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
    if !file.equality_delete_files.is_empty() {
        native_file = native_file.with_equality_delete_files(table_format_equality_delete_files(scan_plan, file)?);
    }
    if requirements.read_columns.is_empty()
        && file.row_groups.is_none()
        && !file.has_row_delete_files()
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

pub(super) fn table_format_equality_delete_files(
    scan_plan: &TableFormatScanPlan,
    file: &TableFormatDataFile,
) -> Result<Vec<NativeParquetEqualityDeleteFile>, CapabilityReason> {
    file.equality_delete_files
        .iter()
        .map(|delete_file| {
            if !delete_file.is_supported_equality_delete() {
                return Err(CapabilityReason::IcebergDeleteFilesUnsupported);
            }
            let equality_ids =
                delete_file.equality_ids.clone().ok_or(CapabilityReason::IcebergDeleteFilesUnsupported)?;
            let equality_columns = equality_delete_column_names(scan_plan, &equality_ids)?;
            let resolved_source =
                delete_file.resolved_parquet_source.clone().ok_or(CapabilityReason::IcebergDeleteFilesUnsupported)?;
            let mut native_delete = NativeParquetEqualityDeleteFile::from_resolved_parquet_source(
                delete_file.id.clone(),
                resolved_source,
                equality_ids,
                equality_columns,
            )
            .map_err(|_error| CapabilityReason::IcebergDeleteFilesUnsupported)?;
            if let Some(size_bytes) = delete_file.size_bytes {
                native_delete = native_delete.with_file_size_bytes(size_bytes);
            }
            if let Some(row_count) = delete_file.row_count {
                native_delete = native_delete.with_row_count(row_count);
            }
            Ok(native_delete)
        })
        .collect()
}

fn equality_delete_column_names(
    scan_plan: &TableFormatScanPlan,
    equality_ids: &[i32],
) -> Result<Vec<String>, CapabilityReason> {
    equality_ids
        .iter()
        .map(|field_id| {
            let column = scan_plan
                .projected_columns
                .iter()
                .chain(scan_plan.pure_filter_columns.iter())
                .find(|column| column.field_id == Some(*field_id))
                .ok_or(CapabilityReason::IcebergDeleteFilesUnsupported)?;
            if column.source != TableFormatColumnSource::DirectTopLevelFileColumn {
                return Err(CapabilityReason::IcebergDeleteFilesUnsupported);
            }
            column.source_name.clone().ok_or(CapabilityReason::IcebergDeleteFilesUnsupported)
        })
        .collect()
}

pub(super) fn table_format_position_delete_files(
    file: &TableFormatDataFile,
) -> Result<Vec<NativeParquetPositionDeleteFile>, CapabilityReason> {
    file.position_delete_files
        .iter()
        .map(|delete_file| {
            if !delete_file.is_supported_position_delete() {
                return Err(CapabilityReason::IcebergDeleteFilesUnsupported);
            }
            let resolved_source =
                delete_file.resolved_parquet_source.clone().ok_or(CapabilityReason::IcebergDeleteFilesUnsupported)?;
            let mut native_delete =
                NativeParquetPositionDeleteFile::from_resolved_parquet_source(delete_file.id.clone(), resolved_source)
                    .map_err(|_error| CapabilityReason::IcebergObjectStoreReadUnsupported)?;
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
) -> Result<Vec<NativeLiteralValue>, CapabilityReason> {
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
                .ok_or(CapabilityReason::NativeSourcePartitionValues)?;
            if value.transform != TableFormatPartitionTransform::Identity {
                return Err(CapabilityReason::IcebergPartitionTransformUnsupported);
            }
            table_format_scalar_value_to_native_literal(&value.value, &column.data_type)
        })
        .collect()
}

pub(super) fn table_format_scalar_value_to_native_literal(
    value: &TableFormatScalarValue,
    data_type: &DataType,
) -> Result<NativeLiteralValue, CapabilityReason> {
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
        TableFormatScalarValue::Int128(_) => return Err(CapabilityReason::UnsupportedType),
        TableFormatScalarValue::UInt32(value) => NativeLiteralValue::UInt32(*value),
        TableFormatScalarValue::UInt64(value) => NativeLiteralValue::UInt64(*value),
        TableFormatScalarValue::Float32(value) => NativeLiteralValue::Float32(*value),
        TableFormatScalarValue::Float64(value) => NativeLiteralValue::Float64(*value),
        TableFormatScalarValue::Utf8(value) if matches!(data_type, DataType::LargeUtf8) => {
            NativeLiteralValue::LargeUtf8(value.clone())
        },
        TableFormatScalarValue::Utf8(value) => NativeLiteralValue::Utf8(value.clone()),
        TableFormatScalarValue::Date32(value) => NativeLiteralValue::Date32(*value),
        TableFormatScalarValue::Time64Micros(value) if matches!(data_type, DataType::Time64(TimeUnit::Microsecond)) => {
            NativeLiteralValue::Time64 { raw: *value, unit: TimeUnit::Microsecond }
        },
        TableFormatScalarValue::TimestampMicros(value)
            if matches!(data_type, DataType::Timestamp(TimeUnit::Microsecond, _)) =>
        {
            NativeLiteralValue::Timestamp { raw: *value, unit: TimeUnit::Microsecond }
        },
        TableFormatScalarValue::TimestampNanos(value)
            if matches!(data_type, DataType::Timestamp(TimeUnit::Nanosecond, _)) =>
        {
            NativeLiteralValue::Timestamp { raw: *value, unit: TimeUnit::Nanosecond }
        },
        TableFormatScalarValue::Time64Micros(_)
        | TableFormatScalarValue::TimestampMicros(_)
        | TableFormatScalarValue::TimestampNanos(_)
        | TableFormatScalarValue::Uuid(_)
        | TableFormatScalarValue::Binary(_) => return Err(CapabilityReason::IcebergScalarValueUnsupported),
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
) -> Result<NativeLocalParquetFile, CapabilityReason> {
    if !file.read_capability.is_native_parquet_capable() {
        return Err(CapabilityReason::IcebergObjectStoreReadUnsupported);
    }
    let resolved_source = match file.resolved_parquet_source.clone() {
        Some(resolved_source) => resolved_source,
        None => {
            let mut metadata =
                TableFormatParquetSourceMetadata::from_size_and_row_count(file.size_bytes, file.row_count);
            if let Some(content_identity) = file.source_content_identity.as_ref() {
                metadata = metadata.with_content_identity(content_identity.clone());
            }
            resolve_table_format_parquet_source_with_metadata(
                &file.location,
                &file.read_capability,
                &TableFormatParquetReadStrategyConfig::from_env(),
                &metadata,
            )
            .map_err(|_error| CapabilityReason::IcebergObjectStoreReadUnsupported)?
        },
    };
    NativeLocalParquetFile::from_resolved_parquet_source(resolved_source)
        .map_err(|_error| CapabilityReason::IcebergObjectStoreReadUnsupported)
}
