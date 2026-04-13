use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TableFormatFieldIdReadBinding {
    pub(super) logical_field_id: i32,
    pub(super) logical_name: String,
    pub(super) physical_name: String,
    pub(super) physical_root_index: usize,
    pub(super) nullable: bool,
    pub(super) data_type: DataType,
}

impl TableFormatFieldIdReadBinding {
    pub(super) fn native_binding(&self) -> NativeParquetFieldIdReadBinding {
        NativeParquetFieldIdReadBinding::new(
            self.logical_field_id,
            self.logical_name.clone(),
            self.physical_name.clone(),
            self.physical_root_index,
            self.nullable,
            self.data_type.clone(),
        )
    }
}

#[derive(Debug, Clone)]
pub(super) struct TableFormatFileReadBindings {
    pub(super) bindings: Vec<TableFormatFieldIdReadBinding>,
}

pub(super) fn validate_table_format_direct_file_column_field_ids(
    scan_plan: &TableFormatScanPlan,
    mappings: &[&TableFormatColumnMapping],
) -> Result<(), FallbackReason> {
    if scan_plan.format != TableFormat::Iceberg {
        return Ok(());
    }
    let direct_mappings = mappings
        .iter()
        .filter(|mapping| mapping.source == TableFormatColumnSource::DirectTopLevelFileColumn)
        .map(|mapping| {
            mapping
                .field_id
                .map(|field_id| (*mapping, field_id))
                .ok_or(FallbackReason::IcebergFieldIdMappingUnsupported)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if direct_mappings.is_empty() {
        return Ok(());
    }

    for file in scan_plan.data_files.iter().filter(|file| file.read_capability.is_native_parquet_capable()) {
        let schema =
            table_format_data_file_footer_arrow_schema(file).ok_or(FallbackReason::IcebergFieldIdMappingUnsupported)?;
        for (mapping, field_id) in &direct_mappings {
            let matched_field = match footer_field_by_id(&schema, *field_id) {
                Some(field) => Some(field),
                None => footer_field_by_mapping_name_without_field_id(&schema, mapping)?,
            };
            let Some((_index, field)) = matched_field else {
                if mapping.nullable {
                    continue;
                }
                return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
            };
            if !table_format_footer_field_matches_mapping(field, mapping) {
                return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
            }
        }
    }

    Ok(())
}

pub(super) fn parquet_field_id(field: &Field) -> Option<i32> {
    field.metadata().get(PARQUET_FIELD_ID_META_KEY).and_then(|value| value.parse::<i32>().ok())
}

pub(super) fn parquet_field_id_metadata_is_absent(field: &Field) -> bool {
    !field.metadata().contains_key(PARQUET_FIELD_ID_META_KEY)
}

pub(super) fn table_format_requires_iceberg_field_id_bindings(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
) -> Result<bool, FallbackReason> {
    if scan_plan.format != TableFormat::Iceberg {
        return Ok(false);
    }
    let mappings = table_format_column_mapping_by_name(scan_plan)?;
    requirements.scan_output_columns.iter().filter_map(|column| mappings.get(column.as_str())).try_fold(
        false,
        |required, mapping| {
            if mapping.source != TableFormatColumnSource::DirectTopLevelFileColumn {
                return Ok(required);
            }
            Ok(required || mapping.field_id.is_some())
        },
    )
}

pub(super) fn table_format_file_field_id_read_bindings(
    _file_index: usize,
    footer_schema: &Schema,
    requirements: &NativeParquetColumnRequirements,
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
) -> Result<TableFormatFileReadBindings, FallbackReason> {
    let mut bindings = Vec::new();
    let mut seen = BTreeSet::new();
    for column in &requirements.scan_output_columns {
        let mapping = mappings.get(column.as_str()).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
        if mapping.source != TableFormatColumnSource::DirectTopLevelFileColumn {
            continue;
        }
        let field_id = mapping.field_id.ok_or(FallbackReason::IcebergFieldIdMappingUnsupported)?;
        if !seen.insert(field_id) {
            continue;
        }
        let matched_field = match footer_field_by_id(footer_schema, field_id) {
            Some(field) => Some(field),
            None => footer_field_by_mapping_name_without_field_id(footer_schema, mapping)?,
        };
        let Some((physical_root_index, field)) = matched_field else {
            if mapping.nullable || mapping.missing_default_value.is_some() {
                continue;
            }
            return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
        };
        if !table_format_footer_field_matches_mapping(field, mapping) {
            return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
        }
        bindings.push(TableFormatFieldIdReadBinding {
            logical_field_id: field_id,
            logical_name: mapping.logical_name.clone(),
            physical_name: field.name().clone(),
            physical_root_index,
            nullable: field.is_nullable(),
            data_type: field.data_type().clone(),
        });
    }
    Ok(TableFormatFileReadBindings { bindings })
}

pub(super) fn footer_field_by_id(schema: &Schema, field_id: i32) -> Option<(usize, &Field)> {
    schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_index, field)| parquet_field_id(field) == Some(field_id))
        .map(|(index, field)| (index, field.as_ref()))
}

pub(super) fn footer_field_by_mapping_name_without_field_id<'a>(
    schema: &'a Schema,
    mapping: &TableFormatColumnMapping,
) -> Result<Option<(usize, &'a Field)>, FallbackReason> {
    let matches = footer_fields_by_mapping_names(schema, mapping);
    if matches.is_empty() {
        return Ok(None);
    }
    if matches.iter().any(|(_index, field)| !parquet_field_id_metadata_is_absent(field)) {
        return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
    }
    let [matched] = matches.as_slice() else {
        return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
    };
    if !table_format_footer_field_matches_mapping(matched.1, mapping) {
        return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
    }
    Ok(Some(*matched))
}

fn footer_fields_by_mapping_names<'a>(
    schema: &'a Schema,
    mapping: &TableFormatColumnMapping,
) -> Vec<(usize, &'a Field)> {
    schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_index, field)| {
            mapping.field_id_physical_name_candidates().any(|candidate| field.name() == candidate)
        })
        .map(|(index, field)| (index, field.as_ref()))
        .collect()
}
