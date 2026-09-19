use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::{
    CapabilityReason, DataType, Field, NativeMissingColumn, NativeParquetColumnRequirements, NativePartitionColumn,
    NativeSourceColumnRef, Schema, SchemaRef, TableFormatColumnMapping, TableFormatColumnSource,
    TableFormatFieldIdReadBinding, TableFormatFileReadBindings, TableFormatScanPlan,
    table_format_column_mapping_by_name, table_format_data_file_footer_arrow_schema,
    table_format_file_field_id_read_bindings, table_format_file_schema,
    table_format_footer_data_type_needs_logical_source_contract, table_format_requires_iceberg_field_id_bindings,
    table_format_scalar_value_to_native_literal,
};

#[derive(Debug, Clone)]
pub(super) struct TableFormatSourceReadGroup {
    pub(super) file_indices: Vec<usize>,
    pub(super) requirements: NativeParquetColumnRequirements,
    pub(super) source_schema: SchemaRef,
    pub(super) output_schema: SchemaRef,
    pub(super) field_id_bindings: Vec<Vec<TableFormatFieldIdReadBinding>>,
}

impl TableFormatSourceReadGroup {
    pub(super) fn has_field_id_aliases(&self) -> bool {
        self.field_id_bindings.iter().flatten().any(|binding| binding.logical_name != binding.physical_name)
    }

    pub(super) fn has_read_coercions(&self) -> bool {
        self.field_id_bindings.iter().flatten().any(|binding| {
            self.source_schema.field_with_name(binding.physical_name.as_str()).is_ok_and(|field| {
                table_format_footer_data_type_needs_logical_source_contract(&binding.data_type, field.data_type())
            })
        })
    }
}

pub(super) fn table_format_source_read_groups(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
    output_schema: SchemaRef,
) -> Result<Vec<TableFormatSourceReadGroup>, CapabilityReason> {
    if !table_format_requires_iceberg_field_id_bindings(scan_plan, requirements)? {
        let file_schema = table_format_file_schema(scan_plan);
        let source_schema = table_format_source_contract_schema(requirements, &output_schema, &file_schema)?;
        return Ok(vec![TableFormatSourceReadGroup {
            file_indices: (0..scan_plan.data_files.len()).collect(),
            requirements: requirements.clone(),
            source_schema,
            output_schema,
            field_id_bindings: vec![Vec::new(); scan_plan.data_files.len()],
        }]);
    }

    let mappings = table_format_column_mapping_by_name(scan_plan)?;
    let mut groups: Vec<TableFormatSourceReadGroup> = Vec::new();
    let mut group_by_key: BTreeMap<String, usize> = BTreeMap::new();

    for (file_index, file) in scan_plan.data_files.iter().enumerate() {
        let footer_schema = table_format_data_file_footer_arrow_schema(file)
            .ok_or(CapabilityReason::IcebergFieldIdMappingUnsupported)?;
        let file_bindings =
            table_format_file_field_id_read_bindings(file_index, &footer_schema, requirements, &mappings)?;
        let file_requirements =
            table_format_requirements_for_file(requirements, &mappings, &file_bindings, &footer_schema)?;
        if requirements.requires_physical_row_presence && file_requirements.read_columns.is_empty() {
            return Err(CapabilityReason::IcebergFieldIdMappingUnsupported);
        }
        let source_schema = table_format_source_contract_schema(&file_requirements, &output_schema, &footer_schema)?;
        let key = table_format_source_group_key(&file_requirements);
        if let Some(group_index) = group_by_key.get(&key).copied() {
            groups[group_index].file_indices.push(file_index);
            groups[group_index].field_id_bindings.push(file_bindings.bindings);
        } else {
            group_by_key.insert(key, groups.len());
            groups.push(TableFormatSourceReadGroup {
                file_indices: vec![file_index],
                requirements: file_requirements,
                source_schema,
                output_schema: Arc::clone(&output_schema),
                field_id_bindings: vec![file_bindings.bindings],
            });
        }
    }

    Ok(groups)
}

pub(super) fn table_format_requirements_for_file(
    base: &NativeParquetColumnRequirements,
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
    file_bindings: &TableFormatFileReadBindings,
    footer_schema: &Schema,
) -> Result<NativeParquetColumnRequirements, CapabilityReason> {
    let mut read_columns = Vec::new();
    let mut read_column_file_indices = Vec::new();
    let mut read_seen = BTreeSet::new();
    let mut missing_columns = base.missing_columns.clone();
    let mut missing_seen = missing_columns.iter().map(|column| column.name.clone()).collect::<BTreeSet<_>>();
    let mut partition_columns = base.partition_columns.clone();
    let mut partition_column_values = base.partition_column_values.clone();
    let mut partition_seen = partition_columns.iter().map(|column| column.name.clone()).collect::<BTreeSet<_>>();

    for column in &base.scan_output_columns {
        let mapping = mappings.get(column.as_str()).ok_or(CapabilityReason::NativeSourceSchemaMapping)?;
        if mapping.source != TableFormatColumnSource::DirectTopLevelFileColumn {
            continue;
        }
        let field_id = mapping.field_id.ok_or(CapabilityReason::IcebergFieldIdMappingUnsupported)?;
        if let Some(binding) = file_bindings.bindings.iter().find(|binding| binding.logical_field_id == field_id) {
            if read_seen.insert(binding.physical_name.clone()) {
                read_columns.push(binding.physical_name.clone());
                read_column_file_indices.push(binding.physical_root_index);
            }
        } else if let Some(default_value) = mapping.missing_default_value.as_ref() {
            if partition_seen.insert(mapping.logical_name.clone()) {
                partition_columns.push(NativePartitionColumn::new(
                    mapping.logical_name.as_str(),
                    mapping.data_type.clone(),
                    mapping.nullable,
                ));
                partition_column_values
                    .push(Some(table_format_scalar_value_to_native_literal(default_value, &mapping.data_type)?));
            }
        } else if mapping.nullable {
            if table_format_mapping_is_predicate_required(mapping, base) {
                return Err(CapabilityReason::IcebergFieldIdMappingUnsupported);
            }
            if missing_seen.insert(mapping.logical_name.clone()) {
                missing_columns.push(NativeMissingColumn::new(
                    mapping.logical_name.as_str(),
                    mapping.data_type.clone(),
                    mapping.nullable,
                ));
            }
        } else {
            return Err(CapabilityReason::IcebergFieldIdMappingUnsupported);
        }
    }

    let projected_output_column_source_refs = table_format_file_column_source_refs(
        mappings,
        file_bindings,
        footer_schema,
        &partition_columns,
        &missing_columns,
        &base.projected_output_columns,
    )?;
    let pure_filter_column_source_refs = table_format_file_column_source_refs(
        mappings,
        file_bindings,
        footer_schema,
        &partition_columns,
        &missing_columns,
        &base.pure_filter_columns,
    )?;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns: base.projected_output_columns.clone(),
        projected_output_column_source_refs,
        pure_filter_columns: base.pure_filter_columns.clone(),
        pure_filter_column_source_refs,
        predicate_columns: base.predicate_columns.clone(),
        partition_columns,
        partition_column_values,
        partition_column_indices: base.partition_column_indices.clone(),
        missing_columns,
        scan_output_columns: base.scan_output_columns.clone(),
        requires_physical_row_presence: base.requires_physical_row_presence,
    })
}

fn table_format_mapping_is_predicate_required(
    mapping: &TableFormatColumnMapping,
    requirements: &NativeParquetColumnRequirements,
) -> bool {
    requirements.predicate_columns.iter().any(|column| {
        column == &mapping.logical_name || mapping.source_name.as_ref().is_some_and(|source_name| column == source_name)
    })
}

pub(super) fn table_format_file_column_source_refs(
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
    file_bindings: &TableFormatFileReadBindings,
    footer_schema: &Schema,
    partition_columns: &[NativePartitionColumn],
    missing_columns: &[NativeMissingColumn],
    columns: &[String],
) -> Result<Vec<NativeSourceColumnRef>, CapabilityReason> {
    columns
        .iter()
        .map(|column| {
            let mapping = mappings.get(column.as_str()).ok_or(CapabilityReason::NativeSourceSchemaMapping)?;
            match mapping.source {
                TableFormatColumnSource::DirectTopLevelFileColumn => {
                    let field_id = mapping.field_id.ok_or(CapabilityReason::IcebergFieldIdMappingUnsupported)?;
                    if let Some(binding) =
                        file_bindings.bindings.iter().find(|binding| binding.logical_field_id == field_id)
                    {
                        Ok(NativeSourceColumnRef::File(binding.physical_root_index))
                    } else if mapping.missing_default_value.is_some() {
                        partition_columns
                            .iter()
                            .position(|field| field.name == mapping.logical_name)
                            .map(NativeSourceColumnRef::Partition)
                            .ok_or(CapabilityReason::NativeSourceSchemaMapping)
                    } else if mapping.nullable {
                        missing_columns
                            .iter()
                            .position(|field| field.name == mapping.logical_name)
                            .map(NativeSourceColumnRef::MissingNull)
                            .ok_or(CapabilityReason::NativeSourceSchemaMapping)
                    } else {
                        Err(CapabilityReason::IcebergFieldIdMappingUnsupported)
                    }
                },
                TableFormatColumnSource::IdentityPartitionValue => partition_columns
                    .iter()
                    .position(|field| field.name == mapping.logical_name)
                    .map(NativeSourceColumnRef::Partition)
                    .ok_or(CapabilityReason::NativeSourceSchemaMapping),
                TableFormatColumnSource::MissingNullableColumn => missing_columns
                    .iter()
                    .position(|field| field.name == mapping.logical_name)
                    .map(NativeSourceColumnRef::MissingNull)
                    .ok_or(CapabilityReason::NativeSourceSchemaMapping),
                TableFormatColumnSource::UnsupportedNestedProjection => {
                    Err(CapabilityReason::IcebergNestedProjectionUnsupported)
                },
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(|refs| {
            for source_ref in &refs {
                if let NativeSourceColumnRef::File(index) = source_ref
                    && footer_schema.fields().get(*index).is_none()
                {
                    return Err(CapabilityReason::NativeSourceSchemaMapping);
                }
            }
            Ok(refs)
        })
}

pub(super) fn table_format_source_contract_schema(
    requirements: &NativeParquetColumnRequirements,
    output_schema: &SchemaRef,
    footer_schema: &Schema,
) -> Result<SchemaRef, CapabilityReason> {
    let mut fields = Vec::new();
    let mut seen = BTreeSet::new();
    let read_fields = requirements
        .read_column_file_indices
        .iter()
        .map(|index| {
            let field = footer_schema.fields().get(*index).ok_or(CapabilityReason::NativeSourceSchemaMapping)?;
            Ok::<_, CapabilityReason>((field.name().clone(), table_format_file_read_contract_field(field)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for field in output_schema.fields() {
        if seen.insert(field.name().clone()) {
            let contract_field = read_fields
                .iter()
                .find_map(|(name, read_field)| {
                    (name == field.name()).then(|| {
                        if table_format_footer_data_type_needs_logical_source_contract(
                            read_field.data_type(),
                            field.data_type(),
                        ) {
                            field.as_ref().clone()
                        } else {
                            read_field.clone()
                        }
                    })
                })
                .unwrap_or_else(|| field.as_ref().clone());
            fields.push(contract_field);
        }
    }
    for (name, field) in read_fields {
        if seen.insert(name) {
            fields.push(field);
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn table_format_file_read_contract_field(field: &Field) -> Field {
    Field::new(field.name(), table_format_file_read_contract_data_type(field.data_type()), field.is_nullable())
}

pub(super) fn table_format_file_read_contract_data_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Timestamp(unit, _) => DataType::Timestamp(*unit, None),
        _ => data_type.clone(),
    }
}

pub(super) fn table_format_source_group_key(requirements: &NativeParquetColumnRequirements) -> String {
    let read = requirements.read_columns.join(",");
    let indices = requirements.read_column_file_indices.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    let missing = requirements.missing_columns.iter().map(|column| column.name.as_str()).collect::<Vec<_>>().join(",");
    let partition_values =
        requirements.partition_column_values.iter().map(|value| format!("{value:?}")).collect::<Vec<_>>().join(",");
    let projected_refs = source_refs_key(&requirements.projected_output_column_source_refs);
    let filter_refs = source_refs_key(&requirements.pure_filter_column_source_refs);
    format!(
        "read={read};indices={indices};missing={missing};partition_values={partition_values};projected={projected_refs};filter={filter_refs}"
    )
}

pub(super) fn source_refs_key(refs: &[NativeSourceColumnRef]) -> String {
    refs.iter()
        .map(|source_ref| match source_ref {
            NativeSourceColumnRef::File(index) => format!("f{index}"),
            NativeSourceColumnRef::Partition(index) => format!("p{index}"),
            NativeSourceColumnRef::MissingNull(index) => format!("m{index}"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn table_format_source_output_schema(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
    file_schema: &SchemaRef,
) -> Result<SchemaRef, CapabilityReason> {
    let mappings = table_format_column_mapping_by_name(scan_plan)?;
    let mut fields =
        Vec::with_capacity(requirements.projected_output_columns.len() + requirements.pure_filter_columns.len());
    let mut seen = BTreeSet::new();
    for (column, source_ref) in requirements
        .projected_output_columns
        .iter()
        .zip(requirements.projected_output_column_source_refs.iter())
        .chain(requirements.pure_filter_columns.iter().zip(requirements.pure_filter_column_source_refs.iter()))
    {
        if seen.insert(column.clone()) {
            fields.push(table_format_source_output_field(column, source_ref, requirements, file_schema, &mappings)?);
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn table_format_source_output_field(
    column: &str,
    source_ref: &NativeSourceColumnRef,
    requirements: &NativeParquetColumnRequirements,
    file_schema: &SchemaRef,
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
) -> Result<Field, CapabilityReason> {
    match source_ref {
        NativeSourceColumnRef::File(index) => {
            if let Some(mapping) = mappings.get(column) {
                return Ok(Field::new(column, mapping.data_type.clone(), mapping.nullable));
            }
            let field = file_schema.fields().get(*index).ok_or(CapabilityReason::NativeSourceSchemaMapping)?;
            Ok(Field::new(column, field.data_type().clone(), field.is_nullable()))
        },
        NativeSourceColumnRef::Partition(index) => {
            let column =
                requirements.partition_columns.get(*index).ok_or(CapabilityReason::NativeSourceSchemaMapping)?;
            Ok(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable))
        },
        NativeSourceColumnRef::MissingNull(index) => {
            let column = requirements.missing_columns.get(*index).ok_or(CapabilityReason::NativeSourceSchemaMapping)?;
            Ok(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable))
        },
    }
}
