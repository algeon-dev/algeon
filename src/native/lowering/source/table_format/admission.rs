use super::*;

pub(in crate::native::lowering) fn table_format_scan_rejection(source: &NormalizedNativeSource<'_>) -> FallbackReason {
    let Some(scan_exec) = source.table_format_scan_exec() else {
        return FallbackReason::TableFormatUnsupported;
    };
    scan_exec
        .scan_plan()
        .unsupported_fallback_reasons()
        .into_iter()
        .next()
        .unwrap_or_else(|| table_format_scan_plan_rejection(scan_exec.scan_plan()))
}

pub(super) fn validate_table_format_scan_plan(plan: &TableFormatScanPlan) -> Result<(), FallbackReason> {
    match table_format_scan_plan_rejection(plan) {
        FallbackReason::TableFormatUnsupported if plan.is_native_admissible() => Ok(()),
        FallbackReason::TableFormatUnsupported => Err(FallbackReason::TableFormatUnsupported),
        reason => Err(reason),
    }
}

pub(super) fn table_format_scan_plan_rejection(plan: &TableFormatScanPlan) -> FallbackReason {
    plan.first_rejection_reason().unwrap_or(FallbackReason::TableFormatUnsupported)
}

pub(super) fn table_format_scan_plan_is_empty_admissible(plan: &TableFormatScanPlan) -> bool {
    plan.unsupported_features.is_empty() && plan.delete_files.is_empty() && plan.data_files.is_empty()
}

pub(super) fn table_format_column_requirements(
    scan_plan: &TableFormatScanPlan,
    file_schema: &SchemaRef,
    scan_schema: SchemaRef,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    required_output_fields: &RequiredFields,
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let required_scan_output_indices =
        required_source_scan_output_indices(scan_schema.clone(), filter, required_output_fields)?;
    let output_columns = required_scan_output_indices
        .iter()
        .map(|index| {
            scan_schema
                .fields()
                .get(*index)
                .map(|field| field.name().clone())
                .ok_or(FallbackReason::UnsupportedExpression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut filter_columns: BTreeSet<String> = BTreeSet::new();
    if let Some(filter) = filter {
        collect_required_columns(filter.predicate(), &filter.input().schema(), &mut filter_columns)?;
    }

    let output_column_set: BTreeSet<String> = output_columns.iter().cloned().collect();
    let predicate_columns =
        table_format_predicate_scan_output_indices(scan_schema.clone(), filter, required_output_fields)?
            .into_iter()
            .map(|index| {
                scan_schema
                    .fields()
                    .get(index)
                    .map(|field| field.name().clone())
                    .ok_or(FallbackReason::UnsupportedExpression)
            })
            .chain(filter_columns.iter().cloned().map(Ok))
            .collect::<Result<BTreeSet<_>, _>>()?;

    // Iterate required columns in scan-projection order so the read/partition/
    // missing lists below preserve that order. The `BTreeSet`-driven order used
    // previously produced an alphabetical source schema, which then failed
    // DataFusion's optimizer-rule schema check against `scan_schema`. Output
    // columns lead (in scan-projection order); filter-only columns follow
    // (in `BTreeSet` iteration order - they never appear in `scan_schema` and
    // their relative order does not affect any schema check).
    let mut ordered_required_columns: Vec<String> = Vec::with_capacity(output_columns.len() + filter_columns.len());
    let mut required_seen: BTreeSet<String> = BTreeSet::new();
    for column in output_columns.iter().chain(filter_columns.iter()) {
        if required_seen.insert(column.clone()) {
            ordered_required_columns.push(column.clone());
        }
    }

    if ordered_required_columns.is_empty() {
        return table_format_row_presence_carrier_requirements(scan_plan, file_schema);
    }

    let mappings = table_format_column_mapping_by_name(scan_plan)?;
    let ordered_required_mappings = ordered_required_columns
        .iter()
        .map(|column| mappings.get(column.as_str()).copied().ok_or(FallbackReason::NativeSourceSchemaMapping))
        .collect::<Result<Vec<_>, _>>()?;
    let mut read_columns: Vec<String> = Vec::new();
    let mut read_columns_seen: BTreeSet<String> = BTreeSet::new();
    let mut partition_columns: Vec<NativePartitionColumn> = Vec::new();
    let mut partition_column_values: Vec<Option<NativeLiteralValue>> = Vec::new();
    let mut partition_columns_seen: BTreeSet<String> = BTreeSet::new();
    let mut missing_columns: Vec<NativeMissingColumn> = Vec::new();
    let mut missing_columns_seen: BTreeSet<String> = BTreeSet::new();
    for mapping in &ordered_required_mappings {
        match mapping.source {
            TableFormatColumnSource::DirectTopLevelFileColumn => {
                let source_name = mapping.source_name.as_ref().unwrap_or(&mapping.logical_name);
                if file_schema.index_of(source_name).is_err()
                    && !(scan_plan.format == TableFormat::Iceberg && mapping.field_id.is_some())
                {
                    return Err(FallbackReason::NativeSourceSchemaMapping);
                }
                if read_columns_seen.insert(mapping.logical_name.clone()) {
                    read_columns.push(mapping.logical_name.clone());
                }
            },
            TableFormatColumnSource::IdentityPartitionValue => {
                if !is_supported_native_type(&mapping.data_type) {
                    return Err(FallbackReason::UnsupportedType);
                }
                if partition_columns_seen.insert(mapping.logical_name.clone()) {
                    partition_columns.push(NativePartitionColumn::new(
                        mapping.logical_name.as_str(),
                        mapping.data_type.clone(),
                        mapping.nullable,
                    ));
                    partition_column_values.push(None);
                }
            },
            TableFormatColumnSource::MissingNullableColumn => {
                if !mapping.nullable {
                    return Err(FallbackReason::NativeSourceSchemaMapping);
                }
                if !is_supported_native_type(&mapping.data_type) {
                    return Err(FallbackReason::UnsupportedType);
                }
                if is_list_carry_type(&mapping.data_type) {
                    return Err(FallbackReason::UnsupportedType);
                }
                if missing_columns_seen.insert(mapping.logical_name.clone()) {
                    missing_columns.push(NativeMissingColumn::new(
                        mapping.logical_name.as_str(),
                        mapping.data_type.clone(),
                        mapping.nullable,
                    ));
                }
            },
            TableFormatColumnSource::UnsupportedNestedProjection => {
                return Err(FallbackReason::IcebergNestedProjectionUnsupported);
            },
        }
    }

    let pure_filter_columns = filter_columns.difference(&output_column_set).cloned().collect::<Vec<_>>();
    let projected_output_columns = output_columns;
    let read_column_file_indices = read_columns
        .iter()
        .map(|column| {
            file_schema.index_of(column).or_else(|_error| {
                let mapping = mappings.get(column.as_str()).ok_or(())?;
                if scan_plan.format == TableFormat::Iceberg && mapping.field_id.is_some() { Ok(0) } else { Err(()) }
            })
        })
        .map(|result| result.map_err(|_error| FallbackReason::NativeSourceSchemaMapping))
        .collect::<Result<Vec<_>, _>>()?;
    let projected_output_column_source_refs = table_format_column_source_refs(
        scan_plan,
        &mappings,
        file_schema,
        &partition_columns,
        &missing_columns,
        &projected_output_columns,
    )?;
    let pure_filter_column_source_refs = table_format_column_source_refs(
        scan_plan,
        &mappings,
        file_schema,
        &partition_columns,
        &missing_columns,
        &pure_filter_columns,
    )?;
    let scan_output_columns = ordered_required_columns;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns,
        projected_output_column_source_refs,
        pure_filter_columns,
        pure_filter_column_source_refs,
        predicate_columns: predicate_columns.into_iter().collect(),
        partition_columns,
        partition_column_values,
        partition_column_indices: Vec::new(),
        missing_columns,
        scan_output_columns,
        requires_physical_row_presence: false,
    })
}

fn table_format_predicate_scan_output_indices(
    scan_schema: SchemaRef,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    required_output_fields: &RequiredFields,
) -> Result<Vec<usize>, FallbackReason> {
    if let Some(filter) = filter
        && let Some(projection) = filter.projection()
    {
        return required_output_fields
            .predicate_indices(filter.schema().fields().len())?
            .into_iter()
            .map(|output_index| projection.get(output_index).copied().ok_or(FallbackReason::UnsupportedExpression))
            .collect();
    }
    required_output_fields.predicate_indices(scan_schema.fields().len())
}

pub(super) fn table_format_row_presence_carrier_requirements(
    scan_plan: &TableFormatScanPlan,
    file_schema: &SchemaRef,
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let carrier = table_format_count_star_fallback_column(scan_plan, file_schema)?;
    let read_column = carrier.read_column;
    let read_column_file_index = carrier.read_column_file_index;
    Ok(NativeParquetColumnRequirements {
        read_columns: vec![read_column.clone()],
        read_column_file_indices: vec![read_column_file_index],
        projected_output_columns: Vec::new(),
        projected_output_column_source_refs: Vec::new(),
        pure_filter_columns: vec![read_column.clone()],
        pure_filter_column_source_refs: vec![NativeSourceColumnRef::File(read_column_file_index)],
        predicate_columns: Vec::new(),
        partition_columns: Vec::new(),
        partition_column_values: Vec::new(),
        partition_column_indices: Vec::new(),
        missing_columns: Vec::new(),
        scan_output_columns: vec![read_column],
        requires_physical_row_presence: true,
    })
}

pub(super) fn table_format_column_mapping_by_name(
    scan_plan: &TableFormatScanPlan,
) -> Result<BTreeMap<&str, &TableFormatColumnMapping>, FallbackReason> {
    let mut mappings = BTreeMap::new();
    for mapping in scan_plan.projected_columns.iter().chain(scan_plan.pure_filter_columns.iter()) {
        for key in [Some(mapping.logical_name.as_str()), mapping.source_name.as_deref()].into_iter().flatten() {
            if let Some(existing) = mappings.get(key)
                && *existing != mapping
            {
                return Err(FallbackReason::NativeSourceSchemaMapping);
            }
            mappings.insert(key, mapping);
        }
    }
    Ok(mappings)
}

pub(super) fn table_format_column_source_refs(
    scan_plan: &TableFormatScanPlan,
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
    file_schema: &SchemaRef,
    partition_columns: &[NativePartitionColumn],
    missing_columns: &[NativeMissingColumn],
    columns: &[String],
) -> Result<Vec<NativeSourceColumnRef>, FallbackReason> {
    columns
        .iter()
        .map(|column| {
            let mapping = mappings.get(column.as_str()).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
            match mapping.source {
                TableFormatColumnSource::DirectTopLevelFileColumn => {
                    let source_name = mapping.source_name.as_ref().unwrap_or(&mapping.logical_name);
                    match file_schema.index_of(source_name) {
                        Ok(index) => Ok(NativeSourceColumnRef::File(index)),
                        Err(_) if scan_plan.format == TableFormat::Iceberg && mapping.field_id.is_some() => {
                            Ok(NativeSourceColumnRef::File(0))
                        },
                        Err(_) => Err(FallbackReason::NativeSourceSchemaMapping),
                    }
                },
                TableFormatColumnSource::IdentityPartitionValue => partition_columns
                    .iter()
                    .position(|field| field.name == mapping.logical_name)
                    .map(NativeSourceColumnRef::Partition)
                    .ok_or(FallbackReason::NativeSourceSchemaMapping),
                TableFormatColumnSource::MissingNullableColumn => missing_columns
                    .iter()
                    .position(|field| field.name == mapping.logical_name)
                    .map(NativeSourceColumnRef::MissingNull)
                    .ok_or(FallbackReason::NativeSourceSchemaMapping),
                TableFormatColumnSource::UnsupportedNestedProjection => {
                    Err(FallbackReason::IcebergNestedProjectionUnsupported)
                },
            }
        })
        .collect()
}
