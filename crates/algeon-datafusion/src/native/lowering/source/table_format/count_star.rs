use std::{collections::BTreeSet, sync::Arc};

use super::{
    AlgeonIcebergScanExec, CapabilityReason, DataType, ExecutionPlan, Field, LoweredNativeInput,
    NATIVE_ZERO_COLUMN_COUNT_COLUMN, NativeLiteralBatchSource, NativeLocalParquetSource,
    NativeParquetColumnRequirements, NativeQueryPlanBuilder, NativeSourceColumnRef, NativeSourceNode, Schema,
    SchemaRef, TableFormat, TableFormatColumnMapping, TableFormatColumnSource, TableFormatDataFile,
    TableFormatScanPlan, available_table_format_scan_plan, is_supported_native_type,
    table_format_arrow_data_types_match_except_list_child_names, table_format_file_schema,
    table_format_local_footer_schema, table_format_local_parquet_files, table_format_local_parquet_files_for_group,
    table_format_scan_plan_is_empty_admissible, table_format_scan_plan_rejection, table_format_source_read_groups,
    validate_schema, validate_table_format_direct_file_column_field_ids, validate_table_format_scan_plan,
};

pub(in crate::native::lowering) fn lower_table_format_zero_column_count_source(
    scan_exec: &AlgeonIcebergScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let scan_plan = available_table_format_scan_plan(scan_exec)?;
    validate_table_format_scan_plan(scan_plan)?;
    let source_node = add_table_format_zero_column_count_source_node(scan_plan, builder)?;
    Ok(LoweredNativeInput::with_bindings(source_node, scan_exec.schema(), Vec::new()))
}

pub(super) fn add_table_format_zero_column_count_source_node(
    scan_plan: &TableFormatScanPlan,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<query_engine::plan::NodeId, CapabilityReason> {
    let requirements = NativeParquetColumnRequirements {
        read_columns: Vec::new(),
        read_column_file_indices: Vec::new(),
        projected_output_columns: Vec::new(),
        projected_output_column_source_refs: Vec::new(),
        pure_filter_columns: Vec::new(),
        pure_filter_column_source_refs: Vec::new(),
        predicate_columns: Vec::new(),
        partition_columns: Vec::new(),
        partition_column_values: Vec::new(),
        partition_column_indices: Vec::new(),
        missing_columns: Vec::new(),
        scan_output_columns: Vec::new(),
        requires_physical_row_presence: false,
    };
    let native_files = table_format_local_parquet_files(scan_plan, &requirements)?;
    let source = NativeLocalParquetSource::zero_column_count(native_files).with_scan_limit(scan_plan.scan_limit);
    Ok(builder.add_source(NativeSourceNode::LocalParquet(source)))
}

pub(in crate::native::lowering) fn table_format_zero_column_count_metadata_is_available(
    scan_plan: &TableFormatScanPlan,
) -> bool {
    !scan_plan.data_files.is_empty()
        && scan_plan
            .data_files
            .iter()
            .all(|file| file.row_count.is_some() && file.row_groups.is_none() && !file.has_row_delete_files())
}

pub(in crate::native::lowering) fn table_format_count_star_mixed_row_count_metadata_can_split(
    scan_plan: &TableFormatScanPlan,
) -> bool {
    table_format_count_star_probe_can_lower(scan_plan)
        && scan_plan.data_files.iter().any(|file| file.row_count.is_some() && !file.has_row_delete_files())
        && scan_plan.data_files.iter().any(|file| file.row_count.is_none() || file.has_row_delete_files())
}

pub(in crate::native::lowering) fn table_format_count_star_probe_can_lower(scan_plan: &TableFormatScanPlan) -> bool {
    scan_plan.scan_limit.is_none()
        && !scan_plan.data_files.is_empty()
        && scan_plan.data_files.iter().all(|file| file.row_groups.is_none())
}

pub(in crate::native::lowering) fn lower_table_format_count_star_mixed_row_count_source(
    scan_exec: &AlgeonIcebergScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let scan_plan = available_table_format_scan_plan(scan_exec)?;
    if !table_format_count_star_mixed_row_count_metadata_can_split(scan_plan) {
        return Err(CapabilityReason::TableFormatUnsupported);
    }
    validate_table_format_scan_plan(scan_plan)?;

    let metadata_plan = table_format_scan_plan_with_data_files(
        scan_plan,
        scan_plan
            .data_files
            .iter()
            .filter(|file| file.row_count.is_some() && !file.has_row_delete_files())
            .cloned()
            .collect(),
    );
    let fallback_plan = table_format_scan_plan_with_data_files(
        scan_plan,
        scan_plan
            .data_files
            .iter()
            .filter(|file| file.row_count.is_none() || file.has_row_delete_files())
            .cloned()
            .collect(),
    );

    let count_weight_schema = table_format_count_weight_schema();
    let mut inputs = Vec::with_capacity(2);
    if !metadata_plan.data_files.is_empty() {
        inputs.push(add_table_format_zero_column_count_source_node(&metadata_plan, builder)?);
    }
    if !fallback_plan.data_files.is_empty() {
        let fallback_input = lower_table_format_count_star_fallback_source_for_plan(&fallback_plan, builder)?;
        inputs.push(fallback_input.node);
    }

    let node = match inputs.as_slice() {
        [] => return Err(CapabilityReason::NativeSourceEmpty),
        [node] => *node,
        _ => builder.add_union(inputs, Arc::clone(&count_weight_schema)),
    };
    Ok(LoweredNativeInput::with_bindings(node, count_weight_schema, Vec::new()))
}

pub(in crate::native::lowering) fn lower_table_format_count_star_fallback_source(
    scan_exec: &AlgeonIcebergScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let scan_plan = available_table_format_scan_plan(scan_exec)?;
    validate_table_format_scan_plan(scan_plan)?;
    lower_table_format_count_star_fallback_source_for_plan(scan_plan, builder)
}

pub(super) fn lower_table_format_count_star_fallback_source_for_plan(
    scan_plan: &TableFormatScanPlan,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let file_schema = table_format_file_schema(scan_plan);
    let fallback_column = table_format_count_star_fallback_column(scan_plan, &file_schema)?;
    if let Some(mapping) = fallback_column.direct_mapping
        && mapping.field_id.is_some()
    {
        validate_table_format_direct_file_column_field_ids(scan_plan, &[mapping])?;
    }
    let read_column = fallback_column.read_column;
    let read_column_file_index = fallback_column.read_column_file_index;
    let requirements = NativeParquetColumnRequirements {
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
        scan_output_columns: vec![read_column.clone()],
        requires_physical_row_presence: true,
    };
    let source_output_field = if let Some(mapping) = fallback_column.direct_mapping {
        Field::new(read_column.as_str(), mapping.data_type.clone(), mapping.nullable)
    } else {
        file_schema
            .fields()
            .get(read_column_file_index)
            .ok_or(CapabilityReason::NativeSourceSchemaMapping)?
            .as_ref()
            .clone()
    };
    let source_output_schema = Arc::new(Schema::new(vec![source_output_field]));
    validate_schema(Arc::clone(&source_output_schema))?;
    let read_groups = table_format_source_read_groups(scan_plan, &requirements, Arc::clone(&source_output_schema))?;
    let mut inputs = Vec::with_capacity(read_groups.len());
    for group in &read_groups {
        if group.requirements.read_columns.is_empty() {
            return Err(CapabilityReason::IcebergFieldIdMappingUnsupported);
        }
        let native_files = table_format_local_parquet_files_for_group(scan_plan, group)?;
        let source = if table_format_count_star_probe_can_lower(scan_plan) {
            NativeLocalParquetSource::count_star_probe(
                native_files,
                Arc::clone(&group.source_schema),
                group.requirements.read_columns.clone(),
            )
        } else {
            NativeLocalParquetSource::from_files(
                native_files,
                Arc::clone(&group.source_schema),
                group.requirements.read_columns.clone(),
            )
            .with_output_schema_override(Arc::clone(&group.output_schema))
        }
        .with_column_roles(Vec::new(), group.requirements.pure_filter_columns.clone())
        .with_column_source_refs(
            group.requirements.read_column_file_indices.clone(),
            group.requirements.projected_output_column_source_refs.clone(),
            group.requirements.pure_filter_column_source_refs.clone(),
        )
        .with_file_column_count(file_schema.fields().len())
        .with_scan_limit(scan_plan.scan_limit);
        inputs.push(builder.add_source(NativeSourceNode::LocalParquet(source)));
    }
    let output_schema = if table_format_count_star_probe_can_lower(scan_plan) {
        table_format_count_weight_schema()
    } else {
        source_output_schema
    };
    let source_node = match inputs.as_slice() {
        [] => return Err(CapabilityReason::NativeSourceEmpty),
        [node] => *node,
        _ => builder.add_union(inputs, Arc::clone(&output_schema)),
    };
    Ok(LoweredNativeInput::with_bindings(source_node, output_schema, Vec::new()))
}

pub(super) fn table_format_scan_plan_with_data_files(
    scan_plan: &TableFormatScanPlan,
    data_files: Vec<TableFormatDataFile>,
) -> TableFormatScanPlan {
    let mut plan = scan_plan.clone();
    let data_file_ids = data_files.iter().map(|file| file.id.as_str()).collect::<BTreeSet<_>>();
    plan.delete_files.retain(|delete_file| data_file_ids.contains(delete_file.data_file_id.as_str()));
    plan.data_files = data_files;
    plan
}

pub(super) fn table_format_count_weight_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(NATIVE_ZERO_COLUMN_COUNT_COLUMN, DataType::Int64, false)]))
}

pub(in crate::native::lowering) fn lower_empty_table_format_count_star_source(
    scan_exec: &AlgeonIcebergScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let scan_plan = available_table_format_scan_plan(scan_exec)?;
    if !table_format_scan_plan_is_empty_admissible(scan_plan) {
        return Err(table_format_scan_plan_rejection(scan_plan));
    }
    let schema = scan_exec.schema();
    validate_schema(Arc::clone(&schema))?;
    let source_node =
        builder.add_source(NativeSourceNode::LiteralBatch(NativeLiteralBatchSource::empty(Arc::clone(&schema))));
    Ok(LoweredNativeInput::with_bindings(source_node, schema, Vec::new()))
}

pub(super) struct TableFormatCountStarFallbackColumn<'a> {
    pub(super) read_column: String,
    pub(super) read_column_file_index: usize,
    pub(super) direct_mapping: Option<&'a TableFormatColumnMapping>,
}

pub(super) fn table_format_count_star_fallback_column<'a>(
    scan_plan: &'a TableFormatScanPlan,
    file_schema: &SchemaRef,
) -> Result<TableFormatCountStarFallbackColumn<'a>, CapabilityReason> {
    if let Some(column) = table_format_explicit_count_star_probe_column(scan_plan, file_schema) {
        return Ok(column);
    }
    if table_format_local_footer_schema(scan_plan).is_some() {
        if scan_plan.format == TableFormat::Iceberg {
            return Err(CapabilityReason::IcebergFieldIdMappingUnsupported);
        }
        return table_format_first_supported_file_column(file_schema);
    }
    Err(CapabilityReason::NativeSourceSchemaMapping)
}

pub(super) fn table_format_explicit_count_star_probe_column<'a>(
    scan_plan: &'a TableFormatScanPlan,
    file_schema: &SchemaRef,
) -> Option<TableFormatCountStarFallbackColumn<'a>> {
    scan_plan
        .pure_filter_columns
        .iter()
        .chain(scan_plan.projected_columns.iter())
        .filter(|mapping| mapping.source == TableFormatColumnSource::DirectTopLevelFileColumn)
        .filter_map(|mapping| {
            let index = table_format_count_star_probe_candidate_file_index(mapping, file_schema);
            let field_supported = index.is_some();
            let field_id_supported = mapping.field_id.is_some() && is_supported_native_type(&mapping.data_type);
            (field_supported || field_id_supported).then(|| TableFormatCountStarFallbackColumn {
                read_column: mapping.logical_name.clone(),
                read_column_file_index: index.unwrap_or(0),
                direct_mapping: Some(mapping),
            })
        })
        .next()
}

fn table_format_count_star_probe_candidate_file_index(
    mapping: &TableFormatColumnMapping,
    file_schema: &SchemaRef,
) -> Option<usize> {
    let compatible_indices = mapping
        .field_id_physical_name_candidates()
        .filter_map(|name| {
            let index = file_schema.index_of(name).ok()?;
            let field = file_schema.fields().get(index)?;
            (table_format_arrow_data_types_match_except_list_child_names(field.data_type(), &mapping.data_type)
                && is_supported_native_type(field.data_type()))
            .then_some(index)
        })
        .collect::<BTreeSet<_>>();
    let mut indices = compatible_indices.into_iter();
    let index = indices.next()?;
    indices.next().is_none().then_some(index)
}

pub(super) fn table_format_first_supported_file_column(
    file_schema: &SchemaRef,
) -> Result<TableFormatCountStarFallbackColumn<'static>, CapabilityReason> {
    file_schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_index, field)| is_supported_native_type(field.data_type()))
        .map(|(index, field)| TableFormatCountStarFallbackColumn {
            read_column: field.name().clone(),
            read_column_file_index: index,
            direct_mapping: None,
        })
        .ok_or(CapabilityReason::UnsupportedType)
}
