use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::ExecutionPlan;
use nexus_query_engine::{
    Expr as NativeExpr, LimitSpec as NativeLimitSpec, LiteralBatchSource as NativeLiteralBatchSource,
    LiteralValue as NativeLiteralValue, LocalParquetFile as NativeLocalParquetFile,
    LocalParquetSource as NativeLocalParquetSource, MissingColumn as NativeMissingColumn,
    ParquetFieldIdReadBinding as NativeParquetFieldIdReadBinding, PartitionColumn as NativePartitionColumn,
    QueryPlanBuilder as NativeQueryPlanBuilder, SourceColumnRef as NativeSourceColumnRef,
    SourceNode as NativeSourceNode, ZERO_COLUMN_COUNT_COLUMN as NATIVE_ZERO_COLUMN_COUNT_COLUMN,
};
use parquet::{arrow::PARQUET_FIELD_ID_META_KEY, file::reader::FileReader};

use super::{
    SourceReaderPushdownPredicate, add_source_filter_projection, combine_scan_limits, project_scan_output_columns,
    required_source_scan_output_indices, source_bindings_for_schema, supported_reader_pushdown_predicate,
};
use crate::{
    native::{
        lowering::{LoweredNativeInput, RequiredFields, expr::lower_physical_expr_with_bindings},
        normalize::NormalizedNativeSource,
        source::{NativeParquetColumnRequirements, attach_row_group_byte_metadata},
        validate::{collect_required_columns, is_supported_native_type, validate_physical_expr, validate_schema},
    },
    policy::FallbackReason,
    source_cardinality::reader_predicate_proves_source_at_most_one_row,
    table_format::{
        CudfTableFormatScanExec, TableFormat, TableFormatColumnMapping, TableFormatColumnSource, TableFormatDataFile,
        TableFormatDataFileFormat, TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata,
        TableFormatPartitionTransform, TableFormatScalarValue, TableFormatScanPlan,
        resolve_table_format_parquet_source_with_metadata,
    },
};

const TABLE_FORMAT_FILTER_ELISION_MIN_SOURCE_ROWS: u64 = 10_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TableFormatFieldIdReadBinding {
    logical_field_id: i32,
    logical_name: String,
    physical_name: String,
    physical_root_index: usize,
    nullable: bool,
    data_type: DataType,
}

impl TableFormatFieldIdReadBinding {
    fn native_binding(&self) -> NativeParquetFieldIdReadBinding {
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
struct TableFormatFileReadBindings {
    bindings: Vec<TableFormatFieldIdReadBinding>,
}

#[derive(Debug, Clone)]
struct TableFormatSourceReadGroup {
    file_indices: Vec<usize>,
    requirements: NativeParquetColumnRequirements,
    source_schema: SchemaRef,
    output_schema: SchemaRef,
    field_id_bindings: Vec<Vec<TableFormatFieldIdReadBinding>>,
}

impl TableFormatSourceReadGroup {
    fn has_field_id_aliases(&self) -> bool {
        self.field_id_bindings.iter().flatten().any(|binding| binding.logical_name != binding.physical_name)
    }
}

pub(super) fn lower_table_format_source_input(
    scan_exec: &CudfTableFormatScanExec,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    pushed_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, FallbackReason> {
    let scan_plan = scan_exec.scan_plan();
    if table_format_scan_plan_is_empty_admissible(scan_plan) {
        return lower_empty_table_format_source_input(scan_exec, filter, builder, required_output_fields);
    }
    validate_table_format_scan_plan(scan_plan)?;
    validate_schema(scan_exec.schema())?;
    if let Some(filter) = filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }

    let file_schema = table_format_file_schema(scan_plan);
    let requirements =
        table_format_column_requirements(scan_plan, &file_schema, scan_exec.schema(), filter, required_output_fields)?;
    let source_output_schema = table_format_source_output_schema(scan_plan, &requirements, &file_schema)?;
    validate_schema(Arc::clone(&source_output_schema))?;
    let read_groups = table_format_source_read_groups(scan_plan, &requirements, Arc::clone(&source_output_schema))?;
    let source_scan_limit =
        if filter.is_none() { combine_scan_limits(scan_plan.scan_limit, pushed_scan_limit) } else { None };
    let (scan_output, reader_pushdown_covers_filter) = if let [group] = read_groups.as_slice() {
        let (native_source, reader_pushdown_covers_filter) = table_format_native_source_for_group(
            scan_plan,
            group,
            filter,
            &file_schema,
            scan_exec.schema(),
            source_scan_limit,
        )?;
        let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
        let scan_output_bindings = source_bindings_for_schema(&source_output_schema, scan_exec.schema());
        (
            LoweredNativeInput::with_bindings(source_node, Arc::clone(&source_output_schema), scan_output_bindings),
            reader_pushdown_covers_filter,
        )
    } else {
        let inputs = read_groups
            .iter()
            .map(|group| {
                table_format_native_source_for_group(scan_plan, group, None, &file_schema, scan_exec.schema(), None)
                    .map(|(source, _pushdown)| builder.add_source(NativeSourceNode::LocalParquet(source)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let union_node = builder.add_union(inputs, Arc::clone(&source_output_schema));
        let scan_output_bindings = source_bindings_for_schema(&source_output_schema, scan_exec.schema());
        (LoweredNativeInput::with_bindings(union_node, Arc::clone(&source_output_schema), scan_output_bindings), false)
    };
    if let Some(filter) = filter {
        if reader_pushdown_covers_filter {
            if let Some(projection) = filter.projection() {
                return add_source_filter_projection(
                    builder,
                    scan_output.node,
                    &scan_output,
                    filter.schema(),
                    projection,
                    required_output_fields,
                );
            }
            return project_scan_output_columns(
                builder,
                scan_output.node,
                scan_output.schema,
                scan_output.bindings,
                required_output_fields,
            );
        }
        let filtered_node = builder.add_filter(
            scan_output.node,
            lower_physical_expr_with_bindings(
                filter.predicate(),
                &filter.input().schema(),
                Some(&scan_output.bindings),
            )?,
            Arc::clone(&scan_output.schema),
        );
        let filtered = if let Some(projection) = filter.projection() {
            add_source_filter_projection(
                builder,
                filtered_node,
                &scan_output,
                filter.schema(),
                projection,
                required_output_fields,
            )
        } else {
            project_scan_output_columns(
                builder,
                filtered_node,
                scan_output.schema,
                scan_output.bindings,
                required_output_fields,
            )
        }?;
        if let Some(scan_limit) = scan_plan.scan_limit {
            let limited_node =
                builder.add_limit(filtered.node, NativeLimitSpec::fetch(scan_limit), Arc::clone(&filtered.schema));
            Ok(LoweredNativeInput::with_bindings(limited_node, filtered.schema, filtered.bindings))
        } else {
            Ok(filtered)
        }
    } else {
        let scan_output = if read_groups.len() > 1
            && let Some(scan_limit) = source_scan_limit
        {
            let limited_node = builder.add_limit(
                scan_output.node,
                NativeLimitSpec::fetch(scan_limit),
                Arc::clone(&scan_output.schema),
            );
            LoweredNativeInput::with_bindings(limited_node, scan_output.schema, scan_output.bindings)
        } else {
            scan_output
        };
        Ok(scan_output)
    }
}

fn table_format_native_source_for_group(
    scan_plan: &TableFormatScanPlan,
    group: &TableFormatSourceReadGroup,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    file_schema: &SchemaRef,
    frontend_schema: SchemaRef,
    source_scan_limit: Option<usize>,
) -> Result<(NativeLocalParquetSource, bool), FallbackReason> {
    let native_files = table_format_local_parquet_files_for_group(scan_plan, group)?;
    let native_file_paths =
        native_files.iter().filter_map(|file| file.location.as_local_path().map(Path::to_path_buf)).collect::<Vec<_>>();
    let mut native_source = NativeLocalParquetSource::from_files(
        native_files,
        Arc::clone(&group.source_schema),
        group.requirements.read_columns.clone(),
    )
    .with_column_roles(
        group.requirements.projected_output_columns.clone(),
        group.requirements.pure_filter_columns.clone(),
    )
    .with_column_source_refs(
        group.requirements.read_column_file_indices.clone(),
        group.requirements.projected_output_column_source_refs.clone(),
        group.requirements.pure_filter_column_source_refs.clone(),
    )
    .with_output_schema_override(Arc::clone(&group.output_schema))
    .with_file_column_count(file_schema.fields().len())
    .with_partition_columns(group.requirements.partition_columns.clone())
    .with_missing_columns(group.requirements.missing_columns.clone())
    .with_scan_limit(source_scan_limit);

    let mut reader_pushdown_covers_filter = false;
    if !group.has_field_id_aliases()
        && let Some(predicate) = table_format_reader_pushdown_predicate(
            filter,
            &group.output_schema,
            Arc::clone(&frontend_schema),
            &native_source,
        )?
    {
        reader_pushdown_covers_filter = predicate.covers_filter;
        let max_rows = if reader_predicate_proves_table_format_source_at_most_one_row(
            scan_plan,
            file_schema,
            &group.output_schema,
            frontend_schema.as_ref(),
            &native_file_paths.iter().map(|path| path.as_path()).collect::<Vec<_>>(),
            &predicate.expr,
        ) {
            Some(1)
        } else {
            None
        };
        native_source = native_source.with_reader_predicate(predicate.expr).with_max_rows(max_rows);
    }

    Ok((native_source, reader_pushdown_covers_filter))
}

fn lower_empty_table_format_source_input(
    scan_exec: &CudfTableFormatScanExec,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let source_schema = filter.map_or_else(|| scan_exec.schema(), |filter| filter.schema());
    validate_schema(Arc::clone(&source_schema))?;
    let source_node =
        builder.add_source(NativeSourceNode::LiteralBatch(NativeLiteralBatchSource::empty(Arc::clone(&source_schema))));
    let source_bindings = source_bindings_for_schema(&source_schema, Arc::clone(&source_schema));
    project_scan_output_columns(builder, source_node, source_schema, source_bindings, required_output_fields)
}

fn reader_predicate_proves_table_format_source_at_most_one_row(
    scan_plan: &TableFormatScanPlan,
    file_schema: &Schema,
    source_output_schema: &Schema,
    frontend_schema: &Schema,
    source_paths: &[&Path],
    predicate: &NativeExpr,
) -> bool {
    reader_predicate_proves_source_at_most_one_row(
        &[
            scan_plan.logical_schema.as_ref(),
            scan_plan.native_source_schema.as_ref(),
            file_schema,
            source_output_schema,
            frontend_schema,
        ],
        source_paths,
        predicate,
    )
}

fn table_format_reader_pushdown_predicate(
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    source_output_schema: &SchemaRef,
    frontend_schema: SchemaRef,
    native_source: &NativeLocalParquetSource,
) -> Result<Option<SourceReaderPushdownPredicate>, FallbackReason> {
    let Some(filter) = filter else {
        return Ok(None);
    };
    let bindings = source_bindings_for_schema(source_output_schema, frontend_schema);
    let predicate =
        match lower_physical_expr_with_bindings(filter.predicate(), &filter.input().schema(), Some(&bindings)) {
            Ok(predicate) => predicate,
            Err(_) => return Ok(None),
        };
    if let Some(predicate) = supported_reader_pushdown_predicate(&predicate, native_source, |expr| {
        !table_format_reader_predicate_references_stats_filter_risky_type(expr, source_output_schema)
    }) {
        Ok(Some(SourceReaderPushdownPredicate {
            covers_filter: predicate.covers_predicate
                && table_format_reader_pushdown_can_elide_native_filter(native_source),
            expr: predicate.expr,
        }))
    } else {
        Ok(None)
    }
}

fn table_format_reader_pushdown_can_elide_native_filter(native_source: &NativeLocalParquetSource) -> bool {
    table_format_source_known_rows(native_source)
        .is_some_and(|rows| rows >= TABLE_FORMAT_FILTER_ELISION_MIN_SOURCE_ROWS)
}

fn table_format_source_known_rows(native_source: &NativeLocalParquetSource) -> Option<u64> {
    native_source.files.iter().try_fold(0u64, |total, file| {
        let rows =
            file.row_group_rows.as_ref()?.iter().try_fold(0u64, |file_total, rows| file_total.checked_add(*rows))?;
        total.checked_add(rows)
    })
}

fn table_format_reader_predicate_references_stats_filter_risky_type(expr: &NativeExpr, schema: &Schema) -> bool {
    match expr {
        NativeExpr::Column(column) => schema
            .field_with_name(column.name())
            .is_ok_and(|field| table_format_reader_predicate_stats_filter_risky_type(field.data_type())),
        NativeExpr::Literal(_) => false,
        NativeExpr::Not(expr)
        | NativeExpr::IsNull(expr)
        | NativeExpr::IsNotNull(expr)
        | NativeExpr::Lower(expr)
        | NativeExpr::Upper(expr)
        | NativeExpr::Abs(expr) => table_format_reader_predicate_references_stats_filter_risky_type(expr, schema),
        NativeExpr::Cast { expr, .. }
        | NativeExpr::TryCast { expr, .. }
        | NativeExpr::Like { expr, .. }
        | NativeExpr::Substring { expr, .. }
        | NativeExpr::StringLength { expr, .. }
        | NativeExpr::Sqrt { expr, .. }
        | NativeExpr::Round { expr, .. }
        | NativeExpr::Extract { expr, .. }
        | NativeExpr::DateTrunc { expr, .. }
        | NativeExpr::Alias { expr, .. } => {
            table_format_reader_predicate_references_stats_filter_risky_type(expr, schema)
        },
        NativeExpr::Binary { left, right, .. }
        | NativeExpr::IsDistinctFrom { left, right }
        | NativeExpr::IsNotDistinctFrom { left, right } => {
            table_format_reader_predicate_references_stats_filter_risky_type(left, schema)
                || table_format_reader_predicate_references_stats_filter_risky_type(right, schema)
        },
        NativeExpr::InList { expr, list, .. } => {
            table_format_reader_predicate_references_stats_filter_risky_type(expr, schema)
                || list
                    .iter()
                    .any(|item| table_format_reader_predicate_references_stats_filter_risky_type(item, schema))
        },
        NativeExpr::Case { branches, else_expr } => {
            branches.iter().any(|branch| {
                table_format_reader_predicate_references_stats_filter_risky_type(&branch.when, schema)
                    || table_format_reader_predicate_references_stats_filter_risky_type(&branch.then, schema)
            }) || table_format_reader_predicate_references_stats_filter_risky_type(else_expr, schema)
        },
        NativeExpr::Concat(exprs) => {
            exprs.iter().any(|expr| table_format_reader_predicate_references_stats_filter_risky_type(expr, schema))
        },
        NativeExpr::RegexpReplace { expr, .. } => {
            table_format_reader_predicate_references_stats_filter_risky_type(expr, schema)
        },
    }
}

fn table_format_reader_predicate_stats_filter_risky_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

pub(in crate::native::lowering) fn lower_table_format_zero_column_count_source(
    scan_exec: &CudfTableFormatScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, FallbackReason> {
    let scan_plan = scan_exec.scan_plan();
    validate_table_format_scan_plan(scan_plan)?;
    let source_node = add_table_format_zero_column_count_source_node(scan_plan, builder)?;
    Ok(LoweredNativeInput::with_bindings(source_node, scan_exec.schema(), Vec::new()))
}

fn add_table_format_zero_column_count_source_node(
    scan_plan: &TableFormatScanPlan,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<nexus_query_engine::NodeId, FallbackReason> {
    let requirements = NativeParquetColumnRequirements {
        read_columns: Vec::new(),
        read_column_file_indices: Vec::new(),
        projected_output_columns: Vec::new(),
        projected_output_column_source_refs: Vec::new(),
        pure_filter_columns: Vec::new(),
        pure_filter_column_source_refs: Vec::new(),
        partition_columns: Vec::new(),
        partition_column_values: Vec::new(),
        partition_column_indices: Vec::new(),
        missing_columns: Vec::new(),
        scan_output_columns: Vec::new(),
    };
    let native_files = table_format_local_parquet_files(scan_plan, &requirements)?;
    let source = NativeLocalParquetSource::zero_column_count(native_files).with_scan_limit(scan_plan.scan_limit);
    Ok(builder.add_source(NativeSourceNode::LocalParquet(source)))
}

pub(in crate::native::lowering) fn table_format_zero_column_count_metadata_is_available(
    scan_plan: &TableFormatScanPlan,
) -> bool {
    !scan_plan.data_files.is_empty()
        && scan_plan.data_files.iter().all(|file| file.row_count.is_some() && file.row_groups.is_none())
}

pub(in crate::native::lowering) fn table_format_count_star_mixed_row_count_metadata_can_split(
    scan_plan: &TableFormatScanPlan,
) -> bool {
    table_format_count_star_probe_can_lower(scan_plan)
        && scan_plan.data_files.iter().any(|file| file.row_count.is_some())
        && scan_plan.data_files.iter().any(|file| file.row_count.is_none())
}

pub(in crate::native::lowering) fn table_format_count_star_probe_can_lower(scan_plan: &TableFormatScanPlan) -> bool {
    scan_plan.scan_limit.is_none()
        && !scan_plan.data_files.is_empty()
        && scan_plan.data_files.iter().all(|file| file.row_groups.is_none())
}

pub(in crate::native::lowering) fn lower_table_format_count_star_mixed_row_count_source(
    scan_exec: &CudfTableFormatScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, FallbackReason> {
    let scan_plan = scan_exec.scan_plan();
    if !table_format_count_star_mixed_row_count_metadata_can_split(scan_plan) {
        return Err(FallbackReason::TableFormatUnsupported);
    }
    validate_table_format_scan_plan(scan_plan)?;

    let metadata_plan = table_format_scan_plan_with_data_files(
        scan_plan,
        scan_plan.data_files.iter().filter(|file| file.row_count.is_some()).cloned().collect(),
    );
    let fallback_plan = table_format_scan_plan_with_data_files(
        scan_plan,
        scan_plan.data_files.iter().filter(|file| file.row_count.is_none()).cloned().collect(),
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
        [] => return Err(FallbackReason::NativeSourceEmpty),
        [node] => *node,
        _ => builder.add_union(inputs, Arc::clone(&count_weight_schema)),
    };
    Ok(LoweredNativeInput::with_bindings(node, count_weight_schema, Vec::new()))
}

pub(in crate::native::lowering) fn lower_table_format_count_star_fallback_source(
    scan_exec: &CudfTableFormatScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, FallbackReason> {
    let scan_plan = scan_exec.scan_plan();
    validate_table_format_scan_plan(scan_plan)?;
    lower_table_format_count_star_fallback_source_for_plan(scan_plan, builder)
}

fn lower_table_format_count_star_fallback_source_for_plan(
    scan_plan: &TableFormatScanPlan,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, FallbackReason> {
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
        partition_columns: Vec::new(),
        partition_column_values: Vec::new(),
        partition_column_indices: Vec::new(),
        missing_columns: Vec::new(),
        scan_output_columns: vec![read_column.clone()],
    };
    let source_output_field = if let Some(mapping) = fallback_column.direct_mapping {
        Field::new(read_column.as_str(), mapping.data_type.clone(), mapping.nullable)
    } else {
        file_schema
            .fields()
            .get(read_column_file_index)
            .ok_or(FallbackReason::NativeSourceSchemaMapping)?
            .as_ref()
            .clone()
    };
    let source_output_schema = Arc::new(Schema::new(vec![source_output_field]));
    validate_schema(Arc::clone(&source_output_schema))?;
    let read_groups = table_format_source_read_groups(scan_plan, &requirements, Arc::clone(&source_output_schema))?;
    let mut inputs = Vec::with_capacity(read_groups.len());
    for group in &read_groups {
        if group.requirements.read_columns.is_empty() {
            return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
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
        [] => return Err(FallbackReason::NativeSourceEmpty),
        [node] => *node,
        _ => builder.add_union(inputs, Arc::clone(&output_schema)),
    };
    Ok(LoweredNativeInput::with_bindings(source_node, output_schema, Vec::new()))
}

fn table_format_scan_plan_with_data_files(
    scan_plan: &TableFormatScanPlan,
    data_files: Vec<TableFormatDataFile>,
) -> TableFormatScanPlan {
    let mut plan = scan_plan.clone();
    plan.data_files = data_files;
    plan
}

fn table_format_count_weight_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(NATIVE_ZERO_COLUMN_COUNT_COLUMN, DataType::Int64, false)]))
}

pub(in crate::native::lowering) fn lower_empty_table_format_count_star_source(
    scan_exec: &CudfTableFormatScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, FallbackReason> {
    if !table_format_scan_plan_is_empty_admissible(scan_exec.scan_plan()) {
        return Err(table_format_scan_plan_rejection(scan_exec.scan_plan()));
    }
    let schema = scan_exec.schema();
    validate_schema(Arc::clone(&schema))?;
    let source_node =
        builder.add_source(NativeSourceNode::LiteralBatch(NativeLiteralBatchSource::empty(Arc::clone(&schema))));
    Ok(LoweredNativeInput::with_bindings(source_node, schema, Vec::new()))
}

struct TableFormatCountStarFallbackColumn<'a> {
    read_column: String,
    read_column_file_index: usize,
    direct_mapping: Option<&'a TableFormatColumnMapping>,
}

fn table_format_count_star_fallback_column<'a>(
    scan_plan: &'a TableFormatScanPlan,
    file_schema: &SchemaRef,
) -> Result<TableFormatCountStarFallbackColumn<'a>, FallbackReason> {
    if let Some(column) = table_format_explicit_count_star_probe_column(scan_plan, file_schema) {
        return Ok(column);
    }
    if table_format_local_footer_schema(scan_plan).is_some() {
        if scan_plan.format == TableFormat::Iceberg {
            return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
        }
        return table_format_first_supported_file_column(file_schema);
    }
    Err(FallbackReason::NativeSourceSchemaMapping)
}

fn table_format_explicit_count_star_probe_column<'a>(
    scan_plan: &'a TableFormatScanPlan,
    file_schema: &SchemaRef,
) -> Option<TableFormatCountStarFallbackColumn<'a>> {
    scan_plan
        .pure_filter_columns
        .iter()
        .chain(scan_plan.projected_columns.iter())
        .filter(|mapping| mapping.source == TableFormatColumnSource::DirectTopLevelFileColumn)
        .filter_map(|mapping| {
            let source_name = mapping.source_name.as_ref().unwrap_or(&mapping.logical_name);
            let index = file_schema.index_of(source_name).ok();
            let field_supported = index.and_then(|index| file_schema.fields().get(index)).is_some_and(|field| {
                field.data_type() == &mapping.data_type && is_supported_native_type(field.data_type())
            });
            let field_id_supported = mapping.field_id.is_some() && is_supported_native_type(&mapping.data_type);
            (field_supported || field_id_supported).then(|| TableFormatCountStarFallbackColumn {
                read_column: mapping.logical_name.clone(),
                read_column_file_index: index.unwrap_or(0),
                direct_mapping: Some(mapping),
            })
        })
        .next()
}

fn table_format_first_supported_file_column(
    file_schema: &SchemaRef,
) -> Result<TableFormatCountStarFallbackColumn<'static>, FallbackReason> {
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
        .ok_or(FallbackReason::UnsupportedType)
}

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

fn validate_table_format_scan_plan(plan: &TableFormatScanPlan) -> Result<(), FallbackReason> {
    match table_format_scan_plan_rejection(plan) {
        FallbackReason::TableFormatUnsupported if plan.is_native_admissible() => Ok(()),
        FallbackReason::TableFormatUnsupported => Err(FallbackReason::TableFormatUnsupported),
        reason => Err(reason),
    }
}

fn table_format_scan_plan_rejection(plan: &TableFormatScanPlan) -> FallbackReason {
    plan.first_rejection_reason().unwrap_or(FallbackReason::TableFormatUnsupported)
}

fn table_format_scan_plan_is_empty_admissible(plan: &TableFormatScanPlan) -> bool {
    plan.unsupported_features.is_empty() && plan.delete_files.is_empty() && plan.data_files.is_empty()
}

fn table_format_column_requirements(
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
        partition_columns,
        partition_column_values,
        partition_column_indices: Vec::new(),
        missing_columns,
        scan_output_columns,
    })
}

fn table_format_column_mapping_by_name(
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

fn validate_table_format_direct_file_column_field_ids(
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
            let Some((_index, field)) = footer_field_by_id(&schema, *field_id) else {
                if mapping.nullable {
                    continue;
                }
                return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
            };
            if field.data_type() != &mapping.data_type {
                return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
            }
        }
    }

    Ok(())
}

fn parquet_field_id(field: &Field) -> Option<i32> {
    field.metadata().get(PARQUET_FIELD_ID_META_KEY).and_then(|value| value.parse::<i32>().ok())
}

fn table_format_column_source_refs(
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

fn table_format_source_read_groups(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
    output_schema: SchemaRef,
) -> Result<Vec<TableFormatSourceReadGroup>, FallbackReason> {
    if !table_format_requires_iceberg_field_id_bindings(scan_plan, requirements)? {
        return Ok(vec![TableFormatSourceReadGroup {
            file_indices: (0..scan_plan.data_files.len()).collect(),
            requirements: requirements.clone(),
            source_schema: output_schema.clone(),
            output_schema,
            field_id_bindings: vec![Vec::new(); scan_plan.data_files.len()],
        }]);
    }

    let mappings = table_format_column_mapping_by_name(scan_plan)?;
    let mut groups: Vec<TableFormatSourceReadGroup> = Vec::new();
    let mut group_by_key: BTreeMap<String, usize> = BTreeMap::new();

    for (file_index, file) in scan_plan.data_files.iter().enumerate() {
        let footer_schema =
            table_format_data_file_footer_arrow_schema(file).ok_or(FallbackReason::IcebergFieldIdMappingUnsupported)?;
        let file_bindings =
            table_format_file_field_id_read_bindings(file_index, &footer_schema, requirements, &mappings)?;
        let file_requirements =
            table_format_requirements_for_file(requirements, &mappings, &file_bindings, &footer_schema)?;
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

fn table_format_requires_iceberg_field_id_bindings(
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

fn table_format_file_field_id_read_bindings(
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
        let Some((physical_root_index, field)) = footer_field_by_id(footer_schema, field_id) else {
            if mapping.nullable || mapping.missing_default_value.is_some() {
                continue;
            }
            return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
        };
        if field.data_type() != &mapping.data_type {
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

fn table_format_requirements_for_file(
    base: &NativeParquetColumnRequirements,
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
    file_bindings: &TableFormatFileReadBindings,
    footer_schema: &Schema,
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let mut read_columns = Vec::new();
    let mut read_column_file_indices = Vec::new();
    let mut read_seen = BTreeSet::new();
    let mut missing_columns = base.missing_columns.clone();
    let mut missing_seen = missing_columns.iter().map(|column| column.name.clone()).collect::<BTreeSet<_>>();
    let mut partition_columns = base.partition_columns.clone();
    let mut partition_column_values = base.partition_column_values.clone();
    let mut partition_seen = partition_columns.iter().map(|column| column.name.clone()).collect::<BTreeSet<_>>();

    for column in &base.scan_output_columns {
        let mapping = mappings.get(column.as_str()).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
        if mapping.source != TableFormatColumnSource::DirectTopLevelFileColumn {
            continue;
        }
        let field_id = mapping.field_id.ok_or(FallbackReason::IcebergFieldIdMappingUnsupported)?;
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
            if missing_seen.insert(mapping.logical_name.clone()) {
                missing_columns.push(NativeMissingColumn::new(
                    mapping.logical_name.as_str(),
                    mapping.data_type.clone(),
                    mapping.nullable,
                ));
            }
        } else {
            return Err(FallbackReason::IcebergFieldIdMappingUnsupported);
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
        partition_columns,
        partition_column_values,
        partition_column_indices: base.partition_column_indices.clone(),
        missing_columns,
        scan_output_columns: base.scan_output_columns.clone(),
    })
}

fn table_format_file_column_source_refs(
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
    file_bindings: &TableFormatFileReadBindings,
    footer_schema: &Schema,
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
                    let field_id = mapping.field_id.ok_or(FallbackReason::IcebergFieldIdMappingUnsupported)?;
                    if let Some(binding) =
                        file_bindings.bindings.iter().find(|binding| binding.logical_field_id == field_id)
                    {
                        Ok(NativeSourceColumnRef::File(binding.physical_root_index))
                    } else if mapping.missing_default_value.is_some() {
                        partition_columns
                            .iter()
                            .position(|field| field.name == mapping.logical_name)
                            .map(NativeSourceColumnRef::Partition)
                            .ok_or(FallbackReason::NativeSourceSchemaMapping)
                    } else if mapping.nullable {
                        missing_columns
                            .iter()
                            .position(|field| field.name == mapping.logical_name)
                            .map(NativeSourceColumnRef::MissingNull)
                            .ok_or(FallbackReason::NativeSourceSchemaMapping)
                    } else {
                        Err(FallbackReason::IcebergFieldIdMappingUnsupported)
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
        .collect::<Result<Vec<_>, _>>()
        .and_then(|refs| {
            for source_ref in &refs {
                if let NativeSourceColumnRef::File(index) = source_ref
                    && footer_schema.fields().get(*index).is_none()
                {
                    return Err(FallbackReason::NativeSourceSchemaMapping);
                }
            }
            Ok(refs)
        })
}

fn table_format_source_contract_schema(
    requirements: &NativeParquetColumnRequirements,
    output_schema: &SchemaRef,
    footer_schema: &Schema,
) -> Result<SchemaRef, FallbackReason> {
    let mut fields = Vec::new();
    let mut seen = BTreeSet::new();
    for index in &requirements.read_column_file_indices {
        let field = footer_schema.fields().get(*index).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
        if seen.insert(field.name().clone()) {
            fields.push(Field::new(field.name(), field.data_type().clone(), field.is_nullable()));
        }
    }
    for field in output_schema.fields() {
        if seen.insert(field.name().clone()) {
            fields.push(field.as_ref().clone());
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn table_format_source_group_key(requirements: &NativeParquetColumnRequirements) -> String {
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

fn source_refs_key(refs: &[NativeSourceColumnRef]) -> String {
    refs.iter()
        .map(|source_ref| match source_ref {
            NativeSourceColumnRef::File(index) => format!("f{index}"),
            NativeSourceColumnRef::Partition(index) => format!("p{index}"),
            NativeSourceColumnRef::MissingNull(index) => format!("m{index}"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn footer_field_by_id(schema: &Schema, field_id: i32) -> Option<(usize, &Field)> {
    schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_index, field)| parquet_field_id(field) == Some(field_id))
        .map(|(index, field)| (index, field.as_ref()))
}

fn table_format_source_output_schema(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
    file_schema: &SchemaRef,
) -> Result<SchemaRef, FallbackReason> {
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

fn table_format_source_output_field(
    column: &str,
    source_ref: &NativeSourceColumnRef,
    requirements: &NativeParquetColumnRequirements,
    file_schema: &SchemaRef,
    mappings: &BTreeMap<&str, &TableFormatColumnMapping>,
) -> Result<Field, FallbackReason> {
    match source_ref {
        NativeSourceColumnRef::File(index) => {
            if let Some(mapping) = mappings.get(column) {
                return Ok(Field::new(column, mapping.data_type.clone(), mapping.nullable));
            }
            let field = file_schema.fields().get(*index).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
            Ok(Field::new(column, field.data_type().clone(), field.is_nullable()))
        },
        NativeSourceColumnRef::Partition(index) => {
            let column = requirements.partition_columns.get(*index).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
            Ok(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable))
        },
        NativeSourceColumnRef::MissingNull(index) => {
            let column = requirements.missing_columns.get(*index).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
            Ok(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable))
        },
    }
}

fn table_format_local_parquet_files(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
) -> Result<Vec<NativeLocalParquetFile>, FallbackReason> {
    if scan_plan.data_files.is_empty() {
        return Err(FallbackReason::NativeSourceEmpty);
    }
    scan_plan.data_files.iter().map(|file| table_format_local_parquet_file(scan_plan, file, requirements)).collect()
}

fn table_format_local_parquet_files_for_group(
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

fn table_format_local_parquet_file(
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
    if requirements.read_columns.is_empty()
        && file.row_groups.is_none()
        && let Some(row_count) = file.row_count
    {
        native_file = native_file.with_row_group_rows(vec![row_count]);
    }
    if !requirements.partition_columns.is_empty() {
        native_file = native_file.with_partition_values(table_format_partition_values(file, requirements)?);
    }
    Ok(attach_row_group_byte_metadata(native_file, requirements))
}

fn table_format_partition_values(
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

fn table_format_scalar_value_to_native_literal(
    value: &TableFormatScalarValue,
    data_type: &DataType,
) -> Result<NativeLiteralValue, FallbackReason> {
    Ok(match value {
        TableFormatScalarValue::Null => NativeLiteralValue::Null(data_type.clone()),
        TableFormatScalarValue::Boolean(value) => NativeLiteralValue::Boolean(*value),
        TableFormatScalarValue::Int32(value) => NativeLiteralValue::Int32(*value),
        TableFormatScalarValue::Int64(value) if matches!(data_type, DataType::Timestamp(_, None)) => {
            let DataType::Timestamp(unit, None) = data_type else { unreachable!("timestamp type checked in guard") };
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

fn table_format_local_path(location: &str) -> PathBuf {
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

fn table_format_native_parquet_file(file: &TableFormatDataFile) -> Result<NativeLocalParquetFile, FallbackReason> {
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

fn table_format_file_schema(scan_plan: &TableFormatScanPlan) -> SchemaRef {
    table_format_local_footer_schema(scan_plan)
        .unwrap_or_else(|| strip_table_format_schema_metadata(&scan_plan.native_source_schema))
}

fn table_format_local_footer_schema(scan_plan: &TableFormatScanPlan) -> Option<SchemaRef> {
    scan_plan
        .data_files
        .iter()
        .filter(|file| file.read_capability.is_native_local_parquet_capable())
        .map(|file| table_format_local_path(&file.location))
        .find_map(|path| table_format_local_footer_arrow_schema(path.as_path()))
        .map(strip_table_format_file_schema_metadata)
}

fn table_format_local_footer_arrow_schema(path: &Path) -> Option<Schema> {
    parquet::file::reader::SerializedFileReader::try_from(path).ok().and_then(|reader| {
        let metadata = reader.metadata().file_metadata();
        parquet::arrow::parquet_to_arrow_schema(metadata.schema_descr(), metadata.key_value_metadata()).ok()
    })
}

fn table_format_data_file_footer_arrow_schema(file: &TableFormatDataFile) -> Option<Schema> {
    let native_file = table_format_native_parquet_file(file).ok()?;
    table_format_local_footer_arrow_schema(native_file.path.as_path())
}

fn strip_table_format_file_schema_metadata(schema: Schema) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), field.is_nullable()))
            .collect::<Vec<_>>(),
    ))
}

fn strip_table_format_schema_metadata(schema: &SchemaRef) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), field.is_nullable()))
            .collect::<Vec<_>>(),
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        fs::File,
        time::{SystemTime, UNIX_EPOCH},
    };

    use arrow::{array::Int64Array, record_batch::RecordBatch};
    use nexus_query_engine::{
        CredentialMode, PlanNode, ResolvedParquetSource, SourceColumnRef as NativeSourceColumnRef,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use super::*;
    use crate::table_format::{
        TableFormat, TableFormatPartitionTransform, TableFormatPartitionValue, TableFormatReadCapability,
        TableFormatScalarValue,
    };

    fn test_column_requirements() -> NativeParquetColumnRequirements {
        NativeParquetColumnRequirements {
            read_columns: vec!["c0".to_string()],
            read_column_file_indices: vec![0],
            projected_output_columns: vec!["c0".to_string()],
            projected_output_column_source_refs: vec![NativeSourceColumnRef::File(0)],
            pure_filter_columns: Vec::new(),
            pure_filter_column_source_refs: Vec::new(),
            partition_columns: Vec::new(),
            partition_column_values: Vec::new(),
            partition_column_indices: Vec::new(),
            missing_columns: Vec::new(),
            scan_output_columns: vec!["c0".to_string()],
        }
    }

    #[test]
    fn table_format_local_footer_field_id_mismatch_rejects_direct_file_column() {
        let dir = unique_table_format_temp_dir("field-id-mismatch");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "status", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
        let file_schema = table_format_file_schema(&scan_plan);

        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();
        let result = table_format_source_read_groups(&scan_plan, &requirements, output_schema);

        assert_eq!(result.unwrap_err(), FallbackReason::IcebergFieldIdMappingUnsupported);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_local_footer_matching_field_id_admits_direct_file_column() {
        let dir = unique_table_format_temp_dir("field-id-match");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "status", 2);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
        let file_schema = table_format_file_schema(&scan_plan);

        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();

        assert_eq!(requirements.read_columns, vec!["status"]);
        assert_eq!(requirements.read_column_file_indices, vec![0]);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_local_footer_matching_field_id_admits_renamed_physical_column() {
        let dir = unique_table_format_temp_dir("field-id-rename");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "old_status", 2);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].requirements.read_columns, vec!["old_status"]);
        assert_eq!(groups[0].requirements.projected_output_columns, vec!["status"]);
        assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
        assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
        assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_status");
        assert_eq!(groups[0].source_schema.field_with_name("old_status").unwrap().data_type(), &DataType::Int64);
        assert_eq!(groups[0].output_schema.field_with_name("status").unwrap().data_type(), &DataType::Int64);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_local_footer_renamed_physical_columns_lower_as_union_groups() {
        let dir = unique_table_format_temp_dir("field-id-rename-union");
        let old_path = dir.join("part-old.parquet");
        let new_path = dir.join("part-new.parquet");
        write_field_id_parquet_file(&old_path, "old_status", 2);
        write_field_id_parquet_file(&new_path, "status", 2);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&scan_schema))
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(
                TableFormatDataFile::new(
                    "old-file",
                    old_path.display().to_string(),
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(old_path.display().to_string())),
            )
            .with_data_file(
                TableFormatDataFile::new(
                    "new-file",
                    new_path.display().to_string(),
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(new_path.display().to_string())),
            )
            .with_projected_column(TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let scan_exec = CudfTableFormatScanExec::new(scan_plan, Arc::clone(&scan_schema));
        let mut builder = NativeQueryPlanBuilder::new();

        let lowered =
            lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
        let plan = builder.finish_with_host_sink(lowered.node);

        assert!(matches!(plan.root_node().kind, PlanNode::Union(_)));
        let sources = plan
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    NativeSourceNode::LocalParquet(source) => Some(source),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sources.len(), 2);
        assert!(sources.iter().any(|source| source.read_columns == vec!["old_status".to_string()]));
        assert!(sources.iter().any(|source| source.read_columns == vec!["status".to_string()]));
        assert!(sources.iter().all(|source| source.output_schema().field_with_name("status").is_ok()));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_local_footer_absent_nullable_field_id_materializes_missing_null_column() {
        let dir = unique_table_format_temp_dir("field-id-missing-null");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "other", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let scan_plan = single_projected_file_scan_plan_with_nullability(&path, Arc::clone(&scan_schema), 2, true);
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

        assert_eq!(groups.len(), 1);
        assert!(groups[0].requirements.read_columns.is_empty());
        assert_eq!(groups[0].requirements.missing_columns.len(), 1);
        assert_eq!(groups[0].requirements.missing_columns[0].name, "status");
        assert_eq!(
            groups[0].requirements.projected_output_column_source_refs,
            vec![NativeSourceColumnRef::MissingNull(0)]
        );
        assert!(groups[0].output_schema.field_with_name("status").unwrap().is_nullable());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_local_footer_absent_field_id_materializes_literal_default_column() {
        let dir = unique_table_format_temp_dir("field-id-missing-default");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "other", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_projected_file_scan_plan_with_default(
            &path,
            Arc::clone(&scan_schema),
            2,
            TableFormatScalarValue::Int64(7),
        );
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
        let native_files = table_format_local_parquet_files_for_group(&scan_plan, &groups[0]).unwrap();

        assert_eq!(groups.len(), 1);
        assert!(groups[0].requirements.read_columns.is_empty());
        assert_eq!(groups[0].requirements.partition_columns.len(), 1);
        assert_eq!(groups[0].requirements.partition_columns[0].name, "status");
        assert_eq!(
            groups[0].requirements.projected_output_column_source_refs,
            vec![NativeSourceColumnRef::Partition(0)]
        );
        assert_eq!(native_files[0].partition_values, vec![NativeLiteralValue::Int64(7)]);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_count_star_fallback_rejects_field_id_mismatch_on_carrier_column() {
        let dir = unique_table_format_temp_dir("count-star-field-id-mismatch");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "status", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
        let mut builder = NativeQueryPlanBuilder::new();

        let error = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
            .err()
            .expect("field-id mismatch must reject COUNT(*) fallback carrier column");

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_count_star_fallback_rejects_iceberg_implicit_local_carrier_column() {
        let dir = unique_table_format_temp_dir("count-star-implicit-carrier");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "status", 2);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_unmapped_file_scan_plan(&path, Arc::clone(&scan_schema));
        let mut builder = NativeQueryPlanBuilder::new();

        let error = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
            .err()
            .expect("Iceberg COUNT(*) fallback must not synthesize a name-based carrier column");

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_count_star_fallback_rejects_missing_nullable_carrier_without_probe_metadata() {
        let dir = unique_table_format_temp_dir("count-star-missing-carrier");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "other", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2).with_scan_limit(Some(1));
        let mut builder = NativeQueryPlanBuilder::new();

        let error = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
            .err()
            .expect("COUNT(*) fallback must not treat missing/null materialization as row-presence proof");

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_count_star_fallback_admits_matching_field_id_on_carrier_column() {
        let dir = unique_table_format_temp_dir("count-star-field-id-match");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "status", 2);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
        let mut builder = NativeQueryPlanBuilder::new();

        let lowered = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
            .expect("matching field id should admit COUNT(*) fallback carrier column");

        assert_eq!(lowered.schema.field(0).name(), NATIVE_ZERO_COLUMN_COUNT_COLUMN);
        fs::remove_dir_all(dir).ok();
    }

    fn single_projected_file_scan_plan(path: &Path, schema: SchemaRef, field_id: i32) -> TableFormatScanPlan {
        single_projected_file_scan_plan_with_nullability(path, schema, field_id, false)
    }

    fn single_projected_file_scan_plan_with_nullability(
        path: &Path,
        schema: SchemaRef,
        field_id: i32,
        nullable: bool,
    ) -> TableFormatScanPlan {
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(
                TableFormatDataFile::new(
                    "file-0",
                    path.display().to_string(),
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
            )
            .with_projected_column(TableFormatColumnMapping::new(
                Some(field_id),
                "status",
                Some("status"),
                DataType::Int64,
                nullable,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ))
    }

    fn single_projected_file_scan_plan_with_default(
        path: &Path,
        schema: SchemaRef,
        field_id: i32,
        default_value: TableFormatScalarValue,
    ) -> TableFormatScanPlan {
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(
                TableFormatDataFile::new(
                    "file-0",
                    path.display().to_string(),
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
            )
            .with_projected_column(
                TableFormatColumnMapping::new(
                    Some(field_id),
                    "status",
                    Some("status"),
                    DataType::Int64,
                    false,
                    TableFormatColumnSource::DirectTopLevelFileColumn,
                    crate::table_format::TableFormatColumnRole::Projected,
                )
                .with_missing_default_value(default_value),
            )
    }

    fn single_unmapped_file_scan_plan(path: &Path, schema: SchemaRef) -> TableFormatScanPlan {
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(
                TableFormatDataFile::new(
                    "file-0",
                    path.display().to_string(),
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
            )
    }

    fn single_pure_filter_file_scan_plan(path: &Path, schema: SchemaRef, field_id: i32) -> TableFormatScanPlan {
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(
                TableFormatDataFile::new(
                    "file-0",
                    path.display().to_string(),
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
            )
            .with_pure_filter_column(TableFormatColumnMapping::new(
                Some(field_id),
                "status",
                Some("status"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::PureFilter,
            ))
    }

    fn write_field_id_parquet_file(path: &Path, column_name: &str, field_id: i32) {
        let field = Field::new(column_name, DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string())]));
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![10, 20]))]).unwrap();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn unique_table_format_temp_dir(label: &str) -> PathBuf {
        let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("datafusion-nexus-table-format-{label}-{id}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn table_format_file_row_count_does_not_create_synthetic_row_group_metadata() {
        let mut scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg);
        scan_plan.data_files.push(
            TableFormatDataFile::new(
                "file-0",
                "s3://bucket/table/part-0.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(1024)
            .with_row_count(123)
            .with_partition_values(vec![TableFormatPartitionValue::new(
                "part",
                DataType::Int32,
                false,
                TableFormatPartitionTransform::Identity,
                TableFormatScalarValue::Int32(7),
            )])
            .with_resolved_parquet_source(ResolvedParquetSource::remote_host_buffer(
                "s3://bucket/table/part-0.parquet",
                "s3://bucket/table/part-0.parquet",
                CredentialMode::SharedProcessEnv,
            )),
        );

        let files = table_format_local_parquet_files(&scan_plan, &test_column_requirements()).unwrap();

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].row_groups, None);
        assert_eq!(files[0].row_group_rows, None);
        assert_eq!(files[0].selected_row_group_count(), 0);
    }

    #[test]
    fn table_format_partition_only_file_row_count_creates_synthetic_row_group_metadata() {
        let mut scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg);
        scan_plan.data_files.push(
            TableFormatDataFile::new(
                "file-0",
                "s3://bucket/table/part-0.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(1024)
            .with_row_count(123)
            .with_partition_values(vec![TableFormatPartitionValue::new(
                "part",
                DataType::Int32,
                false,
                TableFormatPartitionTransform::Identity,
                TableFormatScalarValue::Int32(7),
            )])
            .with_resolved_parquet_source(ResolvedParquetSource::remote_host_buffer(
                "s3://bucket/table/part-0.parquet",
                "s3://bucket/table/part-0.parquet",
                CredentialMode::SharedProcessEnv,
            )),
        );
        let mut requirements = test_column_requirements();
        requirements.read_columns.clear();
        requirements.read_column_file_indices.clear();
        requirements.projected_output_columns = vec!["part".to_string()];
        requirements.projected_output_column_source_refs = vec![NativeSourceColumnRef::Partition(0)];
        requirements.partition_columns = vec![nexus_query_engine::PartitionColumn::new("part", DataType::Int32, false)];
        requirements.partition_column_indices = vec![0];
        requirements.scan_output_columns = vec!["part".to_string()];

        let files = table_format_local_parquet_files(&scan_plan, &requirements).unwrap();

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].row_group_rows.as_deref(), Some(&[123][..]));
    }
}
