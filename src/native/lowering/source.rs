use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    datasource::physical_plan::{FileSource, ParquetSource},
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::ExecutionPlan,
};
use nexus_query_engine::{
    Expr as NativeExpr, LimitSpec as NativeLimitSpec, LiteralValue as NativeLiteralValue,
    LocalParquetFile as NativeLocalParquetFile, LocalParquetSource as NativeLocalParquetSource,
    MissingColumn as NativeMissingColumn, PartitionColumn as NativePartitionColumn,
    QueryPlanBuilder as NativeQueryPlanBuilder, SourceColumnRef as NativeSourceColumnRef,
    SourceNode as NativeSourceNode,
};
use parquet::file::reader::FileReader;

// Keep native filters on small dimension scans. They are cheap and preserve
// join/aggregate planning shape; large fact filters benefit from elision.
const TABLE_FORMAT_FILTER_ELISION_MIN_SOURCE_ROWS: u64 = 10_000_000;

use super::{
    LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, lower_physical_expr, lower_physical_expr_with_bindings},
    projection_outputs,
};
use crate::{
    native::{
        normalize::{NormalizedNativeScan, NormalizedNativeSource},
        pruning::{
            scan_predicate_exactly_matches_filter, scan_predicate_is_dynamic_filter_hint,
            scan_predicate_without_dynamic_filter_hints, validate_pruning_only_parquet_predicate,
        },
        source::{
            NativeParquetColumnRequirements, attach_row_group_byte_metadata, local_parquet_files,
            local_parquet_source_diagnostics, native_source_output_schema,
            required_parquet_column_requirements_for_scan_columns, validate_slice1_parquet_scan,
        },
        validate::{collect_required_columns, is_supported_native_type, validate_physical_expr, validate_schema},
    },
    policy::FallbackReason,
    source_cardinality::reader_predicate_proves_source_at_most_one_row,
    table_format::{
        CudfTableFormatScanExec, TableFormatColumnMapping, TableFormatColumnSource, TableFormatDataFile,
        TableFormatDataFileFormat, TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata,
        TableFormatPartitionTransform, TableFormatScalarValue, TableFormatScanPlan,
        resolve_table_format_parquet_source_with_metadata,
    },
};

pub(super) fn lower_source_input(
    source: &NormalizedNativeSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    pushed_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, FallbackReason> {
    let NormalizedNativeScan::LocalParquet(source_exec) = source.scan else {
        let scan_exec = source.table_format_scan_exec().ok_or(FallbackReason::TableFormatUnsupported)?;
        return lower_table_format_source_input(
            scan_exec,
            source.filter,
            builder,
            required_output_fields,
            pushed_scan_limit,
        );
    };
    let filter = source.filter;
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;
    let required_scan_output_indices =
        required_source_scan_output_indices(source_exec.schema(), filter, required_output_fields)?;
    let column_requirements = required_parquet_column_requirements_for_scan_columns(
        file_scan,
        parquet_source,
        filter,
        &required_scan_output_indices,
    )?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let source_output_schema = native_source_output_schema(file_scan, &column_requirements)?;

    validate_schema(Arc::clone(&source_output_schema))?;
    if let Some(scan_projection) = scan_projection {
        for projection_expr in
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns)
        {
            validate_physical_expr(projection_expr.expr, &table_schema)?;
        }
    }
    if let Some(filter) = filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }
    validate_slice1_parquet_scan(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;

    let native_files =
        local_parquet_files(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    let native_file_paths = native_files.iter().map(|file| file.path.clone()).collect::<Vec<_>>();
    let scan_projection_requires_native = scan_projection.is_some_and(|scan_projection| {
        let required_scan_projection =
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns);
        required_scan_projection_requires_native_projection(&required_scan_projection, &table_schema)
    });
    let mut native_source = NativeLocalParquetSource::from_files(
        native_files,
        Arc::clone(&source_output_schema),
        column_requirements.read_columns.clone(),
    )
    .with_column_roles(
        column_requirements.projected_output_columns.clone(),
        column_requirements.pure_filter_columns.clone(),
    )
    .with_column_source_refs(
        column_requirements.read_column_file_indices.clone(),
        column_requirements.projected_output_column_source_refs.clone(),
        column_requirements.pure_filter_column_source_refs.clone(),
    )
    .with_partition_columns(column_requirements.partition_columns.clone())
    .with_missing_columns(column_requirements.missing_columns.clone())
    .with_scan_limit(if filter.is_none() {
        combine_scan_limits(file_scan.limit, pushed_scan_limit)
    } else {
        file_scan.limit
    });
    let mut reader_pushdown_covers_filter = false;
    if let Some(predicate) = source_reader_pushdown_predicate(
        parquet_source,
        filter,
        file_scan.limit,
        scan_projection_requires_native,
        &source_output_schema,
        if filter.is_none() { Arc::clone(&table_schema) } else { source_exec.schema() },
        &native_source,
    )? {
        reader_pushdown_covers_filter = predicate.covers_filter;
        let max_rows = if reader_predicate_proves_local_parquet_at_most_one_row(
            table_schema.as_ref(),
            file_scan.file_schema().as_ref(),
            &native_file_paths.iter().map(|path| path.as_path()).collect::<Vec<_>>(),
            &predicate.expr,
        ) {
            Some(1)
        } else {
            None
        };
        native_source = native_source.with_reader_predicate(predicate.expr).with_max_rows(max_rows);
    }
    let reader_pushdown = native_source.reader_predicate.is_some();
    let selected_row_group_stats = native_source.diagnostics.selected_row_group_stats.clone();
    native_source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        Arc::clone(&table_schema),
        Some(&column_requirements),
        reader_pushdown,
        false,
    );
    native_source.diagnostics.selected_row_group_stats = selected_row_group_stats;

    let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
    let scan_only_post_scan_predicate = if filter.is_none() && !reader_pushdown {
        parquet_source.filter().and_then(|predicate| scan_predicate_without_dynamic_filter_hints(&predicate))
    } else {
        None
    };
    let (scan_input_node, scan_input_schema) = if let Some(predicate) = scan_only_post_scan_predicate.as_ref() {
        let bindings = source_bindings_for_schema(&source_output_schema, Arc::clone(&table_schema));
        (
            builder.add_filter(
                source_node,
                lower_physical_expr_with_bindings(predicate, &table_schema, Some(&bindings))?,
                Arc::clone(&source_output_schema),
            ),
            Arc::clone(&source_output_schema),
        )
    } else {
        (source_node, Arc::clone(&source_output_schema))
    };
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection {
        let required_scan_projection =
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns);
        if filter.is_none()
            && required_scan_projection_needs_native_projection(
                &required_scan_projection,
                &table_schema,
                &source_output_schema,
                source_exec.schema(),
            )?
        {
            let expressions = lower_required_file_source_projection_exprs(&required_scan_projection, &table_schema)?;
            let schema = required_scan_projection_schema(&required_scan_projection, source_exec.schema())?;
            (builder.add_projection(scan_input_node, projection_outputs(expressions, &schema)?), schema)
        } else {
            (scan_input_node, scan_input_schema)
        }
    } else {
        (scan_input_node, scan_input_schema)
    };
    let scan_output_bindings = source_bindings_for_schema(&scan_output_schema, source_exec.schema());
    let scan_output = LoweredNativeInput::with_bindings(scan_output_node, scan_output_schema, scan_output_bindings);
    if let Some(filter) = filter {
        if reader_pushdown_covers_filter {
            if let Some(projection) = filter.projection() {
                let projected = add_source_filter_projection(
                    builder,
                    scan_output.node,
                    &scan_output,
                    filter.schema(),
                    projection,
                    required_output_fields,
                )?;
                return Ok(projected);
            }
            return Ok(scan_output);
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
        if let Some(projection) = filter.projection() {
            let projected = add_source_filter_projection(
                builder,
                filtered_node,
                &scan_output,
                filter.schema(),
                projection,
                required_output_fields,
            )?;
            Ok(projected)
        } else {
            Ok(LoweredNativeInput::with_bindings(filtered_node, scan_output.schema, scan_output.bindings))
        }
    } else {
        project_scan_output_columns(
            builder,
            scan_output.node,
            scan_output.schema,
            scan_output.bindings,
            required_output_fields,
        )
    }
}

pub(super) fn reader_predicate_proves_local_parquet_at_most_one_row(
    table_schema: &Schema,
    file_schema: &Schema,
    source_paths: &[&Path],
    predicate: &NativeExpr,
) -> bool {
    reader_predicate_proves_source_at_most_one_row(&[table_schema, file_schema], source_paths, predicate)
}

fn lower_table_format_source_input(
    scan_exec: &CudfTableFormatScanExec,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    pushed_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, FallbackReason> {
    let scan_plan = scan_exec.scan_plan();
    validate_table_format_scan_plan(scan_plan)?;
    validate_schema(scan_exec.schema())?;
    if let Some(filter) = filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }

    let file_schema = table_format_file_schema(scan_plan);
    let requirements =
        table_format_column_requirements(scan_plan, &file_schema, scan_exec.schema(), filter, required_output_fields)?;
    let source_output_schema = table_format_source_output_schema(&requirements, &file_schema);
    validate_schema(Arc::clone(&source_output_schema))?;
    let native_files = table_format_local_parquet_files(scan_plan, &requirements)?;
    let native_file_paths =
        native_files.iter().filter_map(|file| file.location.as_local_path().map(Path::to_path_buf)).collect::<Vec<_>>();
    let source_scan_limit =
        if filter.is_none() { combine_scan_limits(scan_plan.scan_limit, pushed_scan_limit) } else { None };
    let mut native_source = NativeLocalParquetSource::from_files(
        native_files,
        Arc::clone(&source_output_schema),
        requirements.read_columns.clone(),
    )
    .with_column_roles(requirements.projected_output_columns.clone(), requirements.pure_filter_columns.clone())
    .with_column_source_refs(
        requirements.read_column_file_indices.clone(),
        requirements.projected_output_column_source_refs.clone(),
        requirements.pure_filter_column_source_refs.clone(),
    )
    .with_partition_columns(requirements.partition_columns.clone())
    .with_missing_columns(requirements.missing_columns.clone())
    .with_scan_limit(source_scan_limit);
    let mut reader_pushdown_covers_filter = false;
    if let Some(predicate) =
        table_format_reader_pushdown_predicate(filter, &source_output_schema, scan_exec.schema(), &native_source)?
    {
        reader_pushdown_covers_filter = predicate.covers_filter;
        let max_rows = if reader_predicate_proves_table_format_source_at_most_one_row(
            scan_plan,
            &file_schema,
            &source_output_schema,
            scan_exec.schema().as_ref(),
            &native_file_paths.iter().map(|path| path.as_path()).collect::<Vec<_>>(),
            &predicate.expr,
        ) {
            Some(1)
        } else {
            None
        };
        native_source = native_source.with_reader_predicate(predicate.expr).with_max_rows(max_rows);
    }

    let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
    let scan_output_bindings = source_bindings_for_schema(&source_output_schema, scan_exec.schema());
    let scan_output = LoweredNativeInput::with_bindings(source_node, source_output_schema, scan_output_bindings);
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
        Ok(scan_output)
    }
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
    if table_format_reader_predicate_references_stats_filter_risky_type(&predicate, source_output_schema) {
        return Ok(None);
    }
    if native_source.reader_predicate_is_supported(&predicate) {
        Ok(Some(SourceReaderPushdownPredicate {
            expr: predicate,
            covers_filter: table_format_reader_pushdown_can_elide_native_filter(native_source),
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

fn combine_scan_limits(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(limit), None) | (None, Some(limit)) => Some(limit),
        (None, None) => None,
    }
}

pub(super) fn lower_table_format_zero_column_count_source(
    scan_exec: &CudfTableFormatScanExec,
    builder: &mut NativeQueryPlanBuilder,
) -> Result<LoweredNativeInput, FallbackReason> {
    let scan_plan = scan_exec.scan_plan();
    validate_table_format_scan_plan(scan_plan)?;
    let requirements = NativeParquetColumnRequirements {
        read_columns: Vec::new(),
        read_column_file_indices: Vec::new(),
        projected_output_columns: Vec::new(),
        projected_output_column_source_refs: Vec::new(),
        pure_filter_columns: Vec::new(),
        pure_filter_column_source_refs: Vec::new(),
        partition_columns: Vec::new(),
        partition_column_indices: Vec::new(),
        missing_columns: Vec::new(),
        scan_output_columns: Vec::new(),
    };
    let native_files = table_format_local_parquet_files(scan_plan, &requirements)?;
    let source = NativeLocalParquetSource::zero_column_count(native_files).with_scan_limit(scan_plan.scan_limit);
    let source_node = builder.add_source(NativeSourceNode::LocalParquet(source));
    Ok(LoweredNativeInput::with_bindings(source_node, scan_exec.schema(), Vec::new()))
}

pub(super) fn table_format_scan_rejection(source: &NormalizedNativeSource<'_>) -> FallbackReason {
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
    if let Some(reason) = plan.unsupported_fallback_reasons().into_iter().next() {
        return reason;
    }
    if !plan.delete_files.is_empty() {
        return FallbackReason::IcebergDeleteFilesUnsupported;
    }
    if plan.data_files.is_empty() {
        return FallbackReason::NativeSourceEmpty;
    }
    if plan.data_files.iter().any(|file| file.format != TableFormatDataFileFormat::Parquet) {
        return FallbackReason::IcebergNonParquetDataFile;
    }
    if plan.data_files.iter().any(|file| !file.read_capability.is_native_parquet_capable()) {
        return FallbackReason::IcebergObjectStoreReadUnsupported;
    }
    FallbackReason::TableFormatUnsupported
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
    // (in `BTreeSet` iteration order — they never appear in `scan_schema` and
    // their relative order does not affect any schema check).
    let mut ordered_required_columns: Vec<String> = Vec::with_capacity(output_columns.len() + filter_columns.len());
    let mut required_seen: BTreeSet<String> = BTreeSet::new();
    for column in output_columns.iter().chain(filter_columns.iter()) {
        if required_seen.insert(column.clone()) {
            ordered_required_columns.push(column.clone());
        }
    }

    let mappings = table_format_column_mapping_by_name(scan_plan);
    let mut read_columns: Vec<String> = Vec::new();
    let mut read_columns_seen: BTreeSet<String> = BTreeSet::new();
    let mut partition_columns: Vec<NativePartitionColumn> = Vec::new();
    let mut partition_columns_seen: BTreeSet<String> = BTreeSet::new();
    let mut missing_columns: Vec<NativeMissingColumn> = Vec::new();
    let mut missing_columns_seen: BTreeSet<String> = BTreeSet::new();
    for column in &ordered_required_columns {
        let mapping = mappings.get(column.as_str()).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
        match mapping.source {
            TableFormatColumnSource::DirectTopLevelFileColumn => {
                let source_name = mapping.source_name.as_ref().unwrap_or(&mapping.logical_name);
                if file_schema.index_of(source_name).is_err() {
                    return Err(FallbackReason::NativeSourceSchemaMapping);
                }
                if read_columns_seen.insert(source_name.clone()) {
                    read_columns.push(source_name.clone());
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
        .map(|column| file_schema.index_of(column).map_err(|_error| FallbackReason::NativeSourceSchemaMapping))
        .collect::<Result<Vec<_>, _>>()?;
    let projected_output_column_source_refs =
        table_format_column_source_refs(file_schema, &partition_columns, &missing_columns, &projected_output_columns)?;
    let pure_filter_column_source_refs =
        table_format_column_source_refs(file_schema, &partition_columns, &missing_columns, &pure_filter_columns)?;
    let scan_output_columns = ordered_required_columns;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns,
        projected_output_column_source_refs,
        pure_filter_columns,
        pure_filter_column_source_refs,
        partition_columns,
        partition_column_indices: Vec::new(),
        missing_columns,
        scan_output_columns,
    })
}

fn table_format_column_mapping_by_name(scan_plan: &TableFormatScanPlan) -> BTreeMap<&str, &TableFormatColumnMapping> {
    scan_plan
        .projected_columns
        .iter()
        .chain(scan_plan.pure_filter_columns.iter())
        .flat_map(|mapping| {
            let source_name = mapping.source_name.as_deref().map(|name| (name, mapping));
            [(mapping.logical_name.as_str(), mapping)].into_iter().chain(source_name)
        })
        .collect()
}

fn table_format_column_source_refs(
    file_schema: &SchemaRef,
    partition_columns: &[NativePartitionColumn],
    missing_columns: &[NativeMissingColumn],
    columns: &[String],
) -> Result<Vec<NativeSourceColumnRef>, FallbackReason> {
    columns
        .iter()
        .map(|column| {
            if let Ok(index) = file_schema.index_of(column) {
                return Ok(NativeSourceColumnRef::File(index));
            }
            partition_columns
                .iter()
                .position(|field| &field.name == column)
                .map(NativeSourceColumnRef::Partition)
                .or_else(|| {
                    missing_columns
                        .iter()
                        .position(|field| &field.name == column)
                        .map(NativeSourceColumnRef::MissingNull)
                })
                .ok_or(FallbackReason::UnsupportedExpression)
        })
        .collect()
}

fn table_format_source_output_schema(
    requirements: &NativeParquetColumnRequirements,
    file_schema: &SchemaRef,
) -> SchemaRef {
    let mut fields = Vec::with_capacity(
        requirements.read_columns.len() + requirements.partition_columns.len() + requirements.missing_columns.len(),
    );
    for column in &requirements.read_columns {
        match file_schema.field_with_name(column) {
            Ok(field) => fields.push(field.clone()),
            Err(_) => fields.push(Field::new(column, DataType::Null, true)),
        }
    }
    for partition_column in &requirements.partition_columns {
        fields.push(Field::new(
            partition_column.name.as_str(),
            partition_column.data_type.clone(),
            partition_column.nullable,
        ));
    }
    for missing_column in &requirements.missing_columns {
        fields.push(Field::new(
            missing_column.name.as_str(),
            missing_column.data_type.clone(),
            missing_column.nullable,
        ));
    }
    Arc::new(Schema::new(fields))
}

pub(super) fn table_format_local_parquet_files(
    scan_plan: &TableFormatScanPlan,
    requirements: &NativeParquetColumnRequirements,
) -> Result<Vec<NativeLocalParquetFile>, FallbackReason> {
    if scan_plan.data_files.is_empty() {
        return Err(FallbackReason::NativeSourceEmpty);
    }
    scan_plan.data_files.iter().map(|file| table_format_local_parquet_file(scan_plan, file, requirements)).collect()
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
        .map(|column| {
            let value = file
                .partition_values
                .iter()
                .find(|value| value.column_name == column.name)
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
        TableFormatScalarValue::Int64(value) => NativeLiteralValue::Int64(*value),
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
    scan_plan
        .data_files
        .iter()
        .filter(|file| file.read_capability.is_native_local_parquet_capable())
        .map(|file| table_format_local_path(&file.location))
        .find_map(|path| {
            parquet::file::reader::SerializedFileReader::try_from(path.as_path()).ok().and_then(|reader| {
                let metadata = reader.metadata().file_metadata();
                parquet::arrow::parquet_to_arrow_schema(metadata.schema_descr(), metadata.key_value_metadata()).ok()
            })
        })
        .map(strip_table_format_file_schema_metadata)
        .unwrap_or_else(|| strip_table_format_schema_metadata(&scan_plan.native_source_schema))
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

fn project_scan_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input_node: nexus_query_engine::NodeId,
    input_schema: SchemaRef,
    input_bindings: Vec<super::FieldBinding>,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let selected_bindings = input_bindings
        .iter()
        .filter(|binding| required_output_fields.contains(binding.frontend_index))
        .cloned()
        .collect::<Vec<_>>();
    if selected_bindings.len() == input_schema.fields().len()
        && selected_bindings.iter().zip(input_bindings.iter()).all(|(selected, input)| {
            selected.frontend_index == input.frontend_index && selected.native_name == input.native_name
        })
    {
        return Ok(LoweredNativeInput::with_bindings(input_node, input_schema, input_bindings));
    }

    let mut expressions = Vec::with_capacity(selected_bindings.len());
    let mut fields = Vec::with_capacity(selected_bindings.len());
    for binding in &selected_bindings {
        expressions.push(NativeExpr::column(binding.native_name.as_str()));
        let field = input_schema
            .field_with_name(binding.native_name.as_str())
            .map_err(|_error| FallbackReason::UnsupportedExpression)?;
        fields.push(field.clone());
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input_node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, selected_bindings))
}

#[derive(Debug, Clone)]
pub(super) struct SourceReaderPushdownPredicate {
    pub(super) expr: NativeExpr,
    pub(super) covers_filter: bool,
}

pub(super) fn source_reader_pushdown_predicate(
    parquet_source: &ParquetSource,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    scan_limit: Option<usize>,
    _scan_projection_requires_native: bool,
    source_output_schema: &SchemaRef,
    frontend_schema: SchemaRef,
    native_source: &NativeLocalParquetSource,
) -> Result<Option<SourceReaderPushdownPredicate>, FallbackReason> {
    if parquet_source.filter().is_none() {
        return Ok(None);
    }
    let scan_predicate = parquet_source.filter().expect("filter presence checked above");
    if filter.is_none() && scan_predicate_is_dynamic_filter_hint(scan_predicate.as_ref()) {
        return Ok(None);
    }
    if filter.is_some() && !parquet_source.table_parquet_options().global.pushdown_filters {
        return Ok(None);
    }
    let reader_scan_predicate = scan_predicate_without_dynamic_filter_hints(&scan_predicate);
    if scan_limit.is_some() {
        return if filter.is_some() { Ok(None) } else { Err(FallbackReason::NativeSourcePushedDownParquetPredicate) };
    }
    if let Some(filter) = filter
        && validate_pruning_only_parquet_predicate(parquet_source, Some(filter), source_output_schema).is_err()
    {
        return Ok(None);
    }
    let covers_filter = scan_predicate_exactly_matches_filter(parquet_source, filter);
    let predicate = if let Some(filter) = filter
        && covers_filter
    {
        let bindings = source_bindings_for_schema(source_output_schema, Arc::clone(&frontend_schema));
        lower_physical_expr_with_bindings(filter.predicate(), &filter.input().schema(), Some(&bindings))
    } else {
        let Some(reader_scan_predicate) = reader_scan_predicate.as_ref() else {
            return Ok(None);
        };
        let bindings = source_bindings_for_schema(source_output_schema, Arc::clone(&frontend_schema));
        lower_physical_expr_with_bindings(reader_scan_predicate, &frontend_schema, Some(&bindings))
    };
    let predicate = match predicate {
        Ok(predicate) => predicate,
        Err(_error) if filter.is_some() => return Ok(None),
        Err(_error) if filter.is_none() => return Ok(None),
        Err(error) => return Err(error),
    };
    if native_source.reader_predicate_is_supported(&predicate) {
        Ok(Some(SourceReaderPushdownPredicate { expr: predicate, covers_filter }))
    } else {
        Ok(None)
    }
}

pub(super) fn source_bindings_for_schema(schema: &SchemaRef, frontend_schema: SchemaRef) -> Vec<super::FieldBinding> {
    let mut used_frontend_indices = BTreeSet::new();
    schema
        .fields()
        .iter()
        .filter_map(|field| {
            let frontend_index = frontend_schema
                .fields()
                .iter()
                .enumerate()
                .find(|(index, frontend_field)| {
                    !used_frontend_indices.contains(index) && frontend_field.name() == field.name()
                })
                .map(|(index, _)| index)?;
            used_frontend_indices.insert(frontend_index);
            Some(super::FieldBinding {
                frontend_index,
                frontend_name: field.name().clone(),
                native_name: field.name().clone(),
            })
        })
        .collect()
}

fn required_source_scan_output_indices(
    source_schema: SchemaRef,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    required_output_fields: &RequiredFields,
) -> Result<Vec<usize>, FallbackReason> {
    if let Some(filter) = filter
        && let Some(projection) = filter.projection()
    {
        return required_output_fields
            .requested_indices(filter.schema().fields().len())?
            .into_iter()
            .map(|output_index| projection.get(output_index).copied().ok_or(FallbackReason::UnsupportedExpression))
            .collect();
    }
    required_output_fields.requested_indices(source_schema.fields().len())
}

pub(super) fn add_source_filter_projection(
    builder: &mut NativeQueryPlanBuilder,
    filtered_node: nexus_query_engine::NodeId,
    scan_output: &LoweredNativeInput,
    filter_schema: SchemaRef,
    projection: &[usize],
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let requested_output_indices = required_output_fields.requested_indices(filter_schema.fields().len())?;
    let requested = RequiredFields::only(requested_output_indices.iter().copied().collect());
    let (schema, bindings) = executable_schema_and_bindings(&filter_schema, &requested)?;
    let expressions = requested_output_indices
        .iter()
        .zip(bindings.iter())
        .map(|(output_index, output_binding)| {
            let input_index = projection.get(*output_index).copied().ok_or(FallbackReason::UnsupportedExpression)?;
            let input_binding = scan_output.binding_for_frontend_index(input_index)?;
            Ok(alias_if_needed(
                nexus_query_engine::Expr::column(input_binding.native_name.as_str()),
                output_binding.native_name.as_str(),
            ))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let node = builder.add_projection(filtered_node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

pub(super) struct RequiredScanProjectionExpr<'a> {
    pub(super) output_index: usize,
    pub(super) expr: &'a Arc<dyn PhysicalExpr>,
    pub(super) alias: &'a str,
}

pub(super) fn required_scan_projection<'a>(
    projection: &'a ProjectionExprs,
    scan_output_schema: SchemaRef,
    required_scan_output_columns: &[String],
) -> Vec<RequiredScanProjectionExpr<'a>> {
    projection
        .iter()
        .enumerate()
        .filter(|projection_expr| {
            let (_output_index, projection_expr) = projection_expr;
            required_scan_output_columns.iter().any(|column| column == &projection_expr.alias)
                || scan_output_schema.index_of(projection_expr.alias.as_str()).is_ok_and(|index| {
                    scan_output_schema
                        .fields()
                        .get(index)
                        .is_some_and(|field| required_scan_output_columns.iter().any(|column| column == field.name()))
                })
        })
        .map(|(output_index, projection_expr)| RequiredScanProjectionExpr {
            output_index,
            expr: &projection_expr.expr,
            alias: projection_expr.alias.as_str(),
        })
        .collect()
}

pub(super) fn lower_required_file_source_projection_exprs(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .iter()
        .map(|projection_expr| {
            let expr = lower_physical_expr(projection_expr.expr, input_schema)?;
            Ok(alias_if_needed(expr, projection_expr.alias))
        })
        .collect()
}

pub(super) fn required_scan_projection_requires_native_projection(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> bool {
    projection.iter().any(|projection_expr| {
        let Some(column) = projection_expr.expr.as_any().downcast_ref::<Column>() else {
            return true;
        };
        input_schema.fields().get(column.index()).is_none_or(|field| field.name() != projection_expr.alias)
    })
}

pub(super) fn required_scan_projection_needs_native_projection(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
    current_source_schema: &SchemaRef,
    scan_output_schema: SchemaRef,
) -> Result<bool, FallbackReason> {
    if required_scan_projection_requires_native_projection(projection, input_schema) {
        return Ok(true);
    }
    let projected_schema = required_scan_projection_schema(projection, scan_output_schema)?;
    Ok(projected_schema.as_ref() != current_source_schema.as_ref())
}

pub(super) fn required_scan_projection_schema(
    projection: &[RequiredScanProjectionExpr<'_>],
    scan_output_schema: SchemaRef,
) -> Result<SchemaRef, FallbackReason> {
    let fields = projection
        .iter()
        .map(|projection_expr| {
            scan_output_schema
                .fields()
                .get(projection_expr.output_index)
                .cloned()
                .ok_or(FallbackReason::UnsupportedExpression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

#[cfg(test)]
mod tests {
    use nexus_query_engine::{CredentialMode, ResolvedParquetSource, SourceColumnRef as NativeSourceColumnRef};

    use super::*;
    use crate::table_format::{TableFormat, TableFormatReadCapability};

    fn test_column_requirements() -> NativeParquetColumnRequirements {
        NativeParquetColumnRequirements {
            read_columns: vec!["c0".to_string()],
            read_column_file_indices: vec![0],
            projected_output_columns: vec!["c0".to_string()],
            projected_output_column_source_refs: vec![NativeSourceColumnRef::File(0)],
            pure_filter_columns: Vec::new(),
            pure_filter_column_source_refs: Vec::new(),
            partition_columns: Vec::new(),
            partition_column_indices: Vec::new(),
            missing_columns: Vec::new(),
            scan_output_columns: vec!["c0".to_string()],
        }
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
}
