use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::ExecutionPlan;
use nexus_query_engine::{
    Expr as NativeExpr, LimitSpec as NativeLimitSpec, LiteralValue as NativeLiteralValue,
    LocalParquetFile as NativeLocalParquetFile, LocalParquetSource as NativeLocalParquetSource,
    MissingColumn as NativeMissingColumn, PartitionColumn as NativePartitionColumn,
    QueryPlanBuilder as NativeQueryPlanBuilder, SourceColumnRef as NativeSourceColumnRef,
    SourceNode as NativeSourceNode,
};
use parquet::file::reader::FileReader;

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
        CudfTableFormatScanExec, TableFormatColumnMapping, TableFormatColumnSource, TableFormatDataFile,
        TableFormatDataFileFormat, TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata,
        TableFormatPartitionTransform, TableFormatScalarValue, TableFormatScanPlan,
        resolve_table_format_parquet_source_with_metadata,
    },
};

const TABLE_FORMAT_FILTER_ELISION_MIN_SOURCE_ROWS: u64 = 10_000_000;

pub(super) fn lower_table_format_source_input(
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
    .with_file_column_count(file_schema.fields().len())
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
    // (in `BTreeSet` iteration order - they never appear in `scan_schema` and
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

fn table_format_local_parquet_files(
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
