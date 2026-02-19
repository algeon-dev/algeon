use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use cudf_query_engine::{
    Expr as NativeExpr, LiteralValue as NativeLiteralValue, LocalParquetFile as NativeLocalParquetFile,
    LocalParquetSource as NativeLocalParquetSource, MissingColumn as NativeMissingColumn,
    PartitionColumn as NativePartitionColumn, QueryPlanBuilder as NativeQueryPlanBuilder,
    SourceColumnRef as NativeSourceColumnRef, SourceNode as NativeSourceNode,
};
use datafusion::{
    datasource::physical_plan::{FileSource, ParquetSource},
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::ExecutionPlan,
};
use parquet::file::reader::FileReader;

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
    table_format::{
        CudfTableFormatScanExec, TableFormatColumnMapping, TableFormatColumnSource, TableFormatDataFile,
        TableFormatDataFileFormat, TableFormatPartitionTransform, TableFormatReadCapability, TableFormatScalarValue,
        TableFormatScanPlan,
    },
};

pub(super) fn lower_source_input(
    source: &NormalizedNativeSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let NormalizedNativeScan::LocalParquet(source_exec) = source.scan else {
        let scan_exec = source.table_format_scan_exec().ok_or(FallbackReason::TableFormatUnsupported)?;
        return lower_table_format_source_input(scan_exec, source.filter, builder, required_output_fields);
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
    .with_scan_limit(file_scan.limit);
    let mut reader_pushdown_covers_filter = false;
    if let Some(predicate) = source_reader_pushdown_predicate(
        parquet_source,
        filter,
        file_scan.limit,
        scan_projection_requires_native,
        &source_output_schema,
        source_exec.schema(),
        &native_source,
    )? {
        reader_pushdown_covers_filter = predicate.covers_filter;
        native_source = native_source.with_reader_predicate(predicate.expr);
    }
    let reader_pushdown = native_source.reader_predicate.is_some();
    native_source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        Arc::clone(&table_schema),
        Some(&column_requirements),
        reader_pushdown,
        false,
    );

    let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection {
        let required_scan_projection =
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns);
        if required_scan_projection_requires_native_projection(&required_scan_projection, &table_schema) {
            let expressions = lower_required_file_source_projection_exprs(&required_scan_projection, &table_schema)?;
            let schema = required_scan_projection_schema(&required_scan_projection, source_exec.schema())?;
            (builder.add_projection(source_node, projection_outputs(expressions, &schema)?), schema)
        } else {
            (source_node, Arc::clone(&source_output_schema))
        }
    } else {
        (source_node, Arc::clone(&source_output_schema))
    };
    let scan_output_bindings = source_bindings_for_schema(&scan_output_schema, source_exec.schema());
    let scan_output = LoweredNativeInput::with_bindings(scan_output_node, scan_output_schema, scan_output_bindings);
    let scan_only_post_scan_predicate = if filter.is_none() && !reader_pushdown {
        parquet_source.filter().and_then(|predicate| scan_predicate_without_dynamic_filter_hints(&predicate))
    } else {
        None
    };
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
    } else if let Some(predicate) = scan_only_post_scan_predicate {
        let filtered_node = builder.add_filter(
            scan_output.node,
            lower_physical_expr(&predicate, &scan_output.schema)?,
            Arc::clone(&scan_output.schema),
        );
        project_scan_output_columns(
            builder,
            filtered_node,
            scan_output.schema,
            scan_output.bindings,
            required_output_fields,
        )
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

fn lower_table_format_source_input(
    scan_exec: &CudfTableFormatScanExec,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let scan_plan = scan_exec.scan_plan();
    validate_table_format_scan_plan(scan_plan)?;
    validate_schema(scan_exec.schema())?;
    if let Some(filter) = filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }

    let requirements = table_format_column_requirements(scan_plan, scan_exec.schema(), filter, required_output_fields)?;
    let source_output_schema = table_format_source_output_schema(&requirements, &table_format_file_schema(scan_plan));
    validate_schema(Arc::clone(&source_output_schema))?;
    let native_files = table_format_local_parquet_files(scan_plan, &requirements)?;
    let native_source = NativeLocalParquetSource::from_files(
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
    .with_missing_columns(requirements.missing_columns.clone());

    let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
    let scan_output_bindings = source_bindings_for_schema(&source_output_schema, scan_exec.schema());
    let scan_output = LoweredNativeInput::with_bindings(source_node, source_output_schema, scan_output_bindings);
    if let Some(filter) = filter {
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
            add_source_filter_projection(
                builder,
                filtered_node,
                &scan_output,
                filter.schema(),
                projection,
                required_output_fields,
            )
        } else {
            Ok(LoweredNativeInput::with_bindings(filtered_node, scan_output.schema, scan_output.bindings))
        }
    } else {
        Ok(scan_output)
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
    let source_node =
        builder.add_source(NativeSourceNode::LocalParquet(NativeLocalParquetSource::zero_column_count(native_files)));
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
    if plan.data_files.iter().any(|file| !file.read_capability.is_direct_native_parquet_capable()) {
        return FallbackReason::IcebergObjectStoreReadUnsupported;
    }
    FallbackReason::TableFormatUnsupported
}

fn table_format_column_requirements(
    scan_plan: &TableFormatScanPlan,
    scan_schema: SchemaRef,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    required_output_fields: &RequiredFields,
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let file_schema = table_format_file_schema(scan_plan);
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
    let mut filter_columns = BTreeSet::new();
    if let Some(filter) = filter {
        collect_required_columns(filter.predicate(), &filter.input().schema(), &mut filter_columns)?;
    }

    let output_column_set = output_columns.iter().cloned().collect::<BTreeSet<_>>();
    let required_columns = filter_columns.union(&output_column_set).cloned().collect::<BTreeSet<_>>();
    let mappings = table_format_column_mapping_by_name(scan_plan);
    let mut read_columns = BTreeSet::new();
    let mut partition_columns = Vec::new();
    let mut missing_columns = Vec::new();
    for column in &required_columns {
        let mapping = mappings.get(column.as_str()).ok_or(FallbackReason::NativeSourceSchemaMapping)?;
        match mapping.source {
            TableFormatColumnSource::DirectTopLevelFileColumn => {
                let source_name = mapping.source_name.as_ref().unwrap_or(&mapping.logical_name);
                if file_schema.index_of(source_name).is_err() {
                    return Err(FallbackReason::NativeSourceSchemaMapping);
                }
                read_columns.insert(source_name.clone());
            },
            TableFormatColumnSource::IdentityPartitionValue => {
                if !is_supported_native_type(&mapping.data_type) {
                    return Err(FallbackReason::UnsupportedType);
                }
                partition_columns.push(NativePartitionColumn::new(
                    mapping.logical_name.as_str(),
                    mapping.data_type.clone(),
                    mapping.nullable,
                ));
            },
            TableFormatColumnSource::MissingNullableColumn => {
                if !mapping.nullable {
                    return Err(FallbackReason::NativeSourceSchemaMapping);
                }
                if !is_supported_native_type(&mapping.data_type) {
                    return Err(FallbackReason::UnsupportedType);
                }
                missing_columns.push(NativeMissingColumn::new(
                    mapping.logical_name.as_str(),
                    mapping.data_type.clone(),
                    mapping.nullable,
                ));
            },
            TableFormatColumnSource::UnsupportedNestedProjection => {
                return Err(FallbackReason::IcebergNestedProjectionUnsupported);
            },
        }
    }
    partition_columns.sort_by(|left, right| left.name.cmp(&right.name));
    partition_columns.dedup_by(|left, right| left.name == right.name);
    missing_columns.sort_by(|left, right| left.name.cmp(&right.name));
    missing_columns.dedup_by(|left, right| left.name == right.name);

    let read_columns = read_columns.into_iter().collect::<Vec<_>>();
    let pure_filter_columns = filter_columns.difference(&output_column_set).cloned().collect::<Vec<_>>();
    let projected_output_columns = output_columns;
    let read_column_file_indices = read_columns
        .iter()
        .map(|column| file_schema.index_of(column).map_err(|_error| FallbackReason::NativeSourceSchemaMapping))
        .collect::<Result<Vec<_>, _>>()?;
    let projected_output_column_source_refs =
        table_format_column_source_refs(&file_schema, &partition_columns, &missing_columns, &projected_output_columns)?;
    let pure_filter_column_source_refs =
        table_format_column_source_refs(&file_schema, &partition_columns, &missing_columns, &pure_filter_columns)?;
    let scan_output_columns = required_columns.iter().cloned().collect::<Vec<_>>();

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
    if !file.read_capability.is_direct_native_parquet_capable() {
        return Err(FallbackReason::IcebergObjectStoreReadUnsupported);
    }
    let stable_id = format!("table-format:{}:{}", scan_plan.format.code(), file.id);
    let mut native_file =
        table_format_native_parquet_file(&file.location, &file.read_capability)?.with_stable_id(stable_id);
    if let Some(file_size_bytes) = file.size_bytes {
        native_file = native_file.with_file_size_bytes(file_size_bytes);
    }
    if let Some(row_groups) = file.row_groups.clone() {
        native_file = native_file.with_row_groups(row_groups);
    }
    if requirements.read_columns.is_empty()
        && requirements.missing_columns.is_empty()
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
    location.strip_prefix("file://").map_or_else(|| PathBuf::from(location), PathBuf::from)
}

fn table_format_native_parquet_file(
    location: &str,
    read_capability: &TableFormatReadCapability,
) -> Result<NativeLocalParquetFile, FallbackReason> {
    match read_capability {
        TableFormatReadCapability::LocalFileSystem => {
            Ok(NativeLocalParquetFile::new(table_format_local_path(location)))
        },
        TableFormatReadCapability::ObjectStore { scheme } if scheme.eq_ignore_ascii_case("s3") => {
            NativeLocalParquetFile::object_store_uri(location.to_string())
                .map_err(|_reason| FallbackReason::IcebergObjectStoreReadUnsupported)
        },
        TableFormatReadCapability::ObjectStore { .. } | TableFormatReadCapability::Unsupported { .. } => {
            Err(FallbackReason::IcebergObjectStoreReadUnsupported)
        },
    }
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
        .map_or_else(|| Arc::clone(&scan_plan.native_source_schema), Arc::new)
}

fn project_scan_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input_node: cudf_query_engine::NodeId,
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
    scan_projection_requires_native: bool,
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
    if scan_limit.is_some() || scan_projection_requires_native {
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
        let bindings = source_bindings_for_schema(source_output_schema, frontend_schema);
        lower_physical_expr_with_bindings(filter.predicate(), &filter.input().schema(), Some(&bindings))
    } else {
        let Some(reader_scan_predicate) = reader_scan_predicate.as_ref() else {
            return Ok(None);
        };
        lower_physical_expr(reader_scan_predicate, source_output_schema)
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
    filtered_node: cudf_query_engine::NodeId,
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
                cudf_query_engine::Expr::column(input_binding.native_name.as_str()),
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
