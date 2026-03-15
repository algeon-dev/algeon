use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::ExecutionPlan;
use nexus_query_engine::{
    Expr as NativeExpr, LimitSpec as NativeLimitSpec, LiteralBatchSource as NativeLiteralBatchSource,
    LiteralValue as NativeLiteralValue, LocalParquetFile as NativeLocalParquetFile,
    LocalParquetSource as NativeLocalParquetSource, MissingColumn as NativeMissingColumn,
    ParquetFieldIdReadBinding as NativeParquetFieldIdReadBinding,
    ParquetPositionDeleteFile as NativeParquetPositionDeleteFile, PartitionColumn as NativePartitionColumn,
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
        validate::{
            collect_required_columns, is_list_carry_type, is_supported_native_type, validate_physical_expr,
            validate_schema,
        },
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

mod admission;
mod count_star;
mod field_id;
mod files;
mod groups;
mod schema_match;

pub(in crate::native::lowering) use admission::table_format_scan_rejection;
use admission::*;
#[cfg(test)]
use count_star::lower_table_format_count_star_fallback_source_for_plan;
use count_star::table_format_count_star_fallback_column;
pub(in crate::native::lowering) use count_star::{
    lower_empty_table_format_count_star_source, lower_table_format_count_star_fallback_source,
    lower_table_format_count_star_mixed_row_count_source, lower_table_format_zero_column_count_source,
    table_format_count_star_mixed_row_count_metadata_can_split, table_format_count_star_probe_can_lower,
    table_format_zero_column_count_metadata_is_available,
};
use field_id::*;
use files::*;
use groups::*;
use schema_match::*;

const TABLE_FORMAT_FILTER_ELISION_MIN_SOURCE_ROWS: u64 = 10_000_000;

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
    if scan_plan.data_files.iter().any(TableFormatDataFile::has_position_delete_files)
        && requirements.read_columns.is_empty()
    {
        return Err(FallbackReason::IcebergDeleteFilesUnsupported);
    }
    let source_output_schema = table_format_source_output_schema(scan_plan, &requirements, &file_schema)?;
    validate_schema(Arc::clone(&source_output_schema))?;
    let read_groups = table_format_source_read_groups(scan_plan, &requirements, Arc::clone(&source_output_schema))?;
    let effective_scan_limit =
        if filter.is_none() { combine_scan_limits(scan_plan.scan_limit, pushed_scan_limit) } else { None };
    // Iceberg position-delete sources treat this as a non-guaranteed visible-row
    // stop bound. The native upper Limit remains above the source, so connector
    // planning can read smaller chunks without changing SQL limit semantics.
    let source_scan_limit = effective_scan_limit;
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
                table_format_native_source_for_group(
                    scan_plan,
                    group,
                    None,
                    &file_schema,
                    scan_exec.schema(),
                    source_scan_limit,
                )
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
        let scan_output = if let Some(scan_limit) = effective_scan_limit {
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
        && !native_source.files.iter().any(NativeLocalParquetFile::has_position_delete_files)
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

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeSet, HashMap},
        fs,
        fs::File,
        time::{SystemTime, UNIX_EPOCH},
    };

    use arrow::{array::Int64Array, record_batch::RecordBatch};
    use nexus_query_engine::{
        CredentialMode, PlanNode, ResolvedParquetSource, SourceColumnRef as NativeSourceColumnRef,
        validate_local_parquet_source,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use super::*;
    use crate::table_format::{
        TableFormat, TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatPartitionTransform,
        TableFormatPartitionValue, TableFormatReadCapability, TableFormatScalarValue,
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
            requires_physical_row_presence: false,
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
    fn table_format_local_footer_missing_field_id_uses_same_name_direct_file_column() {
        let dir = unique_table_format_temp_dir("field-id-missing-name-match");
        let path = dir.join("part.parquet");
        write_plain_parquet_file_with_nullability(&path, "status", true);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let scan_plan = single_projected_file_scan_plan_with_nullability(&path, Arc::clone(&scan_schema), 2, true);
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].requirements.read_columns, vec!["status"]);
        assert_eq!(groups[0].requirements.read_column_file_indices, vec![0]);
        assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
        assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "status");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_footer_malformed_field_id_metadata_rejects_same_name_fallback() {
        let footer_field = Field::new("status", DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "bad-id".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
    }

    #[test]
    fn table_format_direct_timestamp_timezone_uses_timezone_free_source_contract() {
        let timestamp_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into()));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", timestamp_type.clone(), true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&scan_schema))
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                None,
                "event_ts",
                Some("event_ts"),
                timestamp_type.clone(),
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, Arc::clone(&output_schema)).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].source_schema.field_with_name("event_ts").unwrap().data_type(),
            &DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
        );
        assert_eq!(groups[0].output_schema.field_with_name("event_ts").unwrap().data_type(), &timestamp_type);
    }

    #[test]
    fn table_format_timestamp_partition_source_contract_keeps_logical_timezone() {
        let timestamp_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into()));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", timestamp_type.clone(), false)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_row_count(1)
        .with_partition_values(vec![TableFormatPartitionValue::new(
            "event_ts",
            timestamp_type.clone(),
            false,
            TableFormatPartitionTransform::Identity,
            TableFormatScalarValue::Int64(123),
        )])
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&scan_schema))
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                None,
                "event_ts",
                None::<String>,
                timestamp_type.clone(),
                false,
                TableFormatColumnSource::IdentityPartitionValue,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements = table_format_column_requirements(
            &scan_plan,
            &file_schema,
            Arc::clone(&scan_schema),
            None,
            &RequiredFields::all(),
        )
        .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
        let (native_source, _) =
            table_format_native_source_for_group(&scan_plan, &groups[0], None, &file_schema, scan_schema, None)
                .unwrap();

        assert_eq!(groups[0].source_schema.field_with_name("event_ts").unwrap().data_type(), &timestamp_type);
        validate_local_parquet_source(&native_source).unwrap();
    }

    #[test]
    fn table_format_timestamp_missing_source_contract_keeps_logical_timezone() {
        let timestamp_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into()));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", timestamp_type.clone(), true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_row_count(1)
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&scan_schema))
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                None,
                "event_ts",
                None::<String>,
                timestamp_type.clone(),
                true,
                TableFormatColumnSource::MissingNullableColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements = table_format_column_requirements(
            &scan_plan,
            &file_schema,
            Arc::clone(&scan_schema),
            None,
            &RequiredFields::all(),
        )
        .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
        let (native_source, _) =
            table_format_native_source_for_group(&scan_plan, &groups[0], None, &file_schema, scan_schema, None)
                .unwrap();

        assert_eq!(groups[0].source_schema.field_with_name("event_ts").unwrap().data_type(), &timestamp_type);
        validate_local_parquet_source(&native_source).unwrap();
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
    fn table_format_attached_remote_footer_schema_admits_field_id_mapping() {
        let footer_field = Field::new("old_status", DataType::Int64, true)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].requirements.read_columns, vec!["old_status"]);
        assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
        assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
        assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_status");
    }

    #[test]
    fn table_format_field_id_mapping_allows_list_child_field_name_drift() {
        let footer_list_type = DataType::List(Arc::new(
            Field::new("element", DataType::Utf8, true)
                .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "58".to_string())])),
        ));
        let logical_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let footer_field = Field::new("old_tags", footer_list_type, true)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("tags", logical_list_type.clone(), true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(3),
                "tags",
                Some("tags"),
                logical_list_type,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].requirements.read_columns, vec!["old_tags"]);
        assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "tags");
        assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_tags");
    }

    #[test]
    fn table_format_field_id_mapping_rejects_non_parquet_list_child_metadata_drift() {
        let footer_list_type =
            DataType::List(Arc::new(Field::new("element", DataType::Utf8, true).with_metadata(HashMap::from([
                (PARQUET_FIELD_ID_META_KEY.to_string(), "58".to_string()),
                ("extension".to_string(), "a".to_string()),
            ]))));
        let logical_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let footer_field = Field::new("old_tags", footer_list_type, true)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("tags", logical_list_type.clone(), true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(3),
                "tags",
                Some("tags"),
                logical_list_type,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
    }

    #[test]
    fn table_format_field_id_mapping_allows_required_footer_for_nullable_logical_field() {
        let footer_field = Field::new("old_status", DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
        assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_status");
        assert!(!groups[0].field_id_bindings[0][0].nullable);
    }

    #[test]
    fn table_format_field_id_mapping_rejects_nullable_footer_for_required_logical_field() {
        let footer_field = Field::new("old_status", DataType::Int64, true)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
    }

    #[test]
    fn table_format_field_id_mapping_allows_timestamp_timezone_footer_normalization() {
        let footer_field = Field::new(
            "event_ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into())),
            false,
        )
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "8".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let logical_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into()));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", logical_type.clone(), true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(8),
                "event_ts",
                Some("event_ts"),
                logical_type,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "event_ts");
    }

    #[test]
    fn table_format_missing_nullable_list_column_is_rejected_before_native_source_materialization() {
        let footer_field = Field::new("other", DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]));
        let footer_schema = Arc::new(Schema::new(vec![footer_field]));
        let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let scan_schema = Arc::new(Schema::new(vec![Field::new("tags", list_type.clone(), true)]));
        let data_file = TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .with_footer_schema(footer_schema);
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_data_file(data_file)
            .with_projected_column(TableFormatColumnMapping::new(
                Some(3),
                "tags",
                Some("tags"),
                list_type,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            ));
        let file_schema = table_format_file_schema(&scan_plan);
        let requirements =
            table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all())
                .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

        assert_eq!(error, FallbackReason::UnsupportedType);
    }

    #[test]
    fn table_format_field_id_alias_and_position_deletes_lower_into_one_native_source() {
        let dir = unique_table_format_temp_dir("field-id-rename-with-delete");
        let data_path = dir.join("part.parquet");
        let delete_path = dir.join("delete.parquet");
        write_field_id_parquet_file(&data_path, "old_status", 2);
        write_field_id_parquet_file(&delete_path, "pos", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let delete_file = TableFormatDeleteFile::for_data_file(
            "delete-0",
            "file-0",
            delete_path.display().to_string(),
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::PositionDeletes,
        )
        .with_read_capability(TableFormatReadCapability::local_file_system())
        .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(delete_path.display().to_string()));
        let mut scan_plan = single_projected_file_scan_plan(&data_path, Arc::clone(&scan_schema), 2);
        scan_plan.data_files[0] = scan_plan.data_files[0].clone().with_position_delete_files(vec![delete_file.clone()]);
        scan_plan.delete_files.push(delete_file);
        let scan_exec = CudfTableFormatScanExec::new(scan_plan, Arc::clone(&scan_schema));
        let mut builder = NativeQueryPlanBuilder::new();

        let lowered =
            lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
        let plan = builder.finish_with_host_sink(lowered.node);
        let source = plan
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    NativeSourceNode::LocalParquet(source) => Some(source),
                    _ => None,
                },
                _ => None,
            })
            .expect("lowered plan should contain one local parquet source");

        assert_eq!(source.read_columns, vec!["old_status".to_string()]);
        assert_eq!(source.projected_output_columns, vec!["status".to_string()]);
        assert_eq!(source.files[0].position_delete_files.len(), 1);
        assert_eq!(source.files[0].field_id_read_bindings.len(), 1);
        assert_eq!(source.files[0].field_id_read_bindings[0].logical_name, "status");
        assert_eq!(source.files[0].field_id_read_bindings[0].physical_name, "old_status");
        assert_eq!(source.diagnostics.schema_mapping_disposition.as_str(), "field_id_bindings");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_position_delete_scan_limit_is_pushed_with_upper_limit() {
        let dir = unique_table_format_temp_dir("position-delete-scan-limit");
        let data_path = dir.join("part.parquet");
        let delete_path = dir.join("delete.parquet");
        write_field_id_parquet_file(&data_path, "status", 2);
        write_field_id_parquet_file(&delete_path, "pos", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let delete_file = TableFormatDeleteFile::for_data_file(
            "delete-0",
            "file-0",
            delete_path.display().to_string(),
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::PositionDeletes,
        )
        .with_read_capability(TableFormatReadCapability::local_file_system())
        .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(delete_path.display().to_string()))
        .with_referenced_data_file(data_path.display().to_string())
        .with_position_bounds(Some(0), Some(10));
        let mut scan_plan =
            single_projected_file_scan_plan(&data_path, Arc::clone(&scan_schema), 2).with_scan_limit(Some(1));
        scan_plan.data_files[0] = scan_plan.data_files[0].clone().with_position_delete_files(vec![delete_file.clone()]);
        scan_plan.delete_files.push(delete_file);
        let scan_exec = CudfTableFormatScanExec::new(scan_plan, Arc::clone(&scan_schema));
        let mut builder = NativeQueryPlanBuilder::new();

        let lowered =
            lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
        let plan = builder.finish_with_host_sink(lowered.node);
        let source = plan
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    NativeSourceNode::LocalParquet(source) => Some(source),
                    _ => None,
                },
                _ => None,
            })
            .expect("lowered plan should contain one local parquet source");

        assert_eq!(source.scan_limit, Some(1));
        assert_eq!(
            source.files[0].position_delete_files[0].referenced_data_file.as_deref(),
            Some(data_path.display().to_string().as_str())
        );
        assert_eq!(source.files[0].position_delete_files[0].position_lower_bound, Some(0));
        assert_eq!(source.files[0].position_delete_files[0].position_upper_bound, Some(10));
        assert!(matches!(plan.root_node().kind, PlanNode::Limit(_)));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_local_footer_renamed_physical_columns_lower_as_union_groups() {
        let dir = unique_table_format_temp_dir("field-id-rename-union");
        let old_path = dir.join("part-old.parquet");
        let new_path = dir.join("part-new.parquet");
        write_field_id_parquet_file_with_nullability(&old_path, "old_status", 2, true);
        write_field_id_parquet_file_with_nullability(&new_path, "status", 2, true);
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
    fn table_format_multi_group_scan_limit_is_pushed_into_each_source() {
        let dir = unique_table_format_temp_dir("field-id-rename-union-limit");
        let old_path = dir.join("part-old.parquet");
        let new_path = dir.join("part-new.parquet");
        write_field_id_parquet_file_with_nullability(&old_path, "old_status", 2, true);
        write_field_id_parquet_file_with_nullability(&new_path, "status", 2, true);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&scan_schema))
            .with_native_source_schema(Arc::clone(&scan_schema))
            .with_scan_limit(Some(1))
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
        let scan_limits = plan
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    NativeSourceNode::LocalParquet(source) => Some(source.scan_limit),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(scan_limits, vec![Some(1), Some(1)]);
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
            .expect_err("field-id mismatch must reject COUNT(*) fallback carrier column");

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
            .expect_err("Iceberg COUNT(*) fallback must not synthesize a name-based carrier column");

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
            .expect_err("COUNT(*) fallback must not treat missing/null materialization as row-presence proof");

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

    #[test]
    fn table_format_count_star_fallback_admits_same_name_carrier_without_field_id() {
        let dir = unique_table_format_temp_dir("count-star-field-id-missing-name-match");
        let path = dir.join("part.parquet");
        write_plain_parquet_file_with_nullability(&path, "status", false);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
        let mut builder = NativeQueryPlanBuilder::new();

        let lowered = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
            .expect("same-name carrier without a footer field id should admit COUNT(*) fallback");

        assert_eq!(lowered.schema.field(0).name(), NATIVE_ZERO_COLUMN_COUNT_COLUMN);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_empty_required_fields_uses_field_id_carrier_without_projecting_it() {
        let dir = unique_table_format_temp_dir("empty-required-field-id-carrier");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "old_status", 2);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
        let file_schema = table_format_file_schema(&scan_plan);

        let requirements = table_format_column_requirements(
            &scan_plan,
            &file_schema,
            Arc::clone(&scan_schema),
            None,
            &RequiredFields::only(BTreeSet::new()),
        )
        .unwrap();

        assert_eq!(requirements.read_columns, vec!["status"]);
        assert_eq!(requirements.read_column_file_indices, vec![0]);
        assert!(requirements.projected_output_columns.is_empty());
        assert_eq!(requirements.pure_filter_columns, vec!["status"]);
        assert_eq!(requirements.pure_filter_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
        assert_eq!(requirements.scan_output_columns, vec!["status"]);
        assert!(requirements.requires_physical_row_presence);

        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();
        let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
        let (native_source, _) =
            table_format_native_source_for_group(&scan_plan, &groups[0], None, &file_schema, scan_schema, None)
                .unwrap();

        assert_eq!(groups[0].requirements.read_columns, vec!["old_status"]);
        assert!(groups[0].requirements.projected_output_columns.is_empty());
        assert_eq!(groups[0].requirements.pure_filter_columns, vec!["status"]);
        assert_eq!(native_source.read_columns, vec!["old_status"]);
        assert!(native_source.projected_output_columns.is_empty());
        assert_eq!(native_source.pure_filter_columns, vec!["status"]);
        validate_local_parquet_source(&native_source).unwrap();
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_empty_required_fields_rejects_missing_field_id_carrier_column() {
        let dir = unique_table_format_temp_dir("empty-required-missing-carrier");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "other", 1);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
        let scan_plan = single_projected_file_scan_plan_with_nullability(&path, Arc::clone(&scan_schema), 2, true);
        let file_schema = table_format_file_schema(&scan_plan);

        let requirements = table_format_column_requirements(
            &scan_plan,
            &file_schema,
            scan_schema,
            None,
            &RequiredFields::only(BTreeSet::new()),
        )
        .unwrap();
        let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

        let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema)
            .expect_err("row-presence carrier must remain backed by a physical file read");

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn table_format_empty_required_fields_rejects_iceberg_implicit_local_carrier_column() {
        let dir = unique_table_format_temp_dir("empty-required-implicit-carrier");
        let path = dir.join("part.parquet");
        write_field_id_parquet_file(&path, "status", 2);
        let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
        let scan_plan = single_unmapped_file_scan_plan(&path, Arc::clone(&scan_schema));
        let file_schema = table_format_file_schema(&scan_plan);

        let error = table_format_column_requirements(
            &scan_plan,
            &file_schema,
            scan_schema,
            None,
            &RequiredFields::only(BTreeSet::new()),
        )
        .expect_err("Iceberg empty required fields must not synthesize a name-based carrier column");

        assert_eq!(error, FallbackReason::IcebergFieldIdMappingUnsupported);
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
        write_field_id_parquet_file_with_nullability(path, column_name, field_id, false);
    }

    fn write_field_id_parquet_file_with_nullability(path: &Path, column_name: &str, field_id: i32, nullable: bool) {
        let field = Field::new(column_name, DataType::Int64, nullable)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string())]));
        write_parquet_file_with_field(path, field);
    }

    fn write_plain_parquet_file_with_nullability(path: &Path, column_name: &str, nullable: bool) {
        write_parquet_file_with_field(path, Field::new(column_name, DataType::Int64, nullable));
    }

    fn write_parquet_file_with_field(path: &Path, field: Field) {
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
    fn table_format_position_deletes_preserve_contiguous_native_row_group_reads() {
        let delete_file = TableFormatDeleteFile::for_data_file(
            "delete-0",
            "file-0",
            "s3://bucket/table/delete-0.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::PositionDeletes,
        )
        .with_read_capability(TableFormatReadCapability::object_store("s3"))
        .with_resolved_parquet_source(ResolvedParquetSource::remote_host_buffer(
            "s3://bucket/table/delete-0.parquet",
            "s3://bucket/table/delete-0.parquet",
            CredentialMode::SharedProcessEnv,
        ));
        let mut scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg);
        scan_plan.data_files.push(
            TableFormatDataFile::new(
                "file-0",
                "s3://bucket/table/part-0.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(1024)
            .with_row_groups(vec![1, 2])
            .with_position_delete_files(vec![delete_file.clone()])
            .with_resolved_parquet_source(ResolvedParquetSource::remote_host_buffer(
                "s3://bucket/table/part-0.parquet",
                "s3://bucket/table/part-0.parquet",
                CredentialMode::SharedProcessEnv,
            )),
        );
        scan_plan.delete_files.push(delete_file);

        let files = table_format_local_parquet_files(&scan_plan, &test_column_requirements()).unwrap();

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].row_groups, Some(vec![1, 2]));
        assert_eq!(files[0].selected_row_group_count(), 2);
        assert_eq!(files[0].position_delete_files.len(), 1);
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
