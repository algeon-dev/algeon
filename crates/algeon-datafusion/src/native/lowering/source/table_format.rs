use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::{ExecutionPlan, filter::FilterExec};
use parquet::{arrow::PARQUET_FIELD_ID_META_KEY, file::reader::FileReader};
use query_engine::{
    expr::{Expr as NativeExpr, LimitSpec as NativeLimitSpec, LiteralValue as NativeLiteralValue},
    plan::QueryPlanBuilder as NativeQueryPlanBuilder,
    source::{
        LiteralBatchSource as NativeLiteralBatchSource, LocalParquetFile as NativeLocalParquetFile,
        LocalParquetSource as NativeLocalParquetSource, MissingColumn as NativeMissingColumn,
        ParquetEqualityDeleteFile as NativeParquetEqualityDeleteFile,
        ParquetFieldIdReadBinding as NativeParquetFieldIdReadBinding,
        ParquetPositionDeleteFile as NativeParquetPositionDeleteFile, PartitionColumn as NativePartitionColumn,
        SourceColumnRef as NativeSourceColumnRef, SourceNode as NativeSourceNode,
        ZERO_COLUMN_COUNT_COLUMN as NATIVE_ZERO_COLUMN_COUNT_COLUMN,
    },
};

use super::{
    SourceReaderPushdownPredicate, SourceUnprojectedFilterOutput, combine_scan_limits,
    lower_source_filter_and_projection, project_scan_output_columns, required_source_scan_output_indices,
    source_bindings_for_schema, source_unique_key_columns, supported_reader_pushdown_predicate,
};
use crate::{
    capability::CapabilityReason,
    native::{
        expression_contract::validate_physical_expr_contract,
        lowering::{LoweredNativeInput, RequiredFields, expr::lower_physical_expr_with_bindings},
        normalize::NormalizedNativeSource,
        source::{NativeParquetColumnRequirements, attach_row_group_byte_metadata},
        table_format_pruning_policy::TableFormatPruningPolicy,
        validate::{collect_required_columns, validate_schema},
    },
    source_cardinality::reader_predicate_proves_source_at_most_one_row,
    table_format::{
        AlgeonIcebergScanExec, TableFormat, TableFormatColumnMapping, TableFormatColumnSource, TableFormatDataFile,
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

pub(in crate::native::lowering) use admission::{available_table_format_scan_plan, table_format_scan_rejection};
use admission::{
    table_format_column_mapping_by_name, table_format_column_requirements, table_format_scan_plan_is_empty_admissible,
    table_format_scan_plan_rejection, validate_table_format_scan_plan,
};
#[cfg(test)]
use count_star::lower_table_format_count_star_fallback_source_for_plan;
use count_star::table_format_count_star_fallback_column;
pub(in crate::native::lowering) use count_star::{
    lower_empty_table_format_count_star_source, lower_table_format_count_star_fallback_source,
    lower_table_format_count_star_mixed_row_count_source, lower_table_format_zero_column_count_source,
    table_format_count_star_mixed_row_count_metadata_can_split, table_format_count_star_probe_can_lower,
    table_format_zero_column_count_metadata_is_available,
};
use field_id::{
    TableFormatFieldIdReadBinding, TableFormatFileReadBindings, table_format_file_field_id_read_bindings,
    table_format_requires_iceberg_field_id_bindings, validate_table_format_direct_file_column_field_ids,
};
use files::{
    table_format_local_parquet_files, table_format_local_parquet_files_for_group, table_format_local_path,
    table_format_native_parquet_file, table_format_scalar_value_to_native_literal,
};
use groups::{TableFormatSourceReadGroup, table_format_source_output_schema, table_format_source_read_groups};
use schema_match::{
    table_format_arrow_data_types_match_except_list_child_names, table_format_data_file_footer_arrow_schema,
    table_format_file_schema, table_format_footer_data_type_needs_logical_source_contract,
    table_format_footer_field_matches_mapping, table_format_local_footer_schema,
};

pub(super) fn lower_table_format_source_input(
    scan_exec: &AlgeonIcebergScanExec,
    filter: Option<&FilterExec>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    pushed_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let scan_plan = available_table_format_scan_plan(scan_exec)?;
    if table_format_scan_plan_is_empty_admissible(scan_plan) {
        return lower_empty_table_format_source_input(scan_exec, filter, builder, required_output_fields);
    }
    validate_table_format_scan_plan(scan_plan)?;
    validate_schema(scan_exec.schema())?;
    if let Some(filter) = filter {
        validate_physical_expr_contract(filter.predicate(), &filter.input().schema())?;
    }
    if scan_plan.data_files.iter().any(TableFormatDataFile::has_equality_delete_files) {
        return Err(CapabilityReason::IcebergDeleteFilesUnsupported);
    }

    let file_schema = table_format_file_schema(scan_plan);
    let requirements =
        table_format_column_requirements(scan_plan, &file_schema, scan_exec.schema(), filter, required_output_fields)?;
    if scan_plan.data_files.iter().any(TableFormatDataFile::has_row_delete_files)
        && requirements.read_columns.is_empty()
    {
        return Err(CapabilityReason::IcebergDeleteFilesUnsupported);
    }
    let source_output_schema = table_format_source_output_schema(scan_plan, &requirements, &file_schema)?;
    validate_schema(Arc::clone(&source_output_schema))?;
    let scan_output_bindings = source_bindings_for_schema(&source_output_schema, scan_exec.schema());
    let unique_key_columns = source_unique_key_columns(scan_exec, &scan_output_bindings);
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
            &unique_key_columns,
        )?;
        let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
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
                    &unique_key_columns,
                )
                .map(|(source, _pushdown)| builder.add_source(NativeSourceNode::LocalParquet(source)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let union_node = builder.add_union(inputs, Arc::clone(&source_output_schema));
        (LoweredNativeInput::with_bindings(union_node, Arc::clone(&source_output_schema), scan_output_bindings), false)
    };
    if let Some(filter) = filter {
        let filtered = lower_source_filter_and_projection(
            builder,
            scan_output,
            filter,
            reader_pushdown_covers_filter,
            required_output_fields,
            SourceUnprojectedFilterOutput::ProjectRequiredOutput,
        )?;
        if !reader_pushdown_covers_filter && let Some(scan_limit) = scan_plan.scan_limit {
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
    filter: Option<&FilterExec>,
    file_schema: &SchemaRef,
    frontend_schema: SchemaRef,
    source_scan_limit: Option<usize>,
    unique_key_columns: &[BTreeSet<String>],
) -> Result<(NativeLocalParquetSource, bool), CapabilityReason> {
    let native_files = table_format_local_parquet_files_for_group(scan_plan, group)?;
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
    .with_unique_key_columns(unique_key_columns.to_vec())
    .with_scan_limit(source_scan_limit);

    let mut reader_pushdown_covers_filter = false;
    if !group.has_field_id_aliases()
        && !group.has_read_coercions()
        && !native_source.files.iter().any(NativeLocalParquetFile::has_row_delete_files)
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
    scan_exec: &AlgeonIcebergScanExec,
    filter: Option<&FilterExec>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, CapabilityReason> {
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
        predicate,
    )
}

fn table_format_reader_pushdown_predicate(
    filter: Option<&FilterExec>,
    source_output_schema: &SchemaRef,
    frontend_schema: SchemaRef,
    native_source: &NativeLocalParquetSource,
) -> Result<Option<SourceReaderPushdownPredicate>, CapabilityReason> {
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
    TableFormatPruningPolicy::default()
        .reader_pushdown_can_elide_native_filter(table_format_source_known_rows(native_source))
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
        | NativeExpr::Trim { expr, .. }
        | NativeExpr::LTrim { expr, .. }
        | NativeExpr::RTrim { expr, .. }
        | NativeExpr::StringContains { expr, .. }
        | NativeExpr::StartsWith { expr, .. }
        | NativeExpr::EndsWith { expr, .. }
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
mod tests;
