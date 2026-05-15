use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use cudf_query_engine::{
    Expr as NativeExpr, LocalParquetSource as NativeLocalParquetSource, QueryPlanBuilder as NativeQueryPlanBuilder,
    SourceNode as NativeSourceNode,
};
use datafusion::{
    datasource::physical_plan::{FileSource, ParquetSource},
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::ExecutionPlan,
};

use super::{
    LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, lower_physical_expr, lower_physical_expr_with_bindings},
    projection_outputs,
};
use crate::{
    native::{
        normalize::NormalizedNativeSource,
        pruning::{
            scan_predicate_exactly_matches_filter, scan_predicate_is_dynamic_filter_hint,
            scan_predicate_without_dynamic_filter_hints, validate_pruning_only_parquet_predicate,
        },
        source::{
            local_parquet_files, local_parquet_source_diagnostics, native_source_output_schema,
            required_parquet_column_requirements_for_scan_columns, validate_slice1_parquet_scan,
        },
        validate::{validate_physical_expr, validate_schema},
    },
    policy::FallbackReason,
};

pub(super) fn lower_source_input(
    source: &NormalizedNativeSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let source_exec = source.source_exec;
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
        Ok(scan_output)
    }
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
        let Some(reader_scan_predicate) = reader_scan_predicate else {
            return Ok(None);
        };
        lower_physical_expr(&reader_scan_predicate, source_output_schema)
    };
    let predicate = match predicate {
        Ok(predicate) => predicate,
        Err(_error) if filter.is_some() => return Ok(None),
        Err(error) => return Err(error),
    };
    if native_source.reader_predicate_is_supported(&predicate) {
        Ok(Some(SourceReaderPushdownPredicate { expr: predicate, covers_filter }))
    } else if filter.is_some() {
        Ok(None)
    } else {
        Err(FallbackReason::NativeSourcePushedDownParquetPredicate)
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
