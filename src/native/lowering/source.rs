use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use cudf_query_engine::{
    Expr as NativeExpr, LocalParquetSource as NativeLocalParquetSource, QueryPlanBuilder as NativeQueryPlanBuilder,
    SourceNode as NativeSourceNode,
};
use datafusion::{
    datasource::physical_plan::ParquetSource,
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::ExecutionPlan,
};

use super::{
    LoweredNativeInput,
    expr::{alias_if_needed, lower_physical_expr},
    unary::lower_filter_projection_exprs,
};
use crate::{
    native::{
        normalize::NormalizedNativeSource,
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
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    let source_exec = source.source_exec;
    let filter = source.filter;
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;
    let column_requirements =
        required_parquet_column_requirements_for_scan_columns(file_scan, filter, required_output_columns)?;
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
    .with_partition_columns(column_requirements.partition_columns.clone());
    native_source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        Arc::clone(&table_schema),
        Some(&column_requirements),
        false,
    );

    let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection {
        let required_scan_projection =
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns);
        if required_scan_projection_requires_native_projection(&required_scan_projection, &table_schema) {
            let expressions = lower_required_file_source_projection_exprs(&required_scan_projection, &table_schema)?;
            let schema = required_scan_projection_schema(&required_scan_projection, source_exec.schema())?;
            (builder.add_projection(source_node, expressions, Arc::clone(&schema)), schema)
        } else {
            (source_node, Arc::clone(&source_output_schema))
        }
    } else {
        (source_node, Arc::clone(&source_output_schema))
    };
    let filter_input_schema = source_exec.schema();
    let (node, schema) = if let Some(filter) = filter {
        let filtered_node = builder.add_filter(
            scan_output_node,
            lower_physical_expr(filter.predicate(), &scan_output_schema)?,
            Arc::clone(&scan_output_schema),
        );
        if let Some(projection) = filter.projection() {
            (
                builder.add_projection(
                    filtered_node,
                    lower_filter_projection_exprs(projection, &filter_input_schema)?,
                    filter.schema(),
                ),
                filter.schema(),
            )
        } else {
            (filtered_node, scan_output_schema)
        }
    } else {
        (scan_output_node, scan_output_schema)
    };
    Ok(LoweredNativeInput { node, schema })
}

pub(super) struct RequiredScanProjectionExpr<'a> {
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
        .filter(|projection_expr| {
            required_scan_output_columns.iter().any(|column| column == &projection_expr.alias)
                || scan_output_schema.index_of(projection_expr.alias.as_str()).is_ok_and(|index| {
                    scan_output_schema
                        .fields()
                        .get(index)
                        .is_some_and(|field| required_scan_output_columns.iter().any(|column| column == field.name()))
                })
        })
        .map(|projection_expr| RequiredScanProjectionExpr {
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
                .field_with_name(projection_expr.alias)
                .cloned()
                .map_err(|_error| FallbackReason::UnsupportedExpression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(Schema::new(fields)))
}
