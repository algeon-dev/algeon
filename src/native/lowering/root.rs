use std::{collections::BTreeSet, sync::Arc};

use cudf_query_engine::{
    LimitSpec as NativeLimitSpec, LocalParquetSource as NativeLocalParquetSource, QueryPlan as NativeQueryPlan,
    QueryPlanBuilder as NativeQueryPlanBuilder, SourceNode as NativeSourceNode,
};
use datafusion::{
    datasource::physical_plan::ParquetSource,
    physical_plan::{ExecutionPlan, projection::ProjectionExec},
};

use super::{
    aggregate::{
        RequiredScalarExpressions, lower_aggregate_spec, lower_zero_column_count_aggregate_spec,
        required_aggregate_input_columns, synthetic_aggregate_input_projection,
        synthetic_aggregate_input_projection_if_needed, validate_aggregate_root_operator_schemas,
        validate_joined_aggregate_scalar_expressions, validate_required_scalar_expressions,
    },
    expr::lower_physical_expr,
    lower_native_input,
    sort::lower_sort_spec,
    source::{
        lower_required_file_source_projection_exprs, required_scan_projection,
        required_scan_projection_requires_native_projection, required_scan_projection_schema,
    },
    unary::{lower_filter_projection_exprs, lower_projection_exprs},
};
use crate::{
    native::{
        normalize::{NormalizedNativeAggregateRoot, NormalizedNativeInputRoot, NormalizedNativePlan},
        source::{
            local_parquet_files, local_parquet_source_diagnostics, native_source_output_schema,
            required_parquet_column_requirements, validate_slice1_parquet_scan,
        },
        validate::{
            collect_required_columns, native_zero_column_count_source_is_safe, validate_aggregate,
            validate_physical_expr, validate_schema, validate_sort,
        },
    },
    policy::FallbackReason,
};

pub(super) fn lower_slice1_plan(
    normalized: &NormalizedNativePlan<'_>,
    root: &NormalizedNativeAggregateRoot<'_>,
) -> Result<NativeQueryPlan, FallbackReason> {
    let source = normalized.single_source_input().ok_or(FallbackReason::UnsupportedOperator)?;
    let source_exec = source.source_exec;
    let filter = source.filter;
    let pre_aggregate_projection = root.pre_aggregate_projection;
    let aggregate = root.aggregate;
    let post_aggregate_projection = root.post_aggregate_projection;
    let sort = normalized.sort;
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;
    let column_requirements =
        required_parquet_column_requirements(file_scan, filter, pre_aggregate_projection, aggregate)?;
    let zero_column_count_safe = native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate);
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let source_output_schema = if zero_column_count_safe {
        source_exec.schema()
    } else {
        native_source_output_schema(file_scan, &column_requirements)?
    };

    validate_schema(Arc::clone(&source_output_schema))?;
    validate_aggregate_root_operator_schemas(root, pre_aggregate_projection, post_aggregate_projection, sort)?;
    validate_required_scalar_expressions(RequiredScalarExpressions {
        scan_projection,
        table_schema: &table_schema,
        scan_output_schema: source_exec.schema(),
        required_scan_output_columns: &column_requirements.scan_output_columns,
        filter,
        pre_aggregate_projection,
        aggregate,
        post_aggregate_projection,
    })?;
    validate_slice1_parquet_scan(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    validate_aggregate(aggregate, root.aggregate_allows_partial)?;
    if let Some(sort) = sort {
        validate_sort(sort)?;
    }

    let native_files =
        local_parquet_files(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    let mut source = if zero_column_count_safe && column_requirements.read_columns.is_empty() {
        NativeLocalParquetSource::zero_column_count(native_files)
    } else {
        NativeLocalParquetSource::from_files(
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
    };
    source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        Arc::clone(&table_schema),
        Some(&column_requirements),
        zero_column_count_safe,
    );

    let mut builder = NativeQueryPlanBuilder::new();
    let source_node = builder.add_source(NativeSourceNode::LocalParquet(source));
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection
        && !zero_column_count_safe
    {
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
    let filter_input_schema = scan_projection.map_or_else(|| source_exec.schema(), |_| source_exec.schema());
    let (filter_node, filter_output_schema) = if let Some(filter) = filter {
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
    let (aggregate_input_node, aggregate_input_schema, synthetic_aggregate_args) = if zero_column_count_safe {
        (filter_node, filter_output_schema, None)
    } else if let Some(projection) = pre_aggregate_projection {
        let projection_node =
            builder.add_projection(filter_node, lower_projection_exprs(projection)?, projection.schema());
        if let Some(synthetic_projection) =
            synthetic_aggregate_input_projection_if_needed(aggregate, &projection.schema())?
        {
            let synthetic_projection_node = builder.add_projection(
                projection_node,
                synthetic_projection.expressions,
                Arc::clone(&synthetic_projection.schema),
            );
            (synthetic_projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
        } else {
            (projection_node, projection.schema(), None)
        }
    } else {
        let synthetic_projection = synthetic_aggregate_input_projection(aggregate, &filter_output_schema)?;
        let projection_node = builder.add_projection(
            filter_node,
            synthetic_projection.expressions,
            Arc::clone(&synthetic_projection.schema),
        );
        (projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
    };
    let aggregate_spec = if zero_column_count_safe {
        lower_zero_column_count_aggregate_spec(aggregate)?
    } else {
        lower_aggregate_spec(aggregate, &aggregate_input_schema, synthetic_aggregate_args.as_deref())?
    };
    let aggregate_node =
        builder.add_aggregate(aggregate_input_node, aggregate_spec, Arc::clone(&root.aggregate_output_schema));
    let sort_input_node = if let Some(projection) = post_aggregate_projection {
        builder.add_projection(aggregate_node, lower_projection_exprs(projection)?, projection.schema())
    } else {
        aggregate_node
    };
    let sorted_or_projected_node = if let Some(sort) = sort {
        builder.add_sort(sort_input_node, lower_sort_spec(sort)?, sort.schema())
    } else {
        sort_input_node
    };
    let output_node = if let Some(limit_fetch) = normalized.limit_fetch {
        let output_schema = normalized.sort.expect("native limit requires preceding sort").schema();
        builder.add_limit(sorted_or_projected_node, NativeLimitSpec::fetch(limit_fetch), output_schema)
    } else {
        sorted_or_projected_node
    };

    Ok(builder.finish_with_host_sink(output_node))
}

pub(super) fn lower_joined_aggregate_plan(
    normalized: &NormalizedNativePlan<'_>,
    root: &NormalizedNativeAggregateRoot<'_>,
) -> Result<NativeQueryPlan, FallbackReason> {
    let pre_aggregate_projection = root.pre_aggregate_projection;
    let aggregate = root.aggregate;
    let post_aggregate_projection = root.post_aggregate_projection;
    let sort = normalized.sort;

    validate_aggregate_root_operator_schemas(root, pre_aggregate_projection, post_aggregate_projection, sort)?;
    validate_aggregate(aggregate, root.aggregate_allows_partial)?;
    validate_joined_aggregate_scalar_expressions(root)?;
    if let Some(sort) = sort {
        validate_sort(sort)?;
    }

    let required_input_columns = required_aggregate_input_columns(aggregate, pre_aggregate_projection)?;
    let mut builder = NativeQueryPlanBuilder::new();
    let lowered_input = lower_native_input(&root.input, &mut builder, &required_input_columns)?;
    let (aggregate_input_node, aggregate_input_schema, synthetic_aggregate_args) = if let Some(projection) =
        pre_aggregate_projection
    {
        let projection_node =
            builder.add_projection(lowered_input.node, lower_projection_exprs(projection)?, projection.schema());
        if let Some(synthetic_projection) =
            synthetic_aggregate_input_projection_if_needed(aggregate, &projection.schema())?
        {
            let synthetic_projection_node = builder.add_projection(
                projection_node,
                synthetic_projection.expressions,
                Arc::clone(&synthetic_projection.schema),
            );
            (synthetic_projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
        } else {
            (projection_node, projection.schema(), None)
        }
    } else {
        let synthetic_projection = synthetic_aggregate_input_projection(aggregate, &lowered_input.schema)?;
        let projection_node = builder.add_projection(
            lowered_input.node,
            synthetic_projection.expressions,
            Arc::clone(&synthetic_projection.schema),
        );
        (projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
    };
    let aggregate_spec = lower_aggregate_spec(aggregate, &aggregate_input_schema, synthetic_aggregate_args.as_deref())?;
    let aggregate_node =
        builder.add_aggregate(aggregate_input_node, aggregate_spec, Arc::clone(&root.aggregate_output_schema));
    let sort_input_node = if let Some(projection) = post_aggregate_projection {
        builder.add_projection(aggregate_node, lower_projection_exprs(projection)?, projection.schema())
    } else {
        aggregate_node
    };
    let sorted_or_projected_node = if let Some(sort) = sort {
        builder.add_sort(sort_input_node, lower_sort_spec(sort)?, sort.schema())
    } else {
        sort_input_node
    };
    let output_node = if let Some(limit_fetch) = normalized.limit_fetch {
        let output_schema = normalized.sort.expect("native limit requires preceding sort").schema();
        builder.add_limit(sorted_or_projected_node, NativeLimitSpec::fetch(limit_fetch), output_schema)
    } else {
        sorted_or_projected_node
    };

    Ok(builder.finish_with_host_sink(output_node))
}

pub(super) fn lower_input_root_plan(
    normalized: &NormalizedNativePlan<'_>,
    root: &NormalizedNativeInputRoot<'_>,
) -> Result<NativeQueryPlan, FallbackReason> {
    if let Some(projection) = root.projection {
        validate_schema(projection.schema())?;
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    if let Some(sort) = normalized.sort {
        validate_schema(sort.schema())?;
        validate_sort(sort)?;
    }

    let mut builder = NativeQueryPlanBuilder::new();
    let required_input_columns =
        root.projection.map(required_root_projection_input_columns).transpose()?.unwrap_or_default();
    let lowered_input = lower_native_input(&root.input, &mut builder, &required_input_columns)?;
    let projected_node = if let Some(projection) = root.projection {
        builder.add_projection(lowered_input.node, lower_projection_exprs(projection)?, projection.schema())
    } else {
        lowered_input.node
    };
    let sorted_or_projected_node = if let Some(sort) = normalized.sort {
        builder.add_sort(projected_node, lower_sort_spec(sort)?, sort.schema())
    } else {
        projected_node
    };
    let output_node = if let Some(limit_fetch) = normalized.limit_fetch {
        let output_schema = normalized.sort.expect("native limit requires preceding sort").schema();
        builder.add_limit(sorted_or_projected_node, NativeLimitSpec::fetch(limit_fetch), output_schema)
    } else {
        sorted_or_projected_node
    };

    Ok(builder.finish_with_host_sink(output_node))
}

fn required_root_projection_input_columns(projection: &ProjectionExec) -> Result<Vec<String>, FallbackReason> {
    let mut required = BTreeSet::new();
    for projection_expr in projection.expr() {
        collect_required_columns(&projection_expr.expr, &projection.input().schema(), &mut required)?;
    }
    Ok(required.into_iter().collect())
}
