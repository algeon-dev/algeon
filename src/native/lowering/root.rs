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
    LoweredNativeInput, RequiredFields,
    aggregate::{
        RequiredScalarExpressions, lower_aggregate_spec, lower_zero_column_count_aggregate_spec,
        prepare_aggregate_input_with_full_synthetic_projection, prepare_aggregate_input_with_optional_synthetic_args,
        rename_aggregate_spec_outputs, required_aggregate_input_fields, validate_aggregate_root_operator_schemas,
        validate_joined_aggregate_scalar_expressions, validate_required_scalar_expressions,
    },
    executable_schema_and_bindings,
    expr::{collect_required_field_indices, lower_physical_expr_with_bindings},
    lower_native_input, projection_outputs,
    sort::lower_sort_spec_with_input,
    source::{
        add_source_filter_projection, lower_required_file_source_projection_exprs, required_scan_projection,
        required_scan_projection_requires_native_projection, required_scan_projection_schema,
        source_bindings_for_schema,
    },
    unary::add_projection_node_with_input,
};
use crate::{
    native::{
        normalize::{NormalizedNativeAggregateRoot, NormalizedNativeInputRoot, NormalizedNativePlan},
        source::{
            local_parquet_files, local_parquet_source_diagnostics, native_source_output_schema,
            required_parquet_column_requirements, validate_slice1_parquet_scan,
        },
        validate::{
            native_zero_column_count_source_is_safe, validate_aggregate, validate_physical_expr, validate_schema,
            validate_sort,
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
    source = source.with_scan_limit(file_scan.limit);
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
            (builder.add_projection(source_node, projection_outputs(expressions, &schema)?), schema)
        } else {
            (source_node, Arc::clone(&source_output_schema))
        }
    } else {
        (source_node, Arc::clone(&source_output_schema))
    };
    let scan_output_bindings = source_bindings_for_schema(&scan_output_schema, source_exec.schema());
    let scan_output = LoweredNativeInput::with_bindings(scan_output_node, scan_output_schema, scan_output_bindings);
    let required_filter_output_fields = if filter.and_then(|filter| filter.projection().as_ref()).is_some()
        && !zero_column_count_safe
        && pre_aggregate_projection.is_some()
    {
        required_aggregate_input_fields(aggregate, pre_aggregate_projection)?
    } else {
        RequiredFields::all()
    };
    let filtered = if let Some(filter) = filter {
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
                &mut builder,
                filtered_node,
                &scan_output,
                filter.schema(),
                projection,
                &required_filter_output_fields,
            )?
        } else {
            LoweredNativeInput::with_bindings(filtered_node, scan_output.schema, scan_output.bindings)
        }
    } else {
        scan_output
    };
    let prepared_aggregate_input = if zero_column_count_safe {
        super::aggregate::PreparedAggregateInput::new(filtered)
    } else if let Some(projection) = pre_aggregate_projection {
        let projected_input = add_projection_node_with_input(&mut builder, filtered, projection)?;
        prepare_aggregate_input_with_optional_synthetic_args(&mut builder, projected_input, aggregate)?
    } else {
        prepare_aggregate_input_with_full_synthetic_projection(&mut builder, filtered, aggregate)?
    };
    let aggregate_spec = if zero_column_count_safe {
        lower_zero_column_count_aggregate_spec(aggregate)?
    } else {
        lower_aggregate_spec(aggregate, &prepared_aggregate_input)?
    };
    let (aggregate_output_schema, aggregate_output_bindings) =
        executable_schema_and_bindings(&root.aggregate_output_schema, &RequiredFields::all())?;
    let aggregate_spec = rename_aggregate_spec_outputs(aggregate_spec, &aggregate_output_bindings)?;
    let aggregate_output = LoweredNativeInput::with_bindings(
        builder.add_aggregate(
            prepared_aggregate_input.input.node,
            aggregate_spec,
            Arc::clone(&aggregate_output_schema),
        ),
        aggregate_output_schema,
        aggregate_output_bindings,
    );
    let sort_input = if let Some(projection) = post_aggregate_projection {
        add_projection_node_with_input(&mut builder, aggregate_output, projection)?
    } else {
        aggregate_output
    };
    let sorted_or_projected = if let Some(sort) = sort {
        let node = builder.add_sort(
            sort_input.node,
            lower_sort_spec_with_input(sort, &sort_input)?,
            Arc::clone(&sort_input.schema),
        );
        LoweredNativeInput::with_bindings(node, sort_input.schema, sort_input.bindings)
    } else {
        sort_input
    };
    let output = if let Some(limit_fetch) = normalized.limit_fetch {
        let node = builder.add_limit(
            sorted_or_projected.node,
            NativeLimitSpec::fetch(limit_fetch),
            Arc::clone(&sorted_or_projected.schema),
        );
        LoweredNativeInput::with_bindings(node, sorted_or_projected.schema, sorted_or_projected.bindings)
    } else {
        sorted_or_projected
    };

    Ok(builder.finish_with_host_sink_schema(output.node, output.frontend_schema()?))
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

    let required_input_fields = required_aggregate_input_fields(aggregate, pre_aggregate_projection)?;
    let mut builder = NativeQueryPlanBuilder::new();
    let lowered_input = lower_native_input(&root.input, &mut builder, &required_input_fields)?;
    let prepared_aggregate_input = if let Some(projection) = pre_aggregate_projection {
        let projected_input = add_projection_node_with_input(&mut builder, lowered_input, projection)?;
        prepare_aggregate_input_with_optional_synthetic_args(&mut builder, projected_input, aggregate)?
    } else {
        prepare_aggregate_input_with_full_synthetic_projection(&mut builder, lowered_input, aggregate)?
    };
    let (aggregate_output_schema, aggregate_output_bindings) =
        executable_schema_and_bindings(&root.aggregate_output_schema, &RequiredFields::all())?;
    let aggregate_spec = rename_aggregate_spec_outputs(
        lower_aggregate_spec(aggregate, &prepared_aggregate_input)?,
        &aggregate_output_bindings,
    )?;
    let aggregate_output = LoweredNativeInput::with_bindings(
        builder.add_aggregate(
            prepared_aggregate_input.input.node,
            aggregate_spec,
            Arc::clone(&aggregate_output_schema),
        ),
        aggregate_output_schema,
        aggregate_output_bindings,
    );
    let sort_input = if let Some(projection) = post_aggregate_projection {
        add_projection_node_with_input(&mut builder, aggregate_output, projection)?
    } else {
        aggregate_output
    };
    let sorted_or_projected = if let Some(sort) = sort {
        let node = builder.add_sort(
            sort_input.node,
            lower_sort_spec_with_input(sort, &sort_input)?,
            Arc::clone(&sort_input.schema),
        );
        LoweredNativeInput::with_bindings(node, sort_input.schema, sort_input.bindings)
    } else {
        sort_input
    };
    let output = if let Some(limit_fetch) = normalized.limit_fetch {
        let node = builder.add_limit(
            sorted_or_projected.node,
            NativeLimitSpec::fetch(limit_fetch),
            Arc::clone(&sorted_or_projected.schema),
        );
        LoweredNativeInput::with_bindings(node, sorted_or_projected.schema, sorted_or_projected.bindings)
    } else {
        sorted_or_projected
    };

    Ok(builder.finish_with_host_sink_schema(output.node, output.frontend_schema()?))
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
    let required_input_fields =
        root.projection.map(required_root_projection_input_fields).transpose()?.unwrap_or_else(RequiredFields::all);
    let lowered_input = lower_native_input(&root.input, &mut builder, &required_input_fields)?;
    let projected = if let Some(projection) = root.projection {
        add_projection_node_with_input(&mut builder, lowered_input, projection)?
    } else {
        lowered_input
    };
    let sorted_or_projected = if let Some(sort) = normalized.sort {
        let node = builder.add_sort(
            projected.node,
            lower_sort_spec_with_input(sort, &projected)?,
            Arc::clone(&projected.schema),
        );
        LoweredNativeInput::with_bindings(node, projected.schema, projected.bindings)
    } else {
        projected
    };
    let output = if let Some(limit_fetch) = normalized.limit_fetch {
        let node = builder.add_limit(
            sorted_or_projected.node,
            NativeLimitSpec::fetch(limit_fetch),
            Arc::clone(&sorted_or_projected.schema),
        );
        LoweredNativeInput::with_bindings(node, sorted_or_projected.schema, sorted_or_projected.bindings)
    } else {
        sorted_or_projected
    };

    Ok(builder.finish_with_host_sink_schema(output.node, output.frontend_schema()?))
}

fn required_root_projection_input_fields(projection: &ProjectionExec) -> Result<RequiredFields, FallbackReason> {
    let mut required = BTreeSet::new();
    for projection_expr in projection.expr() {
        collect_required_field_indices(&projection_expr.expr, &projection.input().schema(), &mut required)?;
    }
    Ok(RequiredFields::only(required))
}
