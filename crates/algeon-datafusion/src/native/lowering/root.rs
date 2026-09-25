use std::{collections::BTreeSet, sync::Arc};

use datafusion::{
    datasource::physical_plan::{FileSource, ParquetSource},
    physical_plan::{ExecutionPlan, projection::ProjectionExec},
};
use query_engine::{
    expr::LimitSpec as NativeLimitSpec,
    plan::{QueryPlan as NativeQueryPlan, QueryPlanBuilder as NativeQueryPlanBuilder},
    source::{LocalParquetSource as NativeLocalParquetSource, SourceNode as NativeSourceNode},
};

use super::{
    LoweredNativeInput, RequiredFields,
    aggregate::{
        RequiredScalarExpressions, add_native_aggregate_node, lower_aggregate_spec,
        lower_zero_column_count_aggregate_spec, prepare_aggregate_input_with_full_synthetic_projection,
        prepare_aggregate_input_with_optional_synthetic_args, rename_aggregate_spec_outputs,
        required_aggregate_input_fields, validate_aggregate_root_operator_schemas,
        validate_joined_aggregate_scalar_expressions, validate_required_scalar_expressions,
    },
    executable_schema_and_bindings,
    expr::{collect_required_field_indices, lower_physical_expr_with_bindings},
    lower_native_input, lower_native_input_with_scan_limit, projection_outputs,
    sort::lower_sort_ordering_with_input,
    source::{
        SourceUnprojectedFilterOutput, available_table_format_scan_plan, lower_empty_table_format_count_star_source,
        lower_required_file_source_projection_exprs, lower_source_filter_and_projection,
        lower_table_format_count_star_fallback_source, lower_table_format_count_star_mixed_row_count_source,
        lower_table_format_zero_column_count_source, preserve_resolved_source_diagnostics,
        reader_predicate_proves_local_parquet_at_most_one_row, required_scan_projection,
        required_scan_projection_requires_native_projection, required_scan_projection_schema,
        source_bindings_for_schema, source_reader_pushdown_predicate,
        table_format_count_star_mixed_row_count_metadata_can_split, table_format_count_star_probe_can_lower,
        table_format_scan_rejection, table_format_zero_column_count_metadata_is_available,
    },
    source_scan_limit_from_limit_spec,
    unary::add_projection_node_with_input,
};
use crate::{
    capability::CapabilityReason,
    native::{
        expression_contract::validate_physical_expr_contract,
        normalize::{
            NormalizedNativeAggregateRoot, NormalizedNativeInputRoot, NormalizedNativePlan, NormalizedNativeSortExprs,
            PostAggregateProjectionOrder,
        },
        pruning::scan_predicate_without_dynamic_filter_hints,
        source::{
            local_parquet_files, local_parquet_scan_is_empty, local_parquet_source_diagnostics,
            native_source_output_schema, required_parquet_column_requirements, validate_slice1_parquet_scan,
        },
        validate::{
            native_zero_column_count_source_is_safe, validate_aggregate_for_native_lowering, validate_schema,
            validate_sort_ordering,
        },
    },
};

pub(super) fn lower_slice1_plan(
    normalized: &NormalizedNativePlan<'_>,
    root: &NormalizedNativeAggregateRoot<'_>,
) -> Result<NativeQueryPlan, CapabilityReason> {
    let source = normalized.single_source_input().ok_or(CapabilityReason::UnsupportedOperator)?;
    let source_exec = source.local_parquet_source_exec().ok_or_else(|| table_format_scan_rejection(source))?;
    let filter = source.filter;
    let pre_aggregate_projection = root.pre_aggregate_projection;
    let aggregate = root.aggregate;
    let post_aggregate_projection = root.post_aggregate_projection;
    let sort = normalized.sort.as_ref();
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(CapabilityReason::UnsupportedOperator)?;
    let column_requirements =
        required_parquet_column_requirements(file_scan, parquet_source, filter, pre_aggregate_projection, aggregate)?;
    let zero_column_count_safe = native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate)
        && column_requirements.read_columns.is_empty();
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let source_output_schema = if zero_column_count_safe {
        source_exec.schema()
    } else {
        native_source_output_schema(file_scan, &column_requirements)?
    };
    let scan_projection_requires_native = scan_projection.is_some_and(|scan_projection| {
        let required_scan_projection =
            required_scan_projection(scan_projection, &column_requirements.scan_output_columns);
        required_scan_projection_requires_native_projection(&required_scan_projection, &table_schema)
    });

    validate_schema(Arc::clone(&source_output_schema))?;
    validate_aggregate_root_operator_schemas(root, pre_aggregate_projection, post_aggregate_projection, sort)?;
    validate_required_scalar_expressions(RequiredScalarExpressions {
        scan_projection,
        table_schema: &table_schema,
        required_scan_output_columns: &column_requirements.scan_output_columns,
        filter,
        pre_aggregate_projection,
        aggregate,
        post_aggregate_projection,
    })?;
    if local_parquet_scan_is_empty(file_scan) {
        return lower_joined_aggregate_plan(normalized, root);
    }
    validate_slice1_parquet_scan(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    validate_aggregate_for_native_lowering(aggregate, root.aggregate_allows_partial)?;
    if let Some(sort) = sort {
        validate_sort_ordering(sort.ordering(), sort.input_schema())?;
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
        .with_missing_columns(column_requirements.missing_columns.clone())
    };
    source = source.with_scan_limit(file_scan.limit);
    let mut reader_pushdown_covers_filter = false;
    if !zero_column_count_safe
        && let Some(predicate) = source_reader_pushdown_predicate(
            parquet_source,
            filter,
            file_scan.limit,
            scan_projection_requires_native,
            &source_output_schema,
            if filter.is_none() { Arc::clone(&table_schema) } else { source_exec.schema() },
            &source,
        )?
    {
        reader_pushdown_covers_filter = predicate.covers_filter;
        let max_rows = if reader_predicate_proves_local_parquet_at_most_one_row(
            table_schema.as_ref(),
            file_scan.file_schema().as_ref(),
            &predicate.expr,
        ) {
            Some(1)
        } else {
            None
        };
        source = source.with_reader_predicate(predicate.expr).with_max_rows(max_rows);
    }
    let reader_pushdown = source.reader_predicate.is_some();
    let source_contract_diagnostics = source.diagnostics.clone();
    let selected_row_group_stats = source.diagnostics.selected_row_group_stats.clone();
    source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        Arc::clone(&table_schema),
        Some(&column_requirements),
        reader_pushdown,
        zero_column_count_safe,
    );
    preserve_resolved_source_diagnostics(&mut source.diagnostics, &source_contract_diagnostics);
    source.diagnostics.selected_row_group_stats = selected_row_group_stats;

    let mut builder = NativeQueryPlanBuilder::new();
    let source_node = builder.add_source(NativeSourceNode::LocalParquet(source));
    let scan_only_post_scan_predicate = if filter.is_none() && !reader_pushdown_covers_filter {
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
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection
        && !zero_column_count_safe
    {
        let required_scan_projection =
            required_scan_projection(scan_projection, &column_requirements.scan_output_columns);
        if required_scan_projection_requires_native_projection(&required_scan_projection, &table_schema) {
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
    let required_filter_output_fields = if filter.and_then(|filter| filter.projection().as_ref()).is_some()
        && !zero_column_count_safe
        && pre_aggregate_projection.is_some()
    {
        required_aggregate_input_fields(aggregate, pre_aggregate_projection)?
    } else {
        RequiredFields::all()
    };
    let filtered = if let Some(filter) = filter {
        lower_source_filter_and_projection(
            &mut builder,
            scan_output,
            filter,
            reader_pushdown_covers_filter,
            &required_filter_output_fields,
            SourceUnprojectedFilterOutput::PreserveScanOutput,
        )?
    } else {
        scan_output
    };
    let (aggregate_output_schema, aggregate_output_bindings) =
        executable_schema_and_bindings(&root.aggregate_output_schema, &RequiredFields::all())?;
    let aggregate_output = if aggregate.group_expr().has_grouping_set() {
        let aggregate_input = if let Some(projection) = pre_aggregate_projection {
            add_projection_node_with_input(&mut builder, filtered, projection)?
        } else {
            filtered
        };
        LoweredNativeInput::with_bindings(
            add_native_aggregate_node(
                &mut builder,
                aggregate_input,
                aggregate,
                &aggregate_output_schema,
                &aggregate_output_bindings,
            )?,
            aggregate_output_schema,
            aggregate_output_bindings,
        )
    } else {
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
        let aggregate_spec = rename_aggregate_spec_outputs(aggregate_spec, &aggregate_output_bindings)?;
        LoweredNativeInput::with_bindings(
            builder.add_aggregate(
                prepared_aggregate_input.input.node,
                aggregate_spec,
                Arc::clone(&aggregate_output_schema),
            ),
            aggregate_output_schema,
            aggregate_output_bindings,
        )
    };
    super::advisory::attach(&mut builder, aggregate_output.node, &aggregate_output.schema, root.statistics_output);
    finish_with_post_projection_sort_limit(
        builder,
        aggregate_output,
        post_aggregate_projection,
        sort,
        normalized.limit,
        root.projection_order,
    )
}

pub(super) fn lower_table_format_zero_column_count_plan(
    normalized: &NormalizedNativePlan<'_>,
    root: &NormalizedNativeAggregateRoot<'_>,
) -> Result<NativeQueryPlan, CapabilityReason> {
    let source = normalized.single_source_input().ok_or(CapabilityReason::UnsupportedOperator)?;
    let scan_exec = source.table_format_scan_exec().ok_or_else(|| table_format_scan_rejection(source))?;
    let scan_plan = available_table_format_scan_plan(scan_exec)?;
    let aggregate = root.aggregate;
    let post_aggregate_projection = root.post_aggregate_projection;
    let sort = normalized.sort.as_ref();

    validate_aggregate_root_operator_schemas(root, None, post_aggregate_projection, sort)?;
    validate_aggregate_for_native_lowering(aggregate, root.aggregate_allows_partial)?;
    if let Some(sort) = sort {
        validate_sort_ordering(sort.ordering(), sort.input_schema())?;
    }

    let mut builder = NativeQueryPlanBuilder::new();
    let count_weight_input = table_format_zero_column_count_metadata_is_available(scan_plan)
        || table_format_count_star_mixed_row_count_metadata_can_split(scan_plan)
        || table_format_count_star_probe_can_lower(scan_plan);
    let source_input = if scan_plan.data_files.is_empty() {
        lower_empty_table_format_count_star_source(scan_exec, &mut builder)?
    } else if table_format_zero_column_count_metadata_is_available(scan_plan) {
        lower_table_format_zero_column_count_source(scan_exec, &mut builder)?
    } else if table_format_count_star_mixed_row_count_metadata_can_split(scan_plan) {
        lower_table_format_count_star_mixed_row_count_source(scan_exec, &mut builder)?
    } else {
        lower_table_format_count_star_fallback_source(scan_exec, &mut builder)?
    };
    let (aggregate_output_schema, aggregate_output_bindings) =
        executable_schema_and_bindings(&root.aggregate_output_schema, &RequiredFields::all())?;
    let aggregate_spec = if count_weight_input {
        rename_aggregate_spec_outputs(lower_zero_column_count_aggregate_spec(aggregate)?, &aggregate_output_bindings)?
    } else {
        let prepared_aggregate_input =
            prepare_aggregate_input_with_full_synthetic_projection(&mut builder, source_input, aggregate)?;
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
        super::advisory::attach(&mut builder, aggregate_output.node, &aggregate_output.schema, root.statistics_output);
        return finish_with_post_projection_sort_limit(
            builder,
            aggregate_output,
            post_aggregate_projection,
            sort,
            normalized.limit,
            root.projection_order,
        );
    };
    let aggregate_output = LoweredNativeInput::with_bindings(
        builder.add_aggregate(source_input.node, aggregate_spec, Arc::clone(&aggregate_output_schema)),
        aggregate_output_schema,
        aggregate_output_bindings,
    );
    super::advisory::attach(&mut builder, aggregate_output.node, &aggregate_output.schema, root.statistics_output);
    finish_with_post_projection_sort_limit(
        builder,
        aggregate_output,
        post_aggregate_projection,
        sort,
        normalized.limit,
        root.projection_order,
    )
}

pub(super) fn lower_joined_aggregate_plan(
    normalized: &NormalizedNativePlan<'_>,
    root: &NormalizedNativeAggregateRoot<'_>,
) -> Result<NativeQueryPlan, CapabilityReason> {
    let pre_aggregate_projection = root.pre_aggregate_projection;
    let aggregate = root.aggregate;
    let post_aggregate_projection = root.post_aggregate_projection;
    let sort = normalized.sort.as_ref();

    validate_aggregate_root_operator_schemas(root, pre_aggregate_projection, post_aggregate_projection, sort)?;
    validate_aggregate_for_native_lowering(aggregate, root.aggregate_allows_partial)?;
    validate_joined_aggregate_scalar_expressions(root)?;
    if let Some(sort) = sort {
        validate_sort_ordering(sort.ordering(), sort.input_schema())?;
    }

    let required_input_fields = required_aggregate_input_fields(aggregate, pre_aggregate_projection)?;
    let mut builder = NativeQueryPlanBuilder::new();
    let lowered_input = lower_native_input(&root.input, &mut builder, &required_input_fields)?;
    let (aggregate_output_schema, aggregate_output_bindings) =
        executable_schema_and_bindings(&root.aggregate_output_schema, &RequiredFields::all())?;
    let aggregate_output = if aggregate.group_expr().has_grouping_set() {
        let aggregate_input = if let Some(projection) = pre_aggregate_projection {
            add_projection_node_with_input(&mut builder, lowered_input, projection)?
        } else {
            lowered_input
        };
        LoweredNativeInput::with_bindings(
            add_native_aggregate_node(
                &mut builder,
                aggregate_input,
                aggregate,
                &aggregate_output_schema,
                &aggregate_output_bindings,
            )?,
            aggregate_output_schema,
            aggregate_output_bindings,
        )
    } else {
        let prepared_aggregate_input = if let Some(projection) = pre_aggregate_projection {
            let projected_input = add_projection_node_with_input(&mut builder, lowered_input, projection)?;
            prepare_aggregate_input_with_optional_synthetic_args(&mut builder, projected_input, aggregate)?
        } else {
            prepare_aggregate_input_with_full_synthetic_projection(&mut builder, lowered_input, aggregate)?
        };
        let aggregate_spec = rename_aggregate_spec_outputs(
            lower_aggregate_spec(aggregate, &prepared_aggregate_input)?,
            &aggregate_output_bindings,
        )?;
        LoweredNativeInput::with_bindings(
            builder.add_aggregate(
                prepared_aggregate_input.input.node,
                aggregate_spec,
                Arc::clone(&aggregate_output_schema),
            ),
            aggregate_output_schema,
            aggregate_output_bindings,
        )
    };
    super::advisory::attach(&mut builder, aggregate_output.node, &aggregate_output.schema, root.statistics_output);
    finish_with_post_projection_sort_limit(
        builder,
        aggregate_output,
        post_aggregate_projection,
        sort,
        normalized.limit,
        root.projection_order,
    )
}

pub(super) fn lower_input_root_plan(
    normalized: &NormalizedNativePlan<'_>,
    root: &NormalizedNativeInputRoot<'_>,
) -> Result<NativeQueryPlan, CapabilityReason> {
    if let Some(projection) = root.projection {
        validate_schema(projection.schema())?;
        for projection_expr in projection.expr() {
            validate_physical_expr_contract(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    if let Some(sort) = normalized.sort.as_ref() {
        validate_schema(Arc::clone(sort.output_schema()))?;
        validate_sort_ordering(sort.ordering(), sort.input_schema())?;
    }

    let mut builder = NativeQueryPlanBuilder::new();
    let required_input_fields = if root.projection_order == PostAggregateProjectionOrder::AboveSort {
        RequiredFields::all()
    } else {
        root.projection.map(required_root_projection_input_fields).transpose()?.unwrap_or_else(RequiredFields::all)
    };
    let pushed_scan_limit =
        if normalized.sort.is_none() { source_scan_limit_from_limit_spec(normalized.limit) } else { None };
    let lowered_input =
        lower_native_input_with_scan_limit(&root.input, &mut builder, &required_input_fields, pushed_scan_limit)?;
    finish_with_post_projection_sort_limit(
        builder,
        lowered_input,
        root.projection,
        normalized.sort.as_ref(),
        normalized.limit,
        root.projection_order,
    )
}

fn finish_with_post_projection_sort_limit(
    mut builder: NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    projection: Option<&ProjectionExec>,
    sort: Option<&NormalizedNativeSortExprs<'_>>,
    limit: Option<NativeLimitSpec>,
    projection_order: PostAggregateProjectionOrder,
) -> Result<NativeQueryPlan, CapabilityReason> {
    let output = apply_post_projection_sort_limit(&mut builder, input, projection, sort, limit, projection_order)?;
    Ok(builder.finish_with_host_sink_schema(output.node, output.frontend_schema()?))
}

fn apply_post_projection_sort_limit(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    projection: Option<&ProjectionExec>,
    sort: Option<&NormalizedNativeSortExprs<'_>>,
    limit: Option<NativeLimitSpec>,
    projection_order: PostAggregateProjectionOrder,
) -> Result<LoweredNativeInput, CapabilityReason> {
    if projection_order == PostAggregateProjectionOrder::AboveSort {
        let sorted = if let Some(sort) = sort {
            lower_sort_ordering_with_input(
                builder,
                sort.ordering(),
                sort.input_schema(),
                sort.output_schema(),
                input,
                &RequiredFields::all(),
            )?
        } else {
            input
        };
        let projected = if let Some(projection) = projection {
            add_projection_node_with_input(builder, sorted, projection)?
        } else {
            sorted
        };
        if let Some(limit) = limit {
            let node = builder.add_limit(projected.node, limit, Arc::clone(&projected.schema));
            Ok(LoweredNativeInput::with_bindings(node, projected.schema, projected.bindings))
        } else {
            Ok(projected)
        }
    } else {
        let projected = if let Some(projection) = projection {
            add_projection_node_with_input(builder, input, projection)?
        } else {
            input
        };
        let sorted_or_projected = if let Some(sort) = sort {
            lower_sort_ordering_with_input(
                builder,
                sort.ordering(),
                sort.input_schema(),
                sort.output_schema(),
                projected,
                &RequiredFields::all(),
            )?
        } else {
            projected
        };
        if let Some(limit) = limit {
            let node = builder.add_limit(sorted_or_projected.node, limit, Arc::clone(&sorted_or_projected.schema));
            Ok(LoweredNativeInput::with_bindings(node, sorted_or_projected.schema, sorted_or_projected.bindings))
        } else {
            Ok(sorted_or_projected)
        }
    }
}

fn required_root_projection_input_fields(projection: &ProjectionExec) -> Result<RequiredFields, CapabilityReason> {
    let mut required = BTreeSet::new();
    for projection_expr in projection.expr() {
        collect_required_field_indices(&projection_expr.expr, &projection.input().schema(), &mut required)?;
    }
    Ok(RequiredFields::only(required))
}
