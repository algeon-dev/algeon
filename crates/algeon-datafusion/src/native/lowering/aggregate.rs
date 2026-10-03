use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::{
    logical_expr::AggregateUDF,
    physical_expr::{expressions::Column, projection::ProjectionExprs},
    physical_plan::{ExecutionPlan, aggregates::AggregateExec, filter::FilterExec, projection::ProjectionExec},
};
use query_engine::{
    expr::{
        AggregateExpr as NativeAggregateExpr, AggregateFunction as NativeAggregateFunction,
        AggregateSpec as NativeAggregateSpec, Expr as NativeExpr, GroupingSetsSpec as NativeGroupingSetsSpec,
    },
    plan::{NodeId as NativeNodeId, QueryPlanBuilder as NativeQueryPlanBuilder},
    source::ZERO_COLUMN_COUNT_COLUMN as NATIVE_ZERO_COLUMN_COUNT_COLUMN,
};

use super::{
    FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, collect_required_field_indices, lower_physical_expr_with_bindings},
    lower_native_input, projection_outputs,
    source::required_scan_projection,
};
use crate::{
    capability::CapabilityReason,
    native::{
        LoweringFailure,
        expression_contract::{native_aggregate_function_kind, validate_physical_expr_contract},
        normalize::{NormalizedNativeAggregate, NormalizedNativeAggregateRoot, NormalizedNativeSortExprs},
        validate::{aggregate_is_count_star_only, validate_aggregate_for_native_lowering, validate_schema},
    },
};

mod synthetic;

pub(super) use synthetic::{
    PreparedAggregateInput, prepare_aggregate_input_with_full_synthetic_projection,
    prepare_aggregate_input_with_optional_synthetic_args,
};

pub(super) fn validate_joined_aggregate_scalar_expressions(
    root: &NormalizedNativeAggregateRoot<'_>,
) -> Result<(), LoweringFailure> {
    if let Some(projection) = root.pre_aggregate_projection {
        validate_projection_scalar_expressions(projection)?;
    }
    validate_aggregate_scalar_expressions(root.aggregate)?;
    if let Some(projection) = root.post_aggregate_projection {
        validate_projection_scalar_expressions(projection)?;
    }
    Ok(())
}

pub(super) fn required_aggregate_input_fields(
    aggregate: &AggregateExec,
    pre_aggregate_projection: Option<&ProjectionExec>,
) -> Result<RequiredFields, LoweringFailure> {
    let mut required = BTreeSet::new();
    let mut predicate_required = BTreeSet::new();
    if let Some(projection) = pre_aggregate_projection {
        for projection_expr in projection.expr() {
            collect_required_field_indices(&projection_expr.expr, &projection.input().schema(), &mut required)?;
        }
        let mut filter_projection_indices = BTreeSet::new();
        for filter in aggregate.filter_expr().iter().flatten() {
            collect_required_field_indices(filter, &aggregate.input().schema(), &mut filter_projection_indices)?;
        }
        for index in filter_projection_indices {
            let projection_expr = projection.expr().get(index).ok_or(CapabilityReason::UnsupportedExpression)?;
            collect_required_field_indices(
                &projection_expr.expr,
                &projection.input().schema(),
                &mut predicate_required,
            )?;
        }
    } else {
        for aggregate_expr in aggregate.aggr_expr() {
            for expr in aggregate_expr.expressions() {
                collect_required_field_indices(&expr, &aggregate.input().schema(), &mut required)?;
            }
        }
        for filter in aggregate.filter_expr().iter().flatten() {
            collect_required_field_indices(filter, &aggregate.input().schema(), &mut required)?;
            collect_required_field_indices(filter, &aggregate.input().schema(), &mut predicate_required)?;
        }
        for (group_expr, _alias) in aggregate.group_expr().expr() {
            collect_required_field_indices(group_expr, &aggregate.input().schema(), &mut required)?;
        }
    }
    Ok(RequiredFields::only_with_predicates(required, predicate_required))
}

pub(super) fn lower_aggregate_input(
    aggregate: &NormalizedNativeAggregate<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    validate_schema(Arc::clone(&aggregate.aggregate_output_schema))?;
    validate_aggregate_for_native_lowering(aggregate.aggregate, aggregate.aggregate_allows_partial)?;
    validate_aggregate_scalar_expressions(aggregate.aggregate)?;
    let child_required = required_aggregate_input_fields(aggregate.aggregate, None)?;
    let input = lower_native_input(&aggregate.input, builder, &child_required)?;
    let (aggregate_schema, aggregate_bindings) =
        executable_schema_and_bindings(&aggregate.aggregate_output_schema, &RequiredFields::all())?;
    let aggregate_node =
        add_native_aggregate_node(builder, input, aggregate.aggregate, &aggregate_schema, &aggregate_bindings)?;
    project_required_output_columns(
        builder,
        aggregate_node,
        aggregate_schema,
        aggregate_bindings,
        required_output_fields,
    )
}

pub(super) fn add_native_aggregate_node(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    aggregate: &AggregateExec,
    output_schema: &SchemaRef,
    output_bindings: &[FieldBinding],
) -> Result<NativeNodeId, LoweringFailure> {
    let prepared = prepare_aggregate_input_with_optional_synthetic_args(builder, input, aggregate)?;
    let aggregate_spec = rename_aggregate_spec_outputs(lower_aggregate_spec(aggregate, &prepared)?, output_bindings)?;
    Ok(builder.add_aggregate(prepared.input.node, aggregate_spec, Arc::clone(output_schema))?)
}

fn validate_aggregate_scalar_expressions(aggregate: &AggregateExec) -> Result<(), LoweringFailure> {
    for aggregate_expr in aggregate.aggr_expr() {
        for expr in aggregate_expr.expressions() {
            validate_physical_expr_contract(&expr, &aggregate.input().schema())?;
        }
    }
    for filter in aggregate.filter_expr().iter().flatten() {
        validate_physical_expr_contract(filter, &aggregate.input().schema())?;
    }
    for (group_expr, _alias) in aggregate.group_expr().expr() {
        validate_physical_expr_contract(group_expr, &aggregate.input().schema())?;
    }
    Ok(())
}

fn validate_projection_scalar_expressions(projection: &ProjectionExec) -> Result<(), LoweringFailure> {
    for projection_expr in projection.expr() {
        validate_physical_expr_contract(&projection_expr.expr, &projection.input().schema())?;
    }
    Ok(())
}

pub(super) fn project_required_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input_node: NativeNodeId,
    input_schema: SchemaRef,
    input_bindings: Vec<super::FieldBinding>,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    if required_output_fields.is_all() {
        return Ok(LoweredNativeInput::with_bindings(input_node, input_schema, input_bindings));
    }
    let mut expressions = Vec::new();
    let mut fields = Vec::new();
    let mut bindings = Vec::new();
    for binding in &input_bindings {
        if required_output_fields.contains(binding.frontend_index) {
            expressions.push(NativeExpr::column(binding.native_name.as_str()));
            let field = input_schema
                .field_with_name(binding.native_name.as_str())
                .map_err(|_error| CapabilityReason::UnsupportedExpression)?;
            fields.push(field.clone());
            bindings.push(binding.clone());
        }
    }
    if fields.len() == input_schema.fields().len() {
        return Ok(LoweredNativeInput::with_bindings(input_node, input_schema, input_bindings));
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input_node, projection_outputs(expressions, &schema)?)?;
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

pub(super) fn validate_aggregate_root_operator_schemas(
    root: &NormalizedNativeAggregateRoot<'_>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    post_aggregate_projection: Option<&ProjectionExec>,
    sort: Option<&NormalizedNativeSortExprs<'_>>,
) -> Result<(), LoweringFailure> {
    if let Some(projection) = pre_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    if !root.aggregate_allows_partial {
        validate_schema(root.aggregate.schema())?;
    }
    validate_schema(Arc::clone(&root.aggregate_output_schema))?;
    if let Some(projection) = post_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    if let Some(sort) = sort {
        validate_schema(Arc::clone(sort.output_schema()))?;
    }
    Ok(())
}

// Aggregate admission inputs whose scalar expressions must be validated together.
pub(super) struct RequiredScalarExpressions<'a> {
    pub(super) scan_projection: Option<&'a ProjectionExprs>,
    pub(super) table_schema: &'a SchemaRef,
    pub(super) required_scan_output_columns: &'a [String],
    pub(super) filter: Option<&'a FilterExec>,
    pub(super) pre_aggregate_projection: Option<&'a ProjectionExec>,
    pub(super) aggregate: &'a AggregateExec,
    pub(super) post_aggregate_projection: Option<&'a ProjectionExec>,
}

pub(super) fn validate_required_scalar_expressions(
    inputs: RequiredScalarExpressions<'_>,
) -> Result<(), LoweringFailure> {
    if let Some(scan_projection) = inputs.scan_projection {
        for projection_expr in required_scan_projection(scan_projection, inputs.required_scan_output_columns) {
            validate_physical_expr_contract(projection_expr.expr, inputs.table_schema)?;
        }
    }
    if let Some(filter) = inputs.filter {
        validate_physical_expr_contract(filter.predicate(), &filter.input().schema())?;
    }
    if let Some(projection) = inputs.pre_aggregate_projection {
        validate_projection_scalar_expressions(projection)?;
    }
    validate_aggregate_scalar_expressions(inputs.aggregate)?;
    if let Some(projection) = inputs.post_aggregate_projection {
        validate_projection_scalar_expressions(projection)?;
    }
    Ok(())
}

pub(super) fn lower_aggregate_spec(
    aggregate: &AggregateExec,
    prepared: &PreparedAggregateInput,
) -> Result<NativeAggregateSpec, LoweringFailure> {
    let frontend_schema = aggregate.input().schema();
    let group_exprs = aggregate
        .group_expr()
        .expr()
        .iter()
        .enumerate()
        .map(|(index, (expr, alias))| {
            let native_expr = prepared.group_expr_aliases.as_ref().and_then(|aliases| aliases.get(index)).map_or_else(
                || {
                    lower_physical_expr_with_bindings(expr, &frontend_schema, Some(&prepared.input.bindings))
                        .map_err(LoweringFailure::from)
                },
                |alias| Ok::<NativeExpr, LoweringFailure>(NativeExpr::column(alias)),
            )?;
            Ok::<NativeExpr, LoweringFailure>(alias_if_needed(native_expr, alias))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let aggr_exprs = aggregate
        .aggr_expr()
        .iter()
        .enumerate()
        .map(|(index, expr)| {
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Err(LoweringFailure::from(CapabilityReason::UnsupportedExpression));
            };
            let lowered_arg =
                prepared.aggregate_arg_aliases.as_deref().and_then(|aliases| aliases.get(index)).map_or_else(
                    || lower_physical_expr_with_bindings(arg, &frontend_schema, Some(&prepared.input.bindings)),
                    |alias| Ok(NativeExpr::column(alias)),
                )?;
            let lowered_filter =
                aggregate.filter_expr().get(index).and_then(Option::as_ref).map_or(Ok(None), |filter| {
                    prepared
                        .filter_aliases
                        .as_ref()
                        .and_then(|aliases| aliases.get(index))
                        .and_then(Option::as_ref)
                        .map_or_else(
                            || {
                                lower_physical_expr_with_bindings(
                                    filter,
                                    &frontend_schema,
                                    Some(&prepared.input.bindings),
                                )
                            },
                            |alias| Ok(NativeExpr::column(alias)),
                        )
                        .map(Some)
                })?;
            let (function, distinct) = lower_native_aggregate_expr_function(expr.fun(), expr.is_distinct())?;
            Ok(NativeAggregateExpr::new(function, lowered_arg, expr.name())
                .with_distinct(distinct)
                .with_filter(lowered_filter))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut spec = NativeAggregateSpec::new(group_exprs, aggr_exprs);
    if aggregate.group_expr().has_grouping_set() {
        spec = spec
            .with_grouping_sets(NativeGroupingSetsSpec::new(aggregate.group_expr().groups().to_vec(), "__grouping_id"));
    }
    Ok(spec)
}

pub(super) fn rename_aggregate_spec_outputs(
    mut spec: NativeAggregateSpec,
    output_bindings: &[FieldBinding],
) -> Result<NativeAggregateSpec, LoweringFailure> {
    let grouping_id_offset = spec.group_exprs.len();
    let grouping_id_count = usize::from(spec.grouping_sets.is_some());
    let expected_outputs = spec.group_exprs.len() + grouping_id_count + spec.aggr_exprs.len();
    if output_bindings.len() != expected_outputs {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    for (expr, binding) in spec.group_exprs.iter_mut().zip(output_bindings.iter()) {
        if binding.native_name != binding.frontend_name {
            *expr = with_output_alias(expr.clone(), binding.native_name.as_str());
        }
    }
    if let Some(grouping_sets) = &mut spec.grouping_sets {
        let binding = &output_bindings[grouping_id_offset];
        grouping_sets.grouping_id_alias = binding.native_name.clone();
    }
    for (expr, binding) in
        spec.aggr_exprs.iter_mut().zip(output_bindings.iter().skip(spec.group_exprs.len() + grouping_id_count))
    {
        expr.alias = binding.native_name.clone();
    }
    Ok(spec)
}

fn with_output_alias(expr: NativeExpr, alias: &str) -> NativeExpr {
    match expr {
        NativeExpr::Alias { expr, .. } => (*expr).alias(alias),
        expr => expr.alias(alias),
    }
}

pub(super) fn lower_zero_column_count_aggregate_spec(
    aggregate: &AggregateExec,
) -> Result<NativeAggregateSpec, LoweringFailure> {
    if !aggregate_is_count_star_only(aggregate) {
        return Err(CapabilityReason::UnsupportedOperator.into());
    }
    let expr = &aggregate.aggr_expr()[0];
    Ok(NativeAggregateSpec::new(
        Vec::new(),
        vec![NativeAggregateExpr::new(
            NativeAggregateFunction::Sum,
            NativeExpr::column(NATIVE_ZERO_COLUMN_COUNT_COLUMN),
            expr.name(),
        )],
    ))
}

pub(super) fn lower_native_aggregate_function(fun: &AggregateUDF) -> Result<NativeAggregateFunction, LoweringFailure> {
    native_aggregate_function_kind(fun).ok_or_else(|| CapabilityReason::UnsupportedOperator.into())
}

fn lower_native_aggregate_expr_function(
    fun: &AggregateUDF,
    distinct: bool,
) -> Result<(NativeAggregateFunction, bool), LoweringFailure> {
    let function = lower_native_aggregate_function(fun)?;
    if distinct && function == NativeAggregateFunction::Count {
        Ok((NativeAggregateFunction::NUnique, false))
    } else {
        Ok((function, distinct))
    }
}
