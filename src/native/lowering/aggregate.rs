use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};
use cudf_query_engine::{
    AggregateExpr as NativeAggregateExpr, AggregateFunction as NativeAggregateFunction,
    AggregateSpec as NativeAggregateSpec, Expr as NativeExpr, NodeId as NativeNodeId,
    QueryPlanBuilder as NativeQueryPlanBuilder, ZERO_COLUMN_COUNT_COLUMN as NATIVE_ZERO_COLUMN_COUNT_COLUMN,
};
use datafusion::{
    physical_expr::{expressions::Column, projection::ProjectionExprs},
    physical_plan::{ExecutionPlan, aggregates::AggregateExec, projection::ProjectionExec, sorts::sort::SortExec},
};

use super::{
    LoweredNativeInput,
    expr::{alias_if_needed, lower_physical_expr, physical_expr_field},
    lower_native_input,
    source::required_scan_projection,
};
use crate::{
    native::{
        normalize::{NormalizedNativeAggregate, NormalizedNativeAggregateRoot},
        validate::{
            aggregate_is_count_star_only, collect_required_columns, validate_aggregate, validate_physical_expr,
            validate_schema,
        },
    },
    policy::FallbackReason,
};

pub(super) fn validate_joined_aggregate_scalar_expressions(
    root: &NormalizedNativeAggregateRoot<'_>,
) -> Result<(), FallbackReason> {
    if let Some(projection) = root.pre_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    for aggregate_expr in root.aggregate.aggr_expr() {
        for expr in aggregate_expr.expressions() {
            validate_physical_expr(&expr, &root.aggregate.input().schema())?;
        }
    }
    for (group_expr, _alias) in root.aggregate.group_expr().expr() {
        validate_physical_expr(group_expr, &root.aggregate.input().schema())?;
    }
    if let Some(projection) = root.post_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    Ok(())
}

pub(super) fn required_aggregate_input_columns(
    aggregate: &AggregateExec,
    pre_aggregate_projection: Option<&ProjectionExec>,
) -> Result<Vec<String>, FallbackReason> {
    let mut required = BTreeSet::new();
    if let Some(projection) = pre_aggregate_projection {
        for projection_expr in projection.expr() {
            collect_required_columns(&projection_expr.expr, &projection.input().schema(), &mut required)?;
        }
    } else {
        for aggregate_expr in aggregate.aggr_expr() {
            for expr in aggregate_expr.expressions() {
                collect_required_columns(&expr, &aggregate.input().schema(), &mut required)?;
            }
        }
        for (group_expr, _alias) in aggregate.group_expr().expr() {
            collect_required_columns(group_expr, &aggregate.input().schema(), &mut required)?;
        }
    }
    Ok(required.into_iter().collect())
}

pub(super) fn lower_aggregate_input(
    aggregate: &NormalizedNativeAggregate<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    validate_schema(Arc::clone(&aggregate.aggregate_output_schema))?;
    validate_aggregate(aggregate.aggregate, aggregate.aggregate_allows_partial)?;
    validate_aggregate_scalar_expressions(aggregate.aggregate)?;
    let child_required = required_aggregate_input_columns(aggregate.aggregate, None)?;
    let input = lower_native_input(&aggregate.input, builder, &child_required)?;
    let aggregate_node = add_native_aggregate_node(
        builder,
        input.node,
        &input.schema,
        aggregate.aggregate,
        &aggregate.aggregate_output_schema,
    )?;
    project_required_output_columns(
        builder,
        aggregate_node,
        Arc::clone(&aggregate.aggregate_output_schema),
        required_output_columns,
    )
}

fn add_native_aggregate_node(
    builder: &mut NativeQueryPlanBuilder,
    input_node: NativeNodeId,
    input_schema: &SchemaRef,
    aggregate: &AggregateExec,
    output_schema: &SchemaRef,
) -> Result<NativeNodeId, FallbackReason> {
    let (aggregate_input_node, aggregate_input_schema, synthetic_aggregate_args) =
        if let Some(synthetic_projection) = synthetic_aggregate_input_projection_if_needed(aggregate, input_schema)? {
            let synthetic_projection_node = builder.add_projection(
                input_node,
                synthetic_projection.expressions,
                Arc::clone(&synthetic_projection.schema),
            );
            (synthetic_projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
        } else {
            (input_node, Arc::clone(input_schema), None)
        };
    let aggregate_spec = lower_aggregate_spec(aggregate, &aggregate_input_schema, synthetic_aggregate_args.as_deref())?;
    Ok(builder.add_aggregate(aggregate_input_node, aggregate_spec, Arc::clone(output_schema)))
}

fn validate_aggregate_scalar_expressions(aggregate: &AggregateExec) -> Result<(), FallbackReason> {
    for aggregate_expr in aggregate.aggr_expr() {
        for expr in aggregate_expr.expressions() {
            validate_physical_expr(&expr, &aggregate.input().schema())?;
        }
    }
    for (group_expr, _alias) in aggregate.group_expr().expr() {
        validate_physical_expr(group_expr, &aggregate.input().schema())?;
    }
    Ok(())
}

fn project_required_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input_node: NativeNodeId,
    input_schema: SchemaRef,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    if required_output_columns.is_empty() {
        return Ok(LoweredNativeInput { node: input_node, schema: input_schema });
    }
    let required = required_output_columns.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut expressions = Vec::new();
    let mut fields = Vec::new();
    for field in input_schema.fields() {
        if required.contains(field.name().as_str()) {
            expressions.push(NativeExpr::column(field.name()));
            fields.push(field.as_ref().clone());
        }
    }
    if fields.len() != required.len() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if fields.len() == input_schema.fields().len() {
        return Ok(LoweredNativeInput { node: input_node, schema: input_schema });
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input_node, expressions, Arc::clone(&schema));
    Ok(LoweredNativeInput { node, schema })
}

pub(super) fn validate_aggregate_root_operator_schemas(
    root: &NormalizedNativeAggregateRoot<'_>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    post_aggregate_projection: Option<&ProjectionExec>,
    sort: Option<&SortExec>,
) -> Result<(), FallbackReason> {
    if let Some(projection) = pre_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    validate_schema(root.aggregate.schema())?;
    validate_schema(Arc::clone(&root.aggregate_output_schema))?;
    if let Some(projection) = post_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    if let Some(sort) = sort {
        validate_schema(sort.schema())?;
    }
    Ok(())
}

pub(super) struct RequiredScalarExpressions<'a> {
    pub(super) scan_projection: Option<&'a ProjectionExprs>,
    pub(super) table_schema: &'a SchemaRef,
    pub(super) scan_output_schema: SchemaRef,
    pub(super) required_scan_output_columns: &'a [String],
    pub(super) filter: Option<&'a datafusion::physical_plan::filter::FilterExec>,
    pub(super) pre_aggregate_projection: Option<&'a ProjectionExec>,
    pub(super) aggregate: &'a AggregateExec,
    pub(super) post_aggregate_projection: Option<&'a ProjectionExec>,
}

pub(super) fn validate_required_scalar_expressions(
    inputs: RequiredScalarExpressions<'_>,
) -> Result<(), FallbackReason> {
    if let Some(scan_projection) = inputs.scan_projection {
        for projection_expr in
            required_scan_projection(scan_projection, inputs.scan_output_schema, inputs.required_scan_output_columns)
        {
            validate_physical_expr(projection_expr.expr, inputs.table_schema)?;
        }
    }
    if let Some(filter) = inputs.filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }
    if let Some(projection) = inputs.pre_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    for aggregate_expr in inputs.aggregate.aggr_expr() {
        for expr in aggregate_expr.expressions() {
            validate_physical_expr(&expr, &inputs.aggregate.input().schema())?;
        }
    }
    for (group_expr, _alias) in inputs.aggregate.group_expr().expr() {
        validate_physical_expr(group_expr, &inputs.aggregate.input().schema())?;
    }
    if let Some(projection) = inputs.post_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    Ok(())
}

pub(super) struct SyntheticAggregateInputProjection {
    pub(super) expressions: Vec<NativeExpr>,
    pub(super) schema: SchemaRef,
    pub(super) aggregate_arg_aliases: Vec<String>,
}

pub(super) fn synthetic_aggregate_input_projection(
    aggregate: &AggregateExec,
    input_schema: &SchemaRef,
) -> Result<SyntheticAggregateInputProjection, FallbackReason> {
    let mut expressions = Vec::new();
    let mut fields = Vec::new();

    for (group_expr, alias) in aggregate.group_expr().expr() {
        expressions.push(alias_if_needed(lower_physical_expr(group_expr, input_schema)?, alias));
        fields.push(physical_expr_field(group_expr, input_schema, alias)?);
    }

    let mut aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
    for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
        let expressions_for_aggregate = aggregate_expr.expressions();
        let [arg] = expressions_for_aggregate.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        // cuDF aggregate lowering consumes columns, so SQL aggregate arguments
        // such as SUM(m00 + m01) are first materialized as synthetic projection columns.
        let alias = unique_synthetic_aggregate_arg_alias(index, input_schema, &fields);
        expressions.push(alias_if_needed(lower_physical_expr(arg, input_schema)?, alias.as_str()));
        fields.push(physical_expr_field(arg, input_schema, alias.as_str())?);
        aggregate_arg_aliases.push(alias);
    }

    Ok(SyntheticAggregateInputProjection { expressions, schema: Arc::new(Schema::new(fields)), aggregate_arg_aliases })
}

pub(super) fn synthetic_aggregate_input_projection_if_needed(
    aggregate: &AggregateExec,
    input_schema: &SchemaRef,
) -> Result<Option<SyntheticAggregateInputProjection>, FallbackReason> {
    let needs_projection = aggregate.aggr_expr().iter().any(|aggregate_expr| {
        let expressions = aggregate_expr.expressions();
        !matches!(expressions.as_slice(), [arg] if arg.as_any().downcast_ref::<Column>().is_some())
    });
    if !needs_projection {
        return Ok(None);
    }

    let mut expressions =
        input_schema.fields().iter().map(|field| NativeExpr::column(field.name())).collect::<Vec<_>>();
    let mut fields = input_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();

    let mut aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
    for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
        let expressions_for_aggregate = aggregate_expr.expressions();
        let [arg] = expressions_for_aggregate.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        let alias = unique_synthetic_aggregate_arg_alias(index, input_schema, &fields);
        expressions.push(alias_if_needed(lower_physical_expr(arg, input_schema)?, alias.as_str()));
        fields.push(physical_expr_field(arg, input_schema, alias.as_str())?);
        aggregate_arg_aliases.push(alias);
    }

    Ok(Some(SyntheticAggregateInputProjection {
        expressions,
        schema: Arc::new(Schema::new(fields)),
        aggregate_arg_aliases,
    }))
}

fn unique_synthetic_aggregate_arg_alias(index: usize, input_schema: &SchemaRef, fields: &[Field]) -> String {
    let base = format!("__cudf_aggregate_arg_{index}");
    if synthetic_alias_is_available(base.as_str(), input_schema, fields) {
        return base;
    }

    for suffix in 1usize.. {
        let alias = format!("{base}_{suffix}");
        if synthetic_alias_is_available(alias.as_str(), input_schema, fields) {
            return alias;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn synthetic_alias_is_available(alias: &str, input_schema: &SchemaRef, fields: &[Field]) -> bool {
    input_schema.index_of(alias).is_err() && fields.iter().all(|field| field.name() != alias)
}

pub(super) fn lower_aggregate_spec(
    aggregate: &AggregateExec,
    input_schema: &SchemaRef,
    synthetic_aggregate_args: Option<&[String]>,
) -> Result<NativeAggregateSpec, FallbackReason> {
    let group_exprs = aggregate
        .group_expr()
        .expr()
        .iter()
        .map(|(expr, alias)| {
            let native_expr = lower_physical_expr(expr, input_schema)?;
            Ok(alias_if_needed(native_expr, alias))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let aggr_exprs = aggregate
        .aggr_expr()
        .iter()
        .enumerate()
        .map(|(index, expr)| {
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            let lowered_arg = synthetic_aggregate_args
                .and_then(|aliases| aliases.get(index))
                .map_or_else(|| lower_physical_expr(arg, input_schema), |alias| Ok(NativeExpr::column(alias)))?;
            Ok(NativeAggregateExpr::new(lower_native_aggregate_function(expr.fun().name())?, lowered_arg, expr.name()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(NativeAggregateSpec::new(group_exprs, aggr_exprs))
}

pub(super) fn lower_zero_column_count_aggregate_spec(
    aggregate: &AggregateExec,
) -> Result<NativeAggregateSpec, FallbackReason> {
    if !aggregate_is_count_star_only(aggregate) {
        return Err(FallbackReason::UnsupportedOperator);
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

pub(super) fn lower_native_aggregate_function(name: &str) -> Result<NativeAggregateFunction, FallbackReason> {
    if name.eq_ignore_ascii_case("sum") {
        Ok(NativeAggregateFunction::Sum)
    } else if name.eq_ignore_ascii_case("min") {
        Ok(NativeAggregateFunction::Min)
    } else if name.eq_ignore_ascii_case("max") {
        Ok(NativeAggregateFunction::Max)
    } else if name.eq_ignore_ascii_case("count") {
        Ok(NativeAggregateFunction::Count)
    } else if name.eq_ignore_ascii_case("avg") {
        Ok(NativeAggregateFunction::Avg)
    } else {
        Err(FallbackReason::UnsupportedOperator)
    }
}
