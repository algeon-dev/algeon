use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef};
use datafusion::{
    physical_expr::{
        LexOrdering, PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr,
            LikeExpr, Literal, NotExpr, TryCastExpr,
        },
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::FilterExec,
        projection::ProjectionExec,
    },
};
use query_engine::{
    capability::is_cudf_median_input_type,
    datatypes::{is_supported_arrow_type, is_timestamp_with_timezone},
    expr::AggregateFunction as NativeAggregateFunction,
};

use super::{
    LoweringRejection,
    expression_contract::{
        is_list_carry_type, is_narrow_decimal_carry, native_aggregate_function_kind, resolve_column,
        validate_physical_expr_contract,
    },
};
use crate::capability::CapabilityReason;

pub(super) fn validate_schema(schema: SchemaRef) -> Result<(), CapabilityReason> {
    for field in schema.fields() {
        if !is_supported_arrow_type(field.data_type()) {
            return Err(CapabilityReason::UnsupportedType);
        }
    }
    Ok(())
}

pub(super) fn validate_aggregate(aggregate: &AggregateExec, allow_partial: bool) -> Result<(), LoweringRejection> {
    if !(matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned)
        || allow_partial && matches!(aggregate.mode(), AggregateMode::Partial))
    {
        return Err(CapabilityReason::UnsupportedOperator.into());
    }
    let group_by = aggregate.group_expr();
    if group_by.has_grouping_set() {
        validate_grouping_sets(group_by)?;
    } else if !group_by.expr().is_empty()
        && (group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null))
    {
        return Err(CapabilityReason::UnsupportedOperator.into());
    }
    if aggregate_has_median_and_floating_statistical(aggregate) {
        return Err(CapabilityReason::UnsupportedOperator.into());
    }
    for aggregate_expr in aggregate.aggr_expr() {
        let kind = native_aggregate_function_kind(aggregate_expr.fun());
        if aggregate_expr.is_distinct() && !matches!(kind, Some(NativeAggregateFunction::Count)) {
            return Err(CapabilityReason::UnsupportedOperator.into());
        }
        if !aggregate_expr.order_bys().is_empty() {
            return Err(CapabilityReason::UnsupportedExpression.into());
        }
        if kind.is_none() {
            return Err(CapabilityReason::UnsupportedOperator.into());
        }
        if matches!(aggregate.mode(), AggregateMode::Partial) && matches!(kind, Some(NativeAggregateFunction::Median)) {
            return Err(CapabilityReason::UnsupportedOperator.into());
        }
        let expressions = aggregate_expr.expressions();
        let [arg] = expressions.as_slice() else {
            return Err(CapabilityReason::UnsupportedExpression.into());
        };
        validate_physical_expr_contract(arg, &aggregate.input().schema())?;
        validate_operator_value_expr(arg, &aggregate.input().schema())?;
        if matches!(kind, Some(NativeAggregateFunction::Median)) {
            validate_median_input_type(arg, &aggregate.input().schema())?;
        }
    }
    validate_aggregate_filters(aggregate)?;
    for (group_expr, _alias) in group_by.expr() {
        validate_operator_key_expr(group_expr, &aggregate.input().schema())?;
    }
    Ok(())
}

fn aggregate_has_median_and_floating_statistical(aggregate: &AggregateExec) -> bool {
    let has_median = aggregate
        .aggr_expr()
        .iter()
        .any(|expr| matches!(native_aggregate_function_kind(expr.fun()), Some(NativeAggregateFunction::Median)));
    has_median && aggregate_has_floating_statistical(aggregate)
}

fn aggregate_has_floating_statistical(aggregate: &AggregateExec) -> bool {
    aggregate.aggr_expr().iter().any(|expr| {
        matches!(
            native_aggregate_function_kind(expr.fun()),
            Some(
                NativeAggregateFunction::StddevSamp
                    | NativeAggregateFunction::StddevPop
                    | NativeAggregateFunction::VarSamp
                    | NativeAggregateFunction::VarPop
            )
        )
    })
}

/// Whether DataFusion reads the input of a floating statistical aggregate in
/// `plan` through more than one partition. DataFusion's own result then
/// depends on the order partitions deliver rows, which no native evaluation
/// order reproduces.
pub(crate) fn floating_statistical_aggregate_input_is_partitioned(plan: &dyn ExecutionPlan) -> bool {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
        && aggregate_has_floating_statistical(aggregate)
        && subtree_has_multiple_partitions(aggregate.input().as_ref())
    {
        return true;
    }
    plan.children().into_iter().any(|child| floating_statistical_aggregate_input_is_partitioned(child.as_ref()))
}

fn subtree_has_multiple_partitions(plan: &dyn ExecutionPlan) -> bool {
    plan.properties().output_partitioning().partition_count() > 1
        || plan.children().into_iter().any(|child| subtree_has_multiple_partitions(child.as_ref()))
}

pub(super) fn validate_aggregate_for_native_lowering(
    aggregate: &AggregateExec,
    allow_partial: bool,
) -> Result<(), LoweringRejection> {
    if aggregate.group_expr().has_grouping_set() {
        validate_grouping_set_aggregate(aggregate, allow_partial)
    } else {
        validate_aggregate(aggregate, allow_partial)
    }
}

fn validate_grouping_set_aggregate(aggregate: &AggregateExec, allow_partial: bool) -> Result<(), LoweringRejection> {
    if !(matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned)
        || allow_partial && matches!(aggregate.mode(), AggregateMode::Partial))
    {
        return Err(CapabilityReason::UnsupportedOperator.into());
    }
    let group_by = aggregate.group_expr();
    if !group_by.has_grouping_set()
        || group_by.expr().is_empty()
        || group_by.null_expr().len() != group_by.expr().len()
        || group_by.groups().is_empty()
        || group_by.groups().iter().any(|group| group.len() != group_by.expr().len())
    {
        return Err(CapabilityReason::UnsupportedOperator.into());
    }
    for aggregate_expr in aggregate.aggr_expr() {
        if aggregate_expr.is_distinct() || !aggregate_expr.order_bys().is_empty() {
            return Err(CapabilityReason::UnsupportedExpression.into());
        }
        let kind = native_aggregate_function_kind(aggregate_expr.fun());
        if kind.is_none() {
            return Err(CapabilityReason::UnsupportedOperator.into());
        }
        if matches!(kind, Some(NativeAggregateFunction::Median)) {
            return Err(CapabilityReason::UnsupportedOperator.into());
        }
        let expressions = aggregate_expr.expressions();
        let [arg] = expressions.as_slice() else {
            return Err(CapabilityReason::UnsupportedExpression.into());
        };
        validate_physical_expr_contract(arg, &aggregate.input().schema())?;
        validate_operator_value_expr(arg, &aggregate.input().schema())?;
    }
    validate_aggregate_filters(aggregate)?;
    for (group_expr, _alias) in group_by.expr() {
        validate_operator_key_expr(group_expr, &aggregate.input().schema())?;
    }
    for (null_expr, _alias) in group_by.null_expr() {
        validate_physical_expr_contract(null_expr, &aggregate.input().schema())?;
    }
    Ok(())
}

fn validate_aggregate_filters(aggregate: &AggregateExec) -> Result<(), LoweringRejection> {
    for filter in aggregate.filter_expr().iter().flatten() {
        validate_physical_expr_contract(filter, &aggregate.input().schema())?;
        let data_type = filter
            .data_type(aggregate.input().schema().as_ref())
            .map_err(|_error| CapabilityReason::UnsupportedExpression)?;
        if !matches!(data_type, DataType::Boolean) {
            return Err(CapabilityReason::UnsupportedType.into());
        }
    }
    Ok(())
}

fn validate_grouping_sets(group_by: &PhysicalGroupBy) -> Result<(), CapabilityReason> {
    let group_expr_count = group_by.expr().len();
    if group_expr_count == 0 || group_expr_count > 64 {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    if group_by.null_expr().len() != group_expr_count {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    if group_by.groups().is_empty() || group_by.groups().iter().any(|group| group.len() != group_expr_count) {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    Ok(())
}

pub(super) fn validate_sort_ordering(
    ordering: &LexOrdering,
    input_schema: &SchemaRef,
) -> Result<(), LoweringRejection> {
    if ordering.is_empty() {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    for sort_expr in ordering {
        validate_operator_key_expr(&sort_expr.expr, input_schema)?;
    }
    Ok(())
}

fn validate_operator_key_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), LoweringRejection> {
    validate_physical_expr_contract(expr, input_schema)?;
    validate_operator_value_expr(expr, input_schema)
}

fn validate_operator_value_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), LoweringRejection> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    if is_timestamp_with_timezone(&data_type) || is_list_carry_type(&data_type) || is_narrow_decimal_carry(&data_type) {
        Err(CapabilityReason::UnsupportedType.into())
    } else {
        Ok(())
    }
}

fn validate_median_input_type(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), CapabilityReason> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    if is_cudf_median_input_type(&data_type) { Ok(()) } else { Err(CapabilityReason::UnsupportedType) }
}

pub(super) fn native_zero_column_count_source_is_safe(
    filter: Option<&FilterExec>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    aggregate: &AggregateExec,
) -> bool {
    filter.is_none() && pre_aggregate_projection.is_none() && aggregate_is_count_star_only(aggregate)
}

pub(super) fn aggregate_is_count_star_only(aggregate: &AggregateExec) -> bool {
    if !aggregate.group_expr().expr().is_empty()
        || !aggregate.filter_expr().iter().all(Option::is_none)
        || aggregate.aggr_expr().len() != 1
    {
        return false;
    }
    let expr = &aggregate.aggr_expr()[0];
    if !matches!(native_aggregate_function_kind(expr.fun()), Some(NativeAggregateFunction::Count))
        || expr.is_distinct()
        || !expr.order_bys().is_empty()
    {
        return false;
    }
    let expressions = expr.expressions();
    let [arg] = expressions.as_slice() else {
        return false;
    };
    if arg.downcast_ref::<Literal>().is_none_or(|literal| literal.value().is_null()) {
        return false;
    }
    let mut required_columns = BTreeSet::new();
    collect_required_columns(arg, &aggregate.input().schema(), &mut required_columns).is_ok()
        && required_columns.is_empty()
}

pub(super) fn collect_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), CapabilityReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let column = resolve_column(column, input_schema).map_err(|rejection| rejection.capability_reason())?;
        required.insert(column.field().name().to_string());
        return Ok(());
    }
    if expr.downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        collect_required_columns(binary.left(), input_schema, required)?;
        collect_required_columns(binary.right(), input_schema, required)?;
        return Ok(());
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        collect_required_columns(not.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        collect_required_columns(is_null.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        collect_required_columns(is_not_null.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        collect_required_columns(cast.expr(), input_schema, required)?;
        return Ok(());
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        collect_required_columns(like.expr(), input_schema, required)?;
        return Ok(());
    }
    // IN-list admission belongs to the expression contract, which rejects
    // the list during lowering with its own capability.
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        collect_required_columns(in_list.expr(), input_schema, required)?;
        for expr in in_list.list() {
            collect_required_columns(expr, input_schema, required)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if let Some(expr) = case.expr() {
            collect_required_columns(expr, input_schema, required)?;
        }
        for (when, then) in case.when_then_expr() {
            collect_required_columns(when, input_schema, required)?;
            collect_required_columns(then, input_schema, required)?;
        }
        if let Some(else_expr) = case.else_expr() {
            collect_required_columns(else_expr, input_schema, required)?;
        }
        return Ok(());
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| CapabilityReason::UnsupportedExpression)?;
        return collect_required_columns(&current, input_schema, required);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        collect_required_columns_for_scalar_function(function, input_schema, required)?;
        return Ok(());
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        collect_required_columns(try_cast.expr(), input_schema, required)?;
        return Ok(());
    }
    Err(CapabilityReason::UnsupportedExpression)
}

fn collect_required_columns_for_scalar_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), CapabilityReason> {
    for arg in function.args() {
        collect_required_columns(arg, input_schema, required)?;
    }
    Ok(())
}
