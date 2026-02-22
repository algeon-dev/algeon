use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use cudf_query_engine::{is_supported_arrow_cast, is_supported_arrow_type};
use datafusion::{
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr,
            LikeExpr, Literal, NotExpr, TryCastExpr,
        },
    },
    physical_plan::{
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

use crate::policy::FallbackReason;

mod expressions;

pub(super) use expressions::{bounded_nonnegative_integral_cast_is_supported, validate_physical_expr};
use expressions::{literal_string, validate_literal_string, validate_substring_arguments_contract};

pub(super) fn validate_schema(schema: SchemaRef) -> Result<(), FallbackReason> {
    for field in schema.fields() {
        if !is_supported_native_type(field.data_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
    }
    Ok(())
}

pub(super) fn is_supported_native_type(data_type: &DataType) -> bool {
    is_supported_arrow_type(data_type)
}

pub(super) fn is_supported_native_cast(source_type: &DataType, target_type: &DataType) -> bool {
    if is_integer_to_timestamp_cast(source_type, target_type) {
        return false;
    }
    is_supported_arrow_cast(source_type, target_type)
}

pub(super) fn is_supported_to_timestamp_seconds_cast(source_type: &DataType, target_type: &DataType) -> bool {
    matches!(target_type, DataType::Timestamp(TimeUnit::Second, None))
        && is_integer_to_timestamp_cast(source_type, target_type)
        && is_supported_arrow_cast(source_type, target_type)
}

fn is_integer_to_timestamp_cast(source_type: &DataType, target_type: &DataType) -> bool {
    matches!(
        (source_type, target_type),
        (
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32,
            DataType::Timestamp(_, None),
        )
    )
}

pub(super) fn validate_aggregate(aggregate: &AggregateExec, allow_partial: bool) -> Result<(), FallbackReason> {
    if !(matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned)
        || allow_partial && matches!(aggregate.mode(), AggregateMode::Partial))
    {
        return Err(FallbackReason::UnsupportedOperator);
    }
    let group_by = aggregate.group_expr();
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if group_by.has_grouping_set() {
        validate_grouping_sets(group_by)?;
    } else if !group_by.expr().is_empty()
        && (group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null))
    {
        return Err(FallbackReason::UnsupportedOperator);
    }
    for aggregate_expr in aggregate.aggr_expr() {
        if aggregate_expr.is_distinct() && !aggregate_expr.fun().name().eq_ignore_ascii_case("count") {
            return Err(FallbackReason::UnsupportedOperator);
        }
        if !aggregate_expr.order_bys().is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        if !is_supported_native_aggregate_function(aggregate_expr.fun().name()) {
            return Err(FallbackReason::UnsupportedOperator);
        }
        let expressions = aggregate_expr.expressions();
        let [arg] = expressions.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        validate_physical_expr(arg, &aggregate.input().schema())?;
    }
    for (group_expr, _alias) in group_by.expr() {
        validate_physical_expr(group_expr, &aggregate.input().schema())?;
    }
    Ok(())
}

pub(super) fn validate_aggregate_for_native_lowering(
    aggregate: &AggregateExec,
    allow_partial: bool,
) -> Result<(), FallbackReason> {
    if aggregate.group_expr().has_grouping_set() {
        validate_grouping_set_aggregate(aggregate, allow_partial)
    } else {
        validate_aggregate(aggregate, allow_partial)
    }
}

fn validate_grouping_set_aggregate(aggregate: &AggregateExec, allow_partial: bool) -> Result<(), FallbackReason> {
    if !(matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned)
        || allow_partial && matches!(aggregate.mode(), AggregateMode::Partial))
    {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(FallbackReason::UnsupportedExpression);
    }

    let group_by = aggregate.group_expr();
    if !group_by.has_grouping_set()
        || group_by.expr().is_empty()
        || group_by.null_expr().len() != group_by.expr().len()
        || group_by.groups().is_empty()
        || group_by.groups().iter().any(|group| group.len() != group_by.expr().len())
    {
        return Err(FallbackReason::UnsupportedOperator);
    }
    for aggregate_expr in aggregate.aggr_expr() {
        if aggregate_expr.is_distinct() || !aggregate_expr.order_bys().is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        if !is_supported_native_aggregate_function(aggregate_expr.fun().name()) {
            return Err(FallbackReason::UnsupportedOperator);
        }
        let expressions = aggregate_expr.expressions();
        let [arg] = expressions.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        validate_physical_expr(arg, &aggregate.input().schema())?;
    }
    for (group_expr, _alias) in group_by.expr() {
        validate_physical_expr(group_expr, &aggregate.input().schema())?;
    }
    for (null_expr, _alias) in group_by.null_expr() {
        validate_physical_expr(null_expr, &aggregate.input().schema())?;
    }
    Ok(())
}

fn validate_grouping_sets(
    group_by: &datafusion::physical_plan::aggregates::PhysicalGroupBy,
) -> Result<(), FallbackReason> {
    let group_expr_count = group_by.expr().len();
    if group_expr_count == 0 || group_expr_count > 64 {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if group_by.null_expr().len() != group_expr_count {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if group_by.groups().is_empty() || group_by.groups().iter().any(|group| group.len() != group_expr_count) {
        return Err(FallbackReason::UnsupportedOperator);
    }
    Ok(())
}

pub(super) fn validate_sort(sort: &SortExec) -> Result<(), FallbackReason> {
    if sort.expr().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    for sort_expr in sort.expr() {
        validate_physical_expr(&sort_expr.expr, &sort.input().schema())?;
    }
    Ok(())
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
    if !expr.fun().name().eq_ignore_ascii_case("count") || expr.is_distinct() || !expr.order_bys().is_empty() {
        return false;
    }
    let expressions = expr.expressions();
    let [arg] = expressions.as_slice() else {
        return false;
    };
    let mut required_columns = BTreeSet::new();
    collect_required_columns(arg, &aggregate.input().schema(), &mut required_columns).is_ok()
        && required_columns.is_empty()
}

fn is_supported_native_aggregate_function(name: &str) -> bool {
    name.eq_ignore_ascii_case("sum")
        || name.eq_ignore_ascii_case("min")
        || name.eq_ignore_ascii_case("max")
        || name.eq_ignore_ascii_case("count")
        || name.eq_ignore_ascii_case("avg")
        || name.eq_ignore_ascii_case("stddev")
        || name.eq_ignore_ascii_case("stddev_samp")
}

pub(super) fn collect_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        required.insert(resolved_column_name(column, input_schema)?.to_string());
        return Ok(());
    }
    if expr.as_any().downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        collect_required_columns(binary.left(), input_schema, required)?;
        collect_required_columns(binary.right(), input_schema, required)?;
        return Ok(());
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        collect_required_columns(not.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        collect_required_columns(is_null.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        collect_required_columns(is_not_null.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        collect_required_columns(cast.expr(), input_schema, required)?;
        return Ok(());
    }
    if let Some(like) = expr.as_any().downcast_ref::<LikeExpr>() {
        validate_literal_string(like.pattern())?;
        collect_required_columns(like.expr(), input_schema, required)?;
        return Ok(());
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        collect_required_columns(in_list.expr(), input_schema, required)?;
        for expr in in_list.list() {
            let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            if literal.value().is_null() {
                return Err(FallbackReason::UnsupportedExpression);
            }
            collect_required_columns(expr, input_schema, required)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
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
    if let Some(dynamic_filter) = expr.as_any().downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| FallbackReason::UnsupportedExpression)?;
        return collect_required_columns(&current, input_schema, required);
    }
    if let Some(function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        collect_required_columns_for_scalar_function(function, input_schema, required)?;
        return Ok(());
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn collect_required_columns_for_scalar_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    match function.name().to_ascii_lowercase().as_str() {
        "substring" | "substr" => {
            let args = function.args();
            validate_substring_arguments_contract(function)?;
            collect_required_columns(&args[0], input_schema, required)?;
            Ok(())
        },
        "date_part" | "extract" => {
            let [field, expr] = function.args() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            match literal_string(field)?.to_ascii_lowercase().as_str() {
                "year" | "month" | "day" | "minute" => {},
                _ => return Err(FallbackReason::UnsupportedExpression),
            }
            collect_required_columns(expr, input_schema, required)
        },
        "date_trunc" => {
            let [field, expr] = function.args() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            match literal_string(field)?.to_ascii_lowercase().as_str() {
                "minute" => {},
                _ => return Err(FallbackReason::UnsupportedExpression),
            }
            collect_required_columns(expr, input_schema, required)
        },
        "to_timestamp_seconds" => {
            let [expr] = function.args() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            collect_required_columns(expr, input_schema, required)
        },
        _ => {
            for arg in function.args() {
                collect_required_columns(arg, input_schema, required)?;
            }
            Ok(())
        },
    }
}

fn resolved_column_name<'a>(column: &Column, input_schema: &'a SchemaRef) -> Result<&'a str, FallbackReason> {
    input_schema
        .fields()
        .get(column.index())
        .filter(|field| field.name() == column.name())
        .map(|field| field.name().as_str())
        .or_else(|| input_schema.field_with_name(column.name()).ok().map(|field| field.name().as_str()))
        .or_else(|| input_schema.fields().get(column.index()).map(|field| field.name().as_str()))
        .ok_or(FallbackReason::UnsupportedExpression)
}
