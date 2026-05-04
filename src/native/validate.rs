use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef};
use cudf_query_engine::{is_supported_arrow_cast, is_supported_arrow_type};
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, CastExpr, Column, Literal, NotExpr, TryCastExpr},
    },
    physical_plan::{
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

use crate::policy::FallbackReason;

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
    is_supported_arrow_cast(source_type, target_type)
}

pub(super) fn validate_aggregate(
    aggregate: &AggregateExec,
    aggregate_args_must_be_columns: bool,
    allow_partial: bool,
) -> Result<(), FallbackReason> {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned)
        && !(allow_partial && matches!(aggregate.mode(), AggregateMode::Partial))
    {
        return Err(FallbackReason::UnsupportedOperator);
    }
    let group_by = aggregate.group_expr();
    if group_by.expr().is_empty() {
        return if aggregate_is_count_star_only(aggregate) { Ok(()) } else { Err(FallbackReason::UnsupportedOperator) };
    }
    if !group_by.is_single() || group_by.expr().is_empty() {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null) {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(FallbackReason::UnsupportedExpression);
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
        if aggregate_args_must_be_columns
            && !aggregate_expr.fun().name().eq_ignore_ascii_case("count")
            && arg.as_any().downcast_ref::<Column>().is_none()
        {
            return Err(FallbackReason::UnsupportedExpression);
        }
        validate_physical_expr(arg, &aggregate.input().schema())?;
    }
    for (group_expr, _alias) in group_by.expr() {
        if group_expr.as_any().downcast_ref::<Column>().is_none() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }
    Ok(())
}

pub(super) fn validate_sort(sort: &SortExec) -> Result<(), FallbackReason> {
    if sort.expr().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if sort.expr().iter().any(|sort_expr| sort_expr.expr.as_any().downcast_ref::<Column>().is_none()) {
        return Err(FallbackReason::UnsupportedExpression);
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
    collect_required_columns(arg, &mut required_columns).is_ok() && required_columns.is_empty()
}

fn is_supported_native_aggregate_function(name: &str) -> bool {
    name.eq_ignore_ascii_case("sum")
        || name.eq_ignore_ascii_case("min")
        || name.eq_ignore_ascii_case("max")
        || name.eq_ignore_ascii_case("count")
        || name.eq_ignore_ascii_case("avg")
}

pub(super) fn validate_physical_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return validate_column_type(column, input_schema);
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return validate_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        validate_physical_expr(binary.left(), input_schema)?;
        validate_physical_expr(binary.right(), input_schema)?;
        validate_binary_operator(binary.op())?;
        return Ok(());
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return validate_physical_expr(not.arg(), input_schema);
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        return validate_physical_expr(cast.expr(), input_schema);
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn validate_column_type(column: &Column, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let field = input_schema.field_with_name(column.name()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(field.data_type()) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn validate_literal(value: &ScalarValue) -> Result<(), FallbackReason> {
    match value {
        ScalarValue::Boolean(Some(_))
        | ScalarValue::Int8(Some(_))
        | ScalarValue::Int16(Some(_))
        | ScalarValue::Int32(Some(_))
        | ScalarValue::Int64(Some(_))
        | ScalarValue::UInt8(Some(_))
        | ScalarValue::UInt16(Some(_))
        | ScalarValue::UInt32(Some(_))
        | ScalarValue::UInt64(Some(_))
        | ScalarValue::Float32(Some(_))
        | ScalarValue::Float64(Some(_))
        | ScalarValue::Utf8(Some(_))
        | ScalarValue::Utf8View(Some(_))
        | ScalarValue::LargeUtf8(Some(_))
        | ScalarValue::Decimal128(Some(_), _, _)
        | ScalarValue::Date32(Some(_))
        | ScalarValue::TimestampSecond(Some(_), None)
        | ScalarValue::TimestampMillisecond(Some(_), None)
        | ScalarValue::TimestampMicrosecond(Some(_), None)
        | ScalarValue::TimestampNanosecond(Some(_), None)
        | ScalarValue::DurationSecond(Some(_))
        | ScalarValue::DurationMillisecond(Some(_))
        | ScalarValue::DurationMicrosecond(Some(_))
        | ScalarValue::DurationNanosecond(Some(_))
        | ScalarValue::Boolean(None)
        | ScalarValue::Int8(None)
        | ScalarValue::Int16(None)
        | ScalarValue::Int32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::UInt8(None)
        | ScalarValue::UInt16(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::Float32(None)
        | ScalarValue::Float64(None)
        | ScalarValue::Utf8(None)
        | ScalarValue::Utf8View(None)
        | ScalarValue::LargeUtf8(None)
        | ScalarValue::Decimal128(None, _, _)
        | ScalarValue::Date32(None)
        | ScalarValue::TimestampSecond(None, None)
        | ScalarValue::TimestampMillisecond(None, None)
        | ScalarValue::TimestampMicrosecond(None, None)
        | ScalarValue::TimestampNanosecond(None, None)
        | ScalarValue::DurationSecond(None)
        | ScalarValue::DurationMillisecond(None)
        | ScalarValue::DurationMicrosecond(None)
        | ScalarValue::DurationNanosecond(None) => Ok(()),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn validate_binary_operator(op: &Operator) -> Result<(), FallbackReason> {
    match op {
        Operator::Plus
        | Operator::Minus
        | Operator::Multiply
        | Operator::Divide
        | Operator::Eq
        | Operator::NotEq
        | Operator::Lt
        | Operator::LtEq
        | Operator::Gt
        | Operator::GtEq
        | Operator::And
        | Operator::Or => Ok(()),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

pub(super) fn collect_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        required.insert(column.name().to_string());
        return Ok(());
    }
    if expr.as_any().downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        collect_required_columns(binary.left(), required)?;
        collect_required_columns(binary.right(), required)?;
        return Ok(());
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        collect_required_columns(not.arg(), required)?;
        return Ok(());
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        collect_required_columns(cast.expr(), required)?;
        return Ok(());
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Err(FallbackReason::UnsupportedExpression)
}
