use std::sync::Arc;

use arrow::datatypes::DataType;
use cudf_query_engine::is_supported_arrow_type;
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, Column, Literal, NotExpr},
    },
    physical_plan::{
        aggregates::{AggregateExec, AggregateMode},
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

use crate::policy::FallbackReason;

pub(crate) fn projection_rejection(projection: &ProjectionExec) -> Option<FallbackReason> {
    projection.expr().iter().find_map(|expr| expr_rejection(&expr.expr))
}

pub(crate) fn sort_rejection(sort: &SortExec) -> Option<FallbackReason> {
    sort.expr()
        .iter()
        .any(|sort_expr| sort_expr.expr.as_any().downcast_ref::<Column>().is_none())
        .then_some(FallbackReason::UnsupportedExpression)
}

pub(crate) fn aggregate_rejection(aggregate: &AggregateExec) -> Option<FallbackReason> {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned) {
        return Some(FallbackReason::UnsupportedOperator);
    }

    let group_by = aggregate.group_expr();
    if !group_by.is_single() || group_by.expr().is_empty() {
        return Some(FallbackReason::UnsupportedOperator);
    }
    if group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null) {
        return Some(FallbackReason::UnsupportedOperator);
    }
    if group_by.expr().iter().any(|(expr, _alias)| expr.as_any().downcast_ref::<Column>().is_none()) {
        return Some(FallbackReason::UnsupportedExpression);
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Some(FallbackReason::UnsupportedExpression);
    }

    aggregate.aggr_expr().iter().find_map(|expr| aggregate_expr_rejection(expr))
}

pub(crate) fn aggregate_expr_rejection(
    expr: &datafusion::physical_expr::aggregate::AggregateFunctionExpr,
) -> Option<FallbackReason> {
    if expr.is_distinct() || !expr.order_bys().is_empty() {
        return Some(FallbackReason::UnsupportedExpression);
    }
    if !matches!(expr.fun().name().to_ascii_lowercase().as_str(), "sum" | "min" | "max" | "count" | "avg") {
        return Some(FallbackReason::UnsupportedOperator);
    }
    let expressions = expr.expressions();
    if expressions.len() != 1 || expressions[0].as_any().downcast_ref::<Column>().is_none() {
        return Some(FallbackReason::UnsupportedExpression);
    }
    None
}

pub(crate) fn expr_rejection(expr: &Arc<dyn PhysicalExpr>) -> Option<FallbackReason> {
    if expr.as_any().downcast_ref::<Column>().is_some() {
        return None;
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return (!literal_supported(literal.value())).then_some(FallbackReason::UnsupportedExpression);
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if !binary_operator_supported(binary.op()) {
            return Some(FallbackReason::UnsupportedExpression);
        }
        return [expr_rejection(binary.left()), expr_rejection(binary.right())].into_iter().flatten().next();
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return expr_rejection(not.arg());
    }
    Some(FallbackReason::UnsupportedExpression)
}

fn literal_supported(value: &ScalarValue) -> bool {
    matches!(
        value,
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
    ) || matches!(value, ScalarValue::Decimal128(Some(_), precision, scale) if decimal128_literal_supported(*precision, *scale))
}

fn decimal128_literal_supported(precision: u8, scale: i8) -> bool {
    is_supported_arrow_type(&DataType::Decimal128(precision, scale))
}

fn binary_operator_supported(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::And
            | Operator::Or
    )
}
