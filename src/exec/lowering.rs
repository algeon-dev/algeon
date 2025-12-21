use std::sync::Arc;

use arrow::datatypes::DataType;
use cudf::{
    Expr, GroupByAgg,
    ops::{col, expr::lit_decimal128_raw, lit},
};
use cudf_query_engine::is_supported_arrow_type;
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, PhysicalSortExpr,
        expressions::{BinaryExpr, Column, Literal, NotExpr},
    },
    physical_plan::aggregates::{AggregateExec, AggregateMode},
};

use crate::error::{Error, ErrorCode, Result as AdapterResult};

#[derive(Debug)]
pub(super) struct LoweredSortExprs {
    pub(super) columns: Vec<String>,
    pub(super) descending: Vec<bool>,
    pub(super) nulls_first: Vec<bool>,
}

#[derive(Debug, Clone)]
pub(super) struct GpuAggregate {
    pub(super) keys: Vec<GpuGroupKey>,
    pub(super) aggs: Vec<GpuAggregateExpr>,
}

#[derive(Debug, Clone)]
pub(super) struct GpuGroupKey {
    pub(super) input_column: String,
    pub(super) output_name: String,
}

#[derive(Debug, Clone)]
pub(super) struct GpuAggregateExpr {
    pub(super) input_column: String,
    pub(super) agg: GroupByAgg,
    pub(super) output_name: String,
}

pub(super) fn lower_aggregate_exec(aggregate: &AggregateExec) -> AdapterResult<GpuAggregate> {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned) {
        return Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.aggregate.lower",
            "cannot lower DataFusion aggregate mode that consumes or produces partial aggregate state",
        )
        .with_fact("aggregate_mode", format!("{:?}", aggregate.mode())));
    }

    let group_by = aggregate.group_expr();
    if !group_by.is_single() || group_by.expr().is_empty() {
        return Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.aggregate.lower",
            "cannot lower global aggregates or grouping sets to cudf groupby",
        ));
    }
    if group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null) {
        return Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.aggregate.lower",
            "cannot lower grouping-set null masks to cudf groupby",
        ));
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower",
            "cannot lower aggregate FILTER clauses to cudf groupby",
        ));
    }

    let keys = group_by
        .expr()
        .iter()
        .map(|(expr, alias)| {
            let Some(column) = expr.as_any().downcast_ref::<Column>() else {
                return Err(Error::new(
                    ErrorCode::UnsupportedExpression,
                    "datafusion_cudf.aggregate.lower_group_key",
                    "cannot lower non-column group key expression",
                )
                .with_fact("expression", format!("{expr:?}")));
            };
            Ok(GpuGroupKey { input_column: column.name().to_string(), output_name: alias.clone() })
        })
        .collect::<AdapterResult<Vec<_>>>()?;

    let aggs =
        aggregate.aggr_expr().iter().map(|expr| lower_aggregate_expr(expr)).collect::<AdapterResult<Vec<_>>>()?;

    Ok(GpuAggregate { keys, aggs })
}

fn lower_aggregate_expr(
    expr: &datafusion::physical_expr::aggregate::AggregateFunctionExpr,
) -> AdapterResult<GpuAggregateExpr> {
    if expr.is_distinct() {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower DISTINCT aggregate expressions to cudf groupby",
        )
        .with_fact("aggregate", expr.name()));
    }
    if !expr.order_bys().is_empty() {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower order-sensitive aggregate expressions to cudf groupby",
        )
        .with_fact("aggregate", expr.name()));
    }

    let expressions = expr.expressions();
    let [arg] = expressions.as_slice() else {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower aggregate expression with multiple arguments",
        )
        .with_fact("aggregate", expr.name())
        .with_fact("argument_count", expressions.len()));
    };
    let Some(column) = arg.as_any().downcast_ref::<Column>() else {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower aggregate expression over non-column argument",
        )
        .with_fact("aggregate", expr.name())
        .with_fact("argument", format!("{arg:?}")));
    };

    let agg = match expr.fun().name().to_ascii_lowercase().as_str() {
        "sum" => GroupByAgg::Sum,
        "min" => GroupByAgg::Min,
        "max" => GroupByAgg::Max,
        "count" => GroupByAgg::Count { include_nulls: false },
        "avg" => GroupByAgg::Mean,
        _ => {
            return Err(Error::new(
                ErrorCode::UnsupportedOperator,
                "datafusion_cudf.aggregate.lower_expr",
                "cannot lower unsupported aggregate function to cudf groupby",
            )
            .with_fact("aggregate_function", expr.fun().name())
            .with_fact("aggregate", expr.name()));
        },
    };

    Ok(GpuAggregateExpr { input_column: column.name().to_string(), agg, output_name: expr.name().to_string() })
}

pub(super) fn lower_sort_exprs(sort: &[PhysicalSortExpr]) -> AdapterResult<LoweredSortExprs> {
    let mut columns = Vec::with_capacity(sort.len());
    let mut descending = Vec::with_capacity(sort.len());
    let mut nulls_first = Vec::with_capacity(sort.len());
    for sort_expr in sort {
        let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
            return Err(Error::new(
                ErrorCode::UnsupportedExpression,
                "datafusion_cudf.sort.lower",
                "cannot lower unsupported sort expression",
            )
            .with_fact("expression", format!("{:?}", sort_expr.expr)));
        };
        columns.push(column.name().to_string());
        descending.push(sort_expr.options.descending);
        nulls_first.push(sort_expr.options.nulls_first);
    }
    Ok(LoweredSortExprs { columns, descending, nulls_first })
}

pub(super) fn lower_expr(expr: &Arc<dyn PhysicalExpr>) -> AdapterResult<Expr> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return Ok(col(column.name()));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_expr(binary.left())?;
        let right = lower_expr(binary.right())?;
        return lower_binary(left, binary.op(), right);
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(lower_expr(not.arg())?.not());
    }
    Err(Error::new(
        ErrorCode::UnsupportedExpression,
        "datafusion_cudf.expr.lower",
        "cannot lower unsupported physical expression",
    )
    .with_fact("expression", expr))
}

fn lower_literal(value: &ScalarValue) -> AdapterResult<Expr> {
    match value {
        ScalarValue::Boolean(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int8(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int16(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int32(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int64(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt8(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt16(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt32(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt64(Some(value)) => Ok(lit(*value)),
        ScalarValue::Float32(Some(value)) => Ok(lit(*value)),
        ScalarValue::Float64(Some(value)) => Ok(lit(*value)),
        ScalarValue::Decimal128(Some(raw), precision, scale) => {
            Ok(lit_decimal128_raw(*raw, datafusion_decimal_scale_to_cudf_scale(*precision, *scale)))
        },
        _ => Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.expr.lower_literal",
            "cannot lower unsupported or null literal",
        )
        .with_fact("literal", format!("{value:?}"))),
    }
}

fn datafusion_decimal_scale_to_cudf_scale(precision: u8, scale: i8) -> i32 {
    debug_assert!(is_supported_arrow_type(&DataType::Decimal128(precision, scale)));
    -i32::from(scale)
}

fn lower_binary(left: Expr, op: &Operator, right: Expr) -> AdapterResult<Expr> {
    match op {
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        Operator::Divide => Ok(left / right),
        Operator::Modulo => Ok(left % right),
        Operator::Eq => Ok(left.eq(right)),
        Operator::NotEq => Ok(left.ne(right)),
        Operator::Lt => Ok(left.lt(right)),
        Operator::LtEq => Ok(left.le(right)),
        Operator::Gt => Ok(left.gt(right)),
        Operator::GtEq => Ok(left.ge(right)),
        Operator::And => Ok(left & right),
        Operator::Or => Ok(left | right),
        _ => Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.expr.lower_binary",
            "cannot lower unsupported binary operator",
        )
        .with_fact("operator", format!("{op:?}"))),
    }
}
