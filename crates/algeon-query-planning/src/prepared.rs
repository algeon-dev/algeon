use arrow::datatypes::DataType as ArrowDataType;

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, SortSpec},
};

mod aggregate;
mod avg;
mod count;

#[doc(hidden)]
pub use aggregate::{
    DataFusionExactHostAccumulator, aggregate_expr_is_exact_nunique, aggregate_spec_has_avg,
    aggregate_spec_has_distinct, aggregate_spec_has_filters, aggregate_spec_has_floating_statistical_aggregate,
    aggregate_spec_has_median, aggregate_spec_requires_whole_input, datafusion_exact_host_accumulator,
    datafusion_exact_host_group_key_supported,
};
#[doc(hidden)]
pub use avg::{AVG_COUNT_STATE_TYPE, AvgStateAliases, avg_argument_cast_type, avg_final_expr, avg_sum_state_type};
pub(crate) use avg::{avg_declared_sum_state_type, avg_division_type, avg_merge_aggregates, avg_partial_aggregates};
pub(crate) use count::{merged_count_field, zero_filled_count_expr};

// cuDF-ready sort keys split into the parallel vectors expected by execution.
#[doc(hidden)]
pub struct LoweredSort {
    #[doc(hidden)]
    pub columns: Vec<String>,
    #[doc(hidden)]
    pub descending: Vec<bool>,
    #[doc(hidden)]
    pub nulls_first: Vec<bool>,
}

#[doc(hidden)]
pub fn lower_sort_spec(spec: &SortSpec) -> Result<LoweredSort> {
    if spec.expressions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.sort.lower",
            "native execution requires at least one sort expression",
        ));
    }

    let mut columns = Vec::with_capacity(spec.expressions.len());
    let mut descending = Vec::with_capacity(spec.expressions.len());
    let mut nulls_first = Vec::with_capacity(spec.expressions.len());
    for sort_expr in &spec.expressions {
        columns.push(lower_column_expr(&sort_expr.expr, "query_engine.sort.lower")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    Ok(LoweredSort { columns, descending, nulls_first })
}

#[doc(hidden)]
pub fn lower_named_column_expr(expr: &Expr, operation: &'static str) -> Result<(String, String)> {
    match expr {
        Expr::Column(column) => Ok((column.name().to_string(), column.display_name().to_string())),
        Expr::Alias { expr, alias } => Ok((lower_column_expr(expr, operation)?, alias.clone())),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

#[doc(hidden)]
pub fn lower_column_expr(expr: &Expr, operation: &'static str) -> Result<String> {
    match expr {
        Expr::Column(column) => Ok(column.name().to_string()),
        Expr::Alias { expr, .. } => lower_column_expr(expr, operation),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}
