//! Top-level `ORDER BY` of a GPU relation-function input.
//!
//! DataFusion's SQL planner drops a derived table's `ORDER BY` unless a
//! `LIMIT` sits above it, and its physical optimizer drops any sort no parent
//! requires. A relation function numbers its input rows (`row_ordinal`) and
//! fits models in input order, so the input's top-level `ORDER BY` is captured
//! from the SQL text into the function's logical node instead.

use std::fmt;

use datafusion::{
    logical_expr::{Expr, LogicalPlan, SortExpr, planner::RelationPlannerContext},
    sql::{
        planner::NullOrdering,
        sqlparser::ast::{
            Expr as SqlExpr, OrderByExpr, OrderByKind, OrderByOptions, Query as SqlQuery, Value, ValueWithSpan,
        },
    },
};

use crate::error::{Error, Result};

/// Resolves `query`'s top-level `ORDER BY` against `planned`, the logical plan
/// of that same query, as sort keys over the planned output columns.
///
/// A key must be an output column of the input, by name or by 1-based
/// position. Otherwise the call fails with `key_not_projected(key)`, unless the
/// query also bounds its rows (`LIMIT`, `OFFSET`, or `FETCH`): there the
/// `ORDER BY` selects rows, DataFusion keeps its sort, and no input order is
/// declared.
pub(crate) fn resolve_input_ordering(
    query: &SqlQuery,
    planned: &LogicalPlan,
    context: &mut dyn RelationPlannerContext,
    key_not_projected: impl Fn(String) -> Error,
) -> Result<Vec<SortExpr>> {
    let Some(order_by) = &query.order_by else {
        return Ok(Vec::new());
    };
    let resolved = match (&order_by.kind, &order_by.interpolate) {
        (OrderByKind::Expressions(keys), None) => {
            let null_ordering =
                NullOrdering::from(context.context_provider().options().sql_parser.default_null_ordering.as_str());
            keys.iter()
                .map(|key| resolve_key(key, planned, context, null_ordering).ok_or_else(|| key.to_string()))
                .collect::<std::result::Result<Vec<_>, _>>()
        },
        _ => Err(order_by.to_string()),
    };
    match resolved {
        Ok(ordering) => Ok(ordering),
        Err(_) if query.limit_clause.is_some() || query.fetch.is_some() => Ok(Vec::new()),
        Err(key) => Err(key_not_projected(key)),
    }
}

fn resolve_key(
    key: &OrderByExpr,
    planned: &LogicalPlan,
    context: &mut dyn RelationPlannerContext,
    null_ordering: NullOrdering,
) -> Option<SortExpr> {
    let OrderByExpr { expr, options: OrderByOptions { asc, nulls_first }, with_fill: None } = key else {
        return None;
    };
    let schema = planned.schema();
    let index = match expr {
        SqlExpr::Value(ValueWithSpan { value: Value::Number(position, _), .. }) => {
            position.parse::<usize>().ok()?.checked_sub(1).filter(|index| *index < schema.fields().len())?
        },
        expr => match context.sql_expr_to_logical_expr(expr.clone(), schema).ok()? {
            Expr::Column(column) => schema.index_of_column(&column).ok()?,
            _ => return None,
        },
    };
    let column = Expr::Column(schema.qualified_field(index).into());
    let asc = asc.unwrap_or(true);
    Some(SortExpr::new(column, asc, nulls_first.unwrap_or_else(|| null_ordering.nulls_first(asc))))
}

/// Sort-key expressions of every input, flattened in relation-role order, as
/// a logical node exposes them to optimizer rules.
pub(crate) fn input_ordering_expressions(input_orderings: &[Vec<SortExpr>]) -> Vec<Expr> {
    input_orderings.iter().flatten().map(|sort| sort.expr.clone()).collect()
}

/// Rebuilds `input_orderings` with rewritten sort-key expressions in the order
/// [`input_ordering_expressions`] produced them; `None` when the count differs.
pub(crate) fn input_orderings_with_expressions(
    input_orderings: &[Vec<SortExpr>],
    exprs: Vec<Expr>,
) -> Option<Vec<Vec<SortExpr>>> {
    if exprs.len() != input_orderings.iter().map(Vec::len).sum::<usize>() {
        return None;
    }
    let mut exprs = exprs.into_iter();
    Some(
        input_orderings
            .iter()
            .map(|ordering| ordering.iter().zip(exprs.by_ref()).map(|(sort, expr)| sort.with_expr(expr)).collect())
            .collect(),
    )
}

/// Appends `, input_order[<role>]=[<keys>]` for each ordered input.
pub(crate) fn fmt_input_orderings<'a>(
    f: &mut fmt::Formatter,
    roles: impl Iterator<Item = &'a str>,
    input_orderings: &[Vec<SortExpr>],
) -> fmt::Result {
    for (role, ordering) in roles.zip(input_orderings) {
        if !ordering.is_empty() {
            let keys = ordering.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
            write!(f, ", input_order[{role}]=[{keys}]")?;
        }
    }
    Ok(())
}
