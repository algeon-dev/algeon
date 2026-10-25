//! DataFusion filters as Iceberg predicates for manifest pruning.
//!
//! Every filter stays in the plan above the scan (the provider reports
//! `Inexact` pushdown), so a filter that cannot be converted only loses
//! pruning. Within an `AND`, a convertible side still prunes on its own;
//! under `NOT` and `OR` both sides must convert, since dropping one side
//! there would widen or narrow the result.

use std::{ops::Not, sync::Arc};

use datafusion::{
    logical_expr::{Expr, Operator},
    scalar::ScalarValue,
};
use iceberg::{
    expr::{Bind, Predicate, Reference},
    spec::{Datum, SchemaRef},
};

/// The conjunction of the filters that convert and bind to `schema`.
///
/// A filter whose literal does not bind to the column type is dropped, so a
/// type the converter maps imprecisely never fails the scan.
pub(crate) fn filters_to_predicate(filters: &[Expr], schema: &SchemaRef) -> Option<Predicate> {
    filters
        .iter()
        .filter_map(|filter| filter_to_predicate(filter, true))
        .filter(|predicate| predicate.bind(Arc::clone(schema), true).is_ok())
        .reduce(Predicate::and)
}

fn filter_to_predicate(expr: &Expr, allow_partial_and: bool) -> Option<Predicate> {
    match expr {
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And => {
                let left = filter_to_predicate(&binary.left, allow_partial_and);
                let right = filter_to_predicate(&binary.right, allow_partial_and);
                match (left, right) {
                    (Some(left), Some(right)) => Some(left.and(right)),
                    (Some(side), None) | (None, Some(side)) if allow_partial_and => Some(side),
                    _ => None,
                }
            },
            Operator::Or => Some(
                filter_to_predicate(&binary.left, allow_partial_and)?
                    .or(filter_to_predicate(&binary.right, allow_partial_and)?),
            ),
            Operator::Eq | Operator::NotEq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                comparison_to_predicate(&binary.left, binary.op, &binary.right)
                    .or_else(|| comparison_to_predicate(&binary.right, reverse_operator(binary.op)?, &binary.left))
            },
            _ => None,
        },
        Expr::Not(expr) => filter_to_predicate(expr, false).map(Not::not),
        Expr::IsNull(expr) => column_name(expr).map(|column| Reference::new(column).is_null()),
        Expr::IsNotNull(expr) => column_name(expr).map(|column| Reference::new(column).is_not_null()),
        Expr::Column(column) => Some(Reference::new(column.name()).equal_to(Datum::bool(true))),
        _ => None,
    }
}

fn comparison_to_predicate(left: &Expr, operator: Operator, right: &Expr) -> Option<Predicate> {
    let reference = Reference::new(column_name(left)?);
    let Expr::Literal(value, _) = right else {
        return None;
    };
    let datum = scalar_to_datum(value)?;
    Some(match operator {
        Operator::Eq => reference.equal_to(datum),
        Operator::NotEq => reference.not_equal_to(datum),
        Operator::Lt => reference.less_than(datum),
        Operator::LtEq => reference.less_than_or_equal_to(datum),
        Operator::Gt => reference.greater_than(datum),
        Operator::GtEq => reference.greater_than_or_equal_to(datum),
        _ => return None,
    })
}

fn column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.name()),
        _ => None,
    }
}

fn scalar_to_datum(value: &ScalarValue) -> Option<Datum> {
    match value {
        ScalarValue::Boolean(Some(value)) => Some(Datum::bool(*value)),
        ScalarValue::Int8(Some(value)) => Some(Datum::int(i32::from(*value))),
        ScalarValue::Int16(Some(value)) => Some(Datum::int(i32::from(*value))),
        ScalarValue::Int32(Some(value)) => Some(Datum::int(*value)),
        ScalarValue::Int64(Some(value)) => Some(Datum::long(*value)),
        ScalarValue::Float32(Some(value)) => Some(Datum::float(*value)),
        ScalarValue::Float64(Some(value)) => Some(Datum::double(*value)),
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Some(Datum::string(value))
        },
        ScalarValue::Date32(Some(value)) => Some(Datum::date(*value)),
        _ => None,
    }
}

fn reverse_operator(operator: Operator) -> Option<Operator> {
    match operator {
        Operator::Eq => Some(Operator::Eq),
        Operator::NotEq => Some(Operator::NotEq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
