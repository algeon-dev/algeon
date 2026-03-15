use super::*;

pub(super) fn datafusion_filters_to_table_format_predicate(filters: &[Expr]) -> Option<TableFormatPredicate> {
    filters
        .iter()
        .filter_map(datafusion_filter_to_table_format_predicate)
        .reduce(|left, right| TableFormatPredicate::And(Box::new(left), Box::new(right)))
}

#[cfg(test)]
pub(super) fn datafusion_filters_to_iceberg_predicate(filters: &[Expr]) -> Option<iceberg::expr::Predicate> {
    datafusion_filters_to_table_format_predicate(filters).as_ref().and_then(table_format_predicate_to_iceberg_predicate)
}

pub(super) fn datafusion_filter_to_table_format_predicate(expr: &Expr) -> Option<TableFormatPredicate> {
    datafusion_filter_to_table_format_predicate_with_partial_and(expr, true)
}

pub(super) fn datafusion_filter_to_table_format_predicate_exact(expr: &Expr) -> Option<TableFormatPredicate> {
    datafusion_filter_to_table_format_predicate_with_partial_and(expr, false)
}

pub(super) fn datafusion_filter_to_table_format_predicate_with_partial_and(
    expr: &Expr,
    allow_partial_and: bool,
) -> Option<TableFormatPredicate> {
    match expr {
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And => match (
                datafusion_filter_to_table_format_predicate_with_partial_and(&binary.left, allow_partial_and),
                datafusion_filter_to_table_format_predicate_with_partial_and(&binary.right, allow_partial_and),
            ) {
                (Some(left), Some(right)) => Some(TableFormatPredicate::And(Box::new(left), Box::new(right))),
                (Some(left), None) if allow_partial_and => Some(left),
                (None, Some(right)) if allow_partial_and => Some(right),
                (None, None) => None,
                (Some(_), None) | (None, Some(_)) => None,
            },
            Operator::Or => Some(TableFormatPredicate::Or(
                Box::new(datafusion_filter_to_table_format_predicate_with_partial_and(
                    &binary.left,
                    allow_partial_and,
                )?),
                Box::new(datafusion_filter_to_table_format_predicate_with_partial_and(
                    &binary.right,
                    allow_partial_and,
                )?),
            )),
            Operator::Eq | Operator::NotEq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                datafusion_comparison_to_table_format_predicate(&binary.left, binary.op, &binary.right).or_else(|| {
                    datafusion_comparison_to_table_format_predicate(
                        &binary.right,
                        reverse_operator(binary.op)?,
                        &binary.left,
                    )
                })
            },
            _ => None,
        },
        Expr::Not(expr) => datafusion_filter_to_table_format_predicate_exact(expr)
            .map(|predicate| TableFormatPredicate::Not(Box::new(predicate))),
        Expr::IsNull(expr) => datafusion_column_reference(expr).map(TableFormatPredicate::IsNull),
        Expr::IsNotNull(expr) => datafusion_column_reference(expr).map(TableFormatPredicate::IsNotNull),
        Expr::Column(column) => Some(TableFormatPredicate::BooleanColumn(column.name().to_string())),
        _ => None,
    }
}

pub(super) fn datafusion_comparison_to_table_format_predicate(
    left: &Expr,
    operator: Operator,
    right: &Expr,
) -> Option<TableFormatPredicate> {
    let column = datafusion_column_reference(left)?;
    let literal = datafusion_literal_to_table_format_literal(right)?;
    let op = match operator {
        Operator::Eq => TableFormatPredicateOp::Eq,
        Operator::NotEq => TableFormatPredicateOp::NotEq,
        Operator::Lt => TableFormatPredicateOp::Lt,
        Operator::LtEq => TableFormatPredicateOp::LtEq,
        Operator::Gt => TableFormatPredicateOp::Gt,
        Operator::GtEq => TableFormatPredicateOp::GtEq,
        _ => return None,
    };
    Some(TableFormatPredicate::Comparison { column, op, literal })
}

pub(super) fn table_format_predicate_to_iceberg_predicate(
    predicate: &TableFormatPredicate,
) -> Option<iceberg::expr::Predicate> {
    match predicate {
        TableFormatPredicate::And(left, right) => Some(
            table_format_predicate_to_iceberg_predicate(left)?.and(table_format_predicate_to_iceberg_predicate(right)?),
        ),
        TableFormatPredicate::Or(left, right) => Some(
            table_format_predicate_to_iceberg_predicate(left)?.or(table_format_predicate_to_iceberg_predicate(right)?),
        ),
        TableFormatPredicate::Not(predicate) => {
            table_format_predicate_to_iceberg_predicate(predicate).map(std::ops::Not::not)
        },
        TableFormatPredicate::BooleanColumn(column) => Some(Reference::new(column).equal_to(Datum::bool(true))),
        TableFormatPredicate::IsNull(column) => Some(Reference::new(column).is_null()),
        TableFormatPredicate::IsNotNull(column) => Some(Reference::new(column).is_not_null()),
        TableFormatPredicate::Comparison { column, op, literal } => {
            table_format_comparison_to_iceberg_predicate(column, *op, literal)
        },
    }
}

pub(super) fn table_format_comparison_to_iceberg_predicate(
    column: &str,
    operator: TableFormatPredicateOp,
    literal: &TableFormatLiteral,
) -> Option<iceberg::expr::Predicate> {
    let reference = Reference::new(column);
    let datum = table_format_literal_to_iceberg_datum(literal);
    match operator {
        TableFormatPredicateOp::Eq => Some(reference.equal_to(datum)),
        TableFormatPredicateOp::NotEq => Some(reference.not_equal_to(datum)),
        TableFormatPredicateOp::Lt => Some(reference.less_than(datum)),
        TableFormatPredicateOp::LtEq => Some(reference.less_than_or_equal_to(datum)),
        TableFormatPredicateOp::Gt => Some(reference.greater_than(datum)),
        TableFormatPredicateOp::GtEq => Some(reference.greater_than_or_equal_to(datum)),
    }
}

pub(super) fn datafusion_column_reference(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => Some(column.name().to_string()),
        _ => None,
    }
}

pub(super) fn datafusion_literal_to_table_format_literal(expr: &Expr) -> Option<TableFormatLiteral> {
    match expr {
        Expr::Literal(value, _) => scalar_value_to_table_format_literal(value),
        _ => None,
    }
}

pub(super) fn scalar_value_to_table_format_literal(value: &ScalarValue) -> Option<TableFormatLiteral> {
    match value {
        ScalarValue::Boolean(Some(value)) => Some(TableFormatLiteral::Boolean(*value)),
        ScalarValue::Int8(Some(value)) => Some(TableFormatLiteral::Int32(i32::from(*value))),
        ScalarValue::Int16(Some(value)) => Some(TableFormatLiteral::Int32(i32::from(*value))),
        ScalarValue::Int32(Some(value)) => Some(TableFormatLiteral::Int32(*value)),
        ScalarValue::Int64(Some(value)) => Some(TableFormatLiteral::Int64(*value)),
        ScalarValue::Float32(Some(value)) => Some(TableFormatLiteral::Float32(*value)),
        ScalarValue::Float64(Some(value)) => Some(TableFormatLiteral::Float64(*value)),
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Some(TableFormatLiteral::Utf8(value.clone()))
        },
        ScalarValue::Date32(Some(value)) => Some(TableFormatLiteral::Date32(*value)),
        _ => None,
    }
}

pub(super) fn table_format_literal_to_iceberg_datum(literal: &TableFormatLiteral) -> Datum {
    match literal {
        TableFormatLiteral::Boolean(value) => Datum::bool(*value),
        TableFormatLiteral::Int32(value) => Datum::int(*value),
        TableFormatLiteral::Int64(value) => Datum::long(*value),
        TableFormatLiteral::Float32(value) => Datum::float(*value),
        TableFormatLiteral::Float64(value) => Datum::double(*value),
        TableFormatLiteral::Utf8(value) => Datum::string(value),
        TableFormatLiteral::Date32(value) => Datum::date(*value),
    }
}

pub(super) fn reverse_operator(operator: Operator) -> Option<Operator> {
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
