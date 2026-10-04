use crate::expr::{Expr, LiteralValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderPredicateDecimalScaleError {
    ScaleFactorOverflow { exponent: u32 },
    IntegerRawOverflow { target_scale: i32 },
    IntegerInexact { target_scale: i32 },
    ScaleAlignmentOverflow { source_scale: i32, target_scale: i32 },
    DecimalRawOverflow { source_scale: i32, target_scale: i32 },
    DecimalInexact { source_scale: i32, target_scale: i32 },
}

pub fn reader_predicate_integer_literal_as<T>(value: &LiteralValue) -> Option<T>
where T: TryFrom<i128> {
    reader_predicate_integer_literal_raw(value).and_then(|value| T::try_from(value).ok())
}

pub fn reader_predicate_unsigned_literal_as<T>(value: &LiteralValue) -> Option<T>
where T: TryFrom<u128> {
    reader_predicate_unsigned_literal_raw(value).and_then(|value| T::try_from(value).ok())
}

pub(crate) fn reader_predicate_unsigned_literal_raw(value: &LiteralValue) -> Option<u128> {
    match value {
        LiteralValue::UInt8(value) => Some(u128::from(*value)),
        LiteralValue::UInt16(value) => Some(u128::from(*value)),
        LiteralValue::UInt32(value) => Some(u128::from(*value)),
        LiteralValue::UInt64(value) => Some(u128::from(*value)),
        LiteralValue::Int8(value) => u128::try_from(*value).ok(),
        LiteralValue::Int16(value) => u128::try_from(*value).ok(),
        LiteralValue::Int32(value) => u128::try_from(*value).ok(),
        LiteralValue::Int64(value) => u128::try_from(*value).ok(),
        _ => None,
    }
}

pub(crate) fn reader_predicate_integer_literal_raw(value: &LiteralValue) -> Option<i128> {
    match value {
        LiteralValue::Int8(value) => Some(i128::from(*value)),
        LiteralValue::Int16(value) => Some(i128::from(*value)),
        LiteralValue::Int32(value) => Some(i128::from(*value)),
        LiteralValue::Int64(value) => Some(i128::from(*value)),
        LiteralValue::UInt8(value) => Some(i128::from(*value)),
        LiteralValue::UInt16(value) => Some(i128::from(*value)),
        LiteralValue::UInt32(value) => Some(i128::from(*value)),
        LiteralValue::UInt64(value) => Some(i128::from(*value)),
        _ => None,
    }
}

pub fn reader_predicate_integer_literal_as_decimal_raw(
    value: &LiteralValue,
    target_scale: i32,
) -> Result<Option<i128>, ReaderPredicateDecimalScaleError> {
    let Some(raw) = reader_predicate_integer_literal_raw(value) else {
        return Ok(None);
    };
    reader_predicate_decimal_raw_for_integer_value(raw, target_scale).map(Some)
}

pub(crate) fn reader_predicate_decimal_raw_for_integer_value(
    value: i128,
    target_scale: i32,
) -> Result<i128, ReaderPredicateDecimalScaleError> {
    if target_scale <= 0 {
        let multiplier = reader_predicate_checked_pow10(target_scale.unsigned_abs())?;
        value.checked_mul(multiplier).ok_or(ReaderPredicateDecimalScaleError::IntegerRawOverflow { target_scale })
    } else {
        let divisor = reader_predicate_checked_pow10(target_scale as u32)?;
        if value % divisor != 0 {
            return Err(ReaderPredicateDecimalScaleError::IntegerInexact { target_scale });
        }
        Ok(value / divisor)
    }
}

pub fn reader_predicate_rescale_decimal_raw(
    raw: i128,
    source_scale: i32,
    target_scale: i32,
) -> Result<i128, ReaderPredicateDecimalScaleError> {
    let scale_delta = source_scale
        .checked_sub(target_scale)
        .ok_or(ReaderPredicateDecimalScaleError::ScaleAlignmentOverflow { source_scale, target_scale })?;
    if scale_delta >= 0 {
        let multiplier = reader_predicate_checked_pow10(scale_delta as u32)?;
        raw.checked_mul(multiplier)
            .ok_or(ReaderPredicateDecimalScaleError::DecimalRawOverflow { source_scale, target_scale })
    } else {
        let divisor = reader_predicate_checked_pow10(scale_delta.unsigned_abs())?;
        if raw % divisor != 0 {
            return Err(ReaderPredicateDecimalScaleError::DecimalInexact { source_scale, target_scale });
        }
        Ok(raw / divisor)
    }
}

fn reader_predicate_checked_pow10(exponent: u32) -> Result<i128, ReaderPredicateDecimalScaleError> {
    let mut value = 1i128;
    for _ in 0..exponent {
        value = value.checked_mul(10).ok_or(ReaderPredicateDecimalScaleError::ScaleFactorOverflow { exponent })?;
    }
    Ok(value)
}

pub fn reader_predicate_expr_references_decimal_column(
    expr: &Expr,
    mut column_is_decimal: impl FnMut(&str) -> bool,
) -> bool {
    fn visit(expr: &Expr, column_is_decimal: &mut impl FnMut(&str) -> bool) -> bool {
        match expr {
            Expr::Column(column) => column_is_decimal(column.name()),
            Expr::Literal(_) => false,
            Expr::Not(input)
            | Expr::IsNull(input)
            | Expr::IsNotNull(input)
            | Expr::Alias { expr: input, .. }
            | Expr::Cast { expr: input, .. }
            | Expr::TryCast { expr: input, .. } => visit(input, column_is_decimal),
            Expr::Binary { left, right, .. } => visit(left, column_is_decimal) || visit(right, column_is_decimal),
            Expr::InList { expr, list, .. } => {
                visit(expr, column_is_decimal) || list.iter().any(|value| visit(value, column_is_decimal))
            },
            Expr::Like { expr, .. }
            | Expr::Substring { expr, .. }
            | Expr::Lower(expr)
            | Expr::Upper(expr)
            | Expr::Trim { expr, .. }
            | Expr::LTrim { expr, .. }
            | Expr::RTrim { expr, .. }
            | Expr::StringContains { expr, .. }
            | Expr::StartsWith { expr, .. }
            | Expr::EndsWith { expr, .. }
            | Expr::StringLength { expr, .. }
            | Expr::RegexpReplace { expr, .. }
            | Expr::Abs(expr)
            | Expr::UnaryMath { expr, .. }
            | Expr::Round { expr, .. }
            | Expr::Extract { expr, .. }
            | Expr::DateTrunc { expr, .. } => visit(expr, column_is_decimal),
            Expr::Concat(exprs) => exprs.iter().any(|expr| visit(expr, column_is_decimal)),
            Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
                visit(left, column_is_decimal) || visit(right, column_is_decimal)
            },
            Expr::Case { branches, else_expr } => {
                branches
                    .iter()
                    .any(|branch| visit(&branch.when, column_is_decimal) || visit(&branch.then, column_is_decimal))
                    || visit(else_expr, column_is_decimal)
            },
        }
    }

    visit(expr, &mut column_is_decimal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{BinaryOp, CaseWhen};

    #[test]
    fn decimal_integer_scaling_reports_exact_failure_kind() {
        assert_eq!(reader_predicate_decimal_raw_for_integer_value(100, -2), Ok(10000));
        assert_eq!(
            reader_predicate_decimal_raw_for_integer_value(101, 2),
            Err(ReaderPredicateDecimalScaleError::IntegerInexact { target_scale: 2 })
        );
        assert_eq!(
            reader_predicate_decimal_raw_for_integer_value(i128::MAX, -1),
            Err(ReaderPredicateDecimalScaleError::IntegerRawOverflow { target_scale: -1 })
        );
    }

    #[test]
    fn decimal_rescale_reports_exact_failure_kind() {
        assert_eq!(reader_predicate_rescale_decimal_raw(100, -2, -4), Ok(10000));
        assert_eq!(
            reader_predicate_rescale_decimal_raw(101, -4, -2),
            Err(ReaderPredicateDecimalScaleError::DecimalInexact { source_scale: -4, target_scale: -2 })
        );
        assert_eq!(
            reader_predicate_rescale_decimal_raw(i128::MAX, -1, -2),
            Err(ReaderPredicateDecimalScaleError::DecimalRawOverflow { source_scale: -1, target_scale: -2 })
        );
    }

    #[test]
    fn decimal_reference_walks_nested_reader_predicate_expressions() {
        let expr = Expr::binary(
            Expr::lower(Expr::column("name")),
            BinaryOp::Eq,
            Expr::case(
                vec![CaseWhen {
                    when: Expr::is_not_null(Expr::column("amount")),
                    then: Expr::literal(LiteralValue::Utf8("x".to_string())),
                }],
                Expr::literal(LiteralValue::Utf8("y".to_string())),
            ),
        );

        assert!(reader_predicate_expr_references_decimal_column(&expr, |name| name == "amount"));
        assert!(!reader_predicate_expr_references_decimal_column(&expr, |name| name == "missing"));
    }
}
