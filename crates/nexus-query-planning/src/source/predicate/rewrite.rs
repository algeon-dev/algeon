//! Rewrites accepted high-level predicate shapes to the reader expression contract.

use arrow::datatypes::DataType;

use crate::expr::{BinaryOp, Expr, LiteralValue};

pub(in crate::source) fn rewrite_fixed_prefix_substring_reader_predicate(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Binary { left, op, right, output_type } => {
            let left = rewrite_fixed_prefix_substring_reader_predicate(left).unwrap_or_else(|| (**left).clone());
            let right = rewrite_fixed_prefix_substring_reader_predicate(right).unwrap_or_else(|| (**right).clone());
            let rewritten = Expr::Binary {
                left: Box::new(left),
                op: *op,
                right: Box::new(right),
                output_type: output_type.clone(),
            };
            (&rewritten != expr).then_some(rewritten)
        },
        Expr::Not(input) => rewrite_fixed_prefix_substring_reader_predicate(input).map(Expr::logical_not),
        Expr::Alias { expr: input, alias } => rewrite_fixed_prefix_substring_reader_predicate(input)
            .map(|expr| Expr::Alias { expr: Box::new(expr), alias: alias.clone() }),
        Expr::InList { expr: input, list, negated } => rewrite_fixed_prefix_substring_in_list(input, list, *negated),
        _ => None,
    }
}

fn rewrite_fixed_prefix_substring_in_list(expr: &Expr, list: &[Expr], negated: bool) -> Option<Expr> {
    let Expr::Substring { expr: input, start: 1, length: Some(length) } = expr else {
        return None;
    };
    let length = usize::try_from(*length).ok()?;
    if length == 0 || list.is_empty() {
        return None;
    }

    let mut ranges = Vec::with_capacity(list.len());
    for value in list {
        let prefix = fixed_ascii_prefix_literal(value, length)?;
        ranges.push(fixed_ascii_prefix_range((**input).clone(), prefix)?);
    }

    let mut ranges = ranges.into_iter();
    let first = ranges.next()?;
    let combined =
        ranges.fold(first, |left, right| Expr::binary_with_output_type(left, BinaryOp::Or, right, DataType::Boolean));
    Some(if negated { Expr::logical_not(combined) } else { combined })
}

fn fixed_ascii_prefix_literal(expr: &Expr, length: usize) -> Option<&str> {
    let Expr::Literal(LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value)) = expr else {
        return None;
    };
    (value.len() == length && value.is_ascii()).then_some(value.as_str())
}

fn fixed_ascii_prefix_range(column: Expr, prefix: &str) -> Option<Expr> {
    let upper = ascii_prefix_successor(prefix)?;
    let lower_bound = Expr::binary_with_output_type(
        column.clone(),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Utf8(prefix.to_string())),
        DataType::Boolean,
    );
    let upper_bound = Expr::binary_with_output_type(
        column,
        BinaryOp::Lt,
        Expr::literal(LiteralValue::Utf8(upper)),
        DataType::Boolean,
    );
    Some(Expr::binary_with_output_type(lower_bound, BinaryOp::And, upper_bound, DataType::Boolean))
}

fn ascii_prefix_successor(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
    let last = bytes.last_mut()?;
    if !last.is_ascii() || *last == 0x7f {
        return None;
    }
    *last += 1;
    String::from_utf8(bytes).ok()
}
