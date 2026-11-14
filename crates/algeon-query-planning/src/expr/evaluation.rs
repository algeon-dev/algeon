//! Result types and failure behavior of native expressions.
//!
//! A rewrite that evaluates an expression on rows the original plan did not
//! evaluate (before a join instead of after it, below a filter, on a widened
//! shared selection) changes which rows can fail the query. [`Expr::can_fail`]
//! answers whether that matters for an expression, following the checked
//! operations table in
//! `website/docs/reference/native-evaluation-semantics.mdx#checked-operations`.
//! A guard inside the expression does not make it safe to move, because the
//! guard can be true on the added rows.

use arrow::datatypes::{DataType, Schema};

use super::{BinaryOp, ColumnQualifier, ColumnRef, Expr, LiteralValue};
use crate::{
    datatypes::{cast_can_fail_at_runtime, temporal_arithmetic_can_fail_at_runtime},
    validation::is_integral_type,
};

impl Expr {
    /// The Arrow type this expression evaluates to, with each column typed by
    /// `column_type`.
    ///
    /// `None` when a column is unresolved, and for `SUBSTRING`, `CONCAT` and
    /// `CASE`, whose result type the IR does not record.
    pub fn data_type(&self, column_type: &dyn Fn(&ColumnRef) -> Option<DataType>) -> Option<DataType> {
        match self {
            Self::Column(column) => column_type(column),
            Self::Literal(value) => Some(value.data_type()),
            Self::Alias { expr, .. }
            | Self::Lower(expr)
            | Self::Upper(expr)
            | Self::Trim { expr, .. }
            | Self::LTrim { expr, .. }
            | Self::RTrim { expr, .. }
            | Self::Abs(expr) => expr.data_type(column_type),
            Self::Cast { target_type, .. } | Self::TryCast { target_type, .. } => Some(target_type.clone()),
            Self::Binary { left, op, right, output_type } => {
                binary_data_type(left, *op, right, output_type.as_ref(), column_type)
            },
            Self::Not(_)
            | Self::IsNull(_)
            | Self::IsNotNull(_)
            | Self::Like { .. }
            | Self::InList { .. }
            | Self::StringContains { .. }
            | Self::StartsWith { .. }
            | Self::EndsWith { .. }
            | Self::IsDistinctFrom { .. }
            | Self::IsNotDistinctFrom { .. } => Some(DataType::Boolean),
            Self::StringLength { output_type, .. }
            | Self::RegexpReplace { output_type, .. }
            | Self::UnaryMath { output_type, .. }
            | Self::Round { output_type, .. }
            | Self::DateTrunc { output_type, .. } => Some(output_type.clone()),
            Self::Extract { .. } => Some(DataType::Int32),
            Self::Substring { .. } | Self::Concat(_) | Self::Case { .. } => None,
        }
    }

    /// Whether native evaluation of this expression can fail the query on
    /// some row, with each column typed by `column_type`.
    ///
    /// An expression that cannot fail may be evaluated on rows the original
    /// plan never evaluated without changing whether the query succeeds. An
    /// operation whose operand types cannot be determined counts as one that
    /// can fail.
    pub fn can_fail(&self, column_type: &dyn Fn(&ColumnRef) -> Option<DataType>) -> bool {
        self.any(|expr| expr.operation_can_fail(column_type))
    }

    /// Whether this node's own operation, apart from its operands, can fail.
    fn operation_can_fail(&self, column_type: &dyn Fn(&ColumnRef) -> Option<DataType>) -> bool {
        match self {
            Self::Cast { expr, target_type } => {
                expr.data_type(column_type).is_none_or(|source_type| cast_can_fail_at_runtime(&source_type, target_type))
            },
            Self::Binary { left, op: BinaryOp::Divide | BinaryOp::Modulo, right, output_type } => {
                if matches!(output_type, Some(DataType::Decimal128(_, _))) {
                    return true;
                }
                match (left.data_type(column_type), right.data_type(column_type)) {
                    (Some(dividend), Some(divisor)) if dividend.is_floating() && divisor.is_floating() => false,
                    (Some(dividend), Some(divisor)) if is_integral_type(&dividend) && is_integral_type(&divisor) => {
                        !is_infallible_integral_divisor(right)
                    },
                    // Decimal `/` and `%`, and operands of unknown type.
                    _ => true,
                }
            },
            Self::Binary { left, op: op @ (BinaryOp::Add | BinaryOp::Subtract), right, .. } => {
                match (left.data_type(column_type), right.data_type(column_type)) {
                    (Some(left_type), Some(right_type)) => {
                        temporal_arithmetic_can_fail_at_runtime(*op, &left_type, &right_type)
                    },
                    _ => true,
                }
            },
            _ => false,
        }
    }
}

/// The type of `column` among the input schemas of the node that evaluates
/// it: an input-qualified column in its input's schema, an unqualified column
/// in the only input's schema.
pub(crate) fn input_column_type(inputs: &[&Schema], column: &ColumnRef) -> Option<DataType> {
    let schema = match column.qualifier() {
        Some(ColumnQualifier::Input { index }) => inputs.get(*index)?,
        None => match inputs {
            [schema] => schema,
            _ => return None,
        },
        Some(ColumnQualifier::Node { .. } | ColumnQualifier::Relation { .. }) => return None,
    };
    Some(schema.field_with_name(column.name()).ok()?.data_type().clone())
}

/// Result type of a binary node: its declared output type, `Boolean` for a
/// comparison or logical operator, a date or timestamp for interval
/// arithmetic, and otherwise the left operand's type, which lowering has
/// already coerced both operands to.
fn binary_data_type(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&DataType>,
    column_type: &dyn Fn(&ColumnRef) -> Option<DataType>,
) -> Option<DataType> {
    if let Some(output_type) = output_type {
        return Some(output_type.clone());
    }
    if matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::And
            | BinaryOp::Or
    ) {
        return Some(DataType::Boolean);
    }
    let left_interval = is_interval_literal(left);
    let right_interval = is_interval_literal(right);
    let left_type = (!left_interval).then(|| left.data_type(column_type)).flatten();
    let right_type = (!right_interval).then(|| right.data_type(column_type)).flatten();
    match (op, left_interval, right_interval, left_type, right_type) {
        (BinaryOp::Add | BinaryOp::Subtract, false, true, Some(data_type), _) if is_date_or_timestamp(&data_type) => {
            Some(data_type)
        },
        (BinaryOp::Add, true, false, _, Some(data_type)) if is_date_or_timestamp(&data_type) => Some(data_type),
        (_, _, _, Some(data_type), _) => Some(data_type),
        _ => None,
    }
}

fn is_interval_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::IntervalMonthDayNano { .. }))
}

fn is_date_or_timestamp(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Date32 | DataType::Timestamp(_, None))
}

/// An integer literal divisor other than `0` and `-1`, which no dividend
/// divides into a failure.
fn is_infallible_integral_divisor(divisor: &Expr) -> bool {
    let Expr::Literal(literal) = divisor else {
        return false;
    };
    let value = match literal {
        LiteralValue::Int8(value) => i128::from(*value),
        LiteralValue::Int16(value) => i128::from(*value),
        LiteralValue::Int32(value) => i128::from(*value),
        LiteralValue::Int64(value) => i128::from(*value),
        LiteralValue::UInt8(value) => i128::from(*value),
        LiteralValue::UInt16(value) => i128::from(*value),
        LiteralValue::UInt32(value) => i128::from(*value),
        LiteralValue::UInt64(value) => i128::from(*value),
        _ => return false,
    };
    value != 0 && value != -1
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    use crate::expr::{BinaryOp, CaseWhen, ColumnRef, Expr, LiteralValue, input_column_type};

    fn can_fail(expr: &Expr) -> bool {
        let schema = Schema::new(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("f", DataType::Float64, true),
            Field::new("t", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        ]);
        expr.can_fail(&|column: &ColumnRef| input_column_type(&[&schema], column))
    }

    #[test]
    fn can_fail_reads_operand_types_and_ignores_guards() {
        let divide = |divisor: Expr| Expr::binary(Expr::column("i"), BinaryOp::Divide, divisor);
        assert!(!can_fail(&divide(Expr::literal(LiteralValue::Int32(2)))));
        assert!(can_fail(&divide(Expr::literal(LiteralValue::Int32(-1)))));
        assert!(!can_fail(&Expr::binary(Expr::column("f"), BinaryOp::Divide, Expr::column("f"))));

        // A guard limits the rows DataFusion evaluates the division on, and a
        // rewrite that adds rows can add rows where the guard holds.
        let guarded = Expr::case(
            vec![CaseWhen::new(
                Expr::binary(Expr::column("i"), BinaryOp::NotEq, Expr::literal(LiteralValue::Int32(0))),
                divide(Expr::column("i")),
            )],
            Expr::literal(LiteralValue::Null(DataType::Int32)),
        );
        assert!(can_fail(&guarded));

        assert!(!can_fail(&Expr::cast(Expr::column("i"), DataType::Int64)));
        assert!(can_fail(&Expr::cast(Expr::column("f"), DataType::Int64)));
        assert!(can_fail(&Expr::binary(Expr::column("t"), BinaryOp::Subtract, Expr::column("t"))));
        assert!(!can_fail(&Expr::binary(Expr::column("i"), BinaryOp::Add, Expr::column("i"))));

        // An operand of unknown type counts as one that can fail.
        assert!(can_fail(&Expr::cast(Expr::column("missing"), DataType::Int64)));
        assert!(can_fail(&Expr::binary(Expr::column("missing"), BinaryOp::Add, Expr::column("i"))));
    }
}
