//! Expression type checking for single-input nodes.
//!
//! [`validate_expr`] returns the Arrow type an expression evaluates to, or the
//! first contract it misses. Types are derived from the IR and the input
//! schema; the backend is asked only whether it represents a type, supports a
//! cast, or infers a decimal result. A column is resolved against two facts:
//! `available_columns` is what the input produces at execution time and the
//! schema supplies the type, and the two differ for a local Parquet source
//! whose execution frame carries fewer columns than its declared schema. Join
//! residuals have their own checker in `join`, because the cuDF AST that
//! evaluates them admits fewer forms than the column-wise kernels admitted
//! here.

use std::collections::BTreeSet;

use arrow::datatypes::{DataType as ArrowDataType, SchemaRef};

use super::{
    BinaryOp, Expr, LiteralValue, QueryError, QueryErrorCode, Result,
    types::{
        is_decimal_type, is_integral_type, require_arithmetic_type, require_boolean, require_comparable_types,
        require_compatible_types, require_date_or_timestamp, require_integral_type, require_remainder_type,
        require_scalar_operator_value, require_string, require_timestamp_preserving_output, require_timestamp_second,
        temporal_duration_binary_output_type,
    },
};
use crate::{
    backend::AnalysisContext,
    datatypes::is_supported_arrow_cast,
    expr::{CaseWhen, DatePart, UnaryMathOp},
    plan::ProjectionOutput,
};

pub(crate) fn validate_expr(
    expr: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => validate_column(column.name(), input_schema, available_columns, ctx),
        Expr::Literal(value) => validate_literal(value, ctx),
        Expr::Not(expr) => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_boolean(&data_type, "NOT argument")?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::IsNull(expr) | Expr::IsNotNull(expr) => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_scalar_operator_value(&data_type, "IS NULL input")?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::Cast { expr, target_type } => validate_cast_expr(expr, target_type, input_schema, available_columns, ctx),
        Expr::TryCast { expr, target_type } => {
            validate_try_cast_expr(expr, target_type, input_schema, available_columns, ctx)
        },
        Expr::Binary { left, op, right, output_type } => {
            validate_binary_expr(left, *op, right, output_type.as_ref(), input_schema, available_columns, ctx)
        },
        Expr::Like { expr, escape, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_string(&data_type, "LIKE input")?;
            if let Some(escape) = escape {
                validate_like_escape_contract(escape)?;
            }
            Ok(ArrowDataType::Boolean)
        },
        Expr::Substring { expr, start, length } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_string(&data_type, "SUBSTRING input")?;
            validate_substring_bounds(*start, *length)?;
            Ok(ArrowDataType::Utf8)
        },
        Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Trim { expr, .. }
        | Expr::LTrim { expr, .. }
        | Expr::RTrim { expr, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_string(&data_type, "string case input")?;
            Ok(data_type)
        },
        Expr::StringContains { expr, .. } | Expr::StartsWith { expr, .. } | Expr::EndsWith { expr, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_string(&data_type, "string predicate input")?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::StringLength { expr, output_type } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_string(&data_type, "string length input")?;
            require_integral_type(output_type, "string length output")?;
            Ok(output_type.clone())
        },
        Expr::RegexpReplace { expr, output_type, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_string(&data_type, "REGEXP_REPLACE input")?;
            require_string(output_type, "REGEXP_REPLACE output")?;
            Ok(output_type.clone())
        },
        Expr::Concat(exprs) => validate_concat_expr(exprs, input_schema, available_columns, ctx),
        Expr::Abs(expr) => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_arithmetic_type(&data_type, "ABS input")?;
            Ok(data_type)
        },
        Expr::UnaryMath { op, expr, output_type } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            validate_unary_math_types(*op, &data_type, output_type)?;
            let (input_context, output_context) = unary_math_type_contexts(*op);
            require_arithmetic_type(&data_type, input_context)?;
            require_arithmetic_type(output_type, output_context)?;
            if is_decimal_type(&data_type) != is_decimal_type(output_type) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "query_engine.plan.validate_expr",
                    format!("native {op} input and output must both be decimal or both be non-decimal"),
                )
                .with_fact("input_type", format!("{data_type:?}"))
                .with_fact("output_type", format!("{output_type:?}")));
            }
            Ok(output_type.clone())
        },
        Expr::Round { expr, output_type, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            require_arithmetic_type(&data_type, "ROUND input")?;
            require_arithmetic_type(output_type, "ROUND output")?;
            if is_decimal_type(&data_type) != is_decimal_type(output_type) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "query_engine.plan.validate_expr",
                    "native ROUND input and output must both be decimal or both be non-decimal",
                )
                .with_fact("input_type", format!("{data_type:?}"))
                .with_fact("output_type", format!("{output_type:?}")));
            }
            Ok(output_type.clone())
        },
        Expr::InList { expr, list, .. } => validate_in_list_expr(expr, list, input_schema, available_columns, ctx),
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            let left_type = validate_expr(left, input_schema, available_columns, ctx)?;
            let right_type = validate_expr(right, input_schema, available_columns, ctx)?;
            require_comparable_types(&left_type, &right_type)?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::Case { branches, else_expr } => {
            validate_case_expr(branches, else_expr, input_schema, available_columns, ctx)
        },
        Expr::Extract { field, expr } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            if !DatePart::EXTRACT_FIELDS.contains(field) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "query_engine.plan.validate_expr",
                    "native EXTRACT does not support this field",
                )
                .with_fact("field", field.to_string()));
            }
            if *field == DatePart::Minute {
                validate_timestamp_second_integer_cast_expr(expr, input_schema, available_columns, ctx)?;
            } else {
                require_date_or_timestamp(&data_type, "EXTRACT input")?;
            }
            Ok(ArrowDataType::Int32)
        },
        Expr::DateTrunc { field, expr, output_type } => {
            let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
            if !DatePart::DATE_TRUNC_FIELDS.contains(field) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "query_engine.plan.validate_expr",
                    "native DATE_TRUNC does not support this field",
                )
                .with_fact("field", field.to_string()));
            }
            require_timestamp_preserving_output(&data_type, output_type, "DATE_TRUNC input/output")?;
            Ok(output_type.clone())
        },
        Expr::Alias { expr, .. } => validate_expr(expr, input_schema, available_columns, ctx),
    }
}

/// `TRY_CAST` runs natively only from a string to an integer, the one pair the
/// engine implements with null-on-failure semantics by masking rows that fail
/// the integer parse check.
fn validate_try_cast_expr(
    input: &Expr,
    target_type: &ArrowDataType,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    let source_type = validate_expr(input, input_schema, available_columns, ctx)?;
    if !is_string_arrow_type(&source_type) || !is_integral_type(target_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "native TRY_CAST currently supports only string -> integer targets",
        )
        .with_fact("source_type", format!("{source_type:?}"))
        .with_fact("target_type", format!("{target_type:?}")));
    }
    Ok(target_type.clone())
}

fn is_string_arrow_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

/// The cast target doubles as the expected type of a nested decimal division,
/// so `CAST(a / b AS DECIMAL(p, s))` declares the division's result scale.
fn validate_cast_expr(
    expr: &Expr,
    target_type: &ArrowDataType,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    let source_type = validate_expr_with_expected_type(expr, Some(target_type), input_schema, available_columns, ctx)?;
    if !is_supported_arrow_cast(&source_type, target_type, ctx.backend()) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "native cast source and target types are not supported",
        )
        .with_fact("source_type", format!("{source_type:?}"))
        .with_fact("target_type", format!("{target_type:?}")));
    }
    Ok(target_type.clone())
}

/// `EXTRACT(MINUTE)` runs natively only over a timezone-free second timestamp
/// cast from an integer; see `DatePart::EXTRACT_FIELDS`.
fn validate_timestamp_second_integer_cast_expr(
    expr: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<()> {
    let Expr::Cast { expr, target_type } = expr else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native second timestamp minute operations currently require an integer-to-timestamp cast",
        )
        .with_fact("expression", expr));
    };
    require_timestamp_second(target_type, "timestamp cast target")?;
    let source_type = validate_expr(expr, input_schema, available_columns, ctx)?;
    require_integral_type(&source_type, "timestamp cast source")
}

pub(crate) fn validate_projection_output_expr(
    output: &ProjectionOutput,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    validate_expr_with_expected_type(&output.expr, Some(output.field.data_type()), input_schema, available_columns, ctx)
}

/// [`validate_expr`] with the type a projection field or enclosing cast
/// declares for the expression. Only decimal division consumes the hint,
/// because its result scale is a frontend contract the backend cannot infer;
/// the hint passes through aliases and casts and is dropped everywhere else.
fn validate_expr_with_expected_type(
    expr: &Expr,
    expected_type: Option<&ArrowDataType>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    match expr {
        Expr::Cast { expr, target_type } => validate_cast_expr(expr, target_type, input_schema, available_columns, ctx),
        Expr::TryCast { expr, target_type } => {
            validate_try_cast_expr(expr, target_type, input_schema, available_columns, ctx)
        },
        Expr::Binary { left, op, right, output_type } => validate_binary_expr_with_expected_type(
            left,
            *op,
            right,
            output_type.as_ref(),
            expected_type,
            input_schema,
            available_columns,
            ctx,
        ),
        Expr::Alias { expr, .. } => {
            validate_expr_with_expected_type(expr, expected_type, input_schema, available_columns, ctx)
        },
        _ => validate_expr(expr, input_schema, available_columns, ctx),
    }
}

/// Availability is checked before the schema lookup, so a column the schema
/// declares but the input does not produce reports the availability error,
/// which lists what the input does produce.
pub(crate) fn validate_column(
    name: &str,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    if !available_columns.contains(name) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.validate_expr",
            "native expression references a column that is not produced by its input",
        )
        .with_fact("column", name)
        .with_fact("available_columns", comma_separated(available_columns)));
    }
    let field = input_schema.field_with_name(name).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.validate_expr",
            "native expression references a column missing from its input schema",
        )
        .with_fact("column", name)
        .with_fact("arrow_error", error.to_string())
    })?;
    if !ctx.backend().supports_type(field.data_type()) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "native expression references an unsupported column type",
        )
        .with_fact("column", name)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(field.data_type().clone())
}

pub(crate) fn validate_literal(value: &LiteralValue, ctx: AnalysisContext<'_>) -> Result<ArrowDataType> {
    if matches!(value, LiteralValue::IntervalMonthDayNano { .. }) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "interval literals are supported only inside datetime interval arithmetic",
        )
        .with_fact("literal", value));
    }
    let data_type = value.data_type();
    if !ctx.backend().supports_type(&data_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "literal type is not supported",
        )
        .with_fact("literal", value)
        .with_fact("data_type", format!("{data_type:?}")));
    }
    Ok(data_type)
}

fn validate_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    validate_binary_expr_with_expected_type(left, op, right, output_type, None, input_schema, available_columns, ctx)
}

#[expect(
    clippy::too_many_arguments,
    reason = "expression/join residual validation threads the shared schema pair plus the backend context"
)]
fn validate_binary_expr_with_expected_type(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
    expected_type: Option<&ArrowDataType>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    let declared_type = output_type.or(expected_type);
    let left_interval = interval_literal(left);
    let right_interval = interval_literal(right);
    if left_interval.is_some() || right_interval.is_some() {
        return validate_interval_binary_expr(
            left,
            op,
            right,
            left_interval,
            right_interval,
            input_schema,
            available_columns,
            ctx,
        );
    }
    // Same-operator AND / OR chains are flattened iteratively. Recursing
    // through `validate_expr` once per conjunct would make stack depth grow
    // with the number of predicates a frontend conjoins.
    if matches!(op, BinaryOp::And | BinaryOp::Or) && output_type.is_none() {
        return validate_logical_binary_expr(left, op, right, input_schema, available_columns, ctx);
    }

    let left_type = validate_expr(left, input_schema, available_columns, ctx)?;
    let right_type = validate_expr(right, input_schema, available_columns, ctx)?;
    // Temporal arithmetic is resolved before the arithmetic rules below, which
    // would reject a timestamp or duration operand.
    if let Some(output_type) = temporal_duration_binary_output_type(op, &left_type, &right_type)? {
        return Ok(output_type);
    }
    // A declared output type is honored only for decimal division, the one
    // operator whose result scale the frontend owns. Anywhere else it would let
    // the IR assert a type execution does not produce.
    if output_type.is_some()
        && !(op == BinaryOp::Divide && (is_decimal_type(&left_type) || is_decimal_type(&right_type)))
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "declared binary output types are currently supported only for decimal division",
        )
        .with_fact("left_type", format!("{left_type:?}"))
        .with_fact("right_type", format!("{right_type:?}"))
        .with_fact("operator", op.to_string()));
    }
    match op {
        BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide => {
            require_arithmetic_type(&left_type, "binary arithmetic left operand")?;
            require_arithmetic_type(&right_type, "binary arithmetic right operand")?;
            require_compatible_types(&left_type, &right_type, "binary arithmetic operands", ctx)?;
            if is_decimal_type(&left_type) || is_decimal_type(&right_type) {
                decimal_arithmetic_output_type(op, &left_type, &right_type, declared_type, ctx)
            } else {
                Ok(left_type)
            }
        },
        BinaryOp::Modulo => {
            require_remainder_type(&left_type, "modulo left operand")?;
            require_remainder_type(&right_type, "modulo right operand")?;
            require_compatible_types(&left_type, &right_type, "modulo operands", ctx)?;
            Ok(left_type)
        },
        BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor
        | BinaryOp::ShiftLeft
        | BinaryOp::ShiftRight => {
            require_integral_type(&left_type, "bitwise left operand")?;
            require_integral_type(&right_type, "bitwise right operand")?;
            require_compatible_types(&left_type, &right_type, "bitwise operands", ctx)?;
            Ok(left_type)
        },
        BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
            require_comparable_types(&left_type, &right_type)?;
            Ok(ArrowDataType::Boolean)
        },
        BinaryOp::And | BinaryOp::Or => {
            require_boolean(&left_type, "boolean binary left operand")?;
            require_boolean(&right_type, "boolean binary right operand")?;
            Ok(ArrowDataType::Boolean)
        },
    }
}

fn validate_logical_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    let mut stack = vec![right, left];
    while let Some(expr) = stack.pop() {
        match expr {
            Expr::Binary { left, op: nested_op, right, output_type: None } if *nested_op == op => {
                stack.push(right);
                stack.push(left);
            },
            _ => {
                let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
                require_boolean(&data_type, "boolean binary operand")?;
            },
        }
    }
    Ok(ArrowDataType::Boolean)
}

/// Result type of decimal arithmetic. Division requires the declared
/// `Decimal128` output: cuDF fixed-point division derives its result scale from
/// the operand scales, which differs from the SQL result scale DataFusion
/// declares, so execution casts to the declared type. For the other operators
/// the backend infers the result from the operand types.
fn decimal_arithmetic_output_type(
    op: BinaryOp,
    left_type: &ArrowDataType,
    right_type: &ArrowDataType,
    expected_type: Option<&ArrowDataType>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    if !is_decimal_type(left_type) && !is_decimal_type(right_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "not decimal arithmetic",
        ));
    }
    if op == BinaryOp::Divide {
        let Some(expected_type @ ArrowDataType::Decimal128(_, _)) = expected_type else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.plan.validate_expr",
                "native decimal division requires a declared decimal output type",
            )
            .with_fact("left_type", format!("{left_type:?}"))
            .with_fact("right_type", format!("{right_type:?}"))
            .with_fact("operator", op.to_string()));
        };
        if !ctx.backend().supports_type(expected_type) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.plan.validate_expr",
                "native decimal division declared output type is not supported",
            )
            .with_fact("expected_type", format!("{expected_type:?}")));
        }
        return Ok(expected_type.clone());
    }

    if !ctx.backend().supports_type(left_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "native decimal arithmetic left operand type is not supported",
        )
        .with_fact("left_type", format!("{left_type:?}")));
    }
    if !ctx.backend().supports_type(right_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "native decimal arithmetic right operand type is not supported",
        )
        .with_fact("right_type", format!("{right_type:?}")));
    }
    ctx.backend().binary_output_type(op, left_type, right_type)
}

#[expect(
    clippy::too_many_arguments,
    reason = "expression/join residual validation threads the shared schema pair plus the backend context"
)]
/// An interval literal is legal only as one operand of `datetime + interval`,
/// `datetime - interval` or `interval + datetime`, and the result keeps the
/// datetime operand's type. `validate_literal` rejects an interval anywhere
/// else, so no bare interval value reaches a kernel.
fn validate_interval_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    left_interval: Option<(i32, i32, i64)>,
    right_interval: Option<(i32, i32, i64)>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    if left_interval.is_some() == right_interval.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native interval arithmetic requires exactly one interval literal operand",
        ));
    }
    let operand = if left_interval.is_some() {
        if op != BinaryOp::Add {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.plan.validate_expr",
                "native interval arithmetic supports interval + datetime but not interval minus datetime",
            ));
        }
        right
    } else {
        if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.plan.validate_expr",
                "native interval arithmetic supports only datetime plus/minus interval",
            ));
        }
        right_interval.expect("right interval was checked above");
        left
    };
    let operand_type = validate_expr(operand, input_schema, available_columns, ctx)?;
    require_date_or_timestamp(&operand_type, "interval arithmetic datetime operand")?;
    Ok(operand_type)
}

fn validate_in_list_expr(
    expr: &Expr,
    list: &[Expr],
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    if list.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native IN list must not be empty",
        ));
    }
    let expr_type = validate_expr(expr, input_schema, available_columns, ctx)?;
    for item in list {
        let Expr::Literal(value) = item else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.plan.validate_expr",
                "native IN list values must be literals",
            ));
        };
        // A null in the list makes SQL `x IN (...)` null, rather than false,
        // for every non-matching row; the native membership kernel has no
        // three-valued form.
        if matches!(value, LiteralValue::Null(_)) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.plan.validate_expr",
                "native IN list does not yet model SQL null-containing list semantics",
            ));
        }
        let item_type = validate_literal(value, ctx)?;
        require_compatible_types(&expr_type, &item_type, "IN list value", ctx)?;
    }
    Ok(ArrowDataType::Boolean)
}

/// `CONCAT` evaluates to the widest string representation among its inputs:
/// `Utf8View` if any input is one, otherwise `LargeUtf8` if any is, otherwise
/// `Utf8`.
fn validate_concat_expr(
    exprs: &[Expr],
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    if exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native CONCAT requires at least one argument",
        ));
    }
    let mut output_type = ArrowDataType::Utf8;
    for expr in exprs {
        let data_type = validate_expr(expr, input_schema, available_columns, ctx)?;
        require_string(&data_type, "CONCAT input")?;
        if data_type == ArrowDataType::Utf8View {
            output_type = ArrowDataType::Utf8View;
        } else if data_type == ArrowDataType::LargeUtf8 && output_type != ArrowDataType::Utf8View {
            output_type = ArrowDataType::LargeUtf8;
        }
    }
    Ok(output_type)
}

/// A searched `CASE` evaluates to its `ELSE` type; every `THEN` must be
/// compatible with it through an implicit cast.
fn validate_case_expr(
    branches: &[CaseWhen],
    else_expr: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    ctx: AnalysisContext<'_>,
) -> Result<ArrowDataType> {
    if branches.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native CASE requires at least one WHEN branch",
        ));
    }
    let else_type = validate_expr(else_expr, input_schema, available_columns, ctx)?;
    for branch in branches {
        let when_type = validate_expr(&branch.when, input_schema, available_columns, ctx)?;
        require_boolean(&when_type, "CASE WHEN predicate")?;
        let then_type = validate_expr(&branch.then, input_schema, available_columns, ctx)?;
        require_compatible_types(&else_type, &then_type, "CASE branch result", ctx)?;
    }
    Ok(else_type)
}

fn interval_literal(expr: &Expr) -> Option<(i32, i32, i64)> {
    match expr {
        Expr::Literal(LiteralValue::IntervalMonthDayNano { months, days, nanoseconds }) => {
            Some((*months, *days, *nanoseconds))
        },
        _ => None,
    }
}

/// cuDF `slice_strings` takes zero-based `i32` start and stop positions, so
/// `start` must be at least one and `(start - 1) + length` must fit in `i32`.
fn validate_substring_bounds(start: i32, length: Option<i32>) -> Result<()> {
    if start < 1 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native substring requires a one-based positive start position",
        )
        .with_fact("start", start));
    }
    if let Some(length) = length
        && (length < 0 || substring_stop_position(start, length).is_none())
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native substring requires a non-negative in-range length",
        )
        .with_fact("start", start)
        .with_fact("length", length));
    }
    Ok(())
}

pub fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
}

/// cuDF `like` takes one escape character, and an empty string means none.
/// Only ASCII is accepted so the one-character bound holds in bytes as well as
/// code points.
fn validate_like_escape_contract(escape: &str) -> Result<()> {
    if escape.len() <= 1 && escape.is_ascii() {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.validate_expr",
            "native LIKE escape must be empty or one ASCII byte",
        )
        .with_fact("escape", escape))
    }
}

pub(crate) fn comma_separated(columns: &BTreeSet<String>) -> String {
    columns.iter().cloned().collect::<Vec<_>>().join(",")
}

fn unary_math_type_contexts(op: UnaryMathOp) -> (&'static str, &'static str) {
    match op {
        UnaryMathOp::Negate => ("NEGATE input", "NEGATE output"),
        UnaryMathOp::Sqrt => ("SQRT input", "SQRT output"),
        UnaryMathOp::Ln => ("LN input", "LN output"),
        UnaryMathOp::Exp => ("EXP input", "EXP output"),
        UnaryMathOp::Floor => ("FLOOR input", "FLOOR output"),
        UnaryMathOp::Ceil => ("CEIL input", "CEIL output"),
    }
}

/// Input and output type rules per function. Negation is in place over signed
/// integers and floats; an unsigned input would need a wider signed output the
/// IR does not model. `FLOOR` and `CEIL` are floating-point operations that
/// keep the input type. `SQRT`, `LN` and `EXP` accept any arithmetic input and
/// leave the output type to the caller's decimal consistency check.
pub(super) fn validate_unary_math_types(
    op: UnaryMathOp,
    input_type: &ArrowDataType,
    output_type: &ArrowDataType,
) -> Result<()> {
    let (admitted, requirement) = match op {
        UnaryMathOp::Negate => (
            matches!(
                input_type,
                ArrowDataType::Int8
                    | ArrowDataType::Int16
                    | ArrowDataType::Int32
                    | ArrowDataType::Int64
                    | ArrowDataType::Float32
                    | ArrowDataType::Float64
            ),
            "native negation requires matching signed integer or floating-point input and output types",
        ),
        UnaryMathOp::Floor | UnaryMathOp::Ceil => (
            matches!(input_type, ArrowDataType::Float32 | ArrowDataType::Float64),
            "native FLOOR and CEIL require matching Float32 or Float64 input and output types",
        ),
        UnaryMathOp::Sqrt | UnaryMathOp::Ln | UnaryMathOp::Exp => return Ok(()),
    };
    if admitted && input_type == output_type {
        return Ok(());
    }
    Err(QueryError::new(QueryErrorCode::UnsupportedType, "query_engine.plan.validate_expr", requirement)
        .with_fact("input_type", format!("{input_type:?}"))
        .with_fact("output_type", format!("{output_type:?}")))
}
