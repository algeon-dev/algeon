use std::collections::BTreeSet;

use arrow::datatypes::{DataType as ArrowDataType, SchemaRef};
use cudf::ops::binary::{BinaryOperator, infer_binary_output_dtype};

use super::{
    types::{
        is_decimal_type, require_arithmetic_type, require_boolean, require_comparable_types, require_compatible_types,
        require_date_or_timestamp, require_integral_type, require_string, require_timestamp_second,
        temporal_duration_binary_output_type,
    },
    *,
};
use crate::{
    datatypes::{is_supported_arrow_cast, is_supported_arrow_type},
    expr::{CaseWhen, DatePart},
    plan::ProjectionOutput,
};

pub(super) fn validate_expr(
    expr: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => validate_column(column.name(), input_schema, available_columns),
        Expr::Literal(value) => validate_literal(value),
        Expr::Not(expr) => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_boolean(&data_type, "NOT argument")?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::IsNull(expr) | Expr::IsNotNull(expr) => {
            validate_expr(expr, input_schema, available_columns)?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::Cast { expr, target_type } => {
            let source_type =
                validate_expr_with_expected_type(expr, Some(target_type), input_schema, available_columns)?;
            if !is_supported_arrow_cast(&source_type, target_type)
                && !bounded_nonnegative_integral_cast_is_supported(expr, &source_type, target_type)
            {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.plan.validate_expr",
                    "native cast source and target types are not supported",
                )
                .with_fact("source_type", format!("{source_type:?}"))
                .with_fact("target_type", format!("{target_type:?}")));
            }
            Ok(target_type.clone())
        },
        Expr::TryCast { expr, target_type } => {
            validate_try_cast_expr(expr, target_type, input_schema, available_columns)
        },
        Expr::Binary { left, op, right, output_type } => {
            validate_binary_expr(left, *op, right, output_type.as_ref(), input_schema, available_columns)
        },
        Expr::Like { expr, escape, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_string(&data_type, "LIKE input")?;
            if let Some(escape) = escape {
                validate_like_escape_contract(escape)?;
            }
            Ok(ArrowDataType::Boolean)
        },
        Expr::Substring { expr, start, length } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_string(&data_type, "SUBSTRING input")?;
            validate_substring_bounds(*start, *length)?;
            Ok(ArrowDataType::Utf8)
        },
        Expr::Lower(expr) | Expr::Upper(expr) => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_string(&data_type, "string case input")?;
            Ok(data_type)
        },
        Expr::StringLength { expr, output_type } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_string(&data_type, "string length input")?;
            require_integral_type(output_type, "string length output")?;
            Ok(output_type.clone())
        },
        Expr::RegexpReplace { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.validate_regexp_replace",
            "REGEXP_REPLACE requires a bounded output-size contract before native execution",
        )),
        Expr::Concat(exprs) => validate_concat_expr(exprs, input_schema, available_columns),
        Expr::Abs(expr) => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_arithmetic_type(&data_type, "ABS input")?;
            Ok(data_type)
        },
        Expr::Sqrt { expr, output_type } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_arithmetic_type(&data_type, "SQRT input")?;
            require_arithmetic_type(output_type, "SQRT output")?;
            if is_decimal_type(&data_type) != is_decimal_type(output_type) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.plan.validate_expr",
                    "native SQRT input and output must both be decimal or both be non-decimal",
                )
                .with_fact("input_type", format!("{data_type:?}"))
                .with_fact("output_type", format!("{output_type:?}")));
            }
            Ok(output_type.clone())
        },
        Expr::Round { expr, output_type, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_arithmetic_type(&data_type, "ROUND input")?;
            require_arithmetic_type(output_type, "ROUND output")?;
            if is_decimal_type(&data_type) != is_decimal_type(output_type) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.plan.validate_expr",
                    "native ROUND input and output must both be decimal or both be non-decimal",
                )
                .with_fact("input_type", format!("{data_type:?}"))
                .with_fact("output_type", format!("{output_type:?}")));
            }
            Ok(output_type.clone())
        },
        Expr::InList { expr, list, .. } => validate_in_list_expr(expr, list, input_schema, available_columns),
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            let left_type = validate_expr(left, input_schema, available_columns)?;
            let right_type = validate_expr(right, input_schema, available_columns)?;
            require_comparable_types(&left_type, &right_type)?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::Case { branches, else_expr } => validate_case_expr(branches, else_expr, input_schema, available_columns),
        Expr::Extract { field, expr } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            match field {
                DatePart::Year | DatePart::Month | DatePart::Day => {
                    require_date_or_timestamp(&data_type, "EXTRACT input")?
                },
                DatePart::Minute => validate_timestamp_second_integer_cast_expr(expr, input_schema, available_columns)?,
            }
            Ok(ArrowDataType::Int32)
        },
        Expr::DateTrunc { field, expr, output_type } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            match field {
                DatePart::Minute => {
                    require_timestamp_second(&data_type, "DATE_TRUNC minute input")?;
                    require_timestamp_second(output_type, "DATE_TRUNC minute output")?;
                    Ok(output_type.clone())
                },
                DatePart::Year | DatePart::Month | DatePart::Day => Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.plan.validate_expr",
                    "native DATE_TRUNC does not support this field",
                )
                .with_fact("field", field.to_string())),
            }
        },
        Expr::Alias { expr, .. } => validate_expr(expr, input_schema, available_columns),
    }
}

fn validate_try_cast_expr(
    input: &Expr,
    target_type: &ArrowDataType,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    let source_type = validate_expr(input, input_schema, available_columns)?;
    if !is_string_arrow_type(&source_type) || !is_integer_arrow_type(target_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
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

fn is_integer_arrow_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
    )
}

fn validate_timestamp_second_integer_cast_expr(
    expr: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<()> {
    let Expr::Cast { expr, target_type } = expr else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native second timestamp minute operations currently require an integer-to-timestamp cast",
        )
        .with_fact("expression", expr));
    };
    require_timestamp_second(target_type, "timestamp cast target")?;
    let source_type = validate_expr(expr, input_schema, available_columns)?;
    require_integral_type(&source_type, "timestamp cast source")
}

pub(super) fn validate_projection_output_expr(
    output: &ProjectionOutput,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    validate_expr_with_expected_type(&output.expr, Some(output.field.data_type()), input_schema, available_columns)
}

fn validate_expr_with_expected_type(
    expr: &Expr,
    expected_type: Option<&ArrowDataType>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    match expr {
        Expr::Cast { expr, target_type } => {
            let source_type =
                validate_expr_with_expected_type(expr, Some(target_type), input_schema, available_columns)?;
            if !is_supported_arrow_cast(&source_type, target_type)
                && !bounded_nonnegative_integral_cast_is_supported(expr, &source_type, target_type)
            {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.plan.validate_expr",
                    "native cast source and target types are not supported",
                )
                .with_fact("source_type", format!("{source_type:?}"))
                .with_fact("target_type", format!("{target_type:?}")));
            }
            Ok(target_type.clone())
        },
        Expr::TryCast { expr, target_type } => {
            validate_try_cast_expr(expr, target_type, input_schema, available_columns)
        },
        Expr::Binary { left, op, right, output_type } => validate_binary_expr_with_expected_type(
            left,
            *op,
            right,
            output_type.as_ref(),
            expected_type,
            input_schema,
            available_columns,
        ),
        Expr::Alias { expr, .. } => {
            validate_expr_with_expected_type(expr, expected_type, input_schema, available_columns)
        },
        _ => validate_expr(expr, input_schema, available_columns),
    }
}

pub(super) fn validate_column(
    name: &str,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if !available_columns.contains(name) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.plan.validate_expr",
            "native expression references a column that is not produced by its input",
        )
        .with_fact("column", name)
        .with_fact("available_columns", comma_separated(available_columns)));
    }
    let field = input_schema.field_with_name(name).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.plan.validate_expr",
            "native expression references a column missing from its input schema",
        )
        .with_fact("column", name)
        .with_fact("arrow_error", error.to_string())
    })?;
    if !is_supported_arrow_type(field.data_type()) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native expression references an unsupported column type",
        )
        .with_fact("column", name)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(field.data_type().clone())
}

pub(super) fn validate_literal(value: &LiteralValue) -> Result<ArrowDataType> {
    if matches!(value, LiteralValue::IntervalMonthDayNano { .. }) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "interval literals are supported only inside datetime interval arithmetic",
        )
        .with_fact("literal", value));
    }
    let data_type = value.data_type();
    if !is_supported_arrow_type(&data_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
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
) -> Result<ArrowDataType> {
    validate_binary_expr_with_expected_type(left, op, right, output_type, None, input_schema, available_columns)
}

fn validate_binary_expr_with_expected_type(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
    expected_type: Option<&ArrowDataType>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
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
        );
    }

    let left_type = validate_expr(left, input_schema, available_columns)?;
    let right_type = validate_expr(right, input_schema, available_columns)?;
    if let Some(output_type) = temporal_duration_binary_output_type(op, &left_type, &right_type)? {
        return Ok(output_type);
    }
    if output_type.is_some()
        && !(op == BinaryOp::Divide && (is_decimal_type(&left_type) || is_decimal_type(&right_type)))
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
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
            require_compatible_types(&left_type, &right_type, "binary arithmetic operands")?;
            if is_decimal_type(&left_type) || is_decimal_type(&right_type) {
                decimal_arithmetic_output_type(op, &left_type, &right_type, declared_type)
            } else {
                Ok(left_type)
            }
        },
        BinaryOp::Modulo => {
            require_integral_type(&left_type, "modulo left operand")?;
            require_integral_type(&right_type, "modulo right operand")?;
            require_compatible_types(&left_type, &right_type, "modulo operands")?;
            Ok(left_type)
        },
        BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor
        | BinaryOp::ShiftLeft
        | BinaryOp::ShiftRight => {
            require_integral_type(&left_type, "bitwise left operand")?;
            require_integral_type(&right_type, "bitwise right operand")?;
            require_compatible_types(&left_type, &right_type, "bitwise operands")?;
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

fn decimal_arithmetic_output_type(
    op: BinaryOp,
    left_type: &ArrowDataType,
    right_type: &ArrowDataType,
    expected_type: Option<&ArrowDataType>,
) -> Result<ArrowDataType> {
    if !is_decimal_type(left_type) && !is_decimal_type(right_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "not decimal arithmetic",
        ));
    }
    if op == BinaryOp::Divide {
        let Some(expected_type @ ArrowDataType::Decimal128(_, _)) = expected_type else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.plan.validate_expr",
                "native decimal division requires a declared decimal output type",
            )
            .with_fact("left_type", format!("{left_type:?}"))
            .with_fact("right_type", format!("{right_type:?}"))
            .with_fact("operator", op.to_string()));
        };
        if !is_supported_arrow_type(expected_type) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.plan.validate_expr",
                "native decimal division declared output type is not supported",
            )
            .with_fact("expected_type", format!("{expected_type:?}")));
        }
        return Ok(expected_type.clone());
    }

    let left_dtype = cudf_dtype_from_arrow_type(left_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native decimal arithmetic left operand type is not supported",
        )
        .with_fact("left_type", format!("{left_type:?}"))
    })?;
    let right_dtype = cudf_dtype_from_arrow_type(right_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native decimal arithmetic right operand type is not supported",
        )
        .with_fact("right_type", format!("{right_type:?}"))
    })?;
    let output_dtype =
        infer_binary_output_dtype(left_dtype, right_dtype, cudf_binary_operator(op)?).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.plan.validate_expr",
                "failed to infer native decimal arithmetic output type",
                error,
            )
            .with_fact("left_type", format!("{left_type:?}"))
            .with_fact("right_type", format!("{right_type:?}"))
            .with_fact("operator", op.to_string())
        })?;
    cudf::interop::arrow_host::arrow_type_from_cudf_dtype(output_dtype).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.plan.validate_expr",
            "failed to map inferred native decimal arithmetic output type to Arrow",
            error,
        )
        .with_fact("left_type", format!("{left_type:?}"))
        .with_fact("right_type", format!("{right_type:?}"))
        .with_fact("operator", op.to_string())
    })
}

fn cudf_binary_operator(op: BinaryOp) -> Result<BinaryOperator> {
    match op {
        BinaryOp::Add => Ok(BinaryOperator::Add),
        BinaryOp::Subtract => Ok(BinaryOperator::Sub),
        BinaryOp::Multiply => Ok(BinaryOperator::Mul),
        BinaryOp::Divide => Ok(BinaryOperator::Div),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native decimal arithmetic cannot infer output type for this operator",
        )
        .with_fact("operator", op.to_string())),
    }
}

fn validate_interval_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    left_interval: Option<(i32, i32, i64)>,
    right_interval: Option<(i32, i32, i64)>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if left_interval.is_some() == right_interval.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native interval arithmetic requires exactly one interval literal operand",
        ));
    }
    let operand = if left_interval.is_some() {
        if op != BinaryOp::Add {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native interval arithmetic supports interval + datetime but not interval minus datetime",
            ));
        }
        right
    } else {
        if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native interval arithmetic supports only datetime plus/minus interval",
            ));
        }
        right_interval.expect("right interval was checked above");
        left
    };
    let operand_type = validate_expr(operand, input_schema, available_columns)?;
    require_date_or_timestamp(&operand_type, "interval arithmetic datetime operand")?;
    Ok(operand_type)
}

fn validate_in_list_expr(
    expr: &Expr,
    list: &[Expr],
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if list.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native IN list must not be empty",
        ));
    }
    let expr_type = validate_expr(expr, input_schema, available_columns)?;
    for item in list {
        let Expr::Literal(value) = item else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native IN list values must be literals",
            ));
        };
        if matches!(value, LiteralValue::Null(_)) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native IN list does not yet model SQL null-containing list semantics",
            ));
        }
        let item_type = validate_literal(value)?;
        require_compatible_types(&expr_type, &item_type, "IN list value")?;
    }
    Ok(ArrowDataType::Boolean)
}

fn validate_concat_expr(
    exprs: &[Expr],
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native CONCAT requires at least one argument",
        ));
    }
    let mut output_type = ArrowDataType::Utf8;
    for expr in exprs {
        let data_type = validate_expr(expr, input_schema, available_columns)?;
        require_string(&data_type, "CONCAT input")?;
        if data_type == ArrowDataType::Utf8View {
            output_type = ArrowDataType::Utf8View;
        } else if data_type == ArrowDataType::LargeUtf8 && output_type != ArrowDataType::Utf8View {
            output_type = ArrowDataType::LargeUtf8;
        }
    }
    Ok(output_type)
}

fn bounded_nonnegative_integral_cast_is_supported(
    expr: &Expr,
    source_type: &ArrowDataType,
    target_type: &ArrowDataType,
) -> bool {
    if !is_integral_type(source_type) || !is_integral_type(target_type) {
        return false;
    }
    let Some(max_value) = nonnegative_integral_expr_max(expr) else {
        return false;
    };
    target_integral_max(target_type).is_some_and(|target_max| max_value <= target_max)
}

fn nonnegative_integral_expr_max(expr: &Expr) -> Option<u128> {
    match expr {
        Expr::Literal(value) => nonnegative_integral_literal_value(value),
        Expr::Alias { expr, .. } | Expr::Cast { expr, .. } => nonnegative_integral_expr_max(expr),
        Expr::Binary { left, op: BinaryOp::BitwiseAnd, right, .. } => {
            nonnegative_integral_expr_max(left).or_else(|| nonnegative_integral_expr_max(right))
        },
        Expr::Binary { left, op: BinaryOp::ShiftRight, right, .. } => {
            let left_max = nonnegative_integral_expr_max(left)?;
            let shift = nonnegative_integral_expr_literal_value(right)?;
            if shift >= 128 { Some(0) } else { Some(left_max >> shift) }
        },
        Expr::Binary { left, op: BinaryOp::ShiftLeft, right, .. } => {
            let left_max = nonnegative_integral_expr_max(left)?;
            let shift = u32::try_from(nonnegative_integral_expr_literal_value(right)?).ok()?;
            left_max.checked_shl(shift)
        },
        Expr::Binary { left, op: BinaryOp::BitwiseOr | BinaryOp::BitwiseXor, right, .. } => {
            Some(nonnegative_integral_expr_max(left)? | nonnegative_integral_expr_max(right)?)
        },
        _ => None,
    }
}

fn nonnegative_integral_expr_literal_value(expr: &Expr) -> Option<u128> {
    match expr {
        Expr::Literal(value) => nonnegative_integral_literal_value(value),
        Expr::Alias { expr, .. } | Expr::Cast { expr, .. } => nonnegative_integral_expr_literal_value(expr),
        _ => None,
    }
}

fn nonnegative_integral_literal_value(value: &LiteralValue) -> Option<u128> {
    match value {
        LiteralValue::Int8(value) => u128::try_from(*value).ok(),
        LiteralValue::Int16(value) => u128::try_from(*value).ok(),
        LiteralValue::Int32(value) => u128::try_from(*value).ok(),
        LiteralValue::Int64(value) => u128::try_from(*value).ok(),
        LiteralValue::UInt8(value) => Some(u128::from(*value)),
        LiteralValue::UInt16(value) => Some(u128::from(*value)),
        LiteralValue::UInt32(value) => Some(u128::from(*value)),
        LiteralValue::UInt64(value) => Some(u128::from(*value)),
        _ => None,
    }
}

fn target_integral_max(data_type: &ArrowDataType) -> Option<u128> {
    match data_type {
        ArrowDataType::Int8 => Some(i8::MAX as u128),
        ArrowDataType::Int16 => Some(i16::MAX as u128),
        ArrowDataType::Int32 => Some(i32::MAX as u128),
        ArrowDataType::Int64 => Some(i64::MAX as u128),
        ArrowDataType::UInt8 => Some(u128::from(u8::MAX)),
        ArrowDataType::UInt16 => Some(u128::from(u16::MAX)),
        ArrowDataType::UInt32 => Some(u128::from(u32::MAX)),
        ArrowDataType::UInt64 => Some(u128::from(u64::MAX)),
        _ => None,
    }
}

fn is_integral_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
    )
}

fn validate_case_expr(
    branches: &[CaseWhen],
    else_expr: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if branches.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native CASE requires at least one WHEN branch",
        ));
    }
    let else_type = validate_expr(else_expr, input_schema, available_columns)?;
    for branch in branches {
        let when_type = validate_expr(&branch.when, input_schema, available_columns)?;
        require_boolean(&when_type, "CASE WHEN predicate")?;
        let then_type = validate_expr(&branch.then, input_schema, available_columns)?;
        require_compatible_types(&else_type, &then_type, "CASE branch result")?;
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

fn validate_substring_bounds(start: i32, length: Option<i32>) -> Result<()> {
    if start < 1 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native substring requires a one-based positive start position",
        )
        .with_fact("start", start));
    }
    if let Some(length) = length
        && (length < 0 || substring_stop_position(start, length).is_none())
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native substring requires a non-negative in-range length",
        )
        .with_fact("start", start)
        .with_fact("length", length));
    }
    Ok(())
}

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
}

fn validate_like_escape_contract(escape: &str) -> Result<()> {
    if escape.len() <= 1 && escape.is_ascii() {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native LIKE escape must be empty or one ASCII byte",
        )
        .with_fact("escape", escape))
    }
}

pub(super) fn comma_separated(columns: &BTreeSet<String>) -> String {
    columns.iter().cloned().collect::<Vec<_>>().join(",")
}
