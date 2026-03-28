use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr,
            LikeExpr, Literal, NotExpr, TryCastExpr,
        },
    },
};

use super::{
    is_list_carry_type, is_supported_native_cast, is_supported_native_type, is_supported_to_timestamp_seconds_cast,
    is_timestamp_with_timezone_type,
};
pub(super) use crate::native::literal::literal_string;
use crate::{
    native::literal::{
        NativeLiteralContext, is_interval_literal_expr, literal_i32, parse_yyyy_mm_dd_date32, substring_stop_position,
        validate_native_literal,
    },
    policy::FallbackReason,
};

pub(in crate::native) fn validate_physical_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return validate_column_type(column, input_schema);
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return validate_literal(literal.value());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return validate_binary_expr(binary, input_schema);
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return validate_physical_expr(not.arg(), input_schema);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return validate_null_predicate_arg(is_null.arg(), input_schema);
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return validate_null_predicate_arg(is_not_null.arg(), input_schema);
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type())
            && !bounded_nonnegative_integral_cast_is_supported(cast.expr(), &source_type, cast.cast_type())
            && !literal_string_date32_cast_is_supported(cast.expr(), cast.cast_type())
        {
            return Err(FallbackReason::UnsupportedType);
        }
        return validate_physical_expr(cast.expr(), input_schema);
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        return validate_like_expr(like, input_schema);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        return validate_in_list_expr(in_list, input_schema);
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return validate_case_expr(case, input_schema);
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| FallbackReason::UnsupportedExpression)?;
        return validate_physical_expr(&current, input_schema);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return validate_scalar_function_expr(function, input_schema);
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return validate_try_cast_expr(try_cast, input_schema);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn validate_try_cast_expr(try_cast: &TryCastExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let source_type =
        try_cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_try_cast_pair(&source_type, try_cast.cast_type()) {
        // Unsupported type pairs stay on the UnsupportedExpression path so the
        // planner's child-rewrite short-circuit treats TRY_CAST the same way
        // it treated it before native admission of `STRING → integer`.
        return Err(FallbackReason::UnsupportedExpression);
    }
    validate_physical_expr(try_cast.expr(), input_schema)
}

fn validate_null_predicate_arg(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    validate_physical_expr(expr, input_schema)?;
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if is_timestamp_with_timezone_type(&data_type) || is_list_carry_type(&data_type) {
        Err(FallbackReason::UnsupportedType)
    } else {
        Ok(())
    }
}

fn is_supported_try_cast_pair(source_type: &DataType, target_type: &DataType) -> bool {
    is_string_data_type(source_type) && is_integer_data_type(target_type)
}

fn is_string_data_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn is_integer_data_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn literal_string_date32_cast_is_supported(expr: &Arc<dyn PhysicalExpr>, target_type: &DataType) -> bool {
    matches!(target_type, DataType::Date32)
        && expr
            .downcast_ref::<Literal>()
            .and_then(|literal| match literal.value() {
                ScalarValue::Utf8(Some(value))
                | ScalarValue::LargeUtf8(Some(value))
                | ScalarValue::Utf8View(Some(value)) => parse_yyyy_mm_dd_date32(value),
                _ => None,
            })
            .is_some()
}

fn validate_like_expr(like: &LikeExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    if like.case_insensitive() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    validate_literal_string(like.pattern())?;
    validate_physical_expr(like.expr(), input_schema)?;
    validate_string_expr(like.expr(), input_schema)
}

fn validate_in_list_expr(in_list: &InListExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    if in_list.is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    validate_physical_expr(in_list.expr(), input_schema)?;
    let expr_type = physical_expr_data_type(in_list.expr(), input_schema)?;
    for expr in in_list.list() {
        let Some(literal) = expr.downcast_ref::<Literal>() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        if literal.value().is_null() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        validate_physical_expr(expr, input_schema)?;
        let literal_type = physical_expr_data_type(expr, input_schema)?;
        validate_compatible_expr_types(&expr_type, &literal_type)?;
    }
    Ok(())
}

fn validate_case_expr(case: &CaseExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    if case.when_then_expr().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let base_type = case
        .expr()
        .map(|expr| {
            validate_physical_expr(expr, input_schema)?;
            physical_expr_data_type(expr, input_schema)
        })
        .transpose()?;
    let output_type = case
        .else_expr()
        .map(|expr| {
            validate_physical_expr(expr, input_schema)?;
            physical_expr_data_type(expr, input_schema)
        })
        .transpose()?
        .unwrap_or(case.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?);

    for (when, then) in case.when_then_expr() {
        validate_physical_expr(when, input_schema)?;
        let when_type = physical_expr_data_type(when, input_schema)?;
        if let Some(base_type) = &base_type {
            validate_compatible_expr_types(base_type, &when_type)?;
        } else {
            validate_boolean_type(&when_type)?;
        }
        validate_physical_expr(then, input_schema)?;
        let then_type = physical_expr_data_type(then, input_schema)?;
        validate_compatible_expr_types(&output_type, &then_type)?;
    }
    Ok(())
}

fn validate_scalar_function_expr(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    match function.name().to_ascii_lowercase().as_str() {
        "substring" | "substr" => validate_substring_function(function, input_schema),
        "date_part" | "extract" => validate_extract_function(function, input_schema),
        "date_trunc" => validate_date_trunc_function(function, input_schema),
        "to_timestamp_seconds" => validate_to_timestamp_seconds_function(function, input_schema),
        "lower" | "upper" => validate_string_case_function(function, input_schema),
        "character_length" | "length" => validate_string_length_function(function, input_schema),
        "regexp_replace" => validate_regexp_replace_function(function, input_schema),
        "concat" => validate_concat_function(function, input_schema),
        "abs" => validate_abs_function(function, input_schema),
        "sqrt" => validate_sqrt_function(function, input_schema),
        "round" => validate_round_function(function, input_schema),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn validate_substring_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let args = function.args();
    validate_substring_arguments_contract(function)?;
    validate_physical_expr(&args[0], input_schema)?;
    validate_string_expr(&args[0], input_schema)
}

fn validate_extract_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let [field, expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    match literal_string(field)?.to_ascii_lowercase().as_str() {
        "year" | "month" | "day" => {},
        "minute" => {
            validate_physical_expr(expr, input_schema)?;
            return validate_timestamp_second_expr(expr, input_schema);
        },
        _ => return Err(FallbackReason::UnsupportedExpression),
    }
    validate_physical_expr(expr, input_schema)?;
    validate_date_or_timestamp_expr(expr, input_schema)
}

fn validate_date_trunc_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let [field, expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    match literal_string(field)?.to_ascii_lowercase().as_str() {
        "minute" => {},
        _ => return Err(FallbackReason::UnsupportedExpression),
    }
    validate_physical_expr(expr, input_schema)?;
    validate_timestamp_second_expr(expr, input_schema)?;
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if matches!(output_type, DataType::Timestamp(TimeUnit::Second, None)) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_to_timestamp_seconds_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_physical_expr(expr, input_schema)?;
    let source_type = physical_expr_data_type(expr, input_schema)?;
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_to_timestamp_seconds_cast(&source_type, &output_type) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_string_case_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_physical_expr(expr, input_schema)?;
    validate_string_expr(expr, input_schema)
}

fn validate_string_length_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_physical_expr(expr, input_schema)?;
    validate_string_expr(expr, input_schema)?;
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(&output_type) && is_integral_type(&output_type) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_regexp_replace_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let [expr, pattern, replacement] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_physical_expr(expr, input_schema)?;
    validate_string_expr(expr, input_schema)?;
    validate_literal_string(pattern)?;
    validate_literal_string(replacement)?;
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(&output_type) && is_string_data_type(&output_type) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_concat_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    if function.args().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    for expr in function.args() {
        validate_physical_expr(expr, input_schema)?;
        validate_string_expr(expr, input_schema)?;
    }
    Ok(())
}

fn validate_abs_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_physical_expr(expr, input_schema)?;
    validate_numeric_or_decimal_expr(expr, input_schema)?;
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(&output_type) && is_numeric_or_decimal_type(&output_type) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_sqrt_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_physical_expr(expr, input_schema)?;
    validate_numeric_or_decimal_expr(expr, input_schema)?;
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(&output_type) && is_numeric_or_decimal_type(&output_type) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_round_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let args = function.args();
    if !(1..=2).contains(&args.len()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    validate_physical_expr(&args[0], input_schema)?;
    validate_numeric_or_decimal_expr(&args[0], input_schema)?;
    if args.len() == 2 {
        literal_i32(&args[1])?;
    }
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(&output_type) && is_numeric_or_decimal_type(&output_type) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_column_type(column: &Column, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let data_type = input_schema.field_with_name(column.name()).map(|field| field.data_type()).or_else(|_error| {
        input_schema
            .fields()
            .get(column.index())
            .map(|field| field.data_type())
            .ok_or(FallbackReason::UnsupportedExpression)
    })?;
    if is_supported_native_type(data_type) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn validate_literal(value: &ScalarValue) -> Result<(), FallbackReason> {
    if matches!(value, ScalarValue::IntervalMonthDayNano(Some(_))) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    validate_native_literal(value, NativeLiteralContext::Expression)
}

fn validate_binary_expr(binary: &BinaryExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let left_interval = is_interval_literal_expr(binary.left());
    let right_interval = is_interval_literal_expr(binary.right());
    if left_interval || right_interval {
        return validate_interval_binary_expr(binary, input_schema, left_interval, right_interval);
    }

    validate_physical_expr(binary.left(), input_schema)?;
    validate_physical_expr(binary.right(), input_schema)?;
    validate_binary_operator(binary.op())?;
    let left_type = physical_expr_data_type(binary.left(), input_schema)?;
    let right_type = physical_expr_data_type(binary.right(), input_schema)?;
    if binary_operator_rejects_timestamp_timezone(binary.op())
        && (is_timestamp_with_timezone_type(&left_type) || is_timestamp_with_timezone_type(&right_type))
    {
        return Err(FallbackReason::UnsupportedType);
    }
    if binary_operator_rejects_list_carry(binary.op())
        && (is_list_carry_type(&left_type) || is_list_carry_type(&right_type))
    {
        return Err(FallbackReason::UnsupportedType);
    }
    if matches!(binary.op(), Operator::StringConcat) {
        validate_string_expr(binary.left(), input_schema)?;
        validate_string_expr(binary.right(), input_schema)?;
    }
    if matches!(binary.op(), Operator::Modulo) {
        validate_integral_operand_type(binary.left(), input_schema)?;
        validate_integral_operand_type(binary.right(), input_schema)?;
    }
    if matches!(
        binary.op(),
        Operator::BitwiseAnd
            | Operator::BitwiseOr
            | Operator::BitwiseXor
            | Operator::BitwiseShiftLeft
            | Operator::BitwiseShiftRight
    ) {
        validate_integral_operand_type(binary.left(), input_schema)?;
        validate_integral_operand_type(binary.right(), input_schema)?;
    }
    Ok(())
}

fn validate_interval_binary_expr(
    binary: &BinaryExpr,
    input_schema: &SchemaRef,
    left_interval: bool,
    right_interval: bool,
) -> Result<(), FallbackReason> {
    if left_interval == right_interval {
        return Err(FallbackReason::UnsupportedExpression);
    }
    match (binary.op(), left_interval, right_interval) {
        (Operator::Plus, true, false) => validate_datetime_interval_operand(binary.right(), input_schema),
        (Operator::Plus | Operator::Minus, false, true) => {
            validate_datetime_interval_operand(binary.left(), input_schema)
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn validate_datetime_interval_operand(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    validate_physical_expr(expr, input_schema)?;
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if matches!(data_type, DataType::Date32 | DataType::Timestamp(_, None)) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedExpression)
    }
}

fn validate_integral_operand_type(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    ) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_binary_operator(op: &Operator) -> Result<(), FallbackReason> {
    match op {
        Operator::Plus
        | Operator::Minus
        | Operator::Multiply
        | Operator::Divide
        | Operator::Modulo
        | Operator::BitwiseAnd
        | Operator::BitwiseOr
        | Operator::BitwiseXor
        | Operator::BitwiseShiftLeft
        | Operator::BitwiseShiftRight
        | Operator::Eq
        | Operator::NotEq
        | Operator::Lt
        | Operator::LtEq
        | Operator::Gt
        | Operator::GtEq
        | Operator::And
        | Operator::Or
        | Operator::StringConcat
        | Operator::IsDistinctFrom
        | Operator::IsNotDistinctFrom => Ok(()),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn binary_operator_rejects_timestamp_timezone(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom
    )
}

fn binary_operator_rejects_list_carry(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom
    )
}

pub(super) fn validate_literal_string(expr: &Arc<dyn PhysicalExpr>) -> Result<(), FallbackReason> {
    literal_string(expr).map(|_| ())
}

pub(super) fn validate_substring_arguments_contract(function: &ScalarFunctionExpr) -> Result<(), FallbackReason> {
    let args = function.args();
    if !(2..=3).contains(&args.len()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let start = literal_i32(&args[1])?;
    if start < 1 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if args.len() == 3 {
        let length = literal_i32(&args[2])?;
        if length < 0 || substring_stop_position(start, length).is_none() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }
    Ok(())
}

fn validate_string_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_date_or_timestamp_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if matches!(data_type, DataType::Date32 | DataType::Timestamp(_, None)) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_timestamp_second_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if matches!(data_type, DataType::Timestamp(TimeUnit::Second, None)) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn validate_numeric_or_decimal_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if is_numeric_or_decimal_type(&data_type) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

pub(in crate::native) fn bounded_nonnegative_integral_cast_is_supported(
    expr: &Arc<dyn PhysicalExpr>,
    source_type: &DataType,
    target_type: &DataType,
) -> bool {
    if !is_integral_type(source_type) || !is_integral_type(target_type) {
        return false;
    }
    let Some(max_value) = nonnegative_integral_expr_max(expr) else {
        return false;
    };
    target_integral_max(target_type).is_some_and(|target_max| max_value <= target_max)
}

fn nonnegative_integral_expr_max(expr: &Arc<dyn PhysicalExpr>) -> Option<u128> {
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return nonnegative_integral_scalar_value(literal.value());
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return nonnegative_integral_expr_max(cast.expr());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return match binary.op() {
            Operator::BitwiseAnd => {
                nonnegative_integral_expr_max(binary.left()).or_else(|| nonnegative_integral_expr_max(binary.right()))
            },
            Operator::BitwiseOr | Operator::BitwiseXor => {
                Some(nonnegative_integral_expr_max(binary.left())? | nonnegative_integral_expr_max(binary.right())?)
            },
            Operator::BitwiseShiftRight => {
                let left_max = nonnegative_integral_expr_max(binary.left())?;
                let shift = nonnegative_integral_scalar_expr_value(binary.right())?;
                if shift >= 128 { Some(0) } else { Some(left_max >> shift) }
            },
            Operator::BitwiseShiftLeft => {
                let left_max = nonnegative_integral_expr_max(binary.left())?;
                let shift = u32::try_from(nonnegative_integral_scalar_expr_value(binary.right())?).ok()?;
                left_max.checked_shl(shift)
            },
            _ => None,
        };
    }
    None
}

fn nonnegative_integral_scalar_expr_value(expr: &Arc<dyn PhysicalExpr>) -> Option<u128> {
    expr.downcast_ref::<Literal>().and_then(|literal| nonnegative_integral_scalar_value(literal.value()))
}

fn nonnegative_integral_scalar_value(value: &ScalarValue) -> Option<u128> {
    match value {
        ScalarValue::Int8(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::Int16(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::Int32(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::Int64(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::UInt8(Some(value)) => Some(u128::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(u128::from(*value)),
        ScalarValue::UInt32(Some(value)) => Some(u128::from(*value)),
        ScalarValue::UInt64(Some(value)) => Some(u128::from(*value)),
        _ => None,
    }
}

fn target_integral_max(data_type: &DataType) -> Option<u128> {
    match data_type {
        DataType::Int8 => Some(i8::MAX as u128),
        DataType::Int16 => Some(i16::MAX as u128),
        DataType::Int32 => Some(i32::MAX as u128),
        DataType::Int64 => Some(i64::MAX as u128),
        DataType::UInt8 => Some(u128::from(u8::MAX)),
        DataType::UInt16 => Some(u128::from(u16::MAX)),
        DataType::UInt32 => Some(u128::from(u32::MAX)),
        DataType::UInt64 => Some(u128::from(u64::MAX)),
        _ => None,
    }
}

fn is_integral_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn is_numeric_or_decimal_type(data_type: &DataType) -> bool {
    is_numeric_type(data_type) || matches!(data_type, DataType::Decimal128(_, _))
}

fn is_numeric_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn validate_boolean_type(data_type: &DataType) -> Result<(), FallbackReason> {
    if matches!(data_type, DataType::Boolean) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn validate_compatible_expr_types(left: &DataType, right: &DataType) -> Result<(), FallbackReason> {
    if is_timestamp_with_timezone_type(left)
        || is_timestamp_with_timezone_type(right)
        || is_list_carry_type(left)
        || is_list_carry_type(right)
    {
        return Err(FallbackReason::UnsupportedType);
    }
    if left == right || is_supported_native_cast(left, right) || is_supported_native_cast(right, left) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn physical_expr_data_type(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<DataType, FallbackReason> {
    expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)
}
