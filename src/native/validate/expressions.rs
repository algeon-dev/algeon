use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
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

use super::{is_supported_native_cast, is_supported_native_type};
use crate::{
    native::literal::{NativeLiteralContext, validate_native_literal},
    policy::FallbackReason,
};

pub(in crate::native) fn validate_physical_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return validate_column_type(column, input_schema);
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return validate_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        return validate_binary_expr(binary, input_schema);
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return validate_physical_expr(not.arg(), input_schema);
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return validate_physical_expr(is_null.arg(), input_schema);
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return validate_physical_expr(is_not_null.arg(), input_schema);
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        return validate_physical_expr(cast.expr(), input_schema);
    }
    if let Some(like) = expr.as_any().downcast_ref::<LikeExpr>() {
        return validate_like_expr(like, input_schema);
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        return validate_in_list_expr(in_list, input_schema);
    }
    if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
        return validate_case_expr(case, input_schema);
    }
    if let Some(dynamic_filter) = expr.as_any().downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| FallbackReason::UnsupportedExpression)?;
        return validate_physical_expr(&current, input_schema);
    }
    if let Some(function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        return validate_scalar_function_expr(function, input_schema);
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Err(FallbackReason::UnsupportedExpression)
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
        let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
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
        "lower" | "upper" => validate_string_case_function(function, input_schema),
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
        _ => return Err(FallbackReason::UnsupportedExpression),
    }
    validate_physical_expr(expr, input_schema)?;
    validate_date_or_timestamp_expr(expr, input_schema)
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
    if matches!(binary.op(), Operator::Modulo) {
        validate_modulo_operand_type(binary.left(), input_schema)?;
        validate_modulo_operand_type(binary.right(), input_schema)?;
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

fn is_interval_literal_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.as_any()
        .downcast_ref::<Literal>()
        .is_some_and(|literal| matches!(literal.value(), ScalarValue::IntervalMonthDayNano(Some(_))))
}

fn validate_modulo_operand_type(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
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
        | Operator::Eq
        | Operator::NotEq
        | Operator::Lt
        | Operator::LtEq
        | Operator::Gt
        | Operator::GtEq
        | Operator::And
        | Operator::Or => Ok(()),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

pub(super) fn validate_literal_string(expr: &Arc<dyn PhysicalExpr>) -> Result<(), FallbackReason> {
    literal_string(expr).map(|_| ())
}

pub(super) fn literal_string(expr: &Arc<dyn PhysicalExpr>) -> Result<String, FallbackReason> {
    let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    match literal.value() {
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Ok(value.clone())
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn literal_i32(expr: &Arc<dyn PhysicalExpr>) -> Result<i32, FallbackReason> {
    let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    match literal.value() {
        ScalarValue::Int8(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::Int16(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::Int32(Some(value)) => Ok(*value),
        ScalarValue::Int64(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        ScalarValue::UInt8(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::UInt32(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        ScalarValue::UInt64(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
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

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
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

fn validate_boolean_type(data_type: &DataType) -> Result<(), FallbackReason> {
    if matches!(data_type, DataType::Boolean) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn validate_compatible_expr_types(left: &DataType, right: &DataType) -> Result<(), FallbackReason> {
    if left == right || is_supported_native_cast(left, right) || is_supported_native_cast(right, left) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn physical_expr_data_type(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<DataType, FallbackReason> {
    expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)
}
