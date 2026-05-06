use std::sync::Arc;

use arrow::datatypes::{Field, SchemaRef};
use cudf_query_engine::{
    BinaryOp as NativeBinaryOp, CaseWhen as NativeCaseWhen, DatePart as NativeDatePart, Expr as NativeExpr,
};
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal, NotExpr,
            TryCastExpr,
        },
    },
};

use crate::{
    native::{
        literal::{NativeLiteralContext, native_literal_value},
        validate::{is_supported_native_cast, is_supported_native_type},
    },
    policy::FallbackReason,
};

pub(super) fn physical_expr_field(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    name: &str,
) -> Result<Field, FallbackReason> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&data_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    let nullable = expr.nullable(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    Ok(Field::new(name, data_type, nullable))
}

pub(super) fn lower_physical_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        validate_column_type(column, input_schema)?;
        return Ok(NativeExpr::column(resolved_column_name(column, input_schema)?));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if matches!(binary.op(), Operator::Modulo) {
            validate_modulo_operand_type(binary.left(), input_schema)?;
            validate_modulo_operand_type(binary.right(), input_schema)?;
        }
        let left = lower_physical_expr(binary.left(), input_schema)?;
        let right = lower_physical_expr(binary.right(), input_schema)?;
        let op = lower_binary_operator(binary.op())?;
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(!lower_physical_expr(not.arg(), input_schema)?);
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_physical_expr(is_null.arg(), input_schema)?));
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_physical_expr(is_not_null.arg(), input_schema)?));
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        return Ok(NativeExpr::cast(lower_physical_expr(cast.expr(), input_schema)?, cast.cast_type().clone()));
    }
    if let Some(like) = expr.as_any().downcast_ref::<LikeExpr>() {
        return lower_like_expr(like, input_schema);
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        return lower_in_list_expr(in_list, input_schema);
    }
    if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
        return lower_case_expr(case, input_schema);
    }
    if let Some(function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        return lower_scalar_function_expr(function, input_schema);
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_like_expr(like: &LikeExpr, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    if like.case_insensitive() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let pattern = literal_string(like.pattern())?;
    Ok(if like.negated() {
        NativeExpr::not_like_escape(lower_physical_expr(like.expr(), input_schema)?, pattern, "\\")
    } else {
        NativeExpr::like_escape(lower_physical_expr(like.expr(), input_schema)?, pattern, "\\")
    })
}

fn lower_in_list_expr(in_list: &InListExpr, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    if in_list.is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let expr = lower_physical_expr(in_list.expr(), input_schema)?;
    let list = in_list
        .list()
        .iter()
        .map(|expr| {
            let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            if literal.value().is_null() {
                return Err(FallbackReason::UnsupportedExpression);
            }
            lower_physical_expr(expr, input_schema)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if in_list.negated() { NativeExpr::not_in_list(expr, list) } else { NativeExpr::in_list(expr, list) })
}

fn lower_case_expr(case: &CaseExpr, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    let base_expr = case.expr().map(|expr| lower_physical_expr(expr, input_schema)).transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => {
                    NativeExpr::binary(base_expr.clone(), NativeBinaryOp::Eq, lower_physical_expr(when, input_schema)?)
                },
                None => lower_physical_expr(when, input_schema)?,
            };
            Ok(NativeCaseWhen::new(when, lower_physical_expr(then, input_schema)?))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_physical_expr(else_expr, input_schema)?,
        None => {
            let data_type =
                case.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
            NativeExpr::literal(cudf_query_engine::LiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

fn lower_scalar_function_expr(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    match function.name().to_ascii_lowercase().as_str() {
        "substring" | "substr" => lower_substring_function(function, input_schema),
        "date_part" | "extract" => lower_extract_function(function, input_schema),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn lower_substring_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    if !(2..=3).contains(&args.len()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let start = literal_i32(&args[1])?;
    if start < 1 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let length = if args.len() == 3 { Some(literal_i32(&args[2])?) } else { None };
    if let Some(length) = length
        && (length < 0 || substring_stop_position(start, length).is_none())
    {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(NativeExpr::substring(lower_physical_expr(&args[0], input_schema)?, start, length))
}

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
}

fn lower_extract_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    let [field, expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let field = match literal_string(field)?.to_ascii_lowercase().as_str() {
        "year" => NativeDatePart::Year,
        "month" => NativeDatePart::Month,
        "day" => NativeDatePart::Day,
        _ => return Err(FallbackReason::UnsupportedExpression),
    };
    Ok(NativeExpr::extract(field, lower_physical_expr(expr, input_schema)?))
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

fn validate_modulo_operand_type(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if matches!(
        data_type,
        arrow::datatypes::DataType::Int8
            | arrow::datatypes::DataType::Int16
            | arrow::datatypes::DataType::Int32
            | arrow::datatypes::DataType::Int64
            | arrow::datatypes::DataType::UInt8
            | arrow::datatypes::DataType::UInt16
            | arrow::datatypes::DataType::UInt32
            | arrow::datatypes::DataType::UInt64
    ) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

pub(super) fn resolved_column_name<'a>(
    column: &Column,
    input_schema: &'a SchemaRef,
) -> Result<&'a str, FallbackReason> {
    input_schema.field_with_name(column.name()).map(|field| field.name().as_str()).or_else(|_error| {
        input_schema
            .fields()
            .get(column.index())
            .map(|field| field.name().as_str())
            .ok_or(FallbackReason::UnsupportedExpression)
    })
}

pub(super) fn lower_literal(value: &ScalarValue) -> Result<NativeExpr, FallbackReason> {
    Ok(NativeExpr::literal(native_literal_value(value, NativeLiteralContext::Expression)?))
}

pub(super) fn lower_binary_operator(op: &Operator) -> Result<NativeBinaryOp, FallbackReason> {
    match op {
        Operator::Plus => Ok(NativeBinaryOp::Add),
        Operator::Minus => Ok(NativeBinaryOp::Subtract),
        Operator::Multiply => Ok(NativeBinaryOp::Multiply),
        Operator::Divide => Ok(NativeBinaryOp::Divide),
        Operator::Modulo => Ok(NativeBinaryOp::Modulo),
        Operator::Eq => Ok(NativeBinaryOp::Eq),
        Operator::NotEq => Ok(NativeBinaryOp::NotEq),
        Operator::Lt => Ok(NativeBinaryOp::Lt),
        Operator::LtEq => Ok(NativeBinaryOp::LtEq),
        Operator::Gt => Ok(NativeBinaryOp::Gt),
        Operator::GtEq => Ok(NativeBinaryOp::GtEq),
        Operator::And => Ok(NativeBinaryOp::And),
        Operator::Or => Ok(NativeBinaryOp::Or),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn literal_string(expr: &Arc<dyn PhysicalExpr>) -> Result<String, FallbackReason> {
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

pub(super) fn alias_if_needed(expr: NativeExpr, alias: &str) -> NativeExpr {
    match &expr {
        NativeExpr::Column(column) if column.display_name() == alias || column.name() == alias => expr,
        _ => expr.alias(alias),
    }
}
