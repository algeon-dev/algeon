use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef};
use cudf_query_engine::{is_supported_arrow_cast, is_supported_arrow_type};
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
    physical_plan::{
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

use super::literal::{NativeLiteralContext, validate_native_literal};
use crate::policy::FallbackReason;

pub(super) fn validate_schema(schema: SchemaRef) -> Result<(), FallbackReason> {
    for field in schema.fields() {
        if !is_supported_native_type(field.data_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
    }
    Ok(())
}

pub(super) fn is_supported_native_type(data_type: &DataType) -> bool {
    is_supported_arrow_type(data_type)
}

pub(super) fn is_supported_native_cast(source_type: &DataType, target_type: &DataType) -> bool {
    is_supported_arrow_cast(source_type, target_type)
}

pub(super) fn validate_aggregate(
    aggregate: &AggregateExec,
    aggregate_args_must_be_columns: bool,
    allow_partial: bool,
) -> Result<(), FallbackReason> {
    if !(matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned)
        || allow_partial && matches!(aggregate.mode(), AggregateMode::Partial))
    {
        return Err(FallbackReason::UnsupportedOperator);
    }
    let group_by = aggregate.group_expr();
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if !group_by.expr().is_empty() && !group_by.is_single() {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if !group_by.expr().is_empty()
        && (group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null))
    {
        return Err(FallbackReason::UnsupportedOperator);
    }
    for aggregate_expr in aggregate.aggr_expr() {
        if aggregate_expr.is_distinct() || !aggregate_expr.order_bys().is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        if !is_supported_native_aggregate_function(aggregate_expr.fun().name()) {
            return Err(FallbackReason::UnsupportedOperator);
        }
        let expressions = aggregate_expr.expressions();
        let [arg] = expressions.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        if aggregate_args_must_be_columns
            && !aggregate_expr.fun().name().eq_ignore_ascii_case("count")
            && arg.as_any().downcast_ref::<Column>().is_none()
        {
            return Err(FallbackReason::UnsupportedExpression);
        }
        validate_physical_expr(arg, &aggregate.input().schema())?;
    }
    for (group_expr, _alias) in group_by.expr() {
        validate_physical_expr(group_expr, &aggregate.input().schema())?;
    }
    Ok(())
}

pub(super) fn validate_sort(sort: &SortExec) -> Result<(), FallbackReason> {
    if sort.expr().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if sort.expr().iter().any(|sort_expr| sort_expr.expr.as_any().downcast_ref::<Column>().is_none()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(())
}

pub(super) fn native_zero_column_count_source_is_safe(
    filter: Option<&FilterExec>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    aggregate: &AggregateExec,
) -> bool {
    filter.is_none() && pre_aggregate_projection.is_none() && aggregate_is_count_star_only(aggregate)
}

pub(super) fn aggregate_is_count_star_only(aggregate: &AggregateExec) -> bool {
    if !aggregate.group_expr().expr().is_empty()
        || !aggregate.filter_expr().iter().all(Option::is_none)
        || aggregate.aggr_expr().len() != 1
    {
        return false;
    }
    let expr = &aggregate.aggr_expr()[0];
    if !expr.fun().name().eq_ignore_ascii_case("count") || expr.is_distinct() || !expr.order_bys().is_empty() {
        return false;
    }
    let expressions = expr.expressions();
    let [arg] = expressions.as_slice() else {
        return false;
    };
    let mut required_columns = BTreeSet::new();
    collect_required_columns(arg, &aggregate.input().schema(), &mut required_columns).is_ok()
        && required_columns.is_empty()
}

fn is_supported_native_aggregate_function(name: &str) -> bool {
    name.eq_ignore_ascii_case("sum")
        || name.eq_ignore_ascii_case("min")
        || name.eq_ignore_ascii_case("max")
        || name.eq_ignore_ascii_case("count")
        || name.eq_ignore_ascii_case("avg")
}

pub(super) fn validate_physical_expr(
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

pub(super) fn collect_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        required.insert(resolved_column_name(column, input_schema)?.to_string());
        return Ok(());
    }
    if expr.as_any().downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        collect_required_columns(binary.left(), input_schema, required)?;
        collect_required_columns(binary.right(), input_schema, required)?;
        return Ok(());
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        collect_required_columns(not.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        collect_required_columns(is_null.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        collect_required_columns(is_not_null.arg(), input_schema, required)?;
        return Ok(());
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        collect_required_columns(cast.expr(), input_schema, required)?;
        return Ok(());
    }
    if let Some(like) = expr.as_any().downcast_ref::<LikeExpr>() {
        validate_literal_string(like.pattern())?;
        collect_required_columns(like.expr(), input_schema, required)?;
        return Ok(());
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        collect_required_columns(in_list.expr(), input_schema, required)?;
        for expr in in_list.list() {
            let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            if literal.value().is_null() {
                return Err(FallbackReason::UnsupportedExpression);
            }
            collect_required_columns(expr, input_schema, required)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
        if let Some(expr) = case.expr() {
            collect_required_columns(expr, input_schema, required)?;
        }
        for (when, then) in case.when_then_expr() {
            collect_required_columns(when, input_schema, required)?;
            collect_required_columns(then, input_schema, required)?;
        }
        if let Some(else_expr) = case.else_expr() {
            collect_required_columns(else_expr, input_schema, required)?;
        }
        return Ok(());
    }
    if let Some(function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        collect_required_columns_for_scalar_function(function, input_schema, required)?;
        return Ok(());
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn collect_required_columns_for_scalar_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    match function.name().to_ascii_lowercase().as_str() {
        "substring" | "substr" => {
            let args = function.args();
            validate_substring_arguments_contract(function)?;
            collect_required_columns(&args[0], input_schema, required)?;
            Ok(())
        },
        "date_part" | "extract" => {
            let [field, expr] = function.args() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            match literal_string(field)?.to_ascii_lowercase().as_str() {
                "year" | "month" | "day" => {},
                _ => return Err(FallbackReason::UnsupportedExpression),
            }
            collect_required_columns(expr, input_schema, required)
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn resolved_column_name<'a>(column: &Column, input_schema: &'a SchemaRef) -> Result<&'a str, FallbackReason> {
    input_schema.field_with_name(column.name()).map(|field| field.name().as_str()).or_else(|_error| {
        input_schema
            .fields()
            .get(column.index())
            .map(|field| field.name().as_str())
            .ok_or(FallbackReason::UnsupportedExpression)
    })
}

fn validate_literal_string(expr: &Arc<dyn PhysicalExpr>) -> Result<(), FallbackReason> {
    literal_string(expr).map(|_| ())
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

fn validate_substring_arguments_contract(function: &ScalarFunctionExpr) -> Result<(), FallbackReason> {
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
