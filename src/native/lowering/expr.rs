use std::sync::Arc;

use arrow::datatypes::{DataType as ArrowDataType, Field, SchemaRef, TimeUnit};
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
use nexus_query_engine::expr::{
    BinaryOp as NativeBinaryOp, CaseWhen as NativeCaseWhen, DatePart as NativeDatePart, Expr as NativeExpr,
};

use crate::{
    native::{
        literal::{
            NativeLiteralContext, literal_i32, literal_string, native_literal_value, parse_yyyy_mm_dd_date32,
            substring_stop_position,
        },
        lowering::FieldBinding,
        validate::{
            bounded_nonnegative_integral_cast_is_supported, is_list_carry_type, is_supported_native_cast,
            is_supported_native_type,
        },
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
    lower_physical_expr_with_bindings(expr, input_schema, None)
}

pub(super) fn lower_physical_expr_with_bindings(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        validate_column_type(column, input_schema)?;
        return Ok(NativeExpr::column(resolved_column_name(column, input_schema, bindings)?));
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower_physical_expr_with_bindings(binary.left(), input_schema, bindings)?;
        let right = lower_physical_expr_with_bindings(binary.right(), input_schema, bindings)?;
        if matches!(binary.op(), Operator::StringConcat) {
            let output_type =
                binary.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
            if !is_supported_native_type(&output_type) {
                return Err(FallbackReason::UnsupportedType);
            }
            let null_guard = NativeExpr::binary(
                NativeExpr::is_null(left.clone()),
                NativeBinaryOp::Or,
                NativeExpr::is_null(right.clone()),
            );
            return Ok(NativeExpr::case(
                vec![NativeCaseWhen::new(
                    null_guard,
                    NativeExpr::literal(nexus_query_engine::expr::LiteralValue::Null(output_type)),
                )],
                NativeExpr::concat(vec![left, right]),
            ));
        }
        if matches!(binary.op(), Operator::IsDistinctFrom) {
            return Ok(NativeExpr::is_distinct_from(left, right));
        }
        if matches!(binary.op(), Operator::IsNotDistinctFrom) {
            return Ok(NativeExpr::is_not_distinct_from(left, right));
        }
        let op = lower_binary_operator(binary.op())?;
        if matches!(binary.op(), Operator::Divide)
            && let Some(declared_type) = declared_decimal_divide_output_type(binary, input_schema)?
        {
            return Ok(NativeExpr::binary_with_output_type(left, op, right, declared_type));
        }
        // libcudf leaves integer divide/modulo by zero undefined; keep this
        // lowering-wide CPU-fallback backstop for paths that skip validation.
        if matches!(binary.op(), Operator::Divide | Operator::Modulo)
            && binary_operands_are_integral(binary, input_schema)?
        {
            return Err(FallbackReason::UnsupportedExpression);
        }
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return Ok(!lower_physical_expr_with_bindings(not.arg(), input_schema, bindings)?);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_physical_expr_with_bindings(is_null.arg(), input_schema, bindings)?));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_physical_expr_with_bindings(
            is_not_null.arg(),
            input_schema,
            bindings,
        )?));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if let Some(value) = lower_literal_string_date32_cast(cast.expr(), cast.cast_type()) {
            return Ok(NativeExpr::literal(value));
        }
        if !is_supported_native_cast(&source_type, cast.cast_type())
            && !bounded_nonnegative_integral_cast_is_supported(cast.expr(), &source_type, cast.cast_type())
        {
            return Err(FallbackReason::UnsupportedType);
        }
        if source_type == *cast.cast_type() && is_list_carry_type(&source_type) {
            return lower_physical_expr_with_bindings(cast.expr(), input_schema, bindings);
        }
        return Ok(NativeExpr::cast(
            lower_physical_expr_with_bindings(cast.expr(), input_schema, bindings)?,
            cast.cast_type().clone(),
        ));
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        return lower_like_expr(like, input_schema, bindings);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        return lower_in_list_expr(in_list, input_schema, bindings);
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return lower_case_expr(case, input_schema, bindings);
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| FallbackReason::UnsupportedExpression)?;
        return lower_physical_expr_with_bindings(&current, input_schema, bindings);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return lower_scalar_function_expr(function, input_schema, bindings);
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return lower_try_cast_expr(try_cast, input_schema, bindings);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_try_cast_expr(
    try_cast: &TryCastExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let source_type =
        try_cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_try_cast_pair(&source_type, try_cast.cast_type()) {
        return Err(FallbackReason::UnsupportedType);
    }
    let inner = lower_physical_expr_with_bindings(try_cast.expr(), input_schema, bindings)?;
    Ok(NativeExpr::try_cast(inner, try_cast.cast_type().clone()))
}

fn is_supported_try_cast_pair(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(source_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
        && matches!(
            target_type,
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

fn lower_literal_string_date32_cast(
    expr: &Arc<dyn PhysicalExpr>,
    target_type: &ArrowDataType,
) -> Option<nexus_query_engine::expr::LiteralValue> {
    if !matches!(target_type, ArrowDataType::Date32) {
        return None;
    }
    expr.downcast_ref::<Literal>().and_then(|literal| match literal.value() {
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            parse_yyyy_mm_dd_date32(value).map(nexus_query_engine::expr::LiteralValue::Date32)
        },
        _ => None,
    })
}

fn lower_like_expr(
    like: &LikeExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let pattern = literal_string(like.pattern())?;
    if like.case_insensitive() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let expr = lower_physical_expr_with_bindings(like.expr(), input_schema, bindings)?;
    Ok(if like.negated() {
        NativeExpr::not_like_escape(expr, pattern, "\\")
    } else {
        NativeExpr::like_escape(expr, pattern, "\\")
    })
}

fn lower_in_list_expr(
    in_list: &InListExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    if in_list.is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let expr = lower_physical_expr_with_bindings(in_list.expr(), input_schema, bindings)?;
    let list = in_list
        .list()
        .iter()
        .map(|expr| {
            let Some(literal) = expr.downcast_ref::<Literal>() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            if literal.value().is_null() {
                return Err(FallbackReason::UnsupportedExpression);
            }
            lower_physical_expr_with_bindings(expr, input_schema, bindings)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if in_list.negated() { NativeExpr::not_in_list(expr, list) } else { NativeExpr::in_list(expr, list) })
}

fn lower_case_expr(
    case: &CaseExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let base_expr =
        case.expr().map(|expr| lower_physical_expr_with_bindings(expr, input_schema, bindings)).transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => NativeExpr::binary(
                    base_expr.clone(),
                    NativeBinaryOp::Eq,
                    lower_physical_expr_with_bindings(when, input_schema, bindings)?,
                ),
                None => lower_physical_expr_with_bindings(when, input_schema, bindings)?,
            };
            Ok(NativeCaseWhen::new(when, lower_physical_expr_with_bindings(then, input_schema, bindings)?))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_physical_expr_with_bindings(else_expr, input_schema, bindings)?,
        None => {
            let data_type =
                case.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
            NativeExpr::literal(nexus_query_engine::expr::LiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

fn lower_scalar_function_expr(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    match function.name().to_ascii_lowercase().as_str() {
        "substring" | "substr" => lower_substring_function(function, input_schema, bindings),
        "date_part" | "extract" => lower_extract_function(function, input_schema, bindings),
        "date_trunc" => lower_date_trunc_function(function, input_schema, bindings),
        "to_timestamp_seconds" => lower_to_timestamp_seconds_function(function, input_schema, bindings),
        "lower" => lower_string_case_function(function, input_schema, bindings, NativeExpr::lower),
        "upper" => lower_string_case_function(function, input_schema, bindings, NativeExpr::upper),
        "trim" | "btrim" => lower_trim_function(function, input_schema, bindings, NativeExpr::trim),
        "ltrim" => lower_trim_function(function, input_schema, bindings, NativeExpr::ltrim),
        "rtrim" => lower_trim_function(function, input_schema, bindings, NativeExpr::rtrim),
        "contains" => lower_string_predicate_function(function, input_schema, bindings, NativeExpr::string_contains),
        "starts_with" => lower_string_predicate_function(function, input_schema, bindings, NativeExpr::starts_with),
        "ends_with" => lower_string_predicate_function(function, input_schema, bindings, NativeExpr::ends_with),
        "character_length" | "length" => lower_string_length_function(function, input_schema, bindings),
        "regexp_replace" => lower_regexp_replace_function(function, input_schema, bindings),
        "concat" => lower_concat_function(function, input_schema, bindings),
        "abs" => lower_abs_function(function, input_schema, bindings),
        "sqrt" => lower_sqrt_function(function, input_schema, bindings),
        "round" => lower_round_function(function, input_schema, bindings),
        "nullif" => lower_nullif_function(function, input_schema, bindings),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn lower_substring_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
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
    Ok(NativeExpr::substring(lower_physical_expr_with_bindings(&args[0], input_schema, bindings)?, start, length))
}

fn lower_extract_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [field, expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let field = match literal_string(field)?.to_ascii_lowercase().as_str() {
        "year" => NativeDatePart::Year,
        "month" => NativeDatePart::Month,
        "day" => NativeDatePart::Day,
        "minute" => NativeDatePart::Minute,
        _ => return Err(FallbackReason::UnsupportedExpression),
    };
    Ok(NativeExpr::extract(field, lower_physical_expr_with_bindings(expr, input_schema, bindings)?))
}

fn lower_date_trunc_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [field, expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let field = lower_date_trunc_field(&literal_string(field)?.to_ascii_lowercase())?;
    let input_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !matches!(
        (&input_type, &output_type),
        (ArrowDataType::Timestamp(input_unit, None), ArrowDataType::Timestamp(output_unit, None))
            if input_unit == output_unit
    ) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(NativeExpr::date_trunc(field, lower_physical_expr_with_bindings(expr, input_schema, bindings)?, output_type))
}

fn lower_date_trunc_field(field: &str) -> Result<NativeDatePart, FallbackReason> {
    Ok(match field {
        "day" => NativeDatePart::Day,
        "hour" => NativeDatePart::Hour,
        "minute" => NativeDatePart::Minute,
        "second" => NativeDatePart::Second,
        "millisecond" => NativeDatePart::Millisecond,
        "microsecond" => NativeDatePart::Microsecond,
        _ => return Err(FallbackReason::UnsupportedExpression),
    })
}

fn lower_to_timestamp_seconds_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !matches!(output_type, ArrowDataType::Timestamp(TimeUnit::Second, None)) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(NativeExpr::cast(lower_physical_expr_with_bindings(expr, input_schema, bindings)?, output_type))
}

fn lower_string_case_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
    constructor: fn(NativeExpr) -> NativeExpr,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    Ok(constructor(lower_physical_expr_with_bindings(expr, input_schema, bindings)?))
}

fn lower_trim_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
    constructor: fn(NativeExpr, String) -> NativeExpr,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    if !(1..=2).contains(&args.len()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let trim_chars = if args.len() == 2 { literal_string(&args[1])? } else { " ".to_string() };
    let expr = lower_physical_expr_with_bindings(&args[0], input_schema, bindings)?;
    if trim_chars.is_empty() { Ok(expr) } else { Ok(constructor(expr, trim_chars)) }
}

fn lower_string_predicate_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
    constructor: fn(NativeExpr, String) -> NativeExpr,
) -> Result<NativeExpr, FallbackReason> {
    let [expr, pattern] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    Ok(constructor(lower_physical_expr_with_bindings(expr, input_schema, bindings)?, literal_string(pattern)?))
}

fn lower_string_length_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&output_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(NativeExpr::string_length(lower_physical_expr_with_bindings(expr, input_schema, bindings)?, output_type))
}

fn lower_regexp_replace_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr, pattern, replacement] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&output_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(NativeExpr::regexp_replace(
        lower_physical_expr_with_bindings(expr, input_schema, bindings)?,
        literal_string(pattern)?,
        literal_string(replacement)?,
        output_type,
    ))
}

fn lower_concat_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    if function.args().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    function
        .args()
        .iter()
        .map(|expr| lower_physical_expr_with_bindings(expr, input_schema, bindings))
        .collect::<Result<Vec<_>, _>>()
        .map(NativeExpr::concat)
}

fn lower_abs_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    Ok(NativeExpr::abs(lower_physical_expr_with_bindings(expr, input_schema, bindings)?))
}

fn lower_sqrt_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&output_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(NativeExpr::sqrt(lower_physical_expr_with_bindings(expr, input_schema, bindings)?, output_type))
}

fn lower_round_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    if !(1..=2).contains(&args.len()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let decimal_places = if args.len() == 2 { literal_i32(&args[1])? } else { 0 };
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&output_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(NativeExpr::round(
        lower_physical_expr_with_bindings(&args[0], input_schema, bindings)?,
        decimal_places,
        output_type,
    ))
}

fn lower_nullif_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [left, right] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let output_type =
        function.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&output_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    let left = lower_physical_expr_with_bindings(left, input_schema, bindings)?;
    let right = lower_physical_expr_with_bindings(right, input_schema, bindings)?;
    Ok(NativeExpr::case(
        vec![NativeCaseWhen::new(
            NativeExpr::is_not_distinct_from(left.clone(), right),
            NativeExpr::literal(nexus_query_engine::expr::LiteralValue::Null(output_type)),
        )],
        left,
    ))
}

fn validate_column_type(column: &Column, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let data_type = input_schema
        .fields()
        .get(column.index())
        .filter(|field| field.name() == column.name())
        .map(|field| field.data_type())
        .or_else(|| input_schema.field_with_name(column.name()).ok().map(|field| field.data_type()))
        .or_else(|| input_schema.fields().get(column.index()).map(|field| field.data_type()))
        .ok_or(FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(data_type) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

pub(super) fn resolved_column_name<'a>(
    column: &Column,
    input_schema: &'a SchemaRef,
    bindings: Option<&'a [FieldBinding]>,
) -> Result<&'a str, FallbackReason> {
    if let Some(bindings) = bindings {
        return bindings
            .iter()
            .find(|binding| binding.frontend_index == column.index() && binding.frontend_name == column.name())
            .or_else(|| bindings.iter().find(|binding| binding.frontend_index == column.index()))
            .or_else(|| bindings.iter().find(|binding| binding.frontend_name == column.name()))
            .map(|binding| binding.native_name.as_str())
            .ok_or(FallbackReason::UnsupportedExpression);
    }
    input_schema
        .fields()
        .get(column.index())
        .filter(|field| field.name() == column.name())
        .map(|field| field.name().as_str())
        .or_else(|| input_schema.field_with_name(column.name()).ok().map(|field| field.name().as_str()))
        .or_else(|| input_schema.fields().get(column.index()).map(|field| field.name().as_str()))
        .ok_or(FallbackReason::UnsupportedExpression)
}

pub(super) fn collect_required_field_indices(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    required: &mut std::collections::BTreeSet<usize>,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        if input_schema.fields().get(column.index()).is_none() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        required.insert(column.index());
        return Ok(());
    }
    if expr.downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        collect_required_field_indices(binary.left(), input_schema, required)?;
        collect_required_field_indices(binary.right(), input_schema, required)?;
        return Ok(());
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return collect_required_field_indices(not.arg(), input_schema, required);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return collect_required_field_indices(is_null.arg(), input_schema, required);
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return collect_required_field_indices(is_not_null.arg(), input_schema, required);
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return collect_required_field_indices(cast.expr(), input_schema, required);
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        literal_string(like.pattern())?;
        return collect_required_field_indices(like.expr(), input_schema, required);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        collect_required_field_indices(in_list.expr(), input_schema, required)?;
        for expr in in_list.list() {
            let Some(literal) = expr.downcast_ref::<Literal>() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            if literal.value().is_null() {
                return Err(FallbackReason::UnsupportedExpression);
            }
            collect_required_field_indices(expr, input_schema, required)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if let Some(expr) = case.expr() {
            collect_required_field_indices(expr, input_schema, required)?;
        }
        for (when, then) in case.when_then_expr() {
            collect_required_field_indices(when, input_schema, required)?;
            collect_required_field_indices(then, input_schema, required)?;
        }
        if let Some(else_expr) = case.else_expr() {
            collect_required_field_indices(else_expr, input_schema, required)?;
        }
        return Ok(());
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| FallbackReason::UnsupportedExpression)?;
        return collect_required_field_indices(&current, input_schema, required);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        match function.name().to_ascii_lowercase().as_str() {
            "substring" | "substr" => {
                let args = function.args();
                if !(2..=3).contains(&args.len()) {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                collect_required_field_indices(&args[0], input_schema, required)
            },
            "date_part" | "extract" => {
                let [field, expr] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                match literal_string(field)?.to_ascii_lowercase().as_str() {
                    "year" | "month" | "day" | "minute" => {},
                    _ => return Err(FallbackReason::UnsupportedExpression),
                }
                collect_required_field_indices(expr, input_schema, required)
            },
            "date_trunc" => {
                let [field, expr] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                lower_date_trunc_field(&literal_string(field)?.to_ascii_lowercase())?;
                collect_required_field_indices(expr, input_schema, required)
            },
            "to_timestamp_seconds" => {
                let [expr] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                collect_required_field_indices(expr, input_schema, required)
            },
            "lower" | "upper" | "character_length" | "length" => {
                let [expr] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                collect_required_field_indices(expr, input_schema, required)
            },
            "contains" | "starts_with" | "ends_with" => {
                let [expr, pattern] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                literal_string(pattern)?;
                collect_required_field_indices(expr, input_schema, required)
            },
            "trim" | "btrim" | "ltrim" | "rtrim" => {
                let args = function.args();
                if !(1..=2).contains(&args.len()) {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                if args.len() == 2 {
                    literal_string(&args[1])?;
                }
                collect_required_field_indices(&args[0], input_schema, required)
            },
            "regexp_replace" => {
                let [expr, pattern, replacement] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                literal_string(pattern)?;
                literal_string(replacement)?;
                collect_required_field_indices(expr, input_schema, required)
            },
            "concat" => {
                if function.args().is_empty() {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                for expr in function.args() {
                    collect_required_field_indices(expr, input_schema, required)?;
                }
                Ok(())
            },
            "abs" => {
                let [expr] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                collect_required_field_indices(expr, input_schema, required)
            },
            "sqrt" => {
                let [expr] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                collect_required_field_indices(expr, input_schema, required)
            },
            "round" => {
                let args = function.args();
                if !(1..=2).contains(&args.len()) {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                if args.len() == 2 {
                    literal_i32(&args[1])?;
                }
                collect_required_field_indices(&args[0], input_schema, required)
            },
            "nullif" => {
                let [left, right] = function.args() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                collect_required_field_indices(left, input_schema, required)?;
                collect_required_field_indices(right, input_schema, required)
            },
            _ => Err(FallbackReason::UnsupportedExpression),
        }?;
        return Ok(());
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return collect_required_field_indices(try_cast.expr(), input_schema, required);
    }
    Err(FallbackReason::UnsupportedExpression)
}

pub(super) fn lower_literal(value: &ScalarValue) -> Result<NativeExpr, FallbackReason> {
    Ok(NativeExpr::literal(native_literal_value(value, NativeLiteralContext::Expression)?))
}

fn binary_operands_are_integral(binary: &BinaryExpr, input_schema: &SchemaRef) -> Result<bool, FallbackReason> {
    let left_type =
        binary.left().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    let right_type =
        binary.right().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    Ok(is_integral_arrow_type(&left_type) && is_integral_arrow_type(&right_type))
}

fn is_integral_arrow_type(data_type: &ArrowDataType) -> bool {
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

fn declared_decimal_divide_output_type(
    binary: &BinaryExpr,
    input_schema: &SchemaRef,
) -> Result<Option<ArrowDataType>, FallbackReason> {
    let left_type =
        binary.left().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    let right_type =
        binary.right().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !matches!(left_type, ArrowDataType::Decimal128(_, _)) && !matches!(right_type, ArrowDataType::Decimal128(_, _)) {
        return Ok(None);
    }
    let declared_type =
        binary.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !matches!(declared_type, ArrowDataType::Decimal128(_, _)) {
        return Ok(None);
    }
    if !is_supported_native_type(&declared_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(Some(declared_type))
}

pub(super) fn lower_binary_operator(op: &Operator) -> Result<NativeBinaryOp, FallbackReason> {
    match op {
        Operator::Plus => Ok(NativeBinaryOp::Add),
        Operator::Minus => Ok(NativeBinaryOp::Subtract),
        Operator::Multiply => Ok(NativeBinaryOp::Multiply),
        Operator::Divide => Ok(NativeBinaryOp::Divide),
        Operator::Modulo => Ok(NativeBinaryOp::Modulo),
        Operator::BitwiseAnd => Ok(NativeBinaryOp::BitwiseAnd),
        Operator::BitwiseOr => Ok(NativeBinaryOp::BitwiseOr),
        Operator::BitwiseXor => Ok(NativeBinaryOp::BitwiseXor),
        Operator::BitwiseShiftLeft => Ok(NativeBinaryOp::ShiftLeft),
        Operator::BitwiseShiftRight => Ok(NativeBinaryOp::ShiftRight),
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

pub(super) fn alias_if_needed(expr: NativeExpr, alias: &str) -> NativeExpr {
    match &expr {
        NativeExpr::Column(column) if column.display_name() == alias || column.name() == alias => expr,
        _ => expr.alias(alias),
    }
}
