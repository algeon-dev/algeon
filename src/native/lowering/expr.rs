use std::sync::Arc;

use arrow::datatypes::{DataType as ArrowDataType, Field, SchemaRef};
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
        expression_contract::{
            ResolvedColumn, ScalarFunctionKind, SupportedBinaryOperator, ValidatedPhysicalExpr,
            declared_decimal_divide_output_type, is_list_carry_type, is_supported_native_type, physical_expr_data_type,
            resolve_column, validate_physical_expr,
        },
        literal::{
            NativeLiteralContext, literal_i32, literal_string, native_literal_value, parse_yyyy_mm_dd_date32,
            substring_stop_position,
        },
        lowering::FieldBinding,
    },
    policy::FallbackReason,
};

pub(super) fn physical_expr_field(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    name: &str,
) -> Result<Field, FallbackReason> {
    let validated = validate_physical_expr(expr, input_schema)?;
    physical_expr_field_from_validated(&validated, name)
}

pub(super) fn physical_expr_field_from_validated(
    validated: &ValidatedPhysicalExpr,
    name: &str,
) -> Result<Field, FallbackReason> {
    let expr = validated.expr();
    let input_schema = validated.input_schema();
    let data_type =
        physical_expr_data_type(expr, input_schema).expect("validated expression must have a resolved output type");
    debug_assert!(is_supported_native_type(&data_type), "validated expression output must be a native type");
    let nullable = expr.nullable(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    Ok(Field::new(name, data_type, nullable))
}

pub(in crate::native) fn lower_physical_expr(
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
    let validated = validate_physical_expr(expr, input_schema)?;
    lower_validated_physical_expr_with_bindings(&validated, bindings)
}

pub(super) fn lower_validated_physical_expr_with_bindings(
    validated: &crate::native::expression_contract::ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let expr = validated.expr();
    let input_schema = validated.input_schema();
    if let Some(column) = expr.downcast_ref::<Column>() {
        return Ok(NativeExpr::column(resolved_column_name(column, input_schema, bindings)?));
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return Ok(lower_literal(literal.value()).expect("validated literal must lower"));
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower_validated_child_expr_with_bindings(binary.left(), validated, bindings)?;
        let right = lower_validated_child_expr_with_bindings(binary.right(), validated, bindings)?;
        if matches!(binary.op(), Operator::StringConcat) {
            let output_type =
                physical_expr_data_type(expr, input_schema).expect("validated concatenation must have an output type");
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
        let op = lower_binary_operator(binary.op());
        if matches!(binary.op(), Operator::Divide)
            && let Some(declared_type) = declared_decimal_divide_output_type(binary, input_schema)
                .expect("validated decimal division must have a supported output type")
        {
            return Ok(NativeExpr::binary_with_output_type(left, op, right, declared_type));
        }
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return Ok(!lower_validated_child_expr_with_bindings(not.arg(), validated, bindings)?);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_validated_child_expr_with_bindings(is_null.arg(), validated, bindings)?));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_validated_child_expr_with_bindings(
            is_not_null.arg(),
            validated,
            bindings,
        )?));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let source_type = physical_expr_data_type(cast.expr(), input_schema)
            .expect("validated cast source must have a resolved type");
        if let Some(value) = lower_literal_string_date32_cast(cast.expr(), cast.cast_type()) {
            return Ok(NativeExpr::literal(value));
        }
        if source_type == *cast.cast_type() && is_list_carry_type(&source_type) {
            return lower_validated_child_expr_with_bindings(cast.expr(), validated, bindings);
        }
        return Ok(NativeExpr::cast(
            lower_validated_child_expr_with_bindings(cast.expr(), validated, bindings)?,
            cast.cast_type().clone(),
        ));
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        return lower_like_expr(like, validated, bindings);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        return lower_in_list_expr(in_list, validated, bindings);
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return lower_case_expr(case, validated, bindings);
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| FallbackReason::UnsupportedExpression)?;
        // Dynamic filters are mutable. Validate the current snapshot before
        // deriving a new lowering proof rather than trusting the older root.
        let current = validate_physical_expr(&current, input_schema)?;
        return lower_validated_physical_expr_with_bindings(&current, bindings);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return lower_scalar_function_expr(function, validated, bindings);
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return lower_try_cast_expr(try_cast, validated, bindings);
    }
    unreachable!("validated physical expression must have a lowering branch")
}

fn lower_validated_child_expr_with_bindings(
    expr: &Arc<dyn PhysicalExpr>,
    parent: &crate::native::expression_contract::ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let child = parent.trusted_descendant(expr);
    lower_validated_physical_expr_with_bindings(&child, bindings)
}

fn lower_try_cast_expr(
    try_cast: &TryCastExpr,
    validated: &crate::native::expression_contract::ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let inner = lower_validated_child_expr_with_bindings(try_cast.expr(), validated, bindings)?;
    Ok(NativeExpr::try_cast(inner, try_cast.cast_type().clone()))
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
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let pattern = literal_string(like.pattern()).expect("validated LIKE pattern must be a string literal");
    debug_assert!(!like.case_insensitive(), "case-insensitive LIKE cannot carry a validation proof");
    let expr = lower_validated_child_expr_with_bindings(like.expr(), validated, bindings)?;
    Ok(if like.negated() {
        NativeExpr::not_like_escape(expr, pattern, "\\")
    } else {
        NativeExpr::like_escape(expr, pattern, "\\")
    })
}

fn lower_in_list_expr(
    in_list: &InListExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    debug_assert!(!in_list.is_empty(), "an empty IN list cannot carry a validation proof");
    let expr = lower_validated_child_expr_with_bindings(in_list.expr(), validated, bindings)?;
    let list = in_list
        .list()
        .iter()
        .map(|expr| {
            let literal = expr.downcast_ref::<Literal>().expect("validated IN-list entries must be literals");
            debug_assert!(!literal.value().is_null(), "null IN-list entries cannot carry a validation proof");
            lower_validated_child_expr_with_bindings(expr, validated, bindings)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if in_list.negated() { NativeExpr::not_in_list(expr, list) } else { NativeExpr::in_list(expr, list) })
}

fn lower_case_expr(
    case: &CaseExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let input_schema = validated.input_schema();
    let base_expr =
        case.expr().map(|expr| lower_validated_child_expr_with_bindings(expr, validated, bindings)).transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => NativeExpr::binary(
                    base_expr.clone(),
                    NativeBinaryOp::Eq,
                    lower_validated_child_expr_with_bindings(when, validated, bindings)?,
                ),
                None => lower_validated_child_expr_with_bindings(when, validated, bindings)?,
            };
            Ok(NativeCaseWhen::new(when, lower_validated_child_expr_with_bindings(then, validated, bindings)?))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_validated_child_expr_with_bindings(else_expr, validated, bindings)?,
        None => {
            let data_type = physical_expr_data_type(validated.expr(), input_schema)
                .expect("validated CASE expression must have an output type");
            NativeExpr::literal(nexus_query_engine::expr::LiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

fn lower_scalar_function_expr(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let kind = ScalarFunctionKind::from_name(function.name()).expect("validated scalar function kind");
    match kind {
        ScalarFunctionKind::Substring => lower_substring_function(function, validated, bindings),
        ScalarFunctionKind::Extract => lower_extract_function(function, validated, bindings),
        ScalarFunctionKind::DateTrunc => lower_date_trunc_function(function, validated, bindings),
        ScalarFunctionKind::ToTimestampSeconds => lower_to_timestamp_seconds_function(function, validated, bindings),
        ScalarFunctionKind::Lower => lower_string_case_function(function, validated, bindings, NativeExpr::lower),
        ScalarFunctionKind::Upper => lower_string_case_function(function, validated, bindings, NativeExpr::upper),
        ScalarFunctionKind::Trim => lower_trim_function(function, validated, bindings, NativeExpr::trim),
        ScalarFunctionKind::LeftTrim => lower_trim_function(function, validated, bindings, NativeExpr::ltrim),
        ScalarFunctionKind::RightTrim => lower_trim_function(function, validated, bindings, NativeExpr::rtrim),
        ScalarFunctionKind::Contains => {
            lower_string_predicate_function(function, validated, bindings, NativeExpr::string_contains)
        },
        ScalarFunctionKind::StartsWith => {
            lower_string_predicate_function(function, validated, bindings, NativeExpr::starts_with)
        },
        ScalarFunctionKind::EndsWith => {
            lower_string_predicate_function(function, validated, bindings, NativeExpr::ends_with)
        },
        ScalarFunctionKind::StringLength => lower_string_length_function(function, validated, bindings),
        ScalarFunctionKind::RegexpReplace => lower_regexp_replace_function(function, validated, bindings),
        ScalarFunctionKind::Concat => lower_concat_function(function, validated, bindings),
        ScalarFunctionKind::Abs => lower_abs_function(function, validated, bindings),
        ScalarFunctionKind::Sqrt => lower_sqrt_function(function, validated, bindings),
        ScalarFunctionKind::Round => lower_round_function(function, validated, bindings),
        ScalarFunctionKind::NullIf => lower_nullif_function(function, validated, bindings),
    }
}

fn lower_substring_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    debug_assert!((2..=3).contains(&args.len()), "invalid SUBSTRING arity cannot carry a validation proof");
    let start = literal_i32(&args[1]).expect("validated SUBSTRING start must be an i32 literal");
    debug_assert!(start >= 1, "invalid SUBSTRING start cannot carry a validation proof");
    let length = if args.len() == 3 {
        Some(literal_i32(&args[2]).expect("validated SUBSTRING length must be an i32 literal"))
    } else {
        None
    };
    debug_assert!(
        length.is_none_or(|length| length >= 0 && substring_stop_position(start, length).is_some()),
        "invalid SUBSTRING length cannot carry a validation proof"
    );
    Ok(NativeExpr::substring(lower_validated_child_expr_with_bindings(&args[0], validated, bindings)?, start, length))
}

fn lower_extract_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [field, expr] = function.args() else { unreachable!("validated EXTRACT must have two arguments") };
    let field = match literal_string(field)
        .expect("validated EXTRACT field must be a string literal")
        .to_ascii_lowercase()
        .as_str()
    {
        "year" => NativeDatePart::Year,
        "month" => NativeDatePart::Month,
        "day" => NativeDatePart::Day,
        "minute" => NativeDatePart::Minute,
        _ => unreachable!("validated EXTRACT field must be supported"),
    };
    Ok(NativeExpr::extract(field, lower_validated_child_expr_with_bindings(expr, validated, bindings)?))
}

fn lower_date_trunc_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [field, expr] = function.args() else { unreachable!("validated DATE_TRUNC must have two arguments") };
    let field = lower_validated_date_trunc_field(
        &literal_string(field).expect("validated DATE_TRUNC field must be a string literal").to_ascii_lowercase(),
    );
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::date_trunc(field, lower_validated_child_expr_with_bindings(expr, validated, bindings)?, output_type))
}

fn lower_validated_date_trunc_field(field: &str) -> NativeDatePart {
    match field {
        "day" => NativeDatePart::Day,
        "hour" => NativeDatePart::Hour,
        "minute" => NativeDatePart::Minute,
        "second" => NativeDatePart::Second,
        "millisecond" => NativeDatePart::Millisecond,
        "microsecond" => NativeDatePart::Microsecond,
        _ => unreachable!("validated DATE_TRUNC field must be supported"),
    }
}

fn lower_to_timestamp_seconds_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else { unreachable!("validated TO_TIMESTAMP_SECONDS must have one argument") };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::cast(lower_validated_child_expr_with_bindings(expr, validated, bindings)?, output_type))
}

fn lower_string_case_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
    constructor: fn(NativeExpr) -> NativeExpr,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else { unreachable!("validated string case function must have one argument") };
    Ok(constructor(lower_validated_child_expr_with_bindings(expr, validated, bindings)?))
}

fn lower_trim_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
    constructor: fn(NativeExpr, String) -> NativeExpr,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    debug_assert!((1..=2).contains(&args.len()), "invalid TRIM arity cannot carry a validation proof");
    let trim_chars = if args.len() == 2 {
        literal_string(&args[1]).expect("validated TRIM characters must be a string literal")
    } else {
        " ".to_string()
    };
    let expr = lower_validated_child_expr_with_bindings(&args[0], validated, bindings)?;
    if trim_chars.is_empty() { Ok(expr) } else { Ok(constructor(expr, trim_chars)) }
}

fn lower_string_predicate_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
    constructor: fn(NativeExpr, String) -> NativeExpr,
) -> Result<NativeExpr, FallbackReason> {
    let [expr, pattern] = function.args() else { unreachable!("validated string predicate must have two arguments") };
    Ok(constructor(
        lower_validated_child_expr_with_bindings(expr, validated, bindings)?,
        literal_string(pattern).expect("validated string predicate pattern must be a string literal"),
    ))
}

fn lower_string_length_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else { unreachable!("validated string length function must have one argument") };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::string_length(lower_validated_child_expr_with_bindings(expr, validated, bindings)?, output_type))
}

fn lower_regexp_replace_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr, pattern, replacement] = function.args() else {
        unreachable!("validated REGEXP_REPLACE must have three arguments")
    };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::regexp_replace(
        lower_validated_child_expr_with_bindings(expr, validated, bindings)?,
        literal_string(pattern).expect("validated REGEXP_REPLACE pattern must be a string literal"),
        literal_string(replacement).expect("validated REGEXP_REPLACE replacement must be a string literal"),
        output_type,
    ))
}

fn lower_concat_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    debug_assert!(!function.args().is_empty(), "empty CONCAT cannot carry a validation proof");
    function
        .args()
        .iter()
        .map(|expr| lower_validated_child_expr_with_bindings(expr, validated, bindings))
        .collect::<Result<Vec<_>, _>>()
        .map(NativeExpr::concat)
}

fn lower_abs_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else { unreachable!("validated ABS must have one argument") };
    Ok(NativeExpr::abs(lower_validated_child_expr_with_bindings(expr, validated, bindings)?))
}

fn lower_sqrt_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else { unreachable!("validated SQRT must have one argument") };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::sqrt(lower_validated_child_expr_with_bindings(expr, validated, bindings)?, output_type))
}

fn lower_round_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    debug_assert!((1..=2).contains(&args.len()), "invalid ROUND arity cannot carry a validation proof");
    let decimal_places = if args.len() == 2 {
        literal_i32(&args[1]).expect("validated ROUND precision must be an i32 literal")
    } else {
        0
    };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::round(
        lower_validated_child_expr_with_bindings(&args[0], validated, bindings)?,
        decimal_places,
        output_type,
    ))
}

fn lower_nullif_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, FallbackReason> {
    let [left, right] = function.args() else { unreachable!("validated NULLIF must have two arguments") };
    let output_type = validated_scalar_output_type(function, validated);
    let left = lower_validated_child_expr_with_bindings(left, validated, bindings)?;
    let right = lower_validated_child_expr_with_bindings(right, validated, bindings)?;
    Ok(NativeExpr::case(
        vec![NativeCaseWhen::new(
            NativeExpr::is_not_distinct_from(left.clone(), right),
            NativeExpr::literal(nexus_query_engine::expr::LiteralValue::Null(output_type)),
        )],
        left,
    ))
}

fn validated_scalar_output_type(function: &ScalarFunctionExpr, validated: &ValidatedPhysicalExpr) -> ArrowDataType {
    function.data_type(validated.input_schema().as_ref()).expect("validated scalar function must have an output type")
}

pub(super) fn resolved_column_name<'a>(
    column: &Column,
    input_schema: &'a SchemaRef,
    bindings: Option<&'a [FieldBinding]>,
) -> Result<&'a str, FallbackReason> {
    let resolved: ResolvedColumn<'_> = resolve_column(column, input_schema)?;
    if let Some(bindings) = bindings {
        return bindings
            .iter()
            .find(|binding| {
                binding.frontend_index == resolved.index()
                    && binding.frontend_name.as_str() == resolved.field().name().as_str()
            })
            .or_else(|| {
                let mut matching_bindings = bindings
                    .iter()
                    .filter(|binding| binding.frontend_name.as_str() == resolved.field().name().as_str());
                let binding = matching_bindings.next()?;
                matching_bindings.next().is_none().then_some(binding)
            })
            .or_else(|| bindings.iter().find(|binding| binding.frontend_index == resolved.index()))
            .map(|binding| binding.native_name.as_str())
            .ok_or(FallbackReason::UnsupportedExpression);
    }
    Ok(resolved.field().name())
}

pub(super) fn collect_required_field_indices(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    required: &mut std::collections::BTreeSet<usize>,
) -> Result<(), FallbackReason> {
    let validated = validate_physical_expr(expr, input_schema)?;
    collect_validated_required_field_indices(&validated, required)
}

fn collect_validated_required_field_indices(
    validated: &ValidatedPhysicalExpr,
    required: &mut std::collections::BTreeSet<usize>,
) -> Result<(), FallbackReason> {
    let expr = validated.expr();
    let input_schema = validated.input_schema();
    if let Some(column) = expr.downcast_ref::<Column>() {
        required.insert(resolve_column(column, input_schema).expect("validated column must resolve").index());
        return Ok(());
    }
    if expr.downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        collect_validated_required_field_indices(&validated.trusted_descendant(binary.left()), required)?;
        collect_validated_required_field_indices(&validated.trusted_descendant(binary.right()), required)?;
        return Ok(());
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return collect_validated_required_field_indices(&validated.trusted_descendant(not.arg()), required);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return collect_validated_required_field_indices(&validated.trusted_descendant(is_null.arg()), required);
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return collect_validated_required_field_indices(&validated.trusted_descendant(is_not_null.arg()), required);
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return collect_validated_required_field_indices(&validated.trusted_descendant(cast.expr()), required);
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        return collect_validated_required_field_indices(&validated.trusted_descendant(like.expr()), required);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        collect_validated_required_field_indices(&validated.trusted_descendant(in_list.expr()), required)?;
        for expr in in_list.list() {
            collect_validated_required_field_indices(&validated.trusted_descendant(expr), required)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if let Some(expr) = case.expr() {
            collect_validated_required_field_indices(&validated.trusted_descendant(expr), required)?;
        }
        for (when, then) in case.when_then_expr() {
            collect_validated_required_field_indices(&validated.trusted_descendant(when), required)?;
            collect_validated_required_field_indices(&validated.trusted_descendant(then), required)?;
        }
        if let Some(else_expr) = case.else_expr() {
            collect_validated_required_field_indices(&validated.trusted_descendant(else_expr), required)?;
        }
        return Ok(());
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| FallbackReason::UnsupportedExpression)?;
        // Dynamic filters are mutable, so the current snapshot needs its own
        // proof even when the outer expression was validated earlier.
        let current = validate_physical_expr(&current, input_schema)?;
        return collect_validated_required_field_indices(&current, required);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        for arg in function.args() {
            collect_validated_required_field_indices(&validated.trusted_descendant(arg), required)?;
        }
        return Ok(());
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return collect_validated_required_field_indices(&validated.trusted_descendant(try_cast.expr()), required);
    }
    unreachable!("validated physical expression must have a required-field branch")
}

pub(super) fn lower_literal(value: &ScalarValue) -> Result<NativeExpr, FallbackReason> {
    Ok(NativeExpr::literal(native_literal_value(value, NativeLiteralContext::Expression)?))
}

pub(super) fn lower_binary_operator(op: &Operator) -> NativeBinaryOp {
    let operator = SupportedBinaryOperator::from_datafusion(op).expect("validated binary operator must be supported");
    match operator {
        SupportedBinaryOperator::Add => NativeBinaryOp::Add,
        SupportedBinaryOperator::Subtract => NativeBinaryOp::Subtract,
        SupportedBinaryOperator::Multiply => NativeBinaryOp::Multiply,
        SupportedBinaryOperator::Divide => NativeBinaryOp::Divide,
        SupportedBinaryOperator::Modulo => NativeBinaryOp::Modulo,
        SupportedBinaryOperator::BitwiseAnd => NativeBinaryOp::BitwiseAnd,
        SupportedBinaryOperator::BitwiseOr => NativeBinaryOp::BitwiseOr,
        SupportedBinaryOperator::BitwiseXor => NativeBinaryOp::BitwiseXor,
        SupportedBinaryOperator::ShiftLeft => NativeBinaryOp::ShiftLeft,
        SupportedBinaryOperator::ShiftRight => NativeBinaryOp::ShiftRight,
        SupportedBinaryOperator::Eq => NativeBinaryOp::Eq,
        SupportedBinaryOperator::NotEq => NativeBinaryOp::NotEq,
        SupportedBinaryOperator::Lt => NativeBinaryOp::Lt,
        SupportedBinaryOperator::LtEq => NativeBinaryOp::LtEq,
        SupportedBinaryOperator::Gt => NativeBinaryOp::Gt,
        SupportedBinaryOperator::GtEq => NativeBinaryOp::GtEq,
        SupportedBinaryOperator::And => NativeBinaryOp::And,
        SupportedBinaryOperator::Or => NativeBinaryOp::Or,
        SupportedBinaryOperator::StringConcat
        | SupportedBinaryOperator::IsDistinctFrom
        | SupportedBinaryOperator::IsNotDistinctFrom => {
            unreachable!("special validated binary operators have dedicated lowering branches")
        },
    }
}

pub(super) fn alias_if_needed(expr: NativeExpr, alias: &str) -> NativeExpr {
    match &expr {
        NativeExpr::Column(column) if column.display_name() == alias || column.name() == alias => expr,
        _ => expr.alias(alias),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    #[test]
    fn binding_resolution_prefers_unique_name_over_colliding_index() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("payload", DataType::Int32, false),
            Field::new("label", DataType::Utf8, true),
        ]));
        let bindings = vec![
            FieldBinding {
                frontend_index: 1,
                frontend_name: "payload".to_string(),
                native_name: "payload".to_string(),
            },
            FieldBinding { frontend_index: 2, frontend_name: "label".to_string(), native_name: "label".to_string() },
        ];
        let projected_label = Column::new("label", 1);

        let resolved = resolved_column_name(&projected_label, &schema, Some(&bindings)).unwrap();

        assert_eq!(resolved, "label");
    }

    #[test]
    fn nested_try_cast_uses_resolved_source_type() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("amount", DataType::Int32, true),
        ]));
        let bindings = vec![
            FieldBinding { frontend_index: 0, frontend_name: "label".to_string(), native_name: "label".to_string() },
            FieldBinding { frontend_index: 1, frontend_name: "amount".to_string(), native_name: "amount".to_string() },
        ];
        let projected_amount: Arc<dyn PhysicalExpr> = Arc::new(Column::new("amount", 0));
        let try_cast: Arc<dyn PhysicalExpr> = Arc::new(TryCastExpr::new(projected_amount, DataType::Int64));

        let error = lower_physical_expr_with_bindings(&try_cast, &schema, Some(&bindings)).unwrap_err();

        assert_eq!(error, FallbackReason::UnsupportedExpression);
    }

    #[test]
    fn decimal_divide_uses_resolved_operand_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("wrong_left", DataType::Int64, false),
            Field::new("wrong_right", DataType::Int64, false),
            Field::new("amount", DataType::Decimal128(15, 2), false),
            Field::new("discount", DataType::Decimal128(15, 2), false),
        ]));
        let bindings = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(frontend_index, field)| FieldBinding {
                frontend_index,
                frontend_name: field.name().clone(),
                native_name: field.name().clone(),
            })
            .collect::<Vec<_>>();
        let divide: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("amount", 0)),
            Operator::Divide,
            Arc::new(Column::new("discount", 1)),
        ));

        let lowered = lower_physical_expr_with_bindings(&divide, &schema, Some(&bindings)).unwrap();

        assert!(matches!(
            lowered,
            NativeExpr::Binary { op: NativeBinaryOp::Divide, output_type: Some(DataType::Decimal128(_, _)), .. }
        ));
    }
}
