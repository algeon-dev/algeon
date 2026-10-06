use std::sync::Arc;

use arrow::datatypes::{DataType, Field, SchemaRef, TimeUnit};
use cudf::interop::arrow_host::is_supported_arrow_type;
use datafusion::{
    common::ScalarValue,
    logical_expr::{Operator, binary::BinaryTypeCoercer},
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr,
            LikeExpr, Literal, NegativeExpr, NotExpr, TryCastExpr,
        },
        utils::collect_columns,
    },
};
use query_planning::{
    datatypes::is_timestamp_with_timezone,
    expr::DatePart as NativeDatePart,
    validation::{is_integral_type, is_string_type, substring_stop_position},
};

use super::{
    ExpressionCapability, ExpressionRejection, ExpressionRejectionSubject, ScalarFunctionKind, SupportedBinaryOperator,
    cast_capability, compatible_types, is_list_carry_type, is_narrow_decimal_carry, is_numeric_or_decimal_type,
    is_supported_to_timestamp_seconds_cast, is_supported_try_cast_pair,
};
use crate::native::literal::{
    NativeLiteralContext, is_interval_literal_expr, literal_i32, literal_string, validate_native_literal,
};

const fn unsupported_expression() -> ExpressionRejection {
    ExpressionRejection::unsupported_expression(ExpressionCapability::PhysicalExpression)
}

const fn unsupported_type() -> ExpressionRejection {
    ExpressionRejection::unsupported_type(ExpressionCapability::PhysicalExpression)
}

const fn scalar_function_arguments() -> ExpressionRejection {
    ExpressionRejection::unsupported_expression(ExpressionCapability::ScalarFunctionArguments)
}

/// A column identity resolved against the schema used by expression
/// validation. Lowering reuses this identity so admission and execution cannot
/// select different fields when a projected column carries a stale index.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) struct ResolvedColumn<'a> {
    index: usize,
    field: &'a Field,
}

impl<'a> ResolvedColumn<'a> {
    pub(in crate::native) fn index(self) -> usize {
        self.index
    }

    pub(in crate::native) fn field(self) -> &'a Field {
        self.field
    }
}

pub(in crate::native) fn resolve_column<'a>(
    column: &Column,
    input_schema: &'a SchemaRef,
) -> Result<ResolvedColumn<'a>, ExpressionRejection> {
    if let Some(field) = input_schema.fields().get(column.index()).filter(|field| field.name() == column.name()) {
        return Ok(ResolvedColumn { index: column.index(), field });
    }

    let mut matching_fields =
        input_schema.fields().iter().enumerate().filter(|(_, field)| field.name() == column.name());
    if let Some((index, field)) = matching_fields.next()
        && matching_fields.next().is_none()
    {
        return Ok(ResolvedColumn { index, field });
    }

    input_schema
        .fields()
        .get(column.index())
        .map(|field| ResolvedColumn { index: column.index(), field })
        .ok_or(unsupported_expression())
}

/// Proof that a physical expression and all of its children satisfy the
/// shared native expression contract for `input_schema`.
#[derive(Debug, Clone)]
pub(in crate::native) struct ValidatedPhysicalExpr {
    expr: Arc<dyn PhysicalExpr>,
    input_schema: SchemaRef,
}

impl ValidatedPhysicalExpr {
    pub(in crate::native) fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    pub(in crate::native) fn input_schema(&self) -> &SchemaRef {
        &self.input_schema
    }

    /// Derives a proof token for a child of an already validated expression.
    /// Only lowering uses this constructor, after the parent contract has
    /// recursively validated every child.
    pub(in crate::native) fn trusted_descendant(&self, expr: &Arc<dyn PhysicalExpr>) -> Self {
        Self { expr: Arc::clone(expr), input_schema: Arc::clone(&self.input_schema) }
    }
}

pub(in crate::native) fn validate_physical_expr_contract(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<ValidatedPhysicalExpr, ExpressionRejection> {
    validate_physical_expr(expr, input_schema)?;
    Ok(ValidatedPhysicalExpr { expr: Arc::clone(expr), input_schema: Arc::clone(input_schema) })
}

fn validate_physical_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
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
        let source_type = physical_expr_data_type(cast.expr(), input_schema)?;
        cast_capability(cast.expr(), &source_type, cast.cast_type())?;
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
        let current = dynamic_filter.current().map_err(|error| {
            ExpressionRejection::unsupported_expression(ExpressionCapability::DynamicFilter)
                .with_subject(ExpressionRejectionSubject::DynamicFilterError(error.to_string()))
        })?;
        return validate_physical_expr(&current, input_schema);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return validate_scalar_function_expr(function, input_schema);
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return validate_try_cast_expr(try_cast, input_schema);
    }
    if let Some(negative) = expr.downcast_ref::<NegativeExpr>() {
        let data_type = physical_expr_data_type(negative.arg(), input_schema)?;
        if !matches!(
            data_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
        ) {
            return Err(ExpressionRejection::unsupported_type(ExpressionCapability::NegativeExpression));
        }
        // DataFusion checks integer scalars but wraps arrays. Only literal
        // scalars have a checked result available during validation.
        if matches!(data_type, DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64)
            && collect_columns(negative.arg()).is_empty()
        {
            let Some(literal) = negative.arg().downcast_ref::<Literal>() else {
                return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::NegativeExpression));
            };
            if literal.value().arithmetic_negate().is_err() {
                return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::NegativeExpression));
            }
        }
        return validate_physical_expr(negative.arg(), input_schema);
    }
    Err(unsupported_expression())
}

fn validate_try_cast_expr(try_cast: &TryCastExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let source_type = physical_expr_data_type(try_cast.expr(), input_schema)?;
    if !is_supported_try_cast_pair(&source_type, try_cast.cast_type()) {
        // Unsupported type pairs stay on the UnsupportedExpression path so the
        // planner's child-rewrite short-circuit treats TRY_CAST the same way
        // it treated it before native admission of `STRING → integer`.
        return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::TryCastPair).with_subject(
            ExpressionRejectionSubject::Cast { source: source_type, target: try_cast.cast_type().clone() },
        ));
    }
    validate_physical_expr(try_cast.expr(), input_schema)
}

fn validate_null_predicate_arg(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    validate_physical_expr(expr, input_schema)?;
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if is_timestamp_with_timezone(&data_type) || is_list_carry_type(&data_type) || is_narrow_decimal_carry(&data_type) {
        Err(unsupported_type())
    } else {
        Ok(())
    }
}

fn validate_like_expr(like: &LikeExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    validate_literal_string(like.pattern())
        .map_err(|_| ExpressionRejection::unsupported_expression(ExpressionCapability::LikePattern))?;
    if like.case_insensitive() {
        return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::LikeCaseInsensitive));
    }
    validate_physical_expr(like.expr(), input_schema)?;
    validate_string_expr(like.expr(), input_schema)
}

fn validate_in_list_expr(in_list: &InListExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let unsupported_list = || ExpressionRejection::unsupported_expression(ExpressionCapability::InList);
    if in_list.is_empty() {
        return Err(unsupported_list());
    }
    validate_physical_expr(in_list.expr(), input_schema)?;
    let expr_type = physical_expr_data_type(in_list.expr(), input_schema)?;
    for expr in in_list.list() {
        let Some(literal) = expr.downcast_ref::<Literal>() else {
            return Err(unsupported_list());
        };
        if literal.value().is_null() {
            return Err(unsupported_list());
        }
        validate_physical_expr(expr, input_schema)?;
        let literal_type = physical_expr_data_type(expr, input_schema)?;
        validate_compatible_expr_types(&expr_type, &literal_type)?;
    }
    Ok(())
}

fn validate_case_expr(case: &CaseExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    if case.when_then_expr().is_empty() {
        return Err(unsupported_expression());
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
        .unwrap_or(case_expr_data_type(case, input_schema)?);

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
) -> Result<(), ExpressionRejection> {
    let kind = ScalarFunctionKind::from_scalar_function(function)?;
    validate_scalar_function_kind(kind, function, input_schema)
        .map_err(|rejection| rejection.for_scalar_function(function.name()))
}

fn validate_scalar_function_kind(
    kind: ScalarFunctionKind,
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    match kind {
        ScalarFunctionKind::Substring => validate_substring_function(function, input_schema),
        ScalarFunctionKind::Extract => validate_extract_function(function, input_schema),
        ScalarFunctionKind::DateTrunc => validate_date_trunc_function(function, input_schema),
        ScalarFunctionKind::ToTimestampSeconds => validate_to_timestamp_seconds_function(function, input_schema),
        ScalarFunctionKind::Lower | ScalarFunctionKind::Upper => validate_string_case_function(function, input_schema),
        ScalarFunctionKind::Trim | ScalarFunctionKind::LeftTrim | ScalarFunctionKind::RightTrim => {
            validate_trim_function(function, input_schema)
        },
        ScalarFunctionKind::Contains | ScalarFunctionKind::StartsWith | ScalarFunctionKind::EndsWith => {
            validate_string_predicate_function(function, input_schema)
        },
        ScalarFunctionKind::StringLength => validate_string_length_function(function, input_schema),
        ScalarFunctionKind::RegexpReplace => validate_regexp_replace_function(function, input_schema),
        ScalarFunctionKind::Concat => validate_concat_function(function, input_schema),
        ScalarFunctionKind::Abs => validate_abs_function(function, input_schema),
        ScalarFunctionKind::UnaryMath(_) => validate_unary_math_function(function, input_schema),
        ScalarFunctionKind::Round => validate_round_function(function, input_schema),
        ScalarFunctionKind::NullIf => validate_nullif_function(function, input_schema),
    }
}

fn validate_substring_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let args = function.args();
    validate_substring_arguments_contract(function)?;
    validate_physical_expr(&args[0], input_schema)?;
    validate_string_expr(&args[0], input_schema)
}

fn validate_extract_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let unsupported_field = || ExpressionRejection::unsupported_expression(ExpressionCapability::ExtractField);
    let [field, expr] = function.args() else {
        return Err(scalar_function_arguments());
    };
    let field = NativeDatePart::from_sql_name(&literal_string(field).map_err(|_| unsupported_field())?)
        .filter(|field| NativeDatePart::EXTRACT_FIELDS.contains(field))
        .ok_or_else(unsupported_field)?;
    validate_physical_expr(expr, input_schema)?;
    if field == NativeDatePart::Minute {
        validate_timestamp_second_expr(expr, input_schema)
    } else {
        validate_date_or_timestamp_expr(expr, input_schema)
    }
}

fn validate_date_trunc_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [field, expr] = function.args() else {
        return Err(scalar_function_arguments());
    };
    // DataFusion's date_trunc has no nanosecond granularity and fails such a
    // call at execution, so lowering it would answer a query DataFusion rejects.
    NativeDatePart::from_sql_name(&literal_string_argument(field)?)
        .filter(|field| *field != NativeDatePart::Nanosecond && NativeDatePart::DATE_TRUNC_FIELDS.contains(field))
        .ok_or_else(scalar_function_arguments)?;
    validate_physical_expr(expr, input_schema)?;
    let input_type = physical_expr_data_type(expr, input_schema)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if matches!(
        (&input_type, &output_type),
        (DataType::Timestamp(input_unit, None), DataType::Timestamp(output_unit, None)) if input_unit == output_unit
    ) {
        Ok(())
    } else {
        Err(unsupported_type())
    }
}

fn validate_to_timestamp_seconds_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [expr] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(expr, input_schema)?;
    let source_type = physical_expr_data_type(expr, input_schema)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if is_supported_to_timestamp_seconds_cast(&source_type, &output_type) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_string_case_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [expr] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(expr, input_schema)?;
    validate_string_expr(expr, input_schema)
}

fn validate_trim_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let args = function.args();
    if !(1..=2).contains(&args.len()) {
        return Err(scalar_function_arguments());
    }
    validate_physical_expr(&args[0], input_schema)?;
    validate_string_expr(&args[0], input_schema)?;
    if args.len() == 2 {
        validate_literal_string(&args[1])?;
    }
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if is_supported_arrow_type(&output_type) && is_string_type(&output_type) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_string_predicate_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [expr, pattern] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(expr, input_schema)?;
    validate_string_expr(expr, input_schema)?;
    validate_literal_string(pattern)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if matches!(output_type, DataType::Boolean) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_string_length_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [expr] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(expr, input_schema)?;
    validate_string_expr(expr, input_schema)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if is_supported_arrow_type(&output_type) && is_integral_type(&output_type) {
        Ok(())
    } else {
        Err(unsupported_type())
    }
}

fn validate_regexp_replace_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [expr, pattern, replacement] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(expr, input_schema)?;
    validate_string_expr(expr, input_schema)?;
    validate_literal_string(pattern)?;
    validate_literal_string(replacement)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if is_supported_arrow_type(&output_type) && is_string_type(&output_type) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_concat_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    if function.args().is_empty() {
        return Err(scalar_function_arguments());
    }
    for expr in function.args() {
        validate_physical_expr(expr, input_schema)?;
        validate_string_expr(expr, input_schema)?;
    }
    Ok(())
}

fn validate_abs_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let [expr] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(expr, input_schema)?;
    validate_numeric_or_decimal_expr(expr, input_schema)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if is_supported_arrow_type(&output_type) && is_numeric_or_decimal_type(&output_type) {
        Ok(())
    } else {
        Err(unsupported_type())
    }
}

fn validate_unary_math_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [expr] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(expr, input_schema)?;
    validate_numeric_or_decimal_expr(expr, input_schema)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if is_supported_arrow_type(&output_type) && is_numeric_or_decimal_type(&output_type) {
        Ok(())
    } else {
        Err(unsupported_type())
    }
}

fn validate_round_function(function: &ScalarFunctionExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let args = function.args();
    if !(1..=2).contains(&args.len()) {
        return Err(scalar_function_arguments());
    }
    validate_physical_expr(&args[0], input_schema)?;
    validate_numeric_or_decimal_expr(&args[0], input_schema)?;
    if args.len() == 2 {
        literal_i32_argument(&args[1])?;
    }
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if is_supported_arrow_type(&output_type) && is_numeric_or_decimal_type(&output_type) {
        Ok(())
    } else {
        Err(unsupported_type())
    }
}

fn validate_nullif_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let [left, right] = function.args() else {
        return Err(scalar_function_arguments());
    };
    validate_physical_expr(left, input_schema)?;
    validate_physical_expr(right, input_schema)?;
    let left_type = physical_expr_data_type(left, input_schema)?;
    let right_type = physical_expr_data_type(right, input_schema)?;
    validate_compatible_expr_types(&left_type, &right_type)?;
    let output_type = function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression())?;
    if !is_supported_arrow_type(&output_type) {
        return Err(unsupported_type());
    }
    validate_compatible_expr_types(&left_type, &output_type)
}

fn validate_column_type(column: &Column, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let data_type = resolved_column_data_type(column, input_schema)?;
    if is_supported_arrow_type(data_type) { Ok(()) } else { Err(unsupported_type()) }
}

fn resolved_column_data_type<'a>(
    column: &Column,
    input_schema: &'a SchemaRef,
) -> Result<&'a DataType, ExpressionRejection> {
    resolve_column(column, input_schema).map(|resolved| resolved.field().data_type())
}

fn validate_literal(value: &ScalarValue) -> Result<(), ExpressionRejection> {
    if matches!(value, ScalarValue::IntervalMonthDayNano(Some(_))) {
        return Err(unsupported_expression());
    }
    validate_native_literal(value, NativeLiteralContext::Expression).map_err(|_| {
        ExpressionRejection::unsupported_expression(ExpressionCapability::Literal)
            .with_subject(ExpressionRejectionSubject::LiteralType(value.data_type()))
    })
}

fn literal_string_argument(expr: &Arc<dyn PhysicalExpr>) -> Result<String, ExpressionRejection> {
    literal_string(expr)
        .map_err(|_| ExpressionRejection::unsupported_expression(ExpressionCapability::ScalarFunctionArguments))
}

fn literal_i32_argument(expr: &Arc<dyn PhysicalExpr>) -> Result<i32, ExpressionRejection> {
    literal_i32(expr)
        .map_err(|_| ExpressionRejection::unsupported_expression(ExpressionCapability::ScalarFunctionArguments))
}

fn validate_binary_expr(binary: &BinaryExpr, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let left_interval = is_interval_literal_expr(binary.left());
    let right_interval = is_interval_literal_expr(binary.right());
    if left_interval || right_interval {
        return validate_interval_binary_expr(binary, input_schema, left_interval, right_interval);
    }

    validate_physical_expr(binary.left(), input_schema)?;
    validate_physical_expr(binary.right(), input_schema)?;
    let operator = SupportedBinaryOperator::from_datafusion(binary.op())?;
    let left_type = physical_expr_data_type(binary.left(), input_schema)?;
    let right_type = physical_expr_data_type(binary.right(), input_schema)?;
    if operator.rejects_timestamp_timezone()
        && (is_timestamp_with_timezone(&left_type) || is_timestamp_with_timezone(&right_type))
    {
        return Err(unsupported_type());
    }
    if operator.rejects_list_carry()
        && (is_list_carry_type(&left_type)
            || is_list_carry_type(&right_type)
            || is_narrow_decimal_carry(&left_type)
            || is_narrow_decimal_carry(&right_type))
    {
        return Err(unsupported_type());
    }
    if matches!(binary.op(), Operator::StringConcat) {
        validate_string_expr(binary.left(), input_schema)?;
        validate_string_expr(binary.right(), input_schema)?;
    }
    if matches!(binary.op(), Operator::Modulo) {
        validate_integral_operand_type(binary.left(), input_schema)?;
        validate_integral_operand_type(binary.right(), input_schema)?;
    }
    // libcudf has no checked integer divide/modulo semantics, so integer `/`
    // and `%` fall back to CPU; decimal/float division stays admitted.
    if matches!(binary.op(), Operator::Divide | Operator::Modulo)
        && is_integral_type(&left_type)
        && is_integral_type(&right_type)
    {
        return Err(unsupported_expression());
    }
    if matches!(binary.op(), Operator::Divide) {
        declared_decimal_divide_output_type(binary, input_schema)?;
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
) -> Result<(), ExpressionRejection> {
    let unsupported_interval = || ExpressionRejection::unsupported_expression(ExpressionCapability::IntervalArithmetic);
    if left_interval == right_interval {
        return Err(unsupported_interval());
    }
    match (binary.op(), left_interval, right_interval) {
        (Operator::Plus, true, false) => validate_datetime_interval_operand(binary.right(), input_schema),
        (Operator::Plus | Operator::Minus, false, true) => {
            validate_datetime_interval_operand(binary.left(), input_schema)
        },
        _ => Err(unsupported_interval()),
    }
}

fn validate_datetime_interval_operand(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    validate_physical_expr(expr, input_schema)?;
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if matches!(data_type, DataType::Date32 | DataType::Timestamp(_, None)) {
        Ok(())
    } else {
        Err(ExpressionRejection::unsupported_expression(ExpressionCapability::IntervalArithmetic))
    }
}

fn validate_integral_operand_type(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
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
        Err(unsupported_type())
    }
}

pub(super) fn validate_literal_string(expr: &Arc<dyn PhysicalExpr>) -> Result<(), ExpressionRejection> {
    literal_string_argument(expr).map(|_| ())
}

pub(super) fn validate_substring_arguments_contract(function: &ScalarFunctionExpr) -> Result<(), ExpressionRejection> {
    let unsupported_arguments =
        || ExpressionRejection::unsupported_expression(ExpressionCapability::SubstringArguments);
    let args = function.args();
    if !(2..=3).contains(&args.len()) {
        return Err(unsupported_arguments());
    }
    let start = literal_i32(&args[1]).map_err(|_| unsupported_arguments())?;
    if start < 1 {
        return Err(unsupported_arguments());
    }
    if args.len() == 3 {
        let length = literal_i32(&args[2]).map_err(|_| unsupported_arguments())?;
        if length < 0 || substring_stop_position(start, length).is_none() {
            return Err(unsupported_arguments());
        }
    }
    Ok(())
}

fn validate_string_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), ExpressionRejection> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if is_string_type(&data_type) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_date_or_timestamp_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if matches!(data_type, DataType::Date32 | DataType::Timestamp(_, None)) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_timestamp_second_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if matches!(data_type, DataType::Timestamp(TimeUnit::Second, None)) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_numeric_or_decimal_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    let data_type = physical_expr_data_type(expr, input_schema)?;
    if is_numeric_or_decimal_type(&data_type) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_boolean_type(data_type: &DataType) -> Result<(), ExpressionRejection> {
    if matches!(data_type, DataType::Boolean) { Ok(()) } else { Err(unsupported_type()) }
}

fn validate_compatible_expr_types(left: &DataType, right: &DataType) -> Result<(), ExpressionRejection> {
    if compatible_types(left, right) {
        Ok(())
    } else {
        Err(ExpressionRejection::unsupported_type(ExpressionCapability::CompatibleTypes))
    }
}

pub(in crate::native) fn physical_expr_data_type(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<DataType, ExpressionRejection> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return resolved_column_data_type(column, input_schema).cloned();
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return Ok(literal.value().data_type());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left_type = physical_expr_data_type(binary.left(), input_schema)?;
        let right_type = physical_expr_data_type(binary.right(), input_schema)?;
        return BinaryTypeCoercer::new(&left_type, binary.op(), &right_type).get_result_type().map_err(|_error| {
            ExpressionRejection::unsupported_expression(ExpressionCapability::DataFusionTypeResolution)
        });
    }
    if let Some(negative) = expr.downcast_ref::<NegativeExpr>() {
        return physical_expr_data_type(negative.arg(), input_schema);
    }
    if expr.downcast_ref::<NotExpr>().is_some()
        || expr.downcast_ref::<IsNullExpr>().is_some()
        || expr.downcast_ref::<IsNotNullExpr>().is_some()
        || expr.downcast_ref::<LikeExpr>().is_some()
        || expr.downcast_ref::<InListExpr>().is_some()
    {
        return Ok(DataType::Boolean);
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return Ok(cast.cast_type().clone());
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return Ok(try_cast.cast_type().clone());
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return case_expr_data_type(case, input_schema);
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| unsupported_expression())?;
        return physical_expr_data_type(&current, input_schema);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return function.data_type(input_schema.as_ref()).map_err(|_error| unsupported_expression());
    }
    Err(unsupported_expression())
}

pub(in crate::native) fn case_expr_data_type(
    case: &CaseExpr,
    input_schema: &SchemaRef,
) -> Result<DataType, ExpressionRejection> {
    for (_when, then) in case.when_then_expr() {
        let data_type = physical_expr_data_type(then, input_schema)?;
        if data_type != DataType::Null {
            return Ok(data_type);
        }
    }
    case.else_expr().map_or(Ok(DataType::Null), |expr| physical_expr_data_type(expr, input_schema))
}

pub(in crate::native) fn declared_decimal_divide_output_type(
    binary: &BinaryExpr,
    input_schema: &SchemaRef,
) -> Result<Option<DataType>, ExpressionRejection> {
    let left_type = physical_expr_data_type(binary.left(), input_schema)?;
    let right_type = physical_expr_data_type(binary.right(), input_schema)?;
    if !matches!(left_type, DataType::Decimal128(_, _)) && !matches!(right_type, DataType::Decimal128(_, _)) {
        return Ok(None);
    }
    let declared_type =
        BinaryTypeCoercer::new(&left_type, binary.op(), &right_type).get_result_type().map_err(|_error| {
            ExpressionRejection::unsupported_expression(ExpressionCapability::DataFusionTypeResolution)
        })?;
    if !matches!(declared_type, DataType::Decimal128(_, _)) {
        return Ok(None);
    }
    if !is_supported_arrow_type(&declared_type) {
        return Err(ExpressionRejection::unsupported_type(ExpressionCapability::BinaryOperandTypes));
    }
    Ok(Some(declared_type))
}
