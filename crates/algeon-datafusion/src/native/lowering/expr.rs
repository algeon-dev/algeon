use std::{cell::Cell, collections::BTreeMap, sync::Arc};

use arrow::datatypes::{DataType as ArrowDataType, Field, SchemaRef};
use cudf::interop::arrow_host::is_supported_arrow_type;
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr,
            LikeExpr, Literal, NegativeExpr, NotExpr, TryCastExpr,
        },
    },
};
use query_planning::{
    expr::{
        BinaryOp as NativeBinaryOp, CaseWhen as NativeCaseWhen, ColumnRef, DatePart as NativeDatePart,
        Expr as NativeExpr, LiteralValue as NativeLiteralValue, UnaryMathOp,
    },
    validation::substring_stop_position,
};

use crate::{
    capability::CapabilityReason,
    native::{
        LoweringRejection,
        expression_contract::{
            MAX_ACTIVATION_NODES, ResolvedColumn, ScalarFunctionKind, SupportedBinaryOperator, ValidatedPhysicalExpr,
            contains_fallible_operation, declared_decimal_divide_output_type, fallible_operand,
            is_decimal128_to_float32_cast_via_float64, is_list_carry_type, oversized_activation_rejection,
            physical_expr_data_type, resolve_column, validate_physical_expr_contract,
        },
        literal::{NativeLiteralContext, literal_i32, literal_string, native_literal_value, parse_yyyy_mm_dd_date32},
        lowering::FieldBinding,
    },
};

pub(super) fn physical_expr_field(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    name: &str,
) -> Result<Field, LoweringRejection> {
    let validated = validate_physical_expr_contract(expr, input_schema)?;
    physical_expr_field_from_validated(&validated, name)
}

pub(super) fn physical_expr_field_from_validated(
    validated: &ValidatedPhysicalExpr,
    name: &str,
) -> Result<Field, LoweringRejection> {
    let expr = validated.expr();
    let input_schema = validated.input_schema();
    let data_type =
        physical_expr_data_type(expr, input_schema).expect("validated expression must have a resolved output type");
    debug_assert!(is_supported_arrow_type(&data_type), "validated expression output must be a native type");
    let nullable = expr.nullable(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    Ok(Field::new(name, data_type, nullable))
}

pub(in crate::native) fn lower_physical_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, LoweringRejection> {
    lower_physical_expr_with_bindings(expr, input_schema, None)
}

pub(super) fn lower_physical_expr_with_bindings(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, LoweringRejection> {
    let validated = validate_physical_expr_contract(expr, input_schema)?;
    lower_validated_physical_expr_with_bindings(&validated, bindings)
}

pub(super) fn lower_validated_physical_expr_with_bindings(
    validated: &ValidatedPhysicalExpr,
    bindings: Option<&[FieldBinding]>,
) -> Result<NativeExpr, LoweringRejection> {
    let activation_nodes = Cell::new(0);
    lower_validated_physical_expr(
        validated,
        LoweringScope { bindings, join_columns: None, activation: None, activation_nodes: &activation_nodes },
    )
}

pub(super) fn lower_validated_join_residual_expr(
    validated: &ValidatedPhysicalExpr,
    columns: &BTreeMap<usize, ColumnRef>,
) -> Result<NativeExpr, LoweringRejection> {
    let activation_nodes = Cell::new(0);
    lower_validated_physical_expr(
        validated,
        LoweringScope {
            bindings: None,
            join_columns: Some(columns),
            activation: None,
            activation_nodes: &activation_nodes,
        },
    )
}

/// What the lowering of an expression needs to know about where it sits.
#[derive(Clone, Copy)]
struct LoweringScope<'a> {
    bindings: Option<&'a [FieldBinding]>,
    join_columns: Option<&'a BTreeMap<usize, ColumnRef>>,
    /// Set inside a `CASE` branch or the right operand of an `AND` or `OR`,
    /// which DataFusion may evaluate on a subset of rows: the rows where this
    /// boolean is TRUE. Native evaluation visits every row, so an operand that
    /// native evaluation checks for failing values is NULL on every other row.
    activation: Option<&'a NativeExpr>,
    /// Nodes that [`Self::restrict`] has embedded so far in this expression.
    activation_nodes: &'a Cell<usize>,
}

impl<'a> LoweringScope<'a> {
    fn within(self, activation: Option<&'a NativeExpr>) -> Self {
        Self { activation, ..self }
    }

    /// `operand` with a NULL value on every row outside the activation. A
    /// NULL operand exempts its row from the failing-value check. `None` when
    /// the activations of this expression would exceed [`MAX_ACTIVATION_NODES`].
    fn restrict(self, operand: NativeExpr, data_type: ArrowDataType) -> Option<NativeExpr> {
        let Some(activation) = self.activation else {
            return Some(operand);
        };
        let mut size = 0;
        activation.any(|_| {
            size += 1;
            false
        });
        let embedded = self.activation_nodes.get() + size;
        if embedded > MAX_ACTIVATION_NODES {
            return None;
        }
        self.activation_nodes.set(embedded);
        Some(NativeExpr::case(
            vec![NativeCaseWhen::new(activation.clone(), operand)],
            NativeExpr::literal(NativeLiteralValue::Null(data_type)),
        ))
    }
}

fn lower_validated_physical_expr(
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let expr = validated.expr();
    let input_schema = validated.input_schema();
    if let Some(column) = expr.downcast_ref::<Column>() {
        if let Some(columns) = scope.join_columns {
            let resolved = resolve_column(column, input_schema)?;
            let column = columns.get(&resolved.index()).expect("validated join column must have a native binding");
            return Ok(NativeExpr::Column(column.clone()));
        }
        return Ok(NativeExpr::column(resolved_column_name(column, input_schema, scope.bindings)?));
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return Ok(lower_literal(literal.value()).expect("validated literal must lower"));
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower_validated_child_expr(binary.left(), validated, scope)?;
        let mut right = lower_binary_right_operand(binary, &left, validated, scope)?;
        if let Some(divisor) = fallible_operand(expr, input_schema) {
            let divisor_type =
                physical_expr_data_type(divisor, input_schema).expect("validated divisor must have a resolved type");
            right = scope
                .restrict(right, divisor_type)
                .ok_or_else(|| oversized_activation_rejection(expr, input_schema))?;
        }
        if matches!(binary.op(), Operator::StringConcat) {
            let output_type =
                physical_expr_data_type(expr, input_schema).expect("validated concatenation must have an output type");
            let null_guard = NativeExpr::binary(
                NativeExpr::is_null(left.clone()),
                NativeBinaryOp::Or,
                NativeExpr::is_null(right.clone()),
            );
            return Ok(NativeExpr::case(
                vec![NativeCaseWhen::new(null_guard, NativeExpr::literal(NativeLiteralValue::Null(output_type)))],
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
        return Ok(!lower_validated_child_expr(not.arg(), validated, scope)?);
    }
    if let Some(negative) = expr.downcast_ref::<NegativeExpr>() {
        let output_type = physical_expr_data_type(negative.arg(), input_schema)
            .expect("validated negation must have a resolved type");
        return Ok(NativeExpr::unary_math(
            UnaryMathOp::Negate,
            lower_validated_child_expr(negative.arg(), validated, scope)?,
            output_type,
        ));
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_validated_child_expr(is_null.arg(), validated, scope)?));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_validated_child_expr(is_not_null.arg(), validated, scope)?));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let source_type = physical_expr_data_type(cast.expr(), input_schema)
            .expect("validated cast source must have a resolved type");
        if let Some(value) = lower_literal_string_date32_cast(cast.expr(), cast.cast_type()) {
            return Ok(NativeExpr::literal(value));
        }
        if source_type == *cast.cast_type() && is_list_carry_type(&source_type) {
            return lower_validated_child_expr(cast.expr(), validated, scope);
        }
        if is_decimal128_to_float32_cast_via_float64(&source_type, cast.cast_type()) {
            let as_float64 =
                NativeExpr::cast(lower_validated_child_expr(cast.expr(), validated, scope)?, ArrowDataType::Float64);
            return Ok(NativeExpr::cast(as_float64, ArrowDataType::Float32));
        }
        let mut input = lower_validated_child_expr(cast.expr(), validated, scope)?;
        if fallible_operand(expr, input_schema).is_some() {
            input =
                scope.restrict(input, source_type).ok_or_else(|| oversized_activation_rejection(expr, input_schema))?;
        }
        return Ok(NativeExpr::cast(input, cast.cast_type().clone()));
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        return lower_like_expr(like, validated, scope);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        return lower_in_list_expr(in_list, validated, scope);
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return lower_case_expr(case, validated, scope);
    }
    if let Some(dynamic_filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        let current = dynamic_filter.current().map_err(|_error| CapabilityReason::UnsupportedExpression)?;
        // Dynamic filters are mutable. Validate the current snapshot before
        // deriving a new lowering proof. The proof of the older root does not carry over.
        let current = validate_physical_expr_contract(&current, input_schema)?;
        return lower_validated_physical_expr(&current, scope);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return lower_scalar_function_expr(function, validated, scope);
    }
    if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        return lower_try_cast_expr(try_cast, validated, scope);
    }
    unreachable!("validated physical expression must have a lowering branch")
}

fn lower_validated_child_expr(
    expr: &Arc<dyn PhysicalExpr>,
    parent: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let child = parent.trusted_descendant(expr);
    lower_validated_physical_expr(&child, scope)
}

/// Lowers the right operand of `binary`.
///
/// DataFusion skips the right operand of `AND` and `OR` on the rows the left
/// operand decides, which are the FALSE rows for `AND` and the TRUE rows for
/// `OR`, only in some batches. It skips all rows when every left value is
/// decisive, and the decided rows when the left values have no NULL and at most
/// a fifth of them are undecided. Any other batch evaluates the right operand
/// on every row, including the decided ones.
///
/// Native evaluation never fails on a row the left operand decides, because
/// the right operand lowers under an activation that is TRUE only on the other
/// rows. A NULL left value is undecided. Values do not change: a decided row
/// takes its value from the left operand alone. Native therefore fails only
/// where DataFusion fails, and not on a decided row that DataFusion happens to
/// evaluate. An operand with no fallible operation needs no activation and
/// does not build one.
fn lower_binary_right_operand(
    binary: &BinaryExpr,
    left: &NativeExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let deciding_left = match binary.op() {
        Operator::And => !left.clone(),
        Operator::Or => left.clone(),
        _ => return lower_validated_child_expr(binary.right(), validated, scope),
    };
    if !contains_fallible_operation(binary.right(), validated.input_schema()) {
        return lower_validated_child_expr(binary.right(), validated, scope);
    }
    let activation = nested_activation(scope.activation, selected_rows(None, &[deciding_left], None));
    lower_validated_child_expr(binary.right(), validated, scope.within(activation.as_ref()))
}

fn lower_try_cast_expr(
    try_cast: &TryCastExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let inner = lower_validated_child_expr(try_cast.expr(), validated, scope)?;
    Ok(NativeExpr::try_cast(inner, try_cast.cast_type().clone()))
}

fn lower_literal_string_date32_cast(
    expr: &Arc<dyn PhysicalExpr>,
    target_type: &ArrowDataType,
) -> Option<NativeLiteralValue> {
    if !matches!(target_type, ArrowDataType::Date32) {
        return None;
    }
    expr.downcast_ref::<Literal>().and_then(|literal| match literal.value() {
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            parse_yyyy_mm_dd_date32(value).map(NativeLiteralValue::Date32)
        },
        _ => None,
    })
}

fn lower_like_expr(
    like: &LikeExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let pattern = literal_string(like.pattern()).expect("validated LIKE pattern must be a string literal");
    debug_assert!(!like.case_insensitive(), "case-insensitive LIKE cannot carry a validation proof");
    let expr = lower_validated_child_expr(like.expr(), validated, scope)?;
    Ok(if like.negated() {
        NativeExpr::not_like_escape(expr, pattern, "\\")
    } else {
        NativeExpr::like_escape(expr, pattern, "\\")
    })
}

fn lower_in_list_expr(
    in_list: &InListExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    debug_assert!(!in_list.is_empty(), "an empty IN list cannot carry a validation proof");
    let expr = lower_validated_child_expr(in_list.expr(), validated, scope)?;
    let list = in_list
        .list()
        .iter()
        .map(|expr| {
            let literal = expr.downcast_ref::<Literal>().expect("validated IN-list entries must be literals");
            debug_assert!(!literal.value().is_null(), "null IN-list entries cannot carry a validation proof");
            lower_validated_child_expr(expr, validated, scope)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if in_list.negated() { NativeExpr::not_in_list(expr, list) } else { NativeExpr::in_list(expr, list) })
}

/// Lowers a `CASE` so that native evaluation, which visits every row of every
/// branch, fails on the same rows where DataFusion does.
///
/// DataFusion evaluates a `WHEN` condition only on the rows no earlier branch
/// selected (for `CASE base WHEN value`, also only where `base` is not NULL),
/// a `THEN` result only where its condition is TRUE, and `ELSE` only where no
/// condition is TRUE, a NULL condition counting as not TRUE. Each of those
/// expressions lowers under the activation that selects its rows. The
/// activations use the guarded lowering of the earlier conditions, so they
/// cannot fail on a row they exclude. An expression with no fallible
/// operation needs no activation and does not build one.
fn lower_case_expr(
    case: &CaseExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let input_schema = validated.input_schema();
    let activation_of = |child: &Arc<dyn PhysicalExpr>, build: &dyn Fn() -> Option<NativeExpr>| {
        if contains_fallible_operation(child, input_schema) {
            nested_activation(scope.activation, build())
        } else {
            None
        }
    };
    let base = case.expr().map(|expr| lower_validated_child_expr(expr, validated, scope)).transpose()?;
    let base_is_null = base.as_ref().map(|base| NativeExpr::is_null(base.clone()));
    let mut conditions: Vec<NativeExpr> = Vec::new();
    let mut branches = Vec::with_capacity(case.when_then_expr().len());
    for (when, then) in case.when_then_expr() {
        let when_activation = activation_of(when, &|| selected_rows(base_is_null.as_ref(), &conditions, None));
        let when = lower_validated_child_expr(when, validated, scope.within(when_activation.as_ref()))?;
        let condition = match &base {
            Some(base) => NativeExpr::binary(base.clone(), NativeBinaryOp::Eq, when),
            None => when,
        };
        let then_activation = activation_of(then, &|| selected_rows(None, &conditions, Some(&condition)));
        let then = lower_validated_child_expr(then, validated, scope.within(then_activation.as_ref()))?;
        branches.push(NativeCaseWhen::new(condition.clone(), then));
        conditions.push(condition);
    }
    let else_expr = match case.else_expr() {
        Some(else_expr) => {
            let else_activation = activation_of(else_expr, &|| selected_rows(None, &conditions, None));
            lower_validated_child_expr(else_expr, validated, scope.within(else_activation.as_ref()))?
        },
        None => {
            let data_type = physical_expr_data_type(validated.expr(), input_schema)
                .expect("validated CASE expression must have an output type");
            NativeExpr::literal(NativeLiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

/// A boolean that is TRUE on the rows where no `leading` or `exits` condition
/// is TRUE and, when given, `selected` is TRUE, and false elsewhere. `None`
/// selects every row. With no `exits` or `leading`, `selected` is returned as
/// it is, so it may be NULL on a row it does not select.
fn selected_rows(
    leading: Option<&NativeExpr>,
    exits: &[NativeExpr],
    selected: Option<&NativeExpr>,
) -> Option<NativeExpr> {
    if leading.is_none() && exits.is_empty() {
        return selected.cloned();
    }
    let literal = |value| NativeExpr::literal(NativeLiteralValue::Boolean(value));
    let branches = leading
        .into_iter()
        .chain(exits)
        .map(|exit| NativeCaseWhen::new(exit.clone(), literal(false)))
        .chain(selected.map(|selected| NativeCaseWhen::new(selected.clone(), literal(true))))
        .collect();
    Some(NativeExpr::case(branches, literal(selected.is_none())))
}

/// `local` restricted to the rows of `outer`, which is the activation of the
/// enclosing `CASE` branch. A boolean that is NULL or false selects no row.
fn nested_activation(outer: Option<&NativeExpr>, local: Option<NativeExpr>) -> Option<NativeExpr> {
    match (outer, local) {
        (Some(outer), Some(local)) => Some(NativeExpr::case(
            vec![NativeCaseWhen::new(outer.clone(), local)],
            NativeExpr::literal(NativeLiteralValue::Null(ArrowDataType::Boolean)),
        )),
        (Some(outer), None) => Some(outer.clone()),
        (None, local) => local,
    }
}

fn lower_scalar_function_expr(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let kind = ScalarFunctionKind::from_scalar_function(function).expect("validated scalar function kind");
    match kind {
        ScalarFunctionKind::Substring => lower_substring_function(function, validated, scope),
        ScalarFunctionKind::Extract => lower_extract_function(function, validated, scope),
        ScalarFunctionKind::DateTrunc => lower_date_trunc_function(function, validated, scope),
        ScalarFunctionKind::ToTimestampSeconds => lower_to_timestamp_seconds_function(function, validated, scope),
        ScalarFunctionKind::Lower => lower_string_case_function(function, validated, scope, NativeExpr::lower),
        ScalarFunctionKind::Upper => lower_string_case_function(function, validated, scope, NativeExpr::upper),
        ScalarFunctionKind::Trim => lower_trim_function(function, validated, scope, NativeExpr::trim),
        ScalarFunctionKind::LeftTrim => lower_trim_function(function, validated, scope, NativeExpr::ltrim),
        ScalarFunctionKind::RightTrim => lower_trim_function(function, validated, scope, NativeExpr::rtrim),
        ScalarFunctionKind::Contains => {
            lower_string_predicate_function(function, validated, scope, NativeExpr::string_contains)
        },
        ScalarFunctionKind::StartsWith => {
            lower_string_predicate_function(function, validated, scope, NativeExpr::starts_with)
        },
        ScalarFunctionKind::EndsWith => {
            lower_string_predicate_function(function, validated, scope, NativeExpr::ends_with)
        },
        ScalarFunctionKind::StringLength => lower_string_length_function(function, validated, scope),
        ScalarFunctionKind::RegexpReplace => lower_regexp_replace_function(function, validated, scope),
        ScalarFunctionKind::Concat => lower_concat_function(function, validated, scope),
        ScalarFunctionKind::Abs => lower_abs_function(function, validated, scope),
        ScalarFunctionKind::UnaryMath(op) => lower_unary_math_function(op, function, validated, scope),
        ScalarFunctionKind::Round => lower_round_function(function, validated, scope),
        ScalarFunctionKind::NullIf => lower_nullif_function(function, validated, scope),
    }
}

fn lower_substring_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
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
    Ok(NativeExpr::substring(lower_validated_child_expr(&args[0], validated, scope)?, start, length))
}

fn lower_extract_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [field, expr] = function.args() else { unreachable!("validated EXTRACT must have two arguments") };
    let field = NativeDatePart::from_sql_name(
        &literal_string(field).expect("validated EXTRACT field must be a string literal"),
    )
    .expect("validated EXTRACT field must name a date part");
    Ok(NativeExpr::extract(field, lower_validated_child_expr(expr, validated, scope)?))
}

fn lower_date_trunc_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [field, expr] = function.args() else { unreachable!("validated DATE_TRUNC must have two arguments") };
    let field = NativeDatePart::from_sql_name(
        &literal_string(field).expect("validated DATE_TRUNC field must be a string literal"),
    )
    .expect("validated DATE_TRUNC field must name a date part");
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::date_trunc(field, lower_validated_child_expr(expr, validated, scope)?, output_type))
}

fn lower_to_timestamp_seconds_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr] = function.args() else { unreachable!("validated TO_TIMESTAMP_SECONDS must have one argument") };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::cast(lower_validated_child_expr(expr, validated, scope)?, output_type))
}

fn lower_string_case_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
    constructor: fn(NativeExpr) -> NativeExpr,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr] = function.args() else { unreachable!("validated string case function must have one argument") };
    Ok(constructor(lower_validated_child_expr(expr, validated, scope)?))
}

fn lower_trim_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
    constructor: fn(NativeExpr, String) -> NativeExpr,
) -> Result<NativeExpr, LoweringRejection> {
    let args = function.args();
    debug_assert!((1..=2).contains(&args.len()), "invalid TRIM arity cannot carry a validation proof");
    let trim_chars = if args.len() == 2 {
        literal_string(&args[1]).expect("validated TRIM characters must be a string literal")
    } else {
        " ".to_string()
    };
    let expr = lower_validated_child_expr(&args[0], validated, scope)?;
    if trim_chars.is_empty() { Ok(expr) } else { Ok(constructor(expr, trim_chars)) }
}

fn lower_string_predicate_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
    constructor: fn(NativeExpr, String) -> NativeExpr,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr, pattern] = function.args() else { unreachable!("validated string predicate must have two arguments") };
    Ok(constructor(
        lower_validated_child_expr(expr, validated, scope)?,
        literal_string(pattern).expect("validated string predicate pattern must be a string literal"),
    ))
}

fn lower_string_length_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr] = function.args() else { unreachable!("validated string length function must have one argument") };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::string_length(lower_validated_child_expr(expr, validated, scope)?, output_type))
}

fn lower_regexp_replace_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr, pattern, replacement] = function.args() else {
        unreachable!("validated REGEXP_REPLACE must have three arguments")
    };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::regexp_replace(
        lower_validated_child_expr(expr, validated, scope)?,
        literal_string(pattern).expect("validated REGEXP_REPLACE pattern must be a string literal"),
        literal_string(replacement).expect("validated REGEXP_REPLACE replacement must be a string literal"),
        output_type,
    ))
}

fn lower_concat_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    debug_assert!(!function.args().is_empty(), "empty CONCAT cannot carry a validation proof");
    function
        .args()
        .iter()
        .map(|expr| lower_validated_child_expr(expr, validated, scope))
        .collect::<Result<Vec<_>, _>>()
        .map(NativeExpr::concat)
}

fn lower_abs_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr] = function.args() else { unreachable!("validated ABS must have one argument") };
    Ok(NativeExpr::abs(lower_validated_child_expr(expr, validated, scope)?))
}

fn lower_unary_math_function(
    op: UnaryMathOp,
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr] = function.args() else { unreachable!("validated {op} must have one argument") };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::unary_math(op, lower_validated_child_expr(expr, validated, scope)?, output_type))
}

fn lower_round_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let args = function.args();
    debug_assert!((1..=2).contains(&args.len()), "invalid ROUND arity cannot carry a validation proof");
    let decimal_places = if args.len() == 2 {
        literal_i32(&args[1]).expect("validated ROUND precision must be an i32 literal")
    } else {
        0
    };
    let output_type = validated_scalar_output_type(function, validated);
    Ok(NativeExpr::round(lower_validated_child_expr(&args[0], validated, scope)?, decimal_places, output_type))
}

fn lower_nullif_function(
    function: &ScalarFunctionExpr,
    validated: &ValidatedPhysicalExpr,
    scope: LoweringScope<'_>,
) -> Result<NativeExpr, LoweringRejection> {
    let [left, right] = function.args() else { unreachable!("validated NULLIF must have two arguments") };
    let output_type = validated_scalar_output_type(function, validated);
    let left = lower_validated_child_expr(left, validated, scope)?;
    let right = lower_validated_child_expr(right, validated, scope)?;
    Ok(NativeExpr::case(
        vec![NativeCaseWhen::new(
            NativeExpr::is_not_distinct_from(left.clone(), right),
            NativeExpr::literal(NativeLiteralValue::Null(output_type)),
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
) -> Result<&'a str, LoweringRejection> {
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
            .ok_or_else(|| CapabilityReason::UnsupportedExpression.into());
    }
    Ok(resolved.field().name())
}

pub(super) fn collect_required_field_indices(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    required: &mut std::collections::BTreeSet<usize>,
) -> Result<(), LoweringRejection> {
    let validated = validate_physical_expr_contract(expr, input_schema)?;
    collect_validated_required_field_indices(&validated, required)
}

fn collect_validated_required_field_indices(
    validated: &ValidatedPhysicalExpr,
    required: &mut std::collections::BTreeSet<usize>,
) -> Result<(), LoweringRejection> {
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
    if let Some(negative) = expr.downcast_ref::<NegativeExpr>() {
        return collect_validated_required_field_indices(&validated.trusted_descendant(negative.arg()), required);
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
        let current = dynamic_filter.current().map_err(|_error| CapabilityReason::UnsupportedExpression)?;
        // Dynamic filters are mutable, so the current snapshot needs its own
        // proof even when the outer expression was validated earlier.
        let current = validate_physical_expr_contract(&current, input_schema)?;
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

pub(super) fn lower_literal(value: &ScalarValue) -> Result<NativeExpr, LoweringRejection> {
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

        assert_eq!(error.reason(), CapabilityReason::UnsupportedExpression);
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
