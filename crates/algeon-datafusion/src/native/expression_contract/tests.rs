use std::sync::Arc;

use arrow::datatypes::{DataType, Field, IntervalMonthDayNanoType, Schema, TimeUnit};
use datafusion::{
    common::{JoinSide, Result as DataFusionResult, ScalarValue},
    config::ConfigOptions,
    functions::{core, datetime, math, regex, string, unicode},
    functions_aggregate::median::Median,
    logical_expr::{
        Accumulator, AggregateUDF, AggregateUDFImpl, ColumnarValue, Operator, ScalarFunctionArgs, ScalarUDF,
        ScalarUDFImpl, Signature, Volatility, function::AccumulatorArgs,
    },
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, DynamicFilterPhysicalExpr, IsNotNullExpr, IsNullExpr, LikeExpr,
            NegativeExpr, NotExpr, TryCastExpr, binary, col, in_list, lit,
        },
    },
    physical_plan::joins::utils::{ColumnIndex, JoinFilter},
};
use query_planning::expr::{
    AggregateFunction as NativeAggregateFunction, BinaryOp as NativeBinaryOp, Expr as NativeExpr,
    LiteralValue as NativeLiteralValue, UnaryMathOp,
};

use super::{
    ExpressionCapability, ScalarFunctionKind, SupportedBinaryOperator,
    join_residual::validate_join_residual_scalar_function, join_residual_cast_mode, native_aggregate_function_kind,
    validate_join_residual, validate_physical_expr_contract,
};
use crate::{capability::CapabilityReason, native::lowering::lower_physical_expr_for_test};

fn scalar_function(
    name: &str,
    function: Arc<ScalarUDF>,
    args: Vec<Arc<dyn PhysicalExpr>>,
    output_type: DataType,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(ScalarFunctionExpr::new(
        name,
        function,
        args,
        Arc::new(Field::new(name, output_type, true)),
        Arc::new(ConfigOptions::new()),
    ))
}

fn assert_validation_lowering_parity(label: &str, expr: &Arc<dyn PhysicalExpr>, schema: &Arc<Schema>) {
    validate_physical_expr_contract(expr, schema).unwrap_or_else(|error| panic!("{label} must validate: {error:?}"));
    lower_physical_expr_for_test(expr, schema).unwrap_or_else(|error| panic!("{label} must lower: {error:?}"));
}

#[test]
fn binary_capability_rejection_is_structured_and_preserves_public_reason() {
    let rejection = SupportedBinaryOperator::from_datafusion(&Operator::RegexMatch).unwrap_err();
    assert_eq!(rejection.kind(), super::rejection::ExpressionRejectionKind::UnsupportedExpression);
    assert_eq!(rejection.capability(), ExpressionCapability::BinaryOperator);
    assert_eq!(rejection.capability_reason(), CapabilityReason::UnsupportedExpression);
}

#[test]
fn join_residual_cast_targets_are_owned_by_the_contract() {
    assert_eq!(join_residual_cast_mode(&DataType::Int64).unwrap(), super::JoinResidualCastMode::Ast);
    assert_eq!(
        join_residual_cast_mode(&DataType::Decimal128(12, 2)).unwrap(),
        super::JoinResidualCastMode::Materialize
    );
    let rejection = join_residual_cast_mode(&DataType::Date32).unwrap_err();
    assert_eq!(rejection.capability(), ExpressionCapability::JoinResidualCastTarget);
}

#[test]
fn join_residual_scalar_functions_add_a_stricter_contract_layer() {
    validate_join_residual_scalar_function(ScalarFunctionKind::Substring).unwrap();
    let rejection = validate_join_residual_scalar_function(ScalarFunctionKind::Abs).unwrap_err();
    assert_eq!(rejection.capability(), ExpressionCapability::JoinResidualScalarFunction);
}

#[test]
fn join_residual_rejects_a_column_index_missing_from_its_declared_side_schema() {
    let left_schema = Arc::new(Schema::empty());
    let right_schema = Arc::new(Schema::new(vec![Field::new("right", DataType::Int32, false)]));
    let filter_schema = Arc::new(Schema::new(vec![
        Field::new("left", DataType::Int32, false),
        Field::new("right", DataType::Int32, false),
    ]));
    let expr = binary(
        col("left", &filter_schema).unwrap(),
        Operator::Eq,
        col("right", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let filter = JoinFilter::new(
        expr,
        vec![ColumnIndex { index: 1, side: JoinSide::Left }, ColumnIndex { index: 0, side: JoinSide::Right }],
        filter_schema,
    );

    let rejection = validate_join_residual(&filter, &left_schema, &right_schema).unwrap_err();

    assert_eq!(rejection.capability(), ExpressionCapability::JoinResidualSideSchema);
    assert_eq!(rejection.capability_reason(), CapabilityReason::UnsupportedExpression);
}

#[test]
fn validated_expression_and_lowering_have_capability_parity() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let expr: Arc<dyn PhysicalExpr> =
        binary(col("value", &schema).unwrap(), Operator::Plus, lit(ScalarValue::Int64(Some(1))), &schema).unwrap();

    let validated = validate_physical_expr_contract(&expr, &schema).expect("expression should satisfy contract");
    assert!(Arc::ptr_eq(validated.expr(), &expr));
    let lowered = lower_physical_expr_for_test(&expr, &schema).expect("validated expression must lower");
    assert!(matches!(lowered, NativeExpr::Binary { op: NativeBinaryOp::Add, .. }));
}

#[test]
fn every_supported_top_level_expression_shape_validates_then_lowers() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("number", DataType::Int64, true),
        Field::new("text", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("date", DataType::Date32, true),
    ]));
    let number = || col("number", &schema).unwrap();
    let text = || col("text", &schema).unwrap();
    let flag = || col("flag", &schema).unwrap();

    let interval = lit(ScalarValue::IntervalMonthDayNano(Some(IntervalMonthDayNanoType::make_value(0, 1, 0))));
    let case: Arc<dyn PhysicalExpr> = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(flag(), lit(ScalarValue::Int64(Some(1))))],
            Some(lit(ScalarValue::Int64(Some(0)))),
        )
        .unwrap(),
    );
    let dynamic_filter: Arc<dyn PhysicalExpr> = Arc::new(DynamicFilterPhysicalExpr::new(vec![flag()], flag()));
    let shapes: Vec<(&str, Arc<dyn PhysicalExpr>)> = vec![
        ("column", number()),
        ("literal", lit(ScalarValue::Int64(Some(1)))),
        ("binary", binary(number(), Operator::Plus, lit(ScalarValue::Int64(Some(1))), &schema).unwrap()),
        ("interval binary", Arc::new(BinaryExpr::new(col("date", &schema).unwrap(), Operator::Plus, interval))),
        ("not", Arc::new(NotExpr::new(flag()))),
        ("negative", Arc::new(NegativeExpr::new(number()))),
        ("is null", Arc::new(IsNullExpr::new(number()))),
        ("is not null", Arc::new(IsNotNullExpr::new(number()))),
        ("cast", Arc::new(CastExpr::new(number(), DataType::Float64, None))),
        ("like", Arc::new(LikeExpr::new(false, false, text(), lit(ScalarValue::Utf8(Some("a%".into())))))),
        ("in list", in_list(number(), vec![lit(ScalarValue::Int64(Some(1)))], &false, schema.as_ref()).unwrap()),
        ("case", case),
        ("dynamic filter", dynamic_filter),
        ("scalar function", scalar_function("lower", string::lower(), vec![text()], DataType::Utf8)),
        ("try cast", Arc::new(TryCastExpr::new(text(), DataType::Int64))),
    ];

    for (label, expr) in shapes {
        assert_validation_lowering_parity(label, &expr, &schema);
    }
}

#[test]
fn negative_expression_preserves_structured_rejection_for_scalar_overflow_and_temporal_types() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("number", DataType::Int64, false),
        Field::new("date", DataType::Date32, false),
    ]));
    let array_negation = Arc::new(NegativeExpr::new(col("number", &schema).unwrap())) as Arc<dyn PhysicalExpr>;
    assert_validation_lowering_parity("integer array negation", &array_negation, &schema);
    for expr in [
        Arc::new(NegativeExpr::new(lit(ScalarValue::Int64(Some(i64::MIN))))) as Arc<dyn PhysicalExpr>,
        Arc::new(NegativeExpr::new(Arc::new(CastExpr::new(
            lit(ScalarValue::Int64(Some(i64::MIN))),
            DataType::Int64,
            None,
        )))),
        Arc::new(NegativeExpr::new(col("date", &schema).unwrap())),
    ] {
        let rejection = validate_physical_expr_contract(&expr, &schema).unwrap_err();
        assert_eq!(rejection.capability(), ExpressionCapability::NegativeExpression);
        assert!(lower_physical_expr_for_test(&expr, &schema).is_err());
    }
}

#[test]
fn every_admitted_binary_operator_validates_then_lowers() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("integer", DataType::Int64, true),
        Field::new("float", DataType::Float64, true),
        Field::new("text", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let integer = || col("integer", &schema).unwrap();
    let float = || col("float", &schema).unwrap();
    let text = || col("text", &schema).unwrap();
    let flag = || col("flag", &schema).unwrap();
    let cases = [
        ("add", Operator::Plus, integer(), integer()),
        ("subtract", Operator::Minus, integer(), integer()),
        ("multiply", Operator::Multiply, integer(), integer()),
        ("divide", Operator::Divide, float(), float()),
        ("bitwise and", Operator::BitwiseAnd, integer(), integer()),
        ("bitwise or", Operator::BitwiseOr, integer(), integer()),
        ("bitwise xor", Operator::BitwiseXor, integer(), integer()),
        ("shift left", Operator::BitwiseShiftLeft, integer(), integer()),
        ("shift right", Operator::BitwiseShiftRight, integer(), integer()),
        ("equal", Operator::Eq, integer(), integer()),
        ("not equal", Operator::NotEq, integer(), integer()),
        ("less than", Operator::Lt, integer(), integer()),
        ("less than or equal", Operator::LtEq, integer(), integer()),
        ("greater than", Operator::Gt, integer(), integer()),
        ("greater than or equal", Operator::GtEq, integer(), integer()),
        ("and", Operator::And, flag(), flag()),
        ("or", Operator::Or, flag(), flag()),
        ("string concat", Operator::StringConcat, text(), text()),
        ("distinct", Operator::IsDistinctFrom, integer(), integer()),
        ("not distinct", Operator::IsNotDistinctFrom, integer(), integer()),
    ];

    for (label, operator, left, right) in cases {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(left, operator, right));
        assert_validation_lowering_parity(label, &expr, &schema);
    }
}

#[test]
fn floor_and_ceil_admit_only_floating_point_arguments() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("whole", DataType::Int64, true),
        Field::new("exact", DataType::Decimal128(10, 2), true),
        Field::new("single", DataType::Float32, true),
    ]));
    for (name, function) in [("floor", math::floor()), ("ceil", math::ceil())] {
        let single =
            scalar_function(name, Arc::clone(&function), vec![col("single", &schema).unwrap()], DataType::Float32);
        assert_validation_lowering_parity(name, &single, &schema);
        let whole = scalar_function(name, Arc::clone(&function), vec![col("whole", &schema).unwrap()], DataType::Int64);
        assert!(validate_physical_expr_contract(&whole, &schema).is_err(), "{name} over an integer");
        let exact = scalar_function(name, function, vec![col("exact", &schema).unwrap()], DataType::Decimal128(10, 2));
        assert!(validate_physical_expr_contract(&exact, &schema).is_err(), "{name} over a decimal");
    }
}

#[test]
fn integer_division_remainder_and_checked_casts_validate_and_lower_inside_case_branches() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, true),
        Field::new("narrow", DataType::Int32, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let x = || col("x", &schema).unwrap();
    let y = || col("y", &schema).unwrap();
    let flag = || col("flag", &schema).unwrap();
    let zero = || lit(ScalarValue::Int64(Some(0)));
    let case = |when: Arc<dyn PhysicalExpr>, then: Arc<dyn PhysicalExpr>, otherwise: Arc<dyn PhysicalExpr>| {
        Arc::new(CaseExpr::try_new(None, vec![(when, then)], Some(otherwise)).unwrap()) as Arc<dyn PhysicalExpr>
    };
    let divide = |left, divisor| binary(left, Operator::Divide, divisor, &schema).unwrap();
    let narrowing = || Arc::new(CastExpr::new(x(), DataType::Int32, None)) as Arc<dyn PhysicalExpr>;
    let lower = |label: &str, expr: &Arc<dyn PhysicalExpr>| {
        assert_validation_lowering_parity(label, expr, &schema);
        lower_physical_expr_for_test(expr, &schema).unwrap()
    };
    // The operand that native evaluation checks, which a CASE branch restricts to the rows that reach it.
    let checked_operand = |expr: &NativeExpr| match expr {
        NativeExpr::Binary { right, .. } => (**right).clone(),
        NativeExpr::Cast { expr, .. } => (**expr).clone(),
        other => panic!("not a division or cast: {other:?}"),
    };
    let then_of_first_branch = |expr: &NativeExpr| match expr {
        NativeExpr::Case { branches, .. } => branches[0].then.clone(),
        other => panic!("not a CASE: {other:?}"),
    };
    let else_of = |expr: &NativeExpr| match expr {
        NativeExpr::Case { else_expr, .. } => (**else_expr).clone(),
        other => panic!("not a CASE: {other:?}"),
    };
    let is_restricted = |operand: &NativeExpr| matches!(operand, NativeExpr::Case { .. });

    for (label, operator) in [("divide", Operator::Divide), ("modulo", Operator::Modulo)] {
        let expr = binary(x(), operator, y(), &schema).unwrap();
        let lowered = lower(label, &expr);
        assert!(
            !is_restricted(&checked_operand(&lowered)),
            "{label} outside CASE evaluates every row, as DataFusion does"
        );
    }
    let to_text: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(x(), DataType::Utf8, None));
    lower("integer to text", &to_text);

    let divide_in_then = lower("variable divisor in THEN", &case(flag(), divide(x(), y()), zero()));
    assert!(is_restricted(&checked_operand(&then_of_first_branch(&divide_in_then))));
    let divide_in_else = lower("variable divisor in ELSE", &case(flag(), zero(), divide(x(), y())));
    assert!(is_restricted(&checked_operand(&else_of(&divide_in_else))));
    let cast_in_then = lower("narrowing cast in THEN", &case(flag(), narrowing(), lit(ScalarValue::Int32(Some(0)))));
    assert!(is_restricted(&checked_operand(&then_of_first_branch(&cast_in_then))));

    let second_when = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(flag(), zero()), (binary(divide(x(), y()), Operator::Gt, zero(), &schema).unwrap(), zero())],
            Some(zero()),
        )
        .unwrap(),
    ) as Arc<dyn PhysicalExpr>;
    let lowered = lower("variable divisor in the second WHEN", &second_when);
    let NativeExpr::Case { branches, .. } = &lowered else { panic!("not a CASE: {lowered:?}") };
    let NativeExpr::Binary { left: second_quotient, .. } = &branches[1].when else {
        panic!("not a comparison: {:?}", branches[1].when)
    };
    assert!(is_restricted(&checked_operand(second_quotient)));

    let first_when = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(binary(divide(x(), y()), Operator::Gt, zero(), &schema).unwrap(), zero())],
            Some(zero()),
        )
        .unwrap(),
    ) as Arc<dyn PhysicalExpr>;
    let lowered = lower("variable divisor in the first WHEN", &first_when);
    let NativeExpr::Case { branches, .. } = &lowered else { panic!("not a CASE: {lowered:?}") };
    let NativeExpr::Binary { left: first_quotient, .. } = &branches[0].when else {
        panic!("not a comparison: {:?}", branches[0].when)
    };
    assert!(!is_restricted(&checked_operand(first_quotient)), "the first WHEN is evaluated on every row");

    let literal_divisor =
        lower("literal divisor in THEN", &case(flag(), divide(x(), lit(ScalarValue::Int64(Some(20)))), zero()));
    assert!(!is_restricted(&checked_operand(&then_of_first_branch(&literal_divisor))));
    let widening =
        case(flag(), Arc::new(CastExpr::new(col("narrow", &schema).unwrap(), DataType::Int64, None)), zero());
    let lowered = lower("widening cast in THEN", &widening);
    assert!(!is_restricted(&checked_operand(&then_of_first_branch(&lowered))));
}

#[test]
fn a_long_chain_of_fallible_when_conditions_stays_on_datafusion_instead_of_growing_without_bound() {
    let schema =
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true), Field::new("y", DataType::Int64, true)]));
    let zero = || lit(ScalarValue::Int64(Some(0)));
    let case_of_dividing_conditions = |conditions: usize| -> Arc<dyn PhysicalExpr> {
        let whens = (0..conditions)
            .map(|_| {
                let quotient =
                    binary(col("x", &schema).unwrap(), Operator::Divide, col("y", &schema).unwrap(), &schema).unwrap();
                (binary(quotient, Operator::Gt, zero(), &schema).unwrap(), zero())
            })
            .collect();
        Arc::new(CaseExpr::try_new(None, whens, Some(zero())).unwrap())
    };

    assert_validation_lowering_parity("five dividing conditions", &case_of_dividing_conditions(5), &schema);

    let long_chain = case_of_dividing_conditions(16);
    validate_physical_expr_contract(&long_chain, &schema).expect("the contract admits the shape");
    let (reason, detail) = lower_physical_expr_for_test(&long_chain, &schema).unwrap_err().into_detail(None);
    assert_eq!(reason, CapabilityReason::UnsupportedExpression);
    let blocking = detail.blocking().expect("the rejection names the division");
    assert_eq!(blocking.kind(), "binary_operator");
}

fn right_operand(expr: &NativeExpr) -> &NativeExpr {
    let NativeExpr::Binary { right, .. } = expr else { panic!("not a binary expression: {expr:?}") };
    right
}

/// The divisor of the quotient on the left of the comparison `expr`.
fn compared_quotient_divisor(expr: &NativeExpr) -> &NativeExpr {
    let NativeExpr::Binary { left: quotient, .. } = expr else { panic!("not a comparison: {expr:?}") };
    let NativeExpr::Binary { right: divisor, .. } = quotient.as_ref() else { panic!("not a quotient: {quotient:?}") };
    divisor
}

/// Splits a divisor that lowering restricted into its activation and the operand it restricts.
fn restriction_of(divisor: &NativeExpr) -> (&NativeExpr, &NativeExpr) {
    let NativeExpr::Case { branches, .. } = divisor else { panic!("the divisor is not restricted: {divisor:?}") };
    let [branch] = branches.as_slice() else { panic!("a restriction has one branch: {divisor:?}") };
    (&branch.when, &branch.then)
}

/// Evaluates an activation, which is built from CASE, NOT, boolean literals, and columns, in three-valued logic.
fn activation_value(activation: &NativeExpr, columns: &[(&str, Option<bool>)]) -> Option<bool> {
    match activation {
        NativeExpr::Column(column) => columns.iter().find(|(name, _)| *name == column.name()).expect("column value").1,
        NativeExpr::Literal(NativeLiteralValue::Boolean(value)) => Some(*value),
        NativeExpr::Literal(NativeLiteralValue::Null(_)) => None,
        NativeExpr::Not(inner) => activation_value(inner, columns).map(|value| !value),
        NativeExpr::Case { branches, else_expr } => branches
            .iter()
            .find(|branch| activation_value(&branch.when, columns) == Some(true))
            .map_or_else(|| activation_value(else_expr, columns), |branch| activation_value(&branch.then, columns)),
        other => panic!("not an activation: {other:?}"),
    }
}

#[test]
fn a_fallible_right_operand_of_and_or_is_evaluated_only_where_the_left_operand_does_not_decide() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("gate", DataType::Boolean, true),
    ]));
    let x = || col("x", &schema).unwrap();
    let y = || col("y", &schema).unwrap();
    let flag = || col("flag", &schema).unwrap();
    let gate = || col("gate", &schema).unwrap();
    let logical = |left, operator, right| binary(left, operator, right, &schema).unwrap();
    let quotient_exceeds_one = || {
        let quotient = binary(y(), Operator::Divide, x(), &schema).unwrap();
        binary(quotient, Operator::Gt, lit(ScalarValue::Int64(Some(1))), &schema).unwrap()
    };
    let lower = |label: &str, expr: &Arc<dyn PhysicalExpr>| {
        assert_validation_lowering_parity(label, expr, &schema);
        lower_physical_expr_for_test(expr, &schema).unwrap()
    };
    let tri = [Some(true), Some(false), None];

    let and = lower("AND", &logical(flag(), Operator::And, quotient_exceeds_one()));
    let (activation, restricted) = restriction_of(compared_quotient_divisor(right_operand(&and)));
    assert_eq!(restricted, &NativeExpr::column("x"));
    for flag_value in tri {
        // FALSE decides an AND. A NULL left operand leaves the right operand's result open.
        let expected = flag_value != Some(false);
        assert_eq!(activation_value(activation, &[("flag", flag_value)]) == Some(true), expected, "AND {flag_value:?}");
    }

    let or = lower("OR", &logical(flag(), Operator::Or, quotient_exceeds_one()));
    let (activation, _) = restriction_of(compared_quotient_divisor(right_operand(&or)));
    for flag_value in tri {
        let expected = flag_value != Some(true);
        assert_eq!(activation_value(activation, &[("flag", flag_value)]) == Some(true), expected, "OR {flag_value:?}");
    }

    let left_division = lower("division on the left", &logical(quotient_exceeds_one(), Operator::And, flag()));
    let NativeExpr::Binary { left, .. } = &left_division else { panic!("not a binary expression") };
    assert_eq!(compared_quotient_divisor(left), &NativeExpr::column("x"), "the left operand is evaluated on every row");

    let infallible = lower(
        "no fallible operation on the right",
        &logical(flag(), Operator::And, binary(y(), Operator::Gt, lit(ScalarValue::Int64(Some(1))), &schema).unwrap()),
    );
    assert!(
        !infallible.any(|expr| matches!(expr, NativeExpr::Case { .. })),
        "no activation without a fallible operand"
    );

    // Inside a CASE branch the right operand also stays inside the branch.
    let guarded = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(gate(), logical(flag(), Operator::And, quotient_exceeds_one()))],
            Some(lit(ScalarValue::Boolean(Some(false)))),
        )
        .unwrap(),
    ) as Arc<dyn PhysicalExpr>;
    let lowered = lower("AND in a CASE branch", &guarded);
    let NativeExpr::Case { branches, .. } = &lowered else { panic!("not a CASE: {lowered:?}") };
    let (activation, _) = restriction_of(compared_quotient_divisor(right_operand(&branches[0].then)));
    for (gate_value, flag_value) in tri.into_iter().flat_map(|gate| tri.into_iter().map(move |flag| (gate, flag))) {
        let expected = gate_value == Some(true) && flag_value != Some(false);
        assert_eq!(
            activation_value(activation, &[("gate", gate_value), ("flag", flag_value)]) == Some(true),
            expected,
            "gate {gate_value:?}, flag {flag_value:?}"
        );
    }

    // The right operand of an outer AND restricts an inner AND by both left operands.
    let nested = logical(flag(), Operator::And, logical(gate(), Operator::And, quotient_exceeds_one()));
    let lowered = lower("AND in the right operand of an AND", &nested);
    let (activation, _) = restriction_of(compared_quotient_divisor(right_operand(right_operand(&lowered))));
    for (gate_value, flag_value) in tri.into_iter().flat_map(|gate| tri.into_iter().map(move |flag| (gate, flag))) {
        let expected = gate_value != Some(false) && flag_value != Some(false);
        assert_eq!(
            activation_value(activation, &[("gate", gate_value), ("flag", flag_value)]) == Some(true),
            expected,
            "gate {gate_value:?}, flag {flag_value:?}"
        );
    }
}

#[test]
fn a_long_chain_of_fallible_and_conditions_stays_on_datafusion_instead_of_growing_without_bound() {
    let schema =
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true), Field::new("y", DataType::Int64, true)]));
    let zero = || lit(ScalarValue::Int64(Some(0)));
    let conjunction_of_guarded_quotients = |conjuncts: usize| -> Arc<dyn PhysicalExpr> {
        (0..conjuncts)
            .map(|_| {
                let guard = binary(col("x", &schema).unwrap(), Operator::NotEq, zero(), &schema).unwrap();
                let quotient =
                    binary(col("y", &schema).unwrap(), Operator::Divide, col("x", &schema).unwrap(), &schema).unwrap();
                let exceeds = binary(quotient, Operator::Gt, zero(), &schema).unwrap();
                binary(guard, Operator::And, exceeds, &schema).unwrap()
            })
            .reduce(|left, right| binary(left, Operator::And, right, &schema).unwrap())
            .unwrap()
    };

    assert_validation_lowering_parity("three guarded quotients", &conjunction_of_guarded_quotients(3), &schema);

    let long_chain = conjunction_of_guarded_quotients(16);
    validate_physical_expr_contract(&long_chain, &schema).expect("the contract admits the shape");
    let (reason, detail) = lower_physical_expr_for_test(&long_chain, &schema).unwrap_err().into_detail(None);
    assert_eq!(reason, CapabilityReason::UnsupportedExpression);
    let blocking = detail.blocking().expect("the rejection names the division");
    assert_eq!(blocking.kind(), "binary_operator");
}

#[test]
fn decimal_division_is_restricted_inside_case_branches_whatever_the_divisor() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Decimal128(10, 2), true),
        Field::new("b", DataType::Decimal128(10, 2), true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let divisors: [(&str, Arc<dyn PhysicalExpr>); 2] = [
        ("column divisor", col("b", &schema).unwrap()),
        ("literal divisor", lit(ScalarValue::Decimal128(Some(250), 10, 2))),
    ];
    for (label, divisor) in divisors {
        let quotient = binary(col("a", &schema).unwrap(), Operator::Divide, divisor, &schema).unwrap();
        let lowered = lower_physical_expr_for_test(&quotient, &schema).unwrap();
        assert!(
            !matches!(right_operand(&lowered), NativeExpr::Case { .. }),
            "{label}: a division outside CASE evaluates every row, as DataFusion does"
        );

        let guarded = Arc::new(
            CaseExpr::try_new(None, vec![(col("flag", &schema).unwrap(), Arc::clone(&quotient))], None).unwrap(),
        ) as Arc<dyn PhysicalExpr>;
        assert_validation_lowering_parity(label, &guarded, &schema);
        let lowered = lower_physical_expr_for_test(&guarded, &schema).unwrap();
        let NativeExpr::Case { branches, .. } = &lowered else { panic!("not a CASE: {lowered:?}") };
        assert!(
            matches!(right_operand(&branches[0].then), NativeExpr::Case { .. }),
            "{label}: a division in a CASE branch checks only the rows that reach the branch"
        );
    }
}

#[test]
fn join_residual_rejects_decimal_division_because_it_is_not_restricted_to_the_rows_a_case_guards() {
    let decimal = || DataType::Decimal128(10, 2);
    let left_schema = Arc::new(Schema::new(vec![Field::new("l", decimal(), false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r", decimal(), false)]));
    let filter_schema =
        Arc::new(Schema::new(vec![Field::new("l", decimal(), false), Field::new("r", decimal(), false)]));
    let filter = |expr: Arc<dyn PhysicalExpr>| {
        JoinFilter::new(
            expr,
            vec![ColumnIndex { index: 0, side: JoinSide::Left }, ColumnIndex { index: 0, side: JoinSide::Right }],
            Arc::clone(&filter_schema),
        )
    };
    let quotient =
        binary(col("l", &filter_schema).unwrap(), Operator::Divide, col("r", &filter_schema).unwrap(), &filter_schema)
            .unwrap();
    let comparison =
        binary(quotient, Operator::Gt, lit(ScalarValue::Decimal128(Some(100), 10, 2)), &filter_schema).unwrap();

    let rejection = validate_join_residual(&filter(comparison), &left_schema, &right_schema).map(|_| ()).unwrap_err();

    assert_eq!(rejection.capability(), ExpressionCapability::JoinResidualShape);
}

#[test]
fn join_residual_ast_rejects_integer_division_and_casts_that_can_fail() {
    let left_schema = Arc::new(Schema::new(vec![Field::new("l", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r", DataType::Int64, false)]));
    let filter_schema =
        Arc::new(Schema::new(vec![Field::new("l", DataType::Int32, false), Field::new("r", DataType::Int64, false)]));
    let cast = |name: &str, target: DataType| -> Arc<dyn PhysicalExpr> {
        Arc::new(CastExpr::new(col(name, &filter_schema).unwrap(), target, None))
    };
    let filter = |expr: Arc<dyn PhysicalExpr>| {
        JoinFilter::new(
            expr,
            vec![ColumnIndex { index: 0, side: JoinSide::Left }, ColumnIndex { index: 0, side: JoinSide::Right }],
            Arc::clone(&filter_schema),
        )
    };
    let validate =
        |expr: Arc<dyn PhysicalExpr>| validate_join_residual(&filter(expr), &left_schema, &right_schema).map(|_| ());

    let failing_cast =
        binary(cast("l", DataType::UInt64), Operator::Eq, cast("r", DataType::UInt64), &filter_schema).unwrap();
    assert_eq!(validate(failing_cast).unwrap_err().capability(), ExpressionCapability::CastPair);

    let safe_cast =
        binary(cast("l", DataType::Int64), Operator::Eq, col("r", &filter_schema).unwrap(), &filter_schema).unwrap();
    validate(safe_cast).expect("an Int32 to Int64 cast cannot fail");

    for operator in [Operator::Divide, Operator::Modulo] {
        let arithmetic =
            binary(cast("l", DataType::Int64), operator, col("r", &filter_schema).unwrap(), &filter_schema).unwrap();
        let comparison = binary(arithmetic, Operator::Gt, lit(ScalarValue::Int64(Some(1))), &filter_schema).unwrap();
        let rejection = validate(comparison).unwrap_err();
        assert_eq!(rejection.capability(), ExpressionCapability::JoinResidualShape, "{operator:?}");
    }
}

#[test]
fn every_supported_scalar_function_kind_validates_then_lowers() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, true),
        Field::new("number", DataType::Float64, true),
        Field::new("epoch", DataType::Int64, true),
        Field::new("timestamp", DataType::Timestamp(TimeUnit::Second, None), true),
    ]));
    let text = || col("text", &schema).unwrap();
    let number = || col("number", &schema).unwrap();
    let epoch = || col("epoch", &schema).unwrap();
    let timestamp = || col("timestamp", &schema).unwrap();
    let utf8 = |value: &str| lit(ScalarValue::Utf8(Some(value.to_string())));
    let integer = |value| lit(ScalarValue::Int64(Some(value)));
    let functions = vec![
        (
            ScalarFunctionKind::Substring,
            scalar_function("substring", unicode::substr(), vec![text(), integer(1), integer(2)], DataType::Utf8),
        ),
        (
            ScalarFunctionKind::Extract,
            scalar_function("date_part", datetime::date_part(), vec![utf8("year"), timestamp()], DataType::Float64),
        ),
        (
            ScalarFunctionKind::DateTrunc,
            scalar_function(
                "date_trunc",
                datetime::date_trunc(),
                vec![utf8("second"), timestamp()],
                DataType::Timestamp(TimeUnit::Second, None),
            ),
        ),
        (
            ScalarFunctionKind::ToTimestampSeconds,
            scalar_function(
                "to_timestamp_seconds",
                datetime::to_timestamp_seconds(&ConfigOptions::new()),
                vec![epoch()],
                DataType::Timestamp(TimeUnit::Second, None),
            ),
        ),
        (ScalarFunctionKind::Lower, scalar_function("lower", string::lower(), vec![text()], DataType::Utf8)),
        (ScalarFunctionKind::Upper, scalar_function("upper", string::upper(), vec![text()], DataType::Utf8)),
        (ScalarFunctionKind::Trim, scalar_function("btrim", string::btrim(), vec![text()], DataType::Utf8)),
        (
            ScalarFunctionKind::LeftTrim,
            scalar_function("ltrim", string::ltrim(), vec![text(), utf8("x")], DataType::Utf8),
        ),
        (
            ScalarFunctionKind::RightTrim,
            scalar_function("rtrim", string::rtrim(), vec![text(), utf8("x")], DataType::Utf8),
        ),
        (
            ScalarFunctionKind::Contains,
            scalar_function("contains", string::contains(), vec![text(), utf8("x")], DataType::Boolean),
        ),
        (
            ScalarFunctionKind::StartsWith,
            scalar_function("starts_with", string::starts_with(), vec![text(), utf8("x")], DataType::Boolean),
        ),
        (
            ScalarFunctionKind::EndsWith,
            scalar_function("ends_with", string::ends_with(), vec![text(), utf8("x")], DataType::Boolean),
        ),
        (
            ScalarFunctionKind::StringLength,
            scalar_function("character_length", unicode::character_length(), vec![text()], DataType::Int64),
        ),
        (
            ScalarFunctionKind::RegexpReplace,
            scalar_function(
                "regexp_replace",
                regex::regexp_replace(),
                vec![text(), utf8("x"), utf8("y")],
                DataType::Utf8,
            ),
        ),
        (
            ScalarFunctionKind::Concat,
            scalar_function("concat", string::concat(), vec![text(), utf8("x")], DataType::Utf8),
        ),
        (ScalarFunctionKind::Abs, scalar_function("abs", math::abs(), vec![number()], DataType::Float64)),
        (
            ScalarFunctionKind::UnaryMath(UnaryMathOp::Sqrt),
            scalar_function("sqrt", math::sqrt(), vec![number()], DataType::Float64),
        ),
        (
            ScalarFunctionKind::UnaryMath(UnaryMathOp::Ln),
            scalar_function("ln", math::ln(), vec![number()], DataType::Float64),
        ),
        (
            ScalarFunctionKind::UnaryMath(UnaryMathOp::Exp),
            scalar_function("exp", math::exp(), vec![number()], DataType::Float64),
        ),
        (
            ScalarFunctionKind::UnaryMath(UnaryMathOp::Floor),
            scalar_function("floor", math::floor(), vec![number()], DataType::Float64),
        ),
        (
            ScalarFunctionKind::UnaryMath(UnaryMathOp::Ceil),
            scalar_function("ceil", math::ceil(), vec![number()], DataType::Float64),
        ),
        (
            ScalarFunctionKind::Round,
            scalar_function("round", math::round(), vec![number(), integer(2)], DataType::Float64),
        ),
        (
            ScalarFunctionKind::NullIf,
            scalar_function(
                "nullif",
                core::nullif(),
                vec![number(), lit(ScalarValue::Float64(Some(0.0)))],
                DataType::Float64,
            ),
        ),
    ];

    for (kind, expr) in functions {
        let function = expr.downcast_ref::<ScalarFunctionExpr>().unwrap();
        assert_eq!(ScalarFunctionKind::from_scalar_function(function).unwrap(), kind);
        assert_validation_lowering_parity(function.name(), &expr, &schema);
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ShadowingScalarFunc {
    signature: Signature,
}

impl ScalarUDFImpl for ShadowingScalarFunc {
    fn name(&self) -> &str {
        "upper"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        unreachable!("shadowing test UDFs are never executed")
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ShadowingMedianFunc {
    signature: Signature,
}

impl AggregateUDFImpl for ShadowingMedianFunc {
    fn name(&self) -> &str {
        "median"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::Float64)
    }

    fn accumulator(&self, _args: AccumulatorArgs) -> DataFusionResult<Box<dyn Accumulator>> {
        unreachable!("shadowing test UDAFs are never executed")
    }
}

#[test]
fn scalar_function_shadowed_by_name_is_rejected() {
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, true)]));
    let shadowing = Arc::new(ScalarUDF::new_from_impl(ShadowingScalarFunc {
        signature: Signature::string(1, Volatility::Immutable),
    }));
    let expr = scalar_function("upper", shadowing, vec![col("text", &schema).unwrap()], DataType::Utf8);
    let function = expr.downcast_ref::<ScalarFunctionExpr>().unwrap();
    assert_eq!(function.name(), "upper");
    assert!(ScalarFunctionKind::from_scalar_function(function).is_err());
    assert!(validate_physical_expr_contract(&expr, &schema).is_err());
}

#[test]
fn aggregate_function_shadowed_by_name_resolves_to_no_native_kind() {
    let shadowing = AggregateUDF::new_from_impl(ShadowingMedianFunc {
        signature: Signature::uniform(1, vec![DataType::Float64], Volatility::Immutable),
    });
    assert_eq!(shadowing.name(), "median");
    assert_eq!(native_aggregate_function_kind(&shadowing), None);
    assert_eq!(
        native_aggregate_function_kind(&AggregateUDF::new_from_impl(Median::default())),
        Some(NativeAggregateFunction::Median)
    );
}
