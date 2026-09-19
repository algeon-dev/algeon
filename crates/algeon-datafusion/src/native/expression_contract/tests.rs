use std::sync::Arc;

use arrow::datatypes::{DataType, Field, IntervalMonthDayNanoType, Schema, TimeUnit};
use datafusion::{
    common::{JoinSide, ScalarValue},
    config::ConfigOptions,
    functions::{core, datetime, math, regex, string, unicode},
    logical_expr::{Operator, ScalarUDF},
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, DynamicFilterPhysicalExpr, IsNotNullExpr, IsNullExpr, LikeExpr, NotExpr,
            TryCastExpr, binary, col, in_list, lit,
        },
    },
    physical_plan::joins::utils::{ColumnIndex, JoinFilter},
};
use query_engine::expr::{BinaryOp as NativeBinaryOp, Expr as NativeExpr, UnaryMathOp};

use super::{
    ExpressionCapability, ScalarFunctionKind, SupportedBinaryOperator,
    join_residual::validate_join_residual_scalar_function, join_residual_cast_mode, validate_join_residual,
    validate_physical_expr_contract,
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
        assert_eq!(ScalarFunctionKind::from_name(function.name()).unwrap(), kind);
        assert_validation_lowering_parity(function.name(), &expr, &schema);
    }
}
