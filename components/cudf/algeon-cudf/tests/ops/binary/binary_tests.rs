//! Tests for binary operations (numeric and string).
use cudf::{
    DataFrame, DataType, ErrorCode, ErrorKind, ErrorStatus, Exec, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::{
        binary::{self, BinaryOperator},
        expr::{Expr, col},
    },
    testing::prelude::*,
};

use crate::common::{
    explicit_i32_series, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, nullable_series,
    reset_exec_observation, reset_sync_observation, run_named_cases, series,
};

fn test_div_scalar_column() {
    let df = DataFrame::from_series([("v", series::<f64>("v", &[2.0, 4.0, 5.0]).unwrap())]).unwrap();

    let result = (100.0f64 / col("v")).eval(&df).unwrap();

    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![50.0, 25.0, 20.0]);
}

fn test_binary_operation_error_contains_operation_context() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = series::<i32>("b", &[10, 20]).unwrap(); // mismatched length

    let err = a.on(Exec::RETAINED).add(&b).expect_err("expected error");

    assert_eq!(err.operation(), "series::add");
    assert!(err.details().iter().any(|(k, _)| *k == "operator"));
}

fn test_null_logical_or_uses_sql_three_valued_logic() {
    // Rows pair LHS x RHS over {true, false, null}:
    //   (T,T) (T,F) (T,N) (F,T) (F,F) (F,N) (N,T) (N,F) (N,N)
    let lhs_values = [true, true, true, false, false, false, false, false, false];
    let lhs_validity = [true, true, true, true, true, true, false, false, false];
    let a = nullable_series("a", &lhs_values, &lhs_validity).unwrap();
    let rhs_values = [true, false, false, true, false, false, true, false, false];
    let rhs_validity = [true, true, false, true, true, false, true, true, false];
    let b = nullable_series("b", &rhs_values, &rhs_validity).unwrap();
    let df = DataFrame::from_series([("a", a), ("b", b)]).unwrap();
    let expr = Expr::BinaryOp { lhs: Box::new(col("a")), op: BinaryOperator::NullLogicalOr, rhs: Box::new(col("b")) };

    let result = expr.eval(&df).unwrap();

    // true||* = true; false||true = true; false||false = false; false||null = null;
    // null||true = true; null||false = null; null||null = null.
    assert_eq!(
        result.to_optional_vec::<bool>().unwrap(),
        vec![Some(true), Some(true), Some(true), Some(true), Some(false), None, Some(true), None, None]
    );
}

fn test_null_logical_and_uses_sql_three_valued_logic() {
    // Rows pair LHS x RHS over {true, false, null}:
    //   (T,T) (T,F) (T,N) (F,T) (F,F) (F,N) (N,T) (N,F) (N,N)
    let lhs_values = [true, true, true, false, false, false, false, false, false];
    let lhs_validity = [true, true, true, true, true, true, false, false, false];
    let a = nullable_series("a", &lhs_values, &lhs_validity).unwrap();
    let rhs_values = [true, false, false, true, false, false, true, false, false];
    let rhs_validity = [true, true, false, true, true, false, true, true, false];
    let b = nullable_series("b", &rhs_values, &rhs_validity).unwrap();
    let df = DataFrame::from_series([("a", a), ("b", b)]).unwrap();
    let expr = Expr::BinaryOp { lhs: Box::new(col("a")), op: BinaryOperator::NullLogicalAnd, rhs: Box::new(col("b")) };

    let result = expr.eval(&df).unwrap();

    // false&&* = false; true&&true = true; true&&false = false; true&&null = null;
    // null&&true = null; null&&false = false; null&&null = null.
    assert_eq!(
        result.to_optional_vec::<bool>().unwrap(),
        vec![Some(true), Some(false), None, Some(false), Some(false), Some(false), None, Some(false), None]
    );
}

fn test_mixed_domain_binary_operation_vv_uses_lhs_provenance() {
    let lhs = explicit_i32_series("lhs", &[1, 2, 3], 900);
    let rhs = explicit_i32_series("rhs", &[10, 20, 30], 901);
    let expected_stream = lhs.as_column().testing_execution_domain_stream_value();
    let expected_mr = lhs.as_column().testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = lhs.on(Exec::RETAINED).add(&rhs).unwrap();

    assert_eq!(result.as_column().testing_execution_domain_id(), lhs.as_column().testing_execution_domain_id());
    assert_eq!(result.as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

fn test_series_scalar_binary_operation_uses_series_provenance() {
    let lhs = explicit_i32_series("lhs", &[2, 3, 4], 902);
    let expected_stream = lhs.as_column().testing_execution_domain_stream_value();
    let expected_mr = lhs.as_column().testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = lhs.on(Exec::RETAINED).mul_scalar(10i32).unwrap();

    assert_eq!(result.as_column().testing_execution_domain_id(), lhs.as_column().testing_execution_domain_id());
    assert_eq!(result.as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

// ---------------------------------------------------------------------------
// String comparison: series vs series
// ---------------------------------------------------------------------------

fn test_string_equal_vv() {
    let a = series("a", &["foo", "bar", "baz"]).unwrap();
    let b = series("b", &["foo", "bar", "qux"]).unwrap();

    let result = a.on(Exec::RETAINED).eq(&b).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

fn test_string_not_equal_vv() {
    let a = series("a", &["foo", "bar", "baz"]).unwrap();
    let b = series("b", &["foo", "bar", "qux"]).unwrap();

    let result = a.on(Exec::RETAINED).ne(&b).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, false, true]);
}

fn test_string_less_vv() {
    let a = series("a", &["apple", "cherry", "banana"]).unwrap();
    let b = series("b", &["banana", "banana", "banana"]).unwrap();

    let result = a.on(Exec::RETAINED).lt(&b).unwrap();
    // "apple" < "banana" → true, "cherry" < "banana" → false, "banana" < "banana" → false
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn test_string_greater_vv() {
    let a = series("a", &["apple", "cherry", "banana"]).unwrap();
    let b = series("b", &["banana", "banana", "banana"]).unwrap();

    let result = a.on(Exec::RETAINED).gt(&b).unwrap();
    // "apple" > "banana" → false, "cherry" > "banana" → true, "banana" > "banana" → false
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_string_less_equal_vv() {
    let a = series("a", &["apple", "banana", "cherry"]).unwrap();
    let b = series("b", &["banana", "banana", "banana"]).unwrap();

    let result = a.on(Exec::RETAINED).le(&b).unwrap();
    // "apple" <= "banana" → true, "banana" <= "banana" → true, "cherry" <= "banana" → false
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

fn test_string_greater_equal_vv() {
    let a = series("a", &["apple", "banana", "cherry"]).unwrap();
    let b = series("b", &["banana", "banana", "banana"]).unwrap();

    let result = a.on(Exec::RETAINED).ge(&b).unwrap();
    // "apple" >= "banana" → false, "banana" >= "banana" → true, "cherry" >= "banana" → true
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// String edge cases
// ---------------------------------------------------------------------------

fn test_string_equal_empty() {
    let a = series("a", &["", "hello", ""]).unwrap();
    let b = series("b", &["", "", "hello"]).unwrap();

    let result = a.on(Exec::RETAINED).eq(&b).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn test_string_equal_case_sensitive() {
    let a = series("a", &["Foo", "BAR", "baz"]).unwrap();
    let b = series("b", &["foo", "bar", "baz"]).unwrap();

    let result = a.on(Exec::RETAINED).eq(&b).unwrap();
    // Case-sensitive: "Foo" != "foo", "BAR" != "bar", "baz" == "baz"
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, false, true]);
}

// ---------------------------------------------------------------------------
// String arithmetic errors
// ---------------------------------------------------------------------------

fn test_string_add_vv_error() {
    let a = series("a", &["foo", "bar"]).unwrap();
    let b = series("b", &["baz", "qux"]).unwrap();

    let result = a.on(Exec::RETAINED).add(&b);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// BinaryOperator::is_comparison
// ---------------------------------------------------------------------------

fn test_is_comparison_true_for_comparison_ops() {
    assert!(BinaryOperator::Equal.is_comparison());
    assert!(BinaryOperator::NotEqual.is_comparison());
    assert!(BinaryOperator::Less.is_comparison());
    assert!(BinaryOperator::Greater.is_comparison());
    assert!(BinaryOperator::LessEqual.is_comparison());
    assert!(BinaryOperator::GreaterEqual.is_comparison());
    assert!(BinaryOperator::NullEquals.is_comparison());
    assert!(BinaryOperator::NullNotEquals.is_comparison());
}

fn test_is_comparison_false_for_arithmetic_ops() {
    assert!(!BinaryOperator::Add.is_comparison());
    assert!(!BinaryOperator::Sub.is_comparison());
    assert!(!BinaryOperator::Mul.is_comparison());
    assert!(!BinaryOperator::Div.is_comparison());
    assert!(!BinaryOperator::TrueDiv.is_comparison());
    assert!(!BinaryOperator::FloorDiv.is_comparison());
    assert!(!BinaryOperator::Mod.is_comparison());
    assert!(!BinaryOperator::LogicalAnd.is_comparison());
    assert!(!BinaryOperator::BitwiseAnd.is_comparison());
}

// ---------------------------------------------------------------------------
// infer_binary_output_type
// ---------------------------------------------------------------------------

fn decimal64(scale: i32) -> DataType {
    DataType::fixed_point(TypeId::Decimal64, scale).unwrap()
}

fn infer_dtype(lhs: DataType, rhs: DataType, op: BinaryOperator) -> DataType {
    binary::infer_binary_output_dtype(lhs, rhs, op).unwrap()
}

fn operator_name(op: BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::TrueDiv => "true_div",
        BinaryOperator::FloorDiv => "floor_div",
        BinaryOperator::IntPow => "int_pow",
        BinaryOperator::LogBase => "log_base",
        BinaryOperator::ShiftLeft => "shift_left",
        BinaryOperator::ShiftRight => "shift_right",
        BinaryOperator::ShiftRightUnsigned => "shift_right_unsigned",
        BinaryOperator::BitwiseAnd => "bitwise_and",
        BinaryOperator::BitwiseOr => "bitwise_or",
        BinaryOperator::BitwiseXor => "bitwise_xor",
        BinaryOperator::LogicalAnd => "logical_and",
        BinaryOperator::LogicalOr => "logical_or",
        BinaryOperator::NotEqual => "not_equal",
        BinaryOperator::LessEqual => "less_equal",
        BinaryOperator::GreaterEqual => "greater_equal",
        BinaryOperator::NullEquals => "null_equals",
        BinaryOperator::NullNotEquals => "null_not_equals",
        BinaryOperator::NullMax => "null_max",
        BinaryOperator::NullMin => "null_min",
        BinaryOperator::NullLogicalAnd => "null_logical_and",
        BinaryOperator::NullLogicalOr => "null_logical_or",
        BinaryOperator::Add => "add",
        BinaryOperator::Sub => "sub",
        BinaryOperator::Mul => "mul",
        BinaryOperator::Div => "div",
        BinaryOperator::Mod => "mod",
        BinaryOperator::Pmod => "pmod",
        BinaryOperator::Pymod => "pymod",
        BinaryOperator::Pow => "pow",
        BinaryOperator::Atan2 => "atan2",
        BinaryOperator::Equal => "equal",
        BinaryOperator::Less => "less",
        BinaryOperator::Greater => "greater",
    }
}

fn assert_type_mismatch_facts(err: &cudf::Error, operation: &str, lhs: TypeId, rhs: TypeId, op: BinaryOperator) {
    assert_eq!(err.code(), cudf::ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), operation);
    assert_eq!(err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(lhs)));
    assert_eq!(err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(rhs)));
    assert_eq!(err.detail_value("operator"), Some(operator_name(op)));
}

fn test_infer_same_type_arithmetic() {
    // Int32 + Int32 → Int32
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::Int32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int32);
}

fn test_infer_int_float_promotion() {
    // Int32 + Float64 → Float64
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::Float64, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Float64);
}

fn test_infer_float_float_promotion() {
    // Float32 + Float64 → Float64
    let dt = binary::infer_binary_output_type(TypeId::Float32, TypeId::Float64, BinaryOperator::Mul).unwrap();
    assert_eq!(dt.id(), TypeId::Float64);
}

fn test_infer_int_int_promotion() {
    // Int16 + Int32 → Int32
    let dt = binary::infer_binary_output_type(TypeId::Int16, TypeId::Int32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int32);
}

fn test_infer_signed_unsigned_promotion() {
    // Int32 + UInt32 → Int64
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::UInt32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int64);
}

fn test_infer_comparison_always_bool() {
    // Any comparison op → Bool8 regardless of input types
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::Float64, BinaryOperator::Less).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);

    let dt = binary::infer_binary_output_type(TypeId::Int64, TypeId::Int32, BinaryOperator::Equal).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

fn test_infer_string_comparison_bool() {
    // String + String comparison → Bool8
    let dt = binary::infer_binary_output_type(TypeId::String, TypeId::String, BinaryOperator::Equal).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);

    let dt = binary::infer_binary_output_type(TypeId::String, TypeId::String, BinaryOperator::Less).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

fn test_infer_string_arithmetic_error() {
    // String + String arithmetic → error
    let result = binary::infer_binary_output_type(TypeId::String, TypeId::String, BinaryOperator::Add);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code(), cudf::ErrorCode::TypeMismatch);
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "binary::infer_binary_output_type");
}

fn test_infer_string_int_arithmetic_error() {
    // String + Int arithmetic → error
    let result = binary::infer_binary_output_type(TypeId::String, TypeId::Int32, BinaryOperator::Add);
    assert!(result.is_err());
}

fn test_infer_non_numeric_error() {
    // Non-numeric types (List, Struct, etc.) → error
    let result = binary::infer_binary_output_type(TypeId::List, TypeId::Int32, BinaryOperator::Add);
    assert!(result.is_err());

    let result = binary::infer_binary_output_type(TypeId::Struct, TypeId::Struct, BinaryOperator::Add);
    assert!(result.is_err());
}

fn test_infer_bool_arithmetic() {
    // Bool8 + Bool8 → Bool8
    let dt = binary::infer_binary_output_type(TypeId::Bool8, TypeId::Bool8, BinaryOperator::LogicalAnd).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

fn test_infer_error_context_keys() {
    // Canonical type ids are facts; operator remains a diagnostic detail.
    let err = binary::infer_binary_output_type(TypeId::String, TypeId::Int32, BinaryOperator::Add).unwrap_err();
    assert_eq!(err.code(), cudf::ErrorCode::TypeMismatch);
    assert_eq!(err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(err.detail_value("left_type_id"), None);
    assert_eq!(err.detail_value("right_type_id"), None);
    assert_eq!(err.detail_value("operator"), Some("add"));
}

fn test_infer_error_operator_detail_uses_stable_snake_case() {
    let err = binary::infer_binary_output_type(TypeId::String, TypeId::String, BinaryOperator::TrueDiv).unwrap_err();
    assert_eq!(err.detail_value("operator"), Some("true_div"));
}

fn test_infer_decimal_same_and_mixed_scale_add_sub_dtype() {
    let same = infer_dtype(decimal64(-2), decimal64(-2), BinaryOperator::Add);
    assert_eq!(same, decimal64(-2));

    let mixed_add = infer_dtype(decimal64(-2), decimal64(-4), BinaryOperator::Add);
    assert_eq!(mixed_add, decimal64(-4));

    let mixed_sub = infer_dtype(decimal64(-5), decimal64(-3), BinaryOperator::Sub);
    assert_eq!(mixed_sub, decimal64(-5));
}

fn test_infer_decimal_mul_div_scales_dtype() {
    let product = infer_dtype(decimal64(-2), decimal64(-4), BinaryOperator::Mul);
    assert_eq!(product, decimal64(-6));

    let quotient = infer_dtype(decimal64(-6), decimal64(-2), BinaryOperator::Div);
    assert_eq!(quotient, decimal64(-4));
}

fn test_infer_decimal_comparison_bool() {
    let type_dt = binary::infer_binary_output_type(TypeId::Decimal64, TypeId::Decimal64, BinaryOperator::Less).unwrap();
    assert_eq!(type_dt.id(), TypeId::Bool8);

    let dtype_dt = infer_dtype(decimal64(-2), decimal64(-4), BinaryOperator::GreaterEqual);
    assert_eq!(dtype_dt, DataType::new(TypeId::Bool8));
}

fn test_infer_mixed_resolution_temporal_comparison_bool() {
    let timestamp_dt =
        binary::infer_binary_output_type(TypeId::TimestampDays, TypeId::TimestampSeconds, BinaryOperator::Less)
            .unwrap();
    assert_eq!(timestamp_dt.id(), TypeId::Bool8);

    let duration_dt = binary::infer_binary_output_type(
        TypeId::DurationMilliseconds,
        TypeId::DurationMicroseconds,
        BinaryOperator::Equal,
    )
    .unwrap();
    assert_eq!(duration_dt.id(), TypeId::Bool8);
}

fn test_infer_temporal_typeid_rules() {
    let timestamp = TypeId::TimestampMicroseconds;
    let duration = TypeId::DurationMicroseconds;

    assert_eq!(binary::infer_binary_output_type(timestamp, duration, BinaryOperator::Add).unwrap().id(), timestamp);
    assert_eq!(binary::infer_binary_output_type(duration, timestamp, BinaryOperator::Add).unwrap().id(), timestamp);
    assert_eq!(binary::infer_binary_output_type(timestamp, duration, BinaryOperator::Sub).unwrap().id(), timestamp);
    assert_eq!(binary::infer_binary_output_type(timestamp, timestamp, BinaryOperator::Sub).unwrap().id(), duration);
    assert_eq!(binary::infer_binary_output_type(duration, TypeId::Int32, BinaryOperator::Mul).unwrap().id(), duration);
    assert_eq!(binary::infer_binary_output_type(duration, TypeId::Int32, BinaryOperator::Div).unwrap().id(), duration);

    let err = binary::infer_binary_output_type(
        TypeId::TimestampMicroseconds,
        TypeId::DurationMilliseconds,
        BinaryOperator::Add,
    )
    .unwrap_err();
    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_type",
        TypeId::TimestampMicroseconds,
        TypeId::DurationMilliseconds,
        BinaryOperator::Add,
    );
}

fn test_infer_decimal_float_arithmetic_rejected_with_type_facts_dtype() {
    let err = binary::infer_binary_output_dtype(decimal64(-2), DataType::new(TypeId::Float64), BinaryOperator::Add)
        .unwrap_err();

    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_dtype",
        TypeId::Decimal64,
        TypeId::Float64,
        BinaryOperator::Add,
    );
}

fn test_infer_same_resolution_timestamp_duration_arithmetic_dtype() {
    let timestamp = DataType::new(TypeId::TimestampMicroseconds);
    let duration = DataType::new(TypeId::DurationMicroseconds);

    assert_eq!(infer_dtype(timestamp, duration, BinaryOperator::Add), timestamp);
    assert_eq!(infer_dtype(duration, timestamp, BinaryOperator::Add), timestamp);
    assert_eq!(infer_dtype(timestamp, duration, BinaryOperator::Sub), timestamp);
}

fn test_infer_timestamp_timestamp_subtracts_to_matching_duration_dtype() {
    let timestamp = DataType::new(TypeId::TimestampNanoseconds);
    let duration = DataType::new(TypeId::DurationNanoseconds);

    assert_eq!(infer_dtype(timestamp, timestamp, BinaryOperator::Sub), duration);
}

fn test_infer_duration_integer_arithmetic_preserves_duration_dtype() {
    let duration = DataType::new(TypeId::DurationMilliseconds);
    let int = DataType::new(TypeId::Int32);

    assert_eq!(infer_dtype(duration, int, BinaryOperator::Mul), duration);
    assert_eq!(infer_dtype(int, duration, BinaryOperator::Mul), duration);
    assert_eq!(infer_dtype(duration, int, BinaryOperator::Div), duration);
}

fn test_infer_duration_modulo_rules_dtype() {
    let duration = DataType::new(TypeId::DurationMilliseconds);
    let int = DataType::new(TypeId::Int32);

    assert_eq!(infer_dtype(duration, int, BinaryOperator::Mod), duration);
    assert_eq!(infer_dtype(duration, duration, BinaryOperator::Mod), duration);
    assert_eq!(infer_dtype(duration, duration, BinaryOperator::Pymod), duration);

    let err = binary::infer_binary_output_dtype(duration, int, BinaryOperator::Pmod).unwrap_err();
    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_dtype",
        TypeId::DurationMilliseconds,
        TypeId::Int32,
        BinaryOperator::Pmod,
    );

    let err = binary::infer_binary_output_dtype(duration, duration, BinaryOperator::Pmod).unwrap_err();
    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_dtype",
        TypeId::DurationMilliseconds,
        TypeId::DurationMilliseconds,
        BinaryOperator::Pmod,
    );

    let err = binary::infer_binary_output_dtype(duration, int, BinaryOperator::Pymod).unwrap_err();
    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_dtype",
        TypeId::DurationMilliseconds,
        TypeId::Int32,
        BinaryOperator::Pymod,
    );
}

fn test_infer_duration_float_arithmetic_rejected_dtype() {
    let err = binary::infer_binary_output_dtype(
        DataType::new(TypeId::DurationMilliseconds),
        DataType::new(TypeId::Float64),
        BinaryOperator::Mul,
    )
    .unwrap_err();

    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_dtype",
        TypeId::DurationMilliseconds,
        TypeId::Float64,
        BinaryOperator::Mul,
    );
}

fn test_infer_mixed_temporal_resolution_rejected_dtype() {
    let err = binary::infer_binary_output_dtype(
        DataType::new(TypeId::TimestampMicroseconds),
        DataType::new(TypeId::DurationMilliseconds),
        BinaryOperator::Add,
    )
    .unwrap_err();

    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_dtype",
        TypeId::TimestampMicroseconds,
        TypeId::DurationMilliseconds,
        BinaryOperator::Add,
    );
}

fn test_infer_uint8_uint16_promotion() {
    // UInt8 + UInt16 → UInt16 (wider unsigned)
    let dt = binary::infer_binary_output_type(TypeId::UInt8, TypeId::UInt16, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::UInt16);
}

fn test_infer_int8_uint8_promotion() {
    // Int8 + UInt8 → Int16 (signed + unsigned same width → next wider signed)
    let dt = binary::infer_binary_output_type(TypeId::Int8, TypeId::UInt8, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int16);
}

fn test_infer_int64_uint64_error() {
    // Int64 + UInt64 → error (no wider signed integer available)
    let result = binary::infer_binary_output_type(TypeId::Int64, TypeId::UInt64, BinaryOperator::Add);
    assert!(result.is_err());
}

fn test_infer_signed_wider_than_unsigned_keeps_signed() {
    // Int64 + UInt32 should stay Int64 (signed is strictly wider)
    let dt = binary::infer_binary_output_type(TypeId::Int64, TypeId::UInt32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int64);
}

fn test_infer_float32_int64_promotion() {
    // Float32 + Int64 → Float32 (int promotes to float, then wider float wins)
    let dt = binary::infer_binary_output_type(TypeId::Float32, TypeId::Int64, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Float32);
}

fn test_infer_non_numeric_comparison_is_bool() {
    // Any comparison op should infer Bool8, even for non-numeric pairs.
    let dt = binary::infer_binary_output_type(TypeId::List, TypeId::List, BinaryOperator::Equal).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

fn test_infer_mixed_string_numeric_comparison_rejected() {
    let err = binary::infer_binary_output_type(TypeId::String, TypeId::Int32, BinaryOperator::Equal).unwrap_err();
    assert_type_mismatch_facts(
        &err,
        "binary::infer_binary_output_type",
        TypeId::String,
        TypeId::Int32,
        BinaryOperator::Equal,
    );
}

// ---------------------------------------------------------------------------
// Integer division and remainder: arrow-arith row semantics
// ---------------------------------------------------------------------------

fn assert_division_by_zero(result: cudf::Result<cudf::Series>, operator: &str) {
    let error = result.expect_err("a zero divisor in a valid row must fail");
    assert_eq!(error.code(), ErrorCode::DivisionByZero);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.status(), ErrorStatus::Permanent);
    assert_eq!(error.detail_value("operator"), Some(operator));
}

fn assert_division_overflow(result: cudf::Result<cudf::Series>) {
    let error = result.expect_err("the minimum value divided by -1 must fail");
    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.detail_value("reason"), Some("overflow"));
    assert_eq!(error.detail_value("operator"), Some("div"));
}

fn test_integer_division_values_and_failures_follow_arrow_row_semantics() {
    let lhs = series::<i32>("a", &[7, -7, 7, -7, 0]).unwrap();
    let rhs = series::<i32>("b", &[2, 2, -2, -2, 5]).unwrap();
    assert_eq!(lhs.on(Exec::RETAINED).div(&rhs).unwrap().to_vec_strict::<i32>().unwrap(), vec![3, -3, -3, 3, 0]);
    assert_eq!(lhs.on(Exec::RETAINED).rem(&rhs).unwrap().to_vec_strict::<i32>().unwrap(), vec![1, -1, 1, -1, 0]);
    assert_eq!(lhs.on(Exec::RETAINED).div_scalar(2i32).unwrap().to_vec_strict::<i32>().unwrap(), vec![3, -3, 3, -3, 0]);
    assert_eq!(lhs.on(Exec::RETAINED).rem_scalar(3i32).unwrap().to_vec_strict::<i32>().unwrap(), vec![1, -1, 1, -1, 0]);

    let df = DataFrame::from_series([("a", lhs), ("b", rhs)]).unwrap();
    let quotient = (col("a") / col("b")).eval(&df).unwrap();
    assert_eq!(quotient.to_vec_strict::<i32>().unwrap(), vec![3, -3, -3, 3, 0]);
    let scalar_dividend = (100i32 / col("b")).eval(&df).unwrap();
    assert_eq!(scalar_dividend.to_vec_strict::<i32>().unwrap(), vec![50, 50, -50, -50, 20]);

    let wide = series::<u64>("a", &[u64::MAX, 7]).unwrap();
    let divisor = series::<u64>("b", &[1, 2]).unwrap();
    assert_eq!(wide.on(Exec::RETAINED).div(&divisor).unwrap().to_vec_strict::<u64>().unwrap(), vec![u64::MAX, 3]);
    assert_eq!(wide.on(Exec::RETAINED).rem(&divisor).unwrap().to_vec_strict::<u64>().unwrap(), vec![0, 1]);

    let lhs = series::<i32>("a", &[1, 2]).unwrap();
    let rhs = series::<i32>("b", &[1, 0]).unwrap();
    assert_division_by_zero(lhs.on(Exec::RETAINED).div(&rhs), "div");
    assert_division_by_zero(lhs.on(Exec::RETAINED).rem(&rhs), "mod");
    assert_division_by_zero(lhs.on(Exec::RETAINED).div_scalar(0i32), "div");
    assert_division_by_zero(lhs.on(Exec::RETAINED).rem_scalar(0i32), "mod");

    let unsigned = series::<u32>("a", &[9, 9]).unwrap();
    let zero = series::<u32>("b", &[3, 0]).unwrap();
    assert_division_by_zero(unsigned.on(Exec::RETAINED).div(&zero), "div");
    assert_division_by_zero(unsigned.on(Exec::RETAINED).rem_scalar(0u32), "mod");

    let df = DataFrame::from_series([("x", series::<i64>("x", &[1, 0, 3]).unwrap())]).unwrap();
    let scalar_dividend = (5i64 / col("x")).eval(&df).unwrap_err();
    assert_eq!(scalar_dividend.code(), ErrorCode::DivisionByZero);
    let scalar_remainder = (5i64 % col("x")).eval(&df).unwrap_err();
    assert_eq!(scalar_remainder.code(), ErrorCode::DivisionByZero);
    let literal_divisor = (col("x") / 0i64).eval(&df).unwrap_err();
    assert_eq!(literal_divisor.code(), ErrorCode::DivisionByZero);
}

fn test_integer_division_null_rows_and_minimum_value_follow_arrow_row_semantics() {
    let lhs = nullable_series::<i32>("a", &[1, 2, 3], &[true, false, true]).unwrap();
    let rhs = series::<i32>("b", &[1, 0, 1]).unwrap();
    assert_eq!(
        lhs.on(Exec::RETAINED).div(&rhs).unwrap().to_optional_vec::<i32>().unwrap(),
        vec![Some(1), None, Some(3)]
    );
    assert_eq!(
        lhs.on(Exec::RETAINED).rem(&rhs).unwrap().to_optional_vec::<i32>().unwrap(),
        vec![Some(0), None, Some(0)]
    );

    let dividend = series::<i32>("a", &[10, 20, 30]).unwrap();
    let nullable_divisor = nullable_series::<i32>("b", &[2, 0, 2], &[true, false, true]).unwrap();
    assert_eq!(
        dividend.on(Exec::RETAINED).div(&nullable_divisor).unwrap().to_optional_vec::<i32>().unwrap(),
        vec![Some(5), None, Some(15)]
    );

    let lhs = nullable_series::<i32>("a", &[1, 10, 3], &[false, true, true]).unwrap();
    let rhs = nullable_series::<i32>("b", &[0, 5, 0], &[true, true, false]).unwrap();
    assert_eq!(lhs.on(Exec::RETAINED).div(&rhs).unwrap().to_optional_vec::<i32>().unwrap(), vec![None, Some(2), None]);
    assert_eq!(lhs.on(Exec::RETAINED).rem(&rhs).unwrap().to_optional_vec::<i32>().unwrap(), vec![None, Some(0), None]);

    let minimum_under_null = nullable_series::<i32>("a", &[i32::MIN, 6], &[false, true]).unwrap();
    let minus_one = series::<i32>("b", &[-1, -1]).unwrap();
    assert_eq!(
        minimum_under_null.on(Exec::RETAINED).div(&minus_one).unwrap().to_optional_vec::<i32>().unwrap(),
        vec![None, Some(-6)]
    );

    let all_null = nullable_series::<i32>("a", &[4, 5], &[false, false]).unwrap();
    assert_eq!(all_null.on(Exec::RETAINED).div_scalar(0i32).unwrap().null_count(), 2);
    assert_eq!(all_null.on(Exec::RETAINED).rem_scalar(0i32).unwrap().null_count(), 2);
    let empty = series::<i32>("a", &[]).unwrap();
    assert_eq!(empty.on(Exec::RETAINED).div_scalar(0i32).unwrap().len(), 0);
    let some_null = nullable_series::<i32>("a", &[4, 5, 6], &[true, false, true]).unwrap();
    assert_eq!(
        some_null.on(Exec::RETAINED).div_scalar(2i32).unwrap().to_optional_vec::<i32>().unwrap(),
        vec![Some(2), None, Some(3)]
    );

    let lhs = series::<i32>("a", &[i32::MIN, 5, -5]).unwrap();
    let minus_one = series::<i32>("b", &[-1, -1, -1]).unwrap();
    assert_division_overflow(lhs.on(Exec::RETAINED).div(&minus_one));
    assert_eq!(lhs.on(Exec::RETAINED).rem(&minus_one).unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 0, 0]);
    assert_division_overflow(lhs.on(Exec::RETAINED).div_scalar(-1i32));
    assert_eq!(lhs.on(Exec::RETAINED).rem_scalar(-1i32).unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 0, 0]);

    let safe = series::<i32>("a", &[5, -5]).unwrap();
    assert_eq!(safe.on(Exec::RETAINED).div_scalar(-1i32).unwrap().to_vec_strict::<i32>().unwrap(), vec![-5, 5]);
    let two_minus_ones = series::<i32>("b", &[-1, -1]).unwrap();
    assert_eq!(safe.on(Exec::RETAINED).div(&two_minus_ones).unwrap().to_vec_strict::<i32>().unwrap(), vec![-5, 5]);

    let wide = series::<i64>("a", &[i64::MIN]).unwrap();
    let wide_minus_one = series::<i64>("b", &[-1]).unwrap();
    assert_division_overflow(wide.on(Exec::RETAINED).div(&wide_minus_one));
    assert_eq!(wide.on(Exec::RETAINED).rem(&wide_minus_one).unwrap().to_vec_strict::<i64>().unwrap(), vec![0]);

    let narrow_dividend = series::<i32>("a", &[i32::MIN]).unwrap();
    let wide_divisor = series::<i64>("b", &[-1]).unwrap();
    assert_eq!(
        narrow_dividend.on(Exec::RETAINED).div(&wide_divisor).unwrap().to_vec_strict::<i64>().unwrap(),
        vec![2_147_483_648]
    );
    let wide_dividend = series::<i64>("a", &[i64::MIN]).unwrap();
    let narrow_divisor = series::<i32>("b", &[-1]).unwrap();
    assert_division_overflow(wide_dividend.on(Exec::RETAINED).div(&narrow_divisor));
    assert_eq!(wide_dividend.on(Exec::RETAINED).rem(&narrow_divisor).unwrap().to_vec_strict::<i64>().unwrap(), vec![0]);

    let df = DataFrame::from_series([("x", series::<i32>("x", &[i32::MIN, 3]).unwrap())]).unwrap();
    let minus_one = -1i32;
    assert_eq!((col("x") % minus_one).eval(&df).unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 0]);
    let divisors = DataFrame::from_series([("x", series::<i32>("x", &[3, -1]).unwrap())]).unwrap();
    assert_eq!((i32::MIN / col("x")).eval(&divisors).unwrap_err().code(), ErrorCode::ValueOutOfRange);

    let unsigned = series::<u32>("a", &[u32::MAX]).unwrap();
    let same = series::<u32>("b", &[u32::MAX]).unwrap();
    assert_eq!(unsigned.on(Exec::RETAINED).div(&same).unwrap().to_vec_strict::<u32>().unwrap(), vec![1]);
}

fn assert_same_float_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let same = (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits();
        assert!(same, "row {index}: libcudf {actual:?} differs from Rust {expected:?}");
    }
}

fn test_float_remainder_matches_the_rust_remainder_for_edge_values() {
    let lhs = [
        5.5,
        -5.5,
        5.5,
        -5.5,
        0.0,
        -0.0,
        1e300,
        5.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        5.0,
        7.25,
        1e-310,
        123_456_789.123,
        f64::MAX,
        -3.0,
    ];
    let rhs = [
        2.0,
        2.0,
        -2.0,
        -2.0,
        3.0,
        3.0,
        7.0,
        0.0,
        3.0,
        3.0,
        3.0,
        f64::INFINITY,
        0.5,
        3e-311,
        1e-5,
        3.0,
        f64::NEG_INFINITY,
    ];
    let expected: Vec<f64> = lhs.iter().zip(&rhs).map(|(lhs, rhs)| lhs % rhs).collect();
    let (a, b) = (series::<f64>("a", &lhs).unwrap(), series::<f64>("b", &rhs).unwrap());
    let actual = a.on(Exec::RETAINED).rem(&b).unwrap().to_vec_strict::<f64>().unwrap();
    assert_same_float_bits(&actual, &expected);

    let lhs32: Vec<f32> = lhs.iter().map(|value| *value as f32).collect();
    let rhs32: Vec<f32> = rhs.iter().map(|value| *value as f32).collect();
    let expected32: Vec<f64> = lhs32.iter().zip(&rhs32).map(|(lhs, rhs)| f64::from(lhs % rhs)).collect();
    let (a, b) = (series::<f32>("a", &lhs32).unwrap(), series::<f32>("b", &rhs32).unwrap());
    let actual32: Vec<f64> =
        a.on(Exec::RETAINED).rem(&b).unwrap().to_vec_strict::<f32>().unwrap().into_iter().map(f64::from).collect();
    assert_same_float_bits(&actual32, &expected32);
}

#[test]
fn binary_tests_cases() {
    run_named_cases(&[
        ("test_div_scalar_column", test_div_scalar_column),
        (
            "test_binary_operation_error_contains_operation_context",
            test_binary_operation_error_contains_operation_context,
        ),
        ("test_null_logical_or_uses_sql_three_valued_logic", test_null_logical_or_uses_sql_three_valued_logic),
        ("test_null_logical_and_uses_sql_three_valued_logic", test_null_logical_and_uses_sql_three_valued_logic),
        (
            "test_mixed_domain_binary_operation_vv_uses_lhs_provenance",
            test_mixed_domain_binary_operation_vv_uses_lhs_provenance,
        ),
        (
            "test_series_scalar_binary_operation_uses_series_provenance",
            test_series_scalar_binary_operation_uses_series_provenance,
        ),
        (
            "test_integer_division_values_and_failures_follow_arrow_row_semantics",
            test_integer_division_values_and_failures_follow_arrow_row_semantics,
        ),
        (
            "test_integer_division_null_rows_and_minimum_value_follow_arrow_row_semantics",
            test_integer_division_null_rows_and_minimum_value_follow_arrow_row_semantics,
        ),
        (
            "test_float_remainder_matches_the_rust_remainder_for_edge_values",
            test_float_remainder_matches_the_rust_remainder_for_edge_values,
        ),
        ("test_string_equal_vv", test_string_equal_vv),
        ("test_string_not_equal_vv", test_string_not_equal_vv),
        ("test_string_less_vv", test_string_less_vv),
        ("test_string_greater_vv", test_string_greater_vv),
        ("test_string_less_equal_vv", test_string_less_equal_vv),
        ("test_string_greater_equal_vv", test_string_greater_equal_vv),
        ("test_string_equal_empty", test_string_equal_empty),
        ("test_string_equal_case_sensitive", test_string_equal_case_sensitive),
        ("test_string_add_vv_error", test_string_add_vv_error),
        ("test_is_comparison_true_for_comparison_ops", test_is_comparison_true_for_comparison_ops),
        ("test_is_comparison_false_for_arithmetic_ops", test_is_comparison_false_for_arithmetic_ops),
        ("test_infer_same_type_arithmetic", test_infer_same_type_arithmetic),
        ("test_infer_int_float_promotion", test_infer_int_float_promotion),
        ("test_infer_float_float_promotion", test_infer_float_float_promotion),
        ("test_infer_int_int_promotion", test_infer_int_int_promotion),
        ("test_infer_signed_unsigned_promotion", test_infer_signed_unsigned_promotion),
        ("test_infer_comparison_always_bool", test_infer_comparison_always_bool),
        ("test_infer_string_comparison_bool", test_infer_string_comparison_bool),
        ("test_infer_string_arithmetic_error", test_infer_string_arithmetic_error),
        ("test_infer_string_int_arithmetic_error", test_infer_string_int_arithmetic_error),
        ("test_infer_non_numeric_error", test_infer_non_numeric_error),
        ("test_infer_bool_arithmetic", test_infer_bool_arithmetic),
        ("test_infer_error_context_keys", test_infer_error_context_keys),
        (
            "test_infer_error_operator_detail_uses_stable_snake_case",
            test_infer_error_operator_detail_uses_stable_snake_case,
        ),
        (
            "test_infer_decimal_same_and_mixed_scale_add_sub_dtype",
            test_infer_decimal_same_and_mixed_scale_add_sub_dtype,
        ),
        ("test_infer_decimal_mul_div_scales_dtype", test_infer_decimal_mul_div_scales_dtype),
        ("test_infer_decimal_comparison_bool", test_infer_decimal_comparison_bool),
        ("test_infer_mixed_resolution_temporal_comparison_bool", test_infer_mixed_resolution_temporal_comparison_bool),
        ("test_infer_temporal_typeid_rules", test_infer_temporal_typeid_rules),
        (
            "test_infer_decimal_float_arithmetic_rejected_with_type_facts_dtype",
            test_infer_decimal_float_arithmetic_rejected_with_type_facts_dtype,
        ),
        (
            "test_infer_same_resolution_timestamp_duration_arithmetic_dtype",
            test_infer_same_resolution_timestamp_duration_arithmetic_dtype,
        ),
        (
            "test_infer_timestamp_timestamp_subtracts_to_matching_duration_dtype",
            test_infer_timestamp_timestamp_subtracts_to_matching_duration_dtype,
        ),
        (
            "test_infer_duration_integer_arithmetic_preserves_duration_dtype",
            test_infer_duration_integer_arithmetic_preserves_duration_dtype,
        ),
        ("test_infer_duration_modulo_rules_dtype", test_infer_duration_modulo_rules_dtype),
        ("test_infer_duration_float_arithmetic_rejected_dtype", test_infer_duration_float_arithmetic_rejected_dtype),
        ("test_infer_mixed_temporal_resolution_rejected_dtype", test_infer_mixed_temporal_resolution_rejected_dtype),
        ("test_infer_uint8_uint16_promotion", test_infer_uint8_uint16_promotion),
        ("test_infer_int8_uint8_promotion", test_infer_int8_uint8_promotion),
        ("test_infer_int64_uint64_error", test_infer_int64_uint64_error),
        ("test_infer_signed_wider_than_unsigned_keeps_signed", test_infer_signed_wider_than_unsigned_keeps_signed),
        ("test_infer_float32_int64_promotion", test_infer_float32_int64_promotion),
        ("test_infer_non_numeric_comparison_is_bool", test_infer_non_numeric_comparison_is_bool),
        ("test_infer_mixed_string_numeric_comparison_rejected", test_infer_mixed_string_numeric_comparison_rejected),
    ]);
}
