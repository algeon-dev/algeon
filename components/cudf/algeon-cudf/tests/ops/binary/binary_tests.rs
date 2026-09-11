//! Tests for binary operations (numeric and string).
use cudf::{
    DataType, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::binary::{self, BinaryOperator},
    testing::prelude::*,
};

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, explicit_i32_column, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation, scalar,
};

fn test_add_column_column() {
    let a = column_from_slice(&[1i32, 2, 3]).unwrap();
    let b = column_from_slice(&[10i32, 20, 30]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32)).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![11, 22, 33]);
}

fn test_mul_column_scalar() {
    let col = column_from_slice(&[2i32, 3, 4]).unwrap();
    let scalar = scalar(10i32).unwrap();
    let cv = col.testing_view().unwrap();

    let result = binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Mul, &DataType::new(TypeId::Int32)).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![20, 30, 40]);
}

fn test_div_scalar_column() {
    let scalar = scalar(100.0f64).unwrap();
    let col = column_from_slice(&[2.0f64, 4.0, 5.0]).unwrap();
    let cv = col.testing_view().unwrap();

    let result =
        binary::binary_operation_sv(&scalar, &cv, BinaryOperator::Div, &DataType::new(TypeId::Float64)).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![50.0, 25.0, 20.0]);
}

fn test_comparison() {
    let a = column_from_slice(&[1i32, 2, 3]).unwrap();
    let b = column_from_slice(&[2i32, 2, 1]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Less, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, false]);
}

fn test_binary_operation_error_contains_operation_context() {
    let a = column_from_slice(&[1i32, 2, 3]).unwrap();
    let b = column_from_slice(&[10i32, 20]).unwrap(); // mismatched length
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let err = binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32))
        .expect_err("expected error");

    assert_eq!(err.operation(), "binary::binary_operation_vv");
    assert!(err.details().iter().any(|(k, _)| *k == "operator"));
}

fn test_logical_and() {
    let a = column_from_slice(&[true, true, false, false]).unwrap();
    let b = column_from_slice(&[true, false, true, false]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::LogicalAnd, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, false, false]);
}

fn test_null_logical_or_uses_sql_three_valued_logic() {
    // Rows pair LHS x RHS over {true, false, null}:
    //   (T,T) (T,F) (T,N) (F,T) (F,F) (F,N) (N,T) (N,F) (N,N)
    let lhs_values = [true, true, true, false, false, false, false, false, false];
    let lhs_validity = [true, true, true, true, true, true, false, false, false];
    let a = column_from_slice_nullable(&lhs_values, &lhs_validity).unwrap();
    let rhs_values = [true, false, false, true, false, false, true, false, false];
    let rhs_validity = [true, true, false, true, true, false, true, true, false];
    let b = column_from_slice_nullable(&rhs_values, &rhs_validity).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::NullLogicalOr, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // true||* = true; false||true = true; false||false = false; false||null = null;
    // null||true = true; null||false = null; null||null = null.
    assert_eq!(
        rv.to_optional_vec::<bool>().unwrap(),
        vec![Some(true), Some(true), Some(true), Some(true), Some(false), None, Some(true), None, None]
    );
}

fn test_null_logical_and_uses_sql_three_valued_logic() {
    // Rows pair LHS x RHS over {true, false, null}:
    //   (T,T) (T,F) (T,N) (F,T) (F,F) (F,N) (N,T) (N,F) (N,N)
    let lhs_values = [true, true, true, false, false, false, false, false, false];
    let lhs_validity = [true, true, true, true, true, true, false, false, false];
    let a = column_from_slice_nullable(&lhs_values, &lhs_validity).unwrap();
    let rhs_values = [true, false, false, true, false, false, true, false, false];
    let rhs_validity = [true, true, false, true, true, false, true, true, false];
    let b = column_from_slice_nullable(&rhs_values, &rhs_validity).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::NullLogicalAnd, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // false&&* = false; true&&true = true; true&&false = false; true&&null = null;
    // null&&true = null; null&&false = false; null&&null = null.
    assert_eq!(
        rv.to_optional_vec::<bool>().unwrap(),
        vec![Some(true), Some(false), None, Some(false), Some(false), Some(false), None, Some(false), None]
    );
}

fn test_mixed_domain_binary_operation_vv_uses_lhs_provenance() {
    let lhs = explicit_i32_column(&[1, 2, 3], 900);
    let rhs = explicit_i32_column(&[10, 20, 30], 901);
    let expected_stream = lhs.testing_execution_domain_stream_value();
    let expected_mr = lhs.testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = binary::binary_operation_vv(
        &lhs.testing_view().unwrap(),
        &rhs.testing_view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();

    assert_eq!(result.testing_execution_domain_id(), lhs.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

fn test_mixed_domain_binary_operation_vs_uses_column_provenance() {
    let lhs = explicit_i32_column(&[2, 3, 4], 902);
    let rhs = scalar(10i32).unwrap().testing_rebind_execution_domain_explicit(903).unwrap();
    let expected_stream = lhs.testing_execution_domain_stream_value();
    let expected_mr = lhs.testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = binary::binary_operation_vs(
        &lhs.testing_view().unwrap(),
        &rhs,
        BinaryOperator::Mul,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();

    assert_eq!(result.testing_execution_domain_id(), lhs.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

fn test_mixed_domain_binary_operation_sv_uses_column_provenance() {
    let lhs = scalar(100i32).unwrap().testing_rebind_execution_domain_explicit(904).unwrap();
    let rhs = explicit_i32_column(&[2, 4, 5], 905);
    // After the L11 fix, binary_operation_sv uses the column's (rhs) domain
    // as primary, consistent with binary_operation_vs.
    let expected_stream = rhs.testing_execution_domain_stream_value();

    reset_exec_observation();
    let result = binary::binary_operation_sv(
        &lhs,
        &rhs.testing_view().unwrap(),
        BinaryOperator::Div,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();

    assert_eq!(result.testing_execution_domain_id(), rhs.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), last_exec_mr_value());
    assert_eq!(last_exec_stream_value(), expected_stream);

    reset_sync_observation();
    assert_eq!(result.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![50, 25, 20]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

// ---------------------------------------------------------------------------
// String comparison: column vs column
// ---------------------------------------------------------------------------

fn test_string_equal_vv() {
    let a = column_from_strs(&["foo", "bar", "baz"]).unwrap();
    let b = column_from_strs(&["foo", "bar", "qux"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

fn test_string_not_equal_vv() {
    let a = column_from_strs(&["foo", "bar", "baz"]).unwrap();
    let b = column_from_strs(&["foo", "bar", "qux"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::NotEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

fn test_string_less_vv() {
    let a = column_from_strs(&["apple", "cherry", "banana"]).unwrap();
    let b = column_from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Less, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" < "banana" → true, "cherry" < "banana" → false, "banana" < "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

fn test_string_greater_vv() {
    let a = column_from_strs(&["apple", "cherry", "banana"]).unwrap();
    let b = column_from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Greater, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" > "banana" → false, "cherry" > "banana" → true, "banana" > "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

fn test_string_less_equal_vv() {
    let a = column_from_strs(&["apple", "banana", "cherry"]).unwrap();
    let b = column_from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::LessEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" <= "banana" → true, "banana" <= "banana" → true, "cherry" <= "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

fn test_string_greater_equal_vv() {
    let a = column_from_strs(&["apple", "banana", "cherry"]).unwrap();
    let b = column_from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::GreaterEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" >= "banana" → false, "banana" >= "banana" → true, "cherry" >= "banana" → true
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// String comparison: column vs scalar
// ---------------------------------------------------------------------------

fn test_string_equal_vs() {
    let col = column_from_strs(&["active", "inactive", "active"]).unwrap();
    let scalar = crate::common::string_scalar("active").unwrap();
    let cv = col.testing_view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

fn test_string_not_equal_vs() {
    let col = column_from_strs(&["active", "inactive", "active"]).unwrap();
    let scalar = crate::common::string_scalar("active").unwrap();
    let cv = col.testing_view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::NotEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

fn test_string_less_vs() {
    let col = column_from_strs(&["apple", "cherry", "banana"]).unwrap();
    let scalar = crate::common::string_scalar("banana").unwrap();
    let cv = col.testing_view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Less, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" < "banana" → true, "cherry" < "banana" → false, "banana" < "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

fn test_string_greater_vs() {
    let col = column_from_strs(&["apple", "cherry", "banana"]).unwrap();
    let scalar = crate::common::string_scalar("banana").unwrap();
    let cv = col.testing_view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Greater, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" > "banana" → false, "cherry" > "banana" → true, "banana" > "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

fn test_string_less_equal_vs() {
    let col = column_from_strs(&["apple", "banana", "cherry"]).unwrap();
    let scalar = crate::common::string_scalar("banana").unwrap();
    let cv = col.testing_view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::LessEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" <= "banana" → true, "banana" <= "banana" → true, "cherry" <= "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

fn test_string_greater_equal_vs() {
    let col = column_from_strs(&["apple", "banana", "cherry"]).unwrap();
    let scalar = crate::common::string_scalar("banana").unwrap();
    let cv = col.testing_view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::GreaterEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // "apple" >= "banana" → false, "banana" >= "banana" → true, "cherry" >= "banana" → true
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// String edge cases
// ---------------------------------------------------------------------------

fn test_string_equal_empty() {
    let a = column_from_strs(&["", "hello", ""]).unwrap();
    let b = column_from_strs(&["", "", "hello"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

fn test_string_equal_case_sensitive() {
    let a = column_from_strs(&["Foo", "BAR", "baz"]).unwrap();
    let b = column_from_strs(&["foo", "bar", "baz"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.testing_view().unwrap();
    // Case-sensitive: "Foo" != "foo", "BAR" != "bar", "baz" == "baz"
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

// ---------------------------------------------------------------------------
// String arithmetic errors
// ---------------------------------------------------------------------------

fn test_string_add_vv_error() {
    let a = column_from_strs(&["foo", "bar"]).unwrap();
    let b = column_from_strs(&["baz", "qux"]).unwrap();
    let av = a.testing_view().unwrap();
    let bv = b.testing_view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::String));
    assert!(result.is_err());
}

fn test_string_sub_vs_error() {
    let col = column_from_strs(&["foo", "bar"]).unwrap();
    let scalar = crate::common::string_scalar("baz").unwrap();
    let cv = col.testing_view().unwrap();

    let result = binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Sub, &DataType::new(TypeId::String));
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

#[test]
fn binary_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_add_column_column", test_add_column_column),
        ("test_mul_column_scalar", test_mul_column_scalar),
        ("test_div_scalar_column", test_div_scalar_column),
        ("test_comparison", test_comparison),
        (
            "test_binary_operation_error_contains_operation_context",
            test_binary_operation_error_contains_operation_context,
        ),
        ("test_logical_and", test_logical_and),
        ("test_null_logical_or_uses_sql_three_valued_logic", test_null_logical_or_uses_sql_three_valued_logic),
        ("test_null_logical_and_uses_sql_three_valued_logic", test_null_logical_and_uses_sql_three_valued_logic),
        (
            "test_mixed_domain_binary_operation_vv_uses_lhs_provenance",
            test_mixed_domain_binary_operation_vv_uses_lhs_provenance,
        ),
        (
            "test_mixed_domain_binary_operation_vs_uses_column_provenance",
            test_mixed_domain_binary_operation_vs_uses_column_provenance,
        ),
        (
            "test_mixed_domain_binary_operation_sv_uses_column_provenance",
            test_mixed_domain_binary_operation_sv_uses_column_provenance,
        ),
        ("test_string_equal_vv", test_string_equal_vv),
        ("test_string_not_equal_vv", test_string_not_equal_vv),
        ("test_string_less_vv", test_string_less_vv),
        ("test_string_greater_vv", test_string_greater_vv),
        ("test_string_less_equal_vv", test_string_less_equal_vv),
        ("test_string_greater_equal_vv", test_string_greater_equal_vv),
        ("test_string_equal_vs", test_string_equal_vs),
        ("test_string_not_equal_vs", test_string_not_equal_vs),
        ("test_string_less_vs", test_string_less_vs),
        ("test_string_greater_vs", test_string_greater_vs),
        ("test_string_less_equal_vs", test_string_less_equal_vs),
        ("test_string_greater_equal_vs", test_string_greater_equal_vs),
        ("test_string_equal_empty", test_string_equal_empty),
        ("test_string_equal_case_sensitive", test_string_equal_case_sensitive),
        ("test_string_add_vv_error", test_string_add_vv_error),
        ("test_string_sub_vs_error", test_string_sub_vs_error),
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
