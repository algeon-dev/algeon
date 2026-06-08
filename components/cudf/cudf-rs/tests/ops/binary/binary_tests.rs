//! Tests for binary operations (numeric and string).

use cudf::{
    DataType, TypeId,
    core::{Column, Scalar},
    ops::binary::{self, BinaryOperator},
};

use crate::common::{
    explicit_i32_column, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation,
    reset_sync_observation,
};

#[test]
fn test_add_column_column() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32)).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![11, 22, 33]);
}

#[test]
fn test_mul_column_scalar() {
    let col = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let scalar = Scalar::from_value(10i32).unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Mul, &DataType::new(TypeId::Int32)).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![20, 30, 40]);
}

#[test]
fn test_div_scalar_column() {
    let scalar = Scalar::from_value(100.0f64).unwrap();
    let col = Column::from_slice(&[2.0f64, 4.0, 5.0]).unwrap();
    let cv = col.view().unwrap();

    let result =
        binary::binary_operation_sv(&scalar, &cv, BinaryOperator::Div, &DataType::new(TypeId::Float64)).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![50.0, 25.0, 20.0]);
}

#[test]
fn test_comparison() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[2i32, 2, 1]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Less, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, false]);
}

#[test]
fn test_binary_operation_error_contains_operation_context() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20]).unwrap(); // mismatched length
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let err = binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32))
        .err()
        .expect("expected error");

    assert_eq!(err.operation(), "binary::binary_operation_vv");
    assert!(err.context().iter().any(|(k, _)| *k == "operator"));
}

#[test]
fn test_logical_and() {
    let a = Column::from_slice(&[true, true, false, false]).unwrap();
    let b = Column::from_slice(&[true, false, true, false]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::LogicalAnd, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, false, false]);
}

#[test]
fn test_mixed_domain_binary_operation_vv_uses_lhs_provenance() {
    let lhs = explicit_i32_column(&[1, 2, 3], 900);
    let rhs = explicit_i32_column(&[10, 20, 30], 901);
    let expected_stream = lhs.__debug_execution_domain_stream_value();
    let expected_mr = lhs.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = binary::binary_operation_vv(
        &lhs.view().unwrap(),
        &rhs.view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();

    assert_eq!(result.__debug_execution_domain_id(), lhs.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn test_mixed_domain_binary_operation_vs_uses_column_provenance() {
    let lhs = explicit_i32_column(&[2, 3, 4], 902);
    let rhs = Scalar::from_value(10i32).unwrap().__debug_rebind_execution_domain_explicit(903).unwrap();
    let expected_stream = lhs.__debug_execution_domain_stream_value();
    let expected_mr = lhs.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result =
        binary::binary_operation_vs(&lhs.view().unwrap(), &rhs, BinaryOperator::Mul, &DataType::new(TypeId::Int32))
            .unwrap();

    assert_eq!(result.__debug_execution_domain_id(), lhs.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn test_mixed_domain_binary_operation_sv_uses_column_provenance() {
    let lhs = Scalar::from_value(100i32).unwrap().__debug_rebind_execution_domain_explicit(904).unwrap();
    let rhs = explicit_i32_column(&[2, 4, 5], 905);
    // After the L11 fix, binary_operation_sv uses the column's (rhs) domain
    // as primary, consistent with binary_operation_vs.
    let expected_stream = rhs.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let result =
        binary::binary_operation_sv(&lhs, &rhs.view().unwrap(), BinaryOperator::Div, &DataType::new(TypeId::Int32))
            .unwrap();

    assert_eq!(result.__debug_execution_domain_id(), rhs.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), last_exec_mr_value());
    assert_eq!(last_exec_stream_value(), expected_stream);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![50, 25, 20]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

// ---------------------------------------------------------------------------
// String comparison: column vs column
// ---------------------------------------------------------------------------

#[test]
fn test_string_equal_vv() {
    let a = Column::from_strs(&["foo", "bar", "baz"]).unwrap();
    let b = Column::from_strs(&["foo", "bar", "qux"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_string_not_equal_vv() {
    let a = Column::from_strs(&["foo", "bar", "baz"]).unwrap();
    let b = Column::from_strs(&["foo", "bar", "qux"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::NotEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

#[test]
fn test_string_less_vv() {
    let a = Column::from_strs(&["apple", "cherry", "banana"]).unwrap();
    let b = Column::from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Less, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" < "banana" → true, "cherry" < "banana" → false, "banana" < "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_string_greater_vv() {
    let a = Column::from_strs(&["apple", "cherry", "banana"]).unwrap();
    let b = Column::from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Greater, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" > "banana" → false, "cherry" > "banana" → true, "banana" > "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_string_less_equal_vv() {
    let a = Column::from_strs(&["apple", "banana", "cherry"]).unwrap();
    let b = Column::from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::LessEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" <= "banana" → true, "banana" <= "banana" → true, "cherry" <= "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_string_greater_equal_vv() {
    let a = Column::from_strs(&["apple", "banana", "cherry"]).unwrap();
    let b = Column::from_strs(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::GreaterEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" >= "banana" → false, "banana" >= "banana" → true, "cherry" >= "banana" → true
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// String comparison: column vs scalar
// ---------------------------------------------------------------------------

#[test]
fn test_string_equal_vs() {
    let col = Column::from_strs(&["active", "inactive", "active"]).unwrap();
    let scalar = Scalar::from_str("active").unwrap();
    let cv = col.view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_string_not_equal_vs() {
    let col = Column::from_strs(&["active", "inactive", "active"]).unwrap();
    let scalar = Scalar::from_str("active").unwrap();
    let cv = col.view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::NotEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_string_less_vs() {
    let col = Column::from_strs(&["apple", "cherry", "banana"]).unwrap();
    let scalar = Scalar::from_str("banana").unwrap();
    let cv = col.view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Less, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" < "banana" → true, "cherry" < "banana" → false, "banana" < "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_string_greater_vs() {
    let col = Column::from_strs(&["apple", "cherry", "banana"]).unwrap();
    let scalar = Scalar::from_str("banana").unwrap();
    let cv = col.view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Greater, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" > "banana" → false, "cherry" > "banana" → true, "banana" > "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_string_less_equal_vs() {
    let col = Column::from_strs(&["apple", "banana", "cherry"]).unwrap();
    let scalar = Scalar::from_str("banana").unwrap();
    let cv = col.view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::LessEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" <= "banana" → true, "banana" <= "banana" → true, "cherry" <= "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_string_greater_equal_vs() {
    let col = Column::from_strs(&["apple", "banana", "cherry"]).unwrap();
    let scalar = Scalar::from_str("banana").unwrap();
    let cv = col.view().unwrap();

    let result =
        binary::binary_operation_vs(&cv, &scalar, BinaryOperator::GreaterEqual, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // "apple" >= "banana" → false, "banana" >= "banana" → true, "cherry" >= "banana" → true
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// String edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_string_equal_empty() {
    let a = Column::from_strs(&["", "hello", ""]).unwrap();
    let b = Column::from_strs(&["", "", "hello"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_string_equal_case_sensitive() {
    let a = Column::from_strs(&["Foo", "BAR", "baz"]).unwrap();
    let b = Column::from_strs(&["foo", "bar", "baz"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8)).unwrap();
    let rv = result.view().unwrap();
    // Case-sensitive: "Foo" != "foo", "BAR" != "bar", "baz" == "baz"
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

// ---------------------------------------------------------------------------
// String arithmetic errors
// ---------------------------------------------------------------------------

#[test]
fn test_string_add_vv_error() {
    let a = Column::from_strs(&["foo", "bar"]).unwrap();
    let b = Column::from_strs(&["baz", "qux"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::String));
    assert!(result.is_err());
}

#[test]
fn test_string_sub_vs_error() {
    let col = Column::from_strs(&["foo", "bar"]).unwrap();
    let scalar = Scalar::from_str("baz").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(&cv, &scalar, BinaryOperator::Sub, &DataType::new(TypeId::String));
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// BinaryOperator::is_comparison
// ---------------------------------------------------------------------------

#[test]
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

#[test]
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

#[test]
fn test_infer_same_type_arithmetic() {
    // Int32 + Int32 → Int32
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::Int32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int32);
}

#[test]
fn test_infer_int_float_promotion() {
    // Int32 + Float64 → Float64
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::Float64, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Float64);
}

#[test]
fn test_infer_float_float_promotion() {
    // Float32 + Float64 → Float64
    let dt = binary::infer_binary_output_type(TypeId::Float32, TypeId::Float64, BinaryOperator::Mul).unwrap();
    assert_eq!(dt.id(), TypeId::Float64);
}

#[test]
fn test_infer_int_int_promotion() {
    // Int16 + Int32 → Int32
    let dt = binary::infer_binary_output_type(TypeId::Int16, TypeId::Int32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int32);
}

#[test]
fn test_infer_signed_unsigned_promotion() {
    // Int32 + UInt32 → Int64
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::UInt32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int64);
}

#[test]
fn test_infer_comparison_always_bool() {
    // Any comparison op → Bool8 regardless of input types
    let dt = binary::infer_binary_output_type(TypeId::Int32, TypeId::Float64, BinaryOperator::Less).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);

    let dt = binary::infer_binary_output_type(TypeId::Int64, TypeId::Int32, BinaryOperator::Equal).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

#[test]
fn test_infer_string_comparison_bool() {
    // String + String comparison → Bool8
    let dt = binary::infer_binary_output_type(TypeId::String, TypeId::String, BinaryOperator::Equal).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);

    let dt = binary::infer_binary_output_type(TypeId::String, TypeId::String, BinaryOperator::Less).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

#[test]
fn test_infer_string_arithmetic_error() {
    // String + String arithmetic → error
    let result = binary::infer_binary_output_type(TypeId::String, TypeId::String, BinaryOperator::Add);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::NotSupported);
    assert_eq!(err.operation(), "binary::infer_binary_output_type");
}

#[test]
fn test_infer_string_int_arithmetic_error() {
    // String + Int arithmetic → error
    let result = binary::infer_binary_output_type(TypeId::String, TypeId::Int32, BinaryOperator::Add);
    assert!(result.is_err());
}

#[test]
fn test_infer_non_numeric_error() {
    // Non-numeric types (List, Struct, etc.) → error
    let result = binary::infer_binary_output_type(TypeId::List, TypeId::Int32, BinaryOperator::Add);
    assert!(result.is_err());

    let result = binary::infer_binary_output_type(TypeId::Struct, TypeId::Struct, BinaryOperator::Add);
    assert!(result.is_err());
}

#[test]
fn test_infer_bool_arithmetic() {
    // Bool8 + Bool8 → Bool8
    let dt = binary::infer_binary_output_type(TypeId::Bool8, TypeId::Bool8, BinaryOperator::LogicalAnd).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

#[test]
fn test_infer_error_context_keys() {
    // Verify error context contains lhs_type_id, rhs_type_id, operator
    let err = binary::infer_binary_output_type(TypeId::String, TypeId::Int32, BinaryOperator::Add).unwrap_err();
    let context_keys: Vec<&str> = err.context().iter().map(|(k, _)| *k).collect();
    assert!(context_keys.contains(&"lhs_type_id"));
    assert!(context_keys.contains(&"rhs_type_id"));
    assert!(context_keys.contains(&"operator"));
}

#[test]
fn test_infer_uint8_uint16_promotion() {
    // UInt8 + UInt16 → UInt16 (wider unsigned)
    let dt = binary::infer_binary_output_type(TypeId::UInt8, TypeId::UInt16, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::UInt16);
}

#[test]
fn test_infer_int8_uint8_promotion() {
    // Int8 + UInt8 → Int16 (signed + unsigned same width → next wider signed)
    let dt = binary::infer_binary_output_type(TypeId::Int8, TypeId::UInt8, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int16);
}

#[test]
fn test_infer_int64_uint64_error() {
    // Int64 + UInt64 → error (no wider signed integer available)
    let result = binary::infer_binary_output_type(TypeId::Int64, TypeId::UInt64, BinaryOperator::Add);
    assert!(result.is_err());
}

#[test]
fn test_infer_signed_wider_than_unsigned_keeps_signed() {
    // Int64 + UInt32 should stay Int64 (signed is strictly wider)
    let dt = binary::infer_binary_output_type(TypeId::Int64, TypeId::UInt32, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Int64);
}

#[test]
fn test_infer_float32_int64_promotion() {
    // Float32 + Int64 → Float32 (int promotes to float, then wider float wins)
    let dt = binary::infer_binary_output_type(TypeId::Float32, TypeId::Int64, BinaryOperator::Add).unwrap();
    assert_eq!(dt.id(), TypeId::Float32);
}

#[test]
fn test_infer_non_numeric_comparison_is_bool() {
    // Any comparison op should infer Bool8, even for non-numeric pairs.
    let dt = binary::infer_binary_output_type(TypeId::List, TypeId::List, BinaryOperator::Equal).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}

#[test]
fn test_infer_mixed_string_numeric_comparison_is_bool() {
    // Any comparison op should infer Bool8 for mixed type pairs too.
    let dt = binary::infer_binary_output_type(TypeId::String, TypeId::Int32, BinaryOperator::Equal).unwrap();
    assert_eq!(dt.id(), TypeId::Bool8);
}
