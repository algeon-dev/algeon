//! Tests for binary operations (numeric and string).

use cudf::binary::{self, BinaryOperator};
use cudf::column::Column;
use cudf::scalar::Scalar;
use cudf::types::{DataType, TypeId};

#[test]
fn test_add_column_column() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32))
            .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![11, 22, 33]);
}

#[test]
fn test_mul_column_scalar() {
    let col = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let scalar = Scalar::from_value(10i32).unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::Mul,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![20, 30, 40]);
}

#[test]
fn test_div_scalar_column() {
    let scalar = Scalar::from_value(100.0f64).unwrap();
    let col = Column::from_slice(&[2.0f64, 4.0, 5.0]).unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_sv(
        &scalar,
        &cv,
        BinaryOperator::Div,
        &DataType::new(TypeId::Float64),
    )
    .unwrap();
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

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::Less,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
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

    let err =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Add, &DataType::new(TypeId::Int32))
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

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::LogicalAnd,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<bool>().unwrap();
    assert_eq!(values, vec![true, false, false, false]);
}

// ---------------------------------------------------------------------------
// String comparison: column vs column
// ---------------------------------------------------------------------------

#[test]
fn test_string_equal_vv() {
    let a = Column::from_strings(&["foo", "bar", "baz"]).unwrap();
    let b = Column::from_strings(&["foo", "bar", "qux"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8))
            .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_string_not_equal_vv() {
    let a = Column::from_strings(&["foo", "bar", "baz"]).unwrap();
    let b = Column::from_strings(&["foo", "bar", "qux"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::NotEqual,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

#[test]
fn test_string_less_vv() {
    let a = Column::from_strings(&["apple", "cherry", "banana"]).unwrap();
    let b = Column::from_strings(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Less, &DataType::new(TypeId::Bool8))
            .unwrap();
    let rv = result.view().unwrap();
    // "apple" < "banana" → true, "cherry" < "banana" → false, "banana" < "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_string_greater_vv() {
    let a = Column::from_strings(&["apple", "cherry", "banana"]).unwrap();
    let b = Column::from_strings(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::Greater,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    // "apple" > "banana" → false, "cherry" > "banana" → true, "banana" > "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_string_less_equal_vv() {
    let a = Column::from_strings(&["apple", "banana", "cherry"]).unwrap();
    let b = Column::from_strings(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::LessEqual,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    // "apple" <= "banana" → true, "banana" <= "banana" → true, "cherry" <= "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_string_greater_equal_vv() {
    let a = Column::from_strings(&["apple", "banana", "cherry"]).unwrap();
    let b = Column::from_strings(&["banana", "banana", "banana"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::GreaterEqual,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    // "apple" >= "banana" → false, "banana" >= "banana" → true, "cherry" >= "banana" → true
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// String comparison: column vs scalar
// ---------------------------------------------------------------------------

#[test]
fn test_string_equal_vs() {
    let col = Column::from_strings(&["active", "inactive", "active"]).unwrap();
    let scalar = Scalar::from_string("active").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::Equal,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_string_not_equal_vs() {
    let col = Column::from_strings(&["active", "inactive", "active"]).unwrap();
    let scalar = Scalar::from_string("active").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::NotEqual,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_string_less_vs() {
    let col = Column::from_strings(&["apple", "cherry", "banana"]).unwrap();
    let scalar = Scalar::from_string("banana").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::Less,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    // "apple" < "banana" → true, "cherry" < "banana" → false, "banana" < "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_string_greater_vs() {
    let col = Column::from_strings(&["apple", "cherry", "banana"]).unwrap();
    let scalar = Scalar::from_string("banana").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::Greater,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    // "apple" > "banana" → false, "cherry" > "banana" → true, "banana" > "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_string_less_equal_vs() {
    let col = Column::from_strings(&["apple", "banana", "cherry"]).unwrap();
    let scalar = Scalar::from_string("banana").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::LessEqual,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    // "apple" <= "banana" → true, "banana" <= "banana" → true, "cherry" <= "banana" → false
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_string_greater_equal_vs() {
    let col = Column::from_strings(&["apple", "banana", "cherry"]).unwrap();
    let scalar = Scalar::from_string("banana").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::GreaterEqual,
        &DataType::new(TypeId::Bool8),
    )
    .unwrap();
    let rv = result.view().unwrap();
    // "apple" >= "banana" → false, "banana" >= "banana" → true, "cherry" >= "banana" → true
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// String edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_string_equal_empty() {
    let a = Column::from_strings(&["", "hello", ""]).unwrap();
    let b = Column::from_strings(&["", "", "hello"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8))
            .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_string_equal_case_sensitive() {
    let a = Column::from_strings(&["Foo", "BAR", "baz"]).unwrap();
    let b = Column::from_strings(&["foo", "bar", "baz"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result =
        binary::binary_operation_vv(&av, &bv, BinaryOperator::Equal, &DataType::new(TypeId::Bool8))
            .unwrap();
    let rv = result.view().unwrap();
    // Case-sensitive: "Foo" != "foo", "BAR" != "bar", "baz" == "baz"
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

// ---------------------------------------------------------------------------
// String arithmetic errors
// ---------------------------------------------------------------------------

#[test]
fn test_string_add_vv_error() {
    let a = Column::from_strings(&["foo", "bar"]).unwrap();
    let b = Column::from_strings(&["baz", "qux"]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = binary::binary_operation_vv(
        &av,
        &bv,
        BinaryOperator::Add,
        &DataType::new(TypeId::String),
    );
    assert!(result.is_err());
}

#[test]
fn test_string_sub_vs_error() {
    let col = Column::from_strings(&["foo", "bar"]).unwrap();
    let scalar = Scalar::from_string("baz").unwrap();
    let cv = col.view().unwrap();

    let result = binary::binary_operation_vs(
        &cv,
        &scalar,
        BinaryOperator::Sub,
        &DataType::new(TypeId::String),
    );
    assert!(result.is_err());
}
