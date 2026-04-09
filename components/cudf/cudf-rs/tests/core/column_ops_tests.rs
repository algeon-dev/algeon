//! Tests for Column-level arithmetic, comparison, and scalar operations.

use cudf::{
    DataType, TypeId,
    core::{Column, Scalar},
    ops::binary::BinaryOperator,
};

// ---------------------------------------------------------------------------
// Arithmetic methods on Column
// ---------------------------------------------------------------------------

#[test]
fn test_column_add_i32() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let result = a.add(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_column_sub_i32() {
    let a = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let b = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let result = a.sub(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![9, 18, 27]);
}

#[test]
fn test_column_mul_i32() {
    let a = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let b = Column::from_slice(&[5i32, 6, 7]).unwrap();
    let result = a.mul(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![10, 18, 28]);
}

#[test]
fn test_column_div_i32() {
    let a = Column::from_slice(&[10i32, 21, 35]).unwrap();
    let b = Column::from_slice(&[2i32, 7, 5]).unwrap();
    let result = a.div(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![5, 3, 7]);
}

#[test]
fn test_column_true_div_i32() {
    let a = Column::from_slice(&[1i32, 3, 7]).unwrap();
    let b = Column::from_slice(&[2i32, 4, 2]).unwrap();
    let result = a.true_div(&b).unwrap();
    let rv = result.view().unwrap();
    // true_div should produce float output
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Float64);
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![0.5, 0.75, 3.5]);
}

#[test]
fn test_column_modulo_i32() {
    let a = Column::from_slice(&[10i32, 21, 37]).unwrap();
    let b = Column::from_slice(&[3i32, 7, 5]).unwrap();
    let result = a.modulo(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![1, 0, 2]);
}

#[test]
fn test_column_add_mixed_types() {
    // Int32 + Float64 → Float64
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[0.5f64, 1.5, 2.5]).unwrap();
    let result = a.add(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Float64);
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![1.5, 3.5, 5.5]);
}

#[test]
fn test_column_binary_op_with_type() {
    // Explicit output type override
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let result = a.binary_op_with_type(&b, BinaryOperator::Add, &DataType::new(TypeId::Int64)).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(rv.to_vec::<i64>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_column_binary_op_error_operation_context() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20]).unwrap(); // length mismatch
    let err = a.binary_op(&b, BinaryOperator::Add).err().expect("expected error");
    assert_eq!(err.operation(), "column::binary_op");
}

#[test]
fn test_column_binary_op_scalar_error_operation_context() {
    let col = Column::from_strs(&["foo", "bar"]).unwrap();
    let scalar = Scalar::from_str("baz").unwrap();
    let err = col.binary_op_scalar(&scalar, BinaryOperator::Add).err().expect("expected error");
    assert_eq!(err.operation(), "column::binary_op_scalar");
}

#[test]
fn test_column_add_string_error() {
    let a = Column::from_strs(&["foo", "bar"]).unwrap();
    let b = Column::from_strs(&["baz", "qux"]).unwrap();
    let err = a.add(&b).err().expect("expected error");
    assert_eq!(err.operation(), "column::add");
}

#[test]
fn test_column_add_nullable() {
    let a = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let result = a.add(&b).unwrap();
    let rv = result.view().unwrap();
    let values = rv.to_vec::<i32>().unwrap();
    // Element at index 1 should be null; value may be arbitrary but null mask matters
    let validity = rv.validity().unwrap().unwrap();
    assert!(validity[0]);
    assert!(!validity[1]); // null
    assert!(validity[2]);
    assert_eq!(values[0], 11);
    assert_eq!(values[2], 33);
}

#[test]
fn test_column_floor_div_f64() {
    let a = Column::from_slice(&[7.0f64, 10.0, 15.0]).unwrap();
    let b = Column::from_slice(&[2.0f64, 3.0, 4.0]).unwrap();
    let result = a.floor_div(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Float64);
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![3.0, 3.0, 3.0]);
}

// ---------------------------------------------------------------------------
// Comparison methods on Column
// ---------------------------------------------------------------------------

#[test]
fn test_column_eq_i32() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[1i32, 5, 3]).unwrap();
    let result = a.eq(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_column_ne_i32() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[1i32, 5, 3]).unwrap();
    let result = a.ne(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_column_lt_i32() {
    let a = Column::from_slice(&[1i32, 5, 3]).unwrap();
    let b = Column::from_slice(&[2i32, 2, 3]).unwrap();
    let result = a.lt(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_column_gt_i32() {
    let a = Column::from_slice(&[1i32, 5, 3]).unwrap();
    let b = Column::from_slice(&[2i32, 2, 3]).unwrap();
    let result = a.gt(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_column_le_i32() {
    let a = Column::from_slice(&[1i32, 5, 3]).unwrap();
    let b = Column::from_slice(&[2i32, 2, 3]).unwrap();
    let result = a.le(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_column_ge_i32() {
    let a = Column::from_slice(&[1i32, 5, 3]).unwrap();
    let b = Column::from_slice(&[2i32, 2, 3]).unwrap();
    let result = a.ge(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn test_column_eq_string() {
    let a = Column::from_strs(&["foo", "bar", "baz"]).unwrap();
    let b = Column::from_strs(&["foo", "qux", "baz"]).unwrap();
    let result = a.eq(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_column_lt_string() {
    let a = Column::from_strs(&["apple", "cherry", "banana"]).unwrap();
    let b = Column::from_strs(&["banana", "banana", "banana"]).unwrap();
    let result = a.lt(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_column_lt_mixed_numeric() {
    // Int32 < Float64 → Bool8
    let a = Column::from_slice(&[1i32, 5, 3]).unwrap();
    let b = Column::from_slice(&[2.0f64, 2.0, 3.0]).unwrap();
    let result = a.lt(&b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_column_comparison_nullable() {
    let a = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let b = Column::from_slice(&[1i32, 2, 4]).unwrap();
    let result = a.eq(&b).unwrap();
    let rv = result.view().unwrap();
    let validity = rv.validity().unwrap().unwrap();
    assert!(validity[0]); // non-null: 1 == 1 → true
    assert!(!validity[1]); // null
    assert!(validity[2]); // non-null: 3 == 4 → false
}

// ---------------------------------------------------------------------------
// Scalar arithmetic and comparison on Column
// ---------------------------------------------------------------------------

#[test]
fn test_column_add_scalar_i32() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(10i32).unwrap();
    let result = col.add_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![11, 12, 13]);
}

#[test]
fn test_column_sub_scalar_i32() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let scalar = Scalar::from_value(5i32).unwrap();
    let result = col.sub_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![5, 15, 25]);
}

#[test]
fn test_column_mul_scalar_mixed_types() {
    // Int32 * Float64 scalar → Float64
    let col = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let scalar = Scalar::from_value(1.5f64).unwrap();
    let result = col.mul_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Float64);
    let values = rv.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![3.0, 4.5, 6.0]);
}

#[test]
fn test_column_div_scalar_i32() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let scalar = Scalar::from_value(5i32).unwrap();
    let result = col.div_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![2, 4, 6]);
}

#[test]
fn test_column_eq_scalar_i32() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(2i32).unwrap();
    let result = col.eq_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_column_ne_scalar_i32() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(2i32).unwrap();
    let result = col.ne_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_column_lt_scalar_i32() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(2i32).unwrap();
    let result = col.lt_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_column_gt_scalar_i32() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(2i32).unwrap();
    let result = col.gt_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

#[test]
fn test_column_le_scalar_i32() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(2i32).unwrap();
    let result = col.le_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_column_ge_scalar_i32() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(2i32).unwrap();
    let result = col.ge_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn test_column_eq_scalar_string() {
    let col = Column::from_strs(&["active", "inactive", "active"]).unwrap();
    let scalar = Scalar::from_str("active").unwrap();
    let result = col.eq_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_column_lt_scalar_string() {
    let col = Column::from_strs(&["apple", "cherry", "banana"]).unwrap();
    let scalar = Scalar::from_str("banana").unwrap();
    let result = col.lt_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_column_add_scalar_string_error() {
    let col = Column::from_strs(&["foo", "bar"]).unwrap();
    let scalar = Scalar::from_str("baz").unwrap();
    let err = col.add_scalar(&scalar).err().expect("expected error");
    assert_eq!(err.operation(), "column::add_scalar");
}

#[test]
fn test_column_add_scalar_null() {
    // Adding a null scalar should produce all-null output
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::null_numeric::<i32>().unwrap();
    let result = col.add_scalar(&scalar).unwrap();
    let rv = result.view().unwrap();
    let validity = rv.validity().unwrap().unwrap();
    assert!(validity.iter().all(|&v| !v)); // all null
}

#[test]
fn test_column_binary_op_scalar_with_type() {
    // Explicit output type override for scalar ops
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::from_value(10i32).unwrap();
    let result = col.binary_op_scalar_with_type(&scalar, BinaryOperator::Add, &DataType::new(TypeId::Int64)).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(rv.to_vec::<i64>().unwrap(), vec![11, 12, 13]);
}

// ---------------------------------------------------------------------------
// Operator overloading via std::ops
// ---------------------------------------------------------------------------

#[test]
fn test_ops_add_column_f32_scalar() {
    let col = Column::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let result = (&col + 1.0f32).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_ops_mul_column_f64_scalar() {
    let col = Column::from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
    let result = (&col * 100.0f64).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<f64>().unwrap(), vec![100.0, 200.0, 300.0]);
}

#[test]
fn test_ops_sub_column_i32_scalar() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let result = (&col - 5i32).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![5, 15, 25]);
}

#[test]
fn test_ops_div_column_i64_scalar() {
    let col = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let result = (&col / 2i64).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i64>().unwrap(), vec![5, 10, 15]);
}

#[test]
fn test_ops_column_scalar_supports_remaining_integer_types() {
    let u8_col = Column::from_slice(&[1u8, 2, 3]).unwrap();
    let u16_col = Column::from_slice(&[10u16, 20, 30]).unwrap();
    let u32_col = Column::from_slice(&[2u32, 3, 4]).unwrap();
    let u64_col = Column::from_slice(&[20u64, 40, 60]).unwrap();
    let i8_col = Column::from_slice(&[10i8, 20, 30]).unwrap();
    let i16_col = Column::from_slice(&[2i16, 4, 6]).unwrap();

    let add = (&u8_col + 1u8).unwrap();
    assert_eq!(add.view().unwrap().to_vec::<u8>().unwrap(), vec![2, 3, 4]);

    let sub = (&u16_col - 5u16).unwrap();
    assert_eq!(sub.view().unwrap().to_vec::<u16>().unwrap(), vec![5, 15, 25]);

    let mul = (&u32_col * 10u32).unwrap();
    assert_eq!(mul.view().unwrap().to_vec::<u32>().unwrap(), vec![20, 30, 40]);

    let div = (&u64_col / 10u64).unwrap();
    assert_eq!(div.view().unwrap().to_vec::<u64>().unwrap(), vec![2, 4, 6]);

    let signed_add = (&i8_col + 1i8).unwrap();
    assert_eq!(signed_add.view().unwrap().to_vec::<i8>().unwrap(), vec![11, 21, 31]);

    let signed_div = (&i16_col / 2i16).unwrap();
    assert_eq!(signed_div.view().unwrap().to_vec::<i16>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_ops_add_column_column() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let result = (&a + &b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_ops_mul_column_column() {
    let a = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let b = Column::from_slice(&[5i32, 6, 7]).unwrap();
    let result = (&a * &b).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![10, 18, 28]);
}

#[test]
fn test_ops_chain() {
    let col = Column::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let result = (&(&col * 100.0f32).unwrap() + 1.0f32).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<f32>().unwrap(), vec![101.0, 201.0, 301.0]);
}
