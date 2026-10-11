//! Tests for binary and scalar operations that Series and Expr expose beyond the basic arithmetic coverage.

use cudf::{
    DataFrame, DataType, Exec, TypeId,
    ops::{
        binary::BinaryOperator,
        expr::{col, lit_null},
    },
};

use crate::common::series;

fn test_series_div_i32() {
    let a = series::<i32>("a", &[10, 21, 35]).unwrap();
    let b = series::<i32>("b", &[2, 7, 5]).unwrap();
    let result = a.on(Exec::RETAINED).div(&b).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![5, 3, 7]);
}

fn test_expr_true_div_i32_produces_float64() {
    let a = series::<i32>("a", &[1, 3, 7]).unwrap();
    let b = series::<i32>("b", &[2, 4, 2]).unwrap();
    let df = DataFrame::from_series([("a", a), ("b", b)]).unwrap();

    let result = col("a")
        .binary_op_with_type(col("b"), BinaryOperator::TrueDiv, DataType::new(TypeId::Float64))
        .eval(&df)
        .unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![0.5, 0.75, 3.5]);
}

fn test_expr_binary_op_with_type() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = series::<i32>("b", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([("a", a), ("b", b)]).unwrap();

    let result =
        col("a").binary_op_with_type(col("b"), BinaryOperator::Add, DataType::new(TypeId::Int64)).eval(&df).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int64);
    assert_eq!(result.to_vec_strict::<i64>().unwrap(), vec![11, 22, 33]);
}

fn test_expr_floor_div_f64() {
    let a = series::<f64>("a", &[7.0, 10.0, 15.0]).unwrap();
    let b = series::<f64>("b", &[2.0, 3.0, 4.0]).unwrap();
    let df = DataFrame::from_series([("a", a), ("b", b)]).unwrap();

    let result = col("a").floor_div(col("b")).eval(&df).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.0, 3.0, 3.0]);
}

fn test_series_mul_scalar_mixed_types() {
    // Int32 * Float64 scalar → Float64
    let s = series::<i32>("x", &[2, 3, 4]).unwrap();
    let result = s.on(Exec::RETAINED).mul_scalar(1.5f64).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.0, 4.5, 6.0]);
}

fn test_expr_add_null_literal_produces_all_null() {
    let df = DataFrame::from_series([("x", series::<i32>("x", &[1, 2, 3]).unwrap())]).unwrap();

    let result = (col("x") + lit_null(TypeId::Int32)).eval(&df).unwrap();

    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![None, None, None]);
}

fn test_series_scalar_ops_support_narrow_signed_integer_types() {
    let i8s = series::<i8>("i8s", &[10, 20, 30]).unwrap();
    let i16s = series::<i16>("i16s", &[2, 4, 6]).unwrap();

    let add = i8s.on(Exec::RETAINED).add_scalar(1i8).unwrap();
    assert_eq!(add.to_vec_strict::<i8>().unwrap(), vec![11, 21, 31]);

    let div = i16s.on(Exec::RETAINED).div_scalar(2i16).unwrap();
    assert_eq!(div.to_vec_strict::<i16>().unwrap(), vec![1, 2, 3]);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("column_ops_tests::test_series_div_i32", test_series_div_i32),
    ("column_ops_tests::test_expr_true_div_i32_produces_float64", test_expr_true_div_i32_produces_float64),
    ("column_ops_tests::test_expr_binary_op_with_type", test_expr_binary_op_with_type),
    ("column_ops_tests::test_expr_floor_div_f64", test_expr_floor_div_f64),
    ("column_ops_tests::test_series_mul_scalar_mixed_types", test_series_mul_scalar_mixed_types),
    ("column_ops_tests::test_expr_add_null_literal_produces_all_null", test_expr_add_null_literal_produces_all_null),
    (
        "column_ops_tests::test_series_scalar_ops_support_narrow_signed_integer_types",
        test_series_scalar_ops_support_narrow_signed_integer_types,
    ),
];
