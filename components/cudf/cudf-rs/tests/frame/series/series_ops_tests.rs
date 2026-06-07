//! Integration tests for `Series` arithmetic operator overloads.
//!
//! Validates the `Result<Series>`-returning operator contract for:
//! - `Series op Series` (all value/ref combos)
//! - `Series op scalar` (f32, f64, i32, i64)
//! - `scalar op Series` (f32, f64, i32, i64)
//!
//! All tests require a GPU.

use cudf::{Series, TypeId};

// ===========================================================================
// Series op Series — Result<Series> contract
// ===========================================================================

#[test]
fn test_series_add_operator_returns_result() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[10, 20, 30]).unwrap();

    let result: cudf::Result<Series> = &a + &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_series_sub_operator_returns_result() {
    let a = Series::new::<i32>("a", &[10, 20, 30]).unwrap();
    let b = Series::new::<i32>("b", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = &a - &b;
    let s = result.unwrap();

    assert_eq!(s.to_vec::<i32>().unwrap(), vec![9, 18, 27]);
}

#[test]
fn test_series_mul_operator_returns_result() {
    let a = Series::new::<i32>("a", &[2, 3, 4]).unwrap();
    let b = Series::new::<i32>("b", &[10, 10, 10]).unwrap();

    let result: cudf::Result<Series> = &a * &b;
    let s = result.unwrap();

    assert_eq!(s.to_vec::<i32>().unwrap(), vec![20, 30, 40]);
}

#[test]
fn test_series_div_operator_returns_result_f64() {
    let a = Series::new::<f64>("a", &[10.0, 20.0, 30.0]).unwrap();
    let b = Series::new::<f64>("b", &[2.0, 4.0, 5.0]).unwrap();

    let result: cudf::Result<Series> = &a / &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.to_vec::<f64>().unwrap(), vec![5.0, 5.0, 6.0]);
}

// -- owned/ref combo variants --

#[test]
fn test_series_add_owned_owned() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i32>("b", &[10, 20]).unwrap();

    let result: cudf::Result<Series> = a + b;
    assert_eq!(result.unwrap().to_vec::<i32>().unwrap(), vec![11, 22]);
}

#[test]
fn test_series_add_owned_ref() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i32>("b", &[10, 20]).unwrap();

    let result: cudf::Result<Series> = a + &b;
    assert_eq!(result.unwrap().to_vec::<i32>().unwrap(), vec![11, 22]);
}

#[test]
fn test_series_add_ref_owned() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i32>("b", &[10, 20]).unwrap();

    let result: cudf::Result<Series> = &a + b;
    assert_eq!(result.unwrap().to_vec::<i32>().unwrap(), vec![11, 22]);
}

// -- naming behavior: Series op Series produces unnamed result --

#[test]
fn test_series_op_series_produces_unnamed_result() {
    let a = Series::new::<i32>("named_a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("named_b", &[10, 20, 30]).unwrap();

    let result = (&a + &b).unwrap();
    assert_eq!(result.name(), None);
}

// -- dtype promotion --

#[test]
fn test_series_add_operator_mixed_type_promotion() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<f64>("b", &[0.5, 0.5, 0.5]).unwrap();

    let result = (&a + &b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

// ===========================================================================
// Series op scalar — Result<Series> contract
// ===========================================================================

#[test]
fn test_series_mul_scalar_f64_operator() {
    let s = Series::new::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = &s * 2.0_f64;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_series_add_scalar_i32_operator() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = &s + 10_i32;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i32>().unwrap(), vec![11, 12, 13]);
}

#[test]
fn test_series_sub_scalar_i64_operator() {
    let s = Series::new::<i64>("x", &[100, 200, 300]).unwrap();

    let result: cudf::Result<Series> = &s - 50_i64;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i64>().unwrap(), vec![50, 150, 250]);
}

#[test]
fn test_series_div_scalar_f32_operator() {
    let s = Series::new::<f32>("x", &[10.0, 20.0, 30.0]).unwrap();

    let result: cudf::Result<Series> = &s / 5.0_f32;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
}

// -- owned Series op scalar --

#[test]
fn test_series_mul_scalar_owned_operator() {
    let s = Series::new::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = s * 2.0_f64;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

// -- naming behavior: Series op scalar inherits Series name --

#[test]
fn test_series_op_scalar_inherits_name() {
    let s = Series::new::<f64>("my_col", &[1.0, 2.0, 3.0]).unwrap();

    let result = (&s * 2.0_f64).unwrap();
    assert_eq!(result.name(), Some("my_col"));
}

#[test]
fn test_series_op_scalar_unnamed_stays_unnamed() {
    // from_column produces an unnamed Series (name = None).
    let unnamed = Series::from_column(cudf::core::Column::from_slice(&[1i32, 2, 3]).unwrap()).unwrap();
    assert_eq!(unnamed.name(), None);

    let result = (&unnamed + 10_i32).unwrap();
    assert_eq!(result.name(), None);
}

// ===========================================================================
// scalar op Series — Result<Series> contract
// ===========================================================================

#[test]
fn test_scalar_mul_series_f64_operator() {
    let s = Series::new::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = 2.0_f64 * &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_scalar_add_series_i32_operator() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = 10_i32 + &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i32>().unwrap(), vec![11, 12, 13]);
}

// -- non-commutative: scalar - Series --

#[test]
fn test_scalar_sub_series_f64_operator() {
    let s = Series::new::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = 10.0_f64 - &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f64>().unwrap(), vec![9.0, 8.0, 7.0]);
}

#[test]
fn test_scalar_sub_series_i32_operator() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = 100_i32 - &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i32>().unwrap(), vec![99, 98, 97]);
}

// -- non-commutative: scalar / Series --

#[test]
fn test_scalar_div_series_f64_operator() {
    let s = Series::new::<f64>("x", &[2.0, 4.0, 5.0]).unwrap();

    let result: cudf::Result<Series> = 20.0_f64 / &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f64>().unwrap(), vec![10.0, 5.0, 4.0]);
}

#[test]
fn test_scalar_div_series_i64_operator() {
    let s = Series::new::<i64>("x", &[2, 5, 10]).unwrap();

    let result: cudf::Result<Series> = 100_i64 / &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i64>().unwrap(), vec![50, 20, 10]);
}

// -- scalar op owned Series --

#[test]
fn test_scalar_mul_owned_series_operator() {
    let s = Series::new::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = 2.0_f64 * s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

// -- naming behavior: scalar op Series inherits Series name --

#[test]
fn test_scalar_op_series_inherits_name() {
    let s = Series::new::<f64>("my_col", &[1.0, 2.0]).unwrap();

    let result = (2.0_f64 * &s).unwrap();
    assert_eq!(result.name(), Some("my_col"));
}

#[test]
fn test_scalar_sub_series_inherits_name() {
    let s = Series::new::<f64>("revenue", &[100.0, 200.0]).unwrap();

    let result = (1000.0_f64 - &s).unwrap();
    assert_eq!(result.name(), Some("revenue"));
}

// ===========================================================================
// Error handling — surfaces Err, not panic
// ===========================================================================

#[test]
fn test_series_op_series_type_mismatch_returns_err() {
    let nums = Series::new::<i32>("nums", &[1, 2, 3]).unwrap();
    let strs = Series::from_strings("strs", &["a", "b", "c"]).unwrap();

    let result: cudf::Result<Series> = &nums + &strs;
    assert!(result.is_err(), "Expected Err for type mismatch, got Ok");
}

// ===========================================================================
// All four scalar types work for each direction
// ===========================================================================

#[test]
fn test_series_add_all_scalar_types() {
    let i32s = Series::new::<i32>("x", &[1, 2]).unwrap();
    let i64s = Series::new::<i64>("x", &[1, 2]).unwrap();
    let f32s = Series::new::<f32>("x", &[1.0, 2.0]).unwrap();
    let f64s = Series::new::<f64>("x", &[1.0, 2.0]).unwrap();

    // Series + scalar
    assert!((&i32s + 1_i32).is_ok());
    assert!((&i64s + 1_i64).is_ok());
    assert!((&f32s + 1.0_f32).is_ok());
    assert!((&f64s + 1.0_f64).is_ok());

    // scalar + Series
    assert!((1_i32 + &i32s).is_ok());
    assert!((1_i64 + &i64s).is_ok());
    assert!((1.0_f32 + &f32s).is_ok());
    assert!((1.0_f64 + &f64s).is_ok());
}

#[test]
fn test_series_sub_all_scalar_types() {
    let i32s = Series::new::<i32>("x", &[10, 20]).unwrap();
    let f64s = Series::new::<f64>("x", &[10.0, 20.0]).unwrap();

    assert!((&i32s - 1_i32).is_ok());
    assert!((&f64s - 1.0_f64).is_ok());
    assert!((100_i32 - &i32s).is_ok());
    assert!((100.0_f64 - &f64s).is_ok());
}

#[test]
fn test_series_div_all_scalar_types() {
    let i32s = Series::new::<i32>("x", &[10, 20]).unwrap();
    let f64s = Series::new::<f64>("x", &[10.0, 20.0]).unwrap();

    assert!((&i32s / 2_i32).is_ok());
    assert!((&f64s / 2.0_f64).is_ok());
    assert!((100_i32 / &i32s).is_ok());
    assert!((100.0_f64 / &f64s).is_ok());
}

// ===========================================================================
// Rem (modulo) operator — Series % Series and Series % scalar
// ===========================================================================

#[test]
fn test_series_rem_operator_i32() {
    let a = Series::new::<i32>("a", &[10, 23, 17]).unwrap();
    let b = Series::new::<i32>("b", &[3, 5, 4]).unwrap();

    let result: cudf::Result<Series> = &a % &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec::<i32>().unwrap(), vec![1, 3, 1]);
}

#[test]
fn test_series_rem_operator_f64() {
    let a = Series::new::<f64>("a", &[10.5, 7.0]).unwrap();
    let b = Series::new::<f64>("b", &[3.0, 2.5]).unwrap();

    let result: cudf::Result<Series> = &a % &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    let vals = s.to_vec::<f64>().unwrap();
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_series_rem_scalar_i32_operator() {
    let s = Series::new::<i32>("x", &[10, 23, 17]).unwrap();

    let result: cudf::Result<Series> = &s % 3_i32;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i32>().unwrap(), vec![1, 2, 2]);
}

#[test]
fn test_series_rem_scalar_f64_operator() {
    let s = Series::new::<f64>("x", &[10.5, 7.0]).unwrap();

    let result: cudf::Result<Series> = &s % 3.0_f64;
    let out = result.unwrap();

    let vals = out.to_vec::<f64>().unwrap();
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_series_rem_owned_owned() {
    let a = Series::new::<i32>("a", &[10, 7]).unwrap();
    let b = Series::new::<i32>("b", &[3, 4]).unwrap();

    let result: cudf::Result<Series> = a % b;
    assert_eq!(result.unwrap().to_vec::<i32>().unwrap(), vec![1, 3]);
}

#[test]
fn test_series_rem_produces_unnamed_result() {
    let a = Series::new::<i32>("named_a", &[10]).unwrap();
    let b = Series::new::<i32>("named_b", &[3]).unwrap();

    let result = (&a % &b).unwrap();
    assert_eq!(result.name(), None);
}

#[test]
fn test_series_rem_scalar_inherits_name() {
    let s = Series::new::<i32>("my_col", &[10, 23]).unwrap();

    let result = (&s % 3_i32).unwrap();
    assert_eq!(result.name(), Some("my_col"));
}

#[test]
fn test_scalar_rem_series_i32() {
    let s = Series::new::<i32>("x", &[3, 4]).unwrap();

    let result: cudf::Result<Series> = 10_i32 % &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_scalar_rem_series_f64() {
    let s = Series::new::<f64>("x", &[3.0]).unwrap();

    let result: cudf::Result<Series> = 10.0_f64 % &s;
    let out = result.unwrap();

    let vals = out.to_vec::<f64>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

// ===========================================================================
// Neg operator — -Series
// ===========================================================================

#[test]
fn test_series_neg_operator_i32() {
    let s = Series::new::<i32>("x", &[1, -2, 3]).unwrap();

    let result: cudf::Result<Series> = -&s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<i32>().unwrap(), vec![-1, 2, -3]);
}

#[test]
fn test_series_neg_operator_f64() {
    let s = Series::new::<f64>("x", &[1.5, -2.5]).unwrap();

    let result: cudf::Result<Series> = -&s;
    let out = result.unwrap();

    assert_eq!(out.to_vec::<f64>().unwrap(), vec![-1.5, 2.5]);
}

#[test]
fn test_series_neg_owned() {
    let s = Series::new::<i32>("x", &[1, -2]).unwrap();

    let result: cudf::Result<Series> = -s;
    assert_eq!(result.unwrap().to_vec::<i32>().unwrap(), vec![-1, 2]);
}

#[test]
fn test_series_neg_inherits_name() {
    let s = Series::new::<i32>("my_col", &[1, 2]).unwrap();

    let result = (-&s).unwrap();
    assert_eq!(result.name(), Some("my_col"));
}

// ===========================================================================
// Not operator — !Series (logical not for Bool8)
// ===========================================================================

#[test]
fn test_series_not_operator_bool() {
    let s = Series::new::<bool>("x", &[true, false, true]).unwrap();

    let result: cudf::Result<Series> = !&s;
    let out = result.unwrap();

    assert_eq!(out.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(out.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_series_not_owned() {
    let s = Series::new::<bool>("x", &[true, false]).unwrap();

    let result: cudf::Result<Series> = !s;
    assert_eq!(result.unwrap().to_vec::<bool>().unwrap(), vec![false, true]);
}

#[test]
fn test_series_not_inherits_name() {
    let s = Series::new::<bool>("flags", &[true, false]).unwrap();

    let result = (!&s).unwrap();
    assert_eq!(result.name(), Some("flags"));
}

// ===========================================================================
// Named rem / rem_scalar methods
// ===========================================================================

#[test]
fn test_series_rem_method() {
    let a = Series::new::<i32>("a", &[10, 23]).unwrap();
    let b = Series::new::<i32>("b", &[3, 5]).unwrap();

    let result = a.rem(&b).unwrap();
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![1, 3]);
}

#[test]
fn test_series_rem_scalar_method() {
    let s = Series::new::<i32>("x", &[10, 23]).unwrap();

    let result = s.rem_scalar(3i32).unwrap();
    assert_eq!(result.name(), Some("x"));
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![1, 2]);
}

// ===========================================================================
// Non-commutative scalar ops produce correct directional results
// ===========================================================================

#[test]
fn test_scalar_sub_is_not_commutative() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();

    let forward = (&s - 10_i32).unwrap();
    let reverse = (10_i32 - &s).unwrap();

    let fwd_vals = forward.to_vec::<i32>().unwrap();
    let rev_vals = reverse.to_vec::<i32>().unwrap();

    assert_eq!(fwd_vals, vec![-9, -8, -7]);
    assert_eq!(rev_vals, vec![9, 8, 7]);
}

#[test]
fn test_scalar_div_is_not_commutative() {
    let s = Series::new::<f64>("x", &[2.0, 4.0, 5.0]).unwrap();

    let forward = (&s / 20.0_f64).unwrap();
    let reverse = (20.0_f64 / &s).unwrap();

    let fwd_vals = forward.to_vec::<f64>().unwrap();
    let rev_vals = reverse.to_vec::<f64>().unwrap();

    assert_eq!(fwd_vals, vec![0.1, 0.2, 0.25]);
    assert_eq!(rev_vals, vec![10.0, 5.0, 4.0]);
}
