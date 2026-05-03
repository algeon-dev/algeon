//! Integration tests for `Series` arithmetic operator overloads.
//!
//! Validates the `Result<Series>`-returning operator contract for:
//! - `Series op Series` (all value/ref combos for arithmetic operators)
//! - `Series op scalar` (all arithmetic numeric `CudfType`s)
//! - `scalar op Series` (all arithmetic numeric `CudfType`s)
//!
//! All tests require a GPU.

use cudf::{DataType, Series, TypeId};

use crate::common::{column_from_slice, host_export_i32_values, host_export_i64_values};

// ===========================================================================
// Series op Series — Result<Series> contract
// ===========================================================================

#[test]
fn test_series_add_operator_returns_result() {
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20, 30]).unwrap();

    let result: cudf::Result<Series> = &a + &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_series_sub_operator_returns_result() {
    let a = crate::common::series::<i32>("a", &[10, 20, 30]).unwrap();
    let b = crate::common::series::<i32>("b", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = &a - &b;
    let s = result.unwrap();

    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![9, 18, 27]);
}

#[test]
fn test_series_mul_operator_returns_result() {
    let a = crate::common::series::<i32>("a", &[2, 3, 4]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 10, 10]).unwrap();

    let result: cudf::Result<Series> = &a * &b;
    let s = result.unwrap();

    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![20, 30, 40]);
}

#[test]
fn test_series_div_operator_returns_result_f64() {
    let a = crate::common::series::<f64>("a", &[10.0, 20.0, 30.0]).unwrap();
    let b = crate::common::series::<f64>("b", &[2.0, 4.0, 5.0]).unwrap();

    let result: cudf::Result<Series> = &a / &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.to_vec_strict::<f64>().unwrap(), vec![5.0, 5.0, 6.0]);
}

// -- owned/ref combo variants --

#[test]
fn test_series_add_owned_owned() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20]).unwrap();

    let result: cudf::Result<Series> = a + b;
    assert_eq!(result.unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22]);
}

#[test]
fn test_series_add_owned_ref() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20]).unwrap();

    let result: cudf::Result<Series> = a + &b;
    assert_eq!(result.unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22]);
}

#[test]
fn test_series_add_ref_owned() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20]).unwrap();

    let result: cudf::Result<Series> = &a + b;
    assert_eq!(result.unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22]);
}

// -- output contract: Series formatting never exposes schema names --

#[test]
fn test_series_op_series_produces_unnamed_result() {
    let a = crate::common::series::<i32>("named_a", &[1, 2, 3]).unwrap();
    let b = crate::common::series::<i32>("named_b", &[10, 20, 30]).unwrap();

    let result = (&a + &b).unwrap();
    let output = format!("{}", result);
    assert!(
        !output.contains("named_a") && !output.contains("named_b"),
        "series + series output should not expose schema names: {}",
        output
    );
}

// -- dtype promotion --

#[test]
fn test_series_add_operator_mixed_type_promotion() {
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = crate::common::series::<f64>("b", &[0.5, 0.5, 0.5]).unwrap();

    let result = (&a + &b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_series_add_decimal64_same_scale_preserves_scale_and_raw_values() {
    let a = crate::common::series("a", &[12_345i64, 20_000, 0])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal64, -2))
        .unwrap();
    let b = crate::common::series("b", &[100i64, -500, 25])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal64, -2))
        .unwrap();

    let result = (&a + &b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -2));
    assert_eq!(host_export_i64_values(&result), vec![1_244_500, 1_950_000, 2_500]);
}

#[test]
fn test_series_mul_decimal64_same_scale_sums_scales_and_raw_values() {
    let a = crate::common::series("a", &[12_345i64, 20_000])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal64, -2))
        .unwrap();
    let b = crate::common::series("b", &[100i64, -500])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal64, -2))
        .unwrap();

    let result = (&a * &b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4));
    assert_eq!(host_export_i64_values(&result), vec![12_345_000_000, -100_000_000_000]);
}

#[test]
fn test_series_add_decimal64_mixed_scale_uses_finer_scale_and_raw_values() {
    let a =
        crate::common::series("a", &[1i64, 2]).unwrap().astype(DataType::fixed_point(TypeId::Decimal64, -2)).unwrap();
    let b =
        crate::common::series("b", &[3i64, 4]).unwrap().astype(DataType::fixed_point(TypeId::Decimal64, -4)).unwrap();

    let result = (&a + &b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4));
    assert_eq!(host_export_i64_values(&result), vec![40_000, 60_000]);
}

#[test]
fn test_series_add_nullable_decimal64_preserves_scale_and_nulls() {
    let a = crate::common::nullable_series("a", &[1i64, 0, 3], &[true, false, true])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal64, -2))
        .unwrap();
    let b = crate::common::series("b", &[10i64, 20, 30])
        .unwrap()
        .astype(DataType::fixed_point(TypeId::Decimal64, -2))
        .unwrap();

    let result = (&a + &b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -2));
    assert_eq!(result.null_count(), 1);
}

#[test]
fn test_series_timestamp_days_filter_predicate_matches_tpch_date_shape() {
    let dates = crate::common::series("shipdate", &[18_262i32, 18_263, 18_264])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    let cutoff = crate::common::series("cutoff", &[18_263i32, 18_263, 18_263])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();

    let result = dates.lt(&cutoff).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_series_timestamp_minus_timestamp_returns_duration_days() {
    let lhs = crate::common::series("lhs", &[18_265i32, 18_270])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    let rhs = crate::common::series("rhs", &[18_262i32, 18_263])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();

    let result = (&lhs - &rhs).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::DurationDays);
    assert_eq!(host_export_i32_values(&result), vec![3, 7]);
}

// ===========================================================================
// Series op scalar — Result<Series> contract
// ===========================================================================

#[test]
fn test_series_mul_scalar_f64_operator() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = &s * 2.0_f64;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_series_add_scalar_i32_operator() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = &s + 10_i32;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![11, 12, 13]);
}

#[test]
fn test_series_sub_scalar_i64_operator() {
    let s = crate::common::series::<i64>("x", &[100, 200, 300]).unwrap();

    let result: cudf::Result<Series> = &s - 50_i64;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i64>().unwrap(), vec![50, 150, 250]);
}

#[test]
fn test_series_div_scalar_f32_operator() {
    let s = crate::common::series::<f32>("x", &[10.0, 20.0, 30.0]).unwrap();

    let result: cudf::Result<Series> = &s / 5.0_f32;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
}

// -- owned Series op scalar --

#[test]
fn test_series_mul_scalar_owned_operator() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = s * 2.0_f64;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

// -- output contract: scalar ops also omit schema names in Series formatting --

#[test]
fn test_series_op_scalar_display_omits_input_name() {
    let s = crate::common::series::<f64>("my_col", &[1.0, 2.0, 3.0]).unwrap();

    let result = (&s * 2.0_f64).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_col"), "series * scalar output should not expose schema names: {}", output);
}

#[test]
fn test_series_op_scalar_unnamed_stays_unnamed() {
    // from_column produces an unnamed Series.
    let unnamed = Series::from_column(column_from_slice(&[1i32, 2, 3]).unwrap()).unwrap();

    let result = (&unnamed + 10_i32).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("name="), "unnamed series output should stay unnamed: {}", output);
}

// ===========================================================================
// scalar op Series — Result<Series> contract
// ===========================================================================

#[test]
fn test_scalar_mul_series_f64_operator() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = 2.0_f64 * &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_scalar_add_series_i32_operator() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = 10_i32 + &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![11, 12, 13]);
}

// -- non-commutative: scalar - Series --

#[test]
fn test_scalar_sub_series_f64_operator() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = 10.0_f64 - &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![9.0, 8.0, 7.0]);
}

#[test]
fn test_scalar_sub_series_i32_operator() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();

    let result: cudf::Result<Series> = 100_i32 - &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![99, 98, 97]);
}

// -- non-commutative: scalar / Series --

#[test]
fn test_scalar_div_series_f64_operator() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 5.0]).unwrap();

    let result: cudf::Result<Series> = 20.0_f64 / &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![10.0, 5.0, 4.0]);
}

#[test]
fn test_scalar_div_series_i64_operator() {
    let s = crate::common::series::<i64>("x", &[2, 5, 10]).unwrap();

    let result: cudf::Result<Series> = 100_i64 / &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i64>().unwrap(), vec![50, 20, 10]);
}

// -- scalar op owned Series --

#[test]
fn test_scalar_mul_owned_series_operator() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let result: cudf::Result<Series> = 2.0_f64 * s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

// -- output contract: scalar-on-series formatting omits schema names --

#[test]
fn test_scalar_op_series_display_omits_input_name() {
    let s = crate::common::series::<f64>("my_col", &[1.0, 2.0]).unwrap();

    let result = (2.0_f64 * &s).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_col"), "scalar * series output should not expose schema names: {}", output);
}

#[test]
fn test_scalar_sub_series_display_omits_input_name() {
    let s = crate::common::series::<f64>("revenue", &[100.0, 200.0]).unwrap();

    let result = (1000.0_f64 - &s).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("revenue"), "scalar - series output should not expose schema names: {}", output);
}

// ===========================================================================
// Error handling — surfaces Err, not panic
// ===========================================================================

#[test]
fn test_series_op_series_type_mismatch_returns_err() {
    let nums = crate::common::series::<i32>("nums", &[1, 2, 3]).unwrap();
    let strs = crate::common::series("strs", &["a", "b", "c"]).unwrap();

    let result: cudf::Result<Series> = &nums + &strs;
    let err = result.expect_err("Expected Err for type mismatch");
    assert_eq!(err.operation(), "series::add");
    assert_eq!(crate::common::error_context_value(&err, "operator").as_deref(), Some("add"));
}

// ===========================================================================
// All four scalar types work for each direction
// ===========================================================================

#[test]
fn test_series_add_all_scalar_types() {
    let i32s = crate::common::series::<i32>("x", &[1, 2]).unwrap();
    let i64s = crate::common::series::<i64>("x", &[1, 2]).unwrap();
    let f32s = crate::common::series::<f32>("x", &[1.0, 2.0]).unwrap();
    let f64s = crate::common::series::<f64>("x", &[1.0, 2.0]).unwrap();

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
    let i32s = crate::common::series::<i32>("x", &[10, 20]).unwrap();
    let f64s = crate::common::series::<f64>("x", &[10.0, 20.0]).unwrap();

    assert!((&i32s - 1_i32).is_ok());
    assert!((&f64s - 1.0_f64).is_ok());
    assert!((100_i32 - &i32s).is_ok());
    assert!((100.0_f64 - &f64s).is_ok());
}

#[test]
fn test_series_div_all_scalar_types() {
    let i32s = crate::common::series::<i32>("x", &[10, 20]).unwrap();
    let f64s = crate::common::series::<f64>("x", &[10.0, 20.0]).unwrap();

    assert!((&i32s / 2_i32).is_ok());
    assert!((&f64s / 2.0_f64).is_ok());
    assert!((100_i32 / &i32s).is_ok());
    assert!((100.0_f64 / &f64s).is_ok());
}

#[test]
fn test_series_scalar_operators_support_remaining_integer_types() {
    let u8s = crate::common::series::<u8>("u8s", &[1, 2, 3]).unwrap();
    let u16s = crate::common::series::<u16>("u16s", &[10, 20, 30]).unwrap();
    let u32s = crate::common::series::<u32>("u32s", &[2, 3, 4]).unwrap();
    let u64s = crate::common::series::<u64>("u64s", &[20, 40, 60]).unwrap();
    let i8s = crate::common::series::<i8>("i8s", &[10, 20, 30]).unwrap();
    let i16s = crate::common::series::<i16>("i16s", &[2, 4, 6]).unwrap();

    let series_add = (&u8s + 1u8).unwrap();
    assert_eq!(series_add.to_vec_strict::<u8>().unwrap(), vec![2, 3, 4]);

    let series_sub = (&u16s - 5u16).unwrap();
    assert_eq!(series_sub.to_vec_strict::<u16>().unwrap(), vec![5, 15, 25]);

    let series_mul = (&u32s * 10u32).unwrap();
    assert_eq!(series_mul.to_vec_strict::<u32>().unwrap(), vec![20, 30, 40]);

    let series_div = (&u64s / 10u64).unwrap();
    assert_eq!(series_div.to_vec_strict::<u64>().unwrap(), vec![2, 4, 6]);

    let scalar_sub = (100i8 - &i8s).unwrap();
    assert_eq!(scalar_sub.to_vec_strict::<i8>().unwrap(), vec![90, 80, 70]);

    let scalar_mul = (3i16 * &i16s).unwrap();
    assert_eq!(scalar_mul.to_vec_strict::<i16>().unwrap(), vec![6, 12, 18]);
}

// ===========================================================================
// Rem (modulo) operator — Series % Series and Series % scalar
// ===========================================================================

#[test]
fn test_series_rem_operator_i32() {
    let a = crate::common::series::<i32>("a", &[10, 23, 17]).unwrap();
    let b = crate::common::series::<i32>("b", &[3, 5, 4]).unwrap();

    let result: cudf::Result<Series> = &a % &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![1, 3, 1]);
}

#[test]
fn test_series_rem_operator_f64() {
    let a = crate::common::series::<f64>("a", &[10.5, 7.0]).unwrap();
    let b = crate::common::series::<f64>("b", &[3.0, 2.5]).unwrap();

    let result: cudf::Result<Series> = &a % &b;
    let s = result.unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    let vals = s.to_vec_strict::<f64>().unwrap();
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_series_rem_scalar_i32_operator() {
    let s = crate::common::series::<i32>("x", &[10, 23, 17]).unwrap();

    let result: cudf::Result<Series> = &s % 3_i32;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![1, 2, 2]);
}

#[test]
fn test_series_rem_scalar_f64_operator() {
    let s = crate::common::series::<f64>("x", &[10.5, 7.0]).unwrap();

    let result: cudf::Result<Series> = &s % 3.0_f64;
    let out = result.unwrap();

    let vals = out.to_vec_strict::<f64>().unwrap();
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_series_rem_owned_owned() {
    let a = crate::common::series::<i32>("a", &[10, 7]).unwrap();
    let b = crate::common::series::<i32>("b", &[3, 4]).unwrap();

    let result: cudf::Result<Series> = a % b;
    assert_eq!(result.unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3]);
}

#[test]
fn test_series_rem_produces_unnamed_result() {
    let a = crate::common::series::<i32>("named_a", &[10]).unwrap();
    let b = crate::common::series::<i32>("named_b", &[3]).unwrap();

    let result = (&a % &b).unwrap();
    let output = format!("{}", result);
    assert!(
        !output.contains("named_a") && !output.contains("named_b"),
        "series % series output should not expose schema names: {}",
        output
    );
}

#[test]
fn test_series_rem_scalar_display_omits_input_name() {
    let s = crate::common::series::<i32>("my_col", &[10, 23]).unwrap();

    let result = (&s % 3_i32).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_col"), "series % scalar output should not expose schema names: {}", output);
}

#[test]
fn test_scalar_rem_series_i32() {
    let s = crate::common::series::<i32>("x", &[3, 4]).unwrap();

    let result: cudf::Result<Series> = 10_i32 % &s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_scalar_rem_series_f64() {
    let s = crate::common::series::<f64>("x", &[3.0]).unwrap();

    let result: cudf::Result<Series> = 10.0_f64 % &s;
    let out = result.unwrap();

    let vals = out.to_vec_strict::<f64>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

// ===========================================================================
// Neg operator — -Series
// ===========================================================================

#[test]
fn test_series_neg_operator_i32() {
    let s = crate::common::series::<i32>("x", &[1, -2, 3]).unwrap();

    let result: cudf::Result<Series> = -&s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![-1, 2, -3]);
}

#[test]
fn test_series_neg_operator_f64() {
    let s = crate::common::series::<f64>("x", &[1.5, -2.5]).unwrap();

    let result: cudf::Result<Series> = -&s;
    let out = result.unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![-1.5, 2.5]);
}

#[test]
fn test_series_neg_owned() {
    let s = crate::common::series::<i32>("x", &[1, -2]).unwrap();

    let result: cudf::Result<Series> = -s;
    assert_eq!(result.unwrap().to_vec_strict::<i32>().unwrap(), vec![-1, 2]);
}

#[test]
fn test_series_neg_display_omits_input_name() {
    let s = crate::common::series::<i32>("my_col", &[1, 2]).unwrap();

    let result = (-&s).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_col"), "negated series output should not expose schema names: {}", output);
}

// ===========================================================================
// Not operator — !Series (logical not for Bool8)
// ===========================================================================

#[test]
fn test_series_not_operator_bool() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();

    let result: cudf::Result<Series> = !&s;
    let out = result.unwrap();

    assert_eq!(out.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(out.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_series_not_owned() {
    let s = crate::common::series::<bool>("x", &[true, false]).unwrap();

    let result: cudf::Result<Series> = !s;
    assert_eq!(result.unwrap().to_vec_strict::<bool>().unwrap(), vec![false, true]);
}

#[test]
fn test_series_not_display_omits_input_name() {
    let s = crate::common::series::<bool>("flags", &[true, false]).unwrap();

    let result = (!&s).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("flags"), "logical-not series output should not expose schema names: {}", output);
}

#[test]
fn test_series_logical_and_method_bool() {
    let a = crate::common::series::<bool>("lhs", &[true, false, true]).unwrap();
    let b = crate::common::series::<bool>("rhs", &[true, true, false]).unwrap();

    let result = a.logical_and(&b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_series_logical_or_method_bool() {
    let a = crate::common::series::<bool>("lhs", &[true, false, true]).unwrap();
    let b = crate::common::series::<bool>("rhs", &[false, true, false]).unwrap();

    let result = a.logical_or(&b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
}

// ===========================================================================
// Named rem / rem_scalar methods
// ===========================================================================

#[test]
fn test_series_rem_method() {
    let a = crate::common::series::<i32>("a", &[10, 23]).unwrap();
    let b = crate::common::series::<i32>("b", &[3, 5]).unwrap();

    let result = a.rem(&b).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 3]);
}

#[test]
fn test_series_rem_scalar_method() {
    let s = crate::common::series::<i32>("x", &[10, 23]).unwrap();

    let result = s.rem_scalar(3i32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2]);
}

// ===========================================================================
// Non-commutative scalar ops produce correct directional results
// ===========================================================================

#[test]
fn test_scalar_sub_is_not_commutative() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();

    let forward = (&s - 10_i32).unwrap();
    let reverse = (10_i32 - &s).unwrap();

    let fwd_vals = forward.to_vec_strict::<i32>().unwrap();
    let rev_vals = reverse.to_vec_strict::<i32>().unwrap();

    assert_eq!(fwd_vals, vec![-9, -8, -7]);
    assert_eq!(rev_vals, vec![9, 8, 7]);
}

#[test]
fn test_scalar_div_is_not_commutative() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 5.0]).unwrap();

    let forward = (&s / 20.0_f64).unwrap();
    let reverse = (20.0_f64 / &s).unwrap();

    let fwd_vals = forward.to_vec_strict::<f64>().unwrap();
    let rev_vals = reverse.to_vec_strict::<f64>().unwrap();

    assert_eq!(fwd_vals, vec![0.1, 0.2, 0.25]);
    assert_eq!(rev_vals, vec![10.0, 5.0, 4.0]);
}
