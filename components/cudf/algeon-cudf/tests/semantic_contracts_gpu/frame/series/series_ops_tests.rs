//! Covers `Series op Series` and `Series op scalar` through the named
//! methods on `SeriesOps`, including dtype promotion, decimal scales,
//! timestamp differences, and error reporting.

use cudf::{DataType, Exec, Series, TypeId};

use crate::common::{error_context_value, host_export_i32_values, host_export_i64_values, nullable_series, series};

// Series-vs-Series operators return `Result<Series>`.

fn test_series_add_returns_result() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = series::<i32>("b", &[10, 20, 30]).unwrap();

    let s = a.on(Exec::RETAINED).add(&b).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

fn test_series_sub_returns_result() {
    let a = series::<i32>("a", &[10, 20, 30]).unwrap();
    let b = series::<i32>("b", &[1, 2, 3]).unwrap();

    let s = a.on(Exec::RETAINED).sub(&b).unwrap();

    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![9, 18, 27]);
}

fn test_series_mul_returns_result() {
    let a = series::<i32>("a", &[2, 3, 4]).unwrap();
    let b = series::<i32>("b", &[10, 10, 10]).unwrap();

    let s = a.on(Exec::RETAINED).mul(&b).unwrap();

    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![20, 30, 40]);
}

fn test_series_div_returns_result_f64() {
    let a = series::<f64>("a", &[10.0, 20.0, 30.0]).unwrap();
    let b = series::<f64>("b", &[2.0, 4.0, 5.0]).unwrap();

    let s = a.on(Exec::RETAINED).div(&b).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.to_vec_strict::<f64>().unwrap(), vec![5.0, 5.0, 6.0]);
}

fn test_series_add_mixed_type_promotion() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = series::<f64>("b", &[0.5, 0.5, 0.5]).unwrap();

    let result = a.on(Exec::RETAINED).add(&b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

fn test_series_add_decimal64_same_scale_preserves_scale_and_raw_values() {
    let a = series("a", &[12_345i64, 20_000, 0])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();
    let b = series("b", &[100i64, -500, 25])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();

    let result = a.on(Exec::RETAINED).add(&b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -2).unwrap());
    assert_eq!(host_export_i64_values(&result), vec![1_244_500, 1_950_000, 2_500]);
}

fn test_series_mul_decimal64_same_scale_sums_scales_and_raw_values() {
    let a = series("a", &[12_345i64, 20_000])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();
    let b = series("b", &[100i64, -500])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();

    let result = a.on(Exec::RETAINED).mul(&b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4).unwrap());
    assert_eq!(host_export_i64_values(&result), vec![12_345_000_000, -100_000_000_000]);
}

fn test_series_add_decimal64_mixed_scale_uses_finer_scale_and_raw_values() {
    let a = series("a", &[1i64, 2])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();
    let b = series("b", &[3i64, 4])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -4).unwrap())
        .unwrap();

    let result = a.on(Exec::RETAINED).add(&b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4).unwrap());
    assert_eq!(host_export_i64_values(&result), vec![40_000, 60_000]);
}

fn test_series_add_nullable_decimal64_preserves_scale_and_nulls() {
    let a = nullable_series("a", &[1i64, 0, 3], &[true, false, true])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();
    let b = series("b", &[10i64, 20, 30])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();

    let result = a.on(Exec::RETAINED).add(&b).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -2).unwrap());
    assert_eq!(result.null_count(), 1);
}

fn test_series_timestamp_days_filter_predicate_matches_tpch_date_shape() {
    let dates = series("shipdate", &[18_262i32, 18_263, 18_264])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();
    let cutoff = series("cutoff", &[18_263i32, 18_263, 18_263])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();

    let result = dates.on(Exec::RETAINED).lt(&cutoff).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn test_series_timestamp_minus_timestamp_returns_duration_days() {
    let lhs = series("lhs", &[18_265i32, 18_270])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();
    let rhs = series("rhs", &[18_262i32, 18_263])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();

    let result = lhs.on(Exec::RETAINED).sub(&rhs).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::DurationDays);
    assert_eq!(host_export_i32_values(&result), vec![3, 7]);
}

// Series-scalar operators return `Result<Series>`.

fn test_series_mul_scalar_f64() {
    let s = series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();

    let out = s.on(Exec::RETAINED).mul_scalar(2.0_f64).unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

fn test_series_add_scalar_i32() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();

    let out = s.on(Exec::RETAINED).add_scalar(10_i32).unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![11, 12, 13]);
}

fn test_series_sub_scalar_i64() {
    let s = series::<i64>("x", &[100, 200, 300]).unwrap();

    let out = s.on(Exec::RETAINED).sub_scalar(50_i64).unwrap();

    assert_eq!(out.to_vec_strict::<i64>().unwrap(), vec![50, 150, 250]);
}

fn test_series_div_scalar_f32() {
    let s = series::<f32>("x", &[10.0, 20.0, 30.0]).unwrap();

    let out = s.on(Exec::RETAINED).div_scalar(5.0_f32).unwrap();

    assert_eq!(out.to_vec_strict::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
}

// Operator errors are returned as `Err` values.

fn test_series_op_series_type_mismatch_returns_err() {
    let nums = series::<i32>("nums", &[1, 2, 3]).unwrap();
    let strs = series("strs", &["a", "b", "c"]).unwrap();

    let result: cudf::Result<Series> = nums.on(Exec::RETAINED).add(&strs);
    let err = result.expect_err("Expected Err for type mismatch");
    assert_eq!(err.operation(), "series::add");
    assert_eq!(error_context_value(&err, "operator").as_deref(), Some("add"));
}

fn test_series_scalar_ops_support_remaining_integer_types() {
    let u8s = series::<u8>("u8s", &[1, 2, 3]).unwrap();
    let u16s = series::<u16>("u16s", &[10, 20, 30]).unwrap();
    let u32s = series::<u32>("u32s", &[2, 3, 4]).unwrap();
    let u64s = series::<u64>("u64s", &[20, 40, 60]).unwrap();

    let series_add = u8s.on(Exec::RETAINED).add_scalar(1u8).unwrap();
    assert_eq!(series_add.to_vec_strict::<u8>().unwrap(), vec![2, 3, 4]);

    let series_sub = u16s.on(Exec::RETAINED).sub_scalar(5u16).unwrap();
    assert_eq!(series_sub.to_vec_strict::<u16>().unwrap(), vec![5, 15, 25]);

    let series_mul = u32s.on(Exec::RETAINED).mul_scalar(10u32).unwrap();
    assert_eq!(series_mul.to_vec_strict::<u32>().unwrap(), vec![20, 30, 40]);

    let series_div = u64s.on(Exec::RETAINED).div_scalar(10u64).unwrap();
    assert_eq!(series_div.to_vec_strict::<u64>().unwrap(), vec![2, 4, 6]);
}

// Rem (modulo) covers `Series % Series` and `Series % scalar`.

fn test_series_rem_i32() {
    let a = series::<i32>("a", &[10, 23, 17]).unwrap();
    let b = series::<i32>("b", &[3, 5, 4]).unwrap();

    let s = a.on(Exec::RETAINED).rem(&b).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![1, 3, 1]);
}

fn test_series_rem_f64() {
    let a = series::<f64>("a", &[10.5, 7.0]).unwrap();
    let b = series::<f64>("b", &[3.0, 2.5]).unwrap();

    let s = a.on(Exec::RETAINED).rem(&b).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    let vals = s.to_vec_strict::<f64>().unwrap();
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
}

fn test_series_rem_scalar_i32() {
    let s = series::<i32>("x", &[10, 23, 17]).unwrap();

    let out = s.on(Exec::RETAINED).rem_scalar(3_i32).unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![1, 2, 2]);
}

fn test_series_rem_scalar_f64() {
    let s = series::<f64>("x", &[10.5, 7.0]).unwrap();

    let out = s.on(Exec::RETAINED).rem_scalar(3.0_f64).unwrap();

    let vals = out.to_vec_strict::<f64>().unwrap();
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 1.0).abs() < 1e-10);
}

fn test_series_neg_i32() {
    let s = series::<i32>("x", &[1, -2, 3]).unwrap();

    let out = s.on(Exec::RETAINED).neg().unwrap();

    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![-1, 2, -3]);
}

fn test_series_neg_f64() {
    let s = series::<f64>("x", &[1.5, -2.5]).unwrap();

    let out = s.on(Exec::RETAINED).neg().unwrap();

    assert_eq!(out.to_vec_strict::<f64>().unwrap(), vec![-1.5, 2.5]);
}

fn test_series_not_bool() {
    let s = series::<bool>("x", &[true, false, true]).unwrap();

    let out = s.on(Exec::RETAINED).logical_not().unwrap();

    assert_eq!(out.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(out.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_series_logical_and_method_bool() {
    let a = series::<bool>("lhs", &[true, false, true]).unwrap();
    let b = series::<bool>("rhs", &[true, true, false]).unwrap();

    let result = a.on(Exec::RETAINED).logical_and(&b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn test_series_logical_or_method_bool() {
    let a = series::<bool>("lhs", &[true, false, true]).unwrap();
    let b = series::<bool>("rhs", &[false, true, false]).unwrap();

    let result = a.on(Exec::RETAINED).logical_or(&b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("series_ops_tests::test_series_add_returns_result", test_series_add_returns_result),
    ("series_ops_tests::test_series_sub_returns_result", test_series_sub_returns_result),
    ("series_ops_tests::test_series_mul_returns_result", test_series_mul_returns_result),
    ("series_ops_tests::test_series_div_returns_result_f64", test_series_div_returns_result_f64),
    ("series_ops_tests::test_series_add_mixed_type_promotion", test_series_add_mixed_type_promotion),
    (
        "series_ops_tests::test_series_add_decimal64_same_scale_preserves_scale_and_raw_values",
        test_series_add_decimal64_same_scale_preserves_scale_and_raw_values,
    ),
    (
        "series_ops_tests::test_series_mul_decimal64_same_scale_sums_scales_and_raw_values",
        test_series_mul_decimal64_same_scale_sums_scales_and_raw_values,
    ),
    (
        "series_ops_tests::test_series_add_decimal64_mixed_scale_uses_finer_scale_and_raw_values",
        test_series_add_decimal64_mixed_scale_uses_finer_scale_and_raw_values,
    ),
    (
        "series_ops_tests::test_series_add_nullable_decimal64_preserves_scale_and_nulls",
        test_series_add_nullable_decimal64_preserves_scale_and_nulls,
    ),
    (
        "series_ops_tests::test_series_timestamp_days_filter_predicate_matches_tpch_date_shape",
        test_series_timestamp_days_filter_predicate_matches_tpch_date_shape,
    ),
    (
        "series_ops_tests::test_series_timestamp_minus_timestamp_returns_duration_days",
        test_series_timestamp_minus_timestamp_returns_duration_days,
    ),
    ("series_ops_tests::test_series_mul_scalar_f64", test_series_mul_scalar_f64),
    ("series_ops_tests::test_series_add_scalar_i32", test_series_add_scalar_i32),
    ("series_ops_tests::test_series_sub_scalar_i64", test_series_sub_scalar_i64),
    ("series_ops_tests::test_series_div_scalar_f32", test_series_div_scalar_f32),
    (
        "series_ops_tests::test_series_op_series_type_mismatch_returns_err",
        test_series_op_series_type_mismatch_returns_err,
    ),
    (
        "series_ops_tests::test_series_scalar_ops_support_remaining_integer_types",
        test_series_scalar_ops_support_remaining_integer_types,
    ),
    ("series_ops_tests::test_series_rem_i32", test_series_rem_i32),
    ("series_ops_tests::test_series_rem_f64", test_series_rem_f64),
    ("series_ops_tests::test_series_rem_scalar_i32", test_series_rem_scalar_i32),
    ("series_ops_tests::test_series_rem_scalar_f64", test_series_rem_scalar_f64),
    ("series_ops_tests::test_series_neg_i32", test_series_neg_i32),
    ("series_ops_tests::test_series_neg_f64", test_series_neg_f64),
    ("series_ops_tests::test_series_not_bool", test_series_not_bool),
    ("series_ops_tests::test_series_logical_and_method_bool", test_series_logical_and_method_bool),
    ("series_ops_tests::test_series_logical_or_method_bool", test_series_logical_or_method_bool),
];
