//! Integration tests for `Series` reduction, scan, and unary methods.
//!
//! All tests require a GPU.

use cudf::{ErrorCode, ErrorKind, TypeId};

use crate::common::error_context_value;

// ===========================================================================
// sum
// ===========================================================================

#[test]
fn test_sum_i32() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(total.value::<i64>().unwrap(), 60);
}

#[test]
fn test_sum_f64() {
    let s = crate::common::series::<f64>("x", &[1.5, 2.5, 3.0]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(total.value::<f64>().unwrap(), 7.0);
}

#[test]
fn test_sum_bool8() {
    let s = crate::common::series::<bool>("x", &[true, false, true, true]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(total.value::<i64>().unwrap(), 3);
}

#[test]
fn test_sum_with_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.value::<i64>().unwrap(), 40);
}

#[test]
fn test_sum_string_reports_series_operation_on_type_error() {
    let s = crate::common::series("x", &["a", "b"]).unwrap();
    let err = s.sum().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::sum");
    assert_eq!(crate::common::error_context_value(&err, "type_id"), Some("string"));
    assert_eq!(crate::common::error_context_value(&err, "input_type"), None);
}

// ===========================================================================
// min / max
// ===========================================================================

#[test]
fn test_min_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.min().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(m.value::<i32>().unwrap(), 10);
}

#[test]
fn test_max_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.max().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(m.value::<i32>().unwrap(), 30);
}

#[test]
fn test_min_f64() {
    let s = crate::common::series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let m = s.min().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 1.0);
}

#[test]
fn test_max_f64() {
    let s = crate::common::series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let m = s.max().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 3.0);
}

// ===========================================================================
// mean / median
// ===========================================================================

#[test]
fn test_mean_i32() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let m = s.mean().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(m.value::<f64>().unwrap(), 20.0);
}

#[test]
fn test_mean_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let m = s.mean().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 2.0);
}

#[test]
fn test_median_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.median().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(m.value::<f64>().unwrap(), 20.0);
}

// ===========================================================================
// std / var
// ===========================================================================

#[test]
fn test_std_f64() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let ddof: u32 = 0;
    let std_val = s.std(ddof).unwrap();

    assert_eq!(std_val.data_type().unwrap().id(), TypeId::Float64);
    let v = std_val.value::<f64>().unwrap();
    assert!((v - 2.0).abs() < 1e-10, "expected std ≈ 2.0, got {v}");
}

#[test]
fn test_var_f64() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let ddof: u32 = 0;
    let var_val = s.var(ddof).unwrap();

    assert_eq!(var_val.data_type().unwrap().id(), TypeId::Float64);
    let v = var_val.value::<f64>().unwrap();
    assert!((v - 4.0).abs() < 1e-10, "expected var ≈ 4.0, got {v}");
}

#[test]
fn test_std_ddof_above_i32_max_uses_series_operation_contract() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();

    let err = s.std(u32::MAX).expect_err("expected ddof range validation error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::std");
    assert_eq!(error_context_value(&err, "ddof"), Some(u32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "reason"), Some("ddof_exceeds_i32_range"));
    assert_eq!(error_context_value(&err, "called"), Some("reduction::create_reduce_agg_handle(std)"));
}

#[test]
fn test_var_ddof_above_i32_max_uses_series_operation_contract() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();

    let err = s.var(u32::MAX).expect_err("expected ddof range validation error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::var");
    assert_eq!(error_context_value(&err, "ddof"), Some(u32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "reason"), Some("ddof_exceeds_i32_range"));
    assert_eq!(error_context_value(&err, "called"), Some("reduction::create_reduce_agg_handle(variance)"));
}

// ===========================================================================
// any / all
// ===========================================================================

#[test]
fn test_any_true() {
    let s = crate::common::series::<bool>("x", &[false, true, false]).unwrap();
    let result = s.any().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Bool8);
    assert!(result.value::<bool>().unwrap());
}

#[test]
fn test_any_false() {
    let s = crate::common::series::<bool>("x", &[false, false, false]).unwrap();
    let result = s.any().unwrap();

    assert!(!result.value::<bool>().unwrap());
}

#[test]
fn test_all_true() {
    let s = crate::common::series::<bool>("x", &[true, true, true]).unwrap();
    let result = s.all().unwrap();

    assert!(result.value::<bool>().unwrap());
}

#[test]
fn test_all_false() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.all().unwrap();

    assert!(!result.value::<bool>().unwrap());
}

// ===========================================================================
// product
// ===========================================================================

#[test]
fn test_product_i32() {
    let s = crate::common::series::<i32>("x", &[2, 3, 4]).unwrap();
    let result = s.product().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(result.value::<i32>().unwrap(), 24);
}

#[test]
fn test_product_f64() {
    let s = crate::common::series::<f64>("x", &[2.0, 3.0, 4.0]).unwrap();
    let result = s.product().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(result.value::<f64>().unwrap(), 24.0);
}

// ===========================================================================
// minmax
// ===========================================================================

#[test]
fn test_minmax_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let (min, max) = s.minmax().unwrap();

    assert_eq!(min.value::<i32>().unwrap(), 10);
    assert_eq!(max.value::<i32>().unwrap(), 30);
}

#[test]
fn test_minmax_f64() {
    let s = crate::common::series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let (min, max) = s.minmax().unwrap();

    assert_eq!(min.value::<f64>().unwrap(), 1.0);
    assert_eq!(max.value::<f64>().unwrap(), 3.0);
}

// ===========================================================================
// cumsum / cummin / cummax / cumprod
// ===========================================================================

#[test]
fn test_cumsum_i32() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let cs = s.cumsum().unwrap();
    assert_eq!(cs.to_vec_strict::<i32>().unwrap(), vec![10, 30, 60]);
}

#[test]
fn test_cumsum_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let cs = s.cumsum().unwrap();

    assert_eq!(cs.to_vec_strict::<f64>().unwrap(), vec![1.0, 3.0, 6.0]);
}

#[test]
fn test_cummin_i32() {
    let s = crate::common::series::<i32>("x", &[3, 1, 2]).unwrap();
    let cm = s.cummin().unwrap();
    assert_eq!(cm.to_vec_strict::<i32>().unwrap(), vec![3, 1, 1]);
}

#[test]
fn test_cummax_i32() {
    let s = crate::common::series::<i32>("x", &[1, 3, 2]).unwrap();
    let cm = s.cummax().unwrap();
    assert_eq!(cm.to_vec_strict::<i32>().unwrap(), vec![1, 3, 3]);
}

#[test]
fn test_cumprod_i32() {
    let s = crate::common::series::<i32>("x", &[2, 3, 4]).unwrap();
    let cp = s.cumprod().unwrap();
    assert_eq!(cp.to_vec_strict::<i32>().unwrap(), vec![2, 6, 24]);
}

// ===========================================================================
// is_nan / is_not_nan
// ===========================================================================

#[test]
fn test_is_nan_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, f64::NAN, 3.0]).unwrap();
    let mask = s.is_nan().unwrap();
    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_is_not_nan_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, f64::NAN, 3.0]).unwrap();
    let mask = s.is_not_nan().unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

// ===========================================================================
// abs
// ===========================================================================

#[test]
fn test_abs_i32() {
    let s = crate::common::series::<i32>("x", &[-3, -1, 0, 2]).unwrap();
    let result = s.abs().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![3, 1, 0, 2]);
}

#[test]
fn test_abs_f64() {
    let s = crate::common::series::<f64>("x", &[-3.0, -1.5, 0.0, 2.5]).unwrap();
    let result = s.abs().unwrap();

    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.0, 1.5, 0.0, 2.5]);
}

// ===========================================================================
// neg
// ===========================================================================

#[test]
fn test_neg_i32() {
    let s = crate::common::series::<i32>("x", &[1, -2, 3]).unwrap();
    let result = s.neg().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![-1, 2, -3]);
}

#[test]
fn test_neg_f64() {
    let s = crate::common::series::<f64>("x", &[1.5, -2.5]).unwrap();
    let result = s.neg().unwrap();

    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![-1.5, 2.5]);
}

// ===========================================================================
// logical_not
// ===========================================================================

#[test]
fn test_logical_not_bool() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.logical_not().unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}
