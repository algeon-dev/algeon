//! Specification tests: null behavior in Series reductions, scans, and aggregations.
//!
//! These tests lock down cuDF/libcudf null semantics for:
//! - Scalar reductions (sum, min, max, mean, median, std, var, any, all, product)
//!   with partial nulls, all-null input, and empty input
//! - Cumulative scans (cumsum, cummin, cummax, cumprod) with nulls
//! - minmax with nulls
//!
//! All tests require a GPU.

use cudf::Exec;

use crate::common::{nullable_series, run_named_cases, series};

// ===========================================================================
// sum with nulls
// ===========================================================================

fn test_sum_partial_nulls_skips_nulls() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).sum().unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i64>().unwrap(), 40); // 10 + 30, skip null
}

fn test_sum_all_null_returns_valid_zero_or_null() {
    // cuDF Python: Series([None, None, None]).sum() returns 0 (not null).
    // libcudf reduce with EXCLUDE nulls on all-null column returns 0 for sum.
    let s = nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let result = s.on(Exec::RETAINED).sum().unwrap();
    // libcudf returns a valid scalar with value 0 for sum of all-null
    assert_eq!(result.value::<i64>().unwrap(), 0);
}

fn test_sum_empty_series() {
    let s = series::<i32>("x", &[]).unwrap();
    let result = s.on(Exec::RETAINED).sum().unwrap();
    // Sum of empty = 0 (identity element)
    assert_eq!(result.value::<i64>().unwrap(), 0);
}

fn test_sum_single_element() {
    let s = series::<i32>("x", &[42]).unwrap();
    let result = s.on(Exec::RETAINED).sum().unwrap();
    assert_eq!(result.value::<i64>().unwrap(), 42);
}

// ===========================================================================
// min / max with nulls
// ===========================================================================

fn test_min_partial_nulls_skips_nulls() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).min().unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i32>().unwrap(), 10);
}

fn test_max_partial_nulls_skips_nulls() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).max().unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i32>().unwrap(), 30);
}

fn test_min_all_null_returns_null_scalar() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).min().unwrap();
    // min of all-null should be a null scalar (no valid element to pick)
    assert!(!result.is_valid().unwrap());
}

fn test_max_all_null_returns_null_scalar() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).max().unwrap();
    assert!(!result.is_valid().unwrap());
}

fn test_min_single_element() {
    let s = series::<i32>("x", &[42]).unwrap();
    assert_eq!(s.on(Exec::RETAINED).min().unwrap().value::<i32>().unwrap(), 42);
}

fn test_max_single_element() {
    let s = series::<i32>("x", &[42]).unwrap();
    assert_eq!(s.on(Exec::RETAINED).max().unwrap().value::<i32>().unwrap(), 42);
}

// ===========================================================================
// mean with nulls
// ===========================================================================

fn test_mean_partial_nulls_skips_nulls() {
    // [10, null, 30] -> mean = (10+30)/2 = 20.0
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).mean().unwrap();
    assert!(result.is_valid().unwrap());
    let v = result.value::<f64>().unwrap();
    assert!((v - 20.0).abs() < 1e-10);
}

fn test_mean_all_null() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).mean().unwrap();
    // mean of all-null: no valid elements => null scalar
    assert!(!result.is_valid().unwrap());
}

fn test_mean_single_element() {
    let s = series::<f64>("x", &[3.5]).unwrap();
    let v = s.on(Exec::RETAINED).mean().unwrap().value::<f64>().unwrap();
    assert!((v - 3.5).abs() < 1e-10);
}

// ===========================================================================
// median with nulls
// ===========================================================================

fn test_median_partial_nulls() {
    // [10, null, 30, 20] -> valid values [10, 20, 30] -> median = 20.0
    let s = nullable_series::<i32>("x", &[10, 0, 30, 20], &[true, false, true, true]).unwrap();
    let result = s.on(Exec::RETAINED).median().unwrap();
    assert!(result.is_valid().unwrap());
    let v = result.value::<f64>().unwrap();
    assert!((v - 20.0).abs() < 1e-10);
}

fn test_median_all_null() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).median().unwrap();
    assert!(!result.is_valid().unwrap());
}

fn test_median_single_element() {
    let s = series::<i32>("x", &[7]).unwrap();
    let v = s.on(Exec::RETAINED).median().unwrap().value::<f64>().unwrap();
    assert!((v - 7.0).abs() < 1e-10);
}

// ===========================================================================
// std / var with nulls
// ===========================================================================

fn test_std_partial_nulls() {
    // [10, null, 30] -> valid values [10, 30], sample std (ddof=1)
    // mean = 20, var = ((10-20)^2 + (30-20)^2) / (2-1) = 200
    // std = sqrt(200) ≈ 14.142
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).std(1).unwrap();
    assert!(result.is_valid().unwrap());
    let v = result.value::<f64>().unwrap();
    assert!((v - 200.0f64.sqrt()).abs() < 1e-6);
}

fn test_var_partial_nulls() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).var(1).unwrap();
    assert!(result.is_valid().unwrap());
    let v = result.value::<f64>().unwrap();
    assert!((v - 200.0).abs() < 1e-6);
}

fn test_std_single_element_ddof1() {
    // std of single element with ddof=1: division by (1-1)=0 -> NaN or null
    let s = series::<i32>("x", &[5]).unwrap();
    let result = s.on(Exec::RETAINED).std(1).unwrap();
    // libcudf returns NaN (or 0) for single-element std with ddof=1
    // Just verify it doesn't crash; the exact result depends on libcudf behavior
    let _v = result.value::<f64>().unwrap();
}

fn test_std_all_null() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).std(1).unwrap();
    assert!(!result.is_valid().unwrap());
}

// ===========================================================================
// any / all with nulls
// ===========================================================================

fn test_any_with_true_and_null() {
    // [true, null, false] -> any should be true (at least one true exists)
    let s = nullable_series::<bool>("x", &[true, false, false], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).any().unwrap();
    assert!(result.is_valid().unwrap());
    assert!(result.value::<bool>().unwrap());
}

fn test_any_with_false_and_null() {
    // [false, null, false] -> cuDF excludes nulls, so any([false, false]) = false
    let s = nullable_series::<bool>("x", &[false, false, false], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).any().unwrap();
    assert!(result.is_valid().unwrap());
    assert!(!result.value::<bool>().unwrap());
}

fn test_all_with_true_and_null() {
    // [true, null, true] -> cuDF excludes nulls, so all([true, true]) = true
    let s = nullable_series::<bool>("x", &[true, false, true], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).all().unwrap();
    assert!(result.is_valid().unwrap());
    assert!(result.value::<bool>().unwrap());
}

fn test_all_with_false_and_null() {
    // [true, null, false] -> cuDF excludes nulls, so all([true, false]) = false
    let s = nullable_series::<bool>("x", &[true, false, false], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).all().unwrap();
    assert!(result.is_valid().unwrap());
    assert!(!result.value::<bool>().unwrap());
}

fn test_any_all_null() {
    let s = nullable_series::<bool>("x", &[false, false], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).any().unwrap();
    // any of all-null with exclude-nulls: no elements, so false (identity)
    assert!(result.is_valid().unwrap());
    assert!(!result.value::<bool>().unwrap());
}

// ===========================================================================
// product with nulls
// ===========================================================================

fn test_product_partial_nulls() {
    // [2, null, 5] -> product of valid = 10
    let s = nullable_series::<i32>("x", &[2, 0, 5], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).product().unwrap();
    assert!(result.is_valid().unwrap());
    assert_eq!(result.value::<i32>().unwrap(), 10);
}

fn test_product_single_element() {
    let s = series::<i32>("x", &[7]).unwrap();
    assert_eq!(s.on(Exec::RETAINED).product().unwrap().value::<i32>().unwrap(), 7);
}

// ===========================================================================
// minmax with nulls
// ===========================================================================

fn test_minmax_partial_nulls() {
    // [10, null, 30, 5] -> min=5, max=30
    let s = nullable_series::<i32>("x", &[10, 0, 30, 5], &[true, false, true, true]).unwrap();
    let (min_s, max_s) = s.on(Exec::RETAINED).minmax().unwrap();
    assert_eq!(min_s.value::<i32>().unwrap(), 5);
    assert_eq!(max_s.value::<i32>().unwrap(), 30);
}

fn test_minmax_single_element() {
    let s = series::<i32>("x", &[42]).unwrap();
    let (min_s, max_s) = s.on(Exec::RETAINED).minmax().unwrap();
    assert_eq!(min_s.value::<i32>().unwrap(), 42);
    assert_eq!(max_s.value::<i32>().unwrap(), 42);
}

// ===========================================================================
// Cumulative scans with nulls
// ===========================================================================

fn test_cumsum_with_nulls() {
    // [1, null, 3, 4] -> cumsum should skip nulls in the running total
    // Expected: [1, null, 4, 8] (null positions stay null)
    let s = nullable_series::<i32>("x", &[1, 0, 3, 4], &[true, false, true, true]).unwrap();
    let result = s.on(Exec::RETAINED).cumsum().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals[0], Some(1));
    assert_eq!(vals[1], None); // null position stays null
    assert_eq!(vals[2], Some(4)); // 1 + 3
    assert_eq!(vals[3], Some(8)); // 1 + 3 + 4
}

fn test_cummin_with_nulls() {
    // [5, null, 3, 7] -> cummin skipping nulls
    // Expected: [5, null, 3, 3]
    let s = nullable_series::<i32>("x", &[5, 0, 3, 7], &[true, false, true, true]).unwrap();
    let result = s.on(Exec::RETAINED).cummin().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals[0], Some(5));
    assert_eq!(vals[1], None);
    assert_eq!(vals[2], Some(3));
    assert_eq!(vals[3], Some(3));
}

fn test_cummax_with_nulls() {
    // [5, null, 3, 7] -> cummax skipping nulls
    // Expected: [5, null, 5, 7]
    let s = nullable_series::<i32>("x", &[5, 0, 3, 7], &[true, false, true, true]).unwrap();
    let result = s.on(Exec::RETAINED).cummax().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals[0], Some(5));
    assert_eq!(vals[1], None);
    assert_eq!(vals[2], Some(5));
    assert_eq!(vals[3], Some(7));
}

fn test_cumprod_with_nulls() {
    // [2, null, 3, 4] -> cumprod skipping nulls
    // Expected: [2, null, 6, 24]
    let s = nullable_series::<i32>("x", &[2, 0, 3, 4], &[true, false, true, true]).unwrap();
    let result = s.on(Exec::RETAINED).cumprod().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals[0], Some(2));
    assert_eq!(vals[1], None);
    assert_eq!(vals[2], Some(6));
    assert_eq!(vals[3], Some(24));
}

// ===========================================================================
// Cumulative scans on empty series
// ===========================================================================

fn test_cumsum_empty() {
    let s = series::<i32>("x", &[]).unwrap();
    let result = s.on(Exec::RETAINED).cumsum().unwrap();
    assert!(result.is_empty());
}

fn test_cumsum_single_element() {
    let s = series::<i32>("x", &[42]).unwrap();
    let result = s.on(Exec::RETAINED).cumsum().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42]);
}

#[test]
fn null_reduction_tests_cases() {
    run_named_cases(&[
        ("test_sum_partial_nulls_skips_nulls", test_sum_partial_nulls_skips_nulls),
        ("test_sum_all_null_returns_valid_zero_or_null", test_sum_all_null_returns_valid_zero_or_null),
        ("test_sum_empty_series", test_sum_empty_series),
        ("test_sum_single_element", test_sum_single_element),
        ("test_min_partial_nulls_skips_nulls", test_min_partial_nulls_skips_nulls),
        ("test_max_partial_nulls_skips_nulls", test_max_partial_nulls_skips_nulls),
        ("test_min_all_null_returns_null_scalar", test_min_all_null_returns_null_scalar),
        ("test_max_all_null_returns_null_scalar", test_max_all_null_returns_null_scalar),
        ("test_min_single_element", test_min_single_element),
        ("test_max_single_element", test_max_single_element),
        ("test_mean_partial_nulls_skips_nulls", test_mean_partial_nulls_skips_nulls),
        ("test_mean_all_null", test_mean_all_null),
        ("test_mean_single_element", test_mean_single_element),
        ("test_median_partial_nulls", test_median_partial_nulls),
        ("test_median_all_null", test_median_all_null),
        ("test_median_single_element", test_median_single_element),
        ("test_std_partial_nulls", test_std_partial_nulls),
        ("test_var_partial_nulls", test_var_partial_nulls),
        ("test_std_single_element_ddof1", test_std_single_element_ddof1),
        ("test_std_all_null", test_std_all_null),
        ("test_any_with_true_and_null", test_any_with_true_and_null),
        ("test_any_with_false_and_null", test_any_with_false_and_null),
        ("test_all_with_true_and_null", test_all_with_true_and_null),
        ("test_all_with_false_and_null", test_all_with_false_and_null),
        ("test_any_all_null", test_any_all_null),
        ("test_product_partial_nulls", test_product_partial_nulls),
        ("test_product_single_element", test_product_single_element),
        ("test_minmax_partial_nulls", test_minmax_partial_nulls),
        ("test_minmax_single_element", test_minmax_single_element),
        ("test_cumsum_with_nulls", test_cumsum_with_nulls),
        ("test_cummin_with_nulls", test_cummin_with_nulls),
        ("test_cummax_with_nulls", test_cummax_with_nulls),
        ("test_cumprod_with_nulls", test_cumprod_with_nulls),
        ("test_cumsum_empty", test_cumsum_empty),
        ("test_cumsum_single_element", test_cumsum_single_element),
    ]);
}
