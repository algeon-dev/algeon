//! Tests for CU-4: forward fill and backward fill.

use cudf::Exec;

use crate::common::{nullable_series, run_named_cases, series};

// -- Forward fill (ffill) tests -----------------------------------------------

fn test_replace_nulls_policy_ffill_basic() {
    // [1, null, null, 4] → [1, 1, 1, 4]
    let s = nullable_series::<i32>("x", &[1, 0, 0, 4], &[true, false, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).ffill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 1, 1, 4]);
    assert_eq!(result.null_count(), 0);
}

fn test_replace_nulls_policy_ffill_leading_nulls_remain() {
    // [null, null, 3, null, 5] → [null, null, 3, 3, 5]
    let s = nullable_series::<i32>("x", &[0, 0, 3, 0, 5], &[false, false, true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).ffill().unwrap();
    assert_eq!(result.null_count(), 2);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, Some(3), Some(3), Some(5)]);
}

fn test_replace_nulls_policy_ffill_no_nulls() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.on(Exec::RETAINED).ffill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_replace_nulls_policy_ffill_all_null() {
    let s = nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let result = s.on(Exec::RETAINED).ffill().unwrap();
    assert_eq!(result.null_count(), 3);
}

fn test_replace_nulls_policy_ffill_empty() {
    let s = series::<i32>("x", &[]).unwrap();
    let result = s.on(Exec::RETAINED).ffill().unwrap();
    assert_eq!(result.len(), 0);
}

// -- Backward fill (bfill) tests ----------------------------------------------

fn test_replace_nulls_policy_bfill_basic() {
    // [1, null, null, 4] → [1, 4, 4, 4]
    let s = nullable_series::<i32>("x", &[1, 0, 0, 4], &[true, false, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).bfill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 4, 4, 4]);
    assert_eq!(result.null_count(), 0);
}

fn test_replace_nulls_policy_bfill_trailing_nulls_remain() {
    // [1, null, 3, null, null] → [1, 3, 3, null, null]
    let s = nullable_series::<i32>("x", &[1, 0, 3, 0, 0], &[true, false, true, false, false]).unwrap();
    let result = s.on(Exec::RETAINED).bfill().unwrap();
    assert_eq!(result.null_count(), 2);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), Some(3), Some(3), None, None]);
}

fn test_replace_nulls_policy_bfill_all_null() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).bfill().unwrap();
    assert_eq!(result.null_count(), 2);
}

fn test_replace_nulls_policy_float64() {
    // [1.0, null, 3.0] → ffill → [1.0, 1.0, 3.0]
    let s = nullable_series::<f64>("x", &[1.0, 0.0, 3.0], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).ffill().unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.0, 1.0, 3.0]);
}

#[test]
fn replace_tests_cases() {
    run_named_cases(&[
        ("test_replace_nulls_policy_ffill_basic", test_replace_nulls_policy_ffill_basic),
        ("test_replace_nulls_policy_ffill_leading_nulls_remain", test_replace_nulls_policy_ffill_leading_nulls_remain),
        ("test_replace_nulls_policy_ffill_no_nulls", test_replace_nulls_policy_ffill_no_nulls),
        ("test_replace_nulls_policy_ffill_all_null", test_replace_nulls_policy_ffill_all_null),
        ("test_replace_nulls_policy_ffill_empty", test_replace_nulls_policy_ffill_empty),
        ("test_replace_nulls_policy_bfill_basic", test_replace_nulls_policy_bfill_basic),
        (
            "test_replace_nulls_policy_bfill_trailing_nulls_remain",
            test_replace_nulls_policy_bfill_trailing_nulls_remain,
        ),
        ("test_replace_nulls_policy_bfill_all_null", test_replace_nulls_policy_bfill_all_null),
        ("test_replace_nulls_policy_float64", test_replace_nulls_policy_float64),
    ]);
}
