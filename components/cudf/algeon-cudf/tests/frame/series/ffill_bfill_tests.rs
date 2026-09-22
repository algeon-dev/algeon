//! Tests for Series::ffill() and Series::bfill() convenience methods.
//!
//! All tests require a GPU.

use crate::common::{nullable_series, run_named_cases, series};

// ===========================================================================
// ffill (forward fill)
// ===========================================================================

fn test_series_ffill_basic() {
    let s = nullable_series::<i32>("x", &[1, 0, 0, 4], &[true, false, false, true]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.null_count(), 0);
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(1), Some(1), Some(1), Some(4)]);
}

fn test_series_ffill_display_omits_input_name() {
    let s = nullable_series::<i32>("my_col", &[1, 0], &[true, false]).unwrap();
    let result = s.ffill().unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_col"), "ffill output should not expose schema names: {}", output);
}

fn test_series_ffill_leading_nulls_remain() {
    let s = nullable_series::<i32>("x", &[0, 0, 3], &[false, false, true]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.null_count(), 2);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, Some(3)]);
}

fn test_series_ffill_no_nulls() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_series_ffill_all_null() {
    let s = nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.null_count(), 3);
}

fn test_series_ffill_empty() {
    let s = series::<i32>("x", &[]).unwrap();
    let result = s.ffill().unwrap();
    assert!(result.is_empty());
}

fn test_series_ffill_float64() {
    let s = nullable_series::<f64>("x", &[1.5, 0.0, 3.5], &[true, false, true]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.to_optional_vec::<f64>().unwrap(), vec![Some(1.5), Some(1.5), Some(3.5)]);
}

// ===========================================================================
// bfill (backward fill)
// ===========================================================================

fn test_series_bfill_basic() {
    let s = nullable_series::<i32>("x", &[1, 0, 0, 4], &[true, false, false, true]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 0);
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(1), Some(4), Some(4), Some(4)]);
}

fn test_series_bfill_display_omits_input_name() {
    let s = nullable_series::<i32>("my_col", &[0, 1], &[false, true]).unwrap();
    let result = s.bfill().unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_col"), "bfill output should not expose schema names: {}", output);
}

fn test_series_bfill_trailing_nulls_remain() {
    let s = nullable_series::<i32>("x", &[1, 0, 0], &[true, false, false]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 2);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), None, None]);
}

fn test_series_bfill_no_nulls() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_series_bfill_all_null() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 2);
}

fn test_series_bfill_empty() {
    let s = series::<i32>("x", &[]).unwrap();
    let result = s.bfill().unwrap();
    assert!(result.is_empty());
}

fn test_series_bfill_single_element() {
    // Single non-null
    let s = series::<i32>("x", &[42]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42]);

    // Single null
    let s = nullable_series::<i32>("x", &[0], &[false]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 1);
}

// ===========================================================================
// ffill then bfill (common pattern: fill interior nulls)
// ===========================================================================

fn test_series_ffill_then_bfill_fills_all_interior() {
    // [null, 2, null, 4, null] → ffill → [null, 2, 2, 4, 4] → bfill → [2, 2, 2, 4, 4]
    let s = nullable_series::<i32>("x", &[0, 2, 0, 4, 0], &[false, true, false, true, false]).unwrap();
    let after_ffill = s.ffill().unwrap();
    let after_both = after_ffill.bfill().unwrap();
    assert_eq!(after_both.null_count(), 0);
    assert_eq!(after_both.to_optional_vec::<i32>().unwrap(), vec![Some(2), Some(2), Some(2), Some(4), Some(4)]);
}

#[test]
fn ffill_bfill_tests_cases() {
    run_named_cases(&[
        ("test_series_ffill_basic", test_series_ffill_basic),
        ("test_series_ffill_display_omits_input_name", test_series_ffill_display_omits_input_name),
        ("test_series_ffill_leading_nulls_remain", test_series_ffill_leading_nulls_remain),
        ("test_series_ffill_no_nulls", test_series_ffill_no_nulls),
        ("test_series_ffill_all_null", test_series_ffill_all_null),
        ("test_series_ffill_empty", test_series_ffill_empty),
        ("test_series_ffill_float64", test_series_ffill_float64),
        ("test_series_bfill_basic", test_series_bfill_basic),
        ("test_series_bfill_display_omits_input_name", test_series_bfill_display_omits_input_name),
        ("test_series_bfill_trailing_nulls_remain", test_series_bfill_trailing_nulls_remain),
        ("test_series_bfill_no_nulls", test_series_bfill_no_nulls),
        ("test_series_bfill_all_null", test_series_bfill_all_null),
        ("test_series_bfill_empty", test_series_bfill_empty),
        ("test_series_bfill_single_element", test_series_bfill_single_element),
        ("test_series_ffill_then_bfill_fills_all_interior", test_series_ffill_then_bfill_fills_all_interior),
    ]);
}
