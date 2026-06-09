//! Tests for Series::ffill() and Series::bfill() convenience methods.
//!
//! All tests require a GPU.

// ===========================================================================
// ffill (forward fill)
// ===========================================================================

#[test]
fn test_series_ffill_basic() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 0, 4], &[true, false, false, true]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.null_count(), 0);
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(1), Some(1), Some(1), Some(4)]);
}

#[test]
fn test_series_ffill_preserves_name() {
    let s = crate::common::nullable_series::<i32>("my_col", &[1, 0], &[true, false]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.name(), Some("my_col"));
}

#[test]
fn test_series_ffill_leading_nulls_remain() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0, 3], &[false, false, true]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.null_count(), 2);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, Some(3)]);
}

#[test]
fn test_series_ffill_no_nulls() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_series_ffill_all_null() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.null_count(), 3);
}

#[test]
fn test_series_ffill_empty() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.ffill().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_series_ffill_float64() {
    let s = crate::common::nullable_series::<f64>("x", &[1.5, 0.0, 3.5], &[true, false, true]).unwrap();
    let result = s.ffill().unwrap();
    assert_eq!(result.to_optional_vec::<f64>().unwrap(), vec![Some(1.5), Some(1.5), Some(3.5)]);
}

// ===========================================================================
// bfill (backward fill)
// ===========================================================================

#[test]
fn test_series_bfill_basic() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 0, 4], &[true, false, false, true]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 0);
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(1), Some(4), Some(4), Some(4)]);
}

#[test]
fn test_series_bfill_preserves_name() {
    let s = crate::common::nullable_series::<i32>("my_col", &[0, 1], &[false, true]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.name(), Some("my_col"));
}

#[test]
fn test_series_bfill_trailing_nulls_remain() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 0], &[true, false, false]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 2);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), None, None]);
}

#[test]
fn test_series_bfill_no_nulls() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_series_bfill_all_null() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 2);
}

#[test]
fn test_series_bfill_empty() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.bfill().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_series_bfill_single_element() {
    // Single non-null
    let s = crate::common::series::<i32>("x", &[42]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42]);

    // Single null
    let s = crate::common::nullable_series::<i32>("x", &[0], &[false]).unwrap();
    let result = s.bfill().unwrap();
    assert_eq!(result.null_count(), 1);
}

// ===========================================================================
// ffill then bfill (common pattern: fill interior nulls)
// ===========================================================================

#[test]
fn test_series_ffill_then_bfill_fills_all_interior() {
    // [null, 2, null, 4, null] → ffill → [null, 2, 2, 4, 4] → bfill → [2, 2, 2, 4, 4]
    let s = crate::common::nullable_series::<i32>("x", &[0, 2, 0, 4, 0], &[false, true, false, true, false]).unwrap();
    let after_ffill = s.ffill().unwrap();
    let after_both = after_ffill.bfill().unwrap();
    assert_eq!(after_both.null_count(), 0);
    assert_eq!(after_both.to_optional_vec::<i32>().unwrap(), vec![Some(2), Some(2), Some(2), Some(4), Some(4)]);
}
