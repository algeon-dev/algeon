//! Specification tests: null propagation in Series arithmetic and comparisons.
//!
//! These tests lock down cuDF/libcudf null semantics at the Series API level:
//! - Null + value = null (null propagates through arithmetic)
//! - Null > value = null (null propagates through comparisons)
//! - Scalar ops on nullable series preserve null positions
//! - Both-null positions stay null
//! - Null positions at different indices merge correctly
//!
//! All tests require a GPU.

use cudf::Series;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a nullable i32 series: [10, null, 30]
fn series_with_null_middle() -> Series {
    Series::new_nullable::<i32>("a", &[10, 0, 30], &[true, false, true]).unwrap()
}

/// Create a non-null i32 series: [1, 2, 3]
fn series_non_null() -> Series {
    Series::new::<i32>("b", &[1, 2, 3]).unwrap()
}

/// Create a nullable i32 series with null at a different position: [100, 200, null]
fn series_with_null_last() -> Series {
    Series::new_nullable::<i32>("c", &[100, 200, 0], &[true, true, false]).unwrap()
}

// ===========================================================================
// Null propagation in Series-vs-Series arithmetic
// ===========================================================================

#[test]
fn test_add_null_plus_value_propagates_null() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = series_non_null(); // [1, 2, 3]
    let result = a.add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(11), None, Some(33)]);
}

#[test]
fn test_add_value_plus_null_propagates_null() {
    let a = series_non_null(); // [1, 2, 3]
    let b = series_with_null_middle(); // [10, null, 30]
    let result = a.add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(11), None, Some(33)]);
}

#[test]
fn test_add_both_null_at_same_position() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = Series::new_nullable::<i32>("d", &[1, 0, 3], &[true, false, true]).unwrap();
    let result = a.add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(11), None, Some(33)]);
}

#[test]
fn test_add_nulls_at_different_positions_merge() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = series_with_null_last(); // [100, 200, null]
    let result = a.add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    // position 0: both valid -> Some(110)
    // position 1: a is null -> None
    // position 2: b is null -> None
    assert_eq!(vals, vec![Some(110), None, None]);
}

#[test]
fn test_sub_with_nulls() {
    let a = series_with_null_middle();
    let b = series_non_null();
    let result = a.sub(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(9), None, Some(27)]);
}

#[test]
fn test_mul_with_nulls() {
    let a = series_with_null_middle();
    let b = series_non_null();
    let result = a.mul(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(90)]);
}

#[test]
fn test_div_with_nulls() {
    let a = Series::new_nullable::<f64>("a", &[10.0, 0.0, 30.0], &[true, false, true]).unwrap();
    let b = Series::new::<f64>("b", &[2.0, 5.0, 3.0]).unwrap();
    let result = a.div(&b).unwrap();
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(5.0), None, Some(10.0)]);
}

#[test]
fn test_rem_with_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = Series::new::<i32>("b", &[3, 3, 7]).unwrap();
    let result = a.rem(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), None, Some(2)]);
}

// ===========================================================================
// Null propagation in scalar arithmetic
// ===========================================================================

#[test]
fn test_add_scalar_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.add_scalar(5i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(15), None, Some(35)]);
}

#[test]
fn test_sub_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.sub_scalar(3i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(7), None, Some(27)]);
}

#[test]
fn test_mul_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.mul_scalar(2i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(20), None, Some(60)]);
}

#[test]
fn test_div_scalar_preserves_nulls() {
    let a = Series::new_nullable::<f64>("a", &[10.0, 0.0, 30.0], &[true, false, true]).unwrap();
    let result = a.div_scalar(2.0f64).unwrap();
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(5.0), None, Some(15.0)]);
}

#[test]
fn test_rem_scalar_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.rem_scalar(7i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(3), None, Some(2)]);
}

// ===========================================================================
// Operator overloads with nullable series
// ===========================================================================

#[test]
fn test_operator_add_with_nulls() {
    let a = series_with_null_middle();
    let b = series_non_null();
    let result = (&a + &b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(11), None, Some(33)]);
}

#[test]
fn test_operator_sub_with_nulls() {
    let a = series_with_null_middle();
    let b = series_non_null();
    let result = (&a - &b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(9), None, Some(27)]);
}

#[test]
fn test_operator_mul_with_nulls() {
    let a = series_with_null_middle();
    let b = series_non_null();
    let result = (&a * &b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(90)]);
}

// ===========================================================================
// Null propagation in Series-vs-Series comparisons
// ===========================================================================

#[test]
fn test_eq_null_propagates() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = Series::new::<i32>("b", &[10, 2, 99]).unwrap();
    let result = a.eq(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    // position 0: 10==10 -> true
    // position 1: null==2 -> null
    // position 2: 30==99 -> false
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_ne_null_propagates() {
    let a = series_with_null_middle();
    let b = Series::new::<i32>("b", &[10, 2, 99]).unwrap();
    let result = a.ne(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

#[test]
fn test_gt_null_propagates() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = Series::new::<i32>("b", &[5, 2, 50]).unwrap();
    let result = a.gt(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_lt_null_propagates() {
    let a = series_with_null_middle();
    let b = Series::new::<i32>("b", &[5, 2, 50]).unwrap();
    let result = a.lt(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

#[test]
fn test_ge_null_propagates() {
    let a = series_with_null_middle();
    let b = Series::new::<i32>("b", &[10, 2, 30]).unwrap();
    let result = a.ge(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

#[test]
fn test_le_null_propagates() {
    let a = series_with_null_middle();
    let b = Series::new::<i32>("b", &[10, 2, 30]).unwrap();
    let result = a.le(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

#[test]
fn test_eq_null_vs_null_is_null_not_true() {
    // Two nulls compared: null == null should be null, not true.
    let a = series_with_null_middle(); // [10, null, 30]
    let b = Series::new_nullable::<i32>("d", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = a.eq(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

// ===========================================================================
// Null propagation in scalar comparisons
// ===========================================================================

#[test]
fn test_gt_scalar_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.gt_scalar(15i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

#[test]
fn test_lt_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.lt_scalar(15i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_eq_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.eq_scalar(10i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_ne_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.ne_scalar(10i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

#[test]
fn test_ge_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.ge_scalar(10i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

#[test]
fn test_le_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.le_scalar(30i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

// ===========================================================================
// Null propagation in string comparisons
// ===========================================================================

#[test]
fn test_eq_scalar_str_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.eq_scalar_str("hello").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_gt_scalar_str_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["banana", "x", "apple"], &[true, false, true]).unwrap();
    let result = s.gt_scalar_str("avocado").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

// ===========================================================================
// Unary ops on nullable series
// ===========================================================================

#[test]
fn test_neg_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = (-&a).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(-10), None, Some(-30)]);
}

#[test]
fn test_logical_not_preserves_nulls() {
    let a = Series::new_nullable::<bool>("f", &[true, false, true], &[true, false, true]).unwrap();
    let result = (!&a).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(false)]);
}

#[test]
fn test_abs_preserves_nulls() {
    let a = Series::new_nullable::<i32>("a", &[-10, 0, 30], &[true, false, true]).unwrap();
    let result = a.abs().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(30)]);
}

#[test]
fn test_is_null_on_nullable() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.is_null().unwrap();
    let vals = result.to_vec::<bool>().unwrap();
    assert_eq!(vals, vec![false, true, false]);
}

#[test]
fn test_is_not_null_on_nullable() {
    let a = series_with_null_middle();
    let result = a.is_not_null().unwrap();
    let vals = result.to_vec::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

// ===========================================================================
// All-null series in arithmetic
// ===========================================================================

#[test]
fn test_add_all_null_series() {
    let a = Series::new_nullable::<i32>("a", &[0, 0, 0], &[false, false, false]).unwrap();
    let b = series_non_null();
    let result = a.add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, None]);
}

#[test]
fn test_eq_all_null_series() {
    let a = Series::new_nullable::<i32>("a", &[0, 0, 0], &[false, false, false]).unwrap();
    let b = series_non_null();
    let result = a.eq(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![None, None, None]);
}

// ===========================================================================
// NaN vs null distinction in float operations
// ===========================================================================

#[test]
fn test_is_nan_does_not_flag_null_as_nan() {
    let a = Series::new_nullable::<f64>("a", &[1.0, f64::NAN, 0.0], &[true, true, false]).unwrap();
    let result = a.is_nan().unwrap();
    // libcudf is_nan returns a non-nullable bool column: null slots → false.
    let vals = result.to_vec::<bool>().unwrap();
    assert!(!vals[0]); // 1.0 is not NaN
    assert!(vals[1]); // NaN is NaN
    assert!(!vals[2]); // null → false (not flagged as NaN)
}

#[test]
fn test_is_not_nan_does_not_flag_null_as_not_nan() {
    let a = Series::new_nullable::<f64>("a", &[1.0, f64::NAN, 0.0], &[true, true, false]).unwrap();
    let result = a.is_not_nan().unwrap();
    // libcudf is_not_nan returns a non-nullable bool column: null slots → true.
    let vals = result.to_vec::<bool>().unwrap();
    assert!(vals[0]); // 1.0 is not NaN
    assert!(!vals[1]); // NaN is NaN
    assert!(vals[2]); // null → true (null is not NaN)
}
