//! Specification tests for null propagation in Series arithmetic and comparisons.
//!
//! These tests lock down cuDF/libcudf null semantics at the Series API level:
//! - Null + value = null (null propagates through arithmetic)
//! - Null > value = null (null propagates through comparisons)
//! - Scalar ops on nullable series preserve null positions
//! - Both-null positions stay null
//! - Null positions at different indices merge correctly

use cudf::{Exec, Series};

use crate::common::{nullable_series, series};

/// Create a nullable i32 series: [10, null, 30]
fn series_with_null_middle() -> Series {
    nullable_series::<i32>("a", &[10, 0, 30], &[true, false, true]).unwrap()
}

/// Create a non-null i32 series: [1, 2, 3]
fn series_non_null() -> Series {
    series::<i32>("b", &[1, 2, 3]).unwrap()
}

/// Create a nullable i32 series with null at a different position: [100, 200, null]
fn series_with_null_last() -> Series {
    nullable_series::<i32>("c", &[100, 200, 0], &[true, true, false]).unwrap()
}

fn test_add_null_plus_value_propagates_null() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = series_non_null(); // [1, 2, 3]
    let result = a.on(Exec::RETAINED).add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(11), None, Some(33)]);
}

fn test_add_value_plus_null_propagates_null() {
    let a = series_non_null(); // [1, 2, 3]
    let b = series_with_null_middle(); // [10, null, 30]
    let result = a.on(Exec::RETAINED).add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(11), None, Some(33)]);
}

fn test_add_both_null_at_same_position() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = nullable_series::<i32>("d", &[1, 0, 3], &[true, false, true]).unwrap();
    let result = a.on(Exec::RETAINED).add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(11), None, Some(33)]);
}

fn test_add_nulls_at_different_positions_merge() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = series_with_null_last(); // [100, 200, null]
    let result = a.on(Exec::RETAINED).add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    // position 0: both valid -> Some(110)
    // position 1: a is null -> None
    // position 2: b is null -> None
    assert_eq!(vals, vec![Some(110), None, None]);
}

fn test_sub_with_nulls() {
    let a = series_with_null_middle();
    let b = series_non_null();
    let result = a.on(Exec::RETAINED).sub(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(9), None, Some(27)]);
}

fn test_mul_with_nulls() {
    let a = series_with_null_middle();
    let b = series_non_null();
    let result = a.on(Exec::RETAINED).mul(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(90)]);
}

fn test_div_with_nulls() {
    let a = nullable_series::<f64>("a", &[10.0, 0.0, 30.0], &[true, false, true]).unwrap();
    let b = series::<f64>("b", &[2.0, 5.0, 3.0]).unwrap();
    let result = a.on(Exec::RETAINED).div(&b).unwrap();
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(5.0), None, Some(10.0)]);
}

fn test_rem_with_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = series::<i32>("b", &[3, 3, 7]).unwrap();
    let result = a.on(Exec::RETAINED).rem(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), None, Some(2)]);
}

fn test_add_scalar_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.on(Exec::RETAINED).add_scalar(5i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(15), None, Some(35)]);
}

fn test_sub_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).sub_scalar(3i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(7), None, Some(27)]);
}

fn test_mul_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).mul_scalar(2i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(20), None, Some(60)]);
}

fn test_div_scalar_preserves_nulls() {
    let a = nullable_series::<f64>("a", &[10.0, 0.0, 30.0], &[true, false, true]).unwrap();
    let result = a.on(Exec::RETAINED).div_scalar(2.0f64).unwrap();
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(5.0), None, Some(15.0)]);
}

fn test_rem_scalar_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.on(Exec::RETAINED).rem_scalar(7i32).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(3), None, Some(2)]);
}

fn test_eq_null_propagates() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = series::<i32>("b", &[10, 2, 99]).unwrap();
    let result = a.on(Exec::RETAINED).eq(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    // position 0: 10==10 -> true
    // position 1: null==2 -> null
    // position 2: 30==99 -> false
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_ne_null_propagates() {
    let a = series_with_null_middle();
    let b = series::<i32>("b", &[10, 2, 99]).unwrap();
    let result = a.on(Exec::RETAINED).ne(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

fn test_gt_null_propagates() {
    let a = series_with_null_middle(); // [10, null, 30]
    let b = series::<i32>("b", &[5, 2, 50]).unwrap();
    let result = a.on(Exec::RETAINED).gt(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_lt_null_propagates() {
    let a = series_with_null_middle();
    let b = series::<i32>("b", &[5, 2, 50]).unwrap();
    let result = a.on(Exec::RETAINED).lt(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

fn test_ge_null_propagates() {
    let a = series_with_null_middle();
    let b = series::<i32>("b", &[10, 2, 30]).unwrap();
    let result = a.on(Exec::RETAINED).ge(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

fn test_le_null_propagates() {
    let a = series_with_null_middle();
    let b = series::<i32>("b", &[10, 2, 30]).unwrap();
    let result = a.on(Exec::RETAINED).le(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

fn test_eq_null_vs_null_is_null_not_true() {
    // Two nulls compared: null == null evaluates to null, so the result is not true.
    let a = series_with_null_middle(); // [10, null, 30]
    let b = nullable_series::<i32>("d", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = a.on(Exec::RETAINED).eq(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

fn test_gt_scalar_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.on(Exec::RETAINED).gt_scalar(15i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

fn test_lt_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).lt_scalar(15i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_eq_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).eq_scalar(10i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_ne_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).ne_scalar(10i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

fn test_ge_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).ge_scalar(10i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

fn test_le_scalar_preserves_nulls() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).le_scalar(30i32).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

fn test_eq_scalar_string_preserves_nulls() {
    let s = nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).eq_scalar("hello").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_gt_scalar_string_preserves_nulls() {
    let s = nullable_series("s", &["banana", "x", "apple"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).gt_scalar("avocado").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_neg_preserves_nulls() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.on(Exec::RETAINED).neg().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(-10), None, Some(-30)]);
}

fn test_logical_not_preserves_nulls() {
    let a = nullable_series::<bool>("f", &[true, false, true], &[true, false, true]).unwrap();
    let result = a.on(Exec::RETAINED).logical_not().unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(false)]);
}

fn test_logical_and_preserves_nulls() {
    let a = nullable_series::<bool>("lhs", &[true, false, true], &[true, false, true]).unwrap();
    let b = series::<bool>("rhs", &[true, true, false]).unwrap();

    let result = a.on(Exec::RETAINED).logical_and(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_logical_or_preserves_nulls() {
    let a = nullable_series::<bool>("lhs", &[true, false, true], &[true, false, true]).unwrap();
    let b = series::<bool>("rhs", &[false, true, false]).unwrap();

    let result = a.on(Exec::RETAINED).logical_or(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

fn test_abs_preserves_nulls() {
    let a = nullable_series::<i32>("a", &[-10, 0, 30], &[true, false, true]).unwrap();
    let result = a.on(Exec::RETAINED).abs().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(30)]);
}

fn test_is_null_on_nullable() {
    let a = series_with_null_middle(); // [10, null, 30]
    let result = a.on(Exec::RETAINED).is_null().unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![false, true, false]);
}

fn test_is_not_null_on_nullable() {
    let a = series_with_null_middle();
    let result = a.on(Exec::RETAINED).is_not_null().unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

fn test_add_all_null_series() {
    let a = nullable_series::<i32>("a", &[0, 0, 0], &[false, false, false]).unwrap();
    let b = series_non_null();
    let result = a.on(Exec::RETAINED).add(&b).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, None]);
}

fn test_eq_all_null_series() {
    let a = nullable_series::<i32>("a", &[0, 0, 0], &[false, false, false]).unwrap();
    let b = series_non_null();
    let result = a.on(Exec::RETAINED).eq(&b).unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![None, None, None]);
}

fn test_is_nan_does_not_flag_null_as_nan() {
    let a = nullable_series::<f64>("a", &[1.0, f64::NAN, 0.0], &[true, true, false]).unwrap();
    let result = a.on(Exec::RETAINED).is_nan().unwrap();
    // libcudf is_nan returns a non-nullable bool column: null slots → false.
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert!(!vals[0]); // 1.0 is not NaN
    assert!(vals[1]); // NaN is NaN
    assert!(!vals[2]); // null → false (not flagged as NaN)
}

fn test_is_not_nan_does_not_flag_null_as_not_nan() {
    let a = nullable_series::<f64>("a", &[1.0, f64::NAN, 0.0], &[true, true, false]).unwrap();
    let result = a.on(Exec::RETAINED).is_not_nan().unwrap();
    // libcudf is_not_nan returns a non-nullable bool column: null slots → true.
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert!(vals[0]); // 1.0 is not NaN
    assert!(!vals[1]); // NaN is NaN
    assert!(vals[2]); // null → true (null is not NaN)
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("null_arithmetic_tests::test_add_null_plus_value_propagates_null", test_add_null_plus_value_propagates_null),
    ("null_arithmetic_tests::test_add_value_plus_null_propagates_null", test_add_value_plus_null_propagates_null),
    ("null_arithmetic_tests::test_add_both_null_at_same_position", test_add_both_null_at_same_position),
    ("null_arithmetic_tests::test_add_nulls_at_different_positions_merge", test_add_nulls_at_different_positions_merge),
    ("null_arithmetic_tests::test_sub_with_nulls", test_sub_with_nulls),
    ("null_arithmetic_tests::test_mul_with_nulls", test_mul_with_nulls),
    ("null_arithmetic_tests::test_div_with_nulls", test_div_with_nulls),
    ("null_arithmetic_tests::test_rem_with_nulls", test_rem_with_nulls),
    ("null_arithmetic_tests::test_add_scalar_preserves_nulls", test_add_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_sub_scalar_preserves_nulls", test_sub_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_mul_scalar_preserves_nulls", test_mul_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_div_scalar_preserves_nulls", test_div_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_rem_scalar_preserves_nulls", test_rem_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_eq_null_propagates", test_eq_null_propagates),
    ("null_arithmetic_tests::test_ne_null_propagates", test_ne_null_propagates),
    ("null_arithmetic_tests::test_gt_null_propagates", test_gt_null_propagates),
    ("null_arithmetic_tests::test_lt_null_propagates", test_lt_null_propagates),
    ("null_arithmetic_tests::test_ge_null_propagates", test_ge_null_propagates),
    ("null_arithmetic_tests::test_le_null_propagates", test_le_null_propagates),
    ("null_arithmetic_tests::test_eq_null_vs_null_is_null_not_true", test_eq_null_vs_null_is_null_not_true),
    ("null_arithmetic_tests::test_gt_scalar_preserves_nulls", test_gt_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_lt_scalar_preserves_nulls", test_lt_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_eq_scalar_preserves_nulls", test_eq_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_ne_scalar_preserves_nulls", test_ne_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_ge_scalar_preserves_nulls", test_ge_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_le_scalar_preserves_nulls", test_le_scalar_preserves_nulls),
    ("null_arithmetic_tests::test_eq_scalar_string_preserves_nulls", test_eq_scalar_string_preserves_nulls),
    ("null_arithmetic_tests::test_gt_scalar_string_preserves_nulls", test_gt_scalar_string_preserves_nulls),
    ("null_arithmetic_tests::test_neg_preserves_nulls", test_neg_preserves_nulls),
    ("null_arithmetic_tests::test_logical_not_preserves_nulls", test_logical_not_preserves_nulls),
    ("null_arithmetic_tests::test_logical_and_preserves_nulls", test_logical_and_preserves_nulls),
    ("null_arithmetic_tests::test_logical_or_preserves_nulls", test_logical_or_preserves_nulls),
    ("null_arithmetic_tests::test_abs_preserves_nulls", test_abs_preserves_nulls),
    ("null_arithmetic_tests::test_is_null_on_nullable", test_is_null_on_nullable),
    ("null_arithmetic_tests::test_is_not_null_on_nullable", test_is_not_null_on_nullable),
    ("null_arithmetic_tests::test_add_all_null_series", test_add_all_null_series),
    ("null_arithmetic_tests::test_eq_all_null_series", test_eq_all_null_series),
    ("null_arithmetic_tests::test_is_nan_does_not_flag_null_as_nan", test_is_nan_does_not_flag_null_as_nan),
    (
        "null_arithmetic_tests::test_is_not_nan_does_not_flag_null_as_not_nan",
        test_is_not_nan_does_not_flag_null_as_not_nan,
    ),
];
