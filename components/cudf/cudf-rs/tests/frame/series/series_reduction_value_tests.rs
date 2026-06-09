//! Integration tests for high-level host-returning `Series` reduction helpers.
//!
//! All tests require a GPU.

use cudf::ErrorKind;

#[test]
fn test_sum_value_i32_returns_i64_host_value() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 60);
}

#[test]
fn test_sum_value_bool_returns_i64_host_value() {
    let s = crate::common::series::<bool>("x", &[true, false, true, true]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 3);
}

#[test]
fn test_sum_value_all_null_preserves_identity_semantics() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 0);
}

#[test]
fn test_sum_value_empty_preserves_identity_semantics() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 0);
}

#[test]
fn test_sum_value_propagates_output_type_contract() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let err = s.sum_value::<i32>().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::TypeError);
}

#[test]
fn test_min_value_returns_some_for_valid_input() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let min = s.min_value::<i32>().unwrap();

    assert_eq!(min, Some(10));
}

#[test]
fn test_min_value_all_null_returns_none() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let min = s.min_value::<i32>().unwrap();

    assert_eq!(min, None);
}

#[test]
fn test_min_value_empty_returns_none() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let min = s.min_value::<i32>().unwrap();

    assert_eq!(min, None);
}

#[test]
fn test_max_value_returns_some_for_valid_input() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let max = s.max_value::<i32>().unwrap();

    assert_eq!(max, Some(30));
}

#[test]
fn test_max_value_all_null_returns_none() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let max = s.max_value::<i32>().unwrap();

    assert_eq!(max, None);
}

#[test]
fn test_mean_value_returns_some_for_valid_input() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let mean = s.mean_value().unwrap();

    assert_eq!(mean, Some(20.0));
}

#[test]
fn test_mean_value_all_null_returns_none() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let mean = s.mean_value().unwrap();

    assert_eq!(mean, None);
}

#[test]
fn test_mean_value_empty_returns_none() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let mean = s.mean_value().unwrap();

    assert_eq!(mean, None);
}

#[test]
fn test_median_value_returns_some_for_valid_input() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30, 20], &[true, false, true, true]).unwrap();
    let median = s.median_value().unwrap();

    assert_eq!(median, Some(20.0));
}

#[test]
fn test_median_value_all_null_returns_none() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let median = s.median_value().unwrap();

    assert_eq!(median, None);
}

#[test]
fn test_std_value_returns_some_for_valid_input() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let std_dev = s.std_value(1).unwrap();

    assert_eq!(std_dev, Some(200.0f64.sqrt()));
}

#[test]
fn test_std_value_all_null_returns_none() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let std_dev = s.std_value(1).unwrap();

    assert_eq!(std_dev, None);
}

#[test]
fn test_var_value_returns_some_for_valid_input() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let variance = s.var_value(1).unwrap();

    assert_eq!(variance, Some(200.0));
}

#[test]
fn test_var_value_all_null_returns_none() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let variance = s.var_value(1).unwrap();

    assert_eq!(variance, None);
}

#[test]
fn test_any_value_returns_host_bool() {
    let s = crate::common::nullable_series::<bool>("x", &[true, false, false], &[true, false, true]).unwrap();
    let any = s.any_value().unwrap();

    assert!(any);
}

#[test]
fn test_any_value_all_null_preserves_identity_semantics() {
    let s = crate::common::nullable_series::<bool>("x", &[false, false], &[false, false]).unwrap();
    let any = s.any_value().unwrap();

    assert!(!any);
}

#[test]
fn test_all_value_returns_host_bool() {
    let s = crate::common::nullable_series::<bool>("x", &[true, false, true], &[true, false, true]).unwrap();
    let all = s.all_value().unwrap();

    assert!(all);
}

#[test]
fn test_product_value_returns_some_for_valid_input() {
    let s = crate::common::series::<i32>("x", &[2, 3, 4]).unwrap();
    let product = s.product_value::<i32>().unwrap();

    assert_eq!(product, Some(24));
}

#[test]
fn test_product_value_partial_nulls_skip_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[2, 0, 5], &[true, false, true]).unwrap();
    let product = s.product_value::<i32>().unwrap();

    assert_eq!(product, Some(10));
}

#[test]
fn test_minmax_values_returns_host_options() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let bounds = s.minmax_values::<i32>().unwrap();

    assert_eq!(bounds, (Some(10), Some(30)));
}

#[test]
fn test_minmax_values_all_null_returns_nones() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let bounds = s.minmax_values::<i32>().unwrap();

    assert_eq!(bounds, (None, None));
}
