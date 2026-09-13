//! Integration tests for high-level host-returning `Series` reduction helpers.
//!
//! All tests require a GPU.

use cudf::ErrorKind;

use crate::common::{nullable_series, run_named_cases, series};

fn test_sum_value_i32_returns_i64_host_value() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 60);
}

fn test_sum_value_bool_returns_i64_host_value() {
    let s = series::<bool>("x", &[true, false, true, true]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 3);
}

fn test_sum_value_all_null_preserves_identity_semantics() {
    let s = nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 0);
}

fn test_sum_value_empty_preserves_identity_semantics() {
    let s = series::<i32>("x", &[]).unwrap();
    let total = s.sum_value::<i64>().unwrap();

    assert_eq!(total, 0);
}

fn test_sum_value_propagates_output_type_contract() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let err = s.sum_value::<i32>().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

fn test_min_value_returns_some_for_valid_input() {
    let s = series::<i32>("x", &[30, 10, 20]).unwrap();
    let min = s.min_value::<i32>().unwrap();

    assert_eq!(min, Some(10));
}

fn test_min_value_all_null_returns_none() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let min = s.min_value::<i32>().unwrap();

    assert_eq!(min, None);
}

fn test_min_value_empty_returns_none() {
    let s = series::<i32>("x", &[]).unwrap();
    let min = s.min_value::<i32>().unwrap();

    assert_eq!(min, None);
}

fn test_max_value_returns_some_for_valid_input() {
    let s = series::<i32>("x", &[30, 10, 20]).unwrap();
    let max = s.max_value::<i32>().unwrap();

    assert_eq!(max, Some(30));
}

fn test_max_value_all_null_returns_none() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let max = s.max_value::<i32>().unwrap();

    assert_eq!(max, None);
}

fn test_mean_value_returns_some_for_valid_input() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let mean = s.mean_value().unwrap();

    assert_eq!(mean, Some(20.0));
}

fn test_mean_value_all_null_returns_none() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let mean = s.mean_value().unwrap();

    assert_eq!(mean, None);
}

fn test_mean_value_empty_returns_none() {
    let s = series::<i32>("x", &[]).unwrap();
    let mean = s.mean_value().unwrap();

    assert_eq!(mean, None);
}

fn test_median_value_returns_some_for_valid_input() {
    let s = nullable_series::<i32>("x", &[10, 0, 30, 20], &[true, false, true, true]).unwrap();
    let median = s.median_value().unwrap();

    assert_eq!(median, Some(20.0));
}

fn test_median_value_all_null_returns_none() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let median = s.median_value().unwrap();

    assert_eq!(median, None);
}

fn test_std_value_returns_some_for_valid_input() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let std_dev = s.std_value(1).unwrap();

    assert_eq!(std_dev, Some(200.0f64.sqrt()));
}

fn test_std_value_all_null_returns_none() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let std_dev = s.std_value(1).unwrap();

    assert_eq!(std_dev, None);
}

fn test_var_value_returns_some_for_valid_input() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let variance = s.var_value(1).unwrap();

    assert_eq!(variance, Some(200.0));
}

fn test_var_value_all_null_returns_none() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let variance = s.var_value(1).unwrap();

    assert_eq!(variance, None);
}

fn test_any_value_returns_host_bool() {
    let s = nullable_series::<bool>("x", &[true, false, false], &[true, false, true]).unwrap();
    let any = s.any_value().unwrap();

    assert!(any);
}

fn test_any_value_all_null_preserves_identity_semantics() {
    let s = nullable_series::<bool>("x", &[false, false], &[false, false]).unwrap();
    let any = s.any_value().unwrap();

    assert!(!any);
}

fn test_all_value_returns_host_bool() {
    let s = nullable_series::<bool>("x", &[true, false, true], &[true, false, true]).unwrap();
    let all = s.all_value().unwrap();

    assert!(all);
}

fn test_product_value_returns_some_for_valid_input() {
    let s = series::<i32>("x", &[2, 3, 4]).unwrap();
    let product = s.product_value::<i32>().unwrap();

    assert_eq!(product, Some(24));
}

fn test_product_value_partial_nulls_skip_nulls() {
    let s = nullable_series::<i32>("x", &[2, 0, 5], &[true, false, true]).unwrap();
    let product = s.product_value::<i32>().unwrap();

    assert_eq!(product, Some(10));
}

fn test_minmax_values_returns_host_options() {
    let s = series::<i32>("x", &[30, 10, 20]).unwrap();
    let bounds = s.minmax_values::<i32>().unwrap();

    assert_eq!(bounds, (Some(10), Some(30)));
}

fn test_minmax_values_all_null_returns_nones() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let bounds = s.minmax_values::<i32>().unwrap();

    assert_eq!(bounds, (None, None));
}

#[test]
fn series_reduction_value_tests_cases() {
    run_named_cases(&[
        ("test_sum_value_i32_returns_i64_host_value", test_sum_value_i32_returns_i64_host_value),
        ("test_sum_value_bool_returns_i64_host_value", test_sum_value_bool_returns_i64_host_value),
        ("test_sum_value_all_null_preserves_identity_semantics", test_sum_value_all_null_preserves_identity_semantics),
        ("test_sum_value_empty_preserves_identity_semantics", test_sum_value_empty_preserves_identity_semantics),
        ("test_sum_value_propagates_output_type_contract", test_sum_value_propagates_output_type_contract),
        ("test_min_value_returns_some_for_valid_input", test_min_value_returns_some_for_valid_input),
        ("test_min_value_all_null_returns_none", test_min_value_all_null_returns_none),
        ("test_min_value_empty_returns_none", test_min_value_empty_returns_none),
        ("test_max_value_returns_some_for_valid_input", test_max_value_returns_some_for_valid_input),
        ("test_max_value_all_null_returns_none", test_max_value_all_null_returns_none),
        ("test_mean_value_returns_some_for_valid_input", test_mean_value_returns_some_for_valid_input),
        ("test_mean_value_all_null_returns_none", test_mean_value_all_null_returns_none),
        ("test_mean_value_empty_returns_none", test_mean_value_empty_returns_none),
        ("test_median_value_returns_some_for_valid_input", test_median_value_returns_some_for_valid_input),
        ("test_median_value_all_null_returns_none", test_median_value_all_null_returns_none),
        ("test_std_value_returns_some_for_valid_input", test_std_value_returns_some_for_valid_input),
        ("test_std_value_all_null_returns_none", test_std_value_all_null_returns_none),
        ("test_var_value_returns_some_for_valid_input", test_var_value_returns_some_for_valid_input),
        ("test_var_value_all_null_returns_none", test_var_value_all_null_returns_none),
        ("test_any_value_returns_host_bool", test_any_value_returns_host_bool),
        ("test_any_value_all_null_preserves_identity_semantics", test_any_value_all_null_preserves_identity_semantics),
        ("test_all_value_returns_host_bool", test_all_value_returns_host_bool),
        ("test_product_value_returns_some_for_valid_input", test_product_value_returns_some_for_valid_input),
        ("test_product_value_partial_nulls_skip_nulls", test_product_value_partial_nulls_skip_nulls),
        ("test_minmax_values_returns_host_options", test_minmax_values_returns_host_options),
        ("test_minmax_values_all_null_returns_nones", test_minmax_values_all_null_returns_nones),
    ]);
}
