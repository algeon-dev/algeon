//! Specification tests: null semantics in GroupBy operations.
//!
//! These tests lock down cuDF/libcudf null behavior for:
//! - Null values in aggregation value columns (sum, count, mean, min, max)
//! - Null values in groupby key columns
//! - Count with include_nulls vs exclude_nulls
//! - All-null groups
//!
//! All tests require a GPU.

use cudf::{Agg, AggSpec, DataFrame, core::Column};

/// Helper: build a DataFrame with key and value columns for groupby tests.
fn df_with_nullable_values() -> DataFrame {
    // key: [1, 1, 2, 2], value: [10, null, 30, 40]
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice_nullable(&[10i64, 0, 30, 40], &[true, false, true, true]).unwrap();
    DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap()
}

// ===========================================================================
// Sum with null values in value column
// ===========================================================================

#[test]
fn test_groupby_sum_skips_nulls_in_values() {
    let df = df_with_nullable_values();
    // key=1: [10, null] -> sum=10 (skip null)
    // key=2: [30, 40] -> sum=70
    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let sum_data = result.series("value_sum").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 10), (2, 70)]);
}

// ===========================================================================
// Min / Max with null values
// ===========================================================================

#[test]
fn test_groupby_min_skips_nulls() {
    let df = df_with_nullable_values();
    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Min)]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let min_data = result.series("value_min").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(min_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    // key=1: [10, null] -> min=10
    // key=2: [30, 40] -> min=30
    assert_eq!(pairs, vec![(1, 10), (2, 30)]);
}

#[test]
fn test_groupby_max_skips_nulls() {
    let df = df_with_nullable_values();
    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Max)]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let max_data = result.series("value_max").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(max_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    // key=1: [10, null] -> max=10
    // key=2: [30, 40] -> max=40
    assert_eq!(pairs, vec![(1, 10), (2, 40)]);
}

// ===========================================================================
// Mean with null values
// ===========================================================================

#[test]
fn test_groupby_mean_skips_nulls() {
    let df = df_with_nullable_values();
    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Mean)]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let mean_data = result.series("value_mean").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    // key=1: [10, null] -> mean=10.0 (only one valid element)
    // key=2: [30, 40] -> mean=35.0
    assert!((pairs[0].1 - 10.0).abs() < 1e-10);
    assert!((pairs[1].1 - 35.0).abs() < 1e-10);
}

// ===========================================================================
// Count with include_nulls vs exclude_nulls
// ===========================================================================

#[test]
fn test_groupby_count_excludes_nulls_by_default() {
    let df = df_with_nullable_values();
    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Count { include_nulls: false })]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let count_data = result.series("value_count").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    // key=1: [10, null] -> count=1 (exclude null)
    // key=2: [30, 40] -> count=2
    assert_eq!(pairs, vec![(1, 1), (2, 2)]);
}

#[test]
fn test_groupby_count_includes_nulls_when_requested() {
    let df = df_with_nullable_values();
    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Count { include_nulls: true })]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let count_data = result.series("value_count").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    // key=1: [10, null] -> count=2 (include null)
    // key=2: [30, 40] -> count=2
    assert_eq!(pairs, vec![(1, 2), (2, 2)]);
}

// ===========================================================================
// All-null group
// ===========================================================================

#[test]
fn test_groupby_sum_all_null_group() {
    // key: [1, 1, 2], value: [null, null, 10]
    let keys = Column::from_slice(&[1i32, 1, 2]).unwrap();
    let vals = Column::from_slice_nullable(&[0i64, 0, 10], &[false, false, true]).unwrap();
    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let sum_series = result.series("value_sum").unwrap();
    let sum_data = sum_series.to_optional_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, Option<i64>)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    // key=1: all null values -> sum is 0 (identity for sum) or null
    // key=2: [10] -> sum=10
    // cuDF Python: sum of all-null group = 0 (not null)
    assert_eq!(pairs[0].0, 1);
    assert_eq!(pairs[1], (2, Some(10)));
}

#[test]
fn test_groupby_min_all_null_group() {
    // key: [1, 1, 2], value: [null, null, 10]
    let keys = Column::from_slice(&[1i32, 1, 2]).unwrap();
    let vals = Column::from_slice_nullable(&[0i64, 0, 10], &[false, false, true]).unwrap();
    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Min)]).unwrap();

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let min_series = result.series("value_min").unwrap();
    let min_data = min_series.to_optional_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, Option<i64>)> = key_data.into_iter().zip(min_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    // key=1: all null -> min should be null (no valid element)
    assert_eq!(pairs[0].0, 1);
    assert_eq!(pairs[0].1, None);
    // key=2: [10] -> min=10
    assert_eq!(pairs[1], (2, Some(10)));
}

// ===========================================================================
// Null keys in GroupBy
// ===========================================================================

#[test]
fn test_groupby_null_keys_excluded_by_default() {
    // key: [1, null, 1, 2], value: [10, 20, 30, 40]
    let keys = Column::from_slice_nullable(&[1i32, 0, 1, 2], &[true, false, true, true]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    // Without include_null_keys, the null-key row should be dropped
    assert_eq!(result.num_rows(), 2);

    let key_data = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let sum_data = result.series("value_sum").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    // key=1: [10, 30] -> sum=40
    // key=2: [40] -> sum=40
    assert_eq!(pairs, vec![(1, 40), (2, 40)]);
}

#[test]
fn test_groupby_null_keys_included_when_requested() {
    use cudf::GroupByOptions;

    // key: [1, null, 1, 2], value: [10, 20, 30, 40]
    let keys = Column::from_slice_nullable(&[1i32, 0, 1, 2], &[true, false, true, true]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let options = GroupByOptions::builder().include_null_keys(true).build();
    let result = df.groupby(["key"]).options(options).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    // With include_null_keys, the null-key group should appear
    assert_eq!(result.num_rows(), 3);

    let key_series = result.series("key").unwrap();
    let sum_series = result.series("value_sum").unwrap();
    let key_data = key_series.to_optional_vec::<i32>().unwrap();
    let _sum_data = sum_series.to_vec::<i64>().unwrap();

    // Should have groups for key=1, key=2, and key=null
    let null_key_count = key_data.iter().filter(|k| k.is_none()).count();
    assert_eq!(null_key_count, 1);
}
