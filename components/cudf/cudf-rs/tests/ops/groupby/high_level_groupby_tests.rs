//! Tests for the `DataFrame::groupby` high-level API: `GroupBy`, `GroupByAgg`, `AggSpec`,
//! `GroupByOptions`.

use cudf::{AggSpec, DataFrame, ErrorKind, GroupByAgg};

use crate::common::{assert_missing_column_error, column_from_slice};

// ---------------------------------------------------------------------------
// GroupBy::agg with AggSpec — basic aggregations
// ---------------------------------------------------------------------------

#[test]
fn groupby_sum() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let sum_data: Vec<i64> = result.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn groupby_mean() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Mean)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let mean_data: Vec<f64> = result.series("value_mean").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

#[test]
fn groupby_count() {
    let keys = column_from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Count { include_nulls: false })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let count_data: Vec<i32> = result.series("value_count").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

// ---------------------------------------------------------------------------
// Output column naming: alias support
// ---------------------------------------------------------------------------

#[test]
fn groupby_alias() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Sum).with_alias("total")]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "total"]);

    let sum_data: Vec<i64> = result.series("total").unwrap().to_vec_strict::<i64>().unwrap();
    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn groupby_deterministic_naming_without_alias() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let prices = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["id".into(), "price".into()], vec![keys, prices]).unwrap();

    let result = df
        .groupby(["id"])
        .agg(&[
            AggSpec::new("price", GroupByAgg::Sum),
            AggSpec::new("price", GroupByAgg::Mean),
            AggSpec::new("price", GroupByAgg::Count { include_nulls: false }),
            AggSpec::new("price", GroupByAgg::Min),
            AggSpec::new("price", GroupByAgg::Max),
        ])
        .unwrap();

    let names = result.column_names();
    assert_eq!(names[0], "id");
    assert_eq!(names[1], "price_sum");
    assert_eq!(names[2], "price_mean");
    assert_eq!(names[3], "price_count");
    assert_eq!(names[4], "price_min");
    assert_eq!(names[5], "price_max");
}

// ---------------------------------------------------------------------------
// Multi-column aggregation ordering
// ---------------------------------------------------------------------------

#[test]
fn groupby_multi_column_agg_ordering() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = column_from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df =
        crate::common::dataframe_from_columns(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df
        .groupby(["key"])
        .agg(&[
            AggSpec::new("a", GroupByAgg::Sum),
            AggSpec::new("b", GroupByAgg::Sum),
            AggSpec::new("a", GroupByAgg::Min),
        ])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);

    let keys: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let a_sum: Vec<i64> = result.series("a_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let b_sum: Vec<i64> = result.series("b_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let a_min: Vec<i64> = result.series("a_min").unwrap().to_vec_strict::<i64>().unwrap();

    let mut rows: Vec<(i32, i64, i64, i64)> = keys
        .into_iter()
        .zip(a_sum.into_iter().zip(b_sum.into_iter().zip(a_min)))
        .map(|(k, (asum, (bsum, amin)))| (k, asum, bsum, amin))
        .collect();
    rows.sort_by_key(|&(k, _, _, _)| k);

    assert_eq!(rows, vec![(1, 30, 3, 10), (2, 70, 7, 30)]);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn groupby_missing_value_column() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["key"])
        .agg(&[AggSpec::new("nonexistent", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent column");

    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn groupby_missing_key_column() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["missing_key"])
        .agg(&[AggSpec::new("value", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent key column");

    assert_missing_column_error(&err, "missing_key");
}

#[test]
fn groupby_empty_agg_specs() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df.groupby(["key"]).agg(&[]).expect_err("expected error for empty aggregation specs");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// Shortcut methods: sum, mean, count
// ---------------------------------------------------------------------------

#[test]
fn groupby_sum_shortcut() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).sum(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let sum_data: Vec<i64> = result.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn groupby_mean_shortcut() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).mean(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let mean_data: Vec<f64> = result.series("value_mean").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

#[test]
fn groupby_count_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).count(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let count_data: Vec<i32> = result.series("value_count").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

#[test]
fn groupby_shortcut_handle_is_reusable() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let groupby = df.groupby(["key"]);
    let summed = groupby.sum(["value"]).unwrap();
    let counted = groupby.count(["value"]).unwrap();

    assert_eq!(summed.column_names(), vec!["key", "value_sum"]);
    assert_eq!(counted.column_names(), vec!["key", "value_count"]);
}

#[test]
fn scoped_groupby_handle_is_reusable() {
    use cudf::{ExecutionContext, Runtime, runtime::CudaStream};

    let rt = Runtime::builder().build().unwrap();
    let df = DataFrame::from_series([
        rt.series("key", &[1i32, 2, 1, 2]).unwrap(),
        rt.series("value", &[10i64, 20, 30, 40]).unwrap(),
    ])
    .unwrap();

    let stream = CudaStream::new().unwrap();
    let ctx = ExecutionContext::new().with_stream(&stream);
    let groupby = df.groupby(["key"]).in_context(&ctx);
    let summed = groupby.sum(["value"]).unwrap();
    let averaged = groupby.mean(["value"]).unwrap();

    assert_eq!(summed.column_names(), vec!["key", "value_sum"]);
    assert_eq!(averaged.column_names(), vec!["key", "value_mean"]);
}

// ---------------------------------------------------------------------------
// Shortcut with multiple value columns
// ---------------------------------------------------------------------------

#[test]
fn groupby_sum_shortcut_multi_columns() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = column_from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df =
        crate::common::dataframe_from_columns(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df.groupby(["key"]).sum(["a", "b"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum"]);

    let keys: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let a_sum: Vec<i64> = result.series("a_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let b_sum: Vec<i64> = result.series("b_sum").unwrap().to_vec_strict::<i64>().unwrap();

    let mut rows: Vec<(i32, i64, i64)> =
        keys.into_iter().zip(a_sum.into_iter().zip(b_sum)).map(|(k, (a, b))| (k, a, b)).collect();
    rows.sort_by_key(|&(k, _, _)| k);

    assert_eq!(rows, vec![(1, 30, 3), (2, 70, 7)]);
}

// ---------------------------------------------------------------------------
// GroupByOptions: include_null_keys
// ---------------------------------------------------------------------------

#[test]
fn groupby_with_options_include_null_keys() {
    use cudf::GroupByOptions;

    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let options = GroupByOptions::builder().include_null_keys(true).build();
    let result = df.groupby(["key"]).options(options).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 2);
}

// ===========================================================================
// Expanded GroupByAgg enum — Median, NUnique, Std, Variance
// ===========================================================================

// ---------------------------------------------------------------------------
// GroupByAgg::Median
// ---------------------------------------------------------------------------

#[test]
fn groupby_median() {
    // keys: [1, 1, 1, 2, 2], values: [10.0, 20.0, 30.0, 100.0, 200.0]
    // key=1 median=20.0, key=2 median=150.0
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Median)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_median"]);

    let key_data: Vec<Option<i32>> = result.series("key").unwrap().to_optional_vec::<i32>().unwrap();
    let med_data: Vec<Option<f64>> = result.series("value_median").unwrap().to_optional_vec::<f64>().unwrap();

    let mut pairs: Vec<(Option<i32>, Option<f64>)> = key_data.into_iter().zip(med_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(Some(1), Some(20.0)), (Some(2), Some(150.0))]);
}

#[test]
fn groupby_median_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).median(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_median"]);

    let key_data: Vec<Option<i32>> = result.series("key").unwrap().to_optional_vec::<i32>().unwrap();
    let med_data: Vec<Option<f64>> = result.series("value_median").unwrap().to_optional_vec::<f64>().unwrap();

    let mut pairs: Vec<(Option<i32>, Option<f64>)> = key_data.into_iter().zip(med_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(Some(1), Some(20.0)), (Some(2), Some(150.0))]);
}

// ---------------------------------------------------------------------------
// GroupByAgg::NUnique
// ---------------------------------------------------------------------------

#[test]
fn groupby_nunique() {
    // keys: [1, 1, 1, 2, 2], values: [10, 10, 20, 30, 30]
    // key=1 nunique=2, key=2 nunique=1
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 10, 20, 30, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::NUnique)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_nunique"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let nu_data: Vec<i32> = result.series("value_nunique").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(nu_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 1)]);
}

#[test]
fn groupby_nunique_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 10, 20, 30, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).nunique(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_nunique"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let nu_data: Vec<i32> = result.series("value_nunique").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(nu_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 1)]);
}

// ---------------------------------------------------------------------------
// GroupByAgg::Std
// ---------------------------------------------------------------------------

#[test]
fn groupby_std_ddof1() {
    // keys: [1, 1, 2, 2], values: [10.0, 20.0, 100.0, 200.0]
    // key=1: std(ddof=1) = sqrt(((10-15)^2 + (20-15)^2) / 1) = sqrt(50) ≈ 7.0711
    // key=2: std(ddof=1) = sqrt(((100-150)^2 + (200-150)^2) / 1) = sqrt(5000) ≈ 70.7107
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Std { ddof: 1 })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_std"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0, 1);
    assert!((pairs[0].1 - 50.0f64.sqrt()).abs() < 1e-6, "key=1 std mismatch: {}", pairs[0].1);
    assert_eq!(pairs[1].0, 2);
    assert!((pairs[1].1 - 5000.0f64.sqrt()).abs() < 1e-6, "key=2 std mismatch: {}", pairs[1].1);
}

#[test]
fn groupby_std_ddof0() {
    // key=1: std(ddof=0) = sqrt(((10-15)^2 + (20-15)^2) / 2) = sqrt(25) = 5.0
    // key=2: std(ddof=0) = sqrt(((100-150)^2 + (200-150)^2) / 2) = sqrt(2500) = 50.0
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Std { ddof: 0 })]).unwrap();

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 5.0).abs() < 1e-6, "key=1 std(ddof=0) mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 50.0).abs() < 1e-6, "key=2 std(ddof=0) mismatch: {}", pairs[1].1);
}

#[test]
fn groupby_std_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    // Default ddof=1 for the shortcut.
    let result = df.groupby(["key"]).std(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_std"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0f64.sqrt()).abs() < 1e-6);
    assert!((pairs[1].1 - 5000.0f64.sqrt()).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// GroupByAgg::Variance
// ---------------------------------------------------------------------------

#[test]
fn groupby_variance_ddof1() {
    // key=1: var(ddof=1) = 50.0
    // key=2: var(ddof=1) = 5000.0
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Variance { ddof: 1 })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_var"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0).abs() < 1e-6, "key=1 var mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 5000.0).abs() < 1e-6, "key=2 var mismatch: {}", pairs[1].1);
}

#[test]
fn groupby_variance_ddof0() {
    // key=1: var(ddof=0) = 25.0
    // key=2: var(ddof=0) = 2500.0
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Variance { ddof: 0 })]).unwrap();

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 25.0).abs() < 1e-6, "key=1 var(ddof=0) mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 2500.0).abs() < 1e-6, "key=2 var(ddof=0) mismatch: {}", pairs[1].1);
}

#[test]
fn groupby_var_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    // Default ddof=1 for the shortcut.
    let result = df.groupby(["key"]).var(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_var"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0).abs() < 1e-6);
    assert!((pairs[1].1 - 5000.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Extended naming — all GroupByAgg variants in a single groupby
// ---------------------------------------------------------------------------

#[test]
fn groupby_all_agg_variants_naming() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .agg(&[
            AggSpec::new("value", GroupByAgg::Sum),
            AggSpec::new("value", GroupByAgg::Mean),
            AggSpec::new("value", GroupByAgg::Min),
            AggSpec::new("value", GroupByAgg::Max),
            AggSpec::new("value", GroupByAgg::Count { include_nulls: false }),
            AggSpec::new("value", GroupByAgg::Median),
            AggSpec::new("value", GroupByAgg::NUnique),
            AggSpec::new("value", GroupByAgg::Std { ddof: 1 }),
            AggSpec::new("value", GroupByAgg::Variance { ddof: 1 }),
        ])
        .unwrap();

    assert_eq!(
        result.column_names(),
        vec![
            "key",
            "value_sum",
            "value_mean",
            "value_min",
            "value_max",
            "value_count",
            "value_median",
            "value_nunique",
            "value_std",
            "value_var",
        ]
    );
}

// ---------------------------------------------------------------------------
// Alias works with new variants
// ---------------------------------------------------------------------------

#[test]
fn groupby_new_agg_alias() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .agg(&[
            AggSpec::new("value", GroupByAgg::Median).with_alias("mid"),
            AggSpec::new("value", GroupByAgg::Std { ddof: 1 }).with_alias("spread"),
        ])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "mid", "spread"]);
}

#[test]
fn groupby_domain_module_exports_documented_surface() {
    use cudf::{
        core::Table,
        ops::groupby::{AggregationRequest, GroupByAgg, GroupByAggregationOutput, NullPolicy, groupby_aggregate},
    };

    let keys = Table::from_columns(vec![column_from_slice(&[1i32, 2, 1, 2]).unwrap()]).unwrap();
    let values = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let value_view = values.view().unwrap();
    let aggs = [GroupByAgg::Sum];
    let requests = [AggregationRequest { values: &value_view, aggregations: &aggs }];

    let result: GroupByAggregationOutput = groupby_aggregate(&keys.view().unwrap(), &requests, NullPolicy::Exclude).unwrap();

    assert_eq!(result.keys.view().unwrap().num_rows().unwrap(), 2);
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 1);
}
