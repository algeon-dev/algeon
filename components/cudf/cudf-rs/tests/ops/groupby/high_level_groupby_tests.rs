//! Tests for the `DataFrame::groupby` high-level API: `GroupBy`, `Agg`, `AggSpec`,
//! `GroupByOptions`.

use cudf::{Agg, AggSpec, Column, DataFrame, ErrorKind};

// ---------------------------------------------------------------------------
// GroupBy::agg with AggSpec — basic aggregations
// ---------------------------------------------------------------------------

#[test]
fn groupby_sum() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let sum_data: Vec<i64> = result.series("value_sum").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn groupby_mean() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Mean)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let mean_data: Vec<f64> = result.series("value_mean").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

#[test]
fn groupby_count() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Count { include_nulls: false })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let count_data: Vec<i32> = result.series("value_count").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

// ---------------------------------------------------------------------------
// Output column naming: alias support
// ---------------------------------------------------------------------------

#[test]
fn groupby_alias() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Sum).with_alias("total")]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "total"]);

    let sum_data: Vec<i64> = result.series("total").unwrap().to_vec::<i64>().unwrap();
    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn groupby_deterministic_naming_without_alias() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let prices = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = DataFrame::new(vec!["id".into(), "price".into()], vec![keys, prices]).unwrap();

    let result = df
        .groupby(&["id"])
        .agg(&[
            AggSpec::new("price", Agg::Sum),
            AggSpec::new("price", Agg::Mean),
            AggSpec::new("price", Agg::Count { include_nulls: false }),
            AggSpec::new("price", Agg::Min),
            AggSpec::new("price", Agg::Max),
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
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = Column::from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df
        .groupby(&["key"])
        .agg(&[AggSpec::new("a", Agg::Sum), AggSpec::new("b", Agg::Sum), AggSpec::new("a", Agg::Min)])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);

    let keys: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let a_sum: Vec<i64> = result.series("a_sum").unwrap().to_vec::<i64>().unwrap();
    let b_sum: Vec<i64> = result.series("b_sum").unwrap().to_vec::<i64>().unwrap();
    let a_min: Vec<i64> = result.series("a_min").unwrap().to_vec::<i64>().unwrap();

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
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(&["key"])
        .agg(&[AggSpec::new("nonexistent", Agg::Sum)])
        .expect_err("expected error for nonexistent column");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    let msg = format!("{}", err);
    assert!(msg.contains("nonexistent"), "error should mention column name");
}

#[test]
fn groupby_missing_key_column() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(&["missing_key"])
        .agg(&[AggSpec::new("value", Agg::Sum)])
        .expect_err("expected error for nonexistent key column");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn groupby_empty_agg_specs() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df.groupby(&["key"]).agg(&[]).expect_err("expected error for empty aggregation specs");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// Shortcut methods: sum, mean, count
// ---------------------------------------------------------------------------

#[test]
fn groupby_sum_shortcut() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).sum(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let sum_data: Vec<i64> = result.series("value_sum").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn groupby_mean_shortcut() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).mean(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let mean_data: Vec<f64> = result.series("value_mean").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

#[test]
fn groupby_count_shortcut() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).count(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let count_data: Vec<i32> = result.series("value_count").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

// ---------------------------------------------------------------------------
// Shortcut with multiple value columns
// ---------------------------------------------------------------------------

#[test]
fn groupby_sum_shortcut_multi_columns() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = Column::from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df.groupby(&["key"]).sum(&["a", "b"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum"]);

    let keys: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let a_sum: Vec<i64> = result.series("a_sum").unwrap().to_vec::<i64>().unwrap();
    let b_sum: Vec<i64> = result.series("b_sum").unwrap().to_vec::<i64>().unwrap();

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

    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let options = GroupByOptions::builder().include_null_keys(true).build();
    let result = df.groupby(&["key"]).options(options).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 2);
}

// ===========================================================================
// Expanded Agg enum — Median, NUnique, Std, Variance
// ===========================================================================

// ---------------------------------------------------------------------------
// Agg::Median
// ---------------------------------------------------------------------------

#[test]
fn groupby_median() {
    // keys: [1, 1, 1, 2, 2], values: [10.0, 20.0, 30.0, 100.0, 200.0]
    // key=1 median=20.0, key=2 median=150.0
    let keys = Column::from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Median)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_median"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let med_data: Vec<f64> = result.series("value_median").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(med_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 150.0)]);
}

#[test]
fn groupby_median_shortcut() {
    let keys = Column::from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).median(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_median"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let med_data: Vec<f64> = result.series("value_median").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(med_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 150.0)]);
}

// ---------------------------------------------------------------------------
// Agg::NUnique
// ---------------------------------------------------------------------------

#[test]
fn groupby_nunique() {
    // keys: [1, 1, 1, 2, 2], values: [10, 10, 20, 30, 30]
    // key=1 nunique=2, key=2 nunique=1
    let keys = Column::from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10i32, 10, 20, 30, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::NUnique)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_nunique"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let nu_data: Vec<i32> = result.series("value_nunique").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(nu_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 1)]);
}

#[test]
fn groupby_nunique_shortcut() {
    let keys = Column::from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10i32, 10, 20, 30, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).nunique(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_nunique"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let nu_data: Vec<i32> = result.series("value_nunique").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(nu_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 1)]);
}

// ---------------------------------------------------------------------------
// Agg::Std
// ---------------------------------------------------------------------------

#[test]
fn groupby_std_ddof1() {
    // keys: [1, 1, 2, 2], values: [10.0, 20.0, 100.0, 200.0]
    // key=1: std(ddof=1) = sqrt(((10-15)^2 + (20-15)^2) / 1) = sqrt(50) ≈ 7.0711
    // key=2: std(ddof=1) = sqrt(((100-150)^2 + (200-150)^2) / 1) = sqrt(5000) ≈ 70.7107
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Std { ddof: 1 })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_std"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec::<f64>().unwrap();

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
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Std { ddof: 0 })]).unwrap();

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 5.0).abs() < 1e-6, "key=1 std(ddof=0) mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 50.0).abs() < 1e-6, "key=2 std(ddof=0) mismatch: {}", pairs[1].1);
}

#[test]
fn groupby_std_shortcut() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    // Default ddof=1 for the shortcut.
    let result = df.groupby(&["key"]).std(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_std"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0f64.sqrt()).abs() < 1e-6);
    assert!((pairs[1].1 - 5000.0f64.sqrt()).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Agg::Variance
// ---------------------------------------------------------------------------

#[test]
fn groupby_variance_ddof1() {
    // key=1: var(ddof=1) = 50.0
    // key=2: var(ddof=1) = 5000.0
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Variance { ddof: 1 })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_var"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0).abs() < 1e-6, "key=1 var mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 5000.0).abs() < 1e-6, "key=2 var mismatch: {}", pairs[1].1);
}

#[test]
fn groupby_variance_ddof0() {
    // key=1: var(ddof=0) = 25.0
    // key=2: var(ddof=0) = 2500.0
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Variance { ddof: 0 })]).unwrap();

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 25.0).abs() < 1e-6, "key=1 var(ddof=0) mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 2500.0).abs() < 1e-6, "key=2 var(ddof=0) mismatch: {}", pairs[1].1);
}

#[test]
fn groupby_var_shortcut() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    // Default ddof=1 for the shortcut.
    let result = df.groupby(&["key"]).var(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_var"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0).abs() < 1e-6);
    assert!((pairs[1].1 - 5000.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Extended naming — all Agg variants in a single groupby
// ---------------------------------------------------------------------------

#[test]
fn groupby_all_agg_variants_naming() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(&["key"])
        .agg(&[
            AggSpec::new("value", Agg::Sum),
            AggSpec::new("value", Agg::Mean),
            AggSpec::new("value", Agg::Min),
            AggSpec::new("value", Agg::Max),
            AggSpec::new("value", Agg::Count { include_nulls: false }),
            AggSpec::new("value", Agg::Median),
            AggSpec::new("value", Agg::NUnique),
            AggSpec::new("value", Agg::Std { ddof: 1 }),
            AggSpec::new("value", Agg::Variance { ddof: 1 }),
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
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(&["key"])
        .agg(&[
            AggSpec::new("value", Agg::Median).with_alias("mid"),
            AggSpec::new("value", Agg::Std { ddof: 1 }).with_alias("spread"),
        ])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "mid", "spread"]);
}

#[test]
fn groupby_domain_module_exports_documented_surface() {
    use cudf::{
        Table,
        ops::groupby::{Aggregation, AggregationRequest, GroupByResult, NullPolicy, groupby_aggregate},
    };

    let keys = Table::from_columns(vec![Column::from_slice(&[1i32, 2, 1, 2]).unwrap()]).unwrap();
    let values = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let value_view = values.view().unwrap();
    let aggs = [Aggregation::Sum];
    let requests = [AggregationRequest { values: &value_view, aggregations: &aggs }];

    let result: GroupByResult = groupby_aggregate(&keys.view().unwrap(), &requests, NullPolicy::Exclude).unwrap();

    assert_eq!(result.keys.view().unwrap().num_rows().unwrap(), 2);
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 1);
}
