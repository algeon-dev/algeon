//! Tests for façade GroupBy types: Agg, AggSpec, GroupBy, GroupByOptions.
//!
//! These test the cuDF-Python-like groupby API alongside the existing
//! `GroupByBuilder`.

use cudf::{
    column::Column,
    dataframe::DataFrame,
    error::ErrorKind,
    groupby::{Agg, AggSpec},
};

// ---------------------------------------------------------------------------
// GroupBy::agg with AggSpec — basic aggregations
// ---------------------------------------------------------------------------

#[test]
fn facade_groupby_sum() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.facade_groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows().unwrap(), 2);
    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let sum_data: Vec<i64> = result.column("value_sum").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn facade_groupby_mean() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.facade_groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Mean)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let mean_data: Vec<f64> = result.column("value_mean").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

#[test]
fn facade_groupby_count() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result =
        df.facade_groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Count { include_nulls: false })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let count_data: Vec<i32> = result.column("value_count").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

// ---------------------------------------------------------------------------
// Output column naming: alias support
// ---------------------------------------------------------------------------

#[test]
fn facade_groupby_alias() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.facade_groupby(&["key"]).agg(&[AggSpec::new("value", Agg::Sum).with_alias("total")]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "total"]);

    let sum_data: Vec<i64> = result.column("total").unwrap().to_vec::<i64>().unwrap();
    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn facade_groupby_deterministic_naming_without_alias() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let prices = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = DataFrame::new(vec!["id".into(), "price".into()], vec![keys, prices]).unwrap();

    let result = df
        .facade_groupby(&["id"])
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
fn facade_groupby_multi_column_agg_ordering() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = Column::from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df
        .facade_groupby(&["key"])
        .agg(&[AggSpec::new("a", Agg::Sum), AggSpec::new("b", Agg::Sum), AggSpec::new("a", Agg::Min)])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);

    let keys: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let a_sum: Vec<i64> = result.column("a_sum").unwrap().to_vec::<i64>().unwrap();
    let b_sum: Vec<i64> = result.column("b_sum").unwrap().to_vec::<i64>().unwrap();
    let a_min: Vec<i64> = result.column("a_min").unwrap().to_vec::<i64>().unwrap();

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
fn facade_groupby_missing_value_column() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .facade_groupby(&["key"])
        .agg(&[AggSpec::new("nonexistent", Agg::Sum)])
        .expect_err("expected error for nonexistent column");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    let msg = format!("{}", err);
    assert!(msg.contains("nonexistent"), "error should mention column name");
}

#[test]
fn facade_groupby_missing_key_column() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .facade_groupby(&["missing_key"])
        .agg(&[AggSpec::new("value", Agg::Sum)])
        .expect_err("expected error for nonexistent key column");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn facade_groupby_empty_agg_specs() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df.facade_groupby(&["key"]).agg(&[]).expect_err("expected error for empty aggregation specs");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// Shortcut methods: sum, mean, count
// ---------------------------------------------------------------------------

#[test]
fn facade_groupby_sum_shortcut() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.facade_groupby(&["key"]).sum(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let sum_data: Vec<i64> = result.column("value_sum").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn facade_groupby_mean_shortcut() {
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.facade_groupby(&["key"]).mean(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let mean_data: Vec<f64> = result.column("value_mean").unwrap().to_vec::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

#[test]
fn facade_groupby_count_shortcut() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.facade_groupby(&["key"]).count(&["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let count_data: Vec<i32> = result.column("value_count").unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

// ---------------------------------------------------------------------------
// Shortcut with multiple value columns
// ---------------------------------------------------------------------------

#[test]
fn facade_groupby_sum_shortcut_multi_columns() {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = Column::from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df.facade_groupby(&["key"]).sum(&["a", "b"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum"]);

    let keys: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let a_sum: Vec<i64> = result.column("a_sum").unwrap().to_vec::<i64>().unwrap();
    let b_sum: Vec<i64> = result.column("b_sum").unwrap().to_vec::<i64>().unwrap();

    let mut rows: Vec<(i32, i64, i64)> =
        keys.into_iter().zip(a_sum.into_iter().zip(b_sum)).map(|(k, (a, b))| (k, a, b)).collect();
    rows.sort_by_key(|&(k, _, _)| k);

    assert_eq!(rows, vec![(1, 30, 3), (2, 70, 7)]);
}

// ---------------------------------------------------------------------------
// GroupByOptions: include_null_keys
// ---------------------------------------------------------------------------

#[test]
fn facade_groupby_with_options_include_null_keys() {
    use cudf::groupby::GroupByOptions;

    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = DataFrame::new(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let options = GroupByOptions::default().include_null_keys(true);
    let result = df.facade_groupby(&["key"]).options(options).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows().unwrap(), 2);
}
