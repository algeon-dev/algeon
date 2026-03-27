//! Tests for DataFrame groupby aggregation.

use cudf::{column::Column, dataframe::DataFrame, error::ErrorKind, groupby::Aggregation};

// ---------------------------------------------------------------------------
// Groupby aggregation
// ---------------------------------------------------------------------------

#[test]
fn test_groupby_single_key_sum() {
    // keys: [1, 2, 1, 2], values: [10, 20, 30, 40]
    // Expected: key=1 → sum=40, key=2 → sum=60
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();

    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let result = df.groupby_builder(&["key"]).agg("value", Aggregation::Sum).execute().unwrap();

    // Should have 2 columns: "key" and "value_sum"
    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows().unwrap(), 2);

    // Check schema names
    let names = result.column_names();
    assert_eq!(names[0], "key");
    assert_eq!(names[1], "value_sum");

    // Verify data (sort by key for determinism)
    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let sum_data: Vec<i64> = result.column("value_sum").unwrap().to_vec::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn test_groupby_single_key_multiple_aggregations_same_column() {
    // keys: [1, 2, 1, 2], values: [10.0, 20.0, 30.0, 40.0]
    // Expected: key=1 → min=10, max=30; key=2 → min=20, max=40
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let df = DataFrame::new(vec!["key".to_string(), "score".to_string()], vec![keys, vals]).unwrap();

    let result =
        df.groupby_builder(&["key"]).agg("score", Aggregation::Min).agg("score", Aggregation::Max).execute().unwrap();

    // Should have 3 columns: "key", "score_min", "score_max"
    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.num_rows().unwrap(), 2);

    let names = result.column_names();
    assert_eq!(names[0], "key");
    assert_eq!(names[1], "score_min");
    assert_eq!(names[2], "score_max");

    let key_data: Vec<i32> = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let min_data: Vec<f64> = result.column("score_min").unwrap().to_vec::<f64>().unwrap();
    let max_data: Vec<f64> = result.column("score_max").unwrap().to_vec::<f64>().unwrap();

    let mut triples: Vec<(i32, f64, f64)> =
        key_data.into_iter().zip(min_data.into_iter().zip(max_data)).map(|(k, (mn, mx))| (k, mn, mx)).collect();
    triples.sort_by_key(|&(k, _, _)| k);

    assert_eq!(triples, vec![(1, 10.0, 30.0), (2, 20.0, 40.0)]);
}

#[test]
fn test_groupby_multiple_keys() {
    // Two key columns: (k0, k1) with 4 unique combinations
    let k0 = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let k1 = Column::from_slice(&[10i32, 20, 10, 20]).unwrap();
    let vals = Column::from_slice(&[100i64, 200, 300, 400]).unwrap();

    let df = DataFrame::new(vec!["a".to_string(), "b".to_string(), "value".to_string()], vec![k0, k1, vals]).unwrap();

    let result = df.groupby_builder(&["a", "b"]).agg("value", Aggregation::Sum).execute().unwrap();

    // Should have 3 columns: "a", "b", "value_sum"
    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.num_rows().unwrap(), 4);

    let names = result.column_names();
    assert_eq!(names[0], "a");
    assert_eq!(names[1], "b");
    assert_eq!(names[2], "value_sum");
}

#[test]
fn test_groupby_nonexistent_agg_column_returns_error() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();

    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df
        .groupby_builder(&["key"])
        .agg("nonexistent", Aggregation::Sum)
        .execute()
        .err()
        .expect("expected error for nonexistent column");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    let msg = format!("{}", err);
    assert!(msg.contains("nonexistent"), "error should mention column name");
}

#[test]
fn test_groupby_nonexistent_key_column_returns_error() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();

    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df
        .groupby_builder(&["missing_key"])
        .agg("value", Aggregation::Sum)
        .execute()
        .err()
        .expect("expected error for nonexistent key column");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_groupby_result_column_naming_convention() {
    // Verify naming: "{column}_{aggregation}"
    let keys = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let prices = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let df = DataFrame::new(vec!["id".to_string(), "price".to_string()], vec![keys, prices]).unwrap();

    let result = df
        .groupby_builder(&["id"])
        .agg("price", Aggregation::Sum)
        .agg("price", Aggregation::Mean)
        .agg("price", Aggregation::Count { include_nulls: false })
        .agg("price", Aggregation::Min)
        .agg("price", Aggregation::Max)
        .execute()
        .unwrap();

    let names = result.column_names();
    assert_eq!(names[0], "id");
    assert_eq!(names[1], "price_sum");
    assert_eq!(names[2], "price_mean");
    assert_eq!(names[3], "price_count");
    assert_eq!(names[4], "price_min");
    assert_eq!(names[5], "price_max");
}

#[test]
fn test_groupby_interleaved_multi_column_agg_ordering() {
    // Request order intentionally interleaves columns:
    //   a_sum, b_sum, a_min
    // Result columns and values must follow that exact order.
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = Column::from_slice(&[1i64, 2, 3, 4]).unwrap();

    let df = DataFrame::new(vec!["key".to_string(), "a".to_string(), "b".to_string()], vec![keys, a, b]).unwrap();

    let result = df
        .groupby_builder(&["key"])
        .agg("a", Aggregation::Sum)
        .agg("b", Aggregation::Sum)
        .agg("a", Aggregation::Min)
        .execute()
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);

    let keys = result.column("key").unwrap().to_vec::<i32>().unwrap();
    let a_sum = result.column("a_sum").unwrap().to_vec::<i64>().unwrap();
    let b_sum = result.column("b_sum").unwrap().to_vec::<i64>().unwrap();
    let a_min = result.column("a_min").unwrap().to_vec::<i64>().unwrap();

    let mut rows: Vec<(i32, i64, i64, i64)> = keys
        .into_iter()
        .zip(a_sum.into_iter().zip(b_sum.into_iter().zip(a_min)))
        .map(|(k, (asum, (bsum, amin)))| (k, asum, bsum, amin))
        .collect();
    rows.sort_by_key(|&(k, _, _, _)| k);

    // key=1: a_sum=30, b_sum=3, a_min=10
    // key=2: a_sum=70, b_sum=7, a_min=30
    assert_eq!(rows, vec![(1, 30, 3, 10), (2, 70, 7, 30)]);
}

#[test]
fn test_groupby_no_aggregations_returns_error() {
    let keys = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals = Column::from_slice(&[10i64, 20, 30]).unwrap();

    let df = DataFrame::new(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df.groupby_builder(&["key"]).execute().err().expect("expected error for empty aggregation");

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}
