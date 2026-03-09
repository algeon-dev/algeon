//! Tests for groupby operations.

use cudf::column::Column;
use cudf::groupby::{self, Aggregation, AggregationRequest, NullPolicy};
use cudf::table::Table;

#[test]
fn test_groupby_sum() {
    // keys: [1, 2, 1, 2], values: [10, 20, 30, 40]
    // Expected groups: key=1 -> sum=40, key=2 -> sum=60
    let keys_col = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals_col = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.view().unwrap();

    let vals_view = vals_col.view().unwrap();
    let requests = [AggregationRequest {
        values: &vals_view,
        aggregations: &[Aggregation::Sum],
    }];

    let result =
        groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    // Check keys.
    let result_kv = result.keys.view().unwrap();
    assert_eq!(result_kv.num_rows().unwrap(), 2);

    let mut key_vals: Vec<i32> = result_kv.column(0).unwrap().to_vec::<i32>().unwrap();

    // Get sums.
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 1);
    let mut sum_vals: Vec<i64> = result.results[0][0].view().unwrap().to_vec::<i64>().unwrap();

    // Sort by key for deterministic comparison.
    let mut pairs: Vec<(i32, i64)> = key_vals.drain(..).zip(sum_vals.drain(..)).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn test_groupby_multiple_aggregations() {
    let keys_col = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals_col = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.view().unwrap();

    let vals_view = vals_col.view().unwrap();
    let requests = [AggregationRequest {
        values: &vals_view,
        aggregations: &[Aggregation::Min, Aggregation::Max],
    }];

    let result =
        groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    let result_kv = result.keys.view().unwrap();
    assert_eq!(result_kv.num_rows().unwrap(), 2);

    // We have 1 request with 2 aggregations -> results[0] has 2 columns.
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 2);

    let keys: Vec<i32> = result_kv.column(0).unwrap().to_vec::<i32>().unwrap();
    let mins: Vec<f64> = result.results[0][0].view().unwrap().to_vec::<f64>().unwrap();
    let maxs: Vec<f64> = result.results[0][1].view().unwrap().to_vec::<f64>().unwrap();

    // Build sorted pairs.
    let mut triples: Vec<(i32, f64, f64)> = keys
        .into_iter()
        .zip(mins.into_iter().zip(maxs))
        .map(|(k, (mn, mx))| (k, mn, mx))
        .collect();
    triples.sort_by_key(|&(k, _, _)| k);

    assert_eq!(triples, vec![(1, 10.0, 30.0), (2, 20.0, 40.0)]);
}

#[test]
fn test_groupby_count() {
    let keys_col = Column::from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals_col = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.view().unwrap();

    let vals_view = vals_col.view().unwrap();
    let requests = [AggregationRequest {
        values: &vals_view,
        aggregations: &[Aggregation::Count { include_nulls: false }],
    }];

    let result =
        groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    let result_kv = result.keys.view().unwrap();
    let keys: Vec<i32> = result_kv.column(0).unwrap().to_vec::<i32>().unwrap();
    let counts: Vec<i32> = result.results[0][0].view().unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = keys.into_iter().zip(counts).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

#[test]
fn test_groupby_rejects_empty_aggregations_in_request() {
    let keys_col = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = Column::from_slice(&[10i32, 20, 30]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.view().unwrap();
    let vals_view = vals_col.view().unwrap();

    let requests = [AggregationRequest {
        values: &vals_view,
        aggregations: &[],
    }];

    let err = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude)
        .err()
        .expect("expected error");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_groupby_rejects_negative_ddof() {
    let keys_col = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = Column::from_slice(&[10f64, 20.0, 30.0]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.view().unwrap();
    let vals_view = vals_col.view().unwrap();

    let requests = [AggregationRequest {
        values: &vals_view,
        aggregations: &[Aggregation::Std { ddof: -1 }],
    }];

    let err = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude)
        .err()
        .expect("expected error");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_groupby_error_contains_operation_context() {
    let keys_col = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = Column::from_slice(&[10i32, 20, 30]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.view().unwrap();
    let vals_view = vals_col.view().unwrap();

    let requests = [AggregationRequest {
        values: &vals_view,
        aggregations: &[],
    }];

    let err = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude)
        .err()
        .expect("expected error");

    assert_eq!(err.operation(), "groupby::groupby_aggregate");
    let ctx = err.context();
    assert!(ctx.iter().any(|(k, _)| *k == "request_index"));
    assert!(ctx.iter().any(|(k, _)| *k == "num_requests"));
}

#[test]
fn test_groupby_multi_column_keys() {
    // Two key columns.
    let k0 = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let k1 = Column::from_slice(&[10i32, 20, 10, 20]).unwrap();
    let vals = Column::from_slice(&[100i64, 200, 300, 400]).unwrap();

    let keys_table = Table::from_columns(vec![k0, k1]).unwrap();
    let kv = keys_table.view().unwrap();

    let vals_view = vals.view().unwrap();
    let requests = [AggregationRequest {
        values: &vals_view,
        aggregations: &[Aggregation::Sum],
    }];

    let result =
        groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    let result_kv = result.keys.view().unwrap();
    assert_eq!(result_kv.num_columns().unwrap(), 2);
    // 4 unique key combinations.
    assert_eq!(result_kv.num_rows().unwrap(), 4);
}
