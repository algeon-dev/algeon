//! Tests for groupby operations.
use cudf::{
    core::Table,
    ops::groupby::{self, AggregationRequest, GroupByAgg, NullPolicy},
    testing::prelude::*,
};

use crate::common::{
    column_from_slice, explicit_i32_column, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    reset_exec_observation, reset_sync_observation,
};

#[test]
fn test_groupby_sum() {
    // keys: [1, 2, 1, 2], values: [10, 20, 30, 40]
    // Expected groups: key=1 -> sum=40, key=2 -> sum=60
    let keys_col = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals_col = column_from_slice(&[10i64, 20, 30, 40]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();

    let vals_view = vals_col.testing_view().unwrap();
    let requests = [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Sum] }];

    let result = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    // Check keys.
    let result_kv = result.keys.testing_view().unwrap();
    assert_eq!(result_kv.num_rows().unwrap(), 2);

    let mut key_vals: Vec<i32> = result_kv.column(0).unwrap().to_vec::<i32>().unwrap();

    // Get sums.
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 1);
    let mut sum_vals: Vec<i64> = result.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap();

    // Sort by key for deterministic comparison.
    let mut pairs: Vec<(i32, i64)> = key_vals.drain(..).zip(sum_vals.drain(..)).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn test_groupby_multiple_aggregations() {
    let keys_col = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals_col = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();

    let vals_view = vals_col.testing_view().unwrap();
    let requests = [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Min, GroupByAgg::Max] }];

    let result = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    let result_kv = result.keys.testing_view().unwrap();
    assert_eq!(result_kv.num_rows().unwrap(), 2);

    // We have 1 request with 2 aggregations -> results[0] has 2 columns.
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 2);

    let keys: Vec<i32> = result_kv.column(0).unwrap().to_vec::<i32>().unwrap();
    let mins: Vec<f64> = result.results[0][0].testing_view().unwrap().to_vec::<f64>().unwrap();
    let maxs: Vec<f64> = result.results[0][1].testing_view().unwrap().to_vec::<f64>().unwrap();

    // Build sorted pairs.
    let mut triples: Vec<(i32, f64, f64)> =
        keys.into_iter().zip(mins.into_iter().zip(maxs)).map(|(k, (mn, mx))| (k, mn, mx)).collect();
    triples.sort_by_key(|&(k, _, _)| k);

    assert_eq!(triples, vec![(1, 10.0, 30.0), (2, 20.0, 40.0)]);
}

#[test]
fn test_groupby_multiple_requests_preserve_request_grouping_and_agg_order() {
    let keys_col = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let left_vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let right_vals = column_from_slice(&[3i64, 4, 1, 2]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();

    let left_view = left_vals.testing_view().unwrap();
    let right_view = right_vals.testing_view().unwrap();
    let requests = [
        AggregationRequest { values: &left_view, aggregations: &[GroupByAgg::Sum, GroupByAgg::Min] },
        AggregationRequest { values: &right_view, aggregations: &[GroupByAgg::Max] },
    ];

    let result = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    assert_eq!(result.results.len(), 2);
    assert_eq!(result.results[0].len(), 2);
    assert_eq!(result.results[1].len(), 1);

    let keys: Vec<i32> = result.keys.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let left_sum: Vec<i64> = result.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap();
    let left_min: Vec<i64> = result.results[0][1].testing_view().unwrap().to_vec::<i64>().unwrap();
    let right_max: Vec<i64> = result.results[1][0].testing_view().unwrap().to_vec::<i64>().unwrap();

    let mut rows: Vec<(i32, i64, i64, i64)> = keys
        .into_iter()
        .zip(left_sum.into_iter().zip(left_min.into_iter().zip(right_max)))
        .map(|(key, (sum, (min, max)))| (key, sum, min, max))
        .collect();
    rows.sort_by_key(|&(key, _, _, _)| key);

    assert_eq!(rows, vec![(1, 40, 10, 3), (2, 60, 20, 4)]);
}

#[test]
fn test_groupby_count() {
    let keys_col = column_from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals_col = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();

    let vals_view = vals_col.testing_view().unwrap();
    let requests =
        [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Count { include_nulls: false }] }];

    let result = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    let result_kv = result.keys.testing_view().unwrap();
    let keys: Vec<i32> = result_kv.column(0).unwrap().to_vec::<i32>().unwrap();
    let counts: Vec<i32> = result.results[0][0].testing_view().unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = keys.into_iter().zip(counts).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

#[test]
fn test_groupby_rejects_empty_aggregations_in_request() {
    let keys_col = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = column_from_slice(&[10i32, 20, 30]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();
    let vals_view = vals_col.testing_view().unwrap();

    let requests = [AggregationRequest { values: &vals_view, aggregations: &[] }];

    let err = match groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude) {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_groupby_std_accepts_shared_non_negative_ddof_type_boundary() {
    let keys_col = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = column_from_slice(&[10f64, 20.0, 30.0]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();
    let vals_view = vals_col.testing_view().unwrap();

    let requests = [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Std { ddof: 1u32 }] }];

    let result = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].len(), 1);
}

#[test]
fn test_groupby_error_contains_operation_context() {
    let keys_col = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = column_from_slice(&[10i32, 20, 30]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();
    let vals_view = vals_col.testing_view().unwrap();

    let requests = [AggregationRequest { values: &vals_view, aggregations: &[] }];

    let err = match groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude) {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };

    assert_eq!(err.operation(), "groupby::groupby_aggregate");
    let ctx = err.context();
    assert!(ctx.iter().any(|(k, _)| *k == "request_index"));
    assert!(ctx.iter().any(|(k, _)| *k == "num_requests"));
}

#[test]
fn test_groupby_multi_column_keys() {
    // Two key columns.
    let k0 = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let k1 = column_from_slice(&[10i32, 20, 10, 20]).unwrap();
    let vals = column_from_slice(&[100i64, 200, 300, 400]).unwrap();

    let keys_table = Table::from_columns(vec![k0, k1]).unwrap();
    let kv = keys_table.testing_view().unwrap();

    let vals_view = vals.testing_view().unwrap();
    let requests = [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Sum] }];

    let result = groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude).unwrap();

    let result_kv = result.keys.testing_view().unwrap();
    assert_eq!(result_kv.num_columns().unwrap(), 2);
    // 4 unique key combinations.
    assert_eq!(result_kv.num_rows().unwrap(), 4);
}

#[test]
fn test_groupby_succeeds_with_mixed_domains_and_uses_keys_provenance() {
    let keys = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 2], 970)]).unwrap();
    let values = explicit_i32_column(&[10, 20, 30, 40], 971);
    let values_view = values.testing_view().unwrap();
    let requests = [AggregationRequest { values: &values_view, aggregations: &[GroupByAgg::Sum] }];
    let expected_stream = keys.testing_execution_domain_stream_value();
    let expected_mr = keys.testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = groupby::groupby_aggregate(&keys.testing_view().unwrap(), &requests, NullPolicy::Exclude).unwrap();

    assert_eq!(result.keys.testing_execution_domain_id(), keys.testing_execution_domain_id());
    assert_eq!(result.keys.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.results[0][0].testing_execution_domain_id(), keys.testing_execution_domain_id());
    assert_eq!(result.results[0][0].testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    let key_host = result.keys.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let sum_host = result.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap();
    assert_eq!(key_host, vec![1, 2]);
    assert_eq!(sum_host, vec![30, 70]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}
