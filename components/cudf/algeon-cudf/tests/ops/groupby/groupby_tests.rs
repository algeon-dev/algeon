//! Tests for groupby operations.
use cudf::{
    ErrorCode, ErrorKind,
    core::Table,
    ops::groupby::{self, AggregationRequest, GroupByAgg, NullPolicy, PreparedGroupBy},
    testing::{
        groupby::{
            trigger_groupby_request_agg_count_limit_error_for_test, trigger_groupby_request_count_limit_error_for_test,
            trigger_groupby_total_agg_handle_count_limit_error_for_test,
        },
        prelude::*,
    },
};

use crate::common::{
    arm_next_groupby_negative_total_columns_fault, column_from_slice, column_from_slice_nullable,
    dataframe_from_columns, error_context_value, explicit_i32_column, last_exec_mr_value, last_exec_stream_value,
    last_sync_stream_value, reset_exec_observation, reset_sync_observation, run_named_cases,
};

fn assert_groupby_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

fn assert_ddof_range_validation_error(err: &cudf::Error, operation: &str, request_index: &str, expected_ddof: u32) {
    assert_groupby_validation_error(
        err,
        operation,
        "ddof_exceeds_i32_range",
        "use a ddof value within libcudf's i32 range",
    );
    assert_eq!(error_context_value(err, "request_index").as_deref(), Some(request_index));
    assert_eq!(error_context_value(err, "ddof").as_deref(), Some(expected_ddof.to_string().as_str()));
}

fn test_testing_hook_locks_groupby_request_count_limit_contract() {
    let err = trigger_groupby_request_count_limit_error_for_test().unwrap_err();

    assert_groupby_validation_error(
        &err,
        "groupby::groupby_aggregate",
        "exceeds_i32_max",
        "reduce the number of aggregation requests to fit within libcudf's i32 range",
    );
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("aggregation_requests"));
    assert_eq!(error_context_value(&err, "request_count").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "num_requests").as_deref(), None);
}

fn test_testing_hook_locks_groupby_request_agg_count_limit_contract() {
    let err = trigger_groupby_request_agg_count_limit_error_for_test().unwrap_err();

    assert_groupby_validation_error(
        &err,
        "groupby::groupby_aggregate",
        "exceeds_i32_max",
        "reduce the number of aggregations in this request to fit within libcudf's i32 range",
    );
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("request_aggregations"));
    assert_eq!(error_context_value(&err, "request_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "request_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "aggregation_count").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "num_requests").as_deref(), None);
    assert_eq!(error_context_value(&err, "agg_count").as_deref(), None);
}

fn test_testing_hook_locks_groupby_total_agg_handle_count_limit_contract() {
    let err = trigger_groupby_total_agg_handle_count_limit_error_for_test().unwrap_err();

    assert_groupby_validation_error(
        &err,
        "groupby::groupby_aggregate",
        "exceeds_i32_max",
        "reduce the total number of aggregation handles to fit within libcudf's i32 range",
    );
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("aggregation_handles"));
    assert_eq!(error_context_value(&err, "request_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "aggregation_handle_count").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "num_requests").as_deref(), None);
    assert_eq!(error_context_value(&err, "total_agg_handles").as_deref(), None);
}

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

fn test_groupby_aggregate_sorted_uses_sorted_key_contract() {
    // Keys are ascending with duplicate groups contiguous, matching cuDF's
    // sorted-key groupby aggregate contract.
    let keys_col = column_from_slice(&[1i32, 1, 2, 2, 3]).unwrap();
    let vals_col = column_from_slice(&[10i64, 30, 20, 40, 5]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();

    let vals_view = vals_col.testing_view().unwrap();
    let requests = [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Sum] }];

    let result = groupby::groupby_aggregate_sorted(&kv, &requests, NullPolicy::Exclude).unwrap();

    let keys: Vec<i32> = result.keys.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let sums: Vec<i64> = result.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap();
    let mut pairs: Vec<(i32, i64)> = keys.into_iter().zip(sums).collect();
    pairs.sort_by_key(|&(key, _)| key);

    assert_eq!(pairs, vec![(1, 40), (2, 60), (3, 5)]);
}

fn test_prepared_groupby_reuses_key_state_for_multiple_aggregations() {
    let keys_col = column_from_slice(&[1i32, 2, 1, 3, 2, 3]).unwrap();
    let left_vals = column_from_slice(&[10i64, 20, 30, 40, 50, 60]).unwrap();
    let right_vals = column_from_slice(&[1i64, 2, 3, 4, 5, 6]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let keys_view = keys_table.testing_view().unwrap();
    let prepared = PreparedGroupBy::new(&keys_view, NullPolicy::Exclude).unwrap();
    let prepared_facts = prepared.prepare_groups().unwrap();
    assert_eq!(prepared_facts.num_groups, 3);
    assert!(prepared_facts.grouped_key_output_bytes > 0);
    assert_eq!(prepared_facts.max_group_size, 2);

    let left_view = left_vals.testing_view().unwrap();
    let left_requests = [AggregationRequest { values: &left_view, aggregations: &[GroupByAgg::Sum] }];
    let left_result = prepared.aggregate(&left_requests).unwrap();

    let right_view = right_vals.testing_view().unwrap();
    let right_requests =
        [AggregationRequest { values: &right_view, aggregations: &[GroupByAgg::Min, GroupByAgg::Max] }];
    let right_result = prepared.aggregate(&right_requests).unwrap();

    let left_keys: Vec<i32> = left_result.keys.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let left_sums: Vec<i64> = left_result.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap();
    let mut left_pairs: Vec<(i32, i64)> = left_keys.into_iter().zip(left_sums).collect();
    left_pairs.sort_by_key(|&(key, _)| key);
    assert_eq!(left_pairs, vec![(1, 40), (2, 70), (3, 100)]);

    let right_keys: Vec<i32> = right_result.keys.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    let mins: Vec<i64> = right_result.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap();
    let maxes: Vec<i64> = right_result.results[0][1].testing_view().unwrap().to_vec::<i64>().unwrap();
    let mut triples: Vec<(i32, i64, i64)> =
        right_keys.into_iter().zip(mins.into_iter().zip(maxes)).map(|(key, (min, max))| (key, min, max)).collect();
    triples.sort_by_key(|&(key, _, _)| key);
    assert_eq!(triples, vec![(1, 1, 3), (2, 2, 5), (3, 4, 6)]);
}

fn test_prepared_groupby_preserves_include_null_key_contract() {
    let keys_col = column_from_slice_nullable(&[1i32, 0, 1, 0, 2], &[true, false, true, false, true]).unwrap();
    let vals_col = column_from_slice(&[10i64, 20, 30, 40, 5]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let keys_view = keys_table.testing_view().unwrap();
    let prepared = PreparedGroupBy::new(&keys_view, NullPolicy::Include).unwrap();
    let prepared_facts = prepared.prepare_groups().unwrap();
    assert_eq!(prepared_facts.num_groups, 3);
    assert!(prepared_facts.grouped_key_output_bytes > 0);
    assert_eq!(prepared_facts.max_group_size, 2);
    let vals_view = vals_col.testing_view().unwrap();
    let requests = [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Sum] }];

    let result = prepared.aggregate(&requests).unwrap();

    let keys = result.keys.testing_view().unwrap().column(0).unwrap().to_optional_vec::<i32>().unwrap();
    let sums = result.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap();
    let mut pairs: Vec<(Option<i32>, i64)> = keys.into_iter().zip(sums).collect();
    pairs.sort_by_key(|&(key, _)| key);
    assert_eq!(pairs, vec![(None, 60), (Some(1), 40), (Some(2), 5)]);
}

fn test_dataframe_groupby_prepare_groups_reports_null_policy_cardinality() {
    let keys = column_from_slice_nullable(&[1i32, 0, 1, 0, 2], &[true, false, true, false, true]).unwrap();
    let values = column_from_slice(&[10i64, 20, 30, 40, 5]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, values]).unwrap();

    let exclude_facts = df.groupby(["key"]).prepare_groups().unwrap();
    assert_eq!(exclude_facts.num_groups, 2);
    assert!(exclude_facts.grouped_key_output_bytes > 0);
    assert_eq!(exclude_facts.max_group_size, 2);

    let include_facts = df
        .groupby(["key"])
        .options(cudf::GroupByOptions::builder().include_null_keys(true).build())
        .prepare_groups()
        .unwrap();
    assert_eq!(include_facts.num_groups, 3);
    assert!(include_facts.grouped_key_output_bytes > 0);
    assert_eq!(include_facts.max_group_size, 2);
}

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
    assert_groupby_validation_error(
        &err,
        "groupby::groupby_aggregate",
        "empty_request_aggregations",
        "provide at least one GroupByAgg per AggregationRequest",
    );
}

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

fn test_groupby_rejects_ddof_above_i32_max_with_stable_contract() {
    let keys_col = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = column_from_slice(&[10f64, 20.0, 30.0]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();
    let vals_view = vals_col.testing_view().unwrap();

    let requests =
        [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Variance { ddof: u32::MAX }] }];

    let err = match groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude) {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };

    assert_ddof_range_validation_error(&err, "groupby::groupby_aggregate", "0", u32::MAX);
}

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

    assert_groupby_validation_error(
        &err,
        "groupby::groupby_aggregate",
        "empty_request_aggregations",
        "provide at least one GroupByAgg per AggregationRequest",
    );
    assert_eq!(error_context_value(&err, "request_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "request_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "num_requests").as_deref(), None);
}

#[test]
fn test_groupby_rejects_negative_foreign_result_column_count_with_stable_contract() {
    let keys_col = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals_col = column_from_slice(&[10i64, 20, 30]).unwrap();

    let keys_table = Table::from_columns(vec![keys_col]).unwrap();
    let kv = keys_table.testing_view().unwrap();
    let vals_view = vals_col.testing_view().unwrap();
    let requests = [AggregationRequest { values: &vals_view, aggregations: &[GroupByAgg::Sum] }];

    arm_next_groupby_negative_total_columns_fault();
    let err = match groupby::groupby_aggregate(&kv, &requests, NullPolicy::Exclude) {
        Ok(_) => panic!("expected malformed foreign result metadata to fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "groupby::groupby_aggregate");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("result_columns"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("negative_column_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign groupby backend returns a non-negative result column count")
    );
    assert_eq!(error_context_value(&err, "reported_table_column_count").as_deref(), Some("-1"));
    assert_eq!(error_context_value(&err, "out_total_columns").as_deref(), None);
}

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

#[test]
fn groupby_tests_cases() {
    run_named_cases(&[
        (
            "test_testing_hook_locks_groupby_request_count_limit_contract",
            test_testing_hook_locks_groupby_request_count_limit_contract,
        ),
        (
            "test_testing_hook_locks_groupby_request_agg_count_limit_contract",
            test_testing_hook_locks_groupby_request_agg_count_limit_contract,
        ),
        (
            "test_testing_hook_locks_groupby_total_agg_handle_count_limit_contract",
            test_testing_hook_locks_groupby_total_agg_handle_count_limit_contract,
        ),
        ("test_groupby_sum", test_groupby_sum),
        (
            "test_groupby_aggregate_sorted_uses_sorted_key_contract",
            test_groupby_aggregate_sorted_uses_sorted_key_contract,
        ),
        (
            "test_prepared_groupby_reuses_key_state_for_multiple_aggregations",
            test_prepared_groupby_reuses_key_state_for_multiple_aggregations,
        ),
        (
            "test_prepared_groupby_preserves_include_null_key_contract",
            test_prepared_groupby_preserves_include_null_key_contract,
        ),
        (
            "test_dataframe_groupby_prepare_groups_reports_null_policy_cardinality",
            test_dataframe_groupby_prepare_groups_reports_null_policy_cardinality,
        ),
        ("test_groupby_multiple_aggregations", test_groupby_multiple_aggregations),
        (
            "test_groupby_multiple_requests_preserve_request_grouping_and_agg_order",
            test_groupby_multiple_requests_preserve_request_grouping_and_agg_order,
        ),
        ("test_groupby_count", test_groupby_count),
        ("test_groupby_rejects_empty_aggregations_in_request", test_groupby_rejects_empty_aggregations_in_request),
        (
            "test_groupby_std_accepts_shared_non_negative_ddof_type_boundary",
            test_groupby_std_accepts_shared_non_negative_ddof_type_boundary,
        ),
        (
            "test_groupby_rejects_ddof_above_i32_max_with_stable_contract",
            test_groupby_rejects_ddof_above_i32_max_with_stable_contract,
        ),
        ("test_groupby_error_contains_operation_context", test_groupby_error_contains_operation_context),
        ("test_groupby_multi_column_keys", test_groupby_multi_column_keys),
        (
            "test_groupby_succeeds_with_mixed_domains_and_uses_keys_provenance",
            test_groupby_succeeds_with_mixed_domains_and_uses_keys_provenance,
        ),
    ]);
}
