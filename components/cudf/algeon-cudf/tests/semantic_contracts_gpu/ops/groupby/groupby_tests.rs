//! Tests for groupby operations.
use cudf::{
    AggSpec, ErrorCode, ErrorKind, Exec, GroupByAgg, GroupByOptions,
    testing::groupby::{
        trigger_groupby_request_agg_count_limit_error_for_test, trigger_groupby_request_count_limit_error_for_test,
        trigger_groupby_total_agg_handle_count_limit_error_for_test,
    },
};

use crate::common::{column_from_slice, column_from_slice_nullable, dataframe_from_columns, error_context_value};

fn assert_groupby_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
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

fn test_groupby_sorted_keys_option_aggregates_contiguous_groups() {
    // Keys are ascending with duplicate groups contiguous, matching cuDF's
    // sorted-key groupby aggregate contract.
    let df = dataframe_from_columns(
        vec!["key".into(), "value".into()],
        vec![column_from_slice(&[1i32, 1, 2, 2, 3]).unwrap(), column_from_slice(&[10i64, 30, 20, 40, 5]).unwrap()],
    )
    .unwrap();

    let result = df
        .groupby(["key"])
        .options(GroupByOptions::builder().keys_are_sorted(true).build())
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("value", GroupByAgg::Sum)])
        .unwrap();

    let keys = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let sums = result.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let mut pairs: Vec<(i32, i64)> = keys.into_iter().zip(sums).collect();
    pairs.sort_by_key(|&(key, _)| key);

    assert_eq!(pairs, vec![(1, 40), (2, 60), (3, 5)]);
}

fn test_dataframe_groupby_prepare_groups_reports_null_policy_cardinality() {
    let keys = column_from_slice_nullable(&[1i32, 0, 1, 0, 2], &[true, false, true, false, true]).unwrap();
    let values = column_from_slice(&[10i64, 20, 30, 40, 5]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, values]).unwrap();

    let exclude_facts = df.groupby(["key"]).on(Exec::RETAINED).prepare_groups().unwrap();
    assert_eq!(exclude_facts.num_groups, 2);
    assert!(exclude_facts.grouped_key_output_bytes > 0);
    assert_eq!(exclude_facts.max_group_size, 2);

    let include_facts = df
        .groupby(["key"])
        .options(cudf::GroupByOptions::builder().include_null_keys(true).build())
        .on(Exec::RETAINED)
        .prepare_groups()
        .unwrap();
    assert_eq!(include_facts.num_groups, 3);
    assert!(include_facts.grouped_key_output_bytes > 0);
    assert_eq!(include_facts.max_group_size, 2);
}

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "groupby_tests::test_testing_hook_locks_groupby_request_count_limit_contract",
        test_testing_hook_locks_groupby_request_count_limit_contract,
    ),
    (
        "groupby_tests::test_testing_hook_locks_groupby_request_agg_count_limit_contract",
        test_testing_hook_locks_groupby_request_agg_count_limit_contract,
    ),
    (
        "groupby_tests::test_testing_hook_locks_groupby_total_agg_handle_count_limit_contract",
        test_testing_hook_locks_groupby_total_agg_handle_count_limit_contract,
    ),
    (
        "groupby_tests::test_groupby_sorted_keys_option_aggregates_contiguous_groups",
        test_groupby_sorted_keys_option_aggregates_contiguous_groups,
    ),
    (
        "groupby_tests::test_dataframe_groupby_prepare_groups_reports_null_policy_cardinality",
        test_dataframe_groupby_prepare_groups_reports_null_policy_cardinality,
    ),
];
