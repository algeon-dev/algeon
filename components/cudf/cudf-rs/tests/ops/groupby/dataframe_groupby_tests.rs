//! Tests for DataFrame groupby aggregation.

use cudf::{
    AggSpec, ErrorCode, ErrorKind, GroupByAgg,
    core::error::{FactKeyName, FactValue},
};

use crate::common::{
    arm_next_groupby_negative_total_columns_fault, assert_missing_column_error, column_from_slice, error_context_value,
};

fn assert_high_level_groupby_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

// ---------------------------------------------------------------------------
// Groupby aggregation
// ---------------------------------------------------------------------------

#[test]
fn test_groupby_single_key_sum() {
    // keys: [1, 2, 1, 2], values: [10, 20, 30, 40]
    // Expected: key=1 → sum=40, key=2 → sum=60
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    // Should have 2 columns: "key" and "value_sum"
    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 2);

    // Check schema names
    let names = result.column_names();
    assert_eq!(names[0], "key");
    assert_eq!(names[1], "value_sum");

    // Verify data (sort by key for determinism)
    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let sum_data: Vec<i64> = result.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

#[test]
fn test_groupby_rejects_empty_group_keys_with_stable_contract() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df.groupby(Vec::<&str>::new()).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap_err();

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::GroupBy::agg",
        "empty_group_keys",
        "provide at least one group key column",
    );
}

#[test]
fn test_groupby_shortcuts_reject_empty_group_keys_with_stable_contract() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df.groupby(Vec::<&str>::new()).sum(["value"]).unwrap_err();

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::GroupBy::agg",
        "empty_group_keys",
        "provide at least one group key column",
    );
}

#[test]
fn test_groupby_single_key_multiple_aggregations_same_column() {
    // keys: [1, 2, 1, 2], values: [10.0, 20.0, 30.0, 40.0]
    // Expected: key=1 → min=10, max=30; key=2 → min=20, max=40
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "score".to_string()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .agg(&[AggSpec::new("score", GroupByAgg::Min), AggSpec::new("score", GroupByAgg::Max)])
        .unwrap();

    // Should have 3 columns: "key", "score_min", "score_max"
    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.num_rows(), 2);

    let names = result.column_names();
    assert_eq!(names[0], "key");
    assert_eq!(names[1], "score_min");
    assert_eq!(names[2], "score_max");

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let min_data: Vec<f64> = result.series("score_min").unwrap().to_vec_strict::<f64>().unwrap();
    let max_data: Vec<f64> = result.series("score_max").unwrap().to_vec_strict::<f64>().unwrap();

    let mut triples: Vec<(i32, f64, f64)> =
        key_data.into_iter().zip(min_data.into_iter().zip(max_data)).map(|(k, (mn, mx))| (k, mn, mx)).collect();
    triples.sort_by_key(|&(k, _, _)| k);

    assert_eq!(triples, vec![(1, 10.0, 30.0), (2, 20.0, 40.0)]);
}

#[test]
fn test_groupby_multiple_keys() {
    // Two key columns: (k0, k1) with 4 unique combinations
    let k0 = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let k1 = column_from_slice(&[10i32, 20, 10, 20]).unwrap();
    let vals = column_from_slice(&[100i64, 200, 300, 400]).unwrap();

    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "value".to_string()],
        vec![k0, k1, vals],
    )
    .unwrap();

    let result = df.groupby(["a", "b"]).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    // Should have 3 columns: "a", "b", "value_sum"
    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.num_rows(), 4);

    let names = result.column_names();
    assert_eq!(names[0], "a");
    assert_eq!(names[1], "b");
    assert_eq!(names[2], "value_sum");
}

#[test]
fn test_groupby_nonexistent_agg_column_returns_error() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["key"])
        .agg(&[AggSpec::new("nonexistent", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent column");

    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_groupby_nonexistent_key_column_returns_error() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["missing_key"])
        .agg(&[AggSpec::new("value", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent key column");

    assert_missing_column_error(&err, "missing_key");
}

#[test]
fn test_groupby_result_column_naming_convention() {
    // Verify naming: "{column}_{aggregation}"
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let prices = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "price".to_string()], vec![keys, prices]).unwrap();

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

#[test]
fn test_groupby_interleaved_multi_column_agg_ordering() {
    // Request order intentionally interleaves columns:
    //   a_sum, b_sum, a_min
    // Result columns and values must follow that exact order.
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = column_from_slice(&[1i64, 2, 3, 4]).unwrap();

    let df = crate::common::dataframe_from_columns(
        vec!["key".to_string(), "a".to_string(), "b".to_string()],
        vec![keys, a, b],
    )
    .unwrap();

    let result = df
        .groupby(["key"])
        .agg(&[
            AggSpec::new("a", GroupByAgg::Sum),
            AggSpec::new("b", GroupByAgg::Sum),
            AggSpec::new("a", GroupByAgg::Min),
        ])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);

    let keys = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let a_sum = result.series("a_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let b_sum = result.series("b_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let a_min = result.series("a_min").unwrap().to_vec_strict::<i64>().unwrap();

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
fn test_groupby_high_level_rejects_negative_foreign_result_column_count_with_stable_contract() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    arm_next_groupby_negative_total_columns_fault();
    let err = df.groupby(["key"]).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "groupby::GroupBy::agg");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("result_columns"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("negative_column_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign groupby backend returns a non-negative result column count")
    );
    assert_eq!(error_context_value(&err, "reported_table_column_count").as_deref(), Some("-1"));
    assert_eq!(error_context_value(&err, "out_total_columns").as_deref(), None);
}

#[test]
fn test_groupby_no_aggregations_returns_error() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df.groupby(["key"]).agg(&[]).expect_err("expected error for empty aggregation");

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::GroupBy::agg",
        "empty_agg_specs",
        "provide at least one AggSpec or use a shortcut like .sum([...])",
    );
}

#[test]
fn test_testing_hook_locks_groupby_result_group_index_out_of_bounds_contract() {
    let err = cudf::testing::groupby::trigger_groupby_result_group_index_out_of_bounds_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "groupby::GroupBy::agg");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("result_group_index_out_of_bounds"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "ensure the foreign groupby backend returns one result column per requested aggregation in stable request order"
        )
    );
    assert_eq!(error_context_value(&err, "request_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "group_index").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "group_count").as_deref(), Some("1"));
}

#[test]
fn test_testing_hook_locks_groupby_result_column_index_out_of_bounds_contract() {
    let err = cudf::testing::groupby::trigger_groupby_result_column_index_out_of_bounds_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "groupby::GroupBy::agg");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("result_column_index_out_of_bounds"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "ensure the foreign groupby backend returns one result column per requested aggregation in stable request order"
        )
    );
    assert_eq!(error_context_value(&err, "request_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "group_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "agg_index").as_deref(), Some("1"));
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(1)));
    assert_eq!(error_context_value(&err, "result_column_count").as_deref(), Some("1"));
}

#[test]
fn test_testing_hook_locks_groupby_duplicate_result_column_consumption_contract() {
    let err = cudf::testing::groupby::trigger_groupby_duplicate_result_column_consumption_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "groupby::GroupBy::agg");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("duplicate_result_column_consumption"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "ensure the foreign groupby backend returns one result column per requested aggregation in stable request order"
        )
    );
    assert_eq!(error_context_value(&err, "request_index").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "group_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "agg_index").as_deref(), Some("0"));
}
