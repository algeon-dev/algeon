//! Tests for DataFrame groupby aggregation.

use cudf::{
    AggSpec, DataFrame, ErrorCode, ErrorKind, Exec, ExecutionBudgetEnforcement, ExecutionBudgetRequest, GroupByAgg,
    core::error::{FactKeyName, FactValue},
    frame::dataframe::InfoDetail,
    testing::{
        groupby::{
            trigger_groupby_duplicate_result_column_consumption_for_test,
            trigger_groupby_result_column_index_out_of_bounds_for_test,
            trigger_groupby_result_group_index_out_of_bounds_for_test,
        },
        prelude::*,
    },
};

use crate::common::{
    arm_next_groupby_negative_total_columns_fault, assert_missing_column_error, column_from_slice,
    dataframe_from_columns, error_context_value, execution_budget_token, last_exec_mr_value, reset_exec_observation,
    with_test_runtime,
};

fn assert_high_level_groupby_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

fn dataframe_device_allocated_size_bytes(df: &DataFrame) -> usize {
    df.info_with(InfoDetail::DEVICE_ALLOCATED).unwrap().memory_accounting.unwrap().device_allocated_bytes.unwrap()
}

fn assert_budget_report_high_water(report: &cudf::ExecutionBudgetReport, retained_or_output_bytes: usize) {
    let high_water = report.allocator_high_water_bytes().expect("budget report should include allocator high-water");
    assert!(high_water > 0, "allocator high-water should be positive: {report:?}");
    assert!(
        high_water >= retained_or_output_bytes,
        "allocator high-water should cover retained/output bytes: high_water={high_water}, retained_or_output={retained_or_output_bytes}, report={report:?}",
    );
    assert!(
        high_water <= report.request().budget_bytes(),
        "allocator high-water should stay within budget: high_water={high_water}, report={report:?}",
    );
    assert!(
        report.allocator_peak_aligned_bytes().is_some_and(|bytes| bytes <= report.request().budget_bytes()),
        "aligned allocator peak should stay within budget: {report:?}",
    );
}

fn groupby_sum_pairs(result: &DataFrame) -> Vec<(i32, i64)> {
    let key_data = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let sum_data = result.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(key, _)| key);
    pairs
}

fn assert_no_budget_error_details(err: &cudf::Error) {
    assert!(err.budget_exhaustion().is_none(), "an unbudgeted operation carries no budget scope");
}

// ---------------------------------------------------------------------------
// Groupby aggregation
// ---------------------------------------------------------------------------

fn test_groupby_single_key_sum() {
    // keys: [1, 2, 1, 2], values: [10, 20, 30, 40]
    // Expected: key=1 → sum=40, key=2 → sum=60
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();

    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

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

fn test_groupby_agg_with_budget_applies_token_reports_output_and_matches_agg() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();
    let specs = [AggSpec::new("value", GroupByAgg::Sum)];
    let expected = df.groupby(["key"]).on(Exec::RETAINED).agg(&specs).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("grouped_sum"))
            .unwrap()
    });

    reset_exec_observation();
    let outcome = token.run("groupby::agg", |exec| df.groupby(["key"]).on(exec).agg(&specs)).unwrap();
    let budgeted_exec_mr = last_exec_mr_value();
    let result = outcome.value();
    let report = outcome.report();
    let output_bytes = dataframe_device_allocated_size_bytes(result);

    assert_eq!(groupby_sum_pairs(result), groupby_sum_pairs(&expected));
    assert_eq!(report.request().boundary(), "grouped_sum");
    assert_eq!(report.request().budget_bytes(), 64 * 1024 * 1024);
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(report.output_rows(), Some(result.num_rows()));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_budget_report_high_water(report, output_bytes);
    assert_eq!(
        result.testing_execution_domain_mr_value(),
        token.memory_resource().testing_raw_handle_value(),
        "budgeted groupby output should retain the device manager resource",
    );
    assert_eq!(
        budgeted_exec_mr,
        token.memory_resource().testing_raw_handle_value(),
        "budgeted groupby execution should route through the device manager resource",
    );
}

fn test_prepare_sort_groupby_with_budget_reports_exact_group_facts_and_reuses_handle() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();
    let specs = [AggSpec::new("value", GroupByAgg::Sum)];
    let expected = df.groupby(["key"]).on(Exec::RETAINED).agg(&specs).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(
            runtime,
            ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("prepared_sort_groupby"),
        )
        .unwrap()
    });

    reset_exec_observation();
    let outcome = df.groupby(["key"]).prepare_sort_groupby_with_budget(&token).unwrap();
    let prepared_exec_mr = last_exec_mr_value();
    let prepared = outcome.value();
    let facts = prepared.prepared_facts();
    let report = outcome.report();

    assert_eq!(facts.num_groups, 2);
    assert!(facts.grouped_key_output_bytes > 0);
    assert_eq!(facts.max_group_size, 2);
    assert_eq!(report.request().boundary(), "prepared_sort_groupby");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(report.output_rows(), Some(facts.num_groups));
    assert_eq!(report.output_bytes(), None);
    assert_eq!(report.retained_bytes(), None);
    assert_eq!(
        report.context_facts().get("prepared_groupby_strategy").map(String::as_str),
        Some("prepared_sort_groupby")
    );
    assert_eq!(report.context_facts().get("prepared_groupby_num_groups").map(String::as_str), Some("2"));
    assert_eq!(
        report.context_facts().get("prepared_groupby_grouped_key_output_bytes").map(String::as_str),
        Some(facts.grouped_key_output_bytes.to_string().as_str())
    );
    assert_eq!(report.context_facts().get("prepared_groupby_max_group_size").map(String::as_str), Some("2"));
    assert_eq!(
        report.context_facts().get("prepared_groupby_memory_facts_status").map(String::as_str),
        Some("grouped_key_output_only_retained_state_unavailable")
    );
    assert_eq!(
        prepared_exec_mr,
        token.memory_resource().testing_raw_handle_value(),
        "budgeted prepared groupby should route preparation through the device manager resource",
    );

    reset_exec_observation();
    let agg_outcome = prepared.agg_with_budget(&specs, &token).unwrap();
    let prepared_agg_exec_mr = last_exec_mr_value();
    let result = agg_outcome.value();
    let agg_report = agg_outcome.report();
    let output_bytes = dataframe_device_allocated_size_bytes(result);
    assert_eq!(groupby_sum_pairs(result), groupby_sum_pairs(&expected));
    assert_eq!(agg_report.request().boundary(), "prepared_sort_groupby");
    assert_eq!(agg_report.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(agg_report.output_rows(), Some(result.num_rows()));
    assert_eq!(agg_report.output_bytes(), Some(output_bytes));
    assert_eq!(agg_report.retained_bytes(), Some(output_bytes));
    assert_budget_report_high_water(agg_report, output_bytes);
    assert_eq!(
        agg_report.context_facts().get("prepared_groupby_strategy").map(String::as_str),
        Some("prepared_sort_groupby")
    );
    assert_eq!(agg_report.context_facts().get("prepared_groupby_num_groups").map(String::as_str), Some("2"));
    assert_eq!(prepared_agg_exec_mr, token.memory_resource().testing_raw_handle_value());
    assert_eq!(
        result.testing_execution_domain_mr_value(),
        token.memory_resource().testing_raw_handle_value(),
        "budgeted prepared aggregate output should retain the device manager resource",
    );
}

fn test_groupby_q17_avg_quantity_with_budget_reports_current_diagnostics() {
    let partkeys = column_from_slice(&[10i32, 20, 10, 20, 30]).unwrap();
    let quantities = column_from_slice(&[4.0f64, 10.0, 8.0, 30.0, 5.0]).unwrap();
    let df =
        dataframe_from_columns(vec!["l_partkey".to_string(), "l_quantity".to_string()], vec![partkeys, quantities])
            .unwrap();
    let specs = [AggSpec::new("l_quantity", GroupByAgg::Mean)];
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("q17_grouped_avg"))
            .unwrap()
    });

    let outcome = token.run("groupby::agg", |exec| df.groupby(["l_partkey"]).on(exec).agg(&specs)).unwrap();
    let result = outcome.value();
    let report = outcome.report();

    let key_data = result.series("l_partkey").unwrap().to_vec_strict::<i32>().unwrap();
    let mean_data = result.series("l_quantity_mean").unwrap().to_vec_strict::<f64>().unwrap();
    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(key, _)| key);

    assert_eq!(pairs, vec![(10, 6.0), (20, 20.0), (30, 5.0)]);
    assert_eq!(report.request().boundary(), "q17_grouped_avg");
    assert_eq!(report.output_rows(), Some(result.num_rows()));
    let output_bytes = report.output_bytes().expect("groupby report should observe post-execution output bytes");
    let retained_bytes = report.retained_bytes().expect("groupby report should observe post-execution retained bytes");
    assert!(output_bytes > 0, "groupby output bytes are observed after execution, not preflighted");
    assert!(retained_bytes > 0, "groupby retained bytes are observed after execution, not preflighted");
}

fn test_groupby_agg_with_budget_reports_exceeded_budget_details() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(1).with_boundary("grouped_sum")).unwrap()
    });

    let err = token
        .run("groupby::agg", |exec| df.groupby(["key"]).on(exec).agg(&[AggSpec::new("value", GroupByAgg::Sum)]))
        .expect_err("one-byte budget should not cover grouped aggregate allocations");

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "groupby::agg");
    let budget = err.budget_exhaustion().expect("budgeted aggregate exposes its budget scope");
    assert_eq!(budget.boundary(), "grouped_sum");
    let usage = budget.usage().expect("the member was sampled");
    assert_eq!(usage.allocation_limit_bytes, 1);
    assert_eq!(usage.peak_bytes, 0);
    assert_eq!(usage.current_bytes, 0);
    assert_eq!(usage.failed_allocation_count, 1);
    assert!(usage.largest_failed_allocation_bytes > 0);
}

fn test_groupby_agg_with_budget_preserves_empty_specs_validation_without_budget_details() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("grouped_sum"))
            .unwrap()
    });

    let err = token
        .run("groupby::agg", |exec| df.groupby(["key"]).on(exec).agg(&[]))
        .expect_err("expected error for empty aggregation");

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::agg",
        "empty_agg_specs",
        "provide at least one AggSpec or use a shortcut like .sum([...])",
    );
    assert_no_budget_error_details(&err);
}

fn test_groupby_agg_with_budget_preserves_missing_column_validation_without_budget_details() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("grouped_sum"))
            .unwrap()
    });

    let err = token
        .run("groupby::agg", |exec| df.groupby(["key"]).on(exec).agg(&[AggSpec::new("missing_value", GroupByAgg::Sum)]))
        .expect_err("expected error for missing aggregation column");

    assert_missing_column_error(&err, "missing_value");
    assert_eq!(err.operation(), "groupby::agg");
    assert_no_budget_error_details(&err);
}

fn test_groupby_agg_with_budget_preserves_empty_keys_validation_without_budget_details() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("grouped_sum"))
            .unwrap()
    });

    let err = token
        .run("groupby::agg", |exec| {
            df.groupby(Vec::<&str>::new()).on(exec).agg(&[AggSpec::new("value", GroupByAgg::Sum)])
        })
        .expect_err("expected error for empty group keys");

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::agg",
        "empty_group_keys",
        "provide at least one group key column",
    );
    assert_no_budget_error_details(&err);
}

#[test]
fn test_groupby_agg_with_budget_preserves_foreign_output_invariant_without_budget_details() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("grouped_sum"))
            .unwrap()
    });

    arm_next_groupby_negative_total_columns_fault();
    let err = token
        .run("groupby::agg", |exec| df.groupby(["key"]).on(exec).agg(&[AggSpec::new("value", GroupByAgg::Sum)]))
        .expect_err("expected foreign invariant error");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "groupby::agg");
    assert_no_budget_error_details(&err);
}

fn test_groupby_rejects_empty_group_keys_with_stable_contract() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err =
        df.groupby(Vec::<&str>::new()).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap_err();

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::GroupBy::agg",
        "empty_group_keys",
        "provide at least one group key column",
    );
}

fn test_groupby_shortcuts_reject_empty_group_keys_with_stable_contract() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df.groupby(Vec::<&str>::new()).on(Exec::RETAINED).sum(["value"]).unwrap_err();

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::GroupBy::agg",
        "empty_group_keys",
        "provide at least one group key column",
    );
}

fn test_groupby_single_key_multiple_aggregations_same_column() {
    // keys: [1, 2, 1, 2], values: [10.0, 20.0, 30.0, 40.0]
    // Expected: key=1 → min=10, max=30; key=2 → min=20, max=40
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let df = dataframe_from_columns(vec!["key".to_string(), "score".to_string()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
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

fn test_groupby_multiple_keys() {
    // Two key columns: (k0, k1) with 4 unique combinations
    let k0 = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let k1 = column_from_slice(&[10i32, 20, 10, 20]).unwrap();
    let vals = column_from_slice(&[100i64, 200, 300, 400]).unwrap();

    let df = dataframe_from_columns(vec!["a".to_string(), "b".to_string(), "value".to_string()], vec![k0, k1, vals])
        .unwrap();

    let result = df.groupby(["a", "b"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    // Should have 3 columns: "a", "b", "value_sum"
    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.num_rows(), 4);

    let names = result.column_names();
    assert_eq!(names[0], "a");
    assert_eq!(names[1], "b");
    assert_eq!(names[2], "value_sum");
}

fn test_groupby_nonexistent_agg_column_returns_error() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();

    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("nonexistent", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent column");

    assert_missing_column_error(&err, "nonexistent");
}

fn test_groupby_nonexistent_key_column_returns_error() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();

    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["missing_key"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("value", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent key column");

    assert_missing_column_error(&err, "missing_key");
}

fn test_groupby_result_column_naming_convention() {
    // Verify naming: "{column}_{aggregation}"
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let prices = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();

    let df = dataframe_from_columns(vec!["id".to_string(), "price".to_string()], vec![keys, prices]).unwrap();

    let result = df
        .groupby(["id"])
        .on(Exec::RETAINED)
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

fn test_groupby_interleaved_multi_column_agg_ordering() {
    // Request order intentionally interleaves columns:
    //   a_sum, b_sum, a_min
    // Result columns and values must follow that exact order.
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = column_from_slice(&[1i64, 2, 3, 4]).unwrap();

    let df =
        dataframe_from_columns(vec!["key".to_string(), "a".to_string(), "b".to_string()], vec![keys, a, b]).unwrap();

    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
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

    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    arm_next_groupby_negative_total_columns_fault();
    let err = df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap_err();

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

fn test_groupby_no_aggregations_returns_error() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();

    let df = dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, vals]).unwrap();

    let err = df.groupby(["key"]).on(Exec::RETAINED).agg(&[]).expect_err("expected error for empty aggregation");

    assert_high_level_groupby_validation_error(
        &err,
        "groupby::GroupBy::agg",
        "empty_agg_specs",
        "provide at least one AggSpec or use a shortcut like .sum([...])",
    );
}

fn test_testing_hook_locks_groupby_result_group_index_out_of_bounds_contract() {
    let err = trigger_groupby_result_group_index_out_of_bounds_for_test().unwrap_err();

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

fn test_testing_hook_locks_groupby_result_column_index_out_of_bounds_contract() {
    let err = trigger_groupby_result_column_index_out_of_bounds_for_test().unwrap_err();

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

fn test_testing_hook_locks_groupby_duplicate_result_column_consumption_contract() {
    let err = trigger_groupby_duplicate_result_column_consumption_for_test().unwrap_err();

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

pub(super) const CASES: &[(&str, fn())] = &[
    ("dataframe_groupby_tests::test_groupby_single_key_sum", test_groupby_single_key_sum),
    (
        "dataframe_groupby_tests::test_groupby_agg_with_budget_applies_token_reports_output_and_matches_agg",
        test_groupby_agg_with_budget_applies_token_reports_output_and_matches_agg,
    ),
    (
        "dataframe_groupby_tests::test_prepare_sort_groupby_with_budget_reports_exact_group_facts_and_reuses_handle",
        test_prepare_sort_groupby_with_budget_reports_exact_group_facts_and_reuses_handle,
    ),
    (
        "dataframe_groupby_tests::test_groupby_q17_avg_quantity_with_budget_reports_current_diagnostics",
        test_groupby_q17_avg_quantity_with_budget_reports_current_diagnostics,
    ),
    (
        "dataframe_groupby_tests::test_groupby_agg_with_budget_reports_exceeded_budget_details",
        test_groupby_agg_with_budget_reports_exceeded_budget_details,
    ),
    (
        "dataframe_groupby_tests::test_groupby_agg_with_budget_preserves_empty_specs_validation_without_budget_details",
        test_groupby_agg_with_budget_preserves_empty_specs_validation_without_budget_details,
    ),
    (
        "dataframe_groupby_tests::test_groupby_agg_with_budget_preserves_missing_column_validation_without_budget_details",
        test_groupby_agg_with_budget_preserves_missing_column_validation_without_budget_details,
    ),
    (
        "dataframe_groupby_tests::test_groupby_agg_with_budget_preserves_empty_keys_validation_without_budget_details",
        test_groupby_agg_with_budget_preserves_empty_keys_validation_without_budget_details,
    ),
    (
        "dataframe_groupby_tests::test_groupby_rejects_empty_group_keys_with_stable_contract",
        test_groupby_rejects_empty_group_keys_with_stable_contract,
    ),
    (
        "dataframe_groupby_tests::test_groupby_shortcuts_reject_empty_group_keys_with_stable_contract",
        test_groupby_shortcuts_reject_empty_group_keys_with_stable_contract,
    ),
    (
        "dataframe_groupby_tests::test_groupby_single_key_multiple_aggregations_same_column",
        test_groupby_single_key_multiple_aggregations_same_column,
    ),
    ("dataframe_groupby_tests::test_groupby_multiple_keys", test_groupby_multiple_keys),
    (
        "dataframe_groupby_tests::test_groupby_nonexistent_agg_column_returns_error",
        test_groupby_nonexistent_agg_column_returns_error,
    ),
    (
        "dataframe_groupby_tests::test_groupby_nonexistent_key_column_returns_error",
        test_groupby_nonexistent_key_column_returns_error,
    ),
    (
        "dataframe_groupby_tests::test_groupby_result_column_naming_convention",
        test_groupby_result_column_naming_convention,
    ),
    (
        "dataframe_groupby_tests::test_groupby_interleaved_multi_column_agg_ordering",
        test_groupby_interleaved_multi_column_agg_ordering,
    ),
    ("dataframe_groupby_tests::test_groupby_no_aggregations_returns_error", test_groupby_no_aggregations_returns_error),
    (
        "dataframe_groupby_tests::test_testing_hook_locks_groupby_result_group_index_out_of_bounds_contract",
        test_testing_hook_locks_groupby_result_group_index_out_of_bounds_contract,
    ),
    (
        "dataframe_groupby_tests::test_testing_hook_locks_groupby_result_column_index_out_of_bounds_contract",
        test_testing_hook_locks_groupby_result_column_index_out_of_bounds_contract,
    ),
    (
        "dataframe_groupby_tests::test_testing_hook_locks_groupby_duplicate_result_column_consumption_contract",
        test_testing_hook_locks_groupby_duplicate_result_column_consumption_contract,
    ),
];
