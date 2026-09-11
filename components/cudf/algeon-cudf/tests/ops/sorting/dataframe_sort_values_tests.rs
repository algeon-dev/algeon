//! Tests for `SortOptions` and the high-level `DataFrame::sort_values`.

use cudf::{
    DataFrame, ErrorCode, ErrorKind, ExecutionBudgetEnforcement, ExecutionBudgetRequest, ExecutionBudgetStatus,
    SortOptions,
    core::error::{FactKeyName, FactValue},
    frame::dataframe::InfoDetail,
    ops::sorting,
    testing::prelude::*,
};

use crate::common::{
    assert_missing_column_error, column_from_slice, column_from_slice_nullable, error_context_value,
    last_exec_mr_value, observed_exec_mr_values, reset_exec_observation, sample_df, with_test_runtime,
};

fn assert_sort_options_validation_error(
    err: &cudf::Error,
    operation: &str,
    reason: &str,
    remedy: &str,
    actual_value: &str,
) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "resource").as_deref(), Some("sort_options"));
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(1)));
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(actual_value.parse().unwrap())));
    assert_eq!(err.detail_value("keys_count"), None);
    assert_eq!(err.detail_value("actual_count"), None);
    assert_eq!(error_context_value(err, "num_keys").as_deref(), None);
    assert_eq!(error_context_value(err, "descending_len").as_deref(), None);
    assert_eq!(error_context_value(err, "nulls_first_len").as_deref(), None);
}

fn assert_i32_column_eq(left: &DataFrame, right: &DataFrame, column: &str) {
    assert_eq!(
        left.series(column).unwrap().to_optional_vec::<i32>().unwrap(),
        right.series(column).unwrap().to_optional_vec::<i32>().unwrap(),
        "column {column} differed",
    );
}

fn dataframe_device_allocated_size_bytes(df: &DataFrame) -> usize {
    df.info_with(InfoDetail::DEVICE_ALLOCATED).unwrap().memory_accounting.unwrap().device_allocated_bytes.unwrap()
}

fn assert_budget_report_high_water(report: &cudf::ExecutionBudgetReport, lower_bound_bytes: usize) {
    let high_water = report.allocator_high_water_bytes().expect("budget report should include allocator high-water");
    assert!(high_water > 0, "allocator high-water should be positive: {report:?}");
    assert!(
        high_water >= lower_bound_bytes,
        "allocator high-water should cover lower bound bytes: high_water={high_water}, lower_bound={lower_bound_bytes}, report={report:?}",
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

// ---------------------------------------------------------------------------
// SortOptions defaults
// ---------------------------------------------------------------------------

fn test_sort_options_defaults() {
    let opts = SortOptions::new();
    assert!(opts.descending.is_empty(), "default descending should be empty (= all ascending)");
    assert!(opts.nulls_first.is_empty(), "default nulls_first should be empty (= all nulls last)");
    assert!(!opts.stable, "default should be unstable sort");
}

// ---------------------------------------------------------------------------
// SortOptions builder
// ---------------------------------------------------------------------------

fn test_sort_options_builder() {
    let opts =
        SortOptions::new().with_descending(vec![true, false]).with_nulls_first(vec![false, true]).with_stable(true);

    assert_eq!(opts.descending, vec![true, false]);
    assert_eq!(opts.nulls_first, vec![false, true]);
    assert!(opts.stable);
}

// ---------------------------------------------------------------------------
// sort_values — single column ascending (default)
// ---------------------------------------------------------------------------

fn test_sort_values_single_column_ascending() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
    assert_eq!(sorted.column_names(), vec!["id", "score"]);
}

// ---------------------------------------------------------------------------
// sort_values — single column descending
// ---------------------------------------------------------------------------

fn test_sort_values_single_column_descending() {
    let df = sample_df();
    let sorted = df.sort_values(["score"], SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

// ---------------------------------------------------------------------------
// sort_values — multi-column
// ---------------------------------------------------------------------------

fn test_sort_values_multi_column() {
    // Sort by id ascending, then score descending within each id group.
    let ids = crate::common::series::<i32>("id", &[2, 1, 2, 1]).unwrap();
    let scores = crate::common::series::<f64>("score", &[20.0, 10.0, 40.0, 30.0]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();

    let sorted = df.sort_values(["id", "score"], SortOptions::new().with_descending(vec![false, true])).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 2, 2]);
    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 20.0]);
}

fn test_sort_values_multi_column_with_owned_string_keys() {
    let ids = crate::common::series::<i32>("id", &[2, 1, 2, 1]).unwrap();
    let scores = crate::common::series::<f64>("score", &[20.0, 10.0, 40.0, 30.0]).unwrap();
    let tags = crate::common::series::<i32>("tag", &[200, 100, 400, 300]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("score", scores), ("tag", tags)]).unwrap();

    let keys = vec!["id".to_string(), "score".to_string()];
    let sorted = df.sort_values(keys, SortOptions::new().with_descending(vec![false, true])).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 2, 2]);
    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 20.0]);
    assert_eq!(sorted.series("tag").unwrap().to_vec_strict::<i32>().unwrap(), vec![300, 100, 400, 200]);
}

// ---------------------------------------------------------------------------
// sort_values — stable sort preserves insertion order for ties
// ---------------------------------------------------------------------------

fn test_sort_values_stable_preserves_order() {
    // All ids are the same — stable sort should preserve original row order.
    let ids = crate::common::series::<i32>("id", &[1, 1, 1, 1]).unwrap();
    let tags = crate::common::series::<i32>("tag", &[10, 20, 30, 40]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("tag", tags)]).unwrap();

    let sorted = df.sort_values(["id"], SortOptions::new().with_stable(true)).unwrap();

    // Tags should remain in original insertion order since all ids are equal.
    assert_eq!(sorted.series("tag").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40]);
}

// ---------------------------------------------------------------------------
// sort_values — rejects mismatched descending length
// ---------------------------------------------------------------------------

fn test_sort_values_rejects_mismatched_descending_length() {
    let df = sample_df();
    // 1 sort key but 2 descending flags
    let result = df.sort_values(["id"], SortOptions::new().with_descending(vec![true, false]));

    let err = result.unwrap_err();
    assert_sort_options_validation_error(
        &err,
        "dataframe::sort_values",
        "descending_length_mismatch",
        "provide either zero descending flags or exactly one flag per sort key",
        "2",
    );
}

// ---------------------------------------------------------------------------
// sort_values — rejects mismatched nulls_first length
// ---------------------------------------------------------------------------

fn test_sort_values_rejects_mismatched_nulls_first_length() {
    let df = sample_df();
    // 1 sort key but 3 nulls_first flags
    let result = df.sort_values(["id"], SortOptions::new().with_nulls_first(vec![true, false, true]));

    let err = result.unwrap_err();
    assert_sort_options_validation_error(
        &err,
        "dataframe::sort_values",
        "nulls_first_length_mismatch",
        "provide either zero nulls_first flags or exactly one flag per sort key",
        "3",
    );
}

// ---------------------------------------------------------------------------
// sort_values — preserves schema
// ---------------------------------------------------------------------------

fn test_sort_values_preserves_schema() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    assert_eq!(sorted.column_names(), vec!["id", "score"]);
    assert_eq!(sorted.column_index("id"), Some(0));
    assert_eq!(sorted.column_index("score"), Some(1));
}

// ---------------------------------------------------------------------------
// sort_values — nonexistent column returns error
// ---------------------------------------------------------------------------

fn test_sort_values_nonexistent_column() {
    let df = sample_df();
    let result = df.sort_values(["missing"], SortOptions::new());

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_missing_column_error(&err, "missing");
}

// ---------------------------------------------------------------------------
// sort_values — empty descending vec means all ascending (defaults)
// ---------------------------------------------------------------------------

fn test_sort_values_empty_options_means_defaults() {
    let df = sample_df();
    // Empty descending + empty nulls_first → all ascending, nulls last.
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    // Should be same as explicit ascending.
    let expected = df.sort_values(["id"], SortOptions::new().with_descending(vec![false])).unwrap();
    assert_eq!(
        sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
        expected.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// sort_values — nulls_first controls null placement
// ---------------------------------------------------------------------------

fn test_sort_values_nulls_first() {
    let ids = column_from_slice_nullable(&[0i32, 2, 0, 1], &[false, true, false, true]);
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]);
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "val".to_string()],
        vec![ids.unwrap(), vals.unwrap()],
    )
    .unwrap();

    // Sort by id ascending, nulls first.
    let sorted = df.sort_values(["id"], SortOptions::new().with_nulls_first(vec![true])).unwrap();

    let id_vals = sorted.series("id").unwrap().to_optional_vec::<i32>().unwrap();
    // Nulls should appear first (their host value is 0, but they're null).
    // Then the non-null values in ascending order: 1, 2.
    // The null positions have undefined data values, but the validity mask marks them null.
    // We check that non-null values appear at the end in correct order.
    assert_eq!(id_vals[2], Some(1));
    assert_eq!(id_vals[3], Some(2));
}

fn test_sort_values_descending_null_placement() {
    let ids = column_from_slice_nullable(&[0i32, 2, 3, 0, 1], &[false, true, true, false, true]);
    let vals = column_from_slice(&[10i32, 20, 30, 40, 50]);
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "val".to_string()],
        vec![ids.unwrap(), vals.unwrap()],
    )
    .unwrap();

    let nulls_first =
        df.sort_values(["id"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![true])).unwrap();
    assert_eq!(
        nulls_first.series("id").unwrap().to_optional_vec::<i32>().unwrap(),
        vec![None, None, Some(3), Some(2), Some(1)]
    );
    assert_eq!(nulls_first.series("val").unwrap().to_vec_strict::<i32>().unwrap()[2..], [30, 20, 50]);

    let nulls_last =
        df.sort_values(["id"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false])).unwrap();
    assert_eq!(
        nulls_last.series("id").unwrap().to_optional_vec::<i32>().unwrap(),
        vec![Some(3), Some(2), Some(1), None, None]
    );
    assert_eq!(nulls_last.series("val").unwrap().to_vec_strict::<i32>().unwrap()[..3], [30, 20, 50]);
}

fn test_sort_values_default_places_nulls_last() {
    let ids = column_from_slice_nullable(&[0i32, 2, 0, 1], &[false, true, false, true]);
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]);
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "val".to_string()],
        vec![ids.unwrap(), vals.unwrap()],
    )
    .unwrap();

    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
    let id_col = sorted.series("id").unwrap();
    let id_vals = id_col.to_optional_vec::<i32>().unwrap();

    assert_eq!(id_vals[0], Some(1));
    assert_eq!(id_vals[1], Some(2));
    assert_eq!(id_col.testing_view().unwrap().to_host_validity().unwrap().unwrap(), vec![true, true, false, false]);
}

// ---------------------------------------------------------------------------
// sort_values — with_stable(false) is the default (unstable sort)
// ---------------------------------------------------------------------------

fn test_sort_values_unstable_by_default() {
    let df = sample_df();
    // Default unstable sort should still produce correct ordering.
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    let ids = sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    // Check monotonically non-decreasing.
    for w in ids.windows(2) {
        assert!(w[0] <= w[1], "unstable sort produced incorrect order: {:?}", ids);
    }
}

// ---------------------------------------------------------------------------
// sort_values with explicit ascending matches default
// ---------------------------------------------------------------------------

fn test_sort_values_explicit_ascending() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new().with_descending(vec![false])).unwrap();
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
}

fn test_prepare_sort_values_single_key_matches_sort_values() {
    let df = sample_df();
    let options = SortOptions::new();
    let expected = df.sort_values(["id"], options.clone()).unwrap();

    let prepared = df.prepare_sort_values(["id"], options).unwrap();
    let actual = prepared.materialize().unwrap();

    assert_eq!(prepared.output_rows(), df.num_rows());
    assert_i32_column_eq(&actual, &expected, "id");
}

fn test_prepare_sort_values_multi_key_descending_matches_sort_values() {
    let ids = crate::common::series::<i32>("id", &[2, 1, 2, 1]).unwrap();
    let scores = crate::common::series::<i32>("score", &[20, 10, 40, 30]).unwrap();
    let tags = crate::common::series::<i32>("tag", &[200, 100, 400, 300]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("score", scores), ("tag", tags)]).unwrap();
    let options = SortOptions::new().with_descending(vec![false, true]);
    let expected = df.sort_values(["id", "score"], options.clone()).unwrap();

    let prepared = df.prepare_sort_values(["id", "score"], options).unwrap();
    let actual = prepared.execute().unwrap();

    assert_i32_column_eq(&actual, &expected, "id");
    assert_i32_column_eq(&actual, &expected, "score");
    assert_i32_column_eq(&actual, &expected, "tag");
}

fn test_prepare_sort_values_descending_null_placement_matches_sort_values() {
    let ids = column_from_slice_nullable(&[0i32, 2, 3, 0, 1], &[false, true, true, false, true]);
    let vals = column_from_slice(&[10i32, 20, 30, 40, 50]);
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "val".to_string()],
        vec![ids.unwrap(), vals.unwrap()],
    )
    .unwrap();
    let options = SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![true]);
    let expected = df.sort_values(["id"], options.clone()).unwrap();

    let prepared = df.prepare_sort_values(["id"], options).unwrap();
    let actual = prepared.materialize().unwrap();

    assert_i32_column_eq(&actual, &expected, "id");
    assert_i32_column_eq(&actual, &expected, "val");
}

fn test_prepare_sort_values_stable_matches_sort_values_for_ties() {
    let ids = crate::common::series::<i32>("id", &[1, 1, 1, 1]).unwrap();
    let tags = crate::common::series::<i32>("tag", &[10, 20, 30, 40]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("tag", tags)]).unwrap();
    let options = SortOptions::new().with_stable(true);
    let expected = df.sort_values(["id"], options.clone()).unwrap();

    let prepared = df.prepare_sort_values(["id"], options).unwrap();
    let actual = prepared.materialize().unwrap();

    assert_i32_column_eq(&actual, &expected, "id");
    assert_i32_column_eq(&actual, &expected, "tag");
}

fn test_sorted_order_reservation_matches_retained_prepared_sort_allocation() {
    let df = sample_df();
    let prepared = df.prepare_sort_values(["id"], SortOptions::new()).unwrap();
    let expected = sorting::sorted_order_reservation_size_bytes_for_len(prepared.output_rows()).unwrap();

    assert_eq!(expected, prepared.sorted_order_reservation_size_bytes().unwrap());
    assert_eq!(expected, prepared.retained_sorted_order_device_allocated_size_bytes().unwrap());
    assert_eq!(expected, prepared.output_rows() * std::mem::size_of::<i32>());
}

fn test_sorted_order_reservation_rejects_rows_above_cudf_size_type() {
    let output_rows = i32::MAX as usize + 1;

    sorting::sorted_order_reservation_size_bytes_for_len(output_rows)
        .expect_err("sorted-order reservation must reject rows above cudf::size_type");
}

fn test_prepared_sort_materialized_output_reservation_matches_actual_numeric_output() {
    let ids = crate::common::series::<i32>("id", &[3, 1, 2]).unwrap();
    let vals = crate::common::series::<i32>("val", &[30, 10, 20]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("val", vals)]).unwrap();
    let prepared = df.prepare_sort_values(["id"], SortOptions::new()).unwrap();

    let reservation = prepared.materialized_output_reservation_size_bytes().unwrap();
    let materialized = prepared.materialize().unwrap();

    assert_eq!(reservation, dataframe_device_allocated_size_bytes(&materialized));
    assert_eq!(materialized.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_prepare_sort_values_with_budget_applies_token_and_reports_retained_order() {
    let df = sample_df();
    let token = with_test_runtime(|runtime| {
        runtime
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("final_sort"))
            .unwrap()
    });

    reset_exec_observation();
    let outcome = df.prepare_sort_values_with_budget(["id"], SortOptions::new(), &token).unwrap();
    let prepared = outcome.value();
    let report = outcome.report();
    let expected_order_bytes = prepared.sorted_order_reservation_size_bytes().unwrap();

    assert_eq!(report.request().boundary(), "final_sort");
    assert_eq!(report.request().budget_bytes(), 64 * 1024 * 1024);
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::StreamBoundReservation);
    assert_eq!(report.status(), ExecutionBudgetStatus::ReservationEnforced);
    assert_eq!(report.retained_bytes(), Some(expected_order_bytes));
    assert_eq!(report.output_bytes(), None);
    assert_budget_report_high_water(report, expected_order_bytes);
    assert_eq!(
        prepared.sorted_order().as_column().testing_execution_domain_mr_value(),
        token.memory_resource().testing_raw_handle_value(),
        "prepared sorted-order output should retain the device manager resource",
    );
    assert_eq!(
        last_exec_mr_value(),
        token.memory_resource().testing_raw_handle_value(),
        "prepared sort execution should route through the device manager resource",
    );
}

fn test_prepare_sort_values_with_budget_stable_uses_token_and_matches_sort_values_for_ties() {
    let ids = crate::common::series::<i32>("id", &[1, 1, 1, 1]).unwrap();
    let tags = crate::common::series::<i32>("tag", &[10, 20, 30, 40]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("tag", tags)]).unwrap();
    let options = SortOptions::new().with_stable(true);
    let expected = df.sort_values(["id"], options.clone()).unwrap();
    let token = with_test_runtime(|runtime| {
        runtime
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("stable_final_sort"))
            .unwrap()
    });

    reset_exec_observation();
    let outcome = df.prepare_sort_values_with_budget(["id"], options, &token).unwrap();
    let prepare_exec_mrs = observed_exec_mr_values();
    let prepared = outcome.value();
    let actual = prepared.materialize().unwrap();
    let manager_mr = token.memory_resource().testing_raw_handle_value();

    assert_i32_column_eq(&actual, &expected, "id");
    assert_i32_column_eq(&actual, &expected, "tag");
    assert_eq!(actual.series("tag").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40]);
    assert_eq!(
        prepared.sorted_order().as_column().testing_execution_domain_mr_value(),
        manager_mr,
        "stable prepared sorted-order output should retain the device manager resource",
    );
    assert!(
        prepare_exec_mrs.len() >= 3,
        "stable prepared sort should observe at least key copy, tie-breaker sequence, and sorted-order execution; got {prepare_exec_mrs:?}",
    );
    assert!(
        prepare_exec_mrs.iter().all(|&observed| observed == manager_mr),
        "stable prepared sort tie-breaker, stable key copies, and sorted-order execution should route through the device manager resource; got {prepare_exec_mrs:?}, expected {manager_mr}",
    );
    assert_eq!(outcome.report().request().boundary(), "stable_final_sort");
    assert_eq!(outcome.report().retained_bytes(), Some(prepared.sorted_order_reservation_size_bytes().unwrap()));
}

fn test_prepared_sort_materialize_with_budget_reports_output_size_and_allocator() {
    let df = sample_df();
    let prepared = {
        let prepare_token = with_test_runtime(|runtime| {
            runtime
                .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("final_sort_order"))
                .unwrap()
        });
        df.prepare_sort_values_with_budget(["id"], SortOptions::new(), &prepare_token).unwrap().into_value()
    };
    let materialize_token = with_test_runtime(|runtime| {
        runtime
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("final_sort_output"))
            .unwrap()
    });

    reset_exec_observation();
    let outcome = prepared.materialize_with_budget(&materialize_token).unwrap();
    let sorted = outcome.value();
    let report = outcome.report();
    let output_bytes = dataframe_device_allocated_size_bytes(sorted);

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
    assert_eq!(report.request().boundary(), "final_sort_output");
    assert_eq!(report.retained_bytes(), Some(prepared.retained_sorted_order_device_allocated_size_bytes().unwrap()));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_budget_report_high_water(report, output_bytes);
    assert_eq!(
        sorted.testing_execution_domain_mr_value(),
        materialize_token.memory_resource().testing_raw_handle_value(),
        "materialized output should retain the device manager resource",
    );
    assert_eq!(
        last_exec_mr_value(),
        materialize_token.memory_resource().testing_raw_handle_value(),
        "materialize execution should route through the device manager resource",
    );
}

fn test_prepare_sort_values_with_budget_reports_exceeded_budget_honestly() {
    let df = sample_df();
    let token = with_test_runtime(|runtime| {
        runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("final_sort")).unwrap()
    });

    let err = df
        .prepare_sort_values_with_budget(["id"], SortOptions::new(), &token)
        .expect_err("one-byte budget should not cover prepared sort allocations");

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("final_sort"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "budget_enforcement").as_deref(), Some("stream_bound_reservation"),);
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"),);
    assert_eq!(error_context_value(&err, "budget_peak_bytes").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "budget_current_bytes").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "budget_failed_allocation_count").as_deref(), Some("1"));
    assert!(error_context_value(&err, "budget_largest_failed_allocation_bytes").is_some());
}

#[test]
fn dataframe_sort_values_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_sort_options_defaults", test_sort_options_defaults),
        ("test_sort_options_builder", test_sort_options_builder),
        ("test_sort_values_single_column_ascending", test_sort_values_single_column_ascending),
        ("test_sort_values_single_column_descending", test_sort_values_single_column_descending),
        ("test_sort_values_multi_column", test_sort_values_multi_column),
        ("test_sort_values_multi_column_with_owned_string_keys", test_sort_values_multi_column_with_owned_string_keys),
        ("test_sort_values_stable_preserves_order", test_sort_values_stable_preserves_order),
        (
            "test_sort_values_rejects_mismatched_descending_length",
            test_sort_values_rejects_mismatched_descending_length,
        ),
        (
            "test_sort_values_rejects_mismatched_nulls_first_length",
            test_sort_values_rejects_mismatched_nulls_first_length,
        ),
        ("test_sort_values_preserves_schema", test_sort_values_preserves_schema),
        ("test_sort_values_nonexistent_column", test_sort_values_nonexistent_column),
        ("test_sort_values_empty_options_means_defaults", test_sort_values_empty_options_means_defaults),
        ("test_sort_values_nulls_first", test_sort_values_nulls_first),
        ("test_sort_values_descending_null_placement", test_sort_values_descending_null_placement),
        ("test_sort_values_default_places_nulls_last", test_sort_values_default_places_nulls_last),
        ("test_sort_values_unstable_by_default", test_sort_values_unstable_by_default),
        ("test_sort_values_explicit_ascending", test_sort_values_explicit_ascending),
        (
            "test_prepare_sort_values_single_key_matches_sort_values",
            test_prepare_sort_values_single_key_matches_sort_values,
        ),
        (
            "test_prepare_sort_values_multi_key_descending_matches_sort_values",
            test_prepare_sort_values_multi_key_descending_matches_sort_values,
        ),
        (
            "test_prepare_sort_values_descending_null_placement_matches_sort_values",
            test_prepare_sort_values_descending_null_placement_matches_sort_values,
        ),
        (
            "test_prepare_sort_values_stable_matches_sort_values_for_ties",
            test_prepare_sort_values_stable_matches_sort_values_for_ties,
        ),
        (
            "test_sorted_order_reservation_matches_retained_prepared_sort_allocation",
            test_sorted_order_reservation_matches_retained_prepared_sort_allocation,
        ),
        (
            "test_sorted_order_reservation_rejects_rows_above_cudf_size_type",
            test_sorted_order_reservation_rejects_rows_above_cudf_size_type,
        ),
        (
            "test_prepared_sort_materialized_output_reservation_matches_actual_numeric_output",
            test_prepared_sort_materialized_output_reservation_matches_actual_numeric_output,
        ),
        (
            "test_prepare_sort_values_with_budget_applies_token_and_reports_retained_order",
            test_prepare_sort_values_with_budget_applies_token_and_reports_retained_order,
        ),
        (
            "test_prepare_sort_values_with_budget_stable_uses_token_and_matches_sort_values_for_ties",
            test_prepare_sort_values_with_budget_stable_uses_token_and_matches_sort_values_for_ties,
        ),
        (
            "test_prepared_sort_materialize_with_budget_reports_output_size_and_allocator",
            test_prepared_sort_materialize_with_budget_reports_output_size_and_allocator,
        ),
        (
            "test_prepare_sort_values_with_budget_reports_exceeded_budget_honestly",
            test_prepare_sort_values_with_budget_reports_exceeded_budget_honestly,
        ),
    ]);
}
