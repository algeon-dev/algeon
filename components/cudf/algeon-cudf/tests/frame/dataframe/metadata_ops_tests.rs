//! Integration tests for zero-copy metadata ops (cuDF-style signatures).
//!
//! Tests: without_columns, rename_columns, with_column, assign, and Arc sharing verification.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{DataFrame, ErrorCode, ErrorKind, Runtime, Series, testing::prelude::*};

use crate::common::{
    assert_duplicate_name_error, assert_missing_column_error, dataframe_device_ptr as device_ptr, error_context_value,
    last_exec_stream_value, lock_event_observation, reset_exec_observation, run_named_cases, series,
};

/// Helper: build a 3-row, 3-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = series::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = series::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();
    let flags = series::<i32>("flags", &[100, 200, 300]).unwrap();

    DataFrame::from_series([("id", ids), ("score", scores), ("flag", flags)]).unwrap()
}

// ===========================================================================
// without_columns: removes correct columns, preserves others
// ===========================================================================

fn test_without_columns_removes_single_column() {
    let df = sample_df();
    let result = df.without_columns(["score"]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), &["id", "flag"]);
    assert_eq!(result.num_rows(), 3);
}

fn test_without_columns_removes_multiple_columns() {
    let df = sample_df();
    let result = df.without_columns(["id", "flag"]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.column_names(), &["score"]);
}

fn test_without_columns_preserves_arc_sharing_for_retained_columns() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let flag_ptr = device_ptr(&df, "flag");

    let result = df.without_columns(["score"]).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
    assert_eq!(flag_ptr, device_ptr(&result, "flag"));
}

fn test_without_columns_nonexistent_column_returns_error() {
    let df = sample_df();
    let result = df.without_columns(["nonexistent"]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

fn test_without_columns_duplicate_name_returns_canonical_duplicate_error() {
    let df = sample_df();

    let err = df.without_columns(["score", "score"]).expect_err("expected duplicate exclusion error");

    assert_duplicate_name_error(&err, "score");
    assert_eq!(err.operation(), "dataframe::without_columns");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
}

fn test_without_columns_all_columns_preserves_row_count() {
    let df = sample_df();
    let result = df.without_columns(["id", "score", "flag"]).unwrap();

    assert_eq!(result.num_columns(), 0);
    assert_eq!(result.num_rows(), 3);
}

fn test_without_columns_preserves_column_order() {
    let df = sample_df();
    let result = df.without_columns(["score"]).unwrap();

    // "id" should still come before "flag"
    assert_eq!(result.column_names(), &["id", "flag"]);
}

fn test_without_columns_original_dataframe_still_reads_excluded_column() {
    let df = sample_df();
    let result = df.without_columns(["score"]).unwrap();

    assert_eq!(result.column_names(), &["id", "flag"]);
    assert_eq!(df.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// rename_columns: multi-column rename
// ===========================================================================

fn test_rename_columns_single_pair() {
    let df = sample_df();
    let result = df.rename_columns([("id", "identifier")]).unwrap();

    assert_eq!(result.column_names(), &["identifier", "score", "flag"]);
    assert_eq!(result.num_rows(), 3);
}

fn test_rename_columns_multiple_pairs() {
    let df = sample_df();
    let result = df.rename_columns([("id", "a"), ("score", "b"), ("flag", "c")]).unwrap();

    assert_eq!(result.column_names(), &["a", "b", "c"]);
}

fn test_rename_columns_preserves_arc_sharing() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let score_ptr = device_ptr(&df, "score");
    let flag_ptr = device_ptr(&df, "flag");

    let result = df.rename_columns([("id", "a"), ("score", "b")]).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "a"));
    assert_eq!(score_ptr, device_ptr(&result, "b"));
    assert_eq!(flag_ptr, device_ptr(&result, "flag"));
}

fn test_rename_columns_nonexistent_source_returns_error() {
    let df = sample_df();
    let result = df.rename_columns([("nonexistent", "x")]);

    assert!(result.is_err());
    assert_missing_column_error(&result.unwrap_err(), "nonexistent");
}

fn test_rename_columns_duplicate_target_returns_error() {
    let df = sample_df();
    // Both "id" and "score" renamed to the same target
    let err = df.rename_columns([("id", "same"), ("score", "same")]).expect_err("expected duplicate rename target");

    assert_duplicate_name_error(&err, "same");
    assert_eq!(err.operation(), "dataframe::rename_columns");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("3"));
}

fn test_rename_columns_target_conflicts_with_existing_unrenamed_column() {
    let df = sample_df();
    // Rename "id" to "flag"; but "flag" already exists and is not being renamed
    let err = df.rename_columns([("id", "flag")]).expect_err("expected conflicting rename target");

    assert_duplicate_name_error(&err, "flag");
    assert_eq!(err.operation(), "dataframe::rename_columns");
}

fn test_rename_columns_duplicate_source_returns_canonical_duplicate_error() {
    let df = sample_df();

    let err = df.rename_columns([("id", "identifier"), ("id", "identifier2")]).expect_err("expected duplicate source");

    assert_duplicate_name_error(&err, "id");
    assert_eq!(err.operation(), "dataframe::rename_columns");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
}

fn test_rename_columns_swap_is_allowed() {
    let df = sample_df();
    // Swap "id" and "flag" names; both are being renamed, so no conflict
    let result = df.rename_columns([("id", "flag"), ("flag", "id")]).unwrap();

    assert_eq!(result.column_names(), &["flag", "score", "id"]);

    // Data should follow the original column, not the new name
    let original_id_ptr = device_ptr(&df, "id");
    let original_flag_ptr = device_ptr(&df, "flag");
    assert_eq!(original_id_ptr, device_ptr(&result, "flag"));
    assert_eq!(original_flag_ptr, device_ptr(&result, "id"));
}

fn test_rename_columns_empty_iterator() {
    let df = sample_df();
    let result = df.rename_columns(std::iter::empty::<(&str, &str)>()).unwrap();

    assert_eq!(result.column_names(), df.column_names());
}

fn test_select_accepts_iterators() {
    let df = sample_df();
    let result = df.select(vec!["flag", "id"]).unwrap();

    assert_eq!(result.column_names(), &["flag", "id"]);
    assert_eq!(result.num_rows(), 3);
}

fn test_select_empty_preserves_row_count() {
    let df = sample_df();
    let result = df.select(std::iter::empty::<&str>()).unwrap();

    assert_eq!(result.num_columns(), 0);
    assert_eq!(result.num_rows(), 3);
}

fn test_select_wide_projection_reorders_columns_without_copying() {
    let pairs: Vec<(String, Series)> = (0..16)
        .map(|idx| (format!("c{idx}"), series::<i32>(format!("source_{idx}"), &[idx, idx + 100]).unwrap()))
        .collect();
    let df = DataFrame::from_series(pairs).unwrap();

    let selected = df.select(["c15", "c0", "c7", "c3"]).unwrap();

    assert_eq!(selected.column_names(), &["c15", "c0", "c7", "c3"]);
    assert_eq!(selected.num_rows(), 2);
    assert_eq!(device_ptr(&df, "c15"), device_ptr(&selected, "c15"));
    assert_eq!(device_ptr(&df, "c0"), device_ptr(&selected, "c0"));
    assert_eq!(device_ptr(&df, "c7"), device_ptr(&selected, "c7"));
    assert_eq!(device_ptr(&df, "c3"), device_ptr(&selected, "c3"));
}

fn test_select_is_metadata_only_no_execution_or_device_copy() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let flag_ptr = device_ptr(&df, "flag");
    let _event_observation = lock_event_observation();

    reset_exec_observation();
    let selected = df.select(["flag", "id"]).unwrap();

    assert_eq!(last_exec_stream_value(), usize::MAX);
    assert_eq!(device_ptr(&selected, "id"), id_ptr);
    assert_eq!(device_ptr(&selected, "flag"), flag_ptr);
    assert_eq!(selected.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(selected.testing_execution_domain_stream_value(), df.testing_execution_domain_stream_value());
    assert_eq!(selected.testing_execution_domain_mr_value(), df.testing_execution_domain_mr_value());
}

// ===========================================================================
// with_column: accepts both Column and Series via Into<Series>
// ===========================================================================

fn test_with_column_adds_new_column() {
    let df = sample_df();
    let extra = series::<i64>("extra", &[100, 200, 300]).unwrap();

    let result = df.with_column("extra", extra).unwrap();

    assert_eq!(result.num_columns(), 4);
    assert!(result.column_names().contains(&"extra"));
    assert_eq!(result.num_rows(), 3);
}

fn test_with_column_replaces_existing_column() {
    let df = sample_df();
    let new_score = series::<f64>("new_score", &[99.0, 88.0, 77.0]).unwrap();

    let result = df.with_column("score", new_score).unwrap();

    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![99.0, 88.0, 77.0]);
}

fn test_with_column_preserves_arc_sharing_for_unchanged() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let flag_ptr = device_ptr(&df, "flag");

    let extra = series::<i64>("extra", &[100, 200, 300]).unwrap();
    let result = df.with_column("extra", extra).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
    assert_eq!(flag_ptr, device_ptr(&result, "flag"));
}

fn test_with_column_shares_arc_from_input_series() {
    let df = sample_df();
    let extra = series::<i64>("extra", &[100, 200, 300]).unwrap();
    let extra_arc = extra.clone().into_inner();

    let result = df.with_column("extra", extra).unwrap();

    // The inserted column should share the same Arc as the input Series
    let result_series = result.series("extra").unwrap();
    assert!(Arc::ptr_eq(&extra_arc, &result_series.into_inner()));
}

fn test_with_column_row_count_mismatch_returns_error() {
    let df = sample_df(); // 3 rows
    let wrong_len = series::<i32>("x", &[1, 2]).unwrap(); // 2 rows

    let result = df.with_column("x", wrong_len);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "dataframe::with_column");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("x"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
}

// ===========================================================================
// assign: adds/replaces multiple columns
// ===========================================================================

fn test_assign_adds_multiple_new_columns() {
    let df = sample_df();
    let a = series::<i32>("a", &[10, 20, 30]).unwrap();
    let b = series::<f64>("b", &[1.0, 2.0, 3.0]).unwrap();

    let result = df.assign([("a", a), ("b", b)]).unwrap();

    assert_eq!(result.num_columns(), 5);
    assert!(result.column_names().contains(&"a"));
    assert!(result.column_names().contains(&"b"));
}

fn test_assign_replaces_existing_column() {
    let df = sample_df();
    let new_score = series::<f64>("s", &[99.0, 88.0, 77.0]).unwrap();

    let result = df.assign([("score", new_score)]).unwrap();

    assert_eq!(result.num_columns(), 3); // no new columns, just replaced
    assert_eq!(result.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![99.0, 88.0, 77.0]);
}

fn test_assign_mix_of_add_and_replace() {
    let df = sample_df();
    let new_id = series::<i32>("x", &[10, 20, 30]).unwrap();
    let extra = series::<i64>("x", &[100, 200, 300]).unwrap();

    let result = df.assign([("id", new_id), ("extra", extra)]).unwrap();

    assert_eq!(result.num_columns(), 4);
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

fn test_assign_preserves_arc_sharing_for_unchanged() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let flag_ptr = device_ptr(&df, "flag");

    let new_score = series::<f64>("s", &[99.0, 88.0, 77.0]).unwrap();
    let result = df.assign([("score", new_score)]).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
    assert_eq!(flag_ptr, device_ptr(&result, "flag"));
}

fn test_assign_shares_arc_from_input_series() {
    let df = sample_df();
    let extra = series::<i64>("extra", &[100, 200, 300]).unwrap();
    let extra_arc = extra.clone().into_inner();

    let result = df.assign([("extra", extra)]).unwrap();

    let result_series = result.series("extra").unwrap();
    assert!(Arc::ptr_eq(&extra_arc, &result_series.into_inner()));
}

fn test_assign_row_count_mismatch_returns_error() {
    let df = sample_df(); // 3 rows
    let wrong = series::<i32>("x", &[1]).unwrap(); // 1 row

    let result = df.assign([("x", wrong)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "dataframe::assign");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("x"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
}

fn test_assign_empty_iterator_returns_same_schema() {
    let df = sample_df();
    let result = df.assign(std::iter::empty::<(&str, Series)>()).unwrap();

    assert_eq!(result.column_names(), df.column_names());
    assert_eq!(result.num_rows(), df.num_rows());
}

fn test_assign_duplicate_names_last_wins() {
    let df = sample_df();
    let first = series::<i32>("x", &[1, 2, 3]).unwrap();
    let second = series::<i32>("x", &[10, 20, 30]).unwrap();

    let result = df.assign([("extra", first), ("extra", second)]).unwrap();

    // The second value should win
    assert_eq!(result.series("extra").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

fn test_assign_on_zero_column_frame_preserves_logical_row_count() {
    let df = sample_df().without_columns(["id", "score", "flag"]).unwrap();
    let replacement = series::<i32>("replacement", &[7, 8, 9]).unwrap();

    let result = df.assign([("replacement", replacement)]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series("replacement").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
}

fn test_assign_on_zero_column_frame_rejects_row_count_mismatch() {
    let df = sample_df().without_columns(["id", "score", "flag"]).unwrap();
    let wrong = series::<i32>("replacement", &[7, 8]).unwrap();

    let result = df.assign([("replacement", wrong)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "dataframe::assign");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("replacement"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
}

fn test_assign_on_truly_empty_dataframe_seeds_row_count_from_first_column() {
    let rt = Runtime::new().unwrap();
    let df = rt.empty_dataframe();
    let values = rt.series(&[7i32, 8]).unwrap();

    let result = df.assign([("values", values)]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.series("values").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8]);
}

// ===========================================================================
// Combined operations: verify Arc sharing chains
// ===========================================================================

fn test_without_columns_then_assign_preserves_sharing() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");

    let dropped = df.without_columns(["score", "flag"]).unwrap();
    let extra = series::<f64>("extra", &[1.0, 2.0, 3.0]).unwrap();
    let result = dropped.assign([("extra", extra)]).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
}

fn test_rename_columns_then_without_columns_preserves_sharing() {
    let df = sample_df();
    let score_ptr = device_ptr(&df, "score");

    let renamed = df.rename_columns([("id", "identifier")]).unwrap();
    let result = renamed.without_columns(["flag"]).unwrap();

    assert_eq!(score_ptr, device_ptr(&result, "score"));
}

fn test_with_series_then_rename_columns_preserves_sharing() {
    let df = sample_df();
    let extra = series::<i64>("extra", &[100, 200, 300]).unwrap();
    let extra_arc = extra.clone().into_inner();

    let with_extra = df.with_column("extra", extra).unwrap();
    let result = with_extra.rename_columns([("extra", "bonus")]).unwrap();

    let bonus_series = result.series("bonus").unwrap();
    assert!(Arc::ptr_eq(&extra_arc, &bonus_series.into_inner()));
}

fn test_with_column_on_zero_column_frame_preserves_logical_row_count() {
    let df = sample_df().select(std::iter::empty::<&str>()).unwrap();
    let values = series::<i32>("values", &[11, 22, 33]).unwrap();

    let result = df.with_column("values", values).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series("values").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

fn test_with_column_on_zero_column_frame_rejects_row_count_mismatch() {
    let df = sample_df().select(std::iter::empty::<&str>()).unwrap();
    let values = series::<i32>("values", &[11, 22]).unwrap();

    let result = df.with_column("values", values);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn test_with_column_on_zero_column_frame_from_drop_rejects_row_count_mismatch() {
    let df = sample_df().without_columns(["id", "score", "flag"]).unwrap();
    let values = series::<i32>("values", &[11, 22]).unwrap();

    let result = df.with_column("values", values);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn test_with_column_on_truly_empty_dataframe_seeds_row_count_from_first_column() {
    let rt = Runtime::new().unwrap();
    let df = rt.empty_dataframe();
    let values = rt.series(&[11i32, 22]).unwrap();

    let result = df.with_column("values", values).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.series("values").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22]);
}

#[test]
fn metadata_ops_tests_cases() {
    run_named_cases(&[
        ("test_without_columns_removes_single_column", test_without_columns_removes_single_column),
        ("test_without_columns_removes_multiple_columns", test_without_columns_removes_multiple_columns),
        (
            "test_without_columns_preserves_arc_sharing_for_retained_columns",
            test_without_columns_preserves_arc_sharing_for_retained_columns,
        ),
        (
            "test_without_columns_nonexistent_column_returns_error",
            test_without_columns_nonexistent_column_returns_error,
        ),
        (
            "test_without_columns_duplicate_name_returns_canonical_duplicate_error",
            test_without_columns_duplicate_name_returns_canonical_duplicate_error,
        ),
        ("test_without_columns_all_columns_preserves_row_count", test_without_columns_all_columns_preserves_row_count),
        ("test_without_columns_preserves_column_order", test_without_columns_preserves_column_order),
        (
            "test_without_columns_original_dataframe_still_reads_excluded_column",
            test_without_columns_original_dataframe_still_reads_excluded_column,
        ),
        ("test_rename_columns_single_pair", test_rename_columns_single_pair),
        ("test_rename_columns_multiple_pairs", test_rename_columns_multiple_pairs),
        ("test_rename_columns_preserves_arc_sharing", test_rename_columns_preserves_arc_sharing),
        ("test_rename_columns_nonexistent_source_returns_error", test_rename_columns_nonexistent_source_returns_error),
        ("test_rename_columns_duplicate_target_returns_error", test_rename_columns_duplicate_target_returns_error),
        (
            "test_rename_columns_target_conflicts_with_existing_unrenamed_column",
            test_rename_columns_target_conflicts_with_existing_unrenamed_column,
        ),
        (
            "test_rename_columns_duplicate_source_returns_canonical_duplicate_error",
            test_rename_columns_duplicate_source_returns_canonical_duplicate_error,
        ),
        ("test_rename_columns_swap_is_allowed", test_rename_columns_swap_is_allowed),
        ("test_rename_columns_empty_iterator", test_rename_columns_empty_iterator),
        ("test_select_accepts_iterators", test_select_accepts_iterators),
        ("test_select_empty_preserves_row_count", test_select_empty_preserves_row_count),
        (
            "test_select_wide_projection_reorders_columns_without_copying",
            test_select_wide_projection_reorders_columns_without_copying,
        ),
        (
            "test_select_is_metadata_only_no_execution_or_device_copy",
            test_select_is_metadata_only_no_execution_or_device_copy,
        ),
        ("test_with_column_adds_new_column", test_with_column_adds_new_column),
        ("test_with_column_replaces_existing_column", test_with_column_replaces_existing_column),
        ("test_with_column_preserves_arc_sharing_for_unchanged", test_with_column_preserves_arc_sharing_for_unchanged),
        ("test_with_column_shares_arc_from_input_series", test_with_column_shares_arc_from_input_series),
        ("test_with_column_row_count_mismatch_returns_error", test_with_column_row_count_mismatch_returns_error),
        ("test_assign_adds_multiple_new_columns", test_assign_adds_multiple_new_columns),
        ("test_assign_replaces_existing_column", test_assign_replaces_existing_column),
        ("test_assign_mix_of_add_and_replace", test_assign_mix_of_add_and_replace),
        ("test_assign_preserves_arc_sharing_for_unchanged", test_assign_preserves_arc_sharing_for_unchanged),
        ("test_assign_shares_arc_from_input_series", test_assign_shares_arc_from_input_series),
        ("test_assign_row_count_mismatch_returns_error", test_assign_row_count_mismatch_returns_error),
        ("test_assign_empty_iterator_returns_same_schema", test_assign_empty_iterator_returns_same_schema),
        ("test_assign_duplicate_names_last_wins", test_assign_duplicate_names_last_wins),
        (
            "test_assign_on_zero_column_frame_preserves_logical_row_count",
            test_assign_on_zero_column_frame_preserves_logical_row_count,
        ),
        (
            "test_assign_on_zero_column_frame_rejects_row_count_mismatch",
            test_assign_on_zero_column_frame_rejects_row_count_mismatch,
        ),
        (
            "test_assign_on_truly_empty_dataframe_seeds_row_count_from_first_column",
            test_assign_on_truly_empty_dataframe_seeds_row_count_from_first_column,
        ),
        ("test_without_columns_then_assign_preserves_sharing", test_without_columns_then_assign_preserves_sharing),
        (
            "test_rename_columns_then_without_columns_preserves_sharing",
            test_rename_columns_then_without_columns_preserves_sharing,
        ),
        (
            "test_with_series_then_rename_columns_preserves_sharing",
            test_with_series_then_rename_columns_preserves_sharing,
        ),
        (
            "test_with_column_on_zero_column_frame_preserves_logical_row_count",
            test_with_column_on_zero_column_frame_preserves_logical_row_count,
        ),
        (
            "test_with_column_on_zero_column_frame_rejects_row_count_mismatch",
            test_with_column_on_zero_column_frame_rejects_row_count_mismatch,
        ),
        (
            "test_with_column_on_zero_column_frame_from_drop_rejects_row_count_mismatch",
            test_with_column_on_zero_column_frame_from_drop_rejects_row_count_mismatch,
        ),
        (
            "test_with_column_on_truly_empty_dataframe_seeds_row_count_from_first_column",
            test_with_column_on_truly_empty_dataframe_seeds_row_count_from_first_column,
        ),
    ]);
}
