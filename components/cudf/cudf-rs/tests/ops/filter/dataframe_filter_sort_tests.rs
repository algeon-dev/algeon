//! Tests for DataFrame high-level filter, filter_isin, sort, deduplicate.
use cudf::{
    DataFrame, DuplicateKeepOption, ErrorKind, SortOptions,
    frame::dataframe::{DropDuplicatesOptions, DropNaOptions},
    testing::prelude::*,
};

use crate::common::{
    assert_missing_column_error, column_from_slice, column_from_slice_nullable, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation, sample_df,
    sample_runtime,
};

// ---------------------------------------------------------------------------
// filter
// ---------------------------------------------------------------------------

#[test]
fn test_filter_basic() {
    let df = sample_df();
    // Keep rows where id > 2: rows 0 (3), 2 (4), 4 (5)
    let mask = crate::common::series::<bool>("mask", &[true, false, true, false, true]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows(), 3);
    assert_eq!(filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 4, 5]);
    assert_eq!(filtered.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![30.0, 40.0, 50.0]);
}

#[test]
fn test_filter_preserves_schema() {
    let df = sample_df();
    let mask = crate::common::series::<bool>("mask", &[true, true, true, true, true]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.column_names(), vec!["id", "score"]);
    assert_eq!(filtered.column_index("id"), Some(0));
    assert_eq!(filtered.column_index("score"), Some(1));
}

#[test]
fn test_filter_all_false() {
    let df = sample_df();
    let mask = crate::common::series::<bool>("mask", &[false, false, false, false, false]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows(), 0);
    assert_eq!(filtered.column_names(), vec!["id", "score"]);
}

// ---------------------------------------------------------------------------
// filter_isin
// ---------------------------------------------------------------------------

#[test]
fn test_filter_isin_basic() {
    let df = sample_df(); // ids: 3,1,4,1,5
    let values = crate::common::series::<i32>("values", &[1, 5]).unwrap();
    let filtered = df.filter_isin("id", &values).unwrap();

    assert_eq!(filtered.num_rows(), 3); // rows with id 1,1,5
    let mut ids = filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![1, 1, 5]);
}

#[test]
fn test_filter_isin_no_matches() {
    let df = sample_df(); // ids: 3,1,4,1,5
    let values = crate::common::series::<i32>("values", &[99, 100]).unwrap();
    let filtered = df.filter_isin("id", &values).unwrap();

    assert_eq!(filtered.num_rows(), 0);
}

#[test]
fn test_filter_isin_all_match() {
    let df = sample_df(); // ids: 3,1,4,1,5
    let values = crate::common::series::<i32>("values", &[1, 3, 4, 5]).unwrap();
    let filtered = df.filter_isin("id", &values).unwrap();

    assert_eq!(filtered.num_rows(), 5);
}

#[test]
fn test_filter_isin_preserves_schema() {
    let df = sample_df();
    let values = crate::common::series::<i32>("values", &[1]).unwrap();
    let filtered = df.filter_isin("id", &values).unwrap();

    assert_eq!(filtered.column_names(), vec!["id", "score"]);
    assert_eq!(filtered.column_index("id"), Some(0));
    assert_eq!(filtered.column_index("score"), Some(1));
}

#[test]
fn test_filter_isin_nonexistent_column() {
    let df = sample_df();
    let values = crate::common::series::<i32>("values", &[1]).unwrap();
    let err = df.filter_isin("nonexistent", &values).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_filter_isin_float64() {
    let df = sample_df(); // scores: 30.0, 10.0, 40.0, 15.0, 50.0
    let values = crate::common::series::<f64>("values", &[10.0, 50.0]).unwrap();
    let filtered = df.filter_isin("score", &values).unwrap();

    assert_eq!(filtered.num_rows(), 2);
    let mut scores = filtered.series("score").unwrap().to_vec_strict::<f64>().unwrap();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(scores, vec![10.0, 50.0]);
}

#[test]
fn test_filter_isin_empty_values() {
    let df = sample_df();
    let values = crate::common::series::<i32>("values", &[]).unwrap();
    let filtered = df.filter_isin("id", &values).unwrap();

    assert_eq!(filtered.num_rows(), 0);
}

#[test]
fn test_filter_isin_with_series_succeeds_with_mixed_domains_and_orders_secondary_input_before_dataframe_exec() {
    let _event_observation = crate::common::lock_event_observation();
    let df_rt = sample_runtime();
    let values_rt = sample_runtime();
    let df = DataFrame::from_columns([
        ("id", df_rt.series(&[10i32, 20, 30]).unwrap()),
        ("score", df_rt.series(&[100i32, 200, 300]).unwrap()),
    ])
    .unwrap();
    let values = values_rt.series(&[20i32, 30]).unwrap();
    let expected_exec_stream = df.testing_execution_domain_stream_value();
    let expected_exec_mr = df.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let filtered = df.filter_isin("id", &values).unwrap();

    assert_eq!(filtered.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(filtered.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    assert_eq!(filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(filtered.series("score").unwrap().to_vec_strict::<i32>().unwrap(), vec![200, 300]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ---------------------------------------------------------------------------
// sort
// ---------------------------------------------------------------------------

#[test]
fn test_sort_single_column_ascending() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
    assert_eq!(sorted.column_names(), vec!["id", "score"]);
}

#[test]
fn test_sort_single_column_descending() {
    let df = sample_df();
    let sorted = df.sort_values(["score"], SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

#[test]
fn test_sort_by_subset_of_columns() {
    // Sort by id ascending, then score descending within each id group.
    let ids = column_from_slice(&[2i32, 1, 2, 1]).unwrap();
    let scores = column_from_slice(&[20.0f64, 10.0, 40.0, 30.0]).unwrap();
    let df =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap();

    let sorted = df.sort_values(["id", "score"], SortOptions::new().with_descending(vec![false, true])).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 2, 2]);
    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 20.0]);
}

#[test]
fn test_sort_nonexistent_column_returns_error() {
    let df = sample_df();
    let err = df.sort_values(["nonexistent"], SortOptions::new()).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_sort_mismatched_descending_count() {
    let df = sample_df();
    // 2 column names but only 1 descending flag → error
    let result = df.sort_values(["id", "score"], SortOptions::new().with_descending(vec![false]));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
}

#[test]
fn test_sort_preserves_schema() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
    assert_eq!(sorted.column_names(), vec!["id", "score"]);
    assert_eq!(sorted.column_index("id"), Some(0));
    assert_eq!(sorted.column_index("score"), Some(1));
}

// ---------------------------------------------------------------------------
// nlargest / nsmallest
// ---------------------------------------------------------------------------

#[test]
fn test_nlargest() {
    let df = sample_df(); // ids: 3,1,4,1,5  scores: 30,10,40,15,50
    let top2 = df.nlargest(2, "score").unwrap();

    assert_eq!(top2.num_rows(), 2);
    assert_eq!(top2.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![50.0, 40.0]);
}

#[test]
fn test_nsmallest() {
    let df = sample_df();
    let bot2 = df.nsmallest(2, "id").unwrap();

    assert_eq!(bot2.num_rows(), 2);
    assert_eq!(bot2.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1]);
}

#[test]
fn test_nlargest_nonexistent_column() {
    let df = sample_df();
    let err = df.nlargest(2, "missing").unwrap_err();
    assert_missing_column_error(&err, "missing");
}

// ---------------------------------------------------------------------------
// drop_duplicates
// ---------------------------------------------------------------------------

#[test]
fn test_drop_duplicates_by_single_column() {
    // Duplicate ids: 3,1,4,1,5 → distinct ids should be {1,3,4,5}
    let df = sample_df();
    let deduped = df
        .drop_duplicates(DropDuplicatesOptions::builder().subset(&["id"]).keep(DuplicateKeepOption::KeepAny).build())
        .unwrap();

    assert_eq!(deduped.num_rows(), 4); // 4 distinct ids
    assert_eq!(deduped.column_names(), vec!["id", "score"]);
}

#[test]
fn test_drop_duplicates_preserves_schema() {
    let df = sample_df();
    let deduped = df
        .drop_duplicates(DropDuplicatesOptions::builder().subset(&["id"]).keep(DuplicateKeepOption::KeepFirst).build())
        .unwrap();
    assert_eq!(deduped.column_names(), vec!["id", "score"]);
    assert_eq!(deduped.column_index("id"), Some(0));
    assert_eq!(deduped.column_index("score"), Some(1));
}

#[test]
fn test_drop_duplicates_nonexistent_column() {
    let df = sample_df();
    let err = df
        .drop_duplicates(DropDuplicatesOptions::builder().subset(&["nope"]).keep(DuplicateKeepOption::KeepAny).build())
        .unwrap_err();
    assert_missing_column_error(&err, "nope");
}

// ---------------------------------------------------------------------------
// dropna
// ---------------------------------------------------------------------------

#[test]
fn test_dropna_basic() {
    let ids = column_from_slice_nullable(&[1i32, 0, 3, 0, 5], &[true, false, true, false, true]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap();

    let cleaned = df.dropna(DropNaOptions::builder().subset(&["id"]).thresh(1).build()).unwrap();

    assert_eq!(cleaned.num_rows(), 3);
    assert_eq!(cleaned.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(1), Some(3), Some(5)]);
    assert_eq!(
        cleaned.series("val").unwrap().to_optional_vec::<f64>().unwrap(),
        vec![Some(10.0), Some(30.0), Some(50.0)]
    );
}

#[test]
fn test_dropna_preserves_schema_simple() {
    let ids = column_from_slice_nullable(&[1i32, 0], &[true, false]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap();

    let cleaned = df.dropna(DropNaOptions::builder().subset(&["id"]).build()).unwrap();
    assert_eq!(cleaned.column_names(), vec!["id", "val"]);
}

#[test]
fn test_dropna_any_equivalent_to_thresh() {
    let ids = column_from_slice_nullable(&[1i32, 0, 3, 0, 5], &[true, false, true, false, true]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap();

    let expected = df.dropna(DropNaOptions::builder().subset(&["id"]).thresh(1).build()).unwrap();
    let actual = df.dropna(DropNaOptions::builder().subset(&["id"]).build()).unwrap();

    assert_eq!(actual.num_rows(), expected.num_rows());
    assert_eq!(
        actual.series("id").unwrap().to_optional_vec::<i32>().unwrap(),
        expected.series("id").unwrap().to_optional_vec::<i32>().unwrap(),
    );
    assert_eq!(
        actual.series("val").unwrap().to_optional_vec::<f64>().unwrap(),
        expected.series("val").unwrap().to_optional_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_dropna_any_mixed_null_patterns() {
    // Two subset columns with nulls in different positions.
    let a = column_from_slice_nullable(&[1i32, 2, 3, 4], &[true, false, true, true]).unwrap();
    let b = column_from_slice_nullable(&[10i32, 20, 0, 40], &[true, true, false, true]).unwrap();
    let c = column_from_slice(&[100.0f64, 200.0, 300.0, 400.0]).unwrap();
    let df =
        crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![a, b, c])
            .unwrap();

    // dropna on ["a", "b"]: drop rows where ANY of a or b is null.
    // Row 0: a=1, b=10 → kept
    // Row 1: a=null → dropped
    // Row 2: b=null → dropped
    // Row 3: a=4, b=40 → kept
    let cleaned = df.dropna(DropNaOptions::builder().subset(&["a", "b"]).build()).unwrap();

    assert_eq!(cleaned.num_rows(), 2);
    assert_eq!(cleaned.series("a").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(1), Some(4)]);
    assert_eq!(cleaned.series("b").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(10), Some(40)]);
    assert_eq!(cleaned.series("c").unwrap().to_optional_vec::<f64>().unwrap(), vec![Some(100.0), Some(400.0)]);
}

#[test]
fn test_dropna_default_all_columns() {
    let ids = column_from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["id".to_string()], vec![ids]).unwrap();

    // Default options: subset=all columns, how=any
    let result = df.dropna(DropNaOptions::default()).unwrap();
    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_dropna_no_nulls() {
    let c = column_from_slice(&[1i32, 2, 3]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![c]).unwrap();

    let result = df.dropna(DropNaOptions::builder().subset(&["x"]).build()).unwrap();
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_dropna_all_null() {
    let c = column_from_slice_nullable(&[0i32, 0, 0], &[false, false, false]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![c]).unwrap();

    let result = df.dropna(DropNaOptions::builder().subset(&["x"]).build()).unwrap();
    assert_eq!(result.num_rows(), 0);
}

#[test]
fn test_dropna_preserves_schema() {
    let a = column_from_slice_nullable(&[1i32, 0], &[true, false]).unwrap();
    let b = column_from_slice(&[10.0f64, 20.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![a, b]).unwrap();

    let cleaned = df.dropna(DropNaOptions::builder().subset(&["a"]).build()).unwrap();
    assert_eq!(cleaned.column_names(), vec!["a", "b"]);
    assert_eq!(cleaned.column_index("a"), Some(0));
    assert_eq!(cleaned.column_index("b"), Some(1));
}

#[test]
fn test_dropna_nonexistent_column_returns_error() {
    let c = column_from_slice(&[1i32]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![c]).unwrap();

    let err = df.dropna(DropNaOptions::builder().subset(&["missing"]).build()).unwrap_err();
    assert_missing_column_error(&err, "missing");
}
