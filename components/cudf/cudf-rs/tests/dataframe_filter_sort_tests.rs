//! Tests for DataFrame filter, filter_isin, sort, deduplicate.

use cudf::{
    column::Column, dataframe::DataFrame, error::ErrorKind, sorting::Order, stream_compaction::DuplicateKeepOption,
};

/// Helper: build a 5-row DataFrame with id (i32) and score (f64) columns.
fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = Column::from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

// ---------------------------------------------------------------------------
// filter
// ---------------------------------------------------------------------------

#[test]
fn test_filter_basic() {
    let df = sample_df();
    // Keep rows where id > 2: rows 0 (3), 2 (4), 4 (5)
    let mask_col = Column::from_slice(&[true, false, true, false, true]).unwrap();
    let mask = mask_col.view().unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 3);
    assert_eq!(filtered.column("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4, 5]);
    assert_eq!(filtered.column("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 40.0, 50.0]);
}

#[test]
fn test_filter_preserves_schema() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[true, true, true, true, true]).unwrap();
    let mask = mask_col.view().unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.column_names(), vec!["id", "score"]);
    assert_eq!(filtered.column_index("id"), Some(0));
    assert_eq!(filtered.column_index("score"), Some(1));
}

#[test]
fn test_filter_all_false() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[false, false, false, false, false]).unwrap();
    let mask = mask_col.view().unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 0);
    assert_eq!(filtered.column_names(), vec!["id", "score"]);
}

// ---------------------------------------------------------------------------
// filter_isin
// ---------------------------------------------------------------------------

#[test]
fn test_filter_isin_basic() {
    let df = sample_df(); // ids: 3,1,4,1,5
    let values = Column::from_slice(&[1i32, 5]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin("id", &values_view).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 3); // rows with id 1,1,5
    let mut ids = filtered.column("id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![1, 1, 5]);
}

#[test]
fn test_filter_isin_no_matches() {
    let df = sample_df(); // ids: 3,1,4,1,5
    let values = Column::from_slice(&[99i32, 100]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin("id", &values_view).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 0);
}

#[test]
fn test_filter_isin_all_match() {
    let df = sample_df(); // ids: 3,1,4,1,5
    let values = Column::from_slice(&[1i32, 3, 4, 5]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin("id", &values_view).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 5);
}

#[test]
fn test_filter_isin_preserves_schema() {
    let df = sample_df();
    let values = Column::from_slice(&[1i32]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin("id", &values_view).unwrap();

    assert_eq!(filtered.column_names(), vec!["id", "score"]);
    assert_eq!(filtered.column_index("id"), Some(0));
    assert_eq!(filtered.column_index("score"), Some(1));
}

#[test]
fn test_filter_isin_nonexistent_column() {
    let df = sample_df();
    let values = Column::from_slice(&[1i32]).unwrap();
    let values_view = values.view().unwrap();
    let result = df.filter_isin("nonexistent", &values_view);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_filter_isin_float64() {
    let df = sample_df(); // scores: 30.0, 10.0, 40.0, 15.0, 50.0
    let values = Column::from_slice(&[10.0f64, 50.0]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin("score", &values_view).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 2);
    let mut scores = filtered.column("score").unwrap().to_vec::<f64>().unwrap();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(scores, vec![10.0, 50.0]);
}

#[test]
fn test_filter_isin_empty_values() {
    let df = sample_df();
    let values = Column::from_slice::<i32>(&[]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin("id", &values_view).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 0);
}

// ---------------------------------------------------------------------------
// sort
// ---------------------------------------------------------------------------

#[test]
fn test_sort_single_column_ascending() {
    let df = sample_df();
    let sorted = df.sort(&["id"], &[Order::Ascending]).unwrap();

    assert_eq!(sorted.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
    assert_eq!(sorted.column_names(), vec!["id", "score"]);
}

#[test]
fn test_sort_single_column_descending() {
    let df = sample_df();
    let sorted = df.sort(&["score"], &[Order::Descending]).unwrap();

    assert_eq!(sorted.column("score").unwrap().to_vec::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

#[test]
fn test_sort_by_subset_of_columns() {
    // Sort by id ascending, then score descending within each id group.
    let ids = Column::from_slice(&[2i32, 1, 2, 1]).unwrap();
    let scores = Column::from_slice(&[20.0f64, 10.0, 40.0, 30.0]).unwrap();
    let df = DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap();

    let sorted = df.sort(&["id", "score"], &[Order::Ascending, Order::Descending]).unwrap();

    assert_eq!(sorted.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 2, 2]);
    assert_eq!(sorted.column("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 20.0]);
}

#[test]
fn test_sort_nonexistent_column_returns_error() {
    let df = sample_df();
    let result = df.sort(&["nonexistent"], &[Order::Ascending]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_sort_mismatched_orders_count() {
    let df = sample_df();
    // 2 column names but only 1 order → error
    let result = df.sort(&["id", "score"], &[Order::Ascending]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_sort_preserves_schema() {
    let df = sample_df();
    let sorted = df.sort(&["id"], &[Order::Ascending]).unwrap();
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

    assert_eq!(top2.num_rows().unwrap(), 2);
    assert_eq!(top2.column("score").unwrap().to_vec::<f64>().unwrap(), vec![50.0, 40.0]);
}

#[test]
fn test_nsmallest() {
    let df = sample_df();
    let bot2 = df.nsmallest(2, "id").unwrap();

    assert_eq!(bot2.num_rows().unwrap(), 2);
    assert_eq!(bot2.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1]);
}

#[test]
fn test_nlargest_nonexistent_column() {
    let df = sample_df();
    let result = df.nlargest(2, "missing");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// distinct
// ---------------------------------------------------------------------------

#[test]
fn test_distinct_by_single_column() {
    // Duplicate ids: 3,1,4,1,5 → distinct ids should be {1,3,4,5}
    let df = sample_df();
    let deduped = df.distinct(&["id"], DuplicateKeepOption::KeepAny).unwrap();

    assert_eq!(deduped.num_rows().unwrap(), 4); // 4 distinct ids
    assert_eq!(deduped.column_names(), vec!["id", "score"]);
}

#[test]
fn test_distinct_preserves_schema() {
    let df = sample_df();
    let deduped = df.distinct(&["id"], DuplicateKeepOption::KeepFirst).unwrap();
    assert_eq!(deduped.column_names(), vec!["id", "score"]);
    assert_eq!(deduped.column_index("id"), Some(0));
    assert_eq!(deduped.column_index("score"), Some(1));
}

#[test]
fn test_distinct_nonexistent_column() {
    let df = sample_df();
    let result = df.distinct(&["nope"], DuplicateKeepOption::KeepAny);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// drop_nulls
// ---------------------------------------------------------------------------

#[test]
fn test_drop_nulls_basic() {
    let ids = Column::from_slice_nullable(&[1i32, 0, 3, 0, 5], &[true, false, true, false, true]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap();

    let cleaned = df.drop_nulls(&["id"], 1).unwrap();

    assert_eq!(cleaned.num_rows().unwrap(), 3);
    assert_eq!(cleaned.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 3, 5]);
    assert_eq!(cleaned.column("val").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 30.0, 50.0]);
}

#[test]
fn test_drop_nulls_preserves_schema() {
    let ids = Column::from_slice_nullable(&[1i32, 0], &[true, false]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0]).unwrap();
    let df = DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap();

    let cleaned = df.drop_nulls(&["id"], 1).unwrap();
    assert_eq!(cleaned.column_names(), vec!["id", "val"]);
}

#[test]
fn test_drop_nulls_nonexistent_column() {
    let c = Column::from_slice(&[1i32]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![c]).unwrap();

    let result = df.drop_nulls(&["missing"], 1);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// drop_nulls_any
// ---------------------------------------------------------------------------

#[test]
fn test_drop_nulls_any_equivalent_to_drop_nulls_with_full_threshold() {
    let ids = Column::from_slice_nullable(&[1i32, 0, 3, 0, 5], &[true, false, true, false, true]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap();

    let subset = &["id"];
    let expected = df.drop_nulls(subset, subset.len() as i32).unwrap();
    let actual = df.drop_nulls_any(subset).unwrap();

    assert_eq!(actual.num_rows().unwrap(), expected.num_rows().unwrap());
    assert_eq!(
        actual.column("id").unwrap().to_vec::<i32>().unwrap(),
        expected.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        actual.column("val").unwrap().to_vec::<f64>().unwrap(),
        expected.column("val").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_drop_nulls_any_mixed_null_patterns() {
    // Two subset columns with nulls in different positions.
    let a = Column::from_slice_nullable(&[1i32, 2, 3, 4], &[true, false, true, true]).unwrap();
    let b = Column::from_slice_nullable(&[10i32, 20, 0, 40], &[true, true, false, true]).unwrap();
    let c = Column::from_slice(&[100.0f64, 200.0, 300.0, 400.0]).unwrap();
    let df = DataFrame::new(vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![a, b, c]).unwrap();

    // drop_nulls_any on ["a", "b"]: drop rows where ANY of a or b is null.
    // Row 0: a=1, b=10 → kept
    // Row 1: a=null → dropped
    // Row 2: b=null → dropped
    // Row 3: a=4, b=40 → kept
    let cleaned = df.drop_nulls_any(&["a", "b"]).unwrap();

    assert_eq!(cleaned.num_rows().unwrap(), 2);
    assert_eq!(cleaned.column("a").unwrap().to_vec::<i32>().unwrap(), vec![1, 4]);
    assert_eq!(cleaned.column("b").unwrap().to_vec::<i32>().unwrap(), vec![10, 40]);
    assert_eq!(cleaned.column("c").unwrap().to_vec::<f64>().unwrap(), vec![100.0, 400.0]);
}

#[test]
fn test_drop_nulls_any_empty_subset_returns_unchanged() {
    let ids = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::new(vec!["id".to_string()], vec![ids]).unwrap();

    // Empty subset → no columns to check for nulls → all rows kept.
    let result = df.drop_nulls_any(&[]).unwrap();
    assert_eq!(result.num_rows().unwrap(), 3);
}

#[test]
fn test_drop_nulls_any_no_nulls() {
    let c = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![c]).unwrap();

    let result = df.drop_nulls_any(&["x"]).unwrap();
    assert_eq!(result.num_rows().unwrap(), 3);
}

#[test]
fn test_drop_nulls_any_all_null() {
    let c = Column::from_slice_nullable(&[0i32, 0, 0], &[false, false, false]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![c]).unwrap();

    let result = df.drop_nulls_any(&["x"]).unwrap();
    assert_eq!(result.num_rows().unwrap(), 0);
}

#[test]
fn test_drop_nulls_any_preserves_schema() {
    let a = Column::from_slice_nullable(&[1i32, 0], &[true, false]).unwrap();
    let b = Column::from_slice(&[10.0f64, 20.0]).unwrap();
    let df = DataFrame::new(vec!["a".to_string(), "b".to_string()], vec![a, b]).unwrap();

    let cleaned = df.drop_nulls_any(&["a"]).unwrap();
    assert_eq!(cleaned.column_names(), vec!["a", "b"]);
    assert_eq!(cleaned.column_index("a"), Some(0));
    assert_eq!(cleaned.column_index("b"), Some(1));
}

#[test]
fn test_drop_nulls_any_nonexistent_column() {
    let c = Column::from_slice(&[1i32]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![c]).unwrap();

    let result = df.drop_nulls_any(&["missing"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}
