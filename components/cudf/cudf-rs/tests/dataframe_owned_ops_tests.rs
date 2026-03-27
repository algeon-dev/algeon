#![allow(deprecated)] // Tests exercise deprecated DataFrame::column()
//! Tests for consuming `_owned()` DataFrame method variants.
//!
//! Each `_owned` variant takes `self` by value. For libcudf-calling methods
//! (filter, sort, drop_nulls_any, head, tail), this allows the input's
//! `Arc<Column>`s to drop sooner when they're the last reference. For zero-copy
//! methods (select, with_column), the owned variant can `remove()` from the
//! IndexMap instead of `Arc::clone`, avoiding even the refcount bump.

use cudf::{column::Column, dataframe::DataFrame, error::ErrorKind, sorting::Order};

/// Helper: build a 5-row, 2-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = Column::from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

// ---------------------------------------------------------------------------
// filter_by_mask_owned
// ---------------------------------------------------------------------------

#[test]
fn test_filter_by_mask_owned_basic() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[true, false, true, false, true]).unwrap();
    let mask = mask_col.view().unwrap();
    let filtered = df.filter_by_mask_owned(&mask).unwrap();

    assert_eq!(filtered.num_rows(), 3);
    assert_eq!(filtered.column("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4, 5]);
    assert_eq!(filtered.column("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 40.0, 50.0]);
}

#[test]
fn test_filter_by_mask_owned_preserves_schema() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[true, true, true, true, true]).unwrap();
    let mask = mask_col.view().unwrap();
    let filtered = df.filter_by_mask_owned(&mask).unwrap();

    assert_eq!(filtered.column_names(), vec!["id", "score"]);
    assert_eq!(filtered.column_index("id"), Some(0));
    assert_eq!(filtered.column_index("score"), Some(1));
}

#[test]
fn test_filter_by_mask_owned_matches_borrow_variant() {
    let df1 = sample_df();
    let df2 = sample_df();
    let mask_col = Column::from_slice(&[true, false, true, false, true]).unwrap();
    let mask = mask_col.view().unwrap();

    let borrow_result = df1.filter_by_mask(&mask).unwrap();
    let owned_result = df2.filter_by_mask_owned(&mask).unwrap();

    assert_eq!(
        borrow_result.column("id").unwrap().to_vec::<i32>().unwrap(),
        owned_result.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        borrow_result.column("score").unwrap().to_vec::<f64>().unwrap(),
        owned_result.column("score").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_filter_by_mask_owned_rejects_non_bool_mask() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[1i32, 0, 1, 0, 1]).unwrap();
    let mask = mask_col.view().unwrap();

    let err = df.filter_by_mask_owned(&mask).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
    assert!(err.to_string().contains("Bool8"));
}

#[test]
fn test_filter_by_mask_owned_rejects_wrong_length_mask() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[true, false]).unwrap();
    let mask = mask_col.view().unwrap();

    let err = df.filter_by_mask_owned(&mask).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.to_string().contains("length"));
}

// ---------------------------------------------------------------------------
// filter_isin_owned
// ---------------------------------------------------------------------------

#[test]
fn test_filter_isin_owned_basic() {
    let df = sample_df(); // ids: 3,1,4,1,5
    let values = Column::from_slice(&[1i32, 5]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin_owned("id", &values_view).unwrap();

    assert_eq!(filtered.num_rows(), 3); // rows with id 1,1,5
    let mut ids = filtered.column("id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![1, 1, 5]);
}

#[test]
fn test_filter_isin_owned_preserves_schema() {
    let df = sample_df();
    let values = Column::from_slice(&[1i32]).unwrap();
    let values_view = values.view().unwrap();
    let filtered = df.filter_isin_owned("id", &values_view).unwrap();

    assert_eq!(filtered.column_names(), vec!["id", "score"]);
}

#[test]
fn test_filter_isin_owned_nonexistent_column() {
    let df = sample_df();
    let values = Column::from_slice(&[1i32]).unwrap();
    let values_view = values.view().unwrap();
    let result = df.filter_isin_owned("nonexistent", &values_view);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// sort_owned
// ---------------------------------------------------------------------------

#[test]
fn test_sort_owned_ascending() {
    let df = sample_df();
    let sorted = df.sort_owned(&["id"], &[Order::Ascending]).unwrap();

    assert_eq!(sorted.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
    assert_eq!(sorted.column_names(), vec!["id", "score"]);
}

#[test]
fn test_sort_owned_descending() {
    let df = sample_df();
    let sorted = df.sort_owned(&["score"], &[Order::Descending]).unwrap();

    assert_eq!(sorted.column("score").unwrap().to_vec::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

#[test]
fn test_sort_owned_matches_borrow_variant() {
    let df1 = sample_df();
    let df2 = sample_df();

    let borrow_result = df1.sort(&["id"], &[Order::Ascending]).unwrap();
    let owned_result = df2.sort_owned(&["id"], &[Order::Ascending]).unwrap();

    assert_eq!(
        borrow_result.column("id").unwrap().to_vec::<i32>().unwrap(),
        owned_result.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        borrow_result.column("score").unwrap().to_vec::<f64>().unwrap(),
        owned_result.column("score").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_sort_owned_mismatched_orders_count() {
    let df = sample_df();
    let result = df.sort_owned(&["id", "score"], &[Order::Ascending]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// drop_nulls_any_owned
// ---------------------------------------------------------------------------

#[test]
fn test_drop_nulls_any_owned_basic() {
    let ids = Column::from_slice_nullable(&[1i32, 0, 3, 0, 5], &[true, false, true, false, true]).unwrap();
    let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap();

    let cleaned = df.drop_nulls_any_owned(&["id"]).unwrap();

    assert_eq!(cleaned.num_rows(), 3);
    assert_eq!(cleaned.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 3, 5]);
    assert_eq!(cleaned.column("val").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 30.0, 50.0]);
}

#[test]
fn test_drop_nulls_any_owned_matches_borrow_variant() {
    let make_df = || {
        let ids = Column::from_slice_nullable(&[1i32, 0, 3, 0, 5], &[true, false, true, false, true]).unwrap();
        let vals = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
        DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap()
    };

    let borrow_result = make_df().drop_nulls_any(&["id"]).unwrap();
    let owned_result = make_df().drop_nulls_any_owned(&["id"]).unwrap();

    assert_eq!(
        borrow_result.column("id").unwrap().to_vec::<i32>().unwrap(),
        owned_result.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        borrow_result.column("val").unwrap().to_vec::<f64>().unwrap(),
        owned_result.column("val").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_drop_nulls_any_owned_preserves_schema() {
    let a = Column::from_slice_nullable(&[1i32, 0], &[true, false]).unwrap();
    let b = Column::from_slice(&[10.0f64, 20.0]).unwrap();
    let df = DataFrame::new(vec!["a".to_string(), "b".to_string()], vec![a, b]).unwrap();

    let cleaned = df.drop_nulls_any_owned(&["a"]).unwrap();
    assert_eq!(cleaned.column_names(), vec!["a", "b"]);
    assert_eq!(cleaned.column_index("a"), Some(0));
    assert_eq!(cleaned.column_index("b"), Some(1));
}

// ---------------------------------------------------------------------------
// select_owned
// ---------------------------------------------------------------------------

#[test]
fn test_select_owned_basic() {
    let df = sample_df();
    let selected = df.select_owned(&["score"]).unwrap();

    assert_eq!(selected.num_columns(), 1);
    assert_eq!(selected.num_rows(), 5);
    assert_eq!(selected.column_names(), vec!["score"]);
    assert_eq!(selected.column("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 15.0, 50.0]);
}

#[test]
fn test_select_owned_reorders_columns() {
    let df = sample_df();
    let selected = df.select_owned(&["score", "id"]).unwrap();

    assert_eq!(selected.column_names(), vec!["score", "id"]);
    assert_eq!(selected.column_index("score"), Some(0));
    assert_eq!(selected.column_index("id"), Some(1));
}

#[test]
fn test_select_owned_nonexistent_column() {
    let df = sample_df();
    let result = df.select_owned(&["missing"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_owned_matches_borrow_variant() {
    let df1 = sample_df();
    let df2 = sample_df();

    let borrow_result = df1.select(&["score"]).unwrap();
    let owned_result = df2.select_owned(&["score"]).unwrap();

    assert_eq!(
        borrow_result.column("score").unwrap().to_vec::<f64>().unwrap(),
        owned_result.column("score").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_select_owned_duplicate_column_names_returns_error() {
    let df = sample_df();
    let result = df.select_owned(&["id", "id"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// with_column_owned
// ---------------------------------------------------------------------------

#[test]
fn test_with_column_owned_adds_new_column() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let new_col = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let result = df.with_column_owned("b", new_col).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);
    assert_eq!(result.column("a").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(result.column("b").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_with_column_owned_replaces_existing() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let replacement = Column::from_slice(&[99i32, 100]).unwrap();
    let result = df.with_column_owned("b", replacement).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);
    assert_eq!(result.column("b").unwrap().to_vec::<i32>().unwrap(), vec![99, 100]);
}

#[test]
fn test_with_column_owned_mismatched_row_count() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let new_col = Column::from_slice(&[10i32, 20]).unwrap();
    let err = df.with_column_owned("b", new_col).err().expect("expected row count mismatch");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// head_owned
// ---------------------------------------------------------------------------

#[test]
fn test_head_owned_basic() {
    let df = sample_df();
    let h = df.head_owned(2).unwrap();

    assert_eq!(h.num_rows(), 2);
    assert_eq!(h.column("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 1]);
    assert_eq!(h.column("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 10.0]);
}

#[test]
fn test_head_owned_clamps_to_num_rows() {
    let df = sample_df();
    let h = df.head_owned(100).unwrap();
    assert_eq!(h.num_rows(), 5);
}

#[test]
fn test_head_owned_zero_rows() {
    let df = sample_df();
    let h = df.head_owned(0).unwrap();
    assert_eq!(h.num_rows(), 0);
    assert_eq!(h.column_names(), vec!["id", "score"]);
}

#[test]
fn test_head_owned_matches_borrow_variant() {
    let df1 = sample_df();
    let df2 = sample_df();

    let borrow_result = df1.head(3).unwrap();
    let owned_result = df2.head_owned(3).unwrap();

    assert_eq!(
        borrow_result.column("id").unwrap().to_vec::<i32>().unwrap(),
        owned_result.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// tail_owned
// ---------------------------------------------------------------------------

#[test]
fn test_tail_owned_basic() {
    let df = sample_df();
    let t = df.tail_owned(2).unwrap();

    assert_eq!(t.num_rows(), 2);
    assert_eq!(t.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 5]);
    assert_eq!(t.column("score").unwrap().to_vec::<f64>().unwrap(), vec![15.0, 50.0]);
}

#[test]
fn test_tail_owned_clamps_to_num_rows() {
    let df = sample_df();
    let t = df.tail_owned(100).unwrap();
    assert_eq!(t.num_rows(), 5);
}

#[test]
fn test_tail_owned_zero_rows() {
    let df = sample_df();
    let t = df.tail_owned(0).unwrap();
    assert_eq!(t.num_rows(), 0);
    assert_eq!(t.column_names(), vec!["id", "score"]);
}

#[test]
fn test_tail_owned_matches_borrow_variant() {
    let df1 = sample_df();
    let df2 = sample_df();

    let borrow_result = df1.tail(3).unwrap();
    let owned_result = df2.tail_owned(3).unwrap();

    assert_eq!(
        borrow_result.column("id").unwrap().to_vec::<i32>().unwrap(),
        owned_result.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// Method chaining
// ---------------------------------------------------------------------------

#[test]
fn test_owned_method_chaining() {
    let df = sample_df(); // ids: 3,1,4,1,5  scores: 30,10,40,15,50

    // Chain: filter (keep id > 1) → sort by id → head 2
    let mask_col = Column::from_slice(&[true, false, true, false, true]).unwrap();
    let mask = mask_col.view().unwrap();

    let result = df
        .filter_by_mask_owned(&mask)
        .unwrap()
        .sort_owned(&["id"], &[Order::Ascending])
        .unwrap()
        .head_owned(2)
        .unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4]);
}

#[test]
fn test_owned_chain_filter_sort_select() {
    let df = sample_df(); // ids: 3,1,4,1,5  scores: 30,10,40,15,50

    let mask_col = Column::from_slice(&[true, true, true, true, true]).unwrap();
    let mask = mask_col.view().unwrap();

    let result = df
        .filter_by_mask_owned(&mask)
        .unwrap()
        .sort_owned(&["score"], &[Order::Descending])
        .unwrap()
        .select_owned(&["score"])
        .unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.column_names(), vec!["score"]);
    assert_eq!(result.column("score").unwrap().to_vec::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

#[test]
fn test_owned_chain_with_column_then_select() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();

    let new_col = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let result = df.with_column_owned("c", new_col).unwrap().select_owned(&["a", "c"]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "c"]);
    assert_eq!(result.column("a").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(result.column("c").unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
}
