//! Tests verifying that Arc-based column sharing works correctly.
//!
//! These tests demonstrate that zero-copy operations (clone, select, rename,
//! with_column) share GPU memory via `Arc<Column>` rather than deep-copying.
//!
//! Sharing is verified by comparing device pointers: two `ColumnView`s that
//! originate from the same `Arc<Column>` will report the same `as_ptr()`.

use cudf::{column::Column, dataframe::DataFrame};

/// Helper: build a 3-row, 3-column DataFrame with distinct column types.
fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let flags = Column::from_slice(&[true, false, true]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string(), "flag".to_string()], vec![ids, scores, flags]).unwrap()
}

/// Returns the raw device data pointer for a named column in a DataFrame.
///
/// Two columns that share the same underlying GPU allocation will have
/// the same device pointer.
fn device_ptr(df: &DataFrame, name: &str) -> *const std::ffi::c_void {
    df.series_view(name).unwrap().device_data_ptr().unwrap()
}

// ---------------------------------------------------------------------------
// clone() produces shared columns
// ---------------------------------------------------------------------------

#[test]
fn test_clone_shares_all_columns() {
    let df = sample_df();
    let copied = df.clone();

    // Both DataFrames should point to the same GPU memory for every column.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&copied, "id"));
    assert_eq!(device_ptr(&df, "score"), device_ptr(&copied, "score"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&copied, "flag"));
}

#[test]
fn test_clone_both_dfs_independently_readable() {
    let df = sample_df();
    let copied = df.clone();

    // Both DataFrames should be independently readable.
    assert_eq!(df.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(copied.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    assert_eq!(df.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(copied.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);

    assert_eq!(df.series_view("flag").unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(copied.series_view("flag").unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_clone_preserves_schema() {
    let df = sample_df();
    let copied = df.clone();

    assert_eq!(df.num_columns(), copied.num_columns());
    assert_eq!(df.num_rows(), copied.num_rows());
    assert_eq!(df.column_names(), copied.column_names());
}

// ---------------------------------------------------------------------------
// select() shares subset of columns
// ---------------------------------------------------------------------------

#[test]
fn test_select_shares_selected_columns() {
    let df = sample_df();
    let selected = df.select(&["id", "flag"]).unwrap();

    // Selected columns should point to the same GPU memory.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&selected, "id"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&selected, "flag"));
}

#[test]
fn test_select_original_still_usable() {
    let df = sample_df();
    let selected = df.select(&["score"]).unwrap();

    // Original DataFrame still has all columns and correct data.
    assert_eq!(df.num_columns(), 3);
    assert_eq!(df.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(df.series_view("flag").unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);

    // Selected DataFrame has only the chosen column.
    assert_eq!(selected.num_columns(), 1);
    assert_eq!(selected.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_select_reorder_shares_columns() {
    let df = sample_df();
    let reordered = df.select(&["flag", "id"]).unwrap();

    // Even after reordering, the GPU memory is shared.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&reordered, "id"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&reordered, "flag"));
}

// ---------------------------------------------------------------------------
// rename() shares all column data
// ---------------------------------------------------------------------------

#[test]
fn test_rename_shares_all_column_data() {
    let df = sample_df();
    let renamed = df.rename_column("id", "identifier").unwrap();

    // The renamed column should share the same GPU memory.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&renamed, "identifier"));

    // Other columns should also be shared.
    assert_eq!(device_ptr(&df, "score"), device_ptr(&renamed, "score"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&renamed, "flag"));
}

#[test]
fn test_rename_original_unchanged() {
    let df = sample_df();
    let renamed = df.rename_column("score", "points").unwrap();

    // Original still has "score", renamed has "points".
    assert_eq!(df.column_names(), &["id", "score", "flag"]);
    assert_eq!(renamed.column_names(), &["id", "points", "flag"]);

    // Data is the same.
    assert_eq!(df.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(renamed.series_view("points").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// with_column() only allocates for the new/replaced column
// ---------------------------------------------------------------------------

#[test]
fn test_with_column_unchanged_columns_shared() {
    let df = sample_df();
    let new_col = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let result = df.with_column("extra", new_col).unwrap();

    // Existing columns should share GPU memory.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&result, "id"));
    assert_eq!(device_ptr(&df, "score"), device_ptr(&result, "score"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&result, "flag"));
}

#[test]
fn test_with_column_replaced_column_has_new_allocation() {
    let df = sample_df();
    let original_score_ptr = device_ptr(&df, "score");

    let new_score = Column::from_slice(&[99.0f64, 88.0, 77.0]).unwrap();
    let result = df.with_column("score", new_score).unwrap();

    // The replaced column should have a DIFFERENT device pointer (new GPU allocation).
    assert_ne!(original_score_ptr, device_ptr(&result, "score"));

    // Unchanged columns should still be shared.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&result, "id"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&result, "flag"));
}

#[test]
fn test_with_column_replaced_column_has_correct_data() {
    let df = sample_df();
    let new_score = Column::from_slice(&[99.0f64, 88.0, 77.0]).unwrap();
    let result = df.with_column("score", new_score).unwrap();

    assert_eq!(result.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![99.0, 88.0, 77.0]);

    // Original is unaffected.
    assert_eq!(df.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// Multiple clone() calls don't OOM (zero-cost sharing)
// ---------------------------------------------------------------------------

#[test]
fn test_many_clones_are_zero_cost() {
    let df = sample_df();

    // Create 1000 clones. Since clone() is just Arc::clone, this should use
    // virtually no additional GPU memory.
    let copies: Vec<DataFrame> = (0..1000).map(|_| df.clone()).collect();

    // All copies should be independently readable with correct data.
    for (i, c) in copies.iter().enumerate() {
        assert_eq!(c.num_columns(), 3, "copy {} has wrong column count", i);
        assert_eq!(c.num_rows(), 3, "copy {} has wrong row count", i);
    }

    // Spot-check a few copies share the same GPU pointers.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&copies[0], "id"));
    assert_eq!(device_ptr(&df, "id"), device_ptr(&copies[500], "id"));
    assert_eq!(device_ptr(&df, "id"), device_ptr(&copies[999], "id"));
}

#[test]
fn test_many_clones_readable_after_original_dropped() {
    let copies: Vec<DataFrame> = {
        let df = sample_df();
        (0..100).map(|_| df.clone()).collect()
        // df is dropped here
    };

    // All copies should still be readable (Arc refcount keeps Column alive).
    for c in &copies {
        assert_eq!(c.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
        assert_eq!(c.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(c.series_view("flag").unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    }
}

// ---------------------------------------------------------------------------
// Combined operations preserve sharing where expected
// ---------------------------------------------------------------------------

#[test]
fn test_clone_then_select_shares_columns() {
    let df = sample_df();
    let copied = df.clone();
    let selected = copied.select(&["id"]).unwrap();

    // All three DataFrames share the "id" column.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&copied, "id"));
    assert_eq!(device_ptr(&df, "id"), device_ptr(&selected, "id"));
}

#[test]
fn test_rename_then_select_shares_columns() {
    let df = sample_df();
    let renamed = df.rename_column("id", "identifier").unwrap();
    let selected = renamed.select(&["identifier", "flag"]).unwrap();

    // The "identifier" column in selected is the same as "id" in original.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&selected, "identifier"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&selected, "flag"));
}

#[test]
fn test_with_column_then_clone_shares_new_and_old() {
    let df = sample_df();
    let new_col = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let with_extra = df.with_column("extra", new_col).unwrap();
    let copied = with_extra.clone();

    // Original columns are shared across all three DataFrames.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&with_extra, "id"));
    assert_eq!(device_ptr(&df, "id"), device_ptr(&copied, "id"));

    // The new column is shared between with_extra and copied.
    assert_eq!(device_ptr(&with_extra, "extra"), device_ptr(&copied, "extra"));
}

#[test]
fn test_chained_renames_share_all_data() {
    let df = sample_df();
    let r1 = df.rename_column("id", "a").unwrap();
    let r2 = r1.rename_column("score", "b").unwrap();
    let r3 = r2.rename_column("flag", "c").unwrap();

    // Despite three renames, all columns share the same GPU data.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&r3, "a"));
    assert_eq!(device_ptr(&df, "score"), device_ptr(&r3, "b"));
    assert_eq!(device_ptr(&df, "flag"), device_ptr(&r3, "c"));
}

#[test]
fn test_select_subset_then_with_column_shares_retained() {
    let df = sample_df();
    let selected = df.select(&["id", "score"]).unwrap();
    let new_flag = Column::from_slice(&[false, true, false]).unwrap();
    let result = selected.with_column("new_flag", new_flag).unwrap();

    // "id" and "score" are shared all the way through.
    assert_eq!(device_ptr(&df, "id"), device_ptr(&result, "id"));
    assert_eq!(device_ptr(&df, "score"), device_ptr(&result, "score"));
}
