//! High-Level API Semantic Baseline Tests (FC-AUDIT)
//!
//! These tests lock down the current borrow/share/copy semantics of every
//! public accessor and conversion on `DataFrame` and `Series` during naming
//! refactors. They serve as a regression gate: if a refactor
//! silently changes a zero-copy path into a deep-copy path (or vice versa),
//! a test here will fail.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{DataFrame, Series, TypeId, core::Column};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 3-row, 2-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("score", &[10.0, 20.0, 30.0]).unwrap();
    DataFrame::from_series([("id", ids), ("score", scores)]).unwrap()
}

/// Get device data pointer for a named column in a DataFrame.
///
/// Two columns sharing the same GPU buffer will have the same pointer.
fn device_ptr(df: &DataFrame, name: &str) -> *const std::ffi::c_void {
    df.series(name).unwrap().as_column_view().unwrap().device_data_ptr().unwrap()
}

// ===========================================================================
// 1. DataFrame::series() — returns owned Series with Arc-shared Column
// ===========================================================================

/// `series()` returns a named `Series` with correct metadata.
#[test]
fn df_series_returns_owned_with_name() {
    let df = sample_df();
    let s = df.series("id").unwrap();

    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);
    assert_eq!(s.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// Two `series()` calls return Series backed by the same GPU buffer.
#[test]
fn df_series_shares_device_pointer() {
    let df = sample_df();
    let ptr1 = df.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = df.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    assert_eq!(ptr1, ptr2);
}

/// `series()` produces an owned value that can outlive the DataFrame.
/// This proves it's task-portable (can be sent to worker threads).
#[test]
fn df_series_outlives_dataframe() {
    let s = {
        let df = sample_df();
        df.series("id").unwrap()
        // df dropped here
    };
    // Series is still usable
    assert_eq!(s.len(), 3);
    assert_eq!(s.as_column_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// Two `series()` calls for the same column produce Series with the
/// same device pointer — proving Arc-sharing (zero GPU copy).
#[test]
fn df_series_two_calls_share_arc() {
    let df = sample_df();
    let s1 = df.series("id").unwrap();
    let s2 = df.series("id").unwrap();

    // Same device pointer proves same underlying Column
    let ptr1 = s1.as_column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = s2.as_column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
}

// ===========================================================================
// 2. DataFrame::get() / contains() — probe and existence check
// ===========================================================================

/// `get()` returns `None` on miss, `Some(Series)` on hit.
#[test]
fn df_get_probe_access() {
    let df = sample_df();
    assert!(df.get("nonexistent").is_none());

    let s = df.get("id").unwrap();
    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `contains()` returns `true` for existing columns, `false` otherwise.
#[test]
fn df_contains_check() {
    let df = sample_df();
    assert!(df.contains("id"));
    assert!(df.contains("score"));
    assert!(!df.contains("nonexistent"));
}

// ===========================================================================
// 3. DataFrame::to_table() — deep-copies GPU data
// ===========================================================================

/// `to_table()` produces a new table with independent GPU memory.
#[test]
fn df_to_table_deep_copies() {
    let df = sample_df();
    let table = df.to_table_copy().unwrap();

    let tv = table.view().unwrap();
    assert_eq!(tv.num_columns().unwrap(), 2);
    assert_eq!(tv.num_rows().unwrap(), 3);

    // The table's column 0 ("id") must have a DIFFERENT device pointer
    let table_col0 = tv.column(0).unwrap();
    let table_ptr = table_col0.device_data_ptr().unwrap();
    let df_ptr = device_ptr(&df, "id");
    assert_ne!(table_ptr, df_ptr, "to_table() must deep-copy; pointers should differ");

    // Data values should still match
    assert_eq!(table_col0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `to_table()` does not consume the DataFrame — source remains usable.
#[test]
fn df_to_table_source_still_usable() {
    let df = sample_df();
    let _table = df.to_table_copy().unwrap();

    // DataFrame is still fully readable
    assert_eq!(df.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.series("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 4. DataFrame::into_table() — conditional deep-copy
// ===========================================================================

/// When the DataFrame is the sole owner of all columns (Arc strong
/// count == 1), `into_table()` can unwrap without copying.
#[test]
fn df_into_table_unique_owner_no_extra_copy() {
    let df = DataFrame::new(vec!["a".into()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let original_ptr = device_ptr(&df, "a");
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["a"]);
    let tv = table.view().unwrap();
    let table_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();

    assert_eq!(original_ptr, table_ptr, "into_table() with unique Arc should transfer without copy");
}

/// When the DataFrame shares columns (Arc strong count > 1),
/// `into_table()` must deep-copy the shared columns.
#[test]
fn df_into_table_shared_columns_deep_copies() {
    let df = sample_df();

    // Create an external reference to keep the Arc refcount > 1.
    let external_series = df.series("id").unwrap();
    let external_ptr = external_series.as_column_view().unwrap().device_data_ptr().unwrap();

    let original_ptr = device_ptr(&df, "id");
    assert_eq!(original_ptr, external_ptr, "series shares same Arc");

    let (table, names) = df.try_into_table().unwrap();
    assert_eq!(names, vec!["id", "score"]);

    let tv = table.view().unwrap();
    let table_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();

    assert_ne!(table_ptr, external_ptr, "into_table() with shared Arc must deep-copy; pointers should differ");

    // External series is still valid (Arc keeps it alive).
    assert_eq!(external_series.as_column_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `into_table()` consumes the DataFrame — data values are preserved.
#[test]
fn df_into_table_values_preserved() {
    let df = sample_df();
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["id", "score"]);
    let tv = table.view().unwrap();
    assert_eq!(tv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(tv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 5. Schema-only DataFrame metadata ops remain zero-copy
// ===========================================================================

/// `filter(&Series)` produces a new GPU allocation.
#[test]
fn filter_allocates_new_gpu_data() {
    let df = sample_df();
    let original_ptr = device_ptr(&df, "id");

    let mask = cudf::Series::new::<bool>("mask", &[true, false, true]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    let filtered_ptr = device_ptr(&filtered, "id");
    assert_ne!(original_ptr, filtered_ptr);
}

/// `sort_values` produces consistent results across calls.
#[test]
fn sort_values_consistent_results() {
    let df1 = sample_df();
    let df2 = sample_df();

    let result1 = df1.sort_values(["id"], cudf::SortOptions::new().with_descending(vec![true])).unwrap();
    let result2 = df2.sort_values(["id"], cudf::SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(
        result1.series("id").unwrap().to_vec::<i32>().unwrap(),
        result2.series("id").unwrap().to_vec::<i32>().unwrap(),
    );
}

/// `select` is schema-only and Arc-shares selected columns.
#[test]
fn select_shares_arc() {
    let df = sample_df();
    let original_ptr = device_ptr(&df, "id");

    let selected = df.select(["id"]).unwrap();
    let selected_ptr = device_ptr(&selected, "id");

    assert_eq!(original_ptr, selected_ptr);
}

/// `rename_column` is a pure schema rewrite that Arc-shares columns.
#[test]
fn rename_column_shares_arc() {
    let df = sample_df();
    let original_ptr = device_ptr(&df, "id");

    let renamed = df.rename_column("id", "identifier").unwrap();
    let renamed_ptr = device_ptr(&renamed, "identifier");

    assert_eq!(original_ptr, renamed_ptr);
}

// ===========================================================================
// 6. Series accessor semantics
// ===========================================================================

/// `Series::as_column_view()` returns a `ColumnView` that points to the
/// same device data as the underlying Column.
#[test]
fn series_column_view_same_device_ptr_as_column() {
    let s = Series::new::<i32>("x", &[1, 2]).unwrap();
    let cv = s.as_column_view().unwrap();
    let col_cv = s.as_column().view().unwrap();

    assert_eq!(cv.device_data_ptr().unwrap(), col_cv.device_data_ptr().unwrap());
}

/// `Series::as_column()` returns `&Column` — a direct borrow.
#[test]
fn series_column_returns_ref() {
    let s = Series::new::<i32>("c", &[1, 2, 3]).unwrap();
    let col_ref: &Column = s.as_column();
    let cv = col_ref.view().unwrap();
    assert_eq!(cv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `Series::into_inner()` consumes the Series and returns the `Arc<Column>`.
/// A clone of the Series shares the same device data.
#[test]
fn series_into_inner_shares_arc() {
    let s = Series::new::<i32>("c", &[1, 2, 3]).unwrap();

    let arc1 = s.clone().into_inner();
    let arc2 = s.into_inner();

    let ptr1 = arc1.view().unwrap().device_data_ptr().unwrap();
    let ptr2 = arc2.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
    assert!(Arc::ptr_eq(&arc1, &arc2));
}

// ===========================================================================
// Supplementary: clone semantics
// ===========================================================================

/// `Series::clone()` is a cheap Arc bump — same device pointer.
#[test]
fn series_clone_shares_gpu_data() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let cloned = s.clone();

    let ptr1 = s.as_column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = cloned.as_column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
    assert_eq!(cloned.name(), Some("x"));
}

/// `Series::rename()` produces a new Series with a different name but
/// the same GPU data.
#[test]
fn series_rename_shares_gpu_data() {
    let s = Series::new::<i32>("old", &[1, 2, 3]).unwrap();
    let renamed = s.rename("new");

    let ptr1 = s.as_column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = renamed.as_column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
    assert_eq!(renamed.name(), Some("new"));
    assert_eq!(s.name(), Some("old"));
}

// ===========================================================================
// Supplementary: high-level methods' GPU-allocating nature
// ===========================================================================

/// `filter()` with a Series mask produces a new GPU allocation.
#[test]
fn df_filter_gpu_allocates() {
    let df = sample_df();
    let mask = Series::new::<bool>("mask", &[true, false, true]).unwrap();

    let filtered = df.filter(&mask).unwrap();
    assert_eq!(filtered.num_rows(), 2);
    assert_eq!(filtered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 3]);

    // New allocation — different device pointer
    assert_ne!(device_ptr(&df, "id"), device_ptr(&filtered, "id"));
}

/// `head()` / `tail()` / `slice()` all produce new GPU allocations.
#[test]
fn df_slicing_ops_gpu_allocate() {
    let df = sample_df();
    let original_ptr = device_ptr(&df, "id");

    let h = df.head(2).unwrap();
    assert_ne!(original_ptr, device_ptr(&h, "id"));

    let t = df.tail(2).unwrap();
    assert_ne!(original_ptr, device_ptr(&t, "id"));

    let s = df.slice(1, 1).unwrap();
    assert_ne!(original_ptr, device_ptr(&s, "id"));
}

/// GPU-allocating unary ops on Series produce new columns.
#[test]
fn series_ops_gpu_allocate() {
    let s = Series::new::<i32>("v", &[1, 2, 3]).unwrap();
    let original_ptr = s.as_column_view().unwrap().device_data_ptr().unwrap();

    let mask = s.gt_scalar(1i32).unwrap();
    let mask_ptr = mask.as_column_view().unwrap().device_data_ptr().unwrap();
    assert_ne!(original_ptr, mask_ptr, "gt_scalar must allocate new column");

    let added = s.add_scalar(10i32).unwrap();
    let add_ptr = added.as_column_view().unwrap().device_data_ptr().unwrap();
    assert_ne!(original_ptr, add_ptr, "add_scalar must allocate new column");
}
