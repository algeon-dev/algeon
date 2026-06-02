//! Facade Semantic Baseline Tests (FC-AUDIT)
//!
//! These tests lock down the current borrow/share/copy semantics of every
//! public accessor and conversion on `DataFrame` and `Series` **before** any
//! renaming work begins. They serve as a regression gate: if a refactor
//! silently changes a zero-copy path into a deep-copy path (or vice versa),
//! a test here will fail.
//!
//! # Semantic Inventory — DataFrame
//!
//! | Method                     | Cost classification        | Task-portable? | Name assessment        |
//! |----------------------------|----------------------------|----------------|------------------------|
//! | `column(&str)`             | Zero-copy (FFI view)       | No (borrows)   | **Misleading**: returns `SeriesView`, not a column |
//! | `series(&str)`             | Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `view()`                   | Zero-copy (borrows)        | No (borrows)   | OK                     |
//! | `to_table()`               | GPU-allocating (deep-copy) | Yes (owned)    | OK — "to_" implies copy |
//! | `into_table()`             | Mixed (conditional copy)   | Yes (owned)    | **Awkward**: implies zero-cost move but may deep-copy |
//! | `clone()`                  | Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `select()`                 | Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `select_owned()`           | Zero-copy (Arc move)       | Yes (owned)    | OK — avoids refcount bump |
//! | `rename()`                 | Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `rename_owned()`           | Zero-copy (Arc move)       | Yes (owned)    | OK                     |
//! | `rename_columns()`         | Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `with_column()`            | Zero-copy for unchanged    | Yes (owned)    | OK                     |
//! | `with_column_owned()`      | Zero-copy for unchanged    | Yes (owned)    | OK                     |
//! | `assign()`                 | Zero-copy for unchanged    | Yes (owned)    | OK                     |
//! | `with_columns(exprs)`      | GPU-allocating per expr    | Yes (owned)    | OK                     |
//! | `drop()` / `drop_columns()`| Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `filter(&Series)`          | GPU-allocating             | Yes (owned)    | OK                     |
//! | `filter_by_mask()`         | GPU-allocating             | Yes (owned)    | **Awkward**: "filter" already implies mask; `_by_mask` is redundant |
//! | `filter_by_mask_owned()`   | GPU-allocating             | Yes (owned)    | **Awkward**: `_owned` suffix unclear to newcomers |
//! | `filter_isin()`            | GPU-allocating             | Yes (owned)    | OK                     |
//! | `filter_isin_owned()`      | GPU-allocating             | Yes (owned)    | **Awkward**: `_owned` suffix |
//! | `sort()`                   | GPU-allocating             | Yes (owned)    | OK                     |
//! | `sort_owned()`             | GPU-allocating             | Yes (owned)    | **Awkward**: `_owned` suffix |
//! | `sort_values()`            | GPU-allocating             | Yes (owned)    | OK                     |
//! | `head()` / `tail()`        | GPU-allocating             | Yes (owned)    | OK                     |
//! | `head_owned()` / `tail_owned()` | GPU-allocating        | Yes (owned)    | **Awkward**: `_owned` suffix |
//! | `slice()`                  | GPU-allocating             | Yes (owned)    | OK                     |
//! | `gather()`                 | GPU-allocating             | Yes (owned)    | OK                     |
//! | `distinct()`               | GPU-allocating             | Yes (owned)    | OK                     |
//! | `drop_nulls()`             | GPU-allocating             | Yes (owned)    | OK                     |
//! | `drop_nulls_any()`         | GPU-allocating             | Yes (owned)    | OK                     |
//! | `drop_nulls_any_owned()`   | GPU-allocating             | Yes (owned)    | **Awkward**: `_owned` suffix |
//! | `dropna(DropNaOptions)`    | GPU-allocating             | Yes (owned)    | OK (cuDF-style facade) |
//! | `drop_duplicates(DistinctOptions)` | GPU-allocating      | Yes (owned)    | OK (cuDF-style facade) |
//! | `fillna(col, &Scalar)`     | GPU-allocating             | Yes (owned)    | OK (cuDF-style facade) |
//! | `fillna_with_series()`     | GPU-allocating             | Yes (owned)    | OK                     |
//! | `replace_nulls()`          | GPU-allocating             | Yes (owned)    | OK                     |
//! | `cast_column()`            | GPU-allocating             | Yes (owned)    | OK                     |
//! | `concat()`                 | GPU-allocating             | Yes (owned)    | OK                     |
//! | `inner_join()` etc.        | GPU-allocating             | Yes (owned)    | OK                     |
//! | `merge()`                  | GPU-allocating             | Yes (owned)    | OK (cuDF-style facade) |
//! | `groupby()`                | Zero-copy (returns builder)| No (borrows)   | OK                     |
//! | `factorize(col)`           | GPU-allocating             | Yes (owned)    | OK                     |
//! | `read_parquet()` etc.      | Host→GPU (I/O)             | Yes (owned)    | OK                     |
//! | `write_parquet()`          | GPU→Host (I/O)             | N/A (no return)| OK                     |
//! | `num_columns()`            | Free (cached metadata)     | N/A            | OK                     |
//! | `num_rows()`               | Free (cached metadata)     | N/A            | OK                     |
//! | `column_names()`           | Free (metadata)            | N/A            | OK                     |
//! | `column_index()`           | Free (metadata)            | N/A            | OK                     |
//!
//! # Semantic Inventory — Series
//!
//! | Method                     | Cost classification        | Task-portable? | Name assessment        |
//! |----------------------------|----------------------------|----------------|------------------------|
//! | `column()`                 | Zero-copy (ref)            | No (borrows)   | **Misleading**: "column" on a Series returns `&Column`, confusing layer |
//! | `column_arc()`             | Zero-copy (ref to Arc)     | No (borrows)   | **Awkward**: leaks implementation detail |
//! | `column_view()`            | Zero-copy (FFI view)       | No (borrows)   | OK but redundant with `view()` |
//! | `view()`                   | Zero-copy (FFI view)       | No (borrows)   | OK — returns `SeriesView` |
//! | `clone()`                  | Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `rename()`                 | Zero-copy (Arc-share)      | Yes (owned)    | OK                     |
//! | `name()`                   | Free (metadata)            | N/A            | OK                     |
//! | `dtype()`                  | FFI call (no GPU alloc)    | N/A            | OK                     |
//! | `len()` / `is_empty()`     | Free (cached)              | N/A            | OK                     |
//! | `null_count()`             | Free (cached)              | N/A            | OK                     |
//! | `to_optional_strings()`    | Host-materializing         | N/A            | OK — "to_" implies copy |
//! | `gt_scalar()` etc.         | GPU-allocating             | Yes (owned)    | OK                     |
//! | `add()` etc.               | GPU-allocating             | Yes (owned)    | OK                     |
//! | `is_null()` / `is_not_null()` | GPU-allocating          | Yes (owned)    | OK                     |
//! | `astype()`                 | GPU-allocating             | Yes (owned)    | OK (cuDF-style)        |
//! | `fillna()` / `fillna_with_series()` | GPU-allocating    | Yes (owned)    | OK                     |
//! | `isin()`                   | GPU-allocating             | Yes (owned)    | OK                     |
//! | `unique()`                 | GPU-allocating             | Yes (owned)    | OK                     |
//! | `nunique()`                | GPU-allocating             | Yes (owned)    | OK                     |
//! | `factorize()`              | GPU-allocating             | Yes (owned)    | OK                     |
//! | `sort_values()`            | GPU-allocating             | Yes (owned)    | OK                     |
//!
//! # Flagged names (actively misleading)
//!
//! 1. **`DataFrame::column()`** — Returns a `SeriesView`, not a `Column` or `Series`.
//!    A user reading `.column("x")` expects an owned column or at least a `ColumnView`.
//! 2. **`Series::column()`** — Returns `&Column`. Confuses facade/core boundary.
//! 3. **`DataFrame::into_table()`** — Implies a zero-cost ownership transfer,
//!    but may deep-copy shared columns.
//!
//! # Flagged names (merely awkward)
//!
//! 1. **`*_owned` suffix** on DataFrame methods — The name doesn't hint at
//!    *why* consuming self matters (memory lifetime). Could be `*_move` or
//!    the borrow variant could just be removed in favor of always-consuming.
//! 2. **`Series::column_arc()`** — Leaks implementation detail (`Arc`) into
//!    the public facade API.
//! 3. **`DataFrame::filter_by_mask()`** — Redundant with `filter()`.
//! 4. **`Series::column_view()`** — Redundant with `view()` except it returns
//!    bare `ColumnView` instead of `SeriesView`. The name is confusing alongside
//!    `Series::column()`.
//!
//! All tests require a GPU.

#![allow(deprecated)]

use std::sync::Arc;

use cudf::{
    column::Column,
    dataframe::DataFrame,
    series::Series,
    types::TypeId,
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 3-row, 2-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("score", &[10.0, 20.0, 30.0]).unwrap();
    DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap()
}

/// Get device data pointer for a named column in a DataFrame.
///
/// Two columns sharing the same GPU buffer will have the same pointer.
fn device_ptr(df: &DataFrame, name: &str) -> *const std::ffi::c_void {
    df.column(name).unwrap().device_data_ptr().unwrap()
}

// ===========================================================================
// 1. DataFrame::column() — returns borrowed SeriesView
// ===========================================================================

/// `column()` returns a `SeriesView` (borrowed), not an owned `Series`.
/// The return type carries a lifetime tied to the DataFrame.
#[test]
fn df_column_returns_series_view_with_name() {
    let df = sample_df();
    let sv = df.column("id").unwrap();

    // SeriesView carries the column name
    assert_eq!(sv.name(), Some("id"));

    // SeriesView derefs to ColumnView — we can read data
    assert_eq!(sv.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(sv.len().unwrap(), 3);
    assert_eq!(sv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// The device pointer from `column()` matches the one from `series()` —
/// both refer to the same underlying GPU buffer.
#[test]
fn df_column_view_points_to_same_gpu_data_as_series() {
    let df = sample_df();
    let sv = df.column("id").unwrap();
    let view_ptr = sv.device_data_ptr().unwrap();

    let series = df.series("id").unwrap();
    let series_cv = series.column_view().unwrap();
    let series_ptr = series_cv.device_data_ptr().unwrap();

    assert_eq!(view_ptr, series_ptr);
}

/// `column()` borrows from the DataFrame: the returned `SeriesView` has
/// a lifetime tied to `&self`. We verify this indirectly by checking
/// that two calls produce views with the same device pointer.
#[test]
fn df_column_two_calls_same_device_ptr() {
    let df = sample_df();
    let sv1 = df.column("id").unwrap();
    let sv2 = df.column("id").unwrap();

    assert_eq!(sv1.device_data_ptr().unwrap(), sv2.device_data_ptr().unwrap());
}

// ===========================================================================
// 2. DataFrame::series() — returns owned Series with Arc-shared Column
// ===========================================================================

/// `series()` returns an owned `Series` that shares the `Arc<Column>`.
/// No GPU data is copied.
#[test]
fn df_series_returns_owned_with_name() {
    let df = sample_df();
    let s = df.series("score").unwrap();

    assert_eq!(s.name(), Some("score"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
}

/// `series()` shares the same GPU buffer — device pointers match.
#[test]
fn df_series_shares_device_pointer() {
    let df = sample_df();
    let s = df.series("id").unwrap();

    let sv = df.column("id").unwrap();
    assert_eq!(
        s.column_view().unwrap().device_data_ptr().unwrap(),
        sv.device_data_ptr().unwrap(),
    );
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
    assert_eq!(s.column_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// Two `series()` calls for the same column produce Series with the
/// same device pointer — proving Arc-sharing (zero GPU copy).
#[test]
fn df_series_two_calls_share_arc() {
    let df = sample_df();
    let s1 = df.series("id").unwrap();
    let s2 = df.series("id").unwrap();

    // Same device pointer proves same underlying Column
    let ptr1 = s1.column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = s2.column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
}

// ===========================================================================
// 3. DataFrame::to_table() — deep-copies GPU data
// ===========================================================================

/// `to_table()` produces a new table with independent GPU memory.
/// Device pointers in the resulting table columns must differ from
/// the source DataFrame's columns.
#[test]
fn df_to_table_deep_copies() {
    let df = sample_df();
    let table = df.to_table().unwrap();

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
    let _table = df.to_table().unwrap();

    // DataFrame is still fully readable
    assert_eq!(df.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.column("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 4. DataFrame::into_table() — conditional deep-copy
// ===========================================================================

/// When the DataFrame is the sole owner of all columns (Arc strong
/// count == 1), `into_table()` can unwrap without copying.
#[test]
fn df_into_table_unique_owner_no_extra_copy() {
    // Build a DataFrame without any other references to the columns.
    let df = DataFrame::new(
        vec!["a".into()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    let original_ptr = device_ptr(&df, "a");
    let (table, names) = df.into_table().unwrap();

    assert_eq!(names, vec!["a"]);
    let tv = table.view().unwrap();
    let table_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();

    // When the Arc was unique, into_table() can unwrap without copying,
    // so the device pointer should be the same.
    assert_eq!(
        original_ptr, table_ptr,
        "into_table() with unique Arc should transfer without copy"
    );
}

/// When the DataFrame shares columns (Arc strong count > 1),
/// `into_table()` must deep-copy the shared columns.
#[test]
fn df_into_table_shared_columns_deep_copies() {
    let df = sample_df();

    // Create an external reference to keep the Arc refcount > 1.
    let external_series = df.series("id").unwrap();
    let external_ptr = external_series.column_view().unwrap().device_data_ptr().unwrap();

    let original_ptr = device_ptr(&df, "id");
    assert_eq!(original_ptr, external_ptr, "series shares same Arc");

    let (table, names) = df.into_table().unwrap();
    assert_eq!(names, vec!["id", "score"]);

    let tv = table.view().unwrap();
    let table_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();

    // The shared column must have been deep-copied.
    assert_ne!(
        table_ptr, external_ptr,
        "into_table() with shared Arc must deep-copy; pointers should differ"
    );

    // External series is still valid (Arc keeps it alive).
    assert_eq!(
        external_series.column_view().unwrap().to_vec::<i32>().unwrap(),
        vec![1, 2, 3],
    );
}

/// `into_table()` consumes the DataFrame — data values are preserved.
#[test]
fn df_into_table_values_preserved() {
    let df = sample_df();
    let (table, names) = df.into_table().unwrap();

    assert_eq!(names, vec!["id", "score"]);
    let tv = table.view().unwrap();
    assert_eq!(tv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(tv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 5. *_owned methods — what "owned" means
// ===========================================================================

/// `filter_by_mask_owned` consumes self but produces the same result
/// as the borrowing `filter_by_mask` variant.
#[test]
fn filter_by_mask_owned_same_result_as_borrow() {
    let df1 = sample_df();
    let df2 = sample_df();

    let mask = Column::from_slice(&[true, false, true]).unwrap();
    let mask_view = mask.view().unwrap();

    let borrow_result = df1.filter_by_mask(&mask_view).unwrap();
    let owned_result = df2.filter_by_mask_owned(&mask_view).unwrap();

    assert_eq!(
        borrow_result.column("id").unwrap().to_vec::<i32>().unwrap(),
        owned_result.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        borrow_result.column("score").unwrap().to_vec::<f64>().unwrap(),
        owned_result.column("score").unwrap().to_vec::<f64>().unwrap(),
    );
}

/// `filter_by_mask_owned` produces a new GPU allocation — the result
/// has different device pointers from the source.
///
/// # Cost
/// GPU-allocating: a new table is produced containing the filtered rows.
#[test]
fn filter_by_mask_owned_allocates_new_gpu_data() {
    let df = sample_df();
    let original_ptr = device_ptr(&df, "id");

    let mask = Column::from_slice(&[true, false, true]).unwrap();
    let mask_view = mask.view().unwrap();
    let filtered = df.filter_by_mask_owned(&mask_view).unwrap();

    // Different number of rows → must be a new GPU allocation.
    let filtered_ptr = device_ptr(&filtered, "id");
    assert_ne!(original_ptr, filtered_ptr);
}

/// `sort_owned` consumes self and produces the same result as `sort`.
#[test]
fn sort_owned_same_result_as_borrow() {
    let df1 = sample_df();
    let df2 = sample_df();

    let borrow_result = df1.sort(&["id"], &[cudf::sorting::Order::Descending]).unwrap();
    let owned_result = df2.sort_owned(&["id"], &[cudf::sorting::Order::Descending]).unwrap();

    assert_eq!(
        borrow_result.column("id").unwrap().to_vec::<i32>().unwrap(),
        owned_result.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
}

/// `head_owned` / `tail_owned` produce the same results as head/tail.
#[test]
fn head_tail_owned_same_results() {
    let df1 = sample_df();
    let df2 = sample_df();

    let h1 = df1.head(2).unwrap();
    let h2 = df2.head_owned(2).unwrap();
    assert_eq!(
        h1.column("id").unwrap().to_vec::<i32>().unwrap(),
        h2.column("id").unwrap().to_vec::<i32>().unwrap(),
    );

    let df3 = sample_df();
    let df4 = sample_df();
    let t1 = df3.tail(2).unwrap();
    let t2 = df4.tail_owned(2).unwrap();
    assert_eq!(
        t1.column("id").unwrap().to_vec::<i32>().unwrap(),
        t2.column("id").unwrap().to_vec::<i32>().unwrap(),
    );
}

/// `select_owned` avoids the Arc refcount bump — it removes columns
/// from the IndexMap instead of cloning the Arc.
#[test]
fn select_owned_transfers_arc() {
    let df = sample_df();
    let original_ptr = device_ptr(&df, "id");

    // select_owned consumes df and moves the Arc
    let selected = df.select_owned(&["id"]).unwrap();
    let selected_ptr = device_ptr(&selected, "id");

    // Same device pointer — the column was moved, not copied.
    assert_eq!(original_ptr, selected_ptr);
}

/// `rename_owned` avoids the Arc refcount bump.
#[test]
fn rename_owned_transfers_arc() {
    let df = sample_df();
    let original_ptr = device_ptr(&df, "id");

    let renamed = df.rename_owned("id", "identifier").unwrap();
    let renamed_ptr = device_ptr(&renamed, "identifier");

    // Same device pointer — pure schema rewrite.
    assert_eq!(original_ptr, renamed_ptr);
}

// ===========================================================================
// 6. Series::view() — returns borrowed SeriesView
// ===========================================================================

/// `Series::view()` returns a `SeriesView` that borrows from the Series.
/// It carries the series name and derefs to `ColumnView`.
#[test]
fn series_view_borrows_from_series() {
    let s = Series::new::<i32>("val", &[10, 20, 30]).unwrap();
    let sv = s.view().unwrap();

    assert_eq!(sv.name(), Some("val"));
    assert_eq!(sv.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(sv.len().unwrap(), 3);
    assert_eq!(sv.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

/// `view()` points to the same device data as the underlying column.
#[test]
fn series_view_same_device_ptr_as_column() {
    let s = Series::new::<i32>("x", &[1, 2]).unwrap();
    let sv = s.view().unwrap();
    let cv = s.column_view().unwrap();

    assert_eq!(sv.device_data_ptr().unwrap(), cv.device_data_ptr().unwrap());
}

/// An unnamed Series produces a view with `name() == None`.
#[test]
fn series_view_unnamed() {
    let col = Column::from_slice(&[5i32, 6]).unwrap();
    let s = Series::from_column(col).unwrap();
    let sv = s.view().unwrap();

    assert_eq!(sv.name(), None);
}

// ===========================================================================
// 7. Series::column() vs Series::column_view() — borrow vs share
// ===========================================================================

/// `Series::column()` returns `&Column` — a direct borrow of the inner
/// Arc's contents. This is a zero-copy borrow.
#[test]
fn series_column_returns_ref() {
    let s = Series::new::<i32>("c", &[1, 2, 3]).unwrap();

    // column() returns &Column
    let col_ref: &Column = s.column();
    let cv = col_ref.view().unwrap();
    assert_eq!(cv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `Series::column_arc()` returns a reference to the Arc itself,
/// allowing callers to clone it for shared ownership.
#[test]
fn series_column_arc_shares_arc() {
    let s = Series::new::<i32>("c", &[1, 2, 3]).unwrap();

    let arc_ref: &Arc<Column> = s.column_arc();
    let cloned = Arc::clone(arc_ref);

    // Both point to the same device data
    let ptr1 = s.column().view().unwrap().device_data_ptr().unwrap();
    let ptr2 = cloned.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
}

/// `Series::column_view()` returns a bare `ColumnView` (no name),
/// while `Series::view()` returns a `SeriesView` (with name).
/// Both point to the same GPU data.
#[test]
fn series_column_view_vs_view() {
    let s = Series::new::<i32>("named", &[7, 8, 9]).unwrap();

    let cv = s.column_view().unwrap();
    let sv = s.view().unwrap();

    // Same device data
    assert_eq!(cv.device_data_ptr().unwrap(), sv.device_data_ptr().unwrap());

    // But different types: ColumnView has no name, SeriesView has name
    assert_eq!(sv.name(), Some("named"));
}

// ===========================================================================
// Supplementary: clone semantics
// ===========================================================================

/// `Series::clone()` is a cheap Arc bump — same device pointer.
#[test]
fn series_clone_shares_gpu_data() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let cloned = s.clone();

    let ptr1 = s.column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = cloned.column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
    assert_eq!(cloned.name(), Some("x"));
}

/// `Series::rename()` produces a new Series with a different name but
/// the same GPU data.
#[test]
fn series_rename_shares_gpu_data() {
    let s = Series::new::<i32>("old", &[1, 2, 3]).unwrap();
    let renamed = s.rename("new");

    let ptr1 = s.column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = renamed.column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
    assert_eq!(renamed.name(), Some("new"));
    assert_eq!(s.name(), Some("old"));
}

// ===========================================================================
// Supplementary: DataFrame::view() — DataFrameView borrowing
// ===========================================================================

/// `DataFrame::view()` returns a `DataFrameView` that borrows all columns.
#[test]
fn df_view_borrows_all_columns() {
    let df = sample_df();
    let dfv = df.view();

    assert_eq!(dfv.num_columns(), 2);
    assert_eq!(dfv.column_names(), vec!["id", "score"]);

    let id_sv = dfv.column("id").unwrap();
    assert_eq!(id_sv.name(), Some("id"));
    assert_eq!(id_sv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `DataFrameView::column()` returns a `SeriesView` that points to the
/// same GPU data as `DataFrame::column()`.
#[test]
fn df_view_column_same_device_ptr() {
    let df = sample_df();
    let dfv = df.view();

    let dfv_sv = dfv.column("id").unwrap();
    let df_sv = df.column("id").unwrap();

    assert_eq!(
        dfv_sv.device_data_ptr().unwrap(),
        df_sv.device_data_ptr().unwrap(),
    );
}

// ===========================================================================
// Supplementary: facade methods' GPU-allocating nature
// ===========================================================================

/// `filter()` with a Series mask produces a new GPU allocation.
#[test]
fn df_filter_gpu_allocates() {
    let df = sample_df();
    let mask = Series::new::<bool>("mask", &[true, false, true]).unwrap();

    let filtered = df.filter(&mask).unwrap();
    assert_eq!(filtered.num_rows(), 2);
    assert_eq!(filtered.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 3]);

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
    let original_ptr = s.column_view().unwrap().device_data_ptr().unwrap();

    let mask = s.gt_scalar(1i32).unwrap();
    let mask_ptr = mask.column_view().unwrap().device_data_ptr().unwrap();
    assert_ne!(original_ptr, mask_ptr, "gt_scalar must allocate new column");

    let added = s.add_scalar(10i32).unwrap();
    let add_ptr = added.column_view().unwrap().device_data_ptr().unwrap();
    assert_ne!(original_ptr, add_ptr, "add_scalar must allocate new column");
}
