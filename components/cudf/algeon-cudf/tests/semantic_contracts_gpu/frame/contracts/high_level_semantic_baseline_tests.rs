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

use cudf::{DataFrame, ErrorCode, Exec, TypeId, core::Column};

use crate::common::{
    column_from_slice, dataframe_device_ptr, dataframe_from_columns, error_context_value,
    fail_next_column_view_metadata_read, series,
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 3-row, 2-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = series::<i32>("id", &[1, 2, 3]).unwrap();
    let scores = series::<f64>("score", &[10.0, 20.0, 30.0]).unwrap();
    DataFrame::from_series([("id", ids), ("score", scores)]).unwrap()
}

// ===========================================================================
// 1. DataFrame::series(): returns owned Series with Arc-shared Column
// ===========================================================================

/// `series()` returns an owned `Series` with correct metadata.
fn df_series_returns_owned_series() {
    let df = sample_df();
    let s = df.series("id").unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

/// Two `series()` calls return Series backed by the same GPU buffer.
fn df_series_shares_device_pointer() {
    let df = sample_df();
    let ptr1 = df.series("id").unwrap().as_column().device_data_ptr().unwrap();
    let ptr2 = df.series("id").unwrap().as_column().device_data_ptr().unwrap();

    assert_eq!(ptr1, ptr2);
}

/// `series()` produces an owned value that can outlive the DataFrame.
/// This proves it is worker-thread portable.
fn df_series_outlives_dataframe() {
    let s = {
        let df = sample_df();
        df.series("id").unwrap()
        // df dropped here
    };
    // Series is still usable
    assert_eq!(s.len(), 3);
    assert_eq!(s.as_column().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// Two `series()` calls for the same column produce Series with the
/// same device pointer; proving Arc-sharing (zero GPU copy).
fn df_series_two_calls_share_arc() {
    let df = sample_df();
    let s1 = df.series("id").unwrap();
    let s2 = df.series("id").unwrap();

    // Same device pointer proves same underlying Column
    let ptr1 = s1.as_column().device_data_ptr().unwrap();
    let ptr2 = s2.as_column().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
}

// ===========================================================================
// 2. DataFrame::get() / contains(): probe and existence check
// ===========================================================================

/// `get()` returns `None` on miss, `Some(Series)` on hit.
fn df_get_probe_access() {
    let df = sample_df();
    assert!(df.get("nonexistent").unwrap().is_none());

    let s = df.get("id").unwrap().unwrap();
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn df_series_returns_error_when_shared_series_reconstruction_metadata_read_fails() {
    let df = sample_df();

    fail_next_column_view_metadata_read();
    let err = df.series("id").unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dataframe::series");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("id"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_view_metadata"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_foreign_column_view_metadata"));
    assert!(err.message().contains("test metadata read failure"), "unexpected error: {err}");
}

#[test]
fn df_get_preserves_missing_column_vs_reconstruction_failure() {
    let df = sample_df();
    assert!(df.get("missing").unwrap().is_none());

    fail_next_column_view_metadata_read();
    let err = df.get("id").unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dataframe::get");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("id"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_view_metadata"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_foreign_column_view_metadata"));
}

/// `contains()` returns `true` for existing columns, `false` otherwise.
fn df_contains_check() {
    let df = sample_df();
    assert!(df.contains("id"));
    assert!(df.contains("score"));
    assert!(!df.contains("nonexistent"));
}

/// `iter_series()` yields series in schema order without copying.
fn df_iter_series_preserves_schema_order_and_shares_gpu_data() {
    let df = sample_df();
    let items: Vec<_> = df.iter_series().collect::<cudf::Result<Vec<_>>>().unwrap();

    assert_eq!(df.column_names(), vec!["id", "score"]);
    assert_eq!(items[0].to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(items[1].to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);

    let iter_ptr = items[0].as_column().device_data_ptr().unwrap();
    let direct_ptr = df.series("id").unwrap().as_column().device_data_ptr().unwrap();
    assert_eq!(iter_ptr, direct_ptr);
}

#[test]
fn df_iter_series_yields_per_item_errors_without_reordering() {
    let df = sample_df();
    let mut iter = df.iter_series();

    fail_next_column_view_metadata_read();
    let err = iter.next().unwrap().unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dataframe::iter_series");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("id"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_view_metadata"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_foreign_column_view_metadata"));

    let second = iter.next().unwrap().unwrap();
    assert_eq!(second.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 3. DataFrame::to_table(): deep-copies GPU data
// ===========================================================================

/// `to_table()` produces a new table with independent GPU memory.
fn df_to_table_deep_copies() {
    let df = sample_df();
    let table = df.on(Exec::RETAINED).to_table_copy().unwrap();

    assert_eq!(table.num_columns().unwrap(), 2);
    assert_eq!(table.num_rows().unwrap(), 3);
    let table_columns = table.into_columns().unwrap();

    // The table's column 0 ("id") must have a DIFFERENT device pointer
    let table_col0 = &table_columns[0];
    let table_ptr = table_col0.device_data_ptr().unwrap();
    let df_ptr = dataframe_device_ptr(&df, "id");
    assert_ne!(table_ptr, df_ptr, "to_table() must deep-copy; pointers should differ");

    // Data values should still match
    assert_eq!(table_col0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `to_table()` does not consume the DataFrame; source remains usable.
fn df_to_table_source_still_usable() {
    let df = sample_df();
    let _table = df.on(Exec::RETAINED).to_table_copy().unwrap();

    // DataFrame is still fully readable
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 4. DataFrame::into_table(): conditional deep-copy
// ===========================================================================

/// When the DataFrame is the sole owner of all columns (Arc strong
/// count == 1), `into_table()` can unwrap without copying.
fn df_into_table_unique_owner_no_extra_copy() {
    let df = dataframe_from_columns(vec!["a".into()], vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let original_ptr = dataframe_device_ptr(&df, "a");
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["a"]);
    let table_columns = table.into_columns().unwrap();
    let table_ptr = table_columns[0].device_data_ptr().unwrap();

    assert_eq!(original_ptr, table_ptr, "into_table() with unique Arc should transfer without copy");
}

/// When the DataFrame shares columns (Arc strong count > 1),
/// `into_table()` must deep-copy the shared columns.
fn df_into_table_shared_columns_deep_copies() {
    let df = sample_df();

    // Create an external reference to keep the Arc refcount > 1.
    let external_series = df.series("id").unwrap();
    let external_ptr = external_series.as_column().device_data_ptr().unwrap();

    let original_ptr = dataframe_device_ptr(&df, "id");
    assert_eq!(original_ptr, external_ptr, "series shares same Arc");

    let (table, names) = df.try_into_table().unwrap();
    assert_eq!(names, vec!["id", "score"]);

    let table_columns = table.into_columns().unwrap();
    let table_ptr = table_columns[0].device_data_ptr().unwrap();

    assert_ne!(table_ptr, external_ptr, "into_table() with shared Arc must deep-copy; pointers should differ");

    // External series is still valid (Arc keeps it alive).
    assert_eq!(external_series.as_column().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `into_table()` consumes the DataFrame; data values are preserved.
fn df_into_table_values_preserved() {
    let df = sample_df();
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["id", "score"]);
    let columns = table.into_columns().unwrap();
    assert_eq!(columns[0].to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(columns[1].to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 5. Schema-only DataFrame metadata ops remain zero-copy
// ===========================================================================

/// `filter(&Series)` produces a new GPU allocation.
fn filter_allocates_new_gpu_data() {
    let df = sample_df();
    let original_ptr = dataframe_device_ptr(&df, "id");

    let mask = series::<bool>("mask", &[true, false, true]).unwrap();
    let filtered = df.on(Exec::RETAINED).filter(&mask).unwrap();

    let filtered_ptr = dataframe_device_ptr(&filtered, "id");
    assert_ne!(original_ptr, filtered_ptr);
}

/// `sort_values` produces consistent results across calls.
fn sort_values_consistent_results() {
    let df1 = sample_df();
    let df2 = sample_df();

    let result1 =
        df1.on(Exec::RETAINED).sort_values(["id"], cudf::SortOptions::new().with_descending(vec![true])).unwrap();
    let result2 =
        df2.on(Exec::RETAINED).sort_values(["id"], cudf::SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(
        result1.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
        result2.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
    );
}

/// `select` is schema-only and Arc-shares selected columns.
fn select_shares_arc() {
    let df = sample_df();
    let original_ptr = dataframe_device_ptr(&df, "id");

    let selected = df.select(["id"]).unwrap();
    let selected_ptr = dataframe_device_ptr(&selected, "id");

    assert_eq!(original_ptr, selected_ptr);
}

/// `rename_column` is a pure schema rewrite that Arc-shares columns.
fn rename_column_shares_arc() {
    let df = sample_df();
    let original_ptr = dataframe_device_ptr(&df, "id");

    let renamed = df.rename_column("id", "identifier").unwrap();
    let renamed_ptr = dataframe_device_ptr(&renamed, "identifier");

    assert_eq!(original_ptr, renamed_ptr);
}

// ===========================================================================
// 6. Series accessor semantics
// ===========================================================================

/// `Series::as_column()` returns a `Column` reference that points to the
/// same device data across repeated access.
fn series_as_column_same_device_ptr_on_repeated_access() {
    let s = series::<i32>("x", &[1, 2]).unwrap();
    let col = s.as_column();

    assert_eq!(col.device_data_ptr().unwrap(), s.as_column().device_data_ptr().unwrap());
}

/// `Series::as_column()` returns `&Column`; a direct borrow.
fn series_column_returns_ref() {
    let s = series::<i32>("c", &[1, 2, 3]).unwrap();
    let col_ref: &Column = s.as_column();
    assert_eq!(col_ref.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `Series::into_inner()` consumes the Series and returns the `Arc<Column>`.
/// A clone of the Series shares the same device data.
fn series_into_inner_shares_arc() {
    let s = series::<i32>("c", &[1, 2, 3]).unwrap();

    let arc1 = s.clone().into_inner();
    let arc2 = s.into_inner();

    let ptr1 = arc1.device_data_ptr().unwrap();
    let ptr2 = arc2.device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
    assert!(Arc::ptr_eq(&arc1, &arc2));
}

/// `Series::arc()` returns a cloned `Arc<Column>` without consuming the Series.
fn series_arc_accessor_shares_arc_without_consuming() {
    let s = series::<i32>("c", &[1, 2, 3]).unwrap();

    let cloned_arc = s.arc();
    let consumed_arc = s.clone().into_inner();

    let ptr1 = cloned_arc.device_data_ptr().unwrap();
    let ptr2 = consumed_arc.device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
    assert!(Arc::ptr_eq(&cloned_arc, &consumed_arc));

    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// Supplementary: clone semantics
// ===========================================================================

/// `Series::clone()` is a cheap Arc bump; same device pointer.
fn series_clone_shares_gpu_data() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let cloned = s.clone();

    let ptr1 = s.as_column().device_data_ptr().unwrap();
    let ptr2 = cloned.as_column().device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2);
}

// ===========================================================================
// Supplementary: high-level methods' GPU-allocating nature
// ===========================================================================

/// `filter()` with a Series mask produces a new GPU allocation.
fn df_filter_gpu_allocates() {
    let df = sample_df();
    let mask = series::<bool>("mask", &[true, false, true]).unwrap();

    let filtered = df.on(Exec::RETAINED).filter(&mask).unwrap();
    assert_eq!(filtered.num_rows(), 2);
    assert_eq!(filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3]);

    // New allocation; different device pointer
    assert_ne!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&filtered, "id"));
}

/// `head()` / `tail()` / `slice()` all produce new GPU allocations.
fn df_slicing_ops_gpu_allocate() {
    let df = sample_df();
    let original_ptr = dataframe_device_ptr(&df, "id");

    let h = df.on(Exec::RETAINED).head(2).unwrap();
    assert_ne!(original_ptr, dataframe_device_ptr(&h, "id"));

    let t = df.on(Exec::RETAINED).tail(2).unwrap();
    assert_ne!(original_ptr, dataframe_device_ptr(&t, "id"));

    let s = df.on(Exec::RETAINED).slice(1, 1).unwrap();
    assert_ne!(original_ptr, dataframe_device_ptr(&s, "id"));
}

/// GPU-allocating unary ops on Series produce new columns.
fn series_ops_gpu_allocate() {
    let s = series::<i32>("v", &[1, 2, 3]).unwrap();
    let original_ptr = s.as_column().device_data_ptr().unwrap();

    let mask = s.on(Exec::RETAINED).gt_scalar(1i32).unwrap();
    let mask_ptr = mask.as_column().device_data_ptr().unwrap();
    assert_ne!(original_ptr, mask_ptr, "gt_scalar must allocate new column");

    let added = s.on(Exec::RETAINED).add_scalar(10i32).unwrap();
    let add_ptr = added.as_column().device_data_ptr().unwrap();
    assert_ne!(original_ptr, add_ptr, "add_scalar must allocate new column");
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("high_level_semantic_baseline_tests::df_series_returns_owned_series", df_series_returns_owned_series),
    ("high_level_semantic_baseline_tests::df_series_shares_device_pointer", df_series_shares_device_pointer),
    ("high_level_semantic_baseline_tests::df_series_outlives_dataframe", df_series_outlives_dataframe),
    ("high_level_semantic_baseline_tests::df_series_two_calls_share_arc", df_series_two_calls_share_arc),
    ("high_level_semantic_baseline_tests::df_get_probe_access", df_get_probe_access),
    ("high_level_semantic_baseline_tests::df_contains_check", df_contains_check),
    (
        "high_level_semantic_baseline_tests::df_iter_series_preserves_schema_order_and_shares_gpu_data",
        df_iter_series_preserves_schema_order_and_shares_gpu_data,
    ),
    ("high_level_semantic_baseline_tests::df_to_table_deep_copies", df_to_table_deep_copies),
    ("high_level_semantic_baseline_tests::df_to_table_source_still_usable", df_to_table_source_still_usable),
    (
        "high_level_semantic_baseline_tests::df_into_table_unique_owner_no_extra_copy",
        df_into_table_unique_owner_no_extra_copy,
    ),
    (
        "high_level_semantic_baseline_tests::df_into_table_shared_columns_deep_copies",
        df_into_table_shared_columns_deep_copies,
    ),
    ("high_level_semantic_baseline_tests::df_into_table_values_preserved", df_into_table_values_preserved),
    ("high_level_semantic_baseline_tests::filter_allocates_new_gpu_data", filter_allocates_new_gpu_data),
    ("high_level_semantic_baseline_tests::sort_values_consistent_results", sort_values_consistent_results),
    ("high_level_semantic_baseline_tests::select_shares_arc", select_shares_arc),
    ("high_level_semantic_baseline_tests::rename_column_shares_arc", rename_column_shares_arc),
    (
        "high_level_semantic_baseline_tests::series_as_column_same_device_ptr_on_repeated_access",
        series_as_column_same_device_ptr_on_repeated_access,
    ),
    ("high_level_semantic_baseline_tests::series_column_returns_ref", series_column_returns_ref),
    ("high_level_semantic_baseline_tests::series_into_inner_shares_arc", series_into_inner_shares_arc),
    (
        "high_level_semantic_baseline_tests::series_arc_accessor_shares_arc_without_consuming",
        series_arc_accessor_shares_arc_without_consuming,
    ),
    ("high_level_semantic_baseline_tests::series_clone_shares_gpu_data", series_clone_shares_gpu_data),
    ("high_level_semantic_baseline_tests::df_filter_gpu_allocates", df_filter_gpu_allocates),
    ("high_level_semantic_baseline_tests::df_slicing_ops_gpu_allocate", df_slicing_ops_gpu_allocate),
    ("high_level_semantic_baseline_tests::series_ops_gpu_allocate", series_ops_gpu_allocate),
];
