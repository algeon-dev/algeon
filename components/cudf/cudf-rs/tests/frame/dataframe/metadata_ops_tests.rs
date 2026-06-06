//! Integration tests for zero-copy metadata ops (cuDF-style signatures).
//!
//! Tests: without_columns, rename_columns, with_column, assign, and Arc sharing verification.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{DataFrame, ErrorKind, Series};

/// Helper: build a 3-row, 3-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();
    let flags = Series::new::<i32>("flags", &[100, 200, 300]).unwrap();

    DataFrame::from_columns([("id", ids), ("score", scores), ("flag", flags)]).unwrap()
}

/// Returns the raw device data pointer for a named column in a DataFrame.
fn device_ptr(df: &DataFrame, name: &str) -> *const std::ffi::c_void {
    df.series_view(name).unwrap().device_data_ptr().unwrap()
}

// ===========================================================================
// without_columns: removes correct columns, preserves others
// ===========================================================================

#[test]
fn test_without_columns_removes_single_column() {
    let df = sample_df();
    let result = df.without_columns(["score"]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), &["id", "flag"]);
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_without_columns_removes_multiple_columns() {
    let df = sample_df();
    let result = df.without_columns(["id", "flag"]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.column_names(), &["score"]);
}

#[test]
fn test_without_columns_preserves_arc_sharing_for_retained_columns() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let flag_ptr = device_ptr(&df, "flag");

    let result = df.without_columns(["score"]).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
    assert_eq!(flag_ptr, device_ptr(&result, "flag"));
}

#[test]
fn test_without_columns_nonexistent_column_returns_error() {
    let df = sample_df();
    let result = df.without_columns(["nonexistent"]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.to_string().contains("nonexistent"));
}

#[test]
fn test_without_columns_all_columns_preserves_row_count() {
    let df = sample_df();
    let result = df.without_columns(["id", "score", "flag"]).unwrap();

    assert_eq!(result.num_columns(), 0);
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_without_columns_preserves_column_order() {
    let df = sample_df();
    let result = df.without_columns(["score"]).unwrap();

    // "id" should still come before "flag"
    assert_eq!(result.column_names(), &["id", "flag"]);
}

#[test]
fn test_without_columns_original_dataframe_still_reads_excluded_column() {
    let df = sample_df();
    let result = df.without_columns(["score"]).unwrap();

    assert_eq!(result.column_names(), &["id", "flag"]);
    assert_eq!(df.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// rename_columns: multi-column rename
// ===========================================================================

#[test]
fn test_rename_columns_single_pair() {
    let df = sample_df();
    let result = df.rename_columns([("id", "identifier")]).unwrap();

    assert_eq!(result.column_names(), &["identifier", "score", "flag"]);
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_rename_columns_multiple_pairs() {
    let df = sample_df();
    let result = df.rename_columns([("id", "a"), ("score", "b"), ("flag", "c")]).unwrap();

    assert_eq!(result.column_names(), &["a", "b", "c"]);
}

#[test]
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

#[test]
fn test_rename_columns_nonexistent_source_returns_error() {
    let df = sample_df();
    let result = df.rename_columns([("nonexistent", "x")]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_rename_columns_duplicate_target_returns_error() {
    let df = sample_df();
    // Both "id" and "score" renamed to the same target
    let result = df.rename_columns([("id", "same"), ("score", "same")]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_rename_columns_target_conflicts_with_existing_unrenamed_column() {
    let df = sample_df();
    // Rename "id" to "flag" — but "flag" already exists and is not being renamed
    let result = df.rename_columns([("id", "flag")]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_rename_columns_swap_is_allowed() {
    let df = sample_df();
    // Swap "id" and "flag" names — both are being renamed, so no conflict
    let result = df.rename_columns([("id", "flag"), ("flag", "id")]).unwrap();

    assert_eq!(result.column_names(), &["flag", "score", "id"]);

    // Data should follow the original column, not the new name
    let original_id_ptr = device_ptr(&df, "id");
    let original_flag_ptr = device_ptr(&df, "flag");
    assert_eq!(original_id_ptr, device_ptr(&result, "flag"));
    assert_eq!(original_flag_ptr, device_ptr(&result, "id"));
}

#[test]
fn test_rename_columns_empty_iterator() {
    let df = sample_df();
    let result = df.rename_columns(std::iter::empty::<(&str, &str)>()).unwrap();

    assert_eq!(result.column_names(), df.column_names());
}

#[test]
fn test_select_accepts_iterators() {
    let df = sample_df();
    let result = df.select(vec!["flag", "id"]).unwrap();

    assert_eq!(result.column_names(), &["flag", "id"]);
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_select_empty_preserves_row_count() {
    let df = sample_df();
    let result = df.select(std::iter::empty::<&str>()).unwrap();

    assert_eq!(result.num_columns(), 0);
    assert_eq!(result.num_rows(), 3);
}

// ===========================================================================
// with_column: accepts both Column and Series via Into<Series>
// ===========================================================================

#[test]
fn test_with_column_adds_new_column() {
    let df = sample_df();
    let extra = Series::new::<i64>("extra", &[100, 200, 300]).unwrap();

    let result = df.with_column("extra", extra).unwrap();

    assert_eq!(result.num_columns(), 4);
    assert!(result.column_names().contains(&"extra"));
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_with_column_replaces_existing_column() {
    let df = sample_df();
    let new_score = Series::new::<f64>("new_score", &[99.0, 88.0, 77.0]).unwrap();

    let result = df.with_column("score", new_score).unwrap();

    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![99.0, 88.0, 77.0]);
}

#[test]
fn test_with_column_preserves_arc_sharing_for_unchanged() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let flag_ptr = device_ptr(&df, "flag");

    let extra = Series::new::<i64>("extra", &[100, 200, 300]).unwrap();
    let result = df.with_column("extra", extra).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
    assert_eq!(flag_ptr, device_ptr(&result, "flag"));
}

#[test]
fn test_with_column_shares_arc_from_input_series() {
    let df = sample_df();
    let extra = Series::new::<i64>("extra", &[100, 200, 300]).unwrap();
    let extra_arc = extra.clone().into_inner();

    let result = df.with_column("extra", extra).unwrap();

    // The inserted column should share the same Arc as the input Series
    let result_series = result.series("extra").unwrap();
    assert!(Arc::ptr_eq(&extra_arc, &result_series.into_inner()));
}

#[test]
fn test_with_column_row_count_mismatch_returns_error() {
    let df = sample_df(); // 3 rows
    let wrong_len = Series::new::<i32>("x", &[1, 2]).unwrap(); // 2 rows

    let result = df.with_column("x", wrong_len);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ===========================================================================
// assign: adds/replaces multiple columns
// ===========================================================================

#[test]
fn test_assign_adds_multiple_new_columns() {
    let df = sample_df();
    let a = Series::new::<i32>("a", &[10, 20, 30]).unwrap();
    let b = Series::new::<f64>("b", &[1.0, 2.0, 3.0]).unwrap();

    let result = df.assign([("a", a), ("b", b)]).unwrap();

    assert_eq!(result.num_columns(), 5);
    assert!(result.column_names().contains(&"a"));
    assert!(result.column_names().contains(&"b"));
}

#[test]
fn test_assign_replaces_existing_column() {
    let df = sample_df();
    let new_score = Series::new::<f64>("s", &[99.0, 88.0, 77.0]).unwrap();

    let result = df.assign([("score", new_score)]).unwrap();

    assert_eq!(result.num_columns(), 3); // no new columns, just replaced
    assert_eq!(result.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![99.0, 88.0, 77.0]);
}

#[test]
fn test_assign_mix_of_add_and_replace() {
    let df = sample_df();
    let new_id = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let extra = Series::new::<i64>("x", &[100, 200, 300]).unwrap();

    let result = df.assign([("id", new_id), ("extra", extra)]).unwrap();

    assert_eq!(result.num_columns(), 4);
    assert_eq!(result.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_assign_preserves_arc_sharing_for_unchanged() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");
    let flag_ptr = device_ptr(&df, "flag");

    let new_score = Series::new::<f64>("s", &[99.0, 88.0, 77.0]).unwrap();
    let result = df.assign([("score", new_score)]).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
    assert_eq!(flag_ptr, device_ptr(&result, "flag"));
}

#[test]
fn test_assign_shares_arc_from_input_series() {
    let df = sample_df();
    let extra = Series::new::<i64>("extra", &[100, 200, 300]).unwrap();
    let extra_arc = extra.clone().into_inner();

    let result = df.assign([("extra", extra)]).unwrap();

    let result_series = result.series("extra").unwrap();
    assert!(Arc::ptr_eq(&extra_arc, &result_series.into_inner()));
}

#[test]
fn test_assign_row_count_mismatch_returns_error() {
    let df = sample_df(); // 3 rows
    let wrong = Series::new::<i32>("x", &[1]).unwrap(); // 1 row

    let result = df.assign([("x", wrong)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_assign_empty_iterator_returns_same_schema() {
    let df = sample_df();
    let result = df.assign(std::iter::empty::<(&str, Series)>()).unwrap();

    assert_eq!(result.column_names(), df.column_names());
    assert_eq!(result.num_rows(), df.num_rows());
}

#[test]
fn test_assign_duplicate_names_last_wins() {
    let df = sample_df();
    let first = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let second = Series::new::<i32>("x", &[10, 20, 30]).unwrap();

    let result = df.assign([("extra", first), ("extra", second)]).unwrap();

    // The second value should win
    assert_eq!(result.series_view("extra").unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_assign_on_zero_column_frame_preserves_logical_row_count() {
    let df = sample_df().without_columns(["id", "score", "flag"]).unwrap();
    let replacement = Series::new::<i32>("replacement", &[7, 8, 9]).unwrap();

    let result = df.assign([("replacement", replacement)]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series_view("replacement").unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn test_assign_on_zero_column_frame_rejects_row_count_mismatch() {
    let df = sample_df().without_columns(["id", "score", "flag"]).unwrap();
    let wrong = Series::new::<i32>("replacement", &[7, 8]).unwrap();

    let result = df.assign([("replacement", wrong)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_assign_on_truly_empty_dataframe_seeds_row_count_from_first_column() {
    let df = DataFrame::new(vec![], vec![]).unwrap();
    let values = Series::new::<i32>("values", &[7, 8]).unwrap();

    let result = df.assign([("values", values)]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.series_view("values").unwrap().to_vec::<i32>().unwrap(), vec![7, 8]);
}

// ===========================================================================
// Combined operations: verify Arc sharing chains
// ===========================================================================

#[test]
fn test_without_columns_then_assign_preserves_sharing() {
    let df = sample_df();
    let id_ptr = device_ptr(&df, "id");

    let dropped = df.without_columns(["score", "flag"]).unwrap();
    let extra = Series::new::<f64>("extra", &[1.0, 2.0, 3.0]).unwrap();
    let result = dropped.assign([("extra", extra)]).unwrap();

    assert_eq!(id_ptr, device_ptr(&result, "id"));
}

#[test]
fn test_rename_columns_then_without_columns_preserves_sharing() {
    let df = sample_df();
    let score_ptr = device_ptr(&df, "score");

    let renamed = df.rename_columns([("id", "identifier")]).unwrap();
    let result = renamed.without_columns(["flag"]).unwrap();

    assert_eq!(score_ptr, device_ptr(&result, "score"));
}

#[test]
fn test_with_series_then_rename_columns_preserves_sharing() {
    let df = sample_df();
    let extra = Series::new::<i64>("extra", &[100, 200, 300]).unwrap();
    let extra_arc = extra.clone().into_inner();

    let with_extra = df.with_column("extra", extra).unwrap();
    let result = with_extra.rename_columns([("extra", "bonus")]).unwrap();

    let bonus_series = result.series("bonus").unwrap();
    assert!(Arc::ptr_eq(&extra_arc, &bonus_series.into_inner()));
}

#[test]
fn test_with_column_on_zero_column_frame_preserves_logical_row_count() {
    let df = sample_df().select(std::iter::empty::<&str>()).unwrap();
    let values = Series::new::<i32>("values", &[11, 22, 33]).unwrap();

    let result = df.with_column("values", values).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series_view("values").unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_with_column_on_zero_column_frame_rejects_row_count_mismatch() {
    let df = sample_df().select(std::iter::empty::<&str>()).unwrap();
    let values = Series::new::<i32>("values", &[11, 22]).unwrap();

    let result = df.with_column("values", values);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_with_column_on_zero_column_frame_from_drop_rejects_row_count_mismatch() {
    let df = sample_df().without_columns(["id", "score", "flag"]).unwrap();
    let values = Series::new::<i32>("values", &[11, 22]).unwrap();

    let result = df.with_column("values", values);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_with_column_on_truly_empty_dataframe_seeds_row_count_from_first_column() {
    let df = DataFrame::new(vec![], vec![]).unwrap();
    let values = Series::new::<i32>("values", &[11, 22]).unwrap();

    let result = df.with_column("values", values).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.series_view("values").unwrap().to_vec::<i32>().unwrap(), vec![11, 22]);
}
