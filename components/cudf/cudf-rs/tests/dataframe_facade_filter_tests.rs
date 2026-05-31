//! Tests for the façade `DataFrame::filter` that accepts `&Series` masks.

use cudf::{column::Column, dataframe::DataFrame, error::ErrorKind, series::Series};

/// Helper: build a 5-row DataFrame with id (i32) and score (f64) columns.
fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("id", &[3, 1, 4, 1, 5]).unwrap();
    let scores = Series::new::<f64>("score", &[30.0, 10.0, 40.0, 15.0, 50.0]).unwrap();
    DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap()
}

// ---------------------------------------------------------------------------
// filter(&Series) — happy path
// ---------------------------------------------------------------------------

#[test]
fn test_filter_series_mask_basic() {
    let df = sample_df();
    // Build a boolean mask via Series comparison: id > 2
    let id_series = df.series("id").unwrap();
    let mask = id_series.gt_scalar(2i32).unwrap();

    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 3);
    assert_eq!(filtered.column("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4, 5]);
    assert_eq!(filtered.column("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 40.0, 50.0]);
}

#[test]
fn test_filter_series_mask_all_true() {
    let df = sample_df();
    let mask = Series::new::<bool>("mask", &[true, true, true, true, true]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 5);
}

#[test]
fn test_filter_series_mask_all_false() {
    let df = sample_df();
    let mask = Series::new::<bool>("mask", &[false, false, false, false, false]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 0);
    // Schema preserved even when empty.
    assert_eq!(filtered.column_names(), vec!["id", "score"]);
}

// ---------------------------------------------------------------------------
// filter(&Series) — preserves output column names
// ---------------------------------------------------------------------------

#[test]
fn test_filter_series_preserves_column_names() {
    let df = sample_df();
    let mask = Series::new::<bool>("m", &[true, false, true, false, true]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.column_names(), vec!["id", "score"]);
    assert_eq!(filtered.column_index("id"), Some(0));
    assert_eq!(filtered.column_index("score"), Some(1));
}

// ---------------------------------------------------------------------------
// filter(&Series) — rejects wrong-length mask
// ---------------------------------------------------------------------------

#[test]
fn test_filter_series_rejects_wrong_length_mask() {
    let df = sample_df(); // 5 rows
    let mask = Series::new::<bool>("short", &[true, false]).unwrap();

    let result = df.filter(&mask);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.to_string().contains("length"));
}

// ---------------------------------------------------------------------------
// filter(&Series) — rejects non-BOOL8 mask
// ---------------------------------------------------------------------------

#[test]
fn test_filter_series_rejects_non_bool_mask() {
    let df = sample_df();
    // Pass an i32 series as mask — should be rejected.
    let mask = Series::new::<i32>("bad_mask", &[1, 0, 1, 0, 1]).unwrap();

    let result = df.filter(&mask);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
    assert!(err.to_string().contains("Bool8"));
}

// ---------------------------------------------------------------------------
// filter(&Series) — works with comparison-produced masks
// ---------------------------------------------------------------------------

#[test]
fn test_filter_series_with_eq_scalar_mask() {
    let df = sample_df();
    let id_series = df.series("id").unwrap();
    let mask = id_series.eq_scalar(1i32).unwrap();

    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 2);
    assert_eq!(filtered.column("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1]);
}

#[test]
fn test_filter_series_with_le_scalar_mask() {
    let df = sample_df();
    let score_series = df.series("score").unwrap();
    // score <= 15.0 → rows 1 (10.0) and 3 (15.0)
    let mask = score_series.le_scalar(15.0f64).unwrap();

    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 2);
    assert_eq!(filtered.column("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 15.0]);
}

// ---------------------------------------------------------------------------
// Low-level filter_by_mask(&ColumnView) still works (backward compat)
// ---------------------------------------------------------------------------

#[test]
fn test_low_level_filter_by_mask_still_works() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[true, false, true, false, true]).unwrap();
    let mask_view = mask_col.view().unwrap();

    let filtered = df.filter_by_mask(&mask_view).unwrap();

    assert_eq!(filtered.num_rows().unwrap(), 3);
    assert_eq!(filtered.column("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4, 5]);
}

#[test]
fn test_low_level_filter_by_mask_rejects_non_bool_mask() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[1i32, 0, 1, 0, 1]).unwrap();
    let mask_view = mask_col.view().unwrap();

    let err = df.filter_by_mask(&mask_view).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
    assert!(err.to_string().contains("Bool8"));
}

#[test]
fn test_low_level_filter_by_mask_rejects_wrong_length_mask() {
    let df = sample_df();
    let mask_col = Column::from_slice(&[true, false]).unwrap();
    let mask_view = mask_col.view().unwrap();

    let err = df.filter_by_mask(&mask_view).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.to_string().contains("length"));
}
