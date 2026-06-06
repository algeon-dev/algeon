//! Tests for the high-level `DataFrame::filter` that accepts `&Series` masks.

use cudf::{Column, ErrorKind, Series};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
    sample_df, sample_runtime,
};

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

    assert_eq!(filtered.num_rows(), 3);
    assert_eq!(filtered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4, 5]);
    assert_eq!(filtered.series("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 40.0, 50.0]);
}

#[test]
fn test_filter_series_mask_succeeds_with_mixed_domains_and_uses_dataframe_provenance() {
    let df_rt = sample_runtime();
    let mask_rt = sample_runtime();
    let df = cudf::DataFrame::from_series([
        ("id", Series::new_with_runtime(&df_rt, "id", &[3i32, 1, 4, 1, 5]).unwrap()),
        ("score", Series::new_with_runtime(&df_rt, "score", &[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap()),
    ])
    .unwrap();
    let mask = Series::new_with_runtime(&mask_rt, "mask", &[true, false, true, false, true]).unwrap();
    let expected_exec_stream = df.__debug_execution_domain_stream_value();
    let expected_exec_mr = df.__debug_execution_domain_mr_value();
    let expected_sync_stream = mask.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(filtered.__debug_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    assert_eq!(filtered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4, 5]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_filter_series_mask_all_true() {
    let df = sample_df();
    let mask = Series::new::<bool>("mask", &[true, true, true, true, true]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows(), 5);
}

#[test]
fn test_filter_series_mask_all_false() {
    let df = sample_df();
    let mask = Series::new::<bool>("mask", &[false, false, false, false, false]).unwrap();
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows(), 0);
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

    assert_eq!(filtered.num_rows(), 2);
    assert_eq!(filtered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1]);
}

#[test]
fn test_filter_series_with_le_scalar_mask() {
    let df = sample_df();
    let score_series = df.series("score").unwrap();
    // score <= 15.0 → rows 1 (10.0) and 3 (15.0)
    let mask = score_series.le_scalar(15.0f64).unwrap();

    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.num_rows(), 2);
    assert_eq!(filtered.series("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 15.0]);
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

    assert_eq!(filtered.num_rows(), 3);
    assert_eq!(filtered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 4, 5]);
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
