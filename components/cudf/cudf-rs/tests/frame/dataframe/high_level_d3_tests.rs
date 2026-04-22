//! Integration tests for high-level DataFrame methods.
//!
//! Methods under test: `dropna`, `fillna`, `drop_duplicates`.
//! Option types under test: `DropNaOptions`, `DropDuplicatesOptions`, `FillNaValue`.
//!
//! All tests require a GPU.

use cudf::{
    DataFrame, DuplicateKeepOption, ErrorCode, ErrorKind,
    frame::dataframe::{DropDuplicatesOptions, DropNaHow, DropNaOptions},
};

use crate::common::{assert_missing_column_error, error_context_value, scalar};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 5-row DataFrame:
///   id:    [1, 2,    3,    4, 5]      (i32, no nulls)
///   score: [10.0, ?, 30.0, ?, 50.0]  (f64, nulls at rows 1,3)
///   tag:   ["a", "b", ?, "d", ?]      (string, nulls at rows 2,4)
fn sample_df_with_nulls() -> DataFrame {
    let ids = crate::common::series::<i32>("id", &[1, 2, 3, 4, 5]).unwrap();
    let scores = crate::common::nullable_series::<f64>(
        "score",
        &[10.0, 0.0, 30.0, 0.0, 50.0],
        &[true, false, true, false, true],
    )
    .unwrap();
    let tags =
        crate::common::nullable_series("tag", &["a", "b", "", "d", ""], &[true, true, false, true, false]).unwrap();
    DataFrame::from_columns([("id", ids), ("score", scores), ("tag", tags)]).unwrap()
}

/// Build a 5-row DataFrame with duplicate values:
///   a: [1, 2, 1, 2, 3]
///   b: [10, 20, 10, 20, 30]
fn sample_df_with_dupes() -> DataFrame {
    let a = crate::common::series::<i32>("a", &[1, 2, 1, 2, 3]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20, 10, 20, 30]).unwrap();
    DataFrame::from_columns([("a", a), ("b", b)]).unwrap()
}

// ===========================================================================
// dropna — default (any, all columns)
// ===========================================================================

#[test]
fn test_dropna_default_drops_any_null_row() {
    let df = sample_df_with_nulls();
    // Default: how=any, subset=all columns.
    let result = df.dropna(Default::default()).unwrap();

    // Rows 1,2,3,4 have at least one null → only row 0 survives.
    // Wait: row 0 = (1, 10.0, "a") — all valid.
    //       row 1 = (2, null, "b") — null in score → dropped
    //       row 2 = (3, 30.0, null) — null in tag → dropped
    //       row 3 = (4, null, "d") — null in score → dropped
    //       row 4 = (5, 50.0, null) — null in tag → dropped
    assert_eq!(result.num_rows(), 1);
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1]);
}

// ===========================================================================
// dropna — how = all
// ===========================================================================

#[test]
fn test_dropna_all_mode_keeps_partial_nulls() {
    let df = sample_df_with_nulls();
    let opts = DropNaOptions::builder().how(DropNaHow::All).build();
    let result = df.dropna(opts).unwrap();

    // "all" means drop only rows where ALL columns are null.
    // No row has ALL three columns null, so all 5 rows survive.
    assert_eq!(result.num_rows(), 5);
}

// ===========================================================================
// dropna — subset
// ===========================================================================

#[test]
fn test_dropna_subset_single_column() {
    let df = sample_df_with_nulls();
    let opts = DropNaOptions::builder().subset(&["score"]).build();
    let result = df.dropna(opts).unwrap();

    // score is null at rows 1,3 → 3 rows survive.
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3, 5]);
}

#[test]
fn test_dropna_subset_multiple_columns() {
    let df = sample_df_with_nulls();
    let opts = DropNaOptions::builder().subset(&["score", "tag"]).build();
    let result = df.dropna(opts).unwrap();

    // Rows with any null in score OR tag: rows 1,2,3,4 → only row 0 survives.
    assert_eq!(result.num_rows(), 1);
}

// ===========================================================================
// dropna — threshold
// ===========================================================================

#[test]
fn test_dropna_threshold() {
    let df = sample_df_with_nulls();
    // thresh=2 on subset ["score","tag"] means: keep row if at least 2 of the
    // subset columns are non-null.
    let opts = DropNaOptions::builder().subset(&["score", "tag"]).thresh(2).build();
    let result = df.dropna(opts).unwrap();

    // Row 0: score=valid, tag=valid → 2 non-null → keep
    // Row 1: score=null, tag=valid → 1 non-null → drop
    // Row 2: score=valid, tag=null → 1 non-null → drop
    // Row 3: score=null, tag=valid → 1 non-null → drop
    // Row 4: score=valid, tag=null → 1 non-null → drop
    assert_eq!(result.num_rows(), 1);
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1]);
}

#[test]
fn test_dropna_threshold_one_keeps_most_rows() {
    let df = sample_df_with_nulls();
    // thresh=1 on subset ["score","tag"] means: keep row if at least 1 is non-null.
    let opts = DropNaOptions::builder().subset(&["score", "tag"]).thresh(1).build();
    let result = df.dropna(opts).unwrap();

    // All 5 rows have at least 1 non-null in {score, tag}.
    assert_eq!(result.num_rows(), 5);
}

// ===========================================================================
// dropna — error cases
// ===========================================================================

#[test]
fn test_dropna_invalid_column_name() {
    let df = sample_df_with_nulls();
    let opts = DropNaOptions::builder().subset(&["nonexistent"]).build();
    let err = df.dropna(opts).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_dropna_negative_thresh_uses_stable_validation_contract() {
    let df = sample_df_with_nulls();
    let opts = DropNaOptions::builder().subset(&["score", "tag"]).thresh(-1).build();

    let err = df.dropna(opts).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::dropna");
    assert_eq!(error_context_value(&err, "reason"), Some("negative_keep_threshold"));
    assert_eq!(error_context_value(&err, "remedy"), Some("use keep_threshold >= 0"));
    assert_eq!(error_context_value(&err, "keys_count"), Some("2"));
    assert_eq!(error_context_value(&err, "keep_threshold"), Some("-1"));
    assert!(
        err.context().iter().any(|(key, value)| *key == "called" && value == "dataframe::drop_nulls"),
        "expected wrapped drop_nulls call in called context, got {:?}",
        err.context()
    );
}

#[test]
fn test_dropna_subset_count_limit_uses_canonical_contract() {
    let err = cudf::testing::trigger_dataframe_dropna_subset_count_limit_error_for_test().unwrap_err();
    let expected = usize::MAX.to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::dropna");
    assert_eq!(error_context_value(&err, "column_name_count"), Some(expected.as_str()));
    assert_eq!(error_context_value(&err, "resource"), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason"), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy"), Some("reduce column_name_count to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "subset_count"), None);
}

// ===========================================================================
// dropna — empty result preserves schema
// ===========================================================================

#[test]
fn test_dropna_empty_result_preserves_schema() {
    // All rows have nulls in some column.
    let a = crate::common::nullable_series::<i32>("a", &[0, 0], &[false, false]).unwrap();
    let b = crate::common::nullable_series::<i32>("b", &[0, 0], &[false, false]).unwrap();
    let df = DataFrame::from_columns([("a", a), ("b", b)]).unwrap();

    let result = df.dropna(Default::default()).unwrap();
    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), vec!["a", "b"]);
}

// ===========================================================================
// drop_duplicates — basic (all columns)
// ===========================================================================

#[test]
fn test_drop_duplicates_default() {
    let df = sample_df_with_dupes();
    let result = df.drop_duplicates(Default::default()).unwrap();

    // 3 distinct rows: (1,10), (2,20), (3,30)
    assert_eq!(result.num_rows(), 3);
}

// ===========================================================================
// drop_duplicates — subset
// ===========================================================================

#[test]
fn test_drop_duplicates_subset() {
    let df = sample_df_with_dupes();
    let opts = DropDuplicatesOptions::builder().subset(&["a"]).build();
    let result = df.drop_duplicates(opts).unwrap();

    // 3 distinct values of "a": 1, 2, 3
    assert_eq!(result.num_rows(), 3);
}

// ===========================================================================
// drop_duplicates — keep options
// ===========================================================================

#[test]
fn test_drop_duplicates_keep_first() {
    let a = crate::common::series::<i32>("a", &[1, 1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("a", a), ("b", b)]).unwrap();

    let opts = DropDuplicatesOptions::builder().subset(&["a"]).keep(DuplicateKeepOption::KeepFirst).build();
    let result = df.drop_duplicates(opts).unwrap();

    assert_eq!(result.num_rows(), 2);
    // First occurrence of a=1 has b=10.
    let mut rows: Vec<(i32, i32)> = result
        .series("a")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(result.series("b").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 10), (2, 30)]);
}

#[test]
fn test_drop_duplicates_keep_last() {
    let a = crate::common::series::<i32>("a", &[1, 1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("a", a), ("b", b)]).unwrap();

    let opts = DropDuplicatesOptions::builder().subset(&["a"]).keep(DuplicateKeepOption::KeepLast).build();
    let result = df.drop_duplicates(opts).unwrap();

    assert_eq!(result.num_rows(), 2);
    let mut rows: Vec<(i32, i32)> = result
        .series("a")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(result.series("b").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 20), (2, 30)]);
}

#[test]
fn test_drop_duplicates_keep_none() {
    let a = crate::common::series::<i32>("a", &[1, 1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("a", a), ("b", b)]).unwrap();

    let opts = DropDuplicatesOptions::builder().subset(&["a"]).keep(DuplicateKeepOption::KeepNone).build();
    let result = df.drop_duplicates(opts).unwrap();

    // a=1 appears twice → both removed; a=2 appears once → kept.
    assert_eq!(result.num_rows(), 1);
    assert_eq!(result.series("a").unwrap().to_vec_strict::<i32>().unwrap(), vec![2]);
}

// ===========================================================================
// drop_duplicates — error on invalid column
// ===========================================================================

#[test]
fn test_drop_duplicates_invalid_column() {
    let df = sample_df_with_dupes();
    let opts = DropDuplicatesOptions::builder().subset(&["nonexistent"]).build();
    let err = df.drop_duplicates(opts).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// drop_duplicates — preserves all columns
// ===========================================================================

#[test]
fn test_drop_duplicates_preserves_schema() {
    let df = sample_df_with_dupes();
    let opts = DropDuplicatesOptions::builder().subset(&["a"]).build();
    let result = df.drop_duplicates(opts).unwrap();

    assert_eq!(result.column_names(), vec!["a", "b"]);
}

// ===========================================================================
// fillna — scalar, single column
// ===========================================================================

#[test]
fn test_fillna_scalar_single_column() {
    let df = sample_df_with_nulls();
    let fill = scalar(999.0f64).unwrap();
    let result = df.fillna_with_scalar("score", &fill).unwrap();

    // score nulls at rows 1,3 replaced with 999.0
    let scores = result.series("score").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(scores, vec![10.0, 999.0, 30.0, 999.0, 50.0]);

    // Other columns untouched.
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
}

// ===========================================================================
// fillna — with Series
// ===========================================================================

#[test]
fn test_fillna_with_series_single_column() {
    let df = sample_df_with_nulls();
    let fill_series = crate::common::series::<f64>("fill", &[0.0, 20.0, 0.0, 40.0, 0.0]).unwrap();
    let result = df.fillna_with_series("score", &fill_series).unwrap();

    let scores = result.series("score").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(scores, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
}

// ===========================================================================
// fillna — error cases
// ===========================================================================

#[test]
fn test_fillna_invalid_column() {
    let df = sample_df_with_nulls();
    let fill = scalar(0i32).unwrap();
    let err = df.fillna_with_scalar("nonexistent", &fill).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_fillna_with_series_invalid_column() {
    let df = sample_df_with_nulls();
    let fill_series = crate::common::series::<f64>("fill", &[0.0; 5]).unwrap();
    let err = df.fillna_with_series("nonexistent", &fill_series).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// fillna — preserves schema and other columns
// ===========================================================================

#[test]
fn test_fillna_preserves_schema() {
    let df = sample_df_with_nulls();
    let fill = scalar(0.0f64).unwrap();
    let result = df.fillna_with_scalar("score", &fill).unwrap();

    assert_eq!(result.column_names(), vec!["id", "score", "tag"]);
    assert_eq!(result.num_rows(), 5);
}

// ===========================================================================
// DropNaOptions builder API
// ===========================================================================

#[test]
fn test_dropna_options_builder_chaining() {
    // Verify that builder methods can be chained.
    let _opts = DropNaOptions::builder().subset(&["a", "b"]).how(DropNaHow::All).thresh(2).build();

    // thresh takes precedence over how, so this should behave like thresh=2
    let df = sample_df_with_nulls();
    let subset_opts = DropNaOptions::builder().subset(&["score"]).how(DropNaHow::All).build();
    let result = df.dropna(subset_opts).unwrap();
    // how=all on just "score": drop only when score is null.
    // score null at rows 1,3 → 3 rows survive.
    assert_eq!(result.num_rows(), 3);
}

// ===========================================================================
// DropDuplicatesOptions builder API
// ===========================================================================

#[test]
fn test_drop_duplicates_options_default_is_keep_any() {
    let df = sample_df_with_dupes();
    let opts = DropDuplicatesOptions::new();
    let result = df.drop_duplicates(opts).unwrap();
    // Default keep=KeepAny → 3 distinct rows
    assert_eq!(result.num_rows(), 3);
}

// ===========================================================================
// High-level DataFrame option types
// ===========================================================================

#[test]
fn test_high_level_option_types_importable_from_dataframe_module() {
    let _: DropNaOptions = Default::default();
    let _: DropNaHow = Default::default();
    let _: DropDuplicatesOptions = Default::default();
}
