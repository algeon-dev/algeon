//! Tests for `SortOptions` and the high-level `DataFrame::sort_values`.

use cudf::{
    DataFrame, ErrorCode, ErrorKind, SortOptions,
    core::error::{FactKeyName, FactValue},
    testing::prelude::*,
};

use crate::common::{
    assert_missing_column_error, column_from_slice, column_from_slice_nullable, error_context_value, sample_df,
};

fn assert_sort_options_validation_error(
    err: &cudf::Error,
    operation: &str,
    reason: &str,
    remedy: &str,
    actual_value: &str,
) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "resource").as_deref(), Some("sort_options"));
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(1)));
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(actual_value.parse().unwrap())));
    assert_eq!(err.detail_value("keys_count"), None);
    assert_eq!(err.detail_value("actual_count"), None);
    assert_eq!(error_context_value(err, "num_keys").as_deref(), None);
    assert_eq!(error_context_value(err, "descending_len").as_deref(), None);
    assert_eq!(error_context_value(err, "nulls_first_len").as_deref(), None);
}

// ---------------------------------------------------------------------------
// SortOptions defaults
// ---------------------------------------------------------------------------

#[test]
fn test_sort_options_defaults() {
    let opts = SortOptions::new();
    assert!(opts.descending.is_empty(), "default descending should be empty (= all ascending)");
    assert!(opts.nulls_first.is_empty(), "default nulls_first should be empty (= all nulls last)");
    assert!(!opts.stable, "default should be unstable sort");
}

// ---------------------------------------------------------------------------
// SortOptions builder
// ---------------------------------------------------------------------------

#[test]
fn test_sort_options_builder() {
    let opts =
        SortOptions::new().with_descending(vec![true, false]).with_nulls_first(vec![false, true]).with_stable(true);

    assert_eq!(opts.descending, vec![true, false]);
    assert_eq!(opts.nulls_first, vec![false, true]);
    assert!(opts.stable);
}

// ---------------------------------------------------------------------------
// sort_values — single column ascending (default)
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_single_column_ascending() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
    assert_eq!(sorted.column_names(), vec!["id", "score"]);
}

// ---------------------------------------------------------------------------
// sort_values — single column descending
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_single_column_descending() {
    let df = sample_df();
    let sorted = df.sort_values(["score"], SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

// ---------------------------------------------------------------------------
// sort_values — multi-column
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_multi_column() {
    // Sort by id ascending, then score descending within each id group.
    let ids = crate::common::series::<i32>("id", &[2, 1, 2, 1]).unwrap();
    let scores = crate::common::series::<f64>("score", &[20.0, 10.0, 40.0, 30.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    let sorted = df.sort_values(["id", "score"], SortOptions::new().with_descending(vec![false, true])).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 2, 2]);
    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 20.0]);
}

#[test]
fn test_sort_values_multi_column_with_owned_string_keys() {
    let ids = crate::common::series::<i32>("id", &[2, 1, 2, 1]).unwrap();
    let scores = crate::common::series::<f64>("score", &[20.0, 10.0, 40.0, 30.0]).unwrap();
    let tags = crate::common::series::<i32>("tag", &[200, 100, 400, 300]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores), ("tag", tags)]).unwrap();

    let keys = vec!["id".to_string(), "score".to_string()];
    let sorted = df.sort_values(keys, SortOptions::new().with_descending(vec![false, true])).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 2, 2]);
    assert_eq!(sorted.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 20.0]);
    assert_eq!(sorted.series("tag").unwrap().to_vec_strict::<i32>().unwrap(), vec![300, 100, 400, 200]);
}

// ---------------------------------------------------------------------------
// sort_values — stable sort preserves insertion order for ties
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_stable_preserves_order() {
    // All ids are the same — stable sort should preserve original row order.
    let ids = crate::common::series::<i32>("id", &[1, 1, 1, 1]).unwrap();
    let tags = crate::common::series::<i32>("tag", &[10, 20, 30, 40]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("tag", tags)]).unwrap();

    let sorted = df.sort_values(["id"], SortOptions::new().with_stable(true)).unwrap();

    // Tags should remain in original insertion order since all ids are equal.
    assert_eq!(sorted.series("tag").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40]);
}

// ---------------------------------------------------------------------------
// sort_values — rejects mismatched descending length
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_rejects_mismatched_descending_length() {
    let df = sample_df();
    // 1 sort key but 2 descending flags
    let result = df.sort_values(["id"], SortOptions::new().with_descending(vec![true, false]));

    let err = result.unwrap_err();
    assert_sort_options_validation_error(
        &err,
        "dataframe::sort_values",
        "descending_length_mismatch",
        "provide either zero descending flags or exactly one flag per sort key",
        "2",
    );
}

// ---------------------------------------------------------------------------
// sort_values — rejects mismatched nulls_first length
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_rejects_mismatched_nulls_first_length() {
    let df = sample_df();
    // 1 sort key but 3 nulls_first flags
    let result = df.sort_values(["id"], SortOptions::new().with_nulls_first(vec![true, false, true]));

    let err = result.unwrap_err();
    assert_sort_options_validation_error(
        &err,
        "dataframe::sort_values",
        "nulls_first_length_mismatch",
        "provide either zero nulls_first flags or exactly one flag per sort key",
        "3",
    );
}

// ---------------------------------------------------------------------------
// sort_values — preserves schema
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_preserves_schema() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    assert_eq!(sorted.column_names(), vec!["id", "score"]);
    assert_eq!(sorted.column_index("id"), Some(0));
    assert_eq!(sorted.column_index("score"), Some(1));
}

// ---------------------------------------------------------------------------
// sort_values — nonexistent column returns error
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_nonexistent_column() {
    let df = sample_df();
    let result = df.sort_values(["missing"], SortOptions::new());

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_missing_column_error(&err, "missing");
}

// ---------------------------------------------------------------------------
// sort_values — empty descending vec means all ascending (defaults)
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_empty_options_means_defaults() {
    let df = sample_df();
    // Empty descending + empty nulls_first → all ascending, nulls last.
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    // Should be same as explicit ascending.
    let expected = df.sort_values(["id"], SortOptions::new().with_descending(vec![false])).unwrap();
    assert_eq!(
        sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
        expected.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// sort_values — nulls_first controls null placement
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_nulls_first() {
    let ids = column_from_slice_nullable(&[0i32, 2, 0, 1], &[false, true, false, true]);
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]);
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "val".to_string()],
        vec![ids.unwrap(), vals.unwrap()],
    )
    .unwrap();

    // Sort by id ascending, nulls first.
    let sorted = df.sort_values(["id"], SortOptions::new().with_nulls_first(vec![true])).unwrap();

    let id_vals = sorted.series("id").unwrap().to_optional_vec::<i32>().unwrap();
    // Nulls should appear first (their host value is 0, but they're null).
    // Then the non-null values in ascending order: 1, 2.
    // The null positions have undefined data values, but the validity mask marks them null.
    // We check that non-null values appear at the end in correct order.
    assert_eq!(id_vals[2], Some(1));
    assert_eq!(id_vals[3], Some(2));
}

#[test]
fn test_sort_values_descending_null_placement() {
    let ids = column_from_slice_nullable(&[0i32, 2, 3, 0, 1], &[false, true, true, false, true]);
    let vals = column_from_slice(&[10i32, 20, 30, 40, 50]);
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "val".to_string()],
        vec![ids.unwrap(), vals.unwrap()],
    )
    .unwrap();

    let nulls_first =
        df.sort_values(["id"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![true])).unwrap();
    assert_eq!(
        nulls_first.series("id").unwrap().to_optional_vec::<i32>().unwrap(),
        vec![None, None, Some(3), Some(2), Some(1)]
    );
    assert_eq!(nulls_first.series("val").unwrap().to_vec_strict::<i32>().unwrap()[2..], [30, 20, 50]);

    let nulls_last =
        df.sort_values(["id"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false])).unwrap();
    assert_eq!(
        nulls_last.series("id").unwrap().to_optional_vec::<i32>().unwrap(),
        vec![Some(3), Some(2), Some(1), None, None]
    );
    assert_eq!(nulls_last.series("val").unwrap().to_vec_strict::<i32>().unwrap()[..3], [30, 20, 50]);
}

#[test]
fn test_sort_values_default_places_nulls_last() {
    let ids = column_from_slice_nullable(&[0i32, 2, 0, 1], &[false, true, false, true]);
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]);
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "val".to_string()],
        vec![ids.unwrap(), vals.unwrap()],
    )
    .unwrap();

    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
    let id_col = sorted.series("id").unwrap();
    let id_vals = id_col.to_optional_vec::<i32>().unwrap();

    assert_eq!(id_vals[0], Some(1));
    assert_eq!(id_vals[1], Some(2));
    assert_eq!(id_col.testing_view().unwrap().to_host_validity().unwrap().unwrap(), vec![true, true, false, false]);
}

// ---------------------------------------------------------------------------
// sort_values — with_stable(false) is the default (unstable sort)
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_unstable_by_default() {
    let df = sample_df();
    // Default unstable sort should still produce correct ordering.
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();

    let ids = sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    // Check monotonically non-decreasing.
    for w in ids.windows(2) {
        assert!(w[0] <= w[1], "unstable sort produced incorrect order: {:?}", ids);
    }
}

// ---------------------------------------------------------------------------
// sort_values with explicit ascending matches default
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_explicit_ascending() {
    let df = sample_df();
    let sorted = df.sort_values(["id"], SortOptions::new().with_descending(vec![false])).unwrap();
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
}
