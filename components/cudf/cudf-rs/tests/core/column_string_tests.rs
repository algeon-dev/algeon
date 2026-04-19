//! Tests for string column creation and reading.

use cudf::{TypeId, testing::prelude::*};

use crate::common::{column_from_slice, column_from_strs, column_from_strs_nullable};

#[test]
fn test_from_strs_basic() {
    let strings = &["hello", "world", "foo"];
    let col = column_from_strs(strings).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);
    assert_eq!(view.null_count().unwrap(), 0);
}

#[test]
fn test_from_strs_roundtrip() {
    let strings = &["hello", "world", "foo"];
    let col = column_from_strs(strings).unwrap();
    let view = col.testing_view().unwrap();

    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec!["hello", "world", "foo"]);
}

#[test]
fn test_from_strs_empty_strings() {
    let strings = &["", "hello", "", "world", ""];
    let col = column_from_strs(strings).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 5);
    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec!["", "hello", "", "world", ""]);
}

#[test]
fn test_from_strs_single() {
    let strings = &["only"];
    let col = column_from_strs(strings).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 1);
    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec!["only"]);
}

#[test]
fn test_from_strs_empty_column() {
    let strings: &[&str] = &[];
    let col = column_from_strs(strings).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 0);
    assert!(view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);

    let result = view.to_strings_lossy().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_from_strs_nullable_basic() {
    let strings = &["hello", "world", "foo"];
    let validity = &[true, false, true];
    let col = column_from_strs_nullable(strings, validity).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 1);

    let mask = view.to_host_validity().unwrap().unwrap();
    assert_eq!(mask, vec![true, false, true]);
}

#[test]
fn test_from_strs_nullable_roundtrip() {
    let strings = &["alpha", "beta", "gamma", "delta"];
    let validity = &[true, false, true, false];
    let col = column_from_strs_nullable(strings, validity).unwrap();
    let view = col.testing_view().unwrap();

    // Lossy materialization maps nulls to empty strings.
    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result[0], "alpha");
    assert_eq!(result[1], "");
    assert_eq!(result[2], "gamma");
    assert_eq!(result[3], "");

    let mask = view.to_host_validity().unwrap().unwrap();
    assert_eq!(mask, vec![true, false, true, false]);
}

#[test]
fn test_to_host_validity_for_sliced_nullable_string_column_is_normalized_to_slice() {
    let col = column_from_strs_nullable(&["alpha", "beta", "gamma", "delta"], &[true, false, true, false]).unwrap();
    let sliced = col.testing_slice_view(1, 3).unwrap();

    let validity = sliced.to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![false, true, false]);
}

#[test]
fn test_sliced_nullable_string_materialization_preserves_active_range() {
    let col = column_from_strs_nullable(&["alpha", "", "gamma", "delta"], &[true, false, true, true]).unwrap();
    let sliced = col.testing_slice_view(1, 3).unwrap();

    assert_eq!(sliced.to_strings_lossy().unwrap(), vec![String::new(), "gamma".to_string(), "delta".to_string()]);
    assert_eq!(sliced.to_optional_strings().unwrap(), vec![None, Some("gamma".to_string()), Some("delta".to_string())]);
}

#[test]
fn test_from_strs_nullable_mismatched_lengths() {
    let strings = &["a", "b", "c"];
    let validity = &[true, false]; // wrong length
    let result = column_from_strs_nullable(strings, validity);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_from_strs_utf8() {
    let strings = &["héllo", "wörld", "日本語", "🦀"];
    let col = column_from_strs(strings).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec!["héllo", "wörld", "日本語", "🦀"]);
}

#[test]
fn test_from_strs_column_has_children() {
    // String columns have at least 1 child (offsets)
    let col = column_from_strs(&["a", "b"]).unwrap();
    let view = col.testing_view().unwrap();
    assert!(view.num_children().unwrap() >= 1);
}

#[test]
fn test_to_strings_rejects_non_string_column() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let result = view.to_strings_lossy();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
}

#[test]
fn test_to_strings_lossy_maps_nulls_to_empty_strings() {
    let col = column_from_strs_nullable(&["", "placeholder", "suffix"], &[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec![String::new(), String::new(), "suffix".to_string()]);
}

#[test]
fn test_to_optional_strings_distinguishes_null_from_empty() {
    let col = column_from_strs_nullable(&["", "placeholder", ""], &[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    let result = view.to_optional_strings().unwrap();
    assert_eq!(result, vec![Some(String::new()), None, Some(String::new())]);
}

#[test]
fn test_series_to_optional_strings_distinguishes_null_from_empty() {
    let series = crate::common::nullable_series("names", &["", "placeholder", ""], &[true, false, true]).unwrap();

    let result = series.to_optional_strings().unwrap();
    assert_eq!(result, vec![Some(String::new()), None, Some(String::new())]);
}

#[test]
fn test_to_optional_strings_rejects_non_string_column() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let result = view.to_optional_strings();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
}

#[test]
fn test_from_strs_deep_copy_roundtrip() {
    let strings = &["deep", "copy", "test"];
    let col = column_from_strs(strings).unwrap();
    let copy = col.deep_copy().unwrap();

    // Original can be dropped; copy should still be valid
    drop(col);
    let view = copy.testing_view().unwrap();
    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec!["deep", "copy", "test"]);
}

#[test]
fn test_from_strs_all_empty() {
    let strings = &["", "", ""];
    let col = column_from_strs(strings).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec!["", "", ""]);
}

#[test]
fn test_from_strs_nullable_all_null() {
    let strings = &["a", "b", "c"];
    let validity = &[false, false, false];
    let col = column_from_strs_nullable(strings, validity).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.null_count().unwrap(), 3);
    let mask = view.to_host_validity().unwrap().unwrap();
    assert_eq!(mask, vec![false, false, false]);
}

#[test]
fn test_from_strs_nullable_no_nulls() {
    let strings = &["x", "y", "z"];
    let validity = &[true, true, true];
    let col = column_from_strs_nullable(strings, validity).unwrap();
    let view = col.testing_view().unwrap();

    // null_count should be 0, but column is still nullable (has mask)
    assert_eq!(view.null_count().unwrap(), 0);
    let result = view.to_strings_lossy().unwrap();
    assert_eq!(result, vec!["x", "y", "z"]);
}
