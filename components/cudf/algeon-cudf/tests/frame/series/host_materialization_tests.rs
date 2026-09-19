//! Integration tests for `Series` host materialization convenience methods.
//!
//! Methods under test: `to_vec_strict<T>`, `to_optional_vec<T>`,
//! `to_strings_strict`, `to_optional_strings`, `to_strings_lossy`.
//!
//! All tests require a GPU.

use cudf::{ErrorCode, ErrorKind, TypeId, testing::prelude::*};

use crate::common::error_context_value;

// ===========================================================================
// to_vec_strict<T> — numeric host materialization (rejects nullable)
// ===========================================================================

fn test_to_vec_i32() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![10, 20, 30]);
}

fn test_to_vec_f64() {
    let s = crate::common::series::<f64>("x", &[1.5, 2.5, 3.5]).unwrap();
    let vals = s.to_vec_strict::<f64>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

fn test_to_vec_bool() {
    let s = crate::common::series::<bool>("flags", &[true, false, true]).unwrap();
    let vals = s.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

fn test_to_vec_i64() {
    let s = crate::common::series::<i64>("big", &[i64::MAX, 0, i64::MIN]).unwrap();
    let vals = s.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![i64::MAX, 0, i64::MIN]);
}

fn test_to_vec_empty() {
    let s = crate::common::series::<i32>("empty", &[]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert!(vals.is_empty());
}

fn test_to_vec_rejects_nullable_data() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 3], &[true, false, true]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "series::to_vec_strict");
    assert_eq!(error_context_value(&err, "null_count").as_deref(), Some("1"));
}

fn test_to_vec_accepts_masked_column_when_all_rows_valid() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 2, 3], &[true, true, true]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

fn test_to_vec_type_mismatch_returns_error() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "column_view::to_vec");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("float64"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
}

fn test_to_vec_on_string_column_returns_error() {
    let s = crate::common::series("x", &["a", "b"]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

// ===========================================================================
// to_optional_vec<T> — numeric with null awareness
// ===========================================================================

fn test_to_optional_vec_all_valid() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), Some(2), Some(3)]);
}

fn test_to_optional_vec_with_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(30)]);
}

fn test_to_optional_vec_all_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, None]);
}

fn test_to_optional_vec_f64_with_nulls() {
    let s = crate::common::nullable_series::<f64>("x", &[1.5, 0.0, 3.5], &[true, false, true]).unwrap();
    let vals = s.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(1.5), None, Some(3.5)]);
}

fn test_to_optional_vec_empty() {
    let s = crate::common::series::<i32>("empty", &[]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert!(vals.is_empty());
}

fn test_to_optional_vec_type_mismatch_returns_error() {
    let s = crate::common::series::<f64>("x", &[1.0]).unwrap();
    let err = s.to_optional_vec::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "column_view::to_vec");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("float64"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
}

fn test_to_optional_vec_bool_with_nulls() {
    let s = crate::common::nullable_series::<bool>("f", &[true, false, true], &[true, false, true]).unwrap();
    let vals = s.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

fn test_to_optional_vec_bool_preserves_false_and_skips_nulls() {
    let s =
        crate::common::nullable_series::<bool>("f", &[false, true, false, true], &[true, true, false, true]).unwrap();
    let vals = s.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), Some(true), None, Some(true)]);
}

// ===========================================================================
// to_strings_strict — rejects nullable string columns
// ===========================================================================

fn test_to_strings_basic() {
    let s = crate::common::series("x", &["hello", "world"]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello".to_string(), "world".to_string()]);
}

fn test_to_strings_empty() {
    let s = crate::common::series::<&str>("x", &[]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert!(vals.is_empty());
}

fn test_to_strings_rejects_nullable_data() {
    let s = crate::common::nullable_series("x", &["a", "b"], &[true, false]).unwrap();
    let err = s.to_strings_strict().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "series::to_strings_strict");
    assert_eq!(error_context_value(&err, "null_count").as_deref(), Some("1"));
}

fn test_to_strings_accepts_masked_column_when_all_rows_valid() {
    let s = crate::common::nullable_series("x", &["a", "b"], &[true, true]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "b".to_string()]);
}

fn test_to_strings_on_numeric_returns_error() {
    let s = crate::common::series::<i32>("x", &[1, 2]).unwrap();
    let err = s.to_strings_strict().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "column_view::to_strings_lossy");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
}

// ===========================================================================
// to_optional_strings — null-preserving string materialization
// ===========================================================================

fn test_to_optional_strings_with_nulls() {
    let s = crate::common::nullable_series("x", &["a", "b", "c"], &[true, false, true]).unwrap();
    let vals = s.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("a".to_string()), None, Some("c".to_string())]);
}

fn test_to_optional_strings_without_nulls() {
    let s = crate::common::series("x", &["a", "b"]).unwrap();
    let vals = s.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("a".to_string()), Some("b".to_string())]);
}

fn test_to_optional_strings_on_numeric_returns_type_mismatch_error() {
    let s = crate::common::series::<i32>("x", &[1, 2]).unwrap();
    let err = s.to_optional_strings().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "column_view::to_optional_strings");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("string"));
}

// ===========================================================================
// to_strings_lossy — nulls become empty strings
// ===========================================================================

fn test_to_strings_lossy_maps_null_to_empty() {
    let s = crate::common::nullable_series("x", &["a", "b", "c"], &[true, false, true]).unwrap();
    let vals = s.to_strings_lossy().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "".to_string(), "c".to_string()]);
}

fn test_to_strings_lossy_non_nullable() {
    let s = crate::common::series("x", &["a", "b"]).unwrap();
    let vals = s.to_strings_lossy().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "b".to_string()]);
}

fn test_series_to_strings_lossy_matches_column_view_lossy_materialization() {
    let s = crate::common::nullable_series("x", &["", "b", ""], &[true, false, true]).unwrap();

    let series_vals = s.to_strings_lossy().unwrap();
    let view_vals = s.testing_view().unwrap().to_strings_lossy().unwrap();

    assert_eq!(series_vals, view_vals);
}

// ===========================================================================
// materialization methods do not consume the Series
// ===========================================================================

fn test_materialization_preserves_series() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 3], &[true, false, true]).unwrap();

    // to_vec_strict would reject this nullable Series, so use to_optional_vec.
    let _v1 = s.to_optional_vec::<i32>().unwrap();

    // Series is still usable.
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

#[test]
fn host_materialization_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_to_vec_i32", test_to_vec_i32),
        ("test_to_vec_f64", test_to_vec_f64),
        ("test_to_vec_bool", test_to_vec_bool),
        ("test_to_vec_i64", test_to_vec_i64),
        ("test_to_vec_empty", test_to_vec_empty),
        ("test_to_vec_rejects_nullable_data", test_to_vec_rejects_nullable_data),
        (
            "test_to_vec_accepts_masked_column_when_all_rows_valid",
            test_to_vec_accepts_masked_column_when_all_rows_valid,
        ),
        ("test_to_vec_type_mismatch_returns_error", test_to_vec_type_mismatch_returns_error),
        ("test_to_vec_on_string_column_returns_error", test_to_vec_on_string_column_returns_error),
        ("test_to_optional_vec_all_valid", test_to_optional_vec_all_valid),
        ("test_to_optional_vec_with_nulls", test_to_optional_vec_with_nulls),
        ("test_to_optional_vec_all_nulls", test_to_optional_vec_all_nulls),
        ("test_to_optional_vec_f64_with_nulls", test_to_optional_vec_f64_with_nulls),
        ("test_to_optional_vec_empty", test_to_optional_vec_empty),
        ("test_to_optional_vec_type_mismatch_returns_error", test_to_optional_vec_type_mismatch_returns_error),
        ("test_to_optional_vec_bool_with_nulls", test_to_optional_vec_bool_with_nulls),
        (
            "test_to_optional_vec_bool_preserves_false_and_skips_nulls",
            test_to_optional_vec_bool_preserves_false_and_skips_nulls,
        ),
        ("test_to_strings_basic", test_to_strings_basic),
        ("test_to_strings_empty", test_to_strings_empty),
        ("test_to_strings_rejects_nullable_data", test_to_strings_rejects_nullable_data),
        (
            "test_to_strings_accepts_masked_column_when_all_rows_valid",
            test_to_strings_accepts_masked_column_when_all_rows_valid,
        ),
        ("test_to_strings_on_numeric_returns_error", test_to_strings_on_numeric_returns_error),
        ("test_to_optional_strings_with_nulls", test_to_optional_strings_with_nulls),
        ("test_to_optional_strings_without_nulls", test_to_optional_strings_without_nulls),
        (
            "test_to_optional_strings_on_numeric_returns_type_mismatch_error",
            test_to_optional_strings_on_numeric_returns_type_mismatch_error,
        ),
        ("test_to_strings_lossy_maps_null_to_empty", test_to_strings_lossy_maps_null_to_empty),
        ("test_to_strings_lossy_non_nullable", test_to_strings_lossy_non_nullable),
        (
            "test_series_to_strings_lossy_matches_column_view_lossy_materialization",
            test_series_to_strings_lossy_matches_column_view_lossy_materialization,
        ),
        ("test_materialization_preserves_series", test_materialization_preserves_series),
    ]);
}
