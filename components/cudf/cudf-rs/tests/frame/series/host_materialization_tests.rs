//! Integration tests for `Series` host materialization convenience methods.
//!
//! Methods under test: `to_vec_strict<T>`, `to_optional_vec<T>`,
//! `to_strings_strict`, `to_optional_strings`, `to_strings_lossy`.
//!
//! All tests require a GPU.

use cudf::{ErrorKind, Series, TypeId};

// ===========================================================================
// to_vec_strict<T> — numeric host materialization (rejects nullable)
// ===========================================================================

#[test]
fn test_to_vec_i32() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn test_to_vec_f64() {
    let s = Series::new::<f64>("x", &[1.5, 2.5, 3.5]).unwrap();
    let vals = s.to_vec_strict::<f64>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_to_vec_bool() {
    let s = Series::new::<bool>("flags", &[true, false, true]).unwrap();
    let vals = s.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

#[test]
fn test_to_vec_i64() {
    let s = Series::new::<i64>("big", &[i64::MAX, 0, i64::MIN]).unwrap();
    let vals = s.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![i64::MAX, 0, i64::MIN]);
}

#[test]
fn test_to_vec_empty() {
    let s = Series::new::<i32>("empty", &[]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_to_vec_rejects_nullable_data() {
    let s = Series::new_nullable::<i32>("x", &[1, 0, 3], &[true, false, true]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_to_vec_accepts_masked_column_when_all_rows_valid() {
    let s = Series::new_nullable::<i32>("x", &[1, 2, 3], &[true, true, true]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_to_vec_type_mismatch_returns_error() {
    let s = Series::new::<f64>("x", &[1.0, 2.0]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
}

#[test]
fn test_to_vec_on_string_column_returns_error() {
    let s = Series::from_strs("x", &["a", "b"]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
}

// ===========================================================================
// to_optional_vec<T> — numeric with null awareness
// ===========================================================================

#[test]
fn test_to_optional_vec_all_valid() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), Some(2), Some(3)]);
}

#[test]
fn test_to_optional_vec_with_nulls() {
    let s = Series::new_nullable::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(30)]);
}

#[test]
fn test_to_optional_vec_all_nulls() {
    let s = Series::new_nullable::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, None]);
}

#[test]
fn test_to_optional_vec_f64_with_nulls() {
    let s = Series::new_nullable::<f64>("x", &[1.5, 0.0, 3.5], &[true, false, true]).unwrap();
    let vals = s.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(1.5), None, Some(3.5)]);
}

#[test]
fn test_to_optional_vec_empty() {
    let s = Series::new::<i32>("empty", &[]).unwrap();
    let vals = s.to_optional_vec::<i32>().unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_to_optional_vec_type_mismatch_returns_error() {
    let s = Series::new::<f64>("x", &[1.0]).unwrap();
    let err = s.to_optional_vec::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
}

#[test]
fn test_to_optional_vec_bool_with_nulls() {
    let s = Series::new_nullable::<bool>("f", &[true, false, true], &[true, false, true]).unwrap();
    let vals = s.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(true)]);
}

#[test]
fn test_to_optional_vec_bool_preserves_false_and_skips_nulls() {
    let s = Series::new_nullable::<bool>("f", &[false, true, false, true], &[true, true, false, true]).unwrap();
    let vals = s.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), Some(true), None, Some(true)]);
}

// ===========================================================================
// to_strings_strict — rejects nullable string columns
// ===========================================================================

#[test]
fn test_to_strings_basic() {
    let s = Series::from_strs("x", &["hello", "world"]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello".to_string(), "world".to_string()]);
}

#[test]
fn test_to_strings_empty() {
    let s = Series::from_strs("x", &[]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_to_strings_rejects_nullable_data() {
    let s = Series::from_strs_nullable("x", &["a", "b"], &[true, false]).unwrap();
    let err = s.to_strings_strict().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_to_strings_accepts_masked_column_when_all_rows_valid() {
    let s = Series::from_strs_nullable("x", &["a", "b"], &[true, true]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn test_to_strings_on_numeric_returns_error() {
    let s = Series::new::<i32>("x", &[1, 2]).unwrap();
    let err = s.to_strings_strict().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
}

// ===========================================================================
// to_optional_strings — null-preserving string materialization
// ===========================================================================

#[test]
fn test_to_optional_strings_with_nulls() {
    let s = Series::from_strs_nullable("x", &["a", "b", "c"], &[true, false, true]).unwrap();
    let vals = s.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("a".to_string()), None, Some("c".to_string())]);
}

#[test]
fn test_to_optional_strings_without_nulls() {
    let s = Series::from_strs("x", &["a", "b"]).unwrap();
    let vals = s.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("a".to_string()), Some("b".to_string())]);
}

// ===========================================================================
// to_strings_lossy — nulls become empty strings
// ===========================================================================

#[test]
fn test_to_strings_lossy_maps_null_to_empty() {
    let s = Series::from_strs_nullable("x", &["a", "b", "c"], &[true, false, true]).unwrap();
    let vals = s.to_strings_lossy().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "".to_string(), "c".to_string()]);
}

#[test]
fn test_to_strings_lossy_non_nullable() {
    let s = Series::from_strs("x", &["a", "b"]).unwrap();
    let vals = s.to_strings_lossy().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "b".to_string()]);
}

// ===========================================================================
// materialization methods do not consume the Series
// ===========================================================================

#[test]
fn test_materialization_preserves_series() {
    let s = Series::new_nullable::<i32>("x", &[1, 0, 3], &[true, false, true]).unwrap();

    // to_vec_strict would reject this nullable Series, so use to_optional_vec.
    let _v1 = s.to_optional_vec::<i32>().unwrap();

    // Series is still usable.
    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}
