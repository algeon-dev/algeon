//! Tests for the semantic host materialization naming conventions.
//!
//! Validates `_strict` / `_optional` / `_lossy` suffixes on Series
//! host materialization methods.

use cudf::{ErrorKind, Series};

// ===========================================================================
// to_vec_strict — rejects nullable columns
// ===========================================================================

#[test]
fn test_to_vec_strict_non_nullable_ok() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_to_vec_strict_rejects_nullable() {
    let s = Series::new_nullable::<i32>("x", &[1, 0, 3], &[true, false, true]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_to_vec_strict_all_valid_masked_column_ok() {
    let s = Series::new_nullable::<i32>("x", &[1, 2, 3], &[true, true, true]).unwrap();
    let vals = s.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_to_vec_strict_type_mismatch() {
    let s = Series::new::<f64>("x", &[1.0]).unwrap();
    let err = s.to_vec_strict::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TypeError);
}

// ===========================================================================
// to_vec_optional — preserves nulls as Option
// ===========================================================================

#[test]
fn test_to_vec_optional_with_nulls() {
    let s = Series::new_nullable::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let vals = s.to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), None, Some(30)]);
}

#[test]
fn test_to_vec_optional_non_nullable() {
    let s = Series::new::<i32>("x", &[1, 2]).unwrap();
    let vals = s.to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), Some(2)]);
}

// ===========================================================================
// to_strings_strict — rejects nullable string columns
// ===========================================================================

#[test]
fn test_to_strings_strict_non_nullable_ok() {
    let s = Series::from_strings("x", &["hello", "world"]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello".to_string(), "world".to_string()]);
}

#[test]
fn test_to_strings_strict_rejects_nullable() {
    let s = Series::from_strings_nullable("x", &["a", "b"], &[true, false]).unwrap();
    let err = s.to_strings_strict().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_to_strings_strict_all_valid_masked_column_ok() {
    let s = Series::from_strings_nullable("x", &["a", "b"], &[true, true]).unwrap();
    let vals = s.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "b".to_string()]);
}

// ===========================================================================
// to_strings_optional — nulls as None
// ===========================================================================

#[test]
fn test_to_strings_optional_with_nulls() {
    let s = Series::from_strings_nullable("x", &["a", "b", "c"], &[true, false, true]).unwrap();
    let vals = s.to_strings_optional().unwrap();
    assert_eq!(vals, vec![Some("a".to_string()), None, Some("c".to_string())]);
}

#[test]
fn test_to_strings_optional_non_nullable() {
    let s = Series::from_strings("x", &["a", "b"]).unwrap();
    let vals = s.to_strings_optional().unwrap();
    assert_eq!(vals, vec![Some("a".to_string()), Some("b".to_string())]);
}

// ===========================================================================
// to_strings_lossy — nulls become empty strings
// ===========================================================================

#[test]
fn test_to_strings_lossy_maps_null_to_empty() {
    let s = Series::from_strings_nullable("x", &["a", "b", "c"], &[true, false, true]).unwrap();
    let vals = s.to_strings_lossy().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "".to_string(), "c".to_string()]);
}

#[test]
fn test_to_strings_lossy_non_nullable() {
    let s = Series::from_strings("x", &["a", "b"]).unwrap();
    let vals = s.to_strings_lossy().unwrap();
    assert_eq!(vals, vec!["a".to_string(), "b".to_string()]);
}
