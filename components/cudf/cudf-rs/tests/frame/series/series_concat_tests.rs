//! Integration tests for `Series::concat`.
//!
//! All tests require a GPU.

use cudf::{Series, TypeId};

// ===========================================================================
// Basic concatenation
// ===========================================================================

#[test]
fn test_concat_two_i32_series() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i32>("b", &[3, 4, 5]).unwrap();

    let result = Series::concat(&[&a, &b]).unwrap();

    assert_eq!(result.name(), None, "concat produces unnamed Series");
    assert_eq!(result.len(), 5);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_concat_three_f64_series() {
    let a = Series::new::<f64>("a", &[1.0]).unwrap();
    let b = Series::new::<f64>("b", &[2.0, 3.0]).unwrap();
    let c = Series::new::<f64>("c", &[4.0]).unwrap();

    let result = Series::concat(&[&a, &b, &c]).unwrap();

    assert_eq!(result.len(), 4);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_concat_strings() {
    let a = Series::from_strs("a", &["hello"]).unwrap();
    let b = Series::from_strs("b", &["world"]).unwrap();

    let result = Series::concat(&[&a, &b]).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(result.to_strings_strict().unwrap(), vec!["hello", "world"]);
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn test_concat_single_series() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();

    let result = Series::concat(&[&a]).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_concat_with_empty_series() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i32>("b", &[]).unwrap();
    let c = Series::new::<i32>("c", &[3]).unwrap();

    let result = Series::concat(&[&a, &b, &c]).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_concat_preserves_nulls() {
    let a = Series::new_nullable::<i32>("a", &[1, 0], &[true, false]).unwrap();
    let b = Series::new::<i32>("b", &[3]).unwrap();

    let result = Series::concat(&[&a, &b]).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.null_count(), 1);
}

// ===========================================================================
// Error cases
// ===========================================================================

#[test]
fn test_concat_empty_slice_returns_error() {
    let result = Series::concat(&[]);
    assert!(result.is_err(), "concat of empty slice should fail");
}

#[test]
fn test_concat_type_mismatch_returns_error() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::from_strs("b", &["hello"]).unwrap();

    let result = Series::concat(&[&a, &b]);
    assert!(result.is_err(), "concat of mismatched types should fail");
}
