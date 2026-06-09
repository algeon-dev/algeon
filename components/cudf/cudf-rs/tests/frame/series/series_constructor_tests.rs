//! Integration tests for `Series` constructor gap-fillers:
//! `nulls`, `from_scalar`, `empty`.
//!
//! All tests require a GPU.

use cudf::{Series, TypeId, core::Scalar};

// ===========================================================================
// Series::from_optional
// ===========================================================================

#[test]
fn test_from_optional_i32() {
    let s = Series::from_optional("x", &[Some(10i32), None, Some(30)]).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_optional_vec::<i32>().unwrap(), vec![Some(10), None, Some(30)]);
}

#[test]
fn test_from_optional_bool_preserves_false() {
    let s = Series::from_optional("flag", &[Some(false), Some(true), None, Some(false)]).unwrap();

    assert_eq!(s.name(), Some("flag"));
    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.to_optional_vec::<bool>().unwrap(), vec![Some(false), Some(true), None, Some(false)]);
}

#[test]
fn test_from_optional_empty() {
    let s = Series::from_optional::<i32>("empty", &[]).unwrap();

    assert_eq!(s.name(), Some("empty"));
    assert!(s.is_empty());
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

// ===========================================================================
// Series::nulls
// ===========================================================================

#[test]
fn test_nulls_i32_basic() {
    let s = Series::nulls("x", TypeId::Int32, 5).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 5);
    assert_eq!(s.null_count(), 5);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

#[test]
fn test_nulls_f64() {
    let s = Series::nulls("floats", TypeId::Float64, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

#[test]
fn test_nulls_bool8() {
    let s = Series::nulls("flags", TypeId::Bool8, 4).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.len(), 4);
    assert_eq!(s.null_count(), 4);
}

#[test]
fn test_nulls_string() {
    let s = Series::nulls("names", TypeId::String, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

#[test]
fn test_nulls_zero_length() {
    let s = Series::nulls("empty", TypeId::Int32, 0).unwrap();

    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
}

// ===========================================================================
// Series::from_scalar
// ===========================================================================

#[test]
fn test_from_scalar_i32() {
    let scalar = Scalar::from_value(42i32).unwrap();
    let s = Series::from_scalar("x", &scalar, 4).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 4);
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42, 42]);
}

#[test]
fn test_from_scalar_f64() {
    let scalar = Scalar::from_value(2.5f64).unwrap();
    let s = Series::from_scalar("val", &scalar, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.to_vec_strict::<f64>().unwrap(), vec![2.5, 2.5, 2.5]);
}

#[test]
fn test_from_scalar_string() {
    let scalar = Scalar::from_str("hello").unwrap();
    let s = Series::from_scalar("greet", &scalar, 2).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 2);
    assert_eq!(s.to_strings_strict().unwrap(), vec!["hello", "hello"]);
}

#[test]
fn test_from_scalar_null_produces_all_nulls() {
    let scalar = Scalar::null_numeric::<i32>().unwrap();
    let s = Series::from_scalar("x", &scalar, 3).unwrap();

    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

#[test]
fn test_from_scalar_zero_size() {
    let scalar = Scalar::from_value(1i32).unwrap();
    let s = Series::from_scalar("x", &scalar, 0).unwrap();

    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
}

// ===========================================================================
// Series::empty
// ===========================================================================

#[test]
fn test_empty_i32() {
    let s = Series::empty("x", TypeId::Int32).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert!(s.is_empty());
}

#[test]
fn test_empty_f64() {
    let s = Series::empty("y", TypeId::Float64).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

#[test]
fn test_empty_string() {
    let s = Series::empty("z", TypeId::String).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 0);
}

#[test]
fn test_empty_bool8() {
    let s = Series::empty("flags", TypeId::Bool8).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert!(s.is_empty());
}
