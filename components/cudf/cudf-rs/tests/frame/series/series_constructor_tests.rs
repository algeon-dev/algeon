//! Integration tests for `Series` constructor gap-fillers:
//! `nulls`, `from_scalar`, `empty`.
//!
//! All tests require a GPU.

use cudf::{Series, TypeId};

use crate::common::scalar;

// ===========================================================================
// crate::common::series_opt
// ===========================================================================

#[test]
fn test_from_optional_i32() {
    let s = crate::common::series_opt("x", &[Some(10i32), None, Some(30)]).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_optional_vec::<i32>().unwrap(), vec![Some(10), None, Some(30)]);
}

#[test]
fn test_from_optional_bool_preserves_false() {
    let s = crate::common::series_opt("flag", &[Some(false), Some(true), None, Some(false)]).unwrap();

    assert_eq!(s.name(), Some("flag"));
    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.to_optional_vec::<bool>().unwrap(), vec![Some(false), Some(true), None, Some(false)]);
}

#[test]
fn test_from_optional_empty() {
    let s = crate::common::series_opt::<i32>("empty", &[]).unwrap();

    assert_eq!(s.name(), Some("empty"));
    assert!(s.is_empty());
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

// ===========================================================================
// crate::common::null_series
// ===========================================================================

#[test]
fn test_nulls_i32_basic() {
    let s = crate::common::null_series("x", TypeId::Int32, 5).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 5);
    assert_eq!(s.null_count(), 5);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

#[test]
fn test_nulls_f64() {
    let s = crate::common::null_series("floats", TypeId::Float64, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

#[test]
fn test_nulls_bool8() {
    let s = crate::common::null_series("flags", TypeId::Bool8, 4).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.len(), 4);
    assert_eq!(s.null_count(), 4);
}

#[test]
fn test_nulls_string() {
    let s = crate::common::null_series("names", TypeId::String, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

#[test]
fn test_nulls_zero_length() {
    let s = crate::common::null_series("empty", TypeId::Int32, 0).unwrap();

    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
}

// ===========================================================================
// Series::from_scalar
// ===========================================================================

#[test]
fn test_from_scalar_i32() {
    let scalar = scalar(42i32).unwrap();
    let s = Series::from_scalar("x", &scalar, 4).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 4);
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42, 42]);
}

#[test]
fn test_from_scalar_f64() {
    let scalar = scalar(2.5f64).unwrap();
    let s = Series::from_scalar("val", &scalar, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.to_vec_strict::<f64>().unwrap(), vec![2.5, 2.5, 2.5]);
}

#[test]
fn test_from_scalar_string() {
    let scalar = crate::common::string_scalar("hello").unwrap();
    let s = Series::from_scalar("greet", &scalar, 2).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 2);
    assert_eq!(s.to_strings_strict().unwrap(), vec!["hello", "hello"]);
}

#[test]
fn test_from_scalar_null_produces_all_nulls() {
    let scalar = crate::common::null_scalar::<i32>().unwrap();
    let s = Series::from_scalar("x", &scalar, 3).unwrap();

    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

#[test]
fn test_from_scalar_zero_size() {
    let scalar = scalar(1i32).unwrap();
    let s = Series::from_scalar("x", &scalar, 0).unwrap();

    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
}

// ===========================================================================
// crate::common::empty_series
// ===========================================================================

#[test]
fn test_empty_i32() {
    let s = crate::common::empty_series("x", TypeId::Int32).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert!(s.is_empty());
}

#[test]
fn test_empty_f64() {
    let s = crate::common::empty_series("y", TypeId::Float64).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

#[test]
fn test_empty_string() {
    let s = crate::common::empty_series("z", TypeId::String).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 0);
}

#[test]
fn test_empty_bool8() {
    let s = crate::common::empty_series("flags", TypeId::Bool8).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert!(s.is_empty());
}
