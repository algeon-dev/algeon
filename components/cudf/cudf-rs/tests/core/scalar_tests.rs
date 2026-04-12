use cudf::TypeId;

use crate::common::scalar;

#[test]
fn test_string_scalar_value_round_trip() {
    let scalar = crate::common::string_scalar("hello").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "hello");
}

#[test]
fn test_empty_string_scalar_value_round_trip() {
    let scalar = crate::common::string_scalar("").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "");
}

#[test]
fn test_i32_scalar_round_trip() {
    let s = scalar(42i32).unwrap();
    assert_eq!(s.value::<i32>().unwrap(), 42);
}

#[test]
fn test_i64_scalar_round_trip() {
    let s = scalar(123i64).unwrap();
    assert_eq!(s.value::<i64>().unwrap(), 123);
}

#[test]
fn test_f32_scalar_round_trip() {
    let s = scalar(3.25f32).unwrap();
    let v = s.value::<f32>().unwrap();
    assert!((v - 3.25f32).abs() < 1e-6);
}

#[test]
fn test_f64_scalar_round_trip() {
    let s = scalar(2.75f64).unwrap();
    let v = s.value::<f64>().unwrap();
    assert!((v - 2.75f64).abs() < 1e-12);
}

#[test]
fn test_null_numeric_is_not_valid() {
    let s = crate::common::null_scalar::<i32>().unwrap();
    assert!(!s.is_valid().unwrap());
}

#[test]
fn test_null_string_is_not_valid() {
    let s = crate::common::null_string_scalar().unwrap();
    assert!(!s.is_valid().unwrap());
}

#[test]
fn test_valid_numeric_is_valid() {
    let s = scalar(42i32).unwrap();
    assert!(s.is_valid().unwrap());
}

#[test]
fn test_valid_string_is_valid() {
    let s = crate::common::string_scalar("hello").unwrap();
    assert!(s.is_valid().unwrap());
}

#[test]
fn test_numeric_scalar_data_type() {
    let s = scalar(42i32).unwrap();
    assert_eq!(s.data_type().unwrap().id(), TypeId::Int32);
}

#[test]
fn test_string_scalar_data_type() {
    let s = crate::common::string_scalar("x").unwrap();
    assert_eq!(s.data_type().unwrap().id(), TypeId::String);
}

#[test]
fn test_type_mismatch_value_extraction() {
    let s = scalar(42i32).unwrap();
    assert!(s.value::<f64>().is_err());
}

#[test]
fn test_string_scalar_numeric_value_fails() {
    let s = crate::common::string_scalar("x").unwrap();
    assert!(s.value::<i32>().is_err());
}
