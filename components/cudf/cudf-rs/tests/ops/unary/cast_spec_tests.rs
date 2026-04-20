//! Specification tests for type casting (`astype` / `cast`).
//!
//! These tests systematically cover cast behavior across type pairs,
//! including edge cases that are not covered by the basic unary tests.
//!
//! All tests require a GPU.

use cudf::{DataType, TimestampResolution, TypeId};

// ===========================================================================
// Integer width widening (lossless)
// ===========================================================================

#[test]
fn test_cast_i8_to_i16() {
    let s = crate::common::series::<i8>("x", &[i8::MIN, 0, i8::MAX]).unwrap();
    let result = s.astype(TypeId::Int16).unwrap();
    let vals = result.to_vec_strict::<i16>().unwrap();
    assert_eq!(vals, vec![i8::MIN as i16, 0, i8::MAX as i16]);
}

#[test]
fn test_cast_i16_to_i32() {
    let s = crate::common::series::<i16>("x", &[i16::MIN, 0, i16::MAX]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![i16::MIN as i32, 0, i16::MAX as i32]);
}

#[test]
fn test_cast_i32_to_i64() {
    let s = crate::common::series::<i32>("x", &[i32::MIN, 0, i32::MAX]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![i32::MIN as i64, 0, i32::MAX as i64]);
}

// ===========================================================================
// Float ↔ Integer
// ===========================================================================

#[test]
fn test_cast_i32_to_f32() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.astype(TypeId::Float32).unwrap();
    let vals = result.to_vec_strict::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_cast_f32_to_i32_truncates() {
    let s = crate::common::series::<f32>("x", &[1.9, -2.7, 0.1]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, -2, 0]);
}

#[test]
fn test_cast_f64_to_f32() {
    let s = crate::common::series::<f64>("x", &[1.5, 2.5, 3.5]).unwrap();
    let result = s.astype(TypeId::Float32).unwrap();
    let vals = result.to_vec_strict::<f32>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

// ===========================================================================
// Bool ↔ Integer
// ===========================================================================

#[test]
fn test_cast_bool_to_i32() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 0, 1]);
}

#[test]
fn test_cast_i32_to_bool() {
    let s = crate::common::series::<i32>("x", &[0, 1, 42, -1]).unwrap();
    let result = s.astype(TypeId::Bool8).unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![false, true, true, true]);
}

#[test]
fn test_cast_bool_to_i64() {
    let s = crate::common::series::<bool>("x", &[false, true]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![0, 1]);
}

// ===========================================================================
// Unsigned ↔ Signed
// ===========================================================================

#[test]
fn test_cast_u8_to_i16() {
    let s = crate::common::series::<u8>("x", &[0, 127, 255]).unwrap();
    let result = s.astype(TypeId::Int16).unwrap();
    let vals = result.to_vec_strict::<i16>().unwrap();
    assert_eq!(vals, vec![0, 127, 255]);
}

#[test]
fn test_cast_u32_to_i64() {
    let s = crate::common::series::<u32>("x", &[0, u32::MAX]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![0, u32::MAX as i64]);
}

// ===========================================================================
// Null preservation through cast
// ===========================================================================

#[test]
fn test_cast_preserves_null_mask() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    assert_eq!(result.null_count(), 1);
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(10.0), None, Some(30.0)]);
}

#[test]
fn test_cast_all_null_column() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    assert_eq!(result.null_count(), 2);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![None, None]);
}

// ===========================================================================
// Empty column cast
// ===========================================================================

#[test]
fn test_cast_empty_column() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
}

// ===========================================================================
// Same-type cast (noop)
// ===========================================================================

#[test]
fn test_cast_same_type() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// Series display omits schema names
// ===========================================================================

#[test]
fn test_cast_display_omits_input_name() {
    let s = crate::common::series::<i32>("my_column", &[1, 2]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_column"), "cast output should not expose schema names: {}", output);
}

// ===========================================================================
// is_supported_cast
// ===========================================================================

#[test]
fn test_is_supported_cast_numeric_to_float() {
    let from = DataType::new(TypeId::Int32);
    let to = DataType::new(TypeId::Float64);
    assert!(cudf::ops::unary::is_supported_cast(&from, &to).unwrap());
}

#[test]
fn test_is_supported_cast_float_to_int() {
    let from = DataType::new(TypeId::Float64);
    let to = DataType::new(TypeId::Int32);
    assert!(cudf::ops::unary::is_supported_cast(&from, &to).unwrap());
}

#[test]
fn test_is_supported_cast_bool_to_int() {
    let from = DataType::new(TypeId::Bool8);
    let to = DataType::new(TypeId::Int32);
    assert!(cudf::ops::unary::is_supported_cast(&from, &to).unwrap());
}

// ===========================================================================
// Integer-to-timestamp cast (supported by libcudf)
// ===========================================================================

#[test]
fn test_cast_i64_to_timestamp_microseconds() {
    // libcudf requires int64 → duration → timestamp (direct int→timestamp is not supported).
    let s = crate::common::series::<i64>("x", &[0, 1_000_000, 1_618_000_000_000_000]).unwrap();
    let dur = s.astype(TypeId::DurationMicroseconds).unwrap();
    let result = dur.astype(TypeId::TimestampMicroseconds).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_cast_timestamp_to_i64() {
    // Round-trip: i64 -> duration_us -> timestamp_us -> duration_us -> i64
    let s = crate::common::series::<i64>("x", &[0, 1000, 2000]).unwrap();
    let dur = s.astype(TypeId::DurationMicroseconds).unwrap();
    let ts = dur.astype(TypeId::TimestampMicroseconds).unwrap();
    let back_dur = ts.astype(TypeId::DurationMicroseconds).unwrap();
    let back = back_dur.astype(TypeId::Int64).unwrap();
    assert_eq!(back.to_vec_strict::<i64>().unwrap(), vec![0, 1000, 2000]);
}

// ===========================================================================
// Convenience method: to_datetime
// ===========================================================================

#[test]
fn test_to_datetime_default_microseconds() {
    let s = crate::common::series::<i64>("ts", &[0, 1_000_000]).unwrap();
    let result = s.to_datetime(None).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
}

#[test]
fn test_to_datetime_explicit_milliseconds() {
    let s = crate::common::series::<i64>("ts", &[0, 1000]).unwrap();
    let result = s.to_datetime(Some(TimestampResolution::Milliseconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMilliseconds);
}

#[test]
fn test_to_datetime_nanoseconds() {
    let s = crate::common::series::<i64>("ts", &[0, 1_000_000_000]).unwrap();
    let result = s.to_datetime(Some(TimestampResolution::Nanoseconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampNanoseconds);
}

#[test]
fn test_to_datetime_seconds() {
    let s = crate::common::series::<i64>("ts", &[0, 86400]).unwrap();
    let result = s.to_datetime(Some(TimestampResolution::Seconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampSeconds);
}

#[test]
fn test_to_datetime_round_trip() {
    let s = crate::common::series::<i64>("ts", &[0, 1000, 2000]).unwrap();
    let dt = s.to_datetime(Some(TimestampResolution::Microseconds)).unwrap();
    // libcudf requires timestamp → duration → int64 for the reverse path.
    let dur = dt.astype(TypeId::DurationMicroseconds).unwrap();
    let back = dur.astype(TypeId::Int64).unwrap();
    assert_eq!(back.to_vec_strict::<i64>().unwrap(), vec![0, 1000, 2000]);
}

#[test]
fn test_to_datetime_preserves_nulls() {
    let s = crate::common::nullable_series::<i64>("ts", &[100, 0, 300], &[true, false, true]).unwrap();
    let result = s.to_datetime(None).unwrap();
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
}

// ===========================================================================
// Convenience method: to_numeric
// ===========================================================================

#[test]
fn test_to_numeric_f64_to_i64() {
    let s = crate::common::series::<f64>("x", &[1.9, 2.0, 3.1]).unwrap();
    let result = s.to_numeric(TypeId::Int64).unwrap();
    assert_eq!(result.to_vec_strict::<i64>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_to_numeric_i32_to_f64() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.to_numeric(TypeId::Float64).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_numeric_bool_to_i32() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.to_numeric(TypeId::Int32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 0, 1]);
}

#[test]
fn test_to_numeric_rejects_non_numeric_target() {
    let s = crate::common::series::<i32>("x", &[1, 2]).unwrap();
    let err = s.to_numeric(TypeId::String).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::NotSupported);
}

#[test]
fn test_to_numeric_rejects_timestamp_target() {
    let s = crate::common::series::<i32>("x", &[1, 2]).unwrap();
    let err = s.to_numeric(TypeId::TimestampMicroseconds).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::NotSupported);
}
