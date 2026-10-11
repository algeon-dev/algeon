//! Constructor gap-fillers covered here: `nulls`, `from_scalar`, `empty`.

use cudf::{DataType, ErrorCode, ErrorKind, ScalarValue, Series, TypeId};

use crate::common::{empty_series, error_context_value, null_series, scalar, series_opt};

// Runtime constructors create unnamed series.

fn test_runtime_series_from_host_slice() {
    let rt = cudf::Runtime::new().unwrap();
    let s = rt.series(&[10i32, 20, 30]).unwrap();
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

fn test_runtime_null_series_constructors() {
    let rt = cudf::Runtime::new().unwrap();

    let unnamed = rt.nulls(TypeId::Int32, 2).unwrap();
    assert_eq!(unnamed.null_count(), 2);

    let second = rt.nulls(TypeId::Int32, 2).unwrap();
    assert_eq!(second.null_count(), 2);
}

fn test_runtime_scalar_seed_constructor() {
    let rt = cudf::Runtime::new().unwrap();
    let scalar = scalar(42i32).unwrap();

    let unnamed = rt.series_from_scalar(&scalar, 3).unwrap();
    assert_eq!(unnamed.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42]);

    let second = rt.series_from_scalar(&scalar, 2).unwrap();
    assert_eq!(second.to_vec_strict::<i32>().unwrap(), vec![42, 42]);
}

fn test_runtime_optional_series_constructor() {
    let rt = cudf::Runtime::new().unwrap();

    let unnamed = rt.series_opt(&[Some(10i32), None, Some(30)]).unwrap();
    assert_eq!(unnamed.to_optional_vec::<i32>().unwrap(), vec![Some(10), None, Some(30)]);

    let second = rt.series_opt(&[Some(1i32), None, Some(3)]).unwrap();
    assert_eq!(second.to_optional_vec::<i32>().unwrap(), vec![Some(1), None, Some(3)]);
}

fn test_runtime_empty_series_constructor() {
    let rt = cudf::Runtime::new().unwrap();

    let unnamed = rt.empty_series(TypeId::Int32).unwrap();
    assert_eq!(unnamed.dtype().unwrap().id(), TypeId::Int32);
    assert!(unnamed.is_empty());

    let second = rt.empty_series(TypeId::Int32).unwrap();
    assert_eq!(second.dtype().unwrap().id(), TypeId::Int32);
    assert!(second.is_empty());
}

fn test_runtime_sequence_constructor() {
    let rt = cudf::Runtime::new().unwrap();

    let unnamed = rt.sequence(0i32, 2i32, 4).unwrap();
    assert_eq!(unnamed.to_vec_strict::<i32>().unwrap(), vec![0, 2, 4, 6]);

    let second = rt.sequence(1i32, 3i32, 4).unwrap();
    assert_eq!(second.to_vec_strict::<i32>().unwrap(), vec![1, 4, 7, 10]);
}

fn test_runtime_sequence_rejects_size_over_i32_max_with_stable_contract() {
    let rt = cudf::Runtime::new().unwrap();
    let size = (i32::MAX as usize) + 1;

    let err = rt.sequence(0i32, 1i32, size).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "column::from_sequence_with_runtime");
    assert_eq!(error_context_value(&err, "size").as_deref(), Some(size.to_string().as_str()));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("size"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce size to fit within i32::MAX"));
}

fn test_from_optional_i32() {
    let s = series_opt("x", &[Some(10i32), None, Some(30)]).unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_optional_vec::<i32>().unwrap(), vec![Some(10), None, Some(30)]);
}

fn test_from_optional_bool_preserves_false() {
    let s = series_opt("flag", &[Some(false), Some(true), None, Some(false)]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.to_optional_vec::<bool>().unwrap(), vec![Some(false), Some(true), None, Some(false)]);
}

fn test_from_optional_empty() {
    let s = series_opt::<i32>("empty", &[]).unwrap();
    assert!(s.is_empty());
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

fn test_nulls_i32_basic() {
    let s = null_series("x", TypeId::Int32, 5).unwrap();
    assert_eq!(s.len(), 5);
    assert_eq!(s.null_count(), 5);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

fn test_nulls_f64() {
    let s = null_series("floats", TypeId::Float64, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

fn test_nulls_bool8() {
    let s = null_series("flags", TypeId::Bool8, 4).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.len(), 4);
    assert_eq!(s.null_count(), 4);
}

fn test_nulls_string() {
    let s = null_series("names", TypeId::String, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

fn test_nulls_zero_length() {
    let s = null_series("empty", TypeId::Int32, 0).unwrap();

    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
}

fn test_nulls_decimal_preserve_scale() {
    let dtype = DataType::fixed_point(TypeId::Decimal64, -2).unwrap();
    let s = null_series("amount", dtype, 3).unwrap();

    assert_eq!(s.dtype().unwrap(), dtype);
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

fn test_from_scalar_i32() {
    let scalar = scalar(42i32).unwrap();
    let s = Series::from_scalar(&scalar, 4).unwrap();
    assert_eq!(s.len(), 4);
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42, 42]);
}

fn test_from_scalar_f64() {
    let scalar = scalar(2.5f64).unwrap();
    let s = Series::from_scalar(&scalar, 3).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.to_vec_strict::<f64>().unwrap(), vec![2.5, 2.5, 2.5]);
}

fn test_from_scalar_string() {
    let scalar = scalar("hello").unwrap();
    let s = Series::from_scalar(&scalar, 2).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 2);
    assert_eq!(s.to_strings_strict().unwrap(), vec!["hello", "hello"]);
}

fn test_from_scalar_null_produces_all_nulls() {
    let scalar = scalar(ScalarValue::Null(TypeId::Int32.into())).unwrap();
    let s = Series::from_scalar(&scalar, 3).unwrap();

    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 3);
}

fn test_from_scalar_zero_size() {
    let scalar = scalar(1i32).unwrap();
    let s = Series::from_scalar(&scalar, 0).unwrap();

    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
}

fn test_from_scalar_rejects_size_over_i32_max_with_stable_contract() {
    let scalar = scalar(1i32).unwrap();
    let size = (i32::MAX as usize) + 1;

    let err = Series::from_scalar(&scalar, size).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "column::from_scalar");
    assert_eq!(error_context_value(&err, "size").as_deref(), Some(size.to_string().as_str()));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("size"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce size to fit within i32::MAX"));
}

fn test_empty_i32() {
    let s = empty_series("x", TypeId::Int32).unwrap();
    assert_eq!(s.len(), 0);
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert!(s.is_empty());
}

fn test_empty_f64() {
    let s = empty_series("y", TypeId::Float64).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

fn test_empty_string() {
    let s = empty_series("z", TypeId::String).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 0);
}

fn test_empty_bool8() {
    let s = empty_series("flags", TypeId::Bool8).unwrap();

    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert!(s.is_empty());
}

fn test_empty_decimal_preserve_scale() {
    let dtype = DataType::fixed_point(TypeId::Decimal64, -2).unwrap();
    let s = empty_series("amount", dtype).unwrap();

    assert_eq!(s.dtype().unwrap(), dtype);
    assert!(s.is_empty());
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("series_constructor_tests::test_runtime_series_from_host_slice", test_runtime_series_from_host_slice),
    ("series_constructor_tests::test_runtime_null_series_constructors", test_runtime_null_series_constructors),
    ("series_constructor_tests::test_runtime_scalar_seed_constructor", test_runtime_scalar_seed_constructor),
    ("series_constructor_tests::test_runtime_optional_series_constructor", test_runtime_optional_series_constructor),
    ("series_constructor_tests::test_runtime_empty_series_constructor", test_runtime_empty_series_constructor),
    ("series_constructor_tests::test_runtime_sequence_constructor", test_runtime_sequence_constructor),
    (
        "series_constructor_tests::test_runtime_sequence_rejects_size_over_i32_max_with_stable_contract",
        test_runtime_sequence_rejects_size_over_i32_max_with_stable_contract,
    ),
    ("series_constructor_tests::test_from_optional_i32", test_from_optional_i32),
    ("series_constructor_tests::test_from_optional_bool_preserves_false", test_from_optional_bool_preserves_false),
    ("series_constructor_tests::test_from_optional_empty", test_from_optional_empty),
    ("series_constructor_tests::test_nulls_i32_basic", test_nulls_i32_basic),
    ("series_constructor_tests::test_nulls_f64", test_nulls_f64),
    ("series_constructor_tests::test_nulls_bool8", test_nulls_bool8),
    ("series_constructor_tests::test_nulls_string", test_nulls_string),
    ("series_constructor_tests::test_nulls_zero_length", test_nulls_zero_length),
    ("series_constructor_tests::test_nulls_decimal_preserve_scale", test_nulls_decimal_preserve_scale),
    ("series_constructor_tests::test_from_scalar_i32", test_from_scalar_i32),
    ("series_constructor_tests::test_from_scalar_f64", test_from_scalar_f64),
    ("series_constructor_tests::test_from_scalar_string", test_from_scalar_string),
    ("series_constructor_tests::test_from_scalar_null_produces_all_nulls", test_from_scalar_null_produces_all_nulls),
    ("series_constructor_tests::test_from_scalar_zero_size", test_from_scalar_zero_size),
    (
        "series_constructor_tests::test_from_scalar_rejects_size_over_i32_max_with_stable_contract",
        test_from_scalar_rejects_size_over_i32_max_with_stable_contract,
    ),
    ("series_constructor_tests::test_empty_i32", test_empty_i32),
    ("series_constructor_tests::test_empty_f64", test_empty_f64),
    ("series_constructor_tests::test_empty_string", test_empty_string),
    ("series_constructor_tests::test_empty_bool8", test_empty_bool8),
    ("series_constructor_tests::test_empty_decimal_preserve_scale", test_empty_decimal_preserve_scale),
];
