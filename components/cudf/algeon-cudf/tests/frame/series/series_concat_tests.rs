//! Integration tests for `Series::concat`.
//!
//! All tests require a GPU.

use cudf::{ErrorCode, ErrorKind, Exec, Runtime, Series, TypeId};

use crate::common::{error_context_value, nullable_series, run_named_cases, series};

// ===========================================================================
// Basic concatenation
// ===========================================================================

fn test_concat_two_i32_series() {
    let a = series::<i32>("a", &[1, 2]).unwrap();
    let b = series::<i32>("b", &[3, 4, 5]).unwrap();

    let result = Series::concat(&[&a, &b], Exec::RETAINED).unwrap();
    assert_eq!(result.len(), 5);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
}

fn test_concat_in_uses_borrowed_execution_overrides() {
    let runtime = Runtime::new().unwrap();
    let ctx = runtime.exec();
    let a = series::<i32>("a", &[1, 2]).unwrap();
    let b = series::<i32>("b", &[3, 4, 5]).unwrap();

    let result = Series::concat(&[&a, &b], ctx).unwrap();

    assert_eq!(result.len(), 5);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
}

fn test_concat_three_f64_series() {
    let a = series::<f64>("a", &[1.0]).unwrap();
    let b = series::<f64>("b", &[2.0, 3.0]).unwrap();
    let c = series::<f64>("c", &[4.0]).unwrap();

    let result = Series::concat(&[&a, &b, &c], Exec::RETAINED).unwrap();

    assert_eq!(result.len(), 4);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

fn test_concat_strings() {
    let a = series("a", &["hello"]).unwrap();
    let b = series("b", &["world"]).unwrap();

    let result = Series::concat(&[&a, &b], Exec::RETAINED).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(result.to_strings_strict().unwrap(), vec!["hello", "world"]);
}

// ===========================================================================
// Edge cases
// ===========================================================================

fn test_concat_single_series() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();

    let result = Series::concat(&[&a], Exec::RETAINED).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_concat_with_empty_series() {
    let a = series::<i32>("a", &[1, 2]).unwrap();
    let b = series::<i32>("b", &[]).unwrap();
    let c = series::<i32>("c", &[3]).unwrap();

    let result = Series::concat(&[&a, &b, &c], Exec::RETAINED).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_concat_preserves_nulls() {
    let a = nullable_series::<i32>("a", &[1, 0], &[true, false]).unwrap();
    let b = series::<i32>("b", &[3]).unwrap();

    let result = Series::concat(&[&a, &b], Exec::RETAINED).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.null_count(), 1);
}

// ===========================================================================
// Error cases
// ===========================================================================

fn test_concat_empty_slice_returns_error() {
    let err = Series::concat(&[], Exec::RETAINED).expect_err("concat of empty slice should fail");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::concat");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("empty_input"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("provide at least one input series"));
}

fn test_concat_type_mismatch_returns_error() {
    let a = series::<i32>("a", &[1, 2]).unwrap();
    let b = series("b", &["hello"]).unwrap();

    let err = Series::concat(&[&a, &b], Exec::RETAINED).expect_err("concat of mismatched types should fail");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "series::concat");
    assert_eq!(error_context_value(&err, "column_index").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("string"));
    assert!(
        err.call_chain().contains(&"concatenate::concatenate_columns"),
        "expected concatenate::concatenate_columns in call_chain, got {:?}",
        err.call_chain()
    );
}

#[test]
fn series_concat_tests_cases() {
    run_named_cases(&[
        ("test_concat_two_i32_series", test_concat_two_i32_series),
        ("test_concat_in_uses_borrowed_execution_overrides", test_concat_in_uses_borrowed_execution_overrides),
        ("test_concat_three_f64_series", test_concat_three_f64_series),
        ("test_concat_strings", test_concat_strings),
        ("test_concat_single_series", test_concat_single_series),
        ("test_concat_with_empty_series", test_concat_with_empty_series),
        ("test_concat_preserves_nulls", test_concat_preserves_nulls),
        ("test_concat_empty_slice_returns_error", test_concat_empty_slice_returns_error),
        ("test_concat_type_mismatch_returns_error", test_concat_type_mismatch_returns_error),
    ]);
}
