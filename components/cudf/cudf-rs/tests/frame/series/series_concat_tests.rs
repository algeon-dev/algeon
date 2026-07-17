//! Integration tests for `Series::concat`.
//!
//! All tests require a GPU.

use cudf::{ErrorCode, ErrorKind, ExecutionOverrides, Runtime, Series, TypeId};

use crate::common::error_context_value;

// ===========================================================================
// Basic concatenation
// ===========================================================================

#[test]
fn test_concat_two_i32_series() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[3, 4, 5]).unwrap();

    let result = Series::concat(&[&a, &b]).unwrap();
    assert_eq!(result.len(), 5);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_concat_in_uses_borrowed_execution_overrides() {
    let runtime = Runtime::new().unwrap();
    let ctx = ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[3, 4, 5]).unwrap();

    let result = Series::concat_in(&[&a, &b], &ctx).unwrap();

    assert_eq!(result.len(), 5);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_concat_three_f64_series() {
    let a = crate::common::series::<f64>("a", &[1.0]).unwrap();
    let b = crate::common::series::<f64>("b", &[2.0, 3.0]).unwrap();
    let c = crate::common::series::<f64>("c", &[4.0]).unwrap();

    let result = Series::concat(&[&a, &b, &c]).unwrap();

    assert_eq!(result.len(), 4);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_concat_strings() {
    let a = crate::common::series("a", &["hello"]).unwrap();
    let b = crate::common::series("b", &["world"]).unwrap();

    let result = Series::concat(&[&a, &b]).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(result.to_strings_strict().unwrap(), vec!["hello", "world"]);
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn test_concat_single_series() {
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();

    let result = Series::concat(&[&a]).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_concat_with_empty_series() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[]).unwrap();
    let c = crate::common::series::<i32>("c", &[3]).unwrap();

    let result = Series::concat(&[&a, &b, &c]).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_concat_preserves_nulls() {
    let a = crate::common::nullable_series::<i32>("a", &[1, 0], &[true, false]).unwrap();
    let b = crate::common::series::<i32>("b", &[3]).unwrap();

    let result = Series::concat(&[&a, &b]).unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result.null_count(), 1);
}

// ===========================================================================
// Error cases
// ===========================================================================

#[test]
fn test_concat_empty_slice_returns_error() {
    let err = Series::concat(&[]).expect_err("concat of empty slice should fail");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::concat");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("empty_input"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("provide at least one input series"));
}

#[test]
fn test_concat_type_mismatch_returns_error() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series("b", &["hello"]).unwrap();

    let err = Series::concat(&[&a, &b]).expect_err("concat of mismatched types should fail");

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
