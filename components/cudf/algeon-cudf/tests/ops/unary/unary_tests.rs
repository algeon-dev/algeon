//! Tests for unary operations.

use cudf::{DataType, Exec, TypeId, ops::unary};

use crate::common::{error_context_value, run_named_cases, series};

fn test_is_supported_cast() {
    let from = DataType::new(TypeId::Int32);
    let to = DataType::new(TypeId::Float64);
    assert!(unary::is_supported_cast(&from, &to).unwrap());
}

fn test_cast_error_contains_operation_context() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();

    let err = s.on(Exec::RETAINED).astype(TypeId::Struct).expect_err("expected error");

    assert_eq!(err.operation(), "series::astype");
    assert_eq!(error_context_value(&err, "input_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "input_type_scale").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "output_type_id").as_deref(), Some("struct"));
    assert_eq!(error_context_value(&err, "output_type_scale").as_deref(), Some("0"));
}

fn test_logical_not_empty() {
    let s = series::<bool>("x", &[]).unwrap();

    let result = s.on(Exec::RETAINED).logical_not().unwrap();

    assert!(result.is_empty());
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
}

#[test]
fn unary_tests_cases() {
    run_named_cases(&[
        ("test_is_supported_cast", test_is_supported_cast),
        ("test_cast_error_contains_operation_context", test_cast_error_contains_operation_context),
        ("test_logical_not_empty", test_logical_not_empty),
    ]);
}
