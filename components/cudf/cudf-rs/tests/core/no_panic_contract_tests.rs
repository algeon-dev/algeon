use std::panic;

use cudf::{DataFrame, ErrorCode, ErrorKind, Series};

use crate::common::{column_from_slice, error_context_value, string_scalar};

#[test]
fn dataframe_from_series_empty_iterator_returns_error_instead_of_panicking() {
    let result = panic::catch_unwind(|| DataFrame::from_series(std::iter::empty::<(&str, Series)>()));

    let err = result.expect("from_series should not panic").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "dataframe::from_series");
}

#[test]
fn table_view_select_columns_out_of_bounds_returns_error_instead_of_panicking() {
    let table = cudf::core::Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let view = table.testing_view().unwrap();

    let result = panic::catch_unwind(|| view.select_columns(&[1]));

    let err = result.expect("select_columns should not panic").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "table_view::select_columns");
}

#[test]
fn scalar_type_mismatch_returns_error_instead_of_panicking() {
    let scalar = string_scalar("alice").unwrap();

    let result = panic::catch_unwind(|| scalar.value::<i32>());

    let err = result.expect("scalar::value should not panic on type mismatch").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
}
