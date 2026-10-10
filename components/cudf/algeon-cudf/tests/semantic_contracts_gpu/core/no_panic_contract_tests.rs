use std::{
    iter,
    panic::{self, AssertUnwindSafe},
};

use cudf::{DataFrame, ErrorCode, ErrorKind, Series, core::Table, testing::BorrowedViewTestingExt};

use crate::common::{column_from_slice, error_context_value, run_named_cases, scalar};

fn dataframe_from_series_empty_iterator_returns_error_instead_of_panicking() {
    let result = panic::catch_unwind(|| DataFrame::from_series(iter::empty::<(&str, Series)>()));

    let err = result.expect("from_series should not panic").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "dataframe::from_series");
}

fn table_view_select_columns_out_of_bounds_returns_error_instead_of_panicking() {
    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let view = table.testing_view().unwrap();

    let result = panic::catch_unwind(AssertUnwindSafe(|| view.select_columns(&[1])));

    let err = result.expect("select_columns should not panic").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "table_view::select_columns");
}

fn scalar_type_mismatch_returns_error_instead_of_panicking() {
    let scalar = scalar("alice").unwrap();

    let result = panic::catch_unwind(AssertUnwindSafe(|| scalar.value::<i32>()));

    let err = result.expect("scalar::value should not panic on type mismatch").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
}

#[test]
fn no_panic_contract_tests_cases() {
    run_named_cases(&[
        (
            "dataframe_from_series_empty_iterator_returns_error_instead_of_panicking",
            dataframe_from_series_empty_iterator_returns_error_instead_of_panicking,
        ),
        (
            "table_view_select_columns_out_of_bounds_returns_error_instead_of_panicking",
            table_view_select_columns_out_of_bounds_returns_error_instead_of_panicking,
        ),
        (
            "scalar_type_mismatch_returns_error_instead_of_panicking",
            scalar_type_mismatch_returns_error_instead_of_panicking,
        ),
    ]);
}
