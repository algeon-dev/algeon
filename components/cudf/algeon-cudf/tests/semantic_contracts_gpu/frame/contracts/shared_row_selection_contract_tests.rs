use cudf::{
    Error, ErrorCode, ErrorKind, Exec,
    core::{
        error::{FactKeyName, FactValue},
        types::TypeId,
    },
    testing::{
        dataframe::{
            trigger_dataframe_row_range_end_limit_error_for_test, trigger_dataframe_row_range_overflow_error_for_test,
            trigger_dataframe_row_range_start_limit_error_for_test,
        },
        series::{
            trigger_series_row_range_end_limit_error_for_test, trigger_series_row_range_overflow_error_for_test,
            trigger_series_row_range_start_limit_error_for_test,
        },
    },
};

use crate::common::{sample_df, series};

fn assert_same_row_range_contract(lhs: &Error, rhs: &Error) {
    assert_eq!(lhs.kind(), rhs.kind());
    assert_eq!(lhs.code(), rhs.code());
    assert_eq!(lhs.message(), rhs.message());
    assert_eq!(lhs.detail_value("resource"), rhs.detail_value("resource"));
    assert_eq!(lhs.detail_value("reason"), rhs.detail_value("reason"));
    assert_eq!(lhs.detail_value("remedy"), rhs.detail_value("remedy"));
    assert_eq!(lhs.detail_value("start_row"), rhs.detail_value("start_row"));
    assert_eq!(lhs.detail_value("end_row"), rhs.detail_value("end_row"));
    assert_eq!(lhs.detail_value("length"), rhs.detail_value("length"));
}

fn dataframe_and_series_filter_dtype_validation_use_same_canonical_contract() {
    let df = sample_df();
    let values = df.series("id").unwrap();
    let bad_mask = series::<i32>("mask", &[1, 0, 1, 0, 1]).unwrap();

    let dataframe_err = df.on(Exec::RETAINED).filter(&bad_mask).unwrap_err();
    let series_err = values.on(Exec::RETAINED).filter(&bad_mask).unwrap_err();

    assert_eq!(dataframe_err.kind(), ErrorKind::InvalidInput);
    assert_eq!(series_err.kind(), ErrorKind::InvalidInput);
    assert_eq!(dataframe_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(series_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(dataframe_err.message(), series_err.message());
    assert_eq!(dataframe_err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(series_err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(dataframe_err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Bool8)));
    assert_eq!(series_err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Bool8)));
}

fn dataframe_and_series_filter_length_validation_use_same_canonical_facts() {
    let df = sample_df();
    let values = df.series("id").unwrap();
    let short_mask = series::<bool>("mask", &[true, false]).unwrap();

    let dataframe_err = df.on(Exec::RETAINED).filter(&short_mask).unwrap_err();
    let series_err = values.on(Exec::RETAINED).filter(&short_mask).unwrap_err();

    assert_eq!(dataframe_err.kind(), ErrorKind::InvalidInput);
    assert_eq!(series_err.kind(), ErrorKind::InvalidInput);
    assert_eq!(dataframe_err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(series_err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(dataframe_err.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(2)));
    assert_eq!(series_err.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(2)));
    assert_eq!(dataframe_err.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(5)));
    assert_eq!(series_err.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(5)));
}

fn dataframe_and_series_row_range_testing_hooks_share_same_error_shape() {
    let dataframe_start = trigger_dataframe_row_range_start_limit_error_for_test().unwrap_err();
    let series_start = trigger_series_row_range_start_limit_error_for_test().unwrap_err();
    assert_same_row_range_contract(&dataframe_start, &series_start);

    let dataframe_end = trigger_dataframe_row_range_end_limit_error_for_test().unwrap_err();
    let series_end = trigger_series_row_range_end_limit_error_for_test().unwrap_err();
    assert_same_row_range_contract(&dataframe_end, &series_end);

    let dataframe_overflow = trigger_dataframe_row_range_overflow_error_for_test().unwrap_err();
    let series_overflow = trigger_series_row_range_overflow_error_for_test().unwrap_err();
    assert_same_row_range_contract(&dataframe_overflow, &series_overflow);
}

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "shared_row_selection_contract_tests::dataframe_and_series_filter_dtype_validation_use_same_canonical_contract",
        dataframe_and_series_filter_dtype_validation_use_same_canonical_contract,
    ),
    (
        "shared_row_selection_contract_tests::dataframe_and_series_filter_length_validation_use_same_canonical_facts",
        dataframe_and_series_filter_length_validation_use_same_canonical_facts,
    ),
    (
        "shared_row_selection_contract_tests::dataframe_and_series_row_range_testing_hooks_share_same_error_shape",
        dataframe_and_series_row_range_testing_hooks_share_same_error_shape,
    ),
];
