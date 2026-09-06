//! Tests for shared high-level group-key validation across DataFrame APIs.

use cudf::{AggSpec, ErrorCode, ErrorKind, GroupByAgg};

use crate::common::{column_from_slice, error_context_value};

fn assert_empty_group_key_contract(err: &cudf::Error, operation: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "resource").as_deref(), Some("group_keys"));
    assert_eq!(error_context_value(err, "reason").as_deref(), Some("empty_group_keys"));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some("provide at least one group key column"));
    assert_eq!(error_context_value(err, "keys_count").as_deref(), Some("0"));
}

fn high_level_group_key_apis_share_empty_key_contract() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let values = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["key".to_string(), "value".to_string()], vec![keys, values])
        .unwrap();

    let groupby_err = df.groupby(Vec::<&str>::new()).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap_err();
    let cumcount_err = df.group_cumcount(Vec::<&str>::new()).unwrap_err();

    assert_empty_group_key_contract(&groupby_err, "groupby::GroupBy::agg");
    assert_empty_group_key_contract(&cumcount_err, "dataframe::group_cumcount");
}

#[test]
fn shared_group_key_validation_tests_cases() {
    crate::common::run_named_cases(&[(
        "high_level_group_key_apis_share_empty_key_contract",
        high_level_group_key_apis_share_empty_key_contract,
    )]);
}
