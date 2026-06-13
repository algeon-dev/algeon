//! Tests for sorted per-group cumcount workflows on DataFrames.

use cudf::{DataFrame, ErrorCode, ErrorKind, SortOptions};

use crate::common::{error_context_value, sample_runtime};

fn assert_group_cumcount_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason"), Some(reason));
    assert_eq!(error_context_value(err, "remedy"), Some(remedy));
}

#[test]
fn group_cumcount_returns_zero_based_positions_for_each_sorted_group() {
    let runtime = sample_runtime();
    let edges = DataFrame::from_columns([
        ("_sp", runtime.series(&[2i32, 1, 1, 2, 1, 3]).unwrap()),
        ("weight", runtime.series(&[5.0f32, 9.0, 8.0, 4.0, 7.0, 3.0]).unwrap()),
        ("id_fc", runtime.series(&[5i64, 2, 3, 6, 4, 7]).unwrap()),
    ])
    .unwrap();

    let sorted = edges
        .sort_values(
            ["_sp", "weight", "id_fc"],
            SortOptions::new().with_descending(vec![false, true, false]).with_stable(true),
        )
        .unwrap();

    let rank = sorted.group_cumcount(["_sp"]).unwrap();
    assert_eq!(rank.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 0, 1, 0]);
}

#[test]
fn group_cumcount_supports_multi_column_group_keys() {
    let runtime = sample_runtime();
    let df = DataFrame::from_columns([
        ("partition", runtime.series(&[1i32, 1, 1, 2, 2, 2]).unwrap()),
        ("entityType", runtime.series(&["device", "device", "identity", "device", "device", "identity"]).unwrap()),
        ("score", runtime.series(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0]).unwrap()),
    ])
    .unwrap();

    let rank = df.group_cumcount(["partition", "entityType"]).unwrap();

    assert_eq!(rank.to_vec_strict::<i32>().unwrap(), vec![0, 1, 0, 0, 1, 0]);
}

#[test]
fn group_cumcount_rejects_empty_group_keys_with_stable_contract() {
    let runtime = sample_runtime();
    let df = DataFrame::from_columns([
        ("partition", runtime.series(&[1i32, 1, 2]).unwrap()),
        ("score", runtime.series(&[9.0f32, 8.0, 7.0]).unwrap()),
    ])
    .unwrap();

    let err = df.group_cumcount(Vec::<&str>::new()).unwrap_err();

    assert_group_cumcount_validation_error(
        &err,
        "dataframe::group_cumcount",
        "empty_group_keys",
        "provide at least one group key column",
    );
}
