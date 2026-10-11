//! Tests for sorted per-group cumcount workflows on DataFrames.

use cudf::{DataFrame, ErrorCode, ErrorKind, Exec, SortOptions};

use crate::common::{error_context_value, sample_runtime};

fn assert_group_cumcount_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

fn group_cumcount_returns_zero_based_positions_for_each_sorted_group() {
    let runtime = sample_runtime();
    let edges = DataFrame::from_series([
        ("_sp", runtime.series(&[2i32, 1, 1, 2, 1, 3]).unwrap()),
        ("weight", runtime.series(&[5.0f32, 9.0, 8.0, 4.0, 7.0, 3.0]).unwrap()),
        ("id_fc", runtime.series(&[5i64, 2, 3, 6, 4, 7]).unwrap()),
    ])
    .unwrap();

    let sorted = edges
        .on(Exec::RETAINED)
        .sort_values(
            ["_sp", "weight", "id_fc"],
            SortOptions::new().with_descending(vec![false, true, false]).with_stable(true),
        )
        .unwrap();

    let rank = sorted.on(Exec::RETAINED).group_cumcount(["_sp"]).unwrap();
    assert_eq!(rank.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 0, 1, 0]);
}

fn group_cumcount_supports_multi_column_group_keys() {
    let runtime = sample_runtime();
    let df = DataFrame::from_series([
        ("partition", runtime.series(&[1i32, 1, 1, 2, 2, 2]).unwrap()),
        ("entityType", runtime.series(&["device", "device", "identity", "device", "device", "identity"]).unwrap()),
        ("score", runtime.series(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0]).unwrap()),
    ])
    .unwrap();

    let rank = df.on(Exec::RETAINED).group_cumcount(["partition", "entityType"]).unwrap();

    assert_eq!(rank.to_vec_strict::<i32>().unwrap(), vec![0, 1, 0, 0, 1, 0]);
}

fn group_cumcount_treats_repeated_key_names_as_one_group_key() {
    let runtime = sample_runtime();
    let df = DataFrame::from_series([
        ("partition", runtime.series(&[1i32, 1, 1, 2, 2, 3]).unwrap()),
        ("tier", runtime.series(&["a", "a", "b", "b", "b", "b"]).unwrap()),
        ("score", runtime.series(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0]).unwrap()),
    ])
    .unwrap();

    let once = df.on(Exec::RETAINED).group_cumcount(["partition", "tier"]).unwrap();
    let repeated = df.on(Exec::RETAINED).group_cumcount(["partition", "tier", "partition", "partition"]).unwrap();
    let single_repeated = df.on(Exec::RETAINED).group_cumcount(["partition", "partition"]).unwrap();

    assert_eq!(once.to_vec_strict::<i32>().unwrap(), vec![0, 1, 0, 0, 1, 0]);
    assert_eq!(repeated.to_vec_strict::<i32>().unwrap(), vec![0, 1, 0, 0, 1, 0]);
    assert_eq!(single_repeated.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 0, 1, 0]);
}

fn group_cumcount_rejects_empty_group_keys_with_stable_contract() {
    let runtime = sample_runtime();
    let df = DataFrame::from_series([
        ("partition", runtime.series(&[1i32, 1, 2]).unwrap()),
        ("score", runtime.series(&[9.0f32, 8.0, 7.0]).unwrap()),
    ])
    .unwrap();

    let err = df.on(Exec::RETAINED).group_cumcount(Vec::<&str>::new()).unwrap_err();

    assert_group_cumcount_validation_error(
        &err,
        "dataframe::group_cumcount",
        "empty_group_keys",
        "provide at least one group key column",
    );
}

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "dataframe_group_cumcount_tests::group_cumcount_returns_zero_based_positions_for_each_sorted_group",
        group_cumcount_returns_zero_based_positions_for_each_sorted_group,
    ),
    (
        "dataframe_group_cumcount_tests::group_cumcount_supports_multi_column_group_keys",
        group_cumcount_supports_multi_column_group_keys,
    ),
    (
        "dataframe_group_cumcount_tests::group_cumcount_treats_repeated_key_names_as_one_group_key",
        group_cumcount_treats_repeated_key_names_as_one_group_key,
    ),
    (
        "dataframe_group_cumcount_tests::group_cumcount_rejects_empty_group_keys_with_stable_contract",
        group_cumcount_rejects_empty_group_keys_with_stable_contract,
    ),
];
