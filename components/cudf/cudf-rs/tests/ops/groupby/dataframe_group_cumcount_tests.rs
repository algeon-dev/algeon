//! Tests for sorted per-group cumcount workflows on DataFrames.

use cudf::{DataFrame, SortOptions};

use crate::common::sample_runtime;

#[test]
fn group_cumcount_returns_zero_based_positions_for_each_sorted_group() {
    let runtime = sample_runtime();
    let edges = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([
            runtime.series("_sp", &[2i32, 1, 1, 2, 1, 3]).unwrap(),
            runtime.series("weight", &[5.0f32, 9.0, 8.0, 4.0, 7.0, 3.0]).unwrap(),
            runtime.series("id_fc", &[5i64, 2, 3, 6, 4, 7]).unwrap(),
        ])
        .map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();

    let sorted = edges
        .sort_values(
            ["_sp", "weight", "id_fc"],
            SortOptions::new().with_descending(vec![false, true, false]).with_stable(true),
        )
        .unwrap();

    let rank = sorted.group_cumcount(["_sp"], "_rank").unwrap();

    assert_eq!(rank.name(), Some("_rank"));
    assert_eq!(rank.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 0, 1, 0]);
}

#[test]
fn group_cumcount_supports_multi_column_group_keys() {
    let runtime = sample_runtime();
    let df = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([
            runtime.series("partition", &[1i32, 1, 1, 2, 2, 2]).unwrap(),
            runtime.series("entityType", &["device", "device", "identity", "device", "device", "identity"]).unwrap(),
            runtime.series("weight", &[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0]).unwrap(),
        ])
        .map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();

    let rank = df.group_cumcount(["partition", "entityType"], "_rank").unwrap();

    assert_eq!(rank.to_vec_strict::<i32>().unwrap(), vec![0, 1, 0, 0, 1, 0]);
}
