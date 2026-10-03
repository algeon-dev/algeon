//! Tests for sorted per-group cumulative scan workflows on DataFrames.

use cudf::{DataFrame, Exec, SortOptions, ops::reduction::ScanAgg};

use crate::common::{run_named_cases, sample_runtime};

fn group_scan_sorted_returns_row_aligned_cumulative_sum_per_group() {
    let runtime = sample_runtime();
    let df = DataFrame::from_series([
        ("key", runtime.series(&[2i32, 1, 1, 2, 1, 2]).unwrap()),
        ("ord", runtime.series(&[1i32, 1, 3, 2, 2, 3]).unwrap()),
        ("value", runtime.nullable_series(&[7i64, 10, 5, 0, 0, 3], &[true, true, true, false, false, true]).unwrap()),
    ])
    .unwrap();
    let sorted = df.on(Exec::RETAINED).sort_values(["key", "ord"], SortOptions::new().with_stable(true)).unwrap();

    let scanned = sorted.on(Exec::RETAINED).group_scan_sorted(["key"], "value", ScanAgg::Sum).unwrap();

    assert_eq!(
        sorted.series("value").unwrap().to_optional_vec::<i64>().unwrap(),
        vec![Some(10), None, Some(5), Some(7), None, Some(3)]
    );
    assert_eq!(scanned.to_optional_vec::<i64>().unwrap(), vec![Some(10), None, Some(15), Some(7), None, Some(10)]);
}

fn row_positions_are_aligned_to_current_frame_order() {
    let runtime = sample_runtime();
    let df = DataFrame::from_series([
        ("key", runtime.series(&[3i32, 1, 2]).unwrap()),
        ("value", runtime.series(&[30i64, 10, 20]).unwrap()),
    ])
    .unwrap();
    let sorted = df.on(Exec::RETAINED).sort_values(["key"], SortOptions::new().with_stable(true)).unwrap();

    let positions = sorted.on(Exec::RETAINED).row_positions().unwrap();

    assert_eq!(sorted.series("key").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(positions.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2]);
}

#[test]
fn dataframe_group_scan_tests_cases() {
    run_named_cases(&[
        (
            "group_scan_sorted_returns_row_aligned_cumulative_sum_per_group",
            group_scan_sorted_returns_row_aligned_cumulative_sum_per_group,
        ),
        ("row_positions_are_aligned_to_current_frame_order", row_positions_are_aligned_to_current_frame_order),
    ]);
}
