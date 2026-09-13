use cudf::{DataFrame, ErrorCode};

use crate::common::{run_named_cases, series};

fn sample_frame() -> DataFrame {
    let ids = series("id", &[1i32, 2, 3, 4, 5, 6]).unwrap();
    let values = series("value", &[10i32, 20, 30, 40, 50, 60]).unwrap();
    DataFrame::from_series([("id", ids), ("value", values)]).unwrap()
}

fn hash_partition_by_preserves_rows_and_reports_offsets() {
    let df = sample_frame();

    let partitioned = df.hash_partition_by(["id"], 3).unwrap();
    let offsets = partitioned.offsets();
    assert_eq!(offsets.len(), 4);
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[3], df.num_rows());
    assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));

    let out = partitioned.dataframe();
    assert_eq!(out.num_rows(), df.num_rows());
    assert_eq!(out.column_names(), vec!["id", "value"]);

    let mut original: Vec<_> = df
        .series("id")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(df.series("value").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    let mut partitioned_rows: Vec<_> = out
        .series("id")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(out.series("value").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    original.sort_unstable();
    partitioned_rows.sort_unstable();
    assert_eq!(partitioned_rows, original);
}

fn hash_partition_by_rejects_empty_key_list() {
    let df = sample_frame();

    let err = df.hash_partition_by(std::iter::empty::<&str>(), 2).unwrap_err();
    assert_eq!(err.code(), ErrorCode::EmptyInput);
    assert_eq!(err.operation(), "dataframe::hash_partition_by");
    assert_eq!(err.detail_value("reason"), Some("empty_partition_keys"));
}

#[test]
fn partitioning_tests_cases() {
    run_named_cases(&[
        ("hash_partition_by_preserves_rows_and_reports_offsets", hash_partition_by_preserves_rows_and_reports_offsets),
        ("hash_partition_by_rejects_empty_key_list", hash_partition_by_rejects_empty_key_list),
    ]);
}
