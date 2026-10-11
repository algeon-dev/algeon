//! Tests for `DataFrame::groupby(...).shift(...)`.

use cudf::{ErrorCode, Exec, ScalarValue, ShiftSpec, TypeId};

use crate::common::{column_from_slice, dataframe_from_columns, scalar};

/// Restore input row order on the resulting DataFrame by sorting on the key
/// and value columns. libcudf does not promise inter-group output ordering,
/// so tests build deterministic expectations after sorting.
fn sort_pairs<K: Ord + Copy, V: Ord + Copy>(mut pairs: Vec<(K, V)>) -> Vec<(K, V)> {
    pairs.sort();
    pairs
}

fn groupby_shift_forward_with_null_fill() {
    // Two groups, three rows each. LAG(value, 1) within group.
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2, 2]).unwrap();
    let row_id = column_from_slice(&[0i32, 1, 2, 3, 4, 5]).unwrap();
    let vals = column_from_slice(&[10i32, 20, 30, 100, 200, 300]).unwrap();
    let df =
        dataframe_from_columns(vec!["key".into(), "row_id".into(), "value".into()], vec![keys, row_id, vals]).unwrap();

    let fill = scalar(ScalarValue::Null(TypeId::Int32.into())).unwrap();
    let row_fill = scalar(ScalarValue::Null(TypeId::Int32.into())).unwrap();
    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .shift(&[
            ShiftSpec::new("row_id", 1, row_fill).with_alias("row_id_lag"),
            ShiftSpec::new("value", 1, fill).with_alias("value_lag"),
        ])
        .unwrap();

    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.num_rows(), 6);
    assert_eq!(result.column_names(), vec!["key", "row_id_lag", "value_lag"]);

    // Within-group order is preserved by libcudf, so pairing the lagged
    // row_id back to the lagged value is well-defined.
    let row_ids = result.series("row_id_lag").unwrap().to_optional_vec::<i32>().unwrap();
    let lagged = result.series("value_lag").unwrap().to_optional_vec::<i32>().unwrap();
    let keys_out = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();

    // Build (key, row_id_lag, value_lag) triples and sort for determinism.
    let mut triples: Vec<(i32, Option<i32>, Option<i32>)> =
        keys_out.into_iter().zip(row_ids.into_iter().zip(lagged)).map(|(k, (r, v))| (k, r, v)).collect();
    triples.sort_by_key(|(k, r, _)| (*k, *r));

    assert_eq!(
        triples,
        vec![
            (1, None, None),
            (1, Some(0), Some(10)),
            (1, Some(1), Some(20)),
            (2, None, None),
            (2, Some(3), Some(100)),
            (2, Some(4), Some(200)),
        ]
    );
}

fn groupby_shift_lead_with_value_fill() {
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 20, 30, 100, 200]).unwrap();
    let row_id = column_from_slice(&[0i32, 1, 2, 3, 4]).unwrap();
    let df =
        dataframe_from_columns(vec!["key".into(), "row_id".into(), "value".into()], vec![keys, row_id, vals]).unwrap();

    let fill = scalar(-1i32).unwrap();
    let row_fill = scalar(-1i32).unwrap();
    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .shift(&[ShiftSpec::new("row_id", -1, row_fill), ShiftSpec::new("value", -1, fill).with_alias("next_value")])
        .unwrap();

    let row_ids = result.series("row_id_shift").unwrap().to_vec_strict::<i32>().unwrap();
    let next = result.series("next_value").unwrap().to_vec_strict::<i32>().unwrap();
    let keys_out = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();

    let mut triples: Vec<(i32, i32, i32)> =
        keys_out.into_iter().zip(row_ids.into_iter().zip(next)).map(|(k, (r, v))| (k, r, v)).collect();
    triples.sort();

    assert_eq!(
        triples,
        vec![
            // Group 1 (key=1)
            (1, -1, -1), // last row in group: LEAD is filled
            (1, 1, 20),  // row_id=0 → next row_id=1, next value=20
            (1, 2, 30),  // row_id=1 → next row_id=2, next value=30
            // Group 2 (key=2)
            (2, -1, -1), // last row in group: LEAD is filled
            (2, 4, 200), // row_id=3 → next row_id=4, next value=200
        ]
    );
}

fn groupby_shift_sorted_keys_hint_matches_unsorted() {
    // Keys ascending, so the sorted-keys hint is valid.
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let row_id = column_from_slice(&[0i32, 1, 2, 3]).unwrap();
    let vals = column_from_slice(&[10i32, 20, 100, 200]).unwrap();
    let df_a =
        dataframe_from_columns(vec!["key".into(), "row_id".into(), "value".into()], vec![keys, row_id, vals]).unwrap();

    let keys2 = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let row_id2 = column_from_slice(&[0i32, 1, 2, 3]).unwrap();
    let vals2 = column_from_slice(&[10i32, 20, 100, 200]).unwrap();
    let df_b = dataframe_from_columns(vec!["key".into(), "row_id".into(), "value".into()], vec![keys2, row_id2, vals2])
        .unwrap();

    let row_fill_a = scalar(-1i32).unwrap();
    let value_fill_a = scalar(-1i32).unwrap();
    let unsorted = df_a
        .groupby(["key"])
        .on(Exec::RETAINED)
        .shift(&[ShiftSpec::new("row_id", 1, row_fill_a), ShiftSpec::new("value", 1, value_fill_a)])
        .unwrap();
    let row_fill_b = scalar(-1i32).unwrap();
    let value_fill_b = scalar(-1i32).unwrap();
    let sorted_hint = df_b
        .groupby(["key"])
        .on(Exec::RETAINED)
        .shift_sorted(&[ShiftSpec::new("row_id", 1, row_fill_b), ShiftSpec::new("value", 1, value_fill_b)])
        .unwrap();

    let pair_a: Vec<(i32, i32, i32)> = {
        let keys = unsorted.series("key").unwrap().to_vec_strict::<i32>().unwrap();
        let rs = unsorted.series("row_id_shift").unwrap().to_vec_strict::<i32>().unwrap();
        let vs = unsorted.series("value_shift").unwrap().to_vec_strict::<i32>().unwrap();
        sort_pairs(keys.into_iter().zip(rs.into_iter().zip(vs)).map(|(k, (r, v))| ((k, r), v)).collect())
            .into_iter()
            .map(|((k, r), v)| (k, r, v))
            .collect()
    };
    let pair_b: Vec<(i32, i32, i32)> = {
        let keys = sorted_hint.series("key").unwrap().to_vec_strict::<i32>().unwrap();
        let rs = sorted_hint.series("row_id_shift").unwrap().to_vec_strict::<i32>().unwrap();
        let vs = sorted_hint.series("value_shift").unwrap().to_vec_strict::<i32>().unwrap();
        sort_pairs(keys.into_iter().zip(rs.into_iter().zip(vs)).map(|(k, (r, v))| ((k, r), v)).collect())
            .into_iter()
            .map(|((k, r), v)| (k, r, v))
            .collect()
    };
    assert_eq!(pair_a, pair_b);
}

fn groupby_shift_dtype_mismatch_rejects() {
    let keys = column_from_slice(&[1i32, 1]).unwrap();
    let vals = column_from_slice(&[10i32, 20]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let wrong_fill = scalar(1.5f64).unwrap();
    let err = df.groupby(["key"]).on(Exec::RETAINED).shift(&[ShiftSpec::new("value", 1, wrong_fill)]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    let dump = format!("{err:?}");
    assert!(dump.contains("shift_fill_value"), "missing resource fact: {dump}");
    assert!(dump.contains("dtype_mismatch"), "missing reason: {dump}");
}

fn groupby_shift_empty_specs_rejects() {
    let keys = column_from_slice(&[1i32, 1]).unwrap();
    let vals = column_from_slice(&[10i32, 20]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df.groupby(["key"]).on(Exec::RETAINED).shift(&[]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    let dump = format!("{err:?}");
    assert!(dump.contains("empty_shift_specs"), "missing reason: {dump}");
}

fn groupby_shift_missing_column_rejects() {
    let keys = column_from_slice(&[1i32, 1]).unwrap();
    let vals = column_from_slice(&[10i32, 20]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let fill = scalar(ScalarValue::Null(TypeId::Int32.into())).unwrap();
    let err = df.groupby(["key"]).on(Exec::RETAINED).shift(&[ShiftSpec::new("missing", 1, fill)]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::MissingColumn);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("shift_tests::groupby_shift_forward_with_null_fill", groupby_shift_forward_with_null_fill),
    ("shift_tests::groupby_shift_lead_with_value_fill", groupby_shift_lead_with_value_fill),
    ("shift_tests::groupby_shift_sorted_keys_hint_matches_unsorted", groupby_shift_sorted_keys_hint_matches_unsorted),
    ("shift_tests::groupby_shift_dtype_mismatch_rejects", groupby_shift_dtype_mismatch_rejects),
    ("shift_tests::groupby_shift_empty_specs_rejects", groupby_shift_empty_specs_rejects),
    ("shift_tests::groupby_shift_missing_column_rejects", groupby_shift_missing_column_rejects),
];
