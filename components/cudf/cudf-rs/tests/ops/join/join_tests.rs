//! Tests for join operations.
use cudf::{
    DataFrame, MergeOptions, Series,
    core::Table,
    ops::{
        copying,
        join::{self, NullEquality},
    },
    testing::prelude::*,
};

use crate::common::{
    column_from_slice, column_from_slice_nullable, explicit_i32_column, last_exec_mr_value, last_exec_stream_value,
    last_sync_stream_value, reset_exec_observation, reset_sync_observation, sample_runtime,
};

const JOIN_NO_MATCH: i32 = i32::MIN;

/// Helper to create an empty table (0 columns) for use as an empty payload.
fn empty_payload() -> Table {
    sample_runtime().empty_table().unwrap()
}

fn nullable_i32_table_column(table: &Table, index: usize) -> Vec<Option<i32>> {
    let view = table.testing_view().unwrap();
    let column = view.column(index).unwrap();
    let values = column.to_vec::<i32>().unwrap();
    let validity = column.to_host_validity().unwrap().unwrap_or_else(|| vec![true; values.len()]);

    values.into_iter().zip(validity).map(|(value, valid)| if valid { Some(value) } else { None }).collect()
}

fn sorted_join_index_pairs(indices: &join::JoinIndices) -> Vec<(i32, i32)> {
    let left = indices.left.testing_view().unwrap().to_vec::<i32>().unwrap();
    let right = indices.right.testing_view().unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<_> = left.into_iter().zip(right).collect();
    pairs.sort();
    pairs
}

fn sorted_single_side_indices(indices: &join::SingleSideJoinIndices) -> Vec<i32> {
    let mut rows = indices.indices.testing_view().unwrap().to_vec::<i32>().unwrap();
    rows.sort();
    rows
}

fn sorted_dataframe_triples(df: &DataFrame) -> Vec<(i32, i32, i32)> {
    let ids = df.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    let left_vals = df.series("left_val").unwrap().to_vec_strict::<i32>().unwrap();
    let right_vals = df.series("right_val").unwrap().to_vec_strict::<i32>().unwrap();

    let mut triples: Vec<_> = ids
        .into_iter()
        .zip(left_vals.into_iter().zip(right_vals))
        .map(|(id, (left, right))| (id, left, right))
        .collect();
    triples.sort();
    triples
}

#[test]
fn test_inner_join() {
    // Left: keys=[1,2,3], payload=[10,20,30]
    // Right: keys=[2,3,4], payload=[200,300,400]
    // Inner join on keys: matches are 2 and 3.
    let lk = column_from_slice(&[1i32, 2, 3]).unwrap();
    let lp = column_from_slice(&[10i32, 20, 30]).unwrap();
    let rk = column_from_slice(&[2i32, 3, 4]).unwrap();
    let rp = column_from_slice(&[200i32, 300, 400]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let lp_table = Table::from_columns(vec![lp]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();
    let rp_table = Table::from_columns(vec![rp]).unwrap();

    let lkv = lk_table.testing_view().unwrap();
    let lpv = lp_table.testing_view().unwrap();
    let rkv = rk_table.testing_view().unwrap();
    let rpv = rp_table.testing_view().unwrap();

    let result = join::inner_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.testing_view().unwrap();
    let rv = result.right.testing_view().unwrap();

    // Inner join should have 2 rows.
    assert_eq!(lv.num_rows().unwrap(), 2);
    assert_eq!(rv.num_rows().unwrap(), 2);

    // Left result has key + payload = 2 columns.
    assert_eq!(lv.num_columns().unwrap(), 2);
    assert_eq!(rv.num_columns().unwrap(), 2);

    // Verify data: collect and sort for deterministic comparison.
    let left_keys: Vec<i32> = lv.column(0).unwrap().to_vec::<i32>().unwrap();
    let left_vals: Vec<i32> = lv.column(1).unwrap().to_vec::<i32>().unwrap();
    let right_vals: Vec<i32> = rv.column(1).unwrap().to_vec::<i32>().unwrap();

    let mut triples: Vec<(i32, i32, i32)> =
        left_keys.into_iter().zip(left_vals.into_iter().zip(right_vals)).map(|(k, (lv, rv))| (k, lv, rv)).collect();
    triples.sort_by_key(|&(k, _, _)| k);

    assert_eq!(triples, vec![(2, 20, 200), (3, 30, 300)]);
}

#[test]
fn test_left_join() {
    // Left: keys=[1,2,3], Right: keys=[2,4]
    // Left join preserves all left rows.
    let lk = column_from_slice(&[1i32, 2, 3]).unwrap();
    let rk = column_from_slice(&[2i32, 4]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();
    let lp = empty_payload();
    let rp = empty_payload();

    let lkv = lk_table.testing_view().unwrap();
    let rkv = rk_table.testing_view().unwrap();
    let lpv = lp.testing_view().unwrap();
    let rpv = rp.testing_view().unwrap();

    let result = join::left_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.testing_view().unwrap();
    let rv = result.right.testing_view().unwrap();
    // Left join: all 3 left rows should be present.
    assert_eq!(lv.num_rows().unwrap(), 3);
    assert_eq!(rv.num_rows().unwrap(), 3);
    assert_eq!(lv.num_columns().unwrap(), 1);
    assert_eq!(rv.num_columns().unwrap(), 1);

    let mut left_keys = nullable_i32_table_column(&result.left, 0);
    left_keys.sort();
    assert_eq!(left_keys, vec![Some(1), Some(2), Some(3)]);

    let mut right_keys = nullable_i32_table_column(&result.right, 0);
    right_keys.sort();
    assert_eq!(right_keys, vec![None, None, Some(2)]);
}

#[test]
fn test_full_join() {
    // Left: keys=[1,2], Right: keys=[2,3]
    // Full join: rows for keys 1, 2, 3.
    let lk = column_from_slice(&[1i32, 2]).unwrap();
    let rk = column_from_slice(&[2i32, 3]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();
    let lp = empty_payload();
    let rp = empty_payload();

    let lkv = lk_table.testing_view().unwrap();
    let rkv = rk_table.testing_view().unwrap();
    let lpv = lp.testing_view().unwrap();
    let rpv = rp.testing_view().unwrap();

    let result = join::full_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.testing_view().unwrap();
    let rv = result.right.testing_view().unwrap();
    // Full join should have 3 rows (1, 2, 3).
    assert_eq!(lv.num_rows().unwrap(), 3);
    assert_eq!(rv.num_rows().unwrap(), 3);
    assert_eq!(lv.num_columns().unwrap(), 1);
    assert_eq!(rv.num_columns().unwrap(), 1);

    let mut left_keys = nullable_i32_table_column(&result.left, 0);
    left_keys.sort();
    assert_eq!(left_keys, vec![None, Some(1), Some(2)]);

    let mut right_keys = nullable_i32_table_column(&result.right, 0);
    right_keys.sort();
    assert_eq!(right_keys, vec![None, Some(2), Some(3)]);
}

#[test]
fn test_cross_join_materializes_rows_in_libcudf_order() {
    let left_id = column_from_slice(&[1i32, 2]).unwrap();
    let left_val = column_from_slice(&[10i32, 20]).unwrap();
    let right_id = column_from_slice(&[100i32, 200, 300]).unwrap();

    let left = Table::from_columns(vec![left_id, left_val]).unwrap();
    let right = Table::from_columns(vec![right_id]).unwrap();

    let result = join::cross_join(&left.testing_view().unwrap(), &right.testing_view().unwrap()).unwrap();
    let view = result.testing_view().unwrap();

    assert_eq!(view.num_columns().unwrap(), 3);
    assert_eq!(view.num_rows().unwrap(), 6);
    assert_eq!(view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 1, 2, 2, 2]);
    assert_eq!(view.column(1).unwrap().to_vec::<i32>().unwrap(), vec![10, 10, 10, 20, 20, 20]);
    assert_eq!(view.column(2).unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300, 100, 200, 300]);
}

#[test]
fn test_join_error_contains_operation_context() {
    let lk = column_from_slice(&[1i32, 2]).unwrap();
    let rk = column_from_slice(&[2i32, 3]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();

    // Deliberately invalid payload row count for left side.
    let lp_bad = Table::from_columns(vec![column_from_slice(&[10i32]).unwrap()]).unwrap();
    let rp = empty_payload();

    let lkv = lk_table.testing_view().unwrap();
    let rkv = rk_table.testing_view().unwrap();
    let lpv = lp_bad.testing_view().unwrap();
    let rpv = rp.testing_view().unwrap();

    let err = match join::inner_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal) {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };
    assert_eq!(err.operation(), "join::inner_join");
    assert!(err.details().iter().any(|(k, _)| *k == "null_equality"));
}

#[test]
fn test_inner_join_multi_column_keys() {
    // Two key columns.
    let lk0 = column_from_slice(&[1i32, 1, 2]).unwrap();
    let lk1 = column_from_slice(&[10i32, 20, 10]).unwrap();
    let rk0 = column_from_slice(&[1i32, 2, 2]).unwrap();
    let rk1 = column_from_slice(&[10i32, 10, 20]).unwrap();

    let lk_table = Table::from_columns(vec![lk0, lk1]).unwrap();
    let rk_table = Table::from_columns(vec![rk0, rk1]).unwrap();
    let lp = empty_payload();
    let rp = empty_payload();

    let lkv = lk_table.testing_view().unwrap();
    let rkv = rk_table.testing_view().unwrap();
    let lpv = lp.testing_view().unwrap();
    let rpv = rp.testing_view().unwrap();

    let result = join::inner_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.testing_view().unwrap();
    let rv = result.right.testing_view().unwrap();
    // Matches: (1,10) and (2,10) → 2 rows.
    assert_eq!(lv.num_rows().unwrap(), 2);
    assert_eq!(rv.num_rows().unwrap(), 2);
    assert_eq!(lv.num_columns().unwrap(), 2);
    assert_eq!(rv.num_columns().unwrap(), 2);

    let mut left_pairs: Vec<_> = nullable_i32_table_column(&result.left, 0)
        .into_iter()
        .zip(nullable_i32_table_column(&result.left, 1))
        .collect();
    left_pairs.sort();
    assert_eq!(left_pairs, vec![(Some(1), Some(10)), (Some(2), Some(10))]);

    let mut right_pairs: Vec<_> = nullable_i32_table_column(&result.right, 0)
        .into_iter()
        .zip(nullable_i32_table_column(&result.right, 1))
        .collect();
    right_pairs.sort();
    assert_eq!(right_pairs, vec![(Some(1), Some(10)), (Some(2), Some(10))]);
}

#[test]
fn test_join_domain_module_exports_documented_surface() {
    use cudf::ops::join::{JoinSides, JoinType, MergeOptions, NullEquality, full_join, inner_join, left_join};

    let left_keys = column_from_slice(&[1i32, 2, 3]).unwrap();
    let right_keys = column_from_slice(&[2i32, 3, 4]).unwrap();

    let left_keys = Table::from_columns(vec![left_keys]).unwrap();
    let right_keys = Table::from_columns(vec![right_keys]).unwrap();
    let left_payload = empty_payload();
    let right_payload = empty_payload();

    let inner: JoinSides = inner_join(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        &left_payload.testing_view().unwrap(),
        &right_payload.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    assert_eq!(inner.left.testing_view().unwrap().num_rows().unwrap(), 2);

    let left = left_join(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        &left_payload.testing_view().unwrap(),
        &right_payload.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    assert_eq!(left.left.testing_view().unwrap().num_rows().unwrap(), 3);

    let full = full_join(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        &left_payload.testing_view().unwrap(),
        &right_payload.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    assert_eq!(full.left.testing_view().unwrap().num_rows().unwrap(), 4);

    let opts = MergeOptions::new().on(["id"]).how(JoinType::Left);
    assert_eq!(opts.join_type(), JoinType::Left);
}

#[test]
fn test_inner_join_succeeds_with_mixed_domains_and_uses_left_provenance() {
    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 950)]).unwrap();
    let left_payload = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 950)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3, 4], 951)]).unwrap();
    let right_payload = Table::from_columns(vec![explicit_i32_column(&[200, 300, 400], 951)]).unwrap();
    let expected_stream = left_keys.testing_execution_domain_stream_value();
    let expected_mr = left_keys.testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = join::inner_join(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        &left_payload.testing_view().unwrap(),
        &right_payload.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(result.left.testing_execution_domain_id(), left_keys.testing_execution_domain_id());
    assert_eq!(result.right.testing_execution_domain_id(), left_keys.testing_execution_domain_id());
    assert_eq!(result.left.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.right.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    let left_rows: Vec<(i32, i32)> = result
        .left
        .testing_view()
        .unwrap()
        .column(0)
        .unwrap()
        .to_vec::<i32>()
        .unwrap()
        .into_iter()
        .zip(result.left.testing_view().unwrap().column(1).unwrap().to_vec::<i32>().unwrap())
        .collect();
    let right_rows: Vec<(i32, i32)> = result
        .right
        .testing_view()
        .unwrap()
        .column(0)
        .unwrap()
        .to_vec::<i32>()
        .unwrap()
        .into_iter()
        .zip(result.right.testing_view().unwrap().column(1).unwrap().to_vec::<i32>().unwrap())
        .collect();
    assert_eq!(left_rows, vec![(2, 20), (3, 30)]);
    assert_eq!(right_rows, vec![(2, 200), (3, 300)]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn test_inner_join_indices_with_duplicate_keys_returns_all_matching_pairs() {
    let left_keys = Table::from_columns(vec![column_from_slice(&[1i32, 2, 2, 3]).unwrap()]).unwrap();
    let right_keys = Table::from_columns(vec![column_from_slice(&[2i32, 2, 4]).unwrap()]).unwrap();

    let indices = join::inner_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(1, 0), (1, 1), (2, 0), (2, 1)]);
}

#[test]
fn test_left_semi_join_indices_return_each_matching_left_row_once() {
    let left_keys = Table::from_columns(vec![column_from_slice(&[1i32, 2, 2, 3, 4]).unwrap()]).unwrap();
    let right_keys = Table::from_columns(vec![column_from_slice(&[2i32, 2, 4]).unwrap()]).unwrap();

    let indices = join::left_semi_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_single_side_indices(&indices), vec![1, 2, 4]);
}

#[test]
fn test_left_anti_join_indices_return_only_non_matching_left_rows() {
    let left_keys = Table::from_columns(vec![column_from_slice(&[1i32, 2, 2, 3, 4]).unwrap()]).unwrap();
    let right_keys = Table::from_columns(vec![column_from_slice(&[2i32, 4]).unwrap()]).unwrap();

    let indices = join::left_anti_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_single_side_indices(&indices), vec![0, 3]);
}

#[test]
fn test_inner_join_indices_materialize_same_rows_as_dataframe_merge() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 2, 3]).unwrap()),
        ("left_val", rt.series(&[10i32, 20, 21, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[2i32, 2, 4]).unwrap()),
        ("right_val", rt.series(&[200i32, 201, 400]).unwrap()),
    ])
    .unwrap();

    let left_keys = Table::from_columns(vec![left.series("id").unwrap().as_column().deep_copy().unwrap()]).unwrap();
    let right_keys = Table::from_columns(vec![right.series("id").unwrap().as_column().deep_copy().unwrap()]).unwrap();
    let indices = join::inner_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();

    let left_indices = Series::from_column(indices.left).unwrap();
    let right_indices = Series::from_column(indices.right).unwrap();
    let left_gathered = left.gather(&left_indices).unwrap();
    let right_table = right.to_table_copy().unwrap();
    let right_gathered =
        copying::gather(&right_table.testing_view().unwrap(), &right_indices.as_column().testing_view().unwrap())
            .unwrap();
    let mut right_columns = right_gathered.into_columns().unwrap();
    let assembled = DataFrame::from_series([
        ("id", left_gathered.series("id").unwrap()),
        ("left_val", left_gathered.series("left_val").unwrap()),
        ("right_val", Series::from_column(right_columns.remove(1)).unwrap()),
    ])
    .unwrap();

    let merged = left.merge(&right, MergeOptions::new().on(["id"]).how_inner()).unwrap();
    assert_eq!(sorted_dataframe_triples(&assembled), sorted_dataframe_triples(&merged));
}

#[test]
fn test_left_join_indices_with_duplicate_keys_preserves_unmatched_left_rows() {
    let left_keys = Table::from_columns(vec![column_from_slice(&[1i32, 2, 2, 3]).unwrap()]).unwrap();
    let right_keys = Table::from_columns(vec![column_from_slice(&[2i32, 2, 4]).unwrap()]).unwrap();

    let indices = join::left_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(
        sorted_join_index_pairs(&indices),
        vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 0), (2, 1), (3, JOIN_NO_MATCH)]
    );
}

#[test]
fn test_full_join_indices_with_duplicate_keys_preserves_unmatched_rows_from_both_sides() {
    let left_keys = Table::from_columns(vec![column_from_slice(&[1i32, 2, 2, 3]).unwrap()]).unwrap();
    let right_keys = Table::from_columns(vec![column_from_slice(&[2i32, 2, 4]).unwrap()]).unwrap();

    let indices = join::full_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(
        sorted_join_index_pairs(&indices),
        vec![(JOIN_NO_MATCH, 2), (0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 0), (2, 1), (3, JOIN_NO_MATCH),]
    );
}

#[test]
fn test_inner_join_indices_honor_null_equality() {
    let left_keys =
        Table::from_columns(vec![column_from_slice_nullable(&[1i32, 0, 2, 0], &[true, false, true, false]).unwrap()])
            .unwrap();
    let right_keys =
        Table::from_columns(vec![column_from_slice_nullable(&[0i32, 2, 0], &[false, true, false]).unwrap()]).unwrap();

    let equal = join::inner_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    let unequal = join::inner_join_indices(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        NullEquality::Unequal,
    )
    .unwrap();

    assert_eq!(sorted_join_index_pairs(&equal), vec![(1, 0), (1, 2), (2, 1), (3, 0), (3, 2)]);
    assert_eq!(sorted_join_index_pairs(&unequal), vec![(2, 1)]);
}

#[test]
fn test_hash_join_supports_duplicate_build_keys_and_repeated_probes() {
    use cudf::ops::join::{HashJoin, HashJoinOptions};

    let build_keys = Table::from_columns(vec![column_from_slice(&[2i32, 2, 4]).unwrap()]).unwrap();
    let first_probe = Table::from_columns(vec![column_from_slice(&[2i32, 4]).unwrap()]).unwrap();
    let second_probe = Table::from_columns(vec![column_from_slice(&[1i32, 2]).unwrap()]).unwrap();
    let hash_join = HashJoin::new(build_keys, HashJoinOptions::new()).unwrap();

    let first = hash_join.inner_join_indices(&first_probe.testing_view().unwrap()).unwrap();
    let second = hash_join.left_join_indices(&second_probe.testing_view().unwrap()).unwrap();

    assert_eq!(sorted_join_index_pairs(&first), vec![(0, 0), (0, 1), (1, 2)]);
    assert_eq!(sorted_join_index_pairs(&second), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1)]);
}

#[test]
fn test_hash_join_supports_parallel_probes_from_cloned_handle() {
    use cudf::ops::join::{HashJoin, HashJoinOptions};

    let build_keys = Table::from_columns(vec![column_from_slice(&[2i32, 2, 4]).unwrap()]).unwrap();
    let first_probe = Table::from_columns(vec![column_from_slice(&[2i32, 4]).unwrap()]).unwrap();
    let second_probe = Table::from_columns(vec![column_from_slice(&[1i32, 2]).unwrap()]).unwrap();
    let hash_join = HashJoin::new(build_keys, HashJoinOptions::new()).unwrap();

    let first_join = hash_join.clone();
    let first = std::thread::spawn(move || {
        let indices = first_join.inner_join_indices(&first_probe.testing_view().unwrap()).unwrap();
        sorted_join_index_pairs(&indices)
    });

    let second_join = hash_join.clone();
    let second = std::thread::spawn(move || {
        let indices = second_join.left_join_indices(&second_probe.testing_view().unwrap()).unwrap();
        sorted_join_index_pairs(&indices)
    });

    assert_eq!(first.join().unwrap(), vec![(0, 0), (0, 1), (1, 2)]);
    assert_eq!(second.join().unwrap(), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1)]);
}
