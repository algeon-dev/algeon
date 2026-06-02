//! Tests for join operations.

use cudf::{
    column::Column,
    join::{self, NullEquality},
    table::Table,
};

/// Helper to create an empty table (0 columns) for use as an empty payload.
fn empty_payload() -> Table {
    Table::from_columns(vec![]).unwrap()
}

fn nullable_i32_table_column(table: &Table, index: usize) -> Vec<Option<i32>> {
    let view = table.view().unwrap();
    let column = view.column(index.try_into().unwrap()).unwrap();
    let values = column.to_vec::<i32>().unwrap();
    let validity = column.validity().unwrap().unwrap_or_else(|| vec![true; values.len()]);

    values.into_iter().zip(validity).map(|(value, valid)| if valid { Some(value) } else { None }).collect()
}

#[test]
fn test_inner_join() {
    // Left: keys=[1,2,3], payload=[10,20,30]
    // Right: keys=[2,3,4], payload=[200,300,400]
    // Inner join on keys: matches are 2 and 3.
    let lk = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let lp = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let rk = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let rp = Column::from_slice(&[200i32, 300, 400]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let lp_table = Table::from_columns(vec![lp]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();
    let rp_table = Table::from_columns(vec![rp]).unwrap();

    let lkv = lk_table.view().unwrap();
    let lpv = lp_table.view().unwrap();
    let rkv = rk_table.view().unwrap();
    let rpv = rp_table.view().unwrap();

    let result = join::inner_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.view().unwrap();
    let rv = result.right.view().unwrap();

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
    let lk = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let rk = Column::from_slice(&[2i32, 4]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();
    let lp = empty_payload();
    let rp = empty_payload();

    let lkv = lk_table.view().unwrap();
    let rkv = rk_table.view().unwrap();
    let lpv = lp.view().unwrap();
    let rpv = rp.view().unwrap();

    let result = join::left_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.view().unwrap();
    let rv = result.right.view().unwrap();
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
    let lk = Column::from_slice(&[1i32, 2]).unwrap();
    let rk = Column::from_slice(&[2i32, 3]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();
    let lp = empty_payload();
    let rp = empty_payload();

    let lkv = lk_table.view().unwrap();
    let rkv = rk_table.view().unwrap();
    let lpv = lp.view().unwrap();
    let rpv = rp.view().unwrap();

    let result = join::full_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.view().unwrap();
    let rv = result.right.view().unwrap();
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
fn test_join_error_contains_operation_context() {
    let lk = Column::from_slice(&[1i32, 2]).unwrap();
    let rk = Column::from_slice(&[2i32, 3]).unwrap();

    let lk_table = Table::from_columns(vec![lk]).unwrap();
    let rk_table = Table::from_columns(vec![rk]).unwrap();

    // Deliberately invalid payload row count for left side.
    let lp_bad = Table::from_columns(vec![Column::from_slice(&[10i32]).unwrap()]).unwrap();
    let rp = empty_payload();

    let lkv = lk_table.view().unwrap();
    let rkv = rk_table.view().unwrap();
    let lpv = lp_bad.view().unwrap();
    let rpv = rp.view().unwrap();

    let err = join::inner_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).err().expect("expected error");
    assert_eq!(err.operation(), "join::inner_join");
    assert!(err.context().iter().any(|(k, _)| *k == "null_equality"));
}

#[test]
fn test_inner_join_multi_column_keys() {
    // Two key columns.
    let lk0 = Column::from_slice(&[1i32, 1, 2]).unwrap();
    let lk1 = Column::from_slice(&[10i32, 20, 10]).unwrap();
    let rk0 = Column::from_slice(&[1i32, 2, 2]).unwrap();
    let rk1 = Column::from_slice(&[10i32, 10, 20]).unwrap();

    let lk_table = Table::from_columns(vec![lk0, lk1]).unwrap();
    let rk_table = Table::from_columns(vec![rk0, rk1]).unwrap();
    let lp = empty_payload();
    let rp = empty_payload();

    let lkv = lk_table.view().unwrap();
    let rkv = rk_table.view().unwrap();
    let lpv = lp.view().unwrap();
    let rpv = rp.view().unwrap();

    let result = join::inner_join(&lkv, &rkv, &lpv, &rpv, NullEquality::Equal).unwrap();

    let lv = result.left.view().unwrap();
    let rv = result.right.view().unwrap();
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
