//! Tests for CU-1 (unique), CU-3 (drop_nulls), CU-8 (drop_duplicates / distinct).

use cudf::column::Column;
use cudf::stream_compaction;
use cudf::table::Table;

// ===== CU-3: drop_nulls =====================================================

#[test]
fn test_drop_nulls_basic() {
    // Table with scattered nulls in key column 0.
    let c0 = Column::from_slice_nullable(&[1i32, 0, 3, 0], &[true, false, true, false]).unwrap();
    let c1 = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let tv = table.view().unwrap();

    // Drop rows where column 0 has nulls (threshold = num_keys = 1 → require all keys valid).
    let result = stream_compaction::drop_nulls(&tv, &[0], 1).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 2);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 3]);
    assert_eq!(rv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![10.0, 30.0]);
}

#[test]
fn test_drop_nulls_no_nulls() {
    let c0 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c0]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::drop_nulls(&tv, &[0], 1).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3);
}

#[test]
fn test_drop_nulls_all_null() {
    let c0 =
        Column::from_slice_nullable(&[0i32, 0, 0], &[false, false, false]).unwrap();
    let table = Table::from_columns(vec![c0]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::drop_nulls(&tv, &[0], 1).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 0);
}

#[test]
fn test_drop_nulls_multi_key() {
    // Drop rows where ANY key column has null (threshold = 2 means both keys must be valid).
    let c0 =
        Column::from_slice_nullable(&[1i32, 2, 3, 4], &[true, false, true, true]).unwrap();
    let c1 =
        Column::from_slice_nullable(&[10i32, 20, 0, 40], &[true, true, false, true]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let tv = table.view().unwrap();

    // threshold=2: need both keys to be non-null.
    let result = stream_compaction::drop_nulls(&tv, &[0, 1], 2).unwrap();
    let rv = result.view().unwrap();
    // Row 0: both valid → kept
    // Row 1: c0 null → dropped
    // Row 2: c1 null → dropped
    // Row 3: both valid → kept
    assert_eq!(rv.num_rows().unwrap(), 2);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 4]);
}

// ===== CU-1: unique (single-column dedup) ====================================

#[test]
fn test_unique_basic_i32() {
    let col = Column::from_slice(&[3i32, 1, 2, 1, 3, 2]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::distinct(
        &tv,
        &[0],
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .unwrap();
    let rv = result.view().unwrap();
    // Should have 3 distinct values.
    assert_eq!(rv.num_rows().unwrap(), 3);
    let mut vals = rv.column(0).unwrap().to_vec::<i32>().unwrap();
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_unique_no_duplicates() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::distinct(
        &tv,
        &[0],
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3);
}

#[test]
fn test_unique_all_same() {
    let col = Column::from_slice(&[5i32, 5, 5, 5]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::distinct(
        &tv,
        &[0],
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 1);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![5]);
}

#[test]
fn test_unique_empty() {
    let col = Column::from_slice(&[] as &[i32]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::distinct(
        &tv,
        &[0],
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 0);
}

// ===== CU-8: drop_duplicates (distinct on multi-key table) ===================

#[test]
fn test_distinct_multi_column_dedup() {
    // Table: (id, name_code) with duplicate id=1.
    let ids = Column::from_slice(&[1i32, 2, 1, 3, 2]).unwrap();
    let codes = Column::from_slice(&[10i32, 20, 10, 30, 20]).unwrap();
    let table = Table::from_columns(vec![ids, codes]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::distinct(
        &tv,
        &[0], // dedup on first column (id)
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3); // ids 1, 2, 3
    let mut ids_out = rv.column(0).unwrap().to_vec::<i32>().unwrap();
    ids_out.sort();
    assert_eq!(ids_out, vec![1, 2, 3]);
}

#[test]
fn test_distinct_keep_none() {
    let col = Column::from_slice(&[1i32, 2, 1, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let result = stream_compaction::distinct(
        &tv,
        &[0],
        stream_compaction::DuplicateKeepOption::KeepNone,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .unwrap();
    let rv = result.view().unwrap();
    // Only values 2 and 3 are non-duplicates.
    assert_eq!(rv.num_rows().unwrap(), 2);
    let mut vals = rv.column(0).unwrap().to_vec::<i32>().unwrap();
    vals.sort();
    assert_eq!(vals, vec![2, 3]);
}
