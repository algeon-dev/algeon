//! Tests for CU-7: gather (take) and conditional copy.
use cudf::{core::Table, ops::copying, testing::prelude::*};

use crate::common::{
    column_from_slice, column_from_slice_nullable, explicit_bool_column, explicit_i32_column, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation, run_named_cases,
};

fn test_gather_basic() {
    let col = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.testing_view().unwrap();

    // Gather rows 4, 2, 0.
    let map = column_from_slice(&[4i32, 2, 0]).unwrap();
    let map_view = map.testing_view().unwrap();

    let result = copying::gather(&tv, &map_view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![50, 30, 10]);
}

fn test_gather_multi_column() {
    let c0 = column_from_slice(&[1i32, 2, 3, 4]).unwrap();
    let c1 = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let tv = table.testing_view().unwrap();

    let map = column_from_slice(&[3i32, 1]).unwrap();
    let map_view = map.testing_view().unwrap();

    let result = copying::gather(&tv, &map_view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.num_columns().unwrap(), 2);
    assert_eq!(rv.num_rows().unwrap(), 2);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![4, 2]);
    assert_eq!(rv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![40.0, 20.0]);
}

fn test_gather_identity() {
    // Gather in order → identical copy.
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.testing_view().unwrap();

    let map = column_from_slice(&[0i32, 1, 2]).unwrap();
    let map_view = map.testing_view().unwrap();

    let result = copying::gather(&tv, &map_view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_gather_with_duplicates() {
    // Gather the same row multiple times.
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.testing_view().unwrap();

    let map = column_from_slice(&[1i32, 1, 1, 0]).unwrap();
    let map_view = map.testing_view().unwrap();

    let result = copying::gather(&tv, &map_view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![20, 20, 20, 10]);
}

fn test_gather_empty_map() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.testing_view().unwrap();

    let map = column_from_slice(&[] as &[i32]).unwrap();
    let map_view = map.testing_view().unwrap();

    let result = copying::gather(&tv, &map_view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 0);
}

fn test_gather_out_of_bounds_nullify() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.testing_view().unwrap();

    // Index 5 is out of bounds; should become null.
    let map = column_from_slice(&[0i32, 5, 2]).unwrap();
    let map_view = map.testing_view().unwrap();

    let result = copying::gather(&tv, &map_view).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3);
    let cv = rv.column(0).unwrap();
    // Row 1 should be null.
    let validity = cv.to_host_validity().unwrap().expect("should have null mask");
    assert!(validity[0]);
    assert!(!validity[1]); // out-of-bounds → null
    assert!(validity[2]);
}

fn test_gather_unchecked_basic() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.testing_view().unwrap();

    let map = column_from_slice(&[2i32, 0]).unwrap();
    let map_view = map.testing_view().unwrap();

    // SAFETY: indices 2 and 0 are valid for a 3-row table.
    let result = unsafe { copying::gather_unchecked(&tv, &map_view) }.unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 2);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![30, 10]);
}

fn test_gather_succeeds_with_mixed_domains_and_uses_input_provenance() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 930)]).unwrap();
    let gather_map = explicit_i32_column(&[2, 0], 931);
    let expected_stream = table.testing_execution_domain_stream_value();
    let expected_mr = table.testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = copying::gather(&table.testing_view().unwrap(), &gather_map.testing_view().unwrap()).unwrap();

    assert_eq!(result.testing_execution_domain_id(), table.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![30, 10]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

fn test_scatter_writes_source_rows_into_target_and_supports_negative_indices() {
    let source = Table::from_columns(vec![column_from_slice(&[100i32, 200]).unwrap()]).unwrap();
    let target = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30, 40]).unwrap()]).unwrap();
    let scatter_map = column_from_slice(&[1i32, -1]).unwrap();

    let result = copying::scatter(
        &source.testing_view().unwrap(),
        &scatter_map.testing_view().unwrap(),
        &target.testing_view().unwrap(),
    )
    .unwrap();

    let result_view = result.testing_view().unwrap();
    assert_eq!(result_view.num_rows().unwrap(), 4);
    assert_eq!(result_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 100, 30, 200]);
}

fn test_scatter_accepts_partial_scatter_map() {
    let source = Table::from_columns(vec![column_from_slice(&[100i32, 200, 999]).unwrap()]).unwrap();
    let target = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30, 40]).unwrap()]).unwrap();
    let scatter_map = column_from_slice(&[1i32, -1]).unwrap();

    let result = copying::scatter(
        &source.testing_view().unwrap(),
        &scatter_map.testing_view().unwrap(),
        &target.testing_view().unwrap(),
    )
    .unwrap();

    let result_view = result.testing_view().unwrap();
    assert_eq!(result_view.num_rows().unwrap(), 4);
    assert_eq!(result_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 100, 30, 200]);
}

fn test_scatter_accepts_empty_scatter_map_as_target_copy() {
    let source = Table::from_columns(vec![column_from_slice(&[100i32, 200]).unwrap()]).unwrap();
    let target = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let scatter_map = column_from_slice::<i32>(&[]).unwrap();

    let result = copying::scatter(
        &source.testing_view().unwrap(),
        &scatter_map.testing_view().unwrap(),
        &target.testing_view().unwrap(),
    )
    .unwrap();

    let result_view = result.testing_view().unwrap();
    assert_eq!(result_view.num_rows().unwrap(), 3);
    assert_eq!(result_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

fn test_scatter_rejects_scatter_map_longer_than_source() {
    let source = Table::from_columns(vec![column_from_slice(&[100i32, 200]).unwrap()]).unwrap();
    let target = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let scatter_map = column_from_slice(&[0i32, 1, 2]).unwrap();

    let err = copying::scatter(
        &source.testing_view().unwrap(),
        &scatter_map.testing_view().unwrap(),
        &target.testing_view().unwrap(),
    )
    .expect_err("scatter_map longer than source should be rejected");

    assert_eq!(err.operation(), "copying::scatter");
    assert_eq!(err.detail_value("reason"), Some("scatter_map_too_long"));
}

fn test_scatter_rejects_null_scatter_map() {
    let source = Table::from_columns(vec![column_from_slice(&[100i32, 200]).unwrap()]).unwrap();
    let target = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let scatter_map = column_from_slice_nullable(&[0i32, 2], &[true, false]).unwrap();

    let err = copying::scatter(
        &source.testing_view().unwrap(),
        &scatter_map.testing_view().unwrap(),
        &target.testing_view().unwrap(),
    )
    .expect_err("nullable scatter_map should be rejected");

    assert_eq!(err.operation(), "copying::scatter");
}

fn test_scatter_rejects_out_of_bounds_scatter_map() {
    let source = Table::from_columns(vec![column_from_slice(&[100i32, 200]).unwrap()]).unwrap();
    let target = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let scatter_map = column_from_slice(&[0i32, 3]).unwrap();

    let err = copying::scatter(
        &source.testing_view().unwrap(),
        &scatter_map.testing_view().unwrap(),
        &target.testing_view().unwrap(),
    )
    .expect_err("safe scatter should reject out-of-bounds indices");

    assert_eq!(err.operation(), "copying::scatter");
    assert_eq!(err.detail_value("reason"), Some("index_out_of_bounds"));
}

fn test_scatter_rejects_duplicate_scatter_map() {
    let source = Table::from_columns(vec![column_from_slice(&[100i32, 200]).unwrap()]).unwrap();
    let target = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let scatter_map = column_from_slice(&[1i32, -2]).unwrap();

    let err = copying::scatter(
        &source.testing_view().unwrap(),
        &scatter_map.testing_view().unwrap(),
        &target.testing_view().unwrap(),
    )
    .expect_err("safe scatter should reject duplicate normalized target indices");

    assert_eq!(err.operation(), "copying::scatter");
    assert_eq!(err.detail_value("reason"), Some("duplicate_target_index"));
}

fn test_copy_if_else_basic() {
    let lhs = column_from_slice(&[10i32, 20, 30]).unwrap();
    let rhs = column_from_slice(&[1i32, 2, 3]).unwrap();
    let mask = column_from_slice(&[true, false, true]).unwrap();

    let result = copying::copy_if_else(
        &lhs.testing_view().unwrap(),
        &rhs.testing_view().unwrap(),
        &mask.testing_view().unwrap(),
    )
    .unwrap();
    assert_eq!(result.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![10, 2, 30]);
}

fn test_copy_if_else_treats_null_mask_as_false() {
    let lhs = column_from_slice(&[10i32, 20, 30]).unwrap();
    let rhs = column_from_slice(&[1i32, 2, 3]).unwrap();
    let mask = column_from_slice_nullable(&[true, false, false], &[true, false, true]).unwrap();

    let result = copying::copy_if_else(
        &lhs.testing_view().unwrap(),
        &rhs.testing_view().unwrap(),
        &mask.testing_view().unwrap(),
    )
    .unwrap();
    assert_eq!(result.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![10, 2, 3]);
}

fn test_copy_if_else_succeeds_with_mixed_domains_and_uses_lhs_provenance() {
    let lhs = explicit_i32_column(&[10, 20, 30], 940);
    let rhs = explicit_i32_column(&[1, 2, 3], 941);
    let mask = explicit_bool_column(&[true, false, true], 942);
    let expected_stream = lhs.testing_execution_domain_stream_value();
    let expected_mr = lhs.testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = copying::copy_if_else(
        &lhs.testing_view().unwrap(),
        &rhs.testing_view().unwrap(),
        &mask.testing_view().unwrap(),
    )
    .unwrap();

    assert_eq!(result.testing_execution_domain_id(), lhs.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![10, 2, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn copying_tests_cases() {
    run_named_cases(&[
        ("test_gather_basic", test_gather_basic),
        ("test_gather_multi_column", test_gather_multi_column),
        ("test_gather_identity", test_gather_identity),
        ("test_gather_with_duplicates", test_gather_with_duplicates),
        ("test_gather_empty_map", test_gather_empty_map),
        ("test_gather_out_of_bounds_nullify", test_gather_out_of_bounds_nullify),
        ("test_gather_unchecked_basic", test_gather_unchecked_basic),
        (
            "test_gather_succeeds_with_mixed_domains_and_uses_input_provenance",
            test_gather_succeeds_with_mixed_domains_and_uses_input_provenance,
        ),
        (
            "test_scatter_writes_source_rows_into_target_and_supports_negative_indices",
            test_scatter_writes_source_rows_into_target_and_supports_negative_indices,
        ),
        ("test_scatter_accepts_partial_scatter_map", test_scatter_accepts_partial_scatter_map),
        (
            "test_scatter_accepts_empty_scatter_map_as_target_copy",
            test_scatter_accepts_empty_scatter_map_as_target_copy,
        ),
        ("test_scatter_rejects_scatter_map_longer_than_source", test_scatter_rejects_scatter_map_longer_than_source),
        ("test_scatter_rejects_null_scatter_map", test_scatter_rejects_null_scatter_map),
        ("test_scatter_rejects_out_of_bounds_scatter_map", test_scatter_rejects_out_of_bounds_scatter_map),
        ("test_scatter_rejects_duplicate_scatter_map", test_scatter_rejects_duplicate_scatter_map),
        ("test_copy_if_else_basic", test_copy_if_else_basic),
        ("test_copy_if_else_treats_null_mask_as_false", test_copy_if_else_treats_null_mask_as_false),
        (
            "test_copy_if_else_succeeds_with_mixed_domains_and_uses_lhs_provenance",
            test_copy_if_else_succeeds_with_mixed_domains_and_uses_lhs_provenance,
        ),
    ]);
}
