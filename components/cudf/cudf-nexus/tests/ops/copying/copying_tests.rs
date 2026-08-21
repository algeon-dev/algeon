//! Tests for CU-7: gather (take) and conditional copy.
use cudf::{
    DataFrame, DeviceId, ErrorCode, ExecutionBudgetRequest, Series,
    core::{DataType, Table, TypeId},
    ops::copying,
    testing::prelude::*,
};

use crate::common::{
    column_from_slice, column_from_slice_nullable, explicit_bool_column, explicit_i32_column, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation, with_test_runtime,
};

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
fn test_fixed_width_dont_check_preflight_components_sum() {
    let preflight = copying::gather_fixed_width_dont_check_preflight_for_metadata(
        &[(DataType::new(TypeId::Int32), false), (DataType::new(TypeId::Float64), true)],
        5,
        5 * (std::mem::size_of::<i32>() + std::mem::size_of::<f64>()),
        DeviceId::DEFAULT,
    )
    .unwrap();

    assert_eq!(preflight.gather_map_bytes(), 5 * std::mem::size_of::<i32>());
    assert_eq!(preflight.output_frame_bytes(), 5 * (std::mem::size_of::<i32>() + std::mem::size_of::<f64>()));
    assert!(preflight.output_null_mask_bytes() > 0);
    assert_eq!(
        preflight.native_workspace_bytes(),
        preflight.target_mask_pointer_array_bytes()
            + preflight.source_table_device_view_bytes()
            + preflight.valid_count_array_bytes()
    );
    assert_eq!(
        preflight.active_phase_peak_bytes(),
        preflight.gather_map_bytes()
            + preflight.output_frame_bytes()
            + preflight.output_null_mask_bytes()
            + preflight.native_workspace_bytes()
    );
}

#[test]
fn test_fixed_width_dont_check_preflight_supports_bounded_string_and_rejects_nested_dtype_and_row_overflow() {
    let string = copying::gather_fixed_width_dont_check_preflight_for_metadata(
        &[(DataType::new(TypeId::String), false)],
        1,
        19,
        DeviceId::DEFAULT,
    )
    .unwrap();
    assert_eq!(string.output_frame_bytes(), 19);

    let underbound = copying::gather_fixed_width_dont_check_preflight_for_metadata(
        &[(DataType::new(TypeId::String), false)],
        1,
        7,
        DeviceId::DEFAULT,
    )
    .expect_err("a string output bound must cover the offsets allocation");
    assert_eq!(underbound.code(), ErrorCode::ForeignInvalidArgument);

    let decimal = copying::gather_fixed_width_dont_check_preflight_for_metadata(
        &[(DataType::fixed_point(TypeId::Decimal32, -2).unwrap(), false)],
        1,
        4,
        DeviceId::DEFAULT,
    )
    .expect_err("fixed-point dtype must be rejected before native sizing");
    assert_eq!(decimal.code(), ErrorCode::UnsupportedDtype);

    let unsupported = copying::gather_fixed_width_dont_check_preflight_for_metadata(
        &[(DataType::new(TypeId::List), false)],
        1,
        0,
        DeviceId::DEFAULT,
    )
    .expect_err("nested dtype must be rejected before native sizing");
    assert_eq!(unsupported.code(), ErrorCode::UnsupportedDtype);

    let overflow = copying::gather_fixed_width_dont_check_preflight_for_metadata(
        &[(DataType::new(TypeId::Int32), false)],
        i32::MAX as usize + 1,
        0,
        DeviceId::DEFAULT,
    )
    .expect_err("rows above cudf::size_type must be rejected before native sizing");
    assert_eq!(overflow.code(), ErrorCode::SizeOverflow);
}

#[test]
fn test_fixed_width_dont_check_preflight_zero_rows_retains_nullable_workspace() {
    let preflight = copying::gather_fixed_width_dont_check_preflight_for_metadata(
        &[(DataType::new(TypeId::Int32), true)],
        0,
        0,
        DeviceId::DEFAULT,
    )
    .unwrap();
    assert_eq!(preflight.gather_map_bytes(), 0);
    assert_eq!(preflight.output_frame_bytes(), 0);
    assert_eq!(preflight.output_null_mask_bytes(), 0);
    assert!(preflight.native_workspace_bytes() > 0);
    assert_eq!(
        preflight.native_workspace_bytes(),
        preflight.target_mask_pointer_array_bytes()
            + preflight.source_table_device_view_bytes()
            + preflight.valid_count_array_bytes()
    );
    assert_eq!(
        preflight.active_phase_peak_bytes(),
        preflight.gather_map_bytes()
            + preflight.output_frame_bytes()
            + preflight.output_null_mask_bytes()
            + preflight.native_workspace_bytes()
    );
}

#[test]
fn test_fixed_width_dont_check_preflight_matches_budgeted_slice_peak() {
    let (preflight, peak_bytes) = with_test_runtime(|runtime| {
        let input = runtime.column_from_slice_nullable(&[10i32, 20, 30, 40], &[true, false, true, true]).unwrap();
        let frame = DataFrame::from_series([("value", Series::from_column(input).unwrap())]).unwrap();
        let token = runtime
            .execution_budget_token(ExecutionBudgetRequest::new(1 << 20).with_boundary("gather_preflight_peak"))
            .unwrap();
        let preflight = copying::gather_fixed_width_dont_check_preflight_for_metadata(
            &[(DataType::new(TypeId::Int32), true)],
            3,
            3 * std::mem::size_of::<i32>(),
            runtime.device(),
        )
        .unwrap();
        let result = frame.with_execution(&token.execution_overrides()).slice(1, 3).unwrap();
        assert_eq!(result.num_rows(), 3);
        let peak_bytes = token.reservation_usage().unwrap().peak_bytes;
        (preflight, peak_bytes)
    });

    assert_eq!(peak_bytes, preflight.active_phase_peak_bytes() as u64);
}

#[test]
fn test_gather_out_of_bounds_nullify() {
    let col = column_from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.testing_view().unwrap();

    // Index 5 is out of bounds — should become null.
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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
