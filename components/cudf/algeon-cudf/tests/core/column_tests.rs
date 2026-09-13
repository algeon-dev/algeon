//! Tests for Column, ColumnView, and DeviceSlice.

use cudf::{
    DataType, ErrorCode, TypeId,
    core::{DeviceSlice as CudfDeviceSlice, column::DeviceSlice as CudfColumnDeviceSlice},
    ops::dictionary::{encode, indices},
    testing::{
        column::{
            column_view_to_fixed_width_host_bytes_for_test, column_view_to_raw_bytes_for_test,
            trigger_column_view_fixed_width_host_byte_range_overflow_for_test,
        },
        prelude::*,
    },
};

use crate::common::{
    column_from_device_ptr, column_from_slice, column_from_slice_nullable, column_from_strs, empty_column,
    error_context_value, fail_next_column_view_metadata_read, last_exec_mr_value, last_exec_stream_value, null_column,
    reset_exec_observation, run_named_cases,
};

fn test_create_empty_column() {
    let col = empty_column(TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 0);
    assert!(view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap(), DataType::new(TypeId::Int32));
}

fn test_create_empty_decimal_column_preserves_scale() {
    let dtype = DataType::fixed_point(TypeId::Decimal64, -2).unwrap();
    let col = empty_column(dtype).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 0);
    assert_eq!(view.data_type().unwrap(), dtype);
}

fn test_create_i32_column() {
    let data = vec![1i32, 2, 3, 4, 5];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();
    let null_count: usize = view.null_count().unwrap();

    assert_eq!(view.len().unwrap(), 5);
    assert!(!view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(null_count, 0);
    assert!(!view.nullable().unwrap());

    let host_data = view.to_vec::<i32>().unwrap();
    assert_eq!(host_data, data);
}

fn test_create_f64_column() {
    let data = vec![1.5f64, 2.5, 3.5];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.data_type().unwrap().id(), TypeId::Float64);

    let host_data = view.to_vec::<f64>().unwrap();
    assert_eq!(host_data, data);
}

#[test]
fn test_column_view_len_metadata_failure_maps_to_invariant_violation() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    fail_next_column_view_metadata_read();
    let err = view.len().unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "column_view::len");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_view_metadata"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_foreign_column_view_metadata"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign column view exposes valid dtype, size, null-count, child-count, and offset metadata")
    );
}

#[test]
fn test_to_host_validity_metadata_failure_preserves_top_level_operation() {
    let col = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    fail_next_column_view_metadata_read();
    let err = view.to_host_validity().unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "column_view::to_host_validity");
    assert!(
        err.call_chain().contains(&"column_view::len"),
        "expected helper call history in call_chain, got {:?}",
        err.call_chain()
    );
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_view_metadata"));
}

#[test]
fn test_to_host_null_mask_bytes_metadata_failure_preserves_top_level_operation() {
    let col = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    fail_next_column_view_metadata_read();
    let err = view.to_host_null_mask_bytes().unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "column_view::to_host_null_mask_bytes");
    assert!(
        err.call_chain().contains(&"column_view::len"),
        "expected helper call history in call_chain, got {:?}",
        err.call_chain()
    );
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_view_metadata"));
}

#[test]
fn test_column_view_null_count_metadata_failure_maps_to_invariant_violation() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    fail_next_column_view_metadata_read();
    let err = view.null_count().unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "column_view::null_count");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_view_metadata"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_foreign_column_view_metadata"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign column view exposes valid dtype, size, null-count, child-count, and offset metadata")
    );
}

fn test_nullable_column() {
    let data = vec![10i32, 20, 30, 40];
    let validity = vec![true, false, true, false];
    let col = column_from_slice_nullable(&data, &validity).unwrap();
    let view = col.testing_view().unwrap();
    let null_count: usize = view.null_count().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    assert!(view.nullable().unwrap());
    assert_eq!(null_count, 2);

    let host_data = view.to_vec::<i32>().unwrap();
    assert_eq!(host_data, data);

    let mask = view.to_host_validity().unwrap().unwrap();
    assert_eq!(mask, validity);
}

fn test_to_optional_vec_all_valid() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let host_data = view.to_optional_vec::<i32>().unwrap();
    assert_eq!(host_data, vec![Some(1), Some(2), Some(3)]);
}

fn test_to_optional_vec_with_nulls() {
    let col = column_from_slice_nullable(&[10i32, 20, 30, 40], &[true, false, true, false]).unwrap();
    let view = col.testing_view().unwrap();

    let host_data = view.to_optional_vec::<i32>().unwrap();
    assert_eq!(host_data, vec![Some(10), None, Some(30), None]);
}

fn test_to_optional_vec_bool_with_nulls() {
    let col = column_from_slice_nullable(&[true, false, true], &[true, false, true]).unwrap();
    let view = col.testing_view().unwrap();

    let host_data = view.to_optional_vec::<bool>().unwrap();
    assert_eq!(host_data, vec![Some(true), None, Some(true)]);
}

fn test_to_optional_vec_type_mismatch_error() {
    let col = column_from_slice(&[1.0f64, 2.0]).unwrap();
    let view = col.testing_view().unwrap();

    let err = view.to_optional_vec::<i32>().unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("float64"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
    assert_eq!(error_context_value(&err, "requested_type_id").as_deref(), None);
}

fn test_column_deep_copy() {
    let data = vec![100i64, 200, 300];
    let col = column_from_slice(&data).unwrap();
    let copy = col.deep_copy().unwrap();
    let view = copy.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i64>().unwrap(), data);
}

fn test_type_mismatch_error() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let result = view.to_vec::<f64>();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), cudf::ErrorKind::InvalidInput);
}

fn test_u8_column() {
    let data = vec![0u8, 128, 255];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.to_vec::<u8>().unwrap(), data);
}

fn test_mismatched_validity_length() {
    let data = vec![1i32, 2, 3];
    let validity = vec![true, false]; // wrong length
    let err = column_from_slice_nullable(&data, &validity).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "column::from_slice_nullable_with_runtime");
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "data_len").as_deref(), None);
    assert_eq!(error_context_value(&err, "validity_len").as_deref(), None);
}

fn test_to_host_validity_none_for_non_nullable_column() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();
    assert!(view.to_host_validity().unwrap().is_none());
}

fn test_to_host_validity_for_sliced_nullable_column_is_normalized_to_slice() {
    let col =
        column_from_slice_nullable(&[10i32, 20, 30, 40, 50, 60], &[true, false, true, true, false, true]).unwrap();
    let sliced = col.testing_slice_view(1, 4).unwrap();

    let validity = sliced.to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![false, true, true, false]);
}

fn test_testing_slice_view_uses_column_execution_domain_execution_resources() {
    let col = column_from_slice_nullable(&[10i32, 20, 30, 40], &[true, false, true, false]).unwrap();
    let expected_stream = col.testing_execution_domain_stream_value();
    let expected_mr = col.testing_execution_domain_mr_value();

    reset_exec_observation();
    let sliced = col.testing_slice_view(1, 2).unwrap();

    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(sliced.null_count().unwrap(), 1);
}

fn test_to_vec_for_sliced_view_starts_at_sliced_element_offset() {
    let col = column_from_slice(&[10i32, 20, 30, 40, 50, 60]).unwrap();
    let sliced = col.testing_slice_view(2, 3).unwrap();

    assert_eq!(sliced.to_vec::<i32>().unwrap(), vec![30, 40, 50]);
}

fn test_to_optional_vec_for_sliced_nullable_view_starts_at_sliced_element_offset() {
    let col =
        column_from_slice_nullable(&[10i32, 20, 30, 40, 50, 60], &[true, false, true, true, false, true]).unwrap();
    let sliced = col.testing_slice_view(1, 4).unwrap();

    assert_eq!(sliced.to_optional_vec::<i32>().unwrap(), vec![None, Some(30), Some(40), None]);
}

fn test_to_host_null_mask_bytes_for_sliced_nullable_column_is_packed_to_active_range() {
    let col =
        column_from_slice_nullable(&[10i32, 20, 30, 40, 50, 60], &[true, false, true, true, false, true]).unwrap();
    let sliced = col.testing_slice_view(1, 4).unwrap();

    let mask_bytes = sliced.to_host_null_mask_bytes().unwrap().unwrap();
    assert_eq!(mask_bytes, vec![0b0000_0110]);
}

fn test_various_numeric_types() {
    // i8
    let col = column_from_slice(&[1i8, -1, 127]).unwrap();
    assert_eq!(col.testing_view().unwrap().to_vec::<i8>().unwrap(), vec![1i8, -1, 127]);

    // i16
    let col = column_from_slice(&[1i16, -1, 32767]).unwrap();
    assert_eq!(col.testing_view().unwrap().to_vec::<i16>().unwrap(), vec![1i16, -1, 32767]);

    // u16
    let col = column_from_slice(&[0u16, 1000, 65535]).unwrap();
    assert_eq!(col.testing_view().unwrap().to_vec::<u16>().unwrap(), vec![0u16, 1000, 65535]);

    // u32
    let col = column_from_slice(&[0u32, 42, u32::MAX]).unwrap();
    assert_eq!(col.testing_view().unwrap().to_vec::<u32>().unwrap(), vec![0u32, 42, u32::MAX]);

    // u64
    let col = column_from_slice(&[0u64, 99]).unwrap();
    assert_eq!(col.testing_view().unwrap().to_vec::<u64>().unwrap(), vec![0u64, 99]);

    // f32
    let col = column_from_slice(&[1.0f32, -0.5, 3.125]).unwrap();
    let vals = col.testing_view().unwrap().to_vec::<f32>().unwrap();
    assert!((vals[2] - 3.125).abs() < 0.001);
}

fn test_device_data_ptr_i32() {
    let data = vec![10i32, 20, 30];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    let ptr = view.device_data_ptr().unwrap();
    assert!(!ptr.is_null(), "device data pointer for i32 column must be non-null");
}

fn test_device_data_ptr_f64() {
    let data = vec![1.5f64, 2.5, 3.5];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    let ptr = view.device_data_ptr().unwrap();
    assert!(!ptr.is_null(), "device data pointer for f64 column must be non-null");
}

fn test_device_data_ptr_consistent_with_to_vec() {
    // Two views of the same column should return the same device pointer.
    let col = column_from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let view1 = col.testing_view().unwrap();
    let view2 = col.testing_view().unwrap();

    let ptr1 = view1.device_data_ptr().unwrap();
    let ptr2 = view2.device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2, "same column should yield same device pointer");
}

fn test_offset_for_non_sliced_column() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.offset().unwrap(), 0);
}

fn test_device_data_ptr_empty_column() {
    let col = column_from_slice::<i32>(&[]).unwrap();
    let view = col.testing_view().unwrap();
    // Empty columns may or may not have a non-null pointer; just ensure no crash.
    let _ptr = view.device_data_ptr().unwrap();
}

fn test_child_rejects_out_of_bounds_index_without_segfault() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let err = match view.child(0) {
        Ok(_) => panic!("expected out-of-bounds child access to return an error"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(error_context_value(&err, "child_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "num_children").as_deref(), Some("0"));
    assert!(err.to_string().contains("out of bounds"), "unexpected error: {err}");
}

fn test_column_device_slice_type_mismatch_uses_canonical_type_context() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();

    let err = col.device_slice::<f64>().unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "column::device_slice");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("float64"));
    assert_eq!(err.detail_value("left_type_id"), None);
    assert_eq!(err.detail_value("right_type_id"), None);
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
    assert_eq!(error_context_value(&err, "requested_type_id").as_deref(), None);
}

fn test_child_rejects_usize_indices_outside_i32_range() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let err = view.child(usize::MAX).expect_err("expected oversized child index to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(err.operation(), "column_view::child");
    assert_eq!(error_context_value(&err, "child_index").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "num_children").as_deref(), Some("0"));
}

fn test_testing_slice_view_rejects_start_over_i32_max_with_row_range_contract() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();

    let err = col.testing_slice_view(usize::MAX, 1).expect_err("expected oversized slice start to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "testing::column::slice_view");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("start_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "length").as_deref(), None);
    assert_eq!(error_context_value(&err, "row_count").as_deref(), None);
}

fn test_testing_slice_view_rejects_row_count_over_i32_max_with_canonical_vocabulary() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let expected = usize::MAX.to_string();

    let err = col.testing_slice_view(0, usize::MAX).expect_err("expected oversized slice length to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "testing::column::slice_view");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some(expected.as_str()));
    assert_eq!(error_context_value(&err, "length").as_deref(), None);
}

fn test_testing_slice_view_rejects_end_overflow_with_row_range_context() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let expected_end = ((1usize as u128) + (usize::MAX as u128)).to_string();

    let err = col.testing_slice_view(1, usize::MAX).expect_err("expected overflowing slice end to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "testing::column::slice_view");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("end_overflowed_usize"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), Some(expected_end.as_str()));
    assert_eq!(error_context_value(&err, "length").as_deref(), None);
}

fn test_fixed_width_host_byte_range_overflow_hook_uses_row_count_contract() {
    let err = trigger_column_view_fixed_width_host_byte_range_overflow_for_test().unwrap_err();
    let expected = usize::MAX.to_string();

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "testing::column_view::to_fixed_width_host_bytes");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("host_byte_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("size_overflowed_usize"));
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some(expected.as_str()));
    assert_eq!(error_context_value(&err, "elem_size_bytes").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int16"));
    assert_eq!(error_context_value(&err, "len").as_deref(), None);
}

fn test_core_domain_modules_export_documented_surface() {
    use cudf::core::{
        column::{Column as DomainColumn, ColumnView as DomainColumnView, DeviceSlice as DomainDeviceSlice},
        error::{Error as DomainError, ErrorKind as DomainErrorKind, Result as DomainResult},
        scalar::Scalar as DomainScalar,
        table::{Table as DomainTable, TableView as DomainTableView},
        types::{CudfType as DomainCudfType, DataType as DomainDataType, TypeId as DomainTypeId},
    };

    fn assert_domain_result<T>(result: DomainResult<T>) -> DomainResult<T> {
        result
    }

    fn assert_cudf_type<T: DomainCudfType>() {}

    let _: Option<DomainColumn> = None;
    let _: Option<DomainColumnView<'static>> = None;
    let _: Option<DomainDeviceSlice<'static>> = None;
    let _: Option<DomainError> = None;
    let _: Option<DomainScalar> = None;
    let _: Option<DomainTable> = None;
    let _: Option<DomainTableView<'static>> = None;

    let _ = DomainErrorKind::InvalidInput;
    let _ = DomainDataType::new(DomainTypeId::Int32);
    let _ = assert_domain_result(column_from_slice(&[1i32, 2, 3]));
    assert_cudf_type::<i32>();
}

fn test_device_slice_stays_under_core_domain_module() {
    let _: Option<CudfColumnDeviceSlice<'static>> = None;
}

fn test_device_slice_is_reexported_at_crate_root() {
    let _: Option<CudfDeviceSlice<'static>> = None;
}

// --- DeviceSlice tests ---

fn test_device_slice_i32() {
    let col = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let view = col.testing_view().unwrap();

    let slice = view.device_slice::<i32>().unwrap();
    assert_eq!(slice.len(), 5);
    assert!(!slice.is_empty());
    assert_eq!(slice.type_id(), TypeId::Int32);
    assert_eq!(slice.elem_size(), 4);
    assert!(!slice.as_ptr().is_null());
}

fn test_device_slice_f64() {
    let col = column_from_slice(&[1.5f64, 2.5]).unwrap();
    let view = col.testing_view().unwrap();

    let slice = view.device_slice::<f64>().unwrap();
    assert_eq!(slice.len(), 2);
    assert_eq!(slice.type_id(), TypeId::Float64);
    assert_eq!(slice.elem_size(), 8);
}

fn test_device_slice_type_mismatch() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let err = view.device_slice::<f64>().unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
}

fn test_to_raw_bytes_rejects_non_byte_type_with_stable_contract() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();

    let err = column_view_to_raw_bytes_for_test(&view).expect_err("expected Int32 raw-byte export to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "column_view::to_raw_bytes");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
}

fn test_to_fixed_width_host_bytes_rejects_string_type_with_stable_contract() {
    let col = column_from_strs(&["alpha", "beta"]).unwrap();
    let view = col.testing_view().unwrap();

    let err = column_view_to_fixed_width_host_bytes_for_test(&view)
        .expect_err("expected STRING fixed-width host export to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "column_view::to_fixed_width_host_bytes");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
}

fn test_device_slice_split_at() {
    let col = column_from_slice(&[10i32, 20, 30, 40, 50, 60]).unwrap();
    let view = col.testing_view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    let (left, right) = slice.split_at(2);
    assert_eq!(left.len(), 2);
    assert_eq!(right.len(), 4);
    assert_eq!(left.type_id(), TypeId::Int32);
    assert_eq!(right.type_id(), TypeId::Int32);

    // Left pointer should equal the original pointer.
    assert_eq!(left.as_ptr(), slice.as_ptr());

    // Right pointer should be offset by 2 elements = 8 bytes.
    let expected_right = unsafe { (slice.as_ptr() as *const u8).add(2 * 4) as *const std::ffi::c_void };
    assert_eq!(right.as_ptr(), expected_right);
}

fn test_device_slice_split_at_zero() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    let (left, right) = slice.split_at(0);
    assert_eq!(left.len(), 0);
    assert_eq!(right.len(), 3);
}

fn test_device_slice_split_at_end() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    let (left, right) = slice.split_at(3);
    assert_eq!(left.len(), 3);
    assert_eq!(right.len(), 0);
}

#[test]
#[should_panic(expected = "split_at: mid (4) > len (3)")]
fn test_device_slice_split_at_out_of_bounds() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    slice.split_at(4); // should panic
}

fn test_device_slice_empty_column() {
    let col = column_from_slice::<i32>(&[]).unwrap();
    let view = col.testing_view().unwrap();

    let slice = view.device_slice::<i32>().unwrap();
    assert_eq!(slice.len(), 0);
    assert!(slice.is_empty());
}

fn test_device_slice_from_dictionary_indices() {
    // Dictionary-encode and get the indices as a DeviceSlice; the exact
    // workflow that gpu-app needs.
    let data = vec![10i32, 20, 10, 30, 20, 10];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    let dict = encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();
    let indices_view = indices(&dict_view).unwrap();

    let slice = indices_view.device_slice::<i32>().unwrap();
    assert_eq!(slice.len(), 6);
    assert_eq!(slice.type_id(), TypeId::Int32);
    assert!(!slice.as_ptr().is_null());

    // Split into two halves (simulating src/dst edge split).
    let (src, dst) = slice.split_at(3);
    assert_eq!(src.len(), 3);
    assert_eq!(dst.len(), 3);
}

// --- from_device_ptr tests ---

fn test_from_device_ptr_i32_roundtrip() {
    // Create a column from host data, get its device pointer, then create a
    // new column from that device pointer. Verify the data survives the round-trip.
    let original_data = vec![10i32, 20, 30, 40, 50];
    let src_col = column_from_slice(&original_data).unwrap();
    let src_view = src_col.testing_view().unwrap();

    let device_ptr = src_view.device_data_ptr().unwrap();
    let len = src_view.len().unwrap();

    let new_col = unsafe { column_from_device_ptr(device_ptr, len, TypeId::Int32).unwrap() };

    let new_view = new_col.testing_view().unwrap();
    assert_eq!(new_view.len().unwrap(), 5);
    assert_eq!(new_view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(new_view.null_count().unwrap(), 0);

    let host_data = new_view.to_vec::<i32>().unwrap();
    assert_eq!(host_data, original_data);
}

fn test_from_device_ptr_f64_roundtrip() {
    let original_data = vec![1.5f64, 2.5, 3.5, 4.5];
    let src_col = column_from_slice(&original_data).unwrap();
    let src_view = src_col.testing_view().unwrap();

    let device_ptr = src_view.device_data_ptr().unwrap();
    let len = src_view.len().unwrap();

    let new_col = unsafe { column_from_device_ptr(device_ptr, len, TypeId::Float64).unwrap() };

    let host_data = new_col.testing_view().unwrap().to_vec::<f64>().unwrap();
    assert_eq!(host_data, original_data);
}

fn test_from_device_ptr_i64_roundtrip() {
    let original_data = vec![100i64, 200, 300];
    let src_col = column_from_slice(&original_data).unwrap();
    let src_view = src_col.testing_view().unwrap();

    let device_ptr = src_view.device_data_ptr().unwrap();
    let len = src_view.len().unwrap();

    let new_col = unsafe { column_from_device_ptr(device_ptr, len, TypeId::Int64).unwrap() };

    let host_data = new_col.testing_view().unwrap().to_vec::<i64>().unwrap();
    assert_eq!(host_data, original_data);
}

fn test_from_device_ptr_independent_of_source() {
    // The new column should own its own memory; dropping the source should not
    // invalidate the new column (device-to-device copy, not zero-copy).
    let original_data = vec![42i32, 99, 7, 13];

    let new_col = {
        let src_col = column_from_slice(&original_data).unwrap();
        let src_view = src_col.testing_view().unwrap();
        let device_ptr = src_view.device_data_ptr().unwrap();
        let len = src_view.len().unwrap();

        unsafe { column_from_device_ptr(device_ptr, len, TypeId::Int32).unwrap() }
        // src_col is dropped here
    };

    // The new column should still be readable.
    let host_data = new_col.testing_view().unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host_data, original_data);
}

fn test_from_device_ptr_empty() {
    // Edge case: empty column (len = 0). The device pointer may be anything;
    // it should not be dereferenced.
    let src_col = column_from_slice::<i32>(&[]).unwrap();
    let src_view = src_col.testing_view().unwrap();
    let device_ptr = src_view.device_data_ptr().unwrap();

    let new_col = unsafe { column_from_device_ptr(device_ptr, 0, TypeId::Int32).unwrap() };

    let new_view = new_col.testing_view().unwrap();
    assert_eq!(new_view.len().unwrap(), 0);
    assert!(new_view.is_empty().unwrap());
    assert_eq!(new_view.data_type().unwrap().id(), TypeId::Int32);
}

fn test_from_device_ptr_null_pointer_with_zero_len() {
    // Null pointer with len 0 should succeed (no data to copy).
    let new_col = unsafe { column_from_device_ptr(std::ptr::null(), 0, TypeId::Int32).unwrap() };

    let new_view = new_col.testing_view().unwrap();
    assert_eq!(new_view.len().unwrap(), 0);
}

fn test_from_device_ptr_null_pointer_with_nonzero_len() {
    // Null pointer with len > 0 should fail.
    let result = unsafe { column_from_device_ptr(std::ptr::null(), 5, TypeId::Int32) };
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "column::from_device_ptr_with_runtime");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("device_ptr"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("null_pointer_with_positive_row_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure device_ptr is non-null whenever row_count is positive"),
    );
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some("5"));
    assert_eq!(error_context_value(&err, "len").as_deref(), None);
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
}

fn test_from_device_ptr_various_types() {
    // i8
    let data_i8 = vec![1i8, -1, 127];
    let col = column_from_slice(&data_i8).unwrap();
    let v = col.testing_view().unwrap();
    let new_col =
        unsafe { column_from_device_ptr(v.device_data_ptr().unwrap(), v.len().unwrap(), TypeId::Int8).unwrap() };
    assert_eq!(new_col.testing_view().unwrap().to_vec::<i8>().unwrap(), data_i8);

    // u32
    let data_u32 = vec![0u32, 42, u32::MAX];
    let col = column_from_slice(&data_u32).unwrap();
    let v = col.testing_view().unwrap();
    let new_col =
        unsafe { column_from_device_ptr(v.device_data_ptr().unwrap(), v.len().unwrap(), TypeId::UInt32).unwrap() };
    assert_eq!(new_col.testing_view().unwrap().to_vec::<u32>().unwrap(), data_u32);

    // f32
    let data_f32 = vec![1.0f32, -0.5, 3.125];
    let col = column_from_slice(&data_f32).unwrap();
    let v = col.testing_view().unwrap();
    let new_col =
        unsafe { column_from_device_ptr(v.device_data_ptr().unwrap(), v.len().unwrap(), TypeId::Float32).unwrap() };
    let result = new_col.testing_view().unwrap().to_vec::<f32>().unwrap();
    assert_eq!(result, data_f32);
}

fn test_from_device_ptr_rejects_non_fixed_width_type() {
    // from_device_ptr must only accept fixed-width primitive types.
    let result = unsafe { column_from_device_ptr(std::ptr::null(), 0, TypeId::String) };

    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "column::from_device_ptr_with_runtime");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("requires_fixed_width_type"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("string"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use a fixed-width type when importing a raw device pointer as a cudf column"),
    );
}

// --- null_column() tests ---

fn test_nulls_int32_empty() {
    let col = null_column(TypeId::Int32, 0).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 0);
    assert_eq!(view.data_type().unwrap().id(), TypeId::Int32);
}

fn test_nulls_int64() {
    let col = null_column(TypeId::Int64, 100).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 100);
    assert_eq!(view.null_count().unwrap(), 100);
    let validity = view.to_host_validity().unwrap().unwrap();
    assert!(validity.iter().all(|&v| !v));
}

fn test_nulls_float32() {
    let col = null_column(TypeId::Float32, 50).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 50);
    assert_eq!(view.null_count().unwrap(), 50);
    let validity = view.to_host_validity().unwrap().unwrap();
    assert!(validity.iter().all(|&v| !v));
}

fn test_nulls_bool8() {
    let col = null_column(TypeId::Bool8, 10).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 10);
    assert_eq!(view.null_count().unwrap(), 10);
}

fn test_nulls_string() {
    let col = null_column(TypeId::String, 25).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 25);
    assert_eq!(view.null_count().unwrap(), 25);
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);
}

fn test_nulls_decimal64_preserve_scale() {
    let dtype = DataType::fixed_point(TypeId::Decimal64, -2).unwrap();
    let col = null_column(dtype, 5).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 5);
    assert_eq!(view.null_count().unwrap(), 5);
    assert_eq!(view.data_type().unwrap(), dtype);
}

fn test_nulls_preserves_type() {
    for type_id in [
        TypeId::Int8,
        TypeId::Int16,
        TypeId::Int32,
        TypeId::Int64,
        TypeId::Float32,
        TypeId::Float64,
        TypeId::Bool8,
        TypeId::String,
    ] {
        let col = null_column(type_id, 5).unwrap();
        let view = col.testing_view().unwrap();
        assert_eq!(view.data_type().unwrap().id(), type_id);
    }
}

fn test_nulls_rejects_length_over_i32_max() {
    let result = null_column(TypeId::Int8, (i32::MAX as usize) + 1);
    assert!(result.is_err());
    let err = match result {
        Ok(_) => panic!("expected len > i32::MAX to be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "column::nulls_with_runtime");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "len").as_deref(), None);
    let expected = ((i32::MAX as usize) + 1).to_string();
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some(expected.as_str()));
}

fn test_nulls_rejects_list_type_with_stable_contract() {
    let err = null_column(TypeId::List, 5).expect_err("expected List null-column construction to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "column::nulls_with_runtime");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("requires_fixed_width_type"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("list"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use a fixed-width type or STRING when constructing an all-null column"),
    );
}

#[test]
fn column_tests_cases() {
    run_named_cases(&[
        ("test_create_empty_column", test_create_empty_column),
        ("test_create_empty_decimal_column_preserves_scale", test_create_empty_decimal_column_preserves_scale),
        ("test_create_i32_column", test_create_i32_column),
        ("test_create_f64_column", test_create_f64_column),
        ("test_nullable_column", test_nullable_column),
        ("test_to_optional_vec_all_valid", test_to_optional_vec_all_valid),
        ("test_to_optional_vec_with_nulls", test_to_optional_vec_with_nulls),
        ("test_to_optional_vec_bool_with_nulls", test_to_optional_vec_bool_with_nulls),
        ("test_to_optional_vec_type_mismatch_error", test_to_optional_vec_type_mismatch_error),
        ("test_column_deep_copy", test_column_deep_copy),
        ("test_type_mismatch_error", test_type_mismatch_error),
        ("test_u8_column", test_u8_column),
        ("test_mismatched_validity_length", test_mismatched_validity_length),
        ("test_to_host_validity_none_for_non_nullable_column", test_to_host_validity_none_for_non_nullable_column),
        (
            "test_to_host_validity_for_sliced_nullable_column_is_normalized_to_slice",
            test_to_host_validity_for_sliced_nullable_column_is_normalized_to_slice,
        ),
        (
            "test_testing_slice_view_uses_column_execution_domain_execution_resources",
            test_testing_slice_view_uses_column_execution_domain_execution_resources,
        ),
        (
            "test_to_vec_for_sliced_view_starts_at_sliced_element_offset",
            test_to_vec_for_sliced_view_starts_at_sliced_element_offset,
        ),
        (
            "test_to_optional_vec_for_sliced_nullable_view_starts_at_sliced_element_offset",
            test_to_optional_vec_for_sliced_nullable_view_starts_at_sliced_element_offset,
        ),
        (
            "test_to_host_null_mask_bytes_for_sliced_nullable_column_is_packed_to_active_range",
            test_to_host_null_mask_bytes_for_sliced_nullable_column_is_packed_to_active_range,
        ),
        ("test_various_numeric_types", test_various_numeric_types),
        ("test_device_data_ptr_i32", test_device_data_ptr_i32),
        ("test_device_data_ptr_f64", test_device_data_ptr_f64),
        ("test_device_data_ptr_consistent_with_to_vec", test_device_data_ptr_consistent_with_to_vec),
        ("test_offset_for_non_sliced_column", test_offset_for_non_sliced_column),
        ("test_device_data_ptr_empty_column", test_device_data_ptr_empty_column),
        (
            "test_child_rejects_out_of_bounds_index_without_segfault",
            test_child_rejects_out_of_bounds_index_without_segfault,
        ),
        (
            "test_column_device_slice_type_mismatch_uses_canonical_type_context",
            test_column_device_slice_type_mismatch_uses_canonical_type_context,
        ),
        ("test_child_rejects_usize_indices_outside_i32_range", test_child_rejects_usize_indices_outside_i32_range),
        (
            "test_testing_slice_view_rejects_start_over_i32_max_with_row_range_contract",
            test_testing_slice_view_rejects_start_over_i32_max_with_row_range_contract,
        ),
        (
            "test_testing_slice_view_rejects_row_count_over_i32_max_with_canonical_vocabulary",
            test_testing_slice_view_rejects_row_count_over_i32_max_with_canonical_vocabulary,
        ),
        (
            "test_testing_slice_view_rejects_end_overflow_with_row_range_context",
            test_testing_slice_view_rejects_end_overflow_with_row_range_context,
        ),
        (
            "test_fixed_width_host_byte_range_overflow_hook_uses_row_count_contract",
            test_fixed_width_host_byte_range_overflow_hook_uses_row_count_contract,
        ),
        ("test_core_domain_modules_export_documented_surface", test_core_domain_modules_export_documented_surface),
        ("test_device_slice_stays_under_core_domain_module", test_device_slice_stays_under_core_domain_module),
        ("test_device_slice_is_reexported_at_crate_root", test_device_slice_is_reexported_at_crate_root),
        ("test_device_slice_i32", test_device_slice_i32),
        ("test_device_slice_f64", test_device_slice_f64),
        ("test_device_slice_type_mismatch", test_device_slice_type_mismatch),
        (
            "test_to_raw_bytes_rejects_non_byte_type_with_stable_contract",
            test_to_raw_bytes_rejects_non_byte_type_with_stable_contract,
        ),
        (
            "test_to_fixed_width_host_bytes_rejects_string_type_with_stable_contract",
            test_to_fixed_width_host_bytes_rejects_string_type_with_stable_contract,
        ),
        ("test_device_slice_split_at", test_device_slice_split_at),
        ("test_device_slice_split_at_zero", test_device_slice_split_at_zero),
        ("test_device_slice_split_at_end", test_device_slice_split_at_end),
        ("test_device_slice_empty_column", test_device_slice_empty_column),
        ("test_device_slice_from_dictionary_indices", test_device_slice_from_dictionary_indices),
        ("test_from_device_ptr_i32_roundtrip", test_from_device_ptr_i32_roundtrip),
        ("test_from_device_ptr_f64_roundtrip", test_from_device_ptr_f64_roundtrip),
        ("test_from_device_ptr_i64_roundtrip", test_from_device_ptr_i64_roundtrip),
        ("test_from_device_ptr_independent_of_source", test_from_device_ptr_independent_of_source),
        ("test_from_device_ptr_empty", test_from_device_ptr_empty),
        ("test_from_device_ptr_null_pointer_with_zero_len", test_from_device_ptr_null_pointer_with_zero_len),
        ("test_from_device_ptr_null_pointer_with_nonzero_len", test_from_device_ptr_null_pointer_with_nonzero_len),
        ("test_from_device_ptr_various_types", test_from_device_ptr_various_types),
        ("test_from_device_ptr_rejects_non_fixed_width_type", test_from_device_ptr_rejects_non_fixed_width_type),
        ("test_nulls_int32_empty", test_nulls_int32_empty),
        ("test_nulls_int64", test_nulls_int64),
        ("test_nulls_float32", test_nulls_float32),
        ("test_nulls_bool8", test_nulls_bool8),
        ("test_nulls_string", test_nulls_string),
        ("test_nulls_decimal64_preserve_scale", test_nulls_decimal64_preserve_scale),
        ("test_nulls_preserves_type", test_nulls_preserves_type),
        ("test_nulls_rejects_length_over_i32_max", test_nulls_rejects_length_over_i32_max),
        ("test_nulls_rejects_list_type_with_stable_contract", test_nulls_rejects_list_type_with_stable_contract),
    ]);
}
