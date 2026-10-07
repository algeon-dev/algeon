//! Integration tests for DeviceBuffer and zero-copy Column adoption.
use cudf::{
    ErrorCode, ErrorKind, Runtime, TypeId,
    core::error::{FactKeyName, FactValue},
    testing::prelude::*,
};

use crate::common::{
    column_from_device_buffer, column_from_device_buffer_nullable, error_context_value, run_named_cases,
};

// ---------------------------------------------------------------------------
// DeviceBuffer lifecycle
// ---------------------------------------------------------------------------

fn new_allocates_requested_size() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(1024).unwrap();
    assert_eq!(buf.size_bytes().unwrap(), 1024);
}

fn new_zero_bytes_succeeds() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(0).unwrap();
    assert!(buf.is_empty().unwrap());
    assert_eq!(buf.size_bytes().unwrap(), 0);
}

fn const_and_mut_pointers_are_non_null_for_non_empty_buffer() {
    let rt = Runtime::new().unwrap();
    let mut buf = rt.device_buffer(64).unwrap();
    let const_ptr = buf.as_ptr().unwrap();
    assert!(!const_ptr.is_null());
    assert_eq!(buf.as_mut_ptr().unwrap().cast_const(), const_ptr);
}

fn const_pointer_for_zero_size_succeeds_even_if_pointer_is_null() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(0).unwrap();

    let _ptr = buf.as_ptr().unwrap();

    assert!(buf.is_empty().unwrap());
}

fn device_buffer_reports_owning_runtime_binding() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(64).unwrap();

    assert_eq!(buf.device(), rt.device());
    assert_eq!(buf.placement(), rt.placement());
    assert_eq!(buf.binding_id(), rt.binding_id());
}

fn float32_device_buffer_finite_proof_covers_the_exact_allocation_and_binding() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer_from_host_data(&[1.0f32, 2.0, 3.0]).unwrap();
    let raw = buf.as_ptr().unwrap();

    let validated = buf.validate_finite_float32().unwrap();

    assert_eq!(validated.len(), 3);
    assert!(!validated.is_empty());
    assert_eq!(validated.as_ptr().unwrap(), raw);
    assert_eq!(validated.binding_id(), rt.binding_id());
    assert_eq!(validated.placement(), rt.placement());
}

fn float32_device_buffer_finite_proof_rejects_nan_and_infinities() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let rt = Runtime::new().unwrap();
        let buf = rt.device_buffer_from_host_data(&[1.0f32, invalid, 3.0]).unwrap();

        let error = buf.validate_finite_float32().expect_err("non-finite raw buffer must be rejected");

        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(error.operation(), "device_buffer::validate_finite_float32");
        assert_eq!(
            error.fact_value(FactKeyName::ValidationReason),
            Some(&FactValue::Str("non_finite_vector_value".into()))
        );
        assert_eq!(error.fact_value(FactKeyName::ExecutionBindingId), Some(&FactValue::U64(rt.binding_id().get())));
        assert_eq!(error.fact_value(FactKeyName::ControlPlaneScalarBytes), Some(&FactValue::Usize(4)));
        assert_eq!(error.detail_value("non_finite_count"), Some("1"));
        assert_eq!(error.detail_value("validated_values"), Some("3"));
    }
}

fn typed_device_buffer_validation_rejects_misaligned_allocation_size() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(3).unwrap();

    let error = buf.validate_finite_float32().expect_err("unaligned raw buffer must be rejected");

    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error.detail_value("reason"), Some("validation_buffer_size_misaligned"));
}

fn typed_device_buffer_validation_accepts_empty_allocation_with_bounded_proofs() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(0).unwrap();

    {
        let finite = buf.validate_finite_float32().unwrap();
        assert!(finite.is_empty());
    }

    let range = buf.validate_int64_range(0, 1).unwrap();
    assert!(range.is_empty());
}

fn int64_device_buffer_range_proof_covers_the_exact_allocation_and_binding() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer_from_host_data(&[0i64, 2, 4]).unwrap();
    let raw = buf.as_ptr().unwrap();

    let validated = buf.validate_int64_range(0, 5).unwrap();

    assert_eq!(validated.len(), 3);
    assert_eq!(validated.min_inclusive(), 0);
    assert_eq!(validated.max_exclusive(), 5);
    assert_eq!(validated.as_ptr().unwrap(), raw);
    assert_eq!(validated.binding_id(), rt.binding_id());
    assert_eq!(validated.placement(), rt.placement());
}

fn int64_device_buffer_range_proof_rejects_both_bound_violations() {
    for values in [[-1i64, 0, 1], [0i64, 1, 3]] {
        let rt = Runtime::new().unwrap();
        let buf = rt.device_buffer_from_host_data(&values).unwrap();

        let error = buf.validate_int64_range(0, 3).expect_err("out-of-range raw buffer must be rejected");

        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(
            error.fact_value(FactKeyName::ValidationReason),
            Some(&FactValue::Str("integer_value_out_of_range".into()))
        );
        assert_eq!(error.fact_value(FactKeyName::ExecutionBindingId), Some(&FactValue::U64(rt.binding_id().get())));
        assert_eq!(error.fact_value(FactKeyName::ControlPlaneScalarBytes), Some(&FactValue::Usize(4)));
        assert_eq!(error.detail_value("out_of_range_count"), Some("1"));
        assert_eq!(error.detail_value("min_inclusive"), Some("0"));
        assert_eq!(error.detail_value("max_exclusive"), Some("3"));
    }
}

fn int64_device_buffer_range_proof_rejects_empty_interval_before_execution() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer_from_host_data(&[1i64]).unwrap();

    let error = buf.validate_int64_range(2, 2).expect_err("empty validation range must be rejected");

    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error.detail_value("reason"), Some("invalid_validation_range"));
}

fn float32_device_buffer_add_scalar_preserves_infinite_slots() {
    let rt = Runtime::new().unwrap();
    let mut values = rt.device_buffer_from_host_data(&[-0.75_f32, 1.5, f32::INFINITY, f32::NEG_INFINITY]).unwrap();
    values.add_scalar_to_float32(1.0).unwrap();
    let values = rt.column_from_device_buffer(values, 4, TypeId::Float32).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(values, [0.25, 2.5, f32::INFINITY, f32::NEG_INFINITY]);
    let mut invalid = rt.device_buffer_from_host_data(&[0.0_f32]).unwrap();
    assert_eq!(invalid.add_scalar_to_float32(f32::NAN).unwrap_err().code(), ErrorCode::ValueOutOfRange);
}

fn buffer_sentinel_operations_reject_out_of_range_int64_and_non_finite_float32() {
    let rt = Runtime::new().unwrap();
    let mut neighbors = rt.device_buffer_from_host_data(&[0_i64, 0, 2]).unwrap();
    let distances = rt.device_buffer_from_host_data(&[0.0_f32, f32::INFINITY, 1.0]).unwrap();
    neighbors.fill_int64_where_float32_infinite(&distances, i64::MAX).unwrap();
    neighbors.validate_int64_range_or_sentinel(0, 3, i64::MAX).unwrap();
    distances.validate_finite_float32_unless_int64_equals(&neighbors, i64::MAX).unwrap();
    let neighbors = rt.column_from_device_buffer(neighbors, 3, TypeId::Int64).unwrap().to_vec::<i64>().unwrap();
    assert_eq!(neighbors, [0, i64::MAX, 2]);

    let invalid = rt.device_buffer_from_host_data(&[-1_i64, 3, i64::MAX]).unwrap();
    let error = invalid.validate_int64_range_or_sentinel(0, 3, i64::MAX).unwrap_err();
    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    let distances = rt.device_buffer_from_host_data(&[0.0_f32, f32::NAN, f32::INFINITY]).unwrap();
    let valid = rt.device_buffer_from_host_data(&[0_i64, 1, i64::MAX]).unwrap();
    let error = distances.validate_finite_float32_unless_int64_equals(&valid, i64::MAX).unwrap_err();
    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
}

fn from_host_data_creates_correct_size() {
    let rt = Runtime::new().unwrap();
    let data: &[i32] = &[1, 2, 3];
    let buf = rt.device_buffer_from_host_data(data).unwrap();
    assert_eq!(buf.size_bytes().unwrap(), 12); // 3 * 4 bytes
}

fn from_host_data_empty_slice() {
    let rt = Runtime::new().unwrap();
    let data: &[i32] = &[];
    let buf = rt.device_buffer_from_host_data(data).unwrap();
    assert!(buf.is_empty().unwrap());
    assert_eq!(buf.size_bytes().unwrap(), 0);
}

// ---------------------------------------------------------------------------
// Zero-copy adoption; non-nullable
// ---------------------------------------------------------------------------

fn adopt_i32_roundtrip() {
    let rt = Runtime::new().unwrap();
    let data: &[i32] = &[10, 20, 30];
    let buf = rt.device_buffer_from_host_data(data).unwrap();
    let col = column_from_device_buffer(buf, 3, TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

fn adopt_f64_roundtrip() {
    let rt = Runtime::new().unwrap();
    let data: &[f64] = &[1.5, 2.5, 3.5];
    let buf = rt.device_buffer_from_host_data(data).unwrap();
    let col = column_from_device_buffer(buf, 3, TypeId::Float64).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.to_vec::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

fn adopted_bool8_decodes_non_canonical_non_zero_bytes() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer_from_host_data(&[0u8, 1, 2, 0xff]).unwrap();
    let col = rt.column_from_device_buffer(buf, 4, TypeId::Bool8).unwrap();

    assert_eq!(col.to_vec::<bool>().unwrap(), vec![false, true, true, true]);
}

fn adopted_nullable_bool8_never_materializes_null_slot_as_bool() {
    let rt = Runtime::new().unwrap();
    let data_buf = rt.device_buffer_from_host_data(&[0u8, 0xff, 2]).unwrap();
    let mut mask = [0u8; 64];
    mask[0] = 0b101;
    let mask_buf = rt.device_buffer_from_host_data(&mask).unwrap();
    let col = rt.column_from_device_buffer_nullable(data_buf, mask_buf, 1, 3, TypeId::Bool8).unwrap();

    assert_eq!(col.to_optional_vec::<bool>().unwrap(), vec![Some(false), None, Some(true)]);
}

fn adopt_proves_zero_copy() {
    let rt = Runtime::new().unwrap();
    let data: &[i32] = &[42, 99];
    let buf = rt.device_buffer_from_host_data(data).unwrap();
    let original_ptr = buf.as_ptr().unwrap();
    let col = column_from_device_buffer(buf, 2, TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.device_data_ptr().unwrap(), original_ptr);
}

fn adopt_zero_elements() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(0).unwrap();
    let col = column_from_device_buffer(buf, 0, TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.len().unwrap(), 0);
    assert!(view.is_empty().unwrap());
}

fn adopt_rejects_string_type() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(64).unwrap();
    let err = column_from_device_buffer(buf, 4, TypeId::String).expect_err("expected error");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "column::from_device_buffer_with_runtime");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("requires_fixed_width_type"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use a fixed-width type when adopting a device buffer as a cudf column"),
    );
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("string"));
}

fn runtime_adopt_i32_roundtrip_seeds_explicit_residency_without_copy() {
    let data: &[i32] = &[10, 20, 30];
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer_from_host_data(data).unwrap();
    let original_ptr = buf.as_ptr().unwrap();

    let col = rt.column_from_device_buffer(buf, 3, TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(view.device_data_ptr().unwrap(), original_ptr);
    assert_eq!(col.testing_execution_domain_kind(), "explicit");
    assert_eq!(col.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(col.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
}

fn adopt_rejects_size_mismatch() {
    // 12 bytes = 3 i32 elements, but claim 4 rows
    let rt = Runtime::new().unwrap();
    let data: &[i32] = &[1, 2, 3];
    let buf = rt.device_buffer_from_host_data(data).unwrap();
    let err = column_from_device_buffer(buf, 4, TypeId::Int32).expect_err("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("data_buffer"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("size_mismatch_with_row_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure data_buffer has exactly row_count * element_size bytes"),
    );
    assert_eq!(error_context_value(&err, "actual_bytes").as_deref(), Some("12"));
    assert_eq!(error_context_value(&err, "expected_bytes").as_deref(), Some("16"));
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some("4"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "buffer_bytes").as_deref(), None);
}

fn adopt_rejects_row_count_exceeding_i32_max() {
    let rt = Runtime::new().unwrap();
    let buf = rt.device_buffer(0).unwrap();
    let err = column_from_device_buffer(buf, (i32::MAX as usize) + 1, TypeId::Int32).expect_err("expected error");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "column::from_device_buffer_with_runtime");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row_count to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "len").as_deref(), None);
    let expected = ((i32::MAX as usize) + 1).to_string();
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some(expected.as_str()));
}

// ---------------------------------------------------------------------------
// Zero-copy adoption; nullable
// ---------------------------------------------------------------------------

fn adopt_nullable_with_nulls() {
    let rt = Runtime::new().unwrap();
    // Data: [10, 20, 30, 40] with validity [true, false, true, false] → null_count = 2
    let data: &[i32] = &[10, 20, 30, 40];
    let data_buf = rt.device_buffer_from_host_data(data).unwrap();

    // Build a bitmask: bit 0 and bit 2 set → 0b0000_0101 = 0x05
    // cudf requires 64-byte aligned bitmask buffers.
    let mut mask = [0u8; 64];
    mask[0] = 0x05;
    let mask_buf = rt.device_buffer_from_host_data(&mask).unwrap();

    let col = column_from_device_buffer_nullable(data_buf, mask_buf, 2, 4, TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 2);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30, 40]);

    let validity = view.to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true, false]);
}

fn adopt_nullable_all_valid() {
    let rt = Runtime::new().unwrap();
    let data: &[i32] = &[1, 2, 3];
    let data_buf = rt.device_buffer_from_host_data(data).unwrap();

    // All valid: bits 0,1,2 set → 0b0000_0111 = 0x07
    // cudf requires 64-byte aligned bitmask buffers.
    let mut mask = [0u8; 64];
    mask[0] = 0x07;
    let mask_buf = rt.device_buffer_from_host_data(&mask).unwrap();

    let col = column_from_device_buffer_nullable(data_buf, mask_buf, 0, 3, TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();
    assert_eq!(view.null_count().unwrap(), 0);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

fn runtime_adopt_nullable_roundtrip_seeds_explicit_residency_without_copy() {
    let data: &[i32] = &[10, 20, 30, 40];
    let mut mask = [0u8; 64];
    mask[0] = 0x05;
    let rt = Runtime::new().unwrap();
    let data_buf = rt.device_buffer_from_host_data(data).unwrap();
    let original_ptr = data_buf.as_ptr().unwrap();
    let mask_buf = rt.device_buffer_from_host_data(&mask).unwrap();

    let col = rt.column_from_device_buffer_nullable(data_buf, mask_buf, 2, 4, TypeId::Int32).unwrap();
    let view = col.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    assert_eq!(view.null_count().unwrap(), 2);
    assert_eq!(view.to_host_validity().unwrap().unwrap(), vec![true, false, true, false]);
    assert_eq!(view.device_data_ptr().unwrap(), original_ptr);
    assert_eq!(col.testing_execution_domain_kind(), "explicit");
    assert_eq!(col.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(col.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
}

fn adopt_nullable_rejects_null_count_exceeding_row_count() {
    let rt = Runtime::new().unwrap();
    let data_buf = rt.device_buffer_from_host_data(&[1i32, 2]).unwrap();
    let mask_buf = rt.device_buffer_from_host_data(&[0u8; 64]).unwrap();

    let err = column_from_device_buffer_nullable(data_buf, mask_buf, 5, 2, TypeId::Int32).expect_err("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("5"));
}

fn adopt_nullable_rejects_null_count_exceeding_i32_max_with_stable_contract() {
    let rt = Runtime::new().unwrap();
    let data_buf = rt.device_buffer(0).unwrap();
    let mask_buf = rt.device_buffer(0).unwrap();
    let err = column_from_device_buffer_nullable(data_buf, mask_buf, (i32::MAX as usize) + 1, 0, TypeId::Int32)
        .expect_err("expected error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "column::from_device_buffer_nullable_with_runtime");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("null_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce null_count to fit within i32::MAX"));
    let expected = ((i32::MAX as usize) + 1).to_string();
    assert_eq!(error_context_value(&err, "null_count").as_deref(), Some(expected.as_str()));
}

fn adopt_nullable_rejects_mask_too_small() {
    // 4 rows need at least 1 byte of bitmask. Provide 0 bytes.
    let rt = Runtime::new().unwrap();
    let data_buf = rt.device_buffer_from_host_data(&[1i32, 2, 3, 4]).unwrap();
    let mask_buf = rt.device_buffer(0).unwrap();

    let err = column_from_device_buffer_nullable(data_buf, mask_buf, 1, 4, TypeId::Int32).expect_err("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("null_mask_buffer"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("insufficient_allocation_size"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("provide at least cudf_null_mask_allocation_size_bytes(row_count) bytes in null_mask_buffer"),
    );
    assert_eq!(error_context_value(&err, "actual_bytes").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "expected_bytes").as_deref(), Some("64"));
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some("4"));
    assert_eq!(error_context_value(&err, "mask_bytes").as_deref(), None);
    assert_eq!(error_context_value(&err, "min_required_bytes").as_deref(), None);
}

fn adopt_nullable_rejects_mask_smaller_than_cudf_allocation_size() {
    let rt = Runtime::new().unwrap();
    let data_buf = rt.device_buffer_from_host_data(&[1i32]).unwrap();
    let mask_buf = rt.device_buffer_from_host_data(&[0xffu8; 63]).unwrap();

    let err = column_from_device_buffer_nullable(data_buf, mask_buf, 0, 1, TypeId::Int32).expect_err("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("null_mask_buffer"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("insufficient_allocation_size"));
    assert_eq!(error_context_value(&err, "actual_bytes").as_deref(), Some("63"));
    assert_eq!(error_context_value(&err, "expected_bytes").as_deref(), Some("64"));
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some("1"));
}

fn adopt_nullable_rejects_data_buffer_size_mismatch_with_stable_contract() {
    let rt = Runtime::new().unwrap();
    let data_buf = rt.device_buffer_from_host_data(&[1i32, 2, 3]).unwrap();
    let mask_buf = rt.device_buffer_from_host_data(&[0xffu8; 64]).unwrap();

    let err = column_from_device_buffer_nullable(data_buf, mask_buf, 0, 4, TypeId::Int32).expect_err("expected error");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "column::from_device_buffer_nullable_with_runtime");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("data_buffer"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("size_mismatch_with_row_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure data_buffer has exactly row_count * element_size bytes"),
    );
    assert_eq!(error_context_value(&err, "actual_bytes").as_deref(), Some("12"));
    assert_eq!(error_context_value(&err, "expected_bytes").as_deref(), Some("16"));
    assert_eq!(error_context_value(&err, "row_count").as_deref(), Some("4"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
}

#[test]
fn device_buffer_tests_cases() {
    run_named_cases(&[
        ("new_allocates_requested_size", new_allocates_requested_size),
        ("new_zero_bytes_succeeds", new_zero_bytes_succeeds),
        (
            "const_and_mut_pointers_are_non_null_for_non_empty_buffer",
            const_and_mut_pointers_are_non_null_for_non_empty_buffer,
        ),
        (
            "const_pointer_for_zero_size_succeeds_even_if_pointer_is_null",
            const_pointer_for_zero_size_succeeds_even_if_pointer_is_null,
        ),
        ("device_buffer_reports_owning_runtime_binding", device_buffer_reports_owning_runtime_binding),
        (
            "float32_device_buffer_finite_proof_covers_the_exact_allocation_and_binding",
            float32_device_buffer_finite_proof_covers_the_exact_allocation_and_binding,
        ),
        (
            "float32_device_buffer_finite_proof_rejects_nan_and_infinities",
            float32_device_buffer_finite_proof_rejects_nan_and_infinities,
        ),
        (
            "typed_device_buffer_validation_rejects_misaligned_allocation_size",
            typed_device_buffer_validation_rejects_misaligned_allocation_size,
        ),
        (
            "typed_device_buffer_validation_accepts_empty_allocation_with_bounded_proofs",
            typed_device_buffer_validation_accepts_empty_allocation_with_bounded_proofs,
        ),
        (
            "int64_device_buffer_range_proof_covers_the_exact_allocation_and_binding",
            int64_device_buffer_range_proof_covers_the_exact_allocation_and_binding,
        ),
        (
            "int64_device_buffer_range_proof_rejects_both_bound_violations",
            int64_device_buffer_range_proof_rejects_both_bound_violations,
        ),
        (
            "int64_device_buffer_range_proof_rejects_empty_interval_before_execution",
            int64_device_buffer_range_proof_rejects_empty_interval_before_execution,
        ),
        (
            "float32_device_buffer_add_scalar_preserves_infinite_slots",
            float32_device_buffer_add_scalar_preserves_infinite_slots,
        ),
        (
            "buffer_sentinel_operations_reject_out_of_range_int64_and_non_finite_float32",
            buffer_sentinel_operations_reject_out_of_range_int64_and_non_finite_float32,
        ),
        ("from_host_data_creates_correct_size", from_host_data_creates_correct_size),
        ("from_host_data_empty_slice", from_host_data_empty_slice),
        ("adopt_i32_roundtrip", adopt_i32_roundtrip),
        ("adopt_f64_roundtrip", adopt_f64_roundtrip),
        ("adopted_bool8_decodes_non_canonical_non_zero_bytes", adopted_bool8_decodes_non_canonical_non_zero_bytes),
        (
            "adopted_nullable_bool8_never_materializes_null_slot_as_bool",
            adopted_nullable_bool8_never_materializes_null_slot_as_bool,
        ),
        ("adopt_proves_zero_copy", adopt_proves_zero_copy),
        ("adopt_zero_elements", adopt_zero_elements),
        ("adopt_rejects_string_type", adopt_rejects_string_type),
        (
            "runtime_adopt_i32_roundtrip_seeds_explicit_residency_without_copy",
            runtime_adopt_i32_roundtrip_seeds_explicit_residency_without_copy,
        ),
        ("adopt_rejects_size_mismatch", adopt_rejects_size_mismatch),
        ("adopt_rejects_row_count_exceeding_i32_max", adopt_rejects_row_count_exceeding_i32_max),
        ("adopt_nullable_with_nulls", adopt_nullable_with_nulls),
        ("adopt_nullable_all_valid", adopt_nullable_all_valid),
        (
            "runtime_adopt_nullable_roundtrip_seeds_explicit_residency_without_copy",
            runtime_adopt_nullable_roundtrip_seeds_explicit_residency_without_copy,
        ),
        (
            "adopt_nullable_rejects_null_count_exceeding_row_count",
            adopt_nullable_rejects_null_count_exceeding_row_count,
        ),
        (
            "adopt_nullable_rejects_null_count_exceeding_i32_max_with_stable_contract",
            adopt_nullable_rejects_null_count_exceeding_i32_max_with_stable_contract,
        ),
        ("adopt_nullable_rejects_mask_too_small", adopt_nullable_rejects_mask_too_small),
        (
            "adopt_nullable_rejects_mask_smaller_than_cudf_allocation_size",
            adopt_nullable_rejects_mask_smaller_than_cudf_allocation_size,
        ),
        (
            "adopt_nullable_rejects_data_buffer_size_mismatch_with_stable_contract",
            adopt_nullable_rejects_data_buffer_size_mismatch_with_stable_contract,
        ),
    ]);
}
