//! Contracts for runtime-owned dense Float32 matrix packing.

use cudf::{
    BoundaryKind, DenseMatrixLayout, ErrorCode, Runtime, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::reshape::interleave_columns,
    testing::{arrow, boundary, prelude::*},
};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, run_named_cases,
};

fn direct_interleave_columns_uses_the_public_reshape_surface() {
    let runtime = Runtime::new().unwrap();
    let first = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let second = runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let first_view = first.testing_view().unwrap();
    let second_view = second.testing_view().unwrap();

    let interleaved = interleave_columns(&[&first_view, &second_view]).unwrap();

    assert_eq!(interleaved.to_vec::<i32>().unwrap(), vec![1, 10, 2, 20, 3, 30]);
}

fn row_major_pack_interleaves_columns_on_target_runtime() {
    let input_runtime = Runtime::new().unwrap();
    let output_runtime = Runtime::new().unwrap();
    let first = input_runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let second = output_runtime.column_from_slice(&[10.0f32, 20.0, 30.0]).unwrap();

    let matrix = output_runtime.pack_float32_columns(&[&first, &second], DenseMatrixLayout::RowMajor).unwrap();

    assert_eq!(matrix.shape(), [3, 2]);
    assert_eq!(matrix.len(), 6);
    assert!(!matrix.is_empty());
    assert_eq!(matrix.layout(), DenseMatrixLayout::RowMajor);
    assert_eq!(matrix.row_stride_elements(), 2);
    assert_eq!(matrix.column_stride_elements(), 1);
    assert_eq!(matrix.placement(), output_runtime.placement());
    assert_eq!(matrix.binding_id(), output_runtime.binding_id());
    assert_eq!(matrix.as_column().data_type().unwrap().id(), TypeId::Float32);
    assert_eq!(matrix.as_column().to_vec::<f32>().unwrap(), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);

    let validated = matrix.validate_finite().unwrap();
    assert_eq!(validated.shape(), matrix.shape());
    assert_eq!(validated.layout(), matrix.layout());
    assert!(!validated.as_ptr().unwrap().is_null());
    assert_eq!(validated.as_ptr().unwrap(), validated.as_ptr().unwrap());
}

fn column_major_pack_concatenates_columns() {
    let runtime = Runtime::new().unwrap();
    let first = runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let second = runtime.column_from_slice(&[10.0f32, 20.0, 30.0]).unwrap();

    let matrix = runtime.pack_float32_columns(&[&first, &second], DenseMatrixLayout::ColumnMajor).unwrap();

    assert_eq!(matrix.shape(), [3, 2]);
    assert_eq!(matrix.row_stride_elements(), 1);
    assert_eq!(matrix.column_stride_elements(), 3);
    assert_eq!(matrix.into_column().to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
}

fn pack_rejects_empty_input() {
    let runtime = Runtime::new().unwrap();
    let error = runtime
        .pack_float32_columns(&[], DenseMatrixLayout::RowMajor)
        .expect_err("empty matrix input must be rejected");

    assert_eq!(error.code(), ErrorCode::EmptyInput);
    assert_eq!(error.operation(), "runtime::pack_float32_columns");
    assert_eq!(error.detail_value("reason"), Some("empty_input"));
}

fn pack_rejects_non_float32_column_with_index_and_types() {
    let runtime = Runtime::new().unwrap();
    let first = runtime.column_from_slice(&[1.0f32, 2.0]).unwrap();
    let second = runtime.column_from_slice(&[3i32, 4]).unwrap();
    let error = runtime
        .pack_float32_columns(&[&first, &second], DenseMatrixLayout::RowMajor)
        .expect_err("mixed dtypes must be rejected");

    assert_eq!(error.code(), ErrorCode::TypeMismatch);
    assert_eq!(error.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(1)));
    assert_eq!(error.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(error.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Float32)));
}

fn pack_rejects_row_count_mismatch() {
    let runtime = Runtime::new().unwrap();
    let first = runtime.column_from_slice(&[1.0f32, 2.0]).unwrap();
    let second = runtime.column_from_slice(&[3.0f32]).unwrap();
    let error = runtime
        .pack_float32_columns(&[&first, &second], DenseMatrixLayout::ColumnMajor)
        .expect_err("row count mismatch must be rejected");

    assert_eq!(error.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(error.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(1)));
    assert_eq!(error.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(2)));
    assert_eq!(error.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(1)));
}

fn pack_rejects_null_feature_values() {
    let runtime = Runtime::new().unwrap();
    let column = runtime.column_from_slice_nullable(&[1.0f32, 0.0, 3.0], &[true, false, true]).unwrap();
    let error = runtime
        .pack_float32_columns(&[&column], DenseMatrixLayout::RowMajor)
        .expect_err("null feature values must be rejected");

    assert_eq!(error.code(), ErrorCode::NullUnexpected);
    assert_eq!(error.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(0)));
    assert_eq!(error.detail_value("null_count"), Some("1"));
    assert_eq!(error.detail_value("reason"), Some("null_feature_value"));
}

fn pack_preserves_non_null_nan_and_infinity_for_separate_value_validation() {
    let runtime = Runtime::new().unwrap();
    let column = runtime.column_from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]).unwrap();

    let packed = runtime.pack_float32_columns(&[&column], DenseMatrixLayout::RowMajor).unwrap();
    let values = packed.as_column().to_vec::<f32>().unwrap();
    assert!(values[0].is_nan());
    assert_eq!(values[1], f32::INFINITY);
    assert_eq!(values[2], f32::NEG_INFINITY);
}

fn finite_validation_rejects_nan_and_both_infinities() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let runtime = Runtime::new().unwrap();
        let column = runtime.column_from_slice(&[1.0f32, invalid, 3.0]).unwrap();
        let matrix = runtime.pack_float32_columns(&[&column], DenseMatrixLayout::RowMajor).unwrap();

        let error = matrix.validate_finite().expect_err("non-finite matrix input must be rejected");

        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(error.operation(), "packed_float32_matrix::validate_finite");
        assert_eq!(
            error.fact_value(FactKeyName::ValidationReason),
            Some(&FactValue::Str("non_finite_vector_value".into()))
        );
        assert_eq!(error.fact_value(FactKeyName::ExecutionBindingId), Some(&FactValue::U64(matrix.binding_id().get())));
        assert_eq!(
            error.fact_value(FactKeyName::ValidationDevice),
            Some(&FactValue::DeviceId(matrix.placement().device.ordinal() as i32))
        );
        assert_eq!(
            error.fact_value(FactKeyName::ValidationStream),
            Some(&FactValue::Usize(matrix.placement().stream_debug))
        );
        assert_eq!(error.fact_value(FactKeyName::ControlPlaneScalarBytes), Some(&FactValue::Usize(4)));
        assert_eq!(error.detail_value("non_finite_count"), Some("1"));
        assert_eq!(error.detail_value("validation_evidence"), Some("bounded_control_scalar"));
        assert_eq!(error.detail_value("reason"), None);
        assert_eq!(error.detail_value("control_plane_scalar_bytes"), None);
    }
}

fn finite_validation_covers_multi_column_gpu_packed_output() {
    let runtime = Runtime::new().unwrap();
    let first = runtime.column_from_slice(&[1.0f32, 2.0]).unwrap();
    let second = runtime.column_from_slice(&[f32::INFINITY, f32::NEG_INFINITY]).unwrap();
    let matrix = runtime.pack_float32_columns(&[&first, &second], DenseMatrixLayout::ColumnMajor).unwrap();

    let error = matrix.validate_finite().expect_err("all packed columns must participate in finite validation");

    assert_eq!(error.detail_value("non_finite_count"), Some("2"));
    assert_eq!(error.detail_value("validated_values"), Some("4"));
}

fn finite_validation_accepts_empty_matrix_with_bounded_scalar_proof() {
    let runtime = Runtime::new().unwrap();
    let empty = runtime.column_from_slice::<f32>(&[]).unwrap();
    let matrix = runtime.pack_float32_columns(&[&empty], DenseMatrixLayout::RowMajor).unwrap();

    let validated = matrix.validate_finite().unwrap();

    assert_eq!(validated.len(), 0);
    assert!(validated.is_empty());
    assert_eq!(validated.binding_id(), matrix.binding_id());
    assert_eq!(validated.observation().kind, BoundaryKind::BoundedControlScalar);
    assert_eq!(validated.observation().memory.downloaded_bytes, Some(std::mem::size_of::<i32>()));
}

fn finite_validation_uses_exact_binding_and_never_publishes_arrow_host_boundary() {
    let runtime = Runtime::new().unwrap();
    let first = runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let second = runtime.column_from_slice(&[10.0f32, 20.0, 30.0]).unwrap();
    let matrix = runtime.pack_float32_columns(&[&first, &second], DenseMatrixLayout::RowMajor).unwrap();
    let placement = matrix.placement();

    reset_exec_observation();
    boundary::reset_boundary_observation_for_test();
    arrow::reset_arrow_host_interop_observation_for_test();
    let validated = matrix.validate_finite().unwrap();

    assert_eq!(validated.len(), matrix.len());
    assert_eq!(validated.binding_id(), matrix.binding_id());
    assert_eq!(validated.placement(), placement);
    assert_eq!(validated.rows(), matrix.rows());
    assert_eq!(validated.columns(), matrix.columns());
    assert_eq!(validated.row_stride_elements(), matrix.row_stride_elements());
    assert_eq!(validated.column_stride_elements(), matrix.column_stride_elements());
    assert_eq!(last_exec_stream_value(), placement.stream_debug);
    assert_eq!(last_sync_stream_value(), placement.stream_debug);
    assert_eq!(last_exec_mr_value(), placement.memory_resource_debug);

    let observation = validated.observation();
    assert_eq!(observation.kind, BoundaryKind::BoundedControlScalar);
    assert_eq!(observation.placement, placement);
    assert_eq!(observation.rows, matrix.len());
    assert_eq!(observation.columns, 1);
    assert!(observation.synchronized_for_host);
    assert_eq!(observation.memory.downloaded_bytes, Some(std::mem::size_of::<i32>()));
    assert_eq!(boundary::last_boundary_observation_for_test(), Some(observation));
    assert!(arrow::last_arrow_host_boundary_observation_for_test().is_none());
}

#[test]
fn matrix_tests_cases() {
    run_named_cases(&[
        (
            "direct_interleave_columns_uses_the_public_reshape_surface",
            direct_interleave_columns_uses_the_public_reshape_surface,
        ),
        ("row_major_pack_interleaves_columns_on_target_runtime", row_major_pack_interleaves_columns_on_target_runtime),
        ("column_major_pack_concatenates_columns", column_major_pack_concatenates_columns),
        ("pack_rejects_empty_input", pack_rejects_empty_input),
        ("pack_rejects_non_float32_column_with_index_and_types", pack_rejects_non_float32_column_with_index_and_types),
        ("pack_rejects_row_count_mismatch", pack_rejects_row_count_mismatch),
        ("pack_rejects_null_feature_values", pack_rejects_null_feature_values),
        (
            "pack_preserves_non_null_nan_and_infinity_for_separate_value_validation",
            pack_preserves_non_null_nan_and_infinity_for_separate_value_validation,
        ),
        ("finite_validation_rejects_nan_and_both_infinities", finite_validation_rejects_nan_and_both_infinities),
        (
            "finite_validation_covers_multi_column_gpu_packed_output",
            finite_validation_covers_multi_column_gpu_packed_output,
        ),
        (
            "finite_validation_accepts_empty_matrix_with_bounded_scalar_proof",
            finite_validation_accepts_empty_matrix_with_bounded_scalar_proof,
        ),
        (
            "finite_validation_uses_exact_binding_and_never_publishes_arrow_host_boundary",
            finite_validation_uses_exact_binding_and_never_publishes_arrow_host_boundary,
        ),
    ]);
}
