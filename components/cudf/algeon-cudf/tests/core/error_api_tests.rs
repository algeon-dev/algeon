use cudf::{
    ErrorCode, ErrorKind, ErrorStatus,
    core::error::{FactKeyName, FactValue, find_cudf_error_in_chain, kind_for_code},
    testing::{
        core::{
            trigger_invariant_violation_for_test, trigger_null_handle_contract_error_for_test,
            trigger_range_limit_error_for_test, trigger_table_column_count_limit_error_for_test,
        },
        dataframe::new_column_length_mismatch_error_for_test,
        error::from_interop_in_for_test,
        interop::{
            trigger_interop_column_index_limit_for_test, trigger_interop_projection_missing_null_mask_for_test,
            trigger_interop_projection_negative_offset_for_test,
        },
    },
};
use rapids_interop::{InteropError, InteropErrorKind};

use crate::common::run_named_cases;

fn device_fact(err: &cudf::Error, key: FactKeyName) -> i32 {
    match err.fact_value(key) {
        Some(FactValue::DeviceId(v)) => *v,
        other => panic!("expected DeviceId fact for {key:?}, got {other:?}"),
    }
}

fn interop_error_converts_with_kind_context_status_and_source() {
    let interop = InteropError::new(InteropErrorKind::UnsupportedType, "dictionary export is not supported")
        .with_context("operation", "interop::PreparedColumnExport::try_prepare_device_column_export")
        .with_context("type_id", "Dictionary32");

    let err = from_interop_in_for_test(interop, "test::interop_boundary");

    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "dictionary export is not supported");
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(
        err.detail_value("foreign_operation"),
        Some("interop::PreparedColumnExport::try_prepare_device_column_export")
    );
    // `type_id` is reserved for typed facts, so the translator rewrites
    // foreign `type_id` breadcrumbs onto the non-canonical
    // `foreign_type_id` detail.
    assert_eq!(err.detail_value("type_id"), None);
    assert_eq!(err.detail_value("foreign_type_id"), Some("Dictionary32"));
    assert_eq!(
        std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(),
        Some("dictionary export is not supported")
    );
}

fn interop_error_nested_source_chain_is_preserved_when_mapping_into_cudf_error() {
    let interop = InteropError::new(InteropErrorKind::InvalidView, "borrowed view rejected")
        .with_context("column_id", "7")
        .set_source(std::io::Error::other("child cause"));

    let err = from_interop_in_for_test(interop, "test::interop_boundary");
    let source = std::error::Error::source(&err).expect("expected interop source");

    assert_eq!(source.to_string(), "borrowed view rejected");
    assert_eq!(source.source().map(std::string::ToString::to_string).as_deref(), Some("child cause"));
}

fn interop_not_found_maps_to_not_found() {
    let err = from_interop_in_for_test(
        InteropError::new(InteropErrorKind::NotFound, "prepared export referenced a missing buffer binding")
            .with_context("buffer_id", "10"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "prepared export referenced a missing buffer binding");
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("buffer_id", "10".to_string())]);
}

fn interop_readiness_unavailable_without_cudf_source_maps_to_permanent_execution_context_error() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::ReadinessUnavailable,
            "CUDA stream handle must carry explicit device identity",
        )
        .with_context("field", "cuda_stream.device"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::ReadinessUnavailable);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("field", "cuda_stream.device".to_string())]);
}

fn interop_execution_binding_unavailable_without_cudf_source_uses_stable_code() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::ExecutionBindingUnavailable,
            "borrowed table projection requires an explicit producer execution binding",
        )
        .with_context("binding", "execution"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::ExecutionBindingUnavailable);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "borrowed table projection requires an explicit producer execution binding");
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("binding", "execution".to_string())]);
}

fn interop_lease_already_active_without_cudf_source_uses_stable_code() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::LeaseAlreadyActive,
            "prepared table export already has an active read lease",
        )
        .with_context("export_kind", "table"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::LeaseAlreadyActive);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "prepared table export already has an active read lease");
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("export_kind", "table".to_string())]);
}

fn interop_unsupported_layout_maps_to_unsupported() {
    let err = from_interop_in_for_test(
        InteropError::new(InteropErrorKind::UnsupportedLayout, "struct columns are not supported for this export path")
            .with_context("layout", "struct"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::UnsupportedLayout);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "struct columns are not supported for this export path");
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("layout", "struct".to_string())]);
}

fn interop_execution_binding_mismatch_maps_to_runtime_binding_mismatch() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "prepared table import runtime must use the same CUDA device as the producer partition",
        )
        .with_context("producer_device", "0")
        .with_context("runtime_device", "1"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [] as [(&str, String); 0]);
    assert_eq!(device_fact(&err, FactKeyName::LeftDevice), 0);
    assert_eq!(device_fact(&err, FactKeyName::RightDevice), 1);
    assert_eq!(err.detail_value("producer_device"), None);
    assert_eq!(err.detail_value("runtime_device"), None);
    assert_eq!(err.detail_value("left_device"), None);
    assert_eq!(err.detail_value("right_device"), None);
}

fn interop_descriptor_runtime_binding_mismatch_maps_to_canonical_device_context() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "borrowed descriptor import requires the destination runtime to use the same CUDA device",
        )
        .with_context("descriptor_device", "0")
        .with_context("runtime_device", "1"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [] as [(&str, String); 0]);
    assert_eq!(device_fact(&err, FactKeyName::LeftDevice), 0);
    assert_eq!(device_fact(&err, FactKeyName::RightDevice), 1);
    assert_eq!(err.detail_value("descriptor_device"), None);
    assert_eq!(err.detail_value("runtime_device"), None);
    assert_eq!(err.detail_value("left_device"), None);
    assert_eq!(err.detail_value("right_device"), None);
}

fn interop_readiness_mismatch_maps_to_runtime_binding_mismatch() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::ReadinessMismatch,
            "prepared export readiness event must live on the producer CUDA device",
        )
        .with_context("readiness", "device_event_ordered")
        .with_context("producer_device", "0")
        .with_context("event_device", "1"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("readiness", "device_event_ordered".to_string())]);
    assert_eq!(device_fact(&err, FactKeyName::LeftDevice), 0);
    assert_eq!(device_fact(&err, FactKeyName::RightDevice), 1);
    assert_eq!(err.detail_value("producer_device"), None);
    assert_eq!(err.detail_value("event_device"), None);
    assert_eq!(err.detail_value("left_device"), None);
    assert_eq!(err.detail_value("right_device"), None);
}

fn interop_release_stream_mismatch_maps_to_canonical_device_context() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::ReadinessMismatch,
            "runtime-backed owned-transfer release stream must live on the release CUDA device",
        )
        .with_context("completion", "same_device_stream")
        .with_context("release_device", "0")
        .with_context("stream_device", "1"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("completion", "same_device_stream".to_string())]);
    assert_eq!(device_fact(&err, FactKeyName::LeftDevice), 0);
    assert_eq!(device_fact(&err, FactKeyName::RightDevice), 1);
    assert_eq!(err.detail_value("release_device"), None);
    assert_eq!(err.detail_value("stream_device"), None);
    assert_eq!(err.detail_value("left_device"), None);
    assert_eq!(err.detail_value("right_device"), None);
}

fn interop_mixed_device_unsupported_maps_to_cross_device_inputs() {
    let err = from_interop_in_for_test(
        InteropError::new(
            InteropErrorKind::MixedDeviceUnsupported,
            "owned table transfer requires all referenced bindings to live on one CUDA device",
        )
        .with_context("expected_device", "0")
        .with_context("actual_device", "1"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [] as [(&str, String); 0]);
    assert_eq!(device_fact(&err, FactKeyName::LeftDevice), 0);
    assert_eq!(device_fact(&err, FactKeyName::RightDevice), 1);
    assert_eq!(err.detail_value("expected_device"), None);
    assert_eq!(err.detail_value("actual_device"), None);
    assert_eq!(err.detail_value("left_device"), None);
    assert_eq!(err.detail_value("right_device"), None);
}

fn interop_profile_unsupported_maps_to_unsupported_operation_code() {
    let err = from_interop_in_for_test(
        InteropError::new(InteropErrorKind::ProfileUnsupported, "prepared host export requires host-visible readiness")
            .with_context("profile", "prepared_host_export")
            .with_context("readiness", "device_stream_ordered"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(
        err.details(),
        [("profile", "prepared_host_export".to_string()), ("readiness", "device_stream_ordered".to_string()),]
    );
}

fn interop_placement_unsupported_maps_to_unsupported_operation_code() {
    let err = from_interop_in_for_test(
        InteropError::new(InteropErrorKind::PlacementUnsupported, "prepared host export requires host-backed buffers")
            .with_context("binding", "values")
            .with_context("placement", "cuda"),
        "test::interop_boundary",
    );

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.details(), [("binding", "values".to_string()), ("placement", "cuda".to_string())]);
}

fn interop_error_promotes_nested_operation_context_into_operation_and_called() {
    let interop = InteropError::new(InteropErrorKind::InvalidView, "borrowed view rejected")
        .with_context("operation", "interop::ActiveReadLease::complete")
        .with_context("operation", "runtime::completion::wait")
        .with_context("column_id", "7");

    let err = from_interop_in_for_test(interop, "test::interop_boundary");

    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "test::interop_boundary");
    // Every foreign `operation` context entry is routed into `foreign_operation`
    // under the truthful contract; `details()` preserves insertion order.
    assert_eq!(
        err.details(),
        [
            ("foreign_operation", "interop::ActiveReadLease::complete".to_string()),
            ("foreign_operation", "runtime::completion::wait".to_string()),
            ("column_id", "7".to_string()),
        ]
    );
}

fn interop_projection_negative_offset_contract_maps_to_cudf_error_with_canonical_context() {
    let interop = trigger_interop_projection_negative_offset_for_test().unwrap_err();
    let err = from_interop_in_for_test(interop, "test::interop_boundary");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.detail_value("foreign_operation"), Some("testing::interop::projection::negative_offset"));
    assert_eq!(err.detail_value("resource"), Some("offset"));
    assert_eq!(err.detail_value("reason"), Some("negative_value"));
    assert_eq!(err.detail_value("offset"), Some("-1"));
}

fn interop_projection_missing_null_mask_contract_maps_to_cudf_error_with_canonical_context() {
    let interop = trigger_interop_projection_missing_null_mask_for_test().unwrap_err();
    let err = from_interop_in_for_test(interop, "test::interop_boundary");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.detail_value("foreign_operation"), Some("testing::interop::projection::missing_null_mask"));
    assert_eq!(err.detail_value("resource"), Some("null_mask_buffer"));
    assert_eq!(err.detail_value("reason"), Some("missing_for_nullable_column"));
    assert_eq!(err.detail_value("row_count"), Some("3"));
}

fn interop_column_index_limit_contract_maps_to_cudf_error_with_canonical_context() {
    let interop = trigger_interop_column_index_limit_for_test().unwrap_err();
    let err = from_interop_in_for_test(interop, "test::interop_boundary");
    let expected_column_index = usize::MAX.to_string();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.message(), "column_index exceeds i32::MAX");
    assert_eq!(err.detail_value("foreign_operation"), Some("testing::interop::column_index_limit"));
    assert_eq!(err.detail_value("resource"), Some("column_index"));
    assert_eq!(err.detail_value("reason"), Some("exceeds_i32_max"));
    assert_eq!(err.detail_value("remedy"), Some("reduce column_index to fit within i32::MAX"));
    assert_eq!(err.detail_value("foreign_column_index"), Some(expected_column_index.as_str()));
    assert_eq!(err.detail_value("column_index"), None);
}

fn interop_error_with_cudf_source_preserves_cudf_kind_and_status() {
    // Synthesize a real nested cudf `Error` via a public testing hook.
    // Construction helpers are crate-internal, so tests cannot build an
    // `Error` directly. They must exercise a production-shaped path that
    // returns one.
    let inner_cudf_error = trigger_invariant_violation_for_test().unwrap_err();
    let interop = InteropError::new(InteropErrorKind::ExecutionBindingUnavailable, "CUDA runtime busy")
        .with_context("operation", "interop::ActiveReadLease::complete")
        .set_source(inner_cudf_error);

    let err = from_interop_in_for_test(interop, "test::interop_boundary");

    // Truthful path: code/kind/status derive purely from the interop kind, not
    // from the nested cudf error.
    assert_eq!(err.code(), ErrorCode::ExecutionBindingUnavailable);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "test::interop_boundary");
    assert_eq!(err.detail_value("foreign_operation"), Some("interop::ActiveReadLease::complete"));
    // The outer source is the interop error; the inner cudf error lives one
    // level deeper in the source chain.
    let outer_source = std::error::Error::source(&err).expect("expected interop source");
    assert_eq!(outer_source.to_string(), "CUDA runtime busy");
    let inner = find_cudf_error_in_chain(outer_source).expect("expected nested cudf error");
    assert_eq!(inner.code(), ErrorCode::InvariantViolation);
    assert_eq!(inner.operation(), "testing::trigger_invariant_violation_for_test");
}

fn testing_hook_locks_invariant_violation_helper_contract() {
    let err = trigger_invariant_violation_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::trigger_invariant_violation_for_test");
    assert_eq!(err.message(), "injected invariant violation for tests");
}

fn testing_hook_locks_null_handle_on_success_helper_contract() {
    let err = trigger_null_handle_contract_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::trigger_null_handle_contract_error_for_test");
    assert_eq!(err.message(), "FFI returned null test_handle handle on success");
    assert_eq!(err.detail_value("target"), Some("test_handle"));
}

fn testing_hook_locks_shared_range_limit_helper_contract() {
    let err = trigger_range_limit_error_for_test().unwrap_err();
    let expected_row_count = usize::MAX.to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::trigger_range_limit_error_for_test");
    assert_eq!(err.message(), "row_count exceeds i32::MAX");
    assert_eq!(err.detail_value("row_count"), Some(expected_row_count.as_str()));
    assert_eq!(err.detail_value("resource"), Some("row_count"));
    assert_eq!(err.detail_value("reason"), Some("exceeds_i32_max"));
    assert_eq!(err.detail_value("remedy"), Some("reduce row_count to fit within i32::MAX"));
}

fn testing_hook_locks_shared_limit_validation_helper_contract() {
    let err = trigger_table_column_count_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "table::from_columns");
    assert_eq!(err.message(), "number of columns exceeds i32::MAX");
    assert_eq!(err.detail_value("resource"), Some("table_column_count"));
    assert_eq!(err.detail_value("reason"), Some("exceeds_i32_max"));
    assert_eq!(err.detail_value("remedy"), Some("reduce table_column_count to fit within i32::MAX"));
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(usize::MAX)));
    assert_eq!(err.detail_value("table_column_count"), None);
}

fn testing_hook_locks_column_name_count_mismatch_helper_contract() {
    let err = new_column_length_mismatch_error_for_test(2, 3);

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "dataframe::new");
    assert_eq!(err.message(), "number of names must equal number of columns");
    assert_eq!(err.detail_value("column_name_count"), Some("2"));
    assert_eq!(err.detail_value("resource"), Some("column_name_count"));
    assert_eq!(err.detail_value("reason"), Some("mismatch_with_table_column_count"));
    assert_eq!(err.detail_value("remedy"), Some("ensure column_name_count matches table_column_count"));
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(3)));
}

fn invariant_violation_is_exposed_as_a_first_class_kind_and_code() {
    // Under the truthful contract, `InvariantViolation` is both an
    // `ErrorCode` leaf (domestic soft-panic replacement) and a first-class
    // `ErrorKind`. Neither is a projection of the other; `kind_for_code`
    // must map the leaf to the dedicated kind, and `ForeignInvariantViolation`
    // also projects to the same kind so callers can match on `ErrorKind`
    // for coarse policy while still using `ErrorCode` for precise classification.
    assert_eq!(kind_for_code(ErrorCode::InvariantViolation), ErrorKind::InvariantViolation);
    assert_eq!(kind_for_code(ErrorCode::ForeignInvariantViolation), ErrorKind::InvariantViolation);
}

#[test]
fn error_api_tests_cases() {
    run_named_cases(&[
        (
            "interop_error_converts_with_kind_context_status_and_source",
            interop_error_converts_with_kind_context_status_and_source,
        ),
        (
            "interop_error_nested_source_chain_is_preserved_when_mapping_into_cudf_error",
            interop_error_nested_source_chain_is_preserved_when_mapping_into_cudf_error,
        ),
        ("interop_not_found_maps_to_not_found", interop_not_found_maps_to_not_found),
        (
            "interop_readiness_unavailable_without_cudf_source_maps_to_permanent_execution_context_error",
            interop_readiness_unavailable_without_cudf_source_maps_to_permanent_execution_context_error,
        ),
        (
            "interop_execution_binding_unavailable_without_cudf_source_uses_stable_code",
            interop_execution_binding_unavailable_without_cudf_source_uses_stable_code,
        ),
        (
            "interop_lease_already_active_without_cudf_source_uses_stable_code",
            interop_lease_already_active_without_cudf_source_uses_stable_code,
        ),
        ("interop_unsupported_layout_maps_to_unsupported", interop_unsupported_layout_maps_to_unsupported),
        (
            "interop_execution_binding_mismatch_maps_to_runtime_binding_mismatch",
            interop_execution_binding_mismatch_maps_to_runtime_binding_mismatch,
        ),
        (
            "interop_descriptor_runtime_binding_mismatch_maps_to_canonical_device_context",
            interop_descriptor_runtime_binding_mismatch_maps_to_canonical_device_context,
        ),
        (
            "interop_readiness_mismatch_maps_to_runtime_binding_mismatch",
            interop_readiness_mismatch_maps_to_runtime_binding_mismatch,
        ),
        (
            "interop_release_stream_mismatch_maps_to_canonical_device_context",
            interop_release_stream_mismatch_maps_to_canonical_device_context,
        ),
        (
            "interop_mixed_device_unsupported_maps_to_cross_device_inputs",
            interop_mixed_device_unsupported_maps_to_cross_device_inputs,
        ),
        (
            "interop_profile_unsupported_maps_to_unsupported_operation_code",
            interop_profile_unsupported_maps_to_unsupported_operation_code,
        ),
        (
            "interop_placement_unsupported_maps_to_unsupported_operation_code",
            interop_placement_unsupported_maps_to_unsupported_operation_code,
        ),
        (
            "interop_error_promotes_nested_operation_context_into_operation_and_called",
            interop_error_promotes_nested_operation_context_into_operation_and_called,
        ),
        (
            "interop_projection_negative_offset_contract_maps_to_cudf_error_with_canonical_context",
            interop_projection_negative_offset_contract_maps_to_cudf_error_with_canonical_context,
        ),
        (
            "interop_projection_missing_null_mask_contract_maps_to_cudf_error_with_canonical_context",
            interop_projection_missing_null_mask_contract_maps_to_cudf_error_with_canonical_context,
        ),
        (
            "interop_column_index_limit_contract_maps_to_cudf_error_with_canonical_context",
            interop_column_index_limit_contract_maps_to_cudf_error_with_canonical_context,
        ),
        (
            "interop_error_with_cudf_source_preserves_cudf_kind_and_status",
            interop_error_with_cudf_source_preserves_cudf_kind_and_status,
        ),
        (
            "testing_hook_locks_invariant_violation_helper_contract",
            testing_hook_locks_invariant_violation_helper_contract,
        ),
        (
            "testing_hook_locks_null_handle_on_success_helper_contract",
            testing_hook_locks_null_handle_on_success_helper_contract,
        ),
        (
            "testing_hook_locks_shared_range_limit_helper_contract",
            testing_hook_locks_shared_range_limit_helper_contract,
        ),
        (
            "testing_hook_locks_shared_limit_validation_helper_contract",
            testing_hook_locks_shared_limit_validation_helper_contract,
        ),
        (
            "testing_hook_locks_column_name_count_mismatch_helper_contract",
            testing_hook_locks_column_name_count_mismatch_helper_contract,
        ),
        (
            "invariant_violation_is_exposed_as_a_first_class_kind_and_code",
            invariant_violation_is_exposed_as_a_first_class_kind_and_code,
        ),
    ]);
}
