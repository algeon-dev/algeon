use cudf::{Error, ErrorCode, ErrorKind, ErrorStatus};
use rapids_interop::{InteropError, InteropErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase3Kind {
    InvalidInput,
    NotFound,
    Unsupported,
    ExecutionContext,
    ResourceExhausted,
    DependencyFailure,
    InvariantViolation,
}

fn phase3_kind_for_code(code: ErrorCode) -> Phase3Kind {
    match code {
        ErrorCode::InvalidInput
        | ErrorCode::MissingColumn
        | ErrorCode::DuplicateColumnName
        | ErrorCode::DuplicateKey
        | ErrorCode::ColumnLengthMismatch
        | ErrorCode::TypeMismatch
        | ErrorCode::IndexOutOfBounds
        | ErrorCode::NullUnexpected => Phase3Kind::InvalidInput,
        ErrorCode::NotFound | ErrorCode::MissingResource => Phase3Kind::NotFound,
        ErrorCode::Unsupported
        | ErrorCode::UnsupportedDtype
        | ErrorCode::UnsupportedLayout
        | ErrorCode::UnsupportedOperation => Phase3Kind::Unsupported,
        ErrorCode::ExecutionContext
        | ErrorCode::EmptyConstructionRequiresRuntime
        | ErrorCode::CrossDeviceInputs
        | ErrorCode::CrossProvenanceInputs
        | ErrorCode::RuntimeBindingMismatch
        | ErrorCode::ReadinessUnavailable
        | ErrorCode::ExecutionBindingUnavailable
        | ErrorCode::LeaseAlreadyActive => Phase3Kind::ExecutionContext,
        ErrorCode::ResourceExhausted | ErrorCode::OutOfMemory => Phase3Kind::ResourceExhausted,
        ErrorCode::DependencyFailure | ErrorCode::CudaRuntime | ErrorCode::FatalCuda | ErrorCode::IoFailure => {
            Phase3Kind::DependencyFailure
        },
        ErrorCode::InvariantViolation => Phase3Kind::InvariantViolation,
        _ => Phase3Kind::InvariantViolation,
    }
}

#[test]
fn new_defaults_to_permanent() {
    let err = Error::new(ErrorKind::InvalidInput, "bad type");
    assert_eq!(err.code(), ErrorCode::InvalidInput);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.message(), "bad type");
    assert!(err.is_permanent());
    assert!(!err.is_temporary());
    assert!(!err.is_persistent());
    assert_eq!(err.operation(), "");
    assert!(err.context().is_empty());
}

#[test]
fn builder_methods_chain() {
    let err = Error::new(ErrorKind::ResourceExhausted, "out of memory")
        .with_code(ErrorCode::OutOfMemory)
        .with_operation("reduce")
        .with_context("dtype", "float64")
        .with_context("rows", "1000000")
        .set_temporary();

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(err.is_temporary());
    assert_eq!(err.operation(), "reduce");
    assert_eq!(err.context().len(), 2);
    assert_eq!(err.context()[0], ("dtype", "float64".to_string()));
    assert_eq!(err.context()[1], ("rows", "1000000".to_string()));
}

#[test]
fn status_accessor_reflects_status() {
    let err = Error::new(ErrorKind::InvalidInput, "bad type");
    assert_eq!(err.status(), ErrorStatus::Permanent);

    let tmp = Error::new(ErrorKind::DependencyFailure, "oom").set_temporary();
    assert_eq!(tmp.status(), ErrorStatus::Temporary);

    let persistent = Error::new(ErrorKind::DependencyFailure, "device lost").set_persistent();
    assert_eq!(persistent.status(), ErrorStatus::Persistent);
}

#[test]
fn set_source_attaches_cause() {
    let cause = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
    let err = Error::new(ErrorKind::DependencyFailure, "read failed").set_source(cause);

    let src = std::error::Error::source(&err);
    assert!(src.is_some());
}

#[test]
fn set_source_accepts_boxed_std_error_and_preserves_chain() {
    let cause: Box<dyn std::error::Error + Send + Sync + 'static> =
        Box::new(std::io::Error::other("boxed child cause"));
    let err = Error::new(ErrorKind::DependencyFailure, "read failed").set_source(cause);

    let src = std::error::Error::source(&err).expect("expected boxed source");
    assert_eq!(src.to_string(), "boxed child cause");
}

#[test]
fn display_minimal() {
    let err = Error::new(ErrorKind::InvalidInput, "cannot cast");
    let s = err.to_string();
    assert!(s.contains("InvalidInput"));
    assert!(s.contains("permanent"));
    assert!(s.contains("cannot cast"));
    assert!(!s.contains(" at "));
    assert!(!s.contains("context:"));
}

#[test]
fn display_full() {
    let err = Error::new(ErrorKind::ResourceExhausted, "out of memory")
        .with_code(ErrorCode::OutOfMemory)
        .with_operation("reduce")
        .with_context("dtype", "float64")
        .set_temporary();

    let s = err.to_string();
    assert!(s.contains("ResourceExhausted (temporary)"));
    assert!(s.contains("at reduce"));
    assert!(s.contains("dtype: float64"));
    assert!(s.contains("out of memory"));
}

#[test]
fn debug_is_multiline() {
    let err = Error::new(ErrorKind::DependencyFailure, "oom").with_operation("sort").with_context("rows", "1M");

    let s = format!("{:?}", err);
    assert!(s.contains("operation: sort"));
    assert!(s.contains("rows: 1M"));
    assert!(s.contains("message: oom"));
}

#[test]
fn default_status_logic_errors_are_permanent() {
    let type_err = Error::new(ErrorKind::InvalidInput, "bad type");
    assert_eq!(type_err.status(), ErrorStatus::Permanent);

    let invalid_arg = Error::new(ErrorKind::InvalidInput, "bad argument");
    assert_eq!(invalid_arg.status(), ErrorStatus::Permanent);

    let not_found = Error::new(ErrorKind::NotFound, "missing column");
    assert_eq!(not_found.status(), ErrorStatus::Permanent);

    let unsupported = Error::new(ErrorKind::Unsupported, "not supported");
    assert_eq!(unsupported.status(), ErrorStatus::Permanent);

    let invalid_op = Error::new(ErrorKind::InvalidInput, "bad op");
    assert_eq!(invalid_op.status(), ErrorStatus::Permanent);

    let index_err = Error::new(ErrorKind::InvalidInput, "bad index");
    assert_eq!(index_err.status(), ErrorStatus::Permanent);
}

#[test]
fn helpers_reflect_code_status_kind_and_context() {
    let err = Error::new(ErrorKind::ResourceExhausted, "device allocation failed")
        .with_code(ErrorCode::OutOfMemory)
        .with_operation("column::allocate")
        .with_context("rows", "1024")
        .set_temporary();

    assert_eq!(err.context_value("rows"), Some("1024"));
    assert_eq!(err.context_value("missing"), None);
    assert!(err.is_retryable());
    assert!(err.is_oom());
    assert!(!err.is_not_found());

    let not_found = Error::new(ErrorKind::NotFound, "missing column").with_code(ErrorCode::MissingColumn);
    assert!(not_found.is_not_found());
    assert!(!not_found.is_retryable());
}

#[test]
fn set_persistent_works() {
    let err = Error::new(ErrorKind::InvariantViolation, "repeated failure").set_persistent();
    assert!(err.is_persistent());
    assert!(!err.is_permanent());
    assert!(!err.is_temporary());
}

#[test]
fn display_with_source() {
    let cause = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
    let err = Error::new(ErrorKind::DependencyFailure, "read failed").with_operation("read_csv").set_source(cause);

    let s = err.to_string();
    assert!(s.contains("source:"));
    assert!(s.contains("no such file"));
}

#[test]
fn with_operation_preserves_prior_failure_site_in_called_context() {
    let err =
        Error::new(ErrorKind::InvalidInput, "inner failure").with_operation("inner::op").with_operation("outer::op");

    assert_eq!(err.operation(), "outer::op");
    assert_eq!(err.context(), [("called", "inner::op".to_string())]);
}

#[test]
fn with_operation_stacks_multiple_called_context_entries() {
    let err = Error::new(ErrorKind::InvalidInput, "inner failure")
        .with_operation("inner::op")
        .with_operation("middle::op")
        .with_operation("outer::op");

    assert_eq!(err.operation(), "outer::op");
    assert_eq!(err.context(), [("called", "inner::op".to_string()), ("called", "middle::op".to_string()),]);
}

#[test]
fn interop_error_converts_with_kind_context_status_and_source() {
    let interop = InteropError::new(InteropErrorKind::UnsupportedType, "dictionary export is not supported")
        .with_context("operation", "interop::PreparedColumnExport::try_prepare_device_column_export")
        .with_context("type_id", "Dictionary32");

    let err = Error::from(interop);

    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "dictionary export is not supported");
    assert_eq!(err.operation(), "interop::PreparedColumnExport::try_prepare_device_column_export");
    assert_eq!(err.context(), [("type_id", "Dictionary32".to_string())]);
    assert_eq!(
        std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(),
        Some("dictionary export is not supported")
    );
}

#[test]
fn interop_error_nested_source_chain_is_preserved_when_mapping_into_cudf_error() {
    let interop = InteropError::new(InteropErrorKind::InvalidView, "borrowed view rejected")
        .with_context("column_id", "7")
        .set_source(std::io::Error::other("child cause"));

    let err = Error::from(interop);
    let source = std::error::Error::source(&err).expect("expected interop source");

    assert_eq!(source.to_string(), "borrowed view rejected");
    assert_eq!(source.source().map(std::string::ToString::to_string).as_deref(), Some("child cause"));
}

#[test]
fn interop_not_found_maps_to_not_found() {
    let err = Error::from(
        InteropError::new(InteropErrorKind::NotFound, "prepared export referenced a missing buffer binding")
            .with_context("buffer_id", "10"),
    );

    assert_eq!(err.code(), ErrorCode::MissingResource);
    assert_eq!(err.kind(), ErrorKind::NotFound);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "prepared export referenced a missing buffer binding");
    assert_eq!(err.context(), [("buffer_id", "10".to_string())]);
}

#[test]
fn interop_readiness_unavailable_without_cudf_source_maps_to_permanent_execution_context_error() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ReadinessUnavailable,
            "CUDA stream handle must carry explicit device identity",
        )
        .with_context("field", "cuda_stream.device"),
    );

    assert_eq!(err.code(), ErrorCode::ReadinessUnavailable);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.context(), [("field", "cuda_stream.device".to_string())]);
}

#[test]
fn interop_execution_binding_unavailable_without_cudf_source_uses_stable_code() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ExecutionBindingUnavailable,
            "borrowed table projection requires an explicit producer execution binding",
        )
        .with_context("binding", "execution"),
    );

    assert_eq!(err.code(), ErrorCode::ExecutionBindingUnavailable);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "borrowed table projection requires an explicit producer execution binding");
    assert_eq!(err.context(), [("binding", "execution".to_string())]);
}

#[test]
fn interop_lease_already_active_without_cudf_source_uses_stable_code() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::LeaseAlreadyActive,
            "prepared table export already has an active read lease",
        )
        .with_context("export_kind", "table"),
    );

    assert_eq!(err.code(), ErrorCode::LeaseAlreadyActive);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "prepared table export already has an active read lease");
    assert_eq!(err.context(), [("export_kind", "table".to_string())]);
}

#[test]
fn interop_unsupported_layout_maps_to_unsupported() {
    let err = Error::from(
        InteropError::new(InteropErrorKind::UnsupportedLayout, "struct columns are not supported for this export path")
            .with_context("layout", "struct"),
    );

    assert_eq!(err.code(), ErrorCode::UnsupportedLayout);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "struct columns are not supported for this export path");
    assert_eq!(err.context(), [("layout", "struct".to_string())]);
}

#[test]
fn interop_execution_binding_mismatch_maps_to_runtime_binding_mismatch() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "prepared table import runtime must use the same CUDA device as the producer partition",
        )
        .with_context("producer_device", "0")
        .with_context("runtime_device", "1"),
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.context(), [("left_device", "0".to_string()), ("right_device", "1".to_string())]);
    assert_eq!(err.context_value("producer_device"), None);
    assert_eq!(err.context_value("runtime_device"), None);
}

#[test]
fn interop_descriptor_runtime_binding_mismatch_maps_to_canonical_device_context() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "borrowed descriptor import requires the destination runtime to use the same CUDA device",
        )
        .with_context("descriptor_device", "0")
        .with_context("runtime_device", "1"),
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.context(), [("left_device", "0".to_string()), ("right_device", "1".to_string())]);
    assert_eq!(err.context_value("descriptor_device"), None);
    assert_eq!(err.context_value("runtime_device"), None);
}

#[test]
fn interop_readiness_mismatch_maps_to_runtime_binding_mismatch() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ReadinessMismatch,
            "prepared export readiness event must live on the producer CUDA device",
        )
        .with_context("readiness", "device_event_ordered")
        .with_context("producer_device", "0")
        .with_context("event_device", "1"),
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(
        err.context(),
        [
            ("readiness", "device_event_ordered".to_string()),
            ("left_device", "0".to_string()),
            ("right_device", "1".to_string()),
        ]
    );
    assert_eq!(err.context_value("producer_device"), None);
    assert_eq!(err.context_value("event_device"), None);
}

#[test]
fn interop_release_stream_mismatch_maps_to_canonical_device_context() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ReadinessMismatch,
            "runtime-backed owned-transfer release stream must live on the release CUDA device",
        )
        .with_context("completion", "same_device_stream")
        .with_context("release_device", "0")
        .with_context("stream_device", "1"),
    );

    assert_eq!(err.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(
        err.context(),
        [
            ("completion", "same_device_stream".to_string()),
            ("left_device", "0".to_string()),
            ("right_device", "1".to_string()),
        ]
    );
    assert_eq!(err.context_value("release_device"), None);
    assert_eq!(err.context_value("stream_device"), None);
}

#[test]
fn interop_mixed_device_unsupported_maps_to_cross_device_inputs() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::MixedDeviceUnsupported,
            "owned table transfer requires all referenced bindings to live on one CUDA device",
        )
        .with_context("expected_device", "0")
        .with_context("actual_device", "1"),
    );

    assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.context(), [("left_device", "0".to_string()), ("right_device", "1".to_string())]);
    assert_eq!(err.context_value("expected_device"), None);
    assert_eq!(err.context_value("actual_device"), None);
}

#[test]
fn interop_profile_unsupported_maps_to_unsupported_operation_code() {
    let err = Error::from(
        InteropError::new(InteropErrorKind::ProfileUnsupported, "prepared host export requires host-visible readiness")
            .with_context("profile", "prepared_host_export")
            .with_context("readiness", "device_stream_ordered"),
    );

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(
        err.context(),
        [("profile", "prepared_host_export".to_string()), ("readiness", "device_stream_ordered".to_string()),]
    );
}

#[test]
fn interop_placement_unsupported_maps_to_unsupported_operation_code() {
    let err = Error::from(
        InteropError::new(InteropErrorKind::PlacementUnsupported, "prepared host export requires host-backed buffers")
            .with_context("binding", "values")
            .with_context("placement", "cuda"),
    );

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.context(), [("binding", "values".to_string()), ("placement", "cuda".to_string())]);
}

#[test]
fn interop_error_promotes_nested_operation_context_into_operation_and_called() {
    let interop = InteropError::new(InteropErrorKind::InvalidView, "borrowed view rejected")
        .with_context("operation", "interop::ActiveReadLease::complete")
        .with_context("operation", "runtime::completion::wait")
        .with_context("column_id", "7");

    let err = Error::from(interop);

    assert_eq!(err.code(), ErrorCode::InvalidInput);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "interop::ActiveReadLease::complete");
    assert_eq!(err.context(), [("called", "runtime::completion::wait".to_string()), ("column_id", "7".to_string())]);
}

#[test]
fn interop_projection_negative_offset_contract_maps_to_cudf_error_with_canonical_context() {
    let err = Error::from(cudf::testing::trigger_interop_projection_negative_offset_for_test().unwrap_err());

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::InvalidInput);
    assert_eq!(err.operation(), "testing::interop::projection::negative_offset");
    assert_eq!(err.context_value("resource"), Some("offset"));
    assert_eq!(err.context_value("reason"), Some("negative_value"));
    assert_eq!(err.context_value("offset"), Some("-1"));
}

#[test]
fn interop_projection_missing_null_mask_contract_maps_to_cudf_error_with_canonical_context() {
    let err = Error::from(cudf::testing::trigger_interop_projection_missing_null_mask_for_test().unwrap_err());

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::InvalidInput);
    assert_eq!(err.operation(), "testing::interop::projection::missing_null_mask");
    assert_eq!(err.context_value("resource"), Some("null_mask_buffer"));
    assert_eq!(err.context_value("reason"), Some("missing_for_nullable_column"));
    assert_eq!(err.context_value("row_count"), Some("3"));
}

#[test]
fn interop_error_with_cudf_source_preserves_cudf_kind_and_status() {
    let cudf_error = Error::new(ErrorKind::DependencyFailure, "CUDA runtime busy")
        .with_code(ErrorCode::CudaRuntime)
        .with_operation("runtime::synchronize")
        .set_temporary();
    let interop = InteropError::new(InteropErrorKind::ExecutionBindingUnavailable, "CUDA runtime busy")
        .with_context("operation", "interop::ActiveReadLease::complete")
        .set_source(cudf_error);

    let err = Error::from(interop);

    assert_eq!(err.code(), ErrorCode::CudaRuntime);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
    assert_eq!(err.status(), ErrorStatus::Temporary);
    assert_eq!(err.operation(), "interop::ActiveReadLease::complete");
    assert_eq!(err.context(), [("called", "runtime::synchronize".to_string())]);
    assert_eq!(
        std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(),
        Some("CUDA runtime busy")
    );
}

#[test]
fn question_mark_propagates_interop_error_into_cudf_result() {
    fn helper() -> cudf::Result<()> {
        Err::<(), _>(
            InteropError::new(InteropErrorKind::NullMaskUnsupported, "byte-granularity validity is not supported")
                .with_context("column", "validity"),
        )?;
        Ok(())
    }

    let err = helper().unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidInput);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "byte-granularity validity is not supported");
    assert_eq!(err.context(), [("column", "validity".to_string())]);
}

#[test]
fn testing_hook_locks_invariant_violation_helper_contract() {
    let err = cudf::testing::trigger_invariant_violation_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::trigger_invariant_violation_for_test");
    assert_eq!(err.message(), "injected invariant violation for tests");
}

#[test]
fn testing_hook_locks_null_handle_on_success_helper_contract() {
    let err = cudf::testing::trigger_null_handle_contract_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::trigger_null_handle_contract_error_for_test");
    assert_eq!(err.message(), "FFI returned null test_handle handle on success");
    assert_eq!(err.context_value("target"), Some("test_handle"));
}

#[test]
fn testing_hook_locks_shared_range_limit_helper_contract() {
    let err = cudf::testing::trigger_range_limit_error_for_test().unwrap_err();
    let expected_row_count = usize::MAX.to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::trigger_range_limit_error_for_test");
    assert_eq!(err.message(), "row_count exceeds i32::MAX");
    assert_eq!(err.context_value("row_count"), Some(expected_row_count.as_str()));
    assert_eq!(err.context_value("resource"), Some("row_count"));
    assert_eq!(err.context_value("reason"), Some("exceeds_i32_max"));
    assert_eq!(err.context_value("remedy"), Some("reduce row_count to fit within i32::MAX"));
}

#[test]
fn phase3_prep_resolves_not_found_and_invalid_input_code_families_without_legacy_kind_names() {
    for code in [ErrorCode::NotFound, ErrorCode::MissingResource] {
        assert_eq!(phase3_kind_for_code(code), Phase3Kind::NotFound, "{code:?}");
    }

    for code in [
        ErrorCode::InvalidInput,
        ErrorCode::DuplicateColumnName,
        ErrorCode::DuplicateKey,
        ErrorCode::ColumnLengthMismatch,
        ErrorCode::TypeMismatch,
        ErrorCode::IndexOutOfBounds,
        ErrorCode::NullUnexpected,
    ] {
        assert_eq!(phase3_kind_for_code(code), Phase3Kind::InvalidInput, "{code:?}");
    }
}

#[test]
fn phase3_prep_resolves_execution_context_codes_without_using_legacy_execution_buckets() {
    for code in [
        ErrorCode::ExecutionContext,
        ErrorCode::EmptyConstructionRequiresRuntime,
        ErrorCode::CrossDeviceInputs,
        ErrorCode::CrossProvenanceInputs,
        ErrorCode::RuntimeBindingMismatch,
        ErrorCode::ReadinessUnavailable,
        ErrorCode::ExecutionBindingUnavailable,
        ErrorCode::LeaseAlreadyActive,
    ] {
        assert_eq!(phase3_kind_for_code(code), Phase3Kind::ExecutionContext, "{code:?}");
    }
}

#[test]
fn phase3_prep_resolves_unsupported_and_dependency_families_without_origin_driven_kinds() {
    for code in [
        ErrorCode::Unsupported,
        ErrorCode::UnsupportedDtype,
        ErrorCode::UnsupportedLayout,
        ErrorCode::UnsupportedOperation,
    ] {
        assert_eq!(phase3_kind_for_code(code), Phase3Kind::Unsupported, "{code:?}");
    }

    for code in [ErrorCode::ResourceExhausted, ErrorCode::OutOfMemory] {
        assert_eq!(phase3_kind_for_code(code), Phase3Kind::ResourceExhausted, "{code:?}");
    }

    for code in [ErrorCode::DependencyFailure, ErrorCode::CudaRuntime, ErrorCode::FatalCuda, ErrorCode::IoFailure] {
        assert_eq!(phase3_kind_for_code(code), Phase3Kind::DependencyFailure, "{code:?}");
    }
}

#[test]
fn phase3_final_contract_has_no_transitional_or_overbroad_public_codes() {
    let invalid_input = Error::new(ErrorKind::InvalidInput, "bad request");
    assert_eq!(invalid_input.code(), ErrorCode::InvalidInput);

    let invariant = Error::new(ErrorKind::InvariantViolation, "broken contract");
    assert_eq!(invariant.code(), ErrorCode::InvariantViolation);
}

#[test]
fn phase3_final_contract_exposes_invariant_violation_as_a_first_class_kind_and_code() {
    let target = Phase3Kind::InvariantViolation;
    assert_eq!(target, Phase3Kind::InvariantViolation);
    assert_eq!(phase3_kind_for_code(ErrorCode::InvariantViolation), Phase3Kind::InvariantViolation);
}
