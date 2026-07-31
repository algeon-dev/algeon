use std::error::Error as StdError;

use rapids_interop::{InteropError, InteropErrorKind};

use crate::error::{Error, ErrorKind, ErrorStatus};

pub(super) fn map_runtime_ordering_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to establish runtime-stream ordering for cudf interop input",
        classify_runtime_ordering_interop_error,
    )
}

pub(super) fn map_prepared_lease_activation_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to activate prepared export read lease",
        classify_prepared_lease_interop_error,
    )
}

pub(super) fn map_prepared_lease_completion_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to complete prepared export read lease",
        classify_prepared_lease_interop_error,
    )
}

pub(super) fn map_owned_import_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to import shared interop input into the consumer cudf runtime",
        classify_owned_import_interop_error,
    )
}

pub(super) fn map_borrowed_projection_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to project cudf table view into borrowed RAPIDS interop",
        classify_projection_interop_error,
    )
}

pub(super) fn map_prepared_projection_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to prepare cudf-owned input for prepared device export",
        classify_projection_interop_error,
    )
}

fn classify_runtime_ordering_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn classify_prepared_lease_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn classify_owned_import_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn classify_projection_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn map_interop_boundary_error(
    error: InteropError,
    operation: &'static str,
    message: &'static str,
    classify: fn(InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>),
) -> Error {
    let (kind, explicit_status) = classify(error.kind());
    let mut mapped = Error::new(kind, operation).with_message(interop_boundary_message(message, error.message()));
    if let Some(status) = explicit_status.or_else(|| interop_error_source_status(&error)) {
        mapped = apply_error_status(mapped, status);
    }
    mapped = mapped.with_context("interop_kind", format!("{:?}", error.kind()));
    mapped = mapped.with_context("interop_message", error.message().to_string());

    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }

    mapped.set_source(error)
}

fn interop_boundary_message(boundary_message: &'static str, leaf_message: &str) -> String {
    let leaf_message = leaf_message.trim();
    if leaf_message.is_empty() { boundary_message.to_string() } else { format!("{boundary_message}: {leaf_message}") }
}

fn interop_error_source_status(error: &InteropError) -> Option<ErrorStatus> {
    let mut source = error.source();
    while let Some(current) = source {
        if let Some(cudf_error) = current.downcast_ref::<cudf::Error>() {
            return Some(match cudf_error.status() {
                cudf::ErrorStatus::Permanent => ErrorStatus::Permanent,
                cudf::ErrorStatus::Temporary => ErrorStatus::Temporary,
                cudf::ErrorStatus::Persistent => ErrorStatus::Persistent,
            });
        }
        source = current.source();
    }
    None
}

fn apply_error_status(mapped: Error, status: ErrorStatus) -> Error {
    match status {
        ErrorStatus::Permanent => mapped.set_permanent(),
        ErrorStatus::Temporary => mapped.set_temporary(),
        ErrorStatus::Persistent => mapped.set_persistent(),
    }
}

fn map_cudf_error_code(code: cudf::ErrorCode) -> ErrorKind {
    match code {
        cudf::ErrorCode::MissingColumn
        | cudf::ErrorCode::DuplicateColumnName
        | cudf::ErrorCode::DuplicateKey
        | cudf::ErrorCode::ColumnLengthMismatch
        | cudf::ErrorCode::TypeMismatch
        | cudf::ErrorCode::IndexOutOfBounds
        | cudf::ErrorCode::NullUnexpected
        | cudf::ErrorCode::CrossDeviceInputs
        | cudf::ErrorCode::CrossProvenanceInputs
        | cudf::ErrorCode::ForeignInvalidArgument => ErrorKind::InvalidInput,
        cudf::ErrorCode::EmptyConstructionRequiresRuntime
        | cudf::ErrorCode::RuntimeBindingMismatch
        | cudf::ErrorCode::ReadinessUnavailable
        | cudf::ErrorCode::ExecutionBindingUnavailable
        | cudf::ErrorCode::LeaseAlreadyActive
        | cudf::ErrorCode::InvariantViolation
        | cudf::ErrorCode::ForeignInvariantViolation => ErrorKind::InvalidState,
        cudf::ErrorCode::UnsupportedDtype
        | cudf::ErrorCode::UnsupportedLayout
        | cudf::ErrorCode::UnsupportedOperation => ErrorKind::UnsupportedType,
        cudf::ErrorCode::OutOfMemory | cudf::ErrorCode::ForeignResourceExhausted => ErrorKind::OutOfMemory,
        cudf::ErrorCode::IoFailure => ErrorKind::IoError,
        cudf::ErrorCode::CudaRuntime
        | cudf::ErrorCode::FatalCuda
        | cudf::ErrorCode::ForeignTransientDependencyFailure
        | cudf::ErrorCode::ForeignPersistentDependencyFailure => ErrorKind::Unexpected,
        _ => match cudf::core::error::kind_for_code(code) {
            cudf::ErrorKind::InvalidInput => ErrorKind::InvalidInput,
            cudf::ErrorKind::NotFound => ErrorKind::IoError,
            cudf::ErrorKind::Unsupported => ErrorKind::UnsupportedType,
            cudf::ErrorKind::ExecutionContext => ErrorKind::InvalidState,
            cudf::ErrorKind::ResourceExhausted => ErrorKind::OutOfMemory,
            cudf::ErrorKind::DependencyFailure => ErrorKind::Unexpected,
            cudf::ErrorKind::InvariantViolation => ErrorKind::InvalidState,
            _ => ErrorKind::Unexpected,
        },
    }
}

fn cudf_fact_context_key(key: cudf::core::error::FactKeyName) -> &'static str {
    match key {
        cudf::core::error::FactKeyName::ColumnName => "cudf_column_name",
        cudf::core::error::FactKeyName::ColumnIndex => "cudf_column_index",
        cudf::core::error::FactKeyName::ExpectedRows => "cudf_expected_rows",
        cudf::core::error::FactKeyName::ActualRows => "cudf_actual_rows",
        cudf::core::error::FactKeyName::ExpectedCount => "cudf_expected_count",
        cudf::core::error::FactKeyName::ActualCount => "cudf_actual_count",
        cudf::core::error::FactKeyName::LeftTypeId => "cudf_left_type_id",
        cudf::core::error::FactKeyName::RightTypeId => "cudf_right_type_id",
        cudf::core::error::FactKeyName::LeftDevice => "cudf_left_device",
        cudf::core::error::FactKeyName::RightDevice => "cudf_right_device",
        cudf::core::error::FactKeyName::TableRowCount => "cudf_table_row_count",
        cudf::core::error::FactKeyName::TableColumnCount => "cudf_table_column_count",
        cudf::core::error::FactKeyName::KeysCount => "cudf_keys_count",
        cudf::core::error::FactKeyName::DistinctKeysCount => "cudf_distinct_keys_count",
        cudf::core::error::FactKeyName::TypeId => "cudf_type_id",
        _ => "cudf_fact",
    }
}

fn map_cudf_status(mut mapped: Error, status: cudf::ErrorStatus) -> Error {
    match status {
        cudf::ErrorStatus::Permanent => mapped = mapped.set_permanent(),
        cudf::ErrorStatus::Temporary => mapped = mapped.set_temporary(),
        cudf::ErrorStatus::Persistent => mapped = mapped.set_persistent(),
    }
    mapped
}

fn attach_cudf_error_context(mut mapped: Error, error: &cudf::Error) -> Error {
    mapped = mapped.with_context("cudf_code", error.code().to_string());
    mapped = mapped.with_context("cudf_kind", error.kind().to_string());
    mapped = mapped.with_context("cudf_status", error.status().to_string());
    mapped = mapped.with_context("cudf_operation", error.operation().to_string());

    for called in error.call_chain() {
        mapped = mapped.with_context("cudf_called", (*called).to_string());
    }

    for (key, value) in error.facts().iter() {
        mapped = mapped.with_context(cudf_fact_context_key(key), value.to_string());
    }

    for (key, value) in error.details() {
        mapped = mapped.with_context("cudf_detail", format!("{key}={value}"));
    }

    mapped
}

#[cfg(test)]
fn map_cudf_error(error: cudf::Error, operation: &'static str) -> Error {
    map_cudf_error_with(error, operation, "cudf operation failed", std::iter::empty())
}

pub(super) fn map_cudf_error_with<I>(
    error: cudf::Error,
    operation: &'static str,
    message: impl Into<String>,
    context: I,
) -> Error
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    let source_code = error.code();
    let source_status = error.status();
    let mut mapped = Error::new(map_cudf_error_code(source_code), operation).with_message(message.into());
    mapped = map_cudf_status(mapped, source_status);
    mapped = attach_cudf_error_context(mapped, &error);

    for (key, value) in context {
        mapped = mapped.with_context(key, value);
    }

    mapped.set_source(error)
}

#[cfg(test)]
mod tests {
    use rapids_interop::{InteropError, InteropErrorKind};

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn map_cudf_error_missing_column_is_invalid_input() {
        let cudf_err = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::MissingColumn,
            "cudf::testing::missing_column",
            "column 'foo' not found",
        );
        let err =
            map_cudf_error_with(cudf_err, "test_op", "input adapter failed", [("column_role", "src".to_string())]);
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.is_permanent(), "missing columns should be permanent");
        assert_eq!(err.message(), "input adapter failed");
        assert!(err.to_string().contains("cudf_code"), "mapped error should retain cudf metadata");
    }

    #[test]
    fn map_cudf_error_runtime_binding_mismatch_is_invalid_state() {
        let cudf_err = cudf::testing::error::reanchor_for_test(
            cudf::testing::error::new_with_for_test(
                cudf::ErrorCode::RuntimeBindingMismatch,
                "cudf::testing::runtime_binding::inner",
                "runtime must not use ambient provenance",
            ),
            "cudf::testing::runtime_binding::outer",
        );
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "runtime binding mismatches should be permanent");
        assert!(err.to_string().contains("cudf_called"), "call-chain breadcrumbs should be preserved");
    }

    #[test]
    fn map_cudf_error_cuda_runtime_is_temporary_unexpected() {
        let cudf_err = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::CudaRuntime,
            "cudf::testing::cuda_runtime",
            "device launch failed",
        );
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "CUDA runtime failures should be temporary");
    }

    #[test]
    fn map_cudf_error_out_of_memory() {
        let cudf_err =
            cudf::testing::error::new_with_for_test(cudf::ErrorCode::OutOfMemory, "cudf::testing::oom", "alloc failed");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.is_temporary(), "OOM should be temporary");
    }

    #[test]
    fn map_cudf_error_io_error() {
        let cudf_err =
            cudf::testing::error::new_with_for_test(cudf::ErrorCode::IoFailure, "cudf::testing::io", "file not found");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::IoError);
        assert!(err.is_permanent(), "I/O errors should be permanent by default");
    }

    #[test]
    fn map_cudf_error_fatal_cuda_is_persistent_unexpected() {
        let cudf_err = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::FatalCuda,
            "cudf::testing::fatal_cuda",
            "device entered fatal state",
        );
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_persistent(), "fatal CUDA failures should be persistent");
    }

    #[test]
    fn map_cudf_error_foreign_invariant_violation_is_invalid_state() {
        let cudf_err = cudf::testing::error::foreign_invariant_violation_for_test();
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "foreign invariant violations should be permanent");
        assert!(err.to_string().contains("cudf_detail"), "foreign details should be preserved");
    }

    #[test]
    fn runtime_ordering_interop_execution_binding_unavailable_is_invalid_state() {
        let interop_err = InteropError::new(
            InteropErrorKind::ExecutionBindingUnavailable,
            "producer must provide explicit execution provenance",
        );
        let err = map_runtime_ordering_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "missing runtime ordering provenance is a contract failure, not a retryable one");
    }

    #[test]
    fn runtime_ordering_interop_readiness_unavailable_uses_cudf_source_status() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to query producer event")
            .set_source(cudf::testing::error::new_with_for_test(
                cudf::ErrorCode::CudaRuntime,
                "cudf::testing::cuda_runtime",
                "device launch failed",
            ));
        let err = map_runtime_ordering_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "runtime ordering should inherit retryability from the underlying CUDA failure");
    }

    #[test]
    fn runtime_ordering_interop_readiness_unavailable_without_source_is_permanent() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to query producer event");
        let err = map_runtime_ordering_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(
            err.is_permanent(),
            "runtime ordering must not invent retryability when the interop boundary did not prove it"
        );
    }

    #[test]
    fn prepared_lease_activation_interop_lease_already_active_is_invalid_state() {
        let interop_err = InteropError::new(
            InteropErrorKind::LeaseAlreadyActive,
            "prepared export lease activation requires an inactive export root",
        );
        let err = map_prepared_lease_activation_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "lease lifecycle violations should remain permanent invalid-state failures");
    }

    #[test]
    fn prepared_lease_activation_interop_preserves_leaf_failure_message() {
        let interop_err = InteropError::new(InteropErrorKind::InvalidView, "test backend forced activation failure");
        let err = map_prepared_lease_activation_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(
            err.message().contains("test backend forced activation failure"),
            "top-level error message should preserve the leaf failure reason for fast diagnosis"
        );
    }

    #[test]
    fn prepared_lease_activation_interop_without_leaf_message_keeps_boundary_message_clean() {
        let interop_err = InteropError::new(InteropErrorKind::InvalidView, "");
        let err = map_prepared_lease_activation_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.message(), "failed to activate prepared export read lease");
    }

    #[test]
    fn prepared_lease_completion_interop_readiness_unavailable_uses_persistent_source_status() {
        let interop_err =
            InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to finalize producer lease").set_source(
                cudf::testing::error::new_with_for_test(
                    cudf::ErrorCode::FatalCuda,
                    "cudf::testing::fatal_cuda",
                    "device entered fatal state",
                ),
            );
        let err = map_prepared_lease_completion_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_persistent(), "lease completion should preserve persistent dependency failure semantics");
    }

    #[test]
    fn owned_import_interop_execution_binding_mismatch_is_invalid_input() {
        let interop_err = InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "prepared export import requires the destination runtime to use the same CUDA device",
        );
        let err = map_owned_import_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.is_permanent(), "cross-device import mismatches are caller input failures");
    }

    #[test]
    fn borrowed_projection_interop_invalid_view_is_invalid_state() {
        let interop_err =
            InteropError::new(InteropErrorKind::InvalidView, "borrowed table view is internally inconsistent");
        let err = map_borrowed_projection_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "borrowed projection invariants should surface as invalid-state failures");
    }

    #[test]
    fn prepared_projection_interop_readiness_unavailable_uses_temporary_source_status() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to project device export")
            .set_source(cudf::testing::error::new_with_for_test(
                cudf::ErrorCode::OutOfMemory,
                "cudf::testing::oom",
                "allocation failed",
            ));
        let err = map_prepared_projection_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "projection should preserve temporary source failures instead of flattening them");
    }
}
