#[cfg(feature = "cuml")]
use cudf::CudfAllocationAttribution;
#[cfg(feature = "cuml")]
use cuml::{ErrorCode, ErrorKind, ErrorStatus};
use query_planning::error::QueryErrorCode;

#[cfg(feature = "cuml")]
use super::classify_cuml_error;
#[cfg(feature = "cuvs")]
use super::cuvs_query_error;

#[cfg(feature = "cuml")]
#[test]
fn cuml_error_projection_uses_typed_code_kind_status_and_allocation_source() {
    use CudfAllocationAttribution::{
        ConfirmedPhysical, ConfirmedQueryLimit, NonAllocationFailure, UnattributedAllocationFailure,
    };
    let cases = [
        (
            ErrorCode::HostOutOfMemory,
            ErrorKind::ResourceExhausted,
            ErrorStatus::Temporary,
            UnattributedAllocationFailure,
            QueryErrorCode::CudfTemporaryFailure,
        ),
        (
            ErrorCode::OutOfMemory,
            ErrorKind::ResourceExhausted,
            ErrorStatus::Temporary,
            ConfirmedQueryLimit,
            QueryErrorCode::QueryMemoryLimitExceeded,
        ),
        (
            ErrorCode::OutOfMemory,
            ErrorKind::ResourceExhausted,
            ErrorStatus::Temporary,
            ConfirmedPhysical,
            QueryErrorCode::DeviceOutOfMemory,
        ),
        (
            ErrorCode::OutOfMemory,
            ErrorKind::ResourceExhausted,
            ErrorStatus::Temporary,
            UnattributedAllocationFailure,
            QueryErrorCode::DeviceAllocationFailed,
        ),
        (
            ErrorCode::RuntimeBindingMismatch,
            ErrorKind::ExecutionContext,
            ErrorStatus::Permanent,
            NonAllocationFailure,
            QueryErrorCode::InternalInvariantViolation,
        ),
        (
            ErrorCode::CudaRuntime,
            ErrorKind::DependencyFailure,
            ErrorStatus::Temporary,
            NonAllocationFailure,
            QueryErrorCode::CudfTemporaryFailure,
        ),
        (
            ErrorCode::FatalCuda,
            ErrorKind::DependencyFailure,
            ErrorStatus::Persistent,
            NonAllocationFailure,
            QueryErrorCode::DeviceFault,
        ),
        (
            ErrorCode::ForeignUnknown,
            ErrorKind::DependencyFailure,
            ErrorStatus::Permanent,
            NonAllocationFailure,
            QueryErrorCode::CudfDependencyFailure,
        ),
    ];
    for (code, kind, status, attribution, expected) in cases {
        assert_eq!(classify_cuml_error(code, kind, status, attribution), expected, "{code:?} {attribution:?}");
    }
}

#[cfg(feature = "cuvs")]
#[test]
fn cuvs_query_error_maps_only_typed_kind_and_status() {
    let cases = [
        (cuvs::Error::new(cuvs::ErrorKind::InvalidInput, "cuvs.invalid"), QueryErrorCode::CudfInvalidInput),
        (cuvs::Error::new(cuvs::ErrorKind::Unsupported, "cuvs.unsupported"), QueryErrorCode::CudfUnsupported),
        (cuvs::Error::new(cuvs::ErrorKind::OutOfMemory, "cuvs.oom"), QueryErrorCode::DeviceAllocationFailed),
        (cuvs::Error::new(cuvs::ErrorKind::InvalidState, "cuvs.state"), QueryErrorCode::InternalInvariantViolation),
        (
            cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.temporary")
                .with_context("native_status", "CUVS_ERROR")
                .set_temporary(),
            QueryErrorCode::CudfTemporaryFailure,
        ),
        (
            cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.persistent").set_persistent(),
            QueryErrorCode::DeviceFault,
        ),
    ];
    for (source, expected) in cases {
        let error = cuvs_query_error("query_engine.test.from_cuvs", "wrapped cuVS failure", source);
        assert_eq!(error.code(), expected);
        assert!(error.fact_value("cuvs_kind").is_some());
        assert!(error.fact_value("cuvs_status").is_some());
        assert!(error.fact_value("cuvs_operation").is_some());
        if error.fact_value("cuvs_operation") == Some("cuvs.temporary") {
            assert_eq!(error.fact_value("native_status"), Some("CUVS_ERROR"));
        }
    }
}

#[cfg(feature = "cuvs")]
#[test]
fn cuvs_query_error_duplicate_context_projects_effective_value_without_panic() {
    let source = cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.duplicate")
        .with_context("native_status", "CUVS_ERROR")
        .with_context("native_status", "CUVS_ERROR_RETRY")
        .set_temporary();
    assert_eq!(source.context_value("native_status"), Some("CUVS_ERROR_RETRY"));

    let error = cuvs_query_error("query_engine.test.from_cuvs_duplicate", "wrapped cuVS failure", source);

    assert_eq!(error.code(), QueryErrorCode::CudfTemporaryFailure);
    assert_eq!(error.fact_value("native_status"), Some("CUVS_ERROR_RETRY"));
    assert!(error.source_error().is_some());
}
