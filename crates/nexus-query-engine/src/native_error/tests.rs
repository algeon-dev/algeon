use nexus_query_planning::error::QueryErrorCode;

use super::cuvs_query_error;

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
        let error = cuvs_query_error("nexus_query_engine.test.from_cuvs", "wrapped cuVS failure", source);
        assert_eq!(error.code(), expected);
        assert!(error.fact_value("cuvs_kind").is_some());
        assert!(error.fact_value("cuvs_status").is_some());
        assert!(error.fact_value("cuvs_operation").is_some());
        if error.fact_value("cuvs_operation") == Some("cuvs.temporary") {
            assert_eq!(error.fact_value("native_status"), Some("CUVS_ERROR"));
        }
    }
}

#[test]
fn cuvs_query_error_duplicate_context_projects_effective_value_without_panic() {
    let source = cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.duplicate")
        .with_context("native_status", "CUVS_ERROR")
        .with_context("native_status", "CUVS_ERROR_RETRY")
        .set_temporary();
    assert_eq!(source.context_value("native_status"), Some("CUVS_ERROR_RETRY"));

    let error = cuvs_query_error("nexus_query_engine.test.from_cuvs_duplicate", "wrapped cuVS failure", source);

    assert_eq!(error.code(), QueryErrorCode::CudfTemporaryFailure);
    assert_eq!(error.fact_value("native_status"), Some("CUVS_ERROR_RETRY"));
    assert!(error.source_error().is_some());
}
