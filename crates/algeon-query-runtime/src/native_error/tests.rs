use cudf::io::parquet::RemoteParquetSource;
use query_planning::error::{QueryErrorCode, redact_uri_tokens};

use super::{cudf_query_error, cudf_query_error_without_source};

fn sample_cudf_error() -> cudf::Error {
    RemoteParquetSource::s3("gs://bucket/key.parquet").expect_err("non-s3 scheme must be rejected")
}

fn cudf_allocation_error_for_test(
    code: cudf::ErrorCode,
    failure_source: Option<cudf::CudfErrorFailureSource>,
    with_budget_scope: bool,
) -> cudf::Error {
    use cudf::testing::error as cudf_test_error;
    let mut error =
        cudf_test_error::new_with_for_test(code, "cudf.test.allocation_attribution", "synthetic cudf failure");
    if let Some(failure_source) = failure_source {
        error = cudf_test_error::with_ffi_detail_for_test(
            error,
            cudf::CudfErrorDetail {
                status: 0,
                failure_source,
                domain_id: 0,
                member_id: 0,
                generation: 0,
                requested_bytes: 4096,
                alignment: 256,
                cuda_error: -1,
                message: "synthetic allocation failure".to_string(),
            },
        );
    }
    if with_budget_scope {
        error = cudf_test_error::with_budget_exhaustion_for_test(
            error,
            cudf_test_error::budget_exhaustion_detail_for_test(
                1024,
                1024,
                1024,
                512,
                4096,
                1,
                "test_boundary",
                "cudf.test.allocation_attribution",
            ),
        );
    }
    error
}

#[test]
fn cudf_query_error_preserves_foreign_message_as_cudf_message_fact() {
    let source = sample_cudf_error();
    let expected_message = redact_uri_tokens(source.message().trim());

    let error = cudf_query_error("query_engine.test.from_cudf", "wrapped cudf failure", source);

    assert_eq!(error.fact_value("cudf_message"), Some(expected_message.as_str()));
    assert!(error.source_error().is_some());
}

#[test]
fn cudf_query_error_without_source_preserves_foreign_message_without_source_chain() {
    let source = sample_cudf_error();
    let expected_message = redact_uri_tokens(source.message().trim());

    let error =
        cudf_query_error_without_source("query_engine.test.from_cudf_without_source", "wrapped cudf failure", &source);

    assert_eq!(error.fact_value("cudf_message"), Some(expected_message.as_str()));
    assert!(error.source_error().is_none());
}

#[test]
fn cudf_allocation_attribution_drives_native_classification_not_scope_usage() {
    let cases = [
        (
            cudf::ErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::ReservationLimit),
            true,
            QueryErrorCode::QueryMemoryLimitExceeded,
        ),
        (
            cudf::ErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::Physical),
            true,
            QueryErrorCode::DeviceOutOfMemory,
        ),
        (
            cudf::ErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::Unknown),
            true,
            QueryErrorCode::DeviceAllocationFailed,
        ),
        (cudf::ErrorCode::OutOfMemory, None, true, QueryErrorCode::DeviceAllocationFailed),
        (cudf::ErrorCode::IoFailure, None, true, QueryErrorCode::CudfDependencyFailure),
    ];

    for (index, (code, failure_source, with_budget_scope, expected)) in cases.into_iter().enumerate() {
        let error = cudf_query_error(
            "query_engine.test.allocation_attribution",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(code, failure_source, with_budget_scope),
        );
        assert_eq!(error.code(), expected, "native mismatch for case {index}");
    }
}
