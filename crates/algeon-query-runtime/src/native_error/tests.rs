use std::collections::BTreeMap;

use cudf::{io::parquet::RemoteParquetSource, testing::error::new_with_for_test};
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
    use cudf::testing::error;
    let mut error = error::new_with_for_test(code, "cudf.test.allocation_attribution", "synthetic cudf failure");
    if let Some(failure_source) = failure_source {
        error = error::with_ffi_detail_for_test(
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
        error = error::with_budget_exhaustion_for_test(
            error,
            error::budget_exhaustion_detail_for_test(
                1024,
                1024,
                1024,
                512,
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
fn cudf_value_errors_from_checked_arithmetic_classify_as_invalid_input() {
    for code in [cudf::ErrorCode::DivisionByZero, cudf::ErrorCode::ValueOutOfRange] {
        let source = new_with_for_test(code, "cudf.test.checked_arithmetic", "synthetic failure");

        let error = cudf_query_error("query_engine.test.from_cudf", "wrapped cudf failure", source);

        assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput, "{code}");
        assert_eq!(error.fact_value("cudf_code"), Some(code.to_string().as_str()));
        assert_eq!(error.fact_value("cudf_status"), Some("permanent"));
    }
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

#[test]
fn cudf_budget_scope_becomes_typed_memory_context_projected_under_budget_keys() {
    let error = cudf_query_error(
        "query_engine.test.budget_scope",
        "wrapped cudf failure",
        cudf_allocation_error_for_test(
            cudf::ErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::Physical),
            true,
        ),
    );

    // A scope snapshot never turns a physical failure into a cap failure.
    assert_eq!(error.code(), QueryErrorCode::DeviceOutOfMemory);
    let memory = error.context().memory.as_ref().expect("the budget scope is typed context");
    assert_eq!(memory.boundary, "test_boundary");
    assert_eq!(memory.operation, "cudf.test.allocation_attribution");
    let usage = memory.usage.expect("the scope carried member counters");
    assert_eq!((usage.limit_bytes, usage.current_aligned_bytes, usage.failed_allocation_count), (1024, 512, 1));
    let wire = error.wire_facts().map(|(key, value)| (key, value.into_owned())).collect::<BTreeMap<_, _>>();
    for (key, value) in [
        ("budget_boundary", "test_boundary"),
        ("budget_operation", "cudf.test.allocation_attribution"),
        ("budget_enforcement", "query_domain_member"),
        ("budget_status", "exceeded_or_lower_layer_rejected"),
        ("budget_bytes", "1024"),
        ("budget_peak_bytes", "1024"),
        ("budget_peak_aligned_bytes", "1024"),
        ("budget_current_bytes", "512"),
        ("budget_current_aligned_bytes", "512"),
        ("budget_largest_failed_allocation_bytes", "4096"),
        ("budget_failed_allocation_count", "1"),
    ] {
        assert_eq!(wire.get(key).map(String::as_str), Some(value), "{key}");
    }
    assert_eq!(error.fact_value("budget_bytes"), None, "budget keys are projected, not stored");

    let unscoped = cudf_query_error(
        "query_engine.test.budget_scope",
        "wrapped cudf failure",
        cudf_allocation_error_for_test(cudf::ErrorCode::OutOfMemory, None, false),
    );
    assert!(unscoped.context().memory.is_none());
    assert!(unscoped.wire_facts().all(|(key, _)| !key.starts_with("budget_")));
}
