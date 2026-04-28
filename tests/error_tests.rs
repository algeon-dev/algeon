use datafusion::common::DataFusionError;
use datafusion_cudf::{Error, ErrorCode, ErrorKind, ErrorStatus};

#[test]
fn error_code_projections_are_stable_policy_contracts() {
    assert_eq!(ErrorCode::UnsupportedExpression.as_str(), "unsupported_expression");
    assert_eq!(ErrorCode::UnsupportedExpression.kind(), ErrorKind::Unsupported);
    assert_eq!(ErrorCode::UnsupportedExpression.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::CudfResourceExhausted.as_str(), "cudf_resource_exhausted");
    assert_eq!(ErrorCode::CudfResourceExhausted.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::CudfResourceExhausted.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::CudfPersistentFailure.as_str(), "cudf_persistent_failure");
    assert_eq!(ErrorCode::CudfPersistentFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::CudfPersistentFailure.status(), ErrorStatus::Persistent);
}

#[test]
fn error_keeps_machine_code_operation_and_context_facts() {
    let error = Error::new(
        ErrorCode::GpuCandidateRejected,
        "datafusion_cudf.optimize_plan",
        "GPU candidate rejected by adapter policy",
    )
    .with_fact("fallback_reason", "insufficient_island_size")
    .with_fact("candidate_shape", "FilterExec(child)");

    assert_eq!(error.code(), ErrorCode::GpuCandidateRejected);
    assert_eq!(error.kind(), ErrorKind::Planning);
    assert_eq!(error.status(), ErrorStatus::Permanent);
    assert_eq!(error.operation(), "datafusion_cudf.optimize_plan");
    assert_eq!(error.fact_value("fallback_reason"), Some("insufficient_island_size"));

    let display = error.to_string();
    assert!(display.contains("code=gpu_candidate_rejected"));
    assert!(display.contains("operation=datafusion_cudf.optimize_plan"));
    assert!(display.contains("fallback_reason=insufficient_island_size"));
}

#[test]
fn datafusion_conversion_preserves_typed_adapter_error() {
    let error = Error::new(
        ErrorCode::InvalidExecutionPlan,
        "datafusion_cudf.gpu_island.try_new",
        "cannot build GPU island from physical plan",
    )
    .with_fact("plan_node", "EmptyExec");

    let datafusion_error: DataFusionError = error.into();
    let DataFusionError::External(source) = datafusion_error else {
        panic!("adapter errors must cross the DataFusion boundary as typed external errors");
    };
    let adapter_error =
        source.downcast_ref::<Error>().expect("DataFusion external error should preserve datafusion_cudf::Error");

    assert_eq!(adapter_error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(adapter_error.kind(), ErrorKind::Planning);
    assert_eq!(adapter_error.fact_value("plan_node"), Some("EmptyExec"));
}
