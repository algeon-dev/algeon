use std::{collections::HashSet, sync::Arc};

use algeon_datafusion::error::{DataFusionErrorExt, Error, ErrorCode, ErrorKind, ErrorStatus, Result};
use arrow::error::ArrowError;
use datafusion::common::{DataFusionError, Diagnostic, Result as DataFusionResult};
use query_engine::{
    error::{QueryError, QueryErrorCode, QueryErrorStatus},
    runtime::ExecutionCancellation,
    triage::{TriageCode, TriageVerdict},
};

const ADAPTER_STATUS_CASES: &[(ErrorCode, ErrorStatus)] = &[
    (ErrorCode::Cancelled, ErrorStatus::Permanent),
    (ErrorCode::SqlParse, ErrorStatus::Permanent),
    (ErrorCode::InvalidRequest, ErrorStatus::Permanent),
    (ErrorCode::DataFusionPlanning, ErrorStatus::Permanent),
    (ErrorCode::PlanningTimeout, ErrorStatus::Permanent),
    (ErrorCode::ObjectNotFound, ErrorStatus::Permanent),
    (ErrorCode::SourceAccessDenied, ErrorStatus::Permanent),
    (ErrorCode::SourceUnauthenticated, ErrorStatus::Permanent),
    (ErrorCode::SourceObjectNotFound, ErrorStatus::Permanent),
    (ErrorCode::SourceTemporaryFailure, ErrorStatus::Temporary),
    (ErrorCode::SourceDependencyFailure, ErrorStatus::Permanent),
    (ErrorCode::FieldNotFound, ErrorStatus::Permanent),
    (ErrorCode::GraphInputInvalid, ErrorStatus::Permanent),
    (ErrorCode::RequiredGpuOperationUnsupported, ErrorStatus::Permanent),
    (ErrorCode::FinalPlanRequirementUnsatisfied, ErrorStatus::Permanent),
    (ErrorCode::UnsupportedOperator, ErrorStatus::Permanent),
    (ErrorCode::UnsupportedExpression, ErrorStatus::Permanent),
    (ErrorCode::UnsupportedType, ErrorStatus::Permanent),
    (ErrorCode::InvalidExecutionPlan, ErrorStatus::Permanent),
    (ErrorCode::InvalidChildCount, ErrorStatus::Permanent),
    (ErrorCode::CudfInvalidInput, ErrorStatus::Permanent),
    (ErrorCode::CudfUnsupported, ErrorStatus::Permanent),
    (ErrorCode::SizeOverflow, ErrorStatus::Permanent),
    (ErrorCode::HostMaterializationLimitExceeded, ErrorStatus::Permanent),
    (ErrorCode::HostStagingLimitExceeded, ErrorStatus::Permanent),
    (ErrorCode::HostMemoryExhausted, ErrorStatus::Temporary),
    (ErrorCode::CudfTemporaryFailure, ErrorStatus::Temporary),
    (ErrorCode::CudfPersistentFailure, ErrorStatus::Persistent),
    (ErrorCode::CudfDependencyFailure, ErrorStatus::Permanent),
    (ErrorCode::AlgorithmDidNotConverge, ErrorStatus::Permanent),
    (ErrorCode::CudfInvariantViolation, ErrorStatus::Permanent),
    (ErrorCode::MemoryReservationFailed, ErrorStatus::Temporary),
    (ErrorCode::ServiceOverloaded, ErrorStatus::Temporary),
    (ErrorCode::QueryBudgetUnavailable, ErrorStatus::Temporary),
    (ErrorCode::QueryMemoryLimitExceeded, ErrorStatus::Permanent),
    (ErrorCode::DeviceOutOfMemory, ErrorStatus::Temporary),
    (ErrorCode::DeviceAllocationFailed, ErrorStatus::Temporary),
    (ErrorCode::DeviceFault, ErrorStatus::Persistent),
    (ErrorCode::CleanupFailed, ErrorStatus::Persistent),
    (ErrorCode::InternalInvariantViolation, ErrorStatus::Permanent),
    (ErrorCode::FlightFrameTooLarge, ErrorStatus::Permanent),
    (ErrorCode::ServerTransportFailure, ErrorStatus::Permanent),
    (ErrorCode::Internal, ErrorStatus::Permanent),
];

const NATIVE_ADAPTER_STATUS_CASES: &[(QueryErrorCode, ErrorCode, QueryErrorStatus)] = &[
    (QueryErrorCode::Cancelled, ErrorCode::Cancelled, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedPipelineContract, ErrorCode::UnsupportedOperator, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedOperator, ErrorCode::UnsupportedOperator, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedExpression, ErrorCode::UnsupportedExpression, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedType, ErrorCode::UnsupportedType, QueryErrorStatus::Permanent),
    (QueryErrorCode::InvalidPlan, ErrorCode::InvalidExecutionPlan, QueryErrorStatus::Permanent),
    (QueryErrorCode::InvalidAttemptCorrelation, ErrorCode::InvalidRequest, QueryErrorStatus::Permanent),
    (QueryErrorCode::GraphInputInvalid, ErrorCode::GraphInputInvalid, QueryErrorStatus::Permanent),
    (QueryErrorCode::CudfInvalidInput, ErrorCode::CudfInvalidInput, QueryErrorStatus::Permanent),
    (QueryErrorCode::CudfUnsupported, ErrorCode::CudfUnsupported, QueryErrorStatus::Permanent),
    (QueryErrorCode::SizeOverflow, ErrorCode::SizeOverflow, QueryErrorStatus::Permanent),
    (
        QueryErrorCode::HostMaterializationLimitExceeded,
        ErrorCode::HostMaterializationLimitExceeded,
        QueryErrorStatus::Permanent,
    ),
    (QueryErrorCode::CudfTemporaryFailure, ErrorCode::CudfTemporaryFailure, QueryErrorStatus::Temporary),
    (QueryErrorCode::CudfPersistentFailure, ErrorCode::CudfPersistentFailure, QueryErrorStatus::Persistent),
    (QueryErrorCode::CudfDependencyFailure, ErrorCode::CudfDependencyFailure, QueryErrorStatus::Permanent),
    (QueryErrorCode::AlgorithmDidNotConverge, ErrorCode::AlgorithmDidNotConverge, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceAccessDenied, ErrorCode::SourceAccessDenied, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceUnauthenticated, ErrorCode::SourceUnauthenticated, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceObjectNotFound, ErrorCode::SourceObjectNotFound, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceTemporaryFailure, ErrorCode::SourceTemporaryFailure, QueryErrorStatus::Temporary),
    (QueryErrorCode::CudfInvariantViolation, ErrorCode::CudfInvariantViolation, QueryErrorStatus::Permanent),
    (QueryErrorCode::MemoryReservationFailed, ErrorCode::MemoryReservationFailed, QueryErrorStatus::Temporary),
    (QueryErrorCode::ServiceOverloaded, ErrorCode::ServiceOverloaded, QueryErrorStatus::Temporary),
    (QueryErrorCode::QueryBudgetUnavailable, ErrorCode::QueryBudgetUnavailable, QueryErrorStatus::Temporary),
    (QueryErrorCode::QueryMemoryLimitExceeded, ErrorCode::QueryMemoryLimitExceeded, QueryErrorStatus::Permanent),
    (QueryErrorCode::DeviceOutOfMemory, ErrorCode::DeviceOutOfMemory, QueryErrorStatus::Temporary),
    (QueryErrorCode::DeviceAllocationFailed, ErrorCode::DeviceAllocationFailed, QueryErrorStatus::Temporary),
    (QueryErrorCode::DeviceFault, ErrorCode::DeviceFault, QueryErrorStatus::Persistent),
    (QueryErrorCode::CleanupFailed, ErrorCode::CleanupFailed, QueryErrorStatus::Persistent),
    (QueryErrorCode::InternalInvariantViolation, ErrorCode::InternalInvariantViolation, QueryErrorStatus::Permanent),
    (QueryErrorCode::Internal, ErrorCode::Internal, QueryErrorStatus::Permanent),
];

#[test]
fn every_adapter_code_has_the_expected_retry_status() {
    let mapped_codes = ADAPTER_STATUS_CASES.iter().map(|(code, _)| *code).collect::<HashSet<_>>();
    let all_codes = ErrorCode::ALL.iter().copied().collect::<HashSet<_>>();
    assert_eq!(mapped_codes.len(), ADAPTER_STATUS_CASES.len(), "adapter status policy table contains duplicate codes");
    assert_eq!(
        mapped_codes, all_codes,
        "adapter status policy table must cover every ErrorCode::ALL variant exactly once"
    );

    for &(code, expected_status) in ADAPTER_STATUS_CASES {
        let error = Error::new(code, "algeon_datafusion.test.retry_policy", "retry policy test");

        assert_eq!(code.status(), expected_status, "unexpected status for {code}");
        assert_eq!(error.status(), expected_status, "Error disagrees with its code for {code}");
    }
}

#[test]
fn every_native_status_is_preserved_at_the_adapter_boundary() {
    let mapped_codes = NATIVE_ADAPTER_STATUS_CASES.iter().map(|(code, _, _)| *code).collect::<HashSet<_>>();
    let engine_codes = QueryErrorCode::ALL.iter().copied().collect::<HashSet<_>>();
    assert_eq!(mapped_codes.len(), NATIVE_ADAPTER_STATUS_CASES.len(), "native mapping table contains duplicate codes");
    assert_eq!(mapped_codes, engine_codes, "native mapping table must cover every engine error code");

    for &(native_code, expected_adapter_code, expected_native_status) in NATIVE_ADAPTER_STATUS_CASES {
        let native = QueryError::new(native_code, "query_engine.test.retry_policy", "retry policy test");
        assert_eq!(native_code.status(), expected_native_status, "unexpected native status for {native_code}");
        assert_eq!(native.status(), expected_native_status, "QueryError disagrees with its code for {native_code}");

        let adapter = Error::from_native("algeon_datafusion.test.retry_policy", "wrapped native failure", native);
        assert_eq!(adapter.code(), expected_adapter_code, "unexpected adapter mapping for {native_code}");
        assert_eq!(adapter.status().as_str(), expected_native_status.as_str(), "status changed for {native_code}");
        assert_eq!(adapter.fact_value("native_status"), Some(expected_native_status.as_str()));
    }
}

#[test]
fn error_code_projections_are_stable_policy_contracts() {
    assert_eq!(ErrorCode::Cancelled.as_str(), "cancelled");
    assert_eq!(ErrorCode::Cancelled.kind(), ErrorKind::Cancelled);
    assert_eq!(ErrorCode::Cancelled.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SqlParse.as_str(), "sql_parse");
    assert_eq!(ErrorCode::SqlParse.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::SqlParse.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::InvalidRequest.as_str(), "invalid_request");
    assert_eq!(ErrorCode::InvalidRequest.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::InvalidRequest.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::PlanningTimeout.as_str(), "planning_timeout");
    assert_eq!(ErrorCode::PlanningTimeout.kind(), ErrorKind::Planning);
    assert_eq!(ErrorCode::PlanningTimeout.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::ObjectNotFound.as_str(), "object_not_found");
    assert_eq!(ErrorCode::ObjectNotFound.kind(), ErrorKind::NotFound);
    assert_eq!(ErrorCode::ObjectNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceAccessDenied.as_str(), "source_access_denied");
    assert_eq!(ErrorCode::SourceAccessDenied.kind(), ErrorKind::PermissionDenied);
    assert_eq!(ErrorCode::SourceAccessDenied.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceUnauthenticated.as_str(), "source_unauthenticated");
    assert_eq!(ErrorCode::SourceUnauthenticated.kind(), ErrorKind::Unauthenticated);
    assert_eq!(ErrorCode::SourceUnauthenticated.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceObjectNotFound.as_str(), "source_object_not_found");
    assert_eq!(ErrorCode::SourceObjectNotFound.kind(), ErrorKind::NotFound);
    assert_eq!(ErrorCode::SourceObjectNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceTemporaryFailure.as_str(), "source_temporary_failure");
    assert_eq!(ErrorCode::SourceTemporaryFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::SourceTemporaryFailure.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::FieldNotFound.as_str(), "field_not_found");
    assert_eq!(ErrorCode::FieldNotFound.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::FieldNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::RequiredGpuOperationUnsupported.as_str(), "required_gpu_operation_unsupported");
    assert_eq!(ErrorCode::RequiredGpuOperationUnsupported.kind(), ErrorKind::Unsupported);
    assert_eq!(ErrorCode::RequiredGpuOperationUnsupported.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::FinalPlanRequirementUnsatisfied.as_str(), "final_plan_requirement_unsatisfied");
    assert_eq!(ErrorCode::FinalPlanRequirementUnsatisfied.kind(), ErrorKind::Planning);
    assert_eq!(ErrorCode::FinalPlanRequirementUnsatisfied.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::UnsupportedExpression.as_str(), "unsupported_expression");
    assert_eq!(ErrorCode::UnsupportedExpression.kind(), ErrorKind::Unsupported);
    assert_eq!(ErrorCode::UnsupportedExpression.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SizeOverflow.as_str(), "size_overflow");
    assert_eq!(ErrorCode::SizeOverflow.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::SizeOverflow.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::HostMaterializationLimitExceeded.as_str(), "host_materialization_limit_exceeded");
    assert_eq!(ErrorCode::HostMaterializationLimitExceeded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::HostMaterializationLimitExceeded.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::HostStagingLimitExceeded.as_str(), "host_staging_limit_exceeded");
    assert_eq!(ErrorCode::HostStagingLimitExceeded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::HostStagingLimitExceeded.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::HostMemoryExhausted.as_str(), "host_memory_exhausted");
    assert_eq!(ErrorCode::HostMemoryExhausted.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::HostMemoryExhausted.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::CudfPersistentFailure.as_str(), "cudf_persistent_failure");
    assert_eq!(ErrorCode::CudfPersistentFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::CudfPersistentFailure.status(), ErrorStatus::Persistent);

    assert_eq!(ErrorCode::MemoryReservationFailed.as_str(), "memory_reservation_failed");
    assert_eq!(ErrorCode::MemoryReservationFailed.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::MemoryReservationFailed.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::ServiceOverloaded.as_str(), "service_overloaded");
    assert_eq!(ErrorCode::ServiceOverloaded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::ServiceOverloaded.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::QueryBudgetUnavailable.as_str(), "query_budget_unavailable");
    assert_eq!(ErrorCode::QueryBudgetUnavailable.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::QueryBudgetUnavailable.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::QueryMemoryLimitExceeded.as_str(), "query_memory_limit_exceeded");
    assert_eq!(ErrorCode::QueryMemoryLimitExceeded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::QueryMemoryLimitExceeded.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::DeviceOutOfMemory.as_str(), "device_out_of_memory");
    assert_eq!(ErrorCode::DeviceOutOfMemory.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::DeviceOutOfMemory.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::DeviceAllocationFailed.as_str(), "device_allocation_failed");
    assert_eq!(ErrorCode::DeviceAllocationFailed.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::DeviceAllocationFailed.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::DeviceFault.as_str(), "device_fault");
    assert_eq!(ErrorCode::DeviceFault.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::DeviceFault.status(), ErrorStatus::Persistent);

    assert_eq!(ErrorCode::CleanupFailed.as_str(), "cleanup_failed");
    assert_eq!(ErrorCode::CleanupFailed.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::CleanupFailed.status(), ErrorStatus::Persistent);

    assert_eq!(ErrorCode::InternalInvariantViolation.as_str(), "internal_invariant_violation");
    assert_eq!(ErrorCode::InternalInvariantViolation.kind(), ErrorKind::InvariantViolation);
    assert_eq!(ErrorCode::InternalInvariantViolation.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::FlightFrameTooLarge.as_str(), "flight_frame_too_large");
    assert_eq!(ErrorCode::FlightFrameTooLarge.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::FlightFrameTooLarge.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::ServerTransportFailure.as_str(), "server_transport_failure");
    assert_eq!(ErrorCode::ServerTransportFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::ServerTransportFailure.status(), ErrorStatus::Permanent);
}

#[test]
fn native_source_storage_errors_preserve_adapter_status_mapping() {
    for (native_code, adapter_code, adapter_kind, adapter_status) in [
        (
            QueryErrorCode::SourceAccessDenied,
            ErrorCode::SourceAccessDenied,
            ErrorKind::PermissionDenied,
            ErrorStatus::Permanent,
        ),
        (
            QueryErrorCode::SourceUnauthenticated,
            ErrorCode::SourceUnauthenticated,
            ErrorKind::Unauthenticated,
            ErrorStatus::Permanent,
        ),
        (
            QueryErrorCode::SourceObjectNotFound,
            ErrorCode::SourceObjectNotFound,
            ErrorKind::NotFound,
            ErrorStatus::Permanent,
        ),
        (
            QueryErrorCode::SourceTemporaryFailure,
            ErrorCode::SourceTemporaryFailure,
            ErrorKind::DependencyFailure,
            ErrorStatus::Temporary,
        ),
    ] {
        let native_error = QueryError::new(
            native_code,
            "query_engine.source_metadata.object_store_footer",
            "source object storage failure",
        )
        .with_fact("object_store_error_kind", native_code.as_str());

        let adapter_error = Error::from_native(
            "algeon_datafusion.native.execute",
            "Algeon native query execution failed",
            native_error,
        );

        assert_eq!(adapter_error.code(), adapter_code);
        assert_eq!(adapter_error.kind(), adapter_kind);
        assert_eq!(adapter_error.status(), adapter_status);
        assert_eq!(adapter_error.fact_value("native_code"), Some(native_code.as_str()));
    }
}

#[test]
fn native_service_overloaded_error_preserves_adapter_classification_and_facts() {
    let native_error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "query_engine.query_service.admit",
        "native query service is overloaded",
    )
    .with_fact("retry_action", "backoff");

    let adapter_error =
        Error::from_native("algeon_datafusion.native.execute", "Algeon native query execution failed", native_error);

    assert_eq!(adapter_error.code(), ErrorCode::ServiceOverloaded);
    assert_eq!(adapter_error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(adapter_error.status(), ErrorStatus::Temporary);
    assert_eq!(adapter_error.fact_value("native_code"), Some("service_overloaded"));
    assert_eq!(adapter_error.fact_value("retry_action"), Some("backoff"));
}

#[test]
fn native_query_error_json_includes_triage_summary() {
    let native_error = QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        "query_engine.test.memory",
        "memory reservation failed",
    )
    .with_query_id("query-err")
    .with_fact("node_id", "n3")
    .with_fact("compiled_task_id", "task-3");

    let summary = native_error.triage_summary();
    assert_eq!(summary.verdict, TriageVerdict::Failed);
    assert_eq!(summary.signals[0].code, TriageCode::MemoryReservationDenied);

    let json = native_error.to_json();
    assert_eq!(json["triage"]["verdict"], "failed");
    assert_eq!(json["triage"]["signals"][0]["code"], "memory_reservation_denied");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["query_id"], "query-err");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["plan_node_id"], "n3");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["compiled_task_id"], "task-3");
}

#[test]
fn native_query_error_json_uses_fact_only_correlation_ids() {
    let native_error = QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        "query_engine.test.memory",
        "memory reservation failed",
    )
    .with_fact("query_id", "fact-query")
    .with_fact("trace_id", "fact-trace")
    .with_fact("node_id", "n3");

    let json = native_error.to_json();
    assert_eq!(json["query_id"], "fact-query");
    assert_eq!(json["trace_id"], "fact-trace");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["query_id"], "fact-query");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["plan_node_id"], "n3");
}

#[test]
fn native_query_error_triage_codes_preserve_invalid_input_and_unsupported_capability() {
    assert_eq!(TriageCode::from(QueryErrorCode::CudfInvalidInput), TriageCode::InvalidInput);
    assert_eq!(TriageCode::from(QueryErrorCode::CudfUnsupported), TriageCode::UnsupportedOperator);
}

#[test]
fn native_error_message_is_preserved_as_adapter_fact() {
    let native_error = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.graph_algorithm.betweenness_seed_policy",
        "exact betweenness requires explicit seeds or k above the configured vertex threshold",
    );

    let adapter_error = Error::from_native(
        "algeon_datafusion.cugraph_sql.execute_gpu",
        "native cuGraph SQL GPU-resident query execution failed",
        native_error,
    );

    assert_eq!(
        adapter_error.fact_value("native_message"),
        Some("exact betweenness requires explicit seeds or k above the configured vertex threshold")
    );
}

#[test]
fn native_correlation_facts_follow_stamped_query_error() {
    let native_error = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.query_service.context",
        "query service handle was released before context acquisition",
    )
    .with_fact("query_id", "service-local-query")
    .with_query_id("request-query");

    let adapter_error = Error::from_native(
        "algeon_datafusion.native_exec.context",
        "failed to acquire query service algeon native execution context",
        native_error,
    );

    assert_eq!(adapter_error.fact_value("query_id"), Some("request-query"));
}

#[test]
fn native_fact_collisions_preserve_adapter_facts_and_record_skipped_keys() {
    let native_error = QueryError::new(
        QueryErrorCode::UnsupportedPipelineContract,
        "query_engine.plan.lower",
        "native plan shape is not supported",
    )
    .with_fact("native_code", "lower_layer_override")
    .with_fact("native_operation", "lower_layer_operation")
    .with_fact("native_fact_collision_keys", "lower_layer_collision_value")
    .with_fact("fallback_reason", "unsupported_shape");

    let adapter_error =
        Error::from_native("algeon_datafusion.native.lower", "algeon native lowering failed", native_error);

    assert_eq!(adapter_error.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(adapter_error.fact_value("native_code"), Some("unsupported_pipeline_contract"));
    assert_eq!(adapter_error.fact_value("native_operation"), Some("query_engine.plan.lower"));
    assert_eq!(adapter_error.fact_value("fallback_reason"), Some("unsupported_shape"));
    assert_eq!(
        adapter_error.fact_value("native_fact_collision_keys"),
        Some("native_code,native_fact_collision_keys,native_operation")
    );
}

#[test]
fn error_display_is_human_only_while_typed_fields_remain_available() {
    let error = Error::new(
        ErrorCode::FinalPlanRequirementUnsatisfied,
        "algeon_datafusion.planner.final_plan_requirement",
        "final physical plan did not satisfy the configured requirement",
    )
    .with_fact("observed_final_plan_disposition", "datafusion")
    .with_fact("requirement", "no_datafusion_cpu")
    .with_source(std::io::Error::other("sensitive dependency source"));

    assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(error.kind(), ErrorKind::Planning);
    assert_eq!(error.status(), ErrorStatus::Permanent);
    assert_eq!(error.operation(), "algeon_datafusion.planner.final_plan_requirement");
    assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("datafusion"));

    assert_eq!(error.to_string(), "final physical plan did not satisfy the configured requirement");
    for hidden in [
        "final_plan_requirement_unsatisfied",
        "algeon_datafusion.planner.final_plan_requirement",
        "observed_final_plan_disposition",
        "no_datafusion_cpu",
        "sensitive dependency source",
    ] {
        assert!(!error.to_string().contains(hidden), "Display leaked structured or source data: {hidden}");
    }
    let debug = format!("{error:?}");
    assert!(debug.contains("FinalPlanRequirementUnsatisfied"), "{debug}");
    assert!(debug.contains("algeon_datafusion.planner.final_plan_requirement"), "{debug}");
    assert!(!debug.contains("observed_final_plan_disposition"), "{debug}");
    assert!(!debug.contains("no_datafusion_cpu"), "{debug}");
}

fn algeon_error(operation: &'static str, message: &'static str) -> Error {
    Error::new(ErrorCode::CudfDependencyFailure, operation, message)
        .with_fact("device_ordinal", 2)
        .with_source(std::io::Error::other("CUDA dependency detail"))
}

#[test]
fn algeon_error_uses_exact_datafusion_external_prose_and_preserves_structured_fields() {
    let error: DataFusionError = algeon_error("algeon_datafusion.native.execute", "native GPU execution failed").into();

    assert!(matches!(error, DataFusionError::External(_)));
    assert_eq!(error.to_string(), "External error: native GPU execution failed");
    for hidden in ["cudf_dependency_failure", "device_ordinal", "CUDA dependency detail"] {
        assert!(!error.to_string().contains(hidden), "DataFusion Display leaked structured or source data: {hidden}");
    }

    let algeon = error.find_algeon_error().expect("typed Algeon error must remain discoverable");
    assert_eq!(algeon.code(), ErrorCode::CudfDependencyFailure);
    assert_eq!(algeon.kind(), ErrorKind::DependencyFailure);
    assert_eq!(algeon.status(), ErrorStatus::Permanent);
    assert_eq!(algeon.operation(), "algeon_datafusion.native.execute");
    assert_eq!(algeon.fact_value("device_ordinal"), Some("2"));
    assert!(algeon.source_error().is_some());
}

#[test]
fn plain_pre_algeon_datafusion_failure_keeps_its_variant_and_prose() {
    let error = DataFusionError::Plan("unknown function plain_udf".to_string());

    assert!(matches!(error, DataFusionError::Plan(_)));
    assert_eq!(error.to_string(), "Error during planning: unknown function plain_udf");
    assert!(matches!(error.find_root(), DataFusionError::Plan(_)));
    assert!(error.find_algeon_error().is_none());
}

fn algeon_control_plane_failure() -> Result<()> {
    Err(algeon_error("algeon_datafusion.control_plane.open", "failed to open Algeon backend"))
}

fn datafusion_owned_surface() -> DataFusionResult<()> {
    algeon_control_plane_failure()?;
    Ok(())
}

#[test]
fn question_mark_converts_algeon_result_at_datafusion_boundary() {
    let error = datafusion_owned_surface().expect_err("Algeon failure must propagate");

    let DataFusionError::External(source) = &error else {
        panic!("Algeon Result must convert exactly at the DataFusion-owned boundary: {error:?}");
    };
    assert!(source.downcast_ref::<Error>().is_some(), "External must preserve the typed Algeon source");
    assert_eq!(error.to_string(), "External error: failed to open Algeon backend");
    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.control_plane.open"));
}

fn external_algeon_error(operation: &'static str) -> DataFusionError {
    algeon_error(operation, "wrapped native GPU execution failed").into()
}

#[test]
fn datafusion_wrappers_keep_find_root_behavior_and_algeon_discovery() {
    let wrappers = [
        DataFusionError::Context(
            "query context".to_string(),
            Box::new(external_algeon_error("algeon_datafusion.test.context")),
        ),
        DataFusionError::Diagnostic(
            Box::new(Diagnostic::new_error("query diagnostic", None)),
            Box::new(external_algeon_error("algeon_datafusion.test.diagnostic")),
        ),
        DataFusionError::Shared(Arc::new(external_algeon_error("algeon_datafusion.test.shared"))),
        DataFusionError::ArrowError(
            Box::new(ArrowError::from_external_error(Box::new(external_algeon_error("algeon_datafusion.test.arrow")))),
            None,
        ),
    ];

    for (error, expected_operation) in wrappers.into_iter().zip([
        "algeon_datafusion.test.context",
        "algeon_datafusion.test.diagnostic",
        "algeon_datafusion.test.shared",
        "algeon_datafusion.test.arrow",
    ]) {
        assert!(
            matches!(error.find_root(), DataFusionError::External(_)),
            "DataFusion find_root must still stop at its deepest DataFusion error: {error:?}"
        );
        assert_eq!(error.find_algeon_error().map(Error::operation), Some(expected_operation));
    }
}

#[test]
fn collection_preserves_datafusion_root_and_uses_documented_algeon_order() {
    let error = DataFusionError::Collection(vec![
        DataFusionError::Plan("plain first branch".to_string()),
        DataFusionError::Collection(vec![
            DataFusionError::Execution("plain nested branch".to_string()),
            external_algeon_error("algeon_datafusion.test.first"),
        ]),
        external_algeon_error("algeon_datafusion.test.second"),
    ]);

    assert!(matches!(error.find_root(), DataFusionError::Plan(message) if message == "plain first branch"));
    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.first"));
}

#[derive(Debug, PartialEq, Eq)]
enum BackendDisposition {
    Algeon(ErrorCode),
    DataFusionPlanning,
    DataFusionResourceExhausted,
    DataFusionOther,
}

fn existing_datafusion_handler(error: &DataFusionError) -> BackendDisposition {
    match error.find_root() {
        DataFusionError::Plan(_) => BackendDisposition::DataFusionPlanning,
        DataFusionError::ResourcesExhausted(_) => BackendDisposition::DataFusionResourceExhausted,
        _ => BackendDisposition::DataFusionOther,
    }
}

fn algeon_aware_backend_handler(error: &DataFusionError) -> BackendDisposition {
    if let Some(algeon) = error.find_algeon_error() {
        return BackendDisposition::Algeon(algeon.code());
    }
    existing_datafusion_handler(error)
}

#[test]
fn backend_handler_adds_algeon_branch_then_reuses_existing_datafusion_fallback() {
    let algeon = DataFusionError::Context(
        "query".to_string(),
        Box::new(external_algeon_error("algeon_datafusion.test.backend")),
    );
    assert_eq!(algeon_aware_backend_handler(&algeon), BackendDisposition::Algeon(ErrorCode::CudfDependencyFailure));

    for plain in [
        DataFusionError::Plan("plain planning failure".to_string()),
        DataFusionError::ArrowError(
            Box::new(ArrowError::from_external_error(Box::new(DataFusionError::ResourcesExhausted(
                "plain memory pressure".to_string(),
            )))),
            None,
        ),
    ] {
        assert_eq!(algeon_aware_backend_handler(&plain), existing_datafusion_handler(&plain));
    }
}

#[test]
fn datafusion_conversion_preserves_typed_adapter_error() {
    let error = Error::new(
        ErrorCode::InvalidExecutionPlan,
        "algeon_datafusion.native.try_new",
        "cannot build native cuDF execution plan",
    )
    .with_fact("plan_node", "EmptyExec");

    let datafusion_error: DataFusionError = error.into();
    let DataFusionError::External(source) = datafusion_error else {
        panic!("adapter errors must cross the DataFusion boundary as typed external errors");
    };
    let adapter_error = source
        .downcast_ref::<Error>()
        .expect("DataFusion external error should preserve algeon_datafusion::error::Error");

    assert_eq!(adapter_error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(adapter_error.kind(), ErrorKind::Planning);
    assert_eq!(adapter_error.fact_value("plan_node"), Some("EmptyExec"));
}

#[test]
fn native_cancelled_error_preserves_adapter_classification_and_facts() {
    let native_error = ExecutionCancellation::new();
    native_error.cancel();
    let adapter_error = Error::from_native(
        "algeon_datafusion.native_exec.execute",
        "Algeon native query execution failed",
        native_error.check("query_engine.exec.execute_query_plan").unwrap_err(),
    );

    assert_eq!(adapter_error.code(), ErrorCode::Cancelled);
    assert_eq!(adapter_error.kind(), ErrorKind::Cancelled);
    assert_eq!(adapter_error.status(), ErrorStatus::Permanent);
    assert_eq!(adapter_error.fact_value("native_code"), Some("cancelled"));
    assert_eq!(adapter_error.fact_value("cancellation_requested"), Some("true"));
}

#[test]
fn native_retry_error_preserves_resume_boundary_facts() {
    let native_error = QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        "query_engine.final_sort.topk_concat",
        "native final sort exhausted bounded memory",
    )
    .with_fact("retry_action", "reduce_final_sort_fan_in")
    .with_fact("resume_boundary", "final_sort")
    .with_fact("committed_chunks", 4)
    .with_fact("committed_partials", 4)
    .with_fact("merge_level", 2)
    .with_fact("merge_fan_in", 8);

    let adapter_error = Error::from_native(
        "algeon_datafusion.native_exec.execute",
        "Algeon native query execution failed",
        native_error,
    );

    assert_eq!(adapter_error.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(adapter_error.fact_value("native_operation"), Some("query_engine.final_sort.topk_concat"));
    assert_eq!(adapter_error.fact_value("retry_action"), Some("reduce_final_sort_fan_in"));
    assert_eq!(adapter_error.fact_value("resume_boundary"), Some("final_sort"));
    assert_eq!(adapter_error.fact_value("committed_chunks"), Some("4"));
    assert_eq!(adapter_error.fact_value("committed_partials"), Some("4"));
    assert_eq!(adapter_error.fact_value("merge_level"), Some("2"));
    assert_eq!(adapter_error.fact_value("merge_fan_in"), Some("8"));
}
