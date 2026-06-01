use datafusion::common::DataFusionError;
use datafusion_nexus::{Error, ErrorCode, ErrorKind, ErrorStatus};

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

    assert_eq!(ErrorCode::ObjectNotFound.as_str(), "object_not_found");
    assert_eq!(ErrorCode::ObjectNotFound.kind(), ErrorKind::NotFound);
    assert_eq!(ErrorCode::ObjectNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::FieldNotFound.as_str(), "field_not_found");
    assert_eq!(ErrorCode::FieldNotFound.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::FieldNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::UnsupportedExpression.as_str(), "unsupported_expression");
    assert_eq!(ErrorCode::UnsupportedExpression.kind(), ErrorKind::Unsupported);
    assert_eq!(ErrorCode::UnsupportedExpression.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::CudfResourceExhausted.as_str(), "cudf_resource_exhausted");
    assert_eq!(ErrorCode::CudfResourceExhausted.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::CudfResourceExhausted.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::CudfPersistentFailure.as_str(), "cudf_persistent_failure");
    assert_eq!(ErrorCode::CudfPersistentFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::CudfPersistentFailure.status(), ErrorStatus::Persistent);

    assert_eq!(ErrorCode::MemoryReservationFailed.as_str(), "memory_reservation_failed");
    assert_eq!(ErrorCode::MemoryReservationFailed.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::MemoryReservationFailed.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::ServiceOverloaded.as_str(), "service_overloaded");
    assert_eq!(ErrorCode::ServiceOverloaded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::ServiceOverloaded.status(), ErrorStatus::Temporary);
}

#[test]
fn native_service_overloaded_error_preserves_adapter_classification_and_facts() {
    let native_error = nexus_query_engine::QueryError::new(
        nexus_query_engine::QueryErrorCode::ServiceOverloaded,
        "nexus_query_engine.query_service.admit",
        "native query service is overloaded",
    )
    .with_fact("retry_action", "backoff");

    let adapter_error =
        Error::from_native("datafusion_nexus.native.execute", "native cudf query execution failed", native_error);

    assert_eq!(adapter_error.code(), ErrorCode::ServiceOverloaded);
    assert_eq!(adapter_error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(adapter_error.status(), ErrorStatus::Temporary);
    assert_eq!(adapter_error.fact_value("native_code"), Some("service_overloaded"));
    assert_eq!(adapter_error.fact_value("retry_action"), Some("backoff"));
}

#[test]
fn error_keeps_machine_code_operation_and_context_facts() {
    let error = Error::new(
        ErrorCode::NativeCandidateRejected,
        "datafusion_nexus.native.optimize_plan",
        "cuDF native candidate rejected by adapter policy",
    )
    .with_fact("fallback_reason", "unsupported_expression")
    .with_fact("candidate_shape", "SortExec(fetch)(AggregateExec(LocalParquetScan))");

    assert_eq!(error.code(), ErrorCode::NativeCandidateRejected);
    assert_eq!(error.kind(), ErrorKind::Planning);
    assert_eq!(error.status(), ErrorStatus::Permanent);
    assert_eq!(error.operation(), "datafusion_nexus.native.optimize_plan");
    assert_eq!(error.fact_value("fallback_reason"), Some("unsupported_expression"));

    let display = error.to_string();
    assert!(display.contains("code=native_candidate_rejected"));
    assert!(display.contains("operation=datafusion_nexus.native.optimize_plan"));
    assert!(display.contains("fallback_reason=unsupported_expression"));
}

#[test]
fn datafusion_conversion_preserves_typed_adapter_error() {
    let error = Error::new(
        ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.native.try_new",
        "cannot build native cuDF execution plan",
    )
    .with_fact("plan_node", "EmptyExec");

    let datafusion_error: DataFusionError = error.into();
    let DataFusionError::External(source) = datafusion_error else {
        panic!("adapter errors must cross the DataFusion boundary as typed external errors");
    };
    let adapter_error =
        source.downcast_ref::<Error>().expect("DataFusion external error should preserve datafusion_nexus::Error");

    assert_eq!(adapter_error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(adapter_error.kind(), ErrorKind::Planning);
    assert_eq!(adapter_error.fact_value("plan_node"), Some("EmptyExec"));
}

#[test]
fn native_cancelled_error_preserves_adapter_classification_and_facts() {
    let native_error = nexus_query_engine::ExecutionCancellation::new();
    native_error.cancel();
    let adapter_error = Error::from_native(
        "datafusion_nexus.native_exec.execute",
        "native cudf query execution failed",
        native_error.check("nexus_query_engine.exec.execute_query_plan").unwrap_err(),
    );

    assert_eq!(adapter_error.code(), ErrorCode::Cancelled);
    assert_eq!(adapter_error.kind(), ErrorKind::Cancelled);
    assert_eq!(adapter_error.status(), ErrorStatus::Permanent);
    assert_eq!(adapter_error.fact_value("native_code"), Some("cancelled"));
    assert_eq!(adapter_error.fact_value("cancellation_requested"), Some("true"));
}

#[test]
fn native_memory_reservation_error_preserves_adapter_classification_and_facts() {
    let native_error =
        nexus_query_engine::MemoryReservationRequest::new(nexus_query_engine::ReservationKind::FinalSort, 4096, 2048)
            .with_memory_boundary("final_sort")
            .with_node_id("n5")
            .with_stage("final_sort")
            .with_chunk_index(7)
            .with_row_groups(vec![3, 5])
            .with_attempt(1, 3)
            .denied_error(1024);

    let adapter_error =
        Error::from_native("datafusion_nexus.native.execute", "native cudf query execution failed", native_error);

    assert_eq!(adapter_error.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(adapter_error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(adapter_error.status(), ErrorStatus::Temporary);
    assert_eq!(adapter_error.fact_value("native_code"), Some("memory_reservation_failed"));
    assert_eq!(adapter_error.fact_value("reservation_kind"), Some("final_sort"));
    assert_eq!(adapter_error.fact_value("memory_boundary"), Some("final_sort"));
    assert_eq!(adapter_error.fact_value("node_id"), Some("n5"));
    assert_eq!(adapter_error.fact_value("stage"), Some("final_sort"));
    assert_eq!(adapter_error.fact_value("chunk_index"), Some("7"));
    assert_eq!(adapter_error.fact_value("row_groups"), Some("3,5"));
    assert_eq!(adapter_error.fact_value("attempt"), Some("1"));
    assert_eq!(adapter_error.fact_value("max_attempts"), Some("3"));
}

#[test]
fn native_retry_error_preserves_resume_boundary_facts() {
    let native_error = nexus_query_engine::RetryFacts::new()
        .with_retry_action("reduce_final_sort_fan_in")
        .with_resume_boundary("final_sort")
        .with_committed_chunks(4)
        .with_committed_partials(4)
        .with_merge_level(2)
        .with_merge_fan_in(8)
        .attach_to_error(nexus_query_engine::QueryError::new(
            nexus_query_engine::QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.final_sort.topk_concat",
            "native final sort exhausted bounded memory",
        ));

    let adapter_error =
        Error::from_native("datafusion_nexus.native_exec.execute", "native cudf query execution failed", native_error);

    assert_eq!(adapter_error.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(adapter_error.fact_value("native_operation"), Some("nexus_query_engine.final_sort.topk_concat"));
    assert_eq!(adapter_error.fact_value("retry_action"), Some("reduce_final_sort_fan_in"));
    assert_eq!(adapter_error.fact_value("resume_boundary"), Some("final_sort"));
    assert_eq!(adapter_error.fact_value("committed_chunks"), Some("4"));
    assert_eq!(adapter_error.fact_value("committed_partials"), Some("4"));
    assert_eq!(adapter_error.fact_value("merge_level"), Some("2"));
    assert_eq!(adapter_error.fact_value("merge_fan_in"), Some("8"));
}
