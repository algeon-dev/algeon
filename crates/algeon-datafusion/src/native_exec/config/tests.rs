use std::sync::Arc;

use query_planning::error::QueryError;

use super::*;
use crate::{
    attempt::attempt_ticket_for_test, backend::InstalledTestGpuBackend, error::ErrorCode,
    native_exec::admission::task_context_with_query_admission_ticket,
};

#[tokio::test]
async fn async_execution_context_requires_gpu_backend_before_resolving_cuda_state() {
    let error = AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 1, 1).expect("valid source chunking"))
        .with_query_id("query-config")
        .with_trace_id("trace-config")
        .execution_context_async(Arc::new(TaskContext::default()))
        .await
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("reason"), Some("gpu_backend_required"));
    assert_eq!(error.fact_value("query_id"), Some("query-config"));
    assert_eq!(error.fact_value("trace_id"), Some("trace-config"));
}

#[test]
fn native_execution_config_policy_getters_match_bounded_policy() {
    let config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 8, 5, 1).expect("valid source chunking"));
    assert_eq!(config.source_chunking().max_source_chunk_bytes.get(), 512);
    assert_eq!(config.source_chunking().max_row_groups_per_chunk.get(), 8);
    assert_eq!(config.source_chunking().max_retry_attempts.get(), 5);
    assert_eq!(config.max_source_reads_in_flight(), 2);
}

#[test]
fn query_admission_ticket_selection_prefers_ambient_query_admission_ticket() {
    let backend = InstalledTestGpuBackend::new();
    let source = backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority");
    let (_ambient_attempt, ambient_ticket) = attempt_ticket_for_test(source.clone(), None);
    let (_child_attempt, child_ticket) = attempt_ticket_for_test(source.clone(), None);
    let child_config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
            .with_query_admission_source(source)
            .with_existing_query_admission_ticket(Arc::clone(&child_ticket));
    let child_ticket_id = child_config.query_admission_ticket_id();
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&ambient_ticket));

    assert_eq!(child_ticket_id, Some(child_ticket.id()));
    assert_ne!(Some(ambient_ticket.id()), child_ticket_id);
    assert_eq!(
        child_config.query_admission_ticket_id_for_task_context(&task_context),
        Some(ambient_ticket.id()),
        "embedded child config ticket must not win when an ambient root ticket exists"
    );
}

#[test]
fn config_validation_errors_carry_configured_correlation_ids() {
    let error = AlgeonNativeExecutionConfig::default()
        .with_query_id("query-config-error")
        .with_trace_id("trace-config-error")
        .with_source_chunk_planning_options(
            SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(0),
        )
        .validate_query_policies()
        .unwrap_err();

    assert_eq!(error.fact_value("query_id"), Some("query-config-error"));
    assert_eq!(error.fact_value("trace_id"), Some("trace-config-error"));
    let native = error
        .source_error()
        .and_then(|source| source.downcast_ref::<QueryError>())
        .expect("adapter error should retain native source error");
    assert_eq!(native.query_id().map(QueryId::as_str), Some("query-config-error"));
    assert_eq!(native.trace_id().map(TraceId::as_str), Some("trace-config-error"));
}
