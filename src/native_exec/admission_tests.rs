use std::sync::Arc;

use datafusion::execution::context::TaskContext;
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof,
    error::QueryErrorCode,
    plan::QueryPlanBuilder,
    runtime::ExecutionCancellation,
    source::{GpuResidentSource, SourceNode},
};

use super::*;
use crate::{
    backend::{InstalledTestGpuBackend, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
    native_exec::NexusNativeExecutionConfig,
};

fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
    backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority")
}

fn explicit_sink_capability_proof() -> QueryPlanCapabilityProof {
    use arrow::datatypes::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    let plan = builder.finish_with_host_sink(source);
    QueryPlanCapabilityProof::for_query_plan(&plan).expect("an explicit host sink must expose a capability requirement")
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_forwards_opaque_evidence_and_releases_real_ledger_grant_for_successive_admission() {
    let backend = InstalledTestGpuBackend::new();
    let ticket = QueryAdmissionTicket::new(admission_source(&backend), None);
    assert!(ticket.attempt_capability_proof().is_none());

    let context = ticket
        .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
        .await
        .expect("installed backend must acquire a real admission context");
    let facts = context.admission_facts();
    assert!(facts.admission_sequence > 0);
    assert!(facts.query_grant_bytes > 0);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);

    drop(context);
    let next_context = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ticket.acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new()),
    )
    .await
    .expect("dropping the context must eventually return the real ledger grant")
    .expect("released capacity must admit the next context");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);
    drop(next_context);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_propagates_cancellation_to_admission() {
    let backend = InstalledTestGpuBackend::new();
    let ticket = QueryAdmissionTicket::new(admission_source(&backend), None);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), cancellation)
        .await
        .expect_err("cancelled admission must not acquire a real handle");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn planner_capability_evidence_rejects_at_the_native_ticket_boundary_before_admission() {
    let unavailable = explicit_sink_capability_proof();
    let backend = InstalledTestGpuBackend::with_config(
        NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1),
        NexusGpuDeviceProfile::new(0)
            .with_memory_limit(nexus_query_engine::runtime::GpuDeviceMemoryLimit::Explicit {
                limit_bytes: 256 * 1024 * 1024,
            })
            .with_device_capabilities(nexus_query_engine::runtime::DeviceCapabilities::only(Vec::new())),
        NexusGpuQueuePolicy::default(),
    );
    let config = crate::native_exec::NexusNativeExecutionConfig::bounded(256, 512, 1, 1, 1)
        .with_query_admission_source(admission_source(&backend));
    let ticket = config
        .new_plan_query_admission_ticket_with_graph_model_ask_and_fragment_requirements(None, vec![unavailable.clone()])
        .expect("fragment requirements aggregate into a ticket")
        .expect("configured admission source creates a ticket");

    assert_eq!(ticket.attempt_capability_proof(), Some(&unavailable));
    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
        .await
        .expect_err("an aggregated unsupported capability must reject before admission");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
    assert_eq!(
        error.fact_value("required_capabilities"),
        Some("materialization:query_engine_explicit_sink_materialization_reservation")
    );
    assert_eq!(error.fact_value("configured_device_count"), Some("1"));
    assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
    assert_eq!(
        backend.backend().admission_snapshot().unwrap().active_queries,
        0,
        "rejection must precede grant mutation"
    );
    assert_eq!(
        backend.backend().admission_snapshot().unwrap().queued_queries,
        0,
        "rejection must not become a capacity waiter"
    );
}

#[test]
fn standalone_ticket_drop_seals_local_observation() {
    let backend = InstalledTestGpuBackend::new();
    let owner = RuntimeObservationOwner::new("ticket-drop", None, RuntimeObservationConfig::bounded(16, 4096));
    let ticket = QueryAdmissionTicket::new_with_observation(admission_source(&backend), None, owner.clone());
    drop(ticket);
    assert_eq!(owner.sealed_snapshot().unwrap().footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
}
