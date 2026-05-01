use std::sync::Arc;

use datafusion::execution::context::TaskContext;
use nexus_query_engine::{admission::QueryPlanCapabilityProof, error::QueryErrorCode, runtime::ExecutionCancellation};

use super::*;

fn admission_only_controller() -> TestAdmissionController {
    TestAdmissionController::new(TestAdmissionDeviceProfile::new(0, 1024, 1, 256), Some(512)).unwrap()
}

#[test]
fn test_controller_uses_its_fixture_only_hint_without_a_retired_admission_policy() {
    let controller = admission_only_controller();
    assert_eq!(controller.configured_floor_bytes(), 256);
    assert_eq!(controller.admitted_grant_bytes().unwrap(), 512);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_forwards_opaque_evidence_and_releases_after_context_failure() {
    let controller = admission_only_controller();
    let ticket = QueryAdmissionTicket::new(controller.source(), None);
    assert!(ticket.attempt_capability_proof().is_none());

    let result = ticket.acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new()).await;
    assert!(result.is_err(), "the CUDA-free controller cannot create contexts");
    assert_eq!(controller.admission_snapshot().active_queries, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_propagates_cancellation_to_admission() {
    let controller = admission_only_controller();
    let ticket = QueryAdmissionTicket::new(controller.source(), None);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), cancellation)
        .await
        .expect_err("cancelled admission must not acquire a test handle");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(controller.admission_snapshot().active_queries, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn planner_capability_evidence_rejects_at_the_native_ticket_boundary_before_admission() {
    let unavailable = QueryPlanCapabilityProof::for_cugraph_graph_execution();
    let controller = TestAdmissionController::new_with_supported_capabilities(
        TestAdmissionDeviceProfile::new(0, 1024, 1, 256),
        Some(512),
        Vec::new(),
    )
    .expect("restricted CPU-only admission controller");
    let config = crate::native_exec::NexusNativeExecutionConfig::bounded(256, 512, 1, 1, 1)
        .with_query_admission_source(controller.source());
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
    assert_eq!(error.fact_value("required_capabilities"), Some("graph_algorithm:cugraph_graph_execution_contract"));
    assert_eq!(error.fact_value("configured_device_count"), Some("1"));
    assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
    assert_eq!(controller.admission_snapshot().active_queries, 0, "rejection must precede grant mutation");
    assert_eq!(controller.admission_snapshot().queued_queries, 0, "rejection must not become a capacity waiter");
}

#[test]
fn standalone_ticket_drop_seals_local_observation() {
    let controller = admission_only_controller();
    let owner = RuntimeObservationOwner::new("ticket-drop", None, RuntimeObservationConfig::bounded(16, 4096));
    let ticket = QueryAdmissionTicket::new_with_observation(controller.source(), None, owner.clone());
    drop(ticket);
    assert_eq!(owner.sealed_snapshot().unwrap().footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
}
