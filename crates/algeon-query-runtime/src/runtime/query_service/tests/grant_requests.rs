use query_planning::error::QueryErrorCode;

use super::*;

#[test]
fn attempt_requirement_admits_with_the_ledgers_configured_default_grant() {
    let service =
        AdmissionTestController::new_with_ledger(QueueBounds::default(), test_ledger(4096, 1, Default::default()))
            .unwrap();

    let handle = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_attempt_requirement(requirement()))
        .unwrap()
        .expect("requirement should admit");
    assert_eq!(handle.device_budget_bytes(), 512);
}

#[test]
fn admission_without_observation_mints_one_identity_per_committed_grant() {
    let service =
        AdmissionTestController::new_with_ledger(QueueBounds::default(), test_ledger(4096, 2, Default::default()))
            .unwrap();

    let first = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_attempt_requirement(requirement()))
        .unwrap()
        .expect("first request admits");
    let second = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_attempt_requirement(requirement()))
        .unwrap()
        .expect("second request admits");

    assert_ne!(first.grant_identity().attempt_id, second.grant_identity().attempt_id);
    assert_ne!(first.grant_identity().generation, second.grant_identity().generation);
}

#[test]
fn attempt_binding_identity_survives_request_placement_and_handle_commit() {
    let service =
        AdmissionTestController::new_with_ledger(QueueBounds::default(), test_ledger(4096, 1, Default::default()))
            .unwrap();
    let lifecycle = BackendLifecycle::new();
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest::default(),
        lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::new(AttemptBackendCounters::default()),
        None,
    );
    let attempt_id = attempt.owner.attempt_id();
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().unwrap();
    attempt_handle.record_planning_finished().unwrap();
    attempt_handle.set_disposition(AttemptDisposition::Native).unwrap();

    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("attempt-bound request admits");

    assert_eq!(handle.grant_identity().attempt_id, attempt_id);
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
    drop(handle);
    completion
        .wait_timeout(std::time::Duration::from_secs(1))
        .expect("attempt report publishes after the fixture releases its admission");
}

#[test]
fn eager_device_entries_share_capability_requirement_for_fit_grant_and_execution() {
    let service = AdmissionTestController::new(QueueBounds::default()).expect("synthetic admission service");
    let evidence = explicit_sink_capability_evidence();
    let admitted = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence.clone()),
        )
        .expect("submission succeeds")
        .expect("the exact entry fits the synthetic ledger");

    assert_eq!(admitted.admitted_device_requirement(admitted.device_ordinal()), Some(&evidence),);
    assert_eq!(admitted.admitted_capability_evidence().as_deref(), Some(&evidence));
    drop(admitted);
}

#[test]
fn explicit_cap_above_device_capacity_is_rejected_before_enqueue() {
    let service =
        AdmissionTestController::new_with_ledger(QueueBounds::default(), test_ledger(1024, 1, Default::default()))
            .unwrap();

    let error = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(2048))
        .expect_err("cap exceeding every eligible device must reject");
    assert_eq!(error.code(), QueryErrorCode::QueryBudgetUnavailable);
    assert_eq!(error.fact_value("reason"), Some("query_cap_exceeds_resident_capacity"));
    assert_eq!(error.fact_value("requested_query_cap_bytes"), Some("2048"));
}

#[test]
fn plan_capability_evidence_missing_from_exact_device_profile_is_rejected_before_queueing() {
    use query_planning::admission::ExecutionCapability;

    use crate::runtime::{DeviceCapabilities, DeviceCapability};

    let service = AdmissionTestController::new_with_device_capabilities(
        QueueBounds::default(),
        DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()),
    )
    .unwrap();
    let mut evidence = explicit_sink_capability_evidence();
    evidence.required_capabilities = vec![ExecutionCapability::new("unavailable")];
    let error = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence),
        )
        .expect_err("an unavailable device capability must reject locally");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
    assert_eq!(error.fact_value("required_capabilities"), Some("unavailable"));
    assert_eq!(error.fact_value("configured_device_count"), Some("1"));
    assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
    assert_eq!(service.queued_len(), 0, "capability rejection must never enter the admission queue");
}

#[test]
fn fragment_requiring_a_capability_the_granted_device_lacks_cannot_join_the_grant() {
    use query_planning::admission::CUGRAPH_GRAPH_EXECUTION_CAPABILITY;

    use crate::runtime::{DeviceCapabilities, DeviceCapability};

    let service = AdmissionTestController::new_with_device_capabilities(
        QueueBounds::default(),
        DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()),
    )
    .unwrap();
    let claimed = explicit_sink_capability_evidence();
    let admitted = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(claimed.clone()),
        )
        .unwrap()
        .expect("a claim without capability requirements admits");
    let before = service.admission_snapshot().unwrap();
    let ledgers = service.device_ledger_reports().unwrap();

    let mut graph_fragment = AttemptCapabilityEvidence::for_query_plan(&explicit_sink_plan("graph-input"));
    graph_fragment.required_capabilities = vec![CUGRAPH_GRAPH_EXECUTION_CAPABILITY];
    let error = admitted.join_fragment(&graph_fragment).expect_err("the granted device has no cuGraph");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("reason"), Some("granted_device_capability_unavailable"));
    assert_eq!(error.fact_value("required_capabilities"), Some("cugraph_graph_execution"));
    assert_eq!(service.admission_snapshot().unwrap(), before, "a refused fragment does not touch the queue");
    assert_eq!(service.device_ledger_reports().unwrap(), ledgers, "a refused fragment does not touch the ledger");
    assert_eq!(
        admitted.admitted_capability_evidence().as_deref(),
        Some(&claimed),
        "a refused fragment adds no plan identity"
    );

    let plain_fragment = AttemptCapabilityEvidence::for_query_plan(&explicit_sink_plan("plain-input"));
    admitted.join_fragment(&plain_fragment).expect("a fragment without capability requirements joins");
    assert_eq!(service.admission_snapshot().unwrap(), before, "joining does not touch the queue");
    assert_eq!(service.device_ledger_reports().unwrap(), ledgers, "joining does not touch the ledger");
    assert!(admitted.admitted_capability_evidence().expect("the claim's evidence is admitted").covers(&plain_fragment));
}

#[test]
fn submitted_plan_evidence_survives_admission_while_internal_placement_requirement_does_not_become_execution_evidence()
{
    let service = AdmissionTestController::new(QueueBounds::default()).unwrap();
    let evidence = explicit_sink_capability_evidence();
    let submitted = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence.clone()),
        )
        .unwrap()
        .expect("submitted evidence admits on an all-modeled test profile");
    assert_eq!(submitted.admitted_capability_evidence().as_deref(), Some(&evidence));
    drop(submitted);

    let placement_only = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_attempt_requirement(requirement()))
        .unwrap()
        .expect("internal placement requirement admits");
    assert!(placement_only.admitted_capability_evidence().is_none());
}

fn explicit_sink_capability_evidence() -> AttemptCapabilityEvidence {
    let plan = explicit_sink_plan("input");
    AttemptCapabilityEvidence::for_query_plan(&plan)
}

pub(super) fn explicit_sink_plan(input_name: &str) -> QueryPlan {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use query_planning::{
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, SourceNode},
    };

    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source).unwrap()
}

fn restricted_capability_service_and_requirement() -> (AdmissionTestController, AttemptCapabilityEvidence) {
    use query_planning::admission::ExecutionCapability;

    use crate::runtime::{DeviceCapabilities, DeviceCapability};

    let capability = DeviceCapability::new("test_operator");
    let service = AdmissionTestController::new_with_device_capabilities(
        QueueBounds::default(),
        DeviceCapabilities::only([capability]),
    )
    .unwrap();
    let mut attempt_requirement = requirement();
    attempt_requirement.required_capabilities = vec![ExecutionCapability::new("test_operator")];
    (service, attempt_requirement)
}

fn assert_explicit_requirement_required(error: &QueryError) {
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("explicit_attempt_requirement_required"));
    assert_eq!(error.fact_value("restricted_device_count"), Some("1"));
}

#[test]
fn restricted_capability_profile_rejects_missing_blocking_try_and_queued_requirements() {
    let (service, compatible_requirement) = restricted_capability_service_and_requirement();

    assert_explicit_requirement_required(&service.admit_query().expect_err("blocking admission requires a evidence"));
    assert_explicit_requirement_required(&service.try_admit_query().expect_err("try admission requires a evidence"));
    assert_explicit_requirement_required(
        &service.enqueue_admission().expect_err("queued admission requires a evidence"),
    );
    assert_eq!(service.queued_len(), 0);

    let admitted = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::test_attempt().with_attempt_requirement(compatible_requirement),
        )
        .unwrap()
        .expect("an explicit compatible capability evidence admits");
    assert_eq!(admitted.device_ordinal(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restricted_capability_profile_rejects_missing_async_requirement() {
    let (service, _) = restricted_capability_service_and_requirement();

    let error = service.admit_async().await.expect_err("async admission requires a evidence");

    assert_explicit_requirement_required(&error);
    assert_eq!(service.queued_len(), 0);
}

#[test]
fn explicit_query_cap_is_preserved_across_admissions() {
    let service =
        AdmissionTestController::new_with_ledger(QueueBounds::default(), test_ledger(100, 4, Default::default()))
            .unwrap();

    let raised = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(70))
        .unwrap()
        .expect("explicit cap should admit");
    // An explicit cap is never clamped: the admission issues exactly what the
    // request asked for.
    assert_eq!(raised.device_budget_bytes(), 70);
    let second = service.try_admit_query().unwrap().expect("remaining capacity admits unconstrained query");
    assert!(second.device_budget_bytes() > 0);
}
