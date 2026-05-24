#[test]
fn attempt_requirement_hint_selects_grant_for_one_admission() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        test_ledger(4096, 1, Default::default()),
    )
    .unwrap();

    let handle = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(1024, Some(1536))),
        )
        .unwrap()
        .expect("requirement should admit");
    assert_eq!(handle.device_budget_bytes(), 1536);
}

#[test]
fn admission_without_observation_mints_one_identity_per_committed_grant() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
        test_ledger(4096, 2, Default::default()),
    )
    .unwrap();

    let first = service
        .try_admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(requirement(100, Some(100))))
        .unwrap()
        .expect("first request admits");
    let second = service
        .try_admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(requirement(100, Some(100))))
        .unwrap()
        .expect("second request admits");

    assert_ne!(first.grant_identity().attempt_id, second.grant_identity().attempt_id);
    assert_ne!(first.grant_identity().generation, second.grant_identity().generation);
}

#[test]
fn attempt_binding_identity_survives_request_placement_and_handle_commit() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
        test_ledger(4096, 1, Default::default()),
    )
    .unwrap();
    let lifecycle = crate::runtime::BackendLifecycle::new();
    let attempt = crate::attempt::AttemptOwner::begin(
        crate::attempt::BeginAttemptRequest::default(),
        lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default()),
        None,
    );
    let attempt_id = attempt.owner.attempt_id();
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().unwrap();
    attempt_handle.record_planning_finished().unwrap();
    attempt_handle.set_disposition(crate::observability::AttemptDisposition::Native).unwrap();
    attempt_handle.record_admission_requested().unwrap();

    let handle = service
        .service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("attempt-bound request admits");

    assert_eq!(handle.grant_identity().unwrap().attempt_id, attempt_id);
    attempt_handle
        .record_admitted(crate::observability::AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .unwrap();
    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
    handle.finish();
    drop(handle);
    completion
        .wait_timeout(std::time::Duration::from_secs(1))
        .expect("attempt report publishes after admission-only cleanup");
}

#[test]
fn eager_device_entries_share_requirement_envelope_bytes_for_fit_grant_and_execution() {
    let service = AdmissionTestController::new(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
    )
    .expect("synthetic admission service");
    let proof = explicit_sink_capability_proof();
    let policy = service.service.inner.config.attempt_reservation_policy();
    let resolved_proof = proof.requirement_for_device(0).expect("non-sort proof resolves on a synthetic ordinal");
    let envelope = resolved_proof
        .resolve_reservation_envelope(policy)
        .expect("the selected proof resolves its immutable envelope");
    let expected_requirement = resolved_proof.as_requirement().clone().with_reservation_envelope(&envelope);
    let admitted = service
        .try_admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(proof))
        .expect("submission succeeds")
        .expect("the exact entry fits the synthetic ledger");

    let expected_floor = service
        .service
        .inner
        .fallback_device_ledger
        .as_ref()
        .expect("synthetic ledger")
        .required_floor_bytes_for(&expected_requirement)
        .expect("selected requirement has a finite fit floor");
    assert!(admitted.handle.device_budget_bytes() >= expected_floor);
    assert_eq!(
        admitted.handle.lifetime.admitted_device_requirement_for_test(admitted.handle.device_ordinal()),
        Some(&expected_requirement),
    );
    assert_eq!(
        admitted.handle.lifetime.admitted_attempt_proof_for_test().map(|proof| proof.reservation_envelope()),
        Some(&envelope),
    );
    let proof = admitted.handle.lifetime.admitted_attempt_proof_for_test().expect("submitted admission carries its proof");
    assert_eq!(proof.selected_device(), admitted.handle.device_ordinal());
    assert_eq!(proof.grant_identity(), admitted.handle.grant_identity().expect("grant identity"));
    drop(admitted);
}

#[test]
fn retry_floor_uses_the_candidate_requirement_projection() {
    let requirements = BTreeMap::from([
        (0, requirement(10, Some(10))),
        (1, requirement(200, Some(200))),
    ]);
    let ledger = test_ledger_with_explicit_profile(1, 100, 8, 1, Default::default());

    let selected_floor = ledger
        .required_floor_bytes_for(requirements.get(&0).expect("selected requirement"))
        .expect("selected floor resolves");
    let alternate_floor = super::handle::required_floor_bytes_for_candidate(&requirements, 1, &ledger)
        .expect("alternate floor resolves from the ordinal projection");

    assert!(ledger.can_ever_fit_required_floor_bytes(selected_floor).expect("selected floor fit check"));
    assert_eq!(alternate_floor, 200);
    assert!(!ledger.can_ever_fit_required_floor_bytes(alternate_floor).expect("alternate floor fit check"));

    let missing = super::handle::required_floor_bytes_for_candidate(&requirements, 2, &ledger)
        .expect_err("missing candidate requirement must fail closed");
    assert_eq!(missing.code(), QueryErrorCode::Internal);
    assert_eq!(missing.fact_value("reason"), Some("missing_device_specific_admission_resolution"));
}

#[test]
fn impossible_requirement_is_rejected_before_enqueue() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        test_ledger(1024, 1, Default::default()),
    )
    .unwrap();

    let error = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(2048, Some(2048))),
        )
        .expect_err("floor exceeding capacity must reject");
    assert_eq!(error.fact_value("reason"), Some("impossible_requirement"));
    assert_eq!(error.fact_value("minimum_compatible_required_floor_bytes"), Some("2048"));
    assert_eq!(error.fact_value("minimum_compatible_required_floor_device_ordinal"), Some("0"));
    assert_eq!(error.fact_value("minimum_compatible_configured_floor_bytes"), Some("2048"));
    assert_eq!(error.fact_value("minimum_compatible_required_phase_floor_bytes"), Some("0"));
    assert_eq!(error.fact_value("minimum_compatible_capability_floor_raise_bytes"), Some("0"));
}

#[test]
fn plan_capability_proof_missing_from_exact_device_profile_is_rejected_before_queueing() {
    use crate::runtime::{DeviceCapabilities, DeviceCapability};

    let service = AdmissionTestController::new_with_device_capabilities(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()),
    )
    .unwrap();
    let error = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new()
                .with_query_plan_capability_proof(explicit_sink_capability_proof()),
        )
        .expect_err("an unavailable device capability must reject locally");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
    assert_eq!(
        error.fact_value("required_capabilities"),
        Some("materialization:query_engine_explicit_sink_materialization_reservation")
    );
    assert_eq!(error.fact_value("configured_device_count"), Some("1"));
    assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
    assert_eq!(service.queued_len(), 0, "capability rejection must never enter the admission queue");
}

#[test]
fn submitted_plan_proof_survives_admission_while_test_placement_requirement_does_not_become_execution_evidence() {
    let service = AdmissionTestController::new(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
    )
    .unwrap();
    let proof = explicit_sink_capability_proof();
    let submitted = service
        .try_admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(proof.clone()))
        .unwrap()
        .expect("submitted proof admits on an all-modeled test profile");
    assert_eq!(
        submitted.handle.lifetime.admitted_attempt_proof_for_test().map(|proof| proof.submitted_capability_evidence()),
        Some(proof.as_requirement()),
    );
    drop(submitted);

    let placement_only = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(128, Some(128))),
        )
        .unwrap()
        .expect("internal placement requirement admits");
    assert!(placement_only.handle.lifetime.admitted_attempt_proof_for_test().is_none());
}

fn explicit_sink_capability_proof() -> crate::admission::QueryPlanCapabilityProof {
    let plan = explicit_sink_plan("input");
    crate::admission::QueryPlanCapabilityProof::for_query_plan(&plan)
        .expect("an explicit host sink must expose a capability requirement")
}

fn explicit_sink_plan(input_name: &str) -> crate::plan::QueryPlan {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use crate::{plan::QueryPlanBuilder, source::{GpuResidentSource, SourceNode}};

    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source)
}

fn restricted_capability_service_and_requirement() -> (AdmissionTestController, AttemptRequirement) {
    use crate::{
        capability::{ContractQuality, ReservationRequirement},
        runtime::{DeviceCapabilities, DeviceCapability, ReservationKind},
    };

    let capability = DeviceCapability::new(ReservationKind::OperatorOutput, "test_operator_contract");
    let service = AdmissionTestController::new_with_device_capabilities(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        DeviceCapabilities::only([capability]),
    )
    .unwrap();
    let mut attempt_requirement = requirement(128, Some(128));
    attempt_requirement.required_capabilities = vec![ReservationRequirement::modeled(
        ReservationKind::OperatorOutput,
        "test_operator",
        "test_operator_contract",
        ContractQuality::CapabilityProof,
    )];
    (service, attempt_requirement)
}

fn assert_explicit_requirement_required(error: &QueryError) {
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("explicit_attempt_requirement_required"));
    assert_eq!(error.fact_value("restricted_device_count"), Some("1"));
}

#[test]
fn restricted_capability_profile_rejects_missing_blocking_try_and_queued_requirements() {
    let (service, compatible_requirement) = restricted_capability_service_and_requirement();

    assert_explicit_requirement_required(&service.admit_query().expect_err("blocking admission requires a proof"));
    assert_explicit_requirement_required(&service.try_admit_query().expect_err("try admission requires a proof"));
    assert_explicit_requirement_required(&service.enqueue_admission().expect_err("queued admission requires a proof"));
    assert_eq!(service.queued_len(), 0);

    let admitted = service
        .try_admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(compatible_requirement))
        .unwrap()
        .expect("an explicit compatible capability proof admits");
    assert_eq!(admitted.device_ordinal(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restricted_capability_profile_rejects_missing_async_requirement() {
    let (service, _) = restricted_capability_service_and_requirement();

    let error = service.admit_async().await.expect_err("async admission requires a proof");

    assert_explicit_requirement_required(&error);
    assert_eq!(service.queued_len(), 0);
}

#[test]
fn raised_requirement_floor_is_preserved_across_admissions() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        test_ledger(100, 4, Default::default()),
    )
    .unwrap();

    let raised = service
        .try_admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(requirement(40, Some(70))))
        .unwrap()
        .expect("raised floor should admit");
    assert!(raised.device_budget_bytes() >= 40);
    let second = service.try_admit_query().unwrap().expect("remaining capacity admits unconstrained query");
    assert!(second.device_budget_bytes() > 0);
}

#[test]
fn admission_only_service_never_resolves_cuda_device() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        512,
        1,
        1,
    );
    let handle = service.admit_query().expect("admission-only service must not resolve a CUDA device");
    assert_eq!(handle.device_ordinal(), 0);
}
