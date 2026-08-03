#[test]
fn attempt_requirement_hint_selects_grant_for_one_admission() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        non_gds_deployment(),
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
fn impossible_requirement_is_rejected_before_enqueue() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        non_gds_deployment(),
        test_ledger(1024, 1, Default::default()),
    )
    .unwrap();

    let error = service
        .try_admit_query_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(2048, Some(2048))),
        )
        .expect_err("floor exceeding capacity must reject");
    assert_eq!(error.fact_value("reason"), Some("impossible_requirement"));
}

#[test]
fn canonical_graph_proof_missing_from_exact_device_profile_is_rejected_before_queueing() {
    use crate::{
        admission::QueryPlanCapabilityProof,
        runtime::{DeviceCapabilities, DeviceCapability, ReservationKind},
    };

    let service = AdmissionTestController::new_with_device_capabilities(
        QueryExecutionConfig {
            ..QueryExecutionConfig::default()
        },
        QueueBounds::default(),
        non_gds_deployment(),
        DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()),
    )
    .unwrap();
    let error = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new()
                .with_query_plan_capability_proof(QueryPlanCapabilityProof::for_cugraph_graph_execution()),
        )
        .expect_err("an unavailable device capability must reject locally");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
    assert_eq!(error.fact_value("required_capabilities"), Some("graph_algorithm:cugraph_graph_execution_contract"));
    assert_eq!(error.fact_value("configured_device_count"), Some("1"));
    assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
    assert_eq!(service.queued_len(), 0, "capability rejection must never enter the admission queue");
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
        non_gds_deployment(),
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
        non_gds_deployment(),
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
