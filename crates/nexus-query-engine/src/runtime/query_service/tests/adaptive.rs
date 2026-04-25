#[test]
fn adaptive_query_service_context_uses_query_memory_domain_when_policy_enabled() {
    let config = QueryServiceConfig {
        native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
        query_memory_domain: Some(QueryMemoryDomainPolicy::default()),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();
    let grant = handle.lifetime.memory_grant().unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let token = tenant_context
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("test_domain_policy"))
        .unwrap();

    assert_eq!(token.enforcement(), cudf::ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(grant.hard_cap_bytes(), Some(16 * 1024 * 1024));
}
#[test]
fn adaptive_query_memory_domain_follows_grant_growth() {
    let config = QueryServiceConfig {
        native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
        query_memory_domain: Some(QueryMemoryDomainPolicy::default()),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();
    let grant = handle.lifetime.memory_grant().unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    assert!(grant.try_grow_to(14 * 1024 * 1024).unwrap());

    let token = tenant_context
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("test_domain_grow"))
        .unwrap();
    let outcome =
        tenant_context.context().runtime().unwrap().series_from_value_with_budget(17i32, 3_000_000, &token).unwrap();

    assert_eq!(token.enforcement(), cudf::ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(grant.budget_bytes(), 14 * 1024 * 1024);
    assert_eq!(outcome.value().len(), 3_000_000);
}
#[test]
fn adaptive_query_service_grows_partial_active_grant_on_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 95)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();

    assert_eq!(first.device_budget_bytes(), 90);
    assert_eq!(second.device_budget_bytes(), 10);
    assert_eq!(second_grant.budget_bytes(), 10);

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 90);
    let report = first_memory_governor_report(&service);
    assert_eq!(report.outstanding_bytes, 90);
    assert_eq!(report.available_bytes, 10);
}
#[test]
fn strict_query_service_does_not_grow_partial_active_grant_on_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::StrictBounded,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 95)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 10);
    let report = first_memory_governor_report(&service);
    assert_eq!(report.outstanding_bytes, 10);
    assert_eq!(report.available_bytes, 90);
}
#[test]
fn adaptive_query_service_preserves_queued_minimum_before_growing_active_grants() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 80)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();
    let mut queued = service.enqueue_admission("tenant").unwrap();

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 20);
    let admitted = queued.try_admit().unwrap().expect("queued query should receive released capacity");
    assert_eq!(admitted.device_budget_bytes(), 80);
    assert_eq!(second_grant.budget_bytes(), 20);
}
#[test]
fn adaptive_query_service_grows_after_queue_drains_when_capacity_remains() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("wide", 10, 80), QueryServiceTenantConfig::new("small", 10)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("wide").unwrap().expect("first wide query should admit want");
    let second = service.try_admit_query("wide").unwrap().expect("second wide query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();
    let mut queued = service.enqueue_admission("small").unwrap();

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 20);
    let admitted = queued.try_admit().unwrap().expect("small queued query should receive its minimum");
    assert_eq!(admitted.device_budget_bytes(), 10);
    assert_eq!(second_grant.budget_bytes(), 80);
}
#[test]
fn adaptive_growth_allocations_cap_and_redistribute_by_weight() {
    let allocations = weighted_adaptive_growth_allocations(
        &[
            AdaptiveGrowthCandidate { admission_request_id: AdmissionRequestId::new(1), missing_bytes: 10, weight: 10 },
            AdaptiveGrowthCandidate { admission_request_id: AdmissionRequestId::new(2), missing_bytes: 100, weight: 1 },
            AdaptiveGrowthCandidate { admission_request_id: AdmissionRequestId::new(3), missing_bytes: 100, weight: 1 },
        ],
        60,
    );

    assert_eq!(
        allocations,
        vec![(AdmissionRequestId::new(1), 10), (AdmissionRequestId::new(2), 25), (AdmissionRequestId::new(3), 25),]
    );
}
#[test]
fn adaptive_query_service_grows_partial_grants_by_budget_weight() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 4,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 70)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let high = service
        .try_admit_query_with_qos("tenant", QueryQos::new(Priority::Normal).with_budget_weight(4))
        .unwrap()
        .expect("second query should admit minimum");
    let low_a = service
        .try_admit_query_with_qos("tenant", QueryQos::new(Priority::Normal).with_budget_weight(1))
        .unwrap()
        .expect("third query should admit minimum");
    let low_b = service
        .try_admit_query_with_qos("tenant", QueryQos::new(Priority::Normal).with_budget_weight(1))
        .unwrap()
        .expect("fourth query should admit minimum");
    let high_grant = high.lifetime.memory_grant().unwrap();
    let low_a_grant = low_a.lifetime.memory_grant().unwrap();
    let low_b_grant = low_b.lifetime.memory_grant().unwrap();

    assert_eq!(first.device_budget_bytes(), 70);
    assert_eq!(high_grant.budget_bytes(), 10);
    assert_eq!(low_a_grant.budget_bytes(), 10);
    assert_eq!(low_b_grant.budget_bytes(), 10);

    drop(first);

    assert_eq!(high_grant.budget_bytes(), 50);
    assert_eq!(low_a_grant.budget_bytes(), 20);
    assert_eq!(low_b_grant.budget_bytes(), 20);
    let report = first_memory_governor_report(&service);
    assert_eq!(report.outstanding_bytes, 90);
    assert_eq!(report.available_bytes, 10);
}
#[test]
fn adaptive_blocking_query_service_grows_after_queue_drains_when_capacity_remains() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(1),
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("wide", 10, 80), QueryServiceTenantConfig::new("small", 10)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("wide").unwrap().expect("first wide query should admit want");
    let second = service.try_admit_query("wide").unwrap().expect("second wide query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();
    let queued_service = service.clone();
    let queued = std::thread::spawn(move || queued_service.admit_query("small").unwrap());

    wait_for_queued_len(&service, 1);
    drop(first);

    let admitted = queued.join().expect("blocking admission thread should not panic");
    assert_eq!(admitted.device_budget_bytes(), 10);
    assert_eq!(second_grant.budget_bytes(), 80);
}
#[test]
fn adaptive_optimistic_query_service_requests_shrink_for_memory_blocked_waiter() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();

    assert_eq!(first.device_budget_bytes(), 100);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);

    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);
    assert!(waiter.try_admit().unwrap().is_none());

    assert!(first_grant.try_apply_requested_shrink(60).unwrap());
    let admitted = waiter.try_admit().unwrap().expect("waiter should admit after actual refund");

    assert_eq!(first_grant.budget_bytes(), 60);
    assert_eq!(admitted.device_budget_bytes(), 40);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);
}
#[test]
fn adaptive_optimistic_query_service_respects_raised_active_floor_when_requesting_shrink() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service
        .try_admit_query_with_qos_and_grant_request(
            "tenant",
            QueryQos::default(),
            QueryGrantRequest::with_requested_grant_min_bytes(70),
        )
        .unwrap()
        .expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert_eq!(first.device_budget_bytes(), 100);
    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(70));

    assert!(first_grant.try_apply_requested_shrink(70).unwrap());
    assert_eq!(first_grant.budget_bytes(), 70);
    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_memory_governor_report(&service).available_bytes, 30);
}
#[test]
fn query_service_release_shrinks_retained_grant_to_live_bytes() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let _retained = first_grant.retain_live_reservation(40).expect("retention should be tracked");

    assert_eq!(first_grant.budget_bytes(), 100);
    assert_eq!(first_grant.live_reserved_bytes(), 40);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);

    drop(first);

    let report = first_memory_governor_report(&service);
    assert_eq!(report.outstanding_bytes, 40);
    assert_eq!(report.available_bytes, 60);
    assert_eq!(first_grant.budget_bytes(), 40);
    assert_eq!(first_grant.live_reserved_bytes(), 40);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);

    let second = service.try_admit_query("tenant").unwrap().expect("released slot and refunded budget should admit");
    assert_eq!(second.device_budget_bytes(), 60);
}
#[test]
fn adaptive_optimistic_query_service_requests_all_same_device_shrink_candidates() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(120),
            max_concurrent_queries: 3,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 30, 60)],
        non_gds_deployment(),
    )
    .unwrap();
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let second_grant = second.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert_eq!(first.device_budget_bytes(), 60);
    assert_eq!(second.device_budget_bytes(), 60);
    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(30));
    assert_eq!(second_grant.requested_shrink_target_bytes().unwrap(), None);

    assert!(first_grant.try_apply_requested_shrink(30).unwrap());
    let admitted = waiter.try_admit().unwrap().expect("waiter should admit after the deficit-sized refund");

    assert_eq!(admitted.device_budget_bytes(), 30);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(second_grant.requested_shrink_target_bytes().unwrap(), None);
}
#[test]
fn adaptive_optimistic_query_service_rescinds_requested_shrink_when_waiter_drops() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));

    drop(waiter);

    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);
}
#[test]
fn adaptive_optimistic_query_service_keeps_requested_shrink_for_remaining_waiter() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut first_waiter = service.enqueue_admission("tenant").unwrap();
    let second_waiter = service.enqueue_admission("tenant").unwrap();

    assert!(first_waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));

    drop(first_waiter);

    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 1);

    drop(second_waiter);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
}
#[test]
fn adaptive_optimistic_query_service_rescinds_requested_shrink_when_waiter_times_out() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(10),
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 40, 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    std::thread::sleep(Duration::from_millis(20));
    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_wait_timeout"));
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);
}
#[test]
fn adaptive_optimistic_query_service_rescinds_requested_shrink_when_waiter_admits() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    assert!(first_grant.try_apply_requested_shrink(60).unwrap());
    let admitted = waiter.try_admit().unwrap().expect("waiter should admit after shrink");

    assert_eq!(admitted.device_budget_bytes(), 40);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
}
#[test]
fn adaptive_blocking_query_service_wakes_after_requested_shrink_refunds_capacity() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.admit_query("tenant").unwrap();
    let first_grant = first.lifetime.memory_grant().unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (admitted_sender, admitted_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let result = queued_service.admit_query("tenant").map(|handle| handle.device_budget_bytes());
        admitted_sender.send(result.map_err(|error| error.to_string())).unwrap();
    });

    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);
    wait_for_requested_shrink(&first_grant, Some(60));
    assert!(admitted_receiver.recv_timeout(Duration::from_millis(50)).is_err());

    assert!(first_grant.try_apply_requested_shrink(60).unwrap());
    let admitted_budget = admitted_receiver.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();

    assert_eq!(admitted_budget, 40);
    waiter.join().unwrap();
}
#[test]
fn strict_query_service_does_not_request_shrink_for_memory_blocked_waiter() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::StrictBounded);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert_eq!(first.device_budget_bytes(), 100);
    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);
}
#[test]
fn adaptive_shrink_deficit_ignores_sentinel_and_unsatisfiable_floors() {
    assert_eq!(QueryService::adaptive_shrink_deficit_bytes(20, 40, u64::MAX, 100), None);
    assert_eq!(QueryService::adaptive_shrink_deficit_bytes(20, 80, 30, 100), None);
    assert_eq!(QueryService::adaptive_shrink_deficit_bytes(20, u64::MAX, 1, u64::MAX), None);
    assert_eq!(QueryService::adaptive_shrink_deficit_bytes(20, 40, 30, 100), Some(50));
}
