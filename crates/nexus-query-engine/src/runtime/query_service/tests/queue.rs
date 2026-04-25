#[test]
fn query_service_reports_overload_when_queue_is_full() {
    let service = test_service(1, 1);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (finish_sender, finish_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let handle = queued_service.admit_query("large").unwrap();
        finish_receiver.recv().unwrap();
        drop(handle);
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    let error = service.admit_query("small").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.kind(), crate::error::QueryErrorKind::ResourceExhausted);
    assert_eq!(error.status(), crate::error::QueryErrorStatus::Temporary);
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));
    assert_eq!(error.fact_value("tenant_id"), Some("small"));

    drop(active);
    finish_sender.send(()).unwrap();
    waiter.join().unwrap();
}
#[test]
fn query_service_reports_admission_snapshot_by_tenant_order() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let _handle = queued_service.admit_query("small").unwrap();
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 1);
    assert_eq!(snapshot.next_tenant_id.as_ref().map(TenantId::as_str), Some("small"));
    assert_eq!(snapshot.queued_by_tenant.len(), 2);
    assert_eq!(snapshot.queued_by_tenant[0].tenant_id.as_str(), "large");
    assert_eq!(snapshot.queued_by_tenant[0].queued_queries, 0);
    assert_eq!(snapshot.queued_by_tenant[1].tenant_id.as_str(), "small");
    assert_eq!(snapshot.queued_by_tenant[1].queued_queries, 1);

    drop(active);
    waiter.join().unwrap();
}
#[test]
fn query_service_try_admit_does_not_queue_when_saturated() {
    let service = test_service(1, 8);
    let active = service.try_admit_query("large").unwrap().expect("first query should admit");

    assert!(service.try_admit_query("small").unwrap().is_none());
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 0);

    drop(active);
    assert!(service.try_admit_query("small").unwrap().is_some());
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_admit_async_wakes_on_release() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async("small").await });
    wait_for_queued_len(&service, 1);

    drop(active);
    let handle = time::timeout(Duration::from_millis(200), queued).await.unwrap().unwrap().unwrap();

    assert_eq!(handle.tenant_id().as_str(), "small");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_admit_async_notifies_after_queue_head_admits() {
    let service = test_service(3, 8);
    let active_1 = service.admit_query("large").unwrap();
    let active_2 = service.admit_query("large").unwrap();
    let active_3 = service.admit_query("large").unwrap();
    let first_service = service.clone();
    let second_service = service.clone();
    let first = tokio::spawn(async move { first_service.admit_async("small").await });
    let second = tokio::spawn(async move { second_service.admit_async("small").await });
    wait_for_queued_len(&service, 2);

    drop(active_1);
    drop(active_2);
    drop(active_3);
    let first = time::timeout(Duration::from_millis(200), first).await.unwrap().unwrap().unwrap();
    let second = time::timeout(Duration::from_millis(200), second).await.unwrap().unwrap().unwrap();

    assert_eq!(first.tenant_id().as_str(), "small");
    assert_eq!(second.tenant_id().as_str(), "small");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 2);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_cancelled_admit_async_removes_queued_request() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async("small").await });
    wait_for_queued_len(&service, 1);

    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    wait_for_exact_queued_len(&service, 0);
    drop(active);

    assert!(service.try_admit_query("small").unwrap().is_some());
}
#[test]
fn query_service_concurrent_budget_sum_never_exceeds_capacity() {
    let capacity_bytes = 12 * 1024 * 1024;
    let max_concurrent_queries = 4;
    let per_query_budget = capacity_bytes / u64::try_from(max_concurrent_queries).unwrap();
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(capacity_bytes),
            max_concurrent_queries,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", per_query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let handles = (0..max_concurrent_queries)
        .map(|_| service.try_admit_query("tenant").unwrap().expect("capacity should admit query"))
        .collect::<Vec<_>>();

    let admitted_budget_sum: u64 = handles.iter().map(QueryHandle::device_budget_bytes).sum();
    assert_eq!(admitted_budget_sum, capacity_bytes);
    assert!(service.try_admit_query("tenant").unwrap().is_none());
}
#[test]
fn query_service_uses_variable_memory_governor_grants() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 4,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 60)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should preserve remaining minima");
    let third = service.try_admit_query("tenant").unwrap().expect("third query should admit minimum grant");
    let fourth = service.try_admit_query("tenant").unwrap().expect("fourth query should admit minimum grant");

    assert_eq!(first.device_budget_bytes(), 60);
    assert_eq!(second.device_budget_bytes(), 20);
    assert_eq!(third.device_budget_bytes(), 10);
    assert_eq!(fourth.device_budget_bytes(), 10);
    assert!(service.try_admit_query("tenant").unwrap().is_none());
    let report = first_memory_governor_report(&service);
    assert_eq!(report.capacity_bytes, 100);
    assert_eq!(report.outstanding_bytes, 100);
    assert_eq!(report.available_bytes, 0);

    drop(first);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 60);
}
#[test]
fn query_service_waiting_on_memory_grant_wakes_after_grant_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(2),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("tenant").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service.admit_query("tenant").unwrap()
    });
    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);

    drop(active);
    let admitted = waiter.join().unwrap();
    assert_eq!(admitted.device_budget_bytes(), 100);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);
    drop(admitted);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 100);
}
#[test]
fn query_service_waiting_on_memory_grant_wakes_after_retained_grant_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(2),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("tenant").unwrap();
    let retained_grant = active.lifetime.memory_grant().unwrap();
    drop(active);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);

    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service.admit_query("tenant").unwrap()
    });
    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);

    drop(retained_grant);
    let admitted = waiter.join().unwrap();
    assert_eq!(admitted.device_budget_bytes(), 100);
}
#[test]
fn query_admission_waiter_times_out_while_selected_but_memory_blocked() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(20),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let _active = service.admit_query("tenant").unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();
    std::thread::sleep(Duration::from_millis(30));

    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_wait_timeout"));
}
#[test]
fn query_service_work_conserving_admission_skips_memory_blocked_head() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![
            QueryServiceTenantConfig::new("holder", 90),
            QueryServiceTenantConfig::new("large", 100),
            QueryServiceTenantConfig::new("small", 10),
        ],
        non_gds_deployment(),
    )
    .unwrap();
    let _active = service.admit_query("holder").unwrap();
    let mut large = service.enqueue_admission("large").unwrap();
    let mut small = service.enqueue_admission("small").unwrap();

    assert!(large.try_admit().unwrap().is_none());
    let admitted_small = small.try_admit().unwrap().expect("small query should pass memory-blocked large query");

    assert_eq!(admitted_small.tenant_id().as_str(), "small");
    assert_eq!(admitted_small.device_budget_bytes(), 10);
    assert_eq!(admitted_small.admission_sequence(), 2);
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 2);
    assert_eq!(snapshot.queued_queries, 1);
    assert_eq!(snapshot.queued_by_tenant[1].tenant_id.as_str(), "large");
    assert_eq!(snapshot.queued_by_tenant[1].queued_queries, 1);
}
#[test]
fn query_service_admission_priority_passes_lower_priority_queue_head() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 1,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![
            QueryServiceTenantConfig::new("holder", 100),
            QueryServiceTenantConfig::new("normal", 10),
            QueryServiceTenantConfig::new("high", 10),
        ],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("holder").unwrap();
    let mut normal = service.enqueue_admission("normal").unwrap();
    let mut high = service.enqueue_admission_with_qos("high", QueryQos::new(Priority::High)).unwrap();
    drop(active);

    assert!(normal.try_admit().unwrap().is_none());
    let admitted_high = high.try_admit().unwrap().expect("high-priority query should pass normal-priority head");

    assert_eq!(admitted_high.tenant_id().as_str(), "high");
    assert_eq!(admitted_high.admission_sequence(), 2);
    let report = first_memory_governor_report(&service);
    assert_eq!(report.grants.len(), 1);
    assert_eq!(report.grants[0].priority, "high");
}
#[test]
fn query_service_enqueued_waiter_counts_against_queue_limit_until_drop() {
    let service = test_service(1, 1);
    let _active = service.admit_query("large").unwrap();
    let waiter = service.enqueue_admission("small").unwrap();

    let error = service.enqueue_admission("large").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));

    drop(waiter);
    assert!(service.enqueue_admission("large").is_ok());
}
#[test]
fn query_service_sweeps_expired_unpolled_waiter_before_queue_and_try_admit_gates() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 1,
            max_admission_queue_len: 1,
            admission_wait_timeout: Duration::from_millis(10),
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("tenant").unwrap();
    let stale_waiter = service.enqueue_admission("tenant").unwrap();
    std::thread::sleep(Duration::from_millis(20));

    let replacement_waiter = service.enqueue_admission("tenant").expect("expired waiter should be swept");
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 1);
    drop(replacement_waiter);
    drop(active);

    let admitted = service.try_admit_query("tenant").unwrap().expect("expired stale waiter must not block try-admit");
    assert_eq!(admitted.device_budget_bytes(), 100);
    drop(stale_waiter);
}
#[test]
fn query_service_close_wakes_waiting_admission() {
    let service = test_service(1, 8);
    let _active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service.admit_query("small").unwrap_err()
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    service.close().unwrap();
    let error = waiter.join().unwrap();

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("admission_closed"));
}
#[test]
fn query_service_rotates_admission_by_tenant_when_queries_wait() {
    let service = test_service(1, 8);
    let first = service.admit_query("large").unwrap();
    let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
    let (result_sender, result_receiver) = std::sync::mpsc::channel();

    for tenant in ["large", "large", "small"] {
        let service = service.clone();
        let ready_sender = ready_sender.clone();
        let result_sender = result_sender.clone();
        std::thread::spawn(move || {
            ready_sender.send(tenant.to_string()).unwrap();
            let handle = service.admit_query(tenant).unwrap();
            result_sender.send(handle.tenant_id().to_string()).unwrap();
            std::thread::sleep(Duration::from_millis(20));
            drop(handle);
        });
    }
    for _ in 0..3 {
        ready_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    }
    wait_for_queued_len(&service, 3);

    drop(first);
    assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "small");
    assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "large");
    assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "large");
}
