fn observed_admission_events(owner: &RuntimeObservationOwner) -> Vec<(AdmissionRequestId, AdmissionLifecycleState)> {
    let collect = |events: &[crate::observability::RuntimeTimelineEvent]| {
        events
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::Admission(observation) => {
                    Some((observation.admission_request_id, observation.lifecycle.clone()))
                },
                _ => None,
            })
            .collect()
    };
    if let Some(snapshot) = owner.live_snapshot() {
        collect(snapshot.events())
    } else {
        collect(owner.sealed_snapshot().expect("observation is live or sealed").events())
    }
}
fn wait_for_sealed_observation(
    owner: &RuntimeObservationOwner,
) -> crate::observability::SealedRuntimeObservationSnapshot {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if let Some(snapshot) = owner.sealed_snapshot() {
            return snapshot;
        }
        assert!(Instant::now() < deadline, "timed out waiting for observation sealing");
        std::thread::yield_now();
    }
}
fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}
fn test_service(max_concurrent_queries: usize, max_admission_queue_len: usize) -> QueryService {
    let config = QueryServiceConfig {
        max_concurrent_queries,
        max_admission_queue_len,
        admission_wait_timeout: Duration::from_millis(500),
        ..QueryServiceConfig::default()
    };
    QueryService::new_for_admission_tests(
        config,
        vec![
            QueryServiceTenantConfig::new("large", 8 * 1024 * 1024),
            QueryServiceTenantConfig::new("small", 1024 * 1024),
        ],
        non_gds_deployment(),
    )
    .unwrap()
}
fn runtime_test_service(max_concurrent_queries: usize, max_admission_queue_len: usize) -> QueryService {
    let config = QueryServiceConfig {
        max_concurrent_queries,
        max_admission_queue_len,
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    QueryService::new_with_deployment(
        config,
        vec![
            QueryServiceTenantConfig::new("large", 8 * 1024 * 1024),
            QueryServiceTenantConfig::new("small", 1024 * 1024),
        ],
        non_gds_deployment(),
    )
    .unwrap()
}
fn try_guaranteed_runtime_test_service() -> Result<QueryService> {
    try_guaranteed_runtime_test_service_with_concurrency(1)
}
fn try_guaranteed_runtime_test_service_with_concurrency(max_concurrent_queries: usize) -> Result<QueryService> {
    QueryService::new_with_deployment(
        QueryServiceConfig {
            max_concurrent_queries,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(500),
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
}
fn wait_for_queued_len(service: &QueryService, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let state = service.inner.admission.lock().unwrap();
        if state.queued.len() >= expected {
            return;
        }
        drop(state);
        assert!(Instant::now() < deadline, "timed out waiting for queued admissions");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn wait_for_exact_queued_len(service: &QueryService, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let queued = service.admission_snapshot().unwrap().queued_queries;
        if queued == expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {expected} queued admissions; last queued={queued}");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn optimistic_shrink_test_service(native_pipeline_execution_mode: NativePipelineExecutionMode) -> QueryService {
    QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(2),
            native_pipeline_execution_mode,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 40, 100)],
        non_gds_deployment(),
    )
    .unwrap()
}
fn wait_for_requested_shrink(grant: &MemoryGrant, expected: Option<u64>) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let actual = grant.requested_shrink_target_bytes().unwrap();
        if actual == expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for shrink target {expected:?}; last target={actual:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn first_memory_governor_report(service: &QueryService) -> MemoryGovernorReport {
    let reports = service.memory_governor_reports().unwrap();
    assert_eq!(reports.len(), 1);
    reports.into_iter().next().unwrap()
}
