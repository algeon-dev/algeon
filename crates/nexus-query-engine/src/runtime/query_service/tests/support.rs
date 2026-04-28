#[derive(Clone, Debug)]
struct AdmissionTestController {
    service: QueryService,
}

impl AdmissionTestController {
    fn new(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        QueryService::new_for_admission_tests(config, tenants, deployment_io_capability).map(|service| Self { service })
    }

    fn admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<AdmissionTestHandle> {
        self.service.admit_query(tenant_id).map(AdmissionTestHandle::new)
    }

    async fn admit_async(&self, tenant_id: impl Into<TenantId>) -> Result<AdmissionTestHandle> {
        self.service.admit_async(tenant_id).await.map(AdmissionTestHandle::new)
    }

    fn admit_query_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<AdmissionTestHandle> {
        self.service.admit_query_with_request(tenant_id, request).map(AdmissionTestHandle::new)
    }

    async fn admit_async_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<AdmissionTestHandle> {
        self.service.admit_async_with_request(tenant_id, request).await.map(AdmissionTestHandle::new)
    }

    fn try_admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<Option<AdmissionTestHandle>> {
        self.service.try_admit_query(tenant_id).map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn try_admit_query_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<Option<AdmissionTestHandle>> {
        self.service.try_admit_query_with_request(tenant_id, request).map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn try_admit_query_with_qos(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
    ) -> Result<Option<AdmissionTestHandle>> {
        self.service.try_admit_query_with_qos(tenant_id, qos).map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn try_admit_query_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<Option<AdmissionTestHandle>> {
        self.service
            .try_admit_query_with_qos_and_requested_grant_bytes(tenant_id, qos, requested_grant_bytes)
            .map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn try_admit_query_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<Option<AdmissionTestHandle>> {
        self.service
            .try_admit_query_with_qos_and_grant_request(tenant_id, qos, grant_request)
            .map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn admit_query_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<AdmissionTestHandle> {
        self.service.admit_query_with_qos_and_grant_request(tenant_id, qos, grant_request).map(AdmissionTestHandle::new)
    }

    fn enqueue_admission(&self, tenant_id: impl Into<TenantId>) -> Result<AdmissionTestWaiter> {
        self.service.enqueue_admission(tenant_id).map(AdmissionTestWaiter::new)
    }

    fn enqueue_admission_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<AdmissionTestWaiter> {
        self.service.enqueue_admission_with_request(tenant_id, request).map(AdmissionTestWaiter::new)
    }

    fn enqueue_admission_with_qos(&self, tenant_id: impl Into<TenantId>, qos: QueryQos) -> Result<AdmissionTestWaiter> {
        self.service.enqueue_admission_with_qos(tenant_id, qos).map(AdmissionTestWaiter::new)
    }

    fn enqueue_admission_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<AdmissionTestWaiter> {
        self.service
            .enqueue_admission_with_qos_and_grant_request(tenant_id, qos, grant_request)
            .map(AdmissionTestWaiter::new)
    }

    fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        self.service.admission_snapshot()
    }

    fn memory_governor_reports(&self) -> Result<Vec<MemoryGovernorReport>> {
        self.service.memory_governor_reports()
    }

    fn tenant_requested_budget(&self, tenant_id: &TenantId, requested_grant_bytes: Option<u64>) -> Result<u64> {
        self.service.tenant_requested_budget(tenant_id, requested_grant_bytes)
    }

    fn tenant_requested_budget_with_grant_request(
        &self,
        tenant_id: &TenantId,
        grant_request: QueryGrantRequest,
    ) -> Result<u64> {
        self.service.tenant_requested_budget_with_grant_request(tenant_id, grant_request)
    }

    fn queued_len(&self) -> usize {
        self.service.inner.admission.lock().unwrap().queued.len()
    }

    fn queued_len_recovering_poison(&self) -> usize {
        self.service.inner.admission.lock().unwrap_or_else(std::sync::PoisonError::into_inner).queued.len()
    }

    fn poison_admission(&self) {
        let service = self.service.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _state = service.inner.admission.lock().expect("test acquires admission state");
            panic!("injected query service admission mutex poison");
        }));
    }

    fn close(&self) -> Result<()> {
        self.service.close()
    }

    fn close_and_wait_for_quiescence(&self, timeout: Duration) -> Result<()> {
        self.service.close_and_wait_for_quiescence(timeout)
    }

    fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.service.close_and_wait_for_quiescence_until(deadline)
    }

    fn hold_admission_lock(&self) -> Result<AdmissionTestLockGuard<'_>> {
        self.service.lock_admission().map(|state| AdmissionTestLockGuard { state })
    }

    fn shutdown_waiter_count(&self) -> usize {
        self.service.inner.shutdown_waiters.load(Ordering::Acquire)
    }

    fn notify_admission_available(&self) {
        self.service.inner.notify_admission_available();
    }

    fn expire_waiter(&self, waiter: &mut AdmissionTestWaiter) {
        let expired = Instant::now();
        waiter.waiter.deadline = expired;
        let admission_request_id = waiter.waiter.admission_request_id;
        self.service
            .inner
            .admission
            .lock()
            .expect("admission state")
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == admission_request_id)
            .expect("queued request")
            .deadline = expired;
    }

    fn prune_dead_shared_waiters_retains_self(&self) -> bool {
        let mut waiters = vec![Weak::<QueryServiceInner>::new(), Arc::downgrade(&self.service.inner)];
        prune_dead_shared_device_admission_waiters(&mut waiters);
        waiters.len() == 1 && waiters[0].upgrade().is_some_and(|retained| Arc::ptr_eq(&retained, &self.service.inner))
    }

    fn register_shared_device_waiter(&self, device_ordinal: usize) {
        shared_device_admission_waiters()
            .lock()
            .unwrap()
            .insert(device_ordinal, vec![Arc::downgrade(&self.service.inner)]);
    }

    fn with_admission_locked<T>(&self, operation: impl FnOnce() -> T) -> T {
        let _guard = self.service.inner.admission.lock().unwrap();
        operation()
    }
}

#[derive(Debug)]
struct AdmissionTestHandle {
    handle: QueryHandle,
}

impl AdmissionTestHandle {
    fn new(handle: QueryHandle) -> Self {
        Self { handle }
    }

    fn admission_sequence(&self) -> u64 {
        self.handle.admission_sequence()
    }

    fn admission_request_id(&self) -> AdmissionRequestId {
        self.handle.admission_request_id()
    }

    fn admission_wait(&self) -> Duration {
        self.handle.admission_wait()
    }

    fn tenant_id(&self) -> &TenantId {
        self.handle.tenant_id()
    }

    fn device_budget_bytes(&self) -> u64 {
        self.handle.device_budget_bytes()
    }

    fn device_ordinal(&self) -> usize {
        self.handle.device_ordinal()
    }

    fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.handle.memory_control_policy()
    }

    fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.handle.boundary_policy()
    }

    fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.handle.source_scheduler_policy()
    }

    fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.handle.source_chunk_planning_mode()
    }

    fn native_pipeline_execution_mode(&self) -> NativePipelineExecutionMode {
        self.handle.native_pipeline_execution_mode()
    }

    fn stddev_samp_execution_mode(&self) -> StddevSampExecutionMode {
        self.handle.stddev_samp_execution_mode()
    }

    fn dynamic_filter_execution_options(&self) -> crate::dynamic_filter::DynamicFilterExecutionOptions {
        self.handle.dynamic_filter_execution_options()
    }

    fn keyed_hash_join_strategy_options(&self) -> crate::capability::KeyedHashJoinStrategyOptions {
        self.handle.keyed_hash_join_strategy_options()
    }

    fn aggregate_strategy_execution_options(&self) -> crate::capability::AggregateStrategyExecutionOptions {
        self.handle.aggregate_strategy_execution_options()
    }

    fn source_chunk_planning_options(&self) -> crate::chunk::SourceChunkPlanningOptions {
        self.handle.source_chunk_planning_options()
    }

    fn memory_grant(&self) -> Result<MemoryGrant> {
        self.handle.lifetime.memory_grant()
    }

    fn finish(&self) {
        self.handle.finish();
    }
}

#[derive(Debug)]
struct AdmissionTestLockGuard<'a> {
    state: std::sync::MutexGuard<'a, AdmissionState>,
}

impl AdmissionTestLockGuard<'_> {
    fn set_active_queries(&mut self, active_queries: usize) {
        self.state.active_queries = active_queries;
    }
}

#[derive(Debug)]
struct AdmissionTestWaiter {
    waiter: QueryAdmissionWaiter,
}

impl AdmissionTestWaiter {
    fn new(waiter: QueryAdmissionWaiter) -> Self {
        Self { waiter }
    }

    fn try_admit(&mut self) -> Result<Option<AdmissionTestHandle>> {
        self.waiter.try_admit().map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn admission_request_id(&self) -> AdmissionRequestId {
        self.waiter.admission_request_id
    }
}

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
fn test_service(max_concurrent_queries: usize, max_admission_queue_len: usize) -> AdmissionTestController {
    let config = QueryServiceConfig {
        max_concurrent_queries,
        max_admission_queue_len,
        admission_wait_timeout: Duration::from_millis(500),
        ..QueryServiceConfig::default()
    };
    AdmissionTestController::new(
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
fn wait_for_queued_len(service: &AdmissionTestController, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if service.queued_len() >= expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for queued admissions");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn wait_for_exact_queued_len(service: &AdmissionTestController, expected: usize) {
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
fn optimistic_shrink_test_service(
    native_pipeline_execution_mode: NativePipelineExecutionMode,
) -> AdmissionTestController {
    AdmissionTestController::new(
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
fn first_memory_governor_report(service: &AdmissionTestController) -> MemoryGovernorReport {
    let reports = service.memory_governor_reports().unwrap();
    assert_eq!(reports.len(), 1);
    reports.into_iter().next().unwrap()
}
